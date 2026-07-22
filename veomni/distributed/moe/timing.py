"""Lightweight MoE accelerator event spans shared by EP MoE components."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist

from ...utils.accelerator_timing import (
    AcceleratorEvent,
    accelerator_timing_available,
    cuda_nvtx_available,
    record_accelerator_event,
    synchronize_accelerator,
)


_MOE_TIMING_CONTEXT_STACK: list[dict[str, Any]] = []
_MOE_TIMING_SPANS: list[dict[str, Any]] = []


def moe_timing_enabled() -> bool:
    return bool(os.environ.get("VERL_MOE_TIMING_DIR")) and accelerator_timing_available()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _sync_timing_events() -> bool:
    return _env_flag(
        "VEOMNI_MOE_TIMING_SYNC_EVENTS",
        _env_flag("VEOMNI_MOE_TIMING_SYNC_SPANS", False),
    )


def _torch_profiler_requested() -> bool:
    return _env_flag("VEOMNI_TORCH_PROFILE_ENABLE", False) or _env_flag("VERL_TORCH_PROFILE_ENABLE", False)


def _current_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
    else:
        rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("LOCAL_WORLD_RANK", "0")))
    return rank, local_rank


def _rank_allowed(raw: str, rank: int, local_rank: int) -> bool:
    raw = raw.strip().lower()
    if raw in {"", "all", "*"}:
        return True
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return str(rank) in allowed or str(local_rank) in allowed


def _annotation_label(
    meta: dict[str, Any] | None,
    *,
    direction: str,
    component: str,
    section: str,
) -> str:
    parts = ["veomni", "moe", direction, component, section]
    if meta:
        for key in ("phase", "layer", "ep_size", "tokens", "token_expert_assignments", "top_k", "call_index"):
            value = meta.get(key)
            if value is not None:
                text = re.sub(r"\s+", "_", str(value))[:80]
                parts.append(f"{key}={text}")
    return "/".join(parts)


def current_full_profile_phase() -> str | None:
    try:
        from veomni.utils.full_timing_profiler import get_active_full_timing_profiler

        profiler = get_active_full_timing_profiler()
    except Exception:
        profiler = None
    if profiler is None:
        return None
    phase = getattr(profiler, "current_phase", None)
    return str(phase) if phase is not None else None


def with_current_full_profile_phase(meta: dict[str, Any] | None, *, force: bool = False) -> dict[str, Any] | None:
    if meta is None:
        return None
    phase = current_full_profile_phase()
    if phase is None and not force:
        return meta
    updated = dict(meta)
    if phase is not None:
        updated["phase"] = phase
    return updated


def enter_moe_profile_range(
    meta: dict[str, Any] | None,
    *,
    direction: str,
    component: str,
    section: str,
) -> dict[str, Any] | None:
    default = _torch_profiler_requested()
    rank, local_rank = _current_rank()
    annotation_rank_filter = os.environ.get(
        "VEOMNI_FULL_PROFILE_ANNOTATION_RANKS",
        os.environ.get("VEOMNI_TORCH_PROFILE_EXPORT_RANKS", "all"),
    )
    if not _rank_allowed(annotation_rank_filter, rank, local_rank):
        return None
    emit_nvtx = cuda_nvtx_available() and _env_flag(
        "VEOMNI_MOE_PROFILE_NVTX", _env_flag("VEOMNI_FULL_PROFILE_NVTX", default)
    )
    emit_record_function = _env_flag("VEOMNI_MOE_PROFILE_RECORD_FUNCTION", default)
    if not emit_nvtx and not emit_record_function:
        return None

    label = _annotation_label(meta, direction=direction, component=component, section=section)
    token: dict[str, Any] = {"nvtx": False, "record_function": None}
    if emit_nvtx:
        torch.cuda.nvtx.range_push(label)
        token["nvtx"] = True
    if emit_record_function:
        ctx = torch.profiler.record_function(label)
        ctx.__enter__()
        token["record_function"] = ctx
    return token


def exit_moe_profile_range(token: dict[str, Any] | None) -> None:
    if not token:
        return
    ctx = token.get("record_function")
    if ctx is not None:
        ctx.__exit__(None, None, None)
    if token.get("nvtx"):
        torch.cuda.nvtx.range_pop()


@contextmanager
def moe_profile_range(
    meta: dict[str, Any] | None,
    *,
    direction: str,
    component: str,
    section: str,
) -> Iterator[None]:
    token = enter_moe_profile_range(meta, direction=direction, component=component, section=section)
    try:
        yield
    finally:
        exit_moe_profile_range(token)


def moe_timing_event() -> AcceleratorEvent | None:
    if not moe_timing_enabled():
        return None
    if _sync_timing_events():
        synchronize_accelerator()
    return record_accelerator_event()


@contextmanager
def moe_timing_context(
    record: dict[str, Any] | None,
    *,
    component: str,
    section: str,
) -> Iterator[None]:
    if record is None or not moe_timing_enabled():
        yield
        return

    meta = {
        "call_index": record.get("call_index"),
        "layer": record.get("layer"),
        "num_layers": record.get("num_layers"),
        "num_experts": record.get("num_experts"),
        "ep_size": record.get("ep_size"),
        "tokens": record.get("tokens"),
        "token_expert_assignments": record.get("token_expert_assignments"),
        "top_k": record.get("top_k"),
        "phase": record.get("phase") or current_full_profile_phase(),
        "component": component,
        "section": section,
    }
    _MOE_TIMING_CONTEXT_STACK.append(meta)
    try:
        yield
    finally:
        _MOE_TIMING_CONTEXT_STACK.pop()


def current_moe_timing_context() -> dict[str, Any] | None:
    if not _MOE_TIMING_CONTEXT_STACK or not moe_timing_enabled():
        return None
    return dict(_MOE_TIMING_CONTEXT_STACK[-1])


def record_moe_timing_span(
    meta: dict[str, Any] | None,
    *,
    direction: str,
    component: str,
    section: str,
    start_event: AcceleratorEvent | None,
    end_event: AcceleratorEvent | None,
) -> None:
    if meta is None or start_event is None or end_event is None:
        return
    if direction == "backward":
        meta = with_current_full_profile_phase(meta, force=True) or meta
    _MOE_TIMING_SPANS.append(
        {
            "phase": meta.get("phase"),
            "direction": direction,
            "component": component,
            "section": section,
            "start_event": start_event,
            "end_event": end_event,
            "call_index": meta.get("call_index"),
            "micro_batch": (
                int(meta.get("call_index")) // int(meta.get("num_layers"))
                if meta.get("call_index") is not None and meta.get("num_layers")
                else None
            ),
            "layer": meta.get("layer"),
            "num_layers": meta.get("num_layers"),
            "num_experts": meta.get("num_experts"),
            "ep_size": meta.get("ep_size"),
            "tokens": meta.get("tokens"),
            "token_expert_assignments": meta.get("token_expert_assignments"),
            "top_k": meta.get("top_k"),
        }
    )


def _sum_span_rows(spans: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "cuda_ms_sum": 0.0,
            "cuda_ms_max": 0.0,
            "wall_ms_sum": 0.0,
            "wall_ms_max": 0.0,
            "tokens": 0,
            "token_expert_assignments": 0,
        }
    )
    for span in spans:
        cuda_ms = float(span["start_event"].elapsed_time(span["end_event"]))
        wall_ms = float((span["end_event"].wall_time - span["start_event"].wall_time) * 1000.0)
        group_key = tuple(span.get(key) for key in keys)
        row = grouped[group_key]
        row["calls"] += 1
        row["cuda_ms_sum"] += cuda_ms
        row["cuda_ms_max"] = max(float(row["cuda_ms_max"]), cuda_ms)
        row["wall_ms_sum"] += wall_ms
        row["wall_ms_max"] = max(float(row["wall_ms_max"]), wall_ms)
        row["tokens"] += int(span.get("tokens") or 0)
        row["token_expert_assignments"] += int(span.get("token_expert_assignments") or 0)

    rows = []
    for group_key, row in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        item = dict(zip(keys, group_key, strict=True))
        item.update(row)
        calls = max(1, int(item["calls"]))
        item["cuda_ms_avg"] = float(item["cuda_ms_sum"]) / calls
        item["wall_ms_avg"] = float(item["wall_ms_sum"]) / calls
        rows.append(item)
    return rows


def flush_moe_timing_spans() -> dict[str, Any]:
    global _MOE_TIMING_SPANS
    spans = _MOE_TIMING_SPANS
    _MOE_TIMING_SPANS = []
    if not spans:
        return {}

    synchronize_accelerator()
    return {
        "span_layers": _sum_span_rows(spans, ["layer", "direction", "component", "section"]),
        "span_layers_by_phase": _sum_span_rows(spans, ["phase", "layer", "direction", "component", "section"]),
        "span_calls": _sum_span_rows(
            spans, ["call_index", "micro_batch", "layer", "direction", "component", "section"]
        ),
        "span_calls_by_phase": _sum_span_rows(
            spans, ["phase", "call_index", "micro_batch", "layer", "direction", "component", "section"]
        ),
        "span_components": _sum_span_rows(spans, ["direction", "component"]),
        "span_phase_components": _sum_span_rows(spans, ["phase", "direction", "component"]),
    }
