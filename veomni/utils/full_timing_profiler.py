# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-overhead JSONL timing profiler shared by VeOmni pretrain and verl RL.

The profiler is environment-gated and intentionally has no dependency on wandb.
It records accelerator events without synchronizing each section; synchronization
happens once when a sampled step is flushed.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn

from .accelerator_timing import (
    AcceleratorEvent,
    cuda_nvtx_available,
    record_accelerator_event,
    synchronize_accelerator,
)


_GLOBAL_PROFILER: "FullTimingProfiler | None" = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_set(name: str, default: str = "") -> set[str]:
    raw = os.environ.get(name, default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _torch_profiler_requested() -> bool:
    return _env_flag("VEOMNI_TORCH_PROFILE_ENABLE", False) or _env_flag("VERL_TORCH_PROFILE_ENABLE", False)


def _rank_allowed_by_filter(raw: str, rank: int, local_rank: int) -> bool:
    raw = raw.strip().lower()
    if raw in {"", "all", "*"}:
        return True
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return str(rank) in allowed or str(local_rank) in allowed


def _dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("LOCAL_WORLD_RANK", "0")))


def _record_event() -> AcceleratorEvent | None:
    return record_accelerator_event()


def _tensor_token_count(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 3:
            return int(value.shape[0] * value.shape[1])
        if value.ndim >= 1:
            return int(value.shape[0])
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        for item in value:
            tokens = _tensor_token_count(item)
            if tokens is not None:
                return tokens
    if isinstance(value, dict):
        for key in ("hidden_states", "inputs_embeds", "input_ids", "pixel_values"):
            if key in value:
                tokens = _tensor_token_count(value[key])
                if tokens is not None:
                    return tokens
    return None


def _token_expert_assignments_from_output(output: Any) -> int | None:
    if isinstance(output, tuple) and len(output) >= 3 and isinstance(output[2], torch.Tensor):
        return int(output[2].numel())
    if isinstance(output, dict):
        for key in ("selected_experts", "router_indices"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.numel())
    return None


def _layer_idx_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
    if match:
        return int(match.group(1))
    return None


def _clean_label_value(value: Any) -> str:
    text = str(value)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


class FullTimingProfiler:
    """Environment-gated step, module, and coarse RL phase timing profiler."""

    def __init__(
        self,
        *,
        run_kind: str,
        rank: int | None = None,
        local_rank: int | None = None,
        fsdp_size: int | None = None,
        ep_size: int | None = None,
    ) -> None:
        self.run_kind = os.environ.get("VEOMNI_FULL_PROFILE_RUN_KIND", run_kind)
        self.rank = _dist_rank() if rank is None else int(rank)
        self.global_rank = self.rank
        self.local_rank = _local_rank() if local_rank is None else int(local_rank)
        self.fsdp_size = fsdp_size
        self.ep_size = ep_size

        self.enabled = _env_flag("VEOMNI_FULL_PROFILE_ENABLE", False)
        self.profile_dir = Path(os.environ.get("VEOMNI_FULL_PROFILE_DIR", ""))
        self.start_step = _env_int("VEOMNI_FULL_PROFILE_START_STEP", 1)
        self.every_n = max(1, _env_int("VEOMNI_FULL_PROFILE_EVERY_N", 1))
        self.with_backward = _env_flag("VEOMNI_FULL_PROFILE_WITH_BACKWARD", True)
        self.rank_filter = os.environ.get("VEOMNI_FULL_PROFILE_RANKS", "all").strip().lower()
        annotate_default = _torch_profiler_requested()
        annotation_rank_filter = os.environ.get(
            "VEOMNI_FULL_PROFILE_ANNOTATION_RANKS",
            os.environ.get("VEOMNI_TORCH_PROFILE_EXPORT_RANKS", self.rank_filter),
        )
        annotate_this_rank = _rank_allowed_by_filter(annotation_rank_filter, self.rank, self.local_rank)
        self.emit_nvtx = (
            annotate_this_rank and cuda_nvtx_available() and _env_flag("VEOMNI_FULL_PROFILE_NVTX", annotate_default)
        )
        self.emit_record_function = annotate_this_rank and _env_flag(
            "VEOMNI_FULL_PROFILE_RECORD_FUNCTION",
            annotate_default,
        )
        self.record_function_types = _env_set("VEOMNI_FULL_PROFILE_RECORD_FUNCTION_TYPES", "step")
        if self.enabled and (not self.profile_dir or not self._rank_allowed()):
            self.enabled = False

        self.current_step: int | None = None
        self.current_phase: str | None = None
        self.recording = False
        self._step_wall_start = 0.0
        self._step_start_event: AcceleratorEvent | None = None
        self._step_annotation: dict[str, Any] | None = None
        self._step_metadata: dict[str, Any] = {}
        self._records: list[dict[str, Any]] = []
        self._module_forward_stacks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._module_backward_stacks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._handles: list[Any] = []
        self._attached = False
        self._external_step = 0

        if self.enabled:
            self.profile_dir.mkdir(parents=True, exist_ok=True)

    def _rank_allowed(self) -> bool:
        return _rank_allowed_by_filter(self.rank_filter, self.rank, self.local_rank)

    def update_context(
        self,
        *,
        run_kind: str | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        fsdp_size: int | None = None,
        ep_size: int | None = None,
    ) -> None:
        if run_kind is not None:
            self.run_kind = run_kind
        if rank is not None:
            self.rank = int(rank)
            self.global_rank = int(rank)
        if local_rank is not None:
            self.local_rank = int(local_rank)
        if fsdp_size is not None:
            self.fsdp_size = int(fsdp_size)
        if ep_size is not None:
            self.ep_size = int(ep_size)

    def should_sample(self, step: int) -> bool:
        return self.enabled and step >= self.start_step and (step - self.start_step) % self.every_n == 0

    def begin_step(self, step: int, *, metadata: dict[str, Any] | None = None) -> None:
        self.current_step = int(step)
        self.recording = self.should_sample(int(step))
        self._records = []
        self._step_metadata = dict(metadata or {})
        self._step_wall_start = time.perf_counter()
        if self.recording:
            self._step_annotation = self._enter_annotation(
                record_type="step",
                section="train_step_total",
                metadata=self._step_metadata,
            )
            self._step_start_event = _record_event()
        else:
            self._step_annotation = None
            self._step_start_event = None

    def end_step(self, *, metadata: dict[str, Any] | None = None) -> None:
        if not self.recording or self.current_step is None:
            self._exit_annotation(self._step_annotation)
            self._step_annotation = None
            self.current_step = None
            self.recording = False
            return

        record = {
            "record_type": "step",
            "section": "train_step_total",
            "start_event": self._step_start_event,
            "end_event": _record_event(),
            "start_wall": self._step_wall_start,
            "end_wall": time.perf_counter(),
            "metadata": {**self._step_metadata, **(metadata or {})},
        }
        self._exit_annotation(self._step_annotation)
        self._step_annotation = None
        self._records.append(record)
        self.flush()
        self.current_step = None
        self.recording = False

    @contextlib.contextmanager
    def cuda_range(
        self,
        section: str,
        *,
        metadata: dict[str, Any] | None = None,
        record_type: str = "step",
    ) -> Iterator[None]:
        if not self.recording or self.current_step is None:
            yield
            return

        previous_phase = self.current_phase
        self.current_phase = section
        record = self._start_record(record_type=record_type, section=section, metadata=metadata)
        try:
            yield
        finally:
            self._finish_record(record)
            self.current_phase = previous_phase

    def record_external_phase(
        self,
        section: str,
        *,
        wall_ms: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
        record_type: str = "rl_phase",
    ) -> None:
        if not self.enabled:
            return
        if step is None:
            self._external_step += 1
            step = self._external_step
        if not self.should_sample(int(step)):
            return
        payload = self._base_payload(
            step=int(step),
            record_type=record_type,
            section=section,
            cuda_ms=None,
            wall_ms=float(wall_ms),
            metadata=metadata,
        )
        self._write_payload(record_type, payload)

    @contextlib.contextmanager
    def external_phase(
        self,
        section: str,
        *,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
        record_type: str = "rl_phase",
    ) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_external_phase(
                section,
                step=step,
                wall_ms=(time.perf_counter() - start) * 1000.0,
                metadata=metadata,
                record_type=record_type,
            )

    def attach_model_hooks(self, model: nn.Module) -> int:
        if not self.enabled or self._attached:
            return 0
        attached = 0
        for name, module in model.named_modules():
            module_meta = self._module_metadata(name, module)
            if module_meta is None:
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_forward_pre_hook(module_meta)))
            self._handles.append(module.register_forward_hook(self._make_forward_hook(module_meta)))
            attached += 1
            if self.with_backward:
                try:
                    self._handles.append(
                        module.register_full_backward_pre_hook(self._make_backward_pre_hook(module_meta))
                    )
                    self._handles.append(module.register_full_backward_hook(self._make_backward_hook(module_meta)))
                except RuntimeError:
                    # Some wrapped modules may reject backward hooks. Forward
                    # timing is still useful, so keep those hooks.
                    pass
        self._attached = True
        return attached

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached = False

    def flush(self) -> None:
        if not self._records:
            return
        synchronize_accelerator()
        for record in self._records:
            payload = self._payload_from_record(record)
            self._write_payload(record["record_type"], payload)
        self._records = []

    def _start_record(
        self,
        *,
        record_type: str,
        section: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        return {
            "record_type": record_type,
            "section": section,
            "annotation": self._enter_annotation(record_type=record_type, section=section, metadata=metadata),
            "start_event": _record_event(),
            "end_event": None,
            "start_wall": time.perf_counter(),
            "end_wall": None,
            "metadata": metadata,
        }

    def _finish_record(self, record: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        record["end_event"] = _record_event()
        record["end_wall"] = time.perf_counter()
        if metadata:
            record["metadata"].update(metadata)
        self._exit_annotation(record.get("annotation"))
        record["annotation"] = None
        self._records.append(record)

    def _annotation_label(
        self,
        *,
        record_type: str,
        section: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = metadata or {}
        parts = ["veomni", self.run_kind, record_type, section]
        for key in ("module_name", "module_type", "layer_idx", "phase", "rl_stage", "forward_only"):
            value = metadata.get(key)
            if value is not None:
                parts.append(f"{key}={_clean_label_value(value)}")
        if self.current_step is not None:
            parts.append(f"step={int(self.current_step)}")
        parts.append(f"rank={int(self.rank)}")
        return "/".join(parts)

    def _enter_annotation(
        self,
        *,
        record_type: str,
        section: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        record_type_key = str(record_type).lower()
        section_key = str(section).lower()
        emit_record_function = self.emit_record_function and (
            "all" in self.record_function_types
            or record_type_key in self.record_function_types
            or section_key in self.record_function_types
        )
        if not self.emit_nvtx and not emit_record_function:
            return None
        label = self._annotation_label(record_type=record_type, section=section, metadata=metadata)
        token: dict[str, Any] = {"nvtx": False, "record_function": None}
        if self.emit_nvtx and cuda_nvtx_available():
            torch.cuda.nvtx.range_push(label)
            token["nvtx"] = True
        if emit_record_function:
            ctx = torch.profiler.record_function(label)
            ctx.__enter__()
            token["record_function"] = ctx
        return token

    @staticmethod
    def _exit_annotation(token: dict[str, Any] | None) -> None:
        if not token:
            return
        ctx = token.get("record_function")
        if ctx is not None:
            ctx.__exit__(None, None, None)
        if token.get("nvtx"):
            torch.cuda.nvtx.range_pop()

    def _payload_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        start_event = record.get("start_event")
        end_event = record.get("end_event")
        cuda_ms = None
        if start_event is not None and end_event is not None:
            cuda_ms = float(start_event.elapsed_time(end_event))
        wall_ms = None
        if record.get("start_wall") is not None and record.get("end_wall") is not None:
            wall_ms = float((record["end_wall"] - record["start_wall"]) * 1000.0)
        return self._base_payload(
            step=int(self.current_step or 0),
            record_type=record["record_type"],
            section=record["section"],
            cuda_ms=cuda_ms,
            wall_ms=wall_ms,
            metadata=record.get("metadata"),
        )

    def _base_payload(
        self,
        *,
        step: int,
        record_type: str,
        section: str,
        cuda_ms: float | None,
        wall_ms: float | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = {
            "run_kind": self.run_kind,
            "step": int(step),
            "rank": int(self.rank),
            "global_rank": int(self.global_rank),
            "local_rank": int(self.local_rank),
            "fsdp_size": self.fsdp_size,
            "ep_size": self.ep_size,
            "record_type": record_type,
            "section": section,
            "cuda_ms": cuda_ms,
            "wall_ms": wall_ms,
            "module_name": None,
            "module_type": None,
            "layer_idx": None,
            "tokens": None,
            "token_expert_assignments": None,
            "phase": self.current_phase,
        }
        if metadata:
            payload.update(metadata)
        return payload

    def _write_payload(self, record_type: str, payload: dict[str, Any]) -> None:
        if record_type == "module_forward":
            filename = f"module_forward_timing_rank{self.rank}.jsonl"
        elif record_type == "module_backward":
            filename = f"module_backward_timing_rank{self.rank}.jsonl"
        elif record_type == "rl_phase":
            filename = f"rl_phase_timing_rank{self.rank}.jsonl"
        else:
            filename = f"step_timing_rank{self.rank}.jsonl"
        path = self.profile_dir / filename
        with path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _module_metadata(self, name: str, module: nn.Module) -> dict[str, Any] | None:
        module_type = type(module).__name__
        section = None
        if name.endswith("embed_tokens"):
            section = "embedding"
        elif name.endswith("visual") or module_type.endswith("VisionModel"):
            section = "vision_encoder"
        elif module_type.endswith("VisionBlock"):
            section = "vision_block"
        elif module_type.endswith("VisionAttention"):
            section = "vision_attention"
        elif module_type.endswith("VisionMLP"):
            section = "vision_mlp"
        elif module_type.endswith("VisionPatchEmbed"):
            section = "vision_patch_embed"
        elif module_type.endswith("VisionPatchMerger"):
            section = "vision_patch_merger"
        elif module_type.endswith("DecoderLayer"):
            section = "decoder_layer"
        elif name.endswith("self_attn") or module_type.endswith("Attention"):
            section = "attention"
        elif name.endswith("input_layernorm"):
            section = "input_layernorm"
        elif name.endswith("post_attention_layernorm"):
            section = "post_attention_layernorm"
        elif module_type.endswith("SparseMoeBlock"):
            section = "moe_block"
        elif name.endswith("mlp"):
            section = "mlp"
        elif name.endswith("gate") or module_type.endswith("TopKRouter"):
            section = "router_gate"

        if section is None:
            return None
        return {
            "module_name": name,
            "module_type": module_type,
            "layer_idx": _layer_idx_from_name(name),
            "section": section,
        }

    def _make_forward_pre_hook(self, module_meta: dict[str, Any]):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            if not self.recording or self.current_step is None:
                return
            record = self._start_record(
                record_type="module_forward",
                section=module_meta["section"],
                metadata={**module_meta, "phase": self.current_phase, "tokens": _tensor_token_count(args)},
            )
            self._module_forward_stacks[id(module)].append(record)

        return hook

    def _make_forward_hook(self, module_meta: dict[str, Any]):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if not self.recording or self.current_step is None:
                return
            stack = self._module_forward_stacks.get(id(module))
            if not stack:
                return
            record = stack.pop()
            token_expert_assignments = _token_expert_assignments_from_output(output)
            extra: dict[str, Any] = {}
            if token_expert_assignments is not None:
                extra["token_expert_assignments"] = token_expert_assignments
            self._finish_record(record, metadata=extra)

        return hook

    def _make_backward_pre_hook(self, module_meta: dict[str, Any]):
        def hook(module: nn.Module, grad_output: tuple[Any, ...]) -> None:
            if not self.recording or self.current_step is None:
                return
            record = self._start_record(
                record_type="module_backward",
                section=module_meta["section"],
                metadata={**module_meta, "phase": self.current_phase, "tokens": _tensor_token_count(grad_output)},
            )
            self._module_backward_stacks[id(module)].append(record)

        return hook

    def _make_backward_hook(self, module_meta: dict[str, Any]):
        def hook(module: nn.Module, grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            if not self.recording or self.current_step is None:
                return
            stack = self._module_backward_stacks.get(id(module))
            if not stack:
                return
            record = stack.pop()
            self._finish_record(record)

        return hook


def get_full_timing_profiler(
    *,
    run_kind: str,
    rank: int | None = None,
    local_rank: int | None = None,
    fsdp_size: int | None = None,
    ep_size: int | None = None,
) -> FullTimingProfiler:
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is None:
        _GLOBAL_PROFILER = FullTimingProfiler(
            run_kind=run_kind,
            rank=rank,
            local_rank=local_rank,
            fsdp_size=fsdp_size,
            ep_size=ep_size,
        )
    else:
        _GLOBAL_PROFILER.update_context(
            run_kind=run_kind,
            rank=rank,
            local_rank=local_rank,
            fsdp_size=fsdp_size,
            ep_size=ep_size,
        )
    return _GLOBAL_PROFILER


def get_active_full_timing_profiler() -> FullTimingProfiler | None:
    """Return the active profiler only while a sampled step is recording."""
    profiler = _GLOBAL_PROFILER
    if profiler is None or not profiler.enabled or not profiler.recording or profiler.current_step is None:
        return None
    return profiler
