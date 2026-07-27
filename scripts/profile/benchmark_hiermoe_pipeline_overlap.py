#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Measure whether one-layer planning and expert migration are actually hidden."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch import nn

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.expert_swap import (
    _PIPELINE_PREPARE_SUBSTAGES,
    ExpertSwapManager,
    _CoverTensorEntry,
    _LayerSwapPlan,
    _SwapTensorEntry,
)
from veomni.distributed.moe.hiermoe.greedy_planner import assign_tokens_to_copies_greedy
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.state import configure_hiermoe
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.utils.accelerator_timing import AcceleratorEvent, record_accelerator_event


_LAYER_KEY = "benchmark.layers.0.mlp.experts"
_MANAGER_KEEPALIVE: list[ExpertSwapManager] = []
_PREPARE_CHUNKS = (
    ("c1_context", ("planner_setup", "context")),
    ("c1_baseline", ("route_hash", "baseline_route", "occupancy")),
    ("c2_candidate_routes", ("candidate_routes",)),
    ("c3_statistics", ("pair_events", "unary_statistics")),
    (
        "c4_scoring",
        ("unary_scoring", "pair_statistics", "pair_interaction", "candidate_pack", "collective_pack"),
    ),
)


_PREPARE_SCHEDULE_ATOMS = tuple((stage, (stage,)) for stage in _PIPELINE_PREPARE_SUBSTAGES)


@dataclass(frozen=True)
class _UniformChunkSchedule:
    stage1_end: int
    stage2_end: int
    hidden_ms: float
    exposed_ms: float
    min_slack_ms: float

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
            "stage1_chunks": [name for name, _stages in _PREPARE_CHUNKS[: self.stage1_end]],
            "stage2_chunks": [name for name, _stages in _PREPARE_CHUNKS[self.stage1_end : self.stage2_end]],
            "later_chunks": [name for name, _stages in _PREPARE_CHUNKS[self.stage2_end :]],
            "hidden_ms": self.hidden_ms,
            "exposed_ms": self.exposed_ms,
            "hidden_ratio": self.hidden_ms / max(1e-9, self.hidden_ms + self.exposed_ms),
            "min_slack_ms": self.min_slack_ms,
        }


@dataclass(frozen=True)
class _UniformMultiWindowSchedule:
    cut_points: tuple[int, ...]
    hidden_ms: float
    exposed_ms: float
    min_slack_ms: float

    def as_dict(
        self,
        window_names: Sequence[str],
        chunk_definitions: Sequence[tuple[str, Sequence[str]]] = _PREPARE_CHUNKS,
    ) -> dict[str, Any]:
        start = 0
        window_chunks = {}
        for name, end in zip(window_names, self.cut_points, strict=True):
            window_chunks[name] = [chunk_name for chunk_name, _stages in chunk_definitions[start:end]]
            start = end
        return {
            "cut_points": list(self.cut_points),
            "window_chunks": window_chunks,
            "later_chunks": [name for name, _stages in chunk_definitions[start:]],
            "hidden_ms": self.hidden_ms,
            "exposed_ms": self.exposed_ms,
            "hidden_ratio": self.hidden_ms / max(1e-9, self.hidden_ms + self.exposed_ms),
            "min_slack_ms": self.min_slack_ms,
        }


def _select_uniform_multi_window_schedule(
    window_rows_ms: Sequence[Sequence[float]],
    chunk_rows_ms: Sequence[Sequence[float]],
    *,
    window_scale: float = 1.0,
    chunk_scale: float = 1.0,
    guard_ms: float = 0.0,
    chunk_definitions: Sequence[tuple[str, Sequence[str]]] = _PREPARE_CHUNKS,
) -> _UniformMultiWindowSchedule:
    if len(window_rows_ms) != len(chunk_rows_ms):
        raise ValueError("A2A window and planner chunk rows must have identical sample counts.")
    if not chunk_rows_ms:
        raise ValueError("At least one timing sample is required.")
    window_count = len(window_rows_ms[0])
    chunk_count = len(chunk_definitions)
    if window_count <= 0 or any(len(row) != window_count for row in window_rows_ms):
        raise ValueError("Every A2A timing row must contain the same positive number of windows.")
    if any(len(row) != chunk_count for row in chunk_rows_ms):
        raise ValueError(f"Every planner timing row must contain {chunk_count} chunks.")
    if window_scale <= 0.0 or chunk_scale <= 0.0 or guard_ms < 0.0:
        raise ValueError("Schedule safety scales must be positive and guard_ms must be non-negative.")

    best = None
    best_key = None
    for cut_points in combinations_with_replacement(range(chunk_count + 1), window_count):
        slacks = []
        hidden_rows = []
        exposed_rows = []
        for windows, chunks in zip(window_rows_ms, chunk_rows_ms, strict=True):
            start = 0
            for window, end in zip(windows, cut_points, strict=True):
                work = chunk_scale * sum(chunks[start:end])
                slacks.append(window_scale * window - (guard_ms if work > 0.0 else 0.0) - work)
                start = end
            hidden_rows.append(sum(chunks[:start]))
            exposed_rows.append(sum(chunks[start:]))
        min_slack = min(slacks)
        if min_slack < 0.0:
            continue
        hidden_ms = sum(hidden_rows) / len(hidden_rows)
        exposed_ms = sum(exposed_rows) / len(exposed_rows)
        candidate = _UniformMultiWindowSchedule(
            cut_points=tuple(cut_points),
            hidden_ms=hidden_ms,
            exposed_ms=exposed_ms,
            min_slack_ms=min_slack,
        )
        key = (hidden_ms, min_slack, tuple(-value for value in cut_points))
        if best_key is None or key > best_key:
            best = candidate
            best_key = key
    if best is None:
        raise RuntimeError("No safe uniform multi-window planner schedule was found.")
    return best


def _select_uniform_chunk_schedule(
    stage1_windows_ms: Sequence[float],
    stage2_windows_ms: Sequence[float],
    chunk_rows_ms: Sequence[Sequence[float]],
    *,
    window_scale: float = 1.0,
    chunk_scale: float = 1.0,
    guard_ms: float = 0.0,
) -> _UniformChunkSchedule:
    if not (len(stage1_windows_ms) == len(stage2_windows_ms) == len(chunk_rows_ms)):
        raise ValueError("A2A windows and planner chunk rows must have identical sample counts.")
    if not chunk_rows_ms:
        raise ValueError("At least one timing sample is required.")
    chunk_count = len(_PREPARE_CHUNKS)
    if any(len(row) != chunk_count for row in chunk_rows_ms):
        raise ValueError(f"Every planner timing row must contain {chunk_count} chunks.")
    if window_scale <= 0.0 or chunk_scale <= 0.0 or guard_ms < 0.0:
        raise ValueError("Schedule safety scales must be positive and guard_ms must be non-negative.")

    best: _UniformChunkSchedule | None = None
    best_key: tuple[float, float, int] | None = None
    for stage1_end in range(chunk_count + 1):
        for stage2_end in range(stage1_end, chunk_count + 1):
            slacks = []
            hidden_rows = []
            exposed_rows = []
            for stage1_window, stage2_window, chunks in zip(
                stage1_windows_ms, stage2_windows_ms, chunk_rows_ms, strict=True
            ):
                stage1_work = chunk_scale * sum(chunks[:stage1_end])
                stage2_work = chunk_scale * sum(chunks[stage1_end:stage2_end])
                slacks.extend(
                    (
                        window_scale * stage1_window - (guard_ms if stage1_work > 0.0 else 0.0) - stage1_work,
                        window_scale * stage2_window - (guard_ms if stage2_work > 0.0 else 0.0) - stage2_work,
                    )
                )
                hidden_rows.append(sum(chunks[:stage2_end]))
                exposed_rows.append(sum(chunks[stage2_end:]))
            min_slack = min(slacks)
            if min_slack < 0.0:
                continue
            hidden_ms = sum(hidden_rows) / len(hidden_rows)
            exposed_ms = sum(exposed_rows) / len(exposed_rows)
            candidate = _UniformChunkSchedule(
                stage1_end=stage1_end,
                stage2_end=stage2_end,
                hidden_ms=hidden_ms,
                exposed_ms=exposed_ms,
                min_slack_ms=min_slack,
            )
            key = (hidden_ms, min_slack, -stage1_end)
            if best_key is None or key > best_key:
                best = candidate
                best_key = key
    if best is None:
        raise RuntimeError("No safe uniform planner schedule was found.")
    return best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("planner", "swap", "cover", "all"), default="all")
    parser.add_argument("--route-dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--slot-increment", type=int, default=1)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--compute-window-ms", type=float, default=50.0)
    parser.add_argument("--foreground-a2a-mib", type=float, default=64.0)
    parser.add_argument("--foreground-transport", choices=("raw", "rank-dedup"), default="raw")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--planner-budget-ms", type=float, default=50.0)
    parser.add_argument("--min-hidden-ratio", type=float, default=0.90)
    parser.add_argument("--max-dilation-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser.parse_args()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(samples),
        "p90": _percentile(samples, 0.9),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _hidden_ratio(background_ms: float, foreground_ms: float, overlapped_ms: float) -> float:
    if background_ms <= 0.0:
        return 1.0
    exposed_ms = max(0.0, overlapped_ms - foreground_ms)
    return max(0.0, min(1.0, 1.0 - exposed_ms / background_ms))


def _global_max(value: float, device: torch.device) -> float:
    maximum = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def _global_min(value: float, device: torch.device) -> float:
    minimum = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    return float(minimum.item())


def _event_elapsed(start: AcceleratorEvent | None, end: AcceleratorEvent | None) -> float:
    if start is None or end is None:
        return 0.0
    return start.elapsed_time(end)


def _rank_dedup_stage_timings(state: Any) -> dict[str, float]:
    if not isinstance(state, tuple) or len(state) < 2:
        return {}
    context = state[1]
    events = getattr(context, "internal_timing_events", None)
    if not events:
        return {}
    return {
        f"{stage}_device_ms": _event_elapsed(start, end)
        for stage, (start, end) in events.items()
    }


def _rank_dedup_backward_stage_timings(state: Any) -> dict[str, float]:
    if not isinstance(state, tuple) or len(state) < 2:
        return {}
    context = state[1]
    events = getattr(context, "backward_internal_timing_events", None)
    if not events:
        return {}
    timings = {}
    for stage in (
        "backward_combine_stage1_a2a",
        "backward_combine_stage2_a2a",
        "backward_dispatch_stage2_a2a",
        "backward_dispatch_stage1_a2a",
    ):
        start = events.get(f"{stage}_start")
        end = events.get(f"{stage}_end")
        if start is not None and end is not None:
            timings[f"{stage}_device_ms"] = _event_elapsed(start, end)
    return timings


def _barrier_and_synchronize(device: torch.device) -> None:
    dist.barrier()
    torch.npu.synchronize(device)


class _PlannerExperts(nn.Module):
    def __init__(self, num_experts: int, slots_per_rank: int, device: torch.device) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.gate_up_proj = nn.Parameter(torch.zeros((slots_per_rank, 1), device=device))
        self.down_proj = nn.Parameter(torch.zeros((slots_per_rank, 1), device=device))


def _load_route(route_dir: Path, layer: int, rank: int, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or record.get("format") != "veomni.hiermoe.local_route":
        raise ValueError(f"{path} is not a VeOmni local-route snapshot.")
    routes = record["routes"].to(device=device, dtype=torch.long, non_blocking=True).contiguous()
    return routes, record


def _steady_layout(
    num_experts: int,
    ep_size: int,
    slots_per_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = num_experts // ep_size
    experts = torch.arange(num_experts, dtype=torch.long, device=device)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + torch.remainder(experts, base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, experts)
    ranks = torch.arange(ep_size, dtype=torch.long, device=device)
    for replica_offset in range(slots_per_rank - base):
        logical_offset = replica_offset % base
        replica_rank_offset = replica_offset // base + 1
        layout[ranks * slots_per_rank + base + replica_offset] = (
            torch.remainder(ranks + replica_rank_offset, ep_size) * base + logical_offset
        )
    return layout, owners


def _make_manager(
    *,
    ep_size: int,
    ep_rank: int,
    hierarchy: Hierarchy,
    slot_increment: int,
    max_copies: int,
) -> ExpertSwapManager:
    return ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=ep_size,
        ep_rank=ep_rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=slot_increment,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
        fixed_pipeline_overlap=True,
        greedy_max_copies_per_expert=max_copies,
    )


def _calibrate_gemm(
    target_ms: float,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[Callable[[], None], int, float]:
    rows = 4096
    lhs = torch.randn((rows, hidden_size), dtype=torch.bfloat16, device=device)
    rhs = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, intermediate_size), dtype=torch.bfloat16, device=device)

    def run_iterations(count: int) -> None:
        for _ in range(count):
            torch.mm(lhs, rhs, out=output)

    run_iterations(1)
    _barrier_and_synchronize(device)

    iterations = 1
    measured_ms = 0.0
    for _ in range(6):
        _barrier_and_synchronize(device)
        started = time.perf_counter()
        run_iterations(iterations)
        torch.npu.synchronize(device)
        measured_ms = _global_max((time.perf_counter() - started) * 1000.0, device)
        if measured_ms >= max(1.0, target_ms * 0.75):
            break
        iterations = max(iterations + 1, int(math.ceil(iterations * target_ms / max(measured_ms, 0.1))))
    scaled = max(1, int(round(iterations * target_ms / max(measured_ms, 0.1))))
    _barrier_and_synchronize(device)
    started = time.perf_counter()
    run_iterations(scaled)
    torch.npu.synchronize(device)
    calibrated_ms = _global_max((time.perf_counter() - started) * 1000.0, device)

    def run() -> None:
        run_iterations(scaled)

    return run, scaled, calibrated_ms


def _run_planner_benchmark(
    args: argparse.Namespace,
    *,
    rank: int,
    ep_size: int,
    device: torch.device,
    hierarchy: Hierarchy,
    run_compute: Callable[[], None],
) -> dict[str, Any]:
    if args.route_dir is None:
        raise ValueError("--route-dir is required for planner mode.")
    routes, metadata = _load_route(args.route_dir, args.layer, rank, device)
    if int(metadata["num_experts"]) != args.num_experts:
        raise ValueError("Route and --num-experts disagree.")
    base = args.num_experts // ep_size
    slots_per_rank = base + args.slot_increment
    layout, owners = _steady_layout(args.num_experts, ep_size, slots_per_rank, device)

    manager = _make_manager(
        ep_size=ep_size,
        ep_rank=rank,
        hierarchy=hierarchy,
        slot_increment=args.slot_increment,
        max_copies=args.max_copies,
    )
    _MANAGER_KEEPALIVE.append(manager)
    module = _PlannerExperts(args.num_experts, slots_per_rank, device)
    manager.register_layer(_LAYER_KEY, module)
    layer = manager.layers[_LAYER_KEY]
    layer.slot_to_logical = layout.cpu()
    manager._refresh_layer_mapping_from_slots(layer, owners.cpu())

    if args.foreground_transport == "rank-dedup":
        physical_routes = assign_tokens_to_copies_greedy(
            routes,
            layout,
            slots_per_rank=slots_per_rank,
            source_ranks=rank,
            hierarchy_group_sizes=hierarchy.group_sizes,
            num_experts=args.num_experts,
            step=1,
            layer_seed=args.layer,
            max_copies=args.max_copies,
        ).contiguous()
        hidden = torch.zeros(
            (routes.shape[0], args.hidden_size), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        weights = torch.full(routes.shape, 1.0 / int(routes.shape[1]), dtype=torch.float32, device=device)

        def foreground_dispatch() -> tuple[torch.Tensor, Any]:
            permuted, context, _ = rank_dedup_dispatch(
                hidden,
                physical_routes,
                weights,
                int(layout.numel()),
                dist.group.WORLD,
            )
            return permuted, context

        def foreground_combine(state: tuple[torch.Tensor, Any]) -> torch.Tensor:
            return rank_dedup_combine(*state)

    else:
        a2a_numel = max(ep_size, int(args.foreground_a2a_mib * 1024 * 1024 / 2))
        a2a_numel -= a2a_numel % ep_size
        a2a_input = torch.zeros((a2a_numel,), dtype=torch.bfloat16, device=device)
        a2a_output = torch.empty_like(a2a_input)

        def foreground_dispatch() -> None:
            dist.all_to_all_single(a2a_output, a2a_input, group=dist.group.WORLD)

        def foreground_combine(_state: None) -> None:
            dist.all_to_all_single(a2a_output, a2a_input, group=dist.group.WORLD)

    def foreground_only() -> tuple[float, dict[str, float]]:
        _barrier_and_synchronize(device)
        started = time.perf_counter()
        dispatch_start = record_accelerator_event()
        dispatch_host_started = time.perf_counter()
        dispatch_thread_started = time.thread_time()
        foreground_state = foreground_dispatch()
        dispatch_thread_ms = (time.thread_time() - dispatch_thread_started) * 1000.0
        dispatch_host_ms = (time.perf_counter() - dispatch_host_started) * 1000.0
        dispatch_end = record_accelerator_event()
        compute_start = record_accelerator_event()
        compute_host_started = time.perf_counter()
        compute_thread_started = time.thread_time()
        run_compute()
        compute_thread_ms = (time.thread_time() - compute_thread_started) * 1000.0
        compute_host_ms = (time.perf_counter() - compute_host_started) * 1000.0
        compute_end = record_accelerator_event()
        combine_start = record_accelerator_event()
        combine_host_started = time.perf_counter()
        combine_thread_started = time.thread_time()
        foreground_combine(foreground_state)
        combine_thread_ms = (time.thread_time() - combine_thread_started) * 1000.0
        combine_host_ms = (time.perf_counter() - combine_host_started) * 1000.0
        combine_end = record_accelerator_event()
        torch.npu.synchronize(device)
        internal_stages = _rank_dedup_stage_timings(foreground_state)
        return _global_max((time.perf_counter() - started) * 1000.0, device), {
            "dispatch_device_ms": _event_elapsed(dispatch_start, dispatch_end),
            "compute_device_ms": _event_elapsed(compute_start, compute_end),
            "combine_device_ms": _event_elapsed(combine_start, combine_end),
            "dispatch_host_ms": dispatch_host_ms,
            "compute_host_ms": compute_host_ms,
            "combine_host_ms": combine_host_ms,
            "dispatch_thread_cpu_ms": dispatch_thread_ms,
            "compute_thread_cpu_ms": compute_thread_ms,
            "combine_thread_cpu_ms": combine_thread_ms,
            **internal_stages,
        }

    def backward_windows_only() -> tuple[float, dict[str, float]]:
        if args.foreground_transport != "rank-dedup":
            return 0.0, {}
        hidden.grad = None
        _barrier_and_synchronize(device)
        foreground_state = foreground_dispatch()
        combined = foreground_combine(foreground_state)
        backward_start = record_accelerator_event()
        torch.autograd.backward(combined, torch.ones_like(combined))
        backward_end = record_accelerator_event()
        torch.npu.synchronize(device)
        internal_stages = _rank_dedup_backward_stage_timings(foreground_state)
        hidden.grad = None
        return _global_max(_event_elapsed(backward_start, backward_end), device), internal_stages

    step = 0

    def planner_only() -> tuple[float, dict[str, float | int | str]]:
        nonlocal step
        manager._pipeline_pending_plans.clear()
        manager.configure_pipeline_microstep(step=step, micro_step=0, num_micro_steps=1)
        _barrier_and_synchronize(device)
        started = time.perf_counter()
        manager.record_routing(
            layer_key=_LAYER_KEY,
            selected_experts=routes,
            hidden_size=args.hidden_size,
            bytes_per_element=int(metadata["bytes_per_element"]),
            step=step,
        )
        manager.release_pipeline_planner_prepare(_LAYER_KEY)
        manager.open_pipeline_planner_collective_window(_LAYER_KEY)
        manager.close_pipeline_planner_collective_window(_LAYER_KEY)
        manager.open_pipeline_planner_score_window(_LAYER_KEY)
        manager.close_pipeline_planner_score_window(_LAYER_KEY)
        manager.maybe_swap(step)
        torch.npu.synchronize(device)
        elapsed_ms = _global_max((time.perf_counter() - started) * 1000.0, device)
        metrics = manager.placement_metrics()
        manager._pipeline_pending_plans.clear()
        step += 1
        return elapsed_ms, metrics

    def planner_overlapped() -> tuple[float, dict[str, float | int | str], dict[str, float]]:
        nonlocal step
        manager._pipeline_pending_plans.clear()
        manager.configure_pipeline_microstep(step=step, micro_step=0, num_micro_steps=1)
        _barrier_and_synchronize(device)
        started = time.perf_counter()
        manager.record_routing(
            layer_key=_LAYER_KEY,
            selected_experts=routes,
            hidden_size=args.hidden_size,
            bytes_per_element=int(metadata["bytes_per_element"]),
            step=step,
        )
        manager.release_pipeline_planner_prepare(_LAYER_KEY)
        dispatch_start = record_accelerator_event()
        dispatch_host_started = time.perf_counter()
        dispatch_thread_started = time.thread_time()
        foreground_state = foreground_dispatch()
        dispatch_thread_ms = (time.thread_time() - dispatch_thread_started) * 1000.0
        dispatch_host_ms = (time.perf_counter() - dispatch_host_started) * 1000.0
        dispatch_end = record_accelerator_event()
        manager.open_pipeline_planner_collective_window(_LAYER_KEY)
        compute_start = record_accelerator_event()
        compute_host_started = time.perf_counter()
        compute_thread_started = time.thread_time()
        run_compute()
        compute_thread_ms = (time.thread_time() - compute_thread_started) * 1000.0
        compute_host_ms = (time.perf_counter() - compute_host_started) * 1000.0
        compute_end = record_accelerator_event()
        manager.close_pipeline_planner_collective_window(_LAYER_KEY)
        manager.open_pipeline_planner_score_window(_LAYER_KEY)
        combine_start = record_accelerator_event()
        combine_host_started = time.perf_counter()
        combine_thread_started = time.thread_time()
        foreground_combine(foreground_state)
        combine_thread_ms = (time.thread_time() - combine_thread_started) * 1000.0
        combine_host_ms = (time.perf_counter() - combine_host_started) * 1000.0
        combine_end = record_accelerator_event()
        manager.close_pipeline_planner_score_window(_LAYER_KEY)
        manager.maybe_swap(step)
        torch.npu.synchronize(device)
        internal_stages = _rank_dedup_stage_timings(foreground_state)
        elapsed_ms = _global_max((time.perf_counter() - started) * 1000.0, device)
        metrics = manager.placement_metrics()
        manager._pipeline_pending_plans.clear()
        step += 1
        return elapsed_ms, metrics, {
            "dispatch_device_ms": _event_elapsed(dispatch_start, dispatch_end),
            "compute_device_ms": _event_elapsed(compute_start, compute_end),
            "combine_device_ms": _event_elapsed(combine_start, combine_end),
            "dispatch_host_ms": dispatch_host_ms,
            "compute_host_ms": compute_host_ms,
            "combine_host_ms": combine_host_ms,
            "dispatch_thread_cpu_ms": dispatch_thread_ms,
            "compute_thread_cpu_ms": compute_thread_ms,
            "combine_thread_cpu_ms": combine_thread_ms,
            **internal_stages,
        }

    samples = {
        "isolated_wall_ms": [],
        "isolated_algorithm_ms": [],
        "foreground_ms": [],
        "overlapped_ms": [],
        "dilation_ms": [],
        "collective_deadline_exposed_ms": [],
        "final_deadline_exposed_ms": [],
        "foreground_dispatch_device_ms": [],
        "foreground_compute_device_ms": [],
        "foreground_combine_device_ms": [],
        "foreground_backward_device_ms": [],
        "overlap_dispatch_device_ms": [],
        "overlap_compute_device_ms": [],
        "overlap_combine_device_ms": [],
        "isolated_planner_prepare_device_ms": [],
        "isolated_planner_collective_device_ms": [],
        "isolated_planner_score_device_ms": [],
        "planner_prepare_device_ms": [],
        "planner_collective_device_ms": [],
        "planner_score_device_ms": [],
    }
    for a2a_stage in (
        "stage1_a2a",
        "stage2_a2a",
        "combine_stage2_a2a",
        "combine_stage1_a2a",
    ):
        samples[f"foreground_{a2a_stage}_device_ms"] = []
        samples[f"overlap_{a2a_stage}_device_ms"] = []
    for a2a_stage in (
        "backward_combine_stage1_a2a",
        "backward_combine_stage2_a2a",
        "backward_dispatch_stage2_a2a",
        "backward_dispatch_stage1_a2a",
    ):
        samples[f"foreground_{a2a_stage}_device_ms"] = []
    for chunk_name, _stages in _PREPARE_CHUNKS:
        samples[f"isolated_planner_{chunk_name}_device_ms"] = []
    for foreground_stage in ("dispatch", "compute", "combine"):
        for clock in ("host", "thread_cpu"):
            samples[f"foreground_{foreground_stage}_{clock}_ms"] = []
            samples[f"overlap_{foreground_stage}_{clock}_ms"] = []
    for prepare_stage in _PIPELINE_PREPARE_SUBSTAGES:
        for clock in ("device", "host", "thread_cpu"):
            samples[f"isolated_planner_prepare_{prepare_stage}_{clock}_ms"] = []
            samples[f"planner_prepare_{prepare_stage}_{clock}_ms"] = []
    try:
        for iteration in range(args.warmup + args.iterations):
            isolated_ms, isolated_metrics = planner_only()
            foreground_ms, foreground_stages = foreground_only()
            backward_ms, backward_stages = backward_windows_only()
            overlapped_ms, overlap_metrics, overlap_stages = planner_overlapped()
            if iteration >= args.warmup:
                samples["isolated_wall_ms"].append(isolated_ms)
                samples["isolated_algorithm_ms"].append(
                    _global_max(float(isolated_metrics["hiermoe/placement_planning_ms"]), device)
                )
                samples["foreground_ms"].append(foreground_ms)
                samples["foreground_backward_device_ms"].append(backward_ms)
                samples["overlapped_ms"].append(overlapped_ms)
                samples["dilation_ms"].append(overlapped_ms - foreground_ms)
                samples["collective_deadline_exposed_ms"].append(
                    _global_max(
                        float(overlap_metrics.get("hiermoe/pipeline_planner_collective_exposed_ms", 0.0)),
                        device,
                    )
                )
                samples["final_deadline_exposed_ms"].append(
                    _global_max(
                        float(overlap_metrics.get("hiermoe/pipeline_planner_deadline_exposed_ms", 0.0)),
                        device,
                    )
                )
                for a2a_stage in (
                    "stage1_a2a",
                    "stage2_a2a",
                    "combine_stage2_a2a",
                    "combine_stage1_a2a",
                ):
                    samples[f"foreground_{a2a_stage}_device_ms"].append(
                        _global_min(foreground_stages.get(f"{a2a_stage}_device_ms", 0.0), device)
                    )
                    samples[f"overlap_{a2a_stage}_device_ms"].append(
                        _global_min(overlap_stages.get(f"{a2a_stage}_device_ms", 0.0), device)
                    )
                for a2a_stage in (
                    "backward_combine_stage1_a2a",
                    "backward_combine_stage2_a2a",
                    "backward_dispatch_stage2_a2a",
                    "backward_dispatch_stage1_a2a",
                ):
                    samples[f"foreground_{a2a_stage}_device_ms"].append(
                        _global_min(backward_stages.get(f"{a2a_stage}_device_ms", 0.0), device)
                    )
                for chunk_name, chunk_stages in _PREPARE_CHUNKS:
                    chunk_ms = sum(
                        float(
                            isolated_metrics.get(
                                f"hiermoe/pipeline_planner_prepare_{stage}_device_ms",
                                0.0,
                            )
                        )
                        for stage in chunk_stages
                    )
                    samples[f"isolated_planner_{chunk_name}_device_ms"].append(
                        _global_max(chunk_ms, device)
                    )
                for stage in ("dispatch", "compute", "combine"):
                    for clock in ("device", "host", "thread_cpu"):
                        samples[f"foreground_{stage}_{clock}_ms"].append(
                            _global_max(foreground_stages[f"{stage}_{clock}_ms"], device)
                        )
                        samples[f"overlap_{stage}_{clock}_ms"].append(
                            _global_max(overlap_stages[f"{stage}_{clock}_ms"], device)
                        )
                for prepare_stage in _PIPELINE_PREPARE_SUBSTAGES:
                    for clock in ("device", "host", "thread_cpu"):
                        metric = f"hiermoe/pipeline_planner_prepare_{prepare_stage}_{clock}_ms"
                        samples[f"isolated_planner_prepare_{prepare_stage}_{clock}_ms"].append(
                            _global_max(float(isolated_metrics.get(metric, 0.0)), device)
                        )
                        samples[f"planner_prepare_{prepare_stage}_{clock}_ms"].append(
                            _global_max(float(overlap_metrics.get(metric, 0.0)), device)
                        )
                for stage in ("prepare", "collective", "score"):
                    samples[f"isolated_planner_{stage}_device_ms"].append(
                        _global_max(
                            float(
                                isolated_metrics.get(
                                    f"hiermoe/pipeline_planner_{stage}_device_ms",
                                    0.0,
                                )
                            ),
                            device,
                        )
                    )
                    samples[f"planner_{stage}_device_ms"].append(
                        _global_max(
                            float(
                                overlap_metrics.get(
                                    f"hiermoe/pipeline_planner_{stage}_device_ms",
                                    0.0,
                                )
                            ),
                            device,
                        )
                    )
    finally:
        manager._pipeline_pending_plans.clear()
        manager.shutdown_pipeline()

    summaries = {name: _summarize(values) for name, values in samples.items()}
    isolated_median = summaries["isolated_wall_ms"]["median"]
    foreground_median = summaries["foreground_ms"]["median"]
    overlapped_median = summaries["overlapped_ms"]["median"]
    hidden_ratio = _hidden_ratio(isolated_median, foreground_median, overlapped_median)
    schedule_search = None
    stage1_windows = samples["foreground_stage1_a2a_device_ms"]
    stage2_windows = samples["foreground_stage2_a2a_device_ms"]
    if all(value > 0.0 for value in (*stage1_windows, *stage2_windows)):
        chunk_rows = list(
            zip(
                *(samples[f"isolated_planner_{name}_device_ms"] for name, _stages in _PREPARE_CHUNKS),
                strict=True,
            )
        )
        raw_schedule = _select_uniform_chunk_schedule(stage1_windows, stage2_windows, chunk_rows)
        safe_schedule = _select_uniform_chunk_schedule(
            stage1_windows,
            stage2_windows,
            chunk_rows,
            window_scale=0.85,
            chunk_scale=1.20,
            guard_ms=0.5,
        )
        schedule_search = {
            "raw": raw_schedule.as_dict(),
            "safe_15pct_window_20pct_chunk_0p5ms_guard": safe_schedule.as_dict(),
            "prepare_chunk_ms": {
                name: summaries[f"isolated_planner_{name}_device_ms"]
                for name, _stages in _PREPARE_CHUNKS
            },
            "stage1_a2a_ms": summaries["foreground_stage1_a2a_device_ms"],
            "stage2_a2a_ms": summaries["foreground_stage2_a2a_device_ms"],
            "collective_ms": summaries["isolated_planner_collective_device_ms"],
            "cost_model_and_decision_ms": summaries["isolated_planner_score_device_ms"],
        }
    forward_backward_window_names = (
        "forward_dispatch_stage1_a2a",
        "forward_dispatch_stage2_a2a",
        "forward_combine_stage2_a2a",
        "forward_combine_stage1_a2a",
        "backward_combine_stage1_a2a",
        "backward_combine_stage2_a2a",
    )
    forward_backward_window_keys = (
        "foreground_stage1_a2a_device_ms",
        "foreground_stage2_a2a_device_ms",
        "foreground_combine_stage2_a2a_device_ms",
        "foreground_combine_stage1_a2a_device_ms",
        "foreground_backward_combine_stage1_a2a_device_ms",
        "foreground_backward_combine_stage2_a2a_device_ms",
    )
    if all(all(value > 0.0 for value in samples[key]) for key in forward_backward_window_keys):
        window_rows = list(zip(*(samples[key] for key in forward_backward_window_keys), strict=True))
        atom_rows = list(
            zip(
                *(
                    samples[f"isolated_planner_prepare_{stage}_device_ms"]
                    for stage, _substage in _PREPARE_SCHEDULE_ATOMS
                ),
                strict=True,
            )
        )
        strict_schedule = _select_uniform_multi_window_schedule(
            window_rows,
            atom_rows,
            chunk_definitions=_PREPARE_SCHEDULE_ATOMS,
        )
        median_schedule = _select_uniform_multi_window_schedule(
            [[summaries[key]["median"] for key in forward_backward_window_keys]],
            [[
                summaries[f"isolated_planner_prepare_{stage}_device_ms"]["median"]
                for stage, _substage in _PREPARE_SCHEDULE_ATOMS
            ]],
            chunk_definitions=_PREPARE_SCHEDULE_ATOMS,
        )
        if schedule_search is None:
            schedule_search = {}
        schedule_search["forward_backward_six_window"] = {
            "median": median_schedule.as_dict(
                forward_backward_window_names, _PREPARE_SCHEDULE_ATOMS
            ),
            "strict_all_samples": strict_schedule.as_dict(
                forward_backward_window_names, _PREPARE_SCHEDULE_ATOMS
            ),
            "window_ms": {
                name: summaries[key]
                for name, key in zip(
                    forward_backward_window_names, forward_backward_window_keys, strict=True
                )
            },
            "prepare_atom_ms": {
                stage: summaries[f"isolated_planner_prepare_{stage}_device_ms"]
                for stage, _substage in _PREPARE_SCHEDULE_ATOMS
            },
        }
    return {
        "route_shape": list(routes.shape),
        "candidate_layout_slots": int(layout.numel()),
        "samples": samples,
        "timing_ms": summaries,
        "hidden_ratio": hidden_ratio,
        "uniform_schedule_search": schedule_search,
        "acceptance": {
            "isolated_algorithm_below_budget": summaries["isolated_algorithm_ms"]["median"] < args.planner_budget_ms,
            "hidden_ratio_at_least_target": hidden_ratio >= args.min_hidden_ratio,
            "median_dilation_within_budget": summaries["dilation_ms"]["median"] <= args.max_dilation_ms,
        },
    }


def _migration_tensors(
    *,
    slots_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    shapes = (
        (slots_per_rank, 2 * intermediate_size, hidden_size),
        (slots_per_rank, hidden_size, intermediate_size),
    )
    parameters = [torch.empty(shape, dtype=torch.bfloat16, device=device) for shape in shapes]
    optimizer_states = [torch.empty(shape, dtype=torch.float32, device=device) for shape in shapes for _ in range(2)]
    return parameters + optimizer_states


def _run_background_transfer(
    executor: ThreadPoolExecutor,
    stream: Any,
    ready_event: Any,
    device: torch.device,
    transfer: Callable[[], None],
) -> Any:
    def worker() -> float:
        torch.npu.set_device(device)
        started = time.perf_counter()
        with torch.npu.stream(stream):
            stream.wait_event(ready_event)
            transfer()
        stream.synchronize()
        return (time.perf_counter() - started) * 1000.0

    return executor.submit(worker)


def _run_migration_kind(
    kind: str,
    args: argparse.Namespace,
    *,
    rank: int,
    ep_size: int,
    device: torch.device,
    hierarchy: Hierarchy,
    run_compute: Callable[[], None],
) -> dict[str, Any]:
    base = args.num_experts // ep_size
    slots_per_rank = base + args.slot_increment
    tensors = _migration_tensors(
        slots_per_rank=slots_per_rank,
        hidden_size=args.hidden_size,
        intermediate_size=args.moe_intermediate_size,
        device=device,
    )
    source_rank = 0
    destination_rank = ep_size // 2
    source_slot = 0
    destination_slot = 0 if kind == "swap" else slots_per_rank - 1
    payload_bytes = sum(int(tensor[0].numel()) * int(tensor.element_size()) for tensor in tensors)

    manager = _make_manager(
        ep_size=ep_size,
        ep_rank=rank,
        hierarchy=hierarchy,
        slot_increment=args.slot_increment,
        max_copies=args.max_copies,
    )
    _MANAGER_KEEPALIVE.append(manager)
    if kind == "swap":
        entries = tuple(_SwapTensorEntry(tensor, source_slot, destination_slot) for tensor in tensors)
        plan = _LayerSwapPlan(
            layer_key=_LAYER_KEY,
            logical_lhs=0,
            logical_rhs=args.num_experts // 2,
            lhs_rank=source_rank,
            rhs_rank=destination_rank,
            entries=entries,
        )

        def transfer() -> None:
            manager._execute_swap_plan_batch((plan,))

    else:
        cover_entries = tuple(_CoverTensorEntry(tensor, source_slot, destination_slot) for tensor in tensors)

        def transfer() -> None:
            manager._execute_sparse_group_slot_transfers(
                {(source_rank, destination_rank): list(cover_entries)},
                process_group=dist.group.WORLD,
            )

    def reset() -> None:
        if rank == source_rank:
            for tensor in tensors:
                tensor[source_slot].fill_(1.0)
        if rank == destination_rank:
            fill_slot = destination_slot
            for tensor in tensors:
                tensor[fill_slot].fill_(2.0)
        torch.npu.synchronize(device)

    def validate() -> None:
        local_error = torch.zeros((), dtype=torch.float32, device=device)
        if kind == "swap" and rank == source_rank:
            local_error = torch.stack(
                [(tensor[source_slot].reshape(-1)[0].float() - 2.0).abs() for tensor in tensors]
            ).max()
        elif rank == destination_rank:
            expected = 1.0
            local_error = torch.stack(
                [(tensor[destination_slot].reshape(-1)[0].float() - expected).abs() for tensor in tensors]
            ).max()
        dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
        if float(local_error.item()) != 0.0:
            raise RuntimeError(f"{kind} migration validation failed with max error {float(local_error.item())}.")

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"benchmark-{kind}")
    stream = torch.npu.Stream(device=device)

    def foreground_only() -> float:
        _barrier_and_synchronize(device)
        started = time.perf_counter()
        run_compute()
        torch.npu.synchronize(device)
        return _global_max((time.perf_counter() - started) * 1000.0, device)

    def migration_only() -> float:
        reset()
        _barrier_and_synchronize(device)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(device))
        started = time.perf_counter()
        future = _run_background_transfer(executor, stream, ready, device, transfer)
        future.result()
        torch.npu.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        dist.barrier()
        elapsed_ms = _global_max(elapsed_ms, device)
        validate()
        return elapsed_ms

    def migration_overlapped() -> float:
        reset()
        _barrier_and_synchronize(device)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(device))
        started = time.perf_counter()
        future = _run_background_transfer(executor, stream, ready, device, transfer)
        run_compute()
        future.result()
        torch.npu.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        dist.barrier()
        elapsed_ms = _global_max(elapsed_ms, device)
        validate()
        return elapsed_ms

    samples = {"isolated_ms": [], "foreground_ms": [], "overlapped_ms": [], "dilation_ms": []}
    try:
        for iteration in range(args.warmup + args.iterations):
            isolated_ms = migration_only()
            foreground_ms = foreground_only()
            overlapped_ms = migration_overlapped()
            if iteration >= args.warmup:
                samples["isolated_ms"].append(isolated_ms)
                samples["foreground_ms"].append(foreground_ms)
                samples["overlapped_ms"].append(overlapped_ms)
                samples["dilation_ms"].append(overlapped_ms - foreground_ms)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        manager.shutdown_pipeline()

    summaries = {name: _summarize(values) for name, values in samples.items()}
    hidden_ratio = _hidden_ratio(
        summaries["isolated_ms"]["median"],
        summaries["foreground_ms"]["median"],
        summaries["overlapped_ms"]["median"],
    )
    return {
        "payload_mib": payload_bytes / (1024.0 * 1024.0),
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "transport": "deterministic batched P2P per dtype",
        "samples": samples,
        "timing_ms": summaries,
        "hidden_ratio": hidden_ratio,
        "acceptance": {
            "hidden_ratio_at_least_target": hidden_ratio >= args.min_hidden_ratio,
            "median_dilation_within_budget": summaries["dilation_ms"]["median"] <= args.max_dilation_ms,
        },
    }


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive.")
    if args.compute_window_ms <= 0.0:
        raise ValueError("compute-window-ms must be positive.")
    if not 0.0 <= args.min_hidden_ratio <= 1.0:
        raise ValueError("min-hidden-ratio must be between zero and one.")

    importlib.import_module("torch_npu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    try:
        rank = dist.get_rank()
        ep_size = dist.get_world_size()
        if args.num_experts % ep_size:
            raise ValueError("num-experts must be divisible by EP size.")
        if args.group_sizes[-1] != ep_size:
            raise ValueError("The final hierarchy group size must equal EP size.")
        if ep_size % args.ranks_per_node:
            raise ValueError("EP size must be divisible by ranks-per-node.")
        hierarchy = Hierarchy(
            ep_size=ep_size,
            group_sizes=tuple(args.group_sizes),
            source="pipeline-overlap-benchmark",
            local_world_size=args.ranks_per_node,
        )
        if args.foreground_transport == "rank-dedup":
            configure_hiermoe(
                HierMoEConfig(
                    enable=True,
                    token_dedup=True,
                    expert_swap=False,
                    hierarchy_group_sizes=list(args.group_sizes),
                ),
                dist.group.WORLD,
            )
        run_compute, gemm_iterations, calibrated_compute_ms = _calibrate_gemm(
            args.compute_window_ms,
            args.hidden_size,
            args.moe_intermediate_size,
            device,
        )
        result: dict[str, Any] = {
            "metadata": {
                "ep_size": ep_size,
                "num_experts": args.num_experts,
                "hidden_size": args.hidden_size,
                "moe_intermediate_size": args.moe_intermediate_size,
                "group_sizes": list(args.group_sizes),
                "compute_window_target_ms": args.compute_window_ms,
                "compute_window_calibrated_ms": calibrated_compute_ms,
                "gemm_iterations": gemm_iterations,
                "foreground_a2a_mib_per_rank": args.foreground_a2a_mib,
                "foreground_transport": args.foreground_transport,
                "hidden_ratio_definition": "1 - max(0, overlap - foreground) / isolated_background",
            }
        }
        if args.mode in {"planner", "all"}:
            result["planner"] = _run_planner_benchmark(
                args,
                rank=rank,
                ep_size=ep_size,
                device=device,
                hierarchy=hierarchy,
                run_compute=run_compute,
            )
        if args.mode in {"swap", "all"}:
            result["swap"] = _run_migration_kind(
                "swap",
                args,
                rank=rank,
                ep_size=ep_size,
                device=device,
                hierarchy=hierarchy,
                run_compute=run_compute,
            )
        if args.mode in {"cover", "all"}:
            result["cover"] = _run_migration_kind(
                "cover",
                args,
                rank=rank,
                ep_size=ep_size,
                device=device,
                hierarchy=hierarchy,
                run_compute=run_compute,
            )

        acceptance = {
            f"{section}.{name}": bool(value)
            for section in ("planner", "swap", "cover")
            if section in result
            for name, value in result[section]["acceptance"].items()
        }
        result["acceptance"] = acceptance
        result["passed"] = all(acceptance.values())
        if rank == 0:
            encoded = json.dumps(result, indent=2, sort_keys=True)
            print(encoded, flush=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n", encoding="utf-8")
        failed = torch.tensor([int(not result["passed"])], dtype=torch.int32, device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if args.fail_on_threshold and bool(failed.item()):
            raise RuntimeError("One or more pipeline overlap acceptance thresholds failed.")
    finally:
        if dist.is_initialized():
            for manager in _MANAGER_KEEPALIVE:
                manager.destroy_pipeline_process_groups()
            dist.destroy_process_group()
        _MANAGER_KEEPALIVE.clear()


if __name__ == "__main__":
    main()
