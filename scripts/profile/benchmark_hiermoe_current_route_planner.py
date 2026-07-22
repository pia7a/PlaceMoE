# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark the distributed current-route HierMoE planner on a saved route snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.core_planner import CoReMoEPlanner
from veomni.distributed.moe.hiermoe.oracle import load_route_snapshot
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import CurrentRoutePlanner
from veomni.distributed.moe.hiermoe.topology import Hierarchy


_CONFIGS = {
    "P1S0": (1, 0),
    "P4S0": (4, 0),
    "P0S1-auto": (0, 1),
    "P1S1-auto": (1, 1),
    "P4S1-auto": (4, 1),
}


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=repo_root / "tests/data/hiermoe_current_route_ep16_layer24_golden.json",
    )
    parser.add_argument("--configs", nargs="+", choices=tuple(_CONFIGS), default=list(_CONFIGS))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--planner", choices=("legacy", "core"), default="legacy")
    parser.add_argument("--route-sample-size", type=int, default=1024)
    parser.add_argument(
        "--stress", action="store_true", help="Also benchmark a top-k 8 route accepting near-capacity replicas."
    )
    parser.add_argument(
        "--stress-only", action="store_true", help="Skip snapshot configs and run only the stress route."
    )
    parser.add_argument("--stress-tokens", type=int, default=16384)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument(
        "--device-timing",
        action="store_true",
        help="Record synchronized accelerator-event stage timings for diagnosis.",
    )
    return parser.parse_args()


def _initialize(backend: str) -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "hccl":
        importlib.import_module("torch_npu")
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), device


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _owner_layout(
    logical_to_physical: torch.Tensor, ep_size: int, slots_per_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = int(logical_to_physical.numel())
    base_experts_per_rank = num_experts // ep_size
    compact = logical_to_physical.to(torch.long)
    owner_ranks = torch.div(compact, base_experts_per_rank, rounding_mode="floor")
    owner_local_slots = torch.remainder(compact, base_experts_per_rank)
    owner_slots = owner_ranks * slots_per_rank + owner_local_slots
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long)
    layout.scatter_(0, owner_slots, torch.arange(num_experts, dtype=torch.long))
    return layout, owner_slots


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in samples)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[p90_index],
        "max_ms": ordered[-1],
        "median_48_layers_ms": statistics.median(ordered) * 48.0,
        "p90_48_layers_ms": ordered[p90_index] * 48.0,
    }


def _speedup(baseline: float, current: float) -> float:
    if current > 0.0:
        return baseline / current
    return 1.0 if baseline <= 0.0 else math.inf


def _plan_sha256(plan) -> str:
    payload = {
        "actions": [action.format() for action in plan.actions],
        "final_layout": list(plan.final_layout),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _benchmark_config(
    *,
    routes: torch.Tensor,
    source_rank: int,
    hierarchy: Hierarchy,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    logical_to_physical: torch.Tensor,
    max_swaps: int,
    slots_per_rank_increment: int,
    communication_scale: float,
    compute_scale: float,
    step: int,
    layer_seed: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    record_device_timing: bool,
    planner_kind: str,
    route_sample_size: int,
) -> dict[str, object]:
    ep_size = hierarchy.ep_size
    slots_per_rank = num_experts // ep_size + slots_per_rank_increment
    layout, owners = _owner_layout(logical_to_physical, ep_size, slots_per_rank)
    layout = layout.to(device)
    owners = owners.to(device)
    routes = routes.to(device)

    def reducer(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def gather_fixed(tensor: torch.Tensor) -> torch.Tensor:
        gathered = torch.empty((ep_size, tensor.numel()), dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(gathered, tensor)
        return gathered

    if planner_kind == "core":
        planner = CoReMoEPlanner(
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            slots_per_rank=slots_per_rank,
            communication_scale=communication_scale,
            forward_compute_per_assignment=compute_scale,
            reducer=reducer,
            gather_fixed=gather_fixed,
            route_sample_size=route_sample_size,
            record_device_timing=record_device_timing,
        )
    else:
        planner = CurrentRoutePlanner(
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            slots_per_rank=slots_per_rank,
            communication_scale=communication_scale,
            forward_compute_per_assignment=compute_scale,
            reducer=reducer,
            record_device_timing=record_device_timing,
        )
    max_replicas = slots_per_rank_increment * ep_size
    metric_names = (
        "planning_ms",
        "route_stats_ms",
        "swap_score_ms",
        "swap_update_ms",
        "swap_collective_ms",
        "replica_score_ms",
        "replica_update_ms",
        "replica_collective_ms",
        "decision_sync_ms",
        "finalization_ms",
    )
    device_metric_names = (
        "planning",
        "route_stats",
        "route_collective",
        "swap_score",
        "swap_update",
        "swap_collective",
        "replica_init",
        "replica_score",
        "replica_update",
        "replica_collective",
    )
    samples = {
        name: []
        for name in (
            "external_ms",
            *metric_names,
            *(f"device_{name}_ms" for name in device_metric_names if record_device_timing),
        )
    }
    final_plan = None
    peak_memory_bytes = 0
    for iteration in range(warmup + iterations):
        dist.barrier()
        _synchronize(device)
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats(device)
        started = time.perf_counter()
        plan = planner.plan(
            routes,
            layout,
            owners,
            source_ranks=source_rank,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
            step=step,
            layer_seed=layer_seed,
        )
        _synchronize(device)
        external_ms = (time.perf_counter() - started) * 1000.0
        if device.type == "npu":
            peak_memory_bytes = max(peak_memory_bytes, int(torch.npu.max_memory_allocated(device)))
        device_timings = plan.device_timing_ms or {}
        values = torch.tensor(
            [
                external_ms,
                *(float(getattr(plan, name)) for name in metric_names),
                *(float(device_timings.get(name, 0.0)) for name in device_metric_names if record_device_timing),
            ],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        digest = torch.tensor([int(_plan_sha256(plan)[:15], 16)], dtype=torch.int64, device=device)
        gathered = torch.empty((ep_size,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(gathered, digest)
        if bool((gathered != gathered[0]).any().item()):
            raise RuntimeError("Distributed planner benchmark produced inconsistent plans.")
        costs = torch.tensor(
            [plan.final_cost.communication, plan.final_cost.compute, plan.final_cost.total],
            dtype=torch.float32,
            device=device,
        )
        minimum_costs = costs.clone()
        maximum_costs = costs.clone()
        dist.all_reduce(minimum_costs, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum_costs, op=dist.ReduceOp.MAX)
        if not bool(torch.allclose(minimum_costs, maximum_costs, rtol=0.0, atol=1e-4)):
            raise RuntimeError("Distributed planner benchmark produced inconsistent costs.")
        if iteration >= warmup:
            for name, value in zip(samples, values.cpu().tolist(), strict=True):
                samples[name].append(float(value))
        final_plan = plan

    assert final_plan is not None
    peak_memory = torch.tensor([peak_memory_bytes], dtype=torch.int64, device=device)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    peak_memory_bytes = int(peak_memory.item())
    baseline_total = final_plan.baseline_cost.total
    final_total = final_plan.final_cost.total
    return {
        "swap_budget": max_swaps,
        "slot_increment_per_rank": slots_per_rank_increment,
        "effective_replica_rounds": max_replicas,
        "accepted_swaps": final_plan.swap_rounds,
        "accepted_replicas": final_plan.replica_rounds,
        "actions": [action.format() for action in final_plan.actions],
        "final_layout": list(final_plan.final_layout),
        "plan_sha256": _plan_sha256(final_plan),
        "peak_memory_bytes": peak_memory_bytes,
        "baseline": {
            "communication_ms": final_plan.baseline_cost.communication,
            "compute_ms": final_plan.baseline_cost.compute,
            "total_ms": baseline_total,
        },
        "final": {
            "communication_ms": final_plan.final_cost.communication,
            "compute_ms": final_plan.final_cost.compute,
            "total_ms": final_total,
            "communication_speedup": _speedup(
                final_plan.baseline_cost.communication, final_plan.final_cost.communication
            ),
            "compute_speedup": _speedup(final_plan.baseline_cost.compute, final_plan.final_cost.compute),
            "total_speedup": _speedup(baseline_total, final_total),
        },
        "timing": {name: _summary(values) for name, values in samples.items()},
    }


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive.")
    rank, world_size, device = _initialize(args.backend)
    snapshot = load_route_snapshot(args.snapshot)
    if snapshot.ep_size != world_size:
        raise ValueError(f"Snapshot EP size {snapshot.ep_size} does not match distributed world size {world_size}.")
    calibration = json.loads(args.golden.read_text(encoding="utf-8"))["calibration"]
    configs: dict[str, object] = {}
    for name in () if args.stress_only else args.configs:
        max_swaps, slot_increment = _CONFIGS[name]
        configs[name] = _benchmark_config(
            routes=snapshot.routes_by_rank[rank],
            source_rank=rank,
            hierarchy=snapshot.hierarchy,
            num_experts=snapshot.num_experts,
            hidden_size=snapshot.hidden_size,
            bytes_per_element=snapshot.bytes_per_element,
            logical_to_physical=snapshot.logical_to_physical,
            max_swaps=max_swaps,
            slots_per_rank_increment=slot_increment,
            communication_scale=float(calibration["communication_scale"]),
            compute_scale=float(calibration["forward_compute_per_assignment"]),
            step=snapshot.step,
            layer_seed=24,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            record_device_timing=args.device_timing,
            planner_kind=args.planner,
            route_sample_size=args.route_sample_size,
        )
    if args.stress or args.stress_only:
        local_experts = torch.arange(rank * 8, rank * 8 + 8, dtype=torch.long)
        stress_top_k = (
            local_experts if rank == 0 else torch.cat((torch.zeros((1,), dtype=torch.long), local_experts[:7]))
        )
        stress_routes = stress_top_k.view(1, 8).expand(args.stress_tokens, -1).clone()
        configs["P0S1-auto-stress"] = _benchmark_config(
            routes=stress_routes,
            source_rank=rank,
            hierarchy=Hierarchy(ep_size=world_size, group_sizes=(1, world_size), source="stress"),
            num_experts=128,
            hidden_size=snapshot.hidden_size,
            bytes_per_element=snapshot.bytes_per_element,
            logical_to_physical=torch.arange(128, dtype=torch.long),
            max_swaps=0,
            slots_per_rank_increment=1,
            communication_scale=0.0,
            compute_scale=float(calibration["forward_compute_per_assignment"]),
            step=0,
            layer_seed=24,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            record_device_timing=args.device_timing,
            planner_kind=args.planner,
            route_sample_size=args.route_sample_size,
        )
    output = {
        "snapshot": str(args.snapshot),
        "world_size": world_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "planner": args.planner,
        "route_sample_size": args.route_sample_size,
        "configs": configs,
    }
    if rank == 0:
        encoded = json.dumps(output, indent=2, sort_keys=True)
        print(encoded, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
