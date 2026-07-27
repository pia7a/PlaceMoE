# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark the exact CPU planner, layer-owner sharding, and double buffering."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.cpu_planner import (
    AsyncCPULayerOwnerPlanner,
    CPUExactPlanner,
    CPULayerOwnerPlanner,
    assert_exact_plan_match,
    resolve_cpu_planner_resources,
)
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single-align", "layer-owner", "async"), required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0, help="Saved route rank in single-align mode.")
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--slot-increment", type=int, default=4)
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--max-covers", type=int, default=1)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--communication-scale", type=float, default=1.0)
    parser.add_argument("--forward-compute-per-assignment", type=float, default=1.0)
    parser.add_argument("--forward-compute-constant", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu-cores-per-rank", type=int, default=0)
    parser.add_argument("--reserve-cpu-cores", type=int, default=2)
    parser.add_argument("--layer-workers", type=int, default=0)
    parser.add_argument("--intraop-threads", type=int, default=0)
    parser.add_argument("--owner-offset", type=int, default=0)
    parser.add_argument("--compare-full-exact", action="store_true")
    parser.add_argument(
        "--npu-reference",
        action="store_true",
        help="In single-align mode, execute the current NPU full-exact planner and require bit equality.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _initialize_gloo(mode: str, ep_size: int) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if mode == "single-align":
        if world_size != 1:
            raise ValueError("single-align mode must run as one process.")
        return 0, 1
    if world_size != ep_size:
        raise ValueError(f"{mode} requires WORLD_SIZE={ep_size}, got {world_size}.")
    dist.init_process_group(backend="gloo")
    return int(dist.get_rank()), int(dist.get_world_size())


def _route_path(route_dir: Path, layer: int, rank: int) -> Path:
    direct = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
    if direct.exists():
        return direct
    matches = sorted(route_dir.glob(f"layer{layer:02d}_call*_rank{rank:02d}.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No route snapshot for layer={layer}, rank={rank} under {route_dir}.")


def _load_routes(route_dir: Path, layers: tuple[int, ...], rank: int) -> tuple[list[torch.Tensor], dict]:
    routes = []
    first_record = None
    for layer in layers:
        path = _route_path(route_dir, layer, rank)
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, dict) or "routes" not in record:
            raise ValueError(f"{path} is not a VeOmni route snapshot.")
        if first_record is None:
            first_record = record
        routes.append(record["routes"].to(device="cpu", dtype=torch.long).contiguous())
    assert first_record is not None
    return routes, first_record


def _steady_layout(
    *,
    num_experts: int,
    ep_size: int,
    slot_increment: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size.")
    base = num_experts // ep_size
    slots_per_rank = base + slot_increment
    experts = torch.arange(num_experts, dtype=torch.long)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + torch.remainder(experts, base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long)
    layout.scatter_(0, owners, experts)
    ranks = torch.arange(ep_size, dtype=torch.long)
    for offset in range(slot_increment):
        replica = torch.remainder(ranks + offset + 1, ep_size) * base + (offset % base)
        layout[ranks * slots_per_rank + base + offset] = replica
    return layout, owners, slots_per_rank


def _planner(
    args: argparse.Namespace,
    metadata: dict,
    slots_per_rank: int,
    *,
    device: torch.device,
    reducer=None,
) -> GreedyCommunicationPlanner:
    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=tuple(args.group_sizes),
            source="cpu-planner-benchmark",
            local_world_size=args.local_world_size,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=int(metadata["hidden_size"]),
        bytes_per_element=int(metadata["bytes_per_element"]),
        slots_per_rank=slots_per_rank,
        communication_scale=args.communication_scale,
        forward_compute_per_assignment=args.forward_compute_per_assignment,
        forward_compute_constant=args.forward_compute_constant,
        reducer=reducer,
        process_group=dist.group.WORLD if dist.is_initialized() and device.type == "cpu" else None,
        max_copies=args.max_copies,
        candidate_scorer="statistics",
        assume_unique_routes=True,
        adaptive_topk=False,
        early_proxy_topk=0,
        exact_primitive_topk=0,
    )


def _plan_kwargs(args: argparse.Namespace, layers: tuple[int, ...], route_rank: int) -> dict:
    return {
        "source_ranks": route_rank,
        "max_swaps": args.max_swaps,
        "max_replicas": args.max_covers,
        "layer_seeds": layers,
        "step": 1,
        "communication_scales": [args.communication_scale] * len(layers),
        "forward_compute_per_assignment": [args.forward_compute_per_assignment] * len(layers),
        "forward_compute_constant": [args.forward_compute_constant] * len(layers),
    }


def _action_rows(plans) -> list[str]:
    return [
        f"layer{layer_index}:{','.join(action.format() for action in plan.actions) or 'none'}"
        for layer_index, plan in enumerate(plans)
    ]


def _cost_rows(plans) -> list[dict[str, float]]:
    return [
        {
            "baseline_communication": plan.baseline_cost.communication,
            "baseline_compute": plan.baseline_cost.compute,
            "baseline_total": plan.baseline_cost.total,
            "final_communication": plan.final_cost.communication,
            "final_compute": plan.final_cost.compute,
            "final_total": plan.final_cost.total,
        }
        for plan in plans
    ]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _single_align(
    args: argparse.Namespace,
    routes: list[torch.Tensor],
    layout: torch.Tensor,
    owners: torch.Tensor,
    metadata: dict,
    slots_per_rank: int,
    layers: tuple[int, ...],
) -> dict:
    resources = resolve_cpu_planner_resources(
        layer_count=len(layers),
        local_process_count=1,
        cpu_cores_per_rank=args.cpu_cores_per_rank or None,
        reserve_cpu_cores=args.reserve_cpu_cores,
        layer_workers=1,
        intraop_threads=args.intraop_threads or None,
    )
    torch.set_num_threads(resources.intraop_threads)
    planner = _planner(args, metadata, slots_per_rank, device=torch.device("cpu"))
    backend = CPUExactPlanner(planner)
    kwargs = _plan_kwargs(args, layers, args.rank)
    samples = []
    plans = None
    for iteration in range(args.warmup + args.iterations):
        started = time.perf_counter()
        current = backend.plan_layers(routes, [layout] * len(layers), [owners] * len(layers), **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= args.warmup:
            samples.append(elapsed_ms)
        plans = current
    assert plans is not None

    alignment = None
    npu_ms = None
    if args.npu_reference:
        importlib.import_module("torch_npu")
        torch.npu.set_device(0)
        device = torch.device("npu:0")
        npu_routes = [value.to(device=device) for value in routes]
        npu_layout = layout.to(device=device)
        npu_owners = owners.to(device=device)
        npu_planner = _planner(args, metadata, slots_per_rank, device=device)
        torch.npu.synchronize()
        started = time.perf_counter()
        reference = npu_planner.plan_layers(
            npu_routes,
            [npu_layout] * len(layers),
            [npu_owners] * len(layers),
            skip_final_route_update=True,
            **kwargs,
        )
        torch.npu.synchronize()
        npu_ms = (time.perf_counter() - started) * 1000.0
        for expected, actual in zip(reference, plans, strict=True):
            assert_exact_plan_match(expected, actual)
        alignment = {
            "bit_exact": True,
            "action_matches": len(plans),
            "cost_matches": len(plans),
        }
    return {
        "mode": args.mode,
        "cpu_timing_ms": {
            "median": statistics.median(samples),
            "median_per_layer": statistics.median(samples) / len(layers),
            "p90": _percentile(samples, 0.9),
            "samples": samples,
        },
        "npu_reference_ms": npu_ms,
        "alignment": alignment,
        "resources": vars(resources),
        "actions": _action_rows(plans),
        "costs": _cost_rows(plans),
    }


def _max_timing(result, world_size: int) -> dict[str, float]:
    names = (
        "context_ms",
        "local_prepare_ms",
        "statistic_pack_ms",
        "statistic_collective_ms",
        "owner_score_ms",
        "decision_collective_ms",
        "finalization_ms",
        "total_ms",
    )
    values = torch.tensor([float(getattr(result.timing, name)) for name in names], dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return dict(zip(names, values.tolist(), strict=True))


def _make_owner_backend(
    args: argparse.Namespace,
    metadata: dict,
    slots_per_rank: int,
    layer_count: int,
):
    resources = resolve_cpu_planner_resources(
        layer_count=layer_count,
        local_process_count=args.local_world_size,
        cpu_cores_per_rank=args.cpu_cores_per_rank or None,
        reserve_cpu_cores=args.reserve_cpu_cores,
        layer_workers=args.layer_workers or None,
        intraop_threads=args.intraop_threads or None,
    )
    planner = _planner(args, metadata, slots_per_rank, device=torch.device("cpu"))
    backend = CPULayerOwnerPlanner(
        planner,
        process_group=dist.group.WORLD,
        local_process_count=args.local_world_size,
        resources=resources,
    )
    return backend, resources


def _compare_distributed_full_exact(
    args: argparse.Namespace,
    routes: list[torch.Tensor],
    layout: torch.Tensor,
    owners: torch.Tensor,
    metadata: dict,
    slots_per_rank: int,
    layers: tuple[int, ...],
    route_rank: int,
    owner_plans,
) -> dict | None:
    if not args.compare_full_exact:
        return None

    def reducer(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    reference_planner = _planner(
        args,
        metadata,
        slots_per_rank,
        device=torch.device("cpu"),
        reducer=reducer,
    )
    started = time.perf_counter()
    reference = reference_planner.plan_layers(
        routes,
        [layout] * len(layers),
        [owners] * len(layers),
        skip_final_route_update=True,
        **_plan_kwargs(args, layers, route_rank),
    )
    elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    for expected, actual in zip(reference, owner_plans, strict=True):
        assert_exact_plan_match(expected, actual)
    return {"bit_exact": True, "elapsed_ms": float(elapsed.item())}


def _layer_owner(
    args: argparse.Namespace,
    routes: list[torch.Tensor],
    layout: torch.Tensor,
    owners: torch.Tensor,
    metadata: dict,
    slots_per_rank: int,
    layers: tuple[int, ...],
    rank: int,
    world_size: int,
) -> dict:
    backend, resources = _make_owner_backend(args, metadata, slots_per_rank, len(layers))
    kwargs = _plan_kwargs(args, layers, rank)
    samples = []
    breakdowns = []
    result = None
    for iteration in range(args.warmup + args.iterations):
        dist.barrier()
        current = backend.plan_layers(
            routes,
            [layout] * len(layers),
            [owners] * len(layers),
            owner_offset=args.owner_offset,
            **kwargs,
        )
        maximum = _max_timing(current, world_size)
        if iteration >= args.warmup:
            samples.append(maximum["total_ms"])
            breakdowns.append(maximum)
        result = current
    assert result is not None
    comparison = _compare_distributed_full_exact(
        args,
        routes,
        layout,
        owners,
        metadata,
        slots_per_rank,
        layers,
        rank,
        result.plans,
    )
    median_breakdown = {
        name: statistics.median([sample[name] for sample in breakdowns]) for name in breakdowns[0]
    }
    return {
        "mode": args.mode,
        "timing_ms": {
            "median": statistics.median(samples),
            "median_per_layer": statistics.median(samples) / len(layers),
            "p90": _percentile(samples, 0.9),
            "samples": samples,
            "median_breakdown": median_breakdown,
        },
        "resources": vars(resources),
        "owner_ranks": list(result.owner_ranks),
        "owned_layers_this_rank": result.timing.owned_layer_count,
        "local_payload_bytes": result.timing.local_payload_bytes,
        "received_payload_bytes": result.timing.received_payload_bytes,
        "full_exact_alignment": comparison,
        "actions": _action_rows(result.plans),
        "costs": _cost_rows(result.plans),
    }


def _async(
    args: argparse.Namespace,
    routes: list[torch.Tensor],
    layout: torch.Tensor,
    owners: torch.Tensor,
    metadata: dict,
    slots_per_rank: int,
    layers: tuple[int, ...],
    rank: int,
    world_size: int,
) -> dict:
    backend, resources = _make_owner_backend(args, metadata, slots_per_rank, len(layers))
    kwargs = _plan_kwargs(args, layers, rank)
    kwargs.pop("step")
    versions = [0] * len(layers)
    submit_samples = []
    completion_samples = []
    final = None
    asynchronous = AsyncCPULayerOwnerPlanner(backend)
    try:
        dist.barrier()
        total_iterations = args.warmup + args.iterations
        submitted = 0
        completed = 0
        submission_times: deque[float] = deque()
        pipeline_started = time.perf_counter()
        while submitted < min(2, total_iterations):
            started = time.perf_counter()
            accepted = asynchronous.submit(
                submitted,
                routes,
                [layout] * len(layers),
                [owners] * len(layers),
                placement_versions=versions,
                owner_offset=args.owner_offset,
                **kwargs,
            )
            submit_ms = (time.perf_counter() - started) * 1000.0
            if not accepted:
                raise RuntimeError("The initial double-buffer submission was unexpectedly rejected.")
            submission_times.append(started)
            if submitted >= args.warmup:
                submit_samples.append(submit_ms)
            submitted += 1

        while completed < total_iterations:
            current = asynchronous.wait_next()
            if current is None:
                raise RuntimeError("The asynchronous planner lost a submitted result.")
            completion_ms = (time.perf_counter() - submission_times.popleft()) * 1000.0
            if completed >= args.warmup:
                completion_samples.append(completion_ms)
            final = current
            completed += 1
            if submitted < total_iterations:
                started = time.perf_counter()
                accepted = asynchronous.submit(
                    submitted,
                    routes,
                    [layout] * len(layers),
                    [owners] * len(layers),
                    placement_versions=versions,
                    owner_offset=args.owner_offset,
                    **kwargs,
                )
                submit_ms = (time.perf_counter() - started) * 1000.0
                if not accepted:
                    raise RuntimeError("A refill double-buffer submission was unexpectedly rejected.")
                submission_times.append(started)
                if submitted >= args.warmup:
                    submit_samples.append(submit_ms)
                submitted += 1
        pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
    finally:
        asynchronous.close()
    assert final is not None
    maxima = torch.tensor(
        [
            statistics.median(submit_samples),
            _percentile(submit_samples, 0.9),
            statistics.median(completion_samples),
            _percentile(completion_samples, 0.9),
            pipeline_ms,
        ],
        dtype=torch.float64,
    )
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return {
        "mode": args.mode,
        "submit_timing_ms": {
            "median": float(maxima[0].item()),
            "p90": float(maxima[1].item()),
        },
        "completion_latency_ms": {
            "median": float(maxima[2].item()),
            "p90": float(maxima[3].item()),
        },
        "double_buffer_pipeline_ms": float(maxima[4].item()),
        "double_buffer_throughput_ms": float(maxima[4].item()) / total_iterations,
        "planner_breakdown_ms": _max_timing(final, world_size),
        "resources": vars(resources),
        "owner_ranks": list(final.owner_ranks),
        "actions": _action_rows(final.plans),
        "costs": _cost_rows(final.plans),
    }


def main() -> None:
    args = _parse_args()
    if args.layer_count <= 0 or args.iterations <= 0 or args.warmup < 0:
        raise ValueError("layer-count and iterations must be positive; warmup must be non-negative.")
    rank, world_size = _initialize_gloo(args.mode, args.ep_size)
    route_rank = args.rank if args.mode == "single-align" else rank
    layers = tuple(range(args.layer, args.layer + args.layer_count))
    routes, metadata = _load_routes(args.route_dir, layers, route_rank)
    layout, owners, slots_per_rank = _steady_layout(
        num_experts=int(metadata["num_experts"]),
        ep_size=args.ep_size,
        slot_increment=args.slot_increment,
    )
    if args.mode == "single-align":
        result = _single_align(args, routes, layout, owners, metadata, slots_per_rank, layers)
    elif args.mode == "layer-owner":
        result = _layer_owner(
            args,
            routes,
            layout,
            owners,
            metadata,
            slots_per_rank,
            layers,
            rank,
            world_size,
        )
    else:
        result = _async(
            args,
            routes,
            layout,
            owners,
            metadata,
            slots_per_rank,
            layers,
            rank,
            world_size,
        )
    result.update(
        {
            "ep_size": args.ep_size,
            "world_size": world_size,
            "layer_ids": list(layers),
            "route_rank": route_rank if args.mode == "single-align" else "distributed",
        }
    )
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
