# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark exact greedy swap/cover planning on saved per-rank routes."""

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

from veomni.distributed.moe.hiermoe.greedy_planner import (
    GREEDY_COMMUNICATION_PHASE_MULTIPLIER,
    GREEDY_COMPUTE_PHASE_MULTIPLIER,
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.statistical_scorer import statistical_candidate_local_deltas
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--layer-execution", choices=("sequential", "batched"), default="sequential")
    parser.add_argument("--layer-parallel-streams", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0, help="Route rank for a non-distributed run.")
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--slot-increment", type=int, default=1)
    parser.add_argument("--phase", choices=("steady", "initialize", "initialized-steady"), default="steady")
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--max-covers", type=int, default=1)
    parser.add_argument("--max-copies", type=int, default=8)
    parser.add_argument("--communication-scale", type=float, default=1.0)
    parser.add_argument("--forward-compute-per-assignment", type=float, default=0.0)
    parser.add_argument("--forward-compute-constant", type=float, default=0.0)
    parser.add_argument("--candidate-scorer", choices=("statistics", "reference"), default="statistics")
    parser.add_argument("--candidate-collective", choices=("compact", "full"), default="full")
    parser.add_argument("--adaptive-topk", action="store_true")
    parser.add_argument("--adaptive-topk-initial", type=int, default=16)
    parser.add_argument("--adaptive-topk-strict-certificate", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


def _initialize(backend: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "hccl":
        importlib.import_module("torch_npu")
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend=backend)
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, device


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _load_route(route_dir: Path, layer: int, rank: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or "routes" not in record:
        raise ValueError(f"{path} is not a VeOmni local-route snapshot.")
    routes = record["routes"].to(device=device, dtype=torch.long, non_blocking=True)
    return routes, record


def _initial_layout(
    *,
    num_experts: int,
    ep_size: int,
    slot_increment: int,
    phase: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size.")
    base = num_experts // ep_size
    slots_per_rank = base + slot_increment
    experts = torch.arange(num_experts, dtype=torch.long, device=device)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + torch.remainder(experts, base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, experts)
    if phase == "steady":
        ranks = torch.arange(ep_size, dtype=torch.long, device=device)
        for offset in range(slot_increment):
            replica = torch.remainder(ranks + offset + 1, ep_size) * base + (offset % base)
            layout[ranks * slots_per_rank + base + offset] = replica
    return layout, owners, slots_per_rank


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _plans_digest(plans) -> int:
    payload = [
        {
            "actions": [action.format() for action in plan.actions],
            "layout": list(plan.final_layout),
        }
        for plan in plans
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return int(digest[:15], 16)


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive.")
    if not 1 <= args.max_copies <= 8:
        raise ValueError("max-copies must be between 1 and 8.")
    if args.layer_count <= 0:
        raise ValueError("layer-count must be positive.")
    if (
        min(
            args.communication_scale,
            args.forward_compute_per_assignment,
            args.forward_compute_constant,
        )
        < 0.0
    ):
        raise ValueError("All cost-model coefficients must be non-negative.")
    rank, world_size, device = _initialize(args.backend)
    route_rank = rank if world_size > 1 else args.rank
    if world_size > 1 and world_size != args.ep_size:
        raise ValueError(f"Distributed world size {world_size} must equal ep_size {args.ep_size}.")
    layer_ids = tuple(range(args.layer, args.layer + args.layer_count))
    route_records = [_load_route(args.route_dir, layer, route_rank, device) for layer in layer_ids]
    routes, metadata = route_records[0]
    routes_by_layer = [value[0] for value in route_records]
    if any(int(value[1]["num_experts"]) != int(metadata["num_experts"]) for value in route_records):
        raise ValueError("All benchmark layers must use the same number of experts.")
    num_experts = int(metadata["num_experts"])
    layout, owners, slots_per_rank = _initial_layout(
        num_experts=num_experts,
        ep_size=args.ep_size,
        slot_increment=args.slot_increment,
        phase=args.phase,
        device=device,
    )

    def reducer(tensor: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=tuple(args.group_sizes),
            source="benchmark",
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
        process_group=dist.group.WORLD if dist.is_initialized() else None,
        max_copies=args.max_copies,
        candidate_scorer=args.candidate_scorer,
        compact_candidate_collective=args.candidate_collective == "compact",
        assume_unique_routes=True,
        layer_parallel_streams=args.layer_parallel_streams,
        adaptive_topk=args.adaptive_topk,
        adaptive_topk_initial=args.adaptive_topk_initial,
        adaptive_topk_strict_certificate=args.adaptive_topk_strict_certificate,
    )
    initialization_ms = None
    initialization_actions: list[str] = []
    if args.phase == "initialized-steady":
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        initialization = planner.plan(
            routes,
            layout,
            owners,
            source_ranks=route_rank,
            max_swaps=0,
            max_replicas=args.ep_size * args.slot_increment,
            step=0,
            layer_seed=args.layer,
        )
        _synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        initialization_ms = float(elapsed.item())
        initialization_actions = [action.format() for action in initialization.actions]
        layout = torch.tensor(initialization.final_layout, dtype=torch.long, device=device)
        if bool((layout < 0).any().item()):
            raise RuntimeError("Initialization did not fill every redundant slot.")
    all_slots = torch.arange(layout.numel(), dtype=torch.long, device=device)
    owner_mask = torch.zeros_like(layout, dtype=torch.bool)
    owner_mask.scatter_(0, owners, True)
    if args.phase == "initialize":
        destinations = torch.nonzero(layout < 0, as_tuple=False).flatten()
        candidate_rows = planner._cover_rows(layout, owners, destinations)
        max_covers = args.ep_size * args.slot_increment
        max_swaps = 0
    else:
        rows = []
        if args.max_swaps:
            rows.append(planner._swap_rows(layout, owners))
        if args.max_covers:
            destinations = all_slots[(~owner_mask) & (layout >= 0)]
            rows.append(planner._cover_rows(layout, owners, destinations))
        candidate_rows = torch.cat([row for row in rows if row.numel()], dim=0)
        max_covers = args.max_covers
        max_swaps = args.max_swaps
    _synchronize(device)
    candidate_count = int(candidate_rows.shape[0])

    verification = None
    if args.verify_reference:
        if args.layer_count != 1:
            raise ValueError("--verify-reference currently requires --layer-count=1.")
        if device.type != "npu":
            raise ValueError("--verify-reference requires the NPU reference kernel.")
        physical = assign_tokens_to_copies_greedy(
            routes,
            layout,
            slots_per_rank=slots_per_rank,
            source_ranks=route_rank,
            hierarchy_group_sizes=planner.hierarchy.group_sizes,
            num_experts=num_experts,
            step=1,
            layer_seed=args.layer,
            max_copies=args.max_copies,
        )
        occupancies = planner._token_level_occupancies(physical)
        copy_slots = planner._copy_table(layout, num_experts)
        source_tensor = torch.full((routes.shape[0],), route_rank, dtype=torch.long, device=device)
        token_ordinals = torch.arange(routes.shape[0], dtype=torch.long, device=device)
        exact_stats = statistical_candidate_local_deltas(
            planner,
            routes,
            candidate_rows,
            layout=layout,
            copy_slots=copy_slots,
            physical=physical,
            occupancies=occupancies,
            source_ranks=source_tensor,
            token_ordinals=token_ordinals,
            uniform_source_rank=route_rank,
            step=1,
            layer_seed=args.layer,
            num_experts=num_experts,
        )
        reference = planner._fused_candidate_local_deltas(
            routes,
            candidate_rows,
            layout=layout,
            copy_slots=copy_slots,
            physical=physical,
            occupancies=occupancies,
            source_ranks=source_tensor,
            token_ordinals=token_ordinals,
            step=1,
            layer_seed=args.layer,
            num_experts=num_experts,
        )
        available = torch.tensor(
            [int(exact_stats is not None and reference is not None)],
            dtype=torch.int32,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not bool(available.item()):
            raise RuntimeError("The statistical or reference scorer is unavailable on at least one rank.")
        assert exact_stats is not None and reference is not None
        difference = (exact_stats - reference).abs()
        mismatch_count_tensor = exact_stats.ne(reference).sum(dtype=torch.int64).view(1)
        max_abs_tensor = difference.max().view(1) if difference.numel() else difference.new_zeros(1)
        if dist.is_initialized():
            dist.all_reduce(mismatch_count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_abs_tensor, op=dist.ReduceOp.MAX)
        mismatch_count = int(mismatch_count_tensor.item())
        max_abs = float(max_abs_tensor.item())
        verification = {"mismatch_count": mismatch_count, "max_abs": max_abs}
        if mismatch_count:
            raise RuntimeError(f"Statistical scorer differs from reference: {verification}")

    samples: list[float] = []
    final_plans = None
    for iteration in range(args.warmup + args.iterations):
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        if args.layer_execution == "batched":
            plans = planner.plan_layers(
                routes_by_layer,
                [layout] * args.layer_count,
                [owners] * args.layer_count,
                source_ranks=route_rank,
                max_swaps=max_swaps,
                max_replicas=max_covers,
                layer_seeds=layer_ids,
                step=1,
                skip_final_route_update=True,
            )
        else:
            plans = [
                planner.plan(
                    layer_routes,
                    layout,
                    owners,
                    source_ranks=route_rank,
                    max_swaps=max_swaps,
                    max_replicas=max_covers,
                    step=1,
                    layer_seed=layer,
                )
                for layer, layer_routes in zip(layer_ids, routes_by_layer, strict=True)
            ]
        _synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            digest = torch.tensor([_plans_digest(plans)], dtype=torch.int64, device=device)
            gathered = torch.empty((world_size,), dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(gathered, digest)
            if bool((gathered != gathered[0]).any().item()):
                raise RuntimeError("Ranks selected different placement plans.")
        elapsed_ms = float(elapsed.item())
        if iteration >= args.warmup:
            samples.append(elapsed_ms)
        if args.verbose and rank == 0:
            print(json.dumps({"iteration": iteration, "elapsed_ms": elapsed_ms}))
        final_plans = plans

    assert final_plans is not None
    final_plan = final_plans[0]
    aggregate_planning_ms = sum(plan.planning_ms for plan in final_plans)
    aggregate_route_stats_ms = sum(plan.route_stats_ms for plan in final_plans)
    aggregate_score_ms = sum(plan.swap_score_ms + plan.replica_score_ms for plan in final_plans)
    aggregate_decision_sync_ms = sum(plan.decision_sync_ms for plan in final_plans)
    aggregate_finalization_ms = sum(plan.finalization_ms for plan in final_plans)
    formatted_actions = [
        f"layer{layer:02d}:{action.format()}"
        for layer, plan in zip(layer_ids, final_plans, strict=True)
        for action in plan.actions
    ]
    route_shapes = {tuple(value.shape) for value in routes_by_layer}
    result = {
        "route": {
            "layer": args.layer if args.layer_count == 1 else list(layer_ids),
            "layer_count": args.layer_count,
            "rank": route_rank if world_size == 1 else "distributed",
            "shape": list(routes.shape)
            if len(route_shapes) == 1
            else [list(value.shape) for value in routes_by_layer],
        },
        "world_size": world_size,
        "ep_size": args.ep_size,
        "phase": args.phase,
        "max_copies": args.max_copies,
        "candidate_scorer": args.candidate_scorer,
        "candidate_collective": args.candidate_collective,
        "adaptive_topk": args.adaptive_topk,
        "adaptive_topk_strict_certificate": args.adaptive_topk_strict_certificate,
        "adaptive_topk_stats": planner.last_adaptive_topk_stats,
        "layer_execution": args.layer_execution,
        "layer_parallel_streams": args.layer_parallel_streams,
        "cost_model": {
            "communication_phase_multiplier": GREEDY_COMMUNICATION_PHASE_MULTIPLIER,
            "compute_phase_multiplier": GREEDY_COMPUTE_PHASE_MULTIPLIER,
            "communication_scale": args.communication_scale,
            "forward_compute_per_assignment": args.forward_compute_per_assignment,
            "forward_compute_constant": args.forward_compute_constant,
        },
        "initialization_ms": initialization_ms,
        "initialization_actions": initialization_actions,
        "candidate_count": candidate_count,
        "reference_verification": verification,
        "plan_breakdown_ms": {
            "planning": aggregate_planning_ms,
            "route_stats": aggregate_route_stats_ms,
            "score": aggregate_score_ms,
            "decision_sync": aggregate_decision_sync_ms,
            "finalization": aggregate_finalization_ms,
        },
        "timing_ms": {
            "median": statistics.median(samples),
            "median_per_layer": statistics.median(samples) / args.layer_count,
            "p90": _percentile(samples, 0.9),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "actions": [action.format() for action in final_plan.actions] if args.layer_count == 1 else formatted_actions,
        "baseline_communication": sum(plan.baseline_cost.communication for plan in final_plans),
        "final_communication": sum(plan.final_cost.communication for plan in final_plans),
        "baseline_compute": sum(plan.baseline_cost.compute for plan in final_plans),
        "final_compute": sum(plan.final_cost.compute for plan in final_plans),
        "baseline_total": sum(plan.baseline_cost.total for plan in final_plans),
        "final_total": sum(plan.final_cost.total for plan in final_plans),
        "predicted_communication_speedup": (
            sum(plan.baseline_cost.communication for plan in final_plans)
            / sum(plan.final_cost.communication for plan in final_plans)
            if sum(plan.final_cost.communication for plan in final_plans) > 0
            else math.inf
        ),
        "predicted_total_speedup": (
            sum(plan.baseline_cost.total for plan in final_plans) / sum(plan.final_cost.total for plan in final_plans)
            if sum(plan.final_cost.total for plan in final_plans) > 0
            else math.inf
        ),
    }
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
