# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Evaluate max-bottleneck swap/cover candidate filtering on saved EP routes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe import statistical_scorer
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GreedyCommunicationPlanner,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.statistical_scorer import _canonical_route_mask
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=(0,))
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--slot-increment", type=int, default=1)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--communication-scale", type=float, default=1.0)
    parser.add_argument("--forward-compute-per-assignment", type=float, default=1.0)
    parser.add_argument("--forward-compute-constant", type=float, default=0.0)
    parser.add_argument("--m-values", type=int, nargs="+", default=(4, 8, 12, 16, 24, 32))
    parser.add_argument("--d-values", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument("--unary-k-values", type=int, nargs="+", default=(16, 32, 64, 128, 256, 512, 1024))
    parser.add_argument("--timed-m", type=int, default=8)
    parser.add_argument("--timed-d", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--output", type=Path)
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


def _reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _max_elapsed_ms(started: float, device: torch.device) -> float:
    _synchronize(device)
    value = torch.tensor([(time.perf_counter() - started) * 1000.0], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _load_route(route_dir: Path, layer: int, rank: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or "routes" not in record:
        raise ValueError(f"{path} is not a VeOmni local-route snapshot.")
    return record["routes"].to(device=device, dtype=torch.long, non_blocking=True), record


def _initial_layout(
    *,
    num_experts: int,
    ep_size: int,
    slot_increment: int,
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
    ranks = torch.arange(ep_size, dtype=torch.long, device=device)
    for offset in range(slot_increment):
        replica = torch.remainder(ranks + offset + 1, ep_size) * base + (offset % base)
        layout[ranks * slots_per_rank + base + offset] = replica
    return layout, owners, slots_per_rank


def _planner(
    args: argparse.Namespace,
    *,
    hidden_size: int,
    bytes_per_element: int,
    slots_per_rank: int,
    candidate_scorer: str,
) -> GreedyCommunicationPlanner:
    def reducer(tensor: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=tuple(args.group_sizes),
            source="bottleneck-benchmark",
            local_world_size=args.local_world_size,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        slots_per_rank=slots_per_rank,
        communication_scale=args.communication_scale,
        forward_compute_per_assignment=args.forward_compute_per_assignment,
        forward_compute_constant=args.forward_compute_constant,
        reducer=reducer,
        process_group=dist.group.WORLD if dist.is_initialized() else None,
        max_copies=args.max_copies,
        candidate_scorer=candidate_scorer,
        compact_candidate_collective=False,
        assume_unique_routes=True,
    )


def _all_candidate_rows(
    planner: GreedyCommunicationPlanner,
    layout: torch.Tensor,
    owners: torch.Tensor,
) -> torch.Tensor:
    all_slots = torch.arange(layout.numel(), dtype=torch.long, device=layout.device)
    owner_mask = torch.zeros_like(layout, dtype=torch.bool)
    owner_mask.scatter_(0, owners, True)
    cover_slots = all_slots[(~owner_mask) & (layout >= 0)]
    return torch.cat(
        (
            planner._swap_rows(layout, owners),
            planner._cover_rows(layout, owners, cover_slots),
        ),
        dim=0,
    )


def _critical_expert_scores(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    num_experts = int(selected.max().item()) + 1
    packed_local = planner._local_packed_counts(physical)
    assignment_local = planner._local_assignment_counts(physical)
    combined = _reduce_sum(torch.cat((packed_local, assignment_local), dim=1))
    packed_global = combined[:, : packed_local.shape[1]]
    assignment_global = combined[:, packed_local.shape[1] :]

    widths = planner._count_widths()
    global_levels = packed_global.split(widths, dim=1)
    critical_levels = [counts.eq(counts.max(dim=1, keepdim=True).values)[0] for counts in global_levels]
    critical_compute = assignment_global.eq(assignment_global.max(dim=1, keepdim=True).values)[0]

    route_ranks = torch.div(physical, planner.slots_per_rank, rounding_mode="floor")
    occupancies = planner._token_level_occupancies(physical)
    unique = _canonical_route_mask(selected)
    level_sizes = (1,) + tuple(
        int(size) for size in planner.hierarchy.group_sizes[: max(0, planner.hierarchy.selected_dim - 1)]
    )
    communication_scores = torch.zeros((num_experts,), dtype=torch.float32, device=selected.device)
    for size, occupancy, critical in zip(level_sizes, occupancies, critical_levels, strict=True):
        groups = torch.div(route_ranks, size, rounding_mode="floor")
        sole = occupancy.gather(1, groups).eq(1)
        values = (sole & unique & critical.index_select(0, groups.reshape(-1)).view_as(groups)).to(torch.float32)
        communication_scores.scatter_add_(0, selected.reshape(-1), values.reshape(-1))

    compute_scores = torch.zeros_like(communication_scores)
    compute_values = critical_compute.index_select(0, route_ranks.reshape(-1)).to(torch.float32)
    compute_scores.scatter_add_(0, selected.reshape(-1), compute_values)
    scores = _reduce_sum(torch.stack((communication_scores, compute_scores), dim=0))

    rank_counts = global_levels[0][0]
    communication_load = rank_counts / rank_counts.max().clamp_min(1.0)
    offset = widths[0]
    for level_index, size in enumerate(level_sizes[1:], start=1):
        group_counts = packed_global[0, offset : offset + widths[level_index]]
        communication_load += group_counts.index_select(
            0,
            torch.div(torch.arange(planner.ep_size, device=selected.device), size, rounding_mode="floor"),
        ) / group_counts.max().clamp_min(1.0)
        offset += widths[level_index]
    compute_load = assignment_global[0] / assignment_global[0].max().clamp_min(1.0)
    destination_load = communication_load + (compute_load if planner.forward_compute_per_assignment > 0.0 else 0.0)
    details = {
        "critical_rank_count": int(critical_levels[0].sum().item()),
        "critical_group_counts": [int(value.sum().item()) for value in critical_levels[1:]],
        "critical_compute_rank_count": int(critical_compute.sum().item()),
        "max_rank_count": float(rank_counts.max().item()),
        "max_assignment_count": float(assignment_global.max().item()),
    }
    return scores[0], scores[1], destination_load, packed_global, details


def _candidate_mask(
    rows: torch.Tensor,
    owners: torch.Tensor,
    *,
    communication_scores: torch.Tensor,
    compute_scores: torch.Tensor,
    destination_load: torch.Tensor,
    source_count: int,
    destination_count: int,
    slots_per_rank: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    source_count = min(max(1, int(source_count)), communication_scores.numel())
    destination_count = min(max(1, int(destination_count)), destination_load.numel())
    communication_experts = communication_scores.topk(source_count).indices
    compute_experts = compute_scores.topk(source_count).indices
    source_mask = torch.zeros_like(communication_scores, dtype=torch.bool)
    source_mask[communication_experts] = True
    source_mask[compute_experts] = True
    destination_ranks = destination_load.topk(destination_count, largest=False).indices
    destination_mask = torch.zeros_like(destination_load, dtype=torch.bool)
    destination_mask[destination_ranks] = True

    is_swap = rows[:, 0] == 0
    lhs = rows[:, 3]
    rhs = rows[:, 4].clamp_min(0)
    lhs_owner_ranks = torch.div(owners.index_select(0, lhs), slots_per_rank, rounding_mode="floor")
    rhs_owner_ranks = torch.div(owners.index_select(0, rhs), slots_per_rank, rounding_mode="floor")
    destination_slot_ranks = torch.div(rows[:, 2], slots_per_rank, rounding_mode="floor")
    swap_keep = (source_mask.index_select(0, lhs) & destination_mask.index_select(0, rhs_owner_ranks)) | (
        source_mask.index_select(0, rhs) & destination_mask.index_select(0, lhs_owner_ranks)
    )
    cover_keep = source_mask.index_select(0, lhs) & destination_mask.index_select(0, destination_slot_ranks)
    keep = torch.where(is_swap, swap_keep, cover_keep)
    details = {
        "source_experts": int(source_mask.sum().item()),
        "destination_ranks": int(destination_mask.sum().item()),
        "swap_candidates": int((keep & is_swap).sum().item()),
        "cover_candidates": int((keep & ~is_swap).sum().item()),
    }
    return keep, details


def _score_rows(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    layout: torch.Tensor,
    owners: torch.Tensor,
    rows: torch.Tensor,
    *,
    route_rank: int,
    layer: int,
):
    sources = torch.full((selected.shape[0],), route_rank, dtype=torch.long, device=selected.device)
    ordinals = torch.arange(selected.shape[0], dtype=torch.long, device=selected.device)
    return planner._score_actions(
        selected,
        layout,
        rows,
        source_ranks=sources,
        uniform_source_rank=route_rank,
        copy_slots=planner._copy_table(layout, int(owners.numel())),
        affected_groups=None,
        token_ordinals=ordinals,
        step=1,
        layer_seed=layer,
        num_experts=int(owners.numel()),
    )


def _timed_filtered_score(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    layout: torch.Tensor,
    owners: torch.Tensor,
    rows: torch.Tensor,
    *,
    route_rank: int,
    layer: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    samples = []
    for iteration in range(warmup + iterations):
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        scored = _score_rows(
            planner,
            selected,
            layout,
            owners,
            rows,
            route_rank=route_rank,
            layer=layer,
        )
        elapsed = _max_elapsed_ms(started, device)
        if iteration >= warmup:
            samples.append(elapsed)
        del scored
    return {
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def main() -> None:
    args = _parse_args()
    rank, world_size, device = _initialize(args.backend)
    if world_size > 1 and world_size != args.ep_size:
        raise ValueError(f"Distributed world size {world_size} must equal ep_size {args.ep_size}.")
    route_rank = rank if world_size > 1 else 0
    first_routes, first_record = _load_route(args.route_dir, args.layers[0], route_rank, device)
    num_experts = int(first_record["num_experts"])
    layout, owners, slots_per_rank = _initial_layout(
        num_experts=num_experts,
        ep_size=args.ep_size,
        slot_increment=args.slot_increment,
        device=device,
    )
    planner = _planner(
        args,
        hidden_size=int(first_record["hidden_size"]),
        bytes_per_element=int(first_record["bytes_per_element"]),
        slots_per_rank=slots_per_rank,
        candidate_scorer="statistics",
    )
    all_rows = _all_candidate_rows(planner, layout, owners)
    grid = {(m, d): [] for m in args.m_values for d in args.d_values}
    unary_grid = {k: [] for k in args.unary_k_values}
    layer_results = []
    timed_rows = None
    timed_selected = None

    for layer_index, layer in enumerate(args.layers):
        selected, _record = (
            (first_routes, first_record)
            if layer_index == 0
            else _load_route(args.route_dir, layer, route_rank, device)
        )
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        full = _score_rows(
            planner,
            selected,
            layout,
            owners,
            all_rows,
            route_rank=route_rank,
            layer=layer,
        )
        full_score_ms = _max_elapsed_ms(started, device)
        full_costs = full.total
        full_best_relative = int(full_costs[1:].argmin().item())
        full_best_index = full_best_relative + 1
        baseline = float(full_costs[0].item())
        exact = float(full_costs[full_best_index].item())
        exact_gain = max(0.0, baseline - exact)

        original_pair_builder = statistical_scorer._build_dense_pair_events
        statistical_scorer._build_dense_pair_events = lambda *_args, **_kwargs: None
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        unary_started = time.perf_counter()
        try:
            unary = _score_rows(
                planner,
                selected,
                layout,
                owners,
                all_rows,
                route_rank=route_rank,
                layer=layer,
            )
        finally:
            statistical_scorer._build_dense_pair_events = original_pair_builder
        unary_score_ms = _max_elapsed_ms(unary_started, device)
        unary_candidate_costs = unary.total[1:]
        unary_order = unary_candidate_costs.argsort()
        layer_unary_grid = {}
        for raw_k in args.unary_k_values:
            k = min(max(1, int(raw_k)), int(all_rows.shape[0]))
            candidate_indices = unary_order[:k]
            filtered_costs = full_costs[1:].index_select(0, candidate_indices)
            filtered_relative = int(filtered_costs.argmin().item())
            winner_row_index = int(candidate_indices[filtered_relative].item())
            filtered = float(filtered_costs[filtered_relative].item())
            captured_gain = max(0.0, baseline - filtered)
            entry = {
                "candidate_count": k,
                "candidate_fraction": float(k / all_rows.shape[0]),
                "winner_match": winner_row_index == full_best_relative,
                "relative_cost_gap": float((filtered - exact) / max(abs(exact), 1.0)),
                "gain_capture": float(captured_gain / exact_gain) if exact_gain > 0.0 else 1.0,
            }
            layer_unary_grid[f"k{k}"] = entry
            unary_grid[raw_k].append(entry)

        communication_scores, compute_scores, destination_load, _packed, bottlenecks = _critical_expert_scores(
            planner,
            selected,
            full.baseline_physical_routes,
        )
        layer_grid = {}
        for m in args.m_values:
            for d in args.d_values:
                keep, filter_details = _candidate_mask(
                    all_rows,
                    owners,
                    communication_scores=communication_scores,
                    compute_scores=compute_scores,
                    destination_load=destination_load,
                    source_count=m,
                    destination_count=d,
                    slots_per_rank=slots_per_rank,
                )
                candidate_indices = torch.nonzero(keep, as_tuple=False).flatten()
                filtered_costs = full_costs[1:].index_select(0, candidate_indices)
                filtered_relative = int(filtered_costs.argmin().item())
                winner_row_index = int(candidate_indices[filtered_relative].item())
                filtered = float(filtered_costs[filtered_relative].item())
                captured_gain = max(0.0, baseline - filtered)
                entry = {
                    **filter_details,
                    "candidate_count": int(candidate_indices.numel()),
                    "candidate_fraction": float(candidate_indices.numel() / all_rows.shape[0]),
                    "winner_match": winner_row_index == full_best_relative,
                    "relative_cost_gap": float((filtered - exact) / max(abs(exact), 1.0)),
                    "gain_capture": float(captured_gain / exact_gain) if exact_gain > 0.0 else 1.0,
                }
                layer_grid[f"m{m}_d{d}"] = entry
                grid[(m, d)].append(entry)
                if layer_index == 0 and m == args.timed_m and d == args.timed_d:
                    timed_rows = all_rows.index_select(0, candidate_indices)
                    timed_selected = selected
        exact_row = all_rows[full_best_relative]
        bottleneck_key = f"m{args.timed_m}_d{args.timed_d}"
        bottleneck_entry = layer_grid[bottleneck_key]
        bottleneck_keep, _unused_filter_details = _candidate_mask(
            all_rows,
            owners,
            communication_scores=communication_scores,
            compute_scores=compute_scores,
            destination_load=destination_load,
            source_count=args.timed_m,
            destination_count=args.timed_d,
            slots_per_rank=slots_per_rank,
        )
        bottleneck_indices = torch.nonzero(bottleneck_keep, as_tuple=False).flatten()
        bottleneck_costs = full_costs[1:].index_select(0, bottleneck_indices)
        bottleneck_row = all_rows[bottleneck_indices[bottleneck_costs.argmin()]]
        bottleneck_entry["exact_row"] = exact_row.detach().cpu().tolist()
        bottleneck_entry["filtered_row"] = bottleneck_row.detach().cpu().tolist()
        layer_results.append(
            {
                "layer": layer,
                "full_score_ms": full_score_ms,
                "full_candidate_count": int(all_rows.shape[0]),
                "exact_gain": exact_gain,
                "unary_score_ms": unary_score_ms,
                "bottlenecks": bottlenecks,
                "grid": layer_grid,
                "unary_grid": layer_unary_grid,
            }
        )
        del full, unary

    aggregate_grid = {}
    for (m, d), entries in grid.items():
        aggregate_grid[f"m{m}_d{d}"] = {
            "mean_candidates": statistics.mean(value["candidate_count"] for value in entries),
            "mean_candidate_fraction": statistics.mean(value["candidate_fraction"] for value in entries),
            "winner_match_rate": statistics.mean(float(value["winner_match"]) for value in entries),
            "mean_relative_cost_gap": statistics.mean(value["relative_cost_gap"] for value in entries),
            "max_relative_cost_gap": max(value["relative_cost_gap"] for value in entries),
            "mean_gain_capture": statistics.mean(value["gain_capture"] for value in entries),
            "min_gain_capture": min(value["gain_capture"] for value in entries),
        }
    aggregate_unary_grid = {}
    for k, entries in unary_grid.items():
        aggregate_unary_grid[f"k{k}"] = {
            "mean_candidates": statistics.mean(value["candidate_count"] for value in entries),
            "mean_candidate_fraction": statistics.mean(value["candidate_fraction"] for value in entries),
            "winner_match_rate": statistics.mean(float(value["winner_match"]) for value in entries),
            "mean_relative_cost_gap": statistics.mean(value["relative_cost_gap"] for value in entries),
            "max_relative_cost_gap": max(value["relative_cost_gap"] for value in entries),
            "mean_gain_capture": statistics.mean(value["gain_capture"] for value in entries),
            "min_gain_capture": min(value["gain_capture"] for value in entries),
        }

    timing = None
    if timed_rows is not None and timed_selected is not None:
        current_statistics = _timed_filtered_score(
            planner,
            timed_selected,
            layout,
            owners,
            timed_rows,
            route_rank=route_rank,
            layer=args.layers[0],
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        reference = _planner(
            args,
            hidden_size=int(first_record["hidden_size"]),
            bytes_per_element=int(first_record["bytes_per_element"]),
            slots_per_rank=slots_per_rank,
            candidate_scorer="reference",
        )
        reference._fused_candidate_local_deltas = lambda *_args, **_kwargs: None
        on_demand_reference = _timed_filtered_score(
            reference,
            timed_selected,
            layout,
            owners,
            timed_rows,
            route_rank=route_rank,
            layer=args.layers[0],
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        timing = {
            "m": args.timed_m,
            "d": args.timed_d,
            "candidate_count": int(timed_rows.shape[0]),
            "current_dense_statistics": current_statistics,
            "on_demand_reference_no_kernel": on_demand_reference,
        }

    result = {
        "world_size": world_size,
        "layers": args.layers,
        "full_candidate_count": int(all_rows.shape[0]),
        "aggregate_grid": aggregate_grid,
        "aggregate_unary_grid": aggregate_unary_grid,
        "timing": timing,
        "full_score_ms": {
            "mean": statistics.mean(value["full_score_ms"] for value in layer_results),
            "median": statistics.median(value["full_score_ms"] for value in layer_results),
            "minimum": min(value["full_score_ms"] for value in layer_results),
            "maximum": max(value["full_score_ms"] for value in layer_results),
        },
        "unary_score_ms": {
            "mean": statistics.mean(value["unary_score_ms"] for value in layer_results),
            "median": statistics.median(value["unary_score_ms"] for value in layer_results),
            "minimum": min(value["unary_score_ms"] for value in layer_results),
            "maximum": max(value["unary_score_ms"] for value in layer_results),
        },
        "layer_results": layer_results,
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
