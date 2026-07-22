# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark the tensor CoRe-MoE local route-compression path on NPU."""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from veomni.distributed.moe.hiermoe.core_planner import compress_local_route_payload
from veomni.distributed.moe.hiermoe.oracle import load_route_snapshot
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import CurrentRoutePlanner, _SwapStats, assign_tokens_to_copies
from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[p90_index],
        "max_ms": ordered[-1],
    }


def _measure(function: Callable[[], object], *, warmup: int, iterations: int) -> dict[str, float]:
    values: list[float] = []
    for iteration in range(warmup + iterations):
        started = torch.npu.Event(enable_timing=True)
        finished = torch.npu.Event(enable_timing=True)
        started.record()
        output = function()
        finished.record()
        torch.npu.synchronize()
        if iteration >= warmup:
            values.append(float(started.elapsed_time(finished)))
        del output
    return _summary(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-copies", type=int, default=8)
    parser.add_argument("--search-primitives", action="store_true")
    parser.add_argument("--oracle-actions", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_copies <= 8:
        raise ValueError("max-copies must be between one and eight.")

    importlib.import_module("torch_npu")
    torch.npu.set_device(0)
    snapshot = load_route_snapshot(args.snapshot)
    if args.rank < 0 or args.rank >= snapshot.ep_size:
        raise ValueError(f"rank must be in [0, {snapshot.ep_size}), got {args.rank}.")
    routes = snapshot.routes_by_rank[args.rank].to("npu:0")
    result = _measure(
        lambda: compress_local_route_payload(
            routes,
            num_experts=snapshot.num_experts,
            sample_size=args.sample_size,
            source_rank=args.rank,
            step=snapshot.step,
            layer_seed=24,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(
        f"rank={args.rank} tokens={routes.shape[0]} top_k={routes.shape[1]} sample={args.sample_size} "
        f"median={result['median_ms']:.3f}ms p90={result['p90_ms']:.3f}ms max={result['max_ms']:.3f}ms"
    )
    if not args.search_primitives:
        return

    payloads = torch.stack(
        [
            compress_local_route_payload(
                rank_routes,
                num_experts=snapshot.num_experts,
                sample_size=args.sample_size,
                source_rank=source_rank,
                step=snapshot.step,
                layer_seed=24,
            )
            for source_rank, rank_routes in enumerate(snapshot.routes_by_rank)
        ],
        dim=0,
    )
    ordinal_offset = 2 + snapshot.num_experts
    sample_ordinals = payloads[:, ordinal_offset : ordinal_offset + args.sample_size].reshape(-1).to("npu:0")
    offset = ordinal_offset + 2 * args.sample_size
    sample = payloads[:, offset:].view(snapshot.ep_size, args.sample_size, -1).reshape(-1, routes.shape[1])
    sample = sample.to("npu:0")
    equal_routes = sample.unsqueeze(-1).eq(sample.unsqueeze(-2))
    route_positions = torch.arange(sample.shape[1], dtype=torch.long, device="npu:0")
    prior_positions = route_positions.view(1, -1, 1) > route_positions.view(1, 1, -1)
    sample_multiplicity = equal_routes.sum(dim=-1).to(torch.long) * (~(equal_routes & prior_positions).any(dim=-1))
    slots_per_rank = snapshot.num_experts // snapshot.ep_size + 1
    owner_slots = torch.arange(snapshot.num_experts, device="npu:0")
    owner_slots = torch.div(
        owner_slots, snapshot.num_experts // snapshot.ep_size, rounding_mode="floor"
    ) * slots_per_rank + torch.remainder(owner_slots, snapshot.num_experts // snapshot.ep_size)
    layout = torch.full((snapshot.ep_size * slots_per_rank,), -1, dtype=torch.long, device="npu:0")
    layout.scatter_(0, owner_slots, torch.arange(snapshot.num_experts, device="npu:0"))
    candidate = layout.clone()
    candidate[slots_per_rank + slots_per_rank - 1] = 0
    planner = CurrentRoutePlanner(
        hierarchy=snapshot.hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=snapshot.hidden_size,
        bytes_per_element=snapshot.bytes_per_element,
        slots_per_rank=slots_per_rank,
    )

    def swap_stats():
        token_hits = torch.nn.functional.one_hot(sample, num_classes=snapshot.num_experts).amax(dim=1).float()
        stats = planner._initial_swap_stats(token_hits, sample, owner_slots)
        current = planner._current_swap_cost(stats)
        pairs, _ = planner._swap_candidates(
            stats,
            current,
            torch.zeros((snapshot.num_experts,), dtype=torch.bool, device="npu:0"),
        )
        return planner._swap_candidate_costs(stats, pairs)

    layouts = torch.stack((layout, candidate), dim=0)
    logical_ids = torch.arange(snapshot.num_experts, device="npu:0").view(1, 1, -1)
    slot_ids = torch.arange(layouts.shape[1], device="npu:0").view(1, -1, 1)
    matches = layouts.unsqueeze(-1) == logical_ids
    copy_counts = matches.sum(dim=1)
    max_copies = args.max_copies
    copy_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, layouts.shape[1]))
    copy_slots = copy_slots.sort(dim=1).values[:, :max_copies].transpose(1, 2).contiguous()
    copy_indices = torch.arange(max_copies, dtype=torch.long, device="npu:0").view(1, 1, -1)
    copy_slots = torch.where(copy_indices < copy_counts.unsqueeze(-1), copy_slots, -torch.ones_like(copy_slots))
    owner_ranks = torch.div(owner_slots, slots_per_rank, rounding_mode="floor").view(1, -1).expand(2, -1).contiguous()
    mapping_result = _measure(
        lambda: assign_tokens_to_copies(
            routes,
            layouts,
            slots_per_rank=slots_per_rank,
            source_ranks=args.rank,
            hierarchy_group_sizes=snapshot.hierarchy.group_sizes,
            owner_slots=owner_slots,
            step=snapshot.step,
            layer_seed=24,
            max_copies=max_copies,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    swap_result = _measure(swap_stats, warmup=args.warmup, iterations=args.iterations)
    local_sample = sample[: args.sample_size]
    local_token_hits = torch.nn.functional.one_hot(local_sample, num_classes=snapshot.num_experts).amax(dim=1).float()
    local_swap_result = _measure(
        lambda: planner._initial_swap_stats(local_token_hits, local_sample, owner_slots),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    extension = get_hiermoe_planner_npu_ops()
    if extension is None or not all(hasattr(extension, name) for name in ("dual_map", "quota_policy", "quota_map")):
        raise RuntimeError("The CoRe-MoE dual-map, quota-policy, and quota-map NPU ops must be built.")
    hierarchy_levels = snapshot.hierarchy.group_sizes[:-1]
    fused_mapping_result = _measure(
        lambda: extension.dual_map(
            routes,
            copy_slots,
            copy_counts,
            owner_ranks,
            slots_per_rank,
            args.rank,
            snapshot.ep_size,
            len(hierarchy_levels),
            hierarchy_levels[0] if hierarchy_levels else 1,
            hierarchy_levels[1] if len(hierarchy_levels) > 1 else 1,
            snapshot.step,
            24,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    token_counts = payloads[:, 0].to(device="npu:0", dtype=torch.float32)
    assignment_counts = payloads[:, 2 : 2 + snapshot.num_experts].to(device="npu:0", dtype=torch.long)
    sample_counts = torch.full((snapshot.ep_size,), args.sample_size, dtype=torch.float32, device="npu:0")
    sample_sources = torch.arange(snapshot.ep_size, device="npu:0").repeat_interleave(args.sample_size)
    sample_weights = (token_counts / sample_counts).index_select(0, sample_sources).contiguous()
    owner_rows = owner_slots.view(1, -1).expand(2, -1).contiguous()
    policy_levels = tuple(snapshot.hierarchy.group_sizes[: max(0, snapshot.hierarchy.selected_dim - 1)])
    padded_policy_levels = (*policy_levels, 1, 1)[:2]

    def quota_policy():
        return extension.quota_policy(
            sample.contiguous(),
            sample_multiplicity.contiguous(),
            sample_sources.contiguous(),
            sample_ordinals.contiguous(),
            assignment_counts.contiguous(),
            layouts,
            owner_rows,
            slots_per_rank,
            args.rank,
            snapshot.ep_size,
            max_copies,
            args.sample_size,
            len(policy_levels),
            padded_policy_levels[0],
            padded_policy_levels[1],
        )

    quota_policy_result = _measure(quota_policy, warmup=args.warmup, iterations=args.iterations)
    quota_weights, quota_configured, _rows, _row_counts, _digest = quota_policy()
    quota_map_result = _measure(
        lambda: extension.quota_map(
            routes,
            copy_slots,
            copy_counts,
            owner_ranks,
            quota_weights,
            quota_configured,
            torch.arange(routes.shape[0], dtype=torch.long, device="npu:0"),
            slots_per_rank,
            args.rank,
            snapshot.ep_size,
            len(policy_levels),
            padded_policy_levels[0],
            padded_policy_levels[1],
            snapshot.step,
            24,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    level_sizes = tuple(snapshot.hierarchy.group_sizes[: max(0, snapshot.hierarchy.selected_dim - 1)]) + (1,)
    padded_levels = (*level_sizes, 1, 1)[:3]
    link0 = planner.perf_model.inter[0]
    link1 = planner.perf_model.inter[min(1, len(planner.perf_model.inter) - 1)]

    def swap_search(max_swaps: int):
        return extension.swap_search(
            sample,
            sample_weights,
            assignment_counts,
            layout,
            owner_slots,
            max_swaps,
            slots_per_rank,
            snapshot.ep_size,
            len(level_sizes),
            padded_levels[0],
            padded_levels[1],
            padded_levels[2],
            snapshot.hidden_size * snapshot.bytes_per_element,
            1.0,
            0.0,
            planner.perf_model.a2a.alpha,
            planner.perf_model.a2a.beta,
            link0.alpha,
            link0.beta,
            link1.alpha,
            link1.beta,
            planner.perf_model.intra.alpha,
            planner.perf_model.intra.beta,
            planner.perf_model.source != "default",
        )

    swap_p1_result = _measure(lambda: swap_search(1), warmup=args.warmup, iterations=args.iterations)
    swap_p2_result = _measure(lambda: swap_search(2), warmup=args.warmup, iterations=args.iterations)
    swap_p4_result = _measure(lambda: swap_search(4), warmup=args.warmup, iterations=args.iterations)
    token_hits = torch.nn.functional.one_hot(sample, num_classes=snapshot.num_experts).amax(dim=1).float()
    initial_swap_stats = planner._initial_swap_stats(token_hits, sample, owner_slots)
    owner_ranks = torch.div(owner_slots, slots_per_rank, rounding_mode="floor")
    exact_assignment = assignment_counts.sum(dim=0).to(torch.float32)
    rank_assignment = torch.zeros((snapshot.ep_size,), dtype=torch.float32, device="npu:0")
    rank_assignment.scatter_add_(0, owner_ranks, exact_assignment)

    def weighted_stats(oracle_owners: torch.Tensor) -> _SwapStats:
        oracle_ranks = torch.div(oracle_owners, slots_per_rank, rounding_mode="floor")
        base_counts: list[torch.Tensor] = []
        expert_group_counts: list[torch.Tensor] = []
        sole_expert_counts: list[torch.Tensor] = []
        sole_pair_counts: list[torch.Tensor] = []
        local_token_group_counts: list[torch.Tensor] = []
        for level_size in level_sizes:
            group_by_logical = torch.div(oracle_ranks, level_size, rounding_mode="floor")
            num_groups = snapshot.ep_size // level_size
            group_index = group_by_logical.view(1, -1).expand(token_hits.shape[0], -1)
            occupancy = torch.zeros((token_hits.shape[0], num_groups), dtype=torch.float32, device="npu:0")
            occupancy.scatter_add_(1, group_index, token_hits)
            group_hits = (occupancy > 0).to(torch.float32)
            weighted_group_hits = group_hits * sample_weights.unsqueeze(1)
            own_occupancy = occupancy.index_select(1, group_by_logical)
            weighted_sole_hits = token_hits * (own_occupancy == 1).to(torch.float32)
            weighted_sole_hits *= sample_weights.unsqueeze(1)
            base_counts.append(weighted_group_hits.sum(dim=0))
            expert_group_counts.append(token_hits.transpose(0, 1).matmul(weighted_group_hits))
            sole_expert_counts.append(weighted_sole_hits.sum(dim=0))
            sole_pair_counts.append(weighted_sole_hits.transpose(0, 1).matmul(token_hits))
            local_token_group_counts.append(occupancy)
        return _SwapStats(
            owner_ranks=oracle_ranks,
            expert_token_counts=(token_hits * sample_weights.unsqueeze(1)).sum(dim=0),
            expert_assignment_counts=exact_assignment,
            base_counts=tuple(base_counts),
            expert_group_counts=tuple(expert_group_counts),
            sole_expert_counts=tuple(sole_expert_counts),
            sole_pair_counts=tuple(sole_pair_counts),
            local_token_group_counts=tuple(local_token_group_counts),
        )

    def weighted_swap_oracle() -> list[list[int]]:
        oracle_layout = layout.clone()
        oracle_owners = owner_slots.clone()
        stats = weighted_stats(oracle_owners)
        used = torch.zeros((snapshot.num_experts,), dtype=torch.bool, device="npu:0")
        oracle_actions: list[list[int]] = []
        for _ in range(4):
            current = planner._current_swap_cost(stats)
            pairs, valid = planner._swap_candidates(stats, current, used)
            candidate_cost, candidate_groups = planner._swap_candidate_costs(stats, pairs)
            candidate_total = torch.where(
                valid,
                candidate_cost.total,
                torch.full_like(candidate_cost.total, torch.inf),
            )
            best_index = candidate_total.argmin()
            if not oracle_actions:
                print(
                    "weighted-rebuild-first "
                    f"current={float(current.total.reshape(-1)[0].item()):.9f} "
                    f"best={float(candidate_total[best_index].item()):.9f} "
                    f"pair={pairs[best_index].cpu().tolist()} "
                    f"bottlenecks={[int(current.peak_communication_rank.item()), int(current.peak_compute_rank.item())]} "
                    f"valid={int(valid.sum().item())}"
                )
            if not bool((candidate_total[best_index] < current.total.reshape(-1)[0]).item()):
                break
            pair = pairs[best_index]
            lhs = int(pair[0].item())
            rhs = int(pair[1].item())
            lhs_slot = int(oracle_owners[lhs].item())
            rhs_slot = int(oracle_owners[rhs].item())
            oracle_actions.append([lhs, rhs, lhs_slot, rhs_slot, 1])
            used[pair] = True
            oracle_layout[lhs_slot], oracle_layout[rhs_slot] = (
                oracle_layout[rhs_slot].clone(),
                oracle_layout[lhs_slot].clone(),
            )
            oracle_owners[pair] = oracle_owners[pair.flip(0)]
            stats = weighted_stats(oracle_owners)
        return oracle_actions

    aggregate_stats = weighted_stats(owner_slots)
    aggregate_base = torch.cat(aggregate_stats.base_counts).contiguous()
    aggregate_expert_groups = torch.cat(aggregate_stats.expert_group_counts, dim=1).contiguous()
    aggregate_sole_experts = torch.stack(aggregate_stats.sole_expert_counts, dim=1).contiguous()
    aggregate_sole_pairs = torch.stack(aggregate_stats.sole_pair_counts, dim=1).contiguous()
    zero_expert_bytes = torch.zeros((snapshot.num_experts,), dtype=torch.long, device="npu:0")

    def swap_select(max_swaps: int):
        return extension.swap_select(
            aggregate_stats.expert_token_counts.contiguous(),
            aggregate_stats.expert_assignment_counts.contiguous(),
            aggregate_base,
            aggregate_expert_groups,
            aggregate_sole_experts,
            aggregate_sole_pairs,
            sample.contiguous(),
            sample_weights.to(torch.float32).contiguous(),
            layout,
            owner_slots,
            zero_expert_bytes,
            zero_expert_bytes,
            max_swaps,
            slots_per_rank,
            snapshot.ep_size,
            snapshot.hierarchy.local_world_size,
            len(level_sizes),
            padded_levels[0],
            padded_levels[1],
            padded_levels[2],
            snapshot.hidden_size * snapshot.bytes_per_element,
            1.0,
            0.0,
            planner.perf_model.a2a.alpha,
            planner.perf_model.a2a.beta,
            link0.alpha,
            link0.beta,
            link1.alpha,
            link1.beta,
            planner.perf_model.intra.alpha,
            planner.perf_model.intra.beta,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            planner.perf_model.source != "default",
        )

    swap_select_p1_result = _measure(lambda: swap_select(1), warmup=args.warmup, iterations=args.iterations)
    swap_select_p4_result = _measure(lambda: swap_select(4), warmup=args.warmup, iterations=args.iterations)
    selected_layout, selected_owners, selected_actions, selected_metadata = swap_select(4)

    def replica_init():
        return planner._initial_replica_stats(
            sample,
            owner_slots,
            sample_sources,
            torch.arange(sample.shape[0], dtype=torch.long, device="npu:0"),
            step=snapshot.step,
            layer_seed=24,
            base_counts=initial_swap_stats.base_counts,
            assignment_counts=rank_assignment,
        )

    replica_init_result = _measure(replica_init, warmup=args.warmup, iterations=args.iterations)
    replica_stats = replica_init()
    torch.npu.synchronize()
    all_experts = torch.arange(snapshot.num_experts, dtype=torch.long, device="npu:0")
    bottleneck_ranks = torch.stack(
        (
            planner._current_swap_cost(initial_swap_stats).peak_communication_rank.reshape(-1)[0],
            planner._current_swap_cost(initial_swap_stats).peak_compute_rank.reshape(-1)[0],
        )
    )
    hot_experts = all_experts[(owner_ranks.unsqueeze(1) == bottleneck_ranks.view(1, -1)).any(dim=1)]
    replica_score_result = _measure(
        lambda: planner._incremental_replica_candidates(replica_stats, all_experts),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    hot_replica_score_result = _measure(
        lambda: planner._incremental_replica_candidates(replica_stats, hot_experts),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    swap_layout, swap_owners, actions, metadata = swap_search(4)
    torch.npu.synchronize()
    print(
        f"swap-initial-score median={swap_result['median_ms']:.3f}ms p90={swap_result['p90_ms']:.3f}ms "
        f"max={swap_result['max_ms']:.3f}ms"
    )
    print(
        f"swap-local-sample-score median={local_swap_result['median_ms']:.3f}ms "
        f"p90={local_swap_result['p90_ms']:.3f}ms max={local_swap_result['max_ms']:.3f}ms"
    )
    print(
        f"dual-layout-hash-map median={mapping_result['median_ms']:.3f}ms "
        f"p90={mapping_result['p90_ms']:.3f}ms max={mapping_result['max_ms']:.3f}ms"
    )
    print(
        f"fused-dual-layout-map median={fused_mapping_result['median_ms']:.3f}ms "
        f"p90={fused_mapping_result['p90_ms']:.3f}ms max={fused_mapping_result['max_ms']:.3f}ms"
    )
    print(
        f"quota-policy-c{max_copies} median={quota_policy_result['median_ms']:.3f}ms "
        f"p90={quota_policy_result['p90_ms']:.3f}ms max={quota_policy_result['max_ms']:.3f}ms"
    )
    print(
        f"quota-map-c{max_copies} median={quota_map_result['median_ms']:.3f}ms "
        f"p90={quota_map_result['p90_ms']:.3f}ms max={quota_map_result['max_ms']:.3f}ms"
    )
    print(
        f"fused-swap-p1 median={swap_p1_result['median_ms']:.3f}ms "
        f"p90={swap_p1_result['p90_ms']:.3f}ms max={swap_p1_result['max_ms']:.3f}ms"
    )
    print(
        f"fused-swap-p2 median={swap_p2_result['median_ms']:.3f}ms "
        f"p90={swap_p2_result['p90_ms']:.3f}ms max={swap_p2_result['max_ms']:.3f}ms"
    )
    print(
        f"fused-swap-p4 median={swap_p4_result['median_ms']:.3f}ms "
        f"p90={swap_p4_result['p90_ms']:.3f}ms max={swap_p4_result['max_ms']:.3f}ms "
        f"accepted={int(metadata[0].item())} actions={actions[: int(metadata[0].item())].cpu().tolist()} "
        f"owner_changes={int((swap_owners != owner_slots).sum().item())} "
        f"layout_changes={int((swap_layout != layout).sum().item())}"
    )
    print(
        f"aggregate-swap-p1 median={swap_select_p1_result['median_ms']:.3f}ms "
        f"p90={swap_select_p1_result['p90_ms']:.3f}ms max={swap_select_p1_result['max_ms']:.3f}ms"
    )
    print(
        f"aggregate-swap-p4 median={swap_select_p4_result['median_ms']:.3f}ms "
        f"p90={swap_select_p4_result['p90_ms']:.3f}ms max={swap_select_p4_result['max_ms']:.3f}ms "
        f"accepted={int(selected_metadata[0].item())} "
        f"actions={selected_actions[: int(selected_metadata[0].item())].cpu().tolist()} "
        f"owner_changes={int((selected_owners != owner_slots).sum().item())} "
        f"layout_changes={int((selected_layout != layout).sum().item())}"
    )
    print(
        f"replica-init median={replica_init_result['median_ms']:.3f}ms "
        f"p90={replica_init_result['p90_ms']:.3f}ms max={replica_init_result['max_ms']:.3f}ms"
    )
    print(
        f"replica-one-shot-score median={replica_score_result['median_ms']:.3f}ms "
        f"p90={replica_score_result['p90_ms']:.3f}ms max={replica_score_result['max_ms']:.3f}ms"
    )
    print(
        f"replica-hot-score candidates={int(hot_experts.numel())} "
        f"median={hot_replica_score_result['median_ms']:.3f}ms "
        f"p90={hot_replica_score_result['p90_ms']:.3f}ms max={hot_replica_score_result['max_ms']:.3f}ms"
    )
    if args.oracle_actions:
        print(f"weighted-rebuild-actions={weighted_swap_oracle()}")
        oracle = planner.plan(
            sample,
            layout,
            owner_slots,
            source_ranks=sample_sources,
            max_swaps=4,
            max_replicas=0,
            step=snapshot.step,
            layer_seed=24,
        )
        print(f"unweighted-full-rebuild-actions={[action.format() for action in oracle.actions]}")


if __name__ == "__main__":
    main()
