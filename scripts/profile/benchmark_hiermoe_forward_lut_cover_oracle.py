#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Exhaustively score every legal Cover from cached Forward routes.

Unlike the generic redundant-expert scorer, this benchmark does not rebuild a
replica/hash route state space.  Each source rank starts from its persistent
Forward LUT and every candidate applies the exact two-column patch rule:

* the inserted expert is retargeted inside the destination service group;
* the evicted victim falls back only where its LUT referenced the overwritten
  slot.

All Cover candidates are evaluated in parallel from unary/pair sufficient
statistics.  The source-aware hybrid communication and max-assignment cost is
then reduced across EP ranks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.statistical_scorer import (
    prepare_forward_lut_cover_compact_statistics,
    score_forward_lut_cover_compact_statistics,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(2, 3, 4, 5))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(6, 7))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--service-group-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-cost-atol-ms", type=float, default=2e-3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _initialize(ep_size: int) -> tuple[int, int, torch.device]:
    importlib.import_module("torch_npu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != int(ep_size):
        raise ValueError(f"Expected EP size {ep_size}, got distributed world size {world_size}.")
    return rank, world_size, device


def _synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _layer_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _load_route(
    root: Path,
    *,
    step: int,
    layer: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    path = root / f"step{step:04d}" / f"layer{layer:02d}_call0_rank{rank:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    route = payload.get("routes") if isinstance(payload, dict) else None
    if not torch.is_tensor(route) or route.ndim != 2:
        raise ValueError(f"Invalid route capture: {path}.")
    return route.to(device=device, dtype=torch.long, non_blocking=True)


def _layer_state(
    payload: dict[str, object],
    *,
    layer: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Input layout has no layer table.")
    row = layers.get(_layer_name(layer))
    if not isinstance(row, dict):
        raise ValueError(f"Input layout has no state for layer {layer}.")
    layout = torch.tensor(row["slot_to_logical"], dtype=torch.long, device=device)
    owners = torch.tensor(row["owner_slots"], dtype=torch.long, device=device)
    lut = torch.tensor(row["source_logical_to_physical"], dtype=torch.long, device=device)
    if tuple(layout.shape) != (args.ep_size * args.slots_per_rank,):
        raise ValueError(f"Layer {layer} has an invalid physical layout shape.")
    if tuple(owners.shape) != (args.num_experts,):
        raise ValueError(f"Layer {layer} has an invalid owner table shape.")
    if tuple(lut.shape) != (args.ep_size, args.num_experts):
        raise ValueError(f"Layer {layer} has an invalid source LUT shape.")
    logical = torch.arange(args.num_experts, dtype=torch.long, device=device)
    if bool((layout.index_select(0, owners) != logical).any().item()):
        raise ValueError(f"Layer {layer} owner table does not reference its experts.")
    if bool(
        (
            layout.index_select(0, lut.reshape(-1)).view_as(lut)
            != logical.view(1, -1)
        ).any().item()
    ):
        raise ValueError(f"Layer {layer} source LUT does not reference its experts.")
    return layout, owners, lut


def _legal_cover_rows(
    planner: GreedyCommunicationPlanner,
    layout: torch.Tensor,
    owners: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(layout[layout >= 0], minlength=int(num_experts))
    slots = torch.arange(layout.numel(), dtype=torch.long, device=layout.device)
    destinations = slots[(layout >= 0) & (counts.index_select(0, layout.clamp_min(0)) > 1)]
    rows = planner._cover_rows(layout, owners, destinations)
    if rows.numel() == 0:
        return rows, rows.new_empty((0,))

    fallback = owners.index_select(0, rows[:, 4]).clone()
    owner_overwritten = fallback == rows[:, 2]
    if bool(owner_overwritten.any().item()):
        affected = torch.nonzero(owner_overwritten, as_tuple=False).reshape(-1)
        for action_index in affected.detach().to(device="cpu").tolist():
            victim = int(rows[action_index, 4].item())
            destination = int(rows[action_index, 2].item())
            remaining = torch.nonzero(
                (layout == victim)
                & (torch.arange(layout.numel(), device=layout.device) != destination),
                as_tuple=False,
            ).reshape(-1)
            if remaining.numel() == 0:
                raise ValueError("A legal Cover unexpectedly removes the final victim copy.")
            fallback[action_index] = remaining.min()
    return rows, fallback


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _max_rank_time(value_ms: float, device: torch.device) -> float:
    # HCCL on the target Ascend stack does not support float64 reductions.
    value = torch.tensor([value_ms], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _source_endpoint_statistics(
    *,
    unique_counts: torch.Tensor,
    assignment_counts: torch.Tensor,
    ep_size: int,
    ranks_per_node: int,
) -> torch.Tensor:
    """Build globally aggregated endpoint statistics from explicit sources.

    Inputs are ``[source_rank, candidate, destination]``.  This is equivalent
    to summing ``_local_traffic_endpoint_statistics`` over source ranks, but
    evaluates an owner shard without another collective.
    """

    if unique_counts.ndim != 3 or int(unique_counts.shape[0]) != int(ep_size):
        raise ValueError("unique_counts must have shape [ep_size, candidates, packed_width].")
    if assignment_counts.ndim != 3 or tuple(assignment_counts.shape[:2]) != tuple(unique_counts.shape[:2]):
        raise ValueError("assignment_counts must preserve source and candidate dimensions.")
    num_nodes = int(ep_size) // int(ranks_per_node)
    unique_rank = unique_counts[:, :, :ep_size]
    unique_node = unique_counts[:, :, ep_size : ep_size + num_nodes]
    assignment_rank = assignment_counts[:, :, :ep_size]
    assignment_node = assignment_rank.view(
        ep_size,
        assignment_rank.shape[1],
        num_nodes,
        ranks_per_node,
    ).sum(dim=3)
    candidates = int(unique_counts.shape[1])

    def endpoint_rows(
        rank_values: torch.Tensor,
        node_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cross-node stage: one source endpoint per source rank and one
        # receive endpoint per (relay lane, destination node).
        stage1_send = node_values.sum(dim=2).transpose(0, 1).contiguous()
        by_source_node = node_values.view(
            num_nodes,
            ranks_per_node,
            candidates,
            num_nodes,
        )
        stage1_receive = (
            by_source_node.sum(dim=0)
            .permute(1, 0, 2)
            .reshape(candidates, ep_size)
            .contiguous()
        )

        # Node-local stage: sources sharing one relay lane are aggregated for
        # each destination node; receives are the physical destination ranks.
        rank_matrix = rank_values.view(
            num_nodes,
            ranks_per_node,
            candidates,
            num_nodes,
            ranks_per_node,
        )
        by_lane_node = rank_matrix.sum(dim=4).sum(dim=0)
        stage2_send = (
            by_lane_node.permute(1, 2, 0)
            .reshape(candidates, ep_size)
            .contiguous()
        )
        stage2_receive = rank_values.sum(dim=0)
        return stage1_send, stage1_receive, stage2_send, stage2_receive

    unique_stage1_send, unique_stage1_receive, unique_stage2_send, unique_stage2_receive = endpoint_rows(
        unique_rank,
        unique_node,
    )
    (
        assignment_stage1_send,
        assignment_stage1_receive,
        assignment_stage2_send,
        assignment_stage2_receive,
    ) = endpoint_rows(
        assignment_rank,
        assignment_node,
    )
    return torch.cat(
        (
            unique_stage1_send,
            unique_stage1_receive,
            assignment_stage1_send,
            assignment_stage1_receive,
            unique_stage2_send,
            unique_stage2_receive,
            assignment_stage2_send,
            assignment_stage2_receive,
        ),
        dim=1,
    )


def _global_sharded_forward_lut_costs(
    *,
    planner: GreedyCommunicationPlanner,
    baseline_unique: torch.Tensor,
    communication_delta: torch.Tensor,
    baseline_assignments: torch.Tensor,
    assignment_delta: torch.Tensor,
    ep_size: int,
    ranks_per_node: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score exact source-aware costs with candidate shards owned by ranks.

    Every source rank sends only its compact ``rank+node+assignment`` deltas.
    Rank ``r`` receives all source contributions for its candidate shard,
    reconstructs the source-aware endpoint statistics locally, and all-gathers
    only two scalar metrics per candidate.  This replaces the previous dense
    ``candidate × 8 × ep_size`` reduce-scatter payload.
    """

    communication_width = int(communication_delta.shape[1])
    baseline = torch.cat(
        (
            baseline_unique,
            baseline_assignments,
        ),
        dim=1,
    ).contiguous()
    baseline_by_source = baseline.new_empty((ep_size, baseline.shape[1]))
    dist.all_gather_into_tensor(baseline_by_source, baseline, group=dist.group.WORLD)

    local_delta = torch.cat((communication_delta, assignment_delta), dim=1)
    candidates = int(local_delta.shape[0])
    shard_rows = (candidates + ep_size - 1) // ep_size
    padded_rows = shard_rows * ep_size
    if padded_rows != candidates:
        local_delta = torch.cat(
            (
                local_delta,
                local_delta.new_zeros((padded_rows - candidates, local_delta.shape[1])),
            ),
            dim=0,
        )
    received = local_delta.new_empty(local_delta.shape)
    dist.all_to_all_single(
        received,
        local_delta.contiguous(),
        group=dist.group.WORLD,
    )
    source_deltas = received.view(ep_size, shard_rows, -1)
    source_counts = baseline_by_source.unsqueeze(1) + source_deltas
    source_unique = source_counts[:, :, :communication_width]
    source_assignments = source_counts[:, :, communication_width:]
    endpoint = _source_endpoint_statistics(
        unique_counts=source_unique,
        assignment_counts=source_assignments,
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
    )
    shard_details = planner._traffic_endpoint_cost_details(endpoint)
    shard_metrics = torch.stack((shard_details[0], shard_details[1]), dim=1).contiguous()
    gathered_metrics = shard_metrics.new_empty((padded_rows, 2))
    dist.all_gather_into_tensor(
        gathered_metrics,
        shard_metrics,
        group=dist.group.WORLD,
    )
    candidate_metrics = gathered_metrics[:candidates]

    baseline_endpoint = _source_endpoint_statistics(
        unique_counts=baseline_by_source[:, :communication_width].unsqueeze(1),
        assignment_counts=baseline_by_source[:, communication_width:].unsqueeze(1),
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
    )
    baseline_details = planner._traffic_endpoint_cost_details(baseline_endpoint)
    return (
        torch.cat((baseline_details[0], candidate_metrics[:, 0]), dim=0),
        torch.cat((baseline_details[1], candidate_metrics[:, 1]), dim=0),
    )


def _affected_endpoint_ids(
    *,
    rows: torch.Tensor,
    victim_fallback_slots: torch.Tensor,
    source_logical_to_physical: torch.Tensor,
    ep_size: int,
    ranks_per_node: int,
    slots_per_rank: int,
    service_group_size: int,
) -> torch.Tensor:
    """Return the exact endpoint support of every Forward-LUT Cover.

    Endpoint IDs use the same eight ``[ep_size]`` blocks as
    ``GreedyCommunicationPlanner._local_traffic_endpoint_statistics``.  The
    support depends only on the action and the current LUT, never on token
    data, so every EP rank constructs an identical table.  Invalid tail
    entries are ``-1``.

    This intentionally distinguishes rank and node changes.  A within-node
    move cannot change stage-1 endpoint totals, while a cross-node move can
    change both stages.
    """

    if rows.numel() == 0:
        return rows.new_empty((0, 0))
    if int(ep_size) % int(ranks_per_node) != 0:
        raise ValueError("ranks_per_node must divide ep_size.")
    if int(ep_size) % int(service_group_size) != 0:
        raise ValueError("service_group_size must divide ep_size.")
    lut = source_logical_to_physical.to(device=rows.device, dtype=torch.long)
    if tuple(lut.shape[:1]) != (int(ep_size),):
        raise ValueError("source_logical_to_physical must have one row per EP rank.")

    candidates = int(rows.shape[0])
    num_nodes = int(ep_size) // int(ranks_per_node)
    endpoint_mask = torch.zeros(
        (candidates, 8 * int(ep_size)),
        dtype=torch.bool,
        device=rows.device,
    )
    action_indices = torch.arange(candidates, dtype=torch.long, device=rows.device)
    destination_slots = rows[:, 2]
    destination_ranks = torch.div(destination_slots, int(slots_per_rank), rounding_mode="floor")
    destination_nodes = torch.div(destination_ranks, int(ranks_per_node), rounding_mode="floor")
    fallback_ranks = torch.div(
        victim_fallback_slots.to(device=rows.device, dtype=torch.long),
        int(slots_per_rank),
        rounding_mode="floor",
    )
    lhs = rows[:, 3]
    rhs = rows[:, 4]

    for source_rank in range(int(ep_size)):
        lane = source_rank % int(ranks_per_node)
        source_node = source_rank // int(service_group_size)
        source_lut = lut[source_rank]
        lhs_old = torch.div(
            source_lut.index_select(0, lhs),
            int(slots_per_rank),
            rounding_mode="floor",
        )
        rhs_slots = source_lut.index_select(0, rhs)
        rhs_old = torch.div(rhs_slots, int(slots_per_rank), rounding_mode="floor")
        lhs_new = torch.where(destination_nodes == source_node, destination_ranks, lhs_old)
        rhs_new = torch.where(rhs_slots == destination_slots, fallback_ranks, rhs_old)

        rank_groups = torch.stack((lhs_old, lhs_new, rhs_old, rhs_new), dim=1)
        node_groups = torch.div(rank_groups, int(ranks_per_node), rounding_mode="floor")
        rank_changed = (lhs_old != lhs_new) | (rhs_old != rhs_new)
        node_changed = (
            torch.div(lhs_old, int(ranks_per_node), rounding_mode="floor")
            != torch.div(lhs_new, int(ranks_per_node), rounding_mode="floor")
        ) | (
            torch.div(rhs_old, int(ranks_per_node), rounding_mode="floor")
            != torch.div(rhs_new, int(ranks_per_node), rounding_mode="floor")
        )

        if bool(node_changed.any().item()):
            changed_actions = action_indices[node_changed]
            changed_nodes = node_groups[node_changed]
            # stage-1 unique send
            endpoint_mask[changed_actions, source_rank] = True
            # stage-1 unique/assignment receive
            endpoint_mask[
                changed_actions.view(-1, 1),
                int(ep_size) + lane * num_nodes + changed_nodes,
            ] = True
            endpoint_mask[
                changed_actions.view(-1, 1),
                3 * int(ep_size) + lane * num_nodes + changed_nodes,
            ] = True
            # stage-2 assignment send; node totals only change cross-node.
            endpoint_mask[
                changed_actions.view(-1, 1),
                6 * int(ep_size) + changed_nodes * int(ranks_per_node) + lane,
            ] = True

        if bool(rank_changed.any().item()):
            changed_actions = action_indices[rank_changed]
            changed_ranks = rank_groups[rank_changed]
            changed_nodes = node_groups[rank_changed]
            # stage-2 unique send/receive and assignment receive.
            endpoint_mask[
                changed_actions.view(-1, 1),
                4 * int(ep_size) + changed_nodes * int(ranks_per_node) + lane,
            ] = True
            endpoint_mask[
                changed_actions.view(-1, 1),
                5 * int(ep_size) + changed_ranks,
            ] = True
            endpoint_mask[
                changed_actions.view(-1, 1),
                7 * int(ep_size) + changed_ranks,
            ] = True

    affected_count = endpoint_mask.sum(dim=1)
    width = int(affected_count.max().item())
    endpoint_ids = torch.where(
        endpoint_mask,
        torch.arange(8 * int(ep_size), dtype=torch.long, device=rows.device).view(1, -1),
        torch.full((1, 8 * int(ep_size)), 8 * int(ep_size), dtype=torch.long, device=rows.device),
    )
    endpoint_ids = endpoint_ids.sort(dim=1).values[:, :width]
    return torch.where(
        torch.arange(width, dtype=torch.long, device=rows.device).view(1, -1)
        < affected_count.view(-1, 1),
        endpoint_ids,
        torch.full_like(endpoint_ids, -1),
    ).contiguous()


def _pack_local_endpoint_deltas(
    *,
    communication_delta: torch.Tensor,
    assignment_delta: torch.Tensor,
    affected_endpoint_ids: torch.Tensor,
    source_rank: int,
    ep_size: int,
    ranks_per_node: int,
) -> torch.Tensor:
    """Project local rank/node deltas directly onto compact endpoint IDs."""

    if int(ep_size) % int(ranks_per_node) != 0:
        raise ValueError("ranks_per_node must divide ep_size.")
    candidates, width = affected_endpoint_ids.shape
    if int(communication_delta.shape[0]) != candidates or int(assignment_delta.shape[0]) != candidates:
        raise ValueError("Endpoint IDs and local deltas must have the same candidate count.")
    num_nodes = int(ep_size) // int(ranks_per_node)
    unique_rank = communication_delta[:, :ep_size]
    unique_node = communication_delta[:, ep_size : ep_size + num_nodes]
    assignment_rank = assignment_delta[:, :ep_size]
    assignment_node = assignment_rank.view(candidates, num_nodes, ranks_per_node).sum(dim=2)

    valid = affected_endpoint_ids >= 0
    safe_ids = affected_endpoint_ids.clamp_min(0)
    blocks = torch.div(safe_ids, int(ep_size), rounding_mode="floor")
    endpoints = torch.remainder(safe_ids, int(ep_size))
    lane = int(source_rank) % int(ranks_per_node)
    endpoint_nodes = torch.div(endpoints, int(ranks_per_node), rounding_mode="floor")
    stage1_nodes = torch.remainder(endpoints, num_nodes)

    values = communication_delta.new_zeros((candidates, width))

    def assign(block: int, block_values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        selected = blocks == int(block)
        if mask is not None:
            selected &= mask
        values.copy_(torch.where(selected, block_values, values))

    assign(
        0,
        unique_node.sum(dim=1, keepdim=True).expand(-1, width),
        endpoints == int(source_rank),
    )
    assign(
        1,
        unique_node.gather(1, stage1_nodes.clamp_max(num_nodes - 1)),
        torch.div(endpoints, num_nodes, rounding_mode="floor") == lane,
    )
    # Assignment stage-1 send is invariant for a Cover and therefore absent
    # from the affected support.  Keep the projection complete so this helper
    # remains equivalent to the dense endpoint transform for arbitrary IDs.
    assign(
        2,
        assignment_node.sum(dim=1, keepdim=True).expand(-1, width),
        endpoints == int(source_rank),
    )
    assign(
        3,
        assignment_node.gather(1, stage1_nodes.clamp_max(num_nodes - 1)),
        torch.div(endpoints, num_nodes, rounding_mode="floor") == lane,
    )
    assign(
        4,
        unique_rank.view(candidates, num_nodes, ranks_per_node)
        .sum(dim=2)
        .gather(1, endpoint_nodes.clamp_max(num_nodes - 1)),
        torch.remainder(endpoints, int(ranks_per_node)) == lane,
    )
    assign(5, unique_rank.gather(1, endpoints.clamp_max(int(ep_size) - 1)))
    assign(
        6,
        assignment_node.gather(1, endpoint_nodes.clamp_max(num_nodes - 1)),
        torch.remainder(endpoints, int(ranks_per_node)) == lane,
    )
    assign(7, assignment_rank.gather(1, endpoints.clamp_max(int(ep_size) - 1)))
    return values * valid.to(values.dtype)


def _global_compact_endpoint_costs(
    *,
    planner: GreedyCommunicationPlanner,
    baseline_endpoint_local: torch.Tensor,
    packed_endpoint_delta_local: torch.Tensor,
    affected_endpoint_ids: torch.Tensor,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce compact endpoint deltas and score one candidate shard per rank."""

    baseline_global = baseline_endpoint_local.clone()
    dist.all_reduce(baseline_global, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    baseline_details = planner._traffic_endpoint_cost_details(baseline_global.unsqueeze(0))

    candidates, width = packed_endpoint_delta_local.shape
    shard_rows = (candidates + int(ep_size) - 1) // int(ep_size)
    padded_rows = shard_rows * int(ep_size)
    if padded_rows != candidates:
        padding = padded_rows - candidates
        packed_endpoint_delta_local = torch.cat(
            (
                packed_endpoint_delta_local,
                packed_endpoint_delta_local.new_zeros((padding, width)),
            ),
            dim=0,
        )
        affected_endpoint_ids = torch.cat(
            (
                affected_endpoint_ids,
                affected_endpoint_ids.new_full((padding, width), -1),
            ),
            dim=0,
        )
    reduced_shard = packed_endpoint_delta_local.new_empty((shard_rows, width))
    dist.reduce_scatter_tensor(
        reduced_shard,
        packed_endpoint_delta_local.contiguous(),
        op=dist.ReduceOp.SUM,
        group=dist.group.WORLD,
    )
    group_rank = dist.get_rank()
    shard_start = group_rank * shard_rows
    shard_ids = affected_endpoint_ids[shard_start : shard_start + shard_rows]
    valid = shard_ids >= 0
    safe_ids = shard_ids.clamp_min(0)
    shard_endpoints = baseline_global.unsqueeze(0).expand(shard_rows, -1).clone()
    candidate_rows = torch.arange(shard_rows, dtype=torch.long, device=shard_endpoints.device).view(-1, 1)
    shard_endpoints.reshape(-1).index_add_(
        0,
        (candidate_rows * shard_endpoints.shape[1] + safe_ids).reshape(-1),
        (reduced_shard * valid.to(reduced_shard.dtype)).reshape(-1),
    )
    shard_details = planner._traffic_endpoint_cost_details(shard_endpoints)
    shard_metrics = torch.stack((shard_details[0], shard_details[1]), dim=1).contiguous()
    gathered_metrics = shard_metrics.new_empty((padded_rows, 2))
    dist.all_gather_into_tensor(gathered_metrics, shard_metrics, group=dist.group.WORLD)
    candidate_metrics = gathered_metrics[:candidates]
    return (
        torch.cat((baseline_details[0], candidate_metrics[:, 0]), dim=0),
        torch.cat((baseline_details[1], candidate_metrics[:, 1]), dim=0),
    )


def _score_rows(
    *,
    planner: GreedyCommunicationPlanner,
    rows: torch.Tensor,
    fallback_slots: torch.Tensor,
    lut: torch.Tensor,
    steps: tuple[int, ...],
    layer: int,
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    total_communication = rows.new_zeros((rows.shape[0] + 1,), dtype=torch.float32)
    total_compute = rows.new_zeros((rows.shape[0] + 1,), dtype=torch.float32)
    local_statistics_ms = 0.0
    global_scoring_ms = 0.0
    source_lut = lut[rank]
    affected_endpoint_ids = _affected_endpoint_ids(
        rows=rows,
        victim_fallback_slots=fallback_slots,
        source_logical_to_physical=lut,
        ep_size=args.ep_size,
        ranks_per_node=args.ranks_per_node,
        slots_per_rank=args.slots_per_rank,
        service_group_size=args.service_group_size,
    )

    for step in steps:
        selected = _load_route(
            args.route_root,
            step=step,
            layer=layer,
            rank=rank,
            device=device,
        )
        physical = source_lut.index_select(0, selected.reshape(-1)).view_as(selected)

        _synchronize(device)
        started = time.perf_counter()
        baseline_unique = planner._local_packed_counts(physical)
        baseline_assignments = planner._local_packed_assignment_counts(physical)
        compact_statistics = prepare_forward_lut_cover_compact_statistics(
            planner,
            selected,
            source_logical_to_physical=source_lut,
            num_experts=args.num_experts,
        )
        communication_delta, assignment_delta = score_forward_lut_cover_compact_statistics(
            planner,
            compact_statistics,
            rows,
            source_logical_to_physical=source_lut,
            victim_fallback_slots=fallback_slots,
            uniform_source_rank=rank,
            service_group_size=args.service_group_size,
            num_experts=args.num_experts,
        )
        baseline_endpoint = planner._local_traffic_endpoint_statistics(
            baseline_unique,
            baseline_assignments[:, : args.ep_size],
            source_rank=rank,
        ).squeeze(0)
        packed_endpoint_delta = _pack_local_endpoint_deltas(
            communication_delta=communication_delta,
            assignment_delta=assignment_delta,
            affected_endpoint_ids=affected_endpoint_ids,
            source_rank=rank,
            ep_size=args.ep_size,
            ranks_per_node=args.ranks_per_node,
        )
        _synchronize(device)
        local_statistics_ms += (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        communication, compute = _global_compact_endpoint_costs(
            planner=planner,
            baseline_endpoint_local=baseline_endpoint,
            packed_endpoint_delta_local=packed_endpoint_delta,
            affected_endpoint_ids=affected_endpoint_ids,
            ep_size=args.ep_size,
        )
        _synchronize(device)
        global_scoring_ms += (time.perf_counter() - started) * 1000.0
        total_communication += communication
        total_compute += compute

    divisor = float(len(steps))
    return (
        total_communication / divisor,
        total_compute / divisor,
        {
            "local_statistics_ms": _max_rank_time(local_statistics_ms, device),
            "global_scoring_ms": _max_rank_time(global_scoring_ms, device),
        },
    )


def _action_dict(row: torch.Tensor) -> dict[str, int]:
    values = row.detach().to(device="cpu", dtype=torch.long).tolist()
    return {
        "source_slot": int(values[1]),
        "destination_slot": int(values[2]),
        "source_logical": int(values[3]),
        "victim_logical": int(values[4]),
        # The caller adds slots_per_rank when comparing with the CPU anchor;
        # retain the physical slot here and derive the rank at the call site.
    }


def _anchor_expectation(anchor_path: Path | None) -> tuple[int, dict[str, int], float] | None:
    if anchor_path is None:
        return None
    payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("The CPU anchor has no Cover round.")
    first = rounds[0]
    if not isinstance(first, dict) or not isinstance(first.get("action"), dict):
        raise ValueError("The CPU anchor first round has no action.")
    return (
        int(payload["layer"]),
        {key: int(value) for key, value in first["action"].items() if key != "proxy_score"},
        float(first["optimize_gain_mean_ms"]),
    )


def main() -> None:
    args = _parse_args()
    rank, _world_size, device = _initialize(args.ep_size)
    if args.layers <= 0:
        raise ValueError("layers must be positive.")
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    anchor = _anchor_expectation(args.anchor)

    def reducer(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=(args.ranks_per_node, args.ep_size),
            source="forward-lut-cover-oracle",
            local_world_size=args.ranks_per_node,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=args.hidden_size,
        bytes_per_element=args.bytes_per_element,
        slots_per_rank=args.slots_per_rank,
        reducer=reducer,
        process_group=dist.group.WORLD,
        max_copies=args.max_copies,
        traffic_inter_ms_per_byte=args.inter_ms_per_byte,
        traffic_intra_ms_per_byte=args.intra_ms_per_byte,
        traffic_route_ms_per_assignment=args.route_ms_per_assignment,
        traffic_communication_phase_multiplier=args.communication_phase_multiplier,
        traffic_compute_phase_multiplier=args.compute_phase_multiplier,
        forward_compute_per_assignment=args.compute_ms_per_assignment,
        assume_unique_routes=True,
    )

    results: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if args.warmup:
        warmup_layer = args.layer_start
        warmup_layout, warmup_owners, warmup_lut = _layer_state(
            payload,
            layer=warmup_layer,
            device=device,
            args=args,
        )
        warmup_rows, warmup_fallbacks = _legal_cover_rows(
            planner,
            warmup_layout,
            warmup_owners,
            num_experts=args.num_experts,
        )
        for _ in range(args.warmup):
            _score_rows(
                planner=planner,
                rows=warmup_rows,
                fallback_slots=warmup_fallbacks,
                lut=warmup_lut,
                steps=(args.optimize_steps[0],),
                layer=warmup_layer,
                rank=rank,
                device=device,
                args=args,
            )
    for layer in range(args.layer_start, args.layer_start + args.layers):
        layout, owners, lut = _layer_state(
            payload,
            layer=layer,
            device=device,
            args=args,
        )
        rows, fallbacks = _legal_cover_rows(
            planner,
            layout,
            owners,
            num_experts=args.num_experts,
        )
        communication, compute, timing = _score_rows(
            planner=planner,
            rows=rows,
            fallback_slots=fallbacks,
            lut=lut,
            steps=args.optimize_steps,
            layer=layer,
            rank=rank,
            device=device,
            args=args,
        )
        total = communication + compute
        if rows.numel():
            candidate_index = int(total[1:].argmin().item())
            candidate_cost = float(total[candidate_index + 1].item())
            baseline_cost = float(total[0].item())
            optimize_gain = baseline_cost - candidate_cost
            accepted = optimize_gain > 0.0
            winner_row = rows[candidate_index]
            winner_fallback = fallbacks[candidate_index : candidate_index + 1]
            action = _action_dict(winner_row)
            action["target_rank"] = action["destination_slot"] // args.slots_per_rank
        else:
            candidate_index = -1
            baseline_cost = float(total[0].item())
            candidate_cost = baseline_cost
            optimize_gain = 0.0
            accepted = False
            winner_row = rows.new_empty((0, 5))
            winner_fallback = fallbacks.new_empty((0,))
            action = None

        validation = None
        if args.validation_steps and winner_row.numel():
            validation_communication, validation_compute, validation_timing = _score_rows(
                planner=planner,
                rows=winner_row.view(1, 5),
                fallback_slots=winner_fallback,
                lut=lut,
                steps=args.validation_steps,
                layer=layer,
                rank=rank,
                device=device,
                args=args,
            )
            validation_total = validation_communication + validation_compute
            validation = {
                "baseline_ms": float(validation_total[0].item()),
                "candidate_ms": float(validation_total[1].item()),
                "gain_ms": float((validation_total[0] - validation_total[1]).item()),
                "communication_gain_ms": float(
                    (validation_communication[0] - validation_communication[1]).item()
                ),
                "compute_gain_ms": float((validation_compute[0] - validation_compute[1]).item()),
                "timing": validation_timing,
            }

        anchor_check = None
        if anchor is not None and layer == anchor[0]:
            expected_action = anchor[1]
            comparable_action = (
                None
                if action is None
                else {
                    "source_logical": action["source_logical"],
                    "source_slot": action["source_slot"],
                    "destination_slot": action["destination_slot"],
                    "victim_logical": action["victim_logical"],
                    "target_rank": action["destination_slot"] // args.slots_per_rank,
                }
            )
            action_matches = comparable_action == expected_action
            gain_error = abs(optimize_gain - anchor[2])
            anchor_check = {
                "action_matches": action_matches,
                "gain_error_ms": gain_error,
                "expected_action": expected_action,
                "expected_gain_ms": anchor[2],
            }
            if not action_matches or gain_error > float(args.anchor_cost_atol_ms):
                raise RuntimeError(
                    f"Layer {layer} differs from the CPU anchor: action_matches={action_matches}, "
                    f"gain_error_ms={gain_error:.6f}."
                )

        row = {
            "layer": layer,
            "candidate_count": int(rows.shape[0]),
            "accepted": accepted,
            "action": action,
            "optimize": {
                "baseline_ms": baseline_cost,
                "candidate_ms": candidate_cost,
                "gain_ms": optimize_gain,
                "gain_fraction": optimize_gain / max(baseline_cost, 1e-12),
                "communication_gain_ms": float(
                    (communication[0] - communication[candidate_index + 1]).item()
                    if candidate_index >= 0
                    else 0.0
                ),
                "compute_gain_ms": float(
                    (compute[0] - compute[candidate_index + 1]).item()
                    if candidate_index >= 0
                    else 0.0
                ),
            },
            "validation": validation,
            "timing": {
                **timing,
                "total_ms": timing["local_statistics_ms"] + timing["global_scoring_ms"],
                "per_sample_ms": (
                    timing["local_statistics_ms"] + timing["global_scoring_ms"]
                )
                / max(1, len(args.optimize_steps)),
            },
            "anchor_check": anchor_check,
        }
        results.append(row)
        if rank == 0:
            print(json.dumps(row, sort_keys=True), flush=True)

    if rank == 0:
        if args.output is None:
            raise ValueError("Rank 0 requires --output.")
        positive = [row for row in results if bool(row["accepted"])]
        optimize_baseline = sum(float(row["optimize"]["baseline_ms"]) for row in results)
        optimize_gain = sum(max(0.0, float(row["optimize"]["gain_ms"])) for row in results)
        validation_gain = sum(
            float(row["validation"]["gain_ms"])
            for row in positive
            if isinstance(row.get("validation"), dict)
        )
        timing_samples = [float(row["timing"]["per_sample_ms"]) for row in results]
        report = {
            "schema_version": 1,
            "algorithm": "forward-lut-exhaustive-cover-oracle-v1",
            "input_layout": str(args.input_layout.resolve()),
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
            "warmup": args.warmup,
            "layer_start": args.layer_start,
            "layers": args.layers,
            "positive_layers": len(positive),
            "optimize_baseline_ms": optimize_baseline,
            "optimize_positive_gain_ms": optimize_gain,
            "optimize_positive_gain_fraction": optimize_gain / max(optimize_baseline, 1e-12),
            "validation_selected_gain_ms": validation_gain,
            "timing_per_layer_sample_ms": {
                "median": statistics.median(timing_samples),
                "p90": _percentile(timing_samples, 0.9),
                "maximum": max(timing_samples),
            },
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
