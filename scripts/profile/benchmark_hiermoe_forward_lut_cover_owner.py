#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Exact Cover-only planner with source statistics sharded by layer owner.

Each source rank builds candidate-independent ``zero/sole/cohit`` statistics
for every layer from the cached Forward logical route and LUT.  One packed
all-to-all sends each layer's statistics to its owner rank.  Owners score all
legal Cover actions locally; only the tiny per-layer decision tensor is
all-reduced.

This benchmark intentionally contains no candidate-statistic collective.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.profile.benchmark_hiermoe_forward_lut_cover_oracle import (
    _action_dict,
    _anchor_expectation,
    _layer_state,
    _legal_cover_rows,
    _load_route,
    _parse_int_list,
    _source_endpoint_statistics,
)
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.statistical_scorer import (
    ForwardLUTCoverCompactStatistics,
    prepare_forward_lut_cover_compact_statistics,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


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


def _initialize(ep_size: int) -> tuple[int, torch.device]:
    importlib.import_module("torch_npu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != int(ep_size):
        raise ValueError(f"Expected EP size {ep_size}, got {dist.get_world_size()}.")
    return rank, device


def _synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _flatten_statistics(statistics: ForwardLUTCoverCompactStatistics) -> torch.Tensor:
    fields: list[torch.Tensor] = []
    for level in range(len(statistics.baseline_counts)):
        fields.extend(
            (
                statistics.baseline_counts[level].reshape(-1),
                statistics.zero_group_expert_hits[level].reshape(-1),
                statistics.sole_expert_hits[level].reshape(-1),
                statistics.sole_expert_cohits[level].reshape(-1),
            )
        )
    fields.append(statistics.assignment_multiplicity.reshape(-1))
    return torch.cat(fields).contiguous()


def _unpack_statistics(
    flat: torch.Tensor,
    *,
    planner: GreedyCommunicationPlanner,
    source_lut: torch.Tensor,
    num_experts: int,
) -> ForwardLUTCoverCompactStatistics:
    level_sizes = (1,) + tuple(
        int(size)
        for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    baseline: list[torch.Tensor] = []
    zero: list[torch.Tensor] = []
    sole: list[torch.Tensor] = []
    cohit: list[torch.Tensor] = []
    group_by_logical: list[torch.Tensor] = []
    offset = 0
    source_ranks = torch.div(
        source_lut.to(device=flat.device, dtype=torch.long),
        int(planner.slots_per_rank),
        rounding_mode="floor",
    )
    for size in level_sizes:
        num_groups = int(planner.ep_size) // size
        baseline_end = offset + num_groups
        zero_end = baseline_end + num_groups * int(num_experts)
        sole_end = zero_end + int(num_experts)
        cohit_end = sole_end + int(num_experts) * int(num_experts)
        baseline.append(flat[offset:baseline_end])
        zero.append(flat[baseline_end:zero_end].view(num_groups, int(num_experts)))
        sole.append(flat[zero_end:sole_end])
        cohit.append(flat[sole_end:cohit_end].view(int(num_experts), int(num_experts)))
        group_by_logical.append(torch.div(source_ranks, size, rounding_mode="floor"))
        offset = cohit_end
    multiplicity_end = offset + int(num_experts)
    if multiplicity_end != int(flat.numel()):
        raise ValueError(
            f"Compact statistics consumed {multiplicity_end} values, got {int(flat.numel())}."
        )
    return ForwardLUTCoverCompactStatistics(
        baseline_counts=tuple(baseline),
        zero_group_expert_hits=tuple(zero),
        sole_expert_hits=tuple(sole),
        sole_expert_cohits=tuple(cohit),
        group_by_logical=tuple(group_by_logical),
        assignment_multiplicity=flat[offset:multiplicity_end],
    )


@dataclass(frozen=True)
class _BatchedCompactStatistics:
    baseline_counts: tuple[torch.Tensor, ...]
    zero_group_expert_hits: tuple[torch.Tensor, ...]
    sole_expert_hits: tuple[torch.Tensor, ...]
    sole_expert_cohits: tuple[torch.Tensor, ...]
    group_by_logical: tuple[torch.Tensor, ...]
    assignment_multiplicity: torch.Tensor


def _unpack_batched_statistics(
    flat: torch.Tensor,
    *,
    planner: GreedyCommunicationPlanner,
    source_lut: torch.Tensor,
    num_experts: int,
) -> _BatchedCompactStatistics:
    """View ``[source, flat]`` statistics without source-wise Python loops."""

    if flat.ndim != 2:
        raise ValueError("Batched compact statistics must have shape [source, flat].")
    level_sizes = (1,) + tuple(
        int(size)
        for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    baseline: list[torch.Tensor] = []
    zero: list[torch.Tensor] = []
    sole: list[torch.Tensor] = []
    cohit: list[torch.Tensor] = []
    group_by_logical: list[torch.Tensor] = []
    offset = 0
    source_ranks = torch.div(
        source_lut.to(device=flat.device, dtype=torch.long),
        int(planner.slots_per_rank),
        rounding_mode="floor",
    )
    for size in level_sizes:
        num_groups = int(planner.ep_size) // size
        baseline_end = offset + num_groups
        zero_end = baseline_end + num_groups * int(num_experts)
        sole_end = zero_end + int(num_experts)
        cohit_end = sole_end + int(num_experts) * int(num_experts)
        baseline.append(flat[:, offset:baseline_end])
        zero.append(flat[:, baseline_end:zero_end].view(flat.shape[0], num_groups, int(num_experts)))
        sole.append(flat[:, zero_end:sole_end])
        cohit.append(flat[:, sole_end:cohit_end].view(flat.shape[0], int(num_experts), int(num_experts)))
        group_by_logical.append(torch.div(source_ranks, size, rounding_mode="floor"))
        offset = cohit_end
    multiplicity_end = offset + int(num_experts)
    if multiplicity_end != int(flat.shape[1]):
        raise ValueError(
            f"Compact statistics consumed {multiplicity_end} values, got {int(flat.shape[1])}."
        )
    return _BatchedCompactStatistics(
        baseline_counts=tuple(baseline),
        zero_group_expert_hits=tuple(zero),
        sole_expert_hits=tuple(sole),
        sole_expert_cohits=tuple(cohit),
        group_by_logical=tuple(group_by_logical),
        assignment_multiplicity=flat[:, offset:multiplicity_end],
    )


def _batched_global_endpoint_costs(
    *,
    planner: GreedyCommunicationPlanner,
    statistics: _BatchedCompactStatistics,
    rows: torch.Tensor,
    source_logical_to_physical: torch.Tensor,
    victim_fallback_slots: torch.Tensor,
    ep_size: int,
    ranks_per_node: int,
    service_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score all Covers from all source statistics without a source loop.

    Only ``[source, candidate]`` scalar terms are materialized.  Each unary or
    cohit term is scattered directly into the final global endpoint row; the
    much larger ``[source, candidate, group]`` table never exists.
    """

    sources = int(statistics.assignment_multiplicity.shape[0])
    candidates = int(rows.shape[0])
    if sources != int(ep_size):
        raise ValueError(f"Expected {ep_size} source rows, got {sources}.")
    source_lut = source_logical_to_physical.to(device=rows.device, dtype=torch.long)
    fallback_slots = victim_fallback_slots.to(device=rows.device, dtype=torch.long)
    source_ids = torch.arange(sources, dtype=torch.long, device=rows.device).view(-1, 1)
    candidate_ids = torch.arange(candidates, dtype=torch.long, device=rows.device).view(1, -1)
    source_grid = source_ids.expand(-1, candidates)
    candidate_grid = candidate_ids.expand(sources, -1)
    lanes = torch.remainder(source_ids, int(ranks_per_node)).expand(-1, candidates)

    lhs = rows[:, 3]
    rhs = rows[:, 4]
    destination_slots = rows[:, 2]
    destination_ranks = torch.div(
        destination_slots,
        int(planner.slots_per_rank),
        rounding_mode="floor",
    )
    fallback_ranks = torch.div(
        fallback_slots,
        int(planner.slots_per_rank),
        rounding_mode="floor",
    )
    baseline_ranks = torch.div(
        source_lut,
        int(planner.slots_per_rank),
        rounding_mode="floor",
    )
    lhs_old_ranks = baseline_ranks.index_select(1, lhs)
    rhs_old_ranks = baseline_ranks.index_select(1, rhs)
    source_service_groups = torch.div(source_ids, int(service_group_size), rounding_mode="floor")
    destination_service_groups = torch.div(
        destination_ranks,
        int(service_group_size),
        rounding_mode="floor",
    ).view(1, -1)
    lhs_new_ranks = torch.where(
        source_service_groups == destination_service_groups,
        destination_ranks.view(1, -1),
        lhs_old_ranks,
    )
    rhs_slots = source_lut.index_select(1, rhs)
    rhs_new_ranks = torch.where(
        rhs_slots == destination_slots.view(1, -1),
        fallback_ranks.view(1, -1),
        rhs_old_ranks,
    )

    endpoint_delta = rows.new_zeros(
        (candidates, 8 * int(ep_size)),
        dtype=torch.float32,
    )
    endpoint_flat = endpoint_delta.reshape(-1)

    def scatter_endpoint(
        block: int,
        endpoints: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        flat_valid = valid.reshape(-1)
        flat_indices = (
            candidate_grid * (8 * int(ep_size))
            + int(block) * int(ep_size)
            + endpoints
        ).reshape(-1)
        endpoint_flat.index_add_(
            0,
            flat_indices[flat_valid],
            values.reshape(-1)[flat_valid],
        )

    level_sizes = (1,) + tuple(
        int(size)
        for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    for level, size in enumerate(level_sizes):
        group_by_logical = statistics.group_by_logical[level]
        zero = statistics.zero_group_expert_hits[level]
        sole = statistics.sole_expert_hits[level]
        cohit = statistics.sole_expert_cohits[level]
        lhs_old = group_by_logical.index_select(1, lhs)
        rhs_old = group_by_logical.index_select(1, rhs)
        lhs_new = torch.div(lhs_new_ranks, size, rounding_mode="floor")
        rhs_new = torch.div(rhs_new_ranks, size, rounding_mode="floor")
        lhs_moves = lhs_old != lhs_new
        rhs_moves = rhs_old != rhs_new

        lhs_experts = lhs.view(1, -1).expand(sources, -1)
        rhs_experts = rhs.view(1, -1).expand(sources, -1)
        lhs_loss = -sole.gather(1, lhs_experts)
        rhs_loss = -sole.gather(1, rhs_experts)
        lhs_gain = zero[source_grid, lhs_new, lhs_experts]
        rhs_gain = zero[source_grid, rhs_new, rhs_experts]
        rhs_lhs_cohit = cohit[source_grid, rhs_experts, lhs_experts]
        lhs_rhs_cohit = cohit[source_grid, lhs_experts, rhs_experts]
        both_move = lhs_moves & rhs_moves

        terms = (
            (lhs_old, lhs_loss, lhs_moves),
            (lhs_new, lhs_gain, lhs_moves),
            (rhs_old, rhs_loss, rhs_moves),
            (rhs_new, rhs_gain, rhs_moves),
            (rhs_old, rhs_lhs_cohit, both_move & (lhs_new == rhs_old)),
            (lhs_old, lhs_rhs_cohit, both_move & (rhs_new == lhs_old)),
        )
        for groups, values, valid in terms:
            if size == 1:
                group_nodes = torch.div(groups, int(ranks_per_node), rounding_mode="floor")
                scatter_endpoint(
                    4,
                    group_nodes * int(ranks_per_node) + lanes,
                    values,
                    valid,
                )
                scatter_endpoint(5, groups, values, valid)
            else:
                num_nodes = int(ep_size) // int(ranks_per_node)
                scatter_endpoint(0, source_grid, values, valid)
                scatter_endpoint(
                    1,
                    lanes * num_nodes + groups,
                    values,
                    valid,
                )

    lhs_weights = statistics.assignment_multiplicity.index_select(1, lhs)
    rhs_weights = statistics.assignment_multiplicity.index_select(1, rhs)
    for old_ranks, new_ranks, weights in (
        (lhs_old_ranks, lhs_new_ranks, lhs_weights),
        (rhs_old_ranks, rhs_new_ranks, rhs_weights),
    ):
        moves = old_ranks != new_ranks
        for ranks, sign in ((old_ranks, -1.0), (new_ranks, 1.0)):
            values = weights * sign
            nodes = torch.div(ranks, int(ranks_per_node), rounding_mode="floor")
            num_nodes = int(ep_size) // int(ranks_per_node)
            scatter_endpoint(
                3,
                lanes * num_nodes + nodes,
                values,
                moves,
            )
            scatter_endpoint(
                6,
                nodes * int(ranks_per_node) + lanes,
                values,
                moves,
            )
            scatter_endpoint(7, ranks, values, moves)

    baseline_unique = torch.cat(statistics.baseline_counts, dim=1)
    baseline_assignment = baseline_unique.new_zeros((sources, int(ep_size)))
    baseline_assignment.scatter_add_(
        1,
        statistics.group_by_logical[0],
        statistics.assignment_multiplicity,
    )
    # Reuse the source-aware endpoint projection on one baseline row.  The
    # candidate delta above is already globally accumulated.
    baseline_endpoint = _source_endpoint_statistics(
        unique_counts=baseline_unique.unsqueeze(1),
        assignment_counts=baseline_assignment.unsqueeze(1),
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
    ).squeeze(0)
    return _restore_and_score(
        planner=planner,
        baseline_endpoint=baseline_endpoint,
        packed_endpoint_delta=endpoint_delta,
        affected_endpoint_ids=torch.arange(
            8 * int(ep_size),
            dtype=torch.long,
            device=rows.device,
        )
        .view(1, -1)
        .expand(candidates, -1),
    )


def _restore_and_score(
    *,
    planner: GreedyCommunicationPlanner,
    baseline_endpoint: torch.Tensor,
    packed_endpoint_delta: torch.Tensor,
    affected_endpoint_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = int(packed_endpoint_delta.shape[0])
    valid = affected_endpoint_ids >= 0
    safe_ids = affected_endpoint_ids.clamp_min(0)
    candidate_endpoint = baseline_endpoint.unsqueeze(0).expand(candidates, -1).clone()
    candidate_rows = torch.arange(candidates, dtype=torch.long, device=baseline_endpoint.device).view(-1, 1)
    candidate_endpoint.reshape(-1).index_add_(
        0,
        (candidate_rows * candidate_endpoint.shape[1] + safe_ids).reshape(-1),
        (packed_endpoint_delta * valid.to(packed_endpoint_delta.dtype)).reshape(-1),
    )
    baseline_details = planner._traffic_endpoint_cost_details(baseline_endpoint.unsqueeze(0))
    candidate_details = planner._traffic_endpoint_cost_details(candidate_endpoint)
    return (
        torch.cat((baseline_details[0], candidate_details[0]), dim=0),
        torch.cat((baseline_details[1], candidate_details[1]), dim=0),
    )


def _prepare_local_statistics(
    *,
    planner: GreedyCommunicationPlanner,
    payload: dict[str, object],
    steps: tuple[int, ...],
    layer_indices: tuple[int, ...],
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    layer_context: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sample_rows: list[torch.Tensor] = []
    for step in steps:
        layer_rows: list[torch.Tensor] = []
        for layer in layer_indices:
            layout, owners, lut = _layer_state(payload, layer=layer, device=device, args=args)
            if step == steps[0]:
                layer_context.append((layout, owners, lut))
            selected = _load_route(
                args.route_root,
                step=step,
                layer=layer,
                rank=rank,
                device=device,
            )
            statistics = prepare_forward_lut_cover_compact_statistics(
                planner,
                selected,
                source_logical_to_physical=lut[rank],
                num_experts=args.num_experts,
            )
            layer_rows.append(_flatten_statistics(statistics))
        sample_rows.append(torch.stack(layer_rows, dim=0))
    return torch.stack(sample_rows, dim=0).contiguous(), layer_context


def _exchange_to_layer_owners(
    local_statistics: torch.Tensor,
    *,
    layer_indices: tuple[int, ...],
    rank: int,
    ep_size: int,
) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    owners = tuple(index % int(ep_size) for index in range(len(layer_indices)))
    by_owner = [
        [index for index, owner in enumerate(owners) if owner == destination]
        for destination in range(int(ep_size))
    ]
    samples, _layers, flat_size = local_statistics.shape
    chunks = [
        local_statistics[:, indices, :].contiguous().reshape(-1)
        if indices
        else local_statistics.new_empty((0,))
        for indices in by_owner
    ]
    input_splits = [int(chunk.numel()) for chunk in chunks]
    owned_positions = tuple(by_owner[rank])
    per_source_receive = samples * len(owned_positions) * flat_size
    output_splits = [per_source_receive] * int(ep_size)
    send = torch.cat(chunks).contiguous()
    receive = local_statistics.new_empty((sum(output_splits),))
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    if owned_positions:
        receive = receive.view(int(ep_size), samples, len(owned_positions), flat_size)
    else:
        receive = receive.view(int(ep_size), samples, 0, flat_size)
    return receive, owned_positions, int(send.numel() * send.element_size()), int(receive.numel() * receive.element_size())


def _score_owned_layers(
    *,
    planner: GreedyCommunicationPlanner,
    received: torch.Tensor,
    owned_positions: tuple[int, ...],
    layer_indices: tuple[int, ...],
    contexts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    decision = received.new_zeros((len(layer_indices), 7))
    samples = int(received.shape[1])
    for owner_position, layer_position in enumerate(owned_positions):
        _layout, _owners, lut = contexts[layer_position]
        rows, fallbacks = _legal_cover_rows(
            planner,
            contexts[layer_position][0],
            contexts[layer_position][1],
            num_experts=args.num_experts,
        )
        total_communication = rows.new_zeros((rows.shape[0] + 1,), dtype=torch.float32)
        total_compute = rows.new_zeros((rows.shape[0] + 1,), dtype=torch.float32)
        for sample in range(samples):
            source_statistics = _unpack_batched_statistics(
                received[:, sample, owner_position],
                planner=planner,
                source_lut=lut,
                num_experts=args.num_experts,
            )
            communication, compute = _batched_global_endpoint_costs(
                planner=planner,
                statistics=source_statistics,
                rows=rows,
                source_logical_to_physical=lut,
                victim_fallback_slots=fallbacks,
                ep_size=args.ep_size,
                ranks_per_node=args.ranks_per_node,
                service_group_size=args.service_group_size,
            )
            total_communication += communication
            total_compute += compute
        communication = total_communication / float(samples)
        compute = total_compute / float(samples)
        total = communication + compute
        if rows.numel():
            winner = int(total[1:].argmin().item())
            decision[layer_position, 0] = float(winner + 1)
            decision[layer_position, 1] = total[0]
            decision[layer_position, 2] = total[winner + 1]
            decision[layer_position, 3] = communication[0]
            decision[layer_position, 4] = communication[winner + 1]
            decision[layer_position, 5] = compute[0]
            decision[layer_position, 6] = compute[winner + 1]
    return decision


def _run(
    *,
    planner: GreedyCommunicationPlanner,
    payload: dict[str, object],
    steps: tuple[int, ...],
    layer_indices: tuple[int, ...],
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, float | int]]:
    _synchronize(device)
    started = time.perf_counter()
    local_statistics, contexts = _prepare_local_statistics(
        planner=planner,
        payload=payload,
        steps=steps,
        layer_indices=layer_indices,
        rank=rank,
        device=device,
        args=args,
    )
    _synchronize(device)
    prepare_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    received, owned_positions, sent_bytes, received_bytes = _exchange_to_layer_owners(
        local_statistics,
        layer_indices=layer_indices,
        rank=rank,
        ep_size=args.ep_size,
    )
    _synchronize(device)
    collective_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    decision = _score_owned_layers(
        planner=planner,
        received=received,
        owned_positions=owned_positions,
        layer_indices=layer_indices,
        contexts=contexts,
        args=args,
    )
    _synchronize(device)
    owner_score_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    dist.all_reduce(decision, op=dist.ReduceOp.SUM)
    _synchronize(device)
    decision_ms = (time.perf_counter() - started) * 1000.0
    timing_tensor = torch.tensor(
        [prepare_ms, collective_ms, owner_score_ms, decision_ms],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
    timing = {
        "local_prepare_ms": float(timing_tensor[0].item()),
        "statistic_collective_ms": float(timing_tensor[1].item()),
        "owner_score_ms": float(timing_tensor[2].item()),
        "decision_collective_ms": float(timing_tensor[3].item()),
        "total_ms": float(timing_tensor.sum().item()),
        "sent_statistic_bytes": sent_bytes,
        "received_statistic_bytes": received_bytes,
        "owned_layer_count": len(owned_positions),
    }
    return decision, contexts, timing


def main() -> None:
    args = _parse_args()
    rank, device = _initialize(args.ep_size)
    if args.layers <= 0:
        raise ValueError("layers must be positive.")
    layer_indices = tuple(range(args.layer_start, args.layer_start + args.layers))
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    anchor = _anchor_expectation(args.anchor)
    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=(args.ranks_per_node, args.ep_size),
            source="forward-lut-cover-owner",
            local_world_size=args.ranks_per_node,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=args.hidden_size,
        bytes_per_element=args.bytes_per_element,
        slots_per_rank=args.slots_per_rank,
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
    if args.warmup:
        _run(
            planner=planner,
            payload=payload,
            steps=(args.optimize_steps[0],),
            layer_indices=(layer_indices[0],),
            rank=rank,
            device=device,
            args=args,
        )
    wall_started = time.perf_counter()
    decision, contexts, timing = _run(
        planner=planner,
        payload=payload,
        steps=args.optimize_steps,
        layer_indices=layer_indices,
        rank=rank,
        device=device,
        args=args,
    )
    rows_by_layer = [
        _legal_cover_rows(
            planner,
            layout,
            owners,
            num_experts=args.num_experts,
        )[0]
        for layout, owners, _lut in contexts
    ]
    results: list[dict[str, object]] = []
    for position, (layer, rows) in enumerate(zip(layer_indices, rows_by_layer, strict=True)):
        winner = int(decision[position, 0].item()) - 1
        baseline = float(decision[position, 1].item())
        candidate = float(decision[position, 2].item())
        gain = baseline - candidate
        action = None if winner < 0 else _action_dict(rows[winner])
        if action is not None:
            action["target_rank"] = action["destination_slot"] // args.slots_per_rank
        anchor_check = None
        if anchor is not None and layer == anchor[0]:
            comparable = (
                None
                if action is None
                else {
                    "source_logical": action["source_logical"],
                    "source_slot": action["source_slot"],
                    "destination_slot": action["destination_slot"],
                    "victim_logical": action["victim_logical"],
                    "target_rank": action["target_rank"],
                }
            )
            anchor_check = {
                "action_matches": comparable == anchor[1],
                "gain_error_ms": abs(gain - anchor[2]),
                "expected_action": anchor[1],
                "expected_gain_ms": anchor[2],
            }
            if not anchor_check["action_matches"] or anchor_check["gain_error_ms"] > args.anchor_cost_atol_ms:
                raise RuntimeError(f"Layer {layer} differs from CPU anchor: {anchor_check}.")
        row = {
            "layer": layer,
            "candidate_count": int(rows.shape[0]),
            "accepted": gain > 0.0,
            "action": action,
            "baseline_ms": baseline,
            "candidate_ms": candidate,
            "gain_ms": gain,
            "communication_gain_ms": float(decision[position, 3] - decision[position, 4]),
            "compute_gain_ms": float(decision[position, 5] - decision[position, 6]),
            "anchor_check": anchor_check,
        }
        results.append(row)
        if rank == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
    if rank == 0:
        if args.output is None:
            raise ValueError("Rank 0 requires --output.")
        positive = [row for row in results if bool(row["accepted"])]
        gains = [float(row["gain_ms"]) for row in positive]
        per_online_sample = (
            timing["local_prepare_ms"] / max(1, len(args.optimize_steps))
            + timing["statistic_collective_ms"]
            + timing["owner_score_ms"]
            + timing["decision_collective_ms"]
        )
        report = {
            "schema_version": 1,
            "algorithm": "forward-lut-cover-layer-owner-zero-sole-cohit-v1",
            "input_layout": str(args.input_layout.resolve()),
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "layer_start": args.layer_start,
            "layers": args.layers,
            "positive_layers": len(positive),
            "positive_gain_ms": sum(gains),
            "median_positive_gain_ms": statistics.median(gains) if gains else 0.0,
            "timing": timing,
            "estimated_online_one_sample_ms": per_online_sample,
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
