# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Exact NPU layer-owner planning from packed sufficient statistics."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .greedy_planner import (
    GREEDY_COMMUNICATION_PHASE_MULTIPLIER,
    GREEDY_COMPUTE_PHASE_MULTIPLIER,
    GREEDY_COVER_ALGORITHM_VERSION,
    GreedyCommunicationPlanner,
)
from .planner import PlacementPlan


@dataclass(frozen=True)
class NPULayerOwnerTiming:
    context_ms: float
    local_prepare_ms: float
    statistic_pack_ms: float
    statistic_collective_ms: float
    owner_score_ms: float
    decision_collective_ms: float
    finalization_ms: float
    total_ms: float
    sent_statistic_bytes: int
    received_statistic_bytes: int
    owned_layer_count: int


@dataclass(frozen=True)
class NPULayerOwnerResult:
    plans: tuple[PlacementPlan, ...]
    timing: NPULayerOwnerTiming
    owner_ranks: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedLayer:
    context: dict[str, object]
    packed_local: torch.Tensor
    communication_width: int
    has_assignments: bool

    @property
    def flat_size(self) -> int:
        return int(self.packed_local.numel())


def balanced_layer_owner_ranks(
    layer_count: int,
    world_size: int,
    *,
    owner_offset: int = 0,
) -> tuple[int, ...]:
    if layer_count < 0 or world_size <= 0:
        raise ValueError("layer_count must be non-negative and world_size must be positive.")
    return tuple((int(owner_offset) + index) % world_size for index in range(layer_count))


class NPULayerOwnerPlanner:
    """Compute local exact statistics on all ranks and score only owner layers."""

    def __init__(
        self,
        planner: GreedyCommunicationPlanner,
        *,
        process_group: dist.ProcessGroup,
        statistic_collective: str = "reduce_scatter",
    ) -> None:
        if statistic_collective not in {"reduce_scatter", "all_to_all"}:
            raise ValueError("statistic_collective must be 'reduce_scatter' or 'all_to_all'.")
        self.planner = planner
        self.process_group = process_group
        self.statistic_collective = statistic_collective
        self.last_result: NPULayerOwnerResult | None = None

    def _build_contexts(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_rank: int,
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
    ) -> list[dict[str, object]]:
        contexts = []
        metadata_cache: dict[tuple[int, int, torch.device], dict[str, object]] = {}
        for layer_index, (raw_selected, raw_layout, raw_owners, layer_seed) in enumerate(
            zip(selected_experts, slot_to_logical, owner_slots, layer_seeds, strict=True)
        ):
            selected = raw_selected.to(torch.long)
            original_selected_ndim = selected.ndim
            if selected.ndim == 1:
                selected = selected.unsqueeze(-1)
            if selected.ndim != 2:
                raise ValueError(
                    f"selected_experts[{layer_index}] must have rank 1 or 2, got {tuple(selected.shape)}."
                )
            device = selected.device
            metadata_key = (id(raw_layout), id(raw_owners), device)
            metadata = metadata_cache.get(metadata_key)
            if metadata is None:
                host_layout = raw_layout.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
                host_owners = raw_owners.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
                if bool((host_layout < 0).any().item()):
                    raise ValueError("NPU layer-owner planning only supports initialized layouts.")
                all_slots = torch.arange(host_layout.numel(), dtype=torch.long)
                owner_mask = torch.zeros((host_layout.numel(),), dtype=torch.bool)
                owner_mask.scatter_(0, host_owners, True)
                row_blocks = []
                if max(0, int(max_swaps)) > 0:
                    row_blocks.append(self.planner._swap_rows(host_layout, host_owners))
                if max(0, int(max_replicas)) > 0:
                    destinations = all_slots[(~owner_mask) & (host_layout >= 0)]
                    row_blocks.append(self.planner._cover_rows(host_layout, host_owners, destinations))
                nonempty = [block for block in row_blocks if block.numel()]
                host_rows = (
                    torch.cat(nonempty, dim=0)
                    if nonempty
                    else torch.empty((0, 5), dtype=torch.long)
                )
                host_copy_slots = self.planner._copy_table(host_layout, int(host_owners.numel()))
                metadata = {
                    "host_layout": host_layout,
                    "host_owners": host_owners,
                    "host_rows": host_rows,
                    "layout": host_layout.to(device=device, non_blocking=True),
                    "owners": host_owners.to(device=device, non_blocking=True),
                    "rows": host_rows.to(device=device, non_blocking=True),
                    "copy_slots": host_copy_slots.to(device=device, non_blocking=True),
                }
                metadata_cache[metadata_key] = metadata
            contexts.append(
                {
                    "selected": selected,
                    "original_selected_ndim": original_selected_ndim,
                    **metadata,
                    "sources": torch.full(
                        (selected.shape[0],),
                        source_rank,
                        dtype=torch.long,
                        device=device,
                    ),
                    "uniform_source_rank": source_rank,
                    "ordinals": torch.arange(selected.shape[0], dtype=torch.long, device=device),
                    "layer_seed": int(layer_seed),
                }
            )
        return contexts

    def _finalize(
        self,
        prepared: Sequence[_PreparedLayer],
        decision: torch.Tensor,
        *,
        timing: NPULayerOwnerTiming,
    ) -> tuple[PlacementPlan, ...]:
        rows = decision.detach().to(device="cpu").tolist()
        layer_count = max(1, len(prepared))
        per_layer_total = timing.total_ms / layer_count
        per_layer_prepare = (timing.context_ms + timing.local_prepare_ms) / layer_count
        per_layer_decision = timing.decision_collective_ms / layer_count
        per_owned_score = timing.owner_score_ms / max(1, timing.owned_layer_count)
        plans = []
        for layer, row in zip(prepared, rows, strict=True):
            winner = int(row[0]) - 1
            baseline_cost = self.planner._placement_cost_from_values(row[1:7])
            candidate_cost = self.planner._placement_cost_from_values(row[7:13])
            context = layer.context
            host_layout = context["host_layout"]
            host_owners = context["host_owners"]
            final_layout = host_layout.clone()
            final_owners = host_owners.clone()
            actions = ()
            final_cost = baseline_cost
            if winner >= 0 and candidate_cost.total < baseline_cost.total:
                action = self.planner._placement_action(context["host_rows"][winner].tolist())
                actions = (action,)
                final_cost = candidate_cost
                if action.kind == "swap":
                    final_layout[action.src_slot] = action.dst_logical
                    final_layout[action.dst_slot] = action.src_logical
                    source_owner = final_owners[action.src_logical].clone()
                    final_owners[action.src_logical] = final_owners[action.dst_logical]
                    final_owners[action.dst_logical] = source_owner
                else:
                    final_layout[action.dst_slot] = action.src_logical
            chose_swap = bool(actions) and actions[0].kind == "swap"
            chose_cover = bool(actions) and not chose_swap
            plans.append(
                PlacementPlan(
                    actions=actions,
                    initial_layout=tuple(int(value) for value in host_layout.tolist()),
                    final_layout=tuple(int(value) for value in final_layout.tolist()),
                    baseline_cost=baseline_cost,
                    final_cost=final_cost,
                    swap_rounds=int(chose_swap),
                    replica_rounds=int(chose_cover),
                    planning_ms=per_layer_total,
                    route_stats_ms=per_layer_prepare,
                    swap_ms=per_owned_score if chose_swap else 0.0,
                    replica_ms=per_owned_score if chose_cover else 0.0,
                    swap_score_ms=per_owned_score,
                    swap_update_ms=0.0,
                    swap_collective_ms=timing.statistic_collective_ms / layer_count,
                    replica_score_ms=0.0,
                    replica_update_ms=0.0,
                    replica_collective_ms=0.0,
                    decision_sync_ms=per_layer_decision,
                    finalization_ms=timing.finalization_ms / layer_count,
                    algorithm_version=GREEDY_COVER_ALGORITHM_VERSION,
                    local_physical_routes=None,
                    final_owner_slots=tuple(int(value) for value in final_owners.tolist()),
                )
            )
        return tuple(plans)

    def _score_owned_layers(
        self,
        prepared: Sequence[_PreparedLayer],
        global_by_layer: dict[int, torch.Tensor],
        owned_indices: Sequence[int],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
        decision: torch.Tensor,
    ) -> None:
        """Score all locally owned layers in one accelerator batch.

        This mirrors ``GreedyCommunicationPlanner._score_prepared_layers`` after
        its reduction.  In particular, it avoids one eager scoring graph and one
        ``argmin().item()`` synchronization per owned layer.
        """

        if not owned_indices:
            return
        communication_width = prepared[owned_indices[0]].communication_width
        if any(prepared[index].communication_width != communication_width for index in owned_indices):
            raise ValueError("Owned layers must use the same communication-statistic width.")
        include_assignments = prepared[owned_indices[0]].has_assignments
        if any(prepared[index].has_assignments != include_assignments for index in owned_indices):
            raise ValueError("Owned layers must agree on whether assignment statistics are present.")

        communication_blocks = []
        assignment_blocks = []
        row_counts = []
        for layer_index in owned_indices:
            rows = global_by_layer[layer_index]
            communication_blocks.append(rows[:, :communication_width])
            if include_assignments:
                assignment_blocks.append(rows[:, communication_width:])
            row_counts.append(int(rows.shape[0]))
        communication_rows = torch.cat(communication_blocks, dim=0)
        assignment_rows = torch.cat(assignment_blocks, dim=0) if include_assignments else None

        _unused, units, peak_rank, selected_dim = self.planner._communication_cost_details(communication_rows)
        device = communication_rows.device
        layer_indices = torch.repeat_interleave(
            torch.arange(len(owned_indices), dtype=torch.long, device=device),
            torch.tensor(row_counts, dtype=torch.long, device=device),
        )
        scale_rows = torch.tensor(
            [communication_scales[index] for index in owned_indices],
            dtype=units.dtype,
            device=device,
        ).index_select(0, layer_indices)
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * scale_rows * units
        if assignment_rows is None:
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * torch.tensor(
                [forward_compute_constant[index] for index in owned_indices],
                dtype=units.dtype,
                device=device,
            ).index_select(0, layer_indices)
            peak_compute_rank = torch.full_like(peak_rank, -1)
        else:
            peak_assignments, peak_compute_rank = assignment_rows.max(dim=1)
            slope_rows = torch.tensor(
                [forward_compute_per_assignment[index] for index in owned_indices],
                dtype=units.dtype,
                device=device,
            ).index_select(0, layer_indices)
            constant_rows = torch.tensor(
                [forward_compute_constant[index] for index in owned_indices],
                dtype=units.dtype,
                device=device,
            ).index_select(0, layer_indices)
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (slope_rows * peak_assignments + constant_rows)

        metrics = torch.stack(
            (
                communication,
                compute,
                units,
                peak_rank.to(communication.dtype),
                peak_compute_rank.to(communication.dtype),
                selected_dim.to(communication.dtype),
            ),
            dim=1,
        )
        candidate_counts = [row_count - 1 for row_count in row_counts]
        maximum_candidates = max(1, max(candidate_counts))
        padded_total = torch.full(
            (len(owned_indices), maximum_candidates),
            torch.inf,
            dtype=communication.dtype,
            device=device,
        )
        baseline_metrics = []
        candidate_metrics = []
        offset = 0
        for owned_position, candidate_count in enumerate(candidate_counts):
            layer_metrics = metrics[offset : offset + row_counts[owned_position]]
            layer_total = communication[offset : offset + row_counts[owned_position]]
            layer_total = layer_total + compute[offset : offset + row_counts[owned_position]]
            if candidate_count:
                padded_total[owned_position, :candidate_count] = layer_total[1:]
            baseline_metrics.append(layer_metrics[0])
            offset += row_counts[owned_position]
        best_positions = padded_total.argmin(dim=1)

        offset = 0
        for owned_position, candidate_count in enumerate(candidate_counts):
            layer_metrics = metrics[offset : offset + row_counts[owned_position]]
            candidate_metrics.append(
                layer_metrics[best_positions[owned_position] + 1] if candidate_count else layer_metrics[0]
            )
            offset += row_counts[owned_position]
        owned_tensor = torch.tensor(owned_indices, dtype=torch.long, device=device)
        decision[owned_tensor, 0] = (best_positions + 1).to(decision.dtype)
        decision[owned_tensor, 1:7] = torch.stack(baseline_metrics)
        decision[owned_tensor, 7:13] = torch.stack(candidate_metrics)

    @torch.no_grad()
    def plan_layers(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_rank: int,
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
        step: int = 0,
        communication_scales: Sequence[float] | None = None,
        forward_compute_per_assignment: Sequence[float] | None = None,
        forward_compute_constant: Sequence[float] | None = None,
        owner_offset: int = 0,
    ) -> NPULayerOwnerResult:
        started = time.perf_counter()
        layer_count = len(selected_experts)
        if not (len(slot_to_logical) == len(owner_slots) == len(layer_seeds) == layer_count):
            raise ValueError("NPU layer-owner planner inputs must have identical layer counts.")
        if layer_count == 0:
            result = NPULayerOwnerResult(
                (),
                NPULayerOwnerTiming(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
                (),
            )
            self.last_result = result
            return result
        if not dist.is_initialized():
            raise RuntimeError("NPU layer-owner planning requires an initialized process group.")
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        if int(source_rank) != rank:
            raise ValueError(f"source_rank must equal the EP rank, got {source_rank} and {rank}.")
        scales = (
            [self.planner.communication_scale] * layer_count
            if communication_scales is None
            else [float(value) for value in communication_scales]
        )
        compute_slopes = (
            [self.planner.forward_compute_per_assignment] * layer_count
            if forward_compute_per_assignment is None
            else [float(value) for value in forward_compute_per_assignment]
        )
        compute_constants = (
            [self.planner.forward_compute_constant] * layer_count
            if forward_compute_constant is None
            else [float(value) for value in forward_compute_constant]
        )
        if not (len(scales) == len(compute_slopes) == len(compute_constants) == layer_count):
            raise ValueError("NPU layer-owner cost-model arrays must match the number of layers.")
        layer_owners = balanced_layer_owner_ranks(layer_count, world_size, owner_offset=owner_offset)
        owned_indices = tuple(index for index, owner in enumerate(layer_owners) if owner == rank)
        include_assignments = any(value > 0.0 for value in compute_slopes)

        context_started = time.perf_counter()
        contexts = self._build_contexts(
            selected_experts,
            slot_to_logical,
            owner_slots,
            source_rank=source_rank,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
            layer_seeds=layer_seeds,
        )
        context_ms = (time.perf_counter() - context_started) * 1000.0

        prepare_started = time.perf_counter()
        counts = self.planner._prepare_independent_layers(
            contexts,
            step=step,
            include_assignment_counts=include_assignments,
            include_pair_interactions=True,
            include_pair_bounds=False,
        )
        prepared = []
        for context, layer_counts in zip(contexts, counts, strict=True):
            communication = torch.cat(
                (layer_counts.baseline_local, layer_counts.candidate_local),
                dim=0,
            )
            communication_width = int(communication.shape[1])
            if include_assignments:
                if (
                    layer_counts.baseline_assignment_local is None
                    or layer_counts.candidate_assignment_local is None
                ):
                    raise RuntimeError("NPU layer-owner planning requested missing assignment statistics.")
                assignments = torch.cat(
                    (
                        layer_counts.baseline_assignment_local,
                        layer_counts.candidate_assignment_local,
                    ),
                    dim=0,
                )
                packed = torch.cat((communication, assignments), dim=1)
            else:
                packed = communication
            prepared.append(
                _PreparedLayer(
                    context=context,
                    packed_local=packed.contiguous(),
                    communication_width=communication_width,
                    has_assignments=include_assignments,
                )
            )
        local_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        pack_started = time.perf_counter()
        layers_by_owner = [
            [index for index, owner in enumerate(layer_owners) if owner == destination]
            for destination in range(world_size)
        ]
        if self.statistic_collective == "reduce_scatter":
            # Dynamic cover actions can make the legal candidate count differ by
            # layer. Reduce-scatter still requires equal destination chunks, so
            # pad every layer to one batch-wide stride and unpack only its real
            # prefix on the owner.
            flat_size = max(layer.flat_size for layer in prepared)
            maximum_owned_layers = max(len(indices) for indices in layers_by_owner)
            zero_layer = torch.zeros(
                (flat_size,),
                dtype=prepared[0].packed_local.dtype,
                device=prepared[0].packed_local.device,
            )
            send_chunks = []
            for indices in layers_by_owner:
                values = []
                for index in indices:
                    value = prepared[index].packed_local.reshape(-1)
                    if int(value.numel()) < flat_size:
                        value = torch.cat((value, zero_layer[: flat_size - int(value.numel())]))
                    values.append(value)
                values.extend(zero_layer for _ in range(maximum_owned_layers - len(values)))
                send_chunks.append(torch.cat(values))
            send_buffer = torch.cat(send_chunks).contiguous()
            owned_flat_size = maximum_owned_layers * flat_size
            recv_buffer = torch.empty(
                (owned_flat_size,),
                dtype=send_buffer.dtype,
                device=send_buffer.device,
            )
            input_splits = None
        else:
            input_splits = [
                sum(prepared[index].flat_size for index in indices) for indices in layers_by_owner
            ]
            send_chunks = [
                torch.cat([prepared[index].packed_local.reshape(-1) for index in indices])
                if indices
                else torch.empty(
                    (0,),
                    dtype=prepared[0].packed_local.dtype,
                    device=prepared[0].packed_local.device,
                )
                for indices in layers_by_owner
            ]
            send_buffer = torch.cat(send_chunks).contiguous()
            owned_flat_size = input_splits[rank]
            recv_buffer = torch.empty(
                (world_size * owned_flat_size,),
                dtype=send_buffer.dtype,
                device=send_buffer.device,
            )
        statistic_pack_ms = (time.perf_counter() - pack_started) * 1000.0

        collective_started = time.perf_counter()
        if self.statistic_collective == "reduce_scatter":
            dist.reduce_scatter_tensor(
                recv_buffer,
                send_buffer,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
            reduced = recv_buffer
        else:
            assert input_splits is not None
            dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=[owned_flat_size] * world_size,
                input_split_sizes=input_splits,
                group=self.process_group,
            )
            reduced = recv_buffer.view(world_size, owned_flat_size).sum(dim=0)
        global_by_layer = {}
        offset = 0
        for layer_index in owned_indices:
            size = prepared[layer_index].flat_size
            global_by_layer[layer_index] = reduced[offset : offset + size].view_as(
                prepared[layer_index].packed_local
            )
            offset += flat_size if self.statistic_collective == "reduce_scatter" else size
        if self.statistic_collective == "all_to_all" and offset != owned_flat_size:
            raise RuntimeError("NPU layer-owner statistic unpack consumed an unexpected number of values.")
        if self.statistic_collective == "reduce_scatter" and offset > owned_flat_size:
            raise RuntimeError("NPU layer-owner statistic unpack exceeded the reduce-scatter output.")
        statistic_collective_ms = (time.perf_counter() - collective_started) * 1000.0

        owner_score_started = time.perf_counter()
        # Keep the decision payload in the scorer's native dtype.  HCCL does not
        # support float64 all-reduce, and converting float32 cost outputs through
        # float64 would not preserve any additional information.
        decision = torch.zeros(
            (layer_count, 13),
            dtype=send_buffer.dtype,
            device=send_buffer.device,
        )
        self._score_owned_layers(
            prepared,
            global_by_layer,
            owned_indices,
            communication_scales=scales,
            forward_compute_per_assignment=compute_slopes,
            forward_compute_constant=compute_constants,
            decision=decision,
        )
        owner_score_ms = (time.perf_counter() - owner_score_started) * 1000.0

        decision_started = time.perf_counter()
        dist.all_reduce(decision, op=dist.ReduceOp.SUM, group=self.process_group)
        decision_collective_ms = (time.perf_counter() - decision_started) * 1000.0
        pre_finalize_total = (time.perf_counter() - started) * 1000.0
        provisional_timing = NPULayerOwnerTiming(
            context_ms=context_ms,
            local_prepare_ms=local_prepare_ms,
            statistic_pack_ms=statistic_pack_ms,
            statistic_collective_ms=statistic_collective_ms,
            owner_score_ms=owner_score_ms,
            decision_collective_ms=decision_collective_ms,
            finalization_ms=0.0,
            total_ms=pre_finalize_total,
            sent_statistic_bytes=int(send_buffer.numel() * send_buffer.element_size()),
            received_statistic_bytes=int(recv_buffer.numel() * recv_buffer.element_size()),
            owned_layer_count=len(owned_indices),
        )
        finalization_started = time.perf_counter()
        plans = self._finalize(prepared, decision, timing=provisional_timing)
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = NPULayerOwnerTiming(
            **{
                **vars(provisional_timing),
                "finalization_ms": finalization_ms,
                "total_ms": total_ms,
            }
        )
        result = NPULayerOwnerResult(plans, timing, layer_owners)
        self.last_result = result
        return result
