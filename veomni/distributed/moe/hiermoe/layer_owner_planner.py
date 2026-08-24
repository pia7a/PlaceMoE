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

from .greedy_planner import GREEDY_COVER_ALGORITHM_VERSION, GreedyCommunicationPlanner
from .planner import PlacementAction, PlacementCost, PlacementPlan


@dataclass(frozen=True)
class LayerOwnerTiming:
    """Host-observed stage breakdown for one layer-owner planning round."""

    metadata_collective_ms: float
    route_pack_ms: float
    route_collective_ms: float
    route_unpack_ms: float
    owner_planning_ms: float
    action_collective_ms: float
    finalization_ms: float
    total_ms: float
    sent_route_bytes: int
    received_route_bytes: int
    owned_layer_count: int


@dataclass(frozen=True)
class LayerOwnerResult:
    plans: tuple[PlacementPlan, ...]
    timing: LayerOwnerTiming
    owner_ranks: tuple[int, ...]


_PLAN_INTEGER_FIELDS = 9
_PLAN_FLOAT_FIELDS = 28
_PLAN_FIELDS = _PLAN_INTEGER_FIELDS + _PLAN_FLOAT_FIELDS
_ACTION_CODES = {"none": 1, "swap": 2, "replica": 3, "empty": 4}
_ACTION_KINDS = {value: key for key, value in _ACTION_CODES.items()}


def balanced_layer_owner_ranks(
    layer_count: int,
    world_size: int,
    *,
    owner_offset: int = 0,
) -> tuple[int, ...]:
    """Assign consecutive layers round-robin to EP ranks."""

    if layer_count < 0 or world_size <= 0:
        raise ValueError("layer_count must be non-negative and world_size must be positive.")
    return tuple((int(owner_offset) + index) % world_size for index in range(layer_count))


def _clone_for_global_routes(
    planner: GreedyCommunicationPlanner,
) -> GreedyCommunicationPlanner:
    return GreedyCommunicationPlanner(
        hierarchy=planner.hierarchy,
        perf_model=planner.perf_model,
        hidden_size=planner.hidden_size,
        bytes_per_element=planner.bytes_per_element,
        slots_per_rank=planner.slots_per_rank,
        communication_scale=planner.communication_scale,
        forward_compute_per_assignment=planner.forward_compute_per_assignment,
        forward_compute_constant=planner.forward_compute_constant,
        smooth_max_gamma=planner.smooth_max_gamma,
        reducer=None,
        candidate_chunk_size=planner.candidate_chunk_size,
        process_group=None,
        max_copies=planner.max_copies,
        candidate_scorer=planner.candidate_scorer,
        compact_candidate_collective=False,
        assume_unique_routes=planner.assume_unique_routes,
        layer_parallel_streams=planner.layer_parallel_streams,
        adaptive_topk=planner.adaptive_topk,
        adaptive_topk_initial=planner.adaptive_topk_initial,
        adaptive_topk_growth_factor=planner.adaptive_topk_growth_factor,
        adaptive_topk_epsilon=planner.adaptive_topk_epsilon,
        adaptive_topk_strict_certificate=planner.adaptive_topk_strict_certificate,
        early_proxy_topk=planner.early_proxy_topk,
        exact_primitive_topk=planner.exact_primitive_topk,
        post_shortlist_compact_pair=planner.post_shortlist_compact_pair,
        exact_primitive_max_only=planner.exact_primitive_max_only,
    )


def _encode_plan(plan: PlacementPlan) -> torch.Tensor:
    action = plan.actions[0] if plan.actions else None
    action_code = _ACTION_CODES["none" if action is None else action.kind]
    integer_values = (
        1,
        action_code,
        0 if action is None else action.src_slot,
        0 if action is None else action.dst_slot,
        0 if action is None else action.src_logical,
        -1 if action is None else action.dst_logical,
        plan.swap_rounds,
        plan.replica_rounds,
        len(plan.actions),
    )

    def cost_values(cost: PlacementCost) -> tuple[float, ...]:
        return (
            cost.communication,
            cost.compute,
            cost.communication_model_units,
            float(cost.peak_communication_rank),
            float(cost.peak_compute_rank),
            float(cost.selected_dim),
            cost.state_move_exposed,
            cost.gradient_sync,
        )

    timing_values = (
        plan.planning_ms,
        plan.route_stats_ms,
        plan.swap_ms,
        plan.replica_ms,
        plan.swap_score_ms,
        plan.swap_update_ms,
        plan.swap_collective_ms,
        plan.replica_score_ms,
        plan.replica_update_ms,
        plan.replica_collective_ms,
        plan.decision_sync_ms,
        plan.finalization_ms,
    )
    float_values = (*cost_values(plan.baseline_cost), *cost_values(plan.final_cost), *timing_values)
    encoded = torch.empty((_PLAN_FIELDS,), dtype=torch.int64)
    encoded[:_PLAN_INTEGER_FIELDS] = torch.tensor(integer_values, dtype=torch.int64)
    encoded[_PLAN_INTEGER_FIELDS:] = torch.tensor(float_values, dtype=torch.float64).view(torch.int64)
    return encoded


def _decode_plan(
    encoded: torch.Tensor,
    layout: torch.Tensor,
    owners: torch.Tensor,
) -> PlacementPlan:
    words = encoded.to(device="cpu", dtype=torch.int64)
    integers = [int(value) for value in words[:_PLAN_INTEGER_FIELDS].tolist()]
    if integers[0] != 1:
        raise RuntimeError("A layer-owner action row has no unique owner contribution.")
    action_kind = _ACTION_KINDS.get(integers[1])
    if action_kind is None:
        raise RuntimeError(f"Unknown layer-owner action code {integers[1]}.")
    action_count = integers[8]
    if action_count not in {0, 1}:
        raise RuntimeError("Layer-owner full-exact planning supports at most one action per layer.")
    action = (
        None
        if action_count == 0
        else PlacementAction(
            kind=action_kind,
            src_slot=integers[2],
            dst_slot=integers[3],
            src_logical=integers[4],
            dst_logical=integers[5],
        )
    )
    if (action is None) != (action_kind == "none"):
        raise RuntimeError("The layer-owner action code disagrees with its action count.")
    values = words[_PLAN_INTEGER_FIELDS:].contiguous().view(torch.float64).tolist()

    def placement_cost(offset: int) -> PlacementCost:
        return PlacementCost(
            communication=float(values[offset]),
            compute=float(values[offset + 1]),
            communication_model_units=float(values[offset + 2]),
            peak_communication_rank=int(values[offset + 3]),
            peak_compute_rank=int(values[offset + 4]),
            selected_dim=int(values[offset + 5]),
            state_move_exposed=float(values[offset + 6]),
            gradient_sync=float(values[offset + 7]),
        )

    host_layout = layout.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
    host_owners = owners.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
    initial_layout = tuple(int(value) for value in host_layout.tolist())
    if action is not None:
        if action.kind == "swap":
            host_layout[action.src_slot] = action.dst_logical
            host_layout[action.dst_slot] = action.src_logical
            source_owner = host_owners[action.src_logical].clone()
            host_owners[action.src_logical] = host_owners[action.dst_logical]
            host_owners[action.dst_logical] = source_owner
        elif action.kind == "replica":
            host_layout[action.dst_slot] = action.src_logical
        else:
            raise RuntimeError("Empty-slot initialization is not supported by steady-state layer owners.")
    timing_offset = 16
    return PlacementPlan(
        actions=() if action is None else (action,),
        initial_layout=initial_layout,
        final_layout=tuple(int(value) for value in host_layout.tolist()),
        baseline_cost=placement_cost(0),
        final_cost=placement_cost(8),
        swap_rounds=integers[6],
        replica_rounds=integers[7],
        planning_ms=float(values[timing_offset]),
        route_stats_ms=float(values[timing_offset + 1]),
        swap_ms=float(values[timing_offset + 2]),
        replica_ms=float(values[timing_offset + 3]),
        swap_score_ms=float(values[timing_offset + 4]),
        swap_update_ms=float(values[timing_offset + 5]),
        swap_collective_ms=float(values[timing_offset + 6]),
        replica_score_ms=float(values[timing_offset + 7]),
        replica_update_ms=float(values[timing_offset + 8]),
        replica_collective_ms=float(values[timing_offset + 9]),
        decision_sync_ms=float(values[timing_offset + 10]),
        finalization_ms=float(values[timing_offset + 11]),
        algorithm_version=GREEDY_COVER_ALGORITHM_VERSION,
        local_physical_routes=None,
        final_owner_slots=tuple(int(value) for value in host_owners.tolist()),
    )


class AcceleratorLayerOwnerPlanner:
    """Shard exact independent-layer planning across an accelerator EP group."""

    def __init__(
        self,
        planner: GreedyCommunicationPlanner,
        *,
        process_group: dist.ProcessGroup,
    ) -> None:
        self.planner = planner
        self.process_group = process_group
        self.last_result: LayerOwnerResult | None = None

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
    ) -> LayerOwnerResult:
        started = time.perf_counter()
        layer_count = len(selected_experts)
        if not (len(slot_to_logical) == len(owner_slots) == len(layer_seeds) == layer_count):
            raise ValueError("Layer-owner planner inputs must have identical layer counts.")
        if layer_count == 0:
            result = LayerOwnerResult(
                (),
                LayerOwnerTiming(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
                (),
            )
            self.last_result = result
            return result
        if not dist.is_initialized():
            raise RuntimeError("Accelerator layer-owner planning requires an initialized process group.")
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        if int(source_rank) != rank:
            raise ValueError(f"source_rank must equal the EP rank, got {source_rank} and {rank}.")

        routes = []
        for layer_index, raw in enumerate(selected_experts):
            selected = raw.to(torch.long)
            if selected.ndim == 1:
                selected = selected.unsqueeze(-1)
            if selected.ndim != 2:
                raise ValueError(
                    f"selected_experts[{layer_index}] must have rank 1 or 2, got {tuple(selected.shape)}."
                )
            routes.append(selected)
        device = routes[0].device
        if any(route.device != device for route in routes):
            raise ValueError("All layer-owner routes must reside on the same device.")
        if any(bool((layout < 0).any().item()) for layout in slot_to_logical):
            raise ValueError("Layer-owner planning only supports initialized steady-state layouts.")

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
            raise ValueError("Layer-owner cost-model arrays must match the number of layers.")
        layer_owners = balanced_layer_owner_ranks(layer_count, world_size, owner_offset=owner_offset)
        owned_indices = tuple(index for index, owner in enumerate(layer_owners) if owner == rank)

        metadata_started = time.perf_counter()
        local_metadata = torch.tensor(
            [[int(route.shape[0]), int(route.shape[1])] for route in routes],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)
        gathered_metadata = torch.empty(
            (world_size, layer_count, 2),
            dtype=torch.int64,
            device=device,
        )
        dist.all_gather_into_tensor(gathered_metadata, local_metadata, group=self.process_group)
        metadata_host = gathered_metadata.detach().to(device="cpu").tolist()
        for layer_index in range(layer_count):
            top_k = {int(metadata_host[source][layer_index][1]) for source in range(world_size)}
            if len(top_k) != 1:
                raise ValueError(f"Layer {layer_index} has inconsistent router top-k across EP ranks.")
        metadata_collective_ms = (time.perf_counter() - metadata_started) * 1000.0

        pack_started = time.perf_counter()
        send_chunks = []
        input_splits = []
        for destination in range(world_size):
            indices = [index for index, owner in enumerate(layer_owners) if owner == destination]
            chunks = [routes[index].reshape(-1).to(torch.int32) for index in indices]
            chunk = torch.cat(chunks) if chunks else torch.empty((0,), dtype=torch.int32, device=device)
            send_chunks.append(chunk)
            input_splits.append(int(chunk.numel()))
        send_buffer = torch.cat(send_chunks)
        output_splits = [
            sum(int(metadata_host[source][index][0]) * int(metadata_host[source][index][1]) for index in owned_indices)
            for source in range(world_size)
        ]
        recv_buffer = torch.empty((sum(output_splits),), dtype=torch.int32, device=device)
        route_pack_ms = (time.perf_counter() - pack_started) * 1000.0

        collective_started = time.perf_counter()
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.process_group,
        )
        route_collective_ms = (time.perf_counter() - collective_started) * 1000.0

        unpack_started = time.perf_counter()
        selected_chunks: dict[int, list[torch.Tensor]] = {index: [] for index in owned_indices}
        source_chunks: dict[int, list[torch.Tensor]] = {index: [] for index in owned_indices}
        ordinal_chunks: dict[int, list[torch.Tensor]] = {index: [] for index in owned_indices}
        offset = 0
        for source in range(world_size):
            for layer_index in owned_indices:
                token_count = int(metadata_host[source][layer_index][0])
                top_k = int(metadata_host[source][layer_index][1])
                flat_size = token_count * top_k
                selected_chunks[layer_index].append(
                    recv_buffer[offset : offset + flat_size].view(token_count, top_k).to(torch.long)
                )
                source_chunks[layer_index].append(torch.full((token_count,), source, dtype=torch.long, device=device))
                ordinal_chunks[layer_index].append(torch.arange(token_count, dtype=torch.long, device=device))
                offset += flat_size
        if offset != int(recv_buffer.numel()):
            raise RuntimeError("Layer-owner route unpacking did not consume the receive buffer.")
        global_routes = [torch.cat(selected_chunks[index], dim=0) for index in owned_indices]
        global_sources = [torch.cat(source_chunks[index], dim=0) for index in owned_indices]
        global_ordinals = [torch.cat(ordinal_chunks[index], dim=0) for index in owned_indices]
        route_unpack_ms = (time.perf_counter() - unpack_started) * 1000.0

        owner_started = time.perf_counter()
        owner_planner = _clone_for_global_routes(self.planner)
        owner_plans = owner_planner.plan_layers(
            global_routes,
            [slot_to_logical[index] for index in owned_indices],
            [owner_slots[index] for index in owned_indices],
            source_ranks=global_sources,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
            layer_seeds=[layer_seeds[index] for index in owned_indices],
            step=step,
            communication_scales=[scales[index] for index in owned_indices],
            forward_compute_per_assignment=[compute_slopes[index] for index in owned_indices],
            forward_compute_constant=[compute_constants[index] for index in owned_indices],
            token_ordinals=global_ordinals,
            skip_final_route_update=True,
        )
        owner_planning_ms = (time.perf_counter() - owner_started) * 1000.0

        action_started = time.perf_counter()
        encoded = torch.zeros((layer_count, _PLAN_FIELDS), dtype=torch.int64, device=device)
        for layer_index, plan in zip(owned_indices, owner_plans, strict=True):
            encoded[layer_index].copy_(_encode_plan(plan).to(device=device))
        dist.all_reduce(encoded, op=dist.ReduceOp.SUM, group=self.process_group)
        encoded_host = encoded.detach().to(device="cpu")
        action_collective_ms = (time.perf_counter() - action_started) * 1000.0

        finalization_started = time.perf_counter()
        plans = tuple(
            _decode_plan(encoded_host[index], slot_to_logical[index], owner_slots[index])
            for index in range(layer_count)
        )
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = LayerOwnerTiming(
            metadata_collective_ms=metadata_collective_ms,
            route_pack_ms=route_pack_ms,
            route_collective_ms=route_collective_ms,
            route_unpack_ms=route_unpack_ms,
            owner_planning_ms=owner_planning_ms,
            action_collective_ms=action_collective_ms,
            finalization_ms=finalization_ms,
            total_ms=total_ms,
            sent_route_bytes=int(send_buffer.numel() * send_buffer.element_size()),
            received_route_bytes=int(recv_buffer.numel() * recv_buffer.element_size()),
            owned_layer_count=len(owned_indices),
        )
        result = LayerOwnerResult(plans, timing, layer_owners)
        self.last_result = result
        return result
