# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .expert_swap import (
    _USE_GLOBAL_2D_SELECTOR,
    _USE_GLOBAL_HIERARCHY_SELECTOR,
    ExpertLayerState,
    _all_candidate_pairs,
    _candidate_pair_token_counts,
    _candidate_pairs,
    _costs_from_global_2d_stats,
    _costs_from_global_hierarchy_stats,
    _cross_rank_candidate_pairs,
    _env_int,
    _estimate_best_swap_pair_row,
    _full_timing_range,
    _hierarchy_level_group_shapes,
    _local_tensor_view,
    _resolve_candidate_shards,
    _selector_group_stats,
    _selector_stats_2d,
    _shard_candidate_pairs,
    _smooth_cost_tensor,
)
from .perf_model import HierMoEPerfModel
from .topology import Hierarchy


_FULL_ROUTE_GATHER_MAX_TOKENS = _env_int("VEOMNI_HIERMOE_FULL_ROUTE_GATHER_MAX_TOKENS", 16384)


@dataclass
class _SequentialGroupState:
    group_by_logical: torch.Tensor
    group_by_host: list[int]
    base_counts: torch.Tensor
    expert_group_counts: torch.Tensor
    sole_expert_counts: torch.Tensor
    sole_pair_counts: torch.Tensor


@dataclass
class _SequentialLayerSummary:
    canonical_routes: torch.Tensor
    unique_slots: torch.Tensor
    expert_token_counts: torch.Tensor
    assignment_counts: torch.Tensor
    level_states: list[_SequentialGroupState]
    rank_state: _SequentialGroupState


def _canonical_route_slots(selected_experts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    routes = selected_experts.to(torch.long)
    unique = torch.ones_like(routes, dtype=torch.bool)
    for slot in range(1, routes.shape[1]):
        unique[:, slot] = (routes[:, slot : slot + 1] != routes[:, :slot]).all(dim=1)
    return routes, unique


def _compact_group_contributions(
    canonical_routes: torch.Tensor,
    unique_slots: torch.Tensor,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = int(group_by_logical.numel())
    device = canonical_routes.device
    num_tokens = int(canonical_routes.shape[0])
    top_k = int(canonical_routes.shape[1])
    if canonical_routes.numel() == 0:
        return (
            torch.zeros((num_groups,), dtype=torch.float32, device=device),
            torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device),
            torch.zeros((num_experts,), dtype=torch.float32, device=device),
            torch.zeros((num_experts, num_experts), dtype=torch.float32, device=device),
        )

    base_counts = torch.zeros((num_groups,), dtype=torch.float32, device=device)
    expert_group_counts = torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device)
    sole_expert_counts = torch.zeros((num_experts,), dtype=torch.float32, device=device)
    sole_pair_counts = torch.zeros((num_experts * num_experts,), dtype=torch.float32, device=device)
    chunk_size = 32768
    for start in range(0, num_tokens, chunk_size):
        end = min(num_tokens, start + chunk_size)
        routes = canonical_routes[start:end]
        unique = unique_slots[start:end]
        selected_groups = group_by_logical.index_select(0, routes.reshape(-1)).view_as(routes)
        token_group_counts = torch.zeros(
            (routes.shape[0], num_groups),
            dtype=torch.float32,
            device=device,
        )
        token_group_counts.scatter_add_(1, selected_groups, unique.to(torch.float32))
        token_group_hits = (token_group_counts > 0).to(torch.float32)
        base_counts.add_(token_group_hits.sum(dim=0))

        flat_experts = routes.reshape(-1)
        flat_unique = unique.reshape(-1).to(torch.float32)
        group_values = token_group_hits.unsqueeze(1).expand(-1, top_k, -1).reshape(
            -1, num_groups
        ) * flat_unique.unsqueeze(1)
        expert_group_counts.index_add_(0, flat_experts, group_values)

        own_group_counts = token_group_counts.gather(1, selected_groups)
        sole_slots = unique & (own_group_counts == 1)
        sole_expert_counts.index_add_(
            0,
            flat_experts,
            sole_slots.reshape(-1).to(torch.float32),
        )
        pair_indices = (routes.unsqueeze(2) * num_experts + routes.unsqueeze(1)).reshape(-1)
        pair_values = (sole_slots.unsqueeze(2) & unique.unsqueeze(1)).reshape(-1).to(torch.float32)
        sole_pair_counts.index_add_(0, pair_indices, pair_values)

    return (
        base_counts,
        expert_group_counts,
        sole_expert_counts,
        sole_pair_counts.view(num_experts, num_experts),
    )


def _build_sequential_group_state(
    canonical_routes: torch.Tensor,
    unique_slots: torch.Tensor,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> _SequentialGroupState:
    base_counts, expert_group_counts, sole_expert_counts, sole_pair_counts = _compact_group_contributions(
        canonical_routes,
        unique_slots,
        group_by_logical,
        num_groups,
    )
    return _SequentialGroupState(
        group_by_logical=group_by_logical,
        group_by_host=[int(value) for value in group_by_logical.detach().cpu().tolist()],
        base_counts=base_counts,
        expert_group_counts=expert_group_counts,
        sole_expert_counts=sole_expert_counts,
        sole_pair_counts=sole_pair_counts,
    )


def _sequential_group_state_payload(state: _SequentialGroupState) -> torch.Tensor:
    return torch.cat(
        (
            state.base_counts.reshape(-1),
            state.expert_group_counts.reshape(-1),
            state.sole_expert_counts.reshape(-1),
            state.sole_pair_counts.reshape(-1),
        ),
        dim=0,
    ).to(torch.int32)


def _sequential_group_state_from_payload(
    payload: torch.Tensor,
    offset: int,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> tuple[_SequentialGroupState, int]:
    num_experts = int(group_by_logical.numel())
    base_end = offset + num_groups
    expert_group_end = base_end + num_experts * num_groups
    sole_expert_end = expert_group_end + num_experts
    sole_pair_end = sole_expert_end + num_experts * num_experts
    state = _SequentialGroupState(
        group_by_logical=group_by_logical,
        group_by_host=[int(value) for value in group_by_logical.detach().cpu().tolist()],
        base_counts=payload[offset:base_end].to(torch.float32),
        expert_group_counts=payload[base_end:expert_group_end].view(num_experts, num_groups).to(torch.float32),
        sole_expert_counts=payload[expert_group_end:sole_expert_end].to(torch.float32),
        sole_pair_counts=payload[sole_expert_end:sole_pair_end].view(num_experts, num_experts).to(torch.float32),
    )
    return state, sole_pair_end


def _candidate_group_counts_from_summary(
    *,
    expert_token_counts: torch.Tensor,
    state: _SequentialGroupState,
    pairs: torch.Tensor,
) -> torch.Tensor:
    lhs = pairs[:, 0]
    rhs = pairs[:, 1]
    lhs_group = state.group_by_logical.index_select(0, lhs)
    rhs_group = state.group_by_logical.index_select(0, rhs)
    same_group = lhs_group == rhs_group
    lhs_count = expert_token_counts.index_select(0, lhs)
    rhs_count = expert_token_counts.index_select(0, rhs)
    delta_lhs = (
        rhs_count
        - state.expert_group_counts[rhs, lhs_group]
        - state.sole_expert_counts.index_select(0, lhs)
        + state.sole_pair_counts[lhs, rhs]
    )
    delta_rhs = (
        lhs_count
        - state.expert_group_counts[lhs, rhs_group]
        - state.sole_expert_counts.index_select(0, rhs)
        + state.sole_pair_counts[rhs, lhs]
    )
    delta_lhs = torch.where(same_group, torch.zeros_like(delta_lhs), delta_lhs)
    delta_rhs = torch.where(same_group, torch.zeros_like(delta_rhs), delta_rhs)
    counts = state.base_counts.unsqueeze(0).expand(pairs.shape[0], -1).clone()
    counts.scatter_add_(1, lhs_group.unsqueeze(1), delta_lhs.unsqueeze(1))
    counts.scatter_add_(1, rhs_group.unsqueeze(1), delta_rhs.unsqueeze(1))
    return counts


def _sequential_swap_costs_from_summary(
    *,
    expert_token_counts: torch.Tensor,
    pairs: torch.Tensor,
    level_states: list[_SequentialGroupState],
    rank_state: _SequentialGroupState,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    candidate_rank_counts = _candidate_group_counts_from_summary(
        expert_token_counts=expert_token_counts,
        state=rank_state,
        pairs=pairs,
    )
    rank_max = rank_state.base_counts.max()
    candidate_rank_max = candidate_rank_counts.max(dim=1).values
    current_dim_costs = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * rank_max * perf_model.a2a.beta
    ]
    candidate_dim_costs = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * candidate_rank_max * perf_model.a2a.beta
    ]

    level_shapes = _hierarchy_level_group_shapes(hierarchy, expert_token_counts.numel())
    candidate_level_counts = [
        _candidate_group_counts_from_summary(
            expert_token_counts=expert_token_counts,
            state=state,
            pairs=pairs,
        )
        for state in level_states
    ]
    max_dim = max(1, int(hierarchy.selected_dim))
    for dim in range(2, max_dim + 1):
        if len(level_shapes) < dim - 1 or not perf_model.inter:
            break
        current_total = torch.zeros((), dtype=torch.float32, device=expert_token_counts.device)
        candidate_total = torch.zeros((pairs.shape[0],), dtype=torch.float32, device=expert_token_counts.device)
        previous_u = 1
        for level_idx, (u_i, _num_groups) in enumerate(level_shapes[: dim - 1]):
            link = perf_model.inter[min(level_idx, len(perf_model.inter) - 1)]
            scale = float((u_i / previous_u) * hidden_size * bytes_per_element)
            current_total = current_total + link.alpha + scale * level_states[level_idx].base_counts.max() * link.beta
            candidate_total = (
                candidate_total + link.alpha + scale * candidate_level_counts[level_idx].max(dim=1).values * link.beta
            )
            previous_u = u_i

        scale = float((hierarchy.ep_size / previous_u) * hidden_size * bytes_per_element)
        current_total = current_total + perf_model.intra.alpha + scale * rank_max * perf_model.intra.beta
        candidate_total = candidate_total + perf_model.intra.alpha + scale * candidate_rank_max * perf_model.intra.beta
        current_dim_costs.append(current_total)
        candidate_dim_costs.append(candidate_total)

    current_cost = _smooth_cost_tensor(torch.stack(current_dim_costs).unsqueeze(0), gamma)[0]
    candidate_costs = _smooth_cost_tensor(torch.stack(candidate_dim_costs, dim=1), gamma)
    return (
        current_cost.to(torch.float32),
        candidate_costs.to(torch.float32),
        candidate_level_counts,
        candidate_rank_counts,
    )


def _apply_swap_to_sequential_group_state(
    *,
    canonical_routes: torch.Tensor,
    unique_slots: torch.Tensor,
    lhs: int,
    rhs: int,
    state: _SequentialGroupState,
    updated_base_counts: torch.Tensor,
) -> None:
    lhs_group = state.group_by_host[lhs]
    rhs_group = state.group_by_host[rhs]
    if lhs_group == rhs_group:
        return

    affected = ((canonical_routes == lhs) | (canonical_routes == rhs)).any(dim=1)
    affected_indices = torch.nonzero(affected, as_tuple=False).flatten()
    routes = canonical_routes.index_select(0, affected_indices)
    unique = unique_slots.index_select(0, affected_indices)
    old_groups = state.group_by_logical.index_select(0, routes.reshape(-1)).view_as(routes)
    updated_group_by = state.group_by_logical.clone()
    lhs_value = updated_group_by[lhs].clone()
    updated_group_by[lhs] = updated_group_by[rhs]
    updated_group_by[rhs] = lhs_value
    new_groups = updated_group_by.index_select(0, routes.reshape(-1)).view_as(routes)

    tracked_groups = torch.tensor((lhs_group, rhs_group), dtype=torch.long, device=routes.device)
    old_group_hits = ((old_groups.unsqueeze(2) == tracked_groups.view(1, 1, 2)) & unique.unsqueeze(2)).any(dim=1)
    new_group_hits = ((new_groups.unsqueeze(2) == tracked_groups.view(1, 1, 2)) & unique.unsqueeze(2)).any(dim=1)
    group_hit_delta = new_group_hits.to(torch.float32) - old_group_hits.to(torch.float32)
    expert_group_delta = torch.zeros(
        (state.expert_group_counts.shape[0], 2),
        dtype=torch.float32,
        device=routes.device,
    )
    expert_group_delta.index_add_(
        0,
        routes.reshape(-1),
        (group_hit_delta.unsqueeze(1).expand(-1, routes.shape[1], -1) * unique.unsqueeze(2)).reshape(-1, 2),
    )

    old_own_counts = ((old_groups.unsqueeze(2) == old_groups.unsqueeze(1)) & unique.unsqueeze(1)).sum(dim=2)
    new_own_counts = ((new_groups.unsqueeze(2) == new_groups.unsqueeze(1)) & unique.unsqueeze(1)).sum(dim=2)
    sole_delta = (unique & (new_own_counts == 1)).to(torch.float32) - (unique & (old_own_counts == 1)).to(
        torch.float32
    )
    sole_expert_delta = torch.zeros_like(state.sole_expert_counts)
    sole_expert_delta.index_add_(0, routes.reshape(-1), sole_delta.reshape(-1))
    sole_pair_delta = torch.zeros_like(state.sole_pair_counts).reshape(-1)
    sole_pair_delta.index_add_(
        0,
        (routes.unsqueeze(2) * state.sole_pair_counts.shape[0] + routes.unsqueeze(1)).reshape(-1),
        (sole_delta.unsqueeze(2) * unique.unsqueeze(1)).reshape(-1),
    )

    state.base_counts.copy_(updated_base_counts)
    state.expert_group_counts.index_add_(1, tracked_groups, expert_group_delta)
    state.sole_expert_counts.add_(sole_expert_delta)
    state.sole_pair_counts.add_(sole_pair_delta.view_as(state.sole_pair_counts))
    state.group_by_logical.copy_(updated_group_by)
    state.group_by_host[lhs], state.group_by_host[rhs] = rhs_group, lhs_group


class LegacyBatchedSelector:
    """Restore the pre-current-route batched approximate P1/P4 selector."""

    def __init__(self, manager: object) -> None:
        self._manager = manager

    def __getattr__(self, name: str):
        return getattr(self._manager, name)

    @staticmethod
    def _cross_rank_pairs(layer: ExpertLayerState, pairs: torch.Tensor) -> torch.Tensor:
        return _cross_rank_candidate_pairs(
            pairs,
            layer.mapping_for_device(pairs.device),
            layer.num_local_experts,
        )

    def select(self, layers: list[ExpertLayerState]) -> dict[str, list[tuple[int, int]]]:
        if int(self.expert_swap_max_pairs_per_layer) <= 0:
            return {}
        return self._select_global_pair_lists(layers)

    def _local_pair_candidate(
        self,
        layer: ExpertLayerState,
        candidate_pairs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = layer.latest_selected_experts
        if selected is None or selected.numel() == 0:
            device = _local_tensor_view(layer.gate_up_proj).device
            return torch.tensor([float("inf"), -1.0, -1.0], dtype=torch.float32, device=device)
        else:
            logical_to_physical = layer.mapping_for_device(selected.device)
            candidates = _candidate_pairs(selected, layer.num_experts) if candidate_pairs is None else candidate_pairs
            candidates = _cross_rank_candidate_pairs(candidates, logical_to_physical, layer.num_local_experts)
            return _estimate_best_swap_pair_row(
                selected_experts=selected,
                num_experts=layer.num_experts,
                hidden_size=layer.latest_hidden_size,
                bytes_per_element=layer.latest_bytes_per_element,
                hierarchy=self.hierarchy,
                perf_model=self.perf_model,
                gamma=self.smooth_max_gamma,
                logical_to_physical=logical_to_physical,
                candidate_pairs=candidates,
            )

    def _candidate_pairs_by_layer(
        self,
        layers: list[ExpertLayerState],
        *,
        shard: bool = True,
    ) -> list[torch.Tensor | None]:
        candidate_pairs: list[torch.Tensor | None] = []
        for layer in layers:
            selected = layer.latest_selected_experts
            device = selected.device if selected is not None else _local_tensor_view(layer.gate_up_proj).device
            pairs = _all_candidate_pairs(layer.num_experts, device)
            pairs = self._cross_rank_pairs(layer, pairs)
            num_shards = _resolve_candidate_shards(self.ep_size, layer.num_experts)
            shard_idx = self.ep_rank % num_shards
            candidate_pairs.append(
                _shard_candidate_pairs(
                    pairs,
                    shard_idx=shard_idx,
                    num_shards=num_shards,
                )
                if shard
                else pairs
            )
        return candidate_pairs

    def _first_pair_by_layer(
        self, pair_lists: dict[str, list[tuple[int, int]]] | None
    ) -> dict[str, tuple[int, int]] | None:
        if pair_lists is None:
            return None
        return {layer_key: pairs[0] for layer_key, pairs in pair_lists.items() if pairs}

    def _gather_sequential_layer_summaries(
        self,
        layers: list[ExpertLayerState],
    ) -> list[_SequentialLayerSummary] | None:
        local_rows: list[torch.Tensor] = []
        row_meta: list[
            tuple[
                int,
                int,
                int,
                int,
                list[tuple[torch.Tensor, int]],
                tuple[torch.Tensor, int],
            ]
        ] = []
        for layer in layers:
            selected = layer.latest_selected_experts
            if selected is None:
                return None
            if selected.ndim == 1:
                selected = selected.unsqueeze(-1)
            if selected.ndim != 2:
                return None
            selected = selected.to(dtype=torch.long, non_blocking=True).contiguous()
            num_tokens = int(selected.shape[0])
            top_k = int(selected.shape[1])
            capacity_exceeded = num_tokens > _FULL_ROUTE_GATHER_MAX_TOKENS
            payload_selected = selected[:0] if capacity_exceeded else selected

            canonical_routes, unique_slots = _canonical_route_slots(payload_selected)
            num_experts = int(layer.num_experts)
            compact_routes = torch.where(
                unique_slots,
                canonical_routes,
                torch.full_like(canonical_routes, num_experts),
            ).to(torch.int32)
            route_width = _FULL_ROUTE_GATHER_MAX_TOKENS * top_k
            route_payload = torch.full(
                (route_width,),
                num_experts,
                dtype=torch.int32,
                device=selected.device,
            )
            payload_tokens = int(payload_selected.shape[0])
            route_payload[: payload_tokens * top_k].copy_(compact_routes.reshape(-1))

            expert_token_counts = torch.zeros((num_experts,), dtype=torch.float32, device=selected.device)
            expert_token_counts.index_add_(
                0,
                canonical_routes.reshape(-1),
                unique_slots.reshape(-1).to(torch.float32),
            )
            assignment_counts = torch.bincount(selected.reshape(-1), minlength=num_experts).to(torch.float32)
            mapping = layer.mapping_for_device(selected.device).clone()

            level_states: list[_SequentialGroupState] = []
            level_meta: list[tuple[torch.Tensor, int]] = []
            for u_i, num_groups in _hierarchy_level_group_shapes(self.hierarchy, num_experts):
                expert_group_size = max(
                    1,
                    num_experts // max(1, self.hierarchy.ep_size // int(u_i)),
                )
                group_by = torch.div(mapping, expert_group_size, rounding_mode="floor")
                level_states.append(
                    _build_sequential_group_state(
                        canonical_routes,
                        unique_slots,
                        group_by,
                        num_groups,
                    )
                )
                level_meta.append((group_by, num_groups))

            num_local_experts = max(1, num_experts // self.hierarchy.ep_size)
            rank_group_by = torch.div(mapping, num_local_experts, rounding_mode="floor")
            rank_state = _build_sequential_group_state(
                canonical_routes,
                unique_slots,
                rank_group_by,
                self.hierarchy.ep_size,
            )
            summary_parts = [
                expert_token_counts.to(torch.int32),
                assignment_counts.to(torch.int32),
                *(_sequential_group_state_payload(state) for state in level_states),
                _sequential_group_state_payload(rank_state),
            ]
            summary_payload = torch.cat(summary_parts, dim=0)
            encoded_length = -num_tokens - 1 if capacity_exceeded else num_tokens
            header = torch.tensor((encoded_length,), dtype=torch.int32, device=selected.device)
            local_rows.append(torch.cat((header, route_payload, summary_payload), dim=0))
            row_meta.append(
                (
                    top_k,
                    num_experts,
                    route_width,
                    int(summary_payload.numel()),
                    level_meta,
                    (rank_group_by, self.hierarchy.ep_size),
                )
            )

        if not local_rows:
            return []
        max_width = max(int(row.numel()) for row in local_rows)
        local = torch.zeros((len(local_rows), max_width), dtype=torch.int32, device=local_rows[0].device)
        for row_idx, row in enumerate(local_rows):
            local[row_idx, : row.numel()].copy_(row)
        if self.ep_group is None or self.ep_size <= 1:
            gathered = local.unsqueeze(0)
        else:
            gathered_flat = torch.empty(
                (self.ep_size * len(layers), max_width),
                dtype=torch.int32,
                device=local.device,
            )
            dist.all_gather_into_tensor(gathered_flat, local, group=self.ep_group)
            gathered = gathered_flat.view(self.ep_size, len(layers), max_width)

        route_lengths = gathered[:, :, 0].detach().cpu()
        if bool((route_lengths < 0).any()):
            decoded_lengths = torch.where(route_lengths < 0, -route_lengths - 1, route_lengths)
            max_tokens = int(decoded_lengths.max().item())
            raise RuntimeError(
                "Compact route summary capacity exceeded on at least one EP rank: "
                f"{max_tokens} tokens > {_FULL_ROUTE_GATHER_MAX_TOKENS}. "
                "Increase VEOMNI_HIERMOE_FULL_ROUTE_GATHER_MAX_TOKENS consistently on every EP rank."
            )
        summaries: list[_SequentialLayerSummary] = []
        for layer_idx, meta in enumerate(row_meta):
            top_k, num_experts, route_width, summary_width, level_meta, rank_meta = meta
            rank_routes: list[torch.Tensor] = []
            for rank_idx in range(gathered.shape[0]):
                num_tokens = int(route_lengths[rank_idx, layer_idx])
                if num_tokens < 0 or num_tokens > _FULL_ROUTE_GATHER_MAX_TOKENS:
                    raise RuntimeError(f"Invalid gathered route length {num_tokens} from EP rank {rank_idx}.")
                rank_routes.append(
                    gathered[
                        rank_idx,
                        layer_idx,
                        1 : 1 + num_tokens * top_k,
                    ].view(num_tokens, top_k)
                )
            compact_routes = torch.cat(rank_routes, dim=0).contiguous()
            unique_slots = compact_routes != num_experts
            canonical_routes = compact_routes.clamp_max(num_experts - 1).to(torch.long)

            summary_offset = 1 + route_width
            global_payload = gathered[
                :,
                layer_idx,
                summary_offset : summary_offset + summary_width,
            ].sum(dim=0, dtype=torch.int32)
            offset = 0
            expert_token_counts = global_payload[offset : offset + num_experts].to(torch.float32)
            offset += num_experts
            assignment_counts = global_payload[offset : offset + num_experts]
            offset += num_experts
            level_states: list[_SequentialGroupState] = []
            for group_by, num_groups in level_meta:
                state, offset = _sequential_group_state_from_payload(
                    global_payload,
                    offset,
                    group_by,
                    num_groups,
                )
                level_states.append(state)
            rank_group_by, num_rank_groups = rank_meta
            rank_state, offset = _sequential_group_state_from_payload(
                global_payload,
                offset,
                rank_group_by,
                num_rank_groups,
            )
            if offset != summary_width:
                raise RuntimeError(f"Invalid compact summary width: consumed {offset}, expected {summary_width}.")
            summaries.append(
                _SequentialLayerSummary(
                    canonical_routes=canonical_routes,
                    unique_slots=unique_slots,
                    expert_token_counts=expert_token_counts,
                    assignment_counts=assignment_counts,
                    level_states=level_states,
                    rank_state=rank_state,
                )
            )
        return summaries

    def _select_global_pair_lists_from_gathered_routes(
        self, layers: list[ExpertLayerState]
    ) -> dict[str, list[tuple[int, int]]] | None:
        summaries = self._gather_sequential_layer_summaries(layers)
        if summaries is None:
            return None

        max_pairs = int(self.expert_swap_max_pairs_per_layer)
        if max_pairs <= 0:
            return {}
        pair_lists: dict[str, list[tuple[int, int]]] = {}
        for layer, summary in zip(layers, summaries, strict=True):
            canonical_routes = summary.canonical_routes
            unique_slots = summary.unique_slots
            expert_token_counts = summary.expert_token_counts
            level_states = summary.level_states
            rank_state = summary.rank_state
            device = canonical_routes.device
            pairs = _all_candidate_pairs(layer.num_experts, device)
            pairs = self._cross_rank_pairs(layer, pairs)
            if pairs.numel() == 0:
                continue

            used = torch.zeros((layer.num_experts,), dtype=torch.bool, device=device)
            chosen: list[tuple[int, int]] = []
            for _round in range(max_pairs):
                disallowed = used.index_select(0, pairs[:, 0]) | used.index_select(0, pairs[:, 1])
                current_cost, candidate_costs, candidate_level_counts, candidate_rank_counts = (
                    _sequential_swap_costs_from_summary(
                        expert_token_counts=expert_token_counts,
                        pairs=pairs,
                        level_states=level_states,
                        rank_state=rank_state,
                        hidden_size=layer.latest_hidden_size,
                        bytes_per_element=layer.latest_bytes_per_element,
                        hierarchy=self.hierarchy,
                        perf_model=self.perf_model,
                        gamma=self.smooth_max_gamma,
                    )
                )
                candidate_costs = candidate_costs.masked_fill(disallowed, float("inf"))
                best_index = torch.argmin(candidate_costs)
                best_cost = candidate_costs.index_select(0, best_index.view(1))[0]
                best_pair = pairs.index_select(0, best_index.view(1))[0]
                decision = (
                    torch.stack(
                        (
                            best_cost - current_cost,
                            best_pair[0].to(torch.float32),
                            best_pair[1].to(torch.float32),
                        )
                    )
                    .detach()
                    .cpu()
                )
                if float(decision[0]) >= 0.0:
                    break
                lhs = int(decision[1])
                rhs = int(decision[2])
                chosen.append((lhs, rhs))
                used[lhs] = True
                used[rhs] = True
                for state, candidate_counts in zip(
                    level_states,
                    candidate_level_counts,
                    strict=True,
                ):
                    _apply_swap_to_sequential_group_state(
                        canonical_routes=canonical_routes,
                        unique_slots=unique_slots,
                        lhs=lhs,
                        rhs=rhs,
                        state=state,
                        updated_base_counts=candidate_counts.index_select(0, best_index.view(1)).squeeze(0),
                    )
                _apply_swap_to_sequential_group_state(
                    canonical_routes=canonical_routes,
                    unique_slots=unique_slots,
                    lhs=lhs,
                    rhs=rhs,
                    state=rank_state,
                    updated_base_counts=candidate_rank_counts.index_select(0, best_index.view(1)).squeeze(0),
                )
            if chosen:
                pair_lists[layer.key] = chosen
        return pair_lists

    def _select_pair_lists_from_cost_tensors(
        self,
        layers: list[ExpertLayerState],
        cost_tensor: torch.Tensor,
        current_costs: torch.Tensor,
        pair_tensor: torch.Tensor,
    ) -> dict[str, list[tuple[int, int]]]:
        max_pairs = max(1, int(self.expert_swap_max_pairs_per_layer))
        pair_lists: dict[str, list[tuple[int, int]]] = {}

        if max_pairs == 1:
            with _full_timing_range("hiermoe_expert_swap_select_global"):
                best_costs, best_indices = torch.min(cost_tensor, dim=1)
                improved = best_costs < current_costs
                chosen_pairs = pair_tensor[torch.arange(len(layers), device=cost_tensor.device), best_indices]
                chosen_pairs = torch.where(
                    improved.unsqueeze(1),
                    chosen_pairs,
                    torch.full_like(chosen_pairs, -1),
                )

            with _full_timing_range("hiermoe_expert_swap_select_finalize"):
                chosen_cpu = chosen_pairs.detach().to(torch.device("cpu"))
                for layer, row in zip(layers, chosen_cpu, strict=True):
                    lhs = int(row[0].item())
                    rhs = int(row[1].item())
                    if lhs >= 0 and rhs >= 0:
                        pair_lists[layer.key] = [(lhs, rhs)]
            return pair_lists

        with _full_timing_range("hiermoe_expert_swap_select_global"):
            candidate_orders = torch.argsort(cost_tensor, dim=1)

        with _full_timing_range("hiermoe_expert_swap_select_finalize"):
            orders_cpu = candidate_orders.detach().to(torch.device("cpu"))
            costs_cpu = cost_tensor.detach().to(torch.device("cpu"))
            current_cpu = current_costs.detach().to(torch.device("cpu"))
            pairs_cpu = pair_tensor.detach().to(torch.device("cpu"))
            for layer_idx, layer in enumerate(layers):
                chosen: list[tuple[int, int]] = []
                used_logical: set[int] = set()
                current_cost = float(current_cpu[layer_idx].item())
                for candidate_idx in orders_cpu[layer_idx].tolist():
                    cost = float(costs_cpu[layer_idx, candidate_idx].item())
                    if cost >= current_cost:
                        break
                    lhs = int(pairs_cpu[layer_idx, candidate_idx, 0].item())
                    rhs = int(pairs_cpu[layer_idx, candidate_idx, 1].item())
                    if lhs < 0 or rhs < 0 or lhs in used_logical or rhs in used_logical:
                        continue
                    chosen.append((lhs, rhs))
                    used_logical.update((lhs, rhs))
                    if len(chosen) >= max_pairs:
                        break
                if chosen:
                    pair_lists[layer.key] = chosen
        return pair_lists

    def _select_global_pair_lists_fast_2d(
        self, layers: list[ExpertLayerState]
    ) -> dict[str, list[tuple[int, int]]] | None:
        if not _USE_GLOBAL_2D_SELECTOR or max(1, int(self.hierarchy.selected_dim)) > 2:
            return None
        if not layers:
            return {}

        virtual_mappings: list[torch.Tensor] = []
        used_logical: list[torch.Tensor] = []
        for layer in layers:
            selected = layer.latest_selected_experts
            device = (
                selected.device
                if selected is not None and selected.numel() > 0
                else _local_tensor_view(layer.gate_up_proj).device
            )
            virtual_mappings.append(layer.mapping_for_device(device).clone())
            used_logical.append(torch.zeros((layer.num_experts,), dtype=torch.bool, device=device))

        round_result = self._select_global_pair_lists_fast_2d_round(
            layers,
            virtual_mappings,
            used_logical,
            None,
            None,
        )
        return None if round_result is None else round_result[0]

    def _select_global_pair_lists_fast_2d_round(
        self,
        layers: list[ExpertLayerState],
        virtual_mappings: list[torch.Tensor],
        used_logical: list[torch.Tensor],
        pair_rows: list[torch.Tensor] | None,
        global_pair_counts: list[torch.Tensor] | None,
    ) -> (
        tuple[
            dict[str, list[tuple[int, int]]],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | None
    ):

        with _full_timing_range("hiermoe_expert_swap_select_local"):
            layer_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            num_group_rows: list[int] = []
            stat_groups: dict[tuple[int, int, torch.device], list[int]] = defaultdict(list)
            for layer_idx, layer in enumerate(layers):
                selected = layer.latest_selected_experts
                layer_device = (
                    selected.device
                    if selected is not None and selected.numel() > 0
                    else _local_tensor_view(layer.gate_up_proj).device
                )
                logical_to_physical = virtual_mappings[layer_idx]
                u_i = int(self.hierarchy.group_sizes[0])
                expert_group_size = max(1, layer.num_experts // max(1, self.hierarchy.ep_size // u_i))
                if layer.num_experts % expert_group_size != 0:
                    return None
                group_by_logical = torch.div(logical_to_physical, expert_group_size, rounding_mode="floor")
                num_groups = layer.num_experts // expert_group_size
                if selected is None or selected.numel() == 0:
                    selected = torch.empty((0, 1), dtype=torch.long, device=layer_device)
                expert_counts, base_group_counts, expert_group_counts = _selector_stats_2d(
                    selected,
                    layer.num_experts,
                    group_by_logical,
                    num_groups,
                )
                num_local_experts = max(1, layer.num_experts // self.hierarchy.ep_size)
                rank_by_logical = torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")
                base_rank_counts, expert_rank_counts = _selector_group_stats(
                    selected,
                    layer.num_experts,
                    rank_by_logical,
                    self.hierarchy.ep_size,
                )
                flat_stats = torch.cat(
                    (
                        expert_counts.reshape(-1),
                        base_group_counts.reshape(-1),
                        expert_group_counts.reshape(-1),
                        base_rank_counts.reshape(-1),
                        expert_rank_counts.reshape(-1),
                    ),
                    dim=0,
                )
                layer_stats.append((flat_stats, logical_to_physical, group_by_logical, selected))
                num_group_rows.append(num_groups)
                stat_groups[(layer.num_experts, num_groups, flat_stats.device)].append(len(layer_stats) - 1)

        if not layer_stats:
            return {}, [], []

        with _full_timing_range("hiermoe_expert_swap_select_candidates"):
            global_stats: list[torch.Tensor] = [stats[0] for stats in layer_stats]
            for (_num_experts, _num_groups, _device), indices in stat_groups.items():
                stacked = torch.stack([layer_stats[idx][0] for idx in indices], dim=0)
                if self.ep_group is not None and self.ep_size > 1:
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=self.ep_group)
                for row, layer_idx in zip(stacked, indices, strict=True):
                    global_stats[layer_idx] = row

            if pair_rows is None:
                pair_rows = []
                for layer, global_flat in zip(layers, global_stats, strict=True):
                    pairs = _all_candidate_pairs(layer.num_experts, global_flat.device)
                    pairs = self._cross_rank_pairs(layer, pairs)
                    pair_rows.append(pairs)
            max_candidates = max((int(pairs.shape[0]) for pairs in pair_rows), default=0)

        if max_candidates == 0:
            return {}, pair_rows, [pairs.new_zeros((0,), dtype=torch.float32) for pairs in pair_rows]

        with _full_timing_range("hiermoe_expert_swap_select_local"):
            if global_pair_counts is None:
                pair_count_rows: list[torch.Tensor] = []
                for pairs, (_flat_stats, _logical_to_physical, _group_by_logical, selected), layer in zip(
                    pair_rows,
                    layer_stats,
                    layers,
                    strict=True,
                ):
                    pair_count_rows.append(_candidate_pair_token_counts(selected, pairs, layer.num_experts))

                pair_count_groups: dict[torch.device, list[int]] = defaultdict(list)
                for idx, row in enumerate(pair_count_rows):
                    pair_count_groups[row.device].append(idx)
                global_pair_counts = pair_count_rows
                for _device, indices in pair_count_groups.items():
                    stacked = torch.zeros((len(indices), max_candidates), dtype=torch.float32, device=_device)
                    for row_idx, layer_idx in enumerate(indices):
                        row = pair_count_rows[layer_idx]
                        if row.numel() > 0:
                            stacked[row_idx, : row.numel()] = row
                    if self.ep_group is not None and self.ep_size > 1:
                        dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=self.ep_group)
                    for row_idx, layer_idx in enumerate(indices):
                        global_pair_counts[layer_idx] = stacked[row_idx, : pair_count_rows[layer_idx].numel()]

            cost_rows: list[torch.Tensor] = []
            current_cost_rows: list[torch.Tensor] = []
            for layer_idx, layer in enumerate(layers):
                global_flat = global_stats[layer_idx]
                num_experts = layer.num_experts
                num_groups = num_group_rows[layer_idx]
                base_offset = num_experts
                expert_group_offset = base_offset + num_groups
                rank_offset = expert_group_offset + num_experts * num_groups
                expert_rank_offset = rank_offset + self.hierarchy.ep_size
                expert_counts = global_flat[:num_experts]
                base_group_counts = global_flat[base_offset:expert_group_offset]
                expert_group_counts = global_flat[expert_group_offset:rank_offset].view(num_experts, num_groups)
                base_rank_counts = global_flat[rank_offset:expert_rank_offset]
                expert_rank_counts = global_flat[expert_rank_offset:].view(num_experts, self.hierarchy.ep_size)
                current_cost, costs = _costs_from_global_2d_stats(
                    expert_counts=expert_counts,
                    base_group_counts=base_group_counts,
                    expert_group_counts=expert_group_counts,
                    base_rank_counts=base_rank_counts,
                    expert_rank_counts=expert_rank_counts,
                    pair_counts=global_pair_counts[layer_idx],
                    pairs=pair_rows[layer_idx],
                    num_experts=num_experts,
                    hidden_size=layer.latest_hidden_size,
                    bytes_per_element=layer.latest_bytes_per_element,
                    hierarchy=self.hierarchy,
                    perf_model=self.perf_model,
                    gamma=self.smooth_max_gamma,
                    logical_to_physical=layer_stats[layer_idx][1],
                )
                current_cost_rows.append(current_cost)
                cost_rows.append(costs)

        device = cost_rows[0].device
        cost_tensor = torch.full((len(layers), max_candidates), float("inf"), dtype=torch.float32, device=device)
        current_costs = torch.zeros((len(layers),), dtype=torch.float32, device=device)
        pair_tensor = torch.full((len(layers), max_candidates, 2), -1, dtype=torch.long, device=device)
        for idx, (current_cost, costs, pairs) in enumerate(zip(current_cost_rows, cost_rows, pair_rows, strict=True)):
            num_candidates = int(pairs.shape[0])
            current_costs[idx] = current_cost.to(device=device, dtype=torch.float32, non_blocking=True)
            if num_candidates == 0:
                continue
            disallowed = used_logical[idx].index_select(0, pairs[:, 0]) | used_logical[idx].index_select(
                0, pairs[:, 1]
            )
            costs = costs.masked_fill(disallowed, float("inf"))
            cost_tensor[idx, :num_candidates] = costs.to(device=device, dtype=torch.float32, non_blocking=True)
            pair_tensor[idx, :num_candidates] = pairs.to(device=device, dtype=torch.long, non_blocking=True)

        chosen = self._select_pair_lists_from_cost_tensors(layers, cost_tensor, current_costs, pair_tensor)
        return chosen, pair_rows, global_pair_counts

    def _select_global_pairs_fast_2d(self, layers: list[ExpertLayerState]) -> dict[str, tuple[int, int]] | None:
        return self._first_pair_by_layer(self._select_global_pair_lists_fast_2d(layers))

    def _select_global_pair_lists_fast_hierarchy(
        self, layers: list[ExpertLayerState]
    ) -> dict[str, list[tuple[int, int]]] | None:
        if not _USE_GLOBAL_HIERARCHY_SELECTOR or max(1, int(self.hierarchy.selected_dim)) <= 2:
            return None
        max_dim = max(1, int(self.hierarchy.selected_dim))
        if len(self.hierarchy.group_sizes) < max_dim - 1 or not self.perf_model.inter:
            return None

        with _full_timing_range("hiermoe_expert_swap_select_local"):
            layer_stats: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]] = []
            level_group_rows: list[list[int]] = []
            stat_groups: dict[tuple[int, tuple[int, ...], torch.device], list[int]] = defaultdict(list)
            for layer in layers:
                selected = layer.latest_selected_experts
                layer_device = (
                    selected.device
                    if selected is not None and selected.numel() > 0
                    else _local_tensor_view(layer.gate_up_proj).device
                )
                logical_to_physical = layer.mapping_for_device(layer_device)
                level_shapes = _hierarchy_level_group_shapes(self.hierarchy, layer.num_experts)
                if len(level_shapes) < max_dim - 1:
                    return None
                if selected is None or selected.numel() == 0:
                    selected = torch.empty((0, 1), dtype=torch.long, device=layer_device)
                selected = selected.to(torch.long)
                expert_counts = torch.bincount(selected.reshape(-1), minlength=layer.num_experts).to(torch.float32)
                flat_parts: list[torch.Tensor] = [expert_counts.reshape(-1)]
                num_local_experts = max(1, layer.num_experts // self.hierarchy.ep_size)
                rank_by_logical = torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")
                base_rank_counts, expert_rank_counts = _selector_group_stats(
                    selected,
                    layer.num_experts,
                    rank_by_logical,
                    self.hierarchy.ep_size,
                )
                group_by_levels: list[torch.Tensor] = []
                num_group_rows: list[int] = []
                for u_i, num_groups in level_shapes[: max_dim - 1]:
                    expert_group_size = max(1, layer.num_experts // max(1, self.hierarchy.ep_size // u_i))
                    group_by_logical = torch.div(logical_to_physical, expert_group_size, rounding_mode="floor")
                    base_group_counts, expert_group_counts = _selector_group_stats(
                        selected,
                        layer.num_experts,
                        group_by_logical,
                        num_groups,
                    )
                    flat_parts.extend((base_group_counts.reshape(-1), expert_group_counts.reshape(-1)))
                    group_by_levels.append(group_by_logical)
                    num_group_rows.append(num_groups)
                flat_parts.extend((base_rank_counts.reshape(-1), expert_rank_counts.reshape(-1)))
                flat_stats = torch.cat(flat_parts, dim=0)
                layer_stats.append((flat_stats, logical_to_physical, group_by_levels, selected))
                level_group_rows.append(num_group_rows)
                stat_groups[(layer.num_experts, tuple(num_group_rows), flat_stats.device)].append(len(layer_stats) - 1)

        if not layer_stats:
            return {}

        with _full_timing_range("hiermoe_expert_swap_select_candidates"):
            global_stats: list[torch.Tensor] = [stats[0] for stats in layer_stats]
            for (_num_experts, _num_group_rows, _device), indices in stat_groups.items():
                stacked = torch.stack([layer_stats[idx][0] for idx in indices], dim=0)
                if self.ep_group is not None and self.ep_size > 1:
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=self.ep_group)
                for row, layer_idx in zip(stacked, indices, strict=True):
                    global_stats[layer_idx] = row

            pair_rows: list[torch.Tensor] = []
            max_candidates = 0
            for layer, global_flat in zip(layers, global_stats, strict=True):
                pairs = _all_candidate_pairs(layer.num_experts, global_flat.device)
                pairs = self._cross_rank_pairs(layer, pairs)
                pair_rows.append(pairs)
                max_candidates = max(max_candidates, int(pairs.shape[0]))

        if max_candidates == 0:
            return {}

        with _full_timing_range("hiermoe_expert_swap_select_local"):
            pair_count_rows: list[torch.Tensor] = []
            for pairs, (_flat_stats, _logical_to_physical, _group_by_levels, selected), layer in zip(
                pair_rows,
                layer_stats,
                layers,
                strict=True,
            ):
                pair_count_rows.append(_candidate_pair_token_counts(selected, pairs, layer.num_experts))

            pair_count_groups: dict[torch.device, list[int]] = defaultdict(list)
            for idx, row in enumerate(pair_count_rows):
                pair_count_groups[row.device].append(idx)
            global_pair_counts = pair_count_rows
            for _device, indices in pair_count_groups.items():
                stacked = torch.zeros((len(indices), max_candidates), dtype=torch.float32, device=_device)
                for row_idx, layer_idx in enumerate(indices):
                    row = pair_count_rows[layer_idx]
                    if row.numel() > 0:
                        stacked[row_idx, : row.numel()] = row
                if self.ep_group is not None and self.ep_size > 1:
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=self.ep_group)
                for row_idx, layer_idx in enumerate(indices):
                    global_pair_counts[layer_idx] = stacked[row_idx, : pair_count_rows[layer_idx].numel()]

            cost_rows: list[torch.Tensor] = []
            current_cost_rows: list[torch.Tensor] = []
            for layer_idx, layer in enumerate(layers):
                global_flat = global_stats[layer_idx]
                num_experts = layer.num_experts
                expert_counts = global_flat[:num_experts]
                offset = num_experts
                level_base_group_counts: list[torch.Tensor] = []
                level_expert_group_counts: list[torch.Tensor] = []
                for num_groups in level_group_rows[layer_idx]:
                    level_base_group_counts.append(global_flat[offset : offset + num_groups])
                    offset += num_groups
                    group_count = num_experts * num_groups
                    level_expert_group_counts.append(
                        global_flat[offset : offset + group_count].view(num_experts, num_groups)
                    )
                    offset += group_count
                base_rank_counts = global_flat[offset : offset + self.hierarchy.ep_size]
                offset += self.hierarchy.ep_size
                rank_count = num_experts * self.hierarchy.ep_size
                expert_rank_counts = global_flat[offset : offset + rank_count].view(
                    num_experts, self.hierarchy.ep_size
                )
                current_cost, costs = _costs_from_global_hierarchy_stats(
                    expert_counts=expert_counts,
                    base_rank_counts=base_rank_counts,
                    expert_rank_counts=expert_rank_counts,
                    level_base_group_counts=level_base_group_counts,
                    level_expert_group_counts=level_expert_group_counts,
                    pair_counts=global_pair_counts[layer_idx],
                    pairs=pair_rows[layer_idx],
                    num_experts=num_experts,
                    hidden_size=layer.latest_hidden_size,
                    bytes_per_element=layer.latest_bytes_per_element,
                    hierarchy=self.hierarchy,
                    perf_model=self.perf_model,
                    gamma=self.smooth_max_gamma,
                    logical_to_physical=layer_stats[layer_idx][1],
                )
                current_cost_rows.append(current_cost)
                cost_rows.append(costs)

        device = cost_rows[0].device
        cost_tensor = torch.full((len(layers), max_candidates), float("inf"), dtype=torch.float32, device=device)
        current_costs = torch.zeros((len(layers),), dtype=torch.float32, device=device)
        pair_tensor = torch.full((len(layers), max_candidates, 2), -1, dtype=torch.long, device=device)
        for idx, (current_cost, costs, pairs) in enumerate(zip(current_cost_rows, cost_rows, pair_rows, strict=True)):
            num_candidates = int(pairs.shape[0])
            current_costs[idx] = current_cost.to(device=device, dtype=torch.float32, non_blocking=True)
            if num_candidates == 0:
                continue
            cost_tensor[idx, :num_candidates] = costs.to(device=device, dtype=torch.float32, non_blocking=True)
            pair_tensor[idx, :num_candidates] = pairs.to(device=device, dtype=torch.long, non_blocking=True)

        return self._select_pair_lists_from_cost_tensors(layers, cost_tensor, current_costs, pair_tensor)

    def _select_global_pairs_fast_hierarchy(self, layers: list[ExpertLayerState]) -> dict[str, tuple[int, int]] | None:
        return self._first_pair_by_layer(self._select_global_pair_lists_fast_hierarchy(layers))

    def _select_global_pair_lists(self, layers: list[ExpertLayerState]) -> dict[str, list[tuple[int, int]]]:
        if int(self.expert_swap_max_pairs_per_layer) <= 0:
            return {}
        if int(self.expert_swap_max_pairs_per_layer) == 1:
            return {layer_key: [pair] for layer_key, pair in self._select_global_pairs(layers).items()}
        if not layers:
            return {}
        fast_pair_lists = self._select_global_pair_lists_fast_2d(layers)
        if fast_pair_lists is not None:
            return fast_pair_lists
        fast_pair_lists = self._select_global_pair_lists_fast_hierarchy(layers)
        if fast_pair_lists is not None:
            return fast_pair_lists
        return {layer_key: [pair] for layer_key, pair in self._select_global_pairs(layers).items()}

    def _select_global_pairs(self, layers: list[ExpertLayerState]) -> dict[str, tuple[int, int]]:
        if not layers:
            return {}
        fast_pairs = self._select_global_pairs_fast_2d(layers)
        if fast_pairs is not None:
            return fast_pairs
        fast_pairs = self._select_global_pairs_fast_hierarchy(layers)
        if fast_pairs is not None:
            return fast_pairs
        with _full_timing_range("hiermoe_expert_swap_select_candidates"):
            candidate_pairs = self._candidate_pairs_by_layer(layers)
        with _full_timing_range("hiermoe_expert_swap_select_local"):
            local_rows = [
                self._local_pair_candidate(layer, layer_candidate_pairs)
                for layer, layer_candidate_pairs in zip(layers, candidate_pairs, strict=True)
            ]
        device = local_rows[0].device
        local = torch.stack([row.to(device=device) for row in local_rows], dim=0)
        with _full_timing_range("hiermoe_expert_swap_select_global"):
            if self.ep_group is None or self.ep_size <= 1:
                chosen = local
            else:
                gathered_flat = torch.empty((self.ep_size * len(layers), 3), dtype=local.dtype, device=local.device)
                dist.all_gather_into_tensor(gathered_flat, local, group=self.ep_group)
                gathered = gathered_flat.view(self.ep_size, len(layers), 3)
                best_ranks = torch.argmin(gathered[:, :, 0], dim=0)
                chosen = gathered[best_ranks, torch.arange(len(layers), device=local.device)]

        pairs: dict[str, tuple[int, int]] = {}
        with _full_timing_range("hiermoe_expert_swap_select_finalize"):
            chosen_cpu = chosen.detach().to(torch.device("cpu"))
            for layer, row in zip(layers, chosen_cpu, strict=True):
                lhs = int(row[1].item())
                rhs = int(row[2].item())
                if lhs >= 0 and rhs >= 0:
                    pairs[layer.key] = (lhs, rhs)
        return pairs
