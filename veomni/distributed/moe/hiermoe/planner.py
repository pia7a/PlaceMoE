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

"""Current-route planning for HierMoE expert swaps and replicas."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch
import torch.distributed as dist

from veomni.utils.accelerator_timing import AcceleratorEvent, record_accelerator_event, synchronize_accelerator

from .perf_model import HierMoEPerfModel
from .topology import Hierarchy


ReduceSum = Callable[[torch.Tensor], torch.Tensor | None]
_FUSED_REPLICA_MAX_TOKENS = 16_384
_FUSED_REPLICA_OPS_CACHE: dict[tuple[str, int, bool], ModuleType | None] = {}
_EXACT_COPY_MAX_COMBINATIONS = 4_096
_COPY_CHOICE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


@dataclass(frozen=True)
class PlacementAction:
    kind: Literal["swap", "replica", "empty"]
    src_slot: int
    dst_slot: int
    src_logical: int
    dst_logical: int = -1

    def format(self) -> str:
        if self.kind == "swap":
            return f"swap({self.src_logical}<->{self.dst_logical})"
        if self.kind == "empty":
            return f"empty({self.src_logical}@{self.src_slot})"
        return f"replica({self.src_logical}->{self.dst_slot})"


@dataclass(frozen=True)
class PlacementCost:
    communication: float
    compute: float
    communication_model_units: float
    peak_communication_rank: int
    peak_compute_rank: int
    selected_dim: int
    state_move_exposed: float = 0.0
    gradient_sync: float = 0.0

    @property
    def total(self) -> float:
        return self.communication + self.compute + self.state_move_exposed + self.gradient_sync


@dataclass(frozen=True)
class PlacementPlan:
    actions: tuple[PlacementAction, ...]
    initial_layout: tuple[int, ...]
    final_layout: tuple[int, ...]
    baseline_cost: PlacementCost
    final_cost: PlacementCost
    swap_rounds: int
    replica_rounds: int
    planning_ms: float
    route_stats_ms: float
    swap_ms: float
    replica_ms: float
    swap_score_ms: float
    swap_update_ms: float
    swap_collective_ms: float
    replica_score_ms: float
    replica_update_ms: float
    replica_collective_ms: float
    decision_sync_ms: float
    finalization_ms: float
    device_timing_ms: dict[str, float] | None = None
    algorithm_version: str = "hiermoe-incremental-v1"
    quota_policy: tuple[tuple[int, ...], ...] = ()
    layout_digest: str = ""
    local_physical_routes: torch.Tensor | None = None
    final_owner_slots: tuple[int, ...] = ()

    @property
    def swaps(self) -> tuple[PlacementAction, ...]:
        return tuple(action for action in self.actions if action.kind == "swap")

    @property
    def replicas(self) -> tuple[PlacementAction, ...]:
        return tuple(action for action in self.actions if action.kind == "replica")


@dataclass(frozen=True)
class _TensorCost:
    communication: torch.Tensor
    compute: torch.Tensor
    communication_model_units: torch.Tensor
    peak_communication_rank: torch.Tensor
    peak_compute_rank: torch.Tensor
    selected_dim: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.communication + self.compute


@dataclass(frozen=True)
class _SwapStats:
    owner_ranks: torch.Tensor
    expert_token_counts: torch.Tensor
    expert_assignment_counts: torch.Tensor
    base_counts: tuple[torch.Tensor, ...]
    expert_group_counts: tuple[torch.Tensor, ...]
    sole_expert_counts: tuple[torch.Tensor, ...]
    sole_pair_counts: tuple[torch.Tensor, ...]
    local_token_group_counts: tuple[torch.Tensor, ...]


@dataclass
class _ReplicaStats:
    selected: torch.Tensor
    flat_logical: torch.Tensor
    route_scores: torch.Tensor
    route_hashes: torch.Tensor
    route_ranks: torch.Tensor
    minimum_scores: torch.Tensor
    tied_rank_order: torch.Tensor
    tie_count: torch.Tensor
    copy_rank_mask: torch.Tensor
    copy_slots_by_rank: torch.Tensor
    tokens_by_expert: torch.Tensor
    tokens_per_expert: torch.Tensor
    tokens_per_expert_i32: torch.Tensor
    route_indices_by_expert: torch.Tensor
    route_indices_by_expert_i32: torch.Tensor
    multiplicities_by_expert: torch.Tensor
    fused_route_tables: bool
    local_token_group_counts: tuple[torch.Tensor, ...]
    packed_local_token_group_counts: torch.Tensor
    base_counts: tuple[torch.Tensor, ...]
    assignment_counts: torch.Tensor


@dataclass(frozen=True)
class _ReplicaCandidateBatch:
    cost: _TensorCost
    base_counts: tuple[torch.Tensor, ...]
    assignment_counts: torch.Tensor
    route_ranks_by_destination: torch.Tensor


def apply_placement_action(layout: torch.Tensor, action: PlacementAction) -> torch.Tensor:
    updated = layout.clone()
    if action.kind == "swap":
        updated[action.src_slot], updated[action.dst_slot] = (
            updated[action.dst_slot].clone(),
            updated[action.src_slot].clone(),
        )
    elif action.kind == "replica":
        updated[action.dst_slot] = int(action.src_logical)
    elif action.kind == "empty":
        updated[action.dst_slot] = -1
    else:  # pragma: no cover - Literal protects normal callers
        raise ValueError(f"Unknown placement action kind: {action.kind!r}.")
    return updated


def _reduce_sum(tensor: torch.Tensor, reducer: ReduceSum | None) -> torch.Tensor:
    if reducer is None:
        return tensor
    reduced = tensor.clone()
    result = reducer(reduced)
    return reduced if result is None else result


def _hierarchy_distance(source_ranks: torch.Tensor, destination_ranks: torch.Tensor, group_sizes: Sequence[int]):
    source = source_ranks.view(1, -1, 1, 1)
    distance = torch.full_like(destination_ranks, len(group_sizes) + 1, dtype=torch.long)
    distance = torch.where(destination_ranks == source, torch.zeros_like(distance), distance)
    for level, raw_size in reversed(tuple(enumerate(group_sizes, start=1))):
        size = max(1, int(raw_size))
        same_group = torch.div(destination_ranks, size, rounding_mode="floor") == torch.div(
            source, size, rounding_mode="floor"
        )
        distance = torch.where(same_group & (destination_ranks != source), torch.full_like(distance, level), distance)
    return distance


def _route_hash(
    selected: torch.Tensor,
    *,
    token_ordinals: torch.Tensor | None,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    num_tokens = selected.shape[0]
    if token_ordinals is None:
        token_ids = torch.arange(num_tokens, device=selected.device, dtype=torch.long)
    else:
        token_ids = token_ordinals.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if token_ids.numel() != num_tokens:
            raise ValueError(f"token_ordinals has {token_ids.numel()} values for {num_tokens} tokens.")
    token_ids = token_ids.view(num_tokens, 1)
    logical = selected.to(torch.long)
    value = token_ids * 1_000_003 + logical * 65_537 + int(step) * 131 + int(layer_seed) * 17
    value = torch.remainder(value * 48_271 + 1, 2_147_483_647)
    return torch.remainder(value, 1_048_573)


def assign_tokens_to_mirrored_r2(
    selected_experts: torch.Tensor,
    copy_slots: torch.Tensor,
    *,
    source_ranks: int | torch.Tensor,
    num_ranks: int,
) -> torch.Tensor:
    """Map a mirrored R2 layout to the copy in the source rank's EP half."""

    original_ndim = selected_experts.ndim
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if num_ranks <= 1 or num_ranks % 2 != 0:
        raise ValueError(f"Mirrored R2 requires a positive even rank count, got {num_ranks}.")
    if copy_slots.ndim != 2 or copy_slots.shape[1] != 2:
        raise ValueError(f"Mirrored R2 copy_slots must have shape [experts, 2], got {tuple(copy_slots.shape)}.")

    copies = copy_slots.to(device=selected.device, dtype=torch.long, non_blocking=True)
    routed_copies = copies.index_select(0, selected.reshape(-1)).view(*selected.shape, 2)
    num_tokens = int(selected.shape[0])
    if isinstance(source_ranks, int):
        use_second_half = torch.full(
            (num_tokens, 1),
            int(source_ranks) >= num_ranks // 2,
            dtype=torch.bool,
            device=selected.device,
        )
    else:
        source = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if int(source.numel()) != num_tokens:
            raise ValueError(f"source_ranks has {source.numel()} values for {num_tokens} tokens.")
        use_second_half = (source >= num_ranks // 2).view(-1, 1)
    physical = torch.where(use_second_half, routed_copies[..., 1], routed_copies[..., 0])
    return physical.squeeze(-1) if original_ndim == 1 else physical


def _copy_scores(
    selected: torch.Tensor,
    copy_ranks: torch.Tensor,
    source_ranks: torch.Tensor,
    owner_slots: torch.Tensor,
    *,
    slots_per_rank: int,
    num_ranks: int,
    hierarchy_group_sizes: Sequence[int],
) -> torch.Tensor:
    """Return lexicographic copy scores without route-order dependencies."""

    num_tokens, top_k = selected.shape
    owner_ranks = torch.div(owner_slots, max(1, int(slots_per_rank)), rounding_mode="floor")
    route_owner_ranks = owner_ranks.index_select(0, selected.reshape(-1)).view(num_tokens, top_k)
    owner_rank_counts = torch.zeros((num_tokens, max(1, int(num_ranks))), dtype=torch.long, device=selected.device)
    owner_rank_counts.scatter_add_(1, route_owner_ranks, torch.ones_like(route_owner_ranks))

    batch, _, _, num_copies = copy_ranks.shape
    flat_copy_ranks = copy_ranks.reshape(batch, num_tokens, top_k * num_copies)
    needed_counts = owner_rank_counts.unsqueeze(0).expand(batch, -1, -1).gather(2, flat_copy_ranks)
    needed_counts = needed_counts.view(batch, num_tokens, top_k, num_copies)
    is_own_owner = copy_ranks == route_owner_ranks.view(1, num_tokens, top_k, 1)
    already_needed = needed_counts > is_own_owner.to(torch.long)

    distance = _hierarchy_distance(source_ranks, copy_ranks, hierarchy_group_sizes)
    distance_scale = len(hierarchy_group_sizes) + 2
    return (~already_needed).to(torch.long) * distance_scale + distance


def _copy_choice_table(top_k: int, num_copies: int, device: torch.device) -> torch.Tensor:
    cache_key = (str(device), int(top_k), int(num_copies))
    cached = _COPY_CHOICE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    combinations = int(num_copies) ** int(top_k)
    combination_ids = torch.arange(combinations, dtype=torch.long, device=device).view(-1, 1)
    powers = torch.tensor(
        [int(num_copies) ** index for index in range(int(top_k))],
        dtype=torch.long,
        device=device,
    ).view(1, -1)
    cached = torch.remainder(torch.div(combination_ids, powers, rounding_mode="floor"), int(num_copies))
    _COPY_CHOICE_CACHE[cache_key] = cached
    return cached


def _remote_group_count_for_choices(
    selected_copy_slots: torch.Tensor,
    choices: torch.Tensor,
    source_groups: torch.Tensor,
    *,
    slots_per_rank: int,
    group_size: int,
) -> torch.Tensor:
    num_tokens, top_k, _num_copies = selected_copy_slots.shape
    num_combinations = int(choices.shape[0])
    counts = torch.zeros((num_tokens, num_combinations), dtype=torch.long, device=selected_copy_slots.device)
    previous_groups = []
    for route_index in range(top_k):
        option_indices = choices[:, route_index]
        chosen_slots = selected_copy_slots[:, route_index].index_select(1, option_indices)
        chosen_ranks = torch.div(chosen_slots, int(slots_per_rank), rounding_mode="floor")
        chosen_groups = torch.div(chosen_ranks, int(group_size), rounding_mode="floor")
        is_first_remote = chosen_groups != source_groups.view(-1, 1)
        for previous in previous_groups:
            is_first_remote.logical_and_(chosen_groups != previous)
        counts.add_(is_first_remote)
        previous_groups.append(chosen_groups)
    return counts


def _assign_tokens_to_copies_exact_vectorized(
    selected: torch.Tensor,
    selected_copy_slots: torch.Tensor,
    selected_copy_valid: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    num_ranks: int,
    token_ordinals: torch.Tensor | None,
    step: int,
    layer_seed: int,
) -> torch.Tensor | None:
    num_tokens, top_k = selected.shape
    num_copies = int(selected_copy_slots.shape[-1])
    combinations = num_copies**top_k
    if combinations > _EXACT_COPY_MAX_COMBINATIONS:
        return None
    if num_tokens == 0:
        return selected.clone()

    choices = _copy_choice_table(top_k, num_copies, selected.device)
    num_combinations = int(choices.shape[0])
    valid = torch.ones((num_tokens, num_combinations), dtype=torch.bool, device=selected.device)
    for route_index in range(top_k):
        valid.logical_and_(selected_copy_valid[:, route_index].index_select(1, choices[:, route_index]))

    sorted_selected = selected.sort(dim=-1).values
    if top_k > 1 and bool((sorted_selected[:, 1:] == sorted_selected[:, :-1]).any().item()):
        for lhs in range(top_k):
            lhs_slots = selected_copy_slots[:, lhs].index_select(1, choices[:, lhs])
            for rhs in range(lhs + 1, top_k):
                same_logical = selected[:, lhs] == selected[:, rhs]
                if bool(same_logical.any().item()):
                    rhs_slots = selected_copy_slots[:, rhs].index_select(1, choices[:, rhs])
                    valid.logical_and_(
                        torch.logical_or(
                            torch.logical_not(same_logical.view(-1, 1)),
                            lhs_slots == rhs_slots,
                        )
                    )

    hierarchy_levels = sorted(
        {max(1, int(group_size)) for group_size in hierarchy_group_sizes if 1 < int(group_size) < int(num_ranks)},
        reverse=True,
    )
    hierarchy_levels.append(1)
    score = torch.zeros((num_tokens, num_combinations), dtype=torch.long, device=selected.device)
    score_base = top_k + 1
    for group_size in hierarchy_levels:
        source_groups = torch.div(source_ranks, int(group_size), rounding_mode="floor")
        group_count = _remote_group_count_for_choices(
            selected_copy_slots,
            choices,
            source_groups,
            slots_per_rank=slots_per_rank,
            group_size=group_size,
        )
        score.mul_(score_base).add_(group_count)

    invalid_score = torch.iinfo(torch.long).max
    score = torch.where(valid, score, torch.full_like(score, invalid_score))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = valid & (score == minimum)
    route_hash = _route_hash(
        selected,
        token_ordinals=token_ordinals,
        step=step,
        layer_seed=layer_seed,
    )
    tie_modulus = 2_147_483_647
    tie_target = torch.remainder(route_hash.sum(dim=-1, keepdim=True), tie_modulus)
    candidate_hash = torch.zeros_like(score)
    for route_index in range(top_k):
        option_indices = choices[:, route_index]
        chosen_slots = selected_copy_slots[:, route_index].index_select(1, option_indices)
        candidate_hash.mul_(1_000_003).add_(chosen_slots + 1)
        candidate_hash.remainder_(tie_modulus)
    tie_mixed = torch.remainder(
        candidate_hash * (tie_target + 1_000_003) + tie_target * 48_271 + 1,
        tie_modulus,
    )
    tie_score = torch.remainder(tie_mixed * 48_271 + 1, tie_modulus)
    best_combination = torch.where(
        tied,
        tie_score,
        torch.full_like(tie_score, tie_modulus),
    ).argmin(dim=-1)
    best_choices = choices.index_select(0, best_combination)
    return selected_copy_slots.gather(2, best_choices.unsqueeze(-1)).squeeze(-1)


def assign_tokens_to_copies(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    owner_slots: torch.Tensor | None = None,
    token_ordinals: torch.Tensor | None = None,
    step: int = 0,
    layer_seed: int = 0,
    max_copies: int = 2,
    copy_slots: torch.Tensor | None = None,
    copy_mask: torch.Tensor | None = None,
    validate_copy_table: bool = True,
) -> torch.Tensor:
    """Map logical routes to physical slots with deterministic locality priorities.

    The function accepts one layout ``[slots]`` or a candidate batch
    ``[candidates, slots]``. Candidate layouts are evaluated independently.
    """

    original_ndim = selected_experts.ndim
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    layouts = slot_to_logical.to(device=selected.device, dtype=torch.long, non_blocking=True)
    squeeze_layout = layouts.ndim == 1
    if squeeze_layout:
        layouts = layouts.unsqueeze(0)
    if layouts.ndim != 2:
        raise ValueError(f"slot_to_logical must be rank 1 or 2, got shape={tuple(layouts.shape)}.")

    batch, num_slots = layouts.shape
    owners = None
    if owner_slots is not None:
        owners = owner_slots.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        num_experts = int(owners.numel())
    else:
        selected_max = int(selected.max().item()) if selected.numel() else -1
        layout_max = int(layouts.max().item()) if layouts.numel() else -1
        num_experts = max(selected_max, layout_max) + 1
    if num_experts == 0:
        empty = selected.unsqueeze(0).expand(batch, *selected.shape)
        return empty[0] if squeeze_layout else empty
    if (layouts >= num_experts).any():
        raise ValueError("slot_to_logical contains a logical expert outside selected_experts' expert range.")
    if owners is None:
        first_layout = layouts[0]
        slot_index = torch.arange(num_slots, device=selected.device, dtype=torch.long)
        owners = torch.full((num_experts,), num_slots, device=selected.device, dtype=torch.long)
        owners.scatter_reduce_(
            0,
            first_layout.clamp_min(0),
            torch.where(first_layout >= 0, slot_index, torch.full_like(slot_index, num_slots)),
            reduce="amin",
            include_self=True,
        )

    if copy_slots is None:
        copy_limit = max(1, min(int(max_copies), num_slots))
        logical_ids = torch.arange(num_experts, device=selected.device, dtype=torch.long)
        slot_ids = torch.arange(num_slots, device=selected.device, dtype=torch.long).view(1, num_slots, 1)
        matches = layouts.unsqueeze(-1) == logical_ids.view(1, 1, num_experts)
        masked_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
        routed_copy_slots = masked_slots.sort(dim=1).values[:, :copy_limit].transpose(1, 2).contiguous()
        copy_valid = routed_copy_slots < num_slots
    else:
        cached_slots = copy_slots.to(device=selected.device, dtype=torch.long, non_blocking=True)
        if cached_slots.ndim != 2 or cached_slots.shape[0] != num_experts:
            raise ValueError(
                f"copy_slots must have shape [{num_experts}, copies], got shape={tuple(cached_slots.shape)}."
            )
        if cached_slots.shape[1] == 0:
            raise ValueError("copy_slots must contain at least one copy column.")
        routed_copy_slots = cached_slots.unsqueeze(0).expand(batch, -1, -1)
        if copy_mask is None:
            requested_valid = routed_copy_slots >= 0
        else:
            cached_mask = copy_mask.to(device=selected.device, dtype=torch.bool, non_blocking=True)
            if cached_mask.shape != cached_slots.shape:
                raise ValueError(
                    f"copy_mask must match copy_slots shape={tuple(cached_slots.shape)}, got {tuple(cached_mask.shape)}."
                )
            requested_valid = cached_mask.unsqueeze(0).expand(batch, -1, -1)
        in_bounds = (routed_copy_slots >= 0) & (routed_copy_slots < num_slots)
        safe_cached_slots = routed_copy_slots.clamp(min=0, max=max(0, num_slots - 1))
        cached_logicals = layouts.gather(1, safe_cached_slots.reshape(batch, -1)).view_as(routed_copy_slots)
        expected_logicals = torch.arange(num_experts, device=selected.device).view(1, num_experts, 1)
        matches_logical = cached_logicals == expected_logicals
        if validate_copy_table and bool((requested_valid & ~(in_bounds & matches_logical)).any().item()):
            raise ValueError("copy_slots contains a masked slot that does not hold the corresponding logical expert.")
        copy_valid = requested_valid & in_bounds & matches_logical
        copy_limit = int(cached_slots.shape[1])
        routed_copy_slots = torch.where(copy_valid, routed_copy_slots, torch.full_like(routed_copy_slots, num_slots))
    if (routed_copy_slots[:, :, 0] >= num_slots).any():
        raise ValueError("Every logical expert must retain at least one physical copy.")

    num_tokens, top_k = selected.shape
    flat_selected = selected.reshape(-1)
    routed_slots = routed_copy_slots.index_select(1, flat_selected).view(batch, num_tokens, top_k, copy_limit)
    valid = copy_valid.index_select(1, flat_selected).view(batch, num_tokens, top_k, copy_limit)
    safe_slots = routed_slots.clamp(max=max(0, num_slots - 1))
    copy_ranks = torch.div(safe_slots, max(1, int(slots_per_rank)), rounding_mode="floor")
    if isinstance(source_ranks, int):
        source = torch.full((num_tokens,), int(source_ranks), dtype=torch.long, device=selected.device)
    else:
        source = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if int(source.numel()) != num_tokens:
            raise ValueError(f"source_ranks has {source.numel()} values for {num_tokens} tokens.")

    if squeeze_layout:
        exact = _assign_tokens_to_copies_exact_vectorized(
            selected,
            safe_slots[0],
            valid[0],
            slots_per_rank=max(1, int(slots_per_rank)),
            source_ranks=source,
            hierarchy_group_sizes=hierarchy_group_sizes,
            num_ranks=max(1, num_slots // max(1, int(slots_per_rank))),
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
        if exact is not None:
            return exact.squeeze(-1) if original_ndim == 1 else exact

    score = _copy_scores(
        selected,
        copy_ranks,
        source,
        owners,
        slots_per_rank=slots_per_rank,
        num_ranks=max(1, num_slots // max(1, int(slots_per_rank))),
        hierarchy_group_sizes=hierarchy_group_sizes,
    )
    score = torch.where(valid, score, torch.full_like(score, 1 << 50))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = valid & (score == minimum)
    tie_order = tied.to(torch.long).cumsum(dim=-1) - 1
    route_hash = _route_hash(selected, token_ordinals=token_ordinals, step=step, layer_seed=layer_seed)
    target = torch.remainder(
        route_hash.view(1, num_tokens, top_k, 1),
        tied.sum(dim=-1, keepdim=True),
    )
    chosen = (tied & (tie_order == target)).to(torch.long).argmax(dim=-1, keepdim=True)
    physical = safe_slots.gather(-1, chosen).squeeze(-1)
    if squeeze_layout:
        physical = physical[0]
    if original_ndim == 1:
        physical = physical.squeeze(-1)
    return physical


class CurrentRoutePlanner:
    """Greedy swap-then-replica planner using current token routes."""

    def __init__(
        self,
        *,
        hierarchy: Hierarchy,
        perf_model: HierMoEPerfModel,
        hidden_size: int,
        bytes_per_element: int,
        slots_per_rank: int,
        communication_scale: float = 1.0,
        forward_compute_per_assignment: float = 0.0,
        reducer: ReduceSum | None = None,
        candidate_chunk_size: int = 32,
        record_device_timing: bool = False,
    ) -> None:
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.hidden_size = int(hidden_size)
        self.bytes_per_element = int(bytes_per_element)
        self.slots_per_rank = int(slots_per_rank)
        self.communication_scale = float(communication_scale)
        self.forward_compute_per_assignment = float(forward_compute_per_assignment)
        self.reducer = reducer
        self.candidate_chunk_size = max(1, int(candidate_chunk_size))
        self.record_device_timing = bool(record_device_timing)
        self.last_replica_timing_ms: dict[str, float] = {}
        self._last_swap_collective_ms = 0.0
        self._last_replica_collective_ms = 0.0
        self._rank_distance_cache: dict[str, torch.Tensor] = {}
        self._swap_pair_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._device_event_pairs: dict[str, list[tuple[AcceleratorEvent | None, AcceleratorEvent | None]]] | None = (
            None
        )

    def _device_event(self) -> AcceleratorEvent | None:
        return record_accelerator_event() if self._device_event_pairs is not None else None

    def _record_device_interval(
        self,
        name: str,
        start: AcceleratorEvent | None,
        end: AcceleratorEvent | None,
    ) -> None:
        if self._device_event_pairs is not None:
            self._device_event_pairs.setdefault(name, []).append((start, end))

    def _reduce_with_timing(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        start = self._device_event()
        reduced = _reduce_sum(tensor, self.reducer)
        self._record_device_interval(name, start, self._device_event())
        return reduced

    def _finalize_device_timing(self) -> dict[str, float] | None:
        pairs = self._device_event_pairs
        if pairs is None:
            return None
        synchronize_accelerator()
        timings = {
            name: sum(start.elapsed_time(end) for start, end in intervals if start is not None and end is not None)
            for name, intervals in pairs.items()
        }
        self._device_event_pairs = None
        return timings

    @property
    def ep_size(self) -> int:
        return int(self.hierarchy.ep_size)

    @property
    def payload_bytes(self) -> int:
        return self.hidden_size * self.bytes_per_element

    def _rank_distances(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._rank_distance_cache.get(key)
        if cached is not None:
            return cached
        ranks = torch.arange(self.ep_size, dtype=torch.long, device=device)
        destinations = ranks.view(1, 1, 1, self.ep_size).expand(1, self.ep_size, 1, -1)
        distances = _hierarchy_distance(ranks, destinations, self.hierarchy.group_sizes).view(
            self.ep_size, self.ep_size
        )
        self._rank_distance_cache[key] = distances
        return distances

    def _communication_costs(self, base_counts: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rank_counts = base_counts[-1]
        per_dim = [
            self.perf_model.a2a.alpha
            + float(self.ep_size * self.payload_bytes) * rank_counts.max(dim=-1).values * self.perf_model.a2a.beta
        ]
        previous_u = 1
        running = torch.zeros_like(per_dim[0])
        level_count = len(base_counts) - 1
        for level_idx in range(level_count):
            u_i = int(self.hierarchy.group_sizes[level_idx])
            link = self.perf_model.inter[min(level_idx, len(self.perf_model.inter) - 1)]
            running = (
                running
                + link.alpha
                + float((u_i / previous_u) * self.payload_bytes)
                * (base_counts[level_idx].max(dim=-1).values * link.beta)
            )
            intra = self.perf_model.intra.alpha + float((self.ep_size / u_i) * self.payload_bytes) * (
                rank_counts.max(dim=-1).values * self.perf_model.intra.beta
            )
            per_dim.append(running + intra)
            previous_u = u_i
        stacked = torch.stack(per_dim, dim=-1)
        if self.perf_model.source == "default":
            selected = torch.full(
                stacked.shape[:-1],
                min(2, stacked.shape[-1]) - 1,
                dtype=torch.long,
                device=stacked.device,
            )
        else:
            selected = stacked.argmin(dim=-1)
        one_way = stacked.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        return 4.0 * one_way, selected + 1

    def _tensor_cost(self, base_counts: Sequence[torch.Tensor], assignment_counts: torch.Tensor) -> _TensorCost:
        communication_units, selected_dim = self._communication_costs(base_counts)
        communication = communication_units * self.communication_scale
        peak_assignments, peak_compute_rank = assignment_counts.max(dim=-1)
        compute = 3.0 * self.forward_compute_per_assignment * peak_assignments
        peak_communication_rank = base_counts[-1].argmax(dim=-1)
        return _TensorCost(
            communication=communication,
            compute=compute,
            communication_model_units=communication_units,
            peak_communication_rank=peak_communication_rank,
            peak_compute_rank=peak_compute_rank,
            selected_dim=selected_dim,
        )

    def _local_swap_parts(
        self,
        token_hits: torch.Tensor,
        owner_slots: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[tuple[int, int], ...],
        tuple[torch.Tensor, ...],
    ]:
        owner_ranks = torch.div(owner_slots, self.slots_per_rank, rounding_mode="floor")
        group_maps = [
            (
                torch.div(owner_ranks, int(size), rounding_mode="floor"),
                self.ep_size // int(size),
            )
            for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ]
        group_maps.append((owner_ranks, self.ep_size))
        parts: list[torch.Tensor] = []
        shapes: list[tuple[int, int]] = []
        local_token_group_counts: list[torch.Tensor] = []
        for group_by_logical, num_groups in group_maps:
            group_index = group_by_logical.view(1, -1).expand(token_hits.shape[0], -1)
            token_group_counts = torch.zeros(
                (token_hits.shape[0], num_groups), dtype=torch.float32, device=token_hits.device
            )
            token_group_counts.scatter_add_(1, group_index, token_hits)
            local_token_group_counts.append(token_group_counts)
            token_group_hits = (token_group_counts > 0).to(torch.float32)
            own_group_counts = token_group_counts.index_select(1, group_by_logical)
            sole_expert_hits = token_hits * (own_group_counts == 1)
            parts.extend(
                (
                    token_group_hits.sum(dim=0),
                    token_hits.transpose(0, 1).matmul(token_group_hits).reshape(-1),
                    sole_expert_hits.sum(dim=0),
                    sole_expert_hits.transpose(0, 1).matmul(token_hits).reshape(-1),
                )
            )
            shapes.append((num_groups, token_hits.shape[1]))
        return owner_ranks, tuple(parts), tuple(shapes), tuple(local_token_group_counts)

    @staticmethod
    def _unpack_swap_stats(
        owner_ranks: torch.Tensor,
        expert_token_counts: torch.Tensor,
        expert_assignment_counts: torch.Tensor,
        reduced: torch.Tensor,
        shapes: Sequence[tuple[int, int]],
        local_token_group_counts: Sequence[torch.Tensor],
    ) -> _SwapStats:
        base_counts: list[torch.Tensor] = []
        expert_group_counts: list[torch.Tensor] = []
        sole_expert_counts: list[torch.Tensor] = []
        sole_pair_counts: list[torch.Tensor] = []
        offset = 0
        for num_groups, num_experts in shapes:
            base_counts.append(reduced[offset : offset + num_groups])
            offset += num_groups
            expert_group_size = num_experts * num_groups
            expert_group_counts.append(reduced[offset : offset + expert_group_size].view(num_experts, num_groups))
            offset += expert_group_size
            sole_expert_counts.append(reduced[offset : offset + num_experts])
            offset += num_experts
            pair_size = num_experts * num_experts
            sole_pair_counts.append(reduced[offset : offset + pair_size].view(num_experts, num_experts))
            offset += pair_size
        return _SwapStats(
            owner_ranks=owner_ranks,
            expert_token_counts=expert_token_counts,
            expert_assignment_counts=expert_assignment_counts,
            base_counts=tuple(base_counts),
            expert_group_counts=tuple(expert_group_counts),
            sole_expert_counts=tuple(sole_expert_counts),
            sole_pair_counts=tuple(sole_pair_counts),
            local_token_group_counts=tuple(local_token_group_counts),
        )

    def _initial_swap_stats(
        self,
        token_hits: torch.Tensor,
        selected: torch.Tensor,
        owner_slots: torch.Tensor,
    ) -> _SwapStats:
        owner_ranks, local_parts, shapes, local_token_group_counts = self._local_swap_parts(token_hits, owner_slots)
        local_assignment_counts = torch.zeros((token_hits.shape[1],), dtype=torch.float32, device=selected.device)
        local_assignment_counts.scatter_add_(
            0,
            selected.reshape(-1),
            torch.ones(selected.numel(), dtype=torch.float32, device=selected.device),
        )
        local_token_counts = token_hits.sum(dim=0)
        reduced = self._reduce_with_timing(
            torch.cat((local_token_counts, local_assignment_counts, *local_parts), dim=0),
            "route_collective",
        )
        num_experts = local_assignment_counts.numel()
        expert_token_counts = reduced[:num_experts]
        expert_assignment_counts = reduced[num_experts : 2 * num_experts]
        return self._unpack_swap_stats(
            owner_ranks,
            expert_token_counts,
            expert_assignment_counts,
            reduced[2 * num_experts :],
            shapes,
            local_token_group_counts,
        )

    def _swap_candidate_costs(
        self,
        stats: _SwapStats,
        pairs: torch.Tensor,
    ) -> tuple[_TensorCost, tuple[torch.Tensor, ...]]:
        group_maps = [
            torch.div(stats.owner_ranks, int(size), rounding_mode="floor")
            for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ]
        group_maps.append(stats.owner_ranks)
        candidate_groups: list[torch.Tensor] = []
        for level, group_by_logical in enumerate(group_maps):
            lhs, rhs = pairs[:, 0], pairs[:, 1]
            lhs_group = group_by_logical.index_select(0, lhs)
            rhs_group = group_by_logical.index_select(0, rhs)
            same_group = lhs_group == rhs_group
            lhs_count = stats.expert_token_counts.index_select(0, lhs)
            rhs_count = stats.expert_token_counts.index_select(0, rhs)
            expert_group = stats.expert_group_counts[level]
            sole_expert = stats.sole_expert_counts[level]
            sole_pair = stats.sole_pair_counts[level]
            delta_lhs = (
                rhs_count - expert_group[rhs, lhs_group] - sole_expert.index_select(0, lhs) + sole_pair[lhs, rhs]
            )
            delta_rhs = (
                lhs_count - expert_group[lhs, rhs_group] - sole_expert.index_select(0, rhs) + sole_pair[rhs, lhs]
            )
            delta_lhs = torch.where(same_group, torch.zeros_like(delta_lhs), delta_lhs)
            delta_rhs = torch.where(same_group, torch.zeros_like(delta_rhs), delta_rhs)
            counts = stats.base_counts[level].unsqueeze(0).expand(pairs.shape[0], -1).clone()
            counts.scatter_add_(1, lhs_group.unsqueeze(1), delta_lhs.unsqueeze(1))
            counts.scatter_add_(1, rhs_group.unsqueeze(1), delta_rhs.unsqueeze(1))
            candidate_groups.append(counts)

        assignment = torch.zeros(
            (self.ep_size,), dtype=torch.float32, device=stats.expert_assignment_counts.device
        ).scatter_add_(0, stats.owner_ranks, stats.expert_assignment_counts)
        candidate_assignment = assignment.unsqueeze(0).expand(pairs.shape[0], -1).clone()
        lhs, rhs = pairs[:, 0], pairs[:, 1]
        lhs_rank = stats.owner_ranks.index_select(0, lhs)
        rhs_rank = stats.owner_ranks.index_select(0, rhs)
        lhs_count = stats.expert_assignment_counts.index_select(0, lhs)
        rhs_count = stats.expert_assignment_counts.index_select(0, rhs)
        candidate_assignment.scatter_add_(1, lhs_rank.unsqueeze(1), (rhs_count - lhs_count).unsqueeze(1))
        candidate_assignment.scatter_add_(1, rhs_rank.unsqueeze(1), (lhs_count - rhs_count).unsqueeze(1))
        return self._tensor_cost(candidate_groups, candidate_assignment), tuple(candidate_groups)

    def _current_swap_cost(self, stats: _SwapStats) -> _TensorCost:
        assignment = torch.zeros(
            (self.ep_size,), dtype=torch.float32, device=stats.expert_assignment_counts.device
        ).scatter_add_(0, stats.owner_ranks, stats.expert_assignment_counts)
        return self._tensor_cost([row.unsqueeze(0) for row in stats.base_counts], assignment.unsqueeze(0))

    def _update_swap_stats(
        self,
        token_hits: torch.Tensor,
        stats: _SwapStats,
        pair: torch.Tensor,
        updated_base_counts: Sequence[torch.Tensor],
    ) -> _SwapStats:
        updated_owner_ranks = stats.owner_ranks.clone()
        updated_owner_ranks.scatter_(0, pair, stats.owner_ranks.index_select(0, pair).flip(0))
        lhs, rhs = pair[0], pair[1]
        lhs_hits = token_hits.index_select(1, lhs.view(1)).squeeze(1)
        rhs_hits = token_hits.index_select(1, rhs.view(1)).squeeze(1)
        delta = rhs_hits - lhs_hits
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)

        local_payload: list[torch.Tensor] = []
        updated_local_counts: list[torch.Tensor] = []
        level_metadata: list[tuple[torch.Tensor, torch.Tensor]] = []
        num_experts = int(stats.owner_ranks.numel())
        expert_ids = torch.arange(num_experts, dtype=torch.long, device=token_hits.device)
        for level, size in enumerate(level_sizes):
            old_group_by_logical = torch.div(stats.owner_ranks, size, rounding_mode="floor")
            new_group_by_logical = torch.div(updated_owner_ranks, size, rounding_mode="floor")
            groups = old_group_by_logical.index_select(0, pair)

            local_counts = stats.local_token_group_counts[level].clone()
            group_index = groups.view(1, 2).expand(token_hits.shape[0], -1)
            local_counts.scatter_add_(1, group_index, torch.stack((delta, -delta), dim=1))
            updated_local_counts.append(local_counts)

            group_hits = (local_counts.index_select(1, groups) > 0).to(torch.float32)
            local_expert_group_columns = token_hits.transpose(0, 1).matmul(group_hits)

            affected = (new_group_by_logical.unsqueeze(1) == groups.view(1, 2)).any(dim=1)
            experts_per_group = max(1, num_experts // int(local_counts.shape[1]))
            affected_width = min(num_experts, 2 * experts_per_group)
            affected_experts = torch.where(affected, expert_ids, torch.full_like(expert_ids, num_experts))
            affected_experts = affected_experts.sort().values[:affected_width]
            affected_valid = affected_experts < num_experts
            safe_experts = affected_experts.clamp(max=max(0, num_experts - 1))
            affected_groups = new_group_by_logical.index_select(0, safe_experts)
            own_group_counts = local_counts.index_select(1, affected_groups)
            sole_hits = token_hits.index_select(1, safe_experts) * (own_group_counts == 1).to(torch.float32)
            sole_hits = sole_hits * affected_valid.to(torch.float32).view(1, -1)
            local_sole = sole_hits.sum(dim=0)
            local_sole_pairs = sole_hits.transpose(0, 1).matmul(token_hits)

            local_sole_full = torch.zeros((num_experts,), dtype=torch.float32, device=token_hits.device)
            local_pair_full = torch.zeros((num_experts, num_experts), dtype=torch.float32, device=token_hits.device)
            local_sole_full.index_add_(0, safe_experts, local_sole)
            local_pair_full.index_add_(0, safe_experts, local_sole_pairs)
            local_payload.extend(
                (local_expert_group_columns.reshape(-1), local_sole_full, local_pair_full.reshape(-1))
            )
            level_metadata.append((groups, affected))

        collective_started = time.perf_counter()
        reduced = self._reduce_with_timing(torch.cat(local_payload, dim=0), "swap_collective")
        self._last_swap_collective_ms = (time.perf_counter() - collective_started) * 1000.0
        expert_group_counts: list[torch.Tensor] = []
        sole_expert_counts: list[torch.Tensor] = []
        sole_pair_counts: list[torch.Tensor] = []
        offset = 0
        for level, (groups, affected) in enumerate(level_metadata):
            column_size = num_experts * 2
            columns = reduced[offset : offset + column_size].view(num_experts, 2)
            offset += column_size
            sole = reduced[offset : offset + num_experts]
            offset += num_experts
            pair_size = num_experts * num_experts
            pairs = reduced[offset : offset + pair_size].view(num_experts, num_experts)
            offset += pair_size

            updated_expert_group = stats.expert_group_counts[level].clone()
            updated_expert_group.index_copy_(1, groups, columns)
            expert_group_counts.append(updated_expert_group)
            sole_expert_counts.append(torch.where(affected, sole, stats.sole_expert_counts[level]))
            sole_pair_counts.append(torch.where(affected.view(-1, 1), pairs, stats.sole_pair_counts[level]))

        return _SwapStats(
            owner_ranks=updated_owner_ranks,
            expert_token_counts=stats.expert_token_counts,
            expert_assignment_counts=stats.expert_assignment_counts,
            base_counts=tuple(updated_base_counts),
            expert_group_counts=tuple(expert_group_counts),
            sole_expert_counts=tuple(sole_expert_counts),
            sole_pair_counts=tuple(sole_pair_counts),
            local_token_group_counts=tuple(updated_local_counts),
        )

    def _swap_candidates(
        self,
        stats: _SwapStats,
        current_cost: _TensorCost,
        used: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts = int(stats.owner_ranks.numel())
        cache_key = (str(stats.owner_ranks.device), num_experts)
        pairs = self._swap_pair_cache.get(cache_key)
        if pairs is None:
            pairs = torch.triu_indices(
                num_experts,
                num_experts,
                offset=1,
                device=stats.owner_ranks.device,
            ).transpose(0, 1)
            self._swap_pair_cache[cache_key] = pairs
        if pairs.numel() == 0:
            return pairs, torch.empty((0,), dtype=torch.bool, device=stats.owner_ranks.device)
        bottlenecks = torch.stack(
            (
                current_cost.peak_communication_rank.reshape(-1)[0],
                current_cost.peak_compute_rank.reshape(-1)[0],
            )
        )
        hot_mask = (stats.owner_ranks.unsqueeze(1) == bottlenecks.view(1, -1)).any(dim=1) & ~used
        lhs, rhs = pairs[:, 0], pairs[:, 1]
        lhs_hot = hot_mask.index_select(0, lhs)
        rhs_hot = hot_mask.index_select(0, rhs)
        available = ~used.index_select(0, lhs) & ~used.index_select(0, rhs)
        different_rank = stats.owner_ranks.index_select(0, lhs) != stats.owner_ranks.index_select(0, rhs)
        valid = (lhs_hot ^ rhs_hot) & available & different_rank
        experts_per_rank = max(1, num_experts // self.ep_size)
        possible_hot_counts = (experts_per_rank, min(num_experts, 2 * experts_per_rank))
        max_candidates = max(hot * (num_experts - hot) for hot in possible_hot_counts)
        max_candidates = min(int(pairs.shape[0]), max_candidates)
        if max_candidates == 0:
            return pairs[:0], valid[:0]
        pair_order = torch.arange(pairs.shape[0], dtype=torch.long, device=pairs.device)
        priority = torch.where(valid, pairs.shape[0] - pair_order, -pair_order - 1)
        compact_indices = priority.topk(max_candidates, largest=True, sorted=True).indices
        return pairs.index_select(0, compact_indices), valid.index_select(0, compact_indices)

    def _local_layout_stats(
        self,
        selected: torch.Tensor,
        layouts: torch.Tensor,
        owner_slots: torch.Tensor,
        source_ranks: torch.Tensor,
        *,
        token_ordinals: torch.Tensor | None = None,
        step: int,
        layer_seed: int,
        max_copies: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        physical = assign_tokens_to_copies(
            selected,
            layouts,
            slots_per_rank=self.slots_per_rank,
            source_ranks=source_ranks,
            hierarchy_group_sizes=self.hierarchy.group_sizes,
            owner_slots=owner_slots,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            max_copies=max_copies,
        )
        if physical.ndim == selected.ndim:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
        return self._local_rank_stats(ranks)

    def _local_rank_stats(self, ranks: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        batch, num_tokens = ranks.shape[:2]
        top_k = ranks.shape[2]
        assignment = torch.zeros((batch, self.ep_size), dtype=torch.float32, device=ranks.device)
        assignment.scatter_add_(
            1, ranks.reshape(batch, -1), torch.ones_like(ranks.reshape(batch, -1), dtype=torch.float32)
        )

        counts: list[torch.Tensor] = []
        for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]:
            groups = torch.div(ranks, int(size), rounding_mode="floor")
            num_groups = self.ep_size // int(size)
            hits = torch.zeros((batch * num_tokens, num_groups), dtype=torch.bool, device=ranks.device)
            hits.scatter_(1, groups.reshape(batch * num_tokens, top_k), True)
            counts.append(hits.view(batch, num_tokens, num_groups).sum(dim=1).to(torch.float32))
        rank_hits = torch.zeros((batch * num_tokens, self.ep_size), dtype=torch.bool, device=ranks.device)
        rank_hits.scatter_(1, ranks.reshape(batch * num_tokens, top_k), True)
        counts.append(rank_hits.view(batch, num_tokens, self.ep_size).sum(dim=1).to(torch.float32))
        return tuple(counts), assignment

    def _score_layouts(
        self,
        selected: torch.Tensor,
        layouts: torch.Tensor,
        owner_slots: torch.Tensor,
        source_ranks: torch.Tensor,
        *,
        token_ordinals: torch.Tensor | None = None,
        step: int,
        layer_seed: int,
        max_copies: int,
    ) -> _TensorCost:
        if layouts.ndim == 1:
            layouts = layouts.unsqueeze(0)
        all_counts: list[list[torch.Tensor]] = []
        all_assignments: list[torch.Tensor] = []
        for start in range(0, layouts.shape[0], self.candidate_chunk_size):
            local_counts, local_assignment = self._local_layout_stats(
                selected,
                layouts[start : start + self.candidate_chunk_size],
                owner_slots,
                source_ranks,
                token_ordinals=token_ordinals,
                step=step,
                layer_seed=layer_seed,
                max_copies=max_copies,
            )
            all_counts.append(list(local_counts))
            all_assignments.append(local_assignment)
        level_counts = tuple(
            torch.cat([chunk[level] for chunk in all_counts], dim=0) for level in range(len(all_counts[0]))
        )
        assignments = torch.cat(all_assignments, dim=0)
        return self._cost_from_local_stats(level_counts, assignments)

    def _cost_from_local_stats(
        self,
        level_counts: Sequence[torch.Tensor],
        assignments: torch.Tensor,
    ) -> _TensorCost:
        global_counts, global_assignment = self._globalize_local_stats(level_counts, assignments)
        return self._tensor_cost(global_counts, global_assignment)

    def _globalize_local_stats(
        self,
        level_counts: Sequence[torch.Tensor],
        assignments: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        widths = [row.shape[1] for row in level_counts]
        packed = torch.cat((*level_counts, assignments), dim=1)
        packed = self._reduce_with_timing(packed, "route_collective")
        offset = 0
        global_counts: list[torch.Tensor] = []
        for width in widths:
            global_counts.append(packed[:, offset : offset + width])
            offset += width
        global_assignment = packed[:, offset : offset + self.ep_size]
        return tuple(global_counts), global_assignment

    def _token_group_occupancies(self, route_ranks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        num_tokens = int(route_ranks.shape[0])
        counts: list[torch.Tensor] = []
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        for size in level_sizes:
            groups = torch.div(route_ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            level_counts = torch.zeros((num_tokens, num_groups), dtype=torch.int32, device=route_ranks.device)
            level_counts.scatter_add_(1, groups, torch.ones_like(groups, dtype=torch.int32))
            counts.append(level_counts)
        return tuple(counts)

    @staticmethod
    def _candidate_histogram_delta(
        old_bins: torch.Tensor,
        new_bins: torch.Tensor,
        remove_values: torch.Tensor,
        add_values: torch.Tensor,
        num_bins: int,
    ) -> torch.Tensor:
        old_one_hot = torch.nn.functional.one_hot(old_bins, num_classes=num_bins).to(torch.float32)
        removed = torch.bmm(remove_values.transpose(1, 2).to(torch.float32), old_one_hot)
        added_chunks = []
        for start in range(0, new_bins.shape[2], 16):
            stop = min(start + 16, new_bins.shape[2])
            new_one_hot = torch.nn.functional.one_hot(new_bins[:, :, start:stop], num_classes=num_bins).to(
                torch.float32
            )
            added_chunks.append(
                (add_values[:, :, start:stop].unsqueeze(-1).to(torch.float32) * new_one_hot).sum(dim=1)
            )
        added = torch.cat(added_chunks, dim=1)
        return (added - removed).reshape(-1, num_bins)

    def _fused_replica_level_sizes(self) -> tuple[int, ...] | None:
        if self.ep_size > 64:
            return None
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        return level_sizes if 1 <= len(level_sizes) <= 3 else None

    def _get_fused_replica_ops(self, device: torch.device) -> ModuleType | None:
        distributed = dist.is_available() and dist.is_initialized() and self.reducer is not None
        cache_key = (str(device), self.ep_size, distributed)
        if cache_key in _FUSED_REPLICA_OPS_CACHE:
            return _FUSED_REPLICA_OPS_CACHE[cache_key]
        from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops

        extension = get_hiermoe_planner_npu_ops()
        available = extension is not None
        if distributed:
            local = torch.tensor([int(available)], dtype=torch.int32, device=device)
            available = bool((_reduce_sum(local, self.reducer) == self.ep_size).item())
        _FUSED_REPLICA_OPS_CACHE[cache_key] = extension if available else None
        return _FUSED_REPLICA_OPS_CACHE[cache_key]

    def _fused_replica_route_tables(
        self,
        selected: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            selected.device.type != "npu"
            or selected.shape[0] > _FUSED_REPLICA_MAX_TOKENS
            or self._fused_replica_level_sizes() is None
        ):
            return None
        extension = self._get_fused_replica_ops(selected.device)
        if extension is None:
            return None
        route_indices, multiplicities, token_counts = extension.replica_prepare(selected.contiguous(), num_experts)
        return route_indices, multiplicities, token_counts

    def _initial_replica_stats(
        self,
        selected: torch.Tensor,
        owner_slots: torch.Tensor,
        source_ranks: torch.Tensor,
        token_ordinals: torch.Tensor,
        *,
        step: int,
        layer_seed: int,
        base_counts: Sequence[torch.Tensor],
        assignment_counts: torch.Tensor,
    ) -> _ReplicaStats:
        num_tokens, top_k = selected.shape
        num_experts = int(owner_slots.numel())
        owner_ranks = torch.div(owner_slots, self.slots_per_rank, rounding_mode="floor")
        route_ranks = owner_ranks.index_select(0, selected.reshape(-1))
        flat_logical = selected.reshape(-1)
        fused_route_tables = self._fused_replica_route_tables(selected, num_experts)
        if fused_route_tables is not None:
            route_indices_by_expert_i32, multiplicities_by_expert, tokens_per_expert_i32 = fused_route_tables
            tokens_by_expert = torch.empty((num_experts, 0), dtype=torch.long, device=selected.device)
            tokens_per_expert = tokens_per_expert_i32
            route_indices_by_expert = torch.empty((num_experts, 0), dtype=torch.long, device=selected.device)
        else:
            same_logical = selected.unsqueeze(2) == selected.unsqueeze(1)
            route_multiplicity = same_logical.sum(dim=-1).to(torch.int32)
            first_positions_by_token_expert = torch.full(
                (num_tokens, num_experts), top_k, dtype=torch.int32, device=selected.device
            )
            route_positions = torch.arange(top_k, dtype=torch.int32, device=selected.device).view(1, -1)
            first_positions_by_token_expert.scatter_reduce_(
                1,
                selected,
                route_positions.expand_as(selected),
                reduce="amin",
                include_self=True,
            )
            multiplicities_by_token_expert = torch.zeros(
                (num_tokens, num_experts), dtype=torch.int32, device=selected.device
            )
            multiplicities_by_token_expert.scatter_(1, selected, route_multiplicity)
            token_expert_hits = multiplicities_by_token_expert > 0
            tokens_per_expert = token_expert_hits.sum(dim=0)
            max_tokens_per_expert = int(tokens_per_expert.max().item()) if num_tokens else 0
            token_ids = torch.arange(num_tokens, dtype=torch.long, device=selected.device).view(-1, 1)
            tokens_by_expert = torch.where(
                token_expert_hits,
                token_ids,
                torch.full_like(token_expert_hits, num_tokens, dtype=torch.long),
            ).transpose(0, 1)
            tokens_by_expert = tokens_by_expert.sort(dim=1).values[:, :max_tokens_per_expert]
            if max_tokens_per_expert:
                safe_tokens_by_expert = tokens_by_expert.clamp_max(num_tokens - 1)
                expert_rows = torch.arange(num_experts, dtype=torch.long, device=selected.device).view(-1, 1)
                expert_rows = expert_rows.expand_as(safe_tokens_by_expert)
                multiplicities_by_expert = multiplicities_by_token_expert[safe_tokens_by_expert, expert_rows]
                positions_by_expert = first_positions_by_token_expert[safe_tokens_by_expert, expert_rows]
                positions_by_expert = positions_by_expert.clamp_max(max(0, top_k - 1)).to(torch.long)
                route_indices_by_expert = safe_tokens_by_expert * top_k + positions_by_expert
            else:
                route_indices_by_expert = torch.empty((num_experts, 0), dtype=torch.long, device=selected.device)
                multiplicities_by_expert = torch.empty((num_experts, 0), dtype=torch.int32, device=selected.device)
            route_indices_by_expert_i32 = route_indices_by_expert.to(torch.int32)
            tokens_per_expert_i32 = tokens_per_expert.to(torch.int32)

        rank_candidates = torch.arange(self.ep_size, dtype=torch.long, device=selected.device)
        candidate_copy_ranks = rank_candidates.view(1, 1, 1, -1).expand(1, num_tokens, top_k, -1)
        route_scores = (
            _copy_scores(
                selected,
                candidate_copy_ranks,
                source_ranks,
                owner_slots,
                slots_per_rank=self.slots_per_rank,
                num_ranks=self.ep_size,
                hierarchy_group_sizes=self.hierarchy.group_sizes,
            )
            .reshape(-1, self.ep_size)
            .to(torch.int32)
        )
        route_hashes = _route_hash(
            selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        ).reshape(-1)
        minimum_scores = route_scores.gather(1, route_ranks.view(-1, 1)).squeeze(1)
        tied_rank_order = torch.full(route_scores.shape, self.ep_size, dtype=torch.long, device=selected.device)
        tied_rank_order[:, 0] = route_ranks
        tie_count = torch.ones_like(route_ranks, dtype=torch.int32)

        copy_rank_mask = torch.zeros((num_experts, self.ep_size), dtype=torch.bool, device=selected.device)
        copy_rank_mask.scatter_(1, owner_ranks.view(-1, 1), True)
        copy_slots_by_rank = torch.full((num_experts, self.ep_size), -1, dtype=torch.long, device=selected.device)
        copy_slots_by_rank.scatter_(1, owner_ranks.view(-1, 1), owner_slots.view(-1, 1))
        raw_local_token_group_counts = self._token_group_occupancies(route_ranks.view(num_tokens, top_k))
        local_group_widths = [counts.shape[1] for counts in raw_local_token_group_counts]
        packed_local_token_group_counts = torch.cat(raw_local_token_group_counts, dim=1)
        local_token_group_counts = tuple(packed_local_token_group_counts.split(local_group_widths, dim=1))

        normalized_counts = tuple(count.reshape(-1, count.shape[-1])[0] for count in base_counts)
        normalized_assignment = assignment_counts.reshape(-1, assignment_counts.shape[-1])[0]
        return _ReplicaStats(
            selected=selected,
            flat_logical=flat_logical,
            route_scores=route_scores,
            route_hashes=route_hashes,
            route_ranks=route_ranks,
            minimum_scores=minimum_scores,
            tied_rank_order=tied_rank_order,
            tie_count=tie_count,
            copy_rank_mask=copy_rank_mask,
            copy_slots_by_rank=copy_slots_by_rank,
            tokens_by_expert=tokens_by_expert,
            tokens_per_expert=tokens_per_expert,
            tokens_per_expert_i32=tokens_per_expert_i32,
            route_indices_by_expert=route_indices_by_expert,
            route_indices_by_expert_i32=route_indices_by_expert_i32,
            multiplicities_by_expert=multiplicities_by_expert,
            fused_route_tables=fused_route_tables is not None,
            local_token_group_counts=local_token_group_counts,
            packed_local_token_group_counts=packed_local_token_group_counts,
            base_counts=normalized_counts,
            assignment_counts=normalized_assignment,
        )

    def _fused_replica_candidate_deltas(
        self,
        stats: _ReplicaStats,
        logical_experts: torch.Tensor,
    ) -> torch.Tensor | None:
        if stats.route_ranks.device.type != "npu":
            return None
        level_sizes = self._fused_replica_level_sizes()
        if level_sizes is None:
            return None
        extension = self._get_fused_replica_ops(stats.route_ranks.device)
        if extension is None:
            return None
        padded_sizes = level_sizes + (1,) * (3 - len(level_sizes))
        deltas = extension.replica_score(
            stats.route_indices_by_expert_i32,
            stats.multiplicities_by_expert,
            stats.tokens_per_expert_i32,
            stats.route_ranks,
            stats.route_scores,
            stats.minimum_scores,
            stats.tie_count,
            stats.tied_rank_order,
            stats.route_hashes,
            stats.packed_local_token_group_counts,
            logical_experts,
            len(level_sizes),
            padded_sizes[0],
            padded_sizes[1],
            padded_sizes[2],
            stats.selected.shape[1],
        )
        raw_width = stats.packed_local_token_group_counts.shape[1] + self.ep_size
        return deltas[:, :raw_width]

    def _incremental_replica_candidates(
        self,
        stats: _ReplicaStats,
        logical_experts: torch.Tensor | None = None,
    ) -> _ReplicaCandidateBatch:
        num_experts = int(stats.copy_rank_mask.shape[0])
        if logical_experts is None:
            logical_experts = torch.arange(num_experts, dtype=torch.long, device=stats.route_ranks.device)
        num_candidate_experts = int(logical_experts.numel())
        destination_ranks = torch.arange(self.ep_size, dtype=torch.long, device=stats.route_ranks.device)

        fused_deltas = self._fused_replica_candidate_deltas(stats, logical_experts)
        if fused_deltas is not None:
            collective_started = time.perf_counter()
            reduced = self._reduce_with_timing(fused_deltas, "replica_collective").to(torch.float32)
            self._last_replica_collective_ms = (time.perf_counter() - collective_started) * 1000.0
            offset = 0
            candidate_base_counts: list[torch.Tensor] = []
            for base in stats.base_counts:
                width = int(base.numel())
                candidate_base_counts.append(base.view(1, -1) + reduced[:, offset : offset + width])
                offset += width
            candidate_assignment = stats.assignment_counts.view(1, -1) + reduced[:, offset : offset + self.ep_size]
            return _ReplicaCandidateBatch(
                cost=self._tensor_cost(candidate_base_counts, candidate_assignment),
                base_counts=tuple(candidate_base_counts),
                assignment_counts=candidate_assignment,
                route_ranks_by_destination=stats.route_ranks.new_empty((0,)),
            )

        token_width = int(stats.tokens_by_expert.shape[1])
        token_positions = torch.arange(token_width, dtype=torch.long, device=stats.route_ranks.device)
        group_tokens = stats.tokens_by_expert.index_select(0, logical_experts)
        group_valid = token_positions.view(1, -1) < stats.tokens_per_expert.index_select(0, logical_experts).view(
            -1, 1
        )
        safe_tokens = group_tokens.clamp_max(max(0, stats.selected.shape[0] - 1))
        group_multiplicity = stats.multiplicities_by_expert.index_select(0, logical_experts)
        group_valid &= group_multiplicity > 0
        group_route_indices = stats.route_indices_by_expert.index_select(0, logical_experts)
        flat_route_indices = group_route_indices.reshape(-1)
        flat_group_valid = group_valid.reshape(-1)
        flat_multiplicity = group_multiplicity.reshape(-1)

        current_ranks = stats.route_ranks.index_select(0, flat_route_indices)
        route_scores = stats.route_scores.index_select(0, flat_route_indices)
        minimum_scores = stats.minimum_scores.index_select(0, flat_route_indices)
        tie_count = stats.tie_count.index_select(0, flat_route_indices)
        tied_rank_order = stats.tied_rank_order.index_select(0, flat_route_indices)
        route_hashes = stats.route_hashes.index_select(0, flat_route_indices)
        order_positions = destination_ranks.view(1, -1)
        valid_tied_order = order_positions < tie_count.view(-1, 1)
        tied_rank_mask = torch.zeros_like(tied_rank_order, dtype=torch.bool)
        tied_rank_mask.scatter_(
            1,
            tied_rank_order.clamp(max=max(0, self.ep_size - 1)),
            valid_tied_order,
        )
        insertion_position = tied_rank_mask.to(torch.long).cumsum(dim=1) - tied_rank_mask.to(torch.long)
        new_target = torch.remainder(route_hashes.view(-1, 1), tie_count.view(-1, 1) + 1)
        existing_order = new_target - (new_target > insertion_position).to(torch.long)
        existing_rank = tied_rank_order.gather(1, existing_order.clamp(max=max(0, self.ep_size - 1)))
        equal_rank = torch.where(new_target == insertion_position, destination_ranks.view(1, -1), existing_rank)
        candidate_route_ranks = torch.where(
            route_scores < minimum_scores.view(-1, 1),
            destination_ranks.view(1, -1),
            torch.where(route_scores == minimum_scores.view(-1, 1), equal_rank, current_ranks.view(-1, 1)),
        )

        moved = (candidate_route_ranks != current_ranks.view(-1, 1)) & flat_group_valid.view(-1, 1)
        moved_values = moved.to(torch.float32) * flat_multiplicity.view(-1, 1)
        assignment_delta = self._candidate_histogram_delta(
            current_ranks.view(num_candidate_experts, token_width),
            candidate_route_ranks.view(num_candidate_experts, token_width, self.ep_size),
            moved_values.view(num_candidate_experts, token_width, self.ep_size),
            moved_values.view(num_candidate_experts, token_width, self.ep_size),
            self.ep_size,
        )

        local_level_deltas: list[torch.Tensor] = []
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        for level, size in enumerate(level_sizes):
            token_counts = stats.local_token_group_counts[level]
            num_groups = int(token_counts.shape[1])
            old_groups = torch.div(current_ranks, size, rounding_mode="floor")
            new_groups = torch.div(candidate_route_ranks, size, rounding_mode="floor")
            different_group = old_groups.view(-1, 1) != new_groups
            route_token_counts = token_counts.index_select(0, safe_tokens.reshape(-1))
            old_occupancy = route_token_counts.gather(1, old_groups.view(-1, 1)).squeeze(1)
            new_occupancy = route_token_counts.gather(1, new_groups)
            remove = moved & different_group & (old_occupancy.view(-1, 1) == flat_multiplicity.view(-1, 1))
            add = moved & different_group & (new_occupancy == 0)
            level_delta = self._candidate_histogram_delta(
                old_groups.view(num_candidate_experts, token_width),
                new_groups.view(num_candidate_experts, token_width, self.ep_size),
                remove.view(num_candidate_experts, token_width, self.ep_size),
                add.view(num_candidate_experts, token_width, self.ep_size),
                num_groups,
            )
            local_level_deltas.append(level_delta)

        packed = torch.cat((*local_level_deltas, assignment_delta), dim=1)
        collective_started = time.perf_counter()
        reduced = self._reduce_with_timing(packed, "replica_collective")
        self._last_replica_collective_ms = (time.perf_counter() - collective_started) * 1000.0
        offset = 0
        candidate_base_counts: list[torch.Tensor] = []
        for base in stats.base_counts:
            width = int(base.numel())
            candidate_base_counts.append(base.view(1, -1) + reduced[:, offset : offset + width])
            offset += width
        candidate_assignment = stats.assignment_counts.view(1, -1) + reduced[:, offset : offset + self.ep_size]
        return _ReplicaCandidateBatch(
            cost=self._tensor_cost(candidate_base_counts, candidate_assignment),
            base_counts=tuple(candidate_base_counts),
            assignment_counts=candidate_assignment,
            route_ranks_by_destination=candidate_route_ranks,
        )

    def _replica_route_ranks_for_destination(
        self,
        stats: _ReplicaStats,
        destination_rank: torch.Tensor,
    ) -> torch.Tensor:
        destination_scores = stats.route_scores.index_select(1, destination_rank.view(1)).squeeze(1)
        insertion_position = (stats.tied_rank_order < destination_rank).sum(dim=1)
        new_target = torch.remainder(stats.route_hashes, stats.tie_count + 1)
        existing_order = new_target - (new_target > insertion_position).to(torch.long)
        existing_rank = stats.tied_rank_order.gather(1, existing_order.view(-1, 1)).squeeze(1)
        equal_rank = torch.where(new_target == insertion_position, destination_rank, existing_rank)
        return torch.where(
            destination_scores < stats.minimum_scores,
            destination_rank,
            torch.where(destination_scores == stats.minimum_scores, equal_rank, stats.route_ranks),
        )

    def _fused_apply_replica_candidate(
        self,
        stats: _ReplicaStats,
        logical_expert: torch.Tensor,
        destination_rank: torch.Tensor,
    ) -> bool:
        if not stats.fused_route_tables or stats.route_ranks.device.type != "npu":
            return False
        level_sizes = self._fused_replica_level_sizes()
        if level_sizes is None:
            return False
        extension = self._get_fused_replica_ops(stats.route_ranks.device)
        if extension is None or not hasattr(torch.ops.veomni, "hiermoe_replica_apply"):
            return False
        padded_sizes = level_sizes + (1,) * (3 - len(level_sizes))
        torch.ops.veomni.hiermoe_replica_apply(
            stats.route_indices_by_expert_i32,
            stats.tokens_per_expert_i32,
            stats.flat_logical,
            stats.route_ranks,
            stats.route_scores,
            stats.minimum_scores,
            stats.tie_count,
            stats.tied_rank_order,
            stats.route_hashes,
            stats.packed_local_token_group_counts,
            logical_expert,
            destination_rank,
            len(level_sizes),
            padded_sizes[0],
            padded_sizes[1],
            padded_sizes[2],
            stats.selected.shape[1],
        )
        return True

    def _apply_replica_candidate(
        self,
        stats: _ReplicaStats,
        candidates: _ReplicaCandidateBatch,
        best_index: torch.Tensor,
        logical_expert: torch.Tensor,
        destination_rank: torch.Tensor,
        destination_slot: torch.Tensor,
    ) -> None:
        if self._fused_apply_replica_candidate(stats, logical_expert, destination_rank):
            stats.copy_rank_mask[logical_expert, destination_rank] = True
            stats.copy_slots_by_rank[logical_expert, destination_rank] = destination_slot
            stats.base_counts = tuple(
                counts.index_select(0, best_index.view(1)).squeeze(0) for counts in candidates.base_counts
            )
            stats.assignment_counts = candidates.assignment_counts.index_select(0, best_index.view(1)).squeeze(0)
            return
        proposed_ranks = self._replica_route_ranks_for_destination(stats, destination_rank)
        affected_routes = stats.flat_logical == logical_expert
        old_ranks = stats.route_ranks
        new_ranks = torch.where(affected_routes, proposed_ranks, old_ranks)
        destination_scores = stats.route_scores.index_select(1, destination_rank.view(1)).squeeze(1)
        lower_score = destination_scores < stats.minimum_scores
        equal_score = destination_scores == stats.minimum_scores
        order_positions = torch.arange(self.ep_size, dtype=torch.long, device=stats.route_ranks.device).view(1, -1)
        insertion_position = (stats.tied_rank_order < destination_rank).sum(dim=1)
        previous_position = (
            order_positions - (order_positions > insertion_position.view(-1, 1)).to(torch.long)
        ).clamp_min(0)
        equal_order = stats.tied_rank_order.gather(1, previous_position)
        equal_order = torch.where(order_positions == insertion_position.view(-1, 1), destination_rank, equal_order)
        lower_order = torch.full_like(stats.tied_rank_order, self.ep_size)
        lower_order[:, 0] = destination_rank
        update_lower = affected_routes & lower_score
        update_equal = affected_routes & equal_score
        stats.tied_rank_order = torch.where(
            update_lower.view(-1, 1),
            lower_order,
            torch.where(update_equal.view(-1, 1), equal_order, stats.tied_rank_order),
        )
        stats.tie_count = torch.where(
            update_lower,
            torch.ones_like(stats.tie_count),
            torch.where(update_equal, stats.tie_count + 1, stats.tie_count),
        )
        stats.minimum_scores = torch.where(update_lower, destination_scores, stats.minimum_scores)
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        for level, size in enumerate(level_sizes):
            counts = stats.local_token_group_counts[level]
            num_groups = int(counts.shape[1])
            old_groups = torch.div(old_ranks, size, rounding_mode="floor")
            new_groups = torch.div(new_ranks, size, rounding_mode="floor")
            moved = affected_routes & (old_groups != new_groups)
            old_one_hot = torch.nn.functional.one_hot(old_groups, num_classes=num_groups)
            new_one_hot = torch.nn.functional.one_hot(new_groups, num_classes=num_groups)
            group_delta = ((new_one_hot - old_one_hot) * moved.view(-1, 1)).to(torch.int32)
            counts.add_(group_delta.view(counts.shape[0], -1, num_groups).sum(dim=1, dtype=torch.int32))

        stats.route_ranks = new_ranks
        stats.copy_rank_mask[logical_expert, destination_rank] = True
        stats.copy_slots_by_rank[logical_expert, destination_rank] = destination_slot
        stats.base_counts = tuple(
            counts.index_select(0, best_index.view(1)).squeeze(0) for counts in candidates.base_counts
        )
        stats.assignment_counts = candidates.assignment_counts.index_select(0, best_index.view(1)).squeeze(0)

    def _owner_replica_candidate_costs(
        self,
        selected: torch.Tensor,
        owner_slots: torch.Tensor,
        source_ranks: torch.Tensor,
        logical_experts: torch.Tensor,
        destination_slots: torch.Tensor,
        *,
        token_ordinals: torch.Tensor | None = None,
        step: int,
        layer_seed: int,
    ) -> _TensorCost:
        started = time.perf_counter()
        num_tokens = selected.shape[0]
        owner_ranks = torch.div(owner_slots, self.slots_per_rank, rounding_mode="floor")
        routed_ranks = owner_ranks.index_select(0, selected.reshape(-1)).view_as(selected)
        base_counts, base_assignment = self._local_rank_stats(routed_ranks.unsqueeze(0))
        base_ms = (time.perf_counter() - started) * 1000.0

        present = torch.zeros((owner_slots.numel(),), dtype=torch.bool, device=selected.device)
        present.scatter_(0, logical_experts, True)
        unique_logicals = torch.nonzero(present, as_tuple=False).flatten()
        logical_to_unique = torch.zeros_like(owner_slots)
        logical_to_unique.scatter_(
            0,
            unique_logicals,
            torch.arange(unique_logicals.numel(), dtype=torch.long, device=selected.device),
        )
        candidate_to_unique = logical_to_unique.index_select(0, logical_experts)
        unique_owner_ranks = owner_ranks.index_select(0, unique_logicals)
        destination_ranks = torch.arange(self.ep_size, dtype=torch.long, device=selected.device)
        rank_distances = self._rank_distances(selected.device)

        owner_rank_counts = torch.zeros((num_tokens, self.ep_size), dtype=torch.long, device=selected.device)
        owner_rank_counts.scatter_add_(1, routed_ranks, torch.ones_like(routed_ranks))
        expert_hits = torch.zeros((num_tokens, owner_slots.numel()), dtype=torch.bool, device=selected.device)
        expert_hits.scatter_(1, selected, True)
        unique_expert_hits = expert_hits.index_select(1, unique_logicals)

        owner_preferred = owner_rank_counts.index_select(1, unique_owner_ranks) > 1
        destination_preferred = owner_rank_counts > 0
        same_preference = owner_preferred.unsqueeze(-1) == destination_preferred.unsqueeze(1)
        destination_distance = rank_distances[source_ranks.view(-1, 1), destination_ranks.view(1, -1)].unsqueeze(1)
        owner_distance = rank_distances[source_ranks.view(-1, 1), unique_owner_ranks.view(1, -1)].unsqueeze(-1)
        same_distance = destination_distance == owner_distance
        route_hashes = _route_hash(
            unique_logicals.view(1, -1).expand(num_tokens, -1),
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
        destination_tie_position = (destination_ranks.view(1, 1, -1) > unique_owner_ranks.view(1, -1, 1)).to(
            torch.long
        )
        choose_destination_tie = torch.remainder(route_hashes.unsqueeze(-1), 2) == destination_tie_position
        move = unique_expert_hits.unsqueeze(-1) & (
            (destination_preferred.unsqueeze(1) & ~owner_preferred.unsqueeze(-1))
            | (same_preference & (destination_distance < owner_distance))
            | (same_preference & same_distance & choose_destination_tie)
        )
        moved_by_unique_rank = move.to(torch.float32).sum(dim=0)
        index_ms = (time.perf_counter() - started) * 1000.0 - base_ms

        candidate_ranks = torch.div(destination_slots, self.slots_per_rank, rounding_mode="floor")
        moved = moved_by_unique_rank[candidate_to_unique, candidate_ranks]
        candidate_assignment = base_assignment.expand(logical_experts.shape[0], -1).clone()
        candidate_owner_ranks = owner_ranks.index_select(0, logical_experts)
        candidate_assignment.scatter_add_(1, candidate_owner_ranks.view(-1, 1), -moved.view(-1, 1))
        candidate_assignment.scatter_add_(1, candidate_ranks.view(-1, 1), moved.view(-1, 1))

        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        candidate_level_counts: list[torch.Tensor] = []
        for level, size in enumerate(level_sizes):
            routed_groups = torch.div(routed_ranks, size, rounding_mode="floor")
            destination_groups = torch.div(destination_ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            token_group_counts = torch.zeros((num_tokens, num_groups), dtype=torch.long, device=selected.device)
            token_group_counts.scatter_add_(1, routed_groups, torch.ones_like(routed_groups))
            unique_owner_groups = torch.div(unique_owner_ranks, size, rounding_mode="floor")
            owner_group_counts = token_group_counts.index_select(1, unique_owner_groups)
            destination_group_counts = token_group_counts.index_select(1, destination_groups)
            different_group = unique_owner_groups.view(1, -1, 1) != destination_groups.view(1, 1, -1)
            remove = move & different_group & (owner_group_counts.unsqueeze(-1) == 1)
            add = move & different_group & (destination_group_counts.unsqueeze(1) == 0)
            removed_by_unique_rank = remove.to(torch.float32).sum(dim=0)
            added_by_unique_rank = add.to(torch.float32).sum(dim=0)
            removed = removed_by_unique_rank[candidate_to_unique, candidate_ranks]
            added = added_by_unique_rank[candidate_to_unique, candidate_ranks]
            counts = base_counts[level].expand(logical_experts.shape[0], -1).clone()
            candidate_owner_groups = torch.div(candidate_owner_ranks, size, rounding_mode="floor")
            candidate_destination_groups = torch.div(candidate_ranks, size, rounding_mode="floor")
            counts.scatter_add_(1, candidate_owner_groups.view(-1, 1), -removed.view(-1, 1))
            counts.scatter_add_(1, candidate_destination_groups.view(-1, 1), added.view(-1, 1))
            candidate_level_counts.append(counts)
        candidate_ms = (time.perf_counter() - started) * 1000.0 - base_ms - index_ms
        result = self._cost_from_local_stats(candidate_level_counts, candidate_assignment)
        collective_ms = (time.perf_counter() - started) * 1000.0 - base_ms - index_ms - candidate_ms
        self.last_replica_timing_ms = {
            "base": base_ms,
            "index": index_ms,
            "candidate": candidate_ms,
            "collective": collective_ms,
        }
        return result

    def _replica_candidate_costs(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        owner_slots: torch.Tensor,
        source_ranks: torch.Tensor,
        logical_experts: torch.Tensor,
        destination_slots: torch.Tensor,
        *,
        token_ordinals: torch.Tensor | None = None,
        step: int,
        layer_seed: int,
        max_copies: int,
        owner_only_layout: bool = False,
    ) -> _TensorCost:
        if owner_only_layout:
            return self._owner_replica_candidate_costs(
                selected,
                owner_slots,
                source_ranks,
                logical_experts,
                destination_slots,
                token_ordinals=token_ordinals,
                step=step,
                layer_seed=layer_seed,
            )
        candidate_layouts = layout.unsqueeze(0).expand(logical_experts.shape[0], -1).clone()
        candidate_layouts.scatter_(1, destination_slots.unsqueeze(1), logical_experts.unsqueeze(1))
        return self._score_layouts(
            selected,
            candidate_layouts,
            owner_slots,
            source_ranks,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            max_copies=max_copies,
        )

    @staticmethod
    def _index_cost(cost: _TensorCost, index: torch.Tensor) -> _TensorCost:
        selected = index.reshape(1)
        return _TensorCost(
            communication=cost.communication.index_select(0, selected),
            compute=cost.compute.index_select(0, selected),
            communication_model_units=cost.communication_model_units.index_select(0, selected),
            peak_communication_rank=cost.peak_communication_rank.index_select(0, selected),
            peak_compute_rank=cost.peak_compute_rank.index_select(0, selected),
            selected_dim=cost.selected_dim.index_select(0, selected),
        )

    @staticmethod
    def _where_cost(condition: torch.Tensor, accepted: _TensorCost, rejected: _TensorCost) -> _TensorCost:
        return _TensorCost(
            communication=torch.where(condition, accepted.communication, rejected.communication),
            compute=torch.where(condition, accepted.compute, rejected.compute),
            communication_model_units=torch.where(
                condition, accepted.communication_model_units, rejected.communication_model_units
            ),
            peak_communication_rank=torch.where(
                condition, accepted.peak_communication_rank, rejected.peak_communication_rank
            ),
            peak_compute_rank=torch.where(condition, accepted.peak_compute_rank, rejected.peak_compute_rank),
            selected_dim=torch.where(condition, accepted.selected_dim, rejected.selected_dim),
        )

    @staticmethod
    def _to_cost(cost: _TensorCost, index: int = 0) -> PlacementCost:
        values = (
            torch.stack(
                (
                    cost.communication.reshape(-1)[index],
                    cost.compute.reshape(-1)[index],
                    cost.communication_model_units.reshape(-1)[index],
                    cost.peak_communication_rank.reshape(-1)[index].to(cost.communication.dtype),
                    cost.peak_compute_rank.reshape(-1)[index].to(cost.communication.dtype),
                    cost.selected_dim.reshape(-1)[index].to(cost.communication.dtype),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        return PlacementCost(
            communication=float(values[0]),
            compute=float(values[1]),
            communication_model_units=float(values[2]),
            peak_communication_rank=int(values[3]),
            peak_compute_rank=int(values[4]),
            selected_dim=int(values[5]),
        )

    def score_layout(
        self,
        selected_experts: torch.Tensor,
        slot_to_logical: torch.Tensor,
        *,
        source_ranks: int | torch.Tensor,
        owner_slots: torch.Tensor | None = None,
        token_ordinals: torch.Tensor | None = None,
        step: int = 0,
        layer_seed: int = 0,
        max_copies: int = 2,
    ) -> PlacementCost:
        selected = selected_experts.to(torch.long)
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        if isinstance(source_ranks, int):
            sources = torch.full((selected.shape[0],), int(source_ranks), dtype=torch.long, device=selected.device)
        else:
            sources = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        ordinals = (
            torch.arange(selected.shape[0], dtype=torch.long, device=selected.device)
            if token_ordinals is None
            else token_ordinals.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        )
        owners = (
            owner_slots.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
            if owner_slots is not None
            else self._owners_from_layout(slot_to_logical.to(device=selected.device, dtype=torch.long))
        )
        return self._to_cost(
            self._score_layouts(
                selected,
                slot_to_logical,
                owners,
                sources,
                token_ordinals=ordinals,
                step=step,
                layer_seed=layer_seed,
                max_copies=max_copies,
            )
        )

    @staticmethod
    def _owners_from_layout(layout: torch.Tensor) -> torch.Tensor:
        active = layout[layout >= 0]
        if active.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=layout.device)
        num_experts = int(active.max().item()) + 1
        slots = torch.arange(layout.numel(), dtype=torch.long, device=layout.device)
        owners = torch.full((num_experts,), layout.numel(), dtype=torch.long, device=layout.device)
        owners.scatter_reduce_(0, layout.clamp_min(0), slots, reduce="amin", include_self=True)
        return owners

    def plan(
        self,
        selected_experts: torch.Tensor,
        slot_to_logical: torch.Tensor,
        owner_slots: torch.Tensor,
        *,
        source_ranks: int | torch.Tensor,
        max_swaps: int,
        max_replicas: int,
        token_ordinals: torch.Tensor | None = None,
        step: int = 0,
        layer_seed: int = 0,
    ) -> PlacementPlan:
        started = time.perf_counter()
        self._device_event_pairs = {} if self.record_device_timing else None
        planning_device_started = self._device_event()
        selected = selected_experts.to(torch.long)
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        device = selected.device
        layout = slot_to_logical.to(device=device, dtype=torch.long, non_blocking=True).clone()
        owners = owner_slots.to(device=device, dtype=torch.long, non_blocking=True).clone()
        if isinstance(source_ranks, int):
            sources = torch.full((selected.shape[0],), int(source_ranks), dtype=torch.long, device=device)
        else:
            sources = source_ranks.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        ordinals = (
            torch.arange(selected.shape[0], dtype=torch.long, device=device)
            if token_ordinals is None
            else token_ordinals.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        )
        if ordinals.numel() != selected.shape[0]:
            raise ValueError(f"token_ordinals has {ordinals.numel()} values for {selected.shape[0]} tokens.")

        logical_ids = torch.arange(owners.numel(), device=device, dtype=torch.long)
        preserve_existing_replicas = max(0, int(max_replicas)) == 0
        if preserve_existing_replicas:
            virtual_layout = layout.clone()
        else:
            # Replica placement is recomputed for every observed route. Keep only
            # one owner copy per logical expert as the virtual input.
            virtual_layout = torch.full_like(layout, -1)
            virtual_layout.scatter_(0, owners, logical_ids)
        initial_layout_tensor = virtual_layout.clone()

        stats_started = time.perf_counter()
        stats_device_started = self._device_event()
        swap_budget = max(0, int(max_swaps))
        replica_capacity = int((virtual_layout < 0).sum().item())
        replica_budget = min(max(0, int(max_replicas)), replica_capacity)
        token_hits: torch.Tensor | None = None
        swap_stats: _SwapStats | None = None
        current_base_counts: tuple[torch.Tensor, ...]
        current_assignment_counts: torch.Tensor
        if swap_budget:
            token_hits = torch.zeros((selected.shape[0], owners.numel()), dtype=torch.float32, device=device)
            if selected.numel():
                token_hits.scatter_(1, selected, 1.0)
            swap_stats = self._initial_swap_stats(token_hits, selected, owners)
            baseline_tensor_cost = self._current_swap_cost(swap_stats)
            current_base_counts = swap_stats.base_counts
            current_assignment_counts = torch.zeros((self.ep_size,), dtype=torch.float32, device=device).scatter_add_(
                0, swap_stats.owner_ranks, swap_stats.expert_assignment_counts
            )
        else:
            owner_ranks = torch.div(owners, self.slots_per_rank, rounding_mode="floor")
            routed_ranks = owner_ranks.index_select(0, selected.reshape(-1)).view_as(selected)
            local_counts, local_assignment = self._local_rank_stats(routed_ranks.unsqueeze(0))
            global_counts, global_assignment = self._globalize_local_stats(local_counts, local_assignment)
            baseline_tensor_cost = self._tensor_cost(global_counts, global_assignment)
            current_base_counts = tuple(counts[0] for counts in global_counts)
            current_assignment_counts = global_assignment[0]
        self._record_device_interval("route_stats", stats_device_started, self._device_event())
        route_stats_ms = (time.perf_counter() - stats_started) * 1000.0

        swap_action_rows: list[torch.Tensor] = []
        replica_action_rows: list[torch.Tensor] = []
        swap_score_ms = 0.0
        swap_update_ms = 0.0
        swap_collective_ms = 0.0
        replica_score_ms = 0.0
        replica_update_ms = 0.0
        replica_collective_ms = 0.0
        decision_sync_ms = 0.0
        used = torch.zeros((owners.numel(),), dtype=torch.bool, device=device)
        post_swap_cost = baseline_tensor_cost
        swap_started = time.perf_counter()
        for swap_round in range(swap_budget):
            assert swap_stats is not None
            score_started = time.perf_counter()
            score_device_started = self._device_event()
            current = self._current_swap_cost(swap_stats)
            pairs, valid_pairs = self._swap_candidates(swap_stats, current, used)
            if pairs.numel() == 0:
                swap_score_ms += (time.perf_counter() - score_started) * 1000.0
                break
            assert token_hits is not None
            candidate_cost, candidate_group_counts = self._swap_candidate_costs(swap_stats, pairs)
            candidate_total = torch.where(
                valid_pairs, candidate_cost.total, torch.full_like(candidate_cost.total, math.inf)
            )
            best_index = candidate_total.argmin()
            best_total = candidate_total.index_select(0, best_index.view(1))[0]
            pair = pairs.index_select(0, best_index.view(1))[0]
            accepted = best_total < current.total.reshape(-1)[0]
            self._record_device_interval("swap_score", score_device_started, self._device_event())
            swap_score_ms += (time.perf_counter() - score_started) * 1000.0
            decision_started = time.perf_counter()
            if not bool(accepted.item()):
                decision_sync_ms += (time.perf_counter() - decision_started) * 1000.0
                break
            decision_sync_ms += (time.perf_counter() - decision_started) * 1000.0
            post_swap_cost = self._index_cost(candidate_cost, best_index)
            chosen_slots = owners.index_select(0, pair)
            swap_action_rows.append(torch.cat((torch.ones((1,), dtype=torch.long, device=device), chosen_slots, pair)))

            if preserve_existing_replicas:
                virtual_layout.scatter_(0, chosen_slots, virtual_layout.index_select(0, chosen_slots).flip(0))
            owners.scatter_(0, pair, chosen_slots.flip(0))
            used.scatter_(0, pair, True)
            current_base_counts = tuple(
                counts.index_select(0, best_index.view(1)).squeeze(0) for counts in candidate_group_counts
            )
            current_owner_ranks = torch.div(owners, self.slots_per_rank, rounding_mode="floor")
            current_assignment_counts = torch.zeros((self.ep_size,), dtype=torch.float32, device=device).scatter_add_(
                0, current_owner_ranks, swap_stats.expert_assignment_counts
            )

            if swap_round + 1 < swap_budget:
                update_started = time.perf_counter()
                update_device_started = self._device_event()
                swap_stats = self._update_swap_stats(token_hits, swap_stats, pair, current_base_counts)
                self._record_device_interval("swap_update", update_device_started, self._device_event())
                update_elapsed_ms = (time.perf_counter() - update_started) * 1000.0
                swap_collective_ms += self._last_swap_collective_ms
                swap_update_ms += max(0.0, update_elapsed_ms - self._last_swap_collective_ms)

        if not preserve_existing_replicas:
            virtual_layout.fill_(-1)
            virtual_layout.scatter_(0, owners, logical_ids)
        swap_ms = (time.perf_counter() - swap_started) * 1000.0

        replica_started = time.perf_counter()
        current_layout_cost = post_swap_cost
        replica_stats: _ReplicaStats | None = None
        if replica_budget:
            replica_init_device_started = self._device_event()
            replica_stats = self._initial_replica_stats(
                selected,
                owners,
                sources,
                ordinals,
                step=step,
                layer_seed=layer_seed,
                base_counts=current_base_counts,
                assignment_counts=current_assignment_counts,
            )
            self._record_device_interval("replica_init", replica_init_device_started, self._device_event())
        rank_slot_ids = torch.arange(virtual_layout.numel(), dtype=torch.long, device=device).view(
            self.ep_size, self.slots_per_rank
        )
        empty_slot_mask = virtual_layout.view(self.ep_size, self.slots_per_rank) < 0
        empty_slots_by_rank = (
            torch.where(
                empty_slot_mask,
                rank_slot_ids,
                torch.full_like(rank_slot_ids, virtual_layout.numel()),
            )
            .sort(dim=1)
            .values
        )
        replica_capacity_by_rank = empty_slot_mask.sum(dim=1)
        replica_cursors = torch.zeros((self.ep_size,), dtype=torch.long, device=device)
        for _round in range(replica_budget):
            assert replica_stats is not None
            score_started = time.perf_counter()
            score_device_started = self._device_event()
            bottlenecks = torch.stack(
                (
                    current_layout_cost.peak_communication_rank.reshape(-1)[0],
                    current_layout_cost.peak_compute_rank.reshape(-1)[0],
                )
            )
            source_on_bottleneck = replica_stats.copy_rank_mask.index_select(1, bottlenecks).any(dim=1)
            expert_ids = torch.arange(owners.numel(), dtype=torch.long, device=device)
            max_candidate_experts = min(owners.numel(), 2 * self.slots_per_rank)
            candidate_priority = torch.where(
                source_on_bottleneck,
                owners.numel() - expert_ids,
                -expert_ids - 1,
            )
            candidate_indices = candidate_priority.topk(max_candidate_experts, largest=True, sorted=True).indices
            candidate_expert_valid = source_on_bottleneck.index_select(0, candidate_indices)
            candidate_experts = torch.where(
                candidate_expert_valid,
                candidate_indices,
                torch.full_like(candidate_indices, owners.numel()),
            )
            safe_candidate_experts = candidate_experts.clamp_max(max(0, owners.numel() - 1))
            candidate_logicals = safe_candidate_experts.repeat_interleave(self.ep_size)
            candidate_ranks = torch.arange(self.ep_size, dtype=torch.long, device=device).repeat(max_candidate_experts)
            rank_has_capacity = replica_cursors < replica_capacity_by_rank
            valid_candidates = candidate_expert_valid.repeat_interleave(self.ep_size)
            valid_candidates &= rank_has_capacity.index_select(0, candidate_ranks)
            valid_candidates &= ~replica_stats.copy_rank_mask[candidate_logicals, candidate_ranks]

            candidate_batch = self._incremental_replica_candidates(replica_stats, safe_candidate_experts)
            candidate_total = torch.where(
                valid_candidates,
                candidate_batch.cost.total,
                torch.full_like(candidate_batch.cost.total, math.inf),
            )
            best_index = candidate_total.argmin()
            best_total = candidate_total.index_select(0, best_index.view(1))[0]
            accepted = best_total < current_layout_cost.total.reshape(-1)[0]
            self._record_device_interval("replica_score", score_device_started, self._device_event())
            score_elapsed_ms = (time.perf_counter() - score_started) * 1000.0
            replica_collective_ms += self._last_replica_collective_ms
            replica_score_ms += max(0.0, score_elapsed_ms - self._last_replica_collective_ms)
            decision_started = time.perf_counter()
            if not bool(accepted.item()):
                decision_sync_ms += (time.perf_counter() - decision_started) * 1000.0
                break
            decision_sync_ms += (time.perf_counter() - decision_started) * 1000.0
            chosen_logical = candidate_logicals.index_select(0, best_index.view(1))
            chosen_rank = candidate_ranks.index_select(0, best_index.view(1))
            next_slots = empty_slots_by_rank.gather(1, replica_cursors.clamp_max(self.slots_per_rank - 1).view(-1, 1))
            next_slots = next_slots.squeeze(1)
            chosen_slot = next_slots.index_select(0, chosen_rank)
            source_slot = owners.index_select(0, chosen_logical)
            destination_logical = virtual_layout.index_select(0, chosen_slot)
            replica_action_rows.append(
                torch.cat(
                    (
                        accepted.to(torch.long).view(1),
                        source_slot,
                        chosen_slot,
                        chosen_logical,
                        destination_logical,
                    ),
                    dim=0,
                )
            )

            updated_layout = virtual_layout.clone()
            updated_layout.scatter_(0, chosen_slot, chosen_logical)
            virtual_layout = updated_layout
            update_started = time.perf_counter()
            update_device_started = self._device_event()
            self._apply_replica_candidate(
                replica_stats,
                candidate_batch,
                best_index,
                chosen_logical,
                chosen_rank,
                chosen_slot,
            )
            self._record_device_interval("replica_update", update_device_started, self._device_event())
            replica_cursors.scatter_add_(0, chosen_rank, torch.ones_like(chosen_rank))
            current_layout_cost = self._index_cost(candidate_batch.cost, best_index)
            replica_update_ms += (time.perf_counter() - update_started) * 1000.0
        replica_ms = (time.perf_counter() - replica_started) * 1000.0

        finalization_started = time.perf_counter()
        actions: list[PlacementAction] = []
        if swap_action_rows:
            for accepted, src_slot, dst_slot, src_logical, dst_logical in (
                torch.stack(swap_action_rows).detach().cpu().tolist()
            ):
                if accepted:
                    actions.append(PlacementAction("swap", src_slot, dst_slot, src_logical, dst_logical))
        if replica_action_rows:
            for accepted, src_slot, dst_slot, src_logical, dst_logical in (
                torch.stack(replica_action_rows).detach().cpu().tolist()
            ):
                if accepted:
                    actions.append(PlacementAction("replica", src_slot, dst_slot, src_logical, dst_logical))

        baseline_cost = self._to_cost(baseline_tensor_cost)
        final_cost = self._to_cost(current_layout_cost)
        initial_layout = tuple(int(value) for value in initial_layout_tensor.detach().cpu().tolist())
        final_layout = tuple(int(value) for value in virtual_layout.detach().cpu().tolist())
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        self._record_device_interval("planning", planning_device_started, self._device_event())
        device_timing_ms = self._finalize_device_timing()
        planning_ms = (time.perf_counter() - started) * 1000.0
        return PlacementPlan(
            actions=tuple(actions),
            initial_layout=initial_layout,
            final_layout=final_layout,
            baseline_cost=baseline_cost,
            final_cost=final_cost,
            swap_rounds=sum(action.kind == "swap" for action in actions),
            replica_rounds=sum(action.kind == "replica" for action in actions),
            planning_ms=planning_ms,
            route_stats_ms=route_stats_ms,
            swap_ms=swap_ms,
            replica_ms=replica_ms,
            swap_score_ms=swap_score_ms,
            swap_update_ms=swap_update_ms,
            swap_collective_ms=swap_collective_ms,
            replica_score_ms=replica_score_ms,
            replica_update_ms=replica_update_ms,
            replica_collective_ms=replica_collective_ms,
            decision_sync_ms=decision_sync_ms,
            finalization_ms=finalization_ms,
            device_timing_ms=device_timing_ms,
        )


def plan_routes_by_rank(
    routes_by_rank: Sequence[torch.Tensor],
    slot_to_logical: torch.Tensor,
    owner_slots: torch.Tensor,
    *,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    hidden_size: int,
    bytes_per_element: int,
    slots_per_rank: int,
    max_swaps: int,
    max_replicas: int,
    communication_scale: float = 1.0,
    forward_compute_per_assignment: float = 0.0,
    step: int = 0,
    layer_seed: int = 0,
) -> PlacementPlan:
    """Replay a distributed route snapshot in one process."""

    if not routes_by_rank:
        raise ValueError("routes_by_rank must not be empty.")
    device = routes_by_rank[0].device
    routes = torch.cat([route.to(device=device, dtype=torch.long) for route in routes_by_rank], dim=0)
    sources = torch.cat(
        [
            torch.full((route.shape[0],), rank, dtype=torch.long, device=device)
            for rank, route in enumerate(routes_by_rank)
        ]
    )
    token_ordinals = torch.cat(
        [torch.arange(route.shape[0], dtype=torch.long, device=device) for route in routes_by_rank]
    )
    planner = CurrentRoutePlanner(
        hierarchy=hierarchy,
        perf_model=perf_model,
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        slots_per_rank=slots_per_rank,
        communication_scale=communication_scale,
        forward_compute_per_assignment=forward_compute_per_assignment,
    )
    return planner.plan(
        routes,
        slot_to_logical,
        owner_slots,
        source_ranks=sources,
        max_swaps=max_swaps,
        max_replicas=max_replicas,
        token_ordinals=token_ordinals,
        step=step,
        layer_seed=layer_seed,
    )


def speedup(baseline: float, current: float) -> float:
    if baseline <= 0.0:
        return 1.0 if current <= 0.0 else 0.0
    return math.inf if current <= 0.0 else baseline / current
