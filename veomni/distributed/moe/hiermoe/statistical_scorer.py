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

"""Exact sufficient statistics for greedy redundant-expert planning."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .planner import _route_hash


if TYPE_CHECKING:
    from .greedy_planner import GreedyCommunicationPlanner


_MAX_ROUTE_HASH_STATES = 64
_MAX_DENSE_PAIR_KEYS = 4 * 1024 * 1024
_MAX_UNARY_STAT_ELEMENTS = 16 * 1024 * 1024
_MAX_PAIR_EVENTS = 2 * 1024 * 1024
_MAX_PAIR_OCCUPANCY_ELEMENTS = 64 * 1024 * 1024
_MAX_DENSE_PAIR_STAT_ELEMENTS = 256 * 1024 * 1024
_MAX_DENSE_ACTION_INTERACTION_ELEMENTS = 64 * 1024 * 1024
_MAX_BATCHED_PAIR_LOOKUP_ELEMENTS = 64 * 1024 * 1024
_UNIFORM_DISTANCE_ROWS: dict[tuple[int, int, tuple[int, ...]], torch.Tensor] = {}
_DEVICE_DISTANCE_ROWS: dict[tuple[int, int, tuple[int, ...], str, int | None], torch.Tensor] = {}
_DEVICE_LONG_RANGES: dict[tuple[str, int | None, int], torch.Tensor] = {}
_DEVICE_TRIU_PAIRS: dict[tuple[str, int | None, int], torch.Tensor] = {}


@dataclass(frozen=True)
class _PairStateRecords:
    """Unique co-routed expert-state pairs and their token multiplicities."""

    logical_lo: torch.Tensor
    logical_hi: torch.Tensor
    state_lo: torch.Tensor
    state_hi: torch.Tensor
    event_tokens: torch.Tensor
    event_records: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class _PairActionExpansion:
    """Sparse join between pair-state records and two-expert actions."""

    records: torch.Tensor
    actions: torch.Tensor
    lhs_old_ranks: torch.Tensor
    lhs_new_ranks: torch.Tensor
    rhs_old_ranks: torch.Tensor
    rhs_new_ranks: torch.Tensor


@dataclass(frozen=True)
class _DensePairEvents:
    """Fixed-shape token pair events for the uniform-source fast path."""

    pair_states: torch.Tensor
    pseudo_lo: torch.Tensor
    pseudo_hi: torch.Tensor
    event_tokens: torch.Tensor
    valid: torch.Tensor
    key_count: int


@dataclass(frozen=True)
class StatisticalRouteTables:
    """Candidate route tables retained for winner-only token remapping."""

    state_count: int
    baseline_slots: torch.Tensor
    lhs_slots: torch.Tensor
    rhs_slots: torch.Tensor


@dataclass(frozen=True)
class StatisticalPairContext:
    """Reusable exact statistics for sparse two-expert interaction scoring."""

    selected: torch.Tensor
    route_states: torch.Tensor
    unique_routes: torch.Tensor | None
    routes_are_unique: bool
    occupancy: torch.Tensor
    baseline_groups: torch.Tensor
    lhs_new_groups: torch.Tensor
    rhs_new_groups: torch.Tensor
    rhs_valid: torch.Tensor
    pair_events: _DensePairEvents | None
    pair_absence: torch.Tensor | None
    num_experts: int
    state_count: int


@dataclass(frozen=True)
class UniformStatisticalBaseline:
    """Baseline hash-state routes reused by statistics and winner remapping."""

    state_count: int
    route_states: torch.Tensor
    slots_by_state: torch.Tensor
    physical: torch.Tensor


@dataclass(frozen=True)
class StatisticalProxyResult:
    """Cheap state-collapsed unary statistics used only for early pruning."""

    candidate_delta: torch.Tensor
    route_tables: StatisticalRouteTables
    physical: torch.Tensor
    route_hashes: torch.Tensor


@dataclass(frozen=True)
class StatisticalPrimitiveSpec:
    """Layout-only deduplication of the two expert transitions in every action."""

    experts: torch.Tensor
    options: torch.Tensor
    lhs_ids: torch.Tensor
    rhs_ids: torch.Tensor
    rhs_valid: torch.Tensor


@dataclass(frozen=True)
class StatisticalPrimitiveContext:
    """Exact route-state data shared by unary primitives and Top-K pair reranking."""

    spec: StatisticalPrimitiveSpec
    selected: torch.Tensor
    route_states: torch.Tensor
    unique_routes: torch.Tensor | None
    routes_are_unique: bool
    occupancy: torch.Tensor
    baseline_groups: torch.Tensor
    primitive_groups: torch.Tensor
    pair_events: _DensePairEvents | None
    pair_absence: torch.Tensor | None
    baseline_slots: torch.Tensor
    primitive_slots: torch.Tensor
    num_experts: int
    state_count: int


@dataclass(frozen=True)
class StatisticalPrimitiveResult:
    """Exact unary deltas keyed by unique layout transitions."""

    primitive_delta: torch.Tensor
    context: StatisticalPrimitiveContext
    physical: torch.Tensor
    route_hashes: torch.Tensor


def _canonical_route_mask(selected: torch.Tensor) -> torch.Tensor:
    """Keep the first occurrence of every logical expert in each token."""

    top_k = selected.shape[1]
    slots = torch.arange(top_k, dtype=torch.long, device=selected.device)
    earlier = slots.view(1, 1, -1) < slots.view(1, -1, 1)
    duplicate = (selected.unsqueeze(2) == selected.unsqueeze(1)) & earlier
    return ~duplicate.any(dim=2)


def _paired_gather(table: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """Gather paired row/column indices without NPU advanced indexing."""

    width = table.shape[1]
    return table.reshape(-1).index_select(0, rows * width + columns)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _long_range(size: int, device: torch.device) -> torch.Tensor:
    key = (*_device_key(device), int(size))
    values = _DEVICE_LONG_RANGES.get(key)
    if values is None:
        values = torch.arange(size, dtype=torch.long, device=device)
        _DEVICE_LONG_RANGES[key] = values
    return values


def _triu_pairs(size: int, device: torch.device) -> torch.Tensor:
    key = (*_device_key(device), int(size))
    pairs = _DEVICE_TRIU_PAIRS.get(key)
    if pairs is None:
        pairs = torch.triu_indices(size, size, offset=1, device=device)
        _DEVICE_TRIU_PAIRS[key] = pairs
    return pairs


def _uniform_candidate_route_slots(
    planner: GreedyCommunicationPlanner,
    options: torch.Tensor,
    state_hashes: torch.Tensor,
    source_rank: int,
    num_slots: int,
) -> torch.Tensor:
    """Route fixed hash states without recomputing uniform-source distances."""

    cache_key = (planner.ep_size, int(source_rank), tuple(int(size) for size in planner.hierarchy.group_sizes))
    host_distances = _UNIFORM_DISTANCE_ROWS.get(cache_key)
    if host_distances is None:
        values = []
        for destination in range(planner.ep_size):
            distance = len(cache_key[2]) + 1
            if destination == source_rank:
                distance = 0
            else:
                for level, size in reversed(tuple(enumerate(cache_key[2], start=1))):
                    if destination // max(1, size) == source_rank // max(1, size):
                        distance = level

            values.append(distance)
        host_distances = torch.tensor(values, dtype=torch.long)
        _UNIFORM_DISTANCE_ROWS[cache_key] = host_distances

    valid = options < num_slots
    safe_slots = options.clamp(max=max(0, num_slots - 1))
    copy_ranks = torch.div(safe_slots, planner.slots_per_rank, rounding_mode="floor")
    device_cache_key = (*cache_key, *_device_key(options.device))
    distance_row = _DEVICE_DISTANCE_ROWS.get(device_cache_key)
    if distance_row is None:
        distance_row = host_distances.to(device=options.device, non_blocking=True)
        _DEVICE_DISTANCE_ROWS[device_cache_key] = distance_row
    distances = distance_row.index_select(0, copy_ranks.reshape(-1)).view_as(copy_ranks)
    distances = torch.where(valid, distances, torch.full_like(distances, torch.iinfo(torch.long).max))
    minimum = distances.min(dim=1, keepdim=True).values
    tied = valid & distances.eq(minimum)
    tie_order = tied.to(torch.long).cumsum(dim=1) - 1
    tie_count = tied.sum(dim=1, keepdim=True).clamp_min(1)
    targets = torch.remainder(state_hashes.view(1, -1), tie_count)
    chosen = (tied.unsqueeze(1) & tie_order.unsqueeze(1).eq(targets.unsqueeze(2))).to(torch.long).argmax(dim=2)
    return safe_slots.gather(1, chosen)


def _fixed_hash_modulus(max_copies: int) -> int | None:
    """Return one route-state modulus that is exact for every allowed copy count."""

    modulus = 1
    for divisor in range(2, max(1, int(max_copies)) + 1):
        modulus = math.lcm(modulus, divisor)
    return modulus if modulus <= _MAX_ROUTE_HASH_STATES else None


def uniform_statistical_baseline_routes(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    copy_slots: torch.Tensor,
    route_hashes: torch.Tensor,
    *,
    source_rank: int,
) -> UniformStatisticalBaseline | None:
    """Map baseline routes through the same compact hash-state table used for scoring."""

    reachable_copy_count = min(planner.max_copies, int(copy_slots.shape[1]) + 1)
    state_count = _fixed_hash_modulus(reachable_copy_count)
    if state_count is None:
        return None
    route_states = torch.remainder(route_hashes, state_count)
    state_hashes = _long_range(state_count, selected.device)
    slots_by_state = _uniform_candidate_route_slots(
        planner,
        copy_slots,
        state_hashes,
        source_rank,
        planner.ep_size * planner.slots_per_rank,
    )
    flat_table = slots_by_state.reshape(-1)
    physical = flat_table.index_select(
        0,
        (selected * state_count + route_states).reshape(-1),
    ).view_as(selected)
    return UniformStatisticalBaseline(
        state_count=state_count,
        route_states=route_states,
        slots_by_state=slots_by_state,
        physical=physical,
    )


def build_statistical_primitive_spec(
    planner: GreedyCommunicationPlanner,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    rows: torch.Tensor,
) -> StatisticalPrimitiveSpec:
    """Deduplicate exact per-expert copy-set transitions on layout metadata."""

    if rows.numel() == 0:
        empty = rows.new_empty((0,), dtype=torch.long)
        return StatisticalPrimitiveSpec(
            experts=empty,
            options=rows.new_empty((0, int(copy_slots.shape[1]) + 1)),
            lhs_ids=empty,
            rhs_ids=empty,
            rhs_valid=rows.new_empty((0,), dtype=torch.bool),
        )
    lhs_options, rhs_options, rhs_valid = planner._candidate_copy_options(layout, copy_slots, rows)
    action_count = int(rows.shape[0])
    lhs_experts = rows[:, 3]
    rhs_experts = rows[:, 4].clamp_min(0)
    side_experts = torch.cat((lhs_experts, rhs_experts[rhs_valid]))
    side_options = torch.cat((lhs_options, rhs_options[rhs_valid]), dim=0)
    num_slots = int(layout.numel())
    valid_options = side_options < num_slots
    option_ranks = torch.div(
        side_options.clamp(max=max(0, num_slots - 1)),
        planner.slots_per_rank,
        rounding_mode="floor",
    )
    option_ranks = torch.where(
        valid_options,
        option_ranks,
        torch.full_like(option_ranks, planner.ep_size),
    )
    # torch.unique(..., dim=0) is disproportionately expensive on the CPU
    # metadata path (about 100 ms for this EP32 layout).  Encode the exact
    # expert + sorted-rank tuple as one mixed-radix int64 key whenever it fits;
    # 1-D unique is an order of magnitude cheaper and remains collision-free.
    radix = planner.ep_size + 1
    option_width = int(option_ranks.shape[1])
    rank_space = radix**option_width
    maximum_key = (int(side_experts.max().item()) + 1) * rank_space - 1
    if maximum_key <= torch.iinfo(torch.int64).max:
        encoded = side_experts.clone()
        for column in range(option_width):
            encoded = encoded * radix + option_ranks[:, column]
        unique_keys, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
        primitive_experts = torch.div(unique_keys, rank_space, rounding_mode="floor")
        rank_payload = torch.remainder(unique_keys, rank_space)
        powers = torch.tensor(
            [radix ** (option_width - column - 1) for column in range(option_width)],
            dtype=torch.long,
            device=rows.device,
        )
        primitive_ranks = torch.remainder(
            torch.div(rank_payload.view(-1, 1), powers.view(1, -1), rounding_mode="floor"),
            radix,
        )
    else:
        keys = torch.cat((side_experts.view(-1, 1), option_ranks), dim=1)
        primitive_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        primitive_experts = primitive_keys[:, 0]
        primitive_ranks = primitive_keys[:, 1:]
    lhs_ids = inverse[:action_count]
    rhs_ids = torch.full(
        (action_count,),
        -1,
        dtype=torch.long,
        device=rows.device,
    )
    rhs_ids[rhs_valid] = inverse[action_count:]
    return StatisticalPrimitiveSpec(
        experts=primitive_experts,
        options=torch.where(
            primitive_ranks < planner.ep_size,
            primitive_ranks * planner.slots_per_rank,
            torch.full_like(primitive_ranks, num_slots),
        ),
        lhs_ids=lhs_ids,
        rhs_ids=rhs_ids,
        rhs_valid=rhs_valid,
    )


def _pair_state_key_count(num_experts: int, state_count: int) -> int:
    """Return the compact unordered expert-pair and ordered-state key count."""

    return num_experts * max(0, num_experts - 1) // 2 * state_count * state_count


def _pair_state_indices(
    logical_lo: torch.Tensor,
    logical_hi: torch.Tensor,
    state_lo: torch.Tensor,
    state_hi: torch.Tensor,
    *,
    num_experts: int,
    state_count: int,
) -> torch.Tensor:
    """Encode lo < hi expert pairs without reserving diagonal or reverse keys."""

    pair_offsets = logical_lo * (2 * num_experts - logical_lo - 1) // 2
    pair_indices = pair_offsets + logical_hi - logical_lo - 1
    pair_indices = pair_indices.clamp(min=0, max=max(0, num_experts * (num_experts - 1) // 2 - 1))
    return (pair_indices * state_count + state_lo) * state_count + state_hi


def _build_dense_pair_events(
    selected: torch.Tensor,
    route_states: torch.Tensor,
    unique_routes: torch.Tensor | None,
    *,
    routes_are_unique: bool,
    num_experts: int,
    state_count: int,
) -> _DensePairEvents | None:
    """Build all token pair events without nonzero, sorting, or sparse joins."""

    num_tokens, top_k = selected.shape
    if top_k <= 1:
        return None
    route_pairs = _triu_pairs(top_k, selected.device)
    first_slot, second_slot = route_pairs[0], route_pairs[1]
    first_expert = selected.index_select(1, first_slot)
    second_expert = selected.index_select(1, second_slot)
    first_state = route_states.index_select(1, first_slot)
    second_state = route_states.index_select(1, second_slot)
    first_is_lo = first_expert < second_expert
    logical_lo = torch.where(first_is_lo, first_expert, second_expert)
    logical_hi = torch.where(first_is_lo, second_expert, first_expert)
    state_lo = torch.where(first_is_lo, first_state, second_state)
    state_hi = torch.where(first_is_lo, second_state, first_state)
    if routes_are_unique:
        valid = torch.ones_like(first_expert, dtype=torch.bool)
    else:
        assert unique_routes is not None
        valid = unique_routes.index_select(1, first_slot)
        valid &= unique_routes.index_select(1, second_slot)
        valid &= first_expert != second_expert
    token_grid = _long_range(num_tokens, selected.device).view(-1, 1)
    token_grid = token_grid.expand(-1, route_pairs.shape[1])
    pseudo_width = num_experts * state_count
    pair_states = _pair_state_indices(
        logical_lo,
        logical_hi,
        state_lo,
        state_hi,
        num_experts=num_experts,
        state_count=state_count,
    )
    key_count = _pair_state_key_count(num_experts, state_count)
    return _DensePairEvents(
        pair_states=pair_states.reshape(-1),
        pseudo_lo=(logical_lo * state_count + state_lo).reshape(-1).clamp(max=pseudo_width - 1),
        pseudo_hi=(logical_hi * state_count + state_hi).reshape(-1).clamp(max=pseudo_width - 1),
        event_tokens=token_grid.reshape(-1),
        valid=valid.reshape(-1),
        key_count=key_count,
    )


def _dense_unary_absence_statistics(
    selected: torch.Tensor,
    route_states: torch.Tensor,
    unique_routes: torch.Tensor | None,
    groups: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    routes_are_unique: bool,
    num_experts: int,
    state_count: int,
) -> torch.Tensor:
    """Build fixed-shape exact unary sufficient statistics."""

    num_tokens, top_k = selected.shape
    num_groups = occupancy.shape[1]
    token_rows = _long_range(num_tokens, selected.device)
    token_rows = token_rows.view(-1, 1).expand(-1, top_k).reshape(-1)
    level_count = groups.shape[2]
    pseudo = (selected * state_count + route_states).reshape(-1)
    globally_absent = occupancy.eq(0).to(torch.float32)
    absence = torch.zeros(
        (num_experts * state_count, num_groups),
        dtype=torch.float32,
        device=selected.device,
    )
    if routes_are_unique:
        absence.index_add_(0, pseudo, globally_absent.index_select(0, token_rows))
    else:
        assert unique_routes is not None
        weights = unique_routes.reshape(-1).to(torch.float32)
        absence.index_add_(0, pseudo, globally_absent.index_select(0, token_rows) * weights.view(-1, 1))
    own_groups = groups.reshape(-1)
    own_tokens = token_rows.view(-1, 1).expand(-1, level_count).reshape(-1)
    own_pseudo = pseudo.view(-1, 1).expand(-1, level_count).reshape(-1)
    own_occupancy = _paired_gather(occupancy, own_tokens, own_groups)
    sole = own_occupancy.eq(1).to(torch.float32)
    if not routes_are_unique:
        own_weights = weights.view(-1, 1).expand(-1, level_count).reshape(-1)
        sole *= own_weights
    absence.reshape(-1).index_add_(
        0,
        own_pseudo * num_groups + own_groups,
        sole,
    )
    return absence


def _dense_pair_absence_statistics(
    pair_events: _DensePairEvents,
    occupancy: torch.Tensor,
    baseline_group_by_state: torch.Tensor,
) -> torch.Tensor:
    """Build exact pair statistics in a compact unordered-pair state table."""

    num_groups = occupancy.shape[1]
    num_experts, state_count = baseline_group_by_state.shape[:2]
    pseudo_width = num_experts * state_count
    baseline_flat = baseline_group_by_state.reshape(pseudo_width, -1)
    group_lo = baseline_flat.index_select(0, pair_events.pseudo_lo)
    group_hi = baseline_flat.index_select(0, pair_events.pseudo_hi)
    event_occupancy = occupancy.index_select(0, pair_events.event_tokens).clone()
    changed_groups = torch.cat((group_lo, group_hi), dim=1)
    adjustments = -pair_events.valid.to(event_occupancy.dtype).view(-1, 1).expand_as(changed_groups)
    event_occupancy.scatter_add_(1, changed_groups, adjustments)
    absence_values = event_occupancy.eq(0).to(torch.float32)
    absence_values *= pair_events.valid.to(torch.float32).view(-1, 1)
    pair_absence = torch.zeros(
        (pair_events.key_count, num_groups),
        dtype=torch.float32,
        device=occupancy.device,
    )
    pair_absence.index_add_(0, pair_events.pair_states, absence_values)
    return pair_absence


def _route_state_space(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    rows: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    source_ranks: torch.Tensor,
    token_ordinals: torch.Tensor,
    route_hashes: torch.Tensor | None,
    step: int,
    layer_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Return token route states and the canonical source/hash state space."""

    num_slots = int(layout.numel())
    lhs_options, rhs_options, _ = planner._candidate_copy_options(layout, copy_slots, rows)
    valid_option_counts = torch.cat(
        (
            (copy_slots < num_slots).sum(dim=1),
            (lhs_options < num_slots).sum(dim=1),
            (rhs_options < num_slots).sum(dim=1),
        )
    )
    max_ties = max(1, int(valid_option_counts.max().item()))
    hash_modulus = 1
    for divisor in range(2, max_ties + 1):
        hash_modulus = math.lcm(hash_modulus, divisor)
    if hash_modulus > _MAX_ROUTE_HASH_STATES or source_ranks.numel() == 0:
        return None

    if bool((source_ranks == source_ranks[0]).all().item()):
        unique_sources = source_ranks[:1]
        source_inverse = torch.zeros_like(source_ranks)
    else:
        unique_sources, source_inverse = torch.unique(source_ranks, sorted=True, return_inverse=True)
    source_count = int(unique_sources.numel())
    state_count = source_count * hash_modulus
    if state_count > planner.ep_size * _MAX_ROUTE_HASH_STATES:
        return None
    if int(copy_slots.shape[0]) * state_count * sum(planner._count_widths()) > _MAX_UNARY_STAT_ELEMENTS:
        return None
    pair_events = selected.shape[0] * selected.shape[1] * max(0, selected.shape[1] - 1) // 2
    if pair_events > _MAX_PAIR_EVENTS or pair_events * sum(planner._count_widths()) > _MAX_PAIR_OCCUPANCY_ELEMENTS:
        return None

    if route_hashes is None:
        route_hashes = _route_hash(
            selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
    route_states = source_inverse.view(-1, 1) * hash_modulus + torch.remainder(route_hashes, hash_modulus)
    state_sources = unique_sources.repeat_interleave(hash_modulus)
    state_hashes = torch.arange(hash_modulus, dtype=torch.long, device=selected.device).repeat(source_count)
    return route_states, state_sources, state_hashes, state_count


def _build_pair_state_records(
    selected: torch.Tensor,
    route_states: torch.Tensor,
    unique_routes: torch.Tensor,
    *,
    routes_are_unique: bool,
    num_experts: int,
    state_count: int,
) -> _PairStateRecords | None:
    """Compress token pair events into unique expert-state records."""

    num_tokens, top_k = selected.shape
    if top_k <= 1:
        return None
    route_pairs = torch.triu_indices(top_k, top_k, offset=1, device=selected.device)
    first_slot, second_slot = route_pairs[0], route_pairs[1]
    first_expert = selected.index_select(1, first_slot)
    second_expert = selected.index_select(1, second_slot)
    token_grid = torch.arange(num_tokens, dtype=torch.long, device=selected.device).view(-1, 1)
    token_grid = token_grid.expand(-1, route_pairs.shape[1])
    first_state = route_states.index_select(1, first_slot)
    second_state = route_states.index_select(1, second_slot)
    first_is_lo = first_expert < second_expert
    if routes_are_unique:
        logical_lo = torch.where(first_is_lo, first_expert, second_expert).reshape(-1)
        logical_hi = torch.where(first_is_lo, second_expert, first_expert).reshape(-1)
        state_lo = torch.where(first_is_lo, first_state, second_state).reshape(-1)
        state_hi = torch.where(first_is_lo, second_state, first_state).reshape(-1)
        event_tokens = token_grid.reshape(-1)
    else:
        valid = unique_routes.index_select(1, first_slot)
        valid &= unique_routes.index_select(1, second_slot)
        valid &= first_expert != second_expert
        if not bool(valid.any().item()):
            return None
        logical_lo = torch.where(first_is_lo, first_expert, second_expert)[valid]
        logical_hi = torch.where(first_is_lo, second_expert, first_expert)[valid]
        state_lo = torch.where(first_is_lo, first_state, second_state)[valid]
        state_hi = torch.where(first_is_lo, second_state, first_state)[valid]
        event_tokens = token_grid[valid]

    pseudo_width = num_experts * state_count
    pseudo_lo = logical_lo * state_count + state_lo
    pseudo_hi = logical_hi * state_count + state_hi
    encoded = pseudo_lo * pseudo_width + pseudo_hi
    key_space = pseudo_width * pseudo_width
    if key_space <= _MAX_DENSE_PAIR_KEYS:
        dense_counts = torch.zeros(key_space, dtype=torch.float32, device=selected.device)
        dense_counts.index_add_(0, encoded, torch.ones_like(encoded, dtype=torch.float32))
        unique_keys = torch.nonzero(dense_counts, as_tuple=False).flatten()
        counts = dense_counts.index_select(0, unique_keys)
        record_by_key = torch.full((key_space,), -1, dtype=torch.long, device=selected.device)
        record_by_key[unique_keys] = torch.arange(unique_keys.numel(), dtype=torch.long, device=selected.device)
        event_records = record_by_key.index_select(0, encoded)
    else:
        unique_keys, event_records, counts = torch.unique(
            encoded,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        counts = counts.to(torch.float32)
    pseudo_lo = torch.div(unique_keys, pseudo_width, rounding_mode="floor")
    pseudo_hi = torch.remainder(unique_keys, pseudo_width)
    return _PairStateRecords(
        logical_lo=torch.div(pseudo_lo, state_count, rounding_mode="floor"),
        logical_hi=torch.div(pseudo_hi, state_count, rounding_mode="floor"),
        state_lo=torch.remainder(pseudo_lo, state_count),
        state_hi=torch.remainder(pseudo_hi, state_count),
        event_tokens=event_tokens,
        event_records=event_records,
        counts=counts,
    )


def _build_pair_action_expansion(
    rows: torch.Tensor,
    pair_records: _PairStateRecords | None,
    *,
    baseline_rank_by_state: torch.Tensor,
    lhs_new_rank_by_state: torch.Tensor,
    rhs_new_rank_by_state: torch.Tensor,
    rhs_valid: torch.Tensor,
    num_experts: int,
    state_count: int,
) -> _PairActionExpansion | None:
    """Join unique pair-state records with all actions on the same expert pair."""

    if pair_records is None:
        return None
    pair_rows = torch.nonzero(rhs_valid, as_tuple=False).flatten()
    if pair_rows.numel() == 0:
        return None

    action_rows = rows.index_select(0, pair_rows)
    action_lhs = action_rows[:, 3]
    action_rhs = action_rows[:, 4]
    action_pair_ids = torch.minimum(action_lhs, action_rhs) * num_experts + torch.maximum(action_lhs, action_rhs)
    pair_counts = torch.bincount(action_pair_ids, minlength=num_experts * num_experts)
    max_pair_actions = int(pair_counts.max().item())
    if max_pair_actions <= 0:
        return None
    # Keep the join key integral: float32 cannot distinguish every pair id
    # once the logical expert space grows beyond 2**24.
    action_order = torch.argsort(action_pair_ids, stable=True)
    sorted_pair_ids = action_pair_ids.index_select(0, action_order)
    pair_starts = torch.cumsum(pair_counts, dim=0) - pair_counts
    pair_ordinals = torch.arange(pair_rows.numel(), dtype=torch.long, device=rows.device)
    pair_ordinals -= pair_starts.index_select(0, sorted_pair_ids)
    action_map = torch.full(
        (num_experts * num_experts, max_pair_actions),
        -1,
        dtype=torch.long,
        device=rows.device,
    )
    action_map[sorted_pair_ids, pair_ordinals] = pair_rows.index_select(0, action_order)

    record_pair_ids = pair_records.logical_lo * num_experts + pair_records.logical_hi
    mapped_actions = action_map.index_select(0, record_pair_ids).reshape(-1)
    valid_positions = torch.nonzero(mapped_actions >= 0, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return None
    records = torch.div(valid_positions, max_pair_actions, rounding_mode="floor")
    actions = mapped_actions.index_select(0, valid_positions)
    expanded_lo = pair_records.logical_lo.index_select(0, records)
    lhs_is_lo = rows.index_select(0, actions)[:, 3] == expanded_lo
    lo_states = pair_records.state_lo.index_select(0, records)
    hi_states = pair_records.state_hi.index_select(0, records)
    lhs_states = torch.where(lhs_is_lo, lo_states, hi_states)
    rhs_states = torch.where(lhs_is_lo, hi_states, lo_states)
    action_lhs = rows.index_select(0, actions)[:, 3]
    action_rhs = rows.index_select(0, actions)[:, 4]
    baseline_flat = baseline_rank_by_state.reshape(-1)
    lhs_old_ranks = baseline_flat.index_select(0, action_lhs * state_count + lhs_states)
    rhs_old_ranks = baseline_flat.index_select(0, action_rhs * state_count + rhs_states)
    lhs_new_ranks = _paired_gather(lhs_new_rank_by_state, actions, lhs_states)
    rhs_new_ranks = _paired_gather(rhs_new_rank_by_state, actions, rhs_states)
    return _PairActionExpansion(
        records=records,
        actions=actions,
        lhs_old_ranks=lhs_old_ranks,
        lhs_new_ranks=lhs_new_ranks,
        rhs_old_ranks=rhs_old_ranks,
        rhs_new_ranks=rhs_new_ranks,
    )


def _unary_absence_statistics(
    selected: torch.Tensor,
    route_states: torch.Tensor,
    unique_routes: torch.Tensor,
    groups: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    routes_are_unique: bool,
    num_experts: int,
    state_count: int,
) -> torch.Tensor:
    """Count tokens for which one expert-state controls group presence."""

    num_tokens, top_k = selected.shape
    num_groups = occupancy.shape[1]
    token_rows = torch.arange(num_tokens, dtype=torch.long, device=selected.device)
    token_rows = token_rows.view(-1, 1).expand(-1, top_k).reshape(-1)
    if routes_are_unique:
        pseudo = (selected * state_count + route_states).reshape(-1)
        own_groups = groups.reshape(-1)
    else:
        valid = unique_routes.reshape(-1)
        token_rows = token_rows[valid]
        pseudo = (selected * state_count + route_states).reshape(-1)[valid]
        own_groups = groups.reshape(-1)[valid]

    globally_absent = occupancy.eq(0).to(torch.float32)
    absence = torch.zeros(
        (num_experts * state_count, num_groups),
        dtype=torch.float32,
        device=selected.device,
    )
    absence.index_add_(0, pseudo, globally_absent.index_select(0, token_rows))
    own_occupancy = _paired_gather(occupancy, token_rows, own_groups)
    sole = own_occupancy.eq(1).to(torch.float32)
    absence.reshape(-1).index_add_(0, pseudo * num_groups + own_groups, sole)
    return absence


def _pair_absence_statistics(
    pair_records: _PairStateRecords,
    occupancy: torch.Tensor,
    baseline_group_by_state: torch.Tensor,
) -> torch.Tensor:
    """Count pair-state tokens with no third expert present in each group."""

    num_groups = occupancy.shape[1]
    event_occupancy = occupancy.index_select(0, pair_records.event_tokens).clone()
    baseline_flat = baseline_group_by_state.reshape(-1)
    state_count = baseline_group_by_state.shape[1]
    event_lo = pair_records.logical_lo.index_select(0, pair_records.event_records)
    event_hi = pair_records.logical_hi.index_select(0, pair_records.event_records)
    state_lo = pair_records.state_lo.index_select(0, pair_records.event_records)
    state_hi = pair_records.state_hi.index_select(0, pair_records.event_records)
    group_lo = baseline_flat.index_select(0, event_lo * state_count + state_lo)
    group_hi = baseline_flat.index_select(0, event_hi * state_count + state_hi)
    event_occupancy.scatter_add_(
        1,
        torch.stack((group_lo, group_hi), dim=1),
        torch.full(
            (event_occupancy.shape[0], 2),
            -1,
            dtype=event_occupancy.dtype,
            device=event_occupancy.device,
        ),
    )
    blocked = torch.zeros(
        (pair_records.counts.numel(), num_groups),
        dtype=torch.float32,
        device=occupancy.device,
    )
    blocked.index_add_(0, pair_records.event_records, event_occupancy.gt(0).to(torch.float32))
    return pair_records.counts.view(-1, 1) - blocked


def _add_unary_deltas(
    delta: torch.Tensor,
    absence: torch.Tensor,
    experts: torch.Tensor,
    old_groups: torch.Tensor,
    new_groups: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> None:
    """Add exact one-expert deltas to candidate group counts."""

    num_groups = absence.shape[1]
    state_count = old_groups.shape[1]
    tail_dims = old_groups.ndim - 2
    states = torch.arange(state_count, dtype=torch.long, device=delta.device).view(1, state_count, *([1] * tail_dims))
    expert_shape = (-1,) + (1,) * (old_groups.ndim - 1)
    pseudo = experts.view(expert_shape) * state_count + states
    pseudo = pseudo.expand_as(old_groups)
    flat_absence = absence.reshape(-1)
    losses = flat_absence.index_select(0, (pseudo * num_groups + old_groups).reshape(-1)).view_as(old_groups)
    gains = flat_absence.index_select(0, (pseudo * num_groups + new_groups).reshape(-1)).view_as(new_groups)
    if valid is not None:
        mask = valid.view((-1,) + (1,) * (losses.ndim - 1)).to(losses.dtype)
        losses *= mask
        gains *= mask
    delta.scatter_add_(1, old_groups.reshape(old_groups.shape[0], -1), -losses.reshape(losses.shape[0], -1))
    delta.scatter_add_(1, new_groups.reshape(new_groups.shape[0], -1), gains.reshape(gains.shape[0], -1))


def _add_dense_unary_deltas(
    delta: torch.Tensor,
    absence: torch.Tensor,
    rows: torch.Tensor,
    *,
    lhs_old_groups: torch.Tensor,
    lhs_new_groups: torch.Tensor,
    rhs_old_groups: torch.Tensor,
    rhs_new_groups: torch.Tensor,
    rhs_valid: torch.Tensor,
) -> None:
    """Add lhs/rhs unary terms with one fixed-shape indexed accumulation."""

    num_actions = rows.shape[0]
    num_groups = absence.shape[1]
    state_count = lhs_old_groups.shape[1]
    level_count = lhs_old_groups.shape[2]
    experts = torch.stack((rows[:, 3], rows[:, 4].clamp_min(0)), dim=1)
    old_groups = torch.stack((lhs_old_groups, rhs_old_groups), dim=1)
    new_groups = torch.stack((lhs_new_groups, rhs_new_groups), dim=1)
    states = _long_range(state_count, delta.device).view(1, 1, state_count, 1)
    pseudo = experts.view(num_actions, 2, 1, 1) * state_count + states
    pseudo = pseudo.expand(num_actions, 2, state_count, level_count)
    flat_absence = absence.reshape(-1)
    losses = flat_absence.index_select(0, (pseudo * num_groups + old_groups).reshape(-1)).view_as(old_groups)
    gains = flat_absence.index_select(0, (pseudo * num_groups + new_groups).reshape(-1)).view_as(new_groups)
    valid = torch.stack((torch.ones_like(rhs_valid), rhs_valid), dim=1)
    valid = valid.view(num_actions, 2, 1, 1).to(losses.dtype)
    groups = torch.cat((old_groups, new_groups), dim=2)
    values = torch.cat((-losses * valid, gains * valid), dim=2)
    actions = _long_range(num_actions, delta.device).view(-1, 1, 1, 1)
    actions = actions.expand(num_actions, 2, 2 * state_count, level_count)
    delta.reshape(-1).index_add_(
        0,
        (actions * num_groups + groups).reshape(-1),
        values.reshape(-1),
    )


def _add_pair_deltas(
    delta: torch.Tensor,
    pair_absence: torch.Tensor,
    expansion: _PairActionExpansion | None,
    *,
    group_size: int,
) -> None:
    """Add exact two-expert interaction deltas from pair-state statistics."""

    if expansion is None:
        return
    num_groups = pair_absence.shape[1]
    lhs_old = torch.div(expansion.lhs_old_ranks, group_size, rounding_mode="floor")
    lhs_new = torch.div(expansion.lhs_new_ranks, group_size, rounding_mode="floor")
    rhs_old = torch.div(expansion.rhs_old_ranks, group_size, rounding_mode="floor")
    rhs_new = torch.div(expansion.rhs_new_ranks, group_size, rounding_mode="floor")
    changed_groups = (lhs_old, lhs_new, rhs_old, rhs_new)
    pair_flat = pair_absence.reshape(-1)
    delta_flat = delta.reshape(-1)
    for position, group in enumerate(changed_groups):
        baseline = (lhs_old == group) | (rhs_old == group)
        lhs_only = (lhs_new == group) | (rhs_old == group)
        rhs_only = (lhs_old == group) | (rhs_new == group)
        joint = (lhs_new == group) | (rhs_new == group)
        coefficient = (
            joint.to(torch.float32)
            - lhs_only.to(torch.float32)
            - rhs_only.to(torch.float32)
            + baseline.to(torch.float32)
        )
        if position:
            first = torch.ones_like(coefficient, dtype=torch.bool)
            for previous in changed_groups[:position]:
                first &= group != previous
            coefficient *= first.to(coefficient.dtype)
        pair_count = pair_flat.index_select(0, expansion.records * num_groups + group)
        delta_flat.index_add_(
            0,
            expansion.actions * num_groups + group,
            coefficient * pair_count,
        )


def _add_dense_pair_deltas(
    delta: torch.Tensor,
    pair_absence: torch.Tensor,
    rows: torch.Tensor,
    *,
    baseline_groups: torch.Tensor,
    lhs_new_groups: torch.Tensor,
    rhs_new_groups: torch.Tensor,
    rhs_valid: torch.Tensor,
) -> None:
    """Add all two-expert interactions with one fixed action/state expansion."""

    num_actions = rows.shape[0]
    state_count = baseline_groups.shape[1]
    num_groups = pair_absence.shape[1]
    num_experts = baseline_groups.shape[0]
    lhs_experts = rows[:, 3]
    rhs_experts = rows[:, 4].clamp_min(0)
    level_count = baseline_groups.shape[2]
    lhs_old = baseline_groups.index_select(0, lhs_experts).unsqueeze(2).expand(-1, -1, state_count, -1)
    lhs_new = lhs_new_groups.unsqueeze(2).expand(-1, -1, state_count, -1)
    rhs_old = baseline_groups.index_select(0, rhs_experts).unsqueeze(1).expand(-1, state_count, -1, -1)
    rhs_new = rhs_new_groups.unsqueeze(1).expand(-1, state_count, -1, -1)

    states = _long_range(state_count, delta.device)
    lhs_before_rhs = (lhs_experts < rhs_experts).view(-1, 1, 1, 1)
    logical_lo = torch.minimum(lhs_experts, rhs_experts).view(-1, 1, 1, 1)
    logical_hi = torch.maximum(lhs_experts, rhs_experts).view(-1, 1, 1, 1)
    lhs_states = states.view(1, -1, 1, 1)
    rhs_states = states.view(1, 1, -1, 1)
    state_lo = torch.where(lhs_before_rhs, lhs_states, rhs_states)
    state_hi = torch.where(lhs_before_rhs, rhs_states, lhs_states)
    records = _pair_state_indices(
        logical_lo,
        logical_hi,
        state_lo,
        state_hi,
        num_experts=num_experts,
        state_count=state_count,
    ).expand(-1, state_count, state_count, level_count)
    actions = _long_range(num_actions, delta.device).view(-1, 1, 1, 1)
    actions = actions.expand(-1, state_count, state_count, level_count)
    action_valid = rhs_valid.view(-1, 1, 1, 1).to(torch.float32)
    pair_flat = pair_absence.reshape(-1)
    delta_flat = delta.reshape(-1)
    # Presence is f(l, r) = l + r - l*r, so the interaction is exactly
    # -(l_new-l_old)*(r_new-r_old).  Only lhs_new and lhs_old can carry a
    # nonzero coefficient.  Evaluating those two groups directly halves the
    # fixed-shape interaction expansion versus the four-product form.  If lhs
    # does not move, the two entries target the same group and cancel exactly.
    coefficient_new = (rhs_old == lhs_new).to(torch.float32)
    coefficient_new -= (rhs_new == lhs_new).to(torch.float32)
    coefficient_old = (rhs_new == lhs_old).to(torch.float32)
    coefficient_old -= (rhs_old == lhs_old).to(torch.float32)
    lhs_groups = torch.stack((lhs_new, lhs_old), dim=-1)
    coefficient = torch.stack((coefficient_new, coefficient_old), dim=-1)
    coefficient *= action_valid.unsqueeze(-1)
    expanded_records = records.unsqueeze(-1).expand_as(lhs_groups)
    pair_count = pair_flat.index_select(
        0,
        (expanded_records * num_groups + lhs_groups).reshape(-1),
    ).view_as(lhs_groups)
    expanded_actions = actions.unsqueeze(-1).expand_as(lhs_groups)
    delta_flat.index_add_(
        0,
        (expanded_actions * num_groups + lhs_groups).reshape(-1),
        (coefficient * pair_count).reshape(-1),
    )


def _dense_uniform_source_candidate_local_deltas(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    rows: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    physical: torch.Tensor,
    occupancies: tuple[torch.Tensor, ...] | None,
    unique_routes: torch.Tensor | None,
    routes_are_unique: bool,
    token_ordinals: torch.Tensor,
    route_hashes: torch.Tensor | None,
    uniform_baseline: UniformStatisticalBaseline | None,
    uniform_source_rank: int,
    step: int,
    layer_seed: int,
    num_experts: int,
    include_pair_interactions: bool = True,
    include_pair_statistics: bool = True,
    prepare_stage_callback: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, StatisticalRouteTables, StatisticalPairContext] | None:
    """Exact fixed-shape scorer used by the normal rank-local route sample."""

    # A single action can add at most one copy.  Basing the exact modulus on
    # the current fixed table width avoids reserving unreachable states.
    if uniform_baseline is None:
        reachable_copy_count = min(planner.max_copies, int(copy_slots.shape[1]) + 1)
        state_count = _fixed_hash_modulus(reachable_copy_count)
        if state_count is None:
            return None
    else:
        state_count = uniform_baseline.state_count
    widths = planner._count_widths()
    packed_width = sum(widths)
    pair_key_count = _pair_state_key_count(num_experts, state_count)
    if pair_key_count > _MAX_DENSE_PAIR_KEYS:
        return None
    if num_experts * state_count * packed_width > _MAX_UNARY_STAT_ELEMENTS:
        return None
    if pair_key_count * packed_width > _MAX_DENSE_PAIR_STAT_ELEMENTS:
        return None
    if include_pair_statistics:
        interaction_elements = rows.shape[0] * state_count * state_count * len(widths) * 2
        if interaction_elements > _MAX_DENSE_ACTION_INTERACTION_ELEMENTS:
            return None
        pair_events = selected.shape[0] * selected.shape[1] * max(0, selected.shape[1] - 1) // 2
        if pair_events > _MAX_PAIR_EVENTS or pair_events * packed_width > _MAX_PAIR_OCCUPANCY_ELEMENTS:
            return None

    if route_hashes is None:
        route_hashes = _route_hash(
            selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
    route_states = (
        torch.remainder(route_hashes, state_count) if uniform_baseline is None else uniform_baseline.route_states
    )
    state_hashes = _long_range(state_count, selected.device)
    rhs_valid = rows[:, 4] >= 0
    num_slots = int(layout.numel())
    lhs_options, rhs_options, _ = planner._candidate_copy_options(layout, copy_slots, rows)
    if uniform_baseline is None:
        baseline_options = torch.cat(
            (
                copy_slots,
                torch.full(
                    (copy_slots.shape[0], lhs_options.shape[1] - copy_slots.shape[1]),
                    num_slots,
                    dtype=torch.long,
                    device=selected.device,
                ),
            ),
            dim=1,
        )
        option_rows = (baseline_options, lhs_options, rhs_options)
    else:
        option_rows = (lhs_options, rhs_options)
    all_options = torch.cat(option_rows, dim=0)
    all_slots = _uniform_candidate_route_slots(
        planner,
        all_options,
        state_hashes,
        uniform_source_rank,
        num_slots,
    )
    action_count = rows.shape[0]
    if uniform_baseline is None:
        baseline_count = copy_slots.shape[0]
        baseline_slots_by_state = all_slots[:baseline_count]
        action_slots = all_slots[baseline_count:]
    else:
        baseline_slots_by_state = uniform_baseline.slots_by_state
        action_slots = all_slots
    baseline_rank_by_state = torch.div(
        baseline_slots_by_state,
        planner.slots_per_rank,
        rounding_mode="floor",
    )
    lhs_new_slot_by_state = action_slots[:action_count]
    rhs_new_slot_by_state = action_slots[action_count:]
    lhs_new_rank_by_state = torch.div(lhs_new_slot_by_state, planner.slots_per_rank, rounding_mode="floor")
    rhs_new_rank_by_state = torch.div(rhs_new_slot_by_state, planner.slots_per_rank, rounding_mode="floor")
    if prepare_stage_callback is not None:
        prepare_stage_callback("candidate_routes")
    pair_records = None
    if include_pair_statistics:
        pair_records = _build_dense_pair_events(
            selected,
            route_states,
            unique_routes,
            routes_are_unique=routes_are_unique,
            num_experts=num_experts,
            state_count=state_count,
        )
        if prepare_stage_callback is not None:
            prepare_stage_callback("pair_events")

    route_ranks_current = torch.div(physical, planner.slots_per_rank, rounding_mode="floor")
    level_sizes = (1,) + tuple(
        int(size) for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    widths = tuple(planner.ep_size // size for size in level_sizes)
    offsets: list[int] = []
    offset = 0
    for width in widths:
        offsets.append(offset)
        offset += width
    packed_groups = torch.stack(
        tuple(
            torch.div(route_ranks_current, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    packed_width = sum(widths)
    if occupancies is None:
        occupancy = torch.zeros(
            (selected.shape[0], packed_width),
            dtype=torch.int32,
            device=selected.device,
        )
        updates = (
            torch.ones_like(packed_groups, dtype=torch.int32)
            if routes_are_unique
            else unique_routes.to(torch.int32).unsqueeze(2).expand_as(packed_groups)
        )
        occupancy.scatter_add_(1, packed_groups.reshape(selected.shape[0], -1), updates.reshape(selected.shape[0], -1))
    else:
        occupancy = torch.cat(occupancies, dim=1)
    absence = _dense_unary_absence_statistics(
        selected,
        route_states,
        unique_routes,
        packed_groups,
        occupancy,
        routes_are_unique=routes_are_unique,
        num_experts=num_experts,
        state_count=state_count,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("unary_statistics")
    baseline_groups = torch.stack(
        tuple(
            torch.div(baseline_rank_by_state, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    lhs_new_groups = torch.stack(
        tuple(
            torch.div(lhs_new_rank_by_state, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    rhs_new_groups = torch.stack(
        tuple(
            torch.div(rhs_new_rank_by_state, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    delta = torch.zeros((rows.shape[0], packed_width), dtype=torch.float32, device=selected.device)
    lhs_old_groups = baseline_groups.index_select(0, rows[:, 3])
    rhs_old_groups = baseline_groups.index_select(0, rows[:, 4].clamp_min(0))
    _add_dense_unary_deltas(
        delta,
        absence,
        rows,
        lhs_old_groups=lhs_old_groups,
        lhs_new_groups=lhs_new_groups,
        rhs_old_groups=rhs_old_groups,
        rhs_new_groups=rhs_new_groups,
        rhs_valid=rhs_valid,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("unary_scoring")
    pair_absence = None
    if pair_records is not None:
        pair_absence = _dense_pair_absence_statistics(pair_records, occupancy, baseline_groups)
        if prepare_stage_callback is not None:
            prepare_stage_callback("pair_statistics")
        if include_pair_interactions:
            _add_dense_pair_deltas(
                delta,
                pair_absence,
                rows,
                baseline_groups=baseline_groups,
                lhs_new_groups=lhs_new_groups,
                rhs_new_groups=rhs_new_groups,
                rhs_valid=rhs_valid,
            )
            if prepare_stage_callback is not None:
                prepare_stage_callback("pair_interaction")
    route_tables = StatisticalRouteTables(
        state_count=state_count,
        baseline_slots=baseline_slots_by_state,
        lhs_slots=lhs_new_slot_by_state,
        rhs_slots=rhs_new_slot_by_state,
    )
    pair_context = StatisticalPairContext(
        selected=selected,
        route_states=route_states,
        unique_routes=unique_routes,
        routes_are_unique=routes_are_unique,
        occupancy=occupancy,
        baseline_groups=baseline_groups,
        lhs_new_groups=lhs_new_groups,
        rhs_new_groups=rhs_new_groups,
        rhs_valid=rhs_valid,
        pair_events=pair_records,
        pair_absence=pair_absence,
        num_experts=num_experts,
        state_count=state_count,
    )
    return delta, route_tables, pair_context


def statistical_unary_candidate_local_deltas(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    rows: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    physical: torch.Tensor,
    occupancies: tuple[torch.Tensor, ...],
    token_ordinals: torch.Tensor,
    route_hashes: torch.Tensor,
    uniform_source_rank: int | None,
    uniform_baseline: UniformStatisticalBaseline | None,
    routes_are_unique: bool,
    unique_routes: torch.Tensor | None,
    step: int,
    layer_seed: int,
    num_experts: int,
    prepare_stage_callback: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, StatisticalRouteTables, StatisticalPairContext] | None:
    """Build all unary deltas and retain only data needed for sparse pair reranking."""

    if uniform_source_rank is None or rows.numel() == 0:
        return None
    return _dense_uniform_source_candidate_local_deltas(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=occupancies,
        unique_routes=unique_routes,
        routes_are_unique=routes_are_unique,
        token_ordinals=token_ordinals,
        route_hashes=route_hashes,
        uniform_baseline=uniform_baseline,
        uniform_source_rank=uniform_source_rank,
        step=step,
        layer_seed=layer_seed,
        num_experts=num_experts,
        include_pair_interactions=False,
        prepare_stage_callback=prepare_stage_callback,
    )


def statistical_primitive_fast_path_available(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    *,
    copy_slots: torch.Tensor,
    num_experts: int,
    defer_pair_statistics: bool,
    batched_layer_count: int = 1,
) -> bool:
    """Check bounded-memory requirements before primitive preparation."""

    reachable_copy_count = min(planner.max_copies, int(copy_slots.shape[1]) + 1)
    state_count = _fixed_hash_modulus(reachable_copy_count)
    if state_count is None:
        return False
    packed_width = sum(planner._count_widths())
    if int(num_experts) * state_count * packed_width > _MAX_UNARY_STAT_ELEMENTS:
        return False
    pair_events = selected.shape[0] * selected.shape[1] * max(0, selected.shape[1] - 1) // 2
    if pair_events > _MAX_PAIR_EVENTS or pair_events * packed_width > _MAX_PAIR_OCCUPANCY_ELEMENTS:
        return False
    pair_key_count = _pair_state_key_count(int(num_experts), state_count)
    if pair_key_count > _MAX_DENSE_PAIR_KEYS:
        return False
    if pair_key_count * max(1, int(batched_layer_count)) > _MAX_BATCHED_PAIR_LOOKUP_ELEMENTS:
        return False
    if not defer_pair_statistics:
        if pair_key_count * packed_width > _MAX_DENSE_PAIR_STAT_ELEMENTS:
            return False
    return True


def statistical_primitive_unary_local_deltas(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    primitive_spec: StatisticalPrimitiveSpec,
    *,
    copy_slots: torch.Tensor,
    token_ordinals: torch.Tensor,
    uniform_source_rank: int | None,
    routes_are_unique: bool,
    step: int,
    layer_seed: int,
    num_experts: int,
    defer_pair_statistics: bool = False,
    prepare_stage_callback: Callable[[str], None] | None = None,
) -> StatisticalPrimitiveResult | None:
    """Build exact full-state unary deltas once per unique expert transition."""

    if uniform_source_rank is None or primitive_spec.experts.numel() == 0:
        return None
    if not statistical_primitive_fast_path_available(
        planner,
        selected,
        copy_slots=copy_slots,
        num_experts=num_experts,
        defer_pair_statistics=defer_pair_statistics,
    ):
        return None
    route_hashes = _route_hash(
        selected,
        token_ordinals=token_ordinals,
        step=step,
        layer_seed=layer_seed,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("route_hash")
    baseline = uniform_statistical_baseline_routes(
        planner,
        selected,
        copy_slots,
        route_hashes,
        source_rank=uniform_source_rank,
    )
    if baseline is None:
        return None
    physical = baseline.physical
    if prepare_stage_callback is not None:
        prepare_stage_callback("baseline_route")
    unique_routes = None if routes_are_unique else _canonical_route_mask(selected)
    occupancies = planner._token_level_occupancies(
        physical,
        route_weights=None if routes_are_unique else unique_routes,
    )
    occupancy = torch.cat(occupancies, dim=1)
    if prepare_stage_callback is not None:
        prepare_stage_callback("occupancy")

    state_count = baseline.state_count
    state_hashes = _long_range(state_count, selected.device)
    primitive_slots = _uniform_candidate_route_slots(
        planner,
        primitive_spec.options,
        state_hashes,
        uniform_source_rank,
        planner.ep_size * planner.slots_per_rank,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("candidate_routes")
    primitive_ranks = torch.div(primitive_slots, planner.slots_per_rank, rounding_mode="floor")
    baseline_ranks = torch.div(baseline.slots_by_state, planner.slots_per_rank, rounding_mode="floor")
    route_ranks_current = torch.div(physical, planner.slots_per_rank, rounding_mode="floor")
    level_sizes = (1,) + tuple(
        int(size) for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    widths = tuple(planner.ep_size // size for size in level_sizes)
    offsets: list[int] = []
    offset = 0
    for width in widths:
        offsets.append(offset)
        offset += width
    packed_width = sum(widths)
    packed_groups = torch.stack(
        tuple(
            torch.div(route_ranks_current, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    pair_events = _build_dense_pair_events(
        selected,
        baseline.route_states,
        unique_routes,
        routes_are_unique=routes_are_unique,
        num_experts=num_experts,
        state_count=state_count,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("pair_events")
    absence = _dense_unary_absence_statistics(
        selected,
        baseline.route_states,
        unique_routes,
        packed_groups,
        occupancy,
        routes_are_unique=routes_are_unique,
        num_experts=num_experts,
        state_count=state_count,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("unary_statistics")
    baseline_groups = torch.stack(
        tuple(
            torch.div(baseline_ranks, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    primitive_groups = torch.stack(
        tuple(
            torch.div(primitive_ranks, size, rounding_mode="floor") + group_offset
            for size, group_offset in zip(level_sizes, offsets, strict=True)
        ),
        dim=2,
    )
    primitive_delta = torch.zeros(
        (primitive_spec.experts.numel(), packed_width),
        dtype=torch.float32,
        device=selected.device,
    )
    primitive_old_groups = baseline_groups.index_select(0, primitive_spec.experts)
    _add_unary_deltas(
        primitive_delta,
        absence,
        primitive_spec.experts,
        primitive_old_groups,
        primitive_groups,
    )
    if prepare_stage_callback is not None:
        prepare_stage_callback("unary_scoring")

    pair_absence = None
    if pair_events is not None and not defer_pair_statistics:
        pair_absence = _dense_pair_absence_statistics(pair_events, occupancy, baseline_groups)
        if prepare_stage_callback is not None:
            prepare_stage_callback("pair_statistics")
    # When deferred, exact unary scoring retains only pair_events.  Top-K
    # reranking then filters those events by selected expert pairs and builds
    # the compact exact pair_absence table after shortlist selection.
    context = StatisticalPrimitiveContext(
        spec=primitive_spec,
        selected=selected,
        route_states=baseline.route_states,
        unique_routes=unique_routes,
        routes_are_unique=routes_are_unique,
        occupancy=occupancy,
        baseline_groups=baseline_groups,
        primitive_groups=primitive_groups,
        pair_events=pair_events,
        pair_absence=pair_absence,
        baseline_slots=baseline.slots_by_state,
        primitive_slots=primitive_slots,
        num_experts=num_experts,
        state_count=state_count,
    )
    return StatisticalPrimitiveResult(
        primitive_delta=primitive_delta,
        context=context,
        physical=physical,
        route_hashes=route_hashes,
    )


def statistical_primitive_selected_pair_context(
    planner: GreedyCommunicationPlanner,
    context: StatisticalPrimitiveContext,
    rows: torch.Tensor,
    action_indices: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    uniform_source_rank: int,
    materialize_route_tables: bool = True,
) -> tuple[StatisticalPairContext, torch.Tensor, StatisticalRouteTables | None]:
    """Materialize exact pair groups and real physical routes for selected actions."""

    selected_rows = rows.index_select(0, action_indices)
    lhs_ids = context.spec.lhs_ids.index_select(0, action_indices)
    rhs_ids = context.spec.rhs_ids.index_select(0, action_indices)
    rhs_valid = rhs_ids >= 0
    safe_rhs_ids = rhs_ids.clamp_min(0)
    lhs_groups = context.primitive_groups.index_select(0, lhs_ids)
    rhs_groups = context.primitive_groups.index_select(0, safe_rhs_ids)
    rhs_groups = torch.where(
        rhs_valid.view(-1, 1, 1),
        rhs_groups,
        context.baseline_groups.index_select(0, selected_rows[:, 4].clamp_min(0)),
    )
    pair_context = StatisticalPairContext(
        selected=context.selected,
        route_states=context.route_states,
        unique_routes=context.unique_routes,
        routes_are_unique=context.routes_are_unique,
        occupancy=context.occupancy,
        baseline_groups=context.baseline_groups,
        lhs_new_groups=lhs_groups,
        rhs_new_groups=rhs_groups,
        rhs_valid=rhs_valid,
        pair_events=context.pair_events,
        pair_absence=context.pair_absence,
        num_experts=context.num_experts,
        state_count=context.state_count,
    )
    route_tables = None
    if materialize_route_tables:
        # Primitive identities deliberately collapse slots on the same rank
        # because communication and assignment costs cannot distinguish their
        # local slot offsets.  Winner remapping must nevertheless preserve the
        # real physical slot selected by the action, so rebuild only these
        # Top-K route tables when the caller actually requests a final route.
        lhs_options, rhs_options, option_rhs_valid = planner._candidate_copy_options(
            layout,
            copy_slots,
            selected_rows,
        )
        state_hashes = _long_range(context.state_count, selected_rows.device)
        lhs_slots = _uniform_candidate_route_slots(
            planner,
            lhs_options,
            state_hashes,
            uniform_source_rank,
            int(layout.numel()),
        )
        rhs_slots = _uniform_candidate_route_slots(
            planner,
            rhs_options,
            state_hashes,
            uniform_source_rank,
            int(layout.numel()),
        )
        rhs_valid = rhs_valid & option_rhs_valid
        rhs_slots = torch.where(
            rhs_valid.view(-1, 1),
            rhs_slots,
            context.baseline_slots.index_select(0, selected_rows[:, 4].clamp_min(0)),
        )
        route_tables = StatisticalRouteTables(
            state_count=context.state_count,
            baseline_slots=context.baseline_slots,
            lhs_slots=lhs_slots,
            rhs_slots=rhs_slots,
        )
    return pair_context, selected_rows, route_tables


def statistical_proxy_candidate_local_deltas(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    rows: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    uniform_source_rank: int | None,
    routes_are_unique: bool,
    token_ordinals: torch.Tensor,
    step: int,
    layer_seed: int,
    num_experts: int,
    prepare_stage_callback: Callable[[str], None] | None = None,
) -> StatisticalProxyResult | None:
    """Score every action with one deterministic route state and no pair table.

    The proxy deliberately collapses hash-tie states to state zero.  It keeps
    the exact unary presence/absence accounting for that collapsed route, but
    skips all pair-event construction.  This makes it cheap enough to run
    before candidate route tables for the exact scorer are materialized.
    """

    if uniform_source_rank is None or rows.numel() == 0:
        return None
    num_slots = int(layout.numel())
    state_hashes = _long_range(1, selected.device)
    baseline_slots = _uniform_candidate_route_slots(
        planner,
        copy_slots,
        state_hashes,
        uniform_source_rank,
        num_slots,
    )
    proxy_hashes = torch.zeros_like(selected, dtype=torch.long)
    physical = baseline_slots.reshape(-1).index_select(0, selected.reshape(-1)).view_as(selected)
    baseline = UniformStatisticalBaseline(
        state_count=1,
        route_states=proxy_hashes,
        slots_by_state=baseline_slots,
        physical=physical,
    )
    unique_routes = None if routes_are_unique else _canonical_route_mask(selected)
    occupancies = planner._token_level_occupancies(
        physical,
        route_weights=None if routes_are_unique else unique_routes,
    )
    result = _dense_uniform_source_candidate_local_deltas(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=occupancies,
        unique_routes=unique_routes,
        routes_are_unique=routes_are_unique,
        token_ordinals=token_ordinals,
        route_hashes=proxy_hashes,
        uniform_baseline=baseline,
        uniform_source_rank=uniform_source_rank,
        step=step,
        layer_seed=layer_seed,
        num_experts=num_experts,
        include_pair_interactions=False,
        include_pair_statistics=False,
        prepare_stage_callback=prepare_stage_callback,
    )
    if result is None:
        return None
    candidate_delta, route_tables, _pair_context = result
    return StatisticalProxyResult(
        candidate_delta=candidate_delta,
        route_tables=route_tables,
        physical=physical,
        route_hashes=proxy_hashes,
    )


def statistical_pair_interaction_bound_local(
    context: StatisticalPairContext,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Return strict route-state-aware bounds for every action and packed group."""

    pair_events = context.pair_events
    if rows.numel() == 0:
        return rows.new_empty((0, context.occupancy.shape[1]), dtype=torch.float32)
    if pair_events is None:
        return torch.zeros(
            (rows.shape[0], context.occupancy.shape[1]),
            dtype=torch.float32,
            device=rows.device,
        )
    pair_state_counts = torch.zeros(
        (pair_events.key_count,),
        dtype=torch.float32,
        device=rows.device,
    )
    pair_state_counts.index_add_(
        0,
        pair_events.pair_states,
        pair_events.valid.to(torch.float32),
    )
    action_count = rows.shape[0]
    state_count = context.state_count
    level_count = context.baseline_groups.shape[2]
    packed_width = context.occupancy.shape[1]
    lhs_experts = rows[:, 3]
    rhs_valid = rows[:, 4] >= 0
    rhs_experts = rows[:, 4].clamp_min(0)
    lhs_old = context.baseline_groups.index_select(0, lhs_experts).unsqueeze(2)
    lhs_old = lhs_old.expand(-1, -1, state_count, -1)
    rhs_old = context.baseline_groups.index_select(0, rhs_experts).unsqueeze(1)
    rhs_old = rhs_old.expand(-1, state_count, -1, -1)
    lhs_new = context.lhs_new_groups.unsqueeze(2).expand(-1, -1, state_count, -1)
    rhs_new = context.rhs_new_groups.unsqueeze(1).expand(-1, state_count, -1, -1)

    states = _long_range(state_count, rows.device)
    lhs_before_rhs = (lhs_experts < rhs_experts).view(-1, 1, 1)
    logical_lo = torch.minimum(lhs_experts, rhs_experts).view(-1, 1, 1)
    logical_hi = torch.maximum(lhs_experts, rhs_experts).view(-1, 1, 1)
    lhs_states = states.view(1, -1, 1)
    rhs_states = states.view(1, 1, -1)
    state_lo = torch.where(lhs_before_rhs, lhs_states, rhs_states)
    state_hi = torch.where(lhs_before_rhs, rhs_states, lhs_states)
    records = _pair_state_indices(
        logical_lo,
        logical_hi,
        state_lo,
        state_hi,
        num_experts=context.num_experts,
        state_count=state_count,
    ).expand(-1, state_count, state_count)
    record_counts = pair_state_counts.index_select(0, records.reshape(-1))
    record_counts = record_counts.view(action_count, state_count, state_count, 1)

    coefficient_new = (rhs_old == lhs_new).to(torch.float32)
    coefficient_new -= (rhs_new == lhs_new).to(torch.float32)
    coefficient_old = (rhs_new == lhs_old).to(torch.float32)
    coefficient_old -= (rhs_old == lhs_old).to(torch.float32)
    groups = torch.stack((lhs_new, lhs_old), dim=-1)
    coefficient = torch.stack((coefficient_new, coefficient_old), dim=-1).abs()
    coefficient *= rhs_valid.view(-1, 1, 1, 1, 1).to(torch.float32)
    values = coefficient * record_counts.unsqueeze(-1)
    actions = _long_range(action_count, rows.device).view(-1, 1, 1, 1, 1)
    actions = actions.expand(action_count, state_count, state_count, level_count, 2)
    bounds = torch.zeros(
        (action_count, packed_width),
        dtype=torch.float32,
        device=rows.device,
    )
    bounds.reshape(-1).index_add_(
        0,
        (actions * packed_width + groups).reshape(-1),
        values.reshape(-1),
    )
    return bounds


def statistical_selected_pair_local_deltas(
    context: StatisticalPairContext,
    rows: torch.Tensor,
    action_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute exact pair corrections only for the selected action rows."""

    packed_width = context.occupancy.shape[1]
    selected_count = int(action_indices.numel())
    delta = torch.zeros(
        (selected_count, packed_width),
        dtype=torch.float32,
        device=context.occupancy.device,
    )
    pair_events = context.pair_events
    if selected_count == 0 or pair_events is None:
        return delta
    if context.pair_absence is not None:
        selected_rows = rows.index_select(0, action_indices)
        _add_dense_pair_deltas(
            delta,
            context.pair_absence,
            selected_rows,
            baseline_groups=context.baseline_groups,
            lhs_new_groups=context.lhs_new_groups.index_select(0, action_indices),
            rhs_new_groups=context.rhs_new_groups.index_select(0, action_indices),
            rhs_valid=context.rhs_valid.index_select(0, action_indices),
        )
        return delta

    selected_rows = rows.index_select(0, action_indices)
    lhs_experts = selected_rows[:, 3]
    rhs_experts = selected_rows[:, 4].clamp_min(0)
    state_count = context.state_count
    states = _long_range(state_count, delta.device)
    lhs_before_rhs = (lhs_experts < rhs_experts).view(-1, 1, 1)
    logical_lo = torch.minimum(lhs_experts, rhs_experts).view(-1, 1, 1)
    logical_hi = torch.maximum(lhs_experts, rhs_experts).view(-1, 1, 1)
    lhs_states = states.view(1, -1, 1)
    rhs_states = states.view(1, 1, -1)
    state_lo = torch.where(lhs_before_rhs, lhs_states, rhs_states)
    state_hi = torch.where(lhs_before_rhs, rhs_states, lhs_states)
    action_pair_states = _pair_state_indices(
        logical_lo,
        logical_hi,
        state_lo,
        state_hi,
        num_experts=context.num_experts,
        state_count=state_count,
    ).expand(-1, state_count, state_count)
    unique_pair_states, action_records = torch.unique(
        action_pair_states.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    action_records = action_records.view(selected_count, state_count, state_count)

    record_lookup = torch.full(
        (pair_events.key_count,),
        -1,
        dtype=torch.long,
        device=delta.device,
    )
    record_lookup[unique_pair_states] = _long_range(unique_pair_states.numel(), delta.device)
    event_records = record_lookup.index_select(0, pair_events.pair_states)
    selected_events = (event_records >= 0) & pair_events.valid
    event_indices = torch.nonzero(selected_events, as_tuple=False).flatten()
    if event_indices.numel() == 0:
        return delta

    compact_records = event_records.index_select(0, event_indices)
    event_tokens = pair_events.event_tokens.index_select(0, event_indices)
    pseudo_lo = pair_events.pseudo_lo.index_select(0, event_indices)
    pseudo_hi = pair_events.pseudo_hi.index_select(0, event_indices)
    baseline_flat = context.baseline_groups.reshape(context.num_experts * state_count, -1)
    group_lo = baseline_flat.index_select(0, pseudo_lo)
    group_hi = baseline_flat.index_select(0, pseudo_hi)
    event_occupancy = context.occupancy.index_select(0, event_tokens).clone()
    changed_groups = torch.cat((group_lo, group_hi), dim=1)
    event_occupancy.scatter_add_(
        1,
        changed_groups,
        -torch.ones_like(changed_groups, dtype=event_occupancy.dtype),
    )
    absence = event_occupancy.eq(0).to(torch.float32)
    pair_absence = torch.zeros(
        (unique_pair_states.numel(), packed_width),
        dtype=torch.float32,
        device=delta.device,
    )
    pair_absence.index_add_(0, compact_records, absence)

    return _selected_pair_delta_from_absence(
        context,
        rows,
        action_indices,
        action_records,
        pair_absence,
    )


def _selected_pair_delta_from_absence(
    context: StatisticalPairContext,
    rows: torch.Tensor,
    action_indices: torch.Tensor,
    action_records: torch.Tensor,
    pair_absence: torch.Tensor,
) -> torch.Tensor:
    """Apply compact exact pair-absence rows to selected actions."""

    selected_count = int(action_indices.numel())
    packed_width = context.occupancy.shape[1]
    delta = torch.zeros(
        (selected_count, packed_width),
        dtype=torch.float32,
        device=context.occupancy.device,
    )
    if selected_count == 0:
        return delta
    selected_rows = rows.index_select(0, action_indices)
    rhs_valid = selected_rows[:, 4] >= 0
    lhs_experts = selected_rows[:, 3]
    rhs_experts = selected_rows[:, 4].clamp_min(0)
    state_count = context.state_count
    level_count = context.baseline_groups.shape[2]
    lhs_old = context.baseline_groups.index_select(0, lhs_experts).unsqueeze(2)
    lhs_old = lhs_old.expand(-1, -1, state_count, -1)
    rhs_old = context.baseline_groups.index_select(0, rhs_experts).unsqueeze(1)
    rhs_old = rhs_old.expand(-1, state_count, -1, -1)
    lhs_new = context.lhs_new_groups.index_select(0, action_indices).unsqueeze(2)
    lhs_new = lhs_new.expand(-1, -1, state_count, -1)
    rhs_new = context.rhs_new_groups.index_select(0, action_indices).unsqueeze(1)
    rhs_new = rhs_new.expand(-1, state_count, -1, -1)
    coefficient_new = (rhs_old == lhs_new).to(torch.float32)
    coefficient_new -= (rhs_new == lhs_new).to(torch.float32)
    coefficient_old = (rhs_new == lhs_old).to(torch.float32)
    coefficient_old -= (rhs_old == lhs_old).to(torch.float32)
    lhs_groups = torch.stack((lhs_new, lhs_old), dim=-1)
    coefficient = torch.stack((coefficient_new, coefficient_old), dim=-1)
    coefficient *= rhs_valid.view(-1, 1, 1, 1, 1).to(torch.float32)
    records = action_records.unsqueeze(-1).expand(-1, -1, -1, level_count)
    records = records.unsqueeze(-1).expand_as(lhs_groups)
    pair_count_values = (
        pair_absence.reshape(-1)
        .index_select(
            0,
            (records * packed_width + lhs_groups).reshape(-1),
        )
        .view_as(lhs_groups)
    )
    actions = _long_range(selected_count, delta.device).view(-1, 1, 1, 1, 1)
    actions = actions.expand_as(lhs_groups)
    delta.reshape(-1).index_add_(
        0,
        (actions * packed_width + lhs_groups).reshape(-1),
        (coefficient * pair_count_values).reshape(-1),
    )
    return delta


def statistical_batched_selected_pair_local_deltas(
    contexts: Sequence[StatisticalPairContext],
    rows_by_layer: Sequence[torch.Tensor],
    action_indices_by_layer: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute sparse exact pair corrections for independent layers with one event filter."""

    layer_count = len(contexts)
    if not (len(rows_by_layer) == len(action_indices_by_layer) == layer_count):
        raise ValueError("Batched pair scorer inputs must have identical layer counts.")
    if layer_count == 0:
        return []
    if all(context.pair_absence is not None for context in contexts):
        selected_counts = [int(value.numel()) for value in action_indices_by_layer]
        if len(set(selected_counts)) == 1:
            selected_count = selected_counts[0]
            first = contexts[0]
            state_count = first.state_count
            packed_width = first.occupancy.shape[1]
            level_count = first.baseline_groups.shape[2]
            if all(
                context.state_count == state_count
                and context.occupancy.shape[1] == packed_width
                and context.baseline_groups.shape[2] == level_count
                for context in contexts
            ):
                selected_rows = torch.stack(
                    [
                        rows.index_select(0, action_indices)
                        for rows, action_indices in zip(
                            rows_by_layer,
                            action_indices_by_layer,
                            strict=True,
                        )
                    ]
                )
                lhs_experts = selected_rows[:, :, 3]
                rhs_valid = selected_rows[:, :, 4] >= 0
                rhs_experts = selected_rows[:, :, 4].clamp_min(0)
                baseline_groups = torch.stack([context.baseline_groups for context in contexts])
                lhs_old_indices = lhs_experts.unsqueeze(-1).unsqueeze(-1)
                lhs_old_indices = lhs_old_indices.expand(-1, -1, state_count, level_count)
                lhs_old = baseline_groups.gather(1, lhs_old_indices).unsqueeze(3)
                lhs_old = lhs_old.expand(-1, -1, -1, state_count, -1)
                rhs_old_indices = rhs_experts.unsqueeze(-1).unsqueeze(-1)
                rhs_old_indices = rhs_old_indices.expand(-1, -1, state_count, level_count)
                rhs_old = baseline_groups.gather(1, rhs_old_indices).unsqueeze(2)
                rhs_old = rhs_old.expand(-1, -1, state_count, -1, -1)
                lhs_new = torch.stack(
                    [
                        context.lhs_new_groups.index_select(0, action_indices)
                        for context, action_indices in zip(
                            contexts,
                            action_indices_by_layer,
                            strict=True,
                        )
                    ]
                ).unsqueeze(3)
                lhs_new = lhs_new.expand(-1, -1, -1, state_count, -1)
                rhs_new = torch.stack(
                    [
                        context.rhs_new_groups.index_select(0, action_indices)
                        for context, action_indices in zip(
                            contexts,
                            action_indices_by_layer,
                            strict=True,
                        )
                    ]
                ).unsqueeze(2)
                rhs_new = rhs_new.expand(-1, -1, state_count, -1, -1)

                states = _long_range(state_count, first.occupancy.device)
                lhs_before_rhs = (lhs_experts < rhs_experts).view(layer_count, selected_count, 1, 1)
                logical_lo = torch.minimum(lhs_experts, rhs_experts).view(
                    layer_count,
                    selected_count,
                    1,
                    1,
                )
                logical_hi = torch.maximum(lhs_experts, rhs_experts).view(
                    layer_count,
                    selected_count,
                    1,
                    1,
                )
                lhs_states = states.view(1, 1, -1, 1)
                rhs_states = states.view(1, 1, 1, -1)
                state_lo = torch.where(lhs_before_rhs, lhs_states, rhs_states)
                state_hi = torch.where(lhs_before_rhs, rhs_states, lhs_states)
                records = _pair_state_indices(
                    logical_lo,
                    logical_hi,
                    state_lo,
                    state_hi,
                    num_experts=first.num_experts,
                    state_count=state_count,
                ).expand(-1, -1, state_count, state_count)
                coefficient_new = (rhs_old == lhs_new).to(torch.float32)
                coefficient_new -= (rhs_new == lhs_new).to(torch.float32)
                coefficient_old = (rhs_new == lhs_old).to(torch.float32)
                coefficient_old -= (rhs_old == lhs_old).to(torch.float32)
                groups = torch.stack((lhs_new, lhs_old), dim=-1)
                coefficient = torch.stack((coefficient_new, coefficient_old), dim=-1)
                coefficient *= rhs_valid.view(layer_count, selected_count, 1, 1, 1, 1).to(torch.float32)
                expanded_records = records.unsqueeze(-1).unsqueeze(-1).expand_as(groups)
                pair_count_rows = []
                for layer_index, context in enumerate(contexts):
                    assert context.pair_absence is not None
                    pair_count_rows.append(
                        context.pair_absence.reshape(-1)
                        .index_select(
                            0,
                            (expanded_records[layer_index] * packed_width + groups[layer_index]).reshape(-1),
                        )
                        .view_as(groups[layer_index])
                    )
                pair_count_values = torch.stack(pair_count_rows)
                actions = _long_range(layer_count * selected_count, first.occupancy.device)
                actions = actions.view(layer_count, selected_count, 1, 1, 1, 1).expand_as(groups)
                delta = torch.zeros(
                    (layer_count * selected_count, packed_width),
                    dtype=torch.float32,
                    device=first.occupancy.device,
                )
                delta.reshape(-1).index_add_(
                    0,
                    (actions * packed_width + groups).reshape(-1),
                    (coefficient * pair_count_values).reshape(-1),
                )
                return list(delta.view(layer_count, selected_count, packed_width).unbind())
    first = contexts[0]
    state_count = first.state_count
    packed_width = first.occupancy.shape[1]
    num_experts = first.num_experts
    if any(
        context.state_count != state_count
        or context.occupancy.shape[1] != packed_width
        or context.num_experts != num_experts
        or context.pair_events is None
        for context in contexts
    ):
        return [
            statistical_selected_pair_local_deltas(context, rows, action_indices)
            for context, rows, action_indices in zip(
                contexts,
                rows_by_layer,
                action_indices_by_layer,
                strict=True,
            )
        ]
    pair_events = [context.pair_events for context in contexts]
    assert all(value is not None for value in pair_events)
    dense_events = [value for value in pair_events if value is not None]
    key_count = dense_events[0].key_count
    if any(value.key_count != key_count for value in dense_events):
        return [
            statistical_selected_pair_local_deltas(context, rows, action_indices)
            for context, rows, action_indices in zip(
                contexts,
                rows_by_layer,
                action_indices_by_layer,
                strict=True,
            )
        ]

    action_keys: list[torch.Tensor] = []
    action_key_counts: list[int] = []
    states = _long_range(state_count, first.occupancy.device)
    for layer_index, (rows, action_indices) in enumerate(zip(rows_by_layer, action_indices_by_layer, strict=True)):
        selected_rows = rows.index_select(0, action_indices)
        lhs = selected_rows[:, 3]
        rhs = selected_rows[:, 4].clamp_min(0)
        lhs_before_rhs = (lhs < rhs).view(-1, 1, 1)
        logical_lo = torch.minimum(lhs, rhs).view(-1, 1, 1)
        logical_hi = torch.maximum(lhs, rhs).view(-1, 1, 1)
        lhs_states = states.view(1, -1, 1)
        rhs_states = states.view(1, 1, -1)
        state_lo = torch.where(lhs_before_rhs, lhs_states, rhs_states)
        state_hi = torch.where(lhs_before_rhs, rhs_states, lhs_states)
        keys = _pair_state_indices(
            logical_lo,
            logical_hi,
            state_lo,
            state_hi,
            num_experts=num_experts,
            state_count=state_count,
        ).expand(-1, state_count, state_count)
        encoded = keys.reshape(-1) + layer_index * key_count
        action_keys.append(encoded)
        action_key_counts.append(int(encoded.numel()))
    unique_keys, inverse_records = torch.unique(
        torch.cat(action_keys),
        sorted=True,
        return_inverse=True,
    )
    record_lookup = torch.full(
        (layer_count * key_count,),
        -1,
        dtype=torch.long,
        device=first.occupancy.device,
    )
    record_lookup[unique_keys] = _long_range(unique_keys.numel(), first.occupancy.device)

    token_offset = 0
    pseudo_width = num_experts * state_count
    encoded_events = []
    event_valid = []
    event_tokens = []
    pseudo_lo = []
    pseudo_hi = []
    for layer_index, (context, events) in enumerate(zip(contexts, dense_events, strict=True)):
        encoded_events.append(events.pair_states + layer_index * key_count)
        event_valid.append(events.valid)
        event_tokens.append(events.event_tokens + token_offset)
        pseudo_lo.append(events.pseudo_lo + layer_index * pseudo_width)
        pseudo_hi.append(events.pseudo_hi + layer_index * pseudo_width)
        token_offset += context.occupancy.shape[0]
    event_records = record_lookup.index_select(0, torch.cat(encoded_events))
    selected_events = (event_records >= 0) & torch.cat(event_valid)
    event_indices = torch.nonzero(selected_events, as_tuple=False).flatten()
    compact_records = event_records.index_select(0, event_indices)
    occupancy = torch.cat([context.occupancy for context in contexts], dim=0)
    selected_tokens = torch.cat(event_tokens).index_select(0, event_indices)
    selected_pseudo_lo = torch.cat(pseudo_lo).index_select(0, event_indices)
    selected_pseudo_hi = torch.cat(pseudo_hi).index_select(0, event_indices)
    baseline_flat = torch.cat(
        [context.baseline_groups.reshape(pseudo_width, -1) for context in contexts],
        dim=0,
    )
    group_lo = baseline_flat.index_select(0, selected_pseudo_lo)
    group_hi = baseline_flat.index_select(0, selected_pseudo_hi)
    event_occupancy = occupancy.index_select(0, selected_tokens).clone()
    changed_groups = torch.cat((group_lo, group_hi), dim=1)
    event_occupancy.scatter_add_(
        1,
        changed_groups,
        -torch.ones_like(changed_groups, dtype=event_occupancy.dtype),
    )
    pair_absence = torch.zeros(
        (unique_keys.numel(), packed_width),
        dtype=torch.float32,
        device=first.occupancy.device,
    )
    pair_absence.index_add_(
        0,
        compact_records,
        event_occupancy.eq(0).to(torch.float32),
    )

    outputs = []
    inverse_offset = 0
    for context, rows, action_indices, key_elements in zip(
        contexts,
        rows_by_layer,
        action_indices_by_layer,
        action_key_counts,
        strict=True,
    ):
        records = inverse_records[inverse_offset : inverse_offset + key_elements]
        records = records.view(-1, state_count, state_count)
        outputs.append(
            _selected_pair_delta_from_absence(
                context,
                rows,
                action_indices,
                records,
                pair_absence,
            )
        )
        inverse_offset += key_elements
    return outputs


def statistical_candidate_local_deltas(
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    rows: torch.Tensor,
    *,
    layout: torch.Tensor,
    copy_slots: torch.Tensor,
    physical: torch.Tensor,
    occupancies: tuple[torch.Tensor, ...],
    source_ranks: torch.Tensor,
    token_ordinals: torch.Tensor,
    route_hashes: torch.Tensor | None = None,
    uniform_source_rank: int | None = None,
    uniform_baseline: UniformStatisticalBaseline | None = None,
    routes_are_unique: bool = False,
    unique_routes: torch.Tensor | None = None,
    return_route_tables: bool = False,
    step: int,
    layer_seed: int,
    num_experts: int,
    prepare_stage_callback: Callable[[str], None] | None = None,
) -> torch.Tensor | None | tuple[torch.Tensor | None, StatisticalRouteTables | None]:
    """Score all actions exactly from additive unary and pair-state statistics.

    Token routes are scanned only while building statistics. Candidate scoring
    reads the resulting expert-state tables and never remaps candidate tokens.
    """

    if rows.numel() == 0:
        empty = physical.new_empty((0, sum(planner._count_widths())), dtype=torch.float32)
        return (empty, None) if return_route_tables else empty

    reuse_occupancies = unique_routes is not None or routes_are_unique
    if unique_routes is None and not routes_are_unique:
        unique_routes = _canonical_route_mask(selected)

    if uniform_source_rank is not None:
        dense = _dense_uniform_source_candidate_local_deltas(
            planner,
            selected,
            rows,
            layout=layout,
            copy_slots=copy_slots,
            physical=physical,
            occupancies=occupancies if reuse_occupancies else None,
            unique_routes=unique_routes,
            routes_are_unique=routes_are_unique,
            token_ordinals=token_ordinals,
            route_hashes=route_hashes,
            uniform_baseline=uniform_baseline,
            uniform_source_rank=uniform_source_rank,
            step=step,
            layer_seed=layer_seed,
            num_experts=num_experts,
            prepare_stage_callback=prepare_stage_callback,
        )
        if dense is not None:
            deltas, route_tables, _pair_context = dense
            return (deltas, route_tables) if return_route_tables else deltas

    if unique_routes is None:
        unique_routes = torch.ones_like(selected, dtype=torch.bool)
    state_space = _route_state_space(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        source_ranks=source_ranks,
        token_ordinals=token_ordinals,
        route_hashes=route_hashes,
        step=step,
        layer_seed=layer_seed,
    )
    if state_space is None:
        return (None, None) if return_route_tables else None
    route_states, state_sources, state_hashes, state_count = state_space
    num_slots = int(layout.numel())
    lhs_options, rhs_options, rhs_valid = planner._candidate_copy_options(layout, copy_slots, rows)

    def route_ranks(options: torch.Tensor) -> torch.Tensor:
        hashes = state_hashes.view(1, -1).expand(options.shape[0], -1)
        return planner._candidate_route_ranks(options, hashes, state_sources, num_slots)

    baseline_rank_by_state = route_ranks(copy_slots)
    lhs_new_rank_by_state = route_ranks(lhs_options)
    rhs_new_rank_by_state = route_ranks(rhs_options)
    routes_are_unique = bool(unique_routes.all().item())
    pair_records = _build_pair_state_records(
        selected,
        route_states,
        unique_routes,
        routes_are_unique=routes_are_unique,
        num_experts=num_experts,
        state_count=state_count,
    )
    pair_expansion = _build_pair_action_expansion(
        rows,
        pair_records,
        baseline_rank_by_state=baseline_rank_by_state,
        lhs_new_rank_by_state=lhs_new_rank_by_state,
        rhs_new_rank_by_state=rhs_new_rank_by_state,
        rhs_valid=rhs_valid,
        num_experts=num_experts,
        state_count=state_count,
    )

    route_ranks_current = torch.div(physical, planner.slots_per_rank, rounding_mode="floor")
    level_sizes = (1,) + tuple(
        int(size) for size in planner.hierarchy.group_sizes[: max(0, int(planner.hierarchy.selected_dim) - 1)]
    )
    lhs_experts = rows[:, 3]
    rhs_experts = rows[:, 4].clamp_min(0)
    packed_deltas: list[torch.Tensor] = []
    for size in level_sizes:
        groups = torch.div(route_ranks_current, size, rounding_mode="floor")
        num_groups = planner.ep_size // size
        occupancy = torch.zeros(
            (selected.shape[0], num_groups),
            dtype=torch.int32,
            device=selected.device,
        )
        occupancy.scatter_add_(1, groups, unique_routes.to(torch.int32))
        absence = _unary_absence_statistics(
            selected,
            route_states,
            unique_routes,
            groups,
            occupancy,
            routes_are_unique=routes_are_unique,
            num_experts=num_experts,
            state_count=state_count,
        )
        baseline_groups = torch.div(baseline_rank_by_state, size, rounding_mode="floor")
        lhs_old_groups = baseline_groups.index_select(0, lhs_experts)
        rhs_old_groups = baseline_groups.index_select(0, rhs_experts)
        lhs_new_groups = torch.div(lhs_new_rank_by_state, size, rounding_mode="floor")
        rhs_new_groups = torch.div(rhs_new_rank_by_state, size, rounding_mode="floor")
        delta = torch.zeros((rows.shape[0], num_groups), dtype=torch.float32, device=selected.device)
        _add_unary_deltas(delta, absence, lhs_experts, lhs_old_groups, lhs_new_groups)
        _add_unary_deltas(
            delta,
            absence,
            rhs_experts,
            rhs_old_groups,
            rhs_new_groups,
            valid=rhs_valid,
        )
        if pair_records is not None:
            pair_absence = _pair_absence_statistics(pair_records, occupancy, baseline_groups)
            _add_pair_deltas(delta, pair_absence, pair_expansion, group_size=size)
        packed_deltas.append(delta)
    deltas = torch.cat(packed_deltas, dim=1)
    return (deltas, None) if return_route_tables else deltas


__all__ = [
    "StatisticalPairContext",
    "StatisticalPrimitiveContext",
    "StatisticalPrimitiveResult",
    "StatisticalPrimitiveSpec",
    "StatisticalProxyResult",
    "StatisticalRouteTables",
    "UniformStatisticalBaseline",
    "build_statistical_primitive_spec",
    "statistical_candidate_local_deltas",
    "statistical_batched_selected_pair_local_deltas",
    "statistical_pair_interaction_bound_local",
    "statistical_primitive_fast_path_available",
    "statistical_primitive_selected_pair_context",
    "statistical_primitive_unary_local_deltas",
    "statistical_proxy_candidate_local_deltas",
    "statistical_selected_pair_local_deltas",
    "statistical_unary_candidate_local_deltas",
    "uniform_statistical_baseline_routes",
]
