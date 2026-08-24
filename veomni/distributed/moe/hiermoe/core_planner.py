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

"""Reference implementation of CoRe-MoE current-route placement planning."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .perf_model import HierMoEPerfModel, LinkCost
from .planner import (
    CurrentRoutePlanner,
    PlacementAction,
    PlacementCost,
    PlacementPlan,
    _route_hash,
    _SwapStats,
    _TensorCost,
)
from .topology import Hierarchy


CORE_MOE_ALGORITHM_VERSION = "core-moe-v2"
GatherFixed = Callable[[torch.Tensor], torch.Tensor]

_ROUTE_SUMMARY_SCHEMA_VERSION = 4
_FUSED_PLANNER_ABI_VERSION = 2
_FUSED_PROTOCOL_ABI_SHIFT = 16
_FUSED_PROTOCOL_CAPABILITY_MASK = (1 << _FUSED_PROTOCOL_ABI_SHIFT) - 1
_FUSED_CAP_COLLECTIVE = 1 << 0
_FUSED_CAP_SWAP_SELECT = 1 << 1
_FUSED_CAP_REPLICA_PROJECT = 1 << 2
_FUSED_CAP_REPLICA_MATCH = 1 << 3
_FUSED_CAP_QUOTA_MAP = 1 << 4
_FUSED_CAP_QUOTA_POLICY = 1 << 5


@dataclass(frozen=True)
class RouteSummary:
    """Globally replicated exact assignment rows and deterministic route samples."""

    token_counts: torch.Tensor
    assignment_counts: torch.Tensor
    sample_routes: torch.Tensor
    sample_ordinals: torch.Tensor
    sample_valid: torch.Tensor
    sample_weights: torch.Tensor
    sample_sources: torch.Tensor
    sample_digest: str
    sample_multiplicity: torch.Tensor | None = None
    sample_multiplicity_is_canonical: bool = False
    padded_sample_routes: torch.Tensor | None = None
    padded_sample_ordinals: torch.Tensor | None = None
    padded_sample_valid: torch.Tensor | None = None
    padded_sample_multiplicity: torch.Tensor | None = None


@dataclass(frozen=True)
class QuotaPolicyEntry:
    source_rank: int
    logical_expert: int
    destination_ranks: tuple[int, ...]
    quotas: tuple[int, ...]

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.source_rank,
            self.logical_expert,
            len(self.destination_ranks),
            *self.destination_ranks,
            *self.quotas,
        )

    @classmethod
    def from_tuple(cls, values: Sequence[int]) -> "QuotaPolicyEntry":
        row = tuple(int(value) for value in values)
        if len(row) < 3:
            raise ValueError("A quota policy row must contain source, expert, and destination count.")
        count = row[2]
        if count < 1 or len(row) != 3 + 2 * count:
            raise ValueError(f"Invalid quota policy row length: {row!r}.")
        return cls(
            source_rank=row[0],
            logical_expert=row[1],
            destination_ranks=row[3 : 3 + count],
            quotas=row[3 + count :],
        )


@dataclass(frozen=True)
class QuotaMapping:
    physical_slots: torch.Tensor
    policy: tuple[QuotaPolicyEntry, ...]


@dataclass(frozen=True)
class QuotaTensorTables:
    copy_slots: torch.Tensor
    copy_counts: torch.Tensor
    owner_ranks: torch.Tensor
    quota_weights: torch.Tensor
    quota_configured: torch.Tensor


@dataclass(frozen=True)
class _DevicePlanningSummary:
    route: RouteSummary
    swap_stats: _SwapStats
    fused_capabilities: torch.Tensor


@dataclass(frozen=True)
class _ScoredLayout:
    tensor_cost: _TensorCost
    cost: PlacementCost
    mapping: QuotaMapping


@dataclass(frozen=True)
class _DeviceQuotaPolicies:
    """Fixed-width local policy rows retained on device until the final layout is selected."""

    rows: torch.Tensor
    row_counts: torch.Tensor
    max_copies: int


def _sample_hash(
    token_ordinals: torch.Tensor,
    *,
    source_rank: int,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    value = token_ordinals.to(torch.int64)
    value = value * 1_000_003 + int(source_rank) * 65_537
    value = value + int(step) * 131 + int(layer_seed) * 17
    value = torch.remainder(value * 48_271 + 1, 2_147_483_647)
    return value


def _route_multiplicity(routes: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Count each unique token-logical route once and zero later duplicates."""

    if routes.ndim < 2:
        raise ValueError("routes must end in a top-k dimension.")
    top_k = int(routes.shape[-1])
    equal = routes.unsqueeze(-1) == routes.unsqueeze(-2)
    positions = torch.arange(top_k, dtype=torch.long, device=routes.device)
    prior = positions.view(*(1 for _ in routes.shape[:-1]), top_k, 1) > positions.view(
        *(1 for _ in routes.shape[:-1]), 1, top_k
    )
    first = ~(equal & prior).any(dim=-1)
    multiplicity = equal.sum(dim=-1).to(torch.long) * first.to(torch.long)
    if valid is not None:
        multiplicity *= valid.to(device=routes.device, dtype=torch.long).unsqueeze(-1)
    return multiplicity


def _deterministic_sample_indices(
    num_tokens: int,
    budget: int,
    *,
    source_rank: int,
    step: int,
    layer_seed: int,
    device: torch.device,
) -> torch.Tensor:
    sample_count = min(max(0, int(num_tokens)), max(1, int(budget)))
    if sample_count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    ordinals = torch.arange(num_tokens, dtype=torch.long, device=device)
    route_hash = _sample_hash(ordinals, source_rank=source_rank, step=step, layer_seed=layer_seed)
    # Sorting this composite key makes hash collisions deterministic as well.
    key = route_hash * max(1, num_tokens) + ordinals
    return key.topk(sample_count, largest=False, sorted=True).indices


def _pack_local_route_summary(
    selected_experts: torch.Tensor,
    *,
    num_experts: int,
    sample_size: int,
    source_rank: int,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if selected.ndim != 2:
        raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")
    if selected.numel() and (int(selected.min().item()) < 0 or int(selected.max().item()) >= int(num_experts)):
        raise ValueError("selected_experts contains a logical expert outside the configured range.")

    return compress_local_route_payload(
        selected,
        num_experts=num_experts,
        sample_size=sample_size,
        source_rank=source_rank,
        step=step,
        layer_seed=layer_seed,
    )


def compress_local_route_payload(
    selected_experts: torch.Tensor,
    *,
    num_experts: int,
    sample_size: int,
    source_rank: int,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    """Build the fixed route payload without leaving the accelerator."""

    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if selected.ndim != 2:
        raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")
    if num_experts <= 0 or sample_size <= 0:
        raise ValueError("num_experts and sample_size must be positive.")

    device = selected.device
    top_k = int(selected.shape[1])
    assignment = torch.bincount(selected.reshape(-1), minlength=num_experts).to(torch.long)
    sample_indices = _deterministic_sample_indices(
        selected.shape[0],
        sample_size,
        source_rank=source_rank,
        step=step,
        layer_seed=layer_seed,
        device=device,
    )
    sample_count = int(sample_indices.numel())
    sample_ordinals = torch.full((sample_size,), -1, dtype=torch.long, device=device)
    sample_valid = torch.zeros((sample_size,), dtype=torch.long, device=device)
    sample_routes = torch.full((sample_size, top_k), -1, dtype=torch.long, device=device)
    if sample_count:
        sample_ordinals[:sample_count] = sample_indices
        sample_valid[:sample_count] = 1
        sample_routes[:sample_count] = selected.index_select(0, sample_indices)
    return torch.cat(
        (
            torch.tensor([selected.shape[0], top_k], dtype=torch.long, device=device),
            assignment,
            sample_ordinals,
            sample_valid,
            sample_routes.reshape(-1),
        )
    )


def build_route_summary(
    selected_experts: torch.Tensor,
    *,
    num_experts: int,
    ep_size: int,
    sample_size: int,
    source_rank: int,
    step: int,
    layer_seed: int,
    gather_fixed: GatherFixed | None,
) -> RouteSummary:
    """Build one fixed-shape route payload and replicate it across the EP group."""

    local = _pack_local_route_summary(
        selected_experts,
        num_experts=num_experts,
        sample_size=sample_size,
        source_rank=source_rank,
        step=step,
        layer_seed=layer_seed,
    )
    if gather_fixed is None:
        if ep_size != 1:
            raise RuntimeError("CoRe-MoE route compression requires gather_fixed when ep_size > 1.")
        gathered = local.unsqueeze(0)
    else:
        gathered = gather_fixed(local)
        if gathered is None:
            gathered = local.unsqueeze(0)
    if gathered.ndim == 1:
        gathered = gathered.view(ep_size, -1)
    if tuple(gathered.shape) != (ep_size, local.numel()):
        raise ValueError(f"gather_fixed returned shape={tuple(gathered.shape)}, expected {(ep_size, local.numel())}.")

    top_k = int(selected_experts.shape[-1]) if selected_experts.ndim > 1 else 1
    offset = 0
    token_counts = gathered[:, offset]
    gathered_top_k = gathered[:, offset + 1]
    if bool((gathered_top_k != top_k).any().item()):
        raise ValueError("All CoRe-MoE ranks must use the same top-k route width.")
    offset += 2
    assignment = gathered[:, offset : offset + num_experts]
    offset += num_experts
    ordinals = gathered[:, offset : offset + sample_size]
    offset += sample_size
    valid = gathered[:, offset : offset + sample_size].to(torch.bool)
    offset += sample_size
    routes = gathered[:, offset:].view(ep_size, sample_size, top_k)
    multiplicity = _route_multiplicity(routes, valid)

    flat_valid = valid.reshape(-1)
    sample_routes = routes.reshape(-1, top_k)[flat_valid]
    sample_ordinals = ordinals.reshape(-1)[flat_valid]
    source_grid = torch.arange(ep_size, dtype=torch.long, device=gathered.device).view(-1, 1)
    sample_sources = source_grid.expand(-1, sample_size).reshape(-1)[flat_valid]
    per_rank_samples = valid.sum(dim=1).clamp_min(1)
    per_rank_weights = token_counts.to(torch.float64) / per_rank_samples.to(torch.float64)
    sample_weights = per_rank_weights.index_select(0, sample_sources)
    digest = hashlib.sha256(gathered.detach().cpu().numpy().tobytes()).hexdigest()
    return RouteSummary(
        token_counts=token_counts,
        assignment_counts=assignment,
        sample_routes=sample_routes,
        sample_ordinals=sample_ordinals,
        sample_valid=flat_valid,
        sample_weights=sample_weights,
        sample_sources=sample_sources,
        sample_digest=digest,
        sample_multiplicity=multiplicity.reshape(-1, top_k)[flat_valid],
        sample_multiplicity_is_canonical=True,
        padded_sample_routes=routes,
        padded_sample_ordinals=ordinals,
        padded_sample_valid=valid,
        padded_sample_multiplicity=multiplicity,
    )


def _copy_slots_by_logical(layout: Sequence[int], num_experts: int) -> tuple[tuple[int, ...], ...]:
    copies: list[list[int]] = [[] for _ in range(num_experts)]
    for slot, logical in enumerate(layout):
        if logical >= 0:
            if logical >= num_experts:
                raise ValueError(f"Layout slot {slot} contains invalid logical expert {logical}.")
            copies[logical].append(slot)
    if any(not slots for slots in copies):
        raise ValueError("Every logical expert must retain at least one physical copy.")
    return tuple(tuple(slots) for slots in copies)


def build_quota_tensor_tables(
    layouts: torch.Tensor,
    owner_slots: torch.Tensor,
    policies: Sequence[Sequence[QuotaPolicyEntry]],
    *,
    source_rank: int,
    slots_per_rank: int,
) -> QuotaTensorTables:
    """Convert two host quota policies into fixed device tables for the fused mapper."""

    if layouts.ndim != 2 or layouts.shape[0] != 2:
        raise ValueError("layouts must have shape [2, slots].")
    if owner_slots.ndim != 2 or owner_slots.shape[0] != 2:
        raise ValueError("owner_slots must have shape [2, experts].")
    if len(policies) != 2:
        raise ValueError("policies must contain current and candidate entries.")
    device = layouts.device
    layout_rows = layouts.detach().to(device="cpu", dtype=torch.long)
    owner_rows = owner_slots.detach().to(device="cpu", dtype=torch.long)
    num_experts = int(owner_rows.shape[1])
    copies_by_layout = [
        _copy_slots_by_logical(tuple(int(value) for value in row.tolist()), num_experts) for row in layout_rows
    ]
    max_copies = max(len(slots) for rows in copies_by_layout for slots in rows)
    if max_copies > 8:
        raise NotImplementedError("The fused quota mapper supports at most eight copies per logical expert.")
    copy_slots = torch.full((2, num_experts, max_copies), -1, dtype=torch.long)
    copy_counts = torch.empty((2, num_experts), dtype=torch.long)
    mask_count = 1 << max_copies
    quota_weights = torch.zeros((2, num_experts, mask_count, max_copies), dtype=torch.long)
    quota_configured = torch.zeros((2, num_experts, mask_count), dtype=torch.long)

    for layout_index, rows in enumerate(copies_by_layout):
        for logical, slots in enumerate(rows):
            copy_counts[layout_index, logical] = len(slots)
            copy_slots[layout_index, logical, : len(slots)] = torch.tensor(slots, dtype=torch.long)
            for mask in range(1, 1 << len(slots)):
                for copy_index in range(len(slots)):
                    if mask & (1 << copy_index):
                        quota_weights[layout_index, logical, mask, copy_index] = 1

        for entry in policies[layout_index]:
            if entry.source_rank != int(source_rank):
                continue
            logical = int(entry.logical_expert)
            slots = rows[logical]
            rank_to_copy = {int(slot) // int(slots_per_rank): index for index, slot in enumerate(slots)}
            if any(rank not in rank_to_copy for rank in entry.destination_ranks):
                raise ValueError(f"Quota policy references a rank without a copy: {entry!r}.")
            mask = sum(1 << rank_to_copy[rank] for rank in entry.destination_ranks)
            quota_weights[layout_index, logical, mask].zero_()
            quota_configured[layout_index, logical, mask] = 1
            for rank, quota in zip(entry.destination_ranks, entry.quotas, strict=True):
                quota_weights[layout_index, logical, mask, rank_to_copy[rank]] = max(0, int(quota))

    return QuotaTensorTables(
        copy_slots=copy_slots.to(device=device, non_blocking=True),
        copy_counts=copy_counts.to(device=device, non_blocking=True),
        owner_ranks=torch.div(owner_rows, int(slots_per_rank), rounding_mode="floor").to(
            device=device, non_blocking=True
        ),
        quota_weights=quota_weights.to(device=device, non_blocking=True),
        quota_configured=quota_configured.to(device=device, non_blocking=True),
    )


def _communication_class(
    destination_rank: int,
    *,
    source_rank: int,
    other_owner_ranks: Sequence[int],
    hierarchy: Hierarchy,
) -> tuple[int, ...]:
    visited = (int(source_rank), *(int(rank) for rank in other_owner_ranks))
    levels = tuple(int(size) for size in hierarchy.group_sizes[: max(0, hierarchy.selected_dim - 1)])
    values = [int(all(destination_rank // size != rank // size for rank in visited)) for size in reversed(levels)]
    values.append(int(destination_rank not in visited))
    return tuple(values)


def _waterfill_quota(loads: dict[int, float], destinations: Sequence[int], total: int) -> dict[int, int]:
    quota = {int(rank): 0 for rank in destinations}
    remaining = max(0, int(total))
    if remaining == 0 or not destinations:
        return quota
    ordered = sorted((float(loads[int(rank)]), int(rank)) for rank in destinations)
    active = 1
    while active < len(ordered):
        next_level = ordered[active][0]
        current_level = ordered[active - 1][0]
        required = max(0, math.ceil(next_level - current_level)) * active
        if required > remaining:
            break
        if required:
            increment, extra = divmod(required, active)
            for index in range(active):
                quota[ordered[index][1]] += increment + int(index < extra)
            remaining -= required
        active += 1
    increment, extra = divmod(remaining, active)
    for index in range(active):
        quota[ordered[index][1]] += increment + int(index < extra)
    return quota


def assign_tokens_to_copies_with_quota(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy: Hierarchy,
    owner_slots: torch.Tensor,
    token_ordinals: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
    quota_policy: Sequence[QuotaPolicyEntry] | None = None,
    step: int = 0,
    layer_seed: int = 0,
) -> QuotaMapping:
    """Map routes with communication-first, load-aware integer quotas.

    This eager implementation is deliberately scalar and serves as the oracle
    for the fused planner and dispatch kernels.
    """

    original_device = selected_experts.device
    selected = selected_experts.detach().to(device="cpu", dtype=torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    owners = owner_slots.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    num_tokens, top_k = selected.shape
    num_experts = int(owners.numel())
    if isinstance(source_ranks, int):
        sources = torch.full((num_tokens,), int(source_ranks), dtype=torch.long)
    else:
        sources = source_ranks.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if sources.numel() != num_tokens:
        raise ValueError(f"source_ranks has {sources.numel()} values for {num_tokens} tokens.")
    ordinals = (
        torch.arange(num_tokens, dtype=torch.long)
        if token_ordinals is None
        else token_ordinals.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    )
    weights = (
        torch.ones((num_tokens,), dtype=torch.float64)
        if token_weights is None
        else token_weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    )
    if ordinals.numel() != num_tokens or weights.numel() != num_tokens:
        raise ValueError("token_ordinals and token_weights must match the token count.")

    layout_values = [int(value) for value in layout.tolist()]
    copies = _copy_slots_by_logical(layout_values, num_experts)
    owner_ranks = [int(slot) // int(slots_per_rank) for slot in owners.tolist()]
    rank_slots: list[dict[int, int]] = []
    for logical_slots in copies:
        rank_slots.append({slot // int(slots_per_rank): slot for slot in logical_slots})
    route_hashes = _route_hash(
        selected,
        token_ordinals=ordinals,
        step=step,
        layer_seed=layer_seed,
    ).to(device="cpu")

    records: list[tuple[int, int, int, int, tuple[int, ...], float, int]] = []
    route_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for token in range(num_tokens):
        positions: dict[int, list[int]] = defaultdict(list)
        for position, logical in enumerate(selected[token].tolist()):
            positions[int(logical)].append(position)
        for logical, logical_positions in positions.items():
            multiplicity = len(logical_positions)
            other_owners = [owner_ranks[int(other)] for other in positions if int(other) != int(logical)]
            candidate_ranks = tuple(sorted(rank_slots[logical]))
            classes = {
                rank: _communication_class(
                    rank,
                    source_rank=int(sources[token]),
                    other_owner_ranks=other_owners,
                    hierarchy=hierarchy,
                )
                for rank in candidate_ranks
            }
            minimum = min(classes.values())
            eligible = tuple(rank for rank in candidate_ranks if classes[rank] == minimum)
            route_hash = int(route_hashes[token, logical_positions[0]])
            records.append(
                (
                    token,
                    logical,
                    int(sources[token]),
                    multiplicity,
                    eligible,
                    float(weights[token]),
                    route_hash,
                )
            )
            route_positions[(token, logical)] = logical_positions

    rank_loads = dict.fromkeys(range(hierarchy.ep_size), 0.0)
    chosen_rank: dict[tuple[int, int], int] = {}
    buckets: dict[tuple[int, int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        token, logical, source, multiplicity, eligible, weight, _ = record
        if len(eligible) == 1:
            rank = eligible[0]
            chosen_rank[(token, logical)] = rank
            rank_loads[rank] += multiplicity * weight
        else:
            buckets[(source, logical, eligible)].append(index)

    policy_lookup = {
        (entry.source_rank, entry.logical_expert, entry.destination_ranks): entry.quotas
        for entry in (quota_policy or ())
    }
    policy: list[QuotaPolicyEntry] = []
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            -sum(records[index][3] * records[index][5] for index in item[1]),
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    for (source, logical, destinations), indices in ordered_buckets:
        record_units = {index: float(records[index][3]) * float(records[index][5]) for index in indices}
        total = int(round(sum(record_units.values())))
        configured = policy_lookup.get((source, logical, destinations))
        if configured is None or sum(configured) <= 0:
            quotas = _waterfill_quota(rank_loads, destinations, total)
        else:
            raw = [total * int(value) / sum(configured) for value in configured]
            rounded = [math.floor(value) for value in raw]
            remainder = total - sum(rounded)
            order = sorted(
                range(len(destinations)), key=lambda index: (-(raw[index] - rounded[index]), destinations[index])
            )
            for index in order[:remainder]:
                rounded[index] += 1
            quotas = {rank: int(rounded[index]) for index, rank in enumerate(destinations)}
        ordered_indices = sorted(
            indices,
            key=lambda index: (
                records[index][6],
                int(ordinals[records[index][0]].item()),
            ),
        )
        assigned = dict.fromkeys(destinations, 0.0)
        if all(math.isclose(record_units[index], 1.0) for index in ordered_indices):
            # Unit-weight routes can realize the integer quota exactly.  Stable
            # hash ordering selects token identity, while cumulative intervals
            # consume each destination's quota instead of treating it as a
            # probabilistic hash weight.
            quota_total = sum(max(0, int(quotas[destination])) for destination in destinations)
            destination_index = 0
            consumed = 0
            for position, index in enumerate(ordered_indices):
                while destination_index + 1 < len(destinations) and position >= consumed + max(
                    0, int(quotas[destinations[destination_index]])
                ):
                    consumed += max(0, int(quotas[destinations[destination_index]]))
                    destination_index += 1
                rank = destinations[destination_index] if quota_total else destinations[position % len(destinations)]
                token = records[index][0]
                chosen_rank[(token, logical)] = rank
                assigned[rank] += 1.0
        else:
            # A repeated logical expert in top-k is one indivisible route with
            # multiplicity m.  Fill the largest quota deficit first, then the
            # lighter projected rank, with rank id as the deterministic tie.
            for index in ordered_indices:
                token = records[index][0]
                units = record_units[index]
                rank = min(
                    destinations,
                    key=lambda destination: (
                        assigned[destination] + units - float(quotas[destination]),
                        rank_loads[destination] + assigned[destination] + units,
                        destination,
                    ),
                )
                chosen_rank[(token, logical)] = rank
                assigned[rank] += units
        for destination in destinations:
            rank_loads[destination] += assigned[destination]
        policy.append(
            QuotaPolicyEntry(
                source_rank=source,
                logical_expert=logical,
                destination_ranks=tuple(destinations),
                quotas=tuple(int(quotas[rank]) for rank in destinations),
            )
        )

    physical = torch.empty_like(selected)
    for (token, logical), positions in route_positions.items():
        rank = chosen_rank[(token, logical)]
        slot = rank_slots[logical][rank]
        for position in positions:
            physical[token, position] = slot
    if selected_experts.ndim == 1:
        physical = physical.squeeze(-1)
    return QuotaMapping(
        physical_slots=physical.to(device=original_device, non_blocking=True),
        policy=tuple(policy),
    )


def _remap_replica_logical_from_baseline(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    baseline_physical_slots: torch.Tensor,
    *,
    logical_expert: int,
    slots_per_rank: int,
    source_ranks: torch.Tensor,
    hierarchy: Hierarchy,
    owner_slots: torch.Tensor,
    token_ordinals: torch.Tensor,
    token_weights: torch.Tensor,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    """Remap one logical expert while keeping every other baseline route fixed.

    Replica edges are scored independently.  Their water-filling load therefore
    starts from the realized baseline load with only the target expert removed;
    rescanning a whole candidate layout would also move unrelated routes and
    would not match the one-shot NPU projector.
    """

    original_device = selected_experts.device
    selected = selected_experts.detach().to(device="cpu", dtype=torch.long)
    baseline = baseline_physical_slots.detach().to(device="cpu", dtype=torch.long)
    squeezed = selected.ndim == 1
    if squeezed:
        selected = selected.unsqueeze(-1)
        baseline = baseline.unsqueeze(-1)
    if selected.ndim != 2 or baseline.shape != selected.shape:
        raise ValueError("The baseline physical routes must match the sampled logical routes.")

    logical = int(logical_expert)
    owners = owner_slots.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if logical < 0 or logical >= owners.numel():
        raise ValueError(f"Invalid replica edge logical expert {logical}.")
    layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    copies = _copy_slots_by_logical(tuple(int(value) for value in layout.tolist()), int(owners.numel()))
    rank_slots = {slot // int(slots_per_rank): slot for slot in copies[logical]}
    owner_ranks = [int(slot) // int(slots_per_rank) for slot in owners.tolist()]
    sources = source_ranks.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    ordinals = token_ordinals.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    weights = token_weights.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if sources.numel() != selected.shape[0] or ordinals.numel() != selected.shape[0]:
        raise ValueError("Replica edge sources and ordinals must match the sampled token count.")
    if weights.numel() != selected.shape[0]:
        raise ValueError("Replica edge weights must match the sampled token count.")

    route_hashes = _route_hash(
        selected,
        token_ordinals=ordinals,
        step=step,
        layer_seed=layer_seed,
    ).to(device="cpu")
    baseline_ranks = torch.div(baseline, int(slots_per_rank), rounding_mode="floor")
    rank_loads = {
        rank: float(weights.view(-1, 1).expand_as(baseline_ranks).masked_select(baseline_ranks == rank).sum().item())
        for rank in range(hierarchy.ep_size)
    }

    records: list[tuple[int, int, int, tuple[int, ...], float, int]] = []
    positions_by_token: dict[int, list[int]] = {}
    for token, row in enumerate(selected.tolist()):
        positions = [position for position, value in enumerate(row) if int(value) == logical]
        if not positions:
            continue
        positions_by_token[token] = positions
        multiplicity = len(positions)
        units = float(weights[token].item()) * multiplicity
        old_rank = int(baseline_ranks[token, positions[0]].item())
        rank_loads[old_rank] -= units
        other_logicals = {int(value) for value in row if int(value) != logical}
        other_owners = [owner_ranks[value] for value in other_logicals]
        classes = {
            rank: _communication_class(
                rank,
                source_rank=int(sources[token].item()),
                other_owner_ranks=other_owners,
                hierarchy=hierarchy,
            )
            for rank in sorted(rank_slots)
        }
        minimum = min(classes.values())
        eligible = tuple(rank for rank in sorted(rank_slots) if classes[rank] == minimum)
        records.append(
            (
                token,
                int(sources[token].item()),
                multiplicity,
                eligible,
                float(weights[token].item()),
                int(route_hashes[token, positions[0]].item()),
            )
        )

    chosen: dict[int, int] = {}
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, (token, source, multiplicity, eligible, weight, _route_hash_value) in enumerate(records):
        units = multiplicity * weight
        if len(eligible) == 1:
            rank = eligible[0]
            chosen[token] = rank
            rank_loads[rank] += units
        else:
            buckets[(source, eligible)].append(index)

    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            -sum(records[index][2] * records[index][4] for index in item[1]),
            item[0][0],
            logical,
            item[0][1],
        ),
    )
    for (_source, destinations), indices in ordered_buckets:
        record_units = {index: float(records[index][2]) * records[index][4] for index in indices}
        total = int(round(sum(record_units.values())))
        quotas = _waterfill_quota(rank_loads, destinations, total)
        ordered_indices = sorted(indices, key=lambda index: (records[index][5], int(ordinals[records[index][0]])))
        assigned = dict.fromkeys(destinations, 0.0)
        if all(0.999999 < record_units[index] < 1.000001 for index in ordered_indices):
            quota_total = sum(max(0, int(quotas[destination])) for destination in destinations)
            destination_index = 0
            consumed = 0
            for position, index in enumerate(ordered_indices):
                while destination_index + 1 < len(destinations) and position >= consumed + max(
                    0, int(quotas[destinations[destination_index]])
                ):
                    consumed += max(0, int(quotas[destinations[destination_index]]))
                    destination_index += 1
                rank = destinations[destination_index] if quota_total else destinations[position % len(destinations)]
                chosen[records[index][0]] = rank
                assigned[rank] += 1.0
        else:
            for index in ordered_indices:
                units = record_units[index]
                rank = min(
                    destinations,
                    key=lambda destination: (
                        assigned[destination] + units - float(quotas[destination]),
                        rank_loads[destination] + assigned[destination] + units,
                        destination,
                    ),
                )
                chosen[records[index][0]] = rank
                assigned[rank] += units
        for destination in destinations:
            rank_loads[destination] += assigned[destination]

    remapped = baseline.clone()
    for token, positions in positions_by_token.items():
        slot = rank_slots[chosen[token]]
        for position in positions:
            remapped[token, position] = slot
    if squeezed:
        remapped = remapped.squeeze(-1)
    return remapped.to(device=original_device, non_blocking=True)


def _stable_hungarian_maximize(weights: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return one deterministic column per row for a rectangular max-weight matching."""

    rows = len(weights)
    if rows == 0:
        return ()
    columns = len(weights[0])
    if columns < rows or any(len(row) != columns for row in weights):
        raise ValueError("Hungarian input must be rectangular with columns >= rows.")
    finite = [value for row in weights for value in row if math.isfinite(value)]
    largest = max(finite, default=0.0)
    invalid = largest + 1e30
    cost = [[largest - value if math.isfinite(value) else invalid for value in row] for row in weights]
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    parent = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    parent[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = parent[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    result = [-1] * rows
    for column in range(1, columns + 1):
        if matched_row[column]:
            result[matched_row[column] - 1] = column - 1
    return tuple(result)


class CoReMoEPlanner(CurrentRoutePlanner):
    """Current-route swap-then-replica planner with an exact final guard."""

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
        backward_compute_per_assignment: float | None = None,
        reducer: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        gather_fixed: GatherFixed | None = None,
        collective_backend: str | None = None,
        route_sample_size: int = 1024,
        expert_state_bytes: torch.Tensor | Sequence[int] | None = None,
        expert_gradient_bytes: torch.Tensor | Sequence[int] | None = None,
        verify_collective_digest: bool = False,
        record_device_timing: bool = False,
    ) -> None:
        super().__init__(
            hierarchy=hierarchy,
            perf_model=perf_model,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            slots_per_rank=slots_per_rank,
            communication_scale=communication_scale,
            forward_compute_per_assignment=forward_compute_per_assignment,
            reducer=reducer,
            record_device_timing=record_device_timing,
        )
        if route_sample_size <= 0:
            raise ValueError("route_sample_size must be positive.")
        self.gather_fixed = gather_fixed
        backend = None if collective_backend is None else str(collective_backend).lower()
        self.collective_backend = None if backend is None else backend.rsplit(".", maxsplit=1)[-1]
        self.route_sample_size = int(route_sample_size)
        self.backward_compute_per_assignment = (
            2.0 * float(forward_compute_per_assignment)
            if backward_compute_per_assignment is None
            else float(backward_compute_per_assignment)
        )
        self.expert_state_bytes = self._normalize_expert_bytes(expert_state_bytes)
        self.expert_gradient_bytes = self._normalize_expert_bytes(expert_gradient_bytes)
        self.verify_collective_digest = bool(verify_collective_digest)

    @staticmethod
    def _normalize_expert_bytes(values: torch.Tensor | Sequence[int] | None) -> tuple[int, ...]:
        if values is None:
            return ()
        if torch.is_tensor(values):
            rows = values.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        else:
            rows = list(values)
        return tuple(max(0, int(value)) for value in rows)

    def _expert_bytes(self, logical: int, *, gradient: bool) -> int:
        values = self.expert_gradient_bytes if gradient else self.expert_state_bytes
        if not values:
            return 0
        if logical < 0 or logical >= len(values):
            raise ValueError(f"Missing byte accounting for logical expert {logical}.")
        return values[logical]

    def _build_device_planning_summary(
        self,
        selected: torch.Tensor,
        owner_slots: torch.Tensor,
        *,
        source_rank: int,
        step: int,
        layer_seed: int,
        fused_capable: int | bool | torch.Tensor,
        required_fused_capabilities: int = (_FUSED_CAP_COLLECTIVE | _FUSED_CAP_QUOTA_MAP | _FUSED_CAP_QUOTA_POLICY),
    ) -> _DevicePlanningSummary:
        """Build the integer first-collective payload and sampled swap statistics."""

        if self.gather_fixed is None and self.ep_size != 1:
            raise RuntimeError("The fused CoRe-MoE planner requires gather_fixed when ep_size > 1.")
        num_tokens, top_k = selected.shape
        num_experts = int(owner_slots.numel())
        sample_count = min(num_tokens, self.route_sample_size)
        sample_indices = _deterministic_sample_indices(
            num_tokens,
            self.route_sample_size,
            source_rank=source_rank,
            step=step,
            layer_seed=layer_seed,
            device=selected.device,
        )
        sample_routes = torch.full((self.route_sample_size, top_k), -1, dtype=torch.long, device=selected.device)
        sample_ordinals = torch.full((self.route_sample_size,), -1, dtype=torch.long, device=selected.device)
        sample_valid = torch.zeros((self.route_sample_size,), dtype=torch.long, device=selected.device)
        if sample_count:
            sample_routes[:sample_count] = selected.index_select(0, sample_indices)
            sample_ordinals[:sample_count] = sample_indices
            sample_valid[:sample_count] = 1
        sample_multiplicity = _route_multiplicity(sample_routes, sample_valid.to(torch.bool))

        assignment = torch.bincount(selected.reshape(-1), minlength=num_experts).to(torch.long)
        safe_sample_routes = sample_routes.clamp_min(0)
        token_hits = torch.zeros(
            (self.route_sample_size, num_experts),
            dtype=torch.float32,
            device=selected.device,
        )
        token_hits.scatter_(
            1,
            safe_sample_routes,
            sample_valid.to(torch.float32).view(-1, 1).expand_as(safe_sample_routes),
        )
        owner_ranks, local_parts, shapes, _ = self._local_swap_parts(token_hits, owner_slots)
        local_expert_tokens = token_hits.sum(dim=0).to(torch.long)
        required_capabilities = int(required_fused_capabilities)
        if isinstance(fused_capable, bool):
            local_capabilities = required_capabilities if fused_capable else 0
        elif torch.is_tensor(fused_capable) and fused_capable.dtype == torch.bool:
            local_capabilities = torch.where(
                fused_capable,
                torch.as_tensor(required_capabilities, dtype=torch.long, device=selected.device),
                torch.zeros((), dtype=torch.long, device=selected.device),
            )
        else:
            local_capabilities = torch.as_tensor(fused_capable, dtype=torch.long, device=selected.device)
        protocol_word = local_capabilities | (_FUSED_PLANNER_ABI_VERSION << _FUSED_PROTOCOL_ABI_SHIFT)
        header = torch.tensor(
            (_ROUTE_SUMMARY_SCHEMA_VERSION, 0, num_tokens, top_k, sample_count),
            dtype=torch.long,
            device=selected.device,
        )
        header[1] = protocol_word
        local_payload = torch.cat(
            (
                header,
                assignment,
                sample_ordinals,
                sample_valid,
                sample_routes.reshape(-1),
                sample_multiplicity.reshape(-1),
                local_expert_tokens,
                *(part.to(torch.long) for part in local_parts),
            )
        ).contiguous()
        gathered = local_payload.unsqueeze(0) if self.gather_fixed is None else self.gather_fixed(local_payload)
        if gathered is None:
            gathered = local_payload.unsqueeze(0)
        if gathered.ndim == 1:
            gathered = gathered.view(self.ep_size, -1)
        if tuple(gathered.shape) != (self.ep_size, local_payload.numel()):
            raise ValueError(
                f"gather_fixed returned shape={tuple(gathered.shape)}, "
                f"expected {(self.ep_size, local_payload.numel())}."
            )

        offset = 0
        schema_versions = gathered[:, offset]
        protocol_words = gathered[:, offset + 1]
        abi_versions = torch.bitwise_right_shift(protocol_words, _FUSED_PROTOCOL_ABI_SHIFT)
        capability_masks = torch.bitwise_and(protocol_words, _FUSED_PROTOCOL_CAPABILITY_MASK)
        token_counts = gathered[:, offset + 2]
        gathered_top_k = gathered[:, offset + 3]
        sample_counts = gathered[:, offset + 4]
        capability_consistent = capability_masks.eq(capability_masks[:1]).all()
        fused_capabilities = schema_versions.eq(_ROUTE_SUMMARY_SCHEMA_VERSION)
        fused_capabilities &= abi_versions.eq(_FUSED_PLANNER_ABI_VERSION)
        fused_capabilities &= gathered_top_k.eq(top_k)
        fused_capabilities &= capability_masks.bitwise_and(required_capabilities).eq(required_capabilities)
        fused_capabilities &= capability_consistent
        offset += 5
        assignment_by_source = gathered[:, offset : offset + num_experts]
        offset += num_experts
        ordinals = gathered[:, offset : offset + self.route_sample_size].to(torch.long)
        offset += self.route_sample_size
        valid = gathered[:, offset : offset + self.route_sample_size].to(torch.bool)
        offset += self.route_sample_size
        routes = gathered[:, offset : offset + self.route_sample_size * top_k]
        routes = routes.view(self.ep_size, self.route_sample_size, top_k).to(torch.long)
        offset += self.route_sample_size * top_k
        multiplicity = gathered[:, offset : offset + self.route_sample_size * top_k]
        multiplicity = multiplicity.view(self.ep_size, self.route_sample_size, top_k).to(torch.long)
        offset += self.route_sample_size * top_k
        sample_weights_by_rank = token_counts.to(torch.float32) / sample_counts.clamp_min(1).to(torch.float32)
        expert_tokens = (
            gathered[:, offset : offset + num_experts].to(torch.float32) * sample_weights_by_rank.unsqueeze(1)
        ).sum(dim=0)
        offset += num_experts
        reduced_parts = (gathered[:, offset:].to(torch.float32) * sample_weights_by_rank.unsqueeze(1)).sum(dim=0)
        swap_stats = self._unpack_swap_stats(
            owner_ranks,
            expert_tokens,
            assignment_by_source.to(torch.float32).sum(dim=0),
            reduced_parts,
            shapes,
            (),
        )
        flat_valid = valid.reshape(-1)
        source_grid = torch.arange(self.ep_size, dtype=torch.long, device=selected.device).view(-1, 1)
        fixed_sources = source_grid.expand(-1, self.route_sample_size).reshape(-1)
        per_rank_weights = token_counts.to(torch.float32) / valid.sum(dim=1).clamp_min(1).to(torch.float32)
        fixed_weights = per_rank_weights.view(-1, 1).expand_as(valid)
        return _DevicePlanningSummary(
            route=RouteSummary(
                token_counts=token_counts,
                assignment_counts=assignment_by_source,
                sample_routes=routes.reshape(-1, top_k)[flat_valid],
                sample_ordinals=ordinals.reshape(-1)[flat_valid],
                sample_valid=flat_valid,
                sample_weights=fixed_weights.reshape(-1)[flat_valid],
                sample_sources=fixed_sources[flat_valid],
                sample_digest="device-pending",
                sample_multiplicity=multiplicity.reshape(-1, top_k)[flat_valid],
                sample_multiplicity_is_canonical=True,
                padded_sample_routes=routes,
                padded_sample_ordinals=ordinals,
                padded_sample_valid=valid,
                padded_sample_multiplicity=multiplicity,
            ),
            swap_stats=swap_stats,
            fused_capabilities=fused_capabilities,
        )

    @staticmethod
    def _fused_planner_extension(device: torch.device):
        if device.type not in {"npu", "privateuseone"}:
            return None
        try:
            from ....ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops
        except ImportError:
            return None
        return get_hiermoe_planner_npu_ops()

    @classmethod
    def _fused_swap_extension(cls, device: torch.device):
        extension = cls._fused_planner_extension(device)
        return extension if extension is not None and hasattr(extension, "swap_select_with_stats") else None

    @staticmethod
    def _fused_extension_capabilities(extension) -> int:
        if extension is None:
            return 0
        capabilities = 0
        for name, capability in (
            ("swap_select_with_stats", _FUSED_CAP_SWAP_SELECT),
            ("replica_project", _FUSED_CAP_REPLICA_PROJECT),
            ("replica_match", _FUSED_CAP_REPLICA_MATCH),
            ("quota_map", _FUSED_CAP_QUOTA_MAP),
            ("quota_policy", _FUSED_CAP_QUOTA_POLICY),
        ):
            if hasattr(extension, name):
                capabilities |= capability
        return capabilities

    @staticmethod
    def _required_fused_capabilities(*, max_swaps: int, max_replicas: int) -> int:
        capabilities = _FUSED_CAP_COLLECTIVE | _FUSED_CAP_QUOTA_MAP | _FUSED_CAP_QUOTA_POLICY
        if max_swaps > 0 or max_replicas > 0:
            capabilities |= _FUSED_CAP_SWAP_SELECT
        if max_replicas > 0:
            capabilities |= _FUSED_CAP_REPLICA_PROJECT | _FUSED_CAP_REPLICA_MATCH
        return capabilities

    def _local_fused_capabilities(self, extension) -> int:
        capabilities = self._fused_extension_capabilities(extension)
        # A missing backend is allowed for single-rank and injected test collectives. The
        # production manager always supplies the actual EP backend.
        if self.ep_size <= 1 or self.collective_backend in {None, "hccl"}:
            capabilities |= _FUSED_CAP_COLLECTIVE
        return capabilities

    def _fused_cost_arguments(self) -> tuple[int | float | bool, ...]:
        link0 = self.perf_model.inter[0]
        link1 = self.perf_model.inter[min(1, len(self.perf_model.inter) - 1)]
        state = self.perf_model.resolved_state_move()
        gradient = self.perf_model.resolved_gradient_sync()
        runtime_scale = 1.0 if self.perf_model.runtime_cost_status == "complete" else self.communication_scale
        return (
            self.hidden_size * self.bytes_per_element,
            self.communication_scale,
            self.forward_compute_per_assignment + self.backward_compute_per_assignment,
            self.perf_model.a2a.alpha,
            self.perf_model.a2a.beta,
            link0.alpha,
            link0.beta,
            link1.alpha,
            link1.beta,
            self.perf_model.intra.alpha,
            self.perf_model.intra.beta,
            state.intra.alpha,
            state.intra.beta,
            state.inter.alpha,
            state.inter.beta,
            gradient.gather.intra.alpha,
            gradient.gather.intra.beta,
            gradient.gather.inter.alpha,
            gradient.gather.inter.beta,
            gradient.scatter.intra.alpha,
            gradient.scatter.intra.beta,
            gradient.scatter.inter.alpha,
            gradient.scatter.inter.beta,
            runtime_scale,
            self.perf_model.source != "default",
        )

    def _fused_swap_select(
        self,
        stats: _SwapStats,
        layout: torch.Tensor,
        owners: torch.Tensor,
        *,
        max_swaps: int,
        sample_routes: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        extension=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        extension = self._fused_swap_extension(layout.device) if extension is None else extension
        if extension is None:
            return None
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        padded_levels = (*level_sizes, 1, 1)[:3]
        state_bytes = torch.tensor(
            self.expert_state_bytes or (0,) * int(owners.numel()),
            dtype=torch.long,
            device=layout.device,
        )
        gradient_bytes = torch.tensor(
            self.expert_gradient_bytes or (0,) * int(owners.numel()),
            dtype=torch.long,
            device=layout.device,
        )
        if sample_routes is None or sample_weights is None:
            if max_swaps > 1:
                return None
            sample_routes = torch.empty((0, 1), dtype=torch.long, device=layout.device)
            sample_weights = torch.empty((0,), dtype=torch.float32, device=layout.device)
        return extension.swap_select_with_stats(
            stats.expert_token_counts.to(torch.float32).contiguous(),
            stats.expert_assignment_counts.to(torch.float32).contiguous(),
            torch.cat(stats.base_counts).to(torch.float32).contiguous(),
            torch.cat(stats.expert_group_counts, dim=1).to(torch.float32).contiguous(),
            torch.stack(stats.sole_expert_counts, dim=1).to(torch.float32).contiguous(),
            torch.stack(stats.sole_pair_counts, dim=1).to(torch.float32).contiguous(),
            sample_routes.to(dtype=torch.long, device=layout.device).contiguous(),
            sample_weights.to(dtype=torch.float32, device=layout.device).contiguous(),
            layout.contiguous(),
            owners.contiguous(),
            state_bytes,
            gradient_bytes,
            int(max_swaps),
            self.slots_per_rank,
            self.ep_size,
            max(1, int(self.hierarchy.local_world_size)),
            len(level_sizes),
            padded_levels[0],
            padded_levels[1],
            padded_levels[2],
            *self._fused_cost_arguments(),
        )

    def _fused_redundant_slots(self, owner_slots: torch.Tensor) -> torch.Tensor:
        num_experts = int(owner_slots.numel())
        if num_experts % self.ep_size:
            raise RuntimeError("The fused replica planner requires an equal number of owner slots on every EP rank.")
        redundant_per_rank = self.slots_per_rank - num_experts // self.ep_size
        if redundant_per_rank <= 0 or redundant_per_rank > 8:
            raise RuntimeError("The fused replica planner requires one through eight redundant slots per EP rank.")
        slot_ids = torch.arange(
            self.ep_size * self.slots_per_rank,
            dtype=torch.long,
            device=owner_slots.device,
        )
        owner_mask = torch.zeros_like(slot_ids, dtype=torch.bool)
        owner_mask.scatter_(0, owner_slots, True)
        rank_slots = slot_ids.view(self.ep_size, self.slots_per_rank)
        candidates = torch.where(
            owner_mask.view_as(rank_slots),
            torch.full_like(rank_slots, slot_ids.numel()),
            rank_slots,
        )
        return candidates.sort(dim=1).values[:, :redundant_per_rank].contiguous()

    def _fused_replica_candidates(
        self,
        layout: torch.Tensor,
        swap_metadata: torch.Tensor,
        *,
        num_experts: int,
    ) -> torch.Tensor:
        slot_ranks = torch.div(
            torch.arange(layout.numel(), dtype=torch.long, device=layout.device),
            self.slots_per_rank,
            rounding_mode="floor",
        )
        bottleneck_ranks = swap_metadata[1:3].to(dtype=torch.long, device=layout.device)
        hot_slots = (slot_ranks.unsqueeze(1) == bottleneck_ranks.view(1, -1)).any(dim=1)
        valid = hot_slots & layout.ge(0) & layout.lt(num_experts)
        safe_logical = torch.where(valid, layout, torch.zeros_like(layout))
        counts = torch.zeros((num_experts,), dtype=torch.int32, device=layout.device)
        counts.scatter_add_(0, safe_logical, valid.to(torch.int32))
        return counts.gt(0).to(torch.int32).contiguous()

    def _fused_plan_replicas(
        self,
        extension,
        summary: RouteSummary,
        layout: torch.Tensor,
        owners: torch.Tensor,
        swap_metadata: torch.Tensor,
        seed_base_counts: torch.Tensor | None,
        *,
        max_replicas: int,
        step: int,
        layer_seed: int,
    ) -> tuple[torch.Tensor, list[PlacementAction], int]:
        if max_replicas <= 0:
            return layout, [], 0
        if seed_base_counts is None:
            raise RuntimeError("The fused replica planner requires post-swap sampled base counts.")
        if summary.sample_multiplicity is None:
            raise RuntimeError("The fused replica planner requires sampled route multiplicities.")
        if not summary.sample_multiplicity_is_canonical:
            raise RuntimeError("The fused replica planner requires canonical sampled route multiplicities.")
        redundant_slots = self._fused_redundant_slots(owners)
        candidate_experts = self._fused_replica_candidates(
            layout,
            swap_metadata,
            num_experts=int(owners.numel()),
        )
        level_sizes = tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]
        ) + (1,)
        padded_levels = (*level_sizes, 1, 1)[:3]
        projection = extension.replica_project(
            summary.sample_routes.to(dtype=torch.long, device=layout.device).contiguous(),
            summary.sample_multiplicity.to(dtype=torch.long, device=layout.device).contiguous(),
            summary.sample_weights.to(dtype=torch.float32, device=layout.device).contiguous(),
            summary.sample_sources.to(dtype=torch.long, device=layout.device).contiguous(),
            summary.sample_ordinals.to(dtype=torch.long, device=layout.device).contiguous(),
            summary.assignment_counts.to(dtype=torch.long, device=layout.device).contiguous(),
            seed_base_counts.to(dtype=torch.float32, device=layout.device).contiguous(),
            layout.contiguous(),
            owners.contiguous(),
            redundant_slots,
            candidate_experts,
            self.slots_per_rank,
            self.ep_size,
            len(level_sizes),
            padded_levels[0],
            padded_levels[1],
            padded_levels[2],
            int(step),
            int(layer_seed),
        )
        if not isinstance(projection, tuple) or len(projection) != 6:
            raise RuntimeError("The fused replica projector returned an invalid result.")
        state_bytes = torch.tensor(
            self.expert_state_bytes or (0,) * int(owners.numel()),
            dtype=torch.long,
            device=layout.device,
        )
        gradient_bytes = torch.tensor(
            self.expert_gradient_bytes or (0,) * int(owners.numel()),
            dtype=torch.long,
            device=layout.device,
        )
        capacity = min(max(0, int(max_replicas)), int(redundant_slots.numel()))
        matched = extension.replica_match(
            *(tensor.contiguous() for tensor in projection),
            layout.contiguous(),
            owners.contiguous(),
            redundant_slots,
            candidate_experts,
            state_bytes,
            gradient_bytes,
            capacity,
            self.slots_per_rank,
            self.ep_size,
            max(1, int(self.hierarchy.local_world_size)),
            len(level_sizes),
            padded_levels[0],
            padded_levels[1],
            padded_levels[2],
            *self._fused_cost_arguments(),
        )
        if not isinstance(matched, tuple) or len(matched) != 6:
            raise RuntimeError("The fused replica matcher returned an invalid result.")
        updated_layout, action_rows, _gains, _selected_columns, _matrix, metadata = matched
        accepted = int(metadata.reshape(-1)[0].item())
        if accepted < 0 or accepted > min(capacity, int(action_rows.shape[0])):
            raise RuntimeError(f"Fused CoRe-MoE replica matcher returned invalid action count {accepted}.")
        actions: list[PlacementAction] = []
        for kind, source, destination, logical, previous in action_rows[:accepted].detach().cpu().tolist():
            if kind == 1:
                actions.append(PlacementAction("empty", int(destination), int(destination), int(previous), -1))
            elif kind == 2:
                actions.append(PlacementAction("replica", int(source), int(destination), int(logical), int(previous)))
            else:
                raise RuntimeError(f"Fused CoRe-MoE replica matcher returned invalid action kind {kind}.")
        return updated_layout, actions, len(actions)

    def _build_quota_tables(
        self,
        layouts: torch.Tensor,
        owner_slots: torch.Tensor,
        quota_weights: torch.Tensor,
        quota_configured: torch.Tensor,
    ) -> QuotaTensorTables:
        """Build copy tables and attach the sampled policy emitted by the NPU planner op."""

        if layouts.ndim != 2 or tuple(layouts.shape[:1]) != (2,):
            raise ValueError("layouts must have shape [2, slots].")
        if owner_slots.ndim != 2 or tuple(owner_slots.shape[:1]) != (2,):
            raise ValueError("owner_slots must have shape [2, experts].")
        num_experts = int(owner_slots.shape[1])
        if quota_weights.ndim != 4 or tuple(quota_weights.shape[:2]) != (2, num_experts):
            raise ValueError("quota_weights must have shape [2, experts, masks, copies].")
        max_copies = int(quota_weights.shape[-1])
        if not 0 < max_copies <= 8 or int(quota_weights.shape[2]) != 1 << max_copies:
            raise ValueError("quota_weights has an invalid fixed-copy ABI.")
        if tuple(quota_configured.shape) != (2, num_experts, 1 << max_copies):
            raise ValueError("quota_configured must match quota_weights masks.")
        logical_ids = torch.arange(num_experts, dtype=torch.long, device=layouts.device).view(1, 1, -1)
        matches = layouts.unsqueeze(-1).eq(logical_ids)
        copy_counts = matches.sum(dim=1).to(torch.long)
        slot_ids = torch.arange(layouts.shape[1], dtype=torch.long, device=layouts.device).view(1, -1, 1)
        ordered = torch.where(matches, slot_ids, torch.full_like(slot_ids, layouts.shape[1])).sort(dim=1).values
        copy_slots = ordered[:, :max_copies].transpose(1, 2).contiguous()
        valid_copies = torch.arange(max_copies, dtype=torch.long, device=layouts.device).view(1, 1, -1)
        copy_slots = torch.where(valid_copies < copy_counts.unsqueeze(-1), copy_slots, -torch.ones_like(copy_slots))
        return QuotaTensorTables(
            copy_slots=copy_slots,
            copy_counts=copy_counts.contiguous(),
            owner_ranks=torch.div(owner_slots, self.slots_per_rank, rounding_mode="floor").contiguous(),
            quota_weights=quota_weights.contiguous(),
            quota_configured=quota_configured.contiguous(),
        )

    @staticmethod
    def _quota_policy_from_device_rows(
        policies: _DeviceQuotaPolicies,
        layout_index: int,
        *,
        source_rank: int,
    ) -> tuple[QuotaPolicyEntry, ...]:
        """Materialize compact rows for only the layout selected by the strict final comparison."""

        if tuple(policies.rows.shape[:1]) != (2,) or tuple(policies.row_counts.shape) != (2,):
            raise RuntimeError("The fused quota-policy operator returned invalid compact row shapes.")
        count = int(policies.row_counts[layout_index].item())
        if count < 0 or count > int(policies.rows.shape[1]):
            raise RuntimeError(f"The fused quota-policy operator returned invalid row count {count}.")
        result: list[QuotaPolicyEntry] = []
        for fixed_row in policies.rows[layout_index, :count].detach().to(device="cpu", dtype=torch.long).tolist():
            destination_count = int(fixed_row[2])
            if destination_count < 1 or destination_count > policies.max_copies:
                raise RuntimeError(
                    f"The fused quota-policy operator returned invalid destination count {destination_count}."
                )
            entry = QuotaPolicyEntry(
                source_rank=int(fixed_row[0]),
                logical_expert=int(fixed_row[1]),
                destination_ranks=tuple(int(value) for value in fixed_row[3 : 3 + destination_count]),
                quotas=tuple(
                    int(value)
                    for value in fixed_row[3 + policies.max_copies : 3 + policies.max_copies + destination_count]
                ),
            )
            if entry.source_rank != int(source_rank):
                raise RuntimeError("The fused quota-policy operator returned a row owned by another source rank.")
            result.append(entry)
        return tuple(result)

    def _fused_score_exact_pair(
        self,
        extension,
        summary: RouteSummary,
        selected: torch.Tensor,
        token_ordinals: torch.Tensor,
        source_rank: int,
        current_layout: torch.Tensor,
        current_owners: torch.Tensor,
        candidate_layout: torch.Tensor,
        candidate_owners: torch.Tensor,
        candidate_actions: Sequence[PlacementAction],
        *,
        step: int,
        layer_seed: int,
    ) -> tuple[_ScoredLayout, _ScoredLayout, _DeviceQuotaPolicies]:
        layouts = torch.stack((current_layout, candidate_layout), dim=0).contiguous()
        owner_rows = torch.stack((current_owners, candidate_owners), dim=0).contiguous()
        levels = tuple(int(size) for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)])
        padded_levels = (*levels, 1, 1)[:2]
        max_copies = min(8, int(layouts.shape[1]))
        if summary.sample_multiplicity is None:
            raise RuntimeError("The fused quota-policy operator requires sampled route multiplicities.")
        policy_result = extension.quota_policy(
            summary.sample_routes.to(dtype=torch.long, device=layouts.device).contiguous(),
            summary.sample_multiplicity.to(dtype=torch.long, device=layouts.device).contiguous(),
            summary.sample_sources.to(dtype=torch.long, device=layouts.device).contiguous(),
            summary.sample_ordinals.to(dtype=torch.long, device=layouts.device).contiguous(),
            summary.assignment_counts.to(dtype=torch.long, device=layouts.device).contiguous(),
            layouts,
            owner_rows,
            self.slots_per_rank,
            int(source_rank),
            self.ep_size,
            max_copies,
            self.route_sample_size,
            len(levels),
            padded_levels[0],
            padded_levels[1],
        )
        if not isinstance(policy_result, tuple) or len(policy_result) != 5:
            raise RuntimeError("The fused quota-policy operator returned an invalid result.")
        quota_weights, quota_configured, compact_rows, row_counts, policy_digest = policy_result
        num_experts = int(owner_rows.shape[1])
        expected_policy_shapes = (
            (2, num_experts, 1 << max_copies, max_copies),
            (2, num_experts, 1 << max_copies),
            (2, self.route_sample_size * int(selected.shape[1]), 3 + 2 * max_copies),
            (2,),
            (2, 2),
        )
        for tensor, expected_shape in zip(policy_result, expected_policy_shapes, strict=True):
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    "The fused quota-policy operator returned an invalid shape: "
                    f"expected {expected_shape}, got {tuple(tensor.shape)}."
                )
        tables = self._build_quota_tables(layouts, owner_rows, quota_weights, quota_configured)
        mapped = extension.quota_map(
            selected.contiguous(),
            tables.copy_slots,
            tables.copy_counts,
            tables.owner_ranks,
            tables.quota_weights,
            tables.quota_configured,
            token_ordinals.contiguous(),
            self.slots_per_rank,
            int(source_rank),
            self.ep_size,
            len(levels),
            padded_levels[0],
            padded_levels[1],
            int(step),
            int(layer_seed),
        )
        if not isinstance(mapped, tuple) or len(mapped) != 3:
            raise RuntimeError("The fused quota mapper returned an invalid result.")
        physical_routes, group_counts, assignment_counts = mapped
        expected_group_width = sum(self.ep_size // size for size in (*levels, 1))
        if tuple(physical_routes.shape) != (2, selected.shape[0], selected.shape[1]):
            raise RuntimeError("The fused quota mapper returned physical routes with an invalid shape.")
        if tuple(group_counts.shape) != (2, expected_group_width):
            raise RuntimeError("The fused quota mapper returned group counts with an invalid shape.")
        if tuple(assignment_counts.shape) != (2, self.ep_size):
            raise RuntimeError("The fused quota mapper returned assignment counts with an invalid shape.")

        reduced = torch.cat((group_counts, assignment_counts), dim=1)
        digest_width = 0
        if self.verify_collective_digest:
            kind_codes = {"swap": 1, "replica": 2, "empty": 3}
            action_values = [
                value
                for action in candidate_actions
                for value in (
                    kind_codes[action.kind],
                    action.src_slot,
                    action.dst_slot,
                    action.src_logical,
                    action.dst_logical,
                )
            ]
            action_signature = torch.tensor(action_values, dtype=torch.long, device=layouts.device)
            action_positions = torch.arange(1, action_signature.numel() + 1, dtype=torch.long, device=layouts.device)
            action_first = torch.remainder(
                ((action_signature + 2) * (action_positions * 17 + 3)).sum() + 11,
                31,
            )
            action_second = torch.remainder(
                ((action_signature + 5) * (action_positions * action_positions + 29)).sum() + 37,
                29,
            )
            action_first_rows = torch.stack((torch.zeros_like(action_first), action_first))
            action_second_rows = torch.stack((torch.zeros_like(action_second), action_second))
            first = torch.remainder(policy_digest[:, 0] + action_first_rows, 31)
            second = torch.remainder(policy_digest[:, 1] + action_second_rows, 29)
            valid = policy_digest.ge(0).all(dim=1)
            digest = torch.stack((first, first * first, second, second * second, valid.to(first.dtype)), dim=1).to(
                reduced.dtype
            )
            reduced = torch.cat((reduced, digest), dim=1)
            digest_width = int(digest.shape[1])
        if self.reducer is not None:
            result = self.reducer(reduced)
            if result is not None:
                reduced = result
        if digest_width:
            digest_sum = reduced[:, -digest_width:]
            world = float(self.ep_size)
            consistent = (
                (digest_sum[:, 1] * world == digest_sum[:, 0] * digest_sum[:, 0])
                .logical_and(digest_sum[:, 3] * world == digest_sum[:, 2] * digest_sum[:, 2])
                .logical_and(digest_sum[:, 4] == world)
                .all()
            )
            if not bool(consistent.item()):
                raise RuntimeError("CoRe-MoE ranks produced invalid or inconsistent placement plans before migration.")
            reduced = reduced[:, :-digest_width]

        widths = tuple(self.ep_size // size for size in (*levels, 1))
        offset = 0
        level_counts: list[torch.Tensor] = []
        for width in widths:
            level_counts.append(reduced[:, offset : offset + width])
            offset += width
        reduced_assignments = reduced[:, offset : offset + self.ep_size]
        tensor_cost = self._tensor_cost_with_compute_scale(level_counts, reduced_assignments)
        current_tensor = self._index_cost(tensor_cost, torch.tensor(0, dtype=torch.long, device=selected.device))
        candidate_tensor = self._index_cost(tensor_cost, torch.tensor(1, dtype=torch.long, device=selected.device))
        current_mapping = QuotaMapping(physical_routes[0], ())
        candidate_mapping = QuotaMapping(physical_routes[1], ())
        policies = _DeviceQuotaPolicies(rows=compact_rows, row_counts=row_counts, max_copies=max_copies)
        return (
            _ScoredLayout(
                tensor_cost=current_tensor,
                cost=self._placement_cost(
                    current_tensor,
                    layout=current_layout,
                    owner_slots=current_owners,
                    actions=(),
                ),
                mapping=current_mapping,
            ),
            _ScoredLayout(
                tensor_cost=candidate_tensor,
                cost=self._placement_cost(
                    candidate_tensor,
                    layout=candidate_layout,
                    owner_slots=candidate_owners,
                    actions=candidate_actions,
                ),
                mapping=candidate_mapping,
            ),
            policies,
        )

    def _is_intra_peer(self, source_rank: int, destination_rank: int) -> bool:
        inner = max(1, min(int(self.hierarchy.local_world_size), self.ep_size))
        return source_rank // inner == destination_rank // inner

    def _peer_cost(
        self, link_intra: LinkCost, link_inter: LinkCost, source: int, destination: int, size: int
    ) -> float:
        if source == destination or size <= 0:
            return 0.0
        link = link_intra if self._is_intra_peer(source, destination) else link_inter
        return float(link.alpha) + float(link.beta) * float(size)

    def _peer_wave_cost(
        self,
        payloads: dict[tuple[int, int], int],
        *,
        intra: LinkCost,
        inter: LinkCost,
    ) -> float:
        pair_payloads: dict[tuple[int, int], list[int]] = {}
        for (source, destination), payload in payloads.items():
            if source == destination or payload <= 0:
                continue
            lhs, rhs = sorted((int(source), int(destination)))
            directions = pair_payloads.setdefault((lhs, rhs), [0, 0])
            directions[int(source) != lhs] += int(payload)
        rank_costs = [0.0] * self.ep_size
        for (lhs, rhs), directions in pair_payloads.items():
            pair_cost = self._peer_cost(intra, inter, lhs, rhs, max(directions))
            rank_costs[lhs] += pair_cost
            rank_costs[rhs] += pair_cost
        cost = max(rank_costs, default=0.0)
        if self.perf_model.runtime_cost_status == "fallback":
            cost *= self.communication_scale
        return cost

    def _state_move_cost(self, actions: Sequence[PlacementAction]) -> float:
        model = self.perf_model.resolved_state_move()
        swap_payloads: dict[tuple[int, int], int] = defaultdict(int)
        replica_payloads: dict[tuple[int, int], int] = defaultdict(int)
        for action in actions:
            source_rank = int(action.src_slot) // self.slots_per_rank
            destination_rank = int(action.dst_slot) // self.slots_per_rank
            if action.kind == "swap":
                swap_payloads[(source_rank, destination_rank)] += self._expert_bytes(
                    action.src_logical, gradient=False
                )
                swap_payloads[(destination_rank, source_rank)] += self._expert_bytes(
                    action.dst_logical, gradient=False
                )
            elif action.kind == "replica":
                replica_payloads[(source_rank, destination_rank)] += self._expert_bytes(
                    action.src_logical, gradient=False
                )
        return self._peer_wave_cost(
            swap_payloads,
            intra=model.intra,
            inter=model.inter,
        ) + self._peer_wave_cost(
            replica_payloads,
            intra=model.intra,
            inter=model.inter,
        )

    def _gradient_sync_cost(self, layout: torch.Tensor, owner_slots: torch.Tensor) -> float:
        model = self.perf_model.resolved_gradient_sync()
        layout_values = [int(value) for value in layout.detach().to(device="cpu", dtype=torch.long).tolist()]
        owners = [int(value) for value in owner_slots.detach().to(device="cpu", dtype=torch.long).tolist()]
        gather_payloads: dict[tuple[int, int], int] = defaultdict(int)
        scatter_payloads: dict[tuple[int, int], int] = defaultdict(int)
        for slot, logical in enumerate(layout_values):
            if logical < 0 or slot == owners[logical]:
                continue
            source_rank = slot // self.slots_per_rank
            owner_rank = owners[logical] // self.slots_per_rank
            payload = self._expert_bytes(logical, gradient=True)
            gather_payloads[(source_rank, owner_rank)] += payload
            scatter_payloads[(owner_rank, source_rank)] += payload
        return self._peer_wave_cost(
            gather_payloads,
            intra=model.gather.intra,
            inter=model.gather.inter,
        ) + self._peer_wave_cost(
            scatter_payloads,
            intra=model.scatter.intra,
            inter=model.scatter.inter,
        )

    def _placement_cost(
        self,
        tensor_cost: _TensorCost,
        *,
        layout: torch.Tensor,
        owner_slots: torch.Tensor,
        actions: Sequence[PlacementAction],
    ) -> PlacementCost:
        base = self._to_cost(tensor_cost)
        return PlacementCost(
            communication=base.communication,
            compute=base.compute,
            communication_model_units=base.communication_model_units,
            peak_communication_rank=base.peak_communication_rank,
            peak_compute_rank=base.peak_compute_rank,
            selected_dim=base.selected_dim,
            state_move_exposed=self._state_move_cost(actions),
            gradient_sync=self._gradient_sync_cost(layout, owner_slots),
        )

    def _local_weighted_stats(
        self,
        physical_slots: torch.Tensor,
        token_weights: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        physical = physical_slots.to(torch.long)
        if physical.ndim == 1:
            physical = physical.unsqueeze(-1)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
        num_tokens, top_k = ranks.shape
        weights = (
            torch.ones((num_tokens,), dtype=torch.float32, device=ranks.device)
            if token_weights is None
            else token_weights.to(device=ranks.device, dtype=torch.float32)
        )
        assignment = torch.zeros((self.ep_size,), dtype=torch.float32, device=ranks.device)
        assignment.scatter_add_(0, ranks.reshape(-1), weights.view(-1, 1).expand(-1, top_k).reshape(-1))
        counts: list[torch.Tensor] = []
        for size in self.hierarchy.group_sizes[: max(0, self.hierarchy.selected_dim - 1)]:
            groups = torch.div(ranks, int(size), rounding_mode="floor")
            num_groups = self.ep_size // int(size)
            hits = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=ranks.device)
            hits.scatter_(1, groups, True)
            counts.append((hits.to(torch.float32) * weights.view(-1, 1)).sum(dim=0))
        rank_hits = torch.zeros((num_tokens, self.ep_size), dtype=torch.bool, device=ranks.device)
        rank_hits.scatter_(1, ranks, True)
        counts.append((rank_hits.to(torch.float32) * weights.view(-1, 1)).sum(dim=0))
        return tuple(counts), assignment

    def _tensor_cost_with_compute_scale(
        self,
        base_counts: Sequence[torch.Tensor],
        assignment_counts: torch.Tensor,
    ) -> _TensorCost:
        communication_units, selected_dim = self._communication_costs([row.to(torch.float32) for row in base_counts])
        communication = communication_units * self.communication_scale
        peak_assignments, peak_compute_rank = assignment_counts.to(torch.float32).max(dim=-1)
        compute = (self.forward_compute_per_assignment + self.backward_compute_per_assignment) * peak_assignments
        peak_communication_rank = base_counts[-1].to(torch.float32).argmax(dim=-1)
        return _TensorCost(
            communication=communication,
            compute=compute,
            communication_model_units=communication_units,
            peak_communication_rank=peak_communication_rank,
            peak_compute_rank=peak_compute_rank,
            selected_dim=selected_dim,
        )

    def _project_sample_assignments(
        self,
        summary: RouteSummary,
        physical_slots: torch.Tensor,
        owner_slots: torch.Tensor,
    ) -> torch.Tensor:
        destinations = torch.div(physical_slots.to(torch.long), self.slots_per_rank, rounding_mode="floor")
        owner_ranks = torch.div(owner_slots.to(torch.long), self.slots_per_rank, rounding_mode="floor")
        selected = summary.sample_routes
        projected = torch.zeros((self.ep_size,), dtype=torch.float32, device=selected.device)
        for source in range(self.ep_size):
            source_mask = summary.sample_sources == source
            source_routes = selected[source_mask]
            source_destinations = destinations[source_mask]
            for logical in range(summary.assignment_counts.shape[1]):
                exact = int(summary.assignment_counts[source, logical].item())
                if exact == 0:
                    continue
                logical_mask = source_routes == logical
                counts = torch.zeros((self.ep_size,), dtype=torch.float32, device=selected.device)
                counts.scatter_add_(
                    0,
                    source_destinations[logical_mask],
                    torch.ones(int(logical_mask.sum().item()), dtype=torch.float32, device=selected.device),
                )
                total = float(counts.sum().item())
                if total == 0.0:
                    projected[owner_ranks[logical]] += float(exact)
                    continue
                raw = counts * (exact / total)
                rounded = torch.floor(raw)
                remainder = exact - int(rounded.sum().item())
                if remainder:
                    fractions = raw - rounded
                    rank_order = torch.arange(self.ep_size, dtype=torch.float32, device=selected.device)
                    priority = fractions * (self.ep_size + 1.0) - rank_order / (self.ep_size + 1.0)
                    rounded.index_add_(
                        0,
                        priority.topk(remainder, largest=True, sorted=True).indices,
                        torch.ones((remainder,), dtype=torch.float32, device=selected.device),
                    )
                projected += rounded
        return projected

    def _score_sample_layout(
        self,
        summary: RouteSummary,
        layout: torch.Tensor,
        owner_slots: torch.Tensor,
        *,
        actions: Sequence[PlacementAction],
        step: int,
        layer_seed: int,
    ) -> _ScoredLayout:
        mapping = assign_tokens_to_copies_with_quota(
            summary.sample_routes,
            layout,
            slots_per_rank=self.slots_per_rank,
            source_ranks=summary.sample_sources,
            hierarchy=self.hierarchy,
            owner_slots=owner_slots,
            token_ordinals=summary.sample_ordinals,
            token_weights=summary.sample_weights,
            step=step,
            layer_seed=layer_seed,
        )
        counts, _sample_assignment = self._local_weighted_stats(mapping.physical_slots, summary.sample_weights)
        assignment = self._project_sample_assignments(summary, mapping.physical_slots, owner_slots)
        tensor_cost = self._tensor_cost_with_compute_scale(
            [row.unsqueeze(0) for row in counts], assignment.unsqueeze(0)
        )
        cost = self._placement_cost(tensor_cost, layout=layout, owner_slots=owner_slots, actions=actions)
        return _ScoredLayout(tensor_cost=tensor_cost, cost=cost, mapping=mapping)

    def _packed_local_stats(
        self,
        selected: torch.Tensor,
        mapping: QuotaMapping,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        counts, assignment = self._local_weighted_stats(mapping.physical_slots, None)
        widths = tuple(int(row.numel()) for row in counts)
        return torch.cat((*counts, assignment)), widths

    def _score_exact_pair(
        self,
        selected: torch.Tensor,
        source_ranks: int | torch.Tensor,
        token_ordinals: torch.Tensor,
        current_layout: torch.Tensor,
        current_owners: torch.Tensor,
        current_policy: Sequence[QuotaPolicyEntry],
        candidate_layout: torch.Tensor,
        candidate_owners: torch.Tensor,
        candidate_policy: Sequence[QuotaPolicyEntry],
        candidate_actions: Sequence[PlacementAction],
        *,
        step: int,
        layer_seed: int,
    ) -> tuple[_ScoredLayout, _ScoredLayout]:
        current_mapping = assign_tokens_to_copies_with_quota(
            selected,
            current_layout,
            slots_per_rank=self.slots_per_rank,
            source_ranks=source_ranks,
            hierarchy=self.hierarchy,
            owner_slots=current_owners,
            token_ordinals=token_ordinals,
            quota_policy=current_policy,
            step=step,
            layer_seed=layer_seed,
        )
        candidate_mapping = assign_tokens_to_copies_with_quota(
            selected,
            candidate_layout,
            slots_per_rank=self.slots_per_rank,
            source_ranks=source_ranks,
            hierarchy=self.hierarchy,
            owner_slots=candidate_owners,
            token_ordinals=token_ordinals,
            quota_policy=candidate_policy,
            step=step,
            layer_seed=layer_seed,
        )
        current_packed, widths = self._packed_local_stats(selected, current_mapping)
        candidate_packed, candidate_widths = self._packed_local_stats(selected, candidate_mapping)
        if candidate_widths != widths:
            raise RuntimeError("Current and candidate layouts produced incompatible exact statistics.")
        reduced = torch.stack((current_packed, candidate_packed), dim=0)
        digest_width = 0
        if self.verify_collective_digest:
            signature_values = torch.cat(
                (
                    current_layout.reshape(-1),
                    current_owners.reshape(-1),
                    candidate_layout.reshape(-1),
                    candidate_owners.reshape(-1),
                )
            ).to(device=reduced.device, dtype=torch.long)
            positions = torch.arange(1, signature_values.numel() + 1, dtype=torch.long, device=reduced.device)

            def host_sequence_hash(seed: int) -> int:
                value = int(seed)
                rows = (
                    *(entry.as_tuple() for entry in current_policy),
                    *(entry.as_tuple() for entry in candidate_policy),
                    *(
                        (action.src_slot, action.dst_slot, action.src_logical, action.dst_logical)
                        for action in candidate_actions
                    ),
                )
                for row in rows:
                    for item in row:
                        value = (value * 131 + int(item) + 17) % 509
                return value

            first = torch.remainder(
                ((signature_values + 2) * (positions * 17 + 3)).sum() + host_sequence_hash(11), 509
            )
            second = torch.remainder(
                ((signature_values + 5) * (positions * positions + 29)).sum() + host_sequence_hash(37), 509
            )
            digest_row = torch.stack((first, first * first, second, second * second)).to(reduced.dtype)
            reduced = torch.cat((reduced, digest_row.view(1, -1).expand(2, -1)), dim=1)
            digest_width = int(digest_row.numel())
        if self.reducer is not None:
            result = self.reducer(reduced)
            if result is not None:
                reduced = result
        if digest_width:
            digest_sum = reduced[0, -digest_width:]
            world = float(self.ep_size)
            if not bool(
                (digest_sum[1] * world == digest_sum[0] * digest_sum[0])
                .logical_and(digest_sum[3] * world == digest_sum[2] * digest_sum[2])
                .item()
            ):
                raise RuntimeError("CoRe-MoE ranks produced inconsistent placement plans before migration.")
            reduced = reduced[:, :-digest_width]
        offset = 0
        level_counts: list[torch.Tensor] = []
        for width in widths:
            level_counts.append(reduced[:, offset : offset + width])
            offset += width
        assignment = reduced[:, offset : offset + self.ep_size]
        tensor_cost = self._tensor_cost_with_compute_scale(level_counts, assignment)
        current_tensor = self._index_cost(tensor_cost, torch.tensor(0, device=reduced.device))
        candidate_tensor = self._index_cost(tensor_cost, torch.tensor(1, device=reduced.device))
        return (
            _ScoredLayout(
                tensor_cost=current_tensor,
                cost=self._placement_cost(
                    current_tensor, layout=current_layout, owner_slots=current_owners, actions=()
                ),
                mapping=current_mapping,
            ),
            _ScoredLayout(
                tensor_cost=candidate_tensor,
                cost=self._placement_cost(
                    candidate_tensor,
                    layout=candidate_layout,
                    owner_slots=candidate_owners,
                    actions=candidate_actions,
                ),
                mapping=candidate_mapping,
            ),
        )

    @staticmethod
    def _swap_layout(
        layout: torch.Tensor,
        owners: torch.Tensor,
        lhs: int,
        rhs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, PlacementAction]:
        updated_layout = layout.clone()
        updated_owners = owners.clone()
        lhs_slot = int(owners[lhs].item())
        rhs_slot = int(owners[rhs].item())
        updated_layout[lhs_slot], updated_layout[rhs_slot] = (
            updated_layout[rhs_slot].clone(),
            updated_layout[lhs_slot].clone(),
        )
        updated_owners[lhs], updated_owners[rhs] = owners[rhs].clone(), owners[lhs].clone()
        return updated_layout, updated_owners, PlacementAction("swap", lhs_slot, rhs_slot, lhs, rhs)

    def _valid_swap(
        self,
        layout: torch.Tensor,
        owners: torch.Tensor,
        lhs: int,
        rhs: int,
        used: set[int],
        bottlenecks: set[int],
    ) -> bool:
        if lhs in used or rhs in used:
            return False
        lhs_rank = int(owners[lhs].item()) // self.slots_per_rank
        rhs_rank = int(owners[rhs].item()) // self.slots_per_rank
        lhs_hot = lhs_rank in bottlenecks
        rhs_hot = rhs_rank in bottlenecks
        if lhs_rank == rhs_rank or lhs_hot == rhs_hot:
            return False
        values = layout.detach().to(device="cpu", dtype=torch.long).view(self.ep_size, self.slots_per_rank)
        if bool((values[lhs_rank] == rhs).any().item()) or bool((values[rhs_rank] == lhs).any().item()):
            return False
        return True

    def _plan_swaps(
        self,
        summary: RouteSummary,
        layout: torch.Tensor,
        owners: torch.Tensor,
        *,
        max_swaps: int,
        step: int,
        layer_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[PlacementAction], _ScoredLayout, _ScoredLayout]:
        actions: list[PlacementAction] = []
        used: set[int] = set()
        current = self._score_sample_layout(summary, layout, owners, actions=actions, step=step, layer_seed=layer_seed)
        baseline = current
        for _ in range(max(0, int(max_swaps))):
            bottlenecks = {current.cost.peak_communication_rank, current.cost.peak_compute_rank}
            best: tuple[float, int, int, torch.Tensor, torch.Tensor, PlacementAction, _ScoredLayout] | None = None
            for lhs in range(int(owners.numel())):
                for rhs in range(lhs + 1, int(owners.numel())):
                    if not self._valid_swap(layout, owners, lhs, rhs, used, bottlenecks):
                        continue
                    candidate_layout, candidate_owners, action = self._swap_layout(layout, owners, lhs, rhs)
                    candidate = self._score_sample_layout(
                        summary,
                        candidate_layout,
                        candidate_owners,
                        actions=(*actions, action),
                        step=step,
                        layer_seed=layer_seed,
                    )
                    row = (
                        candidate.cost.total,
                        lhs,
                        rhs,
                        candidate_layout,
                        candidate_owners,
                        action,
                        candidate,
                    )
                    if best is None or row[:3] < best[:3]:
                        best = row
            if best is None or not best[0] < current.cost.total:
                break
            _, lhs, rhs, layout, owners, action, current = best
            actions.append(action)
            used.update((lhs, rhs))
        return layout, owners, actions, current, baseline

    def _replica_edge(
        self,
        summary: RouteSummary,
        layout: torch.Tensor,
        owners: torch.Tensor,
        baseline_mapping: QuotaMapping,
        baseline_counts: Sequence[torch.Tensor],
        baseline_assignment: torch.Tensor,
        baseline_total: float,
        slot: int,
        logical: int,
        *,
        step: int,
        layer_seed: int,
    ) -> tuple[float, PlacementAction]:
        updated = layout.clone()
        previous = int(layout[slot].item())
        if logical < 0:
            updated[slot] = -1
            action = PlacementAction("empty", slot, slot, previous, -1)
        else:
            updated[slot] = logical
            action = PlacementAction("replica", int(owners[logical].item()), slot, logical, previous)

        candidate_counts = [row.clone() for row in baseline_counts]
        candidate_assignment = baseline_assignment.clone()
        independent_updates: list[tuple[int, torch.Tensor]] = []
        if previous >= 0:
            remove_layout = layout.clone()
            remove_layout[slot] = -1
            independent_updates.append((previous, remove_layout))
        if logical >= 0:
            add_layout = layout.clone()
            add_layout[slot] = logical
            independent_updates.append((logical, add_layout))
        for changed_logical, edge_layout in independent_updates:
            edge_physical = _remap_replica_logical_from_baseline(
                summary.sample_routes,
                edge_layout,
                baseline_mapping.physical_slots,
                logical_expert=changed_logical,
                slots_per_rank=self.slots_per_rank,
                source_ranks=summary.sample_sources,
                hierarchy=self.hierarchy,
                owner_slots=owners,
                token_ordinals=summary.sample_ordinals,
                token_weights=summary.sample_weights,
                step=step,
                layer_seed=layer_seed,
            )
            edge_counts, _ = self._local_weighted_stats(edge_physical, summary.sample_weights)
            edge_assignment = self._project_sample_assignments(summary, edge_physical, owners)
            for index, row in enumerate(edge_counts):
                candidate_counts[index].add_(row - baseline_counts[index])
            candidate_assignment.add_(edge_assignment - baseline_assignment)

        tensor_cost = self._tensor_cost_with_compute_scale(
            [row.clamp_min(0).unsqueeze(0) for row in candidate_counts],
            candidate_assignment.clamp_min(0).unsqueeze(0),
        )
        scored = self._placement_cost(
            tensor_cost,
            layout=updated,
            owner_slots=owners,
            actions=(action,),
        )
        return baseline_total - scored.total, action

    def _plan_replicas(
        self,
        summary: RouteSummary,
        layout: torch.Tensor,
        owners: torch.Tensor,
        actions: list[PlacementAction],
        current: _ScoredLayout,
        *,
        max_replicas: int,
        step: int,
        layer_seed: int,
    ) -> tuple[torch.Tensor, list[PlacementAction], _ScoredLayout, int]:
        if max_replicas <= 0:
            return layout, actions, current, 0
        baseline_counts, _ = self._local_weighted_stats(current.mapping.physical_slots, summary.sample_weights)
        baseline_assignment = self._project_sample_assignments(summary, current.mapping.physical_slots, owners)
        baseline_tensor = self._tensor_cost_with_compute_scale(
            [row.unsqueeze(0) for row in baseline_counts],
            baseline_assignment.unsqueeze(0),
        )
        baseline_total = self._placement_cost(
            baseline_tensor,
            layout=layout,
            owner_slots=owners,
            actions=(),
        ).total
        owner_slot_set = {int(value) for value in owners.detach().cpu().tolist()}
        bottlenecks = {current.cost.peak_communication_rank, current.cost.peak_compute_rank}
        layout_cpu = layout.detach().to(device="cpu", dtype=torch.long)
        hot_experts = sorted(
            {
                int(logical)
                for slot, logical in enumerate(layout_cpu.tolist())
                if logical >= 0 and slot // self.slots_per_rank in bottlenecks
            }
        )
        selected: list[tuple[float, int, PlacementAction]] = []
        for rank in range(self.ep_size):
            rank_slots = [
                slot
                for slot in range(rank * self.slots_per_rank, (rank + 1) * self.slots_per_rank)
                if slot not in owner_slot_set
            ]
            if not rank_slots:
                continue
            existing = {
                int(layout_cpu[slot])
                for slot in range(rank * self.slots_per_rank, (rank + 1) * self.slots_per_rank)
                if int(layout_cpu[slot]) >= 0
            }
            expert_columns = [logical for logical in hot_experts if logical not in existing]
            dummy_columns = 2 * len(rank_slots)
            column_count = dummy_columns + len(expert_columns)
            weights = [[-math.inf] * column_count for _ in rank_slots]
            edge_actions: dict[tuple[int, int], PlacementAction] = {}
            for row, slot in enumerate(rank_slots):
                keep_column = 2 * row
                empty_column = keep_column + 1
                weights[row][keep_column] = 0.0
                if int(layout_cpu[slot]) >= 0:
                    gain, action = self._replica_edge(
                        summary,
                        layout,
                        owners,
                        current.mapping,
                        baseline_counts,
                        baseline_assignment,
                        baseline_total,
                        slot,
                        -1,
                        step=step,
                        layer_seed=layer_seed,
                    )
                    weights[row][empty_column] = gain if gain > 0.0 else -math.inf
                    edge_actions[(row, empty_column)] = action
                for expert_column, logical in enumerate(expert_columns):
                    column = dummy_columns + expert_column
                    gain, action = self._replica_edge(
                        summary,
                        layout,
                        owners,
                        current.mapping,
                        baseline_counts,
                        baseline_assignment,
                        baseline_total,
                        slot,
                        logical,
                        step=step,
                        layer_seed=layer_seed,
                    )
                    weights[row][column] = gain if gain > 0.0 else -math.inf
                    edge_actions[(row, column)] = action
            matching = _stable_hungarian_maximize(weights)
            for row, column in enumerate(matching):
                action = edge_actions.get((row, column))
                if action is not None and weights[row][column] > 0.0:
                    selected.append((weights[row][column], rank_slots[row], action))

        num_experts = int(owners.numel())
        selected.sort(
            key=lambda row: (
                -row[0],
                row[1],
                row[2].src_logical if row[2].kind == "replica" else num_experts,
                1 if row[2].kind == "empty" else 2,
            )
        )
        copy_counts = torch.bincount(layout_cpu[layout_cpu >= 0], minlength=num_experts).tolist()
        accepted: list[tuple[float, int, PlacementAction]] = []
        for row in selected:
            action = row[2]
            if action.kind == "replica" and copy_counts[action.src_logical] >= 8:
                continue
            previous = action.src_logical if action.kind == "empty" else action.dst_logical
            if previous >= 0:
                copy_counts[previous] -= 1
            if action.kind == "replica":
                copy_counts[action.src_logical] += 1
            accepted.append(row)
            if len(accepted) >= max_replicas:
                break
        if not accepted:
            return layout, actions, current, 0
        updated = layout.clone()
        replica_actions: list[PlacementAction] = []
        for _, _, action in accepted:
            updated[action.dst_slot] = -1 if action.kind == "empty" else action.src_logical
            replica_actions.append(action)
        combined = [*actions, *replica_actions]
        scored = self._score_sample_layout(
            summary,
            updated,
            owners,
            actions=combined,
            step=step,
            layer_seed=layer_seed,
        )
        return updated, combined, scored, len(replica_actions)

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
        selected = selected_experts.to(torch.long)
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        device = selected.device
        current_layout = slot_to_logical.to(device=device, dtype=torch.long, non_blocking=True).clone()
        current_owners = owner_slots.to(device=device, dtype=torch.long, non_blocking=True).clone()
        ordinals = (
            torch.arange(selected.shape[0], dtype=torch.long, device=device)
            if token_ordinals is None
            else token_ordinals.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        )
        source_rank = int(source_ranks) if isinstance(source_ranks, int) else int(source_ranks.reshape(-1)[0].item())

        route_started = time.perf_counter()
        fused_extension = self._fused_planner_extension(device)
        num_experts = int(current_owners.numel())
        redundant_per_rank = (
            self.slots_per_rank - num_experts // self.ep_size if num_experts % self.ep_size == 0 else -1
        )
        replica_shape_supported = max_replicas <= 0 or 0 < redundant_per_rank <= 8
        required_fused_capabilities = self._required_fused_capabilities(
            max_swaps=max_swaps,
            max_replicas=max_replicas,
        )
        logical_ids = torch.arange(num_experts, dtype=torch.long, device=device).view(1, -1)
        current_copy_max = current_layout.view(-1, 1).eq(logical_ids).sum(dim=0).max()
        # The quota kernels have a fixed eight-copy ABI. The matcher enforces that limit per
        # logical expert while applying the globally budgeted replica actions, so the capability
        # gate only needs to reject an already-invalid input layout. Treating the global action
        # budget as if every action targeted one expert would disable the normal EP16 S1 path.
        copy_budget_supported = current_copy_max <= 8
        input_supported = copy_budget_supported.logical_and(
            torch.tensor(replica_shape_supported, dtype=torch.bool, device=device)
        )
        local_fused_capabilities = torch.where(
            input_supported,
            torch.tensor(self._local_fused_capabilities(fused_extension), dtype=torch.long, device=device),
            torch.zeros((), dtype=torch.long, device=device),
        )
        device_summary = self._build_device_planning_summary(
            selected,
            current_owners,
            source_rank=source_rank,
            step=step,
            layer_seed=layer_seed,
            fused_capable=local_fused_capabilities,
            required_fused_capabilities=required_fused_capabilities,
        )
        summary = device_summary.route
        route_stats_ms = (time.perf_counter() - route_started) * 1000.0

        fused_path_enabled = bool(device_summary.fused_capabilities.all().item())
        swap_started = time.perf_counter()
        if fused_path_enabled:
            if max_swaps > 0 or max_replicas > 0:
                fused_swap = self._fused_swap_select(
                    device_summary.swap_stats,
                    current_layout,
                    current_owners,
                    max_swaps=max(0, int(max_swaps)),
                    sample_routes=summary.sample_routes,
                    sample_weights=summary.sample_weights,
                    extension=fused_extension,
                )
                if fused_swap is None:
                    raise RuntimeError("The group enabled fused CoRe-MoE planning without a swap selector.")
                candidate_layout, candidate_owners, action_rows, swap_metadata, replica_seed_base_counts = fused_swap
                accepted = int(swap_metadata[0].item())
                if accepted < 0 or accepted > min(max(0, int(max_swaps)), int(action_rows.shape[0])):
                    raise RuntimeError(f"Fused CoRe-MoE swap selector returned invalid action count {accepted}.")
                actions = [
                    PlacementAction(
                        "swap",
                        int(row[2]),
                        int(row[3]),
                        int(row[0]),
                        int(row[1]),
                    )
                    for row in action_rows[:accepted].detach().to(device="cpu", dtype=torch.long).tolist()
                ]
            else:
                candidate_layout = current_layout.clone()
                candidate_owners = current_owners.clone()
                actions = []
                swap_metadata = torch.zeros((3,), dtype=torch.int32, device=device)
                replica_seed_base_counts = None
            baseline_sample_score = None
            sample_score = None
        else:
            candidate_layout, candidate_owners, actions, sample_score, baseline_sample_score = self._plan_swaps(
                summary,
                current_layout.clone(),
                current_owners.clone(),
                max_swaps=max_swaps,
                step=step,
                layer_seed=layer_seed,
            )
        swap_ms = (time.perf_counter() - swap_started) * 1000.0
        swap_count = len(actions)

        replica_started = time.perf_counter()
        if fused_path_enabled:
            candidate_layout, replica_actions, replica_count = self._fused_plan_replicas(
                fused_extension,
                summary,
                candidate_layout,
                candidate_owners,
                swap_metadata,
                replica_seed_base_counts,
                max_replicas=max_replicas,
                step=step,
                layer_seed=layer_seed,
            )
            actions.extend(replica_actions)
            candidate_sample_score = None
        else:
            candidate_layout, actions, candidate_sample_score, replica_count = self._plan_replicas(
                summary,
                candidate_layout,
                candidate_owners,
                actions,
                sample_score,
                max_replicas=max_replicas,
                step=step,
                layer_seed=layer_seed,
            )
        replica_ms = (time.perf_counter() - replica_started) * 1000.0

        exact_started = time.perf_counter()
        if fused_path_enabled:
            baseline, candidate, device_policies = self._fused_score_exact_pair(
                fused_extension,
                summary,
                selected,
                ordinals,
                source_rank,
                current_layout,
                current_owners,
                candidate_layout,
                candidate_owners,
                actions,
                step=step,
                layer_seed=layer_seed,
            )
            baseline_policy: tuple[QuotaPolicyEntry, ...] | None = None
            candidate_policy: tuple[QuotaPolicyEntry, ...] | None = None
        else:
            baseline, candidate = self._score_exact_pair(
                selected,
                source_ranks,
                ordinals,
                current_layout,
                current_owners,
                baseline_sample_score.mapping.policy,
                candidate_layout,
                candidate_owners,
                candidate_sample_score.mapping.policy,
                actions,
                step=step,
                layer_seed=layer_seed,
            )
            baseline_policy = baseline_sample_score.mapping.policy
            candidate_policy = candidate_sample_score.mapping.policy
        accepted = bool(actions) and candidate.cost.total < baseline.cost.total
        if accepted:
            final_layout_tensor = candidate_layout
            final_owner_tensor = candidate_owners
            final_cost = candidate.cost
            final_mapping = candidate.mapping
            final_policy = (
                self._quota_policy_from_device_rows(device_policies, 1, source_rank=source_rank)
                if fused_path_enabled
                else candidate_policy
            )
            final_actions = tuple(actions)
            accepted_swaps = swap_count
            accepted_replicas = replica_count
        else:
            final_layout_tensor = current_layout
            final_owner_tensor = current_owners
            final_cost = baseline.cost
            final_mapping = baseline.mapping
            final_policy = (
                self._quota_policy_from_device_rows(device_policies, 0, source_rank=source_rank)
                if fused_path_enabled
                else baseline_policy
            )
            final_actions = ()
            accepted_swaps = 0
            accepted_replicas = 0
        exact_ms = (time.perf_counter() - exact_started) * 1000.0

        final_layout = tuple(int(value) for value in final_layout_tensor.detach().cpu().tolist())
        final_owners = tuple(int(value) for value in final_owner_tensor.detach().cpu().tolist())
        quota_policy = tuple(entry.as_tuple() for entry in final_policy)
        layout_digest = hashlib.sha256(
            repr(
                (
                    CORE_MOE_ALGORITHM_VERSION,
                    tuple(action.format() for action in final_actions),
                    final_layout,
                    final_owners,
                    quota_policy,
                )
            ).encode()
        ).hexdigest()
        planning_ms = (time.perf_counter() - started) * 1000.0
        return PlacementPlan(
            actions=final_actions,
            initial_layout=tuple(int(value) for value in current_layout.detach().cpu().tolist()),
            final_layout=final_layout,
            baseline_cost=baseline.cost,
            final_cost=final_cost,
            swap_rounds=accepted_swaps,
            replica_rounds=accepted_replicas,
            planning_ms=planning_ms,
            route_stats_ms=route_stats_ms,
            swap_ms=swap_ms,
            replica_ms=replica_ms,
            swap_score_ms=swap_ms,
            swap_update_ms=0.0,
            swap_collective_ms=0.0,
            replica_score_ms=replica_ms,
            replica_update_ms=0.0,
            replica_collective_ms=0.0,
            decision_sync_ms=0.0,
            finalization_ms=exact_ms,
            algorithm_version=CORE_MOE_ALGORITHM_VERSION,
            quota_policy=quota_policy,
            layout_digest=layout_digest,
            local_physical_routes=final_mapping.physical_slots,
            final_owner_slots=final_owners,
        )


__all__ = [
    "CORE_MOE_ALGORITHM_VERSION",
    "CoReMoEPlanner",
    "QuotaMapping",
    "QuotaPolicyEntry",
    "QuotaTensorTables",
    "RouteSummary",
    "assign_tokens_to_copies_with_quota",
    "build_quota_tensor_tables",
    "build_route_summary",
    "compress_local_route_payload",
]
