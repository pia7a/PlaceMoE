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

"""Fast forward-reuse heuristic for one redundant-expert cover per layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .greedy_planner import assign_tokens_to_copies_greedy
from .planner import PlacementAction


FORWARD_REUSE_COVER_ALGORITHM_VERSION = "hiermoe-forward-reuse-cover-p1"


@dataclass(frozen=True)
class ForwardCoverProposal:
    """One locally scored cover proposal for a layer-owner rank."""

    action: PlacementAction | None
    estimated_gain: float
    communication_gain: float
    assignment_delta: float
    baseline_communication_units: float
    victim_assignment_count: float


@dataclass(frozen=True)
class ForwardCoverHeuristicStatistics:
    """Small per-expert statistics used to propose one node-wide Cover."""

    communication_benefit: torch.Tensor
    expert_assignments: torch.Tensor
    baseline_communication_units: torch.Tensor


@dataclass(frozen=True)
class ForwardCoverLocalValidation:
    """Local exact statistics for one proposed cover action."""

    baseline_communication_counts: torch.Tensor
    communication_count_delta: torch.Tensor
    baseline_assignment_counts: torch.Tensor
    assignment_count_delta: torch.Tensor
    affected_tokens: int


@dataclass(frozen=True)
class ForwardCoverBatchedPatchValidation:
    """Batched local deltas for one patch-remap cover per layer."""

    communication_count_delta: torch.Tensor
    assignment_count_delta: torch.Tensor
    affected_tokens: torch.Tensor


def forward_cover_patch_source_rank_relevant(
    *,
    action: PlacementAction,
    source_rank: int,
    slots_per_rank: int,
    service_group_size: int,
    source_logical_to_physical: torch.Tensor | None,
) -> bool:
    """Return whether one source rank can have any route changed by a Cover.

    An insertion only changes requests originating in the destination service
    group. Evicting an occupied slot additionally changes a source rank only
    when its current route LUT points the victim expert at that exact slot.
    The LUT test avoids scanning affected tokens for ranks whose local delta
    is provably zero. Missing LUT state conservatively returns ``True``.
    """

    if action.kind != "replica":
        raise ValueError("Forward route relevance only accepts replica actions.")
    if int(service_group_size) <= 0:
        raise ValueError("service_group_size must be positive.")
    destination_rank = int(action.dst_slot) // int(slots_per_rank)
    inserted_relevant = (
        int(source_rank) // int(service_group_size)
        == destination_rank // int(service_group_size)
    )
    victim = int(action.dst_logical)
    if victim < 0:
        return inserted_relevant
    if source_logical_to_physical is None:
        return True
    source_mapping = source_logical_to_physical.reshape(-1)
    if not 0 <= victim < int(source_mapping.numel()):
        raise ValueError("Cover victim is outside the source-rank route LUT.")
    victim_relevant = int(source_mapping[victim].item()) == int(action.dst_slot)
    return inserted_relevant or victim_relevant


def _bottleneck_unique_benefit(
    *,
    selected_experts: torch.Tensor,
    destination_groups: torch.Tensor,
    source_group: int,
    num_groups: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count expert routes that uniquely sustain a bottleneck remote group."""

    token_group_counts = torch.zeros(
        (destination_groups.shape[0], num_groups),
        dtype=torch.int32,
        device=destination_groups.device,
    )
    token_group_counts.scatter_add_(
        1,
        destination_groups,
        torch.ones_like(destination_groups, dtype=torch.int32),
    )
    sole = token_group_counts.gather(1, destination_groups) == 1
    remote = destination_groups != int(source_group)
    counts = (token_group_counts > 0).sum(dim=0, dtype=torch.float32)
    if 0 <= int(source_group) < int(num_groups):
        counts[int(source_group)] = 0.0
    maximum = counts.max() if counts.numel() > 0 else counts.new_zeros(())
    bottleneck = counts == maximum
    useful = sole & remote & bottleneck.index_select(0, destination_groups.reshape(-1)).view_as(destination_groups)
    benefit = torch.zeros((num_experts,), dtype=torch.float32, device=destination_groups.device)
    if selected_experts.numel() > 0:
        benefit.scatter_add_(
            0,
            selected_experts.reshape(-1),
            useful.reshape(-1).to(torch.float32),
        )
    return benefit, maximum


def _bottleneck_move_benefit(
    *,
    selected_experts: torch.Tensor,
    destination_groups: torch.Tensor,
    source_group: int,
    target_group: int,
    num_groups: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the unique-payload delta when an expert moves to target_group.

    Unlike ``_bottleneck_unique_benefit``, the new copy need not be local to
    this source rank.  Removing the old destination is useful only when it is
    a remote bottleneck group sustained solely by that assignment.  Adding a
    previously absent remote target group is charged as a loss.
    """

    token_group_counts = torch.zeros(
        (destination_groups.shape[0], num_groups),
        dtype=torch.int32,
        device=destination_groups.device,
    )
    token_group_counts.scatter_add_(
        1,
        destination_groups,
        torch.ones_like(destination_groups, dtype=torch.int32),
    )
    counts = (token_group_counts > 0).sum(dim=0, dtype=torch.float32)
    if 0 <= int(source_group) < int(num_groups):
        counts[int(source_group)] = 0.0
    maximum = counts.max() if counts.numel() > 0 else counts.new_zeros(())
    bottleneck = counts == maximum

    old_is_sole = token_group_counts.gather(1, destination_groups) == 1
    old_is_remote = destination_groups != int(source_group)
    changes_group = destination_groups != int(target_group)
    old_is_bottleneck = bottleneck.index_select(0, destination_groups.reshape(-1)).view_as(destination_groups)
    removed = old_is_sole & old_is_remote & changes_group & old_is_bottleneck

    target_is_remote = int(target_group) != int(source_group)
    target_absent = token_group_counts[:, int(target_group)] == 0
    added = changes_group & target_absent.unsqueeze(1) if target_is_remote else torch.zeros_like(removed)

    benefit = torch.zeros((num_experts,), dtype=torch.float32, device=destination_groups.device)
    if selected_experts.numel() > 0:
        benefit.scatter_add_(
            0,
            selected_experts.reshape(-1),
            (removed.to(torch.float32) - added.to(torch.float32)).reshape(-1),
        )
    return benefit, maximum


def _batched_bottleneck_move_benefit(
    *,
    selected_experts: torch.Tensor,
    destination_groups: torch.Tensor,
    source_groups: torch.Tensor,
    target_groups: torch.Tensor,
    num_groups: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched equivalent of :func:`_bottleneck_move_benefit`."""

    if selected_experts.ndim != 3 or destination_groups.shape != selected_experts.shape:
        raise ValueError("Batched selected experts and destination groups must have the same rank-3 shape.")
    batch, tokens, _top_k = selected_experts.shape
    sources = source_groups.to(
        device=selected_experts.device,
        dtype=torch.long,
        non_blocking=True,
    ).reshape(batch)
    targets = target_groups.to(
        device=selected_experts.device,
        dtype=torch.long,
        non_blocking=True,
    ).reshape(batch)
    token_group_counts = torch.zeros(
        (batch, tokens, int(num_groups)),
        dtype=torch.int32,
        device=selected_experts.device,
    )
    token_group_counts.scatter_add_(
        2,
        destination_groups,
        torch.ones_like(destination_groups, dtype=torch.int32),
    )
    counts = (token_group_counts > 0).sum(dim=1, dtype=torch.float32)
    counts.scatter_(1, sources.view(batch, 1), 0.0)
    maximum = counts.amax(dim=1)
    bottleneck = counts == maximum.view(batch, 1)

    old_is_sole = token_group_counts.gather(2, destination_groups) == 1
    old_is_remote = destination_groups != sources.view(batch, 1, 1)
    changes_group = destination_groups != targets.view(batch, 1, 1)
    old_is_bottleneck = bottleneck.gather(
        1,
        destination_groups.reshape(batch, -1),
    ).view_as(destination_groups)
    removed = old_is_sole & old_is_remote & changes_group & old_is_bottleneck

    target_absent = (
        token_group_counts.gather(
            2,
            targets.view(batch, 1, 1).expand(batch, tokens, 1),
        ).squeeze(2)
        == 0
    )
    target_is_remote = targets != sources
    added = changes_group & target_absent.unsqueeze(2) & target_is_remote.view(batch, 1, 1)

    benefit = torch.zeros(
        (batch, int(num_experts)),
        dtype=torch.float32,
        device=selected_experts.device,
    )
    benefit.scatter_add_(
        1,
        selected_experts.reshape(batch, -1),
        (removed.to(torch.float32) - added.to(torch.float32)).reshape(batch, -1),
    )
    return benefit, maximum


@torch.no_grad()
def forward_cover_local_heuristic_statistics_batched(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    source_rank: int,
    target_ranks: torch.Tensor,
    slots_per_rank: int,
    ep_size: int,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    level_weights: Sequence[float] | None = None,
) -> ForwardCoverHeuristicStatistics:
    """Build several layers' source-rank proposal statistics in one graph."""

    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.ndim != 3 or physical.shape != selected.shape:
        raise ValueError("Batched selected experts and physical routes must have the same rank-3 shape.")
    batch = int(selected.shape[0])
    targets = target_ranks.to(
        device=selected.device,
        dtype=torch.long,
        non_blocking=True,
    ).reshape(-1)
    if int(targets.numel()) != batch:
        raise ValueError("target_ranks must contain one rank per batched layer.")
    if not 0 <= int(source_rank) < int(ep_size) or bool(((targets < 0) | (targets >= int(ep_size))).any()):
        raise ValueError("source_rank and target_ranks must be valid EP ranks.")

    weights = tuple(float(value) for value in level_weights) if level_weights is not None else ()
    active_group_sizes = [
        int(size)
        for size in hierarchy_group_sizes
        if 1 < int(size) < int(ep_size) and int(ep_size) % int(size) == 0
    ]
    expected_levels = 1 + len(active_group_sizes)
    if weights and len(weights) != expected_levels:
        raise ValueError(f"level_weights has {len(weights)} values, expected {expected_levels}.")
    if not weights:
        weights = (1.0,) * expected_levels

    destination_ranks = torch.div(physical, int(slots_per_rank), rounding_mode="floor")
    source_groups = torch.full(
        (batch,),
        int(source_rank),
        dtype=torch.long,
        device=selected.device,
    )
    communication_benefit, baseline_units = _batched_bottleneck_move_benefit(
        selected_experts=selected,
        destination_groups=destination_ranks,
        source_groups=source_groups,
        target_groups=targets,
        num_groups=int(ep_size),
        num_experts=int(num_experts),
    )
    communication_benefit = weights[0] * communication_benefit
    baseline_units = weights[0] * baseline_units
    for level_index, group_size in enumerate(active_group_sizes, start=1):
        group_benefit, group_maximum = _batched_bottleneck_move_benefit(
            selected_experts=selected,
            destination_groups=torch.div(destination_ranks, group_size, rounding_mode="floor"),
            source_groups=torch.div(source_groups, group_size, rounding_mode="floor"),
            target_groups=torch.div(targets, group_size, rounding_mode="floor"),
            num_groups=int(ep_size) // group_size,
            num_experts=int(num_experts),
        )
        communication_benefit = communication_benefit + weights[level_index] * group_benefit
        baseline_units = baseline_units + weights[level_index] * group_maximum

    expert_assignments = torch.zeros(
        (batch, int(num_experts)),
        dtype=torch.float32,
        device=selected.device,
    )
    expert_assignments.scatter_add_(
        1,
        selected.reshape(batch, -1),
        torch.ones_like(selected, dtype=torch.float32).reshape(batch, -1),
    )
    return ForwardCoverHeuristicStatistics(
        communication_benefit=communication_benefit,
        expert_assignments=expert_assignments,
        baseline_communication_units=baseline_units,
    )


@torch.no_grad()
def forward_cover_local_heuristic_statistics(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    source_rank: int,
    target_rank: int,
    slots_per_rank: int,
    ep_size: int,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    level_weights: Sequence[float] | None = None,
) -> ForwardCoverHeuristicStatistics:
    """Build one source rank's contribution to a service-group proposal."""

    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if physical.ndim == 1:
        physical = physical.unsqueeze(-1)
    if selected.ndim != 2 or physical.shape != selected.shape:
        raise ValueError("selected_experts and physical_routes must have the same rank-2 shape.")
    if not 0 <= int(source_rank) < int(ep_size) or not 0 <= int(target_rank) < int(ep_size):
        raise ValueError("source_rank and target_rank must be valid EP ranks.")

    weights = tuple(float(value) for value in level_weights) if level_weights is not None else ()
    active_group_sizes = [
        int(size)
        for size in hierarchy_group_sizes
        if 1 < int(size) < int(ep_size) and int(ep_size) % int(size) == 0
    ]
    expected_levels = 1 + len(active_group_sizes)
    if weights and len(weights) != expected_levels:
        raise ValueError(f"level_weights has {len(weights)} values, expected {expected_levels}.")
    if not weights:
        weights = (1.0,) * expected_levels

    destination_ranks = torch.div(physical, int(slots_per_rank), rounding_mode="floor")
    communication_benefit, baseline_units = _bottleneck_move_benefit(
        selected_experts=selected,
        destination_groups=destination_ranks,
        source_group=int(source_rank),
        target_group=int(target_rank),
        num_groups=int(ep_size),
        num_experts=int(num_experts),
    )
    communication_benefit = weights[0] * communication_benefit
    baseline_units = weights[0] * baseline_units
    for level_index, group_size in enumerate(active_group_sizes, start=1):
        destination_groups = torch.div(destination_ranks, group_size, rounding_mode="floor")
        group_benefit, group_maximum = _bottleneck_move_benefit(
            selected_experts=selected,
            destination_groups=destination_groups,
            source_group=int(source_rank) // group_size,
            target_group=int(target_rank) // group_size,
            num_groups=int(ep_size) // group_size,
            num_experts=int(num_experts),
        )
        communication_benefit = communication_benefit + weights[level_index] * group_benefit
        baseline_units = baseline_units + weights[level_index] * group_maximum

    expert_assignments = torch.bincount(selected.reshape(-1), minlength=int(num_experts)).to(torch.float32)
    return ForwardCoverHeuristicStatistics(
        communication_benefit=communication_benefit,
        expert_assignments=expert_assignments,
        baseline_communication_units=baseline_units,
    )


def _local_packed_counts(
    physical_routes: torch.Tensor,
    *,
    slots_per_rank: int,
    ep_size: int,
    hierarchy_group_sizes: Sequence[int],
) -> torch.Tensor:
    """Count unique token payloads for every active hierarchy destination."""

    ranks = torch.div(physical_routes, int(slots_per_rank), rounding_mode="floor")
    rows: list[torch.Tensor] = []
    level_sizes = (1,) + tuple(
        int(size) for size in hierarchy_group_sizes if 1 < int(size) < int(ep_size) and int(ep_size) % int(size) == 0
    )
    for size in level_sizes:
        groups = torch.div(ranks, size, rounding_mode="floor")
        num_groups = int(ep_size) // size
        hits = torch.zeros((groups.shape[0], num_groups), dtype=torch.bool, device=groups.device)
        hits.scatter_(1, groups, True)
        rows.append(hits.sum(dim=0).to(torch.float32))
    return torch.cat(rows, dim=0)


def _local_assignment_counts(
    physical_routes: torch.Tensor,
    *,
    slots_per_rank: int,
    ep_size: int,
) -> torch.Tensor:
    ranks = torch.div(physical_routes, int(slots_per_rank), rounding_mode="floor")
    return torch.bincount(ranks.reshape(-1), minlength=int(ep_size)).to(torch.float32)


def _batched_presence_delta(
    *,
    before_groups: torch.Tensor,
    after_groups: torch.Tensor,
    changed_mask: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return exact unique-token group deltas for two changed experts.

    A cover changes at most two logical experts: the evicted victim and, on
    the destination source rank, the inserted expert.  The complete top-k row
    is still used to decide whether an old group disappeared or a new group
    appeared, so interactions with every unchanged expert remain exact.
    """

    batch, tokens, _top_k = before_groups.shape
    victim_valid = changed_mask[..., 0, :].any(dim=-1)
    inserted_valid = changed_mask[..., 1, :].any(dim=-1)
    valid = torch.stack((victim_valid, inserted_valid), dim=-1)

    old_groups = torch.stack(
        (
            torch.where(
                changed_mask[..., 0, :],
                before_groups,
                torch.zeros_like(before_groups),
            ).amax(dim=-1),
            torch.where(
                changed_mask[..., 1, :],
                before_groups,
                torch.zeros_like(before_groups),
            ).amax(dim=-1),
        ),
        dim=-1,
    )
    new_groups = torch.stack(
        (
            torch.where(
                changed_mask[..., 0, :],
                after_groups,
                torch.zeros_like(after_groups),
            ).amax(dim=-1),
            torch.where(
                changed_mask[..., 1, :],
                after_groups,
                torch.zeros_like(after_groups),
            ).amax(dim=-1),
        ),
        dim=-1,
    )

    removed = valid & ~(
        after_groups.unsqueeze(-2) == old_groups.unsqueeze(-1)
    ).any(dim=-1)
    added = valid & ~(
        before_groups.unsqueeze(-2) == new_groups.unsqueeze(-1)
    ).any(dim=-1)

    # If both changed experts leave or enter the same group for one token,
    # count the set transition once rather than once per expert.
    removed[..., 1] &= ~(valid[..., 0] & (old_groups[..., 1] == old_groups[..., 0]))
    added[..., 1] &= ~(valid[..., 0] & (new_groups[..., 1] == new_groups[..., 0]))

    delta = torch.zeros(
        (batch, int(num_groups)),
        dtype=torch.float32,
        device=before_groups.device,
    )
    delta.scatter_add_(
        1,
        old_groups.reshape(batch, tokens * 2),
        -removed.reshape(batch, tokens * 2).to(torch.float32),
    )
    delta.scatter_add_(
        1,
        new_groups.reshape(batch, tokens * 2),
        added.reshape(batch, tokens * 2).to(torch.float32),
    )
    return delta


@torch.no_grad()
def forward_cover_patch_validation_stats_batched(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    source_logical: torch.Tensor,
    victim_logical: torch.Tensor,
    destination_slots: torch.Tensor,
    victim_fallback_slots: torch.Tensor,
    source_rank: int,
    slots_per_rank: int,
    ep_size: int,
    hierarchy_group_sizes: Sequence[int],
    service_group_size: int = 1,
) -> ForwardCoverBatchedPatchValidation:
    """Validate one patch-remap cover per layer with one batched tensor graph.

    The inputs have shape ``[layers, tokens, top_k]``.  Only the victim route
    and the inserted expert route can change, but destination-set changes are
    checked against the full top-k row.  This is exactly equivalent to
    patching each layer separately and recounting its affected token rows.
    """

    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.ndim != 3 or physical.shape != selected.shape:
        raise ValueError("Batched selected experts and physical routes must have the same rank-3 shape.")
    batch, _tokens, _top_k = selected.shape

    def action_column(value: torch.Tensor, name: str) -> torch.Tensor:
        result = value.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if int(result.numel()) != int(batch):
            raise ValueError(f"{name} must contain one value per layer.")
        return result.view(batch, 1, 1)

    source = action_column(source_logical, "source_logical")
    victim = action_column(victim_logical, "victim_logical")
    destination = action_column(destination_slots, "destination_slots")
    fallback = action_column(victim_fallback_slots, "victim_fallback_slots")
    destination_rank = torch.div(destination, int(slots_per_rank), rounding_mode="floor")
    if (
        int(service_group_size) <= 0
        or int(service_group_size) > int(ep_size)
        or int(ep_size) % int(service_group_size) != 0
    ):
        raise ValueError("service_group_size must be a positive divisor of ep_size.")
    source_service_group = int(source_rank) // int(service_group_size)
    destination_service_group = torch.div(
        destination_rank,
        int(service_group_size),
        rounding_mode="floor",
    )

    victim_changed = (selected == victim) & (physical == destination)
    inserted_changed = (selected == source) & (destination_service_group == source_service_group)
    candidate = torch.where(victim_changed, fallback, physical)
    candidate = torch.where(inserted_changed, destination, candidate)
    changed = victim_changed | inserted_changed
    changed_by_kind = torch.stack((victim_changed, inserted_changed), dim=-2)

    before_ranks = torch.div(physical, int(slots_per_rank), rounding_mode="floor")
    after_ranks = torch.div(candidate, int(slots_per_rank), rounding_mode="floor")
    rank_delta = _batched_presence_delta(
        before_groups=before_ranks,
        after_groups=after_ranks,
        changed_mask=changed_by_kind,
        num_groups=ep_size,
    )

    valid_sizes = tuple(
        int(size)
        for size in hierarchy_group_sizes
        if 1 < int(size) < int(ep_size) and int(ep_size) % int(size) == 0
    )
    group_deltas = [rank_delta]
    for size in valid_sizes:
        group_deltas.append(
            _batched_presence_delta(
                before_groups=torch.div(before_ranks, size, rounding_mode="floor"),
                after_groups=torch.div(after_ranks, size, rounding_mode="floor"),
                changed_mask=changed_by_kind,
                num_groups=int(ep_size) // size,
            )
        )

    assignment_delta = torch.zeros(
        (batch, int(ep_size)),
        dtype=torch.float32,
        device=selected.device,
    )
    assignment_delta.scatter_add_(
        1,
        before_ranks.reshape(batch, -1),
        -changed.reshape(batch, -1).to(torch.float32),
    )
    assignment_delta.scatter_add_(
        1,
        after_ranks.reshape(batch, -1),
        changed.reshape(batch, -1).to(torch.float32),
    )
    return ForwardCoverBatchedPatchValidation(
        communication_count_delta=torch.cat(group_deltas, dim=1),
        assignment_count_delta=assignment_delta,
        affected_tokens=changed.any(dim=-1).sum(dim=1),
    )


@torch.no_grad()
def patch_forward_cover_routes(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    action: PlacementAction,
    source_rank: int,
    slots_per_rank: int,
    victim_fallback_slot: int,
    service_group_size: int = 1,
) -> torch.Tensor:
    """Patch only routes whose serving copy changes under one Cover.

    The inserted copy serves source ranks in its destination service group.
    With ``service_group_size=1`` this is the destination rank only; with a
    node-sized group it serves every source rank in that node. Routes that
    used the overwritten victim slot fall back to the victim's canonical
    owner. This is the token-level equivalent of updating the persistent
    source-rank LUT.
    """

    if action.kind != "replica":
        raise ValueError("Forward route patching only accepts replica actions.")
    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.shape != physical.shape:
        raise ValueError("selected_experts and physical_routes must have the same shape.")
    if int(service_group_size) <= 0:
        raise ValueError("service_group_size must be positive.")

    destination_rank = int(action.dst_slot) // int(slots_per_rank)
    patched = torch.where(
        (selected == int(action.dst_logical)) & (physical == int(action.dst_slot)),
        torch.full_like(physical, int(victim_fallback_slot)),
        physical,
    )
    if int(source_rank) // int(service_group_size) == destination_rank // int(service_group_size):
        patched = torch.where(
            selected == int(action.src_logical),
            torch.full_like(patched, int(action.dst_slot)),
            patched,
        )
    return patched


@torch.no_grad()
def forward_cover_local_validation_stats(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    slot_to_logical: torch.Tensor,
    action: PlacementAction,
    source_rank: int,
    slots_per_rank: int,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    max_copies: int,
    step: int,
    layer_seed: int,
    patch_remap: bool = False,
    victim_fallback_slot: int | None = None,
    service_group_size: int = 1,
    baseline_communication_counts: torch.Tensor | None = None,
    baseline_assignment_counts: torch.Tensor | None = None,
) -> ForwardCoverLocalValidation:
    """Compute exact local cost deltas by patching only affected tokens.

    A cover can only change routes for its inserted and evicted logical
    experts. Other token rows cancel, so candidate communication deltas are
    obtained from the subset containing either expert while baseline counts
    are built once from the Forward routes.
    """

    if action.kind != "replica":
        raise ValueError("Forward cover validation only accepts replica actions.")
    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if physical.ndim == 1:
        physical = physical.unsqueeze(-1)
    if selected.ndim != 2 or physical.shape != selected.shape:
        raise ValueError("selected_experts and physical_routes must have the same rank-2 shape.")

    layout = slot_to_logical.detach().to(dtype=torch.long).reshape(-1)
    if int(layout.numel()) % int(slots_per_rank) != 0:
        raise ValueError("slot_to_logical does not contain an integral number of ranks.")
    ep_size = int(layout.numel()) // int(slots_per_rank)
    if not 0 <= int(action.dst_slot) < int(layout.numel()):
        raise ValueError("Cover destination slot is outside the physical layout.")
    if int(layout[int(action.dst_slot)].item()) != int(action.dst_logical):
        raise ValueError("Cover victim does not match the current physical layout.")
    if patch_remap:
        if victim_fallback_slot is None or not 0 <= int(victim_fallback_slot) < int(layout.numel()):
            raise ValueError("Victim fallback slot is outside the physical layout.")
        if int(layout[int(victim_fallback_slot)].item()) != int(action.dst_logical):
            raise ValueError("Victim fallback slot does not contain the victim logical expert.")
        if (
            int(service_group_size) <= 0
            or int(service_group_size) > int(ep_size)
            or int(ep_size) % int(service_group_size) != 0
        ):
            raise ValueError("service_group_size must be a positive divisor of ep_size.")

    if baseline_communication_counts is None:
        baseline_communication = _local_packed_counts(
            physical,
            slots_per_rank=slots_per_rank,
            ep_size=ep_size,
            hierarchy_group_sizes=hierarchy_group_sizes,
        )
    else:
        baseline_communication = baseline_communication_counts.to(
            device=physical.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        expected_width = ep_size + sum(
            ep_size // int(size)
            for size in hierarchy_group_sizes
            if 1 < int(size) < ep_size and ep_size % int(size) == 0
        )
        if baseline_communication.ndim != 1 or int(baseline_communication.numel()) != expected_width:
            raise ValueError(
                f"baseline_communication_counts has shape {tuple(baseline_communication.shape)}, "
                f"expected ({expected_width},)."
            )
    if baseline_assignment_counts is None:
        baseline_assignments = _local_assignment_counts(
            physical,
            slots_per_rank=slots_per_rank,
            ep_size=ep_size,
        )
    else:
        baseline_assignments = baseline_assignment_counts.to(
            device=physical.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if baseline_assignments.ndim != 1 or int(baseline_assignments.numel()) != ep_size:
            raise ValueError(
                f"baseline_assignment_counts has shape {tuple(baseline_assignments.shape)}, expected ({ep_size},)."
            )

    if patch_remap:
        destination_rank = int(action.dst_slot) // int(slots_per_rank)
        affected_mask = (selected == int(action.dst_logical)) & (physical == int(action.dst_slot))
        if int(source_rank) // int(service_group_size) == destination_rank // int(service_group_size):
            affected_mask = affected_mask | (selected == int(action.src_logical))
    else:
        affected_mask = (selected == int(action.src_logical)) | (selected == int(action.dst_logical))
    affected_tokens = torch.nonzero(affected_mask.any(dim=1), as_tuple=False).reshape(-1)
    if affected_tokens.numel() == 0:
        return ForwardCoverLocalValidation(
            baseline_communication_counts=baseline_communication,
            communication_count_delta=torch.zeros_like(baseline_communication),
            baseline_assignment_counts=baseline_assignments,
            assignment_count_delta=torch.zeros_like(baseline_assignments),
            affected_tokens=0,
        )

    selected_subset = selected.index_select(0, affected_tokens)
    baseline_subset = physical.index_select(0, affected_tokens)
    if patch_remap:
        assert victim_fallback_slot is not None
        candidate_subset = patch_forward_cover_routes(
            selected_experts=selected_subset,
            physical_routes=baseline_subset,
            action=action,
            source_rank=source_rank,
            slots_per_rank=slots_per_rank,
            victim_fallback_slot=victim_fallback_slot,
            service_group_size=service_group_size,
        )
    else:
        candidate_layout = layout.clone()
        candidate_layout[int(action.dst_slot)] = int(action.src_logical)
        candidate_subset = assign_tokens_to_copies_greedy(
            selected_subset,
            candidate_layout,
            slots_per_rank=slots_per_rank,
            source_ranks=int(source_rank),
            hierarchy_group_sizes=hierarchy_group_sizes,
            num_experts=num_experts,
            token_ordinals=affected_tokens,
            step=int(step),
            layer_seed=int(layer_seed),
            max_copies=max_copies,
        )
    before_subset_communication = _local_packed_counts(
        baseline_subset,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
        hierarchy_group_sizes=hierarchy_group_sizes,
    )
    after_subset_communication = _local_packed_counts(
        candidate_subset,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
        hierarchy_group_sizes=hierarchy_group_sizes,
    )
    before_subset_assignments = _local_assignment_counts(
        baseline_subset,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
    )
    after_subset_assignments = _local_assignment_counts(
        candidate_subset,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
    )
    return ForwardCoverLocalValidation(
        baseline_communication_counts=baseline_communication,
        communication_count_delta=after_subset_communication - before_subset_communication,
        baseline_assignment_counts=baseline_assignments,
        assignment_count_delta=after_subset_assignments - before_subset_assignments,
        affected_tokens=int(affected_tokens.numel()),
    )


@torch.no_grad()
def propose_forward_reuse_covers(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    slot_to_logical: torch.Tensor,
    owner_slots: torch.Tensor,
    local_slot_assignments: torch.Tensor,
    source_rank: int,
    slots_per_rank: int,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    max_copies: int,
    level_weights: Sequence[float] | None = None,
    compute_weight: float = 1.0,
    minimum_gain: float = 0.0,
    victim_mode: str = "minimum",
    service_group_size: int = 1,
    aggregated_statistics: ForwardCoverHeuristicStatistics | None = None,
    max_proposals: int = 1,
) -> tuple[ForwardCoverProposal, ...]:
    """Choose the top positive estimated-gain covers for one target rank.

    The heuristic reuses the current Forward physical routes. Communication
    gain counts routes for which one expert is the sole reason a token reaches
    the current bottleneck rank or hierarchy group. The compute term estimates
    the target-rank assignment change after evicting its least-used redundant
    slot. No candidate route tensor or candidate-by-group table is built.
    """

    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if physical.ndim == 1:
        physical = physical.unsqueeze(-1)
    if selected.ndim != 2 or physical.shape != selected.shape:
        raise ValueError("selected_experts and physical_routes must have the same rank-2 shape.")
    if slots_per_rank <= 0:
        raise ValueError("slots_per_rank must be positive.")
    layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    owners = owner_slots.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if int(layout.numel()) % int(slots_per_rank) != 0:
        raise ValueError("slot_to_logical does not contain an integral number of ranks.")
    ep_size = int(layout.numel()) // int(slots_per_rank)
    if not 0 <= int(source_rank) < ep_size:
        raise ValueError("source_rank is outside the physical layout.")
    if int(owners.numel()) != int(num_experts):
        raise ValueError("owner_slots must contain one slot for every logical expert.")

    service_group_size = int(service_group_size)
    if service_group_size <= 0 or ep_size % service_group_size != 0:
        raise ValueError("service_group_size must be a positive divisor of ep_size.")
    target_start = int(source_rank) * int(slots_per_rank)
    target_end = target_start + int(slots_per_rank)
    active_layout = layout[layout >= 0]
    copy_counts = torch.bincount(active_layout, minlength=num_experts)
    # Any slot whose logical expert has another physical copy can be covered.
    # Canonical ownership is promoted by the executor when such a slot wins.
    empty_slots = [
        slot
        for slot in range(target_start, target_end)
        if int(layout[slot].item()) < 0
    ]
    occupied_cover_slots = [
        slot
        for slot in range(target_start, target_end)
        if int(layout[slot].item()) >= 0 and int(copy_counts[int(layout[slot].item())].item()) > 1
    ]
    # Empty destinations are zero-eviction-loss Covers and therefore take
    # precedence until this target rank has used all of its redundant slots.
    eligible_slots = empty_slots if empty_slots else occupied_cover_slots
    if not eligible_slots:
        return (ForwardCoverProposal(None, 0.0, 0.0, 0.0, 0.0, 0.0),)

    local_counts = local_slot_assignments.detach().to(
        device=selected.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if local_counts.ndim != 1 or int(local_counts.numel()) != int(slots_per_rank):
        raise ValueError("local_slot_assignments must contain one value per local physical slot.")
    eligible_local = torch.tensor(
        [slot - target_start for slot in eligible_slots],
        dtype=torch.long,
        device=selected.device,
    )
    victim_mode = str(victim_mode).strip().lower()
    if victim_mode not in {"minimum", "maximum"}:
        raise ValueError(f"Unsupported Forward-cover victim mode: {victim_mode!r}.")
    if int(max_proposals) <= 0:
        raise ValueError("max_proposals must be positive.")
    eligible_counts = local_counts.index_select(0, eligible_local)
    victim_choice = eligible_counts.argmin() if victim_mode == "minimum" else eligible_counts.argmax()
    victim_local = int(eligible_local[victim_choice].item())
    victim_slot = target_start + victim_local
    victim_logical = int(layout[victim_slot].item())
    victim_count = local_counts[victim_local]

    if aggregated_statistics is None:
        local_statistics = forward_cover_local_heuristic_statistics(
            selected_experts=selected,
            physical_routes=physical,
            source_rank=int(source_rank),
            target_rank=int(source_rank),
            slots_per_rank=int(slots_per_rank),
            ep_size=ep_size,
            hierarchy_group_sizes=hierarchy_group_sizes,
            num_experts=int(num_experts),
            level_weights=level_weights,
        )
    else:
        local_statistics = aggregated_statistics
    communication_benefit = local_statistics.communication_benefit.to(
        device=selected.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    expert_assignments = local_statistics.expert_assignments.to(
        device=selected.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    baseline_units = local_statistics.baseline_communication_units.to(
        device=selected.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if communication_benefit.shape != (int(num_experts),):
        raise ValueError("aggregated communication benefit must contain one value per expert.")
    if expert_assignments.shape != (int(num_experts),):
        raise ValueError("aggregated expert assignments must contain one value per expert.")
    if baseline_units.numel() != 1:
        raise ValueError("aggregated baseline communication units must be scalar.")
    assignment_delta = expert_assignments - victim_count
    normalized_communication_gain = communication_benefit / baseline_units.clamp_min(1.0)
    normalized_assignment_delta = assignment_delta / expert_assignments.sum().clamp_min(1.0)
    estimated_gain = normalized_communication_gain - float(compute_weight) * normalized_assignment_delta

    service_start_rank = (int(source_rank) // service_group_size) * service_group_size
    service_start_slot = service_start_rank * int(slots_per_rank)
    service_end_slot = (service_start_rank + service_group_size) * int(slots_per_rank)
    target_logicals = layout[service_start_slot:service_end_slot]
    already_local = torch.zeros((num_experts,), dtype=torch.bool)
    active_target = target_logicals[target_logicals >= 0]
    if active_target.numel() > 0:
        already_local[active_target] = True
    valid = (~already_local) & (copy_counts < int(max_copies))
    if victim_logical >= 0:
        valid[victim_logical] = False
    valid_device = valid.to(device=selected.device, non_blocking=True)
    negative_infinity = torch.full_like(estimated_gain, -torch.inf)
    masked_gain = torch.where(valid_device, estimated_gain, negative_infinity)
    finite_count = int(torch.isfinite(masked_gain).sum().item())
    proposal_count = min(int(max_proposals), finite_count)
    if proposal_count <= 0:
        return (
            ForwardCoverProposal(
                None,
                0.0,
                0.0,
                0.0,
                float(baseline_units.item()),
                float(victim_count.item()),
            ),
        )
    top_gains, top_experts = torch.topk(masked_gain, k=proposal_count, largest=True, sorted=True)
    proposals: list[ForwardCoverProposal] = []
    for gain_tensor, expert_tensor in zip(top_gains, top_experts, strict=True):
        gain = float(gain_tensor.item())
        # Empty-slot seeding has no eviction loss and is protected by the
        # winner-only exact global cost validation in the caller.  Do not let
        # this intentionally cheap local heuristic suppress every candidate:
        # its assignment penalty is only a ranking signal and can be
        # pessimistic before the first replicas exist.  Occupied Cover keeps
        # the normal positive-estimated-gain filter.
        if not bool(torch.isfinite(gain_tensor).item()) or (
            victim_logical >= 0 and gain <= float(minimum_gain)
        ):
            continue
        expert = int(expert_tensor.item())
        proposals.append(
            ForwardCoverProposal(
                action=PlacementAction(
                    kind="replica",
                    src_slot=int(owners[expert].item()),
                    dst_slot=victim_slot,
                    src_logical=expert,
                    dst_logical=victim_logical,
                ),
                estimated_gain=gain,
                communication_gain=float(normalized_communication_gain[expert].item()),
                assignment_delta=float(assignment_delta[expert].item()),
                baseline_communication_units=float(baseline_units.item()),
                victim_assignment_count=float(victim_count.item()),
            )
        )
    if proposals:
        return tuple(proposals)
    best_gain = float(top_gains[0].item()) if top_gains.numel() > 0 else 0.0
    return (
        ForwardCoverProposal(
            None,
            best_gain,
            0.0,
            0.0,
            float(baseline_units.item()),
            float(victim_count.item()),
        ),
    )


@torch.no_grad()
def propose_forward_reuse_cover(**kwargs: object) -> ForwardCoverProposal:
    """Backward-compatible single-proposal wrapper."""

    return propose_forward_reuse_covers(**kwargs, max_proposals=1)[0]  # type: ignore[arg-type]


def rotating_service_target_rank(
    target_ranks: Sequence[int],
    *,
    layer_index: int,
    step: int,
    service_group_size: int,
) -> int:
    """Rotate across service groups before revisiting another rank in one group."""

    ranks = tuple(int(rank) for rank in target_ranks)
    if not ranks:
        raise ValueError("target_ranks must not be empty.")
    group_size = int(service_group_size)
    if group_size <= 0:
        raise ValueError("service_group_size must be positive.")
    ranks_by_group: dict[int, list[int]] = {}
    for rank in ranks:
        ranks_by_group.setdefault(rank // group_size, []).append(rank)
    group_ids = tuple(sorted(ranks_by_group))
    rotation = int(layer_index) + int(step)
    group_id = group_ids[rotation % len(group_ids)]
    group_ranks = tuple(sorted(ranks_by_group[group_id]))
    lane_round = rotation // len(group_ids)
    return group_ranks[lane_round % len(group_ranks)]
