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
class ForwardCoverLocalValidation:
    """Local exact statistics for one proposed cover action."""

    baseline_communication_counts: torch.Tensor
    communication_count_delta: torch.Tensor
    baseline_assignment_counts: torch.Tensor
    assignment_count_delta: torch.Tensor
    affected_tokens: int


def _first_and_sole_masks(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first occurrence and singleton masks along a token's top-k row."""

    top_k = int(groups.shape[1])
    equal = groups.unsqueeze(-1) == groups.unsqueeze(-2)
    indices = torch.arange(top_k, dtype=torch.long, device=groups.device)
    earlier = indices.view(1, 1, top_k) < indices.view(1, top_k, 1)
    first = ~(equal & earlier).any(dim=-1)
    sole = equal.sum(dim=-1) == 1
    return first, sole


def _bottleneck_unique_benefit(
    *,
    selected_experts: torch.Tensor,
    destination_groups: torch.Tensor,
    source_group: int,
    num_groups: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count expert routes that uniquely sustain a bottleneck remote group."""

    first, sole = _first_and_sole_masks(destination_groups)
    remote = destination_groups != int(source_group)
    first_remote = first & remote
    counts = torch.zeros((num_groups,), dtype=torch.float32, device=destination_groups.device)
    if destination_groups.numel() > 0:
        counts.scatter_add_(
            0,
            destination_groups[first_remote],
            torch.ones_like(destination_groups[first_remote], dtype=torch.float32),
        )
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


@torch.no_grad()
def patch_forward_cover_routes(
    *,
    selected_experts: torch.Tensor,
    physical_routes: torch.Tensor,
    action: PlacementAction,
    source_rank: int,
    slots_per_rank: int,
    victim_fallback_slot: int,
) -> torch.Tensor:
    """Patch only routes whose serving copy changes under one local cover.

    The inserted copy serves its destination rank only. Routes that used the
    overwritten victim slot fall back to the victim's canonical owner. This is
    the token-level equivalent of updating the persistent source-rank LUT.
    """

    if action.kind != "replica":
        raise ValueError("Forward route patching only accepts replica actions.")
    selected = selected_experts.to(dtype=torch.long)
    physical = physical_routes.to(device=selected.device, dtype=torch.long, non_blocking=True)
    if selected.shape != physical.shape:
        raise ValueError("selected_experts and physical_routes must have the same shape.")

    destination_rank = int(action.dst_slot) // int(slots_per_rank)
    patched = torch.where(
        (selected == int(action.dst_logical)) & (physical == int(action.dst_slot)),
        torch.full_like(physical, int(victim_fallback_slot)),
        physical,
    )
    if int(source_rank) == destination_rank:
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

    baseline_communication = _local_packed_counts(
        physical,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
        hierarchy_group_sizes=hierarchy_group_sizes,
    )
    baseline_assignments = _local_assignment_counts(
        physical,
        slots_per_rank=slots_per_rank,
        ep_size=ep_size,
    )

    if patch_remap:
        destination_rank = int(action.dst_slot) // int(slots_per_rank)
        affected_mask = (selected == int(action.dst_logical)) & (physical == int(action.dst_slot))
        if int(source_rank) == destination_rank:
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
def propose_forward_reuse_cover(
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
) -> ForwardCoverProposal:
    """Choose one positive estimated-gain cover for the source/target rank.

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

    target_start = int(source_rank) * int(slots_per_rank)
    target_end = target_start + int(slots_per_rank)
    active_layout = layout[layout >= 0]
    copy_counts = torch.bincount(active_layout, minlength=num_experts)
    # Any slot whose logical expert has another physical copy can be covered.
    # Canonical ownership is promoted by the executor when such a slot wins.
    eligible_slots = [
        slot
        for slot in range(target_start, target_end)
        if int(layout[slot].item()) >= 0 and int(copy_counts[int(layout[slot].item())].item()) > 1
    ]
    if not eligible_slots:
        return ForwardCoverProposal(None, 0.0, 0.0, 0.0, 0.0, 0.0)

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
    victim_choice = local_counts.index_select(0, eligible_local).argmin()
    victim_local = int(eligible_local[victim_choice].item())
    victim_slot = target_start + victim_local
    victim_logical = int(layout[victim_slot].item())
    victim_count = local_counts[victim_local]

    destination_ranks = torch.div(physical, int(slots_per_rank), rounding_mode="floor")
    weights = tuple(float(value) for value in level_weights) if level_weights is not None else ()
    expected_levels = 1 + len(
        [size for size in hierarchy_group_sizes if 1 < int(size) < ep_size and ep_size % int(size) == 0]
    )
    if weights and len(weights) != expected_levels:
        raise ValueError(f"level_weights has {len(weights)} values, expected {expected_levels}.")
    if not weights:
        weights = (1.0,) * expected_levels

    rank_benefit, rank_maximum = _bottleneck_unique_benefit(
        selected_experts=selected,
        destination_groups=destination_ranks,
        source_group=int(source_rank),
        num_groups=ep_size,
        num_experts=num_experts,
    )
    communication_benefit = weights[0] * rank_benefit
    baseline_units = weights[0] * rank_maximum

    level_index = 1
    for raw_size in hierarchy_group_sizes:
        group_size = int(raw_size)
        if not 1 < group_size < ep_size or ep_size % group_size != 0:
            continue
        destination_groups = torch.div(destination_ranks, group_size, rounding_mode="floor")
        group_benefit, group_maximum = _bottleneck_unique_benefit(
            selected_experts=selected,
            destination_groups=destination_groups,
            source_group=int(source_rank) // group_size,
            num_groups=ep_size // group_size,
            num_experts=num_experts,
        )
        communication_benefit = communication_benefit + weights[level_index] * group_benefit
        baseline_units = baseline_units + weights[level_index] * group_maximum
        level_index += 1

    expert_assignments = torch.bincount(selected.reshape(-1), minlength=num_experts).to(torch.float32)
    assignment_delta = expert_assignments - victim_count
    normalized_communication_gain = communication_benefit / baseline_units.clamp_min(1.0)
    normalized_assignment_delta = assignment_delta / local_counts.sum().clamp_min(1.0)
    estimated_gain = normalized_communication_gain - float(compute_weight) * normalized_assignment_delta

    target_logicals = layout[target_start:target_end]
    already_local = torch.zeros((num_experts,), dtype=torch.bool)
    active_target = target_logicals[target_logicals >= 0]
    if active_target.numel() > 0:
        already_local[active_target] = True
    valid = (~already_local) & (copy_counts < int(max_copies))
    valid[victim_logical] = False
    valid_device = valid.to(device=selected.device, non_blocking=True)
    negative_infinity = torch.full_like(estimated_gain, -torch.inf)
    masked_gain = torch.where(valid_device, estimated_gain, negative_infinity)
    best_gain, best_expert_tensor = masked_gain.max(dim=0)
    if not bool(torch.isfinite(best_gain).item()) or float(best_gain.item()) <= float(minimum_gain):
        return ForwardCoverProposal(
            None,
            float(best_gain.item()) if bool(torch.isfinite(best_gain).item()) else 0.0,
            0.0,
            0.0,
            float(baseline_units.item()),
            float(victim_count.item()),
        )

    best_expert = int(best_expert_tensor.item())
    source_slot = int(owners[best_expert].item())
    action = PlacementAction(
        kind="replica",
        src_slot=source_slot,
        dst_slot=victim_slot,
        src_logical=best_expert,
        dst_logical=victim_logical,
    )
    return ForwardCoverProposal(
        action=action,
        estimated_gain=float(best_gain.item()),
        communication_gain=float(normalized_communication_gain[best_expert].item()),
        assignment_delta=float(assignment_delta[best_expert].item()),
        baseline_communication_units=float(baseline_units.item()),
        victim_assignment_count=float(victim_count.item()),
    )
