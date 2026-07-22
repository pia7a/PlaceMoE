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

"""Communication-only greedy planning for swaps and redundant expert covers."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .perf_model import HierMoEPerfModel
from .planner import PlacementAction, PlacementCost, PlacementPlan, _hierarchy_distance, _route_hash
from .topology import Hierarchy


GREEDY_COVER_ALGORITHM_VERSION = "hiermoe-greedy-cover-p1-v1"
_ACTION_SWAP = 0
_ACTION_COVER = 1


@dataclass(frozen=True)
class _ScoredLayouts:
    communication: torch.Tensor
    peak_rank: torch.Tensor
    selected_dim: torch.Tensor
    baseline_physical_routes: torch.Tensor


def _reduce_sum(
    tensor: torch.Tensor,
    reducer: Callable[[torch.Tensor], torch.Tensor | None] | None,
) -> torch.Tensor:
    if reducer is None:
        return tensor
    reduced = tensor.clone()
    result = reducer(reduced)
    return reduced if result is None else result


def _mapping_level_sizes(num_ranks: int, hierarchy_group_sizes: Sequence[int]) -> tuple[int, ...]:
    coarse_levels = sorted(
        {
            int(group_size)
            for group_size in hierarchy_group_sizes
            if 1 < int(group_size) < int(num_ranks) and num_ranks % int(group_size) == 0
        },
        reverse=True,
    )
    return (*coarse_levels, 1)


def assign_tokens_to_copies_greedy(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    token_ordinals: torch.Tensor | None = None,
    step: int = 0,
    layer_seed: int = 0,
    max_copies: int = 8,
) -> torch.Tensor:
    """Map every logical route to its nearest available physical copy.

    Hierarchy distance is compared lexicographically from coarse groups to the
    source rank. Stable route hashes distribute exact ties. The route choice is
    independent across logical experts, which makes swap/cover deltas sparse:
    only routes of the two experts changed by an action can move.
    """

    original_selected_ndim = selected_experts.ndim
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if selected.ndim != 2:
        raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")

    layouts = slot_to_logical.to(device=selected.device, dtype=torch.long, non_blocking=True)
    squeeze_layout = layouts.ndim == 1
    if squeeze_layout:
        layouts = layouts.unsqueeze(0)
    if layouts.ndim != 2:
        raise ValueError(f"slot_to_logical must have rank 1 or 2, got shape={tuple(layouts.shape)}.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if slots_per_rank <= 0 or layouts.shape[1] % int(slots_per_rank) != 0:
        raise ValueError("The physical layout must contain an integral number of ranks.")

    batch, num_slots = layouts.shape
    copy_limit = max(1, min(int(max_copies), num_slots))
    logical_ids = torch.arange(num_experts, dtype=torch.long, device=selected.device)
    slot_ids = torch.arange(num_slots, dtype=torch.long, device=selected.device).view(1, num_slots, 1)
    matches = layouts.unsqueeze(-1) == logical_ids.view(1, 1, num_experts)
    masked_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
    copy_slots = masked_slots.sort(dim=1).values[:, :copy_limit].transpose(1, 2).contiguous()
    copy_valid = copy_slots < num_slots
    if bool((copy_slots[:, :, 0] >= num_slots).any().item()):
        raise ValueError("Every logical expert must retain at least one physical copy.")

    num_tokens, top_k = selected.shape
    if isinstance(source_ranks, int):
        sources = torch.full((num_tokens,), int(source_ranks), dtype=torch.long, device=selected.device)
    else:
        sources = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if sources.numel() != num_tokens:
            raise ValueError(f"source_ranks has {sources.numel()} values for {num_tokens} tokens.")
    ordinals = (
        torch.arange(num_tokens, dtype=torch.long, device=selected.device)
        if token_ordinals is None
        else token_ordinals.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
    )
    if ordinals.numel() != num_tokens:
        raise ValueError(f"token_ordinals has {ordinals.numel()} values for {num_tokens} tokens.")

    selected_slots = copy_slots.index_select(1, selected.reshape(-1)).view(batch, num_tokens, top_k, copy_limit)
    selected_valid = copy_valid.index_select(1, selected.reshape(-1)).view(batch, num_tokens, top_k, copy_limit)
    safe_selected_slots = selected_slots.clamp(max=max(0, num_slots - 1))
    copy_ranks = torch.div(safe_selected_slots, int(slots_per_rank), rounding_mode="floor")
    distance = _hierarchy_distance(sources, copy_ranks, hierarchy_group_sizes)
    invalid_score = torch.iinfo(torch.long).max
    score = torch.where(selected_valid, distance, torch.full_like(distance, invalid_score))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = selected_valid & (score == minimum)
    tie_order = tied.to(torch.long).cumsum(dim=-1) - 1
    route_hashes = _route_hash(
        selected,
        token_ordinals=ordinals,
        step=step,
        layer_seed=layer_seed,
    )
    tie_count = tied.sum(dim=-1, keepdim=True).clamp_min(1)
    tie_target = torch.remainder(route_hashes.view(1, num_tokens, top_k, 1), tie_count)
    chosen_copy = (tied & (tie_order == tie_target)).to(torch.long).argmax(dim=-1)
    physical = safe_selected_slots.gather(3, chosen_copy.unsqueeze(-1)).squeeze(-1)

    if squeeze_layout:
        physical = physical[0]
    if original_selected_ndim == 1:
        physical = physical.squeeze(-1)
    return physical


class GreedyCommunicationPlanner:
    """Evaluate empty covers, swaps, and occupied covers with one shared collective."""

    def __init__(
        self,
        *,
        hierarchy: Hierarchy,
        perf_model: HierMoEPerfModel,
        hidden_size: int,
        bytes_per_element: int,
        slots_per_rank: int,
        smooth_max_gamma: float = 10.0,
        reducer: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        candidate_chunk_size: int = 128,
        process_group: dist.ProcessGroup | None = None,
        max_copies: int = 4,
    ) -> None:
        if smooth_max_gamma <= 0:
            raise ValueError("smooth_max_gamma must be positive.")
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.hidden_size = int(hidden_size)
        self.bytes_per_element = int(bytes_per_element)
        self.slots_per_rank = int(slots_per_rank)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.reducer = reducer
        self.candidate_chunk_size = max(1, int(candidate_chunk_size))
        self.process_group = process_group
        self.max_copies = max(1, int(max_copies))

    @property
    def ep_size(self) -> int:
        return int(self.hierarchy.ep_size)

    @property
    def payload_bytes(self) -> int:
        return self.hidden_size * self.bytes_per_element

    def _count_widths(self) -> tuple[int, ...]:
        widths = [self.ep_size]
        for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(size)
            if size <= 0 or self.ep_size % size != 0:
                raise ValueError(f"Invalid hierarchy group size {size} for EP size {self.ep_size}.")
            widths.append(self.ep_size // size)
        return tuple(widths)

    def _local_packed_counts(self, physical_slots: torch.Tensor) -> torch.Tensor:
        physical = physical_slots
        if physical.ndim == 2:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
        batch, num_tokens, top_k = ranks.shape
        rows: list[torch.Tensor] = []

        rank_hits = torch.zeros((batch * num_tokens, self.ep_size), dtype=torch.bool, device=ranks.device)
        rank_hits.scatter_(1, ranks.reshape(batch * num_tokens, top_k), True)
        rows.append(rank_hits.view(batch, num_tokens, self.ep_size).sum(dim=1).to(torch.float32))
        for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(size)
            groups = torch.div(ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            group_hits = torch.zeros((batch * num_tokens, num_groups), dtype=torch.bool, device=ranks.device)
            group_hits.scatter_(1, groups.reshape(batch * num_tokens, top_k), True)
            rows.append(group_hits.view(batch, num_tokens, num_groups).sum(dim=1).to(torch.float32))
        return torch.cat(rows, dim=1)

    def _communication_cost(self, packed_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        widths = self._count_widths()
        rows = packed_counts.split(widths, dim=1)
        rank_counts = rows[0]
        rank_max = rank_counts.max(dim=1).values
        dimensions = [
            self.perf_model.a2a.alpha + float(self.ep_size * self.payload_bytes) * rank_max * self.perf_model.a2a.beta
        ]
        max_dim = max(1, int(self.hierarchy.selected_dim))
        for dim in range(2, max_dim + 1):
            total = torch.zeros_like(rank_max)
            previous_size = 1
            for level_index, raw_size in enumerate(self.hierarchy.group_sizes[: dim - 1]):
                size = int(raw_size)
                link = self.perf_model.inter[min(level_index, len(self.perf_model.inter) - 1)]
                group_max = rows[level_index + 1].max(dim=1).values
                scale = float((size / previous_size) * self.payload_bytes)
                total = total + link.alpha + scale * group_max * link.beta
                previous_size = size
            intra_scale = float((self.ep_size / previous_size) * self.payload_bytes)
            total = total + self.perf_model.intra.alpha + intra_scale * rank_max * self.perf_model.intra.beta
            dimensions.append(total)
        per_dim = torch.stack(dimensions, dim=1)
        communication = torch.logsumexp(per_dim * self.smooth_max_gamma, dim=1) / self.smooth_max_gamma
        return communication, rank_counts.argmax(dim=1), per_dim.argmax(dim=1) + 1

    def _score_layouts(
        self,
        selected: torch.Tensor,
        layouts: torch.Tensor,
        *,
        source_ranks: torch.Tensor,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
    ) -> _ScoredLayouts:
        local_rows: list[torch.Tensor] = []
        baseline_physical: torch.Tensor | None = None
        for start in range(0, layouts.shape[0], self.candidate_chunk_size):
            physical = assign_tokens_to_copies_greedy(
                selected,
                layouts[start : start + self.candidate_chunk_size],
                slots_per_rank=self.slots_per_rank,
                source_ranks=source_ranks,
                hierarchy_group_sizes=self.hierarchy.group_sizes,
                num_experts=num_experts,
                token_ordinals=token_ordinals,
                step=step,
                layer_seed=layer_seed,
                max_copies=self.max_copies,
            )
            if start == 0:
                baseline_physical = physical[0].clone()
            local_rows.append(self._local_packed_counts(physical))
        assert baseline_physical is not None
        global_counts = _reduce_sum(torch.cat(local_rows, dim=0), self.reducer)
        communication, peak_rank, selected_dim = self._communication_cost(global_counts)
        return _ScoredLayouts(
            communication=communication,
            peak_rank=peak_rank,
            selected_dim=selected_dim,
            baseline_physical_routes=baseline_physical,
        )

    def _use_sharded_candidate_collective(self, device: torch.device) -> bool:
        if self.process_group is None or not dist.is_initialized() or self.ep_size <= 1:
            return False
        if dist.get_world_size(self.process_group) != self.ep_size:
            return False
        backend = str(dist.get_backend(self.process_group)).lower().rsplit(".", maxsplit=1)[-1]
        return backend != "gloo" or device.type == "cpu"

    def _global_action_costs(
        self,
        baseline_local: torch.Tensor,
        candidate_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_local.numel() == 0 or not self._use_sharded_candidate_collective(baseline_local.device):
            local_counts = torch.cat((baseline_local, candidate_local), dim=0)
            global_counts = _reduce_sum(local_counts, self.reducer)
            return self._communication_cost(global_counts)

        assert self.process_group is not None
        baseline_global = baseline_local.clone()
        dist.all_reduce(baseline_global, op=dist.ReduceOp.SUM, group=self.process_group)
        baseline_communication, baseline_peak, baseline_dim = self._communication_cost(baseline_global)

        group_size = dist.get_world_size(self.process_group)
        num_candidates, width = candidate_local.shape
        shard_rows = (num_candidates + group_size - 1) // group_size
        padded_rows = shard_rows * group_size
        if padded_rows != num_candidates:
            padding = torch.zeros(
                (padded_rows - num_candidates, width),
                dtype=candidate_local.dtype,
                device=candidate_local.device,
            )
            reduce_input = torch.cat((candidate_local, padding), dim=0)
        else:
            reduce_input = candidate_local
        reduced_shard = torch.empty(
            (shard_rows, width),
            dtype=candidate_local.dtype,
            device=candidate_local.device,
        )
        dist.reduce_scatter_tensor(
            reduced_shard,
            reduce_input.contiguous(),
            op=dist.ReduceOp.SUM,
            group=self.process_group,
        )
        shard_communication, shard_peak, shard_dim = self._communication_cost(reduced_shard)
        shard_metrics = torch.stack(
            (
                shard_communication,
                shard_peak.to(shard_communication.dtype),
                shard_dim.to(shard_communication.dtype),
            ),
            dim=1,
        ).contiguous()
        gathered_metrics = torch.empty(
            (padded_rows, 3),
            dtype=shard_metrics.dtype,
            device=shard_metrics.device,
        )
        dist.all_gather_into_tensor(gathered_metrics, shard_metrics, group=self.process_group)
        candidate_metrics = gathered_metrics[:num_candidates]
        communication = torch.cat((baseline_communication, candidate_metrics[:, 0]), dim=0)
        peak_rank = torch.cat((baseline_peak, candidate_metrics[:, 1].to(baseline_peak.dtype)), dim=0)
        selected_dim = torch.cat((baseline_dim, candidate_metrics[:, 2].to(baseline_dim.dtype)), dim=0)
        return communication, peak_rank, selected_dim

    def _copy_table(self, layout: torch.Tensor, num_experts: int) -> torch.Tensor:
        num_slots = int(layout.numel())
        logical_ids = torch.arange(num_experts, dtype=torch.long, device=layout.device)
        slot_ids = torch.arange(num_slots, dtype=torch.long, device=layout.device).view(-1, 1)
        matches = layout.view(-1, 1) == logical_ids.view(1, -1)
        counts = matches.sum(dim=0)
        width = max(1, int(counts.max().item()))
        masked = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
        return masked.sort(dim=0).values[:width].transpose(0, 1).contiguous()

    def _candidate_copy_options(
        self,
        layout: torch.Tensor,
        copy_slots: torch.Tensor,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_slots = int(layout.numel())
        lhs = rows[:, 3]
        rhs_valid = rows[:, 4] >= 0
        rhs = rows[:, 4].clamp_min(0)
        invalid = torch.full((rows.shape[0], 1), num_slots, dtype=torch.long, device=layout.device)
        lhs_options = torch.cat((copy_slots.index_select(0, lhs), invalid), dim=1)
        rhs_options = torch.cat((copy_slots.index_select(0, rhs), invalid), dim=1)

        swap = rows[:, 0] == _ACTION_SWAP
        cover = ~swap
        lhs_options = torch.where(
            swap.view(-1, 1) & (lhs_options == rows[:, 1].view(-1, 1)),
            rows[:, 2].view(-1, 1),
            lhs_options,
        )
        lhs_options[:, -1] = torch.where(cover, rows[:, 2], lhs_options[:, -1])
        rhs_options = torch.where(
            swap.view(-1, 1) & (rhs_options == rows[:, 2].view(-1, 1)),
            rows[:, 1].view(-1, 1),
            rhs_options,
        )
        rhs_options = torch.where(
            cover.view(-1, 1) & (rhs_options == rows[:, 2].view(-1, 1)),
            torch.full_like(rhs_options, num_slots),
            rhs_options,
        )
        return lhs_options.sort(dim=1).values, rhs_options.sort(dim=1).values, rhs_valid

    def _candidate_route_slots(
        self,
        options: torch.Tensor,
        route_hashes: torch.Tensor,
        source_ranks: torch.Tensor,
        num_slots: int,
    ) -> torch.Tensor:
        valid = options < num_slots
        safe_slots = options.clamp(max=max(0, num_slots - 1))
        copy_ranks = torch.div(safe_slots, self.slots_per_rank, rounding_mode="floor")
        expanded_ranks = copy_ranks.view(options.shape[0], 1, 1, options.shape[1]).expand(
            -1, source_ranks.numel(), 1, -1
        )
        distance = _hierarchy_distance(source_ranks, expanded_ranks, self.hierarchy.group_sizes).squeeze(2)
        distance = torch.where(
            valid.view(options.shape[0], 1, -1),
            distance,
            torch.full_like(distance, torch.iinfo(torch.long).max),
        )
        minimum = distance.min(dim=-1, keepdim=True).values
        tied = valid.view(options.shape[0], 1, -1) & (distance == minimum)
        tie_order = tied.to(torch.long).cumsum(dim=-1) - 1
        tie_count = tied.sum(dim=-1).clamp_min(1)
        target = torch.remainder(route_hashes, tie_count)
        chosen = (tied & (tie_order == target.unsqueeze(-1))).to(torch.long).argmax(dim=-1)
        return safe_slots.gather(1, chosen)

    def _candidate_route_ranks(
        self,
        options: torch.Tensor,
        route_hashes: torch.Tensor,
        source_ranks: torch.Tensor,
        num_slots: int,
    ) -> torch.Tensor:
        slots = self._candidate_route_slots(options, route_hashes, source_ranks, num_slots)
        return torch.div(slots, self.slots_per_rank, rounding_mode="floor")

    def _apply_action_routes(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        row: torch.Tensor,
        baseline_physical: torch.Tensor,
        *,
        source_ranks: torch.Tensor,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
    ) -> torch.Tensor:
        """Apply one winning action by remapping only its affected experts."""

        row = row.view(1, 5)
        copy_slots = self._copy_table(layout, num_experts)
        lhs_options, rhs_options, rhs_valid = self._candidate_copy_options(layout, copy_slots, row)
        num_tokens = selected.shape[0]
        lhs = row[0, 3]
        lhs_selected = lhs.expand(num_tokens, 1)
        lhs_hash = _route_hash(
            lhs_selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        ).transpose(0, 1)
        lhs_slots = self._candidate_route_slots(
            lhs_options,
            lhs_hash,
            source_ranks,
            int(layout.numel()),
        )[0]
        updated = torch.where(selected == lhs, lhs_slots.view(-1, 1), baseline_physical)

        if bool(rhs_valid[0].item()):
            rhs = row[0, 4]
            rhs_selected = rhs.expand(num_tokens, 1)
            rhs_hash = _route_hash(
                rhs_selected,
                token_ordinals=token_ordinals,
                step=step,
                layer_seed=layer_seed,
            ).transpose(0, 1)
            rhs_slots = self._candidate_route_slots(
                rhs_options,
                rhs_hash,
                source_ranks,
                int(layout.numel()),
            )[0]
            updated = torch.where(selected == rhs, rhs_slots.view(-1, 1), updated)
        return updated

    def _token_level_occupancies(self, physical_slots: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ranks = torch.div(physical_slots, self.slots_per_rank, rounding_mode="floor")
        num_tokens, top_k = ranks.shape
        occupancies = []
        level_sizes = (1,) + tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        )
        for size in level_sizes:
            groups = torch.div(ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            counts = torch.zeros((num_tokens, num_groups), dtype=torch.int32, device=ranks.device)
            counts.scatter_add_(1, groups, torch.ones((num_tokens, top_k), dtype=torch.int32, device=ranks.device))
            occupancies.append(counts)
        return tuple(occupancies)

    def _candidate_local_deltas(
        self,
        selected: torch.Tensor,
        rows: torch.Tensor,
        *,
        layout: torch.Tensor,
        copy_slots: torch.Tensor,
        occupancies: Sequence[torch.Tensor],
        source_ranks: torch.Tensor,
        route_hash_by_expert: torch.Tensor,
        multiplicity_by_expert: torch.Tensor,
        rank_by_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_slots = int(layout.numel())
        lhs = rows[:, 3]
        rhs = rows[:, 4].clamp_min(0)
        lhs_options, rhs_options, rhs_valid = self._candidate_copy_options(layout, copy_slots, rows)

        lhs_hash = route_hash_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_hash = route_hash_by_expert.index_select(1, rhs).transpose(0, 1)
        lhs_multiplicity = multiplicity_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_multiplicity = multiplicity_by_expert.index_select(1, rhs).transpose(0, 1)
        rhs_multiplicity *= rhs_valid.view(-1, 1).to(rhs_multiplicity.dtype)
        lhs_old = rank_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_old = rank_by_expert.index_select(1, rhs).transpose(0, 1)
        lhs_new = self._candidate_route_ranks(lhs_options, lhs_hash, source_ranks, num_slots)
        rhs_new = self._candidate_route_ranks(rhs_options, rhs_hash, source_ranks, num_slots)
        lhs_multiplicity *= lhs_new.ne(lhs_old).to(lhs_multiplicity.dtype)
        rhs_multiplicity *= rhs_new.ne(rhs_old).to(rhs_multiplicity.dtype)

        packed_deltas = []
        level_sizes = (1,) + tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        )
        for occupancy, size in zip(occupancies, level_sizes, strict=True):
            num_groups = int(occupancy.shape[1])
            groups = (
                torch.div(lhs_old, size, rounding_mode="floor"),
                torch.div(lhs_new, size, rounding_mode="floor"),
                torch.div(rhs_old, size, rounding_mode="floor"),
                torch.div(rhs_new, size, rounding_mode="floor"),
            )
            values = (
                -lhs_multiplicity,
                lhs_multiplicity,
                -rhs_multiplicity,
                rhs_multiplicity,
            )
            delta = torch.zeros((rows.shape[0], num_groups), dtype=torch.float32, device=selected.device)
            for position, group in enumerate(groups):
                combined = torch.zeros_like(lhs_multiplicity)
                for other_group, value in zip(groups, values, strict=True):
                    combined += torch.where(other_group == group, value, torch.zeros_like(value))
                current = occupancy.unsqueeze(0).expand(rows.shape[0], -1, -1)
                current = current.gather(2, group.unsqueeze(-1)).squeeze(-1)
                changed = ((current + combined) > 0).to(torch.float32) - (current > 0).to(torch.float32)
                if position:
                    first = torch.ones_like(changed, dtype=torch.bool)
                    for previous in groups[:position]:
                        first &= group != previous
                    changed *= first.to(changed.dtype)
                delta.scatter_add_(1, group, changed)
            packed_deltas.append(delta)
        return torch.cat(packed_deltas, dim=1)

    def _fused_candidate_local_deltas(
        self,
        selected: torch.Tensor,
        rows: torch.Tensor,
        *,
        layout: torch.Tensor,
        copy_slots: torch.Tensor,
        physical: torch.Tensor,
        occupancies: Sequence[torch.Tensor],
        source_ranks: torch.Tensor,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
    ) -> torch.Tensor | None:
        if (
            rows.numel() == 0
            or
            selected.device.type != "npu"
            or selected.shape[0] > 16384
            or self.ep_size > 64
            or copy_slots.shape[1] > 8
            or not 1 <= len(occupancies) <= 3
            or source_ranks.numel() == 0
            or not bool((source_ranks == source_ranks[0]).all().item())
        ):
            return None
        from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops

        extension = get_hiermoe_planner_npu_ops()
        if extension is None or not hasattr(extension, "cover_score"):
            return None
        route_indices, multiplicities, token_counts = extension.replica_prepare(selected.contiguous(), num_experts)
        route_hashes = _route_hash(
            selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
        token_group_counts = torch.cat(tuple(occupancies), dim=1).contiguous()
        level_sizes = (1,) + tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        )
        padded_sizes = level_sizes + (1,) * (3 - len(level_sizes))
        output = extension.cover_score(
            selected.contiguous(),
            route_indices,
            multiplicities,
            token_counts,
            torch.div(physical, self.slots_per_rank, rounding_mode="floor").reshape(-1).contiguous(),
            route_hashes.reshape(-1).contiguous(),
            token_group_counts,
            copy_slots.contiguous(),
            rows.contiguous(),
            int(layout.numel()),
            self.slots_per_rank,
            self.ep_size,
            int(source_ranks[0].item()),
            len(level_sizes),
            padded_sizes[0],
            padded_sizes[1],
            padded_sizes[2],
            selected.shape[1],
        )
        return output[:, : token_group_counts.shape[1]].to(torch.float32)

    def _score_actions(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        rows: torch.Tensor,
        *,
        source_ranks: torch.Tensor,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
    ) -> _ScoredLayouts:
        physical = assign_tokens_to_copies_greedy(
            selected,
            layout,
            slots_per_rank=self.slots_per_rank,
            source_ranks=source_ranks,
            hierarchy_group_sizes=self.hierarchy.group_sizes,
            num_experts=num_experts,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            max_copies=self.max_copies,
        )
        occupancies = self._token_level_occupancies(physical)
        baseline_local = torch.cat(
            tuple((counts > 0).sum(dim=0, keepdim=True).to(torch.float32) for counts in occupancies),
            dim=1,
        )
        copy_slots = self._copy_table(layout, num_experts)
        fused_delta = self._fused_candidate_local_deltas(
            selected,
            rows,
            layout=layout,
            copy_slots=copy_slots,
            physical=physical,
            occupancies=occupancies,
            source_ranks=source_ranks,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            num_experts=num_experts,
        )
        if rows.numel() == 0:
            candidate_local = []
        elif fused_delta is not None:
            candidate_local = [baseline_local + fused_delta]
        else:
            route_hashes = _route_hash(
                selected, token_ordinals=token_ordinals, step=step, layer_seed=layer_seed
            )
            route_hash_by_expert = torch.zeros(
                (selected.shape[0], num_experts), dtype=torch.long, device=selected.device
            )
            route_hash_by_expert.scatter_(1, selected, route_hashes)
            multiplicity_by_expert = torch.zeros(
                (selected.shape[0], num_experts), dtype=torch.int32, device=selected.device
            )
            multiplicity_by_expert.scatter_add_(1, selected, torch.ones_like(selected, dtype=torch.int32))
            route_ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
            rank_by_expert = torch.zeros_like(route_hash_by_expert)
            rank_by_expert.scatter_(1, selected, route_ranks)
            candidate_local = []
            for start in range(0, rows.shape[0], self.candidate_chunk_size):
                delta = self._candidate_local_deltas(
                    selected,
                    rows[start : start + self.candidate_chunk_size],
                    layout=layout,
                    copy_slots=copy_slots,
                    occupancies=occupancies,
                    source_ranks=source_ranks,
                    route_hash_by_expert=route_hash_by_expert,
                    multiplicity_by_expert=multiplicity_by_expert,
                    rank_by_expert=rank_by_expert,
                )
                candidate_local.append(baseline_local + delta)
        candidates = torch.cat(candidate_local, dim=0) if candidate_local else baseline_local.new_empty((0, baseline_local.shape[1]))
        communication, peak_rank, selected_dim = self._global_action_costs(baseline_local, candidates)
        return _ScoredLayouts(
            communication=communication,
            peak_rank=peak_rank,
            selected_dim=selected_dim,
            baseline_physical_routes=physical,
        )

    def _rank_presence(self, layout: torch.Tensor, num_experts: int) -> torch.Tensor:
        slots = torch.arange(layout.numel(), dtype=torch.long, device=layout.device)
        ranks = torch.div(slots, self.slots_per_rank, rounding_mode="floor")
        active = layout >= 0
        presence = torch.zeros((num_experts, self.ep_size), dtype=torch.bool, device=layout.device)
        presence[layout[active], ranks[active]] = True
        return presence

    def _cover_rows(
        self,
        layout: torch.Tensor,
        owners: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = int(owners.numel())
        if destination_slots.numel() == 0:
            return torch.empty((0, 5), dtype=torch.long, device=layout.device)
        logicals = torch.arange(num_experts, dtype=torch.long, device=layout.device)
        slots = destination_slots.repeat_interleave(num_experts)
        experts = logicals.repeat(destination_slots.numel())
        destination_ranks = torch.div(slots, self.slots_per_rank, rounding_mode="floor")
        presence = self._rank_presence(layout, num_experts)
        copy_counts = torch.bincount(layout[layout >= 0], minlength=num_experts)
        victims = layout.index_select(0, slots)
        valid = ~presence[experts, destination_ranks]
        valid &= experts != victims
        valid &= copy_counts.index_select(0, experts) < self.max_copies
        rows = torch.stack(
            (
                torch.full_like(experts, _ACTION_COVER),
                owners.index_select(0, experts),
                slots,
                experts,
                victims,
            ),
            dim=1,
        )
        return rows[valid]

    def _swap_rows(self, layout: torch.Tensor, owners: torch.Tensor) -> torch.Tensor:
        num_experts = int(owners.numel())
        pairs = torch.triu_indices(num_experts, num_experts, offset=1, device=layout.device).transpose(0, 1)
        if pairs.numel() == 0:
            return torch.empty((0, 5), dtype=torch.long, device=layout.device)
        lhs, rhs = pairs[:, 0], pairs[:, 1]
        lhs_slots = owners.index_select(0, lhs)
        rhs_slots = owners.index_select(0, rhs)
        lhs_ranks = torch.div(lhs_slots, self.slots_per_rank, rounding_mode="floor")
        rhs_ranks = torch.div(rhs_slots, self.slots_per_rank, rounding_mode="floor")
        presence = self._rank_presence(layout, num_experts)
        valid = lhs_ranks != rhs_ranks
        valid &= ~presence[rhs, lhs_ranks]
        valid &= ~presence[lhs, rhs_ranks]
        rows = torch.stack(
            (
                torch.full_like(lhs, _ACTION_SWAP),
                lhs_slots,
                rhs_slots,
                lhs,
                rhs,
            ),
            dim=1,
        )
        return rows[valid]

    @staticmethod
    def _apply_rows(layout: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0:
            return layout.new_empty((0, layout.numel()))
        layouts = layout.unsqueeze(0).expand(rows.shape[0], -1).clone()
        swap = rows[:, 0] == _ACTION_SWAP
        if bool(swap.any().item()):
            swap_indices = torch.nonzero(swap, as_tuple=False).flatten()
            swap_rows = rows.index_select(0, swap_indices)
            layouts[swap_indices, swap_rows[:, 1]] = swap_rows[:, 4]
            layouts[swap_indices, swap_rows[:, 2]] = swap_rows[:, 3]
        cover = ~swap
        if bool(cover.any().item()):
            cover_indices = torch.nonzero(cover, as_tuple=False).flatten()
            cover_rows = rows.index_select(0, cover_indices)
            layouts[cover_indices, cover_rows[:, 2]] = cover_rows[:, 3]
        return layouts

    @staticmethod
    def _placement_action(row: Sequence[int]) -> PlacementAction:
        kind, src_slot, dst_slot, src_logical, dst_logical = (int(value) for value in row)
        return PlacementAction(
            "swap" if kind == _ACTION_SWAP else "replica",
            src_slot,
            dst_slot,
            src_logical,
            dst_logical,
        )

    @staticmethod
    def _placement_cost(scored: _ScoredLayouts, index: int) -> PlacementCost:
        values = (
            torch.stack(
                (
                    scored.communication[index],
                    scored.peak_rank[index].to(scored.communication.dtype),
                    scored.selected_dim[index].to(scored.communication.dtype),
                )
            )
            .detach()
            .to(device="cpu")
            .tolist()
        )
        return PlacementCost(
            communication=float(values[0]),
            compute=0.0,
            communication_model_units=float(values[0]),
            peak_communication_rank=int(values[1]),
            peak_compute_rank=-1,
            selected_dim=int(values[2]),
        )

    def _select_empty_actions(
        self,
        rows: torch.Tensor,
        candidate_costs: torch.Tensor,
        layout: torch.Tensor,
        limit: int,
    ) -> tuple[PlacementAction, ...]:
        if rows.numel() == 0 or limit <= 0:
            return ()
        order = torch.argsort(candidate_costs, stable=True)
        ordered_rows = rows.index_select(0, order).detach().to(device="cpu").tolist()
        layout_cpu = layout.detach().to(device="cpu", dtype=torch.long)
        rank_experts = [
            {
                int(value)
                for value in layout_cpu[rank * self.slots_per_rank : (rank + 1) * self.slots_per_rank].tolist()
                if int(value) >= 0
            }
            for rank in range(self.ep_size)
        ]
        copy_counts: dict[int, int] = {}
        for value in layout_cpu.tolist():
            logical = int(value)
            if logical >= 0:
                copy_counts[logical] = copy_counts.get(logical, 0) + 1
        used_slots: set[int] = set()
        actions: list[PlacementAction] = []
        for row in ordered_rows:
            action = self._placement_action(row)
            rank = action.dst_slot // self.slots_per_rank
            if (
                action.dst_slot in used_slots
                or action.src_logical in rank_experts[rank]
                or copy_counts.get(action.src_logical, 0) >= self.max_copies
            ):
                continue
            actions.append(action)
            used_slots.add(action.dst_slot)
            rank_experts[rank].add(action.src_logical)
            copy_counts[action.src_logical] = copy_counts.get(action.src_logical, 0) + 1
            if len(actions) >= limit:
                break
        return tuple(actions)

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
        original_selected_ndim = selected.ndim
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        if selected.ndim != 2:
            raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")
        device = selected.device
        layout = slot_to_logical.to(device=device, dtype=torch.long, non_blocking=True).clone()
        owners = owner_slots.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1).clone()
        if layout.numel() != self.ep_size * self.slots_per_rank:
            raise ValueError("slot_to_logical does not match hierarchy.ep_size * slots_per_rank.")
        if isinstance(source_ranks, int):
            sources = torch.full((selected.shape[0],), int(source_ranks), dtype=torch.long, device=device)
        else:
            sources = source_ranks.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        ordinals = (
            torch.arange(selected.shape[0], dtype=torch.long, device=device)
            if token_ordinals is None
            else token_ordinals.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        )
        if sources.numel() != selected.shape[0] or ordinals.numel() != selected.shape[0]:
            raise ValueError("source_ranks and token_ordinals must match the local token count.")

        candidate_started = time.perf_counter()
        all_slots = torch.arange(layout.numel(), dtype=torch.long, device=device)
        owner_mask = torch.zeros((layout.numel(),), dtype=torch.bool, device=device)
        owner_mask.scatter_(0, owners, True)
        empty_slots = torch.nonzero(layout < 0, as_tuple=False).flatten()
        initializing = empty_slots.numel() > 0 and max(0, int(max_replicas)) > 0
        if initializing:
            rows = self._cover_rows(layout, owners, empty_slots)
        else:
            rows_by_kind = []
            if max(0, int(max_swaps)) > 0:
                rows_by_kind.append(self._swap_rows(layout, owners))
            if max(0, int(max_replicas)) > 0:
                cover_slots = all_slots[(~owner_mask) & (layout >= 0)]
                rows_by_kind.append(self._cover_rows(layout, owners, cover_slots))
            nonempty_rows = [value for value in rows_by_kind if value.numel()]
            rows = (
                torch.cat(nonempty_rows, dim=0)
                if nonempty_rows
                else torch.empty((0, 5), dtype=torch.long, device=device)
            )
        route_stats_ms = (time.perf_counter() - candidate_started) * 1000.0

        score_started = time.perf_counter()
        scored = self._score_actions(
            selected,
            layout,
            rows,
            source_ranks=sources,
            token_ordinals=ordinals,
            step=step,
            layer_seed=layer_seed,
            num_experts=int(owners.numel()),
        )
        score_ms = (time.perf_counter() - score_started) * 1000.0
        baseline_cost = self._placement_cost(scored, 0)
        actions: tuple[PlacementAction, ...] = ()
        final_layout_tensor = layout
        final_owners = owners
        final_cost = baseline_cost
        final_physical = scored.baseline_physical_routes
        finalization_started = time.perf_counter()

        if initializing and rows.numel():
            fill_limit = min(int(empty_slots.numel()), max(0, int(max_replicas)))
            actions = self._select_empty_actions(rows, scored.communication[1:], layout, fill_limit)
            if actions:
                final_layout_tensor = layout.clone()
                for action in actions:
                    final_layout_tensor[action.dst_slot] = action.src_logical
                candidate_physical = assign_tokens_to_copies_greedy(
                    selected,
                    final_layout_tensor,
                    slots_per_rank=self.slots_per_rank,
                    source_ranks=sources,
                    hierarchy_group_sizes=self.hierarchy.group_sizes,
                    num_experts=int(owners.numel()),
                    token_ordinals=ordinals,
                    step=step,
                    layer_seed=layer_seed,
                    max_copies=self.max_copies,
                )
                candidate_counts = _reduce_sum(self._local_packed_counts(candidate_physical), self.reducer)
                candidate_comm, candidate_peak, candidate_dim = self._communication_cost(candidate_counts)
                candidate_scored = _ScoredLayouts(
                    communication=candidate_comm,
                    peak_rank=candidate_peak,
                    selected_dim=candidate_dim,
                    baseline_physical_routes=candidate_physical,
                )
                candidate_cost = self._placement_cost(candidate_scored, 0)
                if candidate_cost.communication <= baseline_cost.communication:
                    final_cost = candidate_cost
                    final_physical = candidate_physical
        elif rows.numel():
            candidate_costs = scored.communication[1:]
            best_index = candidate_costs.argmin()
            best_cost = candidate_costs.index_select(0, best_index.view(1))[0]
            if bool((best_cost < scored.communication[0]).item()):
                row = rows.index_select(0, best_index.view(1))[0]
                action = self._placement_action(row.detach().to(device="cpu").tolist())
                actions = (action,)
                final_layout_tensor = self._apply_rows(layout, row.view(1, -1))[0]
                if action.kind == "swap":
                    final_owners = owners.clone()
                    logical_pair = torch.tensor(
                        (action.src_logical, action.dst_logical), dtype=torch.long, device=device
                    )
                    final_owners.scatter_(0, logical_pair, owners.index_select(0, logical_pair).flip(0))
                final_cost = self._placement_cost(scored, int(best_index.item()) + 1)
                final_physical = self._apply_action_routes(
                    selected,
                    layout,
                    row,
                    scored.baseline_physical_routes,
                    source_ranks=sources,
                    token_ordinals=ordinals,
                    step=step,
                    layer_seed=layer_seed,
                    num_experts=int(owners.numel()),
                )

        if original_selected_ndim == 1:
            final_physical = final_physical.squeeze(-1)
        initial_layout = tuple(int(value) for value in layout.detach().to(device="cpu").tolist())
        final_layout = tuple(int(value) for value in final_layout_tensor.detach().to(device="cpu").tolist())
        final_owner_slots = tuple(int(value) for value in final_owners.detach().to(device="cpu").tolist())
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        planning_ms = (time.perf_counter() - started) * 1000.0
        chose_swap = bool(actions) and actions[0].kind == "swap"
        chose_cover = bool(actions) and not chose_swap
        return PlacementPlan(
            actions=actions,
            initial_layout=initial_layout,
            final_layout=final_layout,
            baseline_cost=baseline_cost,
            final_cost=final_cost,
            swap_rounds=sum(action.kind == "swap" for action in actions),
            replica_rounds=sum(action.kind == "replica" for action in actions),
            planning_ms=planning_ms,
            route_stats_ms=route_stats_ms,
            swap_ms=score_ms if chose_swap else 0.0,
            replica_ms=score_ms if chose_cover else 0.0,
            swap_score_ms=score_ms if not initializing else 0.0,
            swap_update_ms=0.0,
            swap_collective_ms=0.0,
            replica_score_ms=score_ms if initializing else 0.0,
            replica_update_ms=0.0,
            replica_collective_ms=0.0,
            decision_sync_ms=0.0,
            finalization_ms=finalization_ms,
            algorithm_version=GREEDY_COVER_ALGORITHM_VERSION,
            local_physical_routes=final_physical,
            final_owner_slots=final_owner_slots,
        )


__all__ = [
    "GREEDY_COVER_ALGORITHM_VERSION",
    "GreedyCommunicationPlanner",
    "assign_tokens_to_copies_greedy",
]
