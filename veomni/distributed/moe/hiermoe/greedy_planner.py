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
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from ....utils.device import get_torch_device
from .perf_model import HierMoEPerfModel
from .planner import PlacementAction, PlacementCost, PlacementPlan, _hierarchy_distance, _route_hash
from .statistical_scorer import (
    StatisticalPairContext,
    StatisticalPrimitiveContext,
    StatisticalPrimitiveSpec,
    StatisticalRouteTables,
    _canonical_route_mask,
    build_statistical_primitive_spec,
    statistical_batched_selected_pair_local_deltas,
    statistical_candidate_local_deltas,
    statistical_pair_interaction_bound_local,
    statistical_primitive_fast_path_available,
    statistical_primitive_selected_pair_context,
    statistical_primitive_unary_local_deltas,
    statistical_proxy_candidate_local_deltas,
    statistical_selected_pair_local_deltas,
    statistical_unary_candidate_local_deltas,
    uniform_statistical_baseline_routes,
)
from .topology import Hierarchy


GREEDY_COVER_ALGORITHM_VERSION = "hiermoe-greedy-cover-p1-exact-stats-v9-early-proxy-topk"
GREEDY_COMMUNICATION_PHASE_MULTIPLIER = 4.0
GREEDY_COMPUTE_PHASE_MULTIPLIER = 3.0
_ACTION_SWAP = 0
_ACTION_COVER = 1
_LAYER_COMPUTE_STREAMS: dict[tuple[str, int | None, int], tuple[object, ...]] = {}


@dataclass(frozen=True)
class _ScoredLayouts:
    communication: torch.Tensor
    compute: torch.Tensor
    communication_model_units: torch.Tensor
    peak_rank: torch.Tensor
    peak_compute_rank: torch.Tensor
    selected_dim: torch.Tensor
    baseline_physical_routes: torch.Tensor
    route_hashes: torch.Tensor | None = None
    route_tables: StatisticalRouteTables | None = None
    exact_cost_lower_bound: torch.Tensor | None = None

    @property
    def total(self) -> torch.Tensor:
        return self.communication + self.compute


@dataclass(frozen=True)
class _PreparedActionCounts:
    baseline_local: torch.Tensor
    candidate_local: torch.Tensor
    baseline_assignment_local: torch.Tensor | None
    candidate_assignment_local: torch.Tensor | None
    affected_groups: torch.Tensor | None
    affected_assignment_ranks: torch.Tensor | None
    baseline_physical_routes: torch.Tensor
    route_hashes: torch.Tensor
    route_tables: StatisticalRouteTables | None
    pair_context: StatisticalPairContext | None = None
    candidate_pair_bound_local: torch.Tensor | None = None
    certificate_affected_groups: torch.Tensor | None = None


@dataclass(frozen=True)
class _PreparedPrimitiveCounts:
    baseline_local: torch.Tensor
    primitive_delta_local: torch.Tensor
    baseline_assignment_local: torch.Tensor | None
    primitive_assignment_delta_local: torch.Tensor | None
    primitive_affected_ranks: torch.Tensor
    primitive_affected_groups: tuple[torch.Tensor, ...]
    context: StatisticalPrimitiveContext
    baseline_physical_routes: torch.Tensor
    route_hashes: torch.Tensor


@dataclass(frozen=True)
class _GlobalPrimitiveCounts:
    baseline: torch.Tensor
    primitive_delta: torch.Tensor
    baseline_assignment: torch.Tensor | None
    primitive_assignment_delta: torch.Tensor | None


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
    route_hashes: torch.Tensor | None = None,
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
    if route_hashes is None:
        route_hashes = _route_hash(
            selected,
            token_ordinals=ordinals,
            step=step,
            layer_seed=layer_seed,
        )
    else:
        route_hashes = route_hashes.to(device=selected.device, dtype=torch.long, non_blocking=True)
        if route_hashes.shape != selected.shape:
            raise ValueError("route_hashes must match selected_experts after rank normalization.")
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
        communication_scale: float = 1.0,
        forward_compute_per_assignment: float = 0.0,
        forward_compute_constant: float = 0.0,
        smooth_max_gamma: float = 10.0,
        reducer: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        candidate_chunk_size: int = 128,
        process_group: dist.ProcessGroup | None = None,
        max_copies: int = 4,
        candidate_scorer: str = "statistics",
        compact_candidate_collective: bool = False,
        assume_unique_routes: bool = False,
        layer_parallel_streams: int = 8,
        adaptive_topk: bool = False,
        adaptive_topk_initial: int = 16,
        adaptive_topk_growth_factor: int = 2,
        adaptive_topk_epsilon: float = 1e-6,
        adaptive_topk_strict_certificate: bool = False,
        early_proxy_topk: int = 0,
        exact_primitive_topk: int = 0,
        post_shortlist_compact_pair: bool = False,
        exact_primitive_max_only: bool = False,
    ) -> None:
        if smooth_max_gamma <= 0:
            raise ValueError("smooth_max_gamma must be positive.")
        if candidate_scorer not in {"statistics", "reference"}:
            raise ValueError("candidate_scorer must be 'statistics' or 'reference'.")
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.hidden_size = int(hidden_size)
        self.bytes_per_element = int(bytes_per_element)
        self.slots_per_rank = int(slots_per_rank)
        self.communication_scale = float(communication_scale)
        self.forward_compute_per_assignment = float(forward_compute_per_assignment)
        self.forward_compute_constant = float(forward_compute_constant)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.reducer = reducer
        self.candidate_chunk_size = max(1, int(candidate_chunk_size))
        self.process_group = process_group
        self.max_copies = max(1, int(max_copies))

        self.candidate_scorer = candidate_scorer
        self.compact_candidate_collective = bool(compact_candidate_collective)
        self.assume_unique_routes = bool(assume_unique_routes)
        self.layer_parallel_streams = max(1, int(layer_parallel_streams))
        self.adaptive_topk = bool(adaptive_topk)
        self.adaptive_topk_initial = max(1, int(adaptive_topk_initial))
        self.adaptive_topk_growth_factor = max(2, int(adaptive_topk_growth_factor))
        self.adaptive_topk_epsilon = max(0.0, float(adaptive_topk_epsilon))
        self.adaptive_topk_strict_certificate = bool(adaptive_topk_strict_certificate)
        self.early_proxy_topk = max(0, int(early_proxy_topk))
        self.exact_primitive_topk = max(0, int(exact_primitive_topk))
        self.post_shortlist_compact_pair = bool(post_shortlist_compact_pair)
        self.exact_primitive_max_only = bool(exact_primitive_max_only)
        enabled_topk_modes = sum(
            int(value)
            for value in (
                self.adaptive_topk,
                self.early_proxy_topk > 0,
                self.exact_primitive_topk > 0,
            )
        )
        if enabled_topk_modes > 1:
            raise ValueError("adaptive_topk, early_proxy_topk, and exact_primitive_topk are mutually exclusive.")
        self.last_adaptive_topk_stats: dict[str, object] = {}
        self.last_early_proxy_stats: dict[str, object] = {"enabled": False}
        self.last_early_proxy_shortlist_indices: list[torch.Tensor] = []
        self.last_exact_primitive_stats: dict[str, object] = {"enabled": False}
        self.last_exact_primitive_shortlist_indices: list[torch.Tensor] = []
        if self.communication_scale < 0.0:
            raise ValueError("communication_scale must be non-negative.")
        if self.forward_compute_per_assignment < 0.0:
            raise ValueError("forward_compute_per_assignment must be non-negative.")
        if self.forward_compute_constant < 0.0:
            raise ValueError("forward_compute_constant must be non-negative.")

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

    def _local_packed_assignment_counts(self, physical_slots: torch.Tensor) -> torch.Tensor:
        """Count non-deduplicated assignments for every hierarchy destination."""

        physical = physical_slots
        if physical.ndim == 2:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor").to(torch.long)
        batch = int(ranks.shape[0])
        flat_ranks = ranks.reshape(batch, -1)
        ones = torch.ones_like(flat_ranks, dtype=torch.float32)
        rows: list[torch.Tensor] = []

        rank_counts = torch.zeros((batch, self.ep_size), dtype=torch.float32, device=ranks.device)
        rank_counts.scatter_add_(1, flat_ranks, ones)
        rows.append(rank_counts)
        for raw_size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(raw_size)
            groups = torch.div(flat_ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            group_counts = torch.zeros((batch, num_groups), dtype=torch.float32, device=ranks.device)
            group_counts.scatter_add_(1, groups, ones)
            rows.append(group_counts)
        return torch.cat(rows, dim=1)

    @staticmethod
    def _stage_traffic_features(
        unique_matrix: torch.Tensor,
        assignment_matrix: torch.Tensor,
        *,
        hidden_bytes: int,
        metadata_bytes: int,
    ) -> dict[str, torch.Tensor]:
        """Return exact endpoint/edge features for one A2A stage.

        Both matrices have shape ``[batch, group, source, destination]``.
        Dispatch sends unique hidden rows plus assignment metadata, while
        combine reverses only the unique hidden rows.
        """

        if unique_matrix.shape != assignment_matrix.shape or unique_matrix.ndim != 4:
            raise ValueError(
                "Stage traffic matrices must have the same [batch, group, source, destination] shape."
            )
        unique = unique_matrix.to(torch.float32)
        assignments = assignment_matrix.to(torch.float32)
        unique_send = unique.sum(dim=3)
        unique_receive = unique.sum(dim=2)
        assignment_send = assignments.sum(dim=3)
        assignment_receive = assignments.sum(dim=2)

        dispatch_send_bytes = float(hidden_bytes) * unique_send + float(metadata_bytes) * assignment_send
        dispatch_receive_bytes = (
            float(hidden_bytes) * unique_receive + float(metadata_bytes) * assignment_receive
        )
        dispatch_endpoint_bytes = torch.maximum(
            dispatch_send_bytes.amax(dim=(1, 2)),
            dispatch_receive_bytes.amax(dim=(1, 2)),
        )
        unique_endpoint = torch.maximum(
            unique_send.amax(dim=(1, 2)),
            unique_receive.amax(dim=(1, 2)),
        )
        combine_endpoint_bytes = float(hidden_bytes) * unique_endpoint

        dispatch_edge_bytes = (
            float(hidden_bytes) * unique + float(metadata_bytes) * assignments
        ).amax(dim=(1, 2, 3))
        combine_edge_bytes = float(hidden_bytes) * unique.amax(dim=(1, 2, 3))
        active_send_peers = (unique > 0).sum(dim=3, dtype=torch.float32).amax(dim=(1, 2))
        active_receive_peers = (unique > 0).sum(dim=2, dtype=torch.float32).amax(dim=(1, 2))
        if int(unique.shape[2]) != int(unique.shape[3]):
            raise ValueError("A2A stage traffic matrices must have equal source and destination widths.")
        diagonal = torch.eye(
            int(unique.shape[2]),
            dtype=torch.bool,
            device=unique.device,
        ).view(1, 1, int(unique.shape[2]), int(unique.shape[3]))
        remote_unique = unique.masked_fill(diagonal, 0.0)
        remote_assignments = assignments.masked_fill(diagonal, 0.0)
        remote_unique_send = remote_unique.sum(dim=3)
        remote_unique_receive = remote_unique.sum(dim=2)
        remote_assignment_send = remote_assignments.sum(dim=3)
        remote_assignment_receive = remote_assignments.sum(dim=2)
        remote_dispatch_endpoint_bytes = torch.maximum(
            (
                float(hidden_bytes) * remote_unique_send
                + float(metadata_bytes) * remote_assignment_send
            ).amax(dim=(1, 2)),
            (
                float(hidden_bytes) * remote_unique_receive
                + float(metadata_bytes) * remote_assignment_receive
            ).amax(dim=(1, 2)),
        )
        remote_unique_endpoint = torch.maximum(
            remote_unique_send.amax(dim=(1, 2)),
            remote_unique_receive.amax(dim=(1, 2)),
        )
        remote_dispatch_edge_bytes = (
            float(hidden_bytes) * remote_unique + float(metadata_bytes) * remote_assignments
        ).amax(dim=(1, 2, 3))
        remote_combine_edge_bytes = float(hidden_bytes) * remote_unique.amax(dim=(1, 2, 3))
        diagonal_unique = unique.diagonal(dim1=2, dim2=3)
        diagonal_assignments = assignments.diagonal(dim1=2, dim2=3)
        self_endpoint_bytes = (
            2.0 * float(hidden_bytes) * diagonal_unique
            + float(metadata_bytes) * diagonal_assignments
        ).amax(dim=(1, 2))
        return {
            "unique_endpoint_tokens": unique_endpoint,
            "full_endpoint_bytes": dispatch_endpoint_bytes + combine_endpoint_bytes,
            "full_edge_bytes": dispatch_edge_bytes + combine_edge_bytes,
            "remote_full_endpoint_bytes": (
                remote_dispatch_endpoint_bytes + float(hidden_bytes) * remote_unique_endpoint
            ),
            "remote_full_edge_bytes": remote_dispatch_edge_bytes + remote_combine_edge_bytes,
            "self_endpoint_bytes": self_endpoint_bytes,
            "full_total_bytes": (
                2.0 * float(hidden_bytes) * unique.sum(dim=(1, 2, 3))
                + float(metadata_bytes) * assignments.sum(dim=(1, 2, 3))
            ),
            "max_active_peers": torch.maximum(active_send_peers, active_receive_peers),
        }

    def _hierarchical_traffic_features(
        self,
        source_unique_counts: torch.Tensor,
        source_assignment_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Derive traffic features matching the runtime two-stage hierarchy.

        Stage 1 is split into one cross-node A2A group per local-rank lane.
        Stage 2 runs inside each destination node from relay lanes to target
        local ranks. This preserves peer/edge structure that endpoint-only
        hierarchy counts discard.
        """

        if source_unique_counts.shape != source_assignment_counts.shape:
            raise ValueError("Unique and assignment source counts must have identical shapes.")
        if source_unique_counts.ndim != 3 or int(source_unique_counts.shape[0]) != self.ep_size:
            raise ValueError(
                "Hierarchical source counts must have shape "
                f"[{self.ep_size}, batch, packed_width], got {tuple(source_unique_counts.shape)}."
            )
        if int(self.hierarchy.selected_dim) != 2 or len(self.hierarchy.group_sizes) < 2:
            raise ValueError("Traffic-matrix diagnostics currently require a two-stage hierarchy.")
        intra_size = int(self.hierarchy.group_sizes[0])
        if intra_size <= 1 or self.ep_size % intra_size != 0:
            raise ValueError(f"Invalid intra-node size {intra_size} for EP size {self.ep_size}.")
        num_nodes = self.ep_size // intra_size
        widths = self._count_widths()
        if widths[:2] != (self.ep_size, num_nodes):
            raise ValueError(f"Unexpected two-stage packed widths {widths}.")

        unique_rank, unique_node = source_unique_counts.split(widths, dim=2)[:2]
        assignment_rank, assignment_node = source_assignment_counts.split(widths, dim=2)[:2]
        batch = int(source_unique_counts.shape[1])

        # [source_node, lane, batch, destination_node]
        stage1_unique = unique_node.reshape(num_nodes, intra_size, batch, num_nodes)
        stage1_assignments = assignment_node.reshape(num_nodes, intra_size, batch, num_nodes)
        # One independent cross-node group per local-rank lane.
        stage1_unique = stage1_unique.permute(2, 1, 0, 3).contiguous()
        stage1_assignments = stage1_assignments.permute(2, 1, 0, 3).contiguous()

        # Aggregate source nodes sharing the same relay lane at each
        # destination node: [batch, destination_node, relay_lane, local_rank].
        stage2_unique = (
            unique_rank.reshape(num_nodes, intra_size, batch, num_nodes, intra_size)
            .sum(dim=0)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        stage2_assignments = (
            assignment_rank.reshape(num_nodes, intra_size, batch, num_nodes, intra_size)
            .sum(dim=0)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

        hidden_bytes = int(self.payload_bytes)
        # _pack_meta_weights converts every column to float32 for bf16 routes.
        stage1 = self._stage_traffic_features(
            stage1_unique,
            stage1_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=3 * 4,
        )
        stage2 = self._stage_traffic_features(
            stage2_unique,
            stage2_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=2 * 4,
        )

        # A shared node uplink can be bottlenecked by aggregate traffic across
        # all local-rank lanes even though HCCL creates one group per lane.
        stage1_node_unique = stage1_unique.sum(dim=1, keepdim=True)
        stage1_node_assignments = stage1_assignments.sum(dim=1, keepdim=True)
        stage1_node = self._stage_traffic_features(
            stage1_node_unique,
            stage1_node_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=3 * 4,
        )

        inter_link = self.perf_model.inter[0]
        intra_link = self.perf_model.intra
        unique_link_units = (
            2.0 * float(inter_link.beta) * float(hidden_bytes) * stage1["unique_endpoint_tokens"]
            + 2.0 * float(intra_link.beta) * float(hidden_bytes) * stage2["unique_endpoint_tokens"]
        )
        payload_endpoint_link_units = (
            float(inter_link.beta) * stage1["full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["full_endpoint_bytes"]
        )
        payload_edge_link_units = (
            float(inter_link.beta) * stage1["full_edge_bytes"]
            + float(intra_link.beta) * stage2["full_edge_bytes"]
        )
        shared_node_link_units = (
            float(inter_link.beta) * stage1_node["full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["full_endpoint_bytes"]
        )
        remote_payload_endpoint_link_units = (
            float(inter_link.beta) * stage1["remote_full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["remote_full_endpoint_bytes"]
        )
        remote_payload_edge_link_units = (
            float(inter_link.beta) * stage1["remote_full_edge_bytes"]
            + float(intra_link.beta) * stage2["remote_full_edge_bytes"]
        )
        self_payload_link_units = (
            float(inter_link.beta) * stage1["self_endpoint_bytes"]
            + float(intra_link.beta) * stage2["self_endpoint_bytes"]
        )
        return {
            "stage_unique_endpoint_link_units": unique_link_units,
            "stage_payload_endpoint_link_units": payload_endpoint_link_units,
            "stage_payload_edge_link_units": payload_edge_link_units,
            "stage_shared_node_endpoint_link_units": shared_node_link_units,
            "stage_remote_payload_endpoint_link_units": remote_payload_endpoint_link_units,
            "stage_remote_payload_edge_link_units": remote_payload_edge_link_units,
            "stage_self_payload_link_units": self_payload_link_units,
            "stage1_payload_endpoint_bytes": stage1["full_endpoint_bytes"],
            "stage2_payload_endpoint_bytes": stage2["full_endpoint_bytes"],
            "stage1_payload_edge_bytes": stage1["full_edge_bytes"],
            "stage2_payload_edge_bytes": stage2["full_edge_bytes"],
            "stage1_payload_total_bytes": stage1["full_total_bytes"],
            "stage2_payload_total_bytes": stage2["full_total_bytes"],
            "stage1_max_active_peers": stage1["max_active_peers"],
            "stage2_max_active_peers": stage2["max_active_peers"],
        }

    def _communication_cost_details(
        self,
        packed_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        communication_units = torch.logsumexp(per_dim * self.smooth_max_gamma, dim=1) / self.smooth_max_gamma
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * self.communication_scale * communication_units
        return communication, communication_units, rank_counts.argmax(dim=1), per_dim.argmax(dim=1) + 1

    def _source_aware_communication_cost_details(
        self,
        source_packed_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score duplicate-free traffic using both source and destination bottlenecks.

        ``source_packed_counts`` preserves the source-rank dimension that is
        normally removed by the placement-statistics reduction. At every
        hierarchy level, the effective payload is the larger of the maximum
        outgoing source-group payload and maximum incoming destination-group
        payload.
        """

        if source_packed_counts.ndim != 3:
            raise ValueError(
                "Source-aware communication counts must have shape "
                f"[source_rank, batch, packed_width], got {tuple(source_packed_counts.shape)}."
            )
        if int(source_packed_counts.shape[0]) != self.ep_size:
            raise ValueError(f"Expected {self.ep_size} source ranks, got {int(source_packed_counts.shape[0])}.")
        widths = self._count_widths()
        if int(source_packed_counts.shape[2]) != sum(widths):
            raise ValueError(f"Expected packed width {sum(widths)}, got {int(source_packed_counts.shape[2])}.")

        source_rows = source_packed_counts.split(widths, dim=2)
        receive_rows = source_packed_counts.sum(dim=0).split(widths, dim=1)
        receive_maxima = [row.max(dim=1).values for row in receive_rows]

        rank_send_max = source_rows[0].sum(dim=2).max(dim=0).values
        send_maxima = [rank_send_max]
        for level_index, raw_size in enumerate(
            self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        ):
            size = int(raw_size)
            num_source_groups = self.ep_size // size
            source_group_payload = (
                source_rows[level_index + 1]
                .reshape(num_source_groups, size, source_packed_counts.shape[1], widths[level_index + 1])
                .sum(dim=1)
                .sum(dim=2)
            )
            send_maxima.append(source_group_payload.max(dim=0).values)

        bottleneck_rows = [
            torch.maximum(send_max, receive_max)
            for send_max, receive_max in zip(send_maxima, receive_maxima, strict=True)
        ]
        rank_bottleneck = bottleneck_rows[0]
        dimensions = [
            self.perf_model.a2a.alpha
            + float(self.ep_size * self.payload_bytes) * rank_bottleneck * self.perf_model.a2a.beta
        ]
        max_dim = max(1, int(self.hierarchy.selected_dim))
        for dim in range(2, max_dim + 1):
            total = torch.zeros_like(rank_bottleneck)
            previous_size = 1
            for level_index, raw_size in enumerate(self.hierarchy.group_sizes[: dim - 1]):
                size = int(raw_size)
                link = self.perf_model.inter[min(level_index, len(self.perf_model.inter) - 1)]
                scale = float((size / previous_size) * self.payload_bytes)
                total = total + link.alpha + scale * bottleneck_rows[level_index + 1] * link.beta
                previous_size = size
            intra_scale = float((self.ep_size / previous_size) * self.payload_bytes)
            total = total + self.perf_model.intra.alpha + intra_scale * rank_bottleneck * self.perf_model.intra.beta
            dimensions.append(total)
        per_dim = torch.stack(dimensions, dim=1)
        communication_units = torch.logsumexp(per_dim * self.smooth_max_gamma, dim=1) / self.smooth_max_gamma
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * self.communication_scale * communication_units
        return (
            communication,
            communication_units,
            torch.stack(send_maxima, dim=1),
            torch.stack(receive_maxima, dim=1),
            per_dim.argmax(dim=1) + 1,
        )

    def _communication_cost(self, packed_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        communication, _units, peak_rank, selected_dim = self._communication_cost_details(packed_counts)
        return communication, peak_rank, selected_dim

    def _compute_cost(
        self,
        assignment_counts: torch.Tensor | None,
        *,
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if assignment_counts is None:
            compute = torch.full(
                (rows,),
                GREEDY_COMPUTE_PHASE_MULTIPLIER * self.forward_compute_constant,
                dtype=torch.float32,
                device=device,
            )
            peak_rank = torch.full((rows,), -1, dtype=torch.long, device=device)
            return compute, peak_rank
        peak_assignments, peak_rank = assignment_counts.max(dim=1)
        compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (
            self.forward_compute_per_assignment * peak_assignments + self.forward_compute_constant
        )
        return compute, peak_rank

    def _cost_details(
        self,
        packed_counts: torch.Tensor,
        assignment_counts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        communication, units, peak_rank, selected_dim = self._communication_cost_details(packed_counts)
        compute, peak_compute_rank = self._compute_cost(
            assignment_counts,
            rows=packed_counts.shape[0],
            device=packed_counts.device,
        )
        return communication, compute, units, peak_rank, peak_compute_rank, selected_dim

    def _local_assignment_counts(self, physical_slots: torch.Tensor) -> torch.Tensor:
        physical = physical_slots
        if physical.ndim == 2:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
        counts = torch.zeros((ranks.shape[0], self.ep_size), dtype=torch.float32, device=ranks.device)
        updates = torch.ones_like(ranks, dtype=torch.float32)
        counts.scatter_add_(1, ranks.reshape(ranks.shape[0], -1), updates.reshape(ranks.shape[0], -1))
        return counts

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
        local_assignments: list[torch.Tensor] = []
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
            if self.forward_compute_per_assignment > 0.0:
                local_assignments.append(self._local_assignment_counts(physical))
        assert baseline_physical is not None
        local_counts = torch.cat(local_rows, dim=0)
        assignment_counts = torch.cat(local_assignments, dim=0) if local_assignments else None
        if assignment_counts is None:
            global_counts = _reduce_sum(local_counts, self.reducer)
            communication, compute, units, peak_rank, peak_compute_rank, selected_dim = self._cost_details(
                global_counts,
                None,
            )
        else:
            combined = _reduce_sum(torch.cat((local_counts, assignment_counts), dim=1), self.reducer)
            global_counts = combined[:, : local_counts.shape[1]]
            global_assignments = combined[:, local_counts.shape[1] :]
            communication, compute, units, peak_rank, peak_compute_rank, selected_dim = self._cost_details(
                global_counts,
                global_assignments,
            )
        return _ScoredLayouts(
            communication=communication,
            compute=compute,
            communication_model_units=units,
            peak_rank=peak_rank,
            peak_compute_rank=peak_compute_rank,
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
        baseline_assignment_local: torch.Tensor | None = None,
        candidate_assignment_local: torch.Tensor | None = None,
        affected_groups: torch.Tensor | None = None,
        affected_assignment_ranks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        communication_width = baseline_local.shape[1]
        has_assignments = baseline_assignment_local is not None
        if has_assignments != (candidate_assignment_local is not None):
            raise ValueError("Baseline and candidate assignment counts must either both be present or both be absent.")
        if has_assignments:
            assert baseline_assignment_local is not None and candidate_assignment_local is not None
            baseline_combined = torch.cat((baseline_local, baseline_assignment_local), dim=1)
            candidate_combined = torch.cat((candidate_local, candidate_assignment_local), dim=1)
        else:
            baseline_combined = baseline_local
            candidate_combined = candidate_local

        def costs(combined: torch.Tensor):
            assignments = combined[:, communication_width:] if has_assignments else None
            return self._cost_details(combined[:, :communication_width], assignments)

        if candidate_local.numel() == 0:
            return costs(_reduce_sum(baseline_combined, self.reducer))
        if affected_groups is not None and self.reducer is not None and dist.is_initialized():
            combined_affected = affected_groups
            if has_assignments:
                if affected_assignment_ranks is None:
                    raise ValueError("Compact assignment scoring requires affected assignment ranks.")
                offset_assignment_ranks = torch.where(
                    affected_assignment_ranks >= 0,
                    affected_assignment_ranks + communication_width,
                    affected_assignment_ranks,
                )
                combined_affected = torch.cat(
                    (affected_groups, offset_assignment_ranks),
                    dim=1,
                )
            return self._global_compact_action_costs(
                baseline_combined,
                candidate_combined,
                combined_affected,
                communication_width=communication_width,
                has_assignments=has_assignments,
            )
        if not self._use_sharded_candidate_collective(baseline_combined.device):
            local_counts = torch.cat((baseline_combined, candidate_combined), dim=0)
            global_counts = _reduce_sum(local_counts, self.reducer)
            return costs(global_counts)

        assert self.process_group is not None
        baseline_global = baseline_combined.clone()
        dist.all_reduce(baseline_global, op=dist.ReduceOp.SUM, group=self.process_group)
        baseline_metrics = costs(baseline_global)

        group_size = dist.get_world_size(self.process_group)
        num_candidates, width = candidate_combined.shape
        shard_rows = (num_candidates + group_size - 1) // group_size
        padded_rows = shard_rows * group_size
        if padded_rows != num_candidates:
            padding = torch.zeros(
                (padded_rows - num_candidates, width),
                dtype=candidate_combined.dtype,
                device=candidate_combined.device,
            )
            reduce_input = torch.cat((candidate_combined, padding), dim=0)
        else:
            reduce_input = candidate_combined
        reduced_shard = torch.empty(
            (shard_rows, width),
            dtype=candidate_combined.dtype,
            device=candidate_combined.device,
        )
        dist.reduce_scatter_tensor(
            reduced_shard,
            reduce_input.contiguous(),
            op=dist.ReduceOp.SUM,
            group=self.process_group,
        )
        shard_communication, shard_compute, shard_units, shard_peak, shard_compute_peak, shard_dim = costs(
            reduced_shard
        )
        shard_metrics = torch.stack(
            (
                shard_communication,
                shard_compute,
                shard_units,
                shard_peak.to(shard_communication.dtype),
                shard_compute_peak.to(shard_communication.dtype),
                shard_dim.to(shard_communication.dtype),
            ),
            dim=1,
        ).contiguous()
        gathered_metrics = torch.empty(
            (padded_rows, 6),
            dtype=shard_metrics.dtype,
            device=shard_metrics.device,
        )
        dist.all_gather_into_tensor(gathered_metrics, shard_metrics, group=self.process_group)
        candidate_metrics = gathered_metrics[:num_candidates]
        return (
            torch.cat((baseline_metrics[0], candidate_metrics[:, 0]), dim=0),
            torch.cat((baseline_metrics[1], candidate_metrics[:, 1]), dim=0),
            torch.cat((baseline_metrics[2], candidate_metrics[:, 2]), dim=0),
            torch.cat((baseline_metrics[3], candidate_metrics[:, 3].to(baseline_metrics[3].dtype)), dim=0),
            torch.cat((baseline_metrics[4], candidate_metrics[:, 4].to(baseline_metrics[4].dtype)), dim=0),
            torch.cat((baseline_metrics[5], candidate_metrics[:, 5].to(baseline_metrics[5].dtype)), dim=0),
        )

    @staticmethod
    def _restore_candidate_counts(
        baseline: torch.Tensor,
        packed_deltas: torch.Tensor,
        affected_groups: torch.Tensor,
    ) -> torch.Tensor:
        """Restore dense group counts after reducing only affected groups."""

        num_candidates = packed_deltas.shape[0]
        width = baseline.shape[1]
        dense = baseline.expand(num_candidates, -1).clone()
        if packed_deltas.numel() == 0:
            return dense
        valid = affected_groups >= 0
        safe_groups = affected_groups.clamp_min(0)
        candidate_rows = torch.arange(num_candidates, dtype=torch.long, device=dense.device).view(-1, 1)
        flat_indices = candidate_rows * width + safe_groups
        dense.reshape(-1).index_add_(
            0,
            flat_indices.reshape(-1),
            (packed_deltas * valid.to(packed_deltas.dtype)).reshape(-1),
        )
        return dense

    def _global_compact_action_costs(
        self,
        baseline_local: torch.Tensor,
        candidate_local: torch.Tensor,
        affected_groups: torch.Tensor,
        *,
        communication_width: int,
        has_assignments: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce only groups whose counts can change for each action."""

        def costs(combined: torch.Tensor):
            assignments = combined[:, communication_width:] if has_assignments else None
            return self._cost_details(combined[:, :communication_width], assignments)

        num_candidates = candidate_local.shape[0]
        valid = affected_groups >= 0
        safe_groups = affected_groups.clamp_min(0)
        local_deltas = candidate_local - baseline_local
        packed_local = local_deltas.gather(1, safe_groups) * valid.to(local_deltas.dtype)
        if not self._use_sharded_candidate_collective(baseline_local.device):
            payload = torch.cat((baseline_local.reshape(-1), packed_local.reshape(-1)), dim=0)
            global_payload = _reduce_sum(payload, self.reducer)
            baseline_global = global_payload[: baseline_local.numel()].view_as(baseline_local)
            packed_global = global_payload[baseline_local.numel() :].view_as(packed_local)
            candidate_global = self._restore_candidate_counts(
                baseline_global,
                packed_global,
                affected_groups,
            )
            return costs(torch.cat((baseline_global, candidate_global), dim=0))

        assert self.process_group is not None
        baseline_global = baseline_local.clone()
        dist.all_reduce(baseline_global, op=dist.ReduceOp.SUM, group=self.process_group)
        baseline_metrics = costs(baseline_global)

        group_size = dist.get_world_size(self.process_group)
        group_rank = dist.get_rank(self.process_group)
        packed_width = packed_local.shape[1]
        shard_rows = (num_candidates + group_size - 1) // group_size
        padded_rows = shard_rows * group_size
        if padded_rows != num_candidates:
            packed_local = torch.cat(
                (
                    packed_local,
                    torch.zeros(
                        (padded_rows - num_candidates, packed_width),
                        dtype=packed_local.dtype,
                        device=packed_local.device,
                    ),
                ),
                dim=0,
            )
            affected_groups = torch.cat(
                (
                    affected_groups,
                    torch.full(
                        (padded_rows - num_candidates, packed_width),
                        -1,
                        dtype=affected_groups.dtype,
                        device=affected_groups.device,
                    ),
                ),
                dim=0,
            )
        reduced_shard = torch.empty(
            (shard_rows, packed_width),
            dtype=packed_local.dtype,
            device=packed_local.device,
        )
        dist.reduce_scatter_tensor(
            reduced_shard,
            packed_local.contiguous(),
            op=dist.ReduceOp.SUM,
            group=self.process_group,
        )
        shard_start = group_rank * shard_rows
        shard_groups = affected_groups[shard_start : shard_start + shard_rows]
        shard_counts = self._restore_candidate_counts(baseline_global, reduced_shard, shard_groups)
        shard_communication, shard_compute, shard_units, shard_peak, shard_compute_peak, shard_dim = costs(
            shard_counts
        )
        shard_metrics = torch.stack(
            (
                shard_communication,
                shard_compute,
                shard_units,
                shard_peak.to(shard_communication.dtype),
                shard_compute_peak.to(shard_communication.dtype),
                shard_dim.to(shard_communication.dtype),
            ),
            dim=1,
        ).contiguous()
        gathered_metrics = torch.empty(
            (padded_rows, 6),
            dtype=shard_metrics.dtype,
            device=shard_metrics.device,
        )
        dist.all_gather_into_tensor(gathered_metrics, shard_metrics, group=self.process_group)
        candidate_metrics = gathered_metrics[:num_candidates]
        return (
            torch.cat((baseline_metrics[0], candidate_metrics[:, 0]), dim=0),
            torch.cat((baseline_metrics[1], candidate_metrics[:, 1]), dim=0),
            torch.cat((baseline_metrics[2], candidate_metrics[:, 2]), dim=0),
            torch.cat((baseline_metrics[3], candidate_metrics[:, 3].to(baseline_metrics[3].dtype)), dim=0),
            torch.cat((baseline_metrics[4], candidate_metrics[:, 4].to(baseline_metrics[4].dtype)), dim=0),
            torch.cat((baseline_metrics[5], candidate_metrics[:, 5].to(baseline_metrics[5].dtype)), dim=0),
        )

    def _copy_table(self, layout: torch.Tensor, num_experts: int) -> torch.Tensor:
        num_slots = int(layout.numel())
        logical_ids = torch.arange(num_experts, dtype=torch.long, device=layout.device)
        slot_ids = torch.arange(num_slots, dtype=torch.long, device=layout.device).view(-1, 1)
        matches = layout.view(-1, 1) == logical_ids.view(1, -1)
        counts = matches.sum(dim=0)
        width = max(1, int(counts.max().item()))
        masked = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
        return masked.sort(dim=0).values[:width].transpose(0, 1).contiguous()

    def _primitive_spec(
        self,
        layout: torch.Tensor,
        copy_slots: torch.Tensor,
        rows: torch.Tensor,
        *,
        device: torch.device,
    ) -> StatisticalPrimitiveSpec:
        host = build_statistical_primitive_spec(self, layout, copy_slots, rows)
        return StatisticalPrimitiveSpec(
            experts=host.experts.to(device=device, non_blocking=True),
            options=host.options.to(device=device, non_blocking=True),
            lhs_ids=host.lhs_ids.to(device=device, non_blocking=True),
            rhs_ids=host.rhs_ids.to(device=device, non_blocking=True),
            rhs_valid=host.rhs_valid.to(device=device, non_blocking=True),
        )

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
        sorted_options = torch.cat((lhs_options, rhs_options), dim=0).sort(dim=1).values
        return sorted_options[: rows.shape[0]], sorted_options[rows.shape[0] :], rhs_valid

    def _candidate_affected_groups(
        self,
        copy_slots: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Return canonical packed group ids that may change per action."""

        if rows.numel() == 0:
            return rows.new_empty((0, 0))
        num_slots = self.ep_size * self.slots_per_rank
        lhs = rows[:, 3]
        rhs_valid = rows[:, 4] >= 0
        rhs = rows[:, 4].clamp_min(0)
        lhs_before = copy_slots.index_select(0, lhs)
        rhs_before = copy_slots.index_select(0, rhs)
        rhs_before = torch.where(
            rhs_valid.view(-1, 1),
            rhs_before,
            torch.full_like(rhs_before, num_slots),
        )
        lhs_after, rhs_after, _ = self._candidate_copy_options(
            torch.empty(num_slots, dtype=torch.long, device=rows.device),
            copy_slots,
            rows,
        )
        rhs_after = torch.where(
            rhs_valid.view(-1, 1),
            rhs_after,
            torch.full_like(rhs_after, num_slots),
        )
        all_options = torch.cat((lhs_before, lhs_after, rhs_before, rhs_after), dim=1)
        valid = all_options < num_slots
        ranks = torch.div(all_options.clamp(max=num_slots - 1), self.slots_per_rank, rounding_mode="floor")
        level_sizes = (1,) + tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        )
        if rows.device.type == "cpu":
            rank_values = ranks.numpy()
            valid_values = valid.numpy()
            packed_values = []
            offset = 0
            for size in level_sizes:
                num_groups = self.ep_size // size
                groups = np.where(valid_values, np.floor_divide(rank_values, size), num_groups)
                first = valid_values.copy()
                for position in range(1, groups.shape[1]):
                    first[:, position] &= np.all(
                        groups[:, position : position + 1] != groups[:, :position],
                        axis=1,
                    )
                packed_values.append(np.where(first, groups + offset, -1))
                offset += num_groups
            return torch.from_numpy(np.concatenate(packed_values, axis=1))

        packed_levels: list[torch.Tensor] = []
        offset = 0
        option_count = all_options.shape[1]
        positions = torch.arange(option_count, dtype=torch.long, device=rows.device)
        earlier = positions.view(1, 1, -1) < positions.view(1, -1, 1)
        for size in level_sizes:
            num_groups = self.ep_size // size
            groups = torch.div(ranks, size, rounding_mode="floor")
            groups = torch.where(valid, groups, torch.full_like(groups, num_groups))
            duplicate = groups.unsqueeze(2).eq(groups.unsqueeze(1)) & earlier
            first = (groups < num_groups) & ~duplicate.any(dim=2)
            packed_levels.append(torch.where(first, groups + offset, torch.full_like(groups, -1)))
            offset += num_groups
        return torch.cat(packed_levels, dim=1)

    @staticmethod
    def _candidate_affected_assignment_ranks(
        affected_groups: torch.Tensor,
        copy_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse the rank-level affected groups for compact assignment deltas."""

        rank_option_count = 4 * int(copy_slots.shape[1]) + 2
        return affected_groups[:, :rank_option_count]

    def _statistical_assignment_local_deltas(
        self,
        selected: torch.Tensor,
        rows: torch.Tensor,
        route_hashes: torch.Tensor,
        route_tables: StatisticalRouteTables,
    ) -> torch.Tensor:
        """Compute exact non-deduplicated rank-load deltas from hash-state tables."""

        state_count = route_tables.state_count
        num_experts = int(route_tables.baseline_slots.shape[0])
        route_states = torch.remainder(route_hashes, state_count)
        pseudo = selected * state_count + route_states
        multiplicities = torch.zeros(
            (num_experts * state_count,),
            dtype=torch.float32,
            device=selected.device,
        )
        multiplicities.index_add_(
            0,
            pseudo.reshape(-1),
            torch.ones_like(pseudo, dtype=torch.float32).reshape(-1),
        )
        multiplicities = multiplicities.view(num_experts, state_count)
        lhs = rows[:, 3]
        rhs_valid = rows[:, 4] >= 0
        rhs = rows[:, 4].clamp_min(0)
        baseline_ranks = torch.div(
            route_tables.baseline_slots,
            self.slots_per_rank,
            rounding_mode="floor",
        )
        lhs_old = baseline_ranks.index_select(0, lhs)
        rhs_old = baseline_ranks.index_select(0, rhs)
        lhs_new = torch.div(route_tables.lhs_slots, self.slots_per_rank, rounding_mode="floor")
        rhs_new = torch.div(route_tables.rhs_slots, self.slots_per_rank, rounding_mode="floor")
        lhs_weights = multiplicities.index_select(0, lhs)
        rhs_weights = multiplicities.index_select(0, rhs) * rhs_valid.view(-1, 1).to(torch.float32)
        deltas = torch.zeros((rows.shape[0], self.ep_size), dtype=torch.float32, device=selected.device)
        for ranks, values in (
            (lhs_old, -lhs_weights),
            (lhs_new, lhs_weights),
            (rhs_old, -rhs_weights),
            (rhs_new, rhs_weights),
        ):
            deltas.scatter_add_(1, ranks, values)
        return deltas

    def _reference_assignment_local_deltas(
        self,
        selected: torch.Tensor,
        rows: torch.Tensor,
        *,
        layout: torch.Tensor,
        copy_slots: torch.Tensor,
        physical: torch.Tensor,
        source_ranks: torch.Tensor,
        route_hashes: torch.Tensor,
    ) -> torch.Tensor:
        """Exact fallback for non-uniform sources without compact route tables."""

        num_tokens = selected.shape[0]
        num_experts = int(copy_slots.shape[0])
        route_hash_by_expert = torch.zeros(
            (num_tokens, num_experts),
            dtype=torch.long,
            device=selected.device,
        )
        route_hash_by_expert.scatter_(1, selected, route_hashes)
        multiplicity_by_expert = torch.zeros(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device=selected.device,
        )
        multiplicity_by_expert.scatter_add_(1, selected, torch.ones_like(selected, dtype=torch.float32))
        rank_by_expert = torch.zeros_like(route_hash_by_expert)
        rank_by_expert.scatter_(
            1,
            selected,
            torch.div(physical, self.slots_per_rank, rounding_mode="floor"),
        )
        lhs = rows[:, 3]
        rhs_valid = rows[:, 4] >= 0
        rhs = rows[:, 4].clamp_min(0)
        lhs_options, rhs_options, _ = self._candidate_copy_options(layout, copy_slots, rows)
        lhs_hash = route_hash_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_hash = route_hash_by_expert.index_select(1, rhs).transpose(0, 1)
        lhs_old = rank_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_old = rank_by_expert.index_select(1, rhs).transpose(0, 1)
        lhs_new = self._candidate_route_ranks(
            lhs_options,
            lhs_hash,
            source_ranks,
            int(layout.numel()),
        )
        rhs_new = self._candidate_route_ranks(
            rhs_options,
            rhs_hash,
            source_ranks,
            int(layout.numel()),
        )
        lhs_weights = multiplicity_by_expert.index_select(1, lhs).transpose(0, 1)
        rhs_weights = multiplicity_by_expert.index_select(1, rhs).transpose(0, 1)
        rhs_weights *= rhs_valid.view(-1, 1).to(rhs_weights.dtype)
        deltas = torch.zeros((rows.shape[0], self.ep_size), dtype=torch.float32, device=selected.device)
        for ranks, values in (
            (lhs_old, -lhs_weights),
            (lhs_new, lhs_weights),
            (rhs_old, -rhs_weights),
            (rhs_new, rhs_weights),
        ):
            deltas.scatter_add_(1, ranks, values)
        return deltas

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
        route_hashes: torch.Tensor | None = None,
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
        if route_hashes is None:
            lhs_hash = _route_hash(
                lhs_selected,
                token_ordinals=token_ordinals,
                step=step,
                layer_seed=layer_seed,
            ).transpose(0, 1)
        else:
            lhs_hash = (
                torch.where(selected == lhs, route_hashes, torch.zeros_like(route_hashes))
                .max(dim=1, keepdim=True)
                .values.transpose(0, 1)
            )
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
            if route_hashes is None:
                rhs_hash = _route_hash(
                    rhs_selected,
                    token_ordinals=token_ordinals,
                    step=step,
                    layer_seed=layer_seed,
                ).transpose(0, 1)
            else:
                rhs_hash = (
                    torch.where(selected == rhs, route_hashes, torch.zeros_like(route_hashes))
                    .max(dim=1, keepdim=True)
                    .values.transpose(0, 1)
                )
            rhs_slots = self._candidate_route_slots(
                rhs_options,
                rhs_hash,
                source_ranks,
                int(layout.numel()),
            )[0]
            updated = torch.where(selected == rhs, rhs_slots.view(-1, 1), updated)
        return updated

    @staticmethod
    def _apply_statistical_action_routes(
        selected: torch.Tensor,
        row: torch.Tensor,
        action_index: int,
        baseline_physical: torch.Tensor,
        route_hashes: torch.Tensor,
        route_tables: StatisticalRouteTables,
    ) -> torch.Tensor:
        """Install one winner from the route tables already used for scoring."""

        route_states = torch.remainder(route_hashes, route_tables.state_count)
        lhs_table = route_tables.lhs_slots[action_index]
        rhs_table = route_tables.rhs_slots[action_index]
        lhs_slots = lhs_table.index_select(0, route_states.reshape(-1)).view_as(selected)
        rhs_slots = rhs_table.index_select(0, route_states.reshape(-1)).view_as(selected)
        updated = torch.where(selected == row[3], lhs_slots, baseline_physical)
        return torch.where(selected == row[4], rhs_slots, updated)

    def _token_level_occupancies(
        self,
        physical_slots: torch.Tensor,
        *,
        route_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        ranks = torch.div(physical_slots, self.slots_per_rank, rounding_mode="floor")
        num_tokens, top_k = ranks.shape
        occupancies = []
        level_sizes = (1,) + tuple(
            int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        )
        updates = (
            torch.ones((num_tokens, top_k), dtype=torch.int32, device=ranks.device)
            if route_weights is None
            else route_weights.to(dtype=torch.int32, device=ranks.device)
        )
        for size in level_sizes:
            groups = torch.div(ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            counts = torch.zeros((num_tokens, num_groups), dtype=torch.int32, device=ranks.device)
            counts.scatter_add_(1, groups, updates)
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
            or selected.device.type != "npu"
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

    def _prepare_action_counts(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        rows: torch.Tensor,
        *,
        source_ranks: torch.Tensor,
        uniform_source_rank: int | None,
        copy_slots: torch.Tensor | None,
        affected_groups: torch.Tensor | None,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
        include_assignment_counts: bool | None = None,
        include_pair_interactions: bool = True,
        include_pair_bounds: bool = False,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> _PreparedActionCounts:
        route_hashes = _route_hash(
            selected,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
        if prepare_stage_callback is not None:
            prepare_stage_callback("route_hash")
        if copy_slots is None:
            copy_slots = self._copy_table(layout, num_experts)
        uniform_baseline = None
        if self.candidate_scorer == "statistics" and uniform_source_rank is not None:
            uniform_baseline = uniform_statistical_baseline_routes(
                self,
                selected,
                copy_slots,
                route_hashes,
                source_rank=uniform_source_rank,
            )
        physical = None if uniform_baseline is None else uniform_baseline.physical
        if physical is None:
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
                route_hashes=route_hashes,
            )
        if prepare_stage_callback is not None:
            prepare_stage_callback("baseline_route")
        unique_routes = None
        if self.candidate_scorer == "statistics":
            unique_routes = None if self.assume_unique_routes else _canonical_route_mask(selected)
        occupancies = self._token_level_occupancies(
            physical,
            route_weights=None if self.assume_unique_routes else unique_routes,
        )
        baseline_local = (torch.cat(occupancies, dim=1) > 0).sum(dim=0, keepdim=True).to(torch.float32)
        if prepare_stage_callback is not None:
            prepare_stage_callback("occupancy")
        candidate_delta = None
        route_tables = None
        pair_context = None
        candidate_pair_bound_local = None
        if self.candidate_scorer == "statistics":
            unary = None
            if not include_pair_interactions:
                unary = statistical_unary_candidate_local_deltas(
                    self,
                    selected,
                    rows,
                    layout=layout,
                    copy_slots=copy_slots,
                    physical=physical,
                    occupancies=occupancies,
                    token_ordinals=token_ordinals,
                    route_hashes=route_hashes,
                    uniform_source_rank=uniform_source_rank,
                    uniform_baseline=uniform_baseline,
                    routes_are_unique=self.assume_unique_routes,
                    unique_routes=unique_routes,
                    step=step,
                    layer_seed=layer_seed,
                    num_experts=num_experts,
                    prepare_stage_callback=prepare_stage_callback,
                )
            if unary is not None:
                candidate_delta, route_tables, pair_context = unary
                if include_pair_bounds:
                    candidate_pair_bound_local = statistical_pair_interaction_bound_local(
                        pair_context,
                        rows,
                    )
            else:
                statistical = statistical_candidate_local_deltas(
                    self,
                    selected,
                    rows,
                    layout=layout,
                    copy_slots=copy_slots,
                    physical=physical,
                    occupancies=occupancies,
                    source_ranks=source_ranks,
                    token_ordinals=token_ordinals,
                    route_hashes=route_hashes,
                    uniform_source_rank=uniform_source_rank,
                    uniform_baseline=uniform_baseline,
                    routes_are_unique=self.assume_unique_routes,
                    unique_routes=unique_routes,
                    return_route_tables=True,
                    step=step,
                    layer_seed=layer_seed,
                    num_experts=num_experts,
                    prepare_stage_callback=prepare_stage_callback,
                )
                assert isinstance(statistical, tuple)
                candidate_delta, route_tables = statistical
        elif self.candidate_scorer == "reference":
            candidate_delta = self._fused_candidate_local_deltas(
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
        elif candidate_delta is not None:
            candidate_local = [baseline_local + candidate_delta]
        else:
            route_hashes = _route_hash(selected, token_ordinals=token_ordinals, step=step, layer_seed=layer_seed)
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
        candidates = (
            torch.cat(candidate_local, dim=0)
            if candidate_local
            else baseline_local.new_empty((0, baseline_local.shape[1]))
        )
        if (
            affected_groups is None
            and self.compact_candidate_collective
            and self.reducer is not None
            and dist.is_initialized()
        ):
            affected_groups = self._candidate_affected_groups(copy_slots, rows)
        baseline_assignment_local = None
        candidate_assignment_local = None
        affected_assignment_ranks = None
        include_assignments = (
            self.forward_compute_per_assignment > 0.0
            if include_assignment_counts is None
            else bool(include_assignment_counts)
        )
        if include_assignments:
            baseline_assignment_local = self._local_assignment_counts(physical)
            if rows.numel() == 0:
                candidate_assignment_local = baseline_assignment_local.new_empty((0, self.ep_size))
            else:
                if route_tables is not None:
                    assignment_delta = self._statistical_assignment_local_deltas(
                        selected,
                        rows,
                        route_hashes,
                        route_tables,
                    )
                else:
                    assignment_delta = self._reference_assignment_local_deltas(
                        selected,
                        rows,
                        layout=layout,
                        copy_slots=copy_slots,
                        physical=physical,
                        source_ranks=source_ranks,
                        route_hashes=route_hashes,
                    )
                candidate_assignment_local = baseline_assignment_local + assignment_delta
            if affected_groups is not None:
                affected_assignment_ranks = self._candidate_affected_assignment_ranks(
                    affected_groups,
                    copy_slots,
                )
        if prepare_stage_callback is not None:
            prepare_stage_callback("candidate_pack")
        return _PreparedActionCounts(
            baseline_local=baseline_local,
            candidate_local=candidates,
            baseline_assignment_local=baseline_assignment_local,
            candidate_assignment_local=candidate_assignment_local,
            affected_groups=affected_groups,
            affected_assignment_ranks=affected_assignment_ranks,
            baseline_physical_routes=physical,
            route_hashes=route_hashes,
            route_tables=route_tables,
            pair_context=pair_context,
            candidate_pair_bound_local=candidate_pair_bound_local,
        )

    def _prepare_proxy_action_counts(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        rows: torch.Tensor,
        *,
        uniform_source_rank: int | None,
        copy_slots: torch.Tensor,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
        include_assignment_counts: bool,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> _PreparedActionCounts:
        """Build the state-collapsed unary proxy before exact candidate routes."""

        proxy = statistical_proxy_candidate_local_deltas(
            self,
            selected,
            rows,
            layout=layout,
            copy_slots=copy_slots,
            uniform_source_rank=uniform_source_rank,
            routes_are_unique=self.assume_unique_routes,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            num_experts=num_experts,
            prepare_stage_callback=prepare_stage_callback,
        )
        if proxy is None:
            raise RuntimeError("The early proxy requires non-empty rows and one uniform source rank per EP process.")
        baseline_local = self._local_packed_counts(proxy.physical)
        candidate_local = baseline_local + proxy.candidate_delta
        baseline_assignment_local = None
        candidate_assignment_local = None
        if include_assignment_counts:
            baseline_assignment_local = self._local_assignment_counts(proxy.physical)
            assignment_delta = self._statistical_assignment_local_deltas(
                selected,
                rows,
                proxy.route_hashes,
                proxy.route_tables,
            )
            candidate_assignment_local = baseline_assignment_local + assignment_delta
        if prepare_stage_callback is not None:
            prepare_stage_callback("proxy_candidate_pack")
        return _PreparedActionCounts(
            baseline_local=baseline_local,
            candidate_local=candidate_local,
            baseline_assignment_local=baseline_assignment_local,
            candidate_assignment_local=candidate_assignment_local,
            affected_groups=None,
            affected_assignment_ranks=None,
            baseline_physical_routes=proxy.physical,
            route_hashes=proxy.route_hashes,
            route_tables=proxy.route_tables,
        )

    def _primitive_assignment_local_deltas(
        self,
        selected: torch.Tensor,
        route_hashes: torch.Tensor,
        context: StatisticalPrimitiveContext,
    ) -> torch.Tensor:
        """Build exact non-deduplicated rank deltas once per primitive."""

        state_count = context.state_count
        route_states = torch.remainder(route_hashes, state_count)
        pseudo = selected * state_count + route_states
        multiplicities = torch.zeros(
            (context.num_experts * state_count,),
            dtype=torch.float32,
            device=selected.device,
        )
        multiplicities.index_add_(
            0,
            pseudo.reshape(-1),
            torch.ones_like(pseudo, dtype=torch.float32).reshape(-1),
        )
        multiplicities = multiplicities.view(context.num_experts, state_count)
        experts = context.spec.experts
        weights = multiplicities.index_select(0, experts)
        old_ranks = torch.div(
            context.baseline_slots.index_select(0, experts),
            self.slots_per_rank,
            rounding_mode="floor",
        )
        new_ranks = torch.div(context.primitive_slots, self.slots_per_rank, rounding_mode="floor")
        delta = torch.zeros(
            (experts.numel(), self.ep_size),
            dtype=torch.float32,
            device=selected.device,
        )
        delta.scatter_add_(1, old_ranks, -weights)
        delta.scatter_add_(1, new_ranks, weights)
        return delta

    def _primitive_affected_metadata(
        self,
        copy_slots: torch.Tensor,
        spec: StatisticalPrimitiveSpec,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Build layout-only sparse primitive rank/group indices."""

        old_slots = copy_slots.index_select(0, spec.experts)
        option_width = int(spec.options.shape[1])
        if old_slots.shape[1] < option_width:
            old_slots = torch.cat(
                (
                    old_slots,
                    torch.full(
                        (old_slots.shape[0], option_width - old_slots.shape[1]),
                        self.ep_size * self.slots_per_rank,
                        dtype=torch.long,
                        device=old_slots.device,
                    ),
                ),
                dim=1,
            )
        all_slots = torch.cat((old_slots, spec.options), dim=1)
        valid_slots = all_slots < self.ep_size * self.slots_per_rank
        affected_ranks = torch.div(
            all_slots.clamp(max=self.ep_size * self.slots_per_rank - 1),
            self.slots_per_rank,
            rounding_mode="floor",
        )
        affected_ranks = torch.where(
            valid_slots,
            affected_ranks,
            torch.full_like(affected_ranks, self.ep_size),
        )
        affected_ranks = affected_ranks.sort(dim=1).values
        unique_rank = torch.ones_like(affected_ranks, dtype=torch.bool)
        unique_rank[:, 1:] = affected_ranks[:, 1:] != affected_ranks[:, :-1]
        affected_ranks = (
            torch.where(
                unique_rank,
                affected_ranks,
                torch.full_like(affected_ranks, self.ep_size),
            )
            .sort(dim=1)
            .values
        )
        affected_groups = []
        for level_size, width in zip(
            (1,)
            + tuple(int(size) for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]),
            self._count_widths(),
            strict=True,
        ):
            valid_ranks = affected_ranks < self.ep_size
            groups = torch.div(
                affected_ranks.clamp(max=self.ep_size - 1),
                level_size,
                rounding_mode="floor",
            )
            groups = torch.where(valid_ranks, groups, torch.full_like(groups, width))
            groups = groups.sort(dim=1).values
            unique_group = torch.ones_like(groups, dtype=torch.bool)
            unique_group[:, 1:] = groups[:, 1:] != groups[:, :-1]
            groups = (
                torch.where(
                    unique_group,
                    groups,
                    torch.full_like(groups, width),
                )
                .sort(dim=1)
                .values
            )
            affected_groups.append(groups[:, : min(int(groups.shape[1]), width)])
        return affected_ranks, tuple(affected_groups)

    def _prepare_primitive_counts(
        self,
        context: dict[str, object],
        *,
        step: int,
        include_assignment_counts: bool,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> _PreparedPrimitiveCounts:
        result = statistical_primitive_unary_local_deltas(
            self,
            context["selected"],
            context["primitive_spec"],
            copy_slots=context["copy_slots"],
            token_ordinals=context["ordinals"],
            uniform_source_rank=context["uniform_source_rank"],
            routes_are_unique=self.assume_unique_routes,
            step=step,
            layer_seed=context["layer_seed"],
            num_experts=int(context["owners"].numel()),
            defer_pair_statistics=self.post_shortlist_compact_pair,
            prepare_stage_callback=prepare_stage_callback,
        )
        if result is None:
            raise RuntimeError("Exact primitive scoring requires non-empty actions and a uniform source rank.")
        affected_ranks = context.get("primitive_affected_ranks")
        affected_groups = context.get("primitive_affected_groups")
        if affected_ranks is None or affected_groups is None:
            affected_ranks, affected_groups = self._primitive_affected_metadata(
                context["copy_slots"],
                result.context.spec,
            )
        baseline_local = self._local_packed_counts(result.physical)
        baseline_assignment_local = None
        primitive_assignment_delta_local = None
        if include_assignment_counts:
            baseline_assignment_local = self._local_assignment_counts(result.physical)
            primitive_assignment_delta_local = self._primitive_assignment_local_deltas(
                context["selected"],
                result.route_hashes,
                result.context,
            )
        if prepare_stage_callback is not None:
            prepare_stage_callback("candidate_pack")
        return _PreparedPrimitiveCounts(
            baseline_local=baseline_local,
            primitive_delta_local=result.primitive_delta,
            baseline_assignment_local=baseline_assignment_local,
            primitive_assignment_delta_local=primitive_assignment_delta_local,
            primitive_affected_ranks=affected_ranks,
            primitive_affected_groups=affected_groups,
            context=result.context,
            baseline_physical_routes=result.physical,
            route_hashes=result.route_hashes,
        )

    def _score_actions(
        self,
        selected: torch.Tensor,
        layout: torch.Tensor,
        rows: torch.Tensor,
        *,
        source_ranks: torch.Tensor,
        uniform_source_rank: int | None,
        copy_slots: torch.Tensor | None,
        affected_groups: torch.Tensor | None,
        token_ordinals: torch.Tensor,
        step: int,
        layer_seed: int,
        num_experts: int,
    ) -> _ScoredLayouts:
        prepared = self._prepare_action_counts(
            selected,
            layout,
            rows,
            source_ranks=source_ranks,
            uniform_source_rank=uniform_source_rank,
            copy_slots=copy_slots,
            affected_groups=affected_groups,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
            num_experts=num_experts,
        )
        communication, compute, units, peak_rank, peak_compute_rank, selected_dim = self._global_action_costs(
            prepared.baseline_local,
            prepared.candidate_local,
            prepared.baseline_assignment_local,
            prepared.candidate_assignment_local,
            prepared.affected_groups,
            prepared.affected_assignment_ranks,
        )
        return _ScoredLayouts(
            communication=communication,
            compute=compute,
            communication_model_units=units,
            peak_rank=peak_rank,
            peak_compute_rank=peak_compute_rank,
            selected_dim=selected_dim,
            baseline_physical_routes=prepared.baseline_physical_routes,
            route_hashes=prepared.route_hashes,
            route_tables=prepared.route_tables,
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

    def _can_complete_empty_initialization(self, layout: torch.Tensor, requested_fills: int) -> bool:
        """Check replica-fill feasibility with a small CPU bipartite max flow."""

        requested = max(0, int(requested_fills))
        if requested == 0:
            return True
        active = layout[layout >= 0]
        if active.numel() == 0:
            return False
        num_experts = int(active.max().item()) + 1
        copy_counts = torch.bincount(active, minlength=num_experts)
        capacities = (self.max_copies - copy_counts).clamp_min(0).tolist()
        empty_by_rank = (layout.view(self.ep_size, self.slots_per_rank) < 0).sum(dim=1).to(dtype=torch.long).tolist()
        if sum(empty_by_rank) < requested or sum(capacities) < requested:
            return False

        rank_experts = [
            {
                int(value)
                for value in layout[rank * self.slots_per_rank : (rank + 1) * self.slots_per_rank].tolist()
                if int(value) >= 0
            }
            for rank in range(self.ep_size)
        ]
        source = 0
        rank_offset = 1
        expert_offset = rank_offset + self.ep_size
        sink = expert_offset + num_experts
        graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

        def add_edge(lhs: int, rhs: int, capacity: int) -> None:
            graph[lhs].append([rhs, len(graph[rhs]), capacity])
            graph[rhs].append([lhs, len(graph[lhs]) - 1, 0])

        for rank, empty_count in enumerate(empty_by_rank):
            if empty_count <= 0:
                continue
            add_edge(source, rank_offset + rank, empty_count)
            for expert, capacity in enumerate(capacities):
                if capacity > 0 and expert not in rank_experts[rank]:
                    add_edge(rank_offset + rank, expert_offset + expert, 1)
        for expert, capacity in enumerate(capacities):
            if capacity > 0:
                add_edge(expert_offset + expert, sink, capacity)

        flow = 0
        while flow < requested:
            levels = [-1] * len(graph)
            levels[source] = 0
            queue = deque((source,))
            while queue:
                lhs = queue.popleft()
                for rhs, _reverse, capacity in graph[lhs]:
                    if capacity > 0 and levels[rhs] < 0:
                        levels[rhs] = levels[lhs] + 1
                        queue.append(rhs)
            if levels[sink] < 0:
                return False
            positions = [0] * len(graph)

            def send(
                lhs: int,
                available: int,
                levels: list[int] = levels,
                positions: list[int] = positions,
            ) -> int:
                if lhs == sink:
                    return available
                while positions[lhs] < len(graph[lhs]):
                    edge = graph[lhs][positions[lhs]]
                    rhs, reverse, capacity = edge
                    if capacity > 0 and levels[rhs] == levels[lhs] + 1:
                        sent = send(rhs, min(available, capacity))
                        if sent:
                            edge[2] -= sent
                            graph[rhs][reverse][2] += sent
                            return sent
                    positions[lhs] += 1
                return 0

            while flow < requested:
                sent = send(source, requested - flow)
                if not sent:
                    break
                flow += sent
        return flow >= requested

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
                    scored.compute[index],
                    scored.communication_model_units[index],
                    scored.peak_rank[index].to(scored.communication.dtype),
                    scored.peak_compute_rank[index].to(scored.communication.dtype),
                    scored.selected_dim[index].to(scored.communication.dtype),
                )
            )
            .detach()
            .to(device="cpu")
            .tolist()
        )
        return GreedyCommunicationPlanner._placement_cost_from_values(values)

    @staticmethod
    def _placement_cost_from_values(values: Sequence[float]) -> PlacementCost:
        return PlacementCost(
            communication=float(values[0]),
            compute=float(values[1]),
            communication_model_units=float(values[2]),
            peak_communication_rank=int(values[3]),
            peak_compute_rank=int(values[4]),
            selected_dim=int(values[5]),
        )

    def _score_prepared_layers(
        self,
        prepared_layers: Sequence[_PreparedActionCounts],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> list[_ScoredLayouts]:
        """Score independent layers with one full EP reduction."""

        if not prepared_layers:
            return []
        layer_count = len(prepared_layers)
        if not (
            len(communication_scales)
            == len(forward_compute_per_assignment)
            == len(forward_compute_constant)
            == layer_count
        ):
            raise ValueError("Batched layer cost-model arrays must match the number of prepared layers.")
        include_assignments = any(value > 0.0 for value in forward_compute_per_assignment)
        include_pair_bounds = all(prepared.candidate_pair_bound_local is not None for prepared in prepared_layers)
        communication_width = prepared_layers[0].baseline_local.shape[1]
        local_blocks: list[torch.Tensor] = []
        row_counts: list[int] = []
        for prepared in prepared_layers:
            communication = torch.cat((prepared.baseline_local, prepared.candidate_local), dim=0)
            if communication.shape[1] != communication_width:
                raise ValueError("Batched layers must use the same communication-statistic width.")
            row_counts.append(int(communication.shape[0]))
            if include_assignments:
                if prepared.baseline_assignment_local is None or prepared.candidate_assignment_local is None:
                    raise ValueError("Batched joint-cost scoring requires assignment counts for every layer.")
                assignments = torch.cat(
                    (prepared.baseline_assignment_local, prepared.candidate_assignment_local),
                    dim=0,
                )
                communication = torch.cat((communication, assignments), dim=1)
            if include_pair_bounds:
                assert prepared.candidate_pair_bound_local is not None
                pair_bounds = torch.cat(
                    (
                        prepared.candidate_pair_bound_local.new_zeros((1, communication_width)),
                        prepared.candidate_pair_bound_local,
                    ),
                    dim=0,
                )
                communication = torch.cat((communication, pair_bounds), dim=1)
            local_blocks.append(communication)

        local_payload = torch.cat(local_blocks, dim=0)
        if prepare_stage_callback is not None:
            prepare_stage_callback("collective_pack")
        global_rows = _reduce_sum(local_payload, self.reducer)
        communication_rows = global_rows[:, :communication_width]
        assignment_end = communication_width + (self.ep_size if include_assignments else 0)
        assignment_rows = global_rows[:, communication_width:assignment_end] if include_assignments else None
        pair_bound_rows = global_rows[:, -communication_width:] if include_pair_bounds else None
        _unused, units, peak_rank, selected_dim = self._communication_cost_details(communication_rows)
        device = global_rows.device
        layer_indices = torch.repeat_interleave(
            torch.arange(layer_count, dtype=torch.long, device=device),
            torch.tensor(row_counts, dtype=torch.long, device=device),
        )
        communication_scale_rows = torch.tensor(
            communication_scales,
            dtype=units.dtype,
            device=device,
        ).index_select(0, layer_indices)
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * communication_scale_rows * units
        compute_slope_rows = torch.tensor(
            forward_compute_per_assignment,
            dtype=units.dtype,
            device=device,
        ).index_select(0, layer_indices)
        compute_constant_rows = torch.tensor(
            forward_compute_constant,
            dtype=units.dtype,
            device=device,
        ).index_select(0, layer_indices)
        if assignment_rows is None:
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * compute_constant_rows
            peak_compute_rank = torch.full_like(peak_rank, -1)
        else:
            peak_assignments, peak_compute_rank = assignment_rows.max(dim=1)
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (compute_slope_rows * peak_assignments + compute_constant_rows)

        exact_cost_lower_bound = None
        if pair_bound_rows is not None:
            lower_count_rows = (communication_rows - pair_bound_rows).clamp_min_(0)
            _unused, lower_units, _lower_peak, _lower_dim = self._communication_cost_details(lower_count_rows)
            lower_communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * communication_scale_rows * lower_units
            exact_cost_lower_bound = lower_communication + compute

        scored_layers: list[_ScoredLayouts] = []
        offset = 0
        for prepared, row_count in zip(prepared_layers, row_counts, strict=True):
            rows = slice(offset, offset + row_count)
            scored_layers.append(
                _ScoredLayouts(
                    communication=communication[rows],
                    compute=compute[rows],
                    communication_model_units=units[rows],
                    peak_rank=peak_rank[rows],
                    peak_compute_rank=peak_compute_rank[rows],
                    selected_dim=selected_dim[rows],
                    baseline_physical_routes=prepared.baseline_physical_routes,
                    route_hashes=prepared.route_hashes,
                    route_tables=prepared.route_tables,
                    exact_cost_lower_bound=(None if exact_cost_lower_bound is None else exact_cost_lower_bound[rows]),
                )
            )
            offset += row_count
        return scored_layers

    def _score_global_count_rows(
        self,
        communication_rows: torch.Tensor,
        assignment_rows: torch.Tensor | None,
        *,
        communication_scale: float,
        forward_compute_per_assignment: float,
        forward_compute_constant: float,
        baseline_physical_routes: torch.Tensor,
        route_hashes: torch.Tensor,
        route_tables: StatisticalRouteTables | None = None,
    ) -> _ScoredLayouts:
        """Apply the joint cost model to already-global count rows."""

        _unused, units, peak_rank, selected_dim = self._communication_cost_details(communication_rows)
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * float(communication_scale) * units
        if assignment_rows is None:
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * torch.full_like(
                communication,
                float(forward_compute_constant),
            )
            peak_compute_rank = torch.full_like(peak_rank, -1)
        else:
            peak_assignments, peak_compute_rank = assignment_rows.max(dim=1)
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (
                float(forward_compute_per_assignment) * peak_assignments + float(forward_compute_constant)
            )
        return _ScoredLayouts(
            communication=communication,
            compute=compute,
            communication_model_units=units,
            peak_rank=peak_rank,
            peak_compute_rank=peak_compute_rank,
            selected_dim=selected_dim,
            baseline_physical_routes=baseline_physical_routes,
            route_hashes=route_hashes,
            route_tables=route_tables,
        )

    def _score_batched_primitive_actions(
        self,
        prepared_layers: Sequence[_PreparedPrimitiveCounts],
        global_layers: Sequence[_GlobalPrimitiveCounts],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
        batch_layers: bool,
    ) -> list[_ScoredLayouts]:
        """Compose and score the layer/action batch with one eager graph."""

        layer_count = len(prepared_layers)
        if layer_count == 0:
            return []
        primitive_counts = [int(value.primitive_delta.shape[0]) for value in global_layers]
        action_counts = [int(value.context.spec.lhs_ids.numel()) for value in prepared_layers]
        if not batch_layers or len(set(primitive_counts)) != 1 or len(set(action_counts)) != 1:
            scored = []
            for layer_index, (prepared, global_counts) in enumerate(zip(prepared_layers, global_layers, strict=True)):
                spec = prepared.context.spec
                rhs_ids = spec.rhs_ids.clamp_min(0)
                candidate_communication = (
                    global_counts.baseline
                    + global_counts.primitive_delta.index_select(0, spec.lhs_ids)
                    + global_counts.primitive_delta.index_select(0, rhs_ids)
                    * spec.rhs_valid.view(-1, 1).to(global_counts.primitive_delta.dtype)
                )
                communication_rows = torch.cat(
                    (global_counts.baseline, candidate_communication),
                    dim=0,
                )
                assignment_rows = None
                if global_counts.baseline_assignment is not None:
                    assert global_counts.primitive_assignment_delta is not None
                    candidate_assignment = (
                        global_counts.baseline_assignment
                        + global_counts.primitive_assignment_delta.index_select(0, spec.lhs_ids)
                        + global_counts.primitive_assignment_delta.index_select(0, rhs_ids)
                        * spec.rhs_valid.view(-1, 1).to(global_counts.primitive_assignment_delta.dtype)
                    )
                    assignment_rows = torch.cat(
                        (global_counts.baseline_assignment, candidate_assignment),
                        dim=0,
                    )
                scored.append(
                    self._score_global_count_rows(
                        communication_rows,
                        assignment_rows,
                        communication_scale=communication_scales[layer_index],
                        forward_compute_per_assignment=forward_compute_per_assignment[layer_index],
                        forward_compute_constant=forward_compute_constant[layer_index],
                        baseline_physical_routes=prepared.baseline_physical_routes,
                        route_hashes=prepared.route_hashes,
                    )
                )
            return scored

        action_count = action_counts[0]
        communication_width = int(global_layers[0].baseline.shape[1])
        baselines = torch.stack([value.baseline[0] for value in global_layers])
        primitive_deltas = torch.stack([value.primitive_delta for value in global_layers])
        lhs_ids = torch.stack([value.context.spec.lhs_ids for value in prepared_layers])
        rhs_ids = torch.stack([value.context.spec.rhs_ids.clamp_min(0) for value in prepared_layers])
        rhs_valid = torch.stack([value.context.spec.rhs_valid for value in prepared_layers])
        lhs_delta = primitive_deltas.gather(
            1,
            lhs_ids.unsqueeze(2).expand(-1, -1, communication_width),
        )
        rhs_delta = primitive_deltas.gather(
            1,
            rhs_ids.unsqueeze(2).expand(-1, -1, communication_width),
        )
        candidate_communication = (
            baselines.unsqueeze(1) + lhs_delta + rhs_delta * rhs_valid.unsqueeze(2).to(rhs_delta.dtype)
        )
        communication_rows = torch.cat(
            (baselines.unsqueeze(1), candidate_communication),
            dim=1,
        )
        row_count = action_count + 1
        _unused, units, peak_rank, selected_dim = self._communication_cost_details(
            communication_rows.reshape(layer_count * row_count, communication_width)
        )
        communication_scale_rows = (
            torch.tensor(
                communication_scales,
                dtype=units.dtype,
                device=units.device,
            )
            .view(layer_count, 1)
            .expand(-1, row_count)
            .reshape(-1)
        )
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * communication_scale_rows * units

        include_assignments = global_layers[0].baseline_assignment is not None
        if include_assignments:
            if any(
                value.baseline_assignment is None or value.primitive_assignment_delta is None
                for value in global_layers
            ):
                raise ValueError("Batched primitive assignment statistics must be present for every layer.")
            assignment_baselines = torch.stack(
                [value.baseline_assignment[0] for value in global_layers if value.baseline_assignment is not None]
            )
            assignment_deltas = torch.stack(
                [
                    value.primitive_assignment_delta
                    for value in global_layers
                    if value.primitive_assignment_delta is not None
                ]
            )
            lhs_assignment = assignment_deltas.gather(
                1,
                lhs_ids.unsqueeze(2).expand(-1, -1, self.ep_size),
            )
            rhs_assignment = assignment_deltas.gather(
                1,
                rhs_ids.unsqueeze(2).expand(-1, -1, self.ep_size),
            )
            candidate_assignment = (
                assignment_baselines.unsqueeze(1)
                + lhs_assignment
                + rhs_assignment * rhs_valid.unsqueeze(2).to(rhs_assignment.dtype)
            )
            assignment_rows = torch.cat(
                (assignment_baselines.unsqueeze(1), candidate_assignment),
                dim=1,
            ).reshape(layer_count * row_count, self.ep_size)
            peak_assignments, peak_compute_rank = assignment_rows.max(dim=1)
            compute_slopes = (
                torch.tensor(
                    forward_compute_per_assignment,
                    dtype=peak_assignments.dtype,
                    device=peak_assignments.device,
                )
                .view(layer_count, 1)
                .expand(-1, row_count)
                .reshape(-1)
            )
            compute_constants = (
                torch.tensor(
                    forward_compute_constant,
                    dtype=peak_assignments.dtype,
                    device=peak_assignments.device,
                )
                .view(layer_count, 1)
                .expand(-1, row_count)
                .reshape(-1)
            )
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (compute_slopes * peak_assignments + compute_constants)
        else:
            compute = GREEDY_COMPUTE_PHASE_MULTIPLIER * (
                torch.tensor(
                    forward_compute_constant,
                    dtype=units.dtype,
                    device=units.device,
                )
                .view(layer_count, 1)
                .expand(-1, row_count)
                .reshape(-1)
            )
            peak_compute_rank = torch.full_like(peak_rank, -1)

        scored = []
        for layer_index, prepared in enumerate(prepared_layers):
            rows = slice(layer_index * row_count, (layer_index + 1) * row_count)
            scored.append(
                _ScoredLayouts(
                    communication=communication[rows],
                    compute=compute[rows],
                    communication_model_units=units[rows],
                    peak_rank=peak_rank[rows],
                    peak_compute_rank=peak_compute_rank[rows],
                    selected_dim=selected_dim[rows],
                    baseline_physical_routes=prepared.baseline_physical_routes,
                    route_hashes=prepared.route_hashes,
                    route_tables=None,
                )
            )
        return scored

    def _score_primitive_unary_layers(
        self,
        prepared_layers: Sequence[_PreparedPrimitiveCounts],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[_ScoredLayouts], list[_GlobalPrimitiveCounts]]:
        """Reduce primitive deltas once, then compose every action locally."""

        if not prepared_layers:
            return [], []
        include_assignments = any(value > 0.0 for value in forward_compute_per_assignment)
        communication_width = prepared_layers[0].baseline_local.shape[1]
        local_blocks = []
        primitive_counts = []
        for prepared in prepared_layers:
            primitive_count = int(prepared.primitive_delta_local.shape[0])
            primitive_counts.append(primitive_count)
            if self.exact_primitive_max_only:
                parts = [prepared.baseline_local.reshape(-1)]
                level_offset = 0
                for affected_groups, width in zip(
                    prepared.primitive_affected_groups,
                    self._count_widths(),
                    strict=True,
                ):
                    valid = affected_groups < width
                    values = prepared.primitive_delta_local[:, level_offset : level_offset + width].gather(
                        1, affected_groups.clamp(max=width - 1)
                    )
                    parts.append(torch.where(valid, values, torch.zeros_like(values)).reshape(-1))
                    level_offset += width
                if include_assignments:
                    if prepared.baseline_assignment_local is None or prepared.primitive_assignment_delta_local is None:
                        raise ValueError("Primitive joint-cost scoring requires assignment deltas.")
                    affected_ranks = prepared.primitive_affected_ranks
                    valid_ranks = affected_ranks < self.ep_size
                    assignment_values = prepared.primitive_assignment_delta_local.gather(
                        1,
                        affected_ranks.clamp(max=self.ep_size - 1),
                    )
                    parts.extend(
                        (
                            prepared.baseline_assignment_local.reshape(-1),
                            torch.where(
                                valid_ranks,
                                assignment_values,
                                torch.zeros_like(assignment_values),
                            ).reshape(-1),
                        )
                    )
                block = torch.cat(parts)
            else:
                block = torch.cat((prepared.baseline_local, prepared.primitive_delta_local), dim=0)
                if include_assignments:
                    if prepared.baseline_assignment_local is None or prepared.primitive_assignment_delta_local is None:
                        raise ValueError("Primitive joint-cost scoring requires assignment deltas.")
                    assignments = torch.cat(
                        (
                            prepared.baseline_assignment_local,
                            prepared.primitive_assignment_delta_local,
                        ),
                        dim=0,
                    )
                    block = torch.cat((block, assignments), dim=1)
            local_blocks.append(block)
        local_payload = torch.cat(local_blocks, dim=0)
        if prepare_stage_callback is not None:
            prepare_stage_callback("collective_pack")
        global_payload = _reduce_sum(local_payload, self.reducer)

        global_layers = []
        offset = 0
        for prepared, primitive_count in zip(prepared_layers, primitive_counts, strict=True):
            if self.exact_primitive_max_only:
                baseline = global_payload[offset : offset + communication_width].view(
                    1,
                    communication_width,
                )
                offset += communication_width
                primitive_delta = baseline.new_zeros((primitive_count, communication_width))
                level_offset = 0
                for affected_groups, width in zip(
                    prepared.primitive_affected_groups,
                    self._count_widths(),
                    strict=True,
                ):
                    value_count = primitive_count * int(affected_groups.shape[1])
                    values = global_payload[offset : offset + value_count].view(
                        primitive_count,
                        -1,
                    )
                    offset += value_count
                    dense_level = values.new_zeros((primitive_count, width + 1))
                    dense_level.scatter_(1, affected_groups, values)
                    primitive_delta[:, level_offset : level_offset + width] = dense_level[:, :width]
                    level_offset += width
                baseline_assignment = None
                primitive_assignment_delta = None
                if include_assignments:
                    baseline_assignment = global_payload[offset : offset + self.ep_size].view(
                        1,
                        self.ep_size,
                    )
                    offset += self.ep_size
                    affected_ranks = prepared.primitive_affected_ranks
                    assignment_value_count = primitive_count * int(affected_ranks.shape[1])
                    assignment_values = global_payload[offset : offset + assignment_value_count].view(
                        primitive_count, -1
                    )
                    offset += assignment_value_count
                    dense_assignment = assignment_values.new_zeros((primitive_count, self.ep_size + 1))
                    dense_assignment.scatter_(1, affected_ranks, assignment_values)
                    primitive_assignment_delta = dense_assignment[:, : self.ep_size]
            else:
                block = global_payload[offset : offset + primitive_count + 1]
                offset += primitive_count + 1
                baseline = block[:1, :communication_width]
                primitive_delta = block[1:, :communication_width]
                assignment_end = communication_width + (self.ep_size if include_assignments else 0)
                baseline_assignment = block[:1, communication_width:assignment_end] if include_assignments else None
                primitive_assignment_delta = (
                    block[1:, communication_width:assignment_end] if include_assignments else None
                )
            global_layers.append(
                _GlobalPrimitiveCounts(
                    baseline=baseline,
                    primitive_delta=primitive_delta,
                    baseline_assignment=baseline_assignment,
                    primitive_assignment_delta=primitive_assignment_delta,
                )
            )
        scored_layers = self._score_batched_primitive_actions(
            prepared_layers,
            global_layers,
            communication_scales=communication_scales,
            forward_compute_per_assignment=forward_compute_per_assignment,
            forward_compute_constant=forward_compute_constant,
            batch_layers=self.exact_primitive_max_only,
        )
        return scored_layers, global_layers

    def _exact_primitive_topk_layers(
        self,
        prepared_layers: Sequence[_PreparedPrimitiveCounts],
        contexts: Sequence[dict[str, object]],
        unary_scored: Sequence[_ScoredLayouts],
        global_layers: Sequence[_GlobalPrimitiveCounts],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
        materialize_route_tables: bool,
    ) -> tuple[list[_ScoredLayouts], list[torch.Tensor], dict[str, object]]:
        """Select exact-unary Top-K and add only their exact pair corrections."""

        started = time.perf_counter()
        shortlist_indices = [
            torch.topk(
                scored.total[1:],
                k=min(self.exact_primitive_topk, int(scored.total.numel()) - 1),
                largest=False,
                sorted=True,
            ).indices
            for scored in unary_scored
        ]
        selected_pair_contexts = []
        selected_rows = []
        route_tables = []
        pair_action_indices = []
        for prepared, context, indices in zip(
            prepared_layers,
            contexts,
            shortlist_indices,
            strict=True,
        ):
            pair_context, rows, tables = statistical_primitive_selected_pair_context(
                self,
                prepared.context,
                context["rows"],
                indices,
                layout=context["layout"],
                copy_slots=context["copy_slots"],
                uniform_source_rank=context["uniform_source_rank"],
                materialize_route_tables=materialize_route_tables,
            )
            selected_pair_contexts.append(pair_context)
            selected_rows.append(rows)
            route_tables.append(tables)
            pair_action_indices.append(torch.arange(indices.numel(), dtype=torch.long, device=indices.device))
            context["scored_rows"] = rows
        compact_pair_statistics = all(context.pair_absence is None for context in selected_pair_contexts)
        pair_local = statistical_batched_selected_pair_local_deltas(
            selected_pair_contexts,
            selected_rows,
            pair_action_indices,
        )
        pair_execution = "batched_compact" if compact_pair_statistics else "batched_dense"
        pair_counts = [int(value.shape[0]) for value in pair_local]
        pair_global_payload = _reduce_sum(torch.cat(pair_local, dim=0), self.reducer)

        exact_scored = []
        offset = 0
        include_assignments = any(value > 0.0 for value in forward_compute_per_assignment)
        for layer_index, (
            prepared,
            indices,
            tables,
            global_counts,
            selected_count,
        ) in enumerate(
            zip(
                prepared_layers,
                shortlist_indices,
                route_tables,
                global_layers,
                pair_counts,
                strict=True,
            )
        ):
            pair_delta = pair_global_payload[offset : offset + selected_count]
            offset += selected_count
            spec = prepared.context.spec
            lhs_ids = spec.lhs_ids.index_select(0, indices)
            rhs_ids = spec.rhs_ids.index_select(0, indices)
            rhs_valid = rhs_ids >= 0
            rhs_safe = rhs_ids.clamp_min(0)
            unary_delta = global_counts.primitive_delta.index_select(0, lhs_ids)
            rhs_delta = global_counts.primitive_delta.index_select(0, rhs_safe)
            unary_delta += rhs_delta * rhs_valid.view(-1, 1).to(rhs_delta.dtype)
            candidate_communication = global_counts.baseline + unary_delta + pair_delta
            communication_rows = torch.cat((global_counts.baseline, candidate_communication), dim=0)
            assignment_rows = None
            if include_assignments:
                assert (
                    global_counts.baseline_assignment is not None
                    and global_counts.primitive_assignment_delta is not None
                )
                assignment_delta = global_counts.primitive_assignment_delta.index_select(0, lhs_ids)
                rhs_assignment = global_counts.primitive_assignment_delta.index_select(0, rhs_safe)
                assignment_delta += rhs_assignment * rhs_valid.view(-1, 1).to(rhs_assignment.dtype)
                candidate_assignment = global_counts.baseline_assignment + assignment_delta
                assignment_rows = torch.cat(
                    (global_counts.baseline_assignment, candidate_assignment),
                    dim=0,
                )
            exact_scored.append(
                self._score_global_count_rows(
                    communication_rows,
                    assignment_rows,
                    communication_scale=communication_scales[layer_index],
                    forward_compute_per_assignment=forward_compute_per_assignment[layer_index],
                    forward_compute_constant=forward_compute_constant[layer_index],
                    baseline_physical_routes=prepared.baseline_physical_routes,
                    route_hashes=prepared.route_hashes,
                    route_tables=tables,
                )
            )
        stats = {
            "enabled": True,
            "topk": self.exact_primitive_topk,
            "max_only_unary": self.exact_primitive_max_only,
            "unary_selector": ("batched_full_exact_compact" if self.exact_primitive_max_only else "full_exact_unary"),
            "primitive_counts": [int(value.primitive_delta_local.shape[0]) for value in prepared_layers],
            "candidate_counts": [int(context["rows"].shape[0]) for context in contexts],
            "shortlist_counts": [int(value.numel()) for value in shortlist_indices],
            "pair_rerank_host_ms": (time.perf_counter() - started) * 1000.0,
            "pair_statistics_mode": ("post_shortlist_compact" if compact_pair_statistics else "precomputed_dense"),
            "pair_execution": pair_execution,
            "collective_rounds": 2,
        }
        dense_collective_elements = sum(
            (int(value.primitive_delta_local.shape[0]) + 1)
            * (int(value.baseline_local.shape[1]) + (self.ep_size if include_assignments else 0))
            for value in prepared_layers
        )
        compact_collective_elements = sum(
            int(value.baseline_local.shape[1])
            + int(value.primitive_delta_local.shape[0])
            * sum(int(groups.shape[1]) for groups in value.primitive_affected_groups)
            + (
                self.ep_size + int(value.primitive_delta_local.shape[0]) * int(value.primitive_affected_ranks.shape[1])
                if include_assignments
                else 0
            )
            for value in prepared_layers
        )
        stats["unary_collective_elements"] = (
            compact_collective_elements if self.exact_primitive_max_only else dense_collective_elements
        )
        stats["unary_collective_dense_elements"] = dense_collective_elements
        stats["unary_collective_compression"] = float(dense_collective_elements) / float(
            compact_collective_elements if self.exact_primitive_max_only else dense_collective_elements
        )
        return exact_scored, shortlist_indices, stats

    def _selected_exact_prepared(
        self,
        prepared: _PreparedActionCounts,
        rows: torch.Tensor,
        action_indices: torch.Tensor,
        pair_delta: torch.Tensor | None = None,
    ) -> _PreparedActionCounts:
        """Materialize exact counts only for selected unary-ranked actions."""

        if prepared.pair_context is None:
            raise ValueError("Sparse exact reranking requires a statistical pair context.")
        if pair_delta is None:
            pair_delta = statistical_selected_pair_local_deltas(
                prepared.pair_context,
                rows,
                action_indices,
            )
        candidate_local = prepared.candidate_local.index_select(0, action_indices) + pair_delta
        candidate_assignment_local = (
            None
            if prepared.candidate_assignment_local is None
            else prepared.candidate_assignment_local.index_select(0, action_indices)
        )
        affected_groups = (
            None if prepared.affected_groups is None else prepared.affected_groups.index_select(0, action_indices)
        )
        affected_assignment_ranks = (
            None
            if prepared.affected_assignment_ranks is None
            else prepared.affected_assignment_ranks.index_select(0, action_indices)
        )
        return _PreparedActionCounts(
            baseline_local=prepared.baseline_local,
            candidate_local=candidate_local,
            baseline_assignment_local=prepared.baseline_assignment_local,
            candidate_assignment_local=candidate_assignment_local,
            affected_groups=affected_groups,
            affected_assignment_ranks=affected_assignment_ranks,
            baseline_physical_routes=prepared.baseline_physical_routes,
            route_hashes=prepared.route_hashes,
            route_tables=prepared.route_tables,
        )

    def _adaptive_fast_path_available_on_all_ranks(
        self,
        prepared_layers: Sequence[_PreparedActionCounts],
    ) -> bool:
        """Agree on unary/pair fast-path availability before any scoring reduction."""

        local_available = all(
            prepared.pair_context is not None
            and (not self.adaptive_topk_strict_certificate or prepared.candidate_pair_bound_local is not None)
            for prepared in prepared_layers
        )
        if self.reducer is None or not dist.is_initialized() or not prepared_layers:
            return local_available
        available = prepared_layers[0].baseline_local.new_tensor([float(local_available)])
        available_count = _reduce_sum(available, self.reducer)
        return int(available_count.item()) == self.ep_size

    def _fast_path_available_on_all_ranks(
        self,
        local_available: bool,
        *,
        device: torch.device,
    ) -> bool:
        """Require every EP rank to select the same scoring path."""

        if self.reducer is None or not dist.is_initialized():
            return bool(local_available)
        available = torch.tensor([float(local_available)], dtype=torch.float32, device=device)
        available_count = _reduce_sum(available, self.reducer)
        return int(available_count.item()) == self.ep_size

    def _adaptive_score_layers(
        self,
        prepared_layers: Sequence[_PreparedActionCounts],
        contexts: Sequence[dict[str, object]],
        *,
        communication_scales: Sequence[float],
        forward_compute_per_assignment: Sequence[float],
        forward_compute_constant: Sequence[float],
    ) -> tuple[list[_ScoredLayouts], list[torch.Tensor], dict[str, object]]:
        """Rerank unary Top-K exactly, with optional strict-certificate expansion."""

        if any(prepared.pair_context is None for prepared in prepared_layers) or (
            self.adaptive_topk_strict_certificate
            and any(prepared.candidate_pair_bound_local is None for prepared in prepared_layers)
        ):
            raise RuntimeError("Adaptive Top-K scoring requires group-consistent unary/pair statistics.")
        unary_scored = self._score_prepared_layers(
            prepared_layers,
            communication_scales=communication_scales,
            forward_compute_per_assignment=forward_compute_per_assignment,
            forward_compute_constant=forward_compute_constant,
        )

        action_orders = [torch.argsort(scored.total[1:], stable=True) for scored in unary_scored]
        layer_count = len(prepared_layers)
        selected_indices: list[torch.Tensor | None] = [None] * layer_count
        final_scored: list[_ScoredLayouts | None] = [None] * layer_count
        final_k = [0] * layer_count
        rounds = [0] * layer_count
        certified = [False] * layer_count
        pending = list(range(layer_count))
        current_k = [min(self.adaptive_topk_initial, int(context["rows"].shape[0])) for context in contexts]
        rerank_started = time.perf_counter()
        collective_rounds = 0
        while pending:
            round_prepared: list[_PreparedActionCounts] = []
            round_indices: list[torch.Tensor] = []
            for layer_index in pending:
                action_indices = action_orders[layer_index][: current_k[layer_index]]
                round_indices.append(action_indices)
            pair_deltas = statistical_batched_selected_pair_local_deltas(
                [prepared_layers[layer_index].pair_context for layer_index in pending],
                [contexts[layer_index]["rows"] for layer_index in pending],
                round_indices,
            )
            for layer_index, action_indices, pair_delta in zip(
                pending,
                round_indices,
                pair_deltas,
                strict=True,
            ):
                assert prepared_layers[layer_index].pair_context is not None
                round_prepared.append(
                    self._selected_exact_prepared(
                        prepared_layers[layer_index],
                        contexts[layer_index]["rows"],
                        action_indices,
                        pair_delta,
                    )
                )
            round_scored = self._score_prepared_layers(
                round_prepared,
                communication_scales=[communication_scales[index] for index in pending],
                forward_compute_per_assignment=[forward_compute_per_assignment[index] for index in pending],
                forward_compute_constant=[forward_compute_constant[index] for index in pending],
            )
            collective_rounds += 1
            certificate_rows = []
            for layer_index, action_indices, scored in zip(
                pending,
                round_indices,
                round_scored,
                strict=True,
            ):
                candidate_costs = scored.total[1:]
                minimum = candidate_costs.min()
                original_or_sentinel = torch.where(
                    candidate_costs == minimum,
                    action_indices,
                    torch.full_like(action_indices, int(contexts[layer_index]["rows"].shape[0])),
                )
                winner_action = original_or_sentinel.min()
                winner_position = (action_indices == winner_action).to(torch.long).argmax()
                keep = torch.cat(
                    (
                        torch.zeros((1,), dtype=torch.long, device=winner_position.device),
                        winner_position.view(1) + 1,
                    )
                )
                final_scored[layer_index] = _ScoredLayouts(
                    communication=scored.communication.index_select(0, keep),
                    compute=scored.compute.index_select(0, keep),
                    communication_model_units=scored.communication_model_units.index_select(0, keep),
                    peak_rank=scored.peak_rank.index_select(0, keep),
                    peak_compute_rank=scored.peak_compute_rank.index_select(0, keep),
                    selected_dim=scored.selected_dim.index_select(0, keep),
                    baseline_physical_routes=prepared_layers[layer_index].baseline_physical_routes,
                    route_hashes=prepared_layers[layer_index].route_hashes,
                    route_tables=prepared_layers[layer_index].route_tables,
                )
                selected_indices[layer_index] = winner_action.view(1)
                final_k[layer_index] = int(action_indices.numel())
                rounds[layer_index] += 1

                candidate_count = int(contexts[layer_index]["rows"].shape[0])
                if not self.adaptive_topk_strict_certificate:
                    certificate_rows.append(torch.ones((), dtype=torch.bool, device=minimum.device))
                    continue
                if int(action_indices.numel()) >= candidate_count:
                    certificate_rows.append(torch.ones((), dtype=torch.bool, device=minimum.device))
                    continue
                assert unary_scored[layer_index].exact_cost_lower_bound is not None
                rest_lower = unary_scored[layer_index].exact_cost_lower_bound[1:].clone()
                rest_lower.index_fill_(0, action_indices, torch.inf)
                certificate_rows.append(minimum + self.adaptive_topk_epsilon < rest_lower.min())
            certificate_values = torch.stack(certificate_rows).detach().to(device="cpu").tolist()
            next_pending = []
            for layer_index, passed in zip(pending, certificate_values, strict=True):
                if bool(passed):
                    certified[layer_index] = self.adaptive_topk_strict_certificate
                    continue
                candidate_count = int(contexts[layer_index]["rows"].shape[0])
                current_k[layer_index] = min(
                    candidate_count,
                    current_k[layer_index] * self.adaptive_topk_growth_factor,
                )
                next_pending.append(layer_index)
            pending = next_pending

        if any(value is None for value in final_scored) or any(value is None for value in selected_indices):
            raise RuntimeError("Adaptive Top-K scoring did not produce every layer decision.")
        stats = {
            "enabled": True,
            "certificate_mode": ("strict" if self.adaptive_topk_strict_certificate else "empirical_fixed_k"),
            "initial_k": self.adaptive_topk_initial,
            "final_k": final_k,
            "rounds": rounds,
            "certified": certified,
            "collective_rounds": collective_rounds,
            "rerank_ms": (time.perf_counter() - rerank_started) * 1000.0,
        }
        return (
            [value for value in final_scored if value is not None],
            [value for value in selected_indices if value is not None],
            stats,
        )

    def _prepare_independent_layers(
        self,
        contexts: Sequence[dict[str, object]],
        *,
        step: int,
        include_assignment_counts: bool,
        include_pair_interactions: bool = True,
        include_pair_bounds: bool = False,
        early_proxy: bool = False,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> list[_PreparedActionCounts]:
        """Build exact per-layer statistics concurrently on accelerator streams.

        The first layer runs on the caller stream to populate the shared
        fixed-shape route/hash caches. Remaining layers are independent and
        can safely execute on a bounded stream pool. The caller stream waits
        on that pool before concatenating the one batched collective payload.
        """

        def prepare(context: dict[str, object]) -> _PreparedActionCounts:
            if early_proxy:
                return self._prepare_proxy_action_counts(
                    context["selected"],
                    context["layout"],
                    context["rows"],
                    uniform_source_rank=context["uniform_source_rank"],
                    copy_slots=context["copy_slots"],
                    token_ordinals=context["ordinals"],
                    step=step,
                    layer_seed=context["layer_seed"],
                    num_experts=int(context["owners"].numel()),
                    include_assignment_counts=include_assignment_counts,
                    prepare_stage_callback=prepare_stage_callback,
                )
            return self._prepare_action_counts(
                context["selected"],
                context["layout"],
                context["rows"],
                source_ranks=context["sources"],
                uniform_source_rank=context["uniform_source_rank"],
                copy_slots=context["copy_slots"],
                affected_groups=None,
                token_ordinals=context["ordinals"],
                step=step,
                layer_seed=context["layer_seed"],
                num_experts=int(context["owners"].numel()),
                include_assignment_counts=include_assignment_counts,
                include_pair_interactions=include_pair_interactions,
                include_pair_bounds=include_pair_bounds,
                prepare_stage_callback=prepare_stage_callback,
            )

        if not contexts:
            return []
        device = contexts[0]["selected"].device
        if any(context["selected"].device != device for context in contexts):
            raise ValueError("Batched layers must reside on the same device.")
        prepared: list[_PreparedActionCounts | None] = [None] * len(contexts)
        prepared[0] = prepare(contexts[0])
        common_shape = tuple(contexts[0]["selected"].shape)
        shapes_match = all(tuple(context["selected"].shape) == common_shape for context in contexts)
        # Shared fixed-shape route caches are populated by the first layer.
        # If token/top-k shapes differ, use the caller stream so a later layer
        # cannot observe a cache tensor still being produced by another stream.
        stream_count = min(self.layer_parallel_streams, len(contexts) - 1) if shapes_match else 0
        if device.type == "cpu" or stream_count <= 1:
            for index in range(1, len(contexts)):
                prepared[index] = prepare(contexts[index])
        else:
            device_api = get_torch_device()
            try:
                caller_stream = device_api.current_stream(device)
            except TypeError:
                caller_stream = device_api.current_stream()
            cache_key = (device.type, device.index, stream_count)
            compute_streams = _LAYER_COMPUTE_STREAMS.get(cache_key)
            if compute_streams is None:
                values = []
                for _ in range(stream_count):
                    try:
                        values.append(device_api.Stream(device=device))
                    except TypeError:
                        values.append(device_api.Stream())
                compute_streams = tuple(values)
                _LAYER_COMPUTE_STREAMS[cache_key] = compute_streams
            for compute_stream in compute_streams:
                compute_stream.wait_stream(caller_stream)
            for index in range(1, len(contexts)):
                compute_stream = compute_streams[(index - 1) % stream_count]
                with device_api.stream(compute_stream):
                    prepared[index] = prepare(contexts[index])
            for compute_stream in compute_streams:
                caller_stream.wait_stream(compute_stream)
        if any(value is None for value in prepared):
            raise RuntimeError("Batched layer statistic preparation did not produce every layer.")
        return [value for value in prepared if value is not None]

    def _prepare_primitive_layers(
        self,
        contexts: Sequence[dict[str, object]],
        *,
        step: int,
        include_assignment_counts: bool,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> list[_PreparedPrimitiveCounts]:
        """Build exact primitive statistics on the existing bounded stream pool."""

        if not contexts:
            return []

        def prepare(context: dict[str, object]) -> _PreparedPrimitiveCounts:
            return self._prepare_primitive_counts(
                context,
                step=step,
                include_assignment_counts=include_assignment_counts,
                prepare_stage_callback=prepare_stage_callback,
            )

        device = contexts[0]["selected"].device
        prepared: list[_PreparedPrimitiveCounts | None] = [None] * len(contexts)
        prepared[0] = prepare(contexts[0])
        common_shape = tuple(contexts[0]["selected"].shape)
        shapes_match = all(tuple(context["selected"].shape) == common_shape for context in contexts)
        stream_count = min(self.layer_parallel_streams, len(contexts) - 1) if shapes_match else 0
        if device.type == "cpu" or stream_count <= 1:
            for index in range(1, len(contexts)):
                prepared[index] = prepare(contexts[index])
        else:
            device_api = get_torch_device()
            try:
                caller_stream = device_api.current_stream(device)
            except TypeError:
                caller_stream = device_api.current_stream()
            cache_key = (device.type, device.index, stream_count)
            compute_streams = _LAYER_COMPUTE_STREAMS.get(cache_key)
            if compute_streams is None:
                values = []
                for _ in range(stream_count):
                    try:
                        values.append(device_api.Stream(device=device))
                    except TypeError:
                        values.append(device_api.Stream())
                compute_streams = tuple(values)
                _LAYER_COMPUTE_STREAMS[cache_key] = compute_streams
            for compute_stream in compute_streams:
                compute_stream.wait_stream(caller_stream)
            for index in range(1, len(contexts)):
                compute_stream = compute_streams[(index - 1) % stream_count]
                with device_api.stream(compute_stream):
                    prepared[index] = prepare(contexts[index])
            for compute_stream in compute_streams:
                caller_stream.wait_stream(compute_stream)
        if any(value is None for value in prepared):
            raise RuntimeError("Batched primitive preparation did not produce every layer.")
        return [value for value in prepared if value is not None]

    def score_layout(
        self,
        selected_experts: torch.Tensor,
        slot_to_logical: torch.Tensor,
        *,
        source_ranks: int | torch.Tensor,
        owner_slots: torch.Tensor,
        token_ordinals: torch.Tensor | None = None,
        step: int = 0,
        layer_seed: int = 0,
        max_copies: int | None = None,
    ) -> PlacementCost:
        """Evaluate one layout with the same exact route mapping used by planning."""

        if max_copies is not None and int(max_copies) > self.max_copies:
            raise ValueError("score_layout max_copies exceeds the planner copy limit.")
        return self.plan(
            selected_experts,
            slot_to_logical,
            owner_slots,
            source_ranks=source_ranks,
            max_swaps=0,
            max_replicas=0,
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        ).baseline_cost

    def plan_layers(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_ranks: int | Sequence[int | torch.Tensor],
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
        step: int = 0,
        communication_scales: Sequence[float] | None = None,
        forward_compute_per_assignment: Sequence[float] | None = None,
        forward_compute_constant: Sequence[float] | None = None,
        token_ordinals: Sequence[torch.Tensor] | None = None,
        skip_final_route_update: bool = True,
        prepare_stage_callback: Callable[[str], None] | None = None,
    ) -> list[PlacementPlan]:
        """Plan one steady-state action per independent layer with one EP collective."""

        started = time.perf_counter()
        layer_count = len(selected_experts)
        if not (len(slot_to_logical) == len(owner_slots) == len(layer_seeds) == layer_count):
            raise ValueError("Batched planner inputs must have identical layer counts.")
        if layer_count == 0:
            return []
        scales = (
            [self.communication_scale] * layer_count
            if communication_scales is None
            else [float(value) for value in communication_scales]
        )
        compute_slopes = (
            [self.forward_compute_per_assignment] * layer_count
            if forward_compute_per_assignment is None
            else [float(value) for value in forward_compute_per_assignment]
        )
        compute_constants = (
            [self.forward_compute_constant] * layer_count
            if forward_compute_constant is None
            else [float(value) for value in forward_compute_constant]
        )
        if not (len(scales) == len(compute_slopes) == len(compute_constants) == layer_count):
            raise ValueError("Batched planner cost-model arrays must match the number of layers.")
        if isinstance(source_ranks, int):
            source_values: list[int | torch.Tensor] = [int(source_ranks)] * layer_count
        else:
            source_values = list(source_ranks)
            if len(source_values) != layer_count:
                raise ValueError("Batched source_ranks must match the number of layers.")
        ordinal_values: list[torch.Tensor | None] = (
            [None] * layer_count if token_ordinals is None else list(token_ordinals)
        )
        if len(ordinal_values) != layer_count:
            raise ValueError("Batched token_ordinals must match the number of layers.")

        candidate_started = time.perf_counter()
        contexts: list[dict[str, object]] = []
        # Step-mode callers commonly pass the same steady-state layout object
        # to every layer (for example the initial R2 layout).  Candidate rows,
        # copy tables, and primitive ids depend only on that metadata, not on
        # the layer's token routes.  Reuse the immutable tensors instead of
        # repeating CPU copies and an exact torch.unique for all 48 layers.
        layout_metadata_cache: dict[tuple[int, int], dict[str, object]] = {}
        include_assignments = any(value > 0.0 for value in compute_slopes)
        for layer_index, (
            raw_selected,
            raw_layout,
            raw_owners,
            raw_source,
            raw_ordinals,
            layer_seed,
        ) in enumerate(
            zip(
                selected_experts,
                slot_to_logical,
                owner_slots,
                source_values,
                ordinal_values,
                layer_seeds,
                strict=True,
            )
        ):
            selected = raw_selected.to(torch.long)
            original_selected_ndim = selected.ndim
            if selected.ndim == 1:
                selected = selected.unsqueeze(-1)
            if selected.ndim != 2:
                raise ValueError(
                    f"selected_experts[{layer_index}] must have rank 1 or 2, got shape={tuple(selected.shape)}."
                )
            device = selected.device
            if isinstance(raw_source, int):
                uniform_source_rank = int(raw_source)
                sources = torch.full(
                    (selected.shape[0],),
                    uniform_source_rank,
                    dtype=torch.long,
                    device=device,
                )
            else:
                uniform_source_rank = None
                sources = raw_source.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
            if sources.numel() != selected.shape[0]:
                raise ValueError("A batched source_ranks tensor does not match its local token count.")
            ordinals = (
                torch.arange(selected.shape[0], dtype=torch.long, device=device)
                if raw_ordinals is None
                else raw_ordinals.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
            )
            if ordinals.numel() != selected.shape[0]:
                raise ValueError("A batched token_ordinals tensor does not match its local token count.")
            metadata_key = (id(raw_layout), id(raw_owners))
            metadata = layout_metadata_cache.get(metadata_key)
            if metadata is None:
                host_layout = raw_layout.detach().to(device="cpu", dtype=torch.long).clone()
                host_owners = raw_owners.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
                if host_layout.numel() != self.ep_size * self.slots_per_rank:
                    raise ValueError("A batched slot_to_logical does not match ep_size * slots_per_rank.")
                if bool((host_layout < 0).any().item()):
                    raise ValueError("plan_layers currently requires initialized steady-state layouts.")
                all_slots = torch.arange(host_layout.numel(), dtype=torch.long)
                owner_mask = torch.zeros((host_layout.numel(),), dtype=torch.bool)
                owner_mask.scatter_(0, host_owners, True)
                rows_by_kind = []
                if max(0, int(max_swaps)) > 0:
                    rows_by_kind.append(self._swap_rows(host_layout, host_owners))
                if max(0, int(max_replicas)) > 0:
                    cover_slots = all_slots[(~owner_mask) & (host_layout >= 0)]
                    rows_by_kind.append(self._cover_rows(host_layout, host_owners, cover_slots))
                nonempty_rows = [value for value in rows_by_kind if value.numel()]
                host_rows = torch.cat(nonempty_rows, dim=0) if nonempty_rows else torch.empty((0, 5), dtype=torch.long)
                host_copy_slots = self._copy_table(host_layout, int(host_owners.numel()))
                primitive_spec = (
                    self._primitive_spec(
                        host_layout,
                        host_copy_slots,
                        host_rows,
                        device=device,
                    )
                    if self.exact_primitive_topk and host_rows.numel()
                    else None
                )
                device_copy_slots = host_copy_slots.to(device=device, non_blocking=True)
                primitive_affected_ranks = None
                primitive_affected_groups = None
                if primitive_spec is not None and self.exact_primitive_max_only:
                    primitive_affected_ranks, primitive_affected_groups = self._primitive_affected_metadata(
                        device_copy_slots,
                        primitive_spec,
                    )
                metadata = {
                    "host_layout": host_layout,
                    "host_owners": host_owners,
                    "layout": host_layout.to(device=device, non_blocking=True),
                    "owners": host_owners.to(device=device, non_blocking=True),
                    "host_rows": host_rows,
                    "rows": host_rows.to(device=device, non_blocking=True),
                    "copy_slots": device_copy_slots,
                    "primitive_spec": primitive_spec,
                    "primitive_affected_ranks": primitive_affected_ranks,
                    "primitive_affected_groups": primitive_affected_groups,
                }
                layout_metadata_cache[metadata_key] = metadata
            contexts.append(
                {
                    "selected": selected,
                    "original_selected_ndim": original_selected_ndim,
                    **metadata,
                    "sources": sources,
                    "uniform_source_rank": uniform_source_rank,
                    "ordinals": ordinals,
                    "layer_seed": int(layer_seed),
                }
            )
        route_stats_ms = (time.perf_counter() - candidate_started) * 1000.0
        if prepare_stage_callback is not None:
            prepare_stage_callback("context")

        score_started = time.perf_counter()
        early_available = (
            self.early_proxy_topk > 0
            and self.candidate_scorer == "statistics"
            and all(context["uniform_source_rank"] is not None and context["rows"].numel() for context in contexts)
        )
        primitive_local_available = (
            self.exact_primitive_topk > 0
            and self.candidate_scorer == "statistics"
            and all(
                context["uniform_source_rank"] is not None
                and context["rows"].numel()
                and context["primitive_spec"] is not None
                and statistical_primitive_fast_path_available(
                    self,
                    context["selected"],
                    copy_slots=context["copy_slots"],
                    num_experts=int(context["owners"].numel()),
                    defer_pair_statistics=self.post_shortlist_compact_pair,
                    batched_layer_count=len(contexts),
                )
                for context in contexts
            )
        )
        primitive_available = (
            self._fast_path_available_on_all_ranks(
                primitive_local_available,
                device=contexts[0]["selected"].device,
            )
            if self.exact_primitive_topk > 0
            else False
        )
        route_table_index_maps: list[torch.Tensor]
        if primitive_available:
            primitive_started = time.perf_counter()
            prepared_layers = self._prepare_primitive_layers(
                contexts,
                step=step,
                include_assignment_counts=include_assignments,
                prepare_stage_callback=prepare_stage_callback,
            )
            unary_scored, global_primitive = self._score_primitive_unary_layers(
                prepared_layers,
                communication_scales=scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                prepare_stage_callback=prepare_stage_callback,
            )
            scored_layers, candidate_index_maps, primitive_stats = self._exact_primitive_topk_layers(
                prepared_layers,
                contexts,
                unary_scored,
                global_primitive,
                communication_scales=scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                materialize_route_tables=not skip_final_route_update,
            )
            route_table_index_maps = [
                torch.arange(indices.numel(), dtype=torch.long, device=indices.device)
                for indices in candidate_index_maps
            ]
            primitive_stats["total_host_ms"] = (time.perf_counter() - primitive_started) * 1000.0
            self.last_exact_primitive_stats = primitive_stats
            self.last_exact_primitive_shortlist_indices = candidate_index_maps
            self.last_early_proxy_stats = {"enabled": False, "reason": "disabled"}
            self.last_early_proxy_shortlist_indices = []
            self.last_adaptive_topk_stats = {"enabled": False}
        elif early_available:
            proxy_started = time.perf_counter()
            proxy_prepared = self._prepare_independent_layers(
                contexts,
                step=step,
                include_assignment_counts=include_assignments,
                early_proxy=True,
                prepare_stage_callback=prepare_stage_callback,
            )
            proxy_scored = self._score_prepared_layers(
                proxy_prepared,
                communication_scales=scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                prepare_stage_callback=prepare_stage_callback,
            )
            shortlist_indices = [
                torch.topk(
                    scored.total[1:],
                    k=min(self.early_proxy_topk, int(scored.total.numel()) - 1),
                    largest=False,
                    sorted=True,
                ).indices
                for scored in proxy_scored
            ]
            proxy_ms = (time.perf_counter() - proxy_started) * 1000.0
            exact_contexts = []
            for context, indices in zip(contexts, shortlist_indices, strict=True):
                exact_context = dict(context)
                exact_rows = context["rows"].index_select(0, indices)
                exact_context["rows"] = exact_rows
                context["scored_rows"] = exact_rows
                exact_contexts.append(exact_context)
            exact_started = time.perf_counter()
            prepared_layers = self._prepare_independent_layers(
                exact_contexts,
                step=step,
                include_assignment_counts=include_assignments,
                include_pair_interactions=True,
                include_pair_bounds=False,
                prepare_stage_callback=prepare_stage_callback,
            )
            scored_layers = self._score_prepared_layers(
                prepared_layers,
                communication_scales=scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                prepare_stage_callback=prepare_stage_callback,
            )
            exact_ms = (time.perf_counter() - exact_started) * 1000.0
            candidate_index_maps = shortlist_indices
            self.last_early_proxy_shortlist_indices = shortlist_indices
            route_table_index_maps = [
                torch.arange(indices.numel(), dtype=torch.long, device=indices.device) for indices in shortlist_indices
            ]
            self.last_early_proxy_stats = {
                "enabled": True,
                "topk": self.early_proxy_topk,
                "candidate_counts": [int(context["rows"].shape[0]) for context in contexts],
                "shortlist_counts": [int(indices.numel()) for indices in shortlist_indices],
                "proxy_host_ms": proxy_ms,
                "exact_shortlist_host_ms": exact_ms,
            }
            self.last_adaptive_topk_stats = {"enabled": False}
            self.last_exact_primitive_stats = {"enabled": False, "reason": "disabled"}
            self.last_exact_primitive_shortlist_indices = []
        else:
            prepared_layers = self._prepare_independent_layers(
                contexts,
                step=step,
                include_assignment_counts=include_assignments,
                include_pair_interactions=not self.adaptive_topk,
                include_pair_bounds=self.adaptive_topk and self.adaptive_topk_strict_certificate,
                prepare_stage_callback=prepare_stage_callback,
            )
            adaptive_available = self.adaptive_topk and self._adaptive_fast_path_available_on_all_ranks(
                prepared_layers
            )
            if adaptive_available:
                scored_layers, candidate_index_maps, adaptive_stats = self._adaptive_score_layers(
                    prepared_layers,
                    contexts,
                    communication_scales=scales,
                    forward_compute_per_assignment=compute_slopes,
                    forward_compute_constant=compute_constants,
                )
                route_table_index_maps = candidate_index_maps
                self.last_adaptive_topk_stats = adaptive_stats
            else:
                if self.adaptive_topk:
                    prepared_layers = self._prepare_independent_layers(
                        contexts,
                        step=step,
                        include_assignment_counts=include_assignments,
                        include_pair_interactions=True,
                        include_pair_bounds=False,
                    )
                scored_layers = self._score_prepared_layers(
                    prepared_layers,
                    communication_scales=scales,
                    forward_compute_per_assignment=compute_slopes,
                    forward_compute_constant=compute_constants,
                    prepare_stage_callback=prepare_stage_callback,
                )
                candidate_index_maps = [
                    torch.arange(
                        context["rows"].shape[0],
                        dtype=torch.long,
                        device=context["rows"].device,
                    )
                    for context in contexts
                ]
                route_table_index_maps = candidate_index_maps
                self.last_adaptive_topk_stats = (
                    {
                        "enabled": False,
                        "reason": "statistical unary fast path unavailable on at least one EP rank",
                    }
                    if self.adaptive_topk
                    else {"enabled": False}
                )
            self.last_early_proxy_stats = {
                "enabled": False,
                "reason": (
                    "requires statistics scorer, non-empty candidates, and one uniform source rank per EP process"
                    if self.early_proxy_topk
                    else "disabled"
                ),
            }
            self.last_early_proxy_shortlist_indices = []
            self.last_exact_primitive_stats = {
                "enabled": False,
                "reason": (
                    "requires statistics scorer, non-empty candidates, and one uniform source rank per EP process"
                    if self.exact_primitive_topk
                    else "disabled"
                ),
            }
            self.last_exact_primitive_shortlist_indices = []
        score_ms = (time.perf_counter() - score_started) * 1000.0

        decision_started = time.perf_counter()
        candidate_counts = [int(scored.total.numel()) - 1 for scored in scored_layers]
        maximum_candidates = max(1, max(candidate_counts))
        device = scored_layers[0].communication.device
        padded_total = torch.full(
            (layer_count, maximum_candidates),
            torch.inf,
            dtype=scored_layers[0].communication.dtype,
            device=device,
        )
        for layer_index, (scored, candidate_count) in enumerate(zip(scored_layers, candidate_counts, strict=True)):
            if candidate_count:
                padded_total[layer_index, :candidate_count] = scored.total[1:]
        best_positions = padded_total.argmin(dim=1)
        best_indices = torch.stack(
            [
                candidate_index_maps[layer_index][best_positions[layer_index]]
                if candidate_counts[layer_index]
                else torch.full((), -1, dtype=torch.long, device=device)
                for layer_index in range(layer_count)
            ]
        )
        best_route_table_indices = torch.stack(
            [
                route_table_index_maps[layer_index][best_positions[layer_index]]
                if candidate_counts[layer_index]
                else torch.full((), -1, dtype=torch.long, device=device)
                for layer_index in range(layer_count)
            ]
        )
        has_candidate = torch.tensor(
            [count > 0 for count in candidate_counts],
            dtype=torch.bool,
            device=device,
        )
        baseline_metrics = []
        candidate_metrics = []
        for layer_index, scored in enumerate(scored_layers):
            metrics = torch.stack(
                (
                    scored.communication,
                    scored.compute,
                    scored.communication_model_units,
                    scored.peak_rank.to(scored.communication.dtype),
                    scored.peak_compute_rank.to(scored.communication.dtype),
                    scored.selected_dim.to(scored.communication.dtype),
                ),
                dim=1,
            )
            baseline_metrics.append(metrics[0])
            candidate_metrics.append(
                metrics[best_positions[layer_index] + 1] if candidate_counts[layer_index] else metrics[0]
            )
        decision = torch.cat(
            (
                torch.where(has_candidate, best_indices, torch.full_like(best_indices, -1))
                .to(scored_layers[0].communication.dtype)
                .view(-1, 1),
                torch.where(
                    has_candidate,
                    best_route_table_indices,
                    torch.full_like(best_route_table_indices, -1),
                )
                .to(scored_layers[0].communication.dtype)
                .view(-1, 1),
                torch.stack(baseline_metrics),
                torch.stack(candidate_metrics),
            ),
            dim=1,
        )
        decision_rows = decision.detach().to(device="cpu").tolist()
        decision_sync_ms = (time.perf_counter() - decision_started) * 1000.0

        finalization_started = time.perf_counter()
        plans: list[PlacementPlan] = []
        per_layer_planning_ms = (time.perf_counter() - started) * 1000.0 / layer_count
        for context, scored, decision_row in zip(
            contexts,
            scored_layers,
            decision_rows,
            strict=True,
        ):
            winner_index = int(decision_row[0])
            route_table_index = int(decision_row[1])
            baseline_cost = self._placement_cost_from_values(decision_row[2:8])
            candidate_cost = self._placement_cost_from_values(decision_row[8:14])
            final_cost = baseline_cost
            host_layout = context["host_layout"]
            host_owners = context["host_owners"]
            final_host_layout = host_layout.clone()
            final_host_owners = host_owners.clone()
            actions: tuple[PlacementAction, ...] = ()
            final_physical = None
            if winner_index >= 0 and candidate_cost.total < baseline_cost.total:
                host_row = context["host_rows"][winner_index]
                action = self._placement_action(host_row.tolist())
                actions = (action,)
                final_cost = candidate_cost
                if action.kind == "swap":
                    final_host_layout[action.src_slot] = action.dst_logical
                    final_host_layout[action.dst_slot] = action.src_logical
                    final_host_owners[action.src_logical], final_host_owners[action.dst_logical] = (
                        host_owners[action.dst_logical],
                        host_owners[action.src_logical],
                    )
                else:
                    final_host_layout[action.dst_slot] = action.src_logical
                if not skip_final_route_update:
                    scored_rows = context.get("scored_rows", context["rows"])
                    row = scored_rows[route_table_index]
                    if scored.route_tables is not None:
                        final_physical = self._apply_statistical_action_routes(
                            context["selected"],
                            row,
                            route_table_index,
                            scored.baseline_physical_routes,
                            scored.route_hashes,
                            scored.route_tables,
                        )
                    else:
                        final_physical = self._apply_action_routes(
                            context["selected"],
                            context["layout"],
                            row,
                            scored.baseline_physical_routes,
                            source_ranks=context["sources"],
                            token_ordinals=context["ordinals"],
                            route_hashes=scored.route_hashes,
                            step=step,
                            layer_seed=context["layer_seed"],
                            num_experts=int(context["owners"].numel()),
                        )
            if not skip_final_route_update and final_physical is None:
                final_physical = scored.baseline_physical_routes
            if final_physical is not None and int(context["original_selected_ndim"]) == 1:
                final_physical = final_physical.squeeze(-1)
            chose_swap = bool(actions) and actions[0].kind == "swap"
            chose_cover = bool(actions) and not chose_swap
            plans.append(
                PlacementPlan(
                    actions=actions,
                    initial_layout=tuple(int(value) for value in host_layout.tolist()),
                    final_layout=tuple(int(value) for value in final_host_layout.tolist()),
                    baseline_cost=baseline_cost,
                    final_cost=final_cost,
                    swap_rounds=int(chose_swap),
                    replica_rounds=int(chose_cover),
                    planning_ms=per_layer_planning_ms,
                    route_stats_ms=route_stats_ms / layer_count,
                    swap_ms=score_ms / layer_count if chose_swap else 0.0,
                    replica_ms=score_ms / layer_count if chose_cover else 0.0,
                    swap_score_ms=score_ms / layer_count,
                    swap_update_ms=0.0,
                    swap_collective_ms=0.0,
                    replica_score_ms=0.0,
                    replica_update_ms=0.0,
                    replica_collective_ms=0.0,
                    decision_sync_ms=decision_sync_ms / layer_count,
                    finalization_ms=0.0,
                    algorithm_version=GREEDY_COVER_ALGORITHM_VERSION,
                    local_physical_routes=final_physical,
                    final_owner_slots=tuple(int(value) for value in final_host_owners.tolist()),
                )
            )
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        if finalization_ms > 0.0:
            replacement = []
            for plan in plans:
                values = vars(plan).copy()
                values["finalization_ms"] = finalization_ms / layer_count
                replacement.append(PlacementPlan(**values))
            plans = replacement
        return plans

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
        if self.exact_primitive_topk and token_ordinals is None:
            host_layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long)
            if not bool((host_layout < 0).any().item()):
                batched_sources: int | Sequence[int | torch.Tensor] = (
                    int(source_ranks) if isinstance(source_ranks, int) else [source_ranks]
                )
                return self.plan_layers(
                    [selected_experts],
                    [slot_to_logical],
                    [owner_slots],
                    source_ranks=batched_sources,
                    max_swaps=max_swaps,
                    max_replicas=max_replicas,
                    layer_seeds=[layer_seed],
                    step=step,
                    skip_final_route_update=False,
                )[0]
        started = time.perf_counter()
        selected = selected_experts.to(torch.long)
        original_selected_ndim = selected.ndim
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        if selected.ndim != 2:
            raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")
        device = selected.device
        host_layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long).clone()
        host_owners = owner_slots.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
        if host_layout.numel() != self.ep_size * self.slots_per_rank:
            raise ValueError("slot_to_logical does not match hierarchy.ep_size * slots_per_rank.")
        if isinstance(source_ranks, int):
            uniform_source_rank = int(source_ranks)
            sources = torch.full((selected.shape[0],), uniform_source_rank, dtype=torch.long, device=device)
        else:
            uniform_source_rank = None
            sources = source_ranks.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        ordinals = (
            torch.arange(selected.shape[0], dtype=torch.long, device=device)
            if token_ordinals is None
            else token_ordinals.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        )
        if sources.numel() != selected.shape[0] or ordinals.numel() != selected.shape[0]:
            raise ValueError("source_ranks and token_ordinals must match the local token count.")

        candidate_started = time.perf_counter()
        # Layout metadata is tiny and originates on the host. Constructing
        # dynamic-shape candidate rows on the accelerator would make nonzero
        # and boolean indexing synchronize the model-compute stream on every
        # layer. Build the rows on CPU and transfer one compact matrix.
        all_slots = torch.arange(host_layout.numel(), dtype=torch.long)
        owner_mask = torch.zeros((host_layout.numel(),), dtype=torch.bool)
        owner_mask.scatter_(0, host_owners, True)
        empty_slots = torch.nonzero(host_layout < 0, as_tuple=False).flatten()
        initializing = empty_slots.numel() > 0 and max(0, int(max_replicas)) > 0
        fill_limit = min(int(empty_slots.numel()), max(0, int(max_replicas))) if initializing else 0
        if initializing:
            host_rows = self._cover_rows(host_layout, host_owners, empty_slots)
            if host_rows.numel() == 0 or not self._can_complete_empty_initialization(host_layout, fill_limit):
                raise ValueError(
                    "Empty expert slots cannot be initialized under the current max_copies and rank-local "
                    "duplicate constraints."
                )
        else:
            rows_by_kind = []
            if max(0, int(max_swaps)) > 0:
                rows_by_kind.append(self._swap_rows(host_layout, host_owners))
            if max(0, int(max_replicas)) > 0:
                cover_slots = all_slots[(~owner_mask) & (host_layout >= 0)]
                rows_by_kind.append(self._cover_rows(host_layout, host_owners, cover_slots))
            nonempty_rows = [value for value in rows_by_kind if value.numel()]
            host_rows = torch.cat(nonempty_rows, dim=0) if nonempty_rows else torch.empty((0, 5), dtype=torch.long)

        host_copy_slots = self._copy_table(host_layout, int(host_owners.numel()))
        copy_slots = host_copy_slots.to(device=device, non_blocking=True)
        layout = host_layout.to(device=device, non_blocking=True)
        owners = host_owners.to(device=device, non_blocking=True)
        rows = host_rows.to(device=device, non_blocking=True)
        affected_groups = (
            self._candidate_affected_groups(copy_slots, rows)
            if self.compact_candidate_collective and self.reducer is not None
            else None
        )
        route_stats_ms = (time.perf_counter() - candidate_started) * 1000.0

        score_started = time.perf_counter()
        candidate_index_map = torch.arange(rows.shape[0], dtype=torch.long, device=device)
        route_table_index_map = candidate_index_map
        scored_rows = rows
        early_available = (
            self.early_proxy_topk > 0
            and not initializing
            and rows.numel()
            and self.candidate_scorer == "statistics"
            and uniform_source_rank is not None
        )
        if early_available:
            proxy_started = time.perf_counter()
            proxy_prepared = self._prepare_proxy_action_counts(
                selected,
                layout,
                rows,
                uniform_source_rank=uniform_source_rank,
                copy_slots=copy_slots,
                token_ordinals=ordinals,
                step=step,
                layer_seed=layer_seed,
                num_experts=int(owners.numel()),
                include_assignment_counts=self.forward_compute_per_assignment > 0.0,
            )
            proxy_scored = self._score_prepared_layers(
                [proxy_prepared],
                communication_scales=[self.communication_scale],
                forward_compute_per_assignment=[self.forward_compute_per_assignment],
                forward_compute_constant=[self.forward_compute_constant],
            )[0]
            shortlist_count = min(self.early_proxy_topk, int(rows.shape[0]))
            candidate_index_map = torch.topk(
                proxy_scored.total[1:],
                k=shortlist_count,
                largest=False,
                sorted=True,
            ).indices
            self.last_early_proxy_shortlist_indices = [candidate_index_map]
            proxy_ms = (time.perf_counter() - proxy_started) * 1000.0
            scored_rows = rows.index_select(0, candidate_index_map)
            exact_started = time.perf_counter()
            prepared = self._prepare_action_counts(
                selected,
                layout,
                scored_rows,
                source_ranks=sources,
                uniform_source_rank=uniform_source_rank,
                copy_slots=copy_slots,
                affected_groups=None,
                token_ordinals=ordinals,
                step=step,
                layer_seed=layer_seed,
                num_experts=int(owners.numel()),
                include_pair_interactions=True,
                include_pair_bounds=False,
            )
            scored = self._score_prepared_layers(
                [prepared],
                communication_scales=[self.communication_scale],
                forward_compute_per_assignment=[self.forward_compute_per_assignment],
                forward_compute_constant=[self.forward_compute_constant],
            )[0]
            exact_ms = (time.perf_counter() - exact_started) * 1000.0
            route_table_index_map = torch.arange(
                shortlist_count,
                dtype=torch.long,
                device=device,
            )
            self.last_early_proxy_stats = {
                "enabled": True,
                "topk": self.early_proxy_topk,
                "candidate_counts": [int(rows.shape[0])],
                "shortlist_counts": [shortlist_count],
                "proxy_host_ms": proxy_ms,
                "exact_shortlist_host_ms": exact_ms,
            }
            self.last_adaptive_topk_stats = {"enabled": False}
        elif self.adaptive_topk and not initializing and rows.numel():
            prepared = self._prepare_action_counts(
                selected,
                layout,
                rows,
                source_ranks=sources,
                uniform_source_rank=uniform_source_rank,
                copy_slots=copy_slots,
                affected_groups=affected_groups,
                token_ordinals=ordinals,
                step=step,
                layer_seed=layer_seed,
                num_experts=int(owners.numel()),
                include_pair_interactions=False,
                include_pair_bounds=self.adaptive_topk_strict_certificate,
            )
            context = {
                "rows": rows,
            }
            adaptive_available = self._adaptive_fast_path_available_on_all_ranks([prepared])
            if adaptive_available:
                adaptive_scored, candidate_maps, adaptive_stats = self._adaptive_score_layers(
                    [prepared],
                    [context],
                    communication_scales=[self.communication_scale],
                    forward_compute_per_assignment=[self.forward_compute_per_assignment],
                    forward_compute_constant=[self.forward_compute_constant],
                )
                scored = adaptive_scored[0]
                candidate_index_map = candidate_maps[0]
                route_table_index_map = candidate_index_map
                self.last_adaptive_topk_stats = adaptive_stats
            else:
                prepared = self._prepare_action_counts(
                    selected,
                    layout,
                    rows,
                    source_ranks=sources,
                    uniform_source_rank=uniform_source_rank,
                    copy_slots=copy_slots,
                    affected_groups=affected_groups,
                    token_ordinals=ordinals,
                    step=step,
                    layer_seed=layer_seed,
                    num_experts=int(owners.numel()),
                    include_pair_interactions=True,
                    include_pair_bounds=False,
                )
                scored = self._score_prepared_layers(
                    [prepared],
                    communication_scales=[self.communication_scale],
                    forward_compute_per_assignment=[self.forward_compute_per_assignment],
                    forward_compute_constant=[self.forward_compute_constant],
                )[0]
                self.last_adaptive_topk_stats = {
                    "enabled": False,
                    "reason": "statistical unary fast path unavailable on at least one EP rank",
                }
            self.last_early_proxy_stats = {"enabled": False, "reason": "disabled"}
            self.last_early_proxy_shortlist_indices = []
        else:
            scored = self._score_actions(
                selected,
                layout,
                rows,
                source_ranks=sources,
                uniform_source_rank=uniform_source_rank,
                copy_slots=copy_slots,
                affected_groups=affected_groups,
                token_ordinals=ordinals,
                step=step,
                layer_seed=layer_seed,
                num_experts=int(owners.numel()),
            )
            self.last_adaptive_topk_stats = {"enabled": False}
            self.last_early_proxy_stats = {
                "enabled": False,
                "reason": (
                    "requires statistics scorer, steady state candidates, and one uniform source rank"
                    if self.early_proxy_topk
                    else "disabled"
                ),
            }
            self.last_early_proxy_shortlist_indices = []
        score_ms = (time.perf_counter() - score_started) * 1000.0
        baseline_cost = self._placement_cost(scored, 0) if initializing or rows.numel() == 0 else None
        actions: tuple[PlacementAction, ...] = ()
        final_layout_tensor = layout
        final_cost = baseline_cost
        final_physical = scored.baseline_physical_routes
        final_host_layout = host_layout.clone()
        final_host_owners = host_owners.clone()
        finalization_started = time.perf_counter()

        if initializing:
            fill_actions: list[PlacementAction] = []
            working_host_layout = host_layout.clone()
            for round_index in range(fill_limit):
                if round_index:
                    candidate_started = time.perf_counter()
                    remaining_slots = torch.nonzero(working_host_layout < 0, as_tuple=False).flatten()
                    host_rows = self._cover_rows(working_host_layout, host_owners, remaining_slots)
                    host_copy_slots = self._copy_table(working_host_layout, int(host_owners.numel()))
                    copy_slots = host_copy_slots.to(device=device, non_blocking=True)
                    rows = host_rows.to(device=device, non_blocking=True)
                    affected_groups = (
                        self._candidate_affected_groups(copy_slots, rows)
                        if self.compact_candidate_collective and self.reducer is not None
                        else None
                    )
                    route_stats_ms += (time.perf_counter() - candidate_started) * 1000.0
                    score_started = time.perf_counter()
                    scored = self._score_actions(
                        selected,
                        layout,
                        rows,
                        source_ranks=sources,
                        uniform_source_rank=uniform_source_rank,
                        copy_slots=copy_slots,
                        affected_groups=affected_groups,
                        token_ordinals=ordinals,
                        step=step,
                        layer_seed=layer_seed,
                        num_experts=int(owners.numel()),
                    )
                    score_ms += (time.perf_counter() - score_started) * 1000.0
                if rows.numel() == 0:
                    break
                best_index = None
                next_host_layout = None
                ordered_indices = torch.argsort(scored.total[1:], stable=True).detach().to(device="cpu").tolist()
                remaining_fills = fill_limit - round_index - 1
                for candidate_index in ordered_indices:
                    candidate = host_rows[int(candidate_index)]
                    proposal = working_host_layout.clone()
                    proposal[int(candidate[2])] = int(candidate[3])
                    if self._can_complete_empty_initialization(proposal, remaining_fills):
                        best_index = int(candidate_index)
                        next_host_layout = proposal
                        break
                if best_index is None or next_host_layout is None:
                    break
                row = rows[best_index]
                action = self._placement_action(row.detach().to(device="cpu").tolist())
                fill_actions.append(action)
                final_cost = self._placement_cost(scored, best_index + 1)
                if scored.route_tables is not None and scored.route_hashes is not None:
                    final_physical = self._apply_statistical_action_routes(
                        selected,
                        row,
                        best_index,
                        scored.baseline_physical_routes,
                        scored.route_hashes,
                        scored.route_tables,
                    )
                else:
                    final_physical = self._apply_action_routes(
                        selected,
                        layout,
                        row,
                        scored.baseline_physical_routes,
                        source_ranks=sources,
                        token_ordinals=ordinals,
                        route_hashes=scored.route_hashes,
                        step=step,
                        layer_seed=layer_seed,
                        num_experts=int(owners.numel()),
                    )
                final_layout_tensor = self._apply_rows(layout, row.view(1, -1))[0]
                layout = final_layout_tensor
                working_host_layout = next_host_layout
            actions = tuple(fill_actions)
            if len(actions) != fill_limit:
                raise RuntimeError(
                    f"Greedy replica initialization filled {len(actions)} of {fill_limit} requested empty slots."
                )
            final_host_layout = working_host_layout
        elif rows.numel():
            candidate_costs = scored.total[1:]
            best_position = candidate_costs.argmin()
            best_index = candidate_index_map.index_select(0, best_position.view(1))[0]
            route_table_index = route_table_index_map.index_select(0, best_position.view(1))[0]
            metrics = torch.stack(
                (
                    scored.communication,
                    scored.compute,
                    scored.communication_model_units,
                    scored.peak_rank.to(scored.communication.dtype),
                    scored.peak_compute_rank.to(scored.communication.dtype),
                    scored.selected_dim.to(scored.communication.dtype),
                ),
                dim=1,
            )
            metric_indices = torch.stack((torch.zeros_like(best_position), best_position + 1))
            decision = torch.cat(
                (
                    best_index.to(scored.communication.dtype).view(1),
                    route_table_index.to(scored.communication.dtype).view(1),
                    metrics.index_select(0, metric_indices).reshape(-1),
                )
            )
            decision_values = decision.detach().to(device="cpu").tolist()
            winner_index = int(decision_values[0])
            winner_route_table_index = int(decision_values[1])
            baseline_cost = self._placement_cost_from_values(decision_values[2:8])
            candidate_cost = self._placement_cost_from_values(decision_values[8:14])
            final_cost = baseline_cost
            if candidate_cost.total < baseline_cost.total:
                row = scored_rows[winner_route_table_index]
                host_row = host_rows[winner_index]
                action = self._placement_action(host_row.tolist())
                actions = (action,)
                final_cost = candidate_cost
                if action.kind == "swap":
                    final_host_layout[action.src_slot] = action.dst_logical
                    final_host_layout[action.dst_slot] = action.src_logical
                    final_host_owners[action.src_logical], final_host_owners[action.dst_logical] = (
                        host_owners[action.dst_logical],
                        host_owners[action.src_logical],
                    )
                else:
                    final_host_layout[action.dst_slot] = action.src_logical
                if scored.route_tables is not None and scored.route_hashes is not None:
                    final_physical = self._apply_statistical_action_routes(
                        selected,
                        row,
                        winner_route_table_index,
                        scored.baseline_physical_routes,
                        scored.route_hashes,
                        scored.route_tables,
                    )
                else:
                    final_physical = self._apply_action_routes(
                        selected,
                        layout,
                        row,
                        scored.baseline_physical_routes,
                        source_ranks=sources,
                        token_ordinals=ordinals,
                        route_hashes=scored.route_hashes,
                        step=step,
                        layer_seed=layer_seed,
                        num_experts=int(owners.numel()),
                    )

        assert baseline_cost is not None and final_cost is not None
        if original_selected_ndim == 1:
            final_physical = final_physical.squeeze(-1)
        initial_layout = tuple(int(value) for value in host_layout.tolist())
        final_layout = tuple(int(value) for value in final_host_layout.tolist())
        final_owner_slots = tuple(int(value) for value in final_host_owners.tolist())
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
    "GREEDY_COMMUNICATION_PHASE_MULTIPLIER",
    "GREEDY_COMPUTE_PHASE_MULTIPLIER",
    "GREEDY_COVER_ALGORITHM_VERSION",
    "GreedyCommunicationPlanner",
    "assign_tokens_to_copies_greedy",
]
