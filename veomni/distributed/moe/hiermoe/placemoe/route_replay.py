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

"""Complete-route loading and calibrated held-out evaluation for PlaceMoE."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..greedy_planner import GreedyCommunicationPlanner
from ..perf_model import HierMoEPerfModel
from ..topology import Hierarchy, expected_hierarchy_group_sizes


@dataclass(frozen=True)
class HybridCost:
    communication_ms: float
    compute_ms: float
    total_ms: float
    peak_communication_rank: int
    peak_compute_rank: int
    mean_destination_nodes: float
    mean_destination_ranks: float
    peak_assignments: float


def load_routes(
    root: Path,
    *,
    steps: tuple[int, ...],
    layer: int,
    ep_size: int,
    call_indices: tuple[int, ...] = (0,),
    forward_repeats: int = 1,
    layer_stride: int | None = None,
) -> list[list[torch.Tensor]]:
    """Load per-rank token routes from bundled or individual capture files."""
    if not call_indices:
        raise ValueError("call_indices must not be empty.")
    if forward_repeats <= 0:
        raise ValueError("forward_repeats must be positive.")
    if layer_stride is None:
        layer_stride = 0
    if layer_stride < 0:
        raise ValueError("layer_stride must be non-negative.")
    if forward_repeats > 1 and layer_stride == 0:
        raise ValueError("layer_stride must be positive when forward_repeats > 1.")

    samples: list[list[torch.Tensor]] = []
    for step in steps:
        for repeat in range(forward_repeats):
            capture_layer = layer + repeat * layer_stride
            for call_index in call_indices:
                bundle_path = root / f"step{step:04d}" / f"layer{capture_layer:02d}_call{call_index}_all_ranks.pt"
                if bundle_path.is_file():
                    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
                    routes_by_rank = payload.get("routes_by_rank") if isinstance(payload, dict) else None
                    if (
                        not isinstance(payload, dict)
                        or payload.get("format") != "hiermoe-local-route-bundle-v1"
                        or int(payload.get("ep_size", -1)) != ep_size
                        or not isinstance(routes_by_rank, (list, tuple))
                        or len(routes_by_rank) != ep_size
                    ):
                        raise ValueError(f"Invalid bundled route capture: {bundle_path}.")
                    rows = []
                    for route in routes_by_rank:
                        if not torch.is_tensor(route) or route.ndim != 2:
                            raise ValueError(f"Invalid bundled route capture: {bundle_path}.")
                        rows.append(route.to(dtype=torch.long).contiguous())
                    samples.append(rows)
                    continue

                rows: list[torch.Tensor] = []
                for rank in range(ep_size):
                    path = root / f"step{step:04d}" / f"layer{capture_layer:02d}_call{call_index}_rank{rank:02d}.pt"
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    route = payload.get("routes") if isinstance(payload, dict) else None
                    if not torch.is_tensor(route) or route.ndim != 2:
                        raise ValueError(f"Invalid route capture: {path}.")
                    if int(payload.get("ep_size", -1)) != ep_size:
                        raise ValueError(f"Route capture has a different EP size: {path}.")
                    rows.append(route.to(dtype=torch.long).contiguous())
                samples.append(rows)
    return samples


class HybridEvaluator:
    """Replay complete routes to measure the calibrated communication/compute cost."""

    def __init__(self, args: argparse.Namespace) -> None:
        hierarchy_group_sizes = tuple(
            int(size)
            for size in (
                getattr(args, "hierarchy_group_sizes", ())
                or expected_hierarchy_group_sizes(args.ep_size, args.ranks_per_node)
            )
        )
        if hierarchy_group_sizes[-1] != args.ep_size or len(hierarchy_group_sizes) not in {1, 2, 3}:
            raise ValueError("Hybrid evaluator requires a one-, two-, or three-stage EP hierarchy.")
        if len(hierarchy_group_sizes) == 1 and hierarchy_group_sizes != expected_hierarchy_group_sizes(
            args.ep_size, args.ranks_per_node
        ):
            raise ValueError("A one-stage evaluator hierarchy requires all EP ranks to be on one node.")
        hierarchy = Hierarchy(
            ep_size=args.ep_size,
            group_sizes=hierarchy_group_sizes,
            source="placemoe-route-replay",
        )
        mid_ms_per_byte = getattr(args, "mid_ms_per_byte", None)
        if len(hierarchy_group_sizes) == 1:
            self.level_ms_per_byte = (float(args.intra_ms_per_byte),)
        elif len(hierarchy_group_sizes) == 3:
            self.level_ms_per_byte = (
                float(args.inter_ms_per_byte),
                float(args.inter_ms_per_byte if mid_ms_per_byte is None else mid_ms_per_byte),
                float(args.intra_ms_per_byte),
            )
        else:
            self.level_ms_per_byte = (
                float(args.inter_ms_per_byte),
                float(args.intra_ms_per_byte),
            )
        self.planner = GreedyCommunicationPlanner(
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=args.hidden_size,
            bytes_per_element=args.bytes_per_element,
            slots_per_rank=args.slots_per_rank,
            forward_compute_per_assignment=args.compute_ms_per_assignment,
            traffic_inter_ms_per_byte=args.inter_ms_per_byte,
            traffic_intra_ms_per_byte=args.intra_ms_per_byte,
            traffic_route_ms_per_assignment=args.route_ms_per_assignment,
            traffic_communication_phase_multiplier=args.communication_phase_multiplier,
            traffic_compute_phase_multiplier=args.compute_phase_multiplier,
        )
        self.args = args

    def evaluate(self, samples: list[list[torch.Tensor]], source_lut: np.ndarray) -> HybridCost:
        communication = 0.0
        compute = 0.0
        node_destinations = 0.0
        rank_destinations = 0.0
        tokens = 0
        peak_assignments = 0.0
        peak_communication_rank = -1
        peak_compute_rank = -1
        lut = torch.from_numpy(np.array(source_lut, dtype=np.int64, copy=True))
        three_stage = len(self.level_ms_per_byte) == 3
        packed_width = sum(self.planner._count_widths())
        for sample in samples:
            if three_stage:
                unique_by_source = torch.zeros((self.args.ep_size, 1, packed_width), dtype=torch.float32)
                assignments_by_source = torch.zeros_like(unique_by_source)
            else:
                endpoint = torch.zeros((1, 8 * self.args.ep_size), dtype=torch.float32)
            assignment_totals = torch.zeros((self.args.ep_size,), dtype=torch.float32)
            for source_rank, logical in enumerate(sample):
                physical = lut[source_rank].index_select(0, logical.reshape(-1)).view_as(logical)
                unique = self.planner._local_packed_counts(physical)
                assignments = self.planner._local_packed_assignment_counts(physical)
                if three_stage:
                    unique_by_source[source_rank, 0] = unique[0]
                    assignments_by_source[source_rank, 0] = assignments[0]
                else:
                    endpoint += self.planner._local_traffic_endpoint_statistics(
                        unique, assignments, source_rank=source_rank
                    )
                assignment_totals += assignments[0, : self.args.ep_size]
                ranks = torch.div(physical, self.args.slots_per_rank, rounding_mode="floor")
                rank_hits = torch.zeros((logical.shape[0], self.args.ep_size), dtype=torch.bool)
                rank_hits.scatter_(1, ranks, True)
                nodes = torch.div(ranks, self.args.ranks_per_node, rounding_mode="floor")
                node_hits = torch.zeros(
                    (logical.shape[0], self.args.ep_size // self.args.ranks_per_node), dtype=torch.bool
                )
                node_hits.scatter_(1, nodes, True)
                rank_destinations += float(rank_hits.sum().item())
                node_destinations += float(node_hits.sum().item())
                tokens += int(logical.shape[0])

            sample_peak, sample_peak_rank = assignment_totals.max(dim=0)
            sample_peak_assignments = float(sample_peak.item())
            if three_stage:
                traffic = self.planner._hierarchical_traffic_features(unique_by_source, assignments_by_source)
                network_ms = sum(
                    coefficient * float(traffic[f"stage{index}_payload_endpoint_bytes"].item())
                    for index, coefficient in enumerate(self.level_ms_per_byte, start=1)
                )
                sample_communication = float(self.args.communication_phase_multiplier) * (
                    network_ms + float(self.args.route_ms_per_assignment) * sample_peak_assignments
                )
                sample_compute = float(self.args.compute_phase_multiplier) * (
                    float(self.args.compute_ms_per_assignment) * sample_peak_assignments
                    + float(getattr(self.planner, "forward_compute_constant", 0.0))
                )
                communication += sample_communication
                compute += sample_compute
                if sample_peak_assignments >= peak_assignments:
                    peak_assignments = sample_peak_assignments
                    peak_communication_rank = int(sample_peak_rank.item())
                    peak_compute_rank = int(sample_peak_rank.item())
            else:
                details = self.planner._traffic_endpoint_cost_details(endpoint)
                communication += float(details[0].item())
                compute += float(details[1].item())
                peak_communication_rank = int(details[3].item())
                peak_compute_rank = int(details[4].item())
                peak_assignments = max(peak_assignments, sample_peak_assignments)
        return HybridCost(
            communication_ms=communication,
            compute_ms=compute,
            total_ms=communication + compute,
            peak_communication_rank=peak_communication_rank,
            peak_compute_rank=peak_compute_rank,
            mean_destination_nodes=node_destinations / max(tokens, 1),
            mean_destination_ranks=rank_destinations / max(tokens, 1),
            peak_assignments=peak_assignments,
        )
