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

"""Exact one-entry source-LUT correction for a fixed expert layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .greedy_planner import GreedyCommunicationPlanner
from .statistical_scorer import (
    prepare_forward_lut_cover_compact_statistics,
    score_forward_lut_move_compact_statistics,
)


ONLINE_LUT_ALGORITHM_VERSION = "hiermoe-online-lut-p1"


@dataclass(frozen=True)
class OnlineLUTProposal:
    """The best local source-LUT move for one layer and source rank."""

    expert: int
    destination_slot: int
    candidate_cost: float
    candidate_communication: float
    candidate_compute: float


def online_lut_candidates(
    slot_to_logical: torch.Tensor,
    source_lut: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every valid one-entry LUT move without host-side enumeration."""

    device = source_lut.device
    layout = slot_to_logical.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    source = source_lut.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    if int(source.numel()) != int(num_experts):
        raise ValueError("The source-rank LUT must contain one slot per logical expert.")
    expert_rows = torch.arange(int(num_experts), dtype=torch.long, device=device).view(-1, 1)
    slot_columns = torch.arange(layout.numel(), dtype=torch.long, device=device).view(1, -1)
    movable = (layout.view(1, -1) == expert_rows) & (slot_columns != source.view(-1, 1))
    return expert_rows.expand(-1, layout.numel())[movable], slot_columns.expand(int(num_experts), -1)[movable]


@torch.no_grad()
def propose_online_lut_move(
    *,
    planner: GreedyCommunicationPlanner,
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    source_lut: torch.Tensor,
    global_baseline_endpoint: torch.Tensor,
    source_rank: int,
    num_experts: int,
) -> OnlineLUTProposal | None:
    """Select the exact best one-entry move owned by one source rank."""

    selected = selected_experts.to(dtype=torch.long)
    source = source_lut.to(device=selected.device, dtype=torch.long, non_blocking=True)
    experts, destinations = online_lut_candidates(
        slot_to_logical,
        source,
        num_experts=num_experts,
    )
    if experts.numel() == 0:
        return None

    statistics = prepare_forward_lut_cover_compact_statistics(
        planner,
        selected,
        source_logical_to_physical=source,
        num_experts=num_experts,
    )
    communication_delta, assignment_delta = score_forward_lut_move_compact_statistics(
        planner,
        statistics,
        experts,
        destinations,
        source_logical_to_physical=source,
        num_experts=num_experts,
    )
    endpoint_delta = planner._local_traffic_endpoint_statistics(
        communication_delta,
        assignment_delta,
        source_rank=int(source_rank),
    )
    baseline = global_baseline_endpoint.to(
        device=selected.device,
        dtype=torch.float32,
        non_blocking=True,
    ).reshape(1, -1)
    communication, compute, *_details = planner._traffic_endpoint_cost_details(baseline + endpoint_delta)
    total = communication + compute
    winner = int(total.argmin().item())
    return OnlineLUTProposal(
        expert=int(experts[winner].item()),
        destination_slot=int(destinations[winner].item()),
        candidate_cost=float(total[winner].item()),
        candidate_communication=float(communication[winner].item()),
        candidate_compute=float(compute[winner].item()),
    )


__all__ = [
    "ONLINE_LUT_ALGORITHM_VERSION",
    "OnlineLUTProposal",
    "online_lut_candidates",
    "propose_online_lut_move",
]
