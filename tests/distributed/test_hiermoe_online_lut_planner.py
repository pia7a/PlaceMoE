# Copyright 2026 Bytedance Ltd. and/or its affiliates

import torch

from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.online_lut_planner import (
    online_lut_candidates,
    propose_online_lut_move,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _planner() -> GreedyCommunicationPlanner:
    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        forward_compute_per_assignment=0.01,
        traffic_inter_ms_per_byte=0.02,
        traffic_intra_ms_per_byte=0.01,
        traffic_route_ms_per_assignment=0.005,
    )


def test_online_lut_candidates_only_use_existing_alternative_copies():
    layout = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.long)
    source_lut = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    experts, destinations = online_lut_candidates(
        layout,
        source_lut,
        num_experts=4,
    )

    assert list(zip(experts.tolist(), destinations.tolist(), strict=True)) == [
        (0, 4),
        (1, 6),
        (2, 5),
        (3, 7),
    ]


def test_online_lut_proposal_matches_exhaustive_physical_route_replay():
    planner = _planner()
    selected = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=torch.long,
    )
    layout = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.long)
    source_lut = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    baseline_physical = source_lut.index_select(0, selected.reshape(-1)).view_as(selected)
    baseline_counts = planner._local_packed_counts(baseline_physical)
    baseline_assignments = planner._local_packed_assignment_counts(baseline_physical)
    baseline_endpoint = planner._local_traffic_endpoint_statistics(
        baseline_counts,
        baseline_assignments[:, : planner.ep_size],
        source_rank=0,
    ).squeeze(0)

    proposal = propose_online_lut_move(
        planner=planner,
        selected_experts=selected,
        slot_to_logical=layout,
        source_lut=source_lut,
        global_baseline_endpoint=baseline_endpoint,
        source_rank=0,
        num_experts=4,
    )

    assert proposal is not None
    experts, destinations = online_lut_candidates(
        layout,
        source_lut,
        num_experts=4,
    )
    replay_costs = []
    for expert, destination in zip(experts, destinations, strict=True):
        candidate_lut = source_lut.clone()
        candidate_lut[expert] = destination
        candidate_physical = candidate_lut.index_select(0, selected.reshape(-1)).view_as(selected)
        counts = planner._local_packed_counts(candidate_physical)
        assignments = planner._local_packed_assignment_counts(candidate_physical)
        endpoint = planner._local_traffic_endpoint_statistics(
            counts,
            assignments[:, : planner.ep_size],
            source_rank=0,
        )
        communication, compute, *_details = planner._traffic_endpoint_cost_details(endpoint)
        replay_costs.append(float((communication + compute).item()))
    winner = min(range(len(replay_costs)), key=replay_costs.__getitem__)

    assert proposal.expert == int(experts[winner].item())
    assert proposal.destination_slot == int(destinations[winner].item())
    assert proposal.candidate_cost == replay_costs[winner]
