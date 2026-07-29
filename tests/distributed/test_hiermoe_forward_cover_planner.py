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

from types import SimpleNamespace

import torch

from scripts.profile.benchmark_hiermoe_forward_lut_cover_oracle import (
    _source_endpoint_statistics,
)
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.forward_cover_planner import (
    ForwardCoverHeuristicStatistics,
    forward_cover_local_heuristic_statistics,
    forward_cover_local_heuristic_statistics_batched,
    forward_cover_local_validation_stats,
    forward_cover_patch_source_rank_relevant,
    forward_cover_patch_validation_stats_batched,
    patch_forward_cover_routes,
    propose_forward_reuse_cover,
    propose_forward_reuse_covers,
    rotating_service_target_rank,
)
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import PlacementAction
from veomni.distributed.moe.hiermoe.statistical_scorer import (
    prepare_forward_lut_cover_compact_statistics,
    score_forward_lut_cover_compact_statistics,
    score_forward_lut_move_compact_statistics,
    statistical_forward_lut_cover_local_deltas,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


class _FakeExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 2, 2))
        self.down_proj = torch.nn.Parameter(torch.arange(12, 24, dtype=torch.float32).view(3, 2, 2))


def _toy_inputs() -> dict[str, object]:
    return {
        "selected_experts": torch.tensor(
            [
                [2, 3],
                [2, 0],
                [2, 3],
                [1, 3],
            ],
            dtype=torch.long,
        ),
        "physical_routes": torch.tensor(
            [
                [3, 5],
                [3, 0],
                [3, 5],
                [1, 5],
            ],
            dtype=torch.long,
        ),
        "slot_to_logical": torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long),
        "owner_slots": torch.tensor([0, 2, 4, 6], dtype=torch.long),
        "local_slot_assignments": torch.tensor([20.0, 1.0]),
        "source_rank": 0,
        "slots_per_rank": 2,
        "hierarchy_group_sizes": (2,),
        "num_experts": 4,
        "max_copies": 4,
        "level_weights": (1.0, 4.0),
    }


def test_forward_reuse_cover_prefers_expert_reducing_remote_node_bottleneck():
    proposal = propose_forward_reuse_cover(**_toy_inputs(), compute_weight=0.0)

    assert proposal.action is not None
    assert proposal.action.src_logical == 3
    assert proposal.action.src_slot == 6
    assert proposal.action.dst_slot == 1
    assert proposal.action.dst_logical == 1
    assert proposal.estimated_gain > 0.0
    assert proposal.communication_gain > 0.0


def test_patch_source_rank_relevance_uses_service_group_and_victim_lut():
    action = PlacementAction(
        kind="replica",
        src_slot=1,
        dst_slot=19,
        src_logical=1,
        dst_logical=3,
    )
    source_mapping = torch.tensor([0, 1, 2, 19], dtype=torch.long)

    assert forward_cover_patch_source_rank_relevant(
        action=action,
        source_rank=8,
        slots_per_rank=2,
        service_group_size=4,
        source_logical_to_physical=source_mapping,
    )
    assert forward_cover_patch_source_rank_relevant(
        action=action,
        source_rank=0,
        slots_per_rank=2,
        service_group_size=4,
        source_logical_to_physical=source_mapping,
    )

    source_mapping[3] = 6
    assert not forward_cover_patch_source_rank_relevant(
        action=action,
        source_rank=0,
        slots_per_rank=2,
        service_group_size=4,
        source_logical_to_physical=source_mapping,
    )


def test_empty_cover_source_rank_relevance_only_uses_service_group():
    action = PlacementAction(
        kind="replica",
        src_slot=1,
        dst_slot=19,
        src_logical=1,
        dst_logical=-1,
    )

    assert forward_cover_patch_source_rank_relevant(
        action=action,
        source_rank=8,
        slots_per_rank=2,
        service_group_size=4,
        source_logical_to_physical=None,
    )
    assert not forward_cover_patch_source_rank_relevant(
        action=action,
        source_rank=0,
        slots_per_rank=2,
        service_group_size=4,
        source_logical_to_physical=None,
    )


def test_batched_heuristic_statistics_match_scalar_layers():
    inputs = _toy_inputs()
    selected = inputs["selected_experts"]
    physical = inputs["physical_routes"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    target_ranks = torch.tensor([0, 3], dtype=torch.long)

    batched = forward_cover_local_heuristic_statistics_batched(
        selected_experts=torch.stack((selected, selected), dim=0),
        physical_routes=torch.stack((physical, physical), dim=0),
        source_rank=0,
        target_ranks=target_ranks,
        slots_per_rank=2,
        ep_size=4,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        level_weights=(1.0, 4.0),
    )
    scalar = [
        forward_cover_local_heuristic_statistics(
            selected_experts=selected,
            physical_routes=physical,
            source_rank=0,
            target_rank=int(target_rank),
            slots_per_rank=2,
            ep_size=4,
            hierarchy_group_sizes=(2,),
            num_experts=4,
            level_weights=(1.0, 4.0),
        )
        for target_rank in target_ranks
    ]

    torch.testing.assert_close(
        batched.communication_benefit,
        torch.stack([row.communication_benefit for row in scalar]),
    )
    torch.testing.assert_close(
        batched.expert_assignments,
        torch.stack([row.expert_assignments for row in scalar]),
    )
    torch.testing.assert_close(
        batched.baseline_communication_units,
        torch.stack([row.baseline_communication_units for row in scalar]),
    )


def test_forward_reuse_cover_returns_distinct_topk_candidates():
    proposals = propose_forward_reuse_covers(
        **_toy_inputs(),
        compute_weight=0.0,
        max_proposals=2,
    )

    assert len(proposals) == 2
    assert all(proposal.action is not None for proposal in proposals)
    assert [proposal.action.src_logical for proposal in proposals if proposal.action is not None] == [3, 2]
    assert proposals[0].estimated_gain >= proposals[1].estimated_gain


def test_forward_reuse_cover_can_evict_an_owner_with_another_copy():
    inputs = _toy_inputs()
    inputs["local_slot_assignments"] = torch.tensor([0.0, 20.0])

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=0.0)

    assert proposal.action is not None
    assert proposal.action.dst_slot == 0
    assert proposal.action.dst_logical == 0


def test_forward_reuse_cover_prefers_an_empty_slot_without_a_victim():
    inputs = _toy_inputs()
    inputs["slot_to_logical"] = torch.tensor([0, -1, 1, 2, 2, 3, 3, 0], dtype=torch.long)
    inputs["owner_slots"] = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    inputs["local_slot_assignments"] = torch.tensor([20.0, 0.0])

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=0.0)

    assert proposal.action is not None
    assert proposal.action.dst_slot == 1
    assert proposal.action.dst_logical == -1


def test_forward_reuse_empty_slot_reaches_exact_guard_despite_pessimistic_heuristic():
    inputs = _toy_inputs()
    inputs["slot_to_logical"] = torch.tensor([0, -1, 1, 2, 2, 3, 3, 0], dtype=torch.long)
    inputs["owner_slots"] = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    inputs["local_slot_assignments"] = torch.tensor([20.0, 0.0])

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=100.0)

    assert proposal.action is not None
    assert proposal.action.dst_slot == 1
    assert proposal.action.dst_logical == -1
    assert proposal.estimated_gain <= 0.0


def test_forward_reuse_cover_can_target_highest_load_victim():
    proposal = propose_forward_reuse_cover(
        **_toy_inputs(),
        compute_weight=0.0,
        victim_mode="maximum",
    )

    assert proposal.action is not None
    assert proposal.action.dst_slot == 0
    assert proposal.action.dst_logical == 0


def test_forward_reuse_cover_does_not_duplicate_an_expert_in_one_service_group():
    inputs = _toy_inputs()
    inputs["slot_to_logical"] = torch.tensor([0, 3, 1, 2, 2, 1, 3, 0], dtype=torch.long)
    inputs["owner_slots"] = torch.tensor([0, 2, 3, 1], dtype=torch.long)

    proposal = propose_forward_reuse_cover(
        **inputs,
        compute_weight=0.0,
        service_group_size=2,
    )

    assert proposal.action is None


def test_node_scope_target_rotation_visits_every_node_before_another_lane():
    targets = tuple(range(32))

    selected = [
        rotating_service_target_rank(
            targets,
            layer_index=0,
            step=step,
            service_group_size=8,
        )
        for step in range(8)
    ]

    assert selected == [0, 8, 16, 24, 1, 9, 17, 25]


def test_forward_reuse_cover_can_choose_none_when_compute_penalty_dominates():
    proposal = propose_forward_reuse_cover(**_toy_inputs(), compute_weight=100.0)

    assert proposal.action is None
    assert proposal.estimated_gain <= 0.0


def test_forward_cover_service_statistics_charge_remote_target_and_reward_local_node():
    statistics = forward_cover_local_heuristic_statistics(
        selected_experts=torch.tensor([[2], [2]], dtype=torch.long),
        physical_routes=torch.tensor([[6], [6]], dtype=torch.long),
        source_rank=1,
        target_rank=0,
        slots_per_rank=2,
        ep_size=4,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        level_weights=(1.0, 4.0),
    )

    # Moving rank 1's expert from remote node 1 to rank 0 removes one node
    # destination per token. At rank level it trades remote rank 3 for remote
    # rank 0, so the rank benefit is zero.
    assert statistics.communication_benefit.tolist() == [0.0, 0.0, 8.0, 0.0]
    assert statistics.expert_assignments.tolist() == [0.0, 0.0, 2.0, 0.0]


def test_forward_reuse_cover_can_consume_service_group_aggregated_statistics():
    proposal = propose_forward_reuse_cover(
        **_toy_inputs(),
        compute_weight=0.0,
        aggregated_statistics=ForwardCoverHeuristicStatistics(
            communication_benefit=torch.tensor([0.0, 0.0, 8.0, 1.0]),
            expert_assignments=torch.tensor([0.0, 0.0, 2.0, 1.0]),
            baseline_communication_units=torch.tensor(8.0),
        ),
    )

    assert proposal.action is not None
    assert proposal.action.src_logical == 2
    assert proposal.action.dst_slot == 1


def test_forward_reuse_cover_requires_a_non_owner_target_slot():
    inputs = _toy_inputs()
    inputs["slot_to_logical"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["owner_slots"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["local_slot_assignments"] = torch.tensor([20.0])
    inputs["slots_per_rank"] = 1
    inputs["physical_routes"] = torch.tensor(
        [
            [2, 3],
            [2, 0],
            [2, 3],
            [1, 3],
        ],
        dtype=torch.long,
    )

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=0.0)

    assert proposal.action is None


def test_forward_cover_affected_token_validation_matches_full_remap():
    selected = _toy_inputs()["selected_experts"]
    layout = _toy_inputs()["slot_to_logical"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(layout, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )
    baseline = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        step=3,
        layer_seed=17,
        max_copies=4,
    )
    validation = forward_cover_local_validation_stats(
        selected_experts=selected,
        physical_routes=baseline,
        slot_to_logical=layout,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        max_copies=4,
        step=3,
        layer_seed=17,
    )
    candidate_layout = layout.clone()
    candidate_layout[action.dst_slot] = action.src_logical
    candidate = assign_tokens_to_copies_greedy(
        selected,
        candidate_layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        step=3,
        layer_seed=17,
        max_copies=4,
    )

    def packed_counts(physical: torch.Tensor) -> torch.Tensor:
        ranks = physical // 2
        rows = []
        for size, groups in ((1, ranks), (2, ranks // 2)):
            hits = torch.zeros((physical.shape[0], 4 // size), dtype=torch.bool)
            hits.scatter_(1, groups, True)
            rows.append(hits.sum(dim=0).to(torch.float32))
        return torch.cat(rows)

    def assignment_counts(physical: torch.Tensor) -> torch.Tensor:
        return torch.bincount((physical // 2).reshape(-1), minlength=4).to(torch.float32)

    cached_validation = forward_cover_local_validation_stats(
        selected_experts=selected,
        physical_routes=baseline,
        slot_to_logical=layout,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        max_copies=4,
        step=3,
        layer_seed=17,
        baseline_communication_counts=packed_counts(baseline),
        baseline_assignment_counts=assignment_counts(baseline),
    )
    torch.testing.assert_close(
        validation.baseline_communication_counts + validation.communication_count_delta,
        packed_counts(candidate),
    )
    torch.testing.assert_close(
        validation.baseline_assignment_counts + validation.assignment_count_delta,
        assignment_counts(candidate),
    )
    torch.testing.assert_close(
        cached_validation.baseline_communication_counts,
        validation.baseline_communication_counts,
    )
    torch.testing.assert_close(
        cached_validation.communication_count_delta,
        validation.communication_count_delta,
    )
    torch.testing.assert_close(
        cached_validation.baseline_assignment_counts,
        validation.baseline_assignment_counts,
    )
    torch.testing.assert_close(
        cached_validation.assignment_count_delta,
        validation.assignment_count_delta,
    )
    assert validation.affected_tokens == int(
        ((selected == action.src_logical) | (selected == action.dst_logical)).any(dim=1).sum().item()
    )


def test_patch_forward_cover_routes_supports_an_empty_destination():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=-1,
    )

    patched = patch_forward_cover_routes(
        selected_experts=selected,
        physical_routes=physical,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        victim_fallback_slot=1,
    )

    torch.testing.assert_close(patched[selected == 3], torch.ones_like(patched[selected == 3]))


def test_forward_dispatch_endpoint_cache_matches_source_route_statistics():
    ep_size = 4
    group_size = 2
    slots_per_rank = 2
    hierarchy = Hierarchy(ep_size=ep_size, group_sizes=(group_size, ep_size), source="test")
    planner = GreedyCommunicationPlanner(
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
    )
    physical_by_source = (
        torch.tensor([[0, 2], [4, 6], [2, 3]], dtype=torch.long),
        torch.tensor([[1, 3], [4, 5], [6, 7]], dtype=torch.long),
        torch.tensor([[0, 1], [2, 4], [5, 7]], dtype=torch.long),
        torch.tensor([[0, 6], [3, 5], [4, 7]], dtype=torch.long),
    )
    source_unique = [planner._local_packed_counts(physical).squeeze(0) for physical in physical_by_source]
    source_assignments = [
        planner._local_packed_assignment_counts(physical).squeeze(0) for physical in physical_by_source
    ]
    expected = sum(
        (
            planner._local_traffic_endpoint_statistics(
                unique.unsqueeze(0),
                assignments[:ep_size].unsqueeze(0),
                source_rank=source_rank,
            ).squeeze(0)
            for source_rank, (unique, assignments) in enumerate(zip(source_unique, source_assignments, strict=True))
        ),
        torch.zeros((8 * ep_size,), dtype=torch.float32),
    )

    cached_rows = []
    num_nodes = ep_size // group_size
    for rank in range(ep_size):
        manager = expert_swap_module.ExpertSwapManager(
            ep_group=None,
            ep_size=ep_size,
            ep_rank=rank,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=0,
            redundant_slot_increment_per_device=2,
            max_replica_rounds=1,
            smooth_max_gamma=10.0,
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            expert_swap_mode="step",
            expert_swap_selector="hiermoe_greedy_cover_p1",
        )
        manager._forward_reuse_cover = True
        key = "layers.0.mlp.experts"
        manager.register_layer(key, _FakeExperts())
        manager.layers[key].latest_route_step = 1

        unique = source_unique[rank]
        assignments = source_assignments[rank]
        node_unique = unique[ep_size:]
        rank_assignments = assignments[:ep_size]
        stage1_unique_send = node_unique.to(torch.int64).tolist()
        stage1_assignment_send = rank_assignments.view(num_nodes, group_size).sum(dim=1).to(torch.int64).tolist()

        lane = rank % group_size
        destination_node = rank // group_size
        relay_sources = range(lane, ep_size, group_size)
        stage2_unique_send = (
            sum(
                (
                    source_unique[source][destination_node * group_size : (destination_node + 1) * group_size]
                    for source in relay_sources
                ),
                torch.zeros((group_size,), dtype=torch.float32),
            )
            .to(torch.int64)
            .tolist()
        )
        stage2_assignment_send = (
            sum(
                (
                    source_assignments[source][destination_node * group_size : (destination_node + 1) * group_size]
                    for source in relay_sources
                ),
                torch.zeros((group_size,), dtype=torch.float32),
            )
            .to(torch.int64)
            .tolist()
        )
        context = SimpleNamespace(
            mode="hierarchical",
            stage1_unique_send_splits=stage1_unique_send,
            stage1_unique_recv_splits=[0] * num_nodes,
            stage1_assignment_send_splits=stage1_assignment_send,
            stage2_unique_send_splits=stage2_unique_send,
            stage2_unique_recv_splits=[0] * group_size,
            stage2_assignment_send_splits=stage2_assignment_send,
        )
        manager.record_dispatch_statistics(
            layer_key=key,
            step=1,
            dispatch_context=context,
        )
        cached = manager.layers[key].latest_forward_traffic_endpoint_statistics
        assert cached is not None
        cached_rows.append(cached)

    torch.testing.assert_close(torch.stack(cached_rows).sum(dim=0), expected)


def test_source_endpoint_statistics_match_summed_local_rows():
    ep_size = 4
    group_size = 2
    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(group_size, ep_size), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
    )
    physical_by_source = (
        torch.tensor([[0, 2], [4, 6], [2, 3]], dtype=torch.long),
        torch.tensor([[1, 3], [4, 5], [6, 7]], dtype=torch.long),
        torch.tensor([[0, 1], [2, 4], [5, 7]], dtype=torch.long),
        torch.tensor([[0, 6], [3, 5], [4, 7]], dtype=torch.long),
    )
    unique = torch.stack(
        [planner._local_packed_counts(physical)[0] for physical in physical_by_source],
        dim=0,
    ).unsqueeze(1)
    assignments = torch.stack(
        [planner._local_packed_assignment_counts(physical)[0, :ep_size] for physical in physical_by_source],
        dim=0,
    ).unsqueeze(1)
    expected = sum(
        (
            planner._local_traffic_endpoint_statistics(
                unique[source],
                assignments[source],
                source_rank=source,
            )
            for source in range(ep_size)
        ),
        torch.zeros((1, 8 * ep_size), dtype=torch.float32),
    )

    actual = _source_endpoint_statistics(
        unique_counts=unique,
        assignment_counts=assignments,
        ep_size=ep_size,
        ranks_per_node=group_size,
    )

    torch.testing.assert_close(actual, expected)


def test_forward_cover_patch_changes_only_inserted_and_evicted_routes():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )

    patched = patch_forward_cover_routes(
        selected_experts=selected,
        physical_routes=physical,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        victim_fallback_slot=2,
    )

    expected = torch.tensor(
        [
            [3, 1],
            [3, 0],
            [3, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(patched, expected)
    unaffected = (selected != action.src_logical) & (physical != action.dst_slot)
    torch.testing.assert_close(patched[unaffected], physical[unaffected])


def test_forward_cover_patch_can_serve_every_source_rank_in_destination_node():
    selected = torch.tensor([[3, 1], [2, 3]], dtype=torch.long)
    physical = torch.tensor([[6, 1], [3, 6]], dtype=torch.long)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )

    same_node = patch_forward_cover_routes(
        selected_experts=selected,
        physical_routes=physical,
        action=action,
        source_rank=1,
        slots_per_rank=2,
        victim_fallback_slot=2,
        service_group_size=2,
    )
    remote_node = patch_forward_cover_routes(
        selected_experts=selected,
        physical_routes=physical,
        action=action,
        source_rank=2,
        slots_per_rank=2,
        victim_fallback_slot=2,
        service_group_size=2,
    )

    torch.testing.assert_close(same_node, torch.tensor([[1, 2], [3, 1]]))
    torch.testing.assert_close(remote_node, torch.tensor([[6, 2], [3, 6]]))


def test_batched_patch_validation_matches_individual_exact_counts():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    layout = _toy_inputs()["slot_to_logical"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    assert isinstance(layout, torch.Tensor)
    actions = (
        PlacementAction(
            kind="replica",
            src_slot=6,
            dst_slot=1,
            src_logical=3,
            dst_logical=1,
        ),
        PlacementAction(
            kind="replica",
            src_slot=3,
            dst_slot=5,
            src_logical=2,
            dst_logical=3,
        ),
    )
    fallbacks = (2, 6)
    selected_batch = torch.stack((selected, selected.flip(0)))
    physical_batch = torch.stack((physical, physical.flip(0)))

    batched = forward_cover_patch_validation_stats_batched(
        selected_experts=selected_batch,
        physical_routes=physical_batch,
        source_logical=torch.tensor([action.src_logical for action in actions]),
        victim_logical=torch.tensor([action.dst_logical for action in actions]),
        destination_slots=torch.tensor([action.dst_slot for action in actions]),
        victim_fallback_slots=torch.tensor(fallbacks),
        source_rank=0,
        slots_per_rank=2,
        ep_size=4,
        hierarchy_group_sizes=(2,),
    )
    for index, (action, fallback) in enumerate(zip(actions, fallbacks, strict=True)):
        exact = forward_cover_local_validation_stats(
            selected_experts=selected_batch[index],
            physical_routes=physical_batch[index],
            slot_to_logical=layout,
            action=action,
            source_rank=0,
            slots_per_rank=2,
            hierarchy_group_sizes=(2,),
            num_experts=4,
            max_copies=4,
            step=3,
            layer_seed=17,
            patch_remap=True,
            victim_fallback_slot=fallback,
        )
        torch.testing.assert_close(
            batched.communication_count_delta[index],
            exact.communication_count_delta,
        )
        torch.testing.assert_close(
            batched.assignment_count_delta[index],
            exact.assignment_count_delta,
        )
        assert int(batched.affected_tokens[index].item()) == exact.affected_tokens


def test_batched_node_patch_validation_matches_individual_exact_counts():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    layout = _toy_inputs()["slot_to_logical"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    assert isinstance(layout, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )

    batched = forward_cover_patch_validation_stats_batched(
        selected_experts=selected.unsqueeze(0),
        physical_routes=physical.unsqueeze(0),
        source_logical=torch.tensor([action.src_logical]),
        victim_logical=torch.tensor([action.dst_logical]),
        destination_slots=torch.tensor([action.dst_slot]),
        victim_fallback_slots=torch.tensor([2]),
        source_rank=1,
        slots_per_rank=2,
        ep_size=4,
        hierarchy_group_sizes=(2,),
        service_group_size=2,
    )
    exact = forward_cover_local_validation_stats(
        selected_experts=selected,
        physical_routes=physical,
        slot_to_logical=layout,
        action=action,
        source_rank=1,
        slots_per_rank=2,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        max_copies=4,
        step=3,
        layer_seed=17,
        patch_remap=True,
        victim_fallback_slot=2,
        service_group_size=2,
    )

    torch.testing.assert_close(batched.communication_count_delta[0], exact.communication_count_delta)
    torch.testing.assert_close(batched.assignment_count_delta[0], exact.assignment_count_delta)
    assert int(batched.affected_tokens[0].item()) == exact.affected_tokens


def test_statistical_forward_lut_cover_deltas_match_individual_patch_replay():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    layout = _toy_inputs()["slot_to_logical"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    assert isinstance(layout, torch.Tensor)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    planner = GreedyCommunicationPlanner(
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
    )
    actions = (
        PlacementAction(
            kind="replica",
            src_slot=6,
            dst_slot=1,
            src_logical=3,
            dst_logical=1,
        ),
        PlacementAction(
            kind="replica",
            src_slot=3,
            dst_slot=5,
            src_logical=2,
            dst_logical=3,
        ),
    )
    rows = torch.tensor(
        [[1, action.src_slot, action.dst_slot, action.src_logical, action.dst_logical] for action in actions],
        dtype=torch.long,
    )
    fallbacks = torch.tensor([2, 6], dtype=torch.long)
    source_lut = torch.tensor([0, 1, 3, 5], dtype=torch.long)

    communication_delta, assignment_delta = statistical_forward_lut_cover_local_deltas(
        planner,
        selected,
        rows,
        physical=physical,
        source_logical_to_physical=source_lut,
        victim_fallback_slots=fallbacks,
        uniform_source_rank=0,
        service_group_size=2,
        num_experts=4,
    )
    compact_statistics = prepare_forward_lut_cover_compact_statistics(
        planner,
        selected,
        source_logical_to_physical=source_lut,
        num_experts=4,
    )
    compact_communication_delta, compact_assignment_delta = score_forward_lut_cover_compact_statistics(
        planner,
        compact_statistics,
        rows,
        source_logical_to_physical=source_lut,
        victim_fallback_slots=fallbacks,
        uniform_source_rank=0,
        service_group_size=2,
        num_experts=4,
    )
    torch.testing.assert_close(compact_communication_delta, communication_delta)
    torch.testing.assert_close(compact_assignment_delta, assignment_delta)

    for index, (action, fallback) in enumerate(zip(actions, fallbacks, strict=True)):
        exact = forward_cover_local_validation_stats(
            selected_experts=selected,
            physical_routes=physical,
            slot_to_logical=layout,
            action=action,
            source_rank=0,
            slots_per_rank=2,
            hierarchy_group_sizes=(2,),
            num_experts=4,
            max_copies=4,
            step=3,
            layer_seed=17,
            patch_remap=True,
            victim_fallback_slot=int(fallback),
            service_group_size=2,
        )
        torch.testing.assert_close(
            communication_delta[index],
            exact.communication_count_delta,
        )
        torch.testing.assert_close(
            assignment_delta[index],
            exact.assignment_count_delta,
        )


def test_compact_forward_lut_move_statistics_match_exact_physical_routes():
    selected = _toy_inputs()["selected_experts"]
    assert isinstance(selected, torch.Tensor)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    planner = GreedyCommunicationPlanner(
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
    )
    source_lut = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    experts = torch.tensor([0, 3], dtype=torch.long)
    destination_slots = torch.tensor([7, 6], dtype=torch.long)
    statistics = prepare_forward_lut_cover_compact_statistics(
        planner,
        selected,
        source_logical_to_physical=source_lut,
        num_experts=4,
    )

    communication_delta, assignment_delta = score_forward_lut_move_compact_statistics(
        planner,
        statistics,
        experts,
        destination_slots,
        source_logical_to_physical=source_lut,
        num_experts=4,
    )
    baseline_physical = source_lut.index_select(0, selected.reshape(-1)).view_as(selected)
    baseline_communication = planner._local_packed_counts(baseline_physical)
    baseline_assignment = planner._local_packed_assignment_counts(baseline_physical)

    for index, (expert, destination) in enumerate(zip(experts, destination_slots, strict=True)):
        candidate_physical = baseline_physical.clone()
        candidate_physical[selected == expert] = destination
        exact_communication_delta = planner._local_packed_counts(candidate_physical) - baseline_communication
        exact_assignment_delta = planner._local_packed_assignment_counts(candidate_physical) - baseline_assignment
        torch.testing.assert_close(
            communication_delta[index],
            exact_communication_delta.squeeze(0),
        )
        torch.testing.assert_close(
            assignment_delta[index],
            exact_assignment_delta.squeeze(0)[: planner.ep_size],
        )


def test_forward_reuse_cover_executor_batches_transfer_before_layout_publish(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._forward_reuse_cover_patch_remap = True
    layer.source_logical_to_physical = torch.tensor(
        [
            [0, 1, 3, 4],
            [5, 1, 3, 4],
        ],
        dtype=torch.long,
    )

    calls = []

    def capture(grouped_entries, **_kwargs):
        calls.append(grouped_entries)
        assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 2, 2, 3, 0)

    monkeypatch.setattr(manager, "_execute_sparse_group_slot_transfers", capture)
    action = PlacementAction(
        kind="replica",
        src_slot=4,
        dst_slot=2,
        src_logical=3,
        dst_logical=2,
    )

    committed = manager._execute_forward_cover_actions(((layer, action),))

    assert len(calls) == 1
    assert (1, 0) in calls[0]
    assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 3, 2, 3, 0)
    assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 3, 4)
    assert tuple(layer.source_logical_to_physical[0].tolist()) == (0, 1, 3, 2)
    assert tuple(layer.source_logical_to_physical[1].tolist()) == (5, 1, 3, 4)
    selected = torch.tensor([[3, 2, 0]], dtype=torch.long)
    torch.testing.assert_close(
        manager.map_logical_to_physical(key, selected),
        torch.tensor([[2, 3, 0]], dtype=torch.long),
    )
    assert committed == [f"{key}:replica(3->2)"]


def test_forward_reuse_empty_seeding_initializes_source_route_lut():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    manager._forward_reuse_cover_patch_remap = True

    manager.register_layer("layers.0.mlp.experts", _FakeExperts())

    layer = manager.layers["layers.0.mlp.experts"]
    torch.testing.assert_close(
        layer.source_logical_to_physical,
        torch.tensor(
            [
                [0, 1, 3, 4],
                [0, 1, 3, 4],
            ],
            dtype=torch.long,
        ),
    )


def test_forward_reuse_cover_executor_updates_every_source_lut_row_in_service_group(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._forward_reuse_cover_patch_remap = True
    manager._forward_reuse_cover_service_group_size = 2
    layer.source_logical_to_physical = torch.tensor(
        [
            [0, 1, 3, 4],
            [5, 1, 3, 4],
        ],
        dtype=torch.long,
    )
    monkeypatch.setattr(manager, "_execute_sparse_group_slot_transfers", lambda *_args, **_kwargs: None)
    action = PlacementAction(
        kind="replica",
        src_slot=4,
        dst_slot=2,
        src_logical=3,
        dst_logical=2,
    )

    manager._execute_forward_cover_actions(((layer, action),))

    torch.testing.assert_close(
        layer.source_logical_to_physical,
        torch.tensor(
            [
                [0, 1, 3, 2],
                [5, 1, 3, 2],
            ],
            dtype=torch.long,
        ),
    )


def test_forward_reuse_cover_executor_promotes_remaining_copy_when_owner_is_victim(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._forward_reuse_cover_patch_remap = True
    layer.source_logical_to_physical = torch.tensor(
        [
            [0, 1, 3, 4],
            [5, 1, 3, 4],
        ],
        dtype=torch.long,
    )
    monkeypatch.setattr(manager, "_execute_sparse_group_slot_transfers", lambda *_args, **_kwargs: None)
    action = PlacementAction(
        kind="replica",
        src_slot=4,
        dst_slot=0,
        src_logical=3,
        dst_logical=0,
    )

    committed = manager._execute_forward_cover_actions(((layer, action),))

    assert tuple(layer.slot_to_logical.tolist()) == (3, 1, 2, 2, 3, 0)
    assert tuple(layer.logical_to_physical.tolist()) == (5, 1, 3, 4)
    assert tuple(layer.source_logical_to_physical[0].tolist()) == (5, 1, 3, 0)
    assert tuple(layer.source_logical_to_physical[1].tolist()) == (5, 1, 3, 4)
    assert committed == [f"{key}:replica(3->0)"]
