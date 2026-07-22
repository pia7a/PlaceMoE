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

import pytest
import torch

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GREEDY_COVER_ALGORITHM_VERSION,
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _planner(*, chunk_size: int = 8) -> GreedyCommunicationPlanner:
    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=2,
        candidate_chunk_size=chunk_size,
    )


def _full_layout() -> tuple[torch.Tensor, torch.Tensor]:
    # Owners occupy the first slot of each rank. Every second slot is a
    # replaceable replica, and expert 0 intentionally has no replica.
    return (
        torch.tensor([0, 3, 1, 2, 2, 3, 3, 1], dtype=torch.long),
        torch.tensor([0, 2, 4, 6], dtype=torch.long),
    )


def _assert_routes_match_layout(
    selected: torch.Tensor,
    physical: torch.Tensor,
    layout: tuple[int, ...],
) -> None:
    layout_tensor = torch.tensor(layout, dtype=torch.long)
    torch.testing.assert_close(layout_tensor.index_select(0, physical.reshape(-1)).view_as(selected), selected)


def test_greedy_copy_mapping_is_deterministic_and_keeps_duplicate_routes_together():
    layout, _owners = _full_layout()
    selected = torch.tensor([[0, 1, 0], [2, 3, 2], [1, 3, 0]], dtype=torch.long)
    sources = torch.tensor([0, 1, 3], dtype=torch.long)

    first = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=torch.tensor([5, 7, 11]),
        step=13,
        layer_seed=17,
    )
    second = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=torch.tensor([5, 7, 11]),
        step=13,
        layer_seed=17,
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(layout.index_select(0, first.reshape(-1)).view_as(selected), selected)
    assert first[0, 0].item() == first[0, 2].item()
    assert first[1, 0].item() == first[1, 2].item()


def test_incremental_action_mapping_matches_full_remap_for_every_swap_and_cover():
    planner = _planner()
    layout, owners = _full_layout()
    selected = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    sources = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    baseline = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
    )
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))

    for row in rows:
        candidate_layout = planner._apply_rows(layout, row.view(1, -1))[0]
        expected = assign_tokens_to_copies_greedy(
            selected,
            candidate_layout,
            slots_per_rank=2,
            source_ranks=sources,
            hierarchy_group_sizes=(2, 4),
            num_experts=4,
            token_ordinals=ordinals,
            step=7,
            layer_seed=11,
        )
        actual = planner._apply_action_routes(
            selected,
            layout,
            row,
            baseline,
            source_ranks=sources,
            token_ordinals=ordinals,
            step=7,
            layer_seed=11,
            num_experts=4,
        )
        torch.testing.assert_close(actual, expected)


def test_empty_slots_are_filled_before_swap_and_never_increase_communication():
    layout = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.long)
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    routes = torch.zeros((20, 1), dtype=torch.long)
    sources = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

    plan = _planner().plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        max_swaps=1,
        max_replicas=4,
        step=0,
    )

    assert plan.algorithm_version == GREEDY_COVER_ALGORITHM_VERSION
    assert len(plan.actions) == 4
    assert all(action.kind == "replica" and action.dst_logical == -1 for action in plan.actions)
    assert -1 not in plan.final_layout
    assert plan.swap_rounds == 0
    assert plan.replica_rounds == 4
    assert plan.final_cost.communication <= plan.baseline_cost.communication
    assert plan.local_physical_routes is not None
    _assert_routes_match_layout(routes, plan.local_physical_routes, plan.final_layout)


def test_occupied_cover_uses_add_evict_interaction_and_installs_exact_winner_routes():
    layout, owners = _full_layout()
    routes = torch.zeros((20, 1), dtype=torch.long)
    sources = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
    planner = _planner()

    plan = planner.plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        max_swaps=0,
        max_replicas=1,
        step=3,
    )

    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert (action.kind, action.src_logical, action.dst_slot, action.dst_logical) == ("replica", 0, 3, 2)
    assert plan.final_cost.communication < plan.baseline_cost.communication
    assert plan.local_physical_routes is not None
    _assert_routes_match_layout(routes, plan.local_physical_routes, plan.final_layout)
    assert torch.equal(plan.local_physical_routes[:10], torch.zeros((10, 1), dtype=torch.long))
    assert torch.equal(plan.local_physical_routes[10:], torch.full((10, 1), 3, dtype=torch.long))

    exact_routes = assign_tokens_to_copies_greedy(
        routes,
        torch.tensor(plan.final_layout),
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        step=3,
    )
    torch.testing.assert_close(plan.local_physical_routes, exact_routes)
    packed = planner._local_packed_counts(exact_routes)
    communication, _peak, _dim = planner._communication_cost(packed)
    assert plan.final_cost.communication == pytest.approx(float(communication.item()))


def test_swap_and_cover_are_compared_in_one_round_and_cover_wins():
    layout, owners = _full_layout()
    routes = torch.zeros((20, 1), dtype=torch.long)
    sources = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

    plan = _planner().plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        max_swaps=1,
        max_replicas=1,
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "replica"
    assert plan.actions[0].dst_slot == 3


def test_full_layout_rejects_non_improving_swap_and_cover():
    layout, owners = _full_layout()
    routes = torch.arange(4, dtype=torch.long).view(-1, 1)
    sources = torch.arange(4, dtype=torch.long)

    plan = _planner().plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        max_swaps=1,
        max_replicas=1,
    )

    assert plan.actions == ()
    assert plan.final_layout == tuple(layout.tolist())
    assert plan.final_cost.communication == plan.baseline_cost.communication


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"expert_swap_mode": "step", "redundant_slot_increment_per_device": 1}, "requires expert_swap_mode=layer"),
        ({"expert_swap_mode": "layer", "redundant_slot_increment_per_device": 0}, "requires redundant slots"),
        (
            {
                "expert_swap_mode": "layer",
                "redundant_slot_increment_per_device": 1,
                "expert_swap_max_pairs_per_layer": 2,
            },
            "at most one",
        ),
    ),
)
def test_greedy_cover_selector_rejects_incompatible_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HierMoEConfig(expert_swap_selector="hiermoe_greedy_cover_p1", **kwargs)


def test_greedy_cover_selector_accepts_layer_mode_with_redundant_slots():
    config = HierMoEConfig(
        expert_swap_selector="hiermoe_greedy_cover_p1",
        expert_swap_mode="layer",
        redundant_slot_increment_per_device=1,
    )

    assert config.expert_swap_selector == "hiermoe_greedy_cover_p1"
