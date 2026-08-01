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

import numpy as np
import pytest
import torch

from veomni.distributed.moe.hiermoe.placemoe import (
    LayerPlan,
    PlaceMoETopology,
    bounded_group_shortlist,
    build_replica_allocations,
    profile_route_statistics,
    project_statistics_to_copies,
    uniform_copy_statistics,
)


def _profile():
    return profile_route_statistics(
        [
            [
                torch.tensor([[0, 1], [0, 2]]),
                torch.tensor([[2, 3], [1, 3]]),
            ]
        ],
        num_experts=4,
    )


def test_profile_route_statistics_preserves_source_demand_and_affinity():
    statistics = _profile()
    np.testing.assert_array_equal(statistics.demand, [[2, 1, 1, 0], [0, 1, 1, 2]])
    assert statistics.affinity[0, 0, 1] == 1
    assert statistics.affinity[0, 0, 2] == 1
    assert statistics.affinity[1, 1, 3] == 1
    assert statistics.affinity[1, 2, 3] == 1
    np.testing.assert_array_equal(statistics.affinity, statistics.affinity.transpose(0, 2, 1))


def test_uniform_and_mapped_copy_statistics_follow_paper_definitions():
    statistics = _profile()
    copies = np.array([0, 1, 2, 3, 0, 3])
    uniform_demand, uniform_affinity = uniform_copy_statistics(statistics, copies)
    np.testing.assert_array_equal(uniform_demand[:, [0, 4]], [[1, 1], [0, 0]])
    assert uniform_affinity[0, 0, 1] == pytest.approx(0.5)

    mapping = np.array([[0, 1, 2, 3], [4, 1, 2, 5]])
    mapped_demand, mapped_affinity = project_statistics_to_copies(statistics, copies, mapping)
    np.testing.assert_array_equal(mapped_demand[0], [2, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(mapped_demand[1], [0, 1, 1, 0, 0, 2])
    assert mapped_affinity[0, 0, 1] == 1
    assert mapped_affinity[1, 1, 5] == 1


def test_bounded_replica_allocation_uses_exact_budget_and_deterministic_order():
    groups = [np.array([0, 1]), np.array([2, 3])]
    shortlist = bounded_group_shortlist(
        groups,
        np.array([9.0, 8.0, 2.0, 1.0]),
        selected_groups=1,
        candidate_limit=2,
    )
    assert shortlist == [(0,), (1,)]
    allocations = build_replica_allocations(
        [np.array([0, 0, 1, 1])],
        np.array([9.0, 8.0, 2.0, 1.0]),
        additional_copies=2,
        candidate_limit=2,
    )
    assert [allocation.tolist() for allocation in allocations] == [[0, 1], [2, 3]]


def test_layer_plan_validates_budget_mapping_and_runtime_round_trip():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=4, slots_per_rank=3)
    plan = LayerPlan(
        slot_to_logical=[0, 1, 0, 2, 3, 3],
        owner_slots=[0, 1, 3, 4],
        source_logical_to_physical=[[0, 1, 3, 4], [2, 1, 3, 5]],
    )
    plan.validate(topology, additional_copies=2)
    restored = LayerPlan.from_runtime_payload(plan.to_runtime_payload())
    restored.validate(topology, additional_copies=2)
    np.testing.assert_array_equal(restored.slot_to_logical, plan.slot_to_logical)


def test_layer_plan_rejects_mapping_to_wrong_expert():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=4, slots_per_rank=2)
    plan = LayerPlan(
        slot_to_logical=[0, 1, 2, 3],
        owner_slots=[0, 1, 2, 3],
        source_logical_to_physical=[[1, 1, 2, 3], [0, 1, 2, 3]],
    )
    with pytest.raises(ValueError, match="wrong logical expert"):
        plan.validate(topology, additional_copies=0)
