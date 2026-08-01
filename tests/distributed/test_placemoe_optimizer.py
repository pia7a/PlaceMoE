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
    MappingConfig,
    PartitionConfig,
    PlaceMoETopology,
    ProfileStatistics,
    bounded_group_shortlist,
    build_replica_allocations,
    initialize_mapping,
    map_groups_to_locations,
    optimize_mapping,
    partition_items,
    partition_objective,
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


def test_partition_respects_capacity_and_refines_the_calibrated_objective():
    affinity = np.array(
        [
            [0, 8, 1, 0],
            [8, 0, 0, 1],
            [1, 0, 0, 8],
            [0, 1, 8, 0],
        ],
        dtype=np.float64,
    )
    demand = np.array([9, 1, 9, 1], dtype=np.float64)
    config = PartitionConfig(
        capacities=(2, 2),
        ranks_per_group=(1, 1),
        omega=1.0,
        gamma=2.0,
        restarts=3,
        seed=17,
    )
    results = partition_items(affinity, demand, config)
    assert results
    for result in results:
        np.testing.assert_array_equal(np.bincount(result.labels, minlength=2), [2, 2])
        objective, within, peak = partition_objective(affinity, demand, result.labels, config)
        assert objective == pytest.approx(result.objective)
        assert within == pytest.approx(result.within_affinity)
        assert peak == pytest.approx(result.peak_assignments_per_rank)


def test_calibrated_compute_cost_changes_the_preferred_partition():
    affinity = np.array(
        [
            [0, 20, 0, 0],
            [20, 0, 0, 0],
            [0, 0, 0, 20],
            [0, 0, 20, 0],
        ],
        dtype=np.float64,
    )
    demand = np.array([10, 10, 1, 1], dtype=np.float64)
    affinity_labels = np.array([0, 0, 1, 1])
    balanced_labels = np.array([0, 1, 0, 1])
    communication_only = PartitionConfig((2, 2), (1, 1), omega=1.0, gamma=0.0)
    compute_aware = PartitionConfig((2, 2), (1, 1), omega=1.0, gamma=5.0)
    assert (
        partition_objective(affinity, demand, affinity_labels, communication_only)[0]
        < partition_objective(affinity, demand, balanced_labels, communication_only)[0]
    )
    assert (
        partition_objective(affinity, demand, balanced_labels, compute_aware)[0]
        < partition_objective(affinity, demand, affinity_labels, compute_aware)[0]
    )
    communication_result = partition_items(
        affinity,
        demand,
        PartitionConfig((2, 2), (1, 1), omega=1.0, gamma=0.0, restarts=3, seed=17),
    )[0]
    compute_result = partition_items(
        affinity,
        demand,
        PartitionConfig((2, 2), (1, 1), omega=1.0, gamma=5.0, restarts=3, seed=17),
    )[0]
    assert communication_result.within_affinity == 40
    assert communication_result.peak_assignments_per_rank == 20
    assert compute_result.within_affinity == 0
    assert compute_result.peak_assignments_per_rank == 11


def test_locality_matching_maps_abstract_groups_to_source_demand():
    labels = np.array([0, 0, 1, 1])
    demand_by_source = np.array(
        [
            [0, 0, 5, 4],
            [0, 0, 3, 2],
            [7, 6, 0, 0],
            [5, 4, 0, 0],
        ],
        dtype=np.float64,
    )
    locations = map_groups_to_locations(
        labels,
        demand_by_source,
        sources_by_location=(np.array([0, 1]), np.array([2, 3])),
    )
    np.testing.assert_array_equal(locations, [1, 1, 0, 0])


def test_mapping_initialization_prefers_local_copies_and_balances_rank_loads():
    statistics = ProfileStatistics(
        demand=np.array([[5, 1], [1, 5]], dtype=np.float64),
        affinity=np.zeros((2, 2, 2), dtype=np.float64),
    )
    logical_instances = np.array([0, 1, 0, 1])
    instance_ranks = np.array([0, 0, 1, 1])
    mapping = initialize_mapping(
        logical_instances,
        instance_ranks,
        statistics.demand,
        ranks_per_node=1,
    )
    np.testing.assert_array_equal(mapping, [[0, 1], [2, 3]])


def test_calibrated_mapping_score_balances_affinity_reuse_and_compute_load():
    affinity = np.zeros((2, 2, 2), dtype=np.float64)
    affinity[0, 0, 1] = affinity[0, 1, 0] = 30
    statistics = ProfileStatistics(
        demand=np.array([[10, 1], [0, 10]], dtype=np.float64),
        affinity=affinity,
    )
    logical_instances = np.array([0, 0, 1])
    instance_ranks = np.array([0, 1, 1])
    initial = initialize_mapping(
        logical_instances,
        instance_ranks,
        statistics.demand,
        ranks_per_node=1,
    )
    communication_only = optimize_mapping(
        logical_instances,
        instance_ranks,
        initial,
        statistics,
        MappingConfig(ranks_per_node=1, node_omega=1.0, rank_omega=0.0, gamma=0.0),
    )
    compute_aware = optimize_mapping(
        logical_instances,
        instance_ranks,
        initial,
        statistics,
        MappingConfig(ranks_per_node=1, node_omega=1.0, rank_omega=0.0, gamma=3.0),
    )
    assert communication_only.mapping[0, 0] == 1
    assert compute_aware.mapping[0, 0] == 0
    assert communication_only.peak_rank_load == 21
    assert compute_aware.peak_rank_load == 11
