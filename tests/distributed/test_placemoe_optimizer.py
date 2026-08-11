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
    CommunityMappingConfig,
    LayerPlan,
    MappingConfig,
    OptimizerConfig,
    PartitionConfig,
    PlacementConfig,
    PlaceMoETopology,
    ProfileStatistics,
    bounded_group_shortlist,
    build_placemoe_artifact,
    build_replica_allocations,
    community_intersection_hits,
    community_node_placements,
    initialize_mapping,
    map_groups_to_locations,
    materialize_plan,
    mirrored_r2_plan,
    optimize_community_mapping,
    optimize_fixed_layout_mapping,
    optimize_mapping,
    optimize_mapping_normalized,
    optimize_replica_allocation,
    partition_items,
    partition_objective,
    place_instances,
    profile_route_statistics,
    project_statistics_to_copies,
    repair_rank_placement,
    uniform_copy_statistics,
    validate_placemoe_artifact,
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


def test_normalized_seed_can_be_retained_as_an_exact_cost_candidate():
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
    common = dict(
        capacities=(2, 2),
        ranks_per_group=(1, 1),
        omega=1.0,
        gamma=5.0,
        restarts=1,
        seed=17,
        seed_load_weight=0.0,
    )
    calibrated = partition_items(affinity, demand, PartitionConfig(**common))[0]
    normalized = partition_items(
        affinity,
        demand,
        PartitionConfig(**common, calibrated_refinement=False),
    )[0]

    assert calibrated.peak_assignments_per_rank == 11
    assert normalized.within_affinity == 40
    assert normalized.peak_assignments_per_rank == 20


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


@pytest.mark.parametrize("num_nodes", [4, 8])
def test_community_node_placements_are_balanced_and_topology_general(num_nodes):
    num_experts = 2 * num_nodes
    logical_instances = np.tile(np.arange(num_experts, dtype=np.int64), 2)
    communities = np.repeat(np.arange(num_nodes, dtype=np.int64), 2)
    config = PlacementConfig(
        ep_size=2 * num_nodes,
        ranks_per_node=2,
        slots_per_rank=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
    )
    demand = np.ones((config.ep_size, num_experts), dtype=np.float64)
    candidates = community_node_placements(
        logical_instances,
        communities,
        demand,
        config,
        candidate_limit=2,
    )
    assert candidates
    for instance_nodes in candidates:
        np.testing.assert_array_equal(
            np.bincount(instance_nodes, minlength=num_nodes),
            np.full((num_nodes,), 4),
        )
        for expert in range(num_experts):
            expert_nodes = instance_nodes[logical_instances == expert]
            assert len(expert_nodes) == len(np.unique(expert_nodes))
        for community in range(num_nodes):
            experts = np.flatnonzero(communities == community)
            footprints = [set(instance_nodes[logical_instances == expert].tolist()) for expert in experts]
            assert all(footprint == footprints[0] for footprint in footprints[1:])


def test_community_node_placements_support_partial_replica_budgets():
    logical_instances = np.concatenate(
        (
            np.arange(8, dtype=np.int64),
            np.arange(4, dtype=np.int64),
            np.full((4,), -1, dtype=np.int64),
        )
    )
    communities = np.repeat(np.arange(4, dtype=np.int64), 2)
    config = PlacementConfig(
        ep_size=8,
        ranks_per_node=2,
        slots_per_rank=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
    )
    candidates = community_node_placements(
        logical_instances,
        communities,
        np.ones((config.ep_size, 8), dtype=np.float64),
        config,
        candidate_limit=2,
    )
    assert candidates
    for instance_nodes in candidates:
        np.testing.assert_array_equal(np.bincount(instance_nodes, minlength=4), np.full((4,), 4))
        for expert in range(4):
            expert_nodes = instance_nodes[logical_instances == expert]
            assert len(expert_nodes) == len(np.unique(expert_nodes))


def test_community_mapping_moves_co_selected_experts_as_one_block():
    logical_instances = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    instance_nodes = np.array([0, 1, 0, 1, 1, 2, 1, 2])
    communities = np.array([0, 0, 1, 1])
    assignments = np.zeros((4, 2), dtype=np.float64)
    assignments[3] = [20, 20]
    masks = np.zeros((4, 4), dtype=np.int64)
    masks[3, 3] = 10
    result = optimize_community_mapping(
        logical_instances,
        instance_nodes,
        communities,
        assignments,
        masks,
        CommunityMappingConfig(
            ranks_per_node=1,
            communication_ms_per_token=1.0,
            assignment_ms_per_assignment=0.0,
        ),
    )
    np.testing.assert_array_equal(result.destination_nodes[3], [1, 1])
    mapped_nodes = instance_nodes[result.mapping[3]]
    np.testing.assert_array_equal(mapped_nodes, [1, 1, 1, 1])
    assert result.proxy_cost == 10


def test_community_intersection_hits_matches_mask_unions():
    histogram = np.array([[0, 2, 3, 5]], dtype=np.int64)
    np.testing.assert_array_equal(
        community_intersection_hits(histogram),
        [[0, 7, 8, 10]],
    )


def test_community_mapping_bounds_large_source_row_search():
    num_communities = 13
    logical_instances = np.repeat(np.arange(num_communities, dtype=np.int64), 2)
    instance_nodes = np.tile(np.arange(2, dtype=np.int64), num_communities)
    communities = np.arange(num_communities, dtype=np.int64)
    assignments = np.ones((2, num_communities), dtype=np.float64)
    masks = np.zeros((2, 1 << num_communities), dtype=np.int64)
    masks[:, -1] = 1
    result = optimize_community_mapping(
        logical_instances,
        instance_nodes,
        communities,
        assignments,
        masks,
        CommunityMappingConfig(
            ranks_per_node=1,
            communication_ms_per_token=1.0,
            assignment_ms_per_assignment=0.0,
            row_candidate_limit=4,
            beam_width=4,
        ),
    )
    assert result.mapping.shape == (2, num_communities)
    assert result.destination_nodes.shape == (2, num_communities)


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


def test_normalized_mapping_proposals_span_communication_and_compute_tradeoffs():
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

    communication_only = optimize_mapping_normalized(
        logical_instances,
        instance_ranks,
        initial,
        statistics,
        ranks_per_node=1,
        assignment_weight=0.0,
    )
    compute_aware = optimize_mapping_normalized(
        logical_instances,
        instance_ranks,
        initial,
        statistics,
        ranks_per_node=1,
        assignment_weight=100.0,
    )

    assert communication_only.mapping[0, 0] == 1
    assert compute_aware.mapping[0, 0] == 0
    assert communication_only.peak_rank_load == 21
    assert compute_aware.peak_rank_load == 11


def test_rank_repair_separates_copies_without_changing_node_membership():
    config = PlacementConfig(
        ep_size=2,
        ranks_per_node=2,
        slots_per_rank=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
    )
    logical_instances = np.array([0, 0, 1, 1])
    repaired = repair_rank_placement(
        np.zeros((4,), dtype=np.int64),
        np.array([5, 4, 3, 2], dtype=np.float64),
        np.zeros((4, 4), dtype=np.float64),
        logical_instances,
        config,
    )
    np.testing.assert_array_equal(np.bincount(repaired, minlength=2), [2, 2])
    for rank in range(2):
        assert len(np.unique(logical_instances[repaired == rank])) == 2


def test_hierarchical_placement_respects_rank_capacity_and_copy_uniqueness():
    config = PlacementConfig(
        ep_size=4,
        ranks_per_node=2,
        slots_per_rank=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
        node_exchange_limit=4,
        rank_exchange_limit=2,
        seed=7,
    )
    logical_instances = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    demand = np.ones((4, 8), dtype=np.float64)
    affinity = np.zeros((4, 8, 8), dtype=np.float64)
    result = place_instances(demand, affinity, logical_instances, config)
    np.testing.assert_array_equal(np.bincount(result.instance_ranks, minlength=4), [2, 2, 2, 2])
    for rank in range(4):
        members = logical_instances[result.instance_ranks == rank]
        assert len(members) == len(np.unique(members))


def test_explicit_two_level_hierarchy_is_bitwise_legacy_compatible():
    rng = np.random.default_rng(20260806)
    logical_instances = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    copy_demand = rng.integers(0, 11, size=(4, 8)).astype(np.float64)
    raw_copy_affinity = rng.integers(0, 7, size=(4, 8, 8)).astype(np.float64)
    copy_affinity = raw_copy_affinity + raw_copy_affinity.transpose(0, 2, 1)
    for matrix in copy_affinity:
        np.fill_diagonal(matrix, 0.0)
    common = dict(
        ep_size=4,
        ranks_per_node=2,
        slots_per_rank=2,
        node_omega=1.25,
        rank_omega=0.2,
        gamma=0.7,
        node_exchange_limit=4,
        rank_exchange_limit=2,
        seed=23,
    )

    legacy_placement = place_instances(
        copy_demand,
        copy_affinity,
        logical_instances,
        PlacementConfig(**common),
    )
    explicit_placement = place_instances(
        copy_demand,
        copy_affinity,
        logical_instances,
        PlacementConfig(
            **common,
            hierarchy_group_sizes=(2, 4),
            level_omegas=(1.25, 0.2),
        ),
    )

    np.testing.assert_array_equal(explicit_placement.instance_ranks, legacy_placement.instance_ranks)
    assert explicit_placement.repaired == legacy_placement.repaired
    assert explicit_placement.node_objective == legacy_placement.node_objective
    assert explicit_placement.rank_objective == legacy_placement.rank_objective
    assert explicit_placement.level_objectives == legacy_placement.level_objectives

    demand = rng.integers(0, 13, size=(4, 4)).astype(np.float64)
    raw_affinity = rng.integers(0, 9, size=(4, 4, 4)).astype(np.float64)
    affinity = raw_affinity + raw_affinity.transpose(0, 2, 1)
    for matrix in affinity:
        np.fill_diagonal(matrix, 0.0)
    statistics = ProfileStatistics(demand=demand, affinity=affinity)
    legacy_initial = initialize_mapping(
        logical_instances,
        legacy_placement.instance_ranks,
        demand,
        ranks_per_node=2,
    )
    explicit_initial = initialize_mapping(
        logical_instances,
        legacy_placement.instance_ranks,
        demand,
        ranks_per_node=2,
        hierarchy_group_sizes=(2, 4),
    )
    np.testing.assert_array_equal(explicit_initial, legacy_initial)

    legacy_mapping = optimize_mapping(
        logical_instances,
        legacy_placement.instance_ranks,
        legacy_initial,
        statistics,
        MappingConfig(ranks_per_node=2, node_omega=1.25, rank_omega=0.2, gamma=0.7, sweep_limit=4),
    )
    explicit_mapping = optimize_mapping(
        logical_instances,
        legacy_placement.instance_ranks,
        explicit_initial,
        statistics,
        MappingConfig(
            ranks_per_node=2,
            node_omega=1.25,
            rank_omega=0.2,
            gamma=0.7,
            sweep_limit=4,
            hierarchy_group_sizes=(2, 4),
            level_omegas=(1.25, 0.2),
        ),
    )
    np.testing.assert_array_equal(explicit_mapping.mapping, legacy_mapping.mapping)
    assert explicit_mapping.sweeps == legacy_mapping.sweeps
    assert explicit_mapping.changes == legacy_mapping.changes
    assert explicit_mapping.peak_rank_load == legacy_mapping.peak_rank_load


def test_three_level_placement_and_mapping_preserve_capacity_and_copy_identity():
    config = PlacementConfig(
        ep_size=8,
        ranks_per_node=4,
        slots_per_rank=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
        node_exchange_limit=4,
        rank_exchange_limit=2,
        seed=19,
        hierarchy_group_sizes=(2, 4, 8),
        level_omegas=(1.0, 0.3, 0.1),
    )
    logical_instances = np.tile(np.arange(8, dtype=np.int64), 2)
    copy_demand = np.ones((8, 16), dtype=np.float64)
    copy_affinity = np.zeros((8, 16, 16), dtype=np.float64)

    placement = place_instances(copy_demand, copy_affinity, logical_instances, config)

    assert len(placement.level_objectives) == 3
    np.testing.assert_array_equal(
        np.bincount(placement.instance_ranks, minlength=8),
        np.full((8,), 2),
    )
    for rank in range(8):
        members = logical_instances[placement.instance_ranks == rank]
        assert len(members) == len(np.unique(members))

    statistics = ProfileStatistics(
        demand=np.ones((8, 8), dtype=np.float64),
        affinity=np.zeros((8, 8, 8), dtype=np.float64),
    )
    initial = initialize_mapping(
        logical_instances,
        placement.instance_ranks,
        statistics.demand,
        ranks_per_node=4,
        hierarchy_group_sizes=(2, 4, 8),
    )
    refined = optimize_mapping(
        logical_instances,
        placement.instance_ranks,
        initial,
        statistics,
        MappingConfig(
            ranks_per_node=4,
            node_omega=1.0,
            rank_omega=0.1,
            gamma=1.0,
            hierarchy_group_sizes=(2, 4, 8),
            level_omegas=(1.0, 0.3, 0.1),
        ),
    )
    np.testing.assert_array_equal(
        logical_instances[refined.mapping],
        np.broadcast_to(np.arange(8), (8, 8)),
    )


def test_materialize_plan_rewrites_mapping_to_relocated_slots():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=4, slots_per_rank=3)
    statistics = ProfileStatistics(
        demand=np.array([[5, 4, 1, 1], [1, 1, 5, 4]], dtype=np.float64),
        affinity=np.zeros((2, 4, 4), dtype=np.float64),
    )
    logical_instances = np.array([0, 1, 2, 3, 0, 3])
    instance_ranks = np.array([1, 0, 0, 1, 0, 1])
    instance_mapping = np.array([[4, 1, 2, 5], [0, 1, 2, 3]])
    plan = materialize_plan(
        logical_instances,
        instance_ranks,
        instance_mapping,
        statistics.demand,
        topology,
        primary_slots_per_rank=2,
    )
    plan.validate(topology, additional_copies=2)
    np.testing.assert_array_equal(
        plan.slot_to_logical[plan.source_logical_to_physical],
        np.broadcast_to(np.arange(4), (2, 4)),
    )


def test_optimizer_alternates_layout_and_mapping_with_exact_evaluation_callback():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=4, slots_per_rank=3)
    statistics = _profile()
    logical_instances = np.array([0, 1, 2, 3, 0, 3])
    evaluated: list[np.ndarray] = []

    def evaluate(plan: LayerPlan) -> float:
        evaluated.append(plan.source_logical_to_physical.copy())
        destination_ranks = plan.source_logical_to_physical // topology.slots_per_rank
        loads = np.zeros((topology.ep_size,), dtype=np.float64)
        np.add.at(loads, destination_ranks, statistics.demand)
        return float(loads.max())

    result = optimize_replica_allocation(
        statistics,
        logical_instances,
        OptimizerConfig(
            topology=topology,
            primary_slots_per_rank=2,
            node_omega=1.0,
            rank_omega=0.1,
            gamma=1.0,
            rounds=2,
            node_exchange_limit=4,
            rank_exchange_limit=2,
            mapping_sweep_limit=2,
            seed=11,
        ),
        evaluate,
    )
    assert len(result.candidates) >= 2
    assert len(evaluated) == len(result.candidates)
    assert result.best.cost == min(candidate.cost for candidate in result.candidates)
    for candidate in result.candidates:
        candidate.plan.validate(topology, additional_copies=2)


def test_fixed_layout_mapping_preserves_layout_and_keeps_exact_cost_incumbent():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=2, slots_per_rank=2)
    statistics = ProfileStatistics(
        demand=np.array([[8, 2], [2, 8]], dtype=np.float64),
        affinity=np.zeros((2, 2, 2), dtype=np.float64),
    )
    current = LayerPlan(
        slot_to_logical=[0, 1, 0, 1],
        owner_slots=[0, 1],
        source_logical_to_physical=[[2, 3], [0, 1]],
    )

    def evaluate(plan: LayerPlan) -> float:
        destination_ranks = plan.source_logical_to_physical // topology.slots_per_rank
        return float((destination_ranks != np.arange(topology.ep_size)[:, None]).sum())

    result = optimize_fixed_layout_mapping(
        statistics,
        current,
        OptimizerConfig(
            topology=topology,
            primary_slots_per_rank=1,
            node_omega=1.0,
            rank_omega=0.1,
            gamma=1.0,
            mapping_sweep_limit=2,
        ),
        evaluate,
    )

    assert result.best.cost <= evaluate(current)
    np.testing.assert_array_equal(result.best.plan.slot_to_logical, current.slot_to_logical)
    np.testing.assert_array_equal(result.best.plan.owner_slots, current.owner_slots)
    result.best.plan.validate(topology, additional_copies=2)


def test_mirrored_r2_seed_is_a_valid_default_order_plan():
    topology = PlaceMoETopology(ep_size=4, ranks_per_node=2, num_experts=8, slots_per_rank=4)

    plan = mirrored_r2_plan(topology)

    plan.validate(topology, additional_copies=8)
    np.testing.assert_array_equal(plan.slot_to_logical[:8], np.arange(8))
    np.testing.assert_array_equal(plan.slot_to_logical[8:], np.arange(8))
    np.testing.assert_array_equal(plan.source_logical_to_physical[:2], np.tile(np.arange(8), (2, 1)))
    np.testing.assert_array_equal(plan.source_logical_to_physical[2:], np.tile(np.arange(8, 16), (2, 1)))


def test_artifact_schema_round_trip_validates_every_layer_plan():
    topology = PlaceMoETopology(ep_size=2, ranks_per_node=1, num_experts=4, slots_per_rank=3)
    plan = LayerPlan(
        slot_to_logical=[0, 1, 0, 2, 3, 3],
        owner_slots=[0, 1, 3, 4],
        source_logical_to_physical=[[0, 1, 3, 4], [2, 1, 3, 5]],
    )
    payload = build_placemoe_artifact({"layers.0.experts": plan}, topology, source={"route_root": "/tmp/routes"})
    restored = validate_placemoe_artifact(payload)
    np.testing.assert_array_equal(restored["layers.0.experts"].slot_to_logical, plan.slot_to_logical)
    payload["layers"]["layers.0.experts"]["source_logical_to_physical"][0][0] = 1
    with pytest.raises(ValueError, match="wrong logical expert"):
        validate_placemoe_artifact(payload)
