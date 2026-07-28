# Copyright 2026 Bytedance Ltd. and/or its affiliates

import numpy as np

from scripts.profile.build_hiermoe_recursive_classifier_layout import (
    _logical_instances,
    _materialize_layout,
    _partition_proxy_cost,
    _refine_balanced_partition,
    _replica_sets_from_partition,
)


def test_replica_group_enumeration_supports_all_ep32_capacities() -> None:
    labels = np.repeat(np.arange(4, dtype=np.int64), 32)
    expected = {0: 1, 32: 4, 64: 6, 96: 4, 128: 1}
    for replicas, combinations in expected.items():
        results = _replica_sets_from_partition(labels, replicas=replicas)
        assert len(results) == combinations
        assert all(len(row) == replicas for row in results)
        assert all(len(np.unique(row)) == replicas for row in results)


def test_materialized_layout_has_balanced_owners_and_valid_lut() -> None:
    num_experts = 128
    ep_size = 32
    slots_per_rank = 8
    logical_instances = _logical_instances(
        num_experts,
        np.arange(num_experts, dtype=np.int64),
    )
    instance_ranks = np.concatenate(
        [
            np.repeat(np.arange(ep_size, dtype=np.int64), 4),
            np.repeat(np.arange(ep_size, dtype=np.int64), 4),
        ]
    )
    lut_instances = np.empty((ep_size, num_experts), dtype=np.int64)
    for source_rank in range(ep_size):
        lut_instances[source_rank] = np.arange(num_experts, dtype=np.int64) + (
            num_experts if source_rank >= ep_size // 2 else 0
        )
    demand = np.ones((ep_size, num_experts), dtype=np.float64)

    layout, owners, lut = _materialize_layout(
        logical_instances,
        instance_ranks,
        lut_instances,
        demand,
        ep_size=ep_size,
        slots_per_rank=slots_per_rank,
        primary_slots_per_rank=4,
        num_experts=num_experts,
    )

    assert np.all(layout >= 0)
    assert len(np.unique(owners)) == num_experts
    assert np.all(owners % slots_per_rank < 4)
    assert np.all(layout[lut] == np.arange(num_experts, dtype=np.int64)[None, :])


def test_unified_partition_refinement_preserves_capacity_and_reduces_cost() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    demand = np.asarray([10.0, 1.0, 10.0, 1.0])
    affinity = np.zeros((4, 4), dtype=np.float64)
    affinity[0, 2] = affinity[2, 0] = 10.0
    affinity[1, 3] = affinity[3, 1] = 10.0
    initial_cost, _, _ = _partition_proxy_cost(
        affinity,
        demand,
        labels,
        parts=2,
        affinity_ms_per_hit=1.0,
        assignment_ms_per_assignment=0.1,
    )

    result = _refine_balanced_partition(
        affinity,
        demand,
        labels,
        parts=2,
        capacity=2,
        affinity_ms_per_hit=1.0,
        assignment_ms_per_assignment=0.1,
        max_swaps=4,
    )

    assert np.all(np.bincount(result.labels, minlength=2) == 2)
    assert result.proxy_cost < initial_cost
    assert len(result.swaps) == 1


def test_unified_rank_refinement_rejects_duplicate_expert_copies() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    logical = np.asarray([0, 0, 1, 1], dtype=np.int64)
    demand = np.ones((4,), dtype=np.float64)
    affinity = np.zeros((4, 4), dtype=np.float64)
    affinity[0, 1] = affinity[1, 0] = 100.0

    result = _refine_balanced_partition(
        affinity,
        demand,
        labels,
        parts=2,
        capacity=2,
        affinity_ms_per_hit=1.0,
        assignment_ms_per_assignment=0.0,
        max_swaps=4,
        item_kinds=logical,
        forbid_duplicate_kinds=True,
    )

    for part in range(2):
        members = logical[result.labels == part]
        assert len(np.unique(members)) == len(members)
