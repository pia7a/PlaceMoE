# Copyright 2026 Bytedance Ltd. and/or its affiliates

from argparse import Namespace

import numpy as np
import pytest
import torch

from scripts.profile.build_hiermoe_hierarchical_init_layout import HybridCost, _load_routes, _route_statistics
from scripts.profile.build_hiermoe_recursive_classifier_layout import (
    _build_candidate,
    _classify_instances,
    _group_route_statistics,
    _initial_lut_instances,
    _is_e2e_eligible,
    _logical_instances,
    _materialize_layout,
    _partition_proxy_cost,
    _preloaded_replay_payload,
    _rank_assignment_is_device_unique,
    _refine_balanced_partition,
    _replica_allocations,
    _replica_sets_from_partition,
    _source_statistics,
    _structured_instance_node_candidates,
    _uniform_instance_statistics,
    _validate_configuration,
)
from veomni.distributed.moe.hiermoe.placemoe import LayerPlan, PlaceMoETopology


def test_compute_only_initial_lut_does_not_prefer_source_locality() -> None:
    logical_instances = np.asarray([0, 0], dtype=np.int64)
    instance_ranks = np.asarray([0, 2], dtype=np.int64)
    demand = np.asarray([[100.0], [90.0], [0.0], [0.0]], dtype=np.float64)

    local = _initial_lut_instances(
        logical_instances,
        instance_ranks,
        demand,
        ranks_per_node=2,
        prefer_local=True,
    )
    compute_only = _initial_lut_instances(
        logical_instances,
        instance_ranks,
        demand,
        ranks_per_node=2,
        prefer_local=False,
    )

    assert int(local[1, 0]) == 0
    assert int(compute_only[1, 0]) == 1


def test_generic_builder_uses_canonical_placemoe_optimizer() -> None:
    class Evaluator:
        def evaluate(self, _samples, lut):
            peak = float(np.bincount(lut.reshape(-1) // 2, minlength=4).max())
            return HybridCost(0.0, peak, peak, 0, 0, 0.0, 0.0, peak)

    args = Namespace(
        communication_blind_proposals=False,
        hidden_size=8,
        bytes_per_element=2,
        communication_phase_multiplier=1.0,
        inter_ms_per_byte=0.1,
        intra_ms_per_byte=0.01,
        compute_phase_multiplier=1.0,
        compute_ms_per_assignment=0.1,
        ep_size=4,
        ranks_per_node=2,
        num_experts=4,
        slots_per_rank=2,
        primary_slots_per_rank=1,
        alternations=2,
        partition_iterations=4,
        lut_iterations=2,
    )
    logical_instances = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    demand = np.array(
        [
            [5, 4, 1, 1],
            [4, 5, 1, 1],
            [1, 1, 5, 4],
            [1, 1, 4, 5],
        ],
        dtype=np.float64,
    )
    affinity = np.zeros((4, 4, 4), dtype=np.float64)
    candidate = _build_candidate(
        [],
        logical_instances=logical_instances,
        demand_by_source=demand,
        affinity_by_source=affinity,
        evaluator=Evaluator(),
        args=args,
        seed=13,
        strategy="placemoe_test",
    )
    assert candidate is not None
    plan = LayerPlan(candidate.layout, candidate.lut, candidate.owners)
    plan.validate(PlaceMoETopology(4, 2, 4, 2), additional_copies=4)
    assert candidate.alternations in (1, 2)


def test_load_routes_accepts_one_all_rank_bundle(tmp_path) -> None:
    capture_dir = tmp_path / "step0000"
    capture_dir.mkdir()
    expected = [
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        torch.tensor([[3, 2]], dtype=torch.int32),
    ]
    torch.save(
        {
            "format": "hiermoe-local-route-bundle-v1",
            "ep_size": 2,
            "routes_by_rank": expected,
        },
        capture_dir / "layer00_call0_all_ranks.pt",
    )

    samples = _load_routes(tmp_path, steps=(0,), layer=0, ep_size=2)

    assert len(samples) == 1
    assert all(torch.equal(actual, wanted.to(torch.long)) for actual, wanted in zip(samples[0], expected, strict=True))


def test_preloaded_payload_preserves_explicit_runtime_layer_keys(tmp_path) -> None:
    keys = ("model.layers.2.mlp.experts", "model.layers.10.mlp.experts")
    args = Namespace(
        layer_start=0,
        layer_name_template="unused.{layer}",
        layer_keys=keys,
        ep_size=1,
        ranks_per_node=1,
        num_experts=2,
        slots_per_rank=2,
        route_root=tmp_path,
        optimize_steps=(0,),
        validation_steps=(0,),
    )
    layouts = [np.array([0, 1]), np.array([1, 0])]
    owners = [np.array([0, 1]), np.array([1, 0])]
    luts = [np.array([[0, 1]]), np.array([[1, 0]])]

    payload = _preloaded_replay_payload(
        layouts=layouts,
        owners=owners,
        luts=luts,
        args=args,
        algorithm="placemoe-v1",
    )

    assert tuple(payload["layers"]) == keys


@pytest.mark.parametrize("total_layers", [40, 48])
def test_e2e_eligibility_accepts_complete_model_layer_count(total_layers: int) -> None:
    assert _is_e2e_eligible(
        layer_start=0,
        layers=total_layers,
        expected_total_layers=total_layers,
        validation_total_ms=10.0,
        comparison_validation_ms=11.0,
    )


def test_e2e_eligibility_rejects_partial_or_regressed_layout() -> None:
    assert not _is_e2e_eligible(
        layer_start=0,
        layers=20,
        expected_total_layers=40,
        validation_total_ms=10.0,
        comparison_validation_ms=11.0,
    )
    assert not _is_e2e_eligible(
        layer_start=0,
        layers=40,
        expected_total_layers=40,
        validation_total_ms=12.0,
        comparison_validation_ms=11.0,
    )


def test_group_route_statistics_match_direct_token_incidence() -> None:
    logical_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    samples = [
        [
            torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
            torch.tensor([[0, 1, 3], [2, 3, 2]], dtype=torch.long),
            torch.tensor([[0, 1, 0], [0, 2, 3]], dtype=torch.long),
            torch.tensor([[2, 3, 2], [1, 2, 3]], dtype=torch.long),
        ]
    ]

    assignments, masks = _group_route_statistics(
        samples,
        logical_groups,
        ranks_per_node=2,
    )
    expected_assignments = np.zeros_like(assignments)
    expected_masks = np.zeros_like(masks)
    for source_rank, route in enumerate(samples[0]):
        source_node = source_rank // 2
        for row in route.tolist():
            groups = [int(logical_groups[expert]) for expert in row]
            for group in groups:
                expected_assignments[source_node, group] += 1
            mask = 0
            for group in groups:
                mask |= 1 << group
            expected_masks[source_node, mask] += 1

    assert np.array_equal(assignments, expected_assignments)
    assert np.array_equal(masks, expected_masks)


def test_source_statistics_reconstruct_legacy_global_statistics() -> None:
    samples = [
        [
            torch.tensor(
                [
                    [source_rank, (source_rank + 1) % 8, (source_rank + 3) % 8],
                    [(source_rank + 2) % 8, source_rank, (source_rank + 5) % 8],
                ],
                dtype=torch.long,
            )
            for source_rank in range(4)
        ],
        [
            torch.tensor(
                [
                    [(source_rank + 4) % 8, source_rank, (source_rank + 1) % 8],
                    [(source_rank + 6) % 8, (source_rank + 3) % 8, source_rank],
                ],
                dtype=torch.long,
            )
            for source_rank in range(4)
        ],
    ]

    legacy_demand, _, legacy_affinity, _ = _route_statistics(
        samples,
        num_experts=8,
        ranks_per_node=2,
    )
    source_demand, source_affinity = _source_statistics(
        samples,
        num_experts=8,
    )

    assert np.array_equal(source_demand, legacy_demand)
    assert np.array_equal(source_affinity.sum(axis=0), legacy_affinity)


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


@pytest.mark.parametrize(
    ("num_experts", "ep_size"),
    (
        (128, 16),
        (128, 32),
        (128, 64),
        (256, 16),
        (256, 32),
        (256, 64),
    ),
)
def test_capacity_validation_supports_model_and_ep_matrix(num_experts: int, ep_size: int) -> None:
    primary = num_experts // ep_size
    args = Namespace(
        num_experts=num_experts,
        ep_size=ep_size,
        ranks_per_node=8,
        slots_per_rank=None,
        redundant_slots_per_rank=2,
        primary_slots_per_rank=None,
        active_redundant_slots=ep_size + 1,
    )

    capacity = _validate_configuration(args)

    assert capacity.primary_slots_per_rank == primary
    assert capacity.reserved_replicas == 2 * ep_size
    assert capacity.active_replicas == ep_size + 1
    assert capacity.empty_slots == ep_size - 1
    assert args.primary_slots_per_rank == primary
    assert args.slots_per_rank == primary + 2


@pytest.mark.parametrize("replicas", (1, 6, 16, 22))
def test_replica_allocation_supports_exact_arbitrary_budgets(replicas: int) -> None:
    num_experts = 16
    affinity = np.ones((num_experts, num_experts), dtype=np.float64)
    np.fill_diagonal(affinity, 0.0)
    demand = np.arange(1, num_experts + 1, dtype=np.float64)

    allocations = _replica_allocations(
        affinity,
        demand,
        replicas=replicas,
        restarts=1,
        iterations=2,
        seed=17,
        candidate_limit=8,
    )

    assert allocations
    for allocation in allocations:
        assert allocation.shape == (replicas,)
        assert bool(((allocation >= 0) & (allocation < num_experts)).all())


def test_empty_slots_keep_instance_statistics_and_materialization_valid() -> None:
    num_experts = 8
    ep_size = 4
    slots_per_rank = 3
    logical_instances = _logical_instances(
        num_experts,
        np.asarray([0], dtype=np.int64),
        total_slots=ep_size * slots_per_rank,
    )
    instance_ranks = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 3, 0, 1, 2], dtype=np.int64)
    lut_instances = np.tile(np.arange(num_experts, dtype=np.int64), (ep_size, 1))
    lut_instances[2:, 0] = num_experts
    demand = np.ones((ep_size, num_experts), dtype=np.float64)
    affinity = np.zeros((ep_size, num_experts, num_experts), dtype=np.float64)

    instance_demand, instance_affinity = _uniform_instance_statistics(
        logical_instances,
        demand,
        affinity,
    )
    layout, owners, lut = _materialize_layout(
        logical_instances,
        instance_ranks,
        lut_instances,
        demand,
        ep_size=ep_size,
        slots_per_rank=slots_per_rank,
        primary_slots_per_rank=2,
        num_experts=num_experts,
    )

    assert instance_demand.shape == (ep_size, ep_size * slots_per_rank)
    assert instance_affinity.shape == (ep_size, ep_size * slots_per_rank, ep_size * slots_per_rank)
    assert bool((instance_demand[:, logical_instances < 0] == 0).all())
    assert int((layout < 0).sum()) == 3
    assert len(np.unique(owners)) == num_experts
    assert np.all(layout[lut] == np.arange(num_experts, dtype=np.int64)[None, :])


@pytest.mark.parametrize("ep_size", (16, 64))
def test_four_node_structured_seed_is_not_used_for_other_topologies(ep_size: int) -> None:
    num_experts = 16
    labels = np.arange(num_experts, dtype=np.int64) % (ep_size // 8)
    logical_instances = np.tile(np.arange(num_experts, dtype=np.int64), 2)
    demand = np.ones((ep_size, num_experts), dtype=np.float64)

    assert (
        _structured_instance_node_candidates(
            labels,
            logical_instances,
            demand,
            ranks_per_node=8,
        )
        == []
    )


def test_four_node_structured_seed_separates_two_copies_and_preserves_capacity() -> None:
    num_experts = 16
    labels = np.arange(num_experts, dtype=np.int64) % 4
    logical_instances = np.tile(np.arange(num_experts, dtype=np.int64), 2)
    demand = np.ones((32, num_experts), dtype=np.float64)

    candidates = _structured_instance_node_candidates(
        labels,
        logical_instances,
        demand,
        ranks_per_node=8,
    )

    assert candidates
    for _, instance_nodes in candidates:
        np.testing.assert_array_equal(np.bincount(instance_nodes, minlength=4), [8, 8, 8, 8])
        for expert in range(num_experts):
            copies = np.flatnonzero(logical_instances == expert)
            assert len(np.unique(instance_nodes[copies])) == 2


@pytest.mark.parametrize("ep_size", (16, 64))
def test_generic_classifier_places_full_r2_on_two_and_eight_nodes(ep_size: int) -> None:
    num_experts = 128
    logical_instances = _logical_instances(
        num_experts,
        np.arange(num_experts, dtype=np.int64),
    )
    demand = np.ones((ep_size, num_experts), dtype=np.float64)
    affinity = np.zeros((ep_size, num_experts, num_experts), dtype=np.float64)
    instance_demand, instance_affinity = _uniform_instance_statistics(
        logical_instances,
        demand,
        affinity,
    )

    instance_ranks = _classify_instances(
        instance_demand,
        instance_affinity,
        logical_instances,
        ep_size=ep_size,
        ranks_per_node=8,
        slots_per_rank=2 * num_experts // ep_size,
        seed=3,
        iterations=2,
        node_omega=1.0,
        rank_omega=0.1,
        gamma=1.0,
    )

    assert bool((np.bincount(instance_ranks, minlength=ep_size) == 2 * num_experts // ep_size).all())
    assert _rank_assignment_is_device_unique(
        instance_ranks,
        logical_instances,
        ep_size=ep_size,
    )
