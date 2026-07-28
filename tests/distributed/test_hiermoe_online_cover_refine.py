# Copyright 2026 Bytedance Ltd. and/or its affiliates

from argparse import Namespace

import numpy as np

from scripts.profile.refine_hiermoe_online_cover import (
    _CoverAction,
    _LayerState,
    _patch_cover_state,
    _rank_cover_proposals,
)


class _ToyEvaluator:
    def evaluate(self, _samples, lut):
        from scripts.profile.build_hiermoe_hierarchical_init_layout import HybridCost

        total = float(lut.sum())
        return HybridCost(total, 0.0, total, 0, 0, 0.0, 0.0, 0.0)


def _args() -> Namespace:
    return Namespace(
        ep_size=2,
        ranks_per_node=1,
        service_group_size=1,
        num_experts=2,
        slots_per_rank=2,
        max_copies=3,
        hidden_size=8,
        bytes_per_element=2,
        inter_ms_per_byte=1.0,
        intra_ms_per_byte=0.1,
        route_ms_per_assignment=0.01,
        communication_phase_multiplier=1.0,
        compute_ms_per_assignment=0.01,
        compute_phase_multiplier=1.0,
    )


def _state() -> _LayerState:
    from scripts.profile.build_hiermoe_hierarchical_init_layout import HybridCost

    zero = HybridCost(0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)
    return _LayerState(
        layout=np.asarray([0, 1, 0, 1], dtype=np.int64),
        owners=np.asarray([0, 1], dtype=np.int64),
        lut=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        optimize_cost=zero,
        validation_cost=zero,
    )


def test_cover_lut_patch_is_vectorized_and_promotes_victim_owner() -> None:
    action = _CoverAction(
        source_logical=0,
        source_slot=0,
        destination_slot=1,
        victim_logical=1,
        target_rank=0,
        proxy_score=1.0,
    )
    result = _patch_cover_state(
        _state(),
        action,
        optimize_samples=[],
        validation_samples=[],
        evaluator=_ToyEvaluator(),
        args=_args(),
    )

    np.testing.assert_array_equal(result.layout, np.asarray([0, 0, 0, 1]))
    np.testing.assert_array_equal(result.owners, np.asarray([0, 3]))
    np.testing.assert_array_equal(
        result.lut,
        np.asarray([[1, 3], [2, 3]]),
    )


def test_rank_proposals_emit_at_most_one_cover_per_target_rank() -> None:
    state = _state()
    demand = np.asarray([[10.0, 1.0], [1.0, 10.0]])
    affinity = np.zeros((2, 2, 2), dtype=np.float64)
    affinity[:, 0, 1] = 3.0
    affinity[:, 1, 0] = 3.0

    proposals = _rank_cover_proposals(
        state,
        demand,
        affinity,
        args=_args(),
    )

    assert len(proposals) <= 2
    assert len({proposal.target_rank for proposal in proposals}) == len(proposals)
    assert all(np.count_nonzero(state.layout == proposal.victim_logical) > 1 for proposal in proposals)
