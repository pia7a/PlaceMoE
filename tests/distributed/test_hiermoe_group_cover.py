# Copyright 2026 Bytedance Ltd. and/or its affiliates

from argparse import Namespace

import numpy as np
import torch

from scripts.profile.build_hiermoe_hierarchical_init_layout import _HybridEvaluator
from scripts.profile.refine_hiermoe_group_cover import (
    _AffectedRouteEvaluator,
    _LayerState,
    _patch_group_cover,
)


def _args() -> Namespace:
    return Namespace(
        ep_size=4,
        ranks_per_node=2,
        num_experts=8,
        slots_per_rank=4,
        max_copies=4,
        hidden_size=16,
        bytes_per_element=2,
        inter_ms_per_byte=6.765449326279194e-08,
        intra_ms_per_byte=5.02482606728045e-09,
        route_ms_per_assignment=8.746548178958447e-05,
        communication_phase_multiplier=3.1,
        compute_ms_per_assignment=2.82807e-05,
        compute_phase_multiplier=4.19,
    )


def _state() -> _LayerState:
    layout = np.asarray(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        dtype=np.int64,
    )
    owners = np.arange(8, dtype=np.int64)
    lut = np.empty((4, 8), dtype=np.int64)
    lut[:2] = owners
    lut[2:, :4] = np.arange(8, 12, dtype=np.int64)
    lut[2:, 4:] = np.arange(12, 16, dtype=np.int64)
    return _LayerState(layout=layout, owners=owners, lut=lut)


def _samples() -> list[list[torch.Tensor]]:
    rows = [
        [0, 1, 4],
        [0, 2, 5],
        [1, 3, 6],
        [2, 3, 7],
        [0, 4, 5],
        [1, 6, 7],
    ]
    return [[torch.tensor(rows[index:] + rows[:index], dtype=torch.long) for index in range(4)]]


def test_group_cover_affected_replay_matches_full_hybrid_cost():
    args = _args()
    state = _state()
    candidate = _patch_group_cover(
        state,
        source_rank=0,
        target_rank=1,
        service_group_size=2,
        args=args,
    )
    assert candidate is not None
    assert np.array_equal(candidate.layout[4:8], np.arange(4))
    assert np.all(np.bincount(candidate.layout, minlength=8) >= 1)

    samples = _samples()
    evaluator = _HybridEvaluator(args)
    incremental = _AffectedRouteEvaluator(
        samples,
        state.lut,
        evaluator=evaluator,
        args=args,
    )
    baseline, _ = incremental.cost()
    candidate_cost, affected = incremental.cost(candidate.lut)
    expected_baseline = evaluator.evaluate(samples, state.lut)
    expected_candidate = evaluator.evaluate(samples, candidate.lut)

    assert affected > 0
    assert baseline.total_ms == expected_baseline.total_ms
    assert candidate_cost.total_ms == expected_candidate.total_ms


def test_group_cover_incremental_commit_remains_exact():
    args = _args()
    state = _state()
    candidate = _patch_group_cover(
        state,
        source_rank=0,
        target_rank=1,
        service_group_size=1,
        args=args,
    )
    assert candidate is not None

    samples = _samples()
    evaluator = _HybridEvaluator(args)
    incremental = _AffectedRouteEvaluator(
        samples,
        state.lut,
        evaluator=evaluator,
        args=args,
    )
    assert incremental.commit(candidate.lut) > 0
    committed, _ = incremental.cost()
    expected = evaluator.evaluate(samples, candidate.lut)
    assert committed.total_ms == expected.total_ms


def test_group_cover_rejects_removing_the_last_victim_copy():
    args = _args()
    state = _state()
    layout = state.layout.copy()
    layout[12:16] = np.arange(4)
    owners = np.arange(8, dtype=np.int64)
    owners[4:] = np.arange(4, 8)
    lut = np.broadcast_to(owners, (4, 8)).copy()
    unique_victim_state = _LayerState(layout=layout, owners=owners, lut=lut)

    assert (
        _patch_group_cover(
            unique_victim_state,
            source_rank=0,
            target_rank=1,
            service_group_size=1,
            args=args,
        )
        is None
    )
