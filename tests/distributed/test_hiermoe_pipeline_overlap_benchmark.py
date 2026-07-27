# Copyright 2026 Bytedance Ltd. and/or its affiliates

from threading import Lock

import pytest
import torch

from scripts.profile.benchmark_hiermoe_pipeline_overlap import (
    _PREPARE_CHUNKS,
    _hidden_ratio,
    _select_uniform_chunk_schedule,
    _select_uniform_multi_window_schedule,
    _steady_layout,
    _summarize,
)
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager, _PipelinePlannerWindows


def test_pipeline_overlap_hidden_ratio_uses_foreground_dilation():
    assert _hidden_ratio(40.0, 100.0, 104.0) == 0.9
    assert _hidden_ratio(40.0, 100.0, 96.0) == 1.0
    assert _hidden_ratio(40.0, 100.0, 145.0) == 0.0


def test_pipeline_overlap_summary_reports_tail_and_range():
    summary = _summarize([5.0, 1.0, 3.0, 2.0, 4.0])

    assert summary == {
        "median": 3.0,
        "p90": 5.0,
        "minimum": 1.0,
        "maximum": 5.0,
    }


def test_steady_layout_initializes_every_redundant_slot():
    layout, owners = _steady_layout(
        num_experts=8,
        ep_size=2,
        slots_per_rank=8,
        device=torch.device("cpu"),
    )

    assert owners.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert layout.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3]


def test_prepare_chunks_cover_every_timed_substage_in_dependency_order():
    assert tuple(stage for _name, stages in _PREPARE_CHUNKS for stage in stages) == (
        "planner_setup",
        "context",
        "route_hash",
        "baseline_route",
        "occupancy",
        "candidate_routes",
        "pair_events",
        "unary_statistics",
        "unary_scoring",
        "pair_statistics",
        "pair_interaction",
        "candidate_pack",
        "collective_pack",
    )


def test_multi_window_search_uses_ordered_chunks_and_all_windows():
    schedule = _select_uniform_multi_window_schedule(
        window_rows_ms=[[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]],
        chunk_rows_ms=[
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ],
    )

    assert schedule.cut_points == (1, 2, 3, 4)
    assert schedule.hidden_ms == pytest.approx(16.0)
    assert schedule.exposed_ms == pytest.approx(4.0)
    assert schedule.min_slack_ms == pytest.approx(1.0)


def test_six_window_search_can_schedule_every_prepare_atom():
    schedule = _select_uniform_multi_window_schedule(
        window_rows_ms=[[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]],
        chunk_rows_ms=[[4.0, 4.0, 4.0, 4.0, 4.0]],
    )

    assert schedule.cut_points == (0, 1, 2, 3, 4, 5)
    assert schedule.hidden_ms == pytest.approx(20.0)
    assert schedule.exposed_ms == pytest.approx(0.0)
    assert schedule.min_slack_ms == pytest.approx(1.0)


def test_multi_window_search_rejects_a_cut_that_fails_one_sample():
    schedule = _select_uniform_multi_window_schedule(
        window_rows_ms=[[8.0, 4.0, 4.0, 4.0], [5.0, 4.0, 4.0, 4.0]],
        chunk_rows_ms=[
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ],
    )

    assert schedule.cut_points == (1, 2, 3, 4)
    assert schedule.hidden_ms == pytest.approx(12.0)
    assert schedule.exposed_ms == pytest.approx(3.0)


def test_uniform_chunk_search_preserves_order_and_is_safe_for_every_sample():
    schedule = _select_uniform_chunk_schedule(
        stage1_windows_ms=[13.0, 12.5],
        stage2_windows_ms=[3.0, 3.0],
        chunk_rows_ms=[
            [3.0, 3.0, 6.0, 2.0, 10.0],
            [3.0, 3.0, 5.5, 2.0, 10.0],
        ],
    )

    assert (schedule.stage1_end, schedule.stage2_end) == (3, 4)
    assert schedule.hidden_ms == pytest.approx(13.75)
    assert schedule.exposed_ms == pytest.approx(10.0)
    assert schedule.min_slack_ms == pytest.approx(1.0)


def test_uniform_chunk_search_applies_window_and_chunk_safety_margins():
    raw = _select_uniform_chunk_schedule(
        stage1_windows_ms=[13.0],
        stage2_windows_ms=[3.0],
        chunk_rows_ms=[[3.0, 3.0, 6.0, 2.0, 10.0]],
    )
    safe = _select_uniform_chunk_schedule(
        stage1_windows_ms=[13.0],
        stage2_windows_ms=[3.0],
        chunk_rows_ms=[[3.0, 3.0, 6.0, 2.0, 10.0]],
        window_scale=0.85,
        chunk_scale=1.2,
        guard_ms=0.5,
    )

    assert (raw.stage1_end, raw.stage2_end) == (3, 4)
    assert (safe.stage1_end, safe.stage2_end) == (2, 2)
    assert safe.hidden_ms == pytest.approx(6.0)


def _minimal_pipeline_manager(windows, future):
    manager = object.__new__(ExpertSwapManager)
    manager.fixed_pipeline_overlap = True
    manager._pipeline_lock = Lock()
    manager._pipeline_planner_windows = {"layer": windows}
    manager._pipeline_plan_futures = {"layer": future}
    manager._pipeline_planner_dispatch_events = {}
    manager._pipeline_planner_compute_events = {}
    manager._placement_metrics = {}
    manager.layers = {}
    manager._pipeline_stage_event = lambda: None
    return manager


def test_score_window_close_records_deadline_without_waiting_for_stage():
    class FutureMustNotBeObserved:
        @staticmethod
        def done():
            raise AssertionError("score close must not poll the planner future")

        @staticmethod
        def result():
            raise AssertionError("score close must not join the planner future")

    windows = _PipelinePlannerWindows()
    manager = _minimal_pipeline_manager(windows, FutureMustNotBeObserved())
    deadline = object()
    manager._pipeline_stage_event = lambda: deadline

    manager.close_pipeline_planner_score_window("layer")
    assert windows.score_deadline_event is deadline
    assert not windows.score_done.is_set()


def test_collective_stage_does_not_synchronize_the_shared_planner_stream():
    class StreamMustNotSynchronize:
        @staticmethod
        def synchronize():
            raise AssertionError("planner collective must finish through an event dependency")

    windows = _PipelinePlannerWindows()
    windows.collective_result_ready.set()
    windows.score_gate.set()
    manager = _minimal_pipeline_manager(windows, future=None)
    manager._pipeline_stream = lambda _kind, _device: StreamMustNotSynchronize()

    tensor = torch.ones(1)
    result = manager._pipeline_planner_reduce_sum("layer", tensor, tensor.device)

    assert result is tensor
    assert windows.collective_tensor is tensor
    assert windows.collective_tensor_ready.is_set()


def test_ordered_collective_launcher_reduces_published_tensor_and_releases_worker():
    windows = _PipelinePlannerWindows()
    tensor = torch.ones(1)
    windows.collective_tensor = tensor
    windows.collective_device = tensor.device
    windows.collective_tensor_ready.set()
    manager = _minimal_pipeline_manager(windows, future=None)
    manager._pipeline_planner_group = object()
    reduced = []
    manager._planner_reduce_sum = lambda value, group: reduced.append((value, group))

    manager._launch_pipeline_planner_collective("layer", windows)

    assert reduced == [(tensor, manager._pipeline_planner_group)]
    assert windows.collective_done.is_set()
    assert windows.collective_result_ready.is_set()


def test_pipeline_process_group_cleanup_is_noop_when_ep_group_is_reused(monkeypatch):
    manager = object.__new__(ExpertSwapManager)
    manager.ep_group = manager._pipeline_planner_group = object()
    destroyed = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroyed.append)

    manager.destroy_pipeline_process_groups()

    assert destroyed == []
