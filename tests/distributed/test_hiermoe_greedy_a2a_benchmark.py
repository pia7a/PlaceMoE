# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""CPU tests for the greedy swap/cover A2A replay helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[2] / "scripts/profile/benchmark_hiermoe_greedy_a2a.py"
SPEC = importlib.util.spec_from_file_location("benchmark_hiermoe_greedy_a2a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_shifted_initial_replicas_preserve_owners_and_fill_one_slot_per_rank():
    layout, owners, slots_per_rank = benchmark._initial_layout(8, 4, 1, torch.device("cpu"))

    assert slots_per_rank == 3
    torch.testing.assert_close(owners, torch.tensor([0, 1, 3, 4, 6, 7, 9, 10]))
    torch.testing.assert_close(layout, torch.tensor([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]))
    torch.testing.assert_close(layout.index_select(0, owners), torch.arange(8))


def test_empty_initial_layout_preserves_capacity_for_planner_initialization():
    layout, owners, _slots_per_rank = benchmark._initial_layout(8, 4, 1, torch.device("cpu"), fill_replicas=False)

    torch.testing.assert_close(layout, torch.tensor([0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1]))
    torch.testing.assert_close(layout.index_select(0, owners), torch.arange(8))


def test_communication_score_separates_assignment_and_rank_dedup_counts():
    logical = [torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.long)]
    physical = [torch.tensor([[0, 1, 3, 4], [0, 3, 6, 9]], dtype=torch.long)]

    original_nodes, original_ranks = benchmark._communication_score(
        logical,
        slots_per_rank=2,
        source_rank=0,
        ranks_per_node=2,
        deduplicate=False,
    )
    dedup_nodes, dedup_ranks = benchmark._communication_score(
        physical,
        slots_per_rank=3,
        source_rank=0,
        ranks_per_node=2,
        deduplicate=True,
    )

    assert original_nodes == 2
    assert original_ranks == 5
    assert dedup_nodes == 1
    assert dedup_ranks == 4


def test_summary_uses_median_and_upper_rank_p90():
    summary = benchmark._summary([(10.0, 6.0, 4.0, 10.0), (8.0, 5.0, 3.0, 8.0), (12.0, 7.0, 5.0, 12.0)])

    assert summary["wall_ms"]["median"] == pytest.approx(10.0)
    assert summary["a2a_ms"]["p90"] == pytest.approx(12.0)
