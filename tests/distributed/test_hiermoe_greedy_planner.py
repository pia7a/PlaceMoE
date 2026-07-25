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

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe import greedy_planner as greedy_planner_module
from veomni.distributed.moe.hiermoe import statistical_scorer as statistical_scorer_module
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GREEDY_COMMUNICATION_PHASE_MULTIPLIER,
    GREEDY_COMPUTE_PHASE_MULTIPLIER,
    GREEDY_COVER_ALGORITHM_VERSION,
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import _route_hash
from veomni.distributed.moe.hiermoe.statistical_scorer import (
    _pair_state_indices,
    _pair_state_key_count,
    statistical_candidate_local_deltas,
    statistical_pair_interaction_bound_local,
    statistical_primitive_fast_path_available,
    statistical_selected_pair_local_deltas,
    statistical_unary_candidate_local_deltas,
    uniform_statistical_baseline_routes,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _planner(
    *,
    chunk_size: int = 8,
    max_copies: int = 8,
    communication_scale: float = 1.0,
    forward_compute_per_assignment: float = 0.0,
    forward_compute_constant: float = 0.0,
    adaptive_topk: bool = False,
    adaptive_topk_initial: int = 16,
    adaptive_topk_strict_certificate: bool = False,
    early_proxy_topk: int = 0,
    exact_primitive_topk: int = 0,
    post_shortlist_compact_pair: bool = False,
    exact_primitive_max_only: bool = False,
) -> GreedyCommunicationPlanner:
    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=communication_scale,
        forward_compute_per_assignment=forward_compute_per_assignment,
        forward_compute_constant=forward_compute_constant,
        candidate_chunk_size=chunk_size,
        max_copies=max_copies,
        adaptive_topk=adaptive_topk,
        adaptive_topk_initial=adaptive_topk_initial,
        adaptive_topk_strict_certificate=adaptive_topk_strict_certificate,
        early_proxy_topk=early_proxy_topk,
        exact_primitive_topk=exact_primitive_topk,
        post_shortlist_compact_pair=post_shortlist_compact_pair,
        exact_primitive_max_only=exact_primitive_max_only,
    )


def _full_layout() -> tuple[torch.Tensor, torch.Tensor]:
    # Owners occupy the first slot of each rank. Every second slot is a
    # replaceable replica, and expert 0 intentionally has no replica.
    return (
        torch.tensor([0, 3, 1, 2, 2, 3, 3, 1], dtype=torch.long),
        torch.tensor([0, 2, 4, 6], dtype=torch.long),
    )


def test_pipeline_plan_executor_has_one_worker_per_gate_blocked_layer(monkeypatch):
    manager = object.__new__(expert_swap_module.ExpertSwapManager)
    manager.fixed_pipeline_overlap = True
    manager._pipeline_shutdown = False
    manager._pipeline_lock = Lock()
    manager._pipeline_plan_futures = {}
    manager.layers = {f"layer.{index}": None for index in range(3)}
    manager._pipeline_plan_worker_capacity = 1
    manager._pipeline_plan_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(expert_swap_module, "_PIPELINE_PLAN_WORKERS", 1)

    manager._ensure_pipeline_plan_worker_capacity()

    try:
        assert manager._pipeline_plan_worker_capacity == len(manager.layers)
        assert manager._pipeline_plan_executor.submit(lambda: 1).result(timeout=1.0) == 1
    finally:
        manager._pipeline_plan_executor.shutdown(wait=True, cancel_futures=False)


def test_cost_model_uses_explicit_training_phase_multipliers():
    planner = _planner(
        communication_scale=0.25,
        forward_compute_per_assignment=0.5,
        forward_compute_constant=2.0,
    )
    widths = planner._count_widths()
    packed_counts = torch.arange(1, sum(widths) + 1, dtype=torch.float32).view(1, -1)
    assignment_counts = torch.tensor([[3.0, 7.0, 5.0, 2.0]])

    communication, compute, units, _peak_rank, peak_compute_rank, _selected_dim = planner._cost_details(
        packed_counts,
        assignment_counts,
    )

    assert GREEDY_COMMUNICATION_PHASE_MULTIPLIER == 4.0
    assert GREEDY_COMPUTE_PHASE_MULTIPLIER == 3.0
    torch.testing.assert_close(communication, units, rtol=0, atol=0)
    torch.testing.assert_close(compute, torch.tensor([3.0 * (0.5 * 7.0 + 2.0)]), rtol=0, atol=0)
    torch.testing.assert_close(peak_compute_rank, torch.tensor([1]), rtol=0, atol=0)


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


def test_sufficient_statistics_match_full_remap_for_every_swap_and_cover():
    planner = _planner(max_copies=4)
    layout, owners = _full_layout()
    selected = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    sources = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))
    physical = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
    )
    baseline = planner._local_packed_counts(physical)
    copy_slots = planner._copy_table(layout, 4)
    statistical = statistical_candidate_local_deltas(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=planner._token_level_occupancies(physical),
        source_ranks=sources,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        num_experts=4,
    )
    assert statistical is not None

    candidate_layouts = planner._apply_rows(layout, rows)
    candidate_physical = assign_tokens_to_copies_greedy(
        selected,
        candidate_layouts,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
    )
    expected = planner._local_packed_counts(candidate_physical) - baseline
    torch.testing.assert_close(statistical, expected, rtol=0, atol=0)

    affected_groups = planner._candidate_affected_groups(copy_slots, rows)
    valid = affected_groups >= 0
    packed = statistical.gather(1, affected_groups.clamp_min(0)) * valid.to(statistical.dtype)
    restored = planner._restore_candidate_counts(baseline, packed, affected_groups)
    torch.testing.assert_close(restored, baseline + expected, rtol=0, atol=0)


def test_dense_uniform_source_statistics_match_sparse_and_full_remap():
    planner = _planner(max_copies=4)
    layout, owners = _full_layout()
    selected = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    sources = torch.zeros(selected.shape[0], dtype=torch.long)
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))
    physical = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
    )
    baseline = planner._local_packed_counts(physical)
    copy_slots = planner._copy_table(layout, 4)
    common = dict(
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=planner._token_level_occupancies(physical),
        source_ranks=sources,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        num_experts=4,
    )
    dense = statistical_candidate_local_deltas(
        planner,
        selected,
        rows,
        uniform_source_rank=0,
        **common,
    )
    sparse = statistical_candidate_local_deltas(planner, selected, rows, **common)
    assert dense is not None and sparse is not None

    candidates = planner._apply_rows(layout, rows)
    remapped = assign_tokens_to_copies_greedy(
        selected,
        candidates,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
    )
    expected = planner._local_packed_counts(remapped) - baseline
    torch.testing.assert_close(dense, sparse, rtol=0, atol=0)
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)


@pytest.mark.parametrize("uniform_source", (False, True))
def test_non_deduplicated_assignment_cost_matches_full_remap(uniform_source):
    planner = _planner(
        max_copies=4,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
    )
    layout, owners = _full_layout()
    selected = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    sources = (
        torch.zeros(selected.shape[0], dtype=torch.long)
        if uniform_source
        else torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    )
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    copy_slots = planner._copy_table(layout, 4)
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))
    scored = planner._score_actions(
        selected,
        layout,
        rows,
        source_ranks=sources,
        uniform_source_rank=0 if uniform_source else None,
        copy_slots=copy_slots,
        affected_groups=None,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        num_experts=4,
    )

    layouts = torch.cat((layout.view(1, -1), planner._apply_rows(layout, rows)), dim=0)
    remapped = assign_tokens_to_copies_greedy(
        selected,
        layouts,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
    )
    assignment_counts = planner._local_assignment_counts(remapped)
    expected_peak, expected_rank = assignment_counts.max(dim=1)
    expected_compute = 3.0 * (0.25 * expected_peak + 1.5)

    torch.testing.assert_close(scored.compute, expected_compute, rtol=0, atol=0)
    torch.testing.assert_close(scored.peak_compute_rank, expected_rank, rtol=0, atol=0)


def test_sparse_selected_pair_deltas_reconstruct_exact_statistics():
    planner = _planner(max_copies=4)
    layout, owners = _full_layout()
    selected = torch.tensor(
        [[0, 1, 2], [2, 3, 0], [1, 3, 2], [0, 2, 3], [3, 1, 0], [2, 0, 1]],
        dtype=torch.long,
    )
    sources = torch.zeros(selected.shape[0], dtype=torch.long)
    ordinals = torch.arange(selected.shape[0], dtype=torch.long)
    route_hashes = _route_hash(selected, token_ordinals=ordinals, step=7, layer_seed=11)
    copy_slots = planner._copy_table(layout, 4)
    uniform_baseline = uniform_statistical_baseline_routes(
        planner,
        selected,
        copy_slots,
        route_hashes,
        source_rank=0,
    )
    assert uniform_baseline is not None
    physical = uniform_baseline.physical
    occupancies = planner._token_level_occupancies(physical)
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))

    unary_result = statistical_unary_candidate_local_deltas(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=occupancies,
        token_ordinals=ordinals,
        route_hashes=route_hashes,
        uniform_source_rank=0,
        uniform_baseline=uniform_baseline,
        routes_are_unique=True,
        unique_routes=None,
        step=7,
        layer_seed=11,
        num_experts=4,
    )
    assert unary_result is not None
    unary, _route_tables, pair_context = unary_result
    action_indices = torch.arange(rows.shape[0], dtype=torch.long)
    pair = statistical_selected_pair_local_deltas(pair_context, rows, action_indices)
    exact = statistical_candidate_local_deltas(
        planner,
        selected,
        rows,
        layout=layout,
        copy_slots=copy_slots,
        physical=physical,
        occupancies=occupancies,
        source_ranks=sources,
        token_ordinals=ordinals,
        route_hashes=route_hashes,
        uniform_source_rank=0,
        uniform_baseline=uniform_baseline,
        routes_are_unique=True,
        unique_routes=None,
        step=7,
        layer_seed=11,
        num_experts=4,
    )
    assert isinstance(exact, torch.Tensor)
    torch.testing.assert_close(unary + pair, exact, rtol=0, atol=0)
    interaction_bound = statistical_pair_interaction_bound_local(pair_context, rows)
    assert torch.all(pair.abs() <= interaction_bound)


@pytest.mark.parametrize("strict_certificate", (False, True))
def test_adaptive_topk_matches_full_exact_action_and_cost(strict_certificate):
    layout, owners = _full_layout()
    routes = torch.tensor(
        [[0, 1, 2], [2, 3, 0], [1, 3, 2], [0, 2, 3], [3, 1, 0], [2, 0, 1]] * 4,
        dtype=torch.long,
    )
    exact = _planner(
        max_copies=4,
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
    ).plan(
        routes,
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )
    adaptive_planner = _planner(
        max_copies=4,
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        adaptive_topk=True,
        adaptive_topk_initial=1,
        adaptive_topk_strict_certificate=strict_certificate,
    )
    adaptive = adaptive_planner.plan(
        routes,
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )

    assert adaptive.actions == exact.actions
    assert adaptive.final_layout == exact.final_layout
    assert adaptive.final_owner_slots == exact.final_owner_slots
    assert adaptive.baseline_cost == exact.baseline_cost
    assert adaptive.final_cost == exact.final_cost
    torch.testing.assert_close(adaptive.local_physical_routes, exact.local_physical_routes, rtol=0, atol=0)
    assert adaptive_planner.last_adaptive_topk_stats["enabled"] is True
    assert adaptive_planner.last_adaptive_topk_stats["certified"] == [strict_certificate]


def test_joint_cost_selects_compute_improving_cover_when_communication_is_disabled():
    planner = _planner(
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
        forward_compute_constant=2.0,
    )
    layout, owners = _full_layout()
    routes = torch.zeros((20, 1), dtype=torch.long)
    sources = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

    plan = planner.plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        max_swaps=1,
        max_replicas=1,
        step=3,
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "replica"
    assert plan.final_cost.compute < plan.baseline_cost.compute
    assert plan.final_cost.total < plan.baseline_cost.total
    assert plan.baseline_cost.compute == pytest.approx(3.0 * (20.0 + 2.0))
    assert plan.final_cost.compute == pytest.approx(3.0 * (10.0 + 2.0))


def test_early_proxy_full_shortlist_matches_exact_plan_and_route():
    layout, owners = _full_layout()
    routes = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]] * 3,
        dtype=torch.long,
    )
    exact = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
    ).plan(
        routes,
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=5,
        layer_seed=11,
    )
    early_planner = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        early_proxy_topk=1024,
    )
    early = early_planner.plan(
        routes,
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=5,
        layer_seed=11,
    )

    assert early.actions == exact.actions
    assert early.final_layout == exact.final_layout
    assert early.final_owner_slots == exact.final_owner_slots
    assert early.baseline_cost == exact.baseline_cost
    assert early.final_cost == exact.final_cost
    torch.testing.assert_close(early.local_physical_routes, exact.local_physical_routes, rtol=0, atol=0)
    assert early_planner.last_early_proxy_stats["enabled"]
    assert (
        early_planner.last_early_proxy_stats["shortlist_counts"][0]
        == early_planner.last_early_proxy_stats["candidate_counts"][0]
    )


def test_early_proxy_small_shortlist_keeps_exact_acceptance_and_valid_route():
    layout, owners = _full_layout()
    routes = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]] * 3,
        dtype=torch.long,
    )
    planner = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        early_proxy_topk=2,
    )
    plan = planner.plan(
        routes,
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=5,
        layer_seed=11,
    )

    assert planner.last_early_proxy_stats["shortlist_counts"] == [2]
    assert plan.final_cost.total <= plan.baseline_cost.total
    assert plan.local_physical_routes is not None
    _assert_routes_match_layout(routes, plan.local_physical_routes, plan.final_layout)


def test_early_proxy_and_adaptive_topk_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _planner(adaptive_topk=True, early_proxy_topk=16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _planner(early_proxy_topk=16, exact_primitive_topk=16)


def test_batched_layer_planner_matches_sequential_and_reduces_once():
    collective_calls = 0

    def reducer(tensor):
        nonlocal collective_calls
        collective_calls += 1
        return tensor

    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        reducer=reducer,
        candidate_chunk_size=8,
        max_copies=4,
    )
    layout, owners = _full_layout()
    routes = [
        torch.tensor([[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 3, 0], [1, 2, 1], [0, 3, 2], [2, 1, 0]], dtype=torch.long),
    ]
    sources = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([3, 2, 1, 0], dtype=torch.long),
    ]
    sequential = [
        planner.plan(
            route,
            layout,
            owners,
            source_ranks=source,
            max_swaps=1,
            max_replicas=1,
            step=5,
            layer_seed=seed,
        )
        for route, source, seed in zip(routes, sources, (11, 17), strict=True)
    ]
    assert collective_calls == 2
    collective_calls = 0

    batched = planner.plan_layers(
        routes,
        [layout, layout],
        [owners, owners],
        source_ranks=sources,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11, 17),
        step=5,
    )

    assert collective_calls == 1
    for actual, expected in zip(batched, sequential, strict=True):
        assert actual.actions == expected.actions
        assert actual.final_layout == expected.final_layout
        assert actual.final_owner_slots == expected.final_owner_slots
        assert actual.baseline_cost == expected.baseline_cost
        assert actual.final_cost == expected.final_cost

    collective_calls = 0
    uniform_exact = planner.plan_layers(
        routes,
        [layout, layout],
        [owners, owners],
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11, 17),
        step=5,
    )
    assert collective_calls == 1
    collective_calls = 0
    adaptive = GreedyCommunicationPlanner(
        hierarchy=planner.hierarchy,
        perf_model=planner.perf_model,
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        reducer=reducer,
        candidate_chunk_size=8,
        max_copies=4,
        adaptive_topk=True,
        adaptive_topk_initial=16,
    )
    adaptive_plans = adaptive.plan_layers(
        routes,
        [layout, layout],
        [owners, owners],
        source_ranks=(0, 0),
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11, 17),
        step=5,
    )
    assert collective_calls == 2
    for actual, expected in zip(adaptive_plans, uniform_exact, strict=True):
        assert actual.actions == expected.actions
        assert actual.final_layout == expected.final_layout
        assert actual.final_owner_slots == expected.final_owner_slots
        assert actual.baseline_cost == expected.baseline_cost
        assert actual.final_cost == expected.final_cost
        assert actual.local_physical_routes is None


def test_exact_primitive_full_shortlist_matches_batched_full_exact():
    layout, owners = _full_layout()
    routes = [
        torch.tensor([[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3]] * 3, dtype=torch.long),
        torch.tensor([[3, 3, 0], [1, 2, 1], [0, 3, 2], [2, 1, 0]] * 3, dtype=torch.long),
    ]
    kwargs = dict(
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11, 17),
        step=5,
        skip_final_route_update=False,
    )
    exact = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
    ).plan_layers(routes, [layout, layout], [owners, owners], **kwargs)
    primitive_planner = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        exact_primitive_topk=1024,
        post_shortlist_compact_pair=True,
    )
    primitive = primitive_planner.plan_layers(
        routes,
        [layout, layout],
        [owners, owners],
        **kwargs,
    )

    for actual, expected in zip(primitive, exact, strict=True):
        assert actual.actions == expected.actions
        assert actual.final_layout == expected.final_layout
        assert actual.final_owner_slots == expected.final_owner_slots
        assert actual.baseline_cost == expected.baseline_cost
        assert actual.final_cost == expected.final_cost
        torch.testing.assert_close(actual.local_physical_routes, expected.local_physical_routes, rtol=0, atol=0)
    assert primitive_planner.last_exact_primitive_stats["enabled"]
    assert primitive_planner.last_exact_primitive_stats["pair_statistics_mode"] == "post_shortlist_compact"
    assert all(
        primitive_count < 2 * candidate_count
        for primitive_count, candidate_count in zip(
            primitive_planner.last_exact_primitive_stats["primitive_counts"],
            primitive_planner.last_exact_primitive_stats["candidate_counts"],
            strict=True,
        )
    )
    no_route = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        exact_primitive_topk=1024,
        post_shortlist_compact_pair=True,
    ).plan_layers(
        routes,
        [layout, layout],
        [owners, owners],
        **{**kwargs, "skip_final_route_update": True},
    )
    for actual, expected in zip(no_route, exact, strict=True):
        assert actual.actions == expected.actions
        assert actual.final_cost == expected.final_cost
        assert actual.local_physical_routes is None
    single_exact = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
    ).plan(
        routes[0],
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=5,
        layer_seed=11,
    )
    single_primitive = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        exact_primitive_topk=1024,
        post_shortlist_compact_pair=True,
    ).plan(
        routes[0],
        layout,
        owners,
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        step=5,
        layer_seed=11,
    )
    assert single_primitive.actions == single_exact.actions
    assert single_primitive.final_cost == single_exact.final_cost
    torch.testing.assert_close(
        single_primitive.local_physical_routes,
        single_exact.local_physical_routes,
        rtol=0,
        atol=0,
    )


def test_exact_primitive_max_only_matches_dense_final_decision():
    layout, owners = _full_layout()
    routes = [
        torch.tensor([[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3]] * 3, dtype=torch.long),
        torch.tensor([[3, 3, 0], [1, 2, 1], [0, 3, 2], [2, 1, 0]] * 3, dtype=torch.long),
    ]
    kwargs = dict(
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11, 17),
        step=5,
        skip_final_route_update=True,
    )
    dense_planner = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        exact_primitive_topk=4,
    )
    sparse_planner = _planner(
        communication_scale=0.5,
        forward_compute_per_assignment=0.25,
        forward_compute_constant=1.5,
        exact_primitive_topk=4,
        exact_primitive_max_only=True,
    )
    dense = dense_planner.plan_layers(routes, [layout, layout], [owners, owners], **kwargs)
    sparse = sparse_planner.plan_layers(routes, [layout, layout], [owners, owners], **kwargs)

    for actual, expected in zip(sparse, dense, strict=True):
        assert actual.actions == expected.actions
        assert actual.final_layout == expected.final_layout
        assert actual.baseline_cost == expected.baseline_cost
        assert actual.final_cost == expected.final_cost
    assert sparse_planner.last_exact_primitive_stats["unary_selector"] == "batched_full_exact_compact"
    assert sparse_planner.last_exact_primitive_stats["max_only_unary"]


@pytest.mark.parametrize(
    ("limit_name", "defer_pair_statistics"),
    (
        ("_MAX_PAIR_EVENTS", False),
        ("_MAX_BATCHED_PAIR_LOOKUP_ELEMENTS", True),
    ),
)
def test_exact_primitive_falls_back_when_memory_guard_exceeded(
    monkeypatch,
    limit_name,
    defer_pair_statistics,
):
    layout, owners = _full_layout()
    routes = torch.tensor([[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3]] * 3, dtype=torch.long)
    planner = _planner(exact_primitive_topk=4)
    copy_slots = planner._copy_table(layout, int(owners.numel()))
    assert statistical_primitive_fast_path_available(
        planner,
        routes,
        copy_slots=copy_slots,
        num_experts=int(owners.numel()),
        defer_pair_statistics=defer_pair_statistics,
    )

    monkeypatch.setattr(statistical_scorer_module, limit_name, 0)
    planner.post_shortlist_compact_pair = defer_pair_statistics
    fallback = planner.plan_layers(
        [routes],
        [layout],
        [owners],
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11,),
        step=5,
        skip_final_route_update=True,
    )[0]
    exact = _planner().plan_layers(
        [routes],
        [layout],
        [owners],
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11,),
        step=5,
        skip_final_route_update=True,
    )[0]

    assert fallback.actions == exact.actions
    assert fallback.final_cost == exact.final_cost
    assert planner.last_exact_primitive_stats["enabled"] is False


def test_exact_primitive_callback_reports_dependency_order():
    layout, owners = _full_layout()
    routes = torch.tensor([[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3]] * 3, dtype=torch.long)
    stages = []
    _planner(exact_primitive_topk=4).plan_layers(
        [routes],
        [layout],
        [owners],
        source_ranks=0,
        max_swaps=1,
        max_replicas=1,
        layer_seeds=(11,),
        step=5,
        skip_final_route_update=True,
        prepare_stage_callback=stages.append,
    )

    assert stages.index("pair_events") < stages.index("unary_statistics")
    assert stages.index("unary_statistics") < stages.index("unary_scoring")


def test_compact_pair_state_indices_are_dense_and_unique():
    num_experts = 7
    state_count = 6
    encoded = []
    for lhs in range(num_experts):
        for rhs in range(lhs + 1, num_experts):
            lhs_states, rhs_states = torch.meshgrid(
                torch.arange(state_count),
                torch.arange(state_count),
                indexing="ij",
            )
            encoded.append(
                _pair_state_indices(
                    torch.full_like(lhs_states, lhs),
                    torch.full_like(rhs_states, rhs),
                    lhs_states,
                    rhs_states,
                    num_experts=num_experts,
                    state_count=state_count,
                ).reshape(-1)
            )

    keys = torch.cat(encoded)
    key_count = _pair_state_key_count(num_experts, state_count)
    assert keys.numel() == key_count
    torch.testing.assert_close(torch.sort(keys).values, torch.arange(key_count))


def test_compact_uniform_baseline_routes_match_full_mapping():
    planner = _planner(max_copies=4)
    layout, _owners = _full_layout()
    selected = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    hashes = _route_hash(selected, token_ordinals=ordinals, step=7, layer_seed=11)
    compact = uniform_statistical_baseline_routes(
        planner,
        selected,
        planner._copy_table(layout, 4),
        hashes,
        source_rank=0,
    )
    expected = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=7,
        layer_seed=11,
        max_copies=4,
        route_hashes=hashes,
    )

    assert compact is not None
    torch.testing.assert_close(compact.physical, expected, rtol=0, atol=0)


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


def test_empty_slots_are_filled_by_recomputing_each_greedy_marginal():
    planner = _planner(max_copies=2)
    layout = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.long)
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] * 3, dtype=torch.long)
    sources = torch.tensor([0, 0, 1, 1, 2, 3] * 3, dtype=torch.long)
    ordinals = torch.arange(routes.shape[0], dtype=torch.long)

    plan = planner.plan(
        routes,
        layout,
        owners,
        source_ranks=sources,
        token_ordinals=ordinals,
        max_swaps=0,
        max_replicas=4,
        step=0,
        layer_seed=5,
    )

    current = layout.clone()
    expected_actions = []
    for _ in range(4):
        empty_slots = torch.nonzero(current < 0, as_tuple=False).flatten()
        rows = planner._cover_rows(current, owners, empty_slots)
        candidates = planner._apply_rows(current, rows)
        physical = assign_tokens_to_copies_greedy(
            routes,
            candidates,
            slots_per_rank=2,
            source_ranks=sources,
            hierarchy_group_sizes=(2, 4),
            num_experts=4,
            token_ordinals=ordinals,
            step=0,
            layer_seed=5,
            max_copies=2,
        )
        communication, _peak, _dim = planner._communication_cost(planner._local_packed_counts(physical))
        best = communication.argmin()
        row = rows.index_select(0, best.view(1))[0]
        expected_actions.append(planner._placement_action(row.tolist()))
        current = candidates.index_select(0, best.view(1))[0]

    assert plan.actions == tuple(expected_actions)
    assert plan.final_layout == tuple(current.tolist())


def test_empty_slot_batch_selection_respects_fused_copy_capacity():
    layout = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.long)
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    routes = torch.zeros((20, 1), dtype=torch.long)

    plan = _planner(max_copies=2).plan(
        routes,
        layout,
        owners,
        source_ranks=torch.tensor([0] * 10 + [1] * 10, dtype=torch.long),
        max_swaps=0,
        max_replicas=4,
        step=0,
    )

    assert len(plan.actions) == 4
    copy_counts = torch.bincount(torch.tensor(plan.final_layout), minlength=4)
    assert int(copy_counts.max().item()) <= 2
    assert -1 not in plan.final_layout


def test_empty_slot_initialization_rejects_impossible_copy_limit():
    layout = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.long)
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    with pytest.raises(ValueError, match="cannot be initialized"):
        _planner(max_copies=1).plan(
            torch.arange(4, dtype=torch.long).view(-1, 1),
            layout,
            owners,
            source_ranks=torch.arange(4, dtype=torch.long),
            max_swaps=0,
            max_replicas=4,
        )


def test_empty_slot_initialization_preserves_completion_feasibility():
    layout = torch.tensor([0, 1, 1, -1, 2, 0, 3, -1], dtype=torch.long)
    owners = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    routes = torch.full((20, 1), 2, dtype=torch.long)

    plan = _planner(max_copies=2).plan(
        routes,
        layout,
        owners,
        source_ranks=1,
        max_swaps=0,
        max_replicas=2,
    )

    assert plan.final_layout[3] == 3
    assert plan.final_layout[7] == 2
    assert -1 not in plan.final_layout


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


def test_uniform_source_winner_route_table_matches_full_remap():
    layout, owners = _full_layout()
    routes = torch.tensor(
        [[0, 0, 1], [2, 3, 2], [1, 3, 0], [0, 2, 3], [3, 3, 1], [2, 0, 1]],
        dtype=torch.long,
    )
    planner = _planner(max_copies=4)
    sources = torch.ones(routes.shape[0], dtype=torch.long)
    ordinals = torch.tensor([5, 7, 11, 13, 17, 19], dtype=torch.long)
    cover_slots = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    rows = torch.cat((planner._swap_rows(layout, owners), planner._cover_rows(layout, owners, cover_slots)))
    route_hashes = _route_hash(routes, token_ordinals=ordinals, step=3, layer_seed=5)
    physical = assign_tokens_to_copies_greedy(
        routes,
        layout,
        slots_per_rank=2,
        source_ranks=sources,
        hierarchy_group_sizes=(2, 4),
        num_experts=4,
        token_ordinals=ordinals,
        step=3,
        layer_seed=5,
        max_copies=4,
        route_hashes=route_hashes,
    )
    result = statistical_candidate_local_deltas(
        planner,
        routes,
        rows,
        layout=layout,
        copy_slots=planner._copy_table(layout, 4),
        physical=physical,
        occupancies=planner._token_level_occupancies(physical),
        source_ranks=sources,
        token_ordinals=ordinals,
        route_hashes=route_hashes,
        uniform_source_rank=1,
        return_route_tables=True,
        step=3,
        layer_seed=5,
        num_experts=4,
    )
    assert isinstance(result, tuple)
    _deltas, route_tables = result
    assert route_tables is not None

    for action_index, row in enumerate(rows):
        candidate_layout = planner._apply_rows(layout, row.view(1, -1))[0]
        expected = assign_tokens_to_copies_greedy(
            routes,
            candidate_layout,
            slots_per_rank=2,
            source_ranks=sources,
            hierarchy_group_sizes=(2, 4),
            num_experts=4,
            token_ordinals=ordinals,
            step=3,
            layer_seed=5,
            max_copies=4,
            route_hashes=route_hashes,
        )
        actual = planner._apply_statistical_action_routes(
            routes,
            row,
            action_index,
            physical,
            route_hashes,
            route_tables,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def _sharded_collective_parity_worker():
    rank = dist.get_rank()
    layout = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    owners = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    if rank == 0:
        routes = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 3]] * 4, dtype=torch.long)
    else:
        routes = torch.tensor([[2, 3], [1, 3], [0, 2], [1, 2]] * 4, dtype=torch.long)

    def reducer(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    kwargs = dict(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=3,
        reducer=reducer,
        forward_compute_per_assignment=0.2,
        forward_compute_constant=0.3,
        candidate_chunk_size=4,
    )
    exact = GreedyCommunicationPlanner(**kwargs).plan(
        routes,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )
    sharded = GreedyCommunicationPlanner(**kwargs, process_group=dist.group.WORLD).plan(
        routes,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )
    compact = GreedyCommunicationPlanner(
        **kwargs,
        process_group=dist.group.WORLD,
        compact_candidate_collective=True,
    ).plan(
        routes,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )
    no_candidate = GreedyCommunicationPlanner(
        **kwargs,
        process_group=dist.group.WORLD,
        compact_candidate_collective=True,
    ).plan(
        routes,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=0,
        max_replicas=0,
    )

    assert sharded.actions == exact.actions
    assert sharded.final_layout == exact.final_layout
    assert sharded.final_owner_slots == exact.final_owner_slots
    assert sharded.baseline_cost == exact.baseline_cost
    assert sharded.final_cost == exact.final_cost
    torch.testing.assert_close(sharded.local_physical_routes, exact.local_physical_routes, rtol=0, atol=0)
    assert compact.actions == exact.actions
    assert compact.final_layout == exact.final_layout
    assert compact.baseline_cost == exact.baseline_cost
    assert compact.final_cost == exact.final_cost
    torch.testing.assert_close(compact.local_physical_routes, exact.local_physical_routes, rtol=0, atol=0)
    assert no_candidate.actions == ()


def test_reduce_scatter_candidate_scoring_matches_full_all_reduce():
    torchrun(_sharded_collective_parity_worker, world_size=2, backend="gloo")


def _adaptive_fast_path_consensus_worker():
    rank = dist.get_rank()
    layout = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    owners = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    routes = (
        torch.tensor([[0, 1], [0, 2], [1, 3], [0, 3]] * 4, dtype=torch.long)
        if rank == 0
        else torch.empty((0, 2), dtype=torch.long)
    )

    def reducer(tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    if rank == 1:
        greedy_planner_module.statistical_unary_candidate_local_deltas = lambda *_args, **_kwargs: None

    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=16,
        bytes_per_element=2,
        slots_per_rank=3,
        reducer=reducer,
        forward_compute_per_assignment=0.2,
        forward_compute_constant=0.3,
        candidate_chunk_size=4,
        adaptive_topk=True,
        adaptive_topk_initial=4,
    )
    plan = planner.plan(
        routes,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=1,
        max_replicas=1,
        step=7,
        layer_seed=11,
    )

    assert planner.last_adaptive_topk_stats["enabled"] is False
    assert "at least one EP rank" in planner.last_adaptive_topk_stats["reason"]
    signature = (
        tuple((action.kind, action.src_slot, action.dst_slot) for action in plan.actions),
        plan.final_layout,
        plan.final_cost.total,
    )
    gathered = [None, None]
    dist.all_gather_object(gathered, signature)
    assert gathered[0] == gathered[1]
    if rank == 1:
        assert plan.local_physical_routes.numel() == 0


def test_adaptive_topk_availability_and_empty_routes_are_collective_symmetric():
    torchrun(_adaptive_fast_path_consensus_worker, world_size=2, backend="gloo")


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


@pytest.mark.parametrize("mode", ("layer", "step"))
def test_greedy_cover_selector_accepts_both_route_modes_with_redundant_slots(mode):
    config = HierMoEConfig(
        expert_swap_selector="hiermoe_greedy_cover_p1",
        expert_swap_mode=mode,
        redundant_slot_increment_per_device=1,
    )

    assert config.expert_swap_selector == "hiermoe_greedy_cover_p1"
    assert config.expert_swap_mode == mode
