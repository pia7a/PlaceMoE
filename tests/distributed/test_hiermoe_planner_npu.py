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

import types

import pytest
import torch

from veomni.distributed.moe.hiermoe.core_planner import (
    CoReMoEPlanner,
    QuotaPolicyEntry,
    RouteSummary,
    _remap_replica_logical_from_baseline,
    assign_tokens_to_copies_with_quota,
    build_quota_tensor_tables,
)
from veomni.distributed.moe.hiermoe.perf_model import (
    GradientSyncCost,
    HierMoEPerfModel,
    LinkCost,
    PeerTransferCost,
)
from veomni.distributed.moe.hiermoe.planner import CurrentRoutePlanner, PlacementAction
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU is required")


def _extension():
    extension = get_hiermoe_planner_npu_ops()
    if extension is None:
        pytest.skip("HierMoE NPU planner operators are not built")
    return extension


def _identity_layout(ep_size: int, experts_per_rank: int, extra_slots: int):
    slots_per_rank = experts_per_rank + extra_slots
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long)
    owners = torch.empty((ep_size * experts_per_rank,), dtype=torch.long)
    for logical in range(owners.numel()):
        rank, local = divmod(logical, experts_per_rank)
        slot = rank * slots_per_rank + local
        layout[slot] = logical
        owners[logical] = slot
    return layout, owners, slots_per_rank


def _run_replica_match(
    *,
    layout: torch.Tensor,
    owners: torch.Tensor,
    redundant_slots: torch.Tensor,
    base_counts: torch.Tensor,
    assignment_loads: torch.Tensor,
    add_group_deltas: torch.Tensor,
    add_assignment_deltas: torch.Tensor,
    remove_group_deltas: torch.Tensor,
    remove_assignment_deltas: torch.Tensor,
    candidate_experts: torch.Tensor,
    state_bytes: torch.Tensor,
    gradient_bytes: torch.Tensor,
    max_actions: int,
    slots_per_rank: int,
    local_world_size: int = 1,
    level_sizes: tuple[int, ...] = (1,),
    payload_bytes: int = 0,
    a2a_beta: float = 0.0,
    compute_per_assignment: float = 1.0,
    state_inter_beta: float = 0.0,
    gradient_inter_beta: float = 0.0,
):
    ep_size = redundant_slots.shape[0]
    padded_levels = (*level_sizes, 1, 1)[:3]
    return _extension().replica_match(
        base_counts.npu(),
        assignment_loads.npu(),
        add_group_deltas.npu(),
        add_assignment_deltas.npu(),
        remove_group_deltas.npu(),
        remove_assignment_deltas.npu(),
        layout.npu(),
        owners.npu(),
        redundant_slots.npu(),
        candidate_experts.npu(),
        state_bytes.npu(),
        gradient_bytes.npu(),
        max_actions,
        slots_per_rank,
        ep_size,
        local_world_size,
        len(level_sizes),
        padded_levels[0],
        padded_levels[1],
        padded_levels[2],
        payload_bytes,
        1.0,
        compute_per_assignment,
        0.0,
        a2a_beta,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        state_inter_beta,
        0.0,
        0.0,
        0.0,
        gradient_inter_beta,
        0.0,
        0.0,
        0.0,
        gradient_inter_beta,
        1.0,
        True,
    )


def _run_replica_project(
    *,
    sample_routes: torch.Tensor,
    sample_multiplicity: torch.Tensor,
    sample_weights: torch.Tensor,
    sample_sources: torch.Tensor,
    sample_ordinals: torch.Tensor,
    assignment_counts: torch.Tensor,
    layout: torch.Tensor,
    owners: torch.Tensor,
    redundant_slots: torch.Tensor,
    candidate_experts: torch.Tensor,
    slots_per_rank: int,
    seed_base_counts: torch.Tensor | None = None,
    level_sizes: tuple[int, ...] = (1,),
    step: int = 0,
    layer_seed: int = 0,
):
    ep_size = assignment_counts.shape[0]
    padded_levels = (*level_sizes, 1, 1)[:3]
    if seed_base_counts is None:
        group_counts = [torch.zeros((ep_size // size,), dtype=torch.float32) for size in level_sizes]
        for token in range(sample_routes.shape[0]):
            destinations = {
                int(owners[int(logical)].item()) // slots_per_rank
                for logical, multiplicity in zip(
                    sample_routes[token].tolist(),
                    sample_multiplicity[token].tolist(),
                    strict=True,
                )
                if multiplicity > 0
            }
            for level, size in enumerate(level_sizes):
                for group in {rank // size for rank in destinations}:
                    group_counts[level][group] += sample_weights[token]
        seed_base_counts = torch.cat(group_counts)
    return _extension().replica_project(
        sample_routes.npu(),
        sample_multiplicity.npu(),
        sample_weights.npu(),
        sample_sources.npu(),
        sample_ordinals.npu(),
        assignment_counts.npu(),
        seed_base_counts.npu(),
        layout.npu(),
        owners.npu(),
        redundant_slots.npu(),
        candidate_experts.npu(),
        slots_per_rank,
        ep_size,
        len(level_sizes),
        padded_levels[0],
        padded_levels[1],
        padded_levels[2],
        step,
        layer_seed,
    )


def _sample_multiplicity_for_test(routes: torch.Tensor) -> torch.Tensor:
    multiplicity = torch.zeros_like(routes)
    for token, row in enumerate(routes.tolist()):
        seen: set[int] = set()
        for position, logical in enumerate(row):
            if logical in seen:
                continue
            seen.add(logical)
            multiplicity[token, position] = row.count(logical)
    return multiplicity


def _eager_replica_projection(
    planner: CoReMoEPlanner,
    summary: RouteSummary,
    layout: torch.Tensor,
    owners: torch.Tensor,
    *,
    step: int,
    layer_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = assign_tokens_to_copies_with_quota(
        summary.sample_routes,
        layout,
        slots_per_rank=planner.slots_per_rank,
        source_ranks=summary.sample_sources,
        hierarchy=planner.hierarchy,
        owner_slots=owners,
        token_ordinals=summary.sample_ordinals,
        token_weights=summary.sample_weights,
        step=step,
        layer_seed=layer_seed,
    )
    counts, _ = planner._local_weighted_stats(mapping.physical_slots, summary.sample_weights)
    assignments = planner._project_sample_assignments(summary, mapping.physical_slots, owners)
    return torch.cat(counts), assignments


def _eager_independent_replica_projection(
    planner: CoReMoEPlanner,
    summary: RouteSummary,
    layout: torch.Tensor,
    owners: torch.Tensor,
    baseline_physical: torch.Tensor,
    *,
    logical: int,
    step: int,
    layer_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical = _remap_replica_logical_from_baseline(
        summary.sample_routes,
        layout,
        baseline_physical,
        logical_expert=logical,
        slots_per_rank=planner.slots_per_rank,
        source_ranks=summary.sample_sources,
        hierarchy=planner.hierarchy,
        owner_slots=owners,
        token_ordinals=summary.sample_ordinals,
        token_weights=summary.sample_weights,
        step=step,
        layer_seed=layer_seed,
    )
    counts, _ = planner._local_weighted_stats(physical, summary.sample_weights)
    assignments = planner._project_sample_assignments(summary, physical, owners)
    return torch.cat(counts), assignments


def _eager_aggregate_swap_select(
    planner: CoReMoEPlanner,
    selected: torch.Tensor,
    layout: torch.Tensor,
    owners: torch.Tensor,
    *,
    max_swaps: int,
):
    token_hits = torch.nn.functional.one_hot(selected, num_classes=owners.numel()).amax(dim=1).float()
    working_layout = layout.clone()
    working_owners = owners.clone()
    used: set[int] = set()
    actions: list[PlacementAction] = []
    for _ in range(max_swaps):
        stats = planner._initial_swap_stats(token_hits, selected, working_owners)
        current_tensor = planner._current_swap_cost(stats)
        current = planner._placement_cost(
            current_tensor,
            layout=working_layout,
            owner_slots=working_owners,
            actions=actions,
        )
        bottlenecks = {current.peak_communication_rank, current.peak_compute_rank}
        best = None
        for lhs in range(owners.numel()):
            for rhs in range(lhs + 1, owners.numel()):
                if not planner._valid_swap(working_layout, working_owners, lhs, rhs, used, bottlenecks):
                    continue
                pair = torch.tensor([[lhs, rhs]], dtype=torch.long, device=selected.device)
                candidate_batch, _ = planner._swap_candidate_costs(stats, pair)
                candidate_tensor = planner._index_cost(
                    candidate_batch,
                    torch.tensor(0, dtype=torch.long, device=selected.device),
                )
                candidate_layout, candidate_owners, action = planner._swap_layout(
                    working_layout,
                    working_owners,
                    lhs,
                    rhs,
                )
                candidate = planner._placement_cost(
                    candidate_tensor,
                    layout=candidate_layout,
                    owner_slots=candidate_owners,
                    actions=(*actions, action),
                )
                row = (
                    candidate.total,
                    lhs,
                    rhs,
                    candidate_layout,
                    candidate_owners,
                    action,
                )
                if best is None or row[:3] < best[:3]:
                    best = row
        if best is None or not best[0] < current.total:
            break
        _, lhs, rhs, working_layout, working_owners, action = best
        used.update((lhs, rhs))
        actions.append(action)
    return working_layout, working_owners, actions


@pytest.mark.parametrize("max_swaps", (1, 4))
def test_swap_select_matches_eager_with_nonzero_runtime_costs(max_swaps):
    layout, owners, slots_per_rank = _identity_layout(4, 2, 1)
    layout[slots_per_rank + 2] = 0
    selected = torch.tensor(
        [[0, 1]] * 96 + [[0, 2]] * 64 + [[1, 3]] * 48 + [[4, 5]] * 8 + [[6, 7]] * 8,
        dtype=torch.long,
        device="npu",
    )
    state_move = PeerTransferCost(
        intra=LinkCost(alpha=0.01, beta=1.0e-8),
        inter=LinkCost(alpha=0.02, beta=2.0e-8),
    )
    gradient_sync = GradientSyncCost(
        gather=PeerTransferCost(
            intra=LinkCost(alpha=0.005, beta=1.0e-8),
            inter=LinkCost(alpha=0.01, beta=2.0e-8),
        ),
        scatter=PeerTransferCost(
            intra=LinkCost(alpha=0.006, beta=1.0e-8),
            inter=LinkCost(alpha=0.012, beta=2.0e-8),
        ),
    )
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=0.1, beta=1.0e-4),
        inter=(LinkCost(alpha=0.2, beta=2.0e-4),),
        intra=LinkCost(alpha=0.05, beta=5.0e-5),
        source="test-nonzero-runtime-costs",
        state_move=state_move,
        gradient_sync=gradient_sync,
        schema_version=2,
    )
    state_bytes = tuple(100_000 + 1_000 * expert for expert in range(owners.numel()))
    gradient_bytes = tuple(10_000 + 100 * expert for expert in range(owners.numel()))
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test", local_world_size=2),
        perf_model=perf_model,
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        communication_scale=1.0,
        forward_compute_per_assignment=0.25,
        reducer=lambda value: value,
        expert_state_bytes=state_bytes,
        expert_gradient_bytes=gradient_bytes,
    )
    layout = layout.npu()
    owners = owners.npu()
    token_hits = torch.nn.functional.one_hot(selected, num_classes=owners.numel()).amax(dim=1).float()
    stats = planner._initial_swap_stats(token_hits, selected, owners)

    fused = planner._fused_swap_select(
        stats,
        layout,
        owners,
        max_swaps=max_swaps,
        sample_routes=selected,
        sample_weights=torch.ones((selected.shape[0],), dtype=torch.float32, device=selected.device),
    )
    assert fused is not None
    fused_layout, fused_owners, fused_actions, metadata, final_base = fused
    eager_layout, eager_owners, eager_actions = _eager_aggregate_swap_select(
        planner,
        selected,
        layout,
        owners,
        max_swaps=max_swaps,
    )
    torch.npu.synchronize()

    accepted = int(metadata[0].cpu().item())
    expected_actions = [
        [
            action.src_logical,
            action.dst_logical,
            action.src_slot,
            action.dst_slot,
            1,
        ]
        for action in eager_actions
    ]
    assert accepted == len(expected_actions) > 0
    assert fused_actions[:accepted].cpu().tolist() == expected_actions
    assert torch.equal(fused_layout.cpu(), eager_layout.cpu())
    assert torch.equal(fused_owners.cpu(), eager_owners.cpu())
    expected_final_base = torch.cat(
        planner._initial_swap_stats(token_hits, selected, fused_owners).base_counts
    )
    assert torch.equal(final_base.cpu(), expected_final_base.cpu())


def test_swap_select_zero_swaps_returns_identity_and_current_bottlenecks():
    layout, owners, slots_per_rank = _identity_layout(2, 1, 0)
    selected = torch.tensor([[1, 1], [1, 0]], dtype=torch.long, device="npu")
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        reducer=lambda value: value,
    )
    layout = layout.npu()
    owners = owners.npu()
    token_hits = torch.nn.functional.one_hot(selected, num_classes=owners.numel()).amax(dim=1).float()
    stats = planner._initial_swap_stats(token_hits, selected, owners)

    fused = planner._fused_swap_select(
        stats,
        layout,
        owners,
        max_swaps=0,
        sample_routes=selected,
        sample_weights=torch.ones((selected.shape[0],), dtype=torch.float32, device="npu"),
    )
    assert fused is not None
    fused_layout, fused_owners, fused_actions, metadata, final_base = fused
    torch.npu.synchronize()

    assert torch.equal(fused_layout, layout)
    assert torch.equal(fused_owners, owners)
    assert fused_actions.cpu().tolist() == [[-1] * 5]
    assert metadata.cpu().tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert torch.equal(final_base.cpu(), torch.cat(stats.base_counts).cpu())


def test_replica_prepare_matches_packed_route_oracle():
    selected_cpu = torch.tensor(
        [
            [0, 0, 2, 4],
            [1, 3, 1, 7],
            [6, 6, 6, 5],
            [7, 0, 2, 7],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    route_indices, multiplicities, token_counts = _extension().replica_prepare(selected_cpu.npu(), 8)
    route_indices = route_indices.cpu()
    multiplicities = multiplicities.cpu()
    token_counts = token_counts.cpu()

    for expert in range(8):
        matches = selected_cpu == expert
        token_ids = matches.any(dim=1).nonzero(as_tuple=False).reshape(-1)
        expected_indices = token_ids * selected_cpu.shape[1] + matches[token_ids].long().argmax(dim=1)
        expected_multiplicities = matches[token_ids].sum(dim=1).to(torch.int32)
        count = token_ids.numel()
        assert token_counts[expert].item() == count
        assert torch.equal(route_indices[expert, :count], expected_indices.to(torch.int32))
        assert torch.equal(multiplicities[expert, :count], expected_multiplicities)


def test_fused_prepare_normalizes_noncontiguous_routes():
    selected = torch.tensor([[0, 0, 2, 4], [1, 3, 1, 7], [6, 6, 6, 5], [7, 0, 2, 7]], dtype=torch.long, device="npu")
    selected = selected.t().contiguous().t()
    assert not selected.is_contiguous()
    with pytest.raises(RuntimeError, match="selected must be contiguous"):
        _extension().replica_prepare(selected, 8)

    planner = CurrentRoutePlanner(
        hierarchy=Hierarchy(ep_size=8, group_sizes=(4, 8), source="test"),
        perf_model=HierMoEPerfModel.from_path(None),
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=2,
        reducer=lambda value: value,
    )
    packed = planner._fused_replica_route_tables(selected, 8)
    assert packed is not None
    _route_indices, _multiplicities, token_counts = packed
    expected_counts = torch.stack([(selected == expert).any(dim=1).sum() for expert in range(8)]).to(torch.int32)
    assert torch.equal(token_counts, expected_counts)


def test_fused_prepare_falls_back_when_route_exceeds_ub_limit():
    planner = CurrentRoutePlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.from_path(None),
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=3,
        reducer=lambda value: value,
    )

    assert planner._fused_replica_route_tables(torch.zeros((16_385, 1), dtype=torch.long, device="npu"), 8) is None


def test_fused_replica_planner_matches_exact_eager_fallback():
    layout, owners, slots_per_rank = _identity_layout(4, 2, 1)
    selected = torch.tensor(
        [[0, 0, 1, 2]] * 24 + [[3, 4, 5, 6]] * 8 + [[7, 0, 2, 4]] * 8,
        dtype=torch.long,
        device="npu",
    )

    def make_planner():
        return CurrentRoutePlanner(
            hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
            perf_model=HierMoEPerfModel.from_path(None),
            hidden_size=64,
            bytes_per_element=2,
            slots_per_rank=slots_per_rank,
            communication_scale=1.0,
            forward_compute_per_assignment=1.0,
            reducer=lambda value: value,
        )

    fused = make_planner()
    eager = make_planner()
    eager._fused_replica_route_tables = types.MethodType(lambda self, routes, experts: None, eager)
    eager._fused_replica_candidate_deltas = types.MethodType(lambda self, stats, experts: None, eager)
    captured_states = {"fused": [], "eager": []}
    fused_apply_calls = 0

    original_fused_apply = fused._fused_apply_replica_candidate

    def count_fused_apply(self, stats, logical_expert, destination_rank):
        nonlocal fused_apply_calls
        applied = original_fused_apply(stats, logical_expert, destination_rank)
        fused_apply_calls += int(applied)
        return applied

    fused._fused_apply_replica_candidate = types.MethodType(count_fused_apply, fused)

    def capture_apply(planner, name):
        original_apply = planner._apply_replica_candidate

        def wrapped(self, stats, candidates, best_index, logical_expert, destination_rank, destination_slot):
            original_apply(stats, candidates, best_index, logical_expert, destination_rank, destination_slot)
            torch.npu.synchronize()
            captured_states[name].append(
                tuple(
                    value.detach().cpu().clone()
                    for value in (
                        stats.route_ranks,
                        stats.minimum_scores,
                        stats.tie_count,
                        stats.tied_rank_order,
                        stats.packed_local_token_group_counts,
                    )
                )
            )

        planner._apply_replica_candidate = types.MethodType(wrapped, planner)

    capture_apply(fused, "fused")
    capture_apply(eager, "eager")
    kwargs = {
        "source_ranks": 0,
        "max_swaps": 1,
        "max_replicas": 4,
        "step": 3,
        "layer_seed": 11,
    }

    fused_plan = fused.plan(selected, layout.npu(), owners.npu(), **kwargs)
    eager_plan = eager.plan(selected, layout.npu(), owners.npu(), **kwargs)

    assert fused_plan.actions == eager_plan.actions
    assert fused_plan.final_layout == eager_plan.final_layout
    assert fused_plan.final_cost == eager_plan.final_cost
    assert fused_plan.replica_rounds > 0
    assert fused_apply_calls == fused_plan.replica_rounds
    assert len(captured_states["fused"]) == len(captured_states["eager"])
    for fused_state, eager_state in zip(captured_states["fused"], captured_states["eager"], strict=True):
        assert all(torch.equal(fused_value, eager_value) for fused_value, eager_value in zip(fused_state, eager_state))


@pytest.mark.parametrize(
    ("ep_size", "group_sizes"),
    ((16, (8, 16)), (32, (8, 32)), (64, (8, 16, 64))),
)
def test_fused_replica_apply_matches_ep_topologies(ep_size, group_sizes):
    layout, owners, slots_per_rank = _identity_layout(ep_size, 2, 1)
    source_ranks = torch.arange(2 * ep_size, dtype=torch.long, device="npu").remainder(ep_size)
    local_experts = 2 * source_ranks
    selected = torch.stack(
        (
            torch.zeros_like(local_experts),
            torch.zeros_like(local_experts),
            local_experts,
            local_experts,
            (local_experts + 1).remainder(2 * ep_size),
            (local_experts + 1).remainder(2 * ep_size),
            torch.zeros_like(local_experts),
            local_experts,
        ),
        dim=1,
    )

    def make_planner():
        return CurrentRoutePlanner(
            hierarchy=Hierarchy(ep_size=ep_size, group_sizes=group_sizes, source="test"),
            perf_model=HierMoEPerfModel.from_path(None),
            hidden_size=64,
            bytes_per_element=2,
            slots_per_rank=slots_per_rank,
            communication_scale=1.0,
            forward_compute_per_assignment=1.0,
            reducer=lambda value: value,
        )

    fused = make_planner()
    eager = make_planner()
    eager._fused_replica_route_tables = types.MethodType(lambda self, routes, experts: None, eager)
    eager._fused_replica_candidate_deltas = types.MethodType(lambda self, stats, experts: None, eager)
    kwargs = {
        "source_ranks": source_ranks,
        "max_swaps": 0,
        "max_replicas": 2,
        "step": 5,
        "layer_seed": 19,
    }

    fused_plan = fused.plan(selected, layout.npu(), owners.npu(), **kwargs)
    eager_plan = eager.plan(selected, layout.npu(), owners.npu(), **kwargs)

    assert fused_plan.actions == eager_plan.actions
    assert fused_plan.final_layout == eager_plan.final_layout
    assert fused_plan.final_cost == eager_plan.final_cost
    assert fused_plan.replica_rounds > 0


@pytest.mark.parametrize(
    ("ep_size", "group_sizes"),
    ((16, (8, 16)), (32, (8, 32)), (64, (8, 16, 64))),
)
def test_dual_map_matches_unambiguous_quota_mapping(ep_size, group_sizes):
    current, owners, slots_per_rank = _identity_layout(ep_size, 2, 1)
    candidate = current.clone()
    candidate[slots_per_rank + 2] = 0
    layouts = torch.stack((current, candidate), dim=0)
    logical_ids = torch.arange(2 * ep_size).view(1, 1, -1)
    slot_ids = torch.arange(layouts.shape[1]).view(1, -1, 1)
    matches = layouts.unsqueeze(-1) == logical_ids
    copy_counts = matches.sum(dim=1)
    copy_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, layouts.shape[1]))
    copy_slots = copy_slots.sort(dim=1).values[:, :2].transpose(1, 2).contiguous()
    owner_ranks = torch.div(owners, slots_per_rank, rounding_mode="floor").view(1, -1).expand(2, -1).contiguous()
    selected = torch.tensor([[0, 2, 0, 2]] * 8, dtype=torch.long)
    hierarchy = Hierarchy(ep_size=ep_size, group_sizes=group_sizes, source="test", local_world_size=8)
    levels = group_sizes[:-1]

    fused = _extension().dual_map(
        selected.npu(),
        copy_slots.npu(),
        copy_counts.npu(),
        owner_ranks.npu(),
        slots_per_rank,
        1,
        ep_size,
        len(levels),
        levels[0] if levels else 1,
        levels[1] if len(levels) > 1 else 1,
        3,
        11,
    )
    expected = torch.stack(
        tuple(
            assign_tokens_to_copies_with_quota(
                selected,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=1,
                hierarchy=hierarchy,
                owner_slots=owners,
                step=3,
                layer_seed=11,
            ).physical_slots
            for layout in layouts
        ),
        dim=0,
    )

    assert torch.equal(fused.cpu(), expected)


@pytest.mark.parametrize(
    ("ep_size", "group_sizes"),
    ((16, (8, 16)), (32, (8, 32)), (64, (8, 16, 64))),
)
def test_quota_map_matches_eager_waterfill_and_exact_stats(ep_size, group_sizes):
    current, owners, slots_per_rank = _identity_layout(ep_size, 2, 1)
    candidate = current.clone()
    candidate[slots_per_rank + 2] = 0
    candidate[2 * slots_per_rank + 2] = 0
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), ()),
        source_rank=ep_size - 1,
        slots_per_rank=slots_per_rank,
    )
    selected = torch.tensor(
        (
            (0, 0, 2, 4),
            (0, 2, 4, 6),
            (0, 0, 0, 8),
            (0, 10, 12, 14),
            (2, 4, 6, 8),
            (0, 0, 2, 2),
            (0, 4, 8, 12),
            (0, 0, 0, 0),
        ),
        dtype=torch.long,
    )
    ordinals = torch.tensor((91, 7, 44, 3, 105, 18, 63, 29), dtype=torch.long)
    hierarchy = Hierarchy(ep_size=ep_size, group_sizes=group_sizes, source="test", local_world_size=8)
    levels = group_sizes[:-1]

    fused, group_counts, assignment_counts = _extension().quota_map(
        selected.npu(),
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals.npu(),
        slots_per_rank,
        ep_size - 1,
        ep_size,
        len(levels),
        levels[0] if levels else 1,
        levels[1] if len(levels) > 1 else 1,
        5,
        19,
    )
    expected = torch.stack(
        tuple(
            assign_tokens_to_copies_with_quota(
                selected,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=ep_size - 1,
                hierarchy=hierarchy,
                owner_slots=owners,
                token_ordinals=ordinals,
                step=5,
                layer_seed=19,
            ).physical_slots
            for layout in layouts
        ),
        dim=0,
    )

    expected_groups = []
    expected_assignments = []
    for physical in expected:
        ranks = torch.div(physical, slots_per_rank, rounding_mode="floor")
        expected_assignments.append(torch.bincount(ranks.reshape(-1), minlength=ep_size).to(torch.float32))
        packed_groups = []
        for size in (*levels, 1):
            groups = torch.div(ranks, size, rounding_mode="floor")
            hits = torch.zeros((selected.shape[0], ep_size // size), dtype=torch.bool)
            hits.scatter_(1, groups, True)
            packed_groups.append(hits.sum(dim=0).to(torch.float32))
        expected_groups.append(torch.cat(packed_groups))

    assert torch.equal(fused.cpu(), expected)
    assert torch.equal(group_counts.cpu(), torch.stack(expected_groups))
    assert torch.equal(assignment_counts.cpu(), torch.stack(expected_assignments))


def test_quota_map_multicore_preserves_composite_order_and_exact_stats():
    ep_size = 16
    current, owners, slots_per_rank = _identity_layout(ep_size, 2, 2)
    candidate = current.clone()
    for rank in (1, 2, 9):
        candidate[rank * slots_per_rank + 2] = 0
    for rank in (3, 10):
        candidate[rank * slots_per_rank + 3] = 1
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), ()),
        source_rank=15,
        slots_per_rank=slots_per_rank,
    )

    rows: list[list[int]] = []
    for token in range(257):
        row = [int((token * 7 + position * 13) % owners.numel()) for position in range(8)]
        row[0] = token % 2
        if token % 3 == 0:
            row[1] = row[0]
        if token % 5 == 0:
            row[2] = row[0]
        rows.append(row)
    selected = torch.tensor(rows, dtype=torch.long)
    ordinals = torch.tensor([(token * 97) % 1009 for token in range(len(rows))], dtype=torch.long)

    fused, group_counts, assignment_counts = _extension().quota_map(
        selected.npu(),
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals.npu(),
        slots_per_rank,
        15,
        ep_size,
        1,
        8,
        1,
        11,
        37,
    )
    hierarchy = Hierarchy(ep_size=ep_size, group_sizes=(8, 16), source="test", local_world_size=8)
    expected = torch.stack(
        tuple(
            assign_tokens_to_copies_with_quota(
                selected,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=15,
                hierarchy=hierarchy,
                owner_slots=owners,
                token_ordinals=ordinals,
                step=11,
                layer_seed=37,
            ).physical_slots
            for layout in layouts
        ),
        dim=0,
    )

    expected_groups = []
    expected_assignments = []
    for physical in expected:
        ranks = torch.div(physical, slots_per_rank, rounding_mode="floor")
        expected_assignments.append(torch.bincount(ranks.reshape(-1), minlength=ep_size).to(torch.float32))
        packed_groups = []
        for size in (8, 1):
            groups = torch.div(ranks, size, rounding_mode="floor")
            hits = torch.zeros((selected.shape[0], ep_size // size), dtype=torch.bool)
            hits.scatter_(1, groups, True)
            packed_groups.append(hits.sum(dim=0).to(torch.float32))
        expected_groups.append(torch.cat(packed_groups))

    assert torch.equal(fused.cpu(), expected)
    assert torch.equal(group_counts.cpu(), torch.stack(expected_groups))
    assert torch.equal(assignment_counts.cpu(), torch.stack(expected_assignments))


def test_quota_map_projects_configured_quota_by_largest_remainder():
    current, owners, slots_per_rank = _identity_layout(4, 1, 2)
    candidate = current.clone()
    candidate[slots_per_rank + 2] = 0
    candidate[2 * slots_per_rank + 2] = 0
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    policy = QuotaPolicyEntry(3, 0, (0, 1, 2), (1, 1, 1))
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), (policy,)),
        source_rank=3,
        slots_per_rank=slots_per_rank,
    )
    selected = torch.zeros((5, 1), dtype=torch.long)
    ordinals = torch.tensor((41, 5, 27, 1, 13), dtype=torch.long)

    fused, group_counts, assignment_counts = _extension().quota_map(
        selected.npu(),
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals.npu(),
        slots_per_rank,
        3,
        4,
        0,
        1,
        1,
        9,
        23,
    )
    expected = assign_tokens_to_copies_with_quota(
        selected,
        candidate,
        slots_per_rank=slots_per_rank,
        source_ranks=3,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        owner_slots=owners,
        token_ordinals=ordinals,
        quota_policy=(policy,),
        step=9,
        layer_seed=23,
    ).physical_slots

    assert torch.equal(fused[1].cpu(), expected)
    assert group_counts[1].cpu().tolist() == [2.0, 2.0, 1.0, 0.0]
    assert assignment_counts[1].cpu().tolist() == [2.0, 2.0, 1.0, 0.0]


def test_quota_map_uses_policy_for_communication_ties():
    current, owners, slots_per_rank = _identity_layout(4, 1, 1)
    candidate = current.clone()
    candidate[3] = 0
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    policy = QuotaPolicyEntry(3, 0, (0, 1), (0, 8))
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), (policy,)),
        source_rank=3,
        slots_per_rank=slots_per_rank,
    )
    selected = torch.zeros((8, 1), dtype=torch.long, device="npu")
    ordinals = torch.arange(8, dtype=torch.long, device="npu")

    fused, group_counts, assignment_counts = _extension().quota_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals,
        slots_per_rank,
        3,
        4,
        0,
        1,
        1,
        7,
        13,
    )

    assert fused[0].cpu().tolist() == [[0]] * 8
    assert fused[1].cpu().tolist() == [[3]] * 8
    assert group_counts.cpu().tolist() == [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]
    assert assignment_counts.cpu().tolist() == [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]


def test_quota_map_matches_eager_for_partial_quota_and_duplicate_topk():
    current, owners, slots_per_rank = _identity_layout(4, 1, 1)
    candidate = current.clone()
    candidate[3] = 0
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    policy = QuotaPolicyEntry(3, 0, (0, 1), (6, 10))
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), (policy,)),
        source_rank=3,
        slots_per_rank=slots_per_rank,
    )
    selected = torch.zeros((8, 2), dtype=torch.long)
    ordinals = torch.arange(8, dtype=torch.long)

    fused, group_counts, assignment_counts = _extension().quota_map(
        selected.npu(),
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals.npu(),
        slots_per_rank,
        3,
        4,
        0,
        1,
        1,
        7,
        13,
    )
    expected = assign_tokens_to_copies_with_quota(
        selected,
        candidate,
        slots_per_rank=slots_per_rank,
        source_ranks=3,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        owner_slots=owners,
        token_ordinals=ordinals,
        quota_policy=(policy,),
        step=7,
        layer_seed=13,
    ).physical_slots

    assert torch.equal(fused[1].cpu(), expected)
    assert torch.equal(fused[1, :, 0], fused[1, :, 1])
    assert group_counts[1].cpu().tolist() == [3.0, 5.0, 0.0, 0.0]
    assert assignment_counts[1].cpu().tolist() == [6.0, 10.0, 0.0, 0.0]


def test_dual_and_quota_map_handle_empty_routes_without_launching():
    current, owners, slots_per_rank = _identity_layout(4, 1, 1)
    layouts = torch.stack((current, current), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), ()),
        source_rank=0,
        slots_per_rank=slots_per_rank,
    )
    selected = torch.empty((0, 2), dtype=torch.long, device="npu")
    ordinals = torch.empty((0,), dtype=torch.long, device="npu")

    dual = _extension().dual_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    quota, quota_groups, quota_assignments = _extension().quota_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        ordinals,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )

    assert dual.shape == quota.shape == (2, 0, 2)
    assert quota_groups.cpu().tolist() == [[0.0] * 4, [0.0] * 4]
    assert quota_assignments.cpu().tolist() == [[0.0] * 4, [0.0] * 4]


def test_dual_and_quota_map_mark_invalid_value_domains():
    current, owners, slots_per_rank = _identity_layout(4, 1, 1)
    candidate = current.clone()
    candidate[3] = 0
    layouts = torch.stack((current, candidate), dim=0)
    owner_rows = owners.view(1, -1).expand(2, -1).contiguous()
    policy = QuotaPolicyEntry(3, 0, (0, 1), (1, 1))
    tables = build_quota_tensor_tables(
        layouts.npu(),
        owner_rows.npu(),
        ((), (policy,)),
        source_rank=3,
        slots_per_rank=slots_per_rank,
    )
    invalid_selected = torch.full((1, 1), owners.numel(), dtype=torch.long, device="npu")
    invalid_ordinals = torch.full((1,), -1, dtype=torch.long, device="npu")
    selected = torch.zeros((1, 1), dtype=torch.long, device="npu")
    ordinals = torch.zeros((1,), dtype=torch.long, device="npu")
    invalid_counts = tables.copy_counts.clone()
    invalid_counts[:, 0] = 0
    invalid_slots = tables.copy_slots.clone()
    ep_slot_limit = 4 * slots_per_rank
    invalid_slots[:, 0, 0] = ep_slot_limit
    invalid_owners = tables.owner_ranks.clone()
    invalid_owners[:, 0] = 4
    invalid_quotas = tables.quota_weights.clone()
    invalid_quotas[1, 0, 3, 0] = -1

    invalid_dual = _extension().dual_map(
        invalid_selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    invalid_count_dual = _extension().dual_map(
        selected,
        tables.copy_slots,
        invalid_counts,
        tables.owner_ranks,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    invalid_slot_dual = _extension().dual_map(
        selected,
        invalid_slots,
        tables.copy_counts,
        tables.owner_ranks,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    invalid_owner_dual = _extension().dual_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        invalid_owners,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    invalid_quota, _, _ = _extension().quota_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        tables.quota_weights,
        tables.quota_configured,
        invalid_ordinals,
        slots_per_rank,
        0,
        4,
        0,
        1,
        1,
        0,
        0,
    )
    invalid_weight_quota, _, _ = _extension().quota_map(
        selected,
        tables.copy_slots,
        tables.copy_counts,
        tables.owner_ranks,
        invalid_quotas,
        tables.quota_configured,
        ordinals,
        slots_per_rank,
        3,
        4,
        0,
        1,
        1,
        0,
        0,
    )

    assert invalid_dual.cpu().tolist() == [[[-1]], [[-1]]]
    assert invalid_count_dual.cpu().tolist() == [[[-1]], [[-1]]]
    assert invalid_slot_dual.cpu().tolist() == [[[-1]], [[-1]]]
    assert invalid_owner_dual.cpu().tolist() == [[[-1]], [[-1]]]
    assert invalid_quota.cpu().tolist() == [[[-1]], [[-1]]]
    assert invalid_weight_quota.cpu().tolist() == [[[0]], [[-1]]]


def test_swap_search_workspace_covers_platform_block_count():
    ep_size = 4
    layout, owners, slots_per_rank = _identity_layout(ep_size, 2, 0)
    sample_routes = torch.zeros((16_384, 1), dtype=torch.long, device="npu")
    sample_weights = torch.ones((sample_routes.shape[0],), dtype=torch.float32, device="npu")
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long, device="npu")

    updated_layout, updated_owners, actions, metadata = _extension().swap_search(
        sample_routes,
        sample_weights,
        assignment_counts,
        layout.npu(),
        owners.npu(),
        0,
        slots_per_rank,
        ep_size,
        2,
        2,
        1,
        1,
        128,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        True,
    )
    torch.npu.synchronize()

    assert torch.equal(updated_layout.cpu(), layout)
    assert torch.equal(updated_owners.cpu(), owners)
    assert actions.shape == (1, 5)
    assert metadata[0].cpu().item() == 0


def test_swap_select_tracks_gradient_deltas_for_ep64():
    ep_size = 64
    layout, owners, slots_per_rank = _identity_layout(ep_size, 2, 2)
    hot_experts = (96, 97)
    for rank in range(56):
        layout[rank * slots_per_rank + 2] = hot_experts[0]
        layout[rank * slots_per_rank + 3] = hot_experts[1]
    selected = torch.tensor(
        [[hot_experts[0]]] * 128 + [[hot_experts[1]]] * 128,
        dtype=torch.long,
        device="npu",
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    gradient_inter = LinkCost(alpha=25.0, beta=0.0)
    perf_model = HierMoEPerfModel(
        a2a=zero,
        inter=(zero, zero),
        intra=zero,
        source="test-gradient-delta-capacity",
        state_move=PeerTransferCost(intra=zero, inter=zero),
        gradient_sync=GradientSyncCost(
            gather=PeerTransferCost(intra=zero, inter=gradient_inter),
            scatter=PeerTransferCost(intra=zero, inter=gradient_inter),
        ),
        schema_version=2,
    )
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(8, ep_size), source="test", local_world_size=8),
        perf_model=perf_model,
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        communication_scale=1.0,
        forward_compute_per_assignment=1.0,
        reducer=lambda value: value,
        expert_state_bytes=(0,) * owners.numel(),
        expert_gradient_bytes=tuple(1024 + logical for logical in range(owners.numel())),
    )
    layout = layout.npu()
    owners = owners.npu()
    token_hits = torch.nn.functional.one_hot(selected, num_classes=owners.numel()).amax(dim=1).float()
    stats = planner._initial_swap_stats(token_hits, selected, owners)

    fused = planner._fused_swap_select(
        stats,
        layout,
        owners,
        max_swaps=1,
        sample_routes=selected,
        sample_weights=torch.ones((selected.shape[0],), dtype=torch.float32, device=selected.device),
    )
    assert fused is not None
    fused_layout, fused_owners, fused_actions, metadata, _final_base = fused
    eager_layout, eager_owners, eager_actions = _eager_aggregate_swap_select(
        planner,
        selected,
        layout,
        owners,
        max_swaps=1,
    )
    torch.npu.synchronize()

    accepted = int(metadata[0].cpu().item())
    assert accepted == len(eager_actions) == 0
    assert fused_actions[0].cpu().tolist() == [-1] * 5
    assert torch.equal(fused_layout.cpu(), eager_layout.cpu())
    assert torch.equal(fused_owners.cpu(), eager_owners.cpu())


def test_swap_select_p4_recomputes_cooccurrence_after_each_accepted_swap():
    layout, owners, slots_per_rank = _identity_layout(4, 2, 0)
    selected = torch.tensor(
        [
            [5, 3, 7, 7],
            [4, 0, 0, 1],
            [1, 1, 5, 7],
            [1, 4, 4, 1],
            [5, 3, 4, 7],
            [1, 6, 5, 3],
            [4, 2, 3, 2],
            [0, 4, 7, 1],
            [1, 2, 2, 0],
            [3, 6, 4, 7],
            [7, 5, 1, 5],
            [1, 7, 5, 3],
            [3, 5, 2, 5],
            [6, 0, 1, 2],
            [7, 3, 0, 0],
            [6, 1, 4, 1],
            [3, 1, 4, 5],
        ],
        dtype=torch.long,
        device="npu",
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel(
            a2a=LinkCost(alpha=0.0, beta=1.0),
            inter=(zero,),
            intra=zero,
            source="test-p4-cooccurrence",
        ),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=1.0,
        forward_compute_per_assignment=0.0,
        backward_compute_per_assignment=0.0,
        reducer=lambda value: value,
    )
    layout = layout.npu()
    owners = owners.npu()
    token_hits = torch.nn.functional.one_hot(selected, num_classes=owners.numel()).amax(dim=1).float()
    stats = planner._initial_swap_stats(token_hits, selected, owners)

    fused = planner._fused_swap_select(
        stats,
        layout,
        owners,
        max_swaps=4,
        sample_routes=selected,
        sample_weights=torch.ones((selected.shape[0],), dtype=torch.float32, device=selected.device),
    )
    assert fused is not None
    fused_layout, fused_owners, fused_actions, metadata, _final_base = fused
    eager_layout, eager_owners, eager_actions = _eager_aggregate_swap_select(
        planner,
        selected,
        layout,
        owners,
        max_swaps=4,
    )
    torch.npu.synchronize()

    accepted = int(metadata[0].cpu().item())
    assert [(action.src_logical, action.dst_logical) for action in eager_actions] == [(2, 5)]
    assert accepted == 1
    assert fused_actions[:accepted, :2].cpu().tolist() == [[2, 5]]
    assert torch.equal(fused_layout.cpu(), eager_layout.cpu())
    assert torch.equal(fused_owners.cpu(), eager_owners.cpu())


def test_swap_select_p4_rejects_non_improving_global_summary_suffix():
    layout, owners, slots_per_rank = _identity_layout(4, 2, 0)
    selected = torch.tensor(
        [[0, 1]] * 96 + [[0, 2]] * 64 + [[1, 3]] * 48 + [[4, 5]] * 8 + [[6, 7]] * 8,
        dtype=torch.long,
        device="npu",
    )

    def gather_fixed(payload: torch.Tensor) -> torch.Tensor:
        return payload.unsqueeze(0).expand(4, -1).clone()

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test", local_world_size=2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=64,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        communication_scale=1.0,
        forward_compute_per_assignment=0.25,
        reducer=lambda value: value * 4,
        gather_fixed=gather_fixed,
        route_sample_size=256,
    )
    layout = layout.npu()
    owners = owners.npu()
    summary = planner._build_device_planning_summary(
        selected,
        owners,
        source_rank=0,
        step=3,
        layer_seed=11,
        fused_capable=True,
    )

    fused = planner._fused_swap_select(
        summary.swap_stats,
        layout,
        owners,
        max_swaps=4,
        sample_routes=summary.route.sample_routes,
        sample_weights=summary.route.sample_weights,
    )
    assert fused is not None
    fused_layout, fused_owners, fused_actions, metadata, _final_base = fused
    eager_layout, eager_owners, eager_actions, _, _ = planner._plan_swaps(
        summary.route,
        layout,
        owners,
        max_swaps=4,
        step=3,
        layer_seed=11,
    )
    torch.npu.synchronize()

    accepted = int(metadata[0].cpu().item())
    assert [(action.src_logical, action.dst_logical) for action in eager_actions] == [(0, 3)]
    assert accepted == 1
    assert fused_actions[:accepted, :2].cpu().tolist() == [[0, 3]]
    assert torch.equal(fused_layout.cpu(), eager_layout.cpu())
    assert torch.equal(fused_owners.cpu(), eager_owners.cpu())


@pytest.mark.parametrize("old_logical", [-1, 1])
def test_replica_match_cover_and_retarget_follow_preaggregated_deltas(old_logical: int):
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, old_logical, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4]], dtype=torch.long)
    base_counts = torch.zeros((ep_size,), dtype=torch.float32)
    assignment_loads = torch.tensor([10.0, 0.0])
    add_group_deltas = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_assignment_deltas = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_assignment_deltas[0, 1] = torch.tensor([-6.0, 6.0])
    remove_group_deltas = torch.zeros((ep_size, 1, ep_size), dtype=torch.float32)
    remove_assignment_deltas = torch.zeros((ep_size, 1, ep_size), dtype=torch.float32)

    updated, actions, gains, selected, matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=base_counts,
        assignment_loads=assignment_loads,
        add_group_deltas=add_group_deltas,
        add_assignment_deltas=add_assignment_deltas,
        remove_group_deltas=remove_group_deltas,
        remove_assignment_deltas=remove_assignment_deltas,
        candidate_experts=torch.tensor([1, 0, 0], dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=1,
        slots_per_rank=slots_per_rank,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 1
    assert metadata[5].cpu().item() == 1
    assert actions[0].cpu().tolist() == [2, 0, 4, 0, old_logical]
    assert gains[0].cpu().item() == pytest.approx(4.0)
    assert selected[1, 0].cpu().item() == 2
    assert matrix[1, 0, 2].cpu().item() == pytest.approx(4.0)
    expected = layout.clone()
    expected[4] = 0
    assert torch.equal(updated.cpu(), expected)


def test_replica_match_scores_sampled_dedup_communication_delta():
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, -1, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4]], dtype=torch.long)
    add_group_deltas = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_group_deltas[0, 1] = torch.tensor([-6.0, 6.0])

    updated, actions, gains, _selected, matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.tensor([10.0, 0.0]),
        assignment_loads=torch.zeros((ep_size,), dtype=torch.float32),
        add_group_deltas=add_group_deltas,
        add_assignment_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        remove_group_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        candidate_experts=torch.tensor([1, 0, 0], dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=1,
        slots_per_rank=slots_per_rank,
        payload_bytes=1,
        a2a_beta=1.0,
        compute_per_assignment=0.0,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 1
    assert actions[0].cpu().tolist() == [2, 0, 4, 0, -1]
    assert gains[0].cpu().item() == pytest.approx(32.0)
    assert matrix[1, 0, 2].cpu().item() == pytest.approx(32.0)
    expected = layout.clone()
    expected[4] = 0
    assert torch.equal(updated.cpu(), expected)


def test_replica_match_empty_accounts_for_gradient_sync_savings():
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, 0, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4]], dtype=torch.long)
    zeros_by_group = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    zeros_by_slot = torch.zeros((ep_size, 1, ep_size), dtype=torch.float32)

    updated, actions, gains, selected, matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((ep_size,), dtype=torch.float32),
        assignment_loads=torch.zeros((ep_size,), dtype=torch.float32),
        add_group_deltas=zeros_by_group,
        add_assignment_deltas=zeros_by_group.clone(),
        remove_group_deltas=zeros_by_slot,
        remove_assignment_deltas=zeros_by_slot.clone(),
        candidate_experts=torch.zeros((owners.numel(),), dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.tensor([100, 0, 0], dtype=torch.long),
        max_actions=1,
        slots_per_rank=slots_per_rank,
        compute_per_assignment=0.0,
        gradient_inter_beta=0.01,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 1
    assert actions[0].cpu().tolist() == [1, -1, 4, -1, 0]
    assert gains[0].cpu().item() == pytest.approx(2.0)
    assert selected[1, 0].cpu().item() == 1
    assert matrix[1, 0, 1].cpu().item() == pytest.approx(2.0)
    expected = layout.clone()
    expected[4] = -1
    assert torch.equal(updated.cpu(), expected)


def test_replica_match_rejects_cover_when_state_move_exceeds_compute_gain():
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, -1, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4]], dtype=torch.long)
    add_assignment = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_assignment[0, 1] = torch.tensor([-6.0, 6.0])

    updated, actions, gains, selected, matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((ep_size,), dtype=torch.float32),
        assignment_loads=torch.tensor([10.0, 0.0]),
        add_group_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        add_assignment_deltas=add_assignment,
        remove_group_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        candidate_experts=torch.tensor([1, 0, 0], dtype=torch.int32),
        state_bytes=torch.tensor([10, 0, 0], dtype=torch.long),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=1,
        slots_per_rank=slots_per_rank,
        state_inter_beta=1.0,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 0
    assert actions[0].cpu().tolist() == [-1] * 5
    assert gains[0].cpu().item() == 0.0
    assert selected[1, 0].cpu().item() == 0
    assert matrix[1, 0, 2].cpu().item() < -1.0e30
    assert torch.equal(updated.cpu(), layout)


def test_replica_match_hungarian_uses_one_shared_expert_column_deterministically():
    ep_size = 2
    slots_per_rank = 4
    layout = torch.tensor([0, 1, -1, -1, 2, -1, -1, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 4], dtype=torch.long)
    redundant_slots = torch.tensor([[2, 3], [5, 6]], dtype=torch.long)
    add_assignment = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_assignment[0, 1] = torch.tensor([-6.0, 6.0])
    kwargs = dict(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((ep_size,), dtype=torch.float32),
        assignment_loads=torch.tensor([10.0, 0.0]),
        add_group_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        add_assignment_deltas=add_assignment,
        remove_group_deltas=torch.zeros((ep_size, 2, ep_size), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 2, ep_size), dtype=torch.float32),
        candidate_experts=torch.tensor([1, 0, 0], dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=2,
        slots_per_rank=slots_per_rank,
    )

    first = _run_replica_match(**kwargs)
    second = _run_replica_match(**kwargs)
    torch.npu.synchronize()
    first_cpu = tuple(tensor.cpu() for tensor in first)
    second_cpu = tuple(tensor.cpu() for tensor in second)

    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first_cpu, second_cpu, strict=True))
    updated, actions, gains, selected, matrix, metadata = first_cpu
    assert metadata[0].item() == 1
    assert actions[0, 0].item() == 2
    assert gains[0].item() == pytest.approx(4.0)
    expert_column = 2 * redundant_slots.shape[1]
    assert (selected[1] == expert_column).sum().item() == 1
    assert (updated == 0).sum().item() == 2
    assert torch.allclose(matrix[1, :, expert_column], torch.full((2,), 4.0))


def test_replica_match_truncates_after_rank_local_matching_by_slot_id():
    ep_size = 3
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, -1, -1, 3, -1, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3, 6], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4], [7]], dtype=torch.long)
    add_assignment = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    add_assignment[0, 1] = torch.tensor([-8.0, 8.0, 0.0])
    add_assignment[1, 2] = torch.tensor([-8.0, 0.0, 8.0])

    updated, actions, gains, _selected, _matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((ep_size,), dtype=torch.float32),
        assignment_loads=torch.tensor([20.0, 0.0, 0.0]),
        add_group_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        add_assignment_deltas=add_assignment,
        remove_group_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        candidate_experts=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=1,
        slots_per_rank=slots_per_rank,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 1
    assert metadata[6].cpu().item() == 2
    assert actions[0].cpu().tolist() == [2, 0, 4, 0, -1]
    assert gains[0].cpu().item() == pytest.approx(8.0)
    expected = layout.clone()
    expected[4] = 0
    assert torch.equal(updated.cpu(), expected)


def test_replica_match_enforces_global_eight_copy_limit_and_keeps_other_actions():
    ep_size = 16
    layout, owners, slots_per_rank = _identity_layout(ep_size, 1, 1)
    redundant_slots = torch.arange(ep_size, dtype=torch.long).mul(slots_per_rank).add(1).view(ep_size, 1)
    add_assignment = torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32)
    for rank in range(1, ep_size):
        add_assignment[0, rank, 0] = -50.0
        add_assignment[0, rank, rank] = 50.0
    for rank in range(ep_size):
        if rank == 1:
            continue
        destination = 2 if rank == 0 else rank
        add_assignment[1, rank, 0] = -40.0
        add_assignment[1, rank, destination] = 40.0

    updated, actions, _gains, _selected, _matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((ep_size,), dtype=torch.float32),
        assignment_loads=torch.tensor([100.0] + [0.0] * (ep_size - 1)),
        add_group_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        add_assignment_deltas=add_assignment,
        remove_group_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        candidate_experts=torch.tensor([1, 1] + [0] * (owners.numel() - 2), dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=ep_size,
        slots_per_rank=slots_per_rank,
    )
    torch.npu.synchronize()

    accepted = int(metadata[0].cpu().item())
    updated_cpu = updated.cpu()
    actions_cpu = actions[:accepted].cpu()
    copy_counts = torch.bincount(updated_cpu[updated_cpu >= 0], minlength=owners.numel())
    assert accepted == 8
    assert copy_counts[0].item() == 8
    assert copy_counts[1].item() == 2
    assert copy_counts.max().item() <= 8
    assert (actions_cpu[:, 3] == 0).sum().item() == 7
    assert (actions_cpu[:, 3] == 1).sum().item() == 1


def test_replica_match_ep64_dynamic_shape_and_invalid_slot_are_safe():
    ep_size = 64
    layout, owners, slots_per_rank = _identity_layout(ep_size, 1, 1)
    redundant_slots = torch.arange(ep_size, dtype=torch.long).mul(slots_per_rank).add(1).view(ep_size, 1)
    redundant_slots[-1, 0] = layout.numel() + 17
    level_sizes = (8, 1)
    total_groups = sum(ep_size // size for size in level_sizes)

    updated, actions, gains, selected, matrix, metadata = _run_replica_match(
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        base_counts=torch.zeros((total_groups,), dtype=torch.float32),
        assignment_loads=torch.ones((ep_size,), dtype=torch.float32),
        add_group_deltas=torch.zeros((owners.numel(), ep_size, total_groups), dtype=torch.float32),
        add_assignment_deltas=torch.zeros((owners.numel(), ep_size, ep_size), dtype=torch.float32),
        remove_group_deltas=torch.zeros((ep_size, 1, total_groups), dtype=torch.float32),
        remove_assignment_deltas=torch.zeros((ep_size, 1, ep_size), dtype=torch.float32),
        candidate_experts=torch.ones((owners.numel(),), dtype=torch.int32),
        state_bytes=torch.zeros_like(owners),
        gradient_bytes=torch.zeros_like(owners),
        max_actions=ep_size,
        slots_per_rank=slots_per_rank,
        local_world_size=8,
        level_sizes=level_sizes,
    )
    torch.npu.synchronize()

    assert metadata[0].cpu().item() == 0
    assert metadata[3].cpu().item() == owners.numel() + 2
    assert torch.equal(updated.cpu(), layout)
    assert actions.shape == (ep_size, 5)
    assert gains.shape == (ep_size,)
    assert selected.shape == (ep_size, 1)
    assert matrix.shape == (ep_size, 1, owners.numel() + 2)


def test_replica_project_matches_eager_single_add_and_remove_with_duplicate_topk():
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 1, -1, 2, 0, -1], dtype=torch.long)
    owners = torch.tensor([0, 1, 3], dtype=torch.long)
    redundant_slots = torch.tensor([[2], [4]], dtype=torch.long)
    routes = torch.tensor(
        [
            [0, 0, 2],
            [0, 1, 2],
            [2, 2, 1],
            [0, 2, 2],
            [1, 1, 0],
            [2, 1, 0],
        ],
        dtype=torch.long,
    )
    sources = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    ordinals = torch.tensor([5, 2, 8, 3, 9, 1], dtype=torch.long)
    weights = torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32)
    multiplicity = _sample_multiplicity_for_test(routes)
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long)
    for source in range(ep_size):
        source_routes = routes[sources == source].reshape(-1)
        factor = int(weights[sources == source][0].item())
        assignment_counts[source] = torch.bincount(source_routes, minlength=owners.numel()) * factor

    summary = RouteSummary(
        token_counts=torch.tensor([6, 9], dtype=torch.long),
        assignment_counts=assignment_counts,
        sample_routes=routes,
        sample_ordinals=ordinals,
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=weights,
        sample_sources=sources,
        sample_digest="test",
        sample_multiplicity=multiplicity,
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(ep_size,), source="test"),
        perf_model=HierMoEPerfModel(a2a=zero, inter=(zero,), intra=zero, source="test"),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        reducer=lambda value: value,
    )
    step = 7
    layer_seed = 19

    projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=sources,
        sample_ordinals=ordinals,
        assignment_counts=assignment_counts,
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        candidate_experts=torch.tensor([1, 0, 1], dtype=torch.int32),
        slots_per_rank=slots_per_rank,
        seed_base_counts=torch.full((ep_size,), 12_345.0),
        step=step,
        layer_seed=layer_seed,
    )
    torch.npu.synchronize()
    base_counts, assignment_loads, add_groups, add_assignments, remove_groups, remove_assignments = (
        tensor.cpu() for tensor in projected
    )

    baseline_mapping = assign_tokens_to_copies_with_quota(
        summary.sample_routes,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=sources,
        hierarchy=planner.hierarchy,
        owner_slots=owners,
        token_ordinals=ordinals,
        token_weights=weights,
        step=step,
        layer_seed=layer_seed,
    )
    eager_counts, eager_assignments = _eager_replica_projection(
        planner, summary, layout, owners, step=step, layer_seed=layer_seed
    )
    assert torch.allclose(base_counts, eager_counts)
    assert torch.allclose(assignment_loads, eager_assignments)

    add_layout = layout.clone()
    add_layout[2] = 2
    add_counts, add_loads = _eager_independent_replica_projection(
        planner,
        summary,
        add_layout,
        owners,
        baseline_mapping.physical_slots,
        logical=2,
        step=step,
        layer_seed=layer_seed,
    )
    assert torch.allclose(add_groups[2, 0], add_counts - eager_counts)
    assert torch.allclose(add_assignments[2, 0], add_loads - eager_assignments)
    assert torch.count_nonzero(add_groups[1]).item() == 0
    assert torch.count_nonzero(add_assignments[1]).item() == 0
    assert torch.count_nonzero(add_groups[0]).item() == 0
    assert torch.count_nonzero(add_assignments[0]).item() == 0
    assert torch.count_nonzero(add_groups[2, 1]).item() == 0
    assert torch.count_nonzero(add_assignments[2, 1]).item() == 0

    remove_layout = layout.clone()
    remove_layout[4] = -1
    remove_counts, remove_loads = _eager_independent_replica_projection(
        planner,
        summary,
        remove_layout,
        owners,
        baseline_mapping.physical_slots,
        logical=0,
        step=step,
        layer_seed=layer_seed,
    )
    assert torch.allclose(remove_groups[1, 0], remove_counts - eager_counts)
    assert torch.allclose(remove_assignments[1, 0], remove_loads - eager_assignments)
    assert torch.count_nonzero(remove_groups[0, 0]).item() == 0
    assert torch.count_nonzero(remove_assignments[0, 0]).item() == 0


@pytest.mark.parametrize(
    ("ep_size", "group_sizes", "level_sizes"),
    (
        (16, (16,), (1,)),
        (32, (8, 32), (8, 1)),
        (64, (8, 16, 64), (8, 16, 1)),
    ),
)
def test_replica_project_independent_edges_match_eager_dynamic_ep(ep_size, group_sizes, level_sizes):
    layout, owners, slots_per_rank = _identity_layout(ep_size, 2, 1)
    one_copy_layout = layout.clone()
    one_copy_owners = owners.clone()
    for pair in range({16: 0, 32: 1, 64: 4}[ep_size]):
        lhs = 2 * pair
        rhs = lhs + 1
        lhs_slot = int(one_copy_owners[lhs].item())
        rhs_slot = int(one_copy_owners[rhs].item())
        one_copy_layout[lhs_slot] = rhs
        one_copy_layout[rhs_slot] = lhs
        one_copy_owners[lhs] = rhs_slot
        one_copy_owners[rhs] = lhs_slot
    redundant_slots = (
        torch.arange(ep_size, dtype=torch.long).mul(slots_per_rank).add(slots_per_rank - 1).view(ep_size, 1)
    )
    layout[redundant_slots[1, 0]] = 0
    routes = []
    sources = []
    weights = []
    # Leave the final source rank empty to cover source-local dynamic shapes as
    # well as the fully populated ranks in the same projection.
    for source in range(ep_size - 1):
        routes.extend(
            (
                (0, 0, (2 * source + 3) % owners.numel()),
                (1, (2 * source + 5) % owners.numel(), 0),
            )
        )
        sources.extend((source, source))
        weights.extend((1.0, 2.0))
    routes = torch.tensor(routes, dtype=torch.long)
    sources = torch.tensor(sources, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)
    ordinals = torch.arange(routes.shape[0], dtype=torch.long).mul(7).remainder(routes.shape[0] + 11)
    multiplicity = _sample_multiplicity_for_test(routes)
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long)
    for token, (source, weight) in enumerate(zip(sources.tolist(), weights.tolist(), strict=True)):
        assignment_counts[source].add_(torch.bincount(routes[token], minlength=owners.numel()).mul_(int(weight)))
    summary = RouteSummary(
        token_counts=torch.bincount(sources, weights=weights, minlength=ep_size).to(torch.long),
        assignment_counts=assignment_counts,
        sample_routes=routes,
        sample_ordinals=ordinals,
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=weights,
        sample_sources=sources,
        sample_digest="dynamic-independent-edge",
        sample_multiplicity=multiplicity,
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=group_sizes, source="test", local_world_size=8),
        perf_model=HierMoEPerfModel(
            a2a=zero,
            inter=tuple(zero for _ in group_sizes[:-1]) or (zero,),
            intra=zero,
            source="test",
        ),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        reducer=lambda value: value,
    )
    candidates = torch.zeros((owners.numel(),), dtype=torch.int32)
    candidates[0] = 1
    one_copy_projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=sources,
        sample_ordinals=ordinals,
        assignment_counts=assignment_counts,
        layout=one_copy_layout,
        owners=one_copy_owners,
        redundant_slots=redundant_slots,
        candidate_experts=candidates,
        slots_per_rank=slots_per_rank,
        level_sizes=level_sizes,
        step=11,
        layer_seed=23,
    )
    one_copy_counts, one_copy_assignments = _eager_replica_projection(
        planner,
        summary,
        one_copy_layout,
        one_copy_owners,
        step=11,
        layer_seed=23,
    )
    assert torch.equal(one_copy_projected[0].cpu(), one_copy_counts)
    assert torch.equal(one_copy_projected[1].cpu(), one_copy_assignments)

    projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=sources,
        sample_ordinals=ordinals,
        assignment_counts=assignment_counts,
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        candidate_experts=candidates,
        slots_per_rank=slots_per_rank,
        level_sizes=level_sizes,
        step=11,
        layer_seed=23,
    )
    torch.npu.synchronize()
    base_counts, assignment_loads, add_groups, add_assignments, remove_groups, remove_assignments = (
        tensor.cpu() for tensor in projected
    )
    baseline_mapping = assign_tokens_to_copies_with_quota(
        routes,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=sources,
        hierarchy=planner.hierarchy,
        owner_slots=owners,
        token_ordinals=ordinals,
        token_weights=weights,
        step=11,
        layer_seed=23,
    )
    eager_counts, eager_assignments = _eager_replica_projection(
        planner, summary, layout, owners, step=11, layer_seed=23
    )
    assert torch.equal(base_counts, eager_counts)
    assert torch.equal(assignment_loads, eager_assignments)

    add_layout = layout.clone()
    add_layout[redundant_slots[2, 0]] = 0
    add_counts, add_loads = _eager_independent_replica_projection(
        planner,
        summary,
        add_layout,
        owners,
        baseline_mapping.physical_slots,
        logical=0,
        step=11,
        layer_seed=23,
    )
    assert torch.equal(add_groups[0, 2], add_counts - eager_counts)
    assert torch.equal(add_assignments[0, 2], add_loads - eager_assignments)

    remove_layout = layout.clone()
    remove_layout[redundant_slots[1, 0]] = -1
    remove_counts, remove_loads = _eager_independent_replica_projection(
        planner,
        summary,
        remove_layout,
        owners,
        baseline_mapping.physical_slots,
        logical=0,
        step=11,
        layer_seed=23,
    )
    assert torch.equal(remove_groups[1, 0], remove_counts - eager_counts)
    assert torch.equal(remove_assignments[1, 0], remove_loads - eager_assignments)
    assert torch.count_nonzero(add_groups[0, :2]).item() == 0
    assert torch.count_nonzero(add_assignments[0, :2]).item() == 0
    assert torch.count_nonzero(add_groups[1:]).item() == 0
    assert torch.count_nonzero(add_assignments[1:]).item() == 0


def test_replica_project_scores_multiple_expert_destination_edges_independently():
    ep_size = 4
    layout, owners, slots_per_rank = _identity_layout(ep_size, 1, 1)
    redundant_slots = (
        torch.arange(ep_size, dtype=torch.long).mul(slots_per_rank).add(slots_per_rank - 1).view(ep_size, 1)
    )
    routes = torch.tensor(
        [
            [0, 0, 2],
            [0, 1, 3],
            [1, 1, 2],
            [1, 0, 3],
        ]
        * 4,
        dtype=torch.long,
    )
    sources = torch.arange(routes.shape[0], dtype=torch.long).remainder(ep_size)
    weights = torch.arange(routes.shape[0], dtype=torch.float32).remainder(3).add(1)
    ordinals = torch.arange(routes.shape[0], dtype=torch.long).mul(11).remainder(37)
    multiplicity = _sample_multiplicity_for_test(routes)
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long)
    for token, source in enumerate(sources.tolist()):
        assignment_counts[source].add_(
            torch.bincount(routes[token], minlength=owners.numel()).mul_(int(weights[token].item()))
        )
    summary = RouteSummary(
        token_counts=torch.bincount(sources, weights=weights, minlength=ep_size).to(torch.long),
        assignment_counts=assignment_counts,
        sample_routes=routes,
        sample_ordinals=ordinals,
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=weights,
        sample_sources=sources,
        sample_digest="multi-edge-wave",
        sample_multiplicity=multiplicity,
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(ep_size,), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel(a2a=zero, inter=(zero,), intra=zero, source="test"),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        reducer=lambda value: value,
    )
    projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=sources,
        sample_ordinals=ordinals,
        assignment_counts=assignment_counts,
        layout=layout,
        owners=owners,
        redundant_slots=redundant_slots,
        candidate_experts=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        slots_per_rank=slots_per_rank,
        step=13,
        layer_seed=29,
    )
    torch.npu.synchronize()
    base_counts, assignment_loads, add_groups, add_assignments, _, _ = (
        tensor.cpu() for tensor in projected
    )
    baseline_mapping = assign_tokens_to_copies_with_quota(
        routes,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=sources,
        hierarchy=planner.hierarchy,
        owner_slots=owners,
        token_ordinals=ordinals,
        token_weights=weights,
        step=13,
        layer_seed=29,
    )
    eager_counts, eager_assignments = _eager_replica_projection(
        planner, summary, layout, owners, step=13, layer_seed=29
    )
    assert torch.equal(base_counts, eager_counts)
    assert torch.equal(assignment_loads, eager_assignments)

    for logical in (0, 1):
        owner_rank = int(owners[logical].item()) // slots_per_rank
        for destination in range(ep_size):
            if destination == owner_rank:
                assert torch.count_nonzero(add_groups[logical, destination]).item() == 0
                assert torch.count_nonzero(add_assignments[logical, destination]).item() == 0
                continue
            candidate_layout = layout.clone()
            candidate_layout[redundant_slots[destination, 0]] = logical
            candidate_counts, candidate_assignments = _eager_independent_replica_projection(
                planner,
                summary,
                candidate_layout,
                owners,
                baseline_mapping.physical_slots,
                logical=logical,
                step=13,
                layer_seed=29,
            )
            assert torch.equal(add_groups[logical, destination], candidate_counts - eager_counts)
            assert torch.equal(
                add_assignments[logical, destination], candidate_assignments - eager_assignments
            )
    assert torch.count_nonzero(add_groups[2:]).item() == 0
    assert torch.count_nonzero(add_assignments[2:]).item() == 0


def test_replica_project_one_copy_keeps_exact_assignment_for_unsampled_expert():
    slots_per_rank = 2
    layout = torch.tensor([0, -1, 1, -1], dtype=torch.long)
    owners = torch.tensor([0, 2], dtype=torch.long)
    routes = torch.tensor([[0], [0]], dtype=torch.long)
    multiplicity = torch.ones_like(routes)
    weights = torch.ones((2,), dtype=torch.float32)
    projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=torch.tensor([0, 1], dtype=torch.long),
        sample_ordinals=torch.tensor([0, 1], dtype=torch.long),
        assignment_counts=torch.tensor([[1, 0], [1, 7]], dtype=torch.long),
        layout=layout,
        owners=owners,
        redundant_slots=torch.tensor([[1], [3]], dtype=torch.long),
        candidate_experts=torch.tensor([0, 1], dtype=torch.int32),
        slots_per_rank=slots_per_rank,
    )
    torch.npu.synchronize()

    base_counts, assignment_loads, add_groups, add_assignments, _, _ = (
        tensor.cpu() for tensor in projected
    )
    assert torch.equal(base_counts, torch.tensor([2.0, 0.0]))
    assert torch.equal(assignment_loads, torch.tensor([2.0, 7.0]))
    assert torch.count_nonzero(add_groups[1]).item() == 0
    assert torch.count_nonzero(add_assignments[1]).item() == 0


def test_replica_project_same_rank_duplicate_ignores_one_copy_seed():
    ep_size = 2
    slots_per_rank = 3
    layout = torch.tensor([0, 0, -1, 1, -1, -1], dtype=torch.long)
    owners = torch.tensor([0, 3], dtype=torch.long)
    routes = torch.tensor([[0, 1], [0, 0], [1, 0]], dtype=torch.long)
    multiplicity = _sample_multiplicity_for_test(routes)
    sources = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.ones((3,), dtype=torch.float32)
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long)
    for token, source in enumerate(sources.tolist()):
        assignment_counts[source].add_(torch.bincount(routes[token], minlength=owners.numel()))
    summary = RouteSummary(
        token_counts=torch.tensor([2, 1], dtype=torch.long),
        assignment_counts=assignment_counts,
        sample_routes=routes,
        sample_ordinals=torch.arange(routes.shape[0], dtype=torch.long),
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=weights,
        sample_sources=sources,
        sample_digest="same-rank-duplicate-fallback",
        sample_multiplicity=multiplicity,
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(ep_size,), source="test"),
        perf_model=HierMoEPerfModel(a2a=zero, inter=(zero,), intra=zero, source="test"),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        reducer=lambda value: value,
    )
    projected = _run_replica_project(
        sample_routes=routes,
        sample_multiplicity=multiplicity,
        sample_weights=weights,
        sample_sources=sources,
        sample_ordinals=summary.sample_ordinals,
        assignment_counts=assignment_counts,
        layout=layout,
        owners=owners,
        redundant_slots=torch.tensor([[1], [4]], dtype=torch.long),
        candidate_experts=torch.tensor([0, 1], dtype=torch.int32),
        slots_per_rank=slots_per_rank,
        seed_base_counts=torch.full((ep_size,), 54_321.0),
    )
    eager_counts, eager_assignments = _eager_replica_projection(
        planner,
        summary,
        layout,
        owners,
        step=0,
        layer_seed=0,
    )
    assert torch.equal(projected[0].cpu(), eager_counts)
    assert torch.equal(projected[1].cpu(), eager_assignments)


def test_core_moe_one_shot_replica_matches_independent_eager_action_and_layout():
    ep_size = 4
    layout, owners, slots_per_rank = _identity_layout(ep_size, 1, 1)
    routes = torch.zeros((128, 1), dtype=torch.long)
    sources = torch.arange(ep_size, dtype=torch.long).repeat_interleave(32)
    ordinals = torch.arange(routes.shape[0], dtype=torch.long)
    weights = torch.ones((routes.shape[0],), dtype=torch.float32)
    assignment_counts = torch.zeros((ep_size, owners.numel()), dtype=torch.long)
    assignment_counts[:, 0] = 32
    summary = RouteSummary(
        token_counts=torch.full((ep_size,), 32, dtype=torch.long),
        assignment_counts=assignment_counts,
        sample_routes=routes,
        sample_ordinals=ordinals,
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=weights,
        sample_sources=sources,
        sample_digest="one-shot-independent-parity",
        sample_multiplicity=torch.ones_like(routes),
        sample_multiplicity_is_canonical=True,
    )
    zero = LinkCost(alpha=0.0, beta=0.0)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=ep_size, group_sizes=(ep_size,), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel(a2a=zero, inter=(zero,), intra=zero, source="test"),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=1.0,
        reducer=lambda value: value,
    )
    current = planner._score_sample_layout(summary, layout, owners, actions=(), step=5, layer_seed=17)
    eager_layout, eager_actions, _eager_score, eager_count = planner._plan_replicas(
        summary,
        layout.clone(),
        owners,
        [],
        current,
        max_replicas=1,
        step=5,
        layer_seed=17,
    )
    swap_metadata = torch.tensor(
        (0, current.cost.peak_communication_rank, current.cost.peak_compute_rank),
        dtype=torch.int32,
        device="npu",
    )
    sampled_counts, _ = planner._local_weighted_stats(current.mapping.physical_slots, weights)
    fused_layout, fused_actions, fused_count = planner._fused_plan_replicas(
        _extension(),
        summary,
        layout.npu(),
        owners.npu(),
        swap_metadata,
        torch.cat(sampled_counts).npu(),
        max_replicas=1,
        step=5,
        layer_seed=17,
    )
    torch.npu.synchronize()

    assert eager_count == fused_count == 1
    assert torch.equal(eager_layout, fused_layout.cpu())
    assert [action.format() for action in eager_actions] == [action.format() for action in fused_actions]
