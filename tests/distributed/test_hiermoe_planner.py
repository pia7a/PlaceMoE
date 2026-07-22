# Copyright 2026 Bytedance Ltd. and/or its affiliates

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.distributed.moe.hiermoe.oracle import load_route_snapshot
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import (
    CurrentRoutePlanner,
    assign_tokens_to_copies,
    assign_tokens_to_mirrored_r2,
    plan_routes_by_rank,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _hierarchy(ep_size: int, group_sizes: tuple[int, ...] | None = None) -> Hierarchy:
    return Hierarchy(ep_size=ep_size, group_sizes=group_sizes or (ep_size,), source="test")


def _identity_layout(ep_size: int, experts_per_rank: int, extra_slots: int = 0):
    slots_per_rank = experts_per_rank + extra_slots
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long)
    owners = torch.empty((ep_size * experts_per_rank,), dtype=torch.long)
    for logical in range(ep_size * experts_per_rank):
        rank, local = divmod(logical, experts_per_rank)
        slot = rank * slots_per_rank + local
        layout[slot] = logical
        owners[logical] = slot
    return layout, owners, slots_per_rank


def _plan_sha256(plan) -> str:
    payload = {
        "actions": [action.format() for action in plan.actions],
        "final_layout": list(plan.final_layout),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exhaustive_copy_assignment(
    selected: torch.Tensor,
    layout: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: torch.Tensor,
    hierarchy_group_sizes: tuple[int, ...],
    max_copies: int,
    token_ordinals: torch.Tensor,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    num_ranks = layout.numel() // slots_per_rank
    hierarchy_levels = sorted(
        {
            int(group_size)
            for group_size in hierarchy_group_sizes
            if 1 < int(group_size) < num_ranks
        },
        reverse=True,
    )
    hierarchy_levels.append(1)
    output = torch.empty_like(selected)
    for token_index, logical_routes in enumerate(selected.tolist()):
        copies = [
            torch.nonzero(layout == logical_expert, as_tuple=False).flatten().tolist()[:max_copies]
            for logical_expert in logical_routes
        ]
        copy_limit = min(max_copies, layout.numel())
        combination_count = copy_limit ** len(logical_routes)
        scored = []
        for combination_id in range(combination_count):
            choices = [
                (combination_id // (copy_limit**route_index)) % copy_limit
                for route_index in range(len(logical_routes))
            ]
            if any(choice >= len(copies[route_index]) for route_index, choice in enumerate(choices)):
                continue
            slots = tuple(copies[route_index][choice] for route_index, choice in enumerate(choices))
            if any(
                logical_routes[lhs] == logical_routes[rhs] and slots[lhs] != slots[rhs]
                for lhs in range(len(logical_routes))
                for rhs in range(lhs + 1, len(logical_routes))
            ):
                continue
            destination_ranks = tuple(slot // slots_per_rank for slot in slots)
            source_rank = int(source_ranks[token_index])
            score = tuple(
                len(
                    {
                        destination_rank // group_size
                        for destination_rank in destination_ranks
                        if destination_rank // group_size != source_rank // group_size
                    }
                )
                for group_size in hierarchy_levels
            )
            scored.append((score, combination_id, slots))

        minimum_score = min(score for score, _combination_id, _slots in scored)
        tied = [item for item in scored if item[0] == minimum_score]
        route_hash_sum = 0
        for logical_expert in logical_routes:
            value = (
                int(token_ordinals[token_index]) * 1_000_003
                + int(logical_expert) * 65_537
                + step * 131
                + layer_seed * 17
            )
            route_hash_sum += ((value * 48_271 + 1) % 2_147_483_647) % 1_048_573
        tie_modulus = 2_147_483_647
        tie_target = route_hash_sum % tie_modulus
        candidate_hashes = {}
        for _score, combination_id, slots in tied:
            candidate_hash = 0
            for physical_slot in slots:
                candidate_hash = (candidate_hash * 1_000_003 + physical_slot + 1) % tie_modulus
            candidate_hashes[combination_id] = candidate_hash

        scored_ties = []
        for item in tied:
            combination_id = item[1]
            mixed = (
                candidate_hashes[combination_id] * (tie_target + 1_000_003)
                + tie_target * 48_271
                + 1
            ) % tie_modulus
            tie_score = (mixed * 48_271 + 1) % tie_modulus
            scored_ties.append((tie_score, combination_id, item))
        _tie_score, _combination_id, (_score, _original_id, best_slots) = min(scored_ties)
        output[token_index] = torch.tensor(best_slots, dtype=output.dtype)
    return output


@pytest.mark.parametrize(
    "selected",
    (
        torch.tensor([[0], [2], [4], [7]], dtype=torch.long),
        torch.tensor([[0, 4], [2, 0], [5, 7], [4, 5]], dtype=torch.long),
        torch.tensor(
            [
                [0, 2, 4, 5, 6, 7, 0, 2],
                [5, 4, 2, 0, 7, 6, 5, 4],
            ],
            dtype=torch.long,
        ),
    ),
)
def test_vectorized_copy_assignment_matches_exhaustive_oracle(selected):
    slots_per_rank = 3
    layout = torch.tensor([0, 1, 4, 2, 3, 5, 4, 5, 6, 0, 7, 2], dtype=torch.long)
    source_ranks = torch.arange(selected.shape[0], dtype=torch.long).remainder(4)
    token_ordinals = torch.arange(selected.shape[0], dtype=torch.long) * 3 + 1
    expected = _exhaustive_copy_assignment(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=source_ranks,
        hierarchy_group_sizes=(2, 4),
        max_copies=2,
        token_ordinals=token_ordinals,
        step=11,
        layer_seed=23,
    )

    actual = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=source_ranks,
        hierarchy_group_sizes=(2, 4),
        token_ordinals=token_ordinals,
        step=11,
        layer_seed=23,
        max_copies=2,
    )

    torch.testing.assert_close(actual, expected)
    for token_routes, physical_routes in zip(selected.tolist(), actual.tolist()):
        chosen_by_expert = {}
        for logical_expert, physical_slot in zip(token_routes, physical_routes):
            assert chosen_by_expert.setdefault(logical_expert, physical_slot) == physical_slot


def test_vectorized_copy_assignment_supports_three_nonuniform_copies():
    layout = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2], dtype=torch.long)
    selected = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.long)
    source_ranks = torch.tensor([0, 2, 3], dtype=torch.long)
    token_ordinals = torch.tensor([5, 7, 11], dtype=torch.long)
    expected = _exhaustive_copy_assignment(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=source_ranks,
        hierarchy_group_sizes=(2, 4),
        max_copies=3,
        token_ordinals=token_ordinals,
        step=13,
        layer_seed=29,
    )

    actual = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=source_ranks,
        hierarchy_group_sizes=(2, 4),
        token_ordinals=token_ordinals,
        step=13,
        layer_seed=29,
        max_copies=3,
    )

    torch.testing.assert_close(actual, expected)


def _communication_objectives(
    physical_slots: torch.Tensor,
    source_ranks: torch.Tensor,
    *,
    slots_per_rank: int,
    hierarchy_group_sizes: tuple[int, ...],
) -> list[tuple[int, ...]]:
    physical_ranks = torch.div(physical_slots, slots_per_rank, rounding_mode="floor")
    objectives = []
    for token_index, ranks in enumerate(physical_ranks.tolist()):
        source_rank = int(source_ranks[token_index])
        objectives.append(
            tuple(
                len(
                    {
                        rank // group_size
                        for rank in ranks
                        if rank // group_size != source_rank // group_size
                    }
                )
                for group_size in (*hierarchy_group_sizes[:-1], 1)
            )
        )
    return objectives


@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_mirrored_r2_fast_path_matches_exhaustive_communication_optimum(top_k):
    ep_size = 8
    num_experts = 16
    slots_per_rank = 4
    logical = torch.arange(num_experts, dtype=torch.long)
    first_slots = torch.div(logical, slots_per_rank, rounding_mode="floor") * slots_per_rank
    first_slots += torch.remainder(logical, slots_per_rank)
    copy_slots = torch.stack((first_slots, first_slots + (ep_size // 2) * slots_per_rank), dim=-1)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long)
    layout[copy_slots[:, 0]] = logical
    layout[copy_slots[:, 1]] = logical
    selected = torch.stack(
        [torch.remainder(torch.arange(top_k, dtype=torch.long) * 3 + token, num_experts) for token in range(ep_size)]
    )
    if top_k == 8:
        selected[0, -1] = selected[0, 0]
    source_ranks = torch.arange(ep_size, dtype=torch.long)
    token_ordinals = torch.arange(ep_size, dtype=torch.long) * 5
    oracle = _exhaustive_copy_assignment(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=source_ranks,
        hierarchy_group_sizes=(2, ep_size),
        max_copies=2,
        token_ordinals=token_ordinals,
        step=17,
        layer_seed=31,
    )

    actual = assign_tokens_to_mirrored_r2(
        selected,
        copy_slots,
        source_ranks=source_ranks,
        num_ranks=ep_size,
    )

    expected_half = (source_ranks >= ep_size // 2).view(-1, 1)
    selected_copies = copy_slots.index_select(0, selected.reshape(-1)).view(*selected.shape, 2)
    torch.testing.assert_close(actual, torch.where(expected_half, selected_copies[..., 1], selected_copies[..., 0]))
    assert _communication_objectives(
        actual,
        source_ranks,
        slots_per_rank=slots_per_rank,
        hierarchy_group_sizes=(2, ep_size),
    ) == _communication_objectives(
        oracle,
        source_ranks,
        slots_per_rank=slots_per_rank,
        hierarchy_group_sizes=(2, ep_size),
    )


def test_token_to_copy_minimizes_unique_remote_ranks_before_hash_tie_break():
    layout, _owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    layout[5] = 0
    selected = torch.tensor([[0, 2], [0, 1]])

    physical = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=0,
        hierarchy_group_sizes=(2,),
        max_copies=2,
    )

    assert physical[0, 0].item() in (0, 5)
    assert physical[0, 1].item() == 3
    assert physical[1, 0].item() == 0


def test_token_to_copy_existing_rank_is_only_preferred_when_it_reduces_dedup_cost():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    layout[7] = 0  # rank 3 is farther from source rank 0, but expert 3 already requires it.

    physical = assign_tokens_to_copies(
        torch.tensor([[0, 3]]),
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=0,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        max_copies=2,
    )

    # Both choices communicate with exactly one remote rank: expert 0 can stay
    # local while expert 3 goes to rank 3, or both can go to rank 3.  Exact
    # dedup scoring therefore leaves the decision to the stable hash.
    assert physical[0, 0].item() in (0, 7)
    assert physical[0, 1].item() == 6


def test_token_to_copy_prefers_the_nearer_hierarchy_then_is_deterministic():
    layout, _owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    layout[5] = 0  # rank 2 copy; owner rank 0 is nearer to source rank 1.
    selected = torch.tensor([[0], [0], [0]])

    first = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=1,
        hierarchy_group_sizes=(2, 4),
        step=7,
        layer_seed=19,
        max_copies=2,
    )
    second = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=1,
        hierarchy_group_sizes=(2, 4),
        step=7,
        layer_seed=19,
        max_copies=2,
    )

    assert torch.equal(first, second)
    assert torch.equal(first, torch.zeros_like(first))


def test_token_to_copy_uses_deterministic_hash_for_equal_locality():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    layout[3] = 0  # rank 1 copy; ranks 0 and 1 are equally distant from source rank 2.
    selected = torch.zeros((256, 1), dtype=torch.long)

    first = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=2,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        step=7,
        layer_seed=19,
        max_copies=2,
    )
    second = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=2,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        step=7,
        layer_seed=19,
        max_copies=2,
    )

    assert torch.equal(first, second)
    assert set(first.flatten().tolist()) == {0, 3}


def test_replay_token_to_copy_matches_rank_local_runtime_ordinals():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    layout[3] = 0
    routes_by_rank = (
        torch.empty((0, 1), dtype=torch.long),
        torch.empty((0, 1), dtype=torch.long),
        torch.zeros((7, 1), dtype=torch.long),
        torch.zeros((7, 1), dtype=torch.long),
    )
    runtime = torch.cat(
        [
            assign_tokens_to_copies(
                routes,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=rank,
                hierarchy_group_sizes=(2, 4),
                owner_slots=owners,
                step=7,
                layer_seed=19,
                max_copies=2,
            )
            for rank, routes in enumerate(routes_by_rank)
        ]
    )
    replay_routes = torch.cat(routes_by_rank)
    replay_sources = torch.cat(
        [torch.full((routes.shape[0],), rank, dtype=torch.long) for rank, routes in enumerate(routes_by_rank)]
    )
    local_ordinals = torch.cat([torch.arange(routes.shape[0], dtype=torch.long) for routes in routes_by_rank])
    replay = assign_tokens_to_copies(
        replay_routes,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=replay_sources,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        token_ordinals=local_ordinals,
        step=7,
        layer_seed=19,
        max_copies=2,
    )

    torch.testing.assert_close(replay, runtime)


def test_token_to_copy_uses_compact_copy_table_independent_of_round_budget():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=2)
    layout[5] = 0
    selected = torch.zeros((32, 1), dtype=torch.long)
    copy_slots = torch.tensor([[0, 5], [3, -1], [6, -1], [9, -1]])
    copy_mask = copy_slots >= 0

    from_layout = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=2,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        step=7,
        layer_seed=19,
        max_copies=16,
    )
    from_cache = assign_tokens_to_copies(
        selected,
        layout,
        slots_per_rank=slots_per_rank,
        source_ranks=2,
        hierarchy_group_sizes=(2, 4),
        owner_slots=owners,
        step=7,
        layer_seed=19,
        max_copies=1,
        copy_slots=copy_slots,
        copy_mask=copy_mask,
    )

    torch.testing.assert_close(from_cache, from_layout)


def test_token_to_copy_rejects_mismatched_cached_slots():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    selected = torch.tensor([[0]])
    copy_slots = torch.tensor([[0, 1], [1, -1], [3, -1], [4, -1]])
    copy_mask = copy_slots >= 0

    with pytest.raises(ValueError, match="does not hold"):
        assign_tokens_to_copies(
            selected,
            layout,
            slots_per_rank=slots_per_rank,
            source_ranks=0,
            hierarchy_group_sizes=(2,),
            owner_slots=owners,
            copy_slots=copy_slots,
            copy_mask=copy_mask,
        )


def test_cost_deduplicates_communication_but_not_compute():
    layout, owners, slots_per_rank = _identity_layout(2, 2)
    routes = (torch.tensor([[0, 1]]), torch.empty((0, 2), dtype=torch.long))
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=0,
        max_replicas=0,
        communication_scale=1.0,
        forward_compute_per_assignment=1.0,
    )

    # Both assignments execute on rank 0, while rank-level communication sees
    # one duplicate-free token for that rank.
    assert plan.baseline_cost.compute == pytest.approx(6.0)
    assert plan.baseline_cost.communication_model_units == pytest.approx(12.0)


def test_swap_statistics_keep_duplicate_assignments_in_compute_cost():
    layout, owners, slots_per_rank = _identity_layout(2, 1)
    routes = (torch.tensor([[0, 0]]), torch.empty((0, 2), dtype=torch.long))
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=1,
        max_replicas=0,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.baseline_cost.compute == pytest.approx(6.0)


def test_swap_uses_compute_bottleneck_and_stops_on_non_improvement():
    layout, owners, slots_per_rank = _identity_layout(2, 2)
    routes = (
        torch.tensor([[0]] * 10 + [[1]] * 8),
        torch.tensor([[2], [3]]),
    )
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=4,
        max_replicas=0,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.swap_rounds == 1
    assert plan.swaps[0].src_logical == 0
    assert plan.swaps[0].dst_logical in {2, 3}
    assert plan.final_cost.compute < plan.baseline_cost.compute


def test_swap_preserves_existing_replicas_when_replica_budget_is_zero():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    layout[2] = 2
    layout[5] = 0
    routes = (
        torch.tensor([[0]] * 10 + [[1]] * 8),
        torch.tensor([[2], [3]]),
    )
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=1,
        max_replicas=0,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.swap_rounds == 1
    assert plan.initial_layout == tuple(layout.tolist())
    replayed = layout.clone()
    for action in plan.swaps:
        replayed[action.src_slot], replayed[action.dst_slot] = (
            replayed[action.dst_slot].clone(),
            replayed[action.src_slot].clone(),
        )
    assert plan.final_layout == tuple(replayed.tolist())
    assert plan.final_layout[2] == 2
    assert plan.final_layout[5] == 0


def test_equal_cost_keeps_no_op():
    layout, owners, slots_per_rank = _identity_layout(2, 2)
    routes = (torch.tensor([[0], [1]]), torch.tensor([[2], [3]]))
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=2,
        max_replicas=0,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.actions == ()
    assert plan.final_layout == plan.initial_layout


def test_replica_splits_non_deduplicated_compute_by_source_rank():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    routes = (torch.tensor([[0]] * 5), torch.tensor([[0]] * 5))
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=0,
        max_replicas=1,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.swap_rounds == 0
    assert plan.replica_rounds == 1
    assert plan.replicas[0].src_logical == 0
    assert plan.replicas[0].dst_slot == 5
    assert plan.final_cost.compute == pytest.approx(15.0)
    assert plan.baseline_cost.compute == pytest.approx(30.0)


def test_replica_retargets_a_saturated_redundant_slot():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    layout[2] = 1
    layout[5] = 3
    routes = (torch.tensor([[0]] * 4), torch.tensor([[0]] * 4))
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=0,
        max_replicas=1,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.replica_rounds == 1
    assert plan.final_layout[5] == 0
    assert plan.final_layout[2] == -1


@pytest.mark.parametrize(
    ("max_swaps", "max_replicas", "expected_kinds"),
    [
        (0, 0, ()),
        (1, 0, ("swap",)),
        (0, 1, ("replica",)),
        (1, 1, ("swap", "replica")),
    ],
)
def test_budget_combinations(max_swaps: int, max_replicas: int, expected_kinds: tuple[str, ...]):
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    routes = (
        torch.tensor([[0]] * 2 + [[1]] * 8),
        torch.tensor([[0]] * 12 + [[2]] + [[3]] * 3),
    )
    plan = plan_routes_by_rank(
        routes,
        layout,
        owners,
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=max_swaps,
        max_replicas=max_replicas,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    kinds = tuple(action.kind for action in plan.actions)
    assert kinds == expected_kinds


def test_vectorized_layout_cost_matches_scalar_candidate_scoring():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    selected = torch.tensor([[0, 2], [0, 1], [2, 3]])
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=4,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=0.25,
    )
    candidates = layout.repeat(2, 1)
    candidates[0, 5] = 0
    candidates[1, 2] = 2
    sources = torch.zeros((selected.shape[0],), dtype=torch.long)

    batched = planner._score_layouts(  # noqa: SLF001 - scalar oracle for the vectorized kernel
        selected, candidates, owners, sources, step=0, layer_seed=0, max_copies=2
    )
    scalar = [
        planner._score_layouts(  # noqa: SLF001
            selected, candidate, owners, sources, step=0, layer_seed=0, max_copies=2
        )
        for candidate in candidates
    ]

    assert torch.allclose(batched.total, torch.cat([cost.total for cost in scalar]))
    assert torch.equal(
        batched.peak_compute_rank,
        torch.cat([cost.peak_compute_rank for cost in scalar]),
    )


def test_incremental_swap_cost_matches_full_layout_oracle_with_group_collisions():
    layout, owners, slots_per_rank = _identity_layout(4, 2)
    selected = torch.tensor(
        [
            [0, 0, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 5, 6],
            [2, 3, 6, 7],
            [0, 2, 4, 6],
        ]
    )
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(4, (2, 4)),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=4,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=0.25,
    )
    token_hits = torch.zeros((selected.shape[0], owners.numel()), dtype=torch.float32)
    token_hits.scatter_(1, selected, 1.0)
    stats = planner._initial_swap_stats(token_hits, selected, owners)  # noqa: SLF001
    pairs = torch.tensor([[0, 2], [1, 4], [2, 7]])

    incremental, _candidate_groups = planner._swap_candidate_costs(stats, pairs)  # noqa: SLF001
    candidates = layout.repeat(pairs.shape[0], 1)
    for index, (lhs, rhs) in enumerate(pairs.tolist()):
        lhs_slot, rhs_slot = int(owners[lhs]), int(owners[rhs])
        candidates[index, lhs_slot], candidates[index, rhs_slot] = (
            candidates[index, rhs_slot].clone(),
            candidates[index, lhs_slot].clone(),
        )
    full = planner._score_layouts(  # noqa: SLF001
        selected,
        candidates,
        owners,
        torch.zeros(selected.shape[0], dtype=torch.long),
        step=0,
        layer_seed=0,
        max_copies=1,
    )

    assert torch.allclose(incremental.communication, full.communication)
    assert torch.allclose(incremental.compute, full.compute)
    assert torch.equal(incremental.peak_communication_rank, full.peak_communication_rank)
    assert torch.equal(incremental.peak_compute_rank, full.peak_compute_rank)


def test_incremental_swap_state_update_matches_second_round_full_oracle():
    layout, owners, slots_per_rank = _identity_layout(4, 2)
    selected = torch.tensor(
        [
            [0, 0, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 5, 6],
            [2, 3, 6, 7],
            [0, 2, 4, 6],
        ]
    )
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(4, (2, 4)),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=4,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=0.25,
    )
    token_hits = torch.zeros((selected.shape[0], owners.numel()), dtype=torch.float32)
    token_hits.scatter_(1, selected, 1.0)
    initial = planner._initial_swap_stats(token_hits, selected, owners)  # noqa: SLF001
    first_pair = torch.tensor([[0, 4]])
    _first_cost, first_groups = planner._swap_candidate_costs(initial, first_pair)  # noqa: SLF001
    updated = planner._update_swap_stats(  # noqa: SLF001
        token_hits,
        initial,
        first_pair[0],
        tuple(counts[0] for counts in first_groups),
    )

    first_layout = layout.clone()
    first_layout[owners[0]], first_layout[owners[4]] = (
        first_layout[owners[4]].clone(),
        first_layout[owners[0]].clone(),
    )
    updated_owners = owners.clone()
    updated_owners[0], updated_owners[4] = owners[4], owners[0]
    current_full = planner._score_layouts(  # noqa: SLF001
        selected,
        first_layout,
        updated_owners,
        torch.zeros(selected.shape[0], dtype=torch.long),
        step=0,
        layer_seed=0,
        max_copies=1,
    )
    current_incremental = planner._current_swap_cost(updated)  # noqa: SLF001
    assert torch.allclose(current_incremental.communication, current_full.communication)
    assert torch.allclose(current_incremental.compute, current_full.compute)

    second_pairs = torch.tensor([[1, 6], [2, 7]])
    second_incremental, _second_groups = planner._swap_candidate_costs(  # noqa: SLF001
        updated, second_pairs
    )
    candidates = first_layout.repeat(second_pairs.shape[0], 1)
    for index, (lhs, rhs) in enumerate(second_pairs.tolist()):
        lhs_slot, rhs_slot = int(updated_owners[lhs]), int(updated_owners[rhs])
        candidates[index, lhs_slot], candidates[index, rhs_slot] = (
            candidates[index, rhs_slot].clone(),
            candidates[index, lhs_slot].clone(),
        )
    second_full = planner._score_layouts(  # noqa: SLF001
        selected,
        candidates,
        updated_owners,
        torch.zeros(selected.shape[0], dtype=torch.long),
        step=0,
        layer_seed=0,
        max_copies=1,
    )
    assert torch.allclose(second_incremental.communication, second_full.communication)
    assert torch.allclose(second_incremental.compute, second_full.compute)
    assert torch.equal(second_incremental.peak_communication_rank, second_full.peak_communication_rank)
    assert torch.equal(second_incremental.peak_compute_rank, second_full.peak_compute_rank)


def test_swap_statistics_separate_unique_token_hits_from_raw_assignments():
    _layout, owners, slots_per_rank = _identity_layout(2, 2)
    selected = torch.tensor([[0, 0], [0, 1], [2, 3]])
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
    )
    token_hits = torch.zeros((selected.shape[0], owners.numel()), dtype=torch.float32)
    token_hits.scatter_(1, selected, 1.0)
    stats = planner._initial_swap_stats(token_hits, selected, owners)  # noqa: SLF001

    assert stats.expert_token_counts.tolist() == [2.0, 1.0, 1.0, 1.0]
    assert stats.expert_assignment_counts.tolist() == [3.0, 1.0, 1.0, 1.0]


def test_incremental_replica_cost_matches_full_layout_oracle():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    selected = torch.tensor([[0, 2], [0, 1], [2, 3], [3, 0]])
    sources = torch.zeros((selected.shape[0],), dtype=torch.long)
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=4,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=0.25,
    )
    logical = torch.tensor([0, 2])
    destinations = torch.tensor([5, 2])
    candidates = layout.repeat(2, 1)
    candidates.scatter_(1, destinations.unsqueeze(1), logical.unsqueeze(1))

    incremental = planner._replica_candidate_costs(  # noqa: SLF001
        selected,
        layout,
        owners,
        sources,
        logical,
        destinations,
        step=3,
        layer_seed=7,
        max_copies=2,
    )
    owner_fast_path = planner._replica_candidate_costs(  # noqa: SLF001
        selected,
        layout,
        owners,
        sources,
        logical,
        destinations,
        step=3,
        layer_seed=7,
        max_copies=2,
        owner_only_layout=True,
    )
    full = planner._score_layouts(  # noqa: SLF001
        selected,
        candidates,
        owners,
        sources,
        step=3,
        layer_seed=7,
        max_copies=2,
    )

    assert torch.allclose(incremental.communication, full.communication)
    assert torch.allclose(incremental.compute, full.compute)
    assert torch.equal(incremental.peak_communication_rank, full.peak_communication_rank)
    assert torch.equal(incremental.peak_compute_rank, full.peak_compute_rank)
    assert torch.allclose(owner_fast_path.communication, full.communication)
    assert torch.allclose(owner_fast_path.compute, full.compute)


def test_incremental_replica_cost_matches_oracle_after_an_existing_replica():
    layout, owners, slots_per_rank = _identity_layout(2, 2, extra_slots=1)
    layout[5] = 0
    selected = torch.tensor([[0, 2], [0, 1], [2, 3], [3, 0]])
    sources = torch.zeros((selected.shape[0],), dtype=torch.long)
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=4,
        bytes_per_element=2,
        slots_per_rank=slots_per_rank,
        forward_compute_per_assignment=0.25,
    )
    logical = torch.tensor([2])
    destinations = torch.tensor([2])
    candidate = layout.clone()
    candidate[2] = 2

    incremental = planner._replica_candidate_costs(  # noqa: SLF001
        selected,
        layout,
        owners,
        sources,
        logical,
        destinations,
        step=3,
        layer_seed=7,
        max_copies=3,
    )
    full = planner._score_layouts(  # noqa: SLF001
        selected,
        candidate,
        owners,
        sources,
        step=3,
        layer_seed=7,
        max_copies=3,
    )

    assert torch.allclose(incremental.communication, full.communication)
    assert torch.allclose(incremental.compute, full.compute)
    assert torch.equal(incremental.peak_communication_rank, full.peak_communication_rank)
    assert torch.equal(incremental.peak_compute_rank, full.peak_compute_rank)


def test_incremental_replica_state_matches_full_oracle_across_two_rounds():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    selected = torch.zeros((256, 2), dtype=torch.long)
    sources = torch.full((selected.shape[0],), 2, dtype=torch.long)
    ordinals = torch.arange(selected.shape[0], dtype=torch.long)
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(4),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )
    owner_ranks = torch.div(owners, slots_per_rank, rounding_mode="floor")
    routed_ranks = owner_ranks.index_select(0, selected.reshape(-1)).view_as(selected)
    local_counts, local_assignment = planner._local_rank_stats(routed_ranks.unsqueeze(0))  # noqa: SLF001
    stats = planner._initial_replica_stats(  # noqa: SLF001
        selected,
        owners,
        sources,
        ordinals,
        step=7,
        layer_seed=19,
        base_counts=tuple(count[0] for count in local_counts),
        assignment_counts=local_assignment[0],
    )

    first = planner._incremental_replica_candidates(stats)  # noqa: SLF001
    first_index = torch.tensor(1)  # logical expert 0 -> rank 1
    first_layout = layout.clone()
    first_layout[3] = 0
    first_full = planner._score_layouts(  # noqa: SLF001
        selected,
        first_layout,
        owners,
        sources,
        token_ordinals=ordinals,
        step=7,
        layer_seed=19,
        max_copies=2,
    )
    assert torch.allclose(first.cost.communication[first_index], first_full.communication[0])
    assert torch.allclose(first.cost.compute[first_index], first_full.compute[0])
    planner._apply_replica_candidate(  # noqa: SLF001
        stats,
        first,
        first_index,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([3]),
    )

    second = planner._incremental_replica_candidates(stats)  # noqa: SLF001
    second_index = torch.tensor(3)  # logical expert 0 -> rank 3
    second_layout = first_layout.clone()
    second_layout[7] = 0
    second_full = planner._score_layouts(  # noqa: SLF001
        selected,
        second_layout,
        owners,
        sources,
        token_ordinals=ordinals,
        step=7,
        layer_seed=19,
        max_copies=3,
    )
    assert torch.allclose(second.cost.communication[second_index], second_full.communication[0])
    assert torch.allclose(second.cost.compute[second_index], second_full.compute[0])
    assert torch.equal(second.cost.peak_compute_rank[second_index], second_full.peak_compute_rank[0])


def test_multiround_replica_plan_does_not_use_full_layout_scoring(monkeypatch):
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    selected = torch.zeros((256, 1), dtype=torch.long)
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(4),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    def fail_full_layout(*args, **kwargs):
        raise AssertionError("production replica planning must remain incremental")

    monkeypatch.setattr(planner, "_score_layouts", fail_full_layout)
    plan = planner.plan(
        selected,
        layout,
        owners,
        source_ranks=2,
        max_swaps=0,
        max_replicas=4,
        step=7,
        layer_seed=19,
    )

    assert plan.replica_rounds >= 2
    for field in (
        "swap_score_ms",
        "swap_update_ms",
        "swap_collective_ms",
        "replica_score_ms",
        "replica_update_ms",
        "replica_collective_ms",
        "decision_sync_ms",
        "finalization_ms",
    ):
        assert getattr(plan, field) >= 0.0


def test_replica_planning_uses_one_reduction_per_attempted_round_and_stops_on_no_gain():
    layout, owners, slots_per_rank = _identity_layout(4, 1, extra_slots=1)
    selected = torch.arange(4, dtype=torch.long).repeat(64).view(-1, 1)
    reduction_shapes = []

    def counting_reducer(tensor):
        reduction_shapes.append(tuple(tensor.shape))
        return tensor

    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(4),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
        reducer=counting_reducer,
    )
    plan = planner.plan(
        selected,
        layout,
        owners,
        source_ranks=0,
        max_swaps=0,
        max_replicas=4,
        step=7,
        layer_seed=19,
    )

    assert plan.replica_rounds == 0
    assert len(reduction_shapes) == 2
    assert reduction_shapes[0][0] == 1
    assert reduction_shapes[1][0] == owners.numel() * 4


def test_replica_scoring_compacts_candidates_to_bottleneck_rank_experts():
    layout, owners, slots_per_rank = _identity_layout(16, 8, extra_slots=1)
    selected = torch.zeros((64, 1), dtype=torch.long)
    reduction_shapes = []

    def counting_reducer(tensor):
        reduction_shapes.append(tuple(tensor.shape))
        return tensor

    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(16, (8, 16)),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
        reducer=counting_reducer,
    )
    planner.plan(
        selected,
        layout,
        owners,
        source_ranks=1,
        max_swaps=0,
        max_replicas=1,
        step=7,
        layer_seed=19,
    )

    assert reduction_shapes[1][0] == 2 * slots_per_rank * 16
    assert reduction_shapes[1][0] < owners.numel() * 16


def _empty_route_rank_replica_planner_worker():
    rank = dist.get_rank()
    layout, owners, slots_per_rank = _identity_layout(2, 1, extra_slots=1)
    selected = torch.empty((0, 1), dtype=torch.long) if rank == 0 else torch.zeros((32, 1), dtype=torch.long)

    def reducer(tensor):
        dist.all_reduce(tensor)

    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
        reducer=reducer,
    )
    plan = planner.plan(
        selected,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=0,
        max_replicas=1,
        step=7,
        layer_seed=19,
    )

    gathered_actions = [None, None]
    dist.all_gather_object(gathered_actions, plan.actions)
    assert gathered_actions[0] == gathered_actions[1]


def test_replica_planner_handles_an_empty_route_rank_without_collective_mismatch():
    torchrun(_empty_route_rank_replica_planner_worker, world_size=2, backend="gloo")


def test_replica_planning_uses_actual_empty_slots_with_noncontiguous_owners():
    slots_per_rank = 3
    owners = torch.tensor([2, 0, 5, 3], dtype=torch.long)
    layout = torch.tensor([1, -1, 0, 3, -1, 2], dtype=torch.long)
    selected = torch.zeros((256, 1), dtype=torch.long)
    source_ranks = torch.arange(2, dtype=torch.long).repeat_interleave(128)
    planner = CurrentRoutePlanner(
        hierarchy=_hierarchy(2),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    plan = planner.plan(
        selected,
        layout,
        owners,
        source_ranks=source_ranks,
        max_swaps=0,
        max_replicas=1,
        step=7,
        layer_seed=19,
    )

    assert plan.replica_rounds == 1
    assert plan.final_layout[4] == 0
    assert [plan.final_layout[slot] for slot in owners.tolist()] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("ep_size", "group_sizes"),
    ((16, (8, 16)), (32, (8, 32)), (64, (8, 16, 64)), (16, (16,))),
)
def test_auto_replica_capacity_supports_large_ep_topologies(ep_size, group_sizes):
    num_experts = 128
    experts_per_rank = num_experts // ep_size
    layout, owners, slots_per_rank = _identity_layout(ep_size, experts_per_rank, extra_slots=1)
    routes_by_rank = []
    for rank in range(ep_size):
        local_expert = max(1, rank * experts_per_rank)
        routes_by_rank.append(torch.tensor([[0, 0, 0, 0, 0, 0, 0, local_expert]], dtype=torch.long))

    plan = plan_routes_by_rank(
        routes_by_rank,
        layout,
        owners,
        hierarchy=_hierarchy(ep_size, group_sizes),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=1,
        bytes_per_element=1,
        slots_per_rank=slots_per_rank,
        max_swaps=0,
        max_replicas=ep_size,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    assert plan.replica_rounds == ep_size - 1
    replica_ranks = {action.dst_slot // slots_per_rank for action in plan.replicas}
    assert replica_ranks == set(range(1, ep_size))
    assert plan.final_layout[experts_per_rank] == -1


@pytest.mark.skipif(
    "VEOMNI_HIERMOE_EP16_SNAPSHOT" not in os.environ,
    reason="Set VEOMNI_HIERMOE_EP16_SNAPSHOT to run the external EP16 route replay.",
)
def test_external_ep16_route_snapshot_matches_golden():
    snapshot_path = Path(os.environ["VEOMNI_HIERMOE_EP16_SNAPSHOT"])
    golden_path = Path(__file__).parents[1] / "data" / "hiermoe_current_route_ep16_layer24_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == golden["snapshot"]["sha256"]
    snapshot = load_route_snapshot(snapshot_path)
    assert [routes.shape[0] for routes in snapshot.routes_by_rank] == golden["snapshot"]["tokens_per_rank"]

    calibration = golden["calibration"]
    for config in ("P1S0", "P4S0", "P0S1-auto", "P1S1-auto", "P4S1-auto"):
        expected = golden["configs"][config]
        layout, owners, slots_per_rank = _identity_layout(
            snapshot.ep_size,
            snapshot.num_experts // snapshot.ep_size,
            expected["slot_increment_per_rank"],
        )
        plan = plan_routes_by_rank(
            snapshot.routes_by_rank,
            layout,
            owners,
            hierarchy=snapshot.hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=snapshot.hidden_size,
            bytes_per_element=snapshot.bytes_per_element,
            slots_per_rank=slots_per_rank,
            max_swaps=expected["max_swaps"],
            max_replicas=expected["max_replicas"],
            communication_scale=calibration["communication_scale"],
            forward_compute_per_assignment=calibration["forward_compute_per_assignment"],
            step=snapshot.step,
            layer_seed=24,
        )
        assert [action.format() for action in plan.actions] == expected["actions"]
        assert list(plan.final_layout) == expected["final_layout"]
        assert plan.final_cost.communication == pytest.approx(expected["communication_ms"], abs=1e-3)
        assert plan.final_cost.compute == pytest.approx(expected["compute_ms"], abs=1e-3)
        assert plan.final_cost.total == pytest.approx(expected["total_ms"], abs=1e-3)
        assert _plan_sha256(plan) == expected["plan_sha256"]
