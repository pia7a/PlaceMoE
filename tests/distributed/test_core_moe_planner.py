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

import pytest
import torch

from veomni.distributed.moe.hiermoe.core_planner import (
    _FUSED_CAP_QUOTA_POLICY,
    _FUSED_PROTOCOL_ABI_SHIFT,
    CORE_MOE_ALGORITHM_VERSION,
    CoReMoEPlanner,
    QuotaPolicyEntry,
    RouteSummary,
    _stable_hungarian_maximize,
    assign_tokens_to_copies_with_quota,
    build_quota_tensor_tables,
    build_route_summary,
)
from veomni.distributed.moe.hiermoe.perf_model import (
    GradientSyncCost,
    HierMoEPerfModel,
    LinkCost,
    PeerTransferCost,
    _peer_for_probe,
)
from veomni.distributed.moe.hiermoe.planner import PlacementAction, apply_placement_action
from veomni.distributed.moe.hiermoe.topology import Hierarchy, infer_hierarchy


def _first_copy_quota_map(
    selected,
    copy_slots,
    _copy_counts,
    _owner_ranks,
    _quota_weights,
    _quota_configured,
    _token_ordinals,
    slots_per_rank,
    _source_rank,
    ep_size,
    num_levels,
    level_size0,
    level_size1,
    _step,
    _layer_seed,
):
    physical = copy_slots[..., 0][:, selected]
    level_sizes = (level_size0, level_size1)[:num_levels] + (1,)
    packed_groups = []
    packed_assignments = []
    for layout_physical in physical:
        ranks = torch.div(layout_physical, slots_per_rank, rounding_mode="floor")
        groups = []
        for size in level_sizes:
            hits = torch.zeros((selected.shape[0], ep_size // size), dtype=torch.bool, device=selected.device)
            hits.scatter_(1, torch.div(ranks, size, rounding_mode="floor"), True)
            groups.append(hits.sum(dim=0).to(torch.float32))
        packed_groups.append(torch.cat(groups))
        packed_assignments.append(torch.bincount(ranks.reshape(-1), minlength=ep_size).to(torch.float32))
    return physical, torch.stack(packed_groups), torch.stack(packed_assignments)


def _empty_quota_policy(
    sample_routes,
    _sample_multiplicity,
    _sample_sources,
    _sample_ordinals,
    assignment_counts,
    layouts,
    owner_slots,
    _slots_per_rank,
    _source_rank,
    _ep_size,
    max_copies,
    samples_per_source,
    _num_levels,
    _level_size0,
    _level_size1,
):
    num_experts = assignment_counts.shape[1]
    mask_count = 1 << max_copies
    weights = torch.zeros((2, num_experts, mask_count, max_copies), dtype=torch.long, device=layouts.device)
    configured = torch.zeros((2, num_experts, mask_count), dtype=torch.long, device=layouts.device)
    rows = torch.zeros(
        (2, samples_per_source * sample_routes.shape[1], 3 + 2 * max_copies),
        dtype=torch.long,
        device=layouts.device,
    )
    row_counts = torch.zeros((2,), dtype=torch.long, device=layouts.device)
    digest = torch.stack(
        (
            torch.remainder(layouts.clamp_min(0).sum(dim=1) + 17, 1048573),
            torch.remainder(owner_slots.sum(dim=1) + 29, 1000003),
        ),
        dim=1,
    )
    return weights, configured, rows, row_counts, digest


def test_route_summary_preserves_exact_assignments_and_is_deterministic():
    selected = torch.tensor([[0, 1, 1], [2, 1, 0], [2, 2, 1], [0, 2, 1]])

    first = build_route_summary(
        selected,
        num_experts=3,
        ep_size=1,
        sample_size=2,
        source_rank=0,
        step=7,
        layer_seed=11,
        gather_fixed=None,
    )
    second = build_route_summary(
        selected,
        num_experts=3,
        ep_size=1,
        sample_size=2,
        source_rank=0,
        step=7,
        layer_seed=11,
        gather_fixed=None,
    )

    assert first.assignment_counts.tolist() == [[3, 5, 4]]
    assert first.sample_routes.shape == (2, 3)
    assert first.sample_digest == second.sample_digest
    assert torch.equal(first.sample_routes, second.sample_routes)
    assert torch.equal(first.sample_ordinals, second.sample_ordinals)
    assert first.sample_multiplicity is not None
    for route, multiplicity in zip(first.sample_routes.tolist(), first.sample_multiplicity.tolist(), strict=True):
        for position, logical in enumerate(route):
            expected = route.count(logical) if route.index(logical) == position else 0
            assert multiplicity[position] == expected


def test_quota_mapper_balances_only_after_communication_class_tie():
    hierarchy = Hierarchy(ep_size=2, group_sizes=(1, 2), source="test")
    selected = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    # Expert 0 has copies on both ranks; expert 1 is owned by rank 1.
    layout = torch.tensor([0, -1, 1, 0])
    owners = torch.tensor([0, 2])

    mapping = assign_tokens_to_copies_with_quota(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy=hierarchy,
        owner_slots=owners,
        step=3,
        layer_seed=5,
    )
    ranks = torch.div(mapping.physical_slots, 2, rounding_mode="floor")

    assert ranks[:, 0].tolist() == [0, 0, 0, 0]
    assert ranks[:, 1].tolist() == [1, 1, 1, 1]
    assert mapping.policy[0].destination_ranks == (0, 1)
    assert mapping.policy[0].quotas == (4, 0)


def test_quota_mapper_keeps_duplicate_topk_assignments_on_one_copy():
    hierarchy = Hierarchy(ep_size=2, group_sizes=(1, 2), source="test")
    selected = torch.tensor([[0, 0, 1], [0, 1, 0]])
    layout = torch.tensor([0, -1, 1, 0])
    owners = torch.tensor([0, 2])

    mapping = assign_tokens_to_copies_with_quota(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy=hierarchy,
        owner_slots=owners,
    )

    assert mapping.physical_slots[0, 0] == mapping.physical_slots[0, 1]
    assert mapping.physical_slots[1, 0] == mapping.physical_slots[1, 2]


def test_quota_mapper_reuses_global_sample_policy_for_exact_routes():
    hierarchy = Hierarchy(ep_size=2, group_sizes=(1, 2), source="test")
    selected = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    layout = torch.tensor([0, -1, 1, 0])
    owners = torch.tensor([0, 2])
    policy = (QuotaPolicyEntry(0, 0, (0, 1), (0, 4)),)

    mapping = assign_tokens_to_copies_with_quota(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy=hierarchy,
        owner_slots=owners,
        quota_policy=policy,
    )
    ranks = torch.div(mapping.physical_slots, 2, rounding_mode="floor")

    assert ranks[:, 0].tolist() == [1, 1, 1, 1]
    assert mapping.policy[0].quotas == (0, 4)


def test_quota_mapper_consumes_partial_integer_quota_exactly():
    hierarchy = Hierarchy(ep_size=2, group_sizes=(1, 2), source="test")
    selected = torch.tensor([[0, 1]] * 16, dtype=torch.long)
    layout = torch.tensor([0, -1, 1, 0])
    owners = torch.tensor([0, 2])
    policy = (QuotaPolicyEntry(0, 0, (0, 1), (6, 10)),)

    mapping = assign_tokens_to_copies_with_quota(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy=hierarchy,
        owner_slots=owners,
        token_ordinals=torch.arange(16),
        quota_policy=policy,
        step=7,
        layer_seed=13,
    )
    ranks = torch.div(mapping.physical_slots[:, 0], 2, rounding_mode="floor").reshape(-1)

    assert torch.bincount(ranks, minlength=2).tolist() == [6, 10]
    assert mapping.policy[0].quotas == (6, 10)


def test_quota_policy_tuple_round_trip():
    entry = QuotaPolicyEntry(3, 17, (2, 5, 9), (4, 7, 11))

    assert QuotaPolicyEntry.from_tuple(entry.as_tuple()) == entry


def test_quota_tensor_table_indexes_policy_by_eligible_copy_mask():
    layouts = torch.tensor([[0, 1, -1, -1], [0, 1, 0, -1]])
    owners = torch.tensor([[0, 1], [0, 1]])
    policy = QuotaPolicyEntry(1, 0, (0, 1), (2, 5))

    tables = build_quota_tensor_tables(
        layouts,
        owners,
        ((), (policy,)),
        source_rank=1,
        slots_per_rank=2,
    )

    assert tables.copy_slots[1, 0].tolist() == [0, 2]
    assert tables.quota_weights[1, 0, 0b11].tolist() == [2, 5]


def test_empty_action_only_clears_redundant_slot():
    layout = torch.tensor([0, 1, 0, -1])

    updated = apply_placement_action(layout, PlacementAction("empty", 2, 2, 0, -1))

    assert updated.tolist() == [0, 1, -1, -1]
    assert layout.tolist() == [0, 1, 0, -1]


def test_hierarchy_retains_node_size_for_ep16_ep32_ep64(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    ep16 = infer_hierarchy(16)
    ep32 = infer_hierarchy(32)
    ep64 = infer_hierarchy(64)

    assert (ep16.group_sizes, ep16.local_world_size) == ((8, 16), 8)
    assert (ep32.group_sizes, ep32.local_world_size) == ((8, 32), 8)
    assert (ep64.group_sizes, ep64.local_world_size) == ((8, 64), 8)


def test_startup_probe_pairs_are_symmetric_with_odd_and_partial_nodes():
    for world_size, local_world_size in ((20, 8), (24, 8), (17, 8)):
        for intra in (False, True):
            peers = [_peer_for_probe(rank, world_size, local_world_size, intra=intra) for rank in range(world_size)]
            for rank, peer in enumerate(peers):
                if peer is not None:
                    assert peers[peer] == rank


def test_state_move_cost_batches_replica_payloads_in_one_peer_wave():
    intra = LinkCost(alpha=0.0, beta=0.0)
    inter = LinkCost(alpha=1.0, beta=1.0)
    transfer = PeerTransferCost(intra=intra, inter=inter)
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel(
            a2a=inter,
            inter=(inter,),
            intra=intra,
            source="test",
            state_move=transfer,
            gradient_sync=GradientSyncCost(gather=transfer, scatter=transfer),
            schema_version=2,
        ),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        route_sample_size=4,
        expert_state_bytes=(10, 20),
    )
    actions = (
        PlacementAction("replica", 0, 2, 0, -1),
        PlacementAction("replica", 1, 3, 1, -1),
    )

    assert planner._state_move_cost(actions) == 31.0


def test_fallback_runtime_links_use_communication_calibration_units():
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test", local_world_size=1),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=0.01,
        expert_state_bytes=(10, 20),
    )

    cost = planner._state_move_cost((PlacementAction("replica", 0, 2, 0, -1),))

    assert cost == pytest.approx(0.083)


def test_stable_hungarian_uses_lowest_column_on_equal_gain():
    assert _stable_hungarian_maximize([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]) == (0, 1)


def test_stable_hungarian_replica_columns_follow_keep_empty_cover_order():
    # Per-slot KEEP/EMPTY dummies precede the shared expert column, matching the NPU matcher ABI.
    weights = [[0.0, 1.0, -torch.inf, -torch.inf, 1.0], [-torch.inf, -torch.inf, 0.0, 1.0, 1.0]]

    assert _stable_hungarian_maximize(weights) == (1, 3)


def test_eager_one_shot_replica_uses_independent_edge_semantics():
    ep_size = 4
    slots_per_rank = 2
    layout = torch.tensor((0, -1, 1, -1, 2, -1, 3, -1), dtype=torch.long)
    owners = torch.tensor((0, 2, 4, 6), dtype=torch.long)
    routes = torch.zeros((128, 1), dtype=torch.long)
    sources = torch.arange(ep_size, dtype=torch.long).repeat_interleave(32)
    summary = RouteSummary(
        token_counts=torch.full((ep_size,), 32, dtype=torch.long),
        assignment_counts=torch.tensor(
            ((32, 0, 0, 0), (32, 0, 0, 0), (32, 0, 0, 0), (32, 0, 0, 0)),
            dtype=torch.long,
        ),
        sample_routes=routes,
        sample_ordinals=torch.arange(routes.shape[0], dtype=torch.long),
        sample_valid=torch.ones((routes.shape[0],), dtype=torch.bool),
        sample_weights=torch.ones((routes.shape[0],), dtype=torch.float32),
        sample_sources=sources,
        sample_digest="eager-independent-edge",
        sample_multiplicity=torch.ones_like(routes),
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
    updated, actions, _score, count = planner._plan_replicas(
        summary,
        layout.clone(),
        owners,
        [],
        current,
        max_replicas=1,
        step=5,
        layer_seed=17,
    )

    assert count == 1
    assert len(actions) == 1
    assert actions[0].kind == "replica"
    assert actions[0].src_logical == 0
    assert torch.equal(updated, apply_placement_action(layout, actions[0]))


def test_core_planner_noop_uses_exact_current_mapping():
    hierarchy = Hierarchy(ep_size=1, group_sizes=(1,), source="test")
    planner = CoReMoEPlanner(
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        communication_scale=1.0,
        forward_compute_per_assignment=1.0,
        route_sample_size=4,
    )
    selected = torch.tensor([[0, 1], [1, 0]])

    plan = planner.plan(
        selected,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=0,
        max_replicas=0,
    )

    assert plan.algorithm_version == CORE_MOE_ALGORITHM_VERSION
    assert plan.actions == ()
    assert plan.initial_layout == (0, 1)
    assert plan.final_layout == (0, 1)
    assert plan.final_owner_slots == (0, 1)
    assert torch.equal(plan.local_physical_routes, selected)


def test_device_route_summary_uses_integer_payload_and_excludes_padding():
    captured = []

    def gather_fixed(local):
        captured.append(local)
        return local.unsqueeze(0)

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        gather_fixed=gather_fixed,
        route_sample_size=4,
    )

    summary = planner._build_device_planning_summary(
        torch.tensor([[1]], dtype=torch.long),
        torch.tensor([0, 1]),
        source_rank=0,
        step=3,
        layer_seed=5,
        fused_capable=True,
    )

    assert captured[0].dtype == torch.long
    assert int(captured[0][1].item()) & _FUSED_CAP_QUOTA_POLICY
    assert summary.route.sample_routes.tolist() == [[1]]
    assert summary.route.sample_ordinals.tolist() == [0]
    assert summary.route.sample_multiplicity.tolist() == [[1]]
    assert summary.route.padded_sample_valid.tolist() == [[True, False, False, False]]
    assert summary.swap_stats.expert_token_counts.tolist() == [0.0, 1.0]
    assert summary.fused_capabilities.tolist() == [True]


def test_plan_uses_fused_swap_only_when_every_rank_is_capable(monkeypatch):
    fused_calls = []

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        route_sample_size=4,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()
    extension.quota_policy = _empty_quota_policy
    extension.quota_map = _first_copy_quota_map
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)

    def fused_swap(_stats, layout, owners, *, max_swaps, sample_routes, sample_weights, extension):
        assert extension is not None
        assert sample_routes.shape == sample_weights.shape + (2,)
        fused_calls.append(max_swaps)
        return (
            layout.clone(),
            owners.clone(),
            torch.full((max_swaps, 5), -1, dtype=torch.long),
            torch.tensor([0, -1, -1], dtype=torch.long),
            torch.cat(_stats.base_counts).clone(),
        )

    monkeypatch.setattr(planner, "_fused_swap_select", fused_swap)
    planner.plan(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )

    assert fused_calls == [1]

    def gather_with_mismatch(local):
        gathered = local.view(1, -1).expand(2, -1).clone()
        gathered[1, 1] = 0
        return gathered

    fallback = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
        gather_fixed=gather_with_mismatch,
        reducer=lambda value: value * 2,
        route_sample_size=2,
    )
    monkeypatch.setattr(fallback, "_fused_planner_extension", lambda _device: extension)
    monkeypatch.setattr(
        fallback,
        "_fused_swap_select",
        lambda *_args, **_kwargs: pytest.fail("capability mismatch must use the group-wide eager fallback"),
    )

    fallback.plan(
        torch.tensor([[0], [1]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )


def test_missing_quota_policy_capability_uses_eager_fallback(monkeypatch):
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        route_sample_size=2,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()
    extension.quota_map = lambda *_args: pytest.fail("missing quota_policy must use eager fallback")
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)
    monkeypatch.setattr(
        planner,
        "_fused_swap_select",
        lambda *_args, **_kwargs: pytest.fail("missing quota_policy must use eager fallback"),
    )

    planner.plan(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )


@pytest.mark.parametrize("mismatch", ["schema", "abi", "quota_policy_capability"])
def test_fused_protocol_mismatch_uses_existing_gather_for_group_fallback(monkeypatch, mismatch):
    gather_calls = []

    def gather_with_protocol_mismatch(local):
        gather_calls.append(local.clone())
        gathered = local.view(1, -1).expand(2, -1).clone()
        if mismatch == "schema":
            gathered[1, 0] += 1
        elif mismatch == "abi":
            gathered[1, 1] += 1 << _FUSED_PROTOCOL_ABI_SHIFT
        else:
            gathered[1, 1] -= _FUSED_CAP_QUOTA_POLICY
        return gathered

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
        gather_fixed=gather_with_protocol_mismatch,
        collective_backend="hccl",
        reducer=lambda value: value * 2,
        route_sample_size=2,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()
    extension.quota_policy = lambda *_args: pytest.fail("protocol mismatch must not call quota_policy")
    extension.quota_map = lambda *_args: pytest.fail("protocol mismatch must not call quota_map")
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)
    monkeypatch.setattr(
        planner,
        "_fused_swap_select",
        lambda *_args, **_kwargs: pytest.fail("protocol mismatch must use the group-wide eager fallback"),
    )

    planner.plan(
        torch.tensor([[0], [1]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )

    assert len(gather_calls) == 1


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_non_hccl_backend_uses_group_wide_eager_fallback(monkeypatch, backend):
    gather_calls = []

    def gather_fixed(local):
        gather_calls.append(local.clone())
        return local.view(1, -1).expand(2, -1).clone()

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
        gather_fixed=gather_fixed,
        collective_backend=backend,
        reducer=lambda value: value * 2,
        route_sample_size=2,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()
    extension.quota_policy = lambda *_args: pytest.fail("non-HCCL backend must not call quota_policy")
    extension.quota_map = lambda *_args: pytest.fail("non-HCCL backend must not call quota_map")
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)
    monkeypatch.setattr(
        planner,
        "_fused_swap_select",
        lambda *_args, **_kwargs: pytest.fail("non-HCCL backend must use the group-wide eager fallback"),
    )

    planner.plan(
        torch.tensor([[0], [1]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )

    assert len(gather_calls) == 1


def test_fused_path_skips_eager_scoring_reduces_once_and_reuses_physical_routes(monkeypatch):
    reduce_calls = []
    quota_calls = []

    def reducer(value):
        reduce_calls.append(value.clone())
        return value

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        reducer=reducer,
        route_sample_size=4,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()

    def quota_policy(*args):
        quota_calls.append(args)
        weights, configured, rows, row_counts, digest = _empty_quota_policy(*args)
        max_copies = args[10]
        weights[1, 0, 1, 0] = 2
        configured[1, 0, 1] = 1
        rows[1, 0, :4] = torch.tensor([0, 0, 1, 0])
        rows[1, 0, 3 + max_copies] = 2
        row_counts[1] = 1
        return weights, configured, rows, row_counts, digest

    extension.quota_policy = quota_policy

    def quota_map(selected, _copy_slots, _copy_counts, _owners, weights, configured, *_args):
        assert weights[1, 0, 1, 0].item() == 2
        assert configured[1, 0, 1].item() == 1
        physical = torch.stack((selected, selected.flip(-1)), dim=0)
        groups = torch.tensor([[8.0], [1.0]])
        assignments = torch.tensor([[8.0], [1.0]])
        return physical, groups, assignments

    extension.quota_map = quota_map
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)

    def fused_swap(_stats, layout, owners, **_kwargs):
        candidate_layout = layout.flip(0)
        candidate_owners = owners.flip(0)
        rows = torch.tensor([[0, 1, 0, 1, 1]], dtype=torch.long)
        return (
            candidate_layout,
            candidate_owners,
            rows,
            torch.tensor([1, 0, 0], dtype=torch.int32),
            torch.cat(_stats.base_counts).clone(),
        )

    monkeypatch.setattr(planner, "_fused_swap_select", fused_swap)
    monkeypatch.setattr(planner, "_score_sample_layout", lambda *_args, **_kwargs: pytest.fail("eager sample score"))
    monkeypatch.setattr(planner, "_plan_replicas", lambda *_args, **_kwargs: pytest.fail("eager replica plan"))
    monkeypatch.setattr(planner, "_score_exact_pair", lambda *_args, **_kwargs: pytest.fail("eager exact score"))

    plan = planner.plan(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        source_ranks=0,
        max_swaps=1,
        max_replicas=0,
    )

    assert len(reduce_calls) == 1
    assert len(quota_calls) == 1
    assert reduce_calls[0].shape == (2, 7)
    assert plan.actions == (PlacementAction("swap", 0, 1, 0, 1),)
    assert plan.final_layout == (1, 0)
    assert torch.equal(plan.local_physical_routes, torch.tensor([[1, 0], [0, 1]]))
    assert plan.quota_policy == ((0, 0, 1, 0, 2),)


def test_fused_replica_path_builds_hot_mask_and_owner_complement(monkeypatch):
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        gather_fixed=lambda local: local.view(1, -1).expand(2, -1).clone(),
        reducer=lambda value: value * 2,
        route_sample_size=2,
    )
    captured = {}
    extension = type("FakeFusedExtension", (), {})()
    extension.swap_select_with_stats = object()
    extension.quota_policy = _empty_quota_policy
    extension.quota_map = lambda selected, *_args: (
        torch.stack((selected, torch.full_like(selected, 1)), dim=0),
        torch.tensor([[8.0, 0.0, 8.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        torch.tensor([[8.0, 0.0], [0.0, 1.0]]),
    )

    def replica_project(*args):
        captured["project_seed"] = args[6].clone()
        captured["project_redundant"] = args[9].clone()
        captured["project_candidates"] = args[10].clone()
        return tuple(torch.zeros((1,), dtype=torch.float32) for _ in range(6))

    def replica_match(*args):
        captured["match_redundant"] = args[8].clone()
        captured["match_candidates"] = args[9].clone()
        updated = args[6].clone()
        updated[1] = 1
        rows = torch.full((2, 5), -1, dtype=torch.long)
        rows[0] = torch.tensor([2, 2, 1, 1, -1])
        return (
            updated,
            rows,
            torch.tensor([1.0, 0.0]),
            torch.zeros((2, 1), dtype=torch.int32),
            torch.zeros((2, 1, 4)),
            torch.tensor([1, 0, 0], dtype=torch.int32),
        )

    extension.replica_project = replica_project
    extension.replica_match = replica_match
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)
    monkeypatch.setattr(
        planner,
        "_fused_swap_select",
        lambda _stats, layout, owners, **_kwargs: (
            layout.clone(),
            owners.clone(),
            torch.full((1, 5), -1, dtype=torch.long),
            torch.tensor([0, 1, 1], dtype=torch.int32),
            torch.cat(_stats.base_counts).clone(),
        ),
    )
    monkeypatch.setattr(planner, "_score_sample_layout", lambda *_args, **_kwargs: pytest.fail("eager sample score"))
    monkeypatch.setattr(planner, "_plan_replicas", lambda *_args, **_kwargs: pytest.fail("eager replica plan"))
    monkeypatch.setattr(planner, "_score_exact_pair", lambda *_args, **_kwargs: pytest.fail("eager exact score"))

    plan = planner.plan(
        torch.tensor([[0], [1]]),
        torch.tensor([0, -1, 1, -1]),
        torch.tensor([0, 2]),
        source_ranks=0,
        max_swaps=0,
        max_replicas=16,
    )

    assert captured["project_redundant"].tolist() == [[1], [3]]
    assert captured["match_redundant"].tolist() == [[1], [3]]
    assert captured["project_candidates"].tolist() == [0, 1]
    assert captured["match_candidates"].tolist() == [0, 1]
    assert plan.actions == (PlacementAction("replica", 2, 1, 1, -1),)
    assert plan.final_layout == (0, 1, 1, -1)
    assert torch.equal(plan.local_physical_routes, torch.full((2, 1), 1, dtype=torch.long))


def test_fused_replica_path_rejects_untrusted_multiplicity():
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        reducer=lambda value: value,
    )
    summary = RouteSummary(
        token_counts=torch.tensor([1, 0]),
        assignment_counts=torch.tensor([[1, 0], [0, 0]]),
        sample_routes=torch.tensor([[0]]),
        sample_ordinals=torch.tensor([0]),
        sample_valid=torch.tensor([True]),
        sample_weights=torch.tensor([1.0]),
        sample_sources=torch.tensor([0]),
        sample_digest="untrusted-multiplicity",
        sample_multiplicity=torch.tensor([[1]]),
    )
    with pytest.raises(RuntimeError, match="canonical sampled route multiplicities"):
        planner._fused_plan_replicas(
            object(),
            summary,
            torch.tensor([0, -1, 1, -1]),
            torch.tensor([0, 2]),
            torch.tensor([0, 0, 0], dtype=torch.int32),
            torch.zeros((4,), dtype=torch.float32),
            max_replicas=1,
            step=0,
            layer_seed=0,
        )


def test_sample_assignment_projection_keeps_unsampled_exact_counts_on_owner():
    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=2,
        reducer=lambda value: value,
    )
    summary = RouteSummary(
        token_counts=torch.tensor([1, 1]),
        assignment_counts=torch.tensor([[1, 0], [1, 7]]),
        sample_routes=torch.tensor([[0], [0]]),
        sample_ordinals=torch.tensor([0, 1]),
        sample_valid=torch.tensor([True, True]),
        sample_weights=torch.ones((2,)),
        sample_sources=torch.tensor([0, 1]),
        sample_digest="missing-sample-assignment",
        sample_multiplicity=torch.ones((2, 1), dtype=torch.long),
        sample_multiplicity_is_canonical=True,
    )
    projected = planner._project_sample_assignments(
        summary,
        torch.tensor([[0], [0]]),
        torch.tensor([0, 2]),
    )
    assert torch.equal(projected, torch.tensor([2.0, 7.0]))


def test_fused_collective_guard_rejects_inconsistent_policy_digest_when_enabled(monkeypatch):
    def corrupt_reducer(value):
        reduced = value * 2
        reduced[1, -4] += 1
        return reduced

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
        gather_fixed=lambda local: local.view(1, -1).expand(2, -1).clone(),
        reducer=corrupt_reducer,
        route_sample_size=2,
        verify_collective_digest=True,
    )
    extension = type("FakeFusedExtension", (), {})()
    extension.quota_policy = _empty_quota_policy
    extension.quota_map = _first_copy_quota_map
    monkeypatch.setattr(planner, "_fused_planner_extension", lambda _device: extension)

    with pytest.raises(RuntimeError, match="inconsistent placement plans"):
        planner.plan(
            torch.tensor([[0]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            source_ranks=0,
            max_swaps=0,
            max_replicas=0,
        )


@pytest.mark.parametrize("debug_digest", [True])
def test_exact_collective_guard_rejects_inconsistent_plan_digest_when_enabled(debug_digest):
    def gather_fixed(local):
        return local.view(1, -1).expand(2, -1).clone()

    def corrupt_reducer(value):
        reduced = value * 2
        reduced[0, -3] += 1
        return reduced

    planner = CoReMoEPlanner(
        hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
        reducer=corrupt_reducer,
        gather_fixed=gather_fixed,
        route_sample_size=2,
        verify_collective_digest=debug_digest,
    )

    with pytest.raises(RuntimeError, match="inconsistent placement plans"):
        planner.plan(
            torch.tensor([[0], [1]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            source_ranks=0,
            max_swaps=0,
            max_replicas=0,
        )
