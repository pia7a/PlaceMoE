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

import torch

from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.forward_cover_planner import (
    forward_cover_local_validation_stats,
    patch_forward_cover_routes,
    propose_forward_reuse_cover,
)
from veomni.distributed.moe.hiermoe.greedy_planner import assign_tokens_to_copies_greedy
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import PlacementAction
from veomni.distributed.moe.hiermoe.topology import Hierarchy


class _FakeExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 2, 2))
        self.down_proj = torch.nn.Parameter(torch.arange(12, 24, dtype=torch.float32).view(3, 2, 2))


def _toy_inputs() -> dict[str, object]:
    return {
        "selected_experts": torch.tensor(
            [
                [2, 3],
                [2, 0],
                [2, 3],
                [1, 3],
            ],
            dtype=torch.long,
        ),
        "physical_routes": torch.tensor(
            [
                [3, 5],
                [3, 0],
                [3, 5],
                [1, 5],
            ],
            dtype=torch.long,
        ),
        "slot_to_logical": torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long),
        "owner_slots": torch.tensor([0, 2, 4, 6], dtype=torch.long),
        "local_slot_assignments": torch.tensor([20.0, 1.0]),
        "source_rank": 0,
        "slots_per_rank": 2,
        "hierarchy_group_sizes": (2,),
        "num_experts": 4,
        "max_copies": 4,
        "level_weights": (1.0, 4.0),
    }


def test_forward_reuse_cover_prefers_expert_reducing_remote_node_bottleneck():
    proposal = propose_forward_reuse_cover(**_toy_inputs(), compute_weight=0.0)

    assert proposal.action is not None
    assert proposal.action.src_logical == 3
    assert proposal.action.src_slot == 6
    assert proposal.action.dst_slot == 1
    assert proposal.action.dst_logical == 1
    assert proposal.estimated_gain > 0.0
    assert proposal.communication_gain > 0.0


def test_forward_reuse_cover_can_evict_an_owner_with_another_copy():
    inputs = _toy_inputs()
    inputs["local_slot_assignments"] = torch.tensor([0.0, 20.0])

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=0.0)

    assert proposal.action is not None
    assert proposal.action.dst_slot == 0
    assert proposal.action.dst_logical == 0


def test_forward_reuse_cover_can_choose_none_when_compute_penalty_dominates():
    proposal = propose_forward_reuse_cover(**_toy_inputs(), compute_weight=100.0)

    assert proposal.action is None
    assert proposal.estimated_gain <= 0.0


def test_forward_reuse_cover_requires_a_non_owner_target_slot():
    inputs = _toy_inputs()
    inputs["slot_to_logical"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["owner_slots"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["local_slot_assignments"] = torch.tensor([20.0])
    inputs["slots_per_rank"] = 1
    inputs["physical_routes"] = torch.tensor(
        [
            [2, 3],
            [2, 0],
            [2, 3],
            [1, 3],
        ],
        dtype=torch.long,
    )

    proposal = propose_forward_reuse_cover(**inputs, compute_weight=0.0)

    assert proposal.action is None


def test_forward_cover_affected_token_validation_matches_full_remap():
    selected = _toy_inputs()["selected_experts"]
    layout = _toy_inputs()["slot_to_logical"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(layout, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )
    baseline = assign_tokens_to_copies_greedy(
        selected,
        layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        step=3,
        layer_seed=17,
        max_copies=4,
    )
    validation = forward_cover_local_validation_stats(
        selected_experts=selected,
        physical_routes=baseline,
        slot_to_logical=layout,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        max_copies=4,
        step=3,
        layer_seed=17,
    )
    candidate_layout = layout.clone()
    candidate_layout[action.dst_slot] = action.src_logical
    candidate = assign_tokens_to_copies_greedy(
        selected,
        candidate_layout,
        slots_per_rank=2,
        source_ranks=0,
        hierarchy_group_sizes=(2,),
        num_experts=4,
        step=3,
        layer_seed=17,
        max_copies=4,
    )

    def packed_counts(physical: torch.Tensor) -> torch.Tensor:
        ranks = physical // 2
        rows = []
        for size, groups in ((1, ranks), (2, ranks // 2)):
            hits = torch.zeros((physical.shape[0], 4 // size), dtype=torch.bool)
            hits.scatter_(1, groups, True)
            rows.append(hits.sum(dim=0).to(torch.float32))
        return torch.cat(rows)

    def assignment_counts(physical: torch.Tensor) -> torch.Tensor:
        return torch.bincount((physical // 2).reshape(-1), minlength=4).to(torch.float32)

    torch.testing.assert_close(
        validation.baseline_communication_counts + validation.communication_count_delta,
        packed_counts(candidate),
    )
    torch.testing.assert_close(
        validation.baseline_assignment_counts + validation.assignment_count_delta,
        assignment_counts(candidate),
    )
    assert validation.affected_tokens == int(
        ((selected == action.src_logical) | (selected == action.dst_logical)).any(dim=1).sum().item()
    )


def test_forward_cover_patch_changes_only_inserted_and_evicted_routes():
    selected = _toy_inputs()["selected_experts"]
    physical = _toy_inputs()["physical_routes"]
    assert isinstance(selected, torch.Tensor)
    assert isinstance(physical, torch.Tensor)
    action = PlacementAction(
        kind="replica",
        src_slot=6,
        dst_slot=1,
        src_logical=3,
        dst_logical=1,
    )

    patched = patch_forward_cover_routes(
        selected_experts=selected,
        physical_routes=physical,
        action=action,
        source_rank=0,
        slots_per_rank=2,
        victim_fallback_slot=2,
    )

    expected = torch.tensor(
        [
            [3, 1],
            [3, 0],
            [3, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(patched, expected)
    unaffected = (selected != action.src_logical) & (physical != action.dst_slot)
    torch.testing.assert_close(patched[unaffected], physical[unaffected])


def test_forward_reuse_cover_executor_batches_transfer_before_layout_publish(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._forward_reuse_cover_patch_remap = True
    layer.source_logical_to_physical = torch.tensor(
        [
            [0, 1, 3, 4],
            [5, 1, 3, 4],
        ],
        dtype=torch.long,
    )

    calls = []

    def capture(grouped_entries, **_kwargs):
        calls.append(grouped_entries)
        assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 2, 2, 3, 0)

    monkeypatch.setattr(manager, "_execute_sparse_group_slot_transfers", capture)
    action = PlacementAction(
        kind="replica",
        src_slot=4,
        dst_slot=2,
        src_logical=3,
        dst_logical=2,
    )

    committed = manager._execute_forward_cover_actions(((layer, action),))

    assert len(calls) == 1
    assert (1, 0) in calls[0]
    assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 3, 2, 3, 0)
    assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 3, 4)
    assert tuple(layer.source_logical_to_physical[0].tolist()) == (0, 1, 3, 2)
    assert tuple(layer.source_logical_to_physical[1].tolist()) == (5, 1, 3, 4)
    selected = torch.tensor([[3, 2, 0]], dtype=torch.long)
    torch.testing.assert_close(
        manager.map_logical_to_physical(key, selected),
        torch.tensor([[2, 3, 0]], dtype=torch.long),
    )
    assert committed == [f"{key}:replica(3->2)"]


def test_forward_reuse_cover_executor_promotes_remaining_copy_when_owner_is_victim(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._forward_reuse_cover_patch_remap = True
    layer.source_logical_to_physical = torch.tensor(
        [
            [0, 1, 3, 4],
            [5, 1, 3, 4],
        ],
        dtype=torch.long,
    )
    monkeypatch.setattr(manager, "_execute_sparse_group_slot_transfers", lambda *_args, **_kwargs: None)
    action = PlacementAction(
        kind="replica",
        src_slot=4,
        dst_slot=0,
        src_logical=3,
        dst_logical=0,
    )

    committed = manager._execute_forward_cover_actions(((layer, action),))

    assert tuple(layer.slot_to_logical.tolist()) == (3, 1, 2, 2, 3, 0)
    assert tuple(layer.logical_to_physical.tolist()) == (5, 1, 3, 4)
    assert tuple(layer.source_logical_to_physical[0].tolist()) == (5, 1, 3, 0)
    assert tuple(layer.source_logical_to_physical[1].tolist()) == (5, 1, 3, 4)
    assert committed == [f"{key}:replica(3->0)"]
