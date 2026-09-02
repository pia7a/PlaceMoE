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

from __future__ import annotations

import pytest
import torch
from torch import nn

from placemoe import register_moe_model_adapter, resolve_moe_model_adapter
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager, expand_redundant_expert_slots
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.distributed.parallel_plan import _is_hiermoe_redundant_slot_expert_param as parallel_slot_param
from veomni.models.module_utils import _is_hiermoe_redundant_slot_expert_param as checkpoint_slot_param
from veomni.ops.kernels.moe import _make_moe_experts_adapter
from veomni.utils.accelerator_timing import AcceleratorEvent


class _SplitProjectionExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_proj = nn.Parameter(torch.ones(2, 3, 2))
        self.up_proj = nn.Parameter(torch.full((2, 3, 2), 2.0))
        self.down_proj = nn.Parameter(torch.full((2, 2, 3), 3.0))


class _SingleRankExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.gate_proj = nn.Parameter(torch.ones(2, 3, 2))
        self.up_proj = nn.Parameter(torch.full((2, 3, 2), 2.0))
        self.down_proj = nn.Parameter(torch.full((2, 2, 3), 3.0))


def test_public_adapter_api_is_available_without_hiermoe_import_path() -> None:
    assert callable(register_moe_model_adapter)
    assert resolve_moe_model_adapter(_SplitProjectionExperts()).name == "stacked-split-gate-up"


def test_model_registration_fails_when_no_expert_adapter_matches() -> None:
    manager = object.__new__(ExpertSwapManager)

    with pytest.raises(RuntimeError, match="did not find a supported expert module"):
        manager.register_model(nn.Sequential(nn.Linear(2, 2)))


def _manager() -> ExpertSwapManager:
    return ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
    )


def test_split_projection_adapter_expands_and_registers_all_expert_parameters() -> None:
    module = _SplitProjectionExperts()

    assert expand_redundant_expert_slots(module, ep_size=2, redundant_slot_increment_per_device=1) == 1
    assert tuple(module.gate_proj.shape) == (3, 3, 2)
    assert tuple(module.up_proj.shape) == (3, 3, 2)
    assert tuple(module.down_proj.shape) == (3, 2, 3)
    assert torch.count_nonzero(module.gate_proj[-1]) == 0
    assert torch.count_nonzero(module.up_proj[-1]) == 0
    assert torch.count_nonzero(module.down_proj[-1]) == 0

    manager = _manager()
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]

    assert layer.expert_parameter_names == ("gate_proj", "up_proj", "down_proj")
    assert tuple(map(id, layer.expert_parameters)) == tuple(
        map(id, (module.gate_proj, module.up_proj, module.down_proj))
    )
    assert all(manager.param_id_to_key[id(parameter)] == layer.key for parameter in layer.expert_parameters)


def test_cost_model_normalizes_compact_identity_routes_for_expanded_slots() -> None:
    module = _SplitProjectionExperts()
    expand_redundant_expert_slots(module, ep_size=2, redundant_slot_increment_per_device=1)
    manager = _manager()
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.latest_hidden_size = 3
    layer.latest_bytes_per_element = 4
    compact_routes = torch.arange(4, dtype=torch.long).view(1, 4, 1)

    planner_routes = manager._routes_for_cost_model_planner(layer, compact_routes)

    assert planner_routes.flatten().tolist() == [0, 1, 3, 4]
    planner = manager._cpu_exact_planner_for_layer(layer)
    assignment_counts = planner._local_packed_assignment_counts(planner_routes)
    assert assignment_counts[:, : manager.ep_size].tolist() == [[2.0, 2.0]]


def test_cost_model_preserves_routes_after_expanded_layout_activation() -> None:
    module = _SplitProjectionExperts()
    expand_redundant_expert_slots(module, ep_size=2, redundant_slot_increment_per_device=1)
    manager = _manager()
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.slot_to_logical is not None
    layer.slot_to_logical[2] = 0
    layer.refresh_identity()
    expanded_routes = torch.tensor([[[0], [1], [3], [4]]], dtype=torch.long)

    planner_routes = manager._routes_for_cost_model_planner(layer, expanded_routes)

    assert planner_routes is expanded_routes


def test_split_projection_adapter_normalizes_fused_kernel_arguments() -> None:
    module = _SplitProjectionExperts()
    captured = {}

    def raw_forward(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    hidden = torch.zeros(2, 2)
    selected = torch.zeros(2, 1, dtype=torch.long)
    weights = torch.ones(2, 1)

    result = _make_moe_experts_adapter(raw_forward)(module, hidden, selected, weights)

    assert result is hidden
    assert captured["fc1_1_weight"] is module.gate_proj
    assert captured["fc1_2_weight"] is module.up_proj
    assert captured["fc2_weight"] is module.down_proj
    assert captured["fc1_1_2_weight"] is None
    assert resolve_moe_model_adapter(module).name == "stacked-split-gate-up"


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj", "gate_up_proj"])
def test_checkpoint_paths_recognize_fused_and_split_expert_parameters(projection: str) -> None:
    name = f"model.layers.0.mlp.experts.{projection}"
    global_shape = torch.Size((8, 4, 2))
    local_shape = torch.Size((3, 4, 2))

    assert parallel_slot_param("ep", name, global_shape, tuple(local_shape), 4)
    assert checkpoint_slot_param("ep", name, global_shape, local_shape, 4)


def test_hot_update_seeds_source_mapping_without_initial_artifact(monkeypatch) -> None:
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE", True)
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER", False)
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER_PATCH_REMAP", False)
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_MODE", "off")

    module = _SplitProjectionExperts()
    expand_redundant_expert_slots(module, ep_size=2, redundant_slot_increment_per_device=1)
    manager = ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=0,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
        fixed_pipeline_overlap=True,
    )
    manager.register_layer("layers.0.mlp.experts", module)

    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.source_logical_to_physical is not None
    assert layer.source_logical_to_physical.tolist() == [[0, 1, 3, 4], [0, 1, 3, 4]]


def test_zero_replica_hot_update_constructs_manager_and_seeds_mapping(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE", True)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_WORK_ROOT", str(tmp_path))
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER", False)
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER_PATCH_REMAP", False)
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_MODE", "off")
    monkeypatch.setattr(expert_swap_module, "_COST_MODEL_VERIFY", False)

    manager = ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="current_joint",
        fixed_pipeline_overlap=False,
    )
    module = _SplitProjectionExperts()
    manager.register_layer("layers.0.mlp.experts", module)

    layer = manager.layers["layers.0.mlp.experts"]
    assert manager.placement_planning_enabled()
    assert layer.source_logical_to_physical is not None
    assert layer.source_logical_to_physical.tolist() == [list(range(4)), list(range(4))]
    assert not manager.gradient_overlap_enabled


def test_zero_replica_cost_model_calibration_accepts_current_joint(monkeypatch) -> None:
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE", False)
    monkeypatch.setattr(expert_swap_module, "_COST_MODEL_VERIFY", True)
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER", False)
    monkeypatch.setattr(expert_swap_module, "_ONLINE_FREEZE_COST_MODE", "off")

    manager = ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="current_joint",
        fixed_pipeline_overlap=False,
    )

    assert manager._cost_model_verify
    assert manager.placement_planning_enabled()


def test_rank_only_replication_keeps_gradient_overlap_without_fixed_pipeline(monkeypatch) -> None:
    ep_group = object()
    gradient_group = object()
    monkeypatch.setattr(
        expert_swap_module,
        "_create_expert_swap_process_group",
        lambda *_args, **_kwargs: gradient_group,
    )
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE", False)
    monkeypatch.setattr(expert_swap_module, "_COST_MODEL_VERIFY", False)

    manager = ExpertSwapManager(
        ep_group=ep_group,
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
        fixed_pipeline_overlap=False,
    )

    assert manager.gradient_overlap_enabled
    assert manager._pipeline_grad_group is gradient_group
    assert not manager.fixed_pipeline_overlap


def test_rank_only_cost_model_collects_direct_a2a_observations(monkeypatch) -> None:
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE", False)
    monkeypatch.setattr(expert_swap_module, "_COST_MODEL_VERIFY", True)
    monkeypatch.setattr(expert_swap_module, "_FORWARD_REUSE_COVER", False)
    monkeypatch.setattr(expert_swap_module, "_ONLINE_FREEZE_COST_MODE", "off")
    monkeypatch.setattr(expert_swap_module, "synchronize", lambda: None)
    manager = ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="current_joint",
        fixed_pipeline_overlap=False,
    )
    manager.register_layer("layers.0.mlp.experts", _SingleRankExperts())
    layer = manager.layers["layers.0.mlp.experts"]
    routes = torch.tensor([[0], [1]], dtype=torch.long)
    layer.latest_physical_routes = routes
    layer.latest_hidden_size = 3
    layer.latest_bytes_per_element = 4

    def event(milliseconds: float) -> AcceleratorEvent:
        return AcceleratorEvent(device_type="cpu", event=None, wall_time=milliseconds / 1000.0)

    manager.record_layer_timing(
        layer_key=layer.key,
        step=1,
        selected_experts=routes,
        tokens_per_local_expert=torch.tensor([1, 1]),
        dispatch_start=event(0.0),
        dispatch_end=event(2.0),
        compute_start=event(2.0),
        compute_end=event(5.0),
        combine_start=event(5.0),
        combine_end=event(7.0),
        communication_events={
            "stage2_a2a": (event(0.5), event(1.5)),
            "combine_stage2_a2a": (event(5.5), event(6.5)),
        },
    )

    observations = manager._cost_model_step_observations([layer], step=1)

    assert observations["actual_stage_a2a_names"] == ["stage2_a2a", "combine_stage2_a2a"]
    assert observations["actual_stage_a2a_ms"][0] == pytest.approx([1.0, 1.0])
    assert observations["traffic_features"]["stage1_payload_endpoint_bytes"] == [0.0]
    assert observations["traffic_features"]["stage2_payload_endpoint_bytes"][0] > 0.0
    alignment = observations["sample_alignment"]
    assert alignment["row_layer_indices"] == [0]
    assert alignment["row_call_indices"] == [0]
    assert alignment["source_assignment_totals"] == pytest.approx([2.0])
    assert alignment["destination_assignment_totals"] == pytest.approx([2.0])
    assert alignment["destination_rank_mismatch_counts"] == [0]
    assert alignment["destination_rank_max_abs_deltas"] == pytest.approx([0.0])
