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

from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager, expand_redundant_expert_slots
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.placemoe.model_adapter import resolve_moe_model_adapter
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.distributed.parallel_plan import _is_hiermoe_redundant_slot_expert_param as parallel_slot_param
from veomni.models.module_utils import _is_hiermoe_redundant_slot_expert_param as checkpoint_slot_param
from veomni.ops.kernels.moe import _make_moe_experts_adapter


class _SplitProjectionExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_proj = nn.Parameter(torch.ones(2, 3, 2))
        self.up_proj = nn.Parameter(torch.full((2, 3, 2), 2.0))
        self.down_proj = nn.Parameter(torch.full((2, 2, 3), 3.0))


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
