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

import json
from types import SimpleNamespace

import pytest
import torch

from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.expert_swap import (
    ExpertSwapManager,
    _encode_periodic_full_replan_layer_keys,
)
from veomni.distributed.moe.hiermoe.placemoe import (
    LayerPlan,
    PlaceMoETopology,
    build_placemoe_artifact,
)


def _runtime_manager() -> ExpertSwapManager:
    manager = object.__new__(ExpertSwapManager)
    manager.ep_rank = 0
    manager.ep_size = 1
    manager.ep_group = None
    return manager


def test_periodic_full_replan_uses_canonical_placemoe_cli(monkeypatch):
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_FULL_REPLAN_BUILDER", "")

    path = _runtime_manager()._periodic_full_replan_builder_path()

    assert path.endswith("/scripts/profile/plan_placemoe.py")


def test_periodic_full_replan_passes_runtime_layer_keys() -> None:
    layers = [
        SimpleNamespace(key="model.layers.2.mlp.experts"),
        SimpleNamespace(key="model.layers.10.mlp.experts"),
    ]

    encoded = _encode_periodic_full_replan_layer_keys(layers)

    assert encoded == "model.layers.2.mlp.experts,model.layers.10.mlp.experts"


def test_periodic_full_replan_validates_canonical_artifact(tmp_path):
    topology = PlaceMoETopology(ep_size=1, ranks_per_node=1, num_experts=2, slots_per_rank=3)
    plan = LayerPlan(
        slot_to_logical=[0, 1, 0],
        owner_slots=[0, 1],
        source_logical_to_physical=[[2, 1]],
    )
    payload = build_placemoe_artifact({"layers.0.experts": plan}, topology)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(layout_path=str(layout_path))

    loaded = _runtime_manager()._broadcast_periodic_full_replan_payload(state, torch.device("cpu"))

    assert loaded == payload


def test_periodic_full_replan_rejects_legacy_artifact(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps({"schema_version": 1, "layers": {}}), encoding="utf-8")
    state = SimpleNamespace(layout_path=str(layout_path))

    with pytest.raises(RuntimeError, match="invalid PlaceMoE artifact"):
        _runtime_manager()._broadcast_periodic_full_replan_payload(state, torch.device("cpu"))


def test_periodic_full_replan_rejects_non_placemoe_schema_v2_artifact(tmp_path):
    topology = PlaceMoETopology(ep_size=1, ranks_per_node=1, num_experts=2, slots_per_rank=3)
    plan = LayerPlan(
        slot_to_logical=[0, 1, 0],
        owner_slots=[0, 1],
        source_logical_to_physical=[[2, 1]],
    )
    payload = build_placemoe_artifact({"layers.0.experts": plan}, topology)
    payload["source"]["algorithm"] = "legacy-structured"
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(payload), encoding="utf-8")
    state = SimpleNamespace(layout_path=str(layout_path))

    with pytest.raises(RuntimeError, match="invalid PlaceMoE artifact"):
        _runtime_manager()._broadcast_periodic_full_replan_payload(state, torch.device("cpu"))
