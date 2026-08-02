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


@pytest.mark.parametrize(
    ("layout_interval", "mapping_interval", "placement_step", "expected_mode"),
    [
        (0, 0, 99, None),
        (100, 100, 99, "full"),
        (100, 20, 19, "mapping"),
        (100, 0, 99, "full"),
        (0, 100, 99, "mapping"),
    ],
)
def test_periodic_replan_schedules_independent_layout_and_mapping_intervals(
    monkeypatch,
    layout_interval,
    mapping_interval,
    placement_step,
    expected_mode,
):
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_FULL_REPLAN_LAST_STEP", 10_000)
    manager = _runtime_manager()
    manager._periodic_layout_refresh_interval = layout_interval
    manager._periodic_mapping_refresh_interval = mapping_interval
    manager._periodic_pending_layout = False
    manager._periodic_pending_mapping = False
    manager._periodic_full_replan_state = None
    manager.latest_pair = ""
    launched = []
    manager._launch_periodic_full_replan = lambda **kwargs: launched.append(kwargs)

    result = manager._run_periodic_full_replan_step(placement_step)

    if expected_mode is None:
        assert result == "none"
        assert not launched
    else:
        assert launched[0]["update_mode"] == expected_mode
        assert result == f"periodic_{expected_mode}_replan_submitted:{placement_step + 1}"


def test_periodic_replan_passes_calibration_coefficients_to_planner(monkeypatch, tmp_path):
    manager = _runtime_manager()
    layer = SimpleNamespace(
        key="layers.0.experts",
        num_experts=2,
        num_local_experts=2,
        latest_hidden_size=16,
        latest_bytes_per_element=2,
        placement_version=3,
    )
    manager.layers = {layer.key: layer}
    manager.hierarchy = SimpleNamespace(local_world_size=1)
    manager._periodic_full_replan_state = None
    manager._periodic_full_replan_last_source_step = -1
    manager._periodic_full_replan_last_snapshot_ms = 0.0
    manager._capture_periodic_full_routes = lambda *_args: 1.0
    manager._periodic_full_replan_builder_path = lambda: "/bin/true"
    manager._periodic_full_replan_event = lambda *_args, **_kwargs: None
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_FULL_REPLAN_WORK_ROOT", str(tmp_path))
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_FULL_REPLAN_CPU_IDS", "")
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_INTER_MS_PER_BYTE", 1.25)
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_INTRA_MS_PER_BYTE", 2.5)
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_ROUTE_MS_PER_ASSIGNMENT", 3.75)
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_COMMUNICATION_MULTIPLIER", 4.5)
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_COMPUTE_MS_PER_ASSIGNMENT", 5.25)
    monkeypatch.setattr(expert_swap_module, "_PERIODIC_REPLAN_COMPUTE_MULTIPLIER", 6.5)
    commands = []

    class _Process:
        pid = 123

        def __init__(self, command, **_kwargs):
            commands.append(command)

    monkeypatch.setattr(expert_swap_module.subprocess, "Popen", _Process)

    manager._launch_periodic_full_replan(
        placement_step=9,
        training_step=10,
        update_mode="full",
    )

    command = commands[0]
    expected = {
        "--inter-ms-per-byte": "1.25",
        "--intra-ms-per-byte": "2.5",
        "--route-ms-per-assignment": "3.75",
        "--communication-phase-multiplier": "4.5",
        "--compute-ms-per-assignment": "5.25",
        "--compute-phase-multiplier": "6.5",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
