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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from placemoe import planner as placemoe_planner
from veomni.distributed.moe.hiermoe import all_to_all as all_to_all_module
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager
from veomni.distributed.moe.hiermoe.placemoe import (
    LayerPlan,
    PlaceMoETopology,
    build_placemoe_artifact,
)
from veomni.distributed.moe.hiermoe.placemoe.runtime import HotUpdateController
from veomni.distributed.moe.hiermoe.placemoe.runtime.config import PlaceMoEPlannerResources


def _runtime_manager() -> ExpertSwapManager:
    manager = object.__new__(ExpertSwapManager)
    manager._hot_update_resources = PlaceMoEPlannerResources()
    manager.ep_rank = 0
    manager.ep_size = 1
    manager.ep_group = None
    return manager


def test_hot_update_uses_canonical_placemoe_cli(monkeypatch):
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_BUILDER", "")

    path = _runtime_manager()._hot_update_builder_path()

    assert Path(path).resolve() == Path(placemoe_planner.__file__).resolve()


def test_hot_update_enables_route_capture_without_redundant_slots():
    manager = _runtime_manager()
    manager._hot_update = True
    manager._online_lut_update = False
    manager._initial_layout_path = ""
    manager._ablation_replay_mode = "off"
    manager.expert_swap_max_pairs_per_layer = 0
    manager.redundant_slot_increment_per_device = 0

    assert manager.placement_planning_enabled()


def test_planner_uses_only_intra_node_cost_for_single_node_hierarchy():
    args = SimpleNamespace(
        hierarchy_group_sizes=(4,),
        ep_size=4,
        ranks_per_node=4,
        inter_ms_per_byte=9.0,
        intra_ms_per_byte=2.0,
        mid_ms_per_byte=None,
        hidden_size=8,
        bytes_per_element=2,
        communication_phase_multiplier=3.0,
        compute_phase_multiplier=4.0,
        compute_ms_per_assignment=5.0,
    )

    hierarchy, omegas, gamma = placemoe_planner._hierarchy_coefficients(args)

    assert hierarchy == (4,)
    assert omegas == (96.0,)
    assert gamma == 20.0


def test_rank_only_dispatch_records_one_stage_a2a_events(monkeypatch):
    monkeypatch.setattr(all_to_all_module, "_HIERMOE_INTERNAL_TIMING", True)
    monkeypatch.setattr(all_to_all_module, "_hiermoe_internal_event", object)
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    selected = torch.tensor([[0], [1]], dtype=torch.long)
    weights = torch.ones((2, 1), dtype=torch.float32)

    permuted, context, _counts = all_to_all_module.rank_dedup_dispatch(
        hidden,
        selected,
        weights,
        num_experts=2,
        ep_group=None,
        layer_key="layers.0.experts",
    )
    output = all_to_all_module.rank_dedup_combine(permuted, context)

    assert torch.equal(output, hidden)
    assert context.layer_key == "layers.0.experts"
    assert context.internal_timing_events is not None
    assert "stage2_a2a" in context.internal_timing_events
    assert "combine_stage2_a2a" in context.internal_timing_events


def test_hot_update_validates_canonical_artifact(tmp_path):
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

    loaded = _runtime_manager()._broadcast_hot_update_payload(state, torch.device("cpu"))

    assert loaded == payload


def test_hot_update_rejects_legacy_artifact(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps({"schema_version": 1, "layers": {}}), encoding="utf-8")
    state = SimpleNamespace(layout_path=str(layout_path))

    with pytest.raises(RuntimeError, match="invalid PlaceMoE artifact"):
        _runtime_manager()._broadcast_hot_update_payload(state, torch.device("cpu"))


def test_hot_update_rejects_non_placemoe_schema_v2_artifact(tmp_path):
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
        _runtime_manager()._broadcast_hot_update_payload(state, torch.device("cpu"))


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
def test_hot_update_schedules_independent_layout_and_mapping_intervals(
    monkeypatch,
    layout_interval,
    mapping_interval,
    placement_step,
    expected_mode,
):
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_LAST_STEP", 10_000)
    manager = _runtime_manager()
    manager._hot_update_layout_interval = layout_interval
    manager._hot_update_mapping_interval = mapping_interval
    manager._hot_update_controller = HotUpdateController(
        layout_interval_steps=layout_interval,
        mapping_interval_steps=mapping_interval,
        last_update_step=10_000,
    )
    manager.latest_pair = ""
    launched = []
    manager._launch_hot_update = lambda **kwargs: launched.append(kwargs)

    result = manager._run_hot_update_step(placement_step)

    if expected_mode is None:
        assert result == "none"
        assert not launched
    else:
        assert launched[0]["update_mode"] == expected_mode
        assert result == f"placemoe_{expected_mode}_update_submitted:{placement_step + 1}"


def test_hot_update_passes_calibration_coefficients_to_planner(monkeypatch, tmp_path):
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
    manager.hierarchy = SimpleNamespace(local_world_size=1, group_sizes=(1,))
    manager._hot_update_controller = HotUpdateController(100, 20, 500)
    manager._hot_update_last_source_step = -1
    manager._hot_update_last_snapshot_ms = 0.0
    manager._capture_hot_update_routes = lambda *_args: 1.0
    manager._hot_update_builder_path = lambda: "/bin/true"
    manager._hot_update_event = lambda *_args, **_kwargs: None
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_WORK_ROOT", str(tmp_path))
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_INTER_MS_PER_BYTE", 1.25)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_INTRA_MS_PER_BYTE", 2.5)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT", 3.75)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_COMMUNICATION_MULTIPLIER", 4.5)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT", 5.25)
    monkeypatch.setattr(expert_swap_module, "_HOT_UPDATE_COMPUTE_MULTIPLIER", 6.5)
    commands = []

    class _Process:
        pid = 123

        def __init__(self, command, **_kwargs):
            commands.append(command)

    monkeypatch.setattr(expert_swap_module.subprocess, "Popen", _Process)

    manager._launch_hot_update(
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
        "--hierarchy-group-sizes": "1",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value


def test_canonical_hot_update_keeps_current_pair_when_planner_fails(monkeypatch) -> None:
    manager = _runtime_manager()
    manager._hot_update_controller = HotUpdateController(100, 20, 500)
    process = object()
    manager._hot_update_controller.active_job = SimpleNamespace(
        update_mode="mapping",
        source_step=20,
        planner_log_path="planner.log",
        process=process,
    )
    manager.layers = {"layer": object()}
    manager.latest_pair = ""
    manager._pipeline_device = lambda _layer: torch.device("cpu")
    manager._hot_update_status = lambda _state, _device: 2
    manager._hot_update_event = lambda *_args, **_kwargs: None
    terminated = []
    monkeypatch.setattr(expert_swap_module, "terminate_planner_process", terminated.append)

    result = manager._run_hot_update_step(20)

    assert result == "placemoe_hot_update_failed:20"
    assert terminated == [process]
    assert manager._hot_update_controller.active_job is None


def test_hot_update_validates_all_layers_before_migration() -> None:
    manager = _runtime_manager()
    manager.layers = {
        "layer.0": SimpleNamespace(key="layer.0", placement_version=0),
        "layer.1": SimpleNamespace(key="layer.1", placement_version=0),
    }
    manager._pipeline_device = lambda _layer: torch.device("cpu")
    manager._broadcast_hot_update_payload = lambda _state, _device: {"layers": {"layer.0": {}, "layer.1": {}}}
    prepared = []

    def prepare(layer, _payload, _mode):
        prepared.append(layer)
        if layer is manager.layers["layer.1"]:
            raise RuntimeError("invalid second layer")

    installed = []
    manager._prepare_hot_update_layer = prepare
    manager._install_hot_update_layout = lambda layer, _payload: installed.append(layer)
    state = SimpleNamespace(placement_versions=(0, 0), update_mode="full")

    with pytest.raises(RuntimeError, match="invalid second layer"):
        manager._apply_hot_update(state, training_step=10)

    assert prepared == [manager.layers["layer.0"], manager.layers["layer.1"]]
    assert installed == []
