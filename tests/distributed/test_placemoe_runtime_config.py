# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from veomni.arguments.arguments_types import HierMoEConfig, PlaceMoEArguments
from veomni.distributed.moe.hiermoe.placemoe.runtime import (
    HotUpdateController,
    HotUpdateScheduler,
    PlaceMoERuntimeConfig,
    UpdateKind,
    launch_planner_process,
    planner_process,
    set_current_runtime_config,
    terminate_planner_process,
)
from veomni.distributed.moe.hiermoe.placemoe.runtime.config import PlaceMoEConfigurationError
from veomni.distributed.moe.hiermoe.state import _resolve_placemoe_runtime_config
from veomni.distributed.parallel_plan import _hiermoe_initial_layout_path


def test_runtime_config_loads_single_file_and_resolves_paths(tmp_path) -> None:
    artifact = tmp_path / "initial.json"
    artifact.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text(
        """
placemoe:
  initial_artifact: initial.json
  runtime_perf_model: perf.json
  hot_update:
    enabled: true
    layout_interval_steps: 100
    mapping_interval_steps: 20
    last_update_step: 500
    work_root: work
  calibration:
    inter_ms_per_byte: 1.0e-8
    intra_ms_per_byte: 2.0e-9
    route_ms_per_assignment: 3.0e-5
    communication_multiplier: 3.1
    compute_ms_per_assignment: 4.0e-5
    compute_multiplier: 4.19
  resources:
    workers: 8
    candidate_workers: 2
    worker_threads: 1
    planner_cpu_ids: 8-15
    training_cpu_ids: 0-7
""",
        encoding="utf-8",
    )

    config = PlaceMoERuntimeConfig.from_file(config_path)

    assert config.initial_artifact == str(artifact)
    assert config.runtime_perf_model == str(tmp_path / "perf.json")
    assert config.hot_update.layout_interval_steps == 100
    assert config.hot_update.mapping_interval_steps == 20
    assert config.hot_update.work_root == str(tmp_path / "work")
    assert config.hot_update.failure_policy == "continue"
    assert config.resources.planner_cpu_ids == "8-15"
    assert config.calibration.compute_ms_per_assignment == pytest.approx(4.0e-5)


def test_inline_static_config_ignores_stale_legacy_environment(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("placemoe:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("VEOMNI_PLACEMOE_CONFIG", str(legacy))

    config = _resolve_placemoe_runtime_config(PlaceMoEArguments(enabled=False))

    assert not config.enabled


def test_legacy_environment_requires_explicit_opt_in(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("placemoe:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("VEOMNI_PLACEMOE_CONFIG", str(legacy))
    monkeypatch.setenv("VEOMNI_PLACEMOE_USE_LEGACY_CONFIG", "1")

    config = _resolve_placemoe_runtime_config(PlaceMoEArguments())

    assert config.enabled


def test_model_sharding_prefers_canonical_initial_artifact(tmp_path, monkeypatch) -> None:
    canonical_artifact = tmp_path / "canonical.json"
    canonical_artifact.write_text("{}", encoding="utf-8")
    hiermoe_artifact = tmp_path / "hiermoe.json"
    hiermoe_artifact.write_text("{}", encoding="utf-8")
    config = PlaceMoERuntimeConfig.from_training_config(
        {
            "enabled": True,
            "initial_artifact": str(canonical_artifact),
        }
    )
    set_current_runtime_config(config)
    monkeypatch.setenv("VEOMNI_PLACEMOE_CONFIG", str(tmp_path / "stale.yaml"))
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(hiermoe_artifact))
    _hiermoe_initial_layout_path.cache_clear()

    try:
        assert _hiermoe_initial_layout_path() == str(canonical_artifact)
    finally:
        set_current_runtime_config(PlaceMoERuntimeConfig())
        _hiermoe_initial_layout_path.cache_clear()


def test_model_sharding_ignores_stale_legacy_artifact_for_inline_static_config(tmp_path, monkeypatch) -> None:
    legacy_artifact = tmp_path / "legacy.json"
    legacy_artifact.write_text("{}", encoding="utf-8")
    set_current_runtime_config(PlaceMoERuntimeConfig.from_training_config({"enabled": True}))
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(legacy_artifact))
    _hiermoe_initial_layout_path.cache_clear()

    try:
        assert _hiermoe_initial_layout_path() == ""
    finally:
        set_current_runtime_config(PlaceMoERuntimeConfig())
        _hiermoe_initial_layout_path.cache_clear()


def test_model_sharding_legacy_artifact_requires_explicit_opt_in(tmp_path, monkeypatch) -> None:
    legacy_artifact = tmp_path / "legacy.json"
    legacy_artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(legacy_artifact))
    monkeypatch.setenv("VEOMNI_PLACEMOE_USE_LEGACY_CONFIG", "1")
    set_current_runtime_config(_resolve_placemoe_runtime_config(PlaceMoEArguments()))
    _hiermoe_initial_layout_path.cache_clear()

    try:
        assert _hiermoe_initial_layout_path() == str(legacy_artifact)
    finally:
        set_current_runtime_config(PlaceMoERuntimeConfig())
        _hiermoe_initial_layout_path.cache_clear()


def test_runtime_config_loads_accepted_calibration_artifact(tmp_path) -> None:
    initial = tmp_path / "initial.json"
    initial.write_text("{}", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "accepted",
                "coefficients": {
                    "inter_ms_per_byte": 1.0e-8,
                    "intra_ms_per_byte": 2.0e-9,
                    "route_ms_per_assignment": 3.0e-5,
                    "communication_multiplier": 3.1,
                    "compute_ms_per_assignment": 4.0e-5,
                    "compute_multiplier": 4.19,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps(
            {
                "initial_artifact": "initial.json",
                "hot_update": {"enabled": True, "layout_interval_steps": 100},
                "calibration": {"artifact": "calibration.json"},
            }
        ),
        encoding="utf-8",
    )

    config = PlaceMoERuntimeConfig.from_file(config_path)

    assert config.calibration.artifact == str(calibration)
    assert config.calibration.compute_ms_per_assignment == pytest.approx(4.0e-5)


def test_runtime_config_rejects_wrong_calibration_artifact_type(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "artifact_type": "hiermoe_runtime_performance_model",
                "status": "accepted",
                "coefficients": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="expected 'placemoe_planner_calibration'"):
        PlaceMoERuntimeConfig.from_training_config(
            {
                "base_directory": str(tmp_path),
                "calibration": {"artifact": "calibration.json"},
            }
        )


def test_inline_dataclass_uses_calibration_artifact_coefficients(tmp_path) -> None:
    _write_scoped_calibration(tmp_path, {"model_id": "deepseek"})
    training = PlaceMoEArguments(enabled=True, base_directory=str(tmp_path))
    training.calibration.artifact = "calibration.json"

    config = PlaceMoERuntimeConfig.from_training_config(training)

    assert config.calibration.inter_ms_per_byte == pytest.approx(1.0e-8)
    assert config.calibration.compute_ms_per_assignment == pytest.approx(4.0e-5)


def test_runtime_config_rejects_negative_intervals(tmp_path) -> None:
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text(
        """
placemoe:
  hot_update:
    layout_interval_steps: -1
""",
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="must be non-negative"):
        PlaceMoERuntimeConfig.from_file(config_path)


def test_runtime_config_is_disabled_without_canonical_config() -> None:
    config = PlaceMoERuntimeConfig.from_environment({})

    assert not config.source_path
    assert not config.initial_artifact
    assert not config.hot_update.enabled
    assert config.hot_update.layout_interval_steps == 0


def test_training_config_path_rejects_mixed_inline_fields(tmp_path) -> None:
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text("placemoe:\n  enabled: true\n", encoding="utf-8")
    training = PlaceMoEArguments(config_path=str(config_path), enabled=True)

    with pytest.raises(PlaceMoEConfigurationError, match="exclusive input"):
        PlaceMoERuntimeConfig.from_training_config(training)


def test_training_config_path_accepts_default_inline_fields(tmp_path) -> None:
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text("placemoe:\n  enabled: true\n", encoding="utf-8")
    training = PlaceMoEArguments(config_path=str(config_path))

    config = PlaceMoERuntimeConfig.from_training_config(training)

    assert config.enabled
    assert config.source_path == str(config_path)


def test_runtime_config_loads_inline_training_config_without_initial_artifact(tmp_path) -> None:
    training_config = PlaceMoEArguments(enabled=True, base_directory=str(tmp_path))
    training_config.hot_update.enabled = True
    training_config.hot_update.layout_interval_steps = 100
    training_config.hot_update.mapping_interval_steps = 20
    training_config.hot_update.work_root = "runtime"
    training_config.resources.workers = 8

    config = PlaceMoERuntimeConfig.from_training_config(training_config)

    assert config.source_path == "train.hiermoe.placemoe"
    assert config.enabled
    assert config.initial_artifact == ""
    assert config.hot_update.enabled
    assert config.hot_update.work_root == str(tmp_path / "runtime")
    assert config.resources.workers == 8


def test_placemoe_preset_selects_canonical_hiermoe_runtime() -> None:
    config = HierMoEConfig(
        placemoe=PlaceMoEArguments(enabled=True),
        redundant_slot_increment_per_device=1,
    )

    assert config.enable
    assert config.token_dedup
    assert config.communication_mode == "hierarchical"
    assert config.expert_swap
    assert config.expert_swap_max_pairs_per_layer == 0
    assert config.expert_swap_selector == "hiermoe_greedy_cover_p1"
    assert config.expert_swap_mode == "step"
    assert config.fixed_pipeline_overlap
    assert config.max_slot_op_search_rounds == 0


def test_scheduler_coalesces_equal_intervals_into_full_update() -> None:
    scheduler = HotUpdateScheduler(100, 100, 500)

    scheduler.observe_step(100)

    assert scheduler.pop_next() is UpdateKind.FULL
    assert scheduler.pop_next() is None


def test_scheduler_preserves_full_update_observed_while_job_runs() -> None:
    scheduler = HotUpdateScheduler(100, 20, 500)

    scheduler.observe_step(80)
    assert scheduler.pop_next() is UpdateKind.MAPPING_ONLY
    scheduler.observe_step(100)

    assert scheduler.pop_next() is UpdateKind.FULL


def test_controller_preserves_pending_update_while_a_job_runs() -> None:
    controller = HotUpdateController(100, 20, 500)
    controller.active_job = object()  # type: ignore[assignment]
    controller.observe_step(100)

    assert controller.next_update() is None
    controller.finish()
    assert controller.next_update() is UpdateKind.FULL


def test_planner_process_starts_in_a_new_session(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    process = launch_planner_process(["python", "planner.py"], stdout=subprocess.DEVNULL, environment={})

    assert process is sentinel
    assert captured["command"] == [
        sys.executable,
        str(Path(planner_process.__file__).with_name("planner_supervisor.py")),
        "--",
        "python",
        "planner.py",
    ]
    assert captured["start_new_session"] is True
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["env"]["PLACEMOE_SUPERVISOR_PARENT_PID"] == str(os.getpid())


def test_planner_termination_targets_the_process_group(monkeypatch) -> None:
    signals = []

    class FakeProcess:
        pid = 123
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("planner", timeout)
            return -signal.SIGKILL

        def terminate(self):
            raise AssertionError("process-group termination should be used")

        def kill(self):
            raise AssertionError("process-group kill should be used")

    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)) if sig else None)

    terminate_planner_process(FakeProcess(), timeout=0.01)

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_planner_termination_cleans_workers_after_leader_exit(monkeypatch) -> None:
    signals = []

    class ExitedLeader:
        pid = 456

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)) if sig else None)

    terminate_planner_process(ExitedLeader(), timeout=0.0)

    assert signals == [(456, signal.SIGTERM), (456, signal.SIGKILL)]


def _write_scoped_calibration(tmp_path, scope: dict, *, artifact_type: str | None = None) -> None:
    payload = {
        "status": "accepted",
        "scope": scope,
        "coefficients": {
            "inter_ms_per_byte": 1.0e-8,
            "intra_ms_per_byte": 2.0e-9,
            "route_ms_per_assignment": 3.0e-5,
            "communication_multiplier": 3.1,
            "compute_ms_per_assignment": 4.0e-5,
            "compute_multiplier": 4.19,
        },
    }
    if artifact_type is not None:
        payload["artifact_type"] = artifact_type
    (tmp_path / "calibration.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_runtime_config_accepts_exact_calibration_scope(tmp_path) -> None:
    expected_scope = {
        "device_type": "cuda",
        "accelerator_model": "NVIDIA RTX A6000",
        "communication_backend": "nccl",
        "world_size": 32,
        "ranks_per_node": 8,
        "model_id": "Qwen3-VL-30B-A3B-Instruct",
        "dataset_id": "sharegpt4v",
        "moe_implementation": "fused_triton",
    }
    _write_scoped_calibration(tmp_path, expected_scope)
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps(
            {
                "calibration": {
                    "artifact": "calibration.json",
                    "require_scope": True,
                    "expected_scope": expected_scope,
                }
            }
        ),
        encoding="utf-8",
    )

    config = PlaceMoERuntimeConfig.from_file(config_path)

    assert config.calibration.require_scope
    assert config.calibration.artifact_scope == expected_scope


def test_runtime_config_rejects_mismatched_calibration_scope(tmp_path) -> None:
    _write_scoped_calibration(
        tmp_path,
        {
            "device_type": "npu",
            "accelerator_model": "Ascend 910B",
            "world_size": 32,
        },
    )
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps(
            {
                "calibration": {
                    "artifact": "calibration.json",
                    "require_scope": True,
                    "expected_scope": {
                        "device_type": "cuda",
                        "accelerator_model": "NVIDIA RTX A6000",
                        "world_size": 32,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="does not match expected scope"):
        PlaceMoERuntimeConfig.from_file(config_path)


def test_runtime_config_automatically_validates_artifact_scope(tmp_path) -> None:
    _write_scoped_calibration(
        tmp_path,
        {
            "model_id": "qwen",
            "ep_size": 16,
            "ranks_per_node": 8,
            "hierarchy_group_sizes": [8, 16],
        },
        artifact_type="placemoe_planner_calibration",
    )
    config = PlaceMoERuntimeConfig.from_training_config(
        {
            "base_directory": str(tmp_path),
            "calibration": {"artifact": "calibration.json"},
        }
    )

    with pytest.raises(PlaceMoEConfigurationError, match="mismatched values"):
        config.calibration.validate_artifact_scope(
            {
                "model_id": "deepseek",
                "ep_size": 16,
                "ranks_per_node": 8,
                "hierarchy_group_sizes": [8, 16],
            }
        )


def test_runtime_config_rejects_required_scope_without_artifact(tmp_path) -> None:
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps(
            {
                "calibration": {
                    "require_scope": True,
                    "expected_scope": {"device_type": "cuda"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="artifact is required"):
        PlaceMoERuntimeConfig.from_file(config_path)


def test_runtime_config_rejects_non_mapping_calibration_artifact(tmp_path) -> None:
    (tmp_path / "calibration.json").write_text("[]", encoding="utf-8")
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps({"calibration": {"artifact": "calibration.json"}}),
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="must contain a mapping"):
        PlaceMoERuntimeConfig.from_file(config_path)


def test_runtime_config_ignores_legacy_string_scope_unless_strict(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "accepted",
                "scope": "cluster_topology",
                "coefficients": {"inter_ms_per_byte": 1.0e-8},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "placemoe.json"
    config_path.write_text(
        json.dumps({"calibration": {"artifact": "calibration.json"}}),
        encoding="utf-8",
    )

    config = PlaceMoERuntimeConfig.from_file(config_path)

    assert config.calibration.artifact_scope == {}

    config_path.write_text(
        json.dumps(
            {
                "calibration": {
                    "artifact": "calibration.json",
                    "require_scope": True,
                    "expected_scope": {"device_type": "cuda"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PlaceMoEConfigurationError, match="scope must be a mapping"):
        PlaceMoERuntimeConfig.from_file(config_path)


def test_runtime_config_requires_explicit_cpu_masks_together(tmp_path) -> None:
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text(
        """
placemoe:
  resources:
    planner_cpu_ids: 8-15
""",
        encoding="utf-8",
    )

    with pytest.raises(PlaceMoEConfigurationError, match="must be configured together"):
        PlaceMoERuntimeConfig.from_file(config_path)
