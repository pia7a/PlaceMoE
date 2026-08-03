# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json

import pytest

from veomni.distributed.moe.hiermoe.placemoe.runtime import (
    HotUpdateController,
    HotUpdateScheduler,
    PlaceMoERuntimeConfig,
    UpdateKind,
)
from veomni.distributed.moe.hiermoe.placemoe.runtime.config import PlaceMoEConfigurationError
from veomni.distributed.parallel_plan import _hiermoe_static_preload_layout_path


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


def test_model_sharding_prefers_canonical_initial_artifact(tmp_path, monkeypatch) -> None:
    canonical_artifact = tmp_path / "canonical.json"
    canonical_artifact.write_text("{}", encoding="utf-8")
    legacy_artifact = tmp_path / "legacy.json"
    legacy_artifact.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "placemoe.yaml"
    config_path.write_text(
        "placemoe:\n  initial_artifact: canonical.json\n  hot_update:\n    enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VEOMNI_PLACEMOE_CONFIG", str(config_path))
    monkeypatch.setenv("VEOMNI_HIERMOE_STATIC_PRELOAD_LAYOUT_PATH", str(legacy_artifact))
    _hiermoe_static_preload_layout_path.cache_clear()

    assert _hiermoe_static_preload_layout_path() == str(canonical_artifact)

    _hiermoe_static_preload_layout_path.cache_clear()


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


def test_legacy_environment_preserves_historical_defaults() -> None:
    config = PlaceMoERuntimeConfig.from_environment(
        {
            "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN": "1",
            "VEOMNI_HIERMOE_LAYOUT_REFRESH_INTERVAL": "100",
            "VEOMNI_HIERMOE_MAPPING_REFRESH_INTERVAL": "20",
            "VEOMNI_HIERMOE_PLACEMOE_COMPUTE_MS_PER_ASSIGNMENT": "0.25",
        }
    )

    assert config.hot_update.enabled
    assert config.hot_update.layout_interval_steps == 100
    assert config.hot_update.mapping_interval_steps == 20
    assert config.calibration.compute_ms_per_assignment == pytest.approx(0.25)


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
