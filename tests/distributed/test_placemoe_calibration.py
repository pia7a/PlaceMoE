# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from veomni.distributed.moe.hiermoe.placemoe.calibration import (
    CalibrationThresholds,
    ModelCalibrationError,
    ModelCalibrationSchedule,
    build_planner_calibration_artifact,
    load_local_phase_timing_summary,
    materialize_model_calibration_config,
    parse_cost_model_reports,
)


def _report(step: int, *, communication_scale: float = 1.0) -> dict:
    stage1_bytes = [10.0, 20.0, 30.0, 45.0]
    stage2_bytes = [4.0, 13.0, 7.0, 20.0]
    peak_assignments = [2.0, 9.0, 4.0, 13.0]
    communication = [
        communication_scale * (0.5 * (2.0 * inter + intra) + 0.1 * assignments + 1.0)
        for inter, intra, assignments in zip(stage1_bytes, stage2_bytes, peak_assignments, strict=True)
    ]
    paired_assignments = [5.0, 17.0, 8.0, 21.0]
    paired_compute = [0.2 * value + 0.5 for value in paired_assignments]
    peak_compute = [0.2 * value + 0.5 for value in peak_assignments]
    return {
        "step": step,
        "offline_scorer_samples": {
            "stage1_payload_endpoint_bytes": stage1_bytes,
            "stage2_payload_endpoint_bytes": stage2_bytes,
            "peak_assignments": peak_assignments,
            "actual_stage1_a2a_ms": [0.6 * value for value in communication],
            "actual_stage2_a2a_ms": [0.4 * value for value in communication],
            "paired_assignments": paired_assignments,
            "paired_compute_ms": paired_compute,
        },
        "sample_data": {
            "network_joint": {
                "measured_ms": [
                    network + compute for network, compute in zip(communication, peak_compute, strict=True)
                ]
            }
        },
    }


def _training_log(*reports: dict) -> str:
    lines = []
    for index, report in enumerate(reports):
        phase = "calibration" if index == 0 else "validation"
        lines.append(f"INFO HierMoE cost model {phase} report: {json.dumps(report)}")
    return "\n".join(lines)


def _write_timing(directory: Path, ranks: tuple[int, ...]) -> None:
    directory.mkdir()
    for rank in ranks:
        rows = []
        rank_scale = 1.0 if rank == 0 else 0.8
        for step in (3, 4, 5):
            rows.append(
                {
                    "step": step,
                    "rank": rank,
                    "span_layers": [
                        {
                            "layer": "model.layers.0.mlp",
                            "direction": "forward",
                            "component": "all_to_all",
                            "cuda_ms_sum": 10.0 * rank_scale,
                        },
                        {
                            "layer": "model.layers.0.mlp",
                            "direction": "backward",
                            "component": "all_to_all",
                            "cuda_ms_sum": 20.0 * rank_scale,
                        },
                        {
                            "layer": "model.layers.0.mlp",
                            "direction": "forward",
                            "component": "expert_compute",
                            "cuda_ms_sum": 20.0 * rank_scale,
                        },
                        {
                            "layer": "model.layers.0.mlp",
                            "direction": "backward",
                            "component": "expert_compute",
                            "cuda_ms_sum": 20.0 * rank_scale,
                        },
                    ],
                }
            )
        (directory / f"moe_timing_rank{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )


def _config() -> dict:
    return {
        "model": {"model_path": "/models/demo-moe"},
        "train": {
            "accelerator": {"ep_size": 4},
            "optimizer": {"lr": 1.0e-5},
            "hiermoe": {
                "hierarchy_group_sizes": [2, 4],
                "redundant_slot_increment_per_device": 1,
            },
        },
    }


def _runtime_model() -> dict:
    return {
        "schema_version": 2,
        "source": "bench_hiermoe_perf_model",
        "metadata": {
            "ep_size": 4,
            "ranks_per_node": 2,
            "hierarchy_group_sizes": [2, 4],
        },
        "inter": [{"alpha": 0.0, "beta": 2.0}],
        "intra": {"alpha": 0.0, "beta": 1.0},
    }


def _phase_summaries(tmp_path: Path) -> list[dict]:
    summaries = []
    for node, ranks in enumerate(((0, 1), (2, 3))):
        timing_directory = tmp_path / f"timing-{node}"
        _write_timing(timing_directory, ranks)
        summaries.append(
            load_local_phase_timing_summary(
                timing_directory,
                expected_ranks=ranks,
                expected_steps=(3, 4, 5),
            )
        )
    return summaries


def test_default_model_calibration_schedule_uses_five_steps() -> None:
    schedule = ModelCalibrationSchedule()

    assert schedule.warmup_steps == 2
    assert schedule.calibration_step == 2
    assert schedule.validation_steps == 2
    assert schedule.max_steps == 5


def test_materialized_config_isolated_short_default_layout_run(tmp_path) -> None:
    source = _config()
    source["train"]["num_train_epochs"] = 9
    source["train"]["checkpoint"] = {"load_path": "/checkpoint/resume"}
    source["train"]["hiermoe"]["placemoe"] = {
        "enabled": True,
        "hot_update": {"enabled": True, "layout_interval_steps": 100},
    }

    result = materialize_model_calibration_config(
        source,
        runtime_perf_model=tmp_path / "runtime.json",
        work_directory=tmp_path,
        schedule=ModelCalibrationSchedule(),
    )

    assert result["train"]["max_steps"] == 5
    assert result["train"]["num_train_epochs"] == 1
    assert result["train"]["checkpoint"]["load_path"] is None
    assert result["train"]["optimizer"]["lr"] == 0.0
    assert result["train"]["hiermoe"]["enable"] is True
    assert result["train"]["hiermoe"]["token_dedup"] is True
    assert result["train"]["hiermoe"]["expert_swap"] is True
    assert result["train"]["hiermoe"]["expert_swap_max_pairs_per_layer"] == 0
    assert result["train"]["hiermoe"]["max_slot_op_search_rounds"] == 0
    assert result["train"]["hiermoe"]["redundant_slot_increment_per_device"] == 1
    assert result["train"]["hiermoe"]["placemoe"]["hot_update"]["enabled"] is False
    assert source["train"]["optimizer"]["lr"] == 1.0e-5


def test_builds_scoped_accepted_artifact_from_fit_and_held_out_steps(tmp_path) -> None:
    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), _report(3), _report(4)),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(),
    )

    assert artifact["status"] == "accepted"
    assert artifact["scope"] == {
        "model_id": "demo-moe",
        "ep_size": 4,
        "ranks_per_node": 2,
        "hierarchy_group_sizes": [2, 4],
    }
    coefficients = artifact["coefficients"]
    assert math.isclose(coefficients["inter_ms_per_byte"], 1.0)
    assert math.isclose(coefficients["intra_ms_per_byte"], 0.5)
    assert math.isclose(coefficients["route_ms_per_assignment"], 0.1)
    assert math.isclose(coefficients["compute_ms_per_assignment"], 0.2)
    assert coefficients["communication_multiplier"] == 3.0
    assert coefficients["compute_multiplier"] == 2.0
    assert artifact["fit"]["total_training_steps"] == 5
    assert artifact["held_out_validation"]["checks"] == {
        "communication": True,
        "compute": True,
        "joint": True,
    }
    assert artifact["held_out_validation"]["basis"].startswith("serialized variable-cost")


def test_rejects_artifact_when_held_out_error_exceeds_threshold(tmp_path) -> None:
    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), _report(3, communication_scale=3.0)),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
        thresholds=CalibrationThresholds(
            compute_mape_percent=5.0,
            communication_mape_percent=10.0,
            joint_mape_percent=10.0,
        ),
    )

    assert artifact["status"] == "rejected"
    assert artifact["held_out_validation"]["checks"]["communication"] is False


def test_report_parser_deduplicates_identical_distributed_log_lines() -> None:
    fit = _report(2)
    validation = _report(3)
    text = "\n".join((_training_log(fit, validation), _training_log(fit, validation)))

    parsed_fit, parsed_validations = parse_cost_model_reports(text)

    assert parsed_fit == fit
    assert parsed_validations == [validation]


def test_local_phase_summary_requires_every_rank_and_step(tmp_path) -> None:
    timing_directory = tmp_path / "timing"
    _write_timing(timing_directory, (0, 1))

    with pytest.raises(ModelCalibrationError, match="incomplete phase timing matrix"):
        load_local_phase_timing_summary(
            timing_directory,
            expected_ranks=(0, 1),
            expected_steps=(3, 4, 5, 6),
        )


def test_runtime_performance_model_requires_topology_scope(tmp_path) -> None:
    runtime_model = _runtime_model()
    runtime_model["metadata"].pop("ep_size")

    with pytest.raises(ModelCalibrationError, match="metadata is missing"):
        build_planner_calibration_artifact(
            training_config=_config(),
            runtime_perf_model=runtime_model,
            runtime_perf_model_sha256="runtime-sha",
            training_log_text=_training_log(_report(2), _report(3), _report(4)),
            training_log_sha256="training-sha",
            phase_timing_summaries=_phase_summaries(tmp_path),
            ranks_per_node=2,
            schedule=ModelCalibrationSchedule(),
        )


def test_runtime_hierarchy_must_match_planner_launch_shape(tmp_path) -> None:
    config = _config()
    config["train"]["hiermoe"]["hierarchy_group_sizes"] = [1, 4]
    runtime_model = _runtime_model()
    runtime_model["metadata"]["hierarchy_group_sizes"] = [1, 4]

    with pytest.raises(ModelCalibrationError, match="planner runtime requires"):
        build_planner_calibration_artifact(
            training_config=config,
            runtime_perf_model=runtime_model,
            runtime_perf_model_sha256="runtime-sha",
            training_log_text=_training_log(_report(2), _report(3), _report(4)),
            training_log_sha256="training-sha",
            phase_timing_summaries=_phase_summaries(tmp_path),
            ranks_per_node=2,
            schedule=ModelCalibrationSchedule(),
        )
