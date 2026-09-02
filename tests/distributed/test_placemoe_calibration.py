# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from veomni.distributed.moe.hiermoe.placemoe.calibration import (
    ModelCalibrationError,
    ModelCalibrationSchedule,
    build_planner_calibration_artifact,
    load_local_phase_timing_summary,
    materialize_model_calibration_config,
    parse_cost_model_reports,
    summarize_phase_timing_rows,
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
            "paired_expert_token_counts": [[2.0, 3.0], [10.0, 7.0], [3.0, 5.0], [20.0, 1.0]],
            "paired_assignments": paired_assignments,
            "paired_compute_ms": paired_compute,
        },
        "sample_alignment": {
            "ep_size": 4,
            "row_count_per_rank": 4,
            "layer_keys": ["model.layers.0.mlp"],
            "row_layer_indices": [0, 0, 0, 0],
            "row_call_indices": [0, 1, 2, 3],
            "source_assignment_totals": [20.0, 36.0, 24.0, 52.0],
            "destination_assignment_totals": [20.0, 36.0, 24.0, 52.0],
            "destination_rank_mismatch_counts": [0, 0, 0, 0],
            "destination_rank_max_abs_deltas": [0.0, 0.0, 0.0, 0.0],
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
            "optimizer": {"lr": 1.0e-5, "lr_min": 1.0e-7},
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
            "message_bytes_requested": [4, 10, 20, 50],
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


def test_in_memory_phase_timing_matches_file_summary(tmp_path) -> None:
    timing_directory = tmp_path / "timing"
    _write_timing(timing_directory, (0, 1))
    rows = [
        json.loads(line)
        for path in sorted(timing_directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    in_memory = summarize_phase_timing_rows(
        rows,
        expected_ranks=(0, 1),
        expected_steps=(3, 4, 5),
    )
    from_files = load_local_phase_timing_summary(
        timing_directory,
        expected_ranks=(0, 1),
        expected_steps=(3, 4, 5),
    )

    assert in_memory == from_files


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
    assert result["train"]["optimizer"]["lr_min"] == 0.0
    assert result["train"]["optimizer"]["lr_warmup_ratio"] == 0.0
    assert result["train"]["optimizer"]["lr_decay_style"] == "constant"
    assert result["train"]["hiermoe"]["enable"] is True
    assert result["train"]["hiermoe"]["token_dedup"] is True
    assert result["train"]["hiermoe"]["expert_swap"] is True
    assert result["train"]["hiermoe"]["expert_swap_max_pairs_per_layer"] == 0
    assert result["train"]["hiermoe"]["max_slot_op_search_rounds"] == 0
    assert result["train"]["hiermoe"]["redundant_slot_increment_per_device"] == 1
    assert result["train"]["hiermoe"]["placemoe"]["hot_update"]["enabled"] is False
    assert source["train"]["optimizer"]["lr"] == 1.0e-5
    assert source["train"]["optimizer"]["lr_min"] == 1.0e-7


def test_materialized_config_preserves_zero_replica_budget(tmp_path) -> None:
    source = _config()
    source["train"]["hiermoe"]["redundant_slot_increment_per_device"] = 0

    result = materialize_model_calibration_config(
        source,
        runtime_perf_model=tmp_path / "runtime.json",
        work_directory=tmp_path,
        schedule=ModelCalibrationSchedule(),
    )

    hiermoe = result["train"]["hiermoe"]
    assert hiermoe["redundant_slot_increment_per_device"] == 0
    assert hiermoe["expert_swap_selector"] == "current_joint"
    assert hiermoe["fixed_pipeline_overlap"] is False


def test_materialized_single_node_config_does_not_enable_two_level_pipeline(tmp_path) -> None:
    source = _config()
    source["train"]["hiermoe"]["hierarchy_group_sizes"] = [4]

    result = materialize_model_calibration_config(
        source,
        runtime_perf_model=tmp_path / "runtime.json",
        work_directory=tmp_path,
        schedule=ModelCalibrationSchedule(),
    )

    assert result["train"]["hiermoe"]["fixed_pipeline_overlap"] is False


def test_builds_scoped_artifact_with_mape_diagnostics(tmp_path) -> None:
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

    assert "status" not in artifact
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
    assert "checks" not in artifact["held_out_validation"]
    assert "thresholds" not in artifact["held_out_validation"]
    assert artifact["held_out_validation"]["communication"]["mape_percent"] == pytest.approx(0.0)
    assert artifact["held_out_validation"]["basis"].startswith("serialized variable-cost")
    context = artifact["held_out_validation"]["diagnostic_context"]
    assert context["gate_effect"] == "none"
    assert context["fit_step"]["step"] == 2
    assert context["fit_step"]["communication"]["feature_matrix"]["full_rank"] is True
    assert context["compute"]["raw_affine"]["mape_percent"] == pytest.approx(0.0)
    assert context["compute"]["serialized_variable"]["zero_truth_sample_count"] == 0
    assert context["compute"]["feature_target_pearson"]["paired_assignments"] == pytest.approx(1.0)
    assert context["sample_alignment"]["expert_count_sums"]["mismatch_sample_count"] == 0
    assert context["sample_alignment"]["route_assignment_conservation"]["mismatch_sample_count"] == 0
    assert context["sample_alignment"]["destination_rank_assignment_alignment"]["mismatch_sample_count"] == 0
    assert context["runtime_message_coverage"]["stage1"]["outside_requested_range_sample_count"] == 0
    assert context["runtime_message_coverage"]["stage2"]["outside_requested_range_sample_count"] == 0
    assert context["signals"]["validation_feature_outside_fit_range_sample_count"] == 0


def test_builds_single_node_artifact_from_intra_node_calibration(tmp_path) -> None:
    config = _config()
    config["train"]["hiermoe"]["hierarchy_group_sizes"] = [4]
    runtime_model = _runtime_model()
    runtime_model["metadata"]["ranks_per_node"] = 4
    runtime_model["metadata"]["hierarchy_group_sizes"] = [4]
    runtime_model["inter"] = []

    artifact = build_planner_calibration_artifact(
        training_config=config,
        runtime_perf_model=runtime_model,
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), _report(3), _report(4)),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=4,
        schedule=ModelCalibrationSchedule(),
    )

    assert "status" not in artifact
    assert artifact["scope"]["hierarchy_group_sizes"] == [4]
    assert artifact["coefficients"]["inter_ms_per_byte"] == artifact["coefficients"]["intra_ms_per_byte"]


def test_reports_held_out_error_without_acceptance_gate(tmp_path) -> None:
    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), _report(3, communication_scale=3.0)),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
    )

    assert "status" not in artifact
    assert "checks" not in artifact["held_out_validation"]
    assert artifact["held_out_validation"]["communication"]["mape_percent"] > 10.0
    context = artifact["held_out_validation"]["diagnostic_context"]
    assert context["signals"]["communication_raw_affine_r_squared_negative"] is True
    assert context["trainer_validation_reports"][0]["step"] == 3


def test_nonfinite_trainer_diagnostics_are_serialized_as_null(tmp_path) -> None:
    fit = _report(2)
    validation = _report(3)
    fit["communication"] = {"r_squared": float("nan")}
    validation["joint"] = {"r_squared": float("inf")}

    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(fit, validation),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
    )

    context = artifact["held_out_validation"]["diagnostic_context"]
    assert context["trainer_calibration_report"]["communication"]["r_squared"] is None
    assert context["trainer_validation_reports"][0]["joint"]["r_squared"] is None
    json.dumps(artifact, allow_nan=False)


def test_reports_alignment_mismatches_as_diagnostics(tmp_path) -> None:
    validation = _report(3)
    validation["offline_scorer_samples"]["paired_expert_token_counts"][0][0] += 1.0
    validation["sample_alignment"]["source_assignment_totals"][0] += 2.0
    validation["sample_alignment"]["destination_rank_mismatch_counts"][0] = 1
    validation["sample_alignment"]["destination_rank_max_abs_deltas"][0] = 2.0

    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), validation),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
    )

    assert "status" not in artifact
    context = artifact["held_out_validation"]["diagnostic_context"]
    assert context["gate_effect"] == "none"
    assert context["sample_alignment"]["expert_count_sums"]["mismatch_sample_count"] == 1
    assert context["sample_alignment"]["route_assignment_conservation"]["mismatch_sample_count"] == 1
    assert context["sample_alignment"]["destination_rank_assignment_alignment"]["mismatch_sample_count"] == 1
    mismatch = context["sample_alignment"]["per_step"][0]["destination_rank_alignment"]["mismatch_examples"][0]
    assert mismatch["layer_key"] == "model.layers.0.mlp"
    assert mismatch["call_index"] == 0
    assert context["signals"]["sample_alignment_mismatch"] is True


def test_diagnostics_remain_backward_compatible_with_older_reports(tmp_path) -> None:
    fit = _report(2)
    validation = _report(3)
    for report in (fit, validation):
        report["offline_scorer_samples"].pop("paired_expert_token_counts")
        report.pop("sample_alignment")

    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(fit, validation),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
    )

    assert "status" not in artifact
    alignment = artifact["held_out_validation"]["diagnostic_context"]["sample_alignment"]
    assert alignment["expert_count_sums"]["available"] is False
    assert alignment["route_assignment_conservation"]["available"] is False
    assert alignment["destination_rank_assignment_alignment"]["available"] is False


def test_diagnostics_identify_intercept_clamping_as_mape_risk(tmp_path) -> None:
    validation = _report(3)
    validation["offline_scorer_samples"]["paired_compute_ms"][0] = 0.25

    artifact = build_planner_calibration_artifact(
        training_config=_config(),
        runtime_perf_model=_runtime_model(),
        runtime_perf_model_sha256="runtime-sha",
        training_log_text=_training_log(_report(2), validation),
        training_log_sha256="training-sha",
        phase_timing_summaries=_phase_summaries(tmp_path),
        ranks_per_node=2,
        schedule=ModelCalibrationSchedule(validation_steps=1),
    )

    context = artifact["held_out_validation"]["diagnostic_context"]
    variable = context["compute"]["serialized_variable"]
    assert variable["zero_truth_sample_count"] == 1
    assert variable["zero_truth_fraction"] == pytest.approx(0.25)
    assert variable["absolute_percentage_error_percent"]["max"] > 1_000_000.0
    assert context["signals"]["compute_zero_truth_sample_count"] == 1


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
