# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Portable model calibration for the PlaceMoE planner."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_CALIBRATION_MARKER = "HierMoE cost model calibration report: "
_VALIDATION_MARKER = "HierMoE cost model validation report: "
_COEFFICIENT_FLOOR = 1.0e-15


class ModelCalibrationError(RuntimeError):
    """Raised when a production planner calibration cannot be constructed."""


@dataclass(frozen=True)
class ModelCalibrationSchedule:
    """Short default-layout run used to fit and validate planner costs."""

    warmup_steps: int = 2
    validation_steps: int = 2

    def __post_init__(self) -> None:
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if self.validation_steps < 1:
            raise ValueError("validation_steps must be positive")

    @property
    def calibration_step(self) -> int:
        return self.warmup_steps

    @property
    def max_steps(self) -> int:
        return self.warmup_steps + 1 + self.validation_steps


@dataclass(frozen=True)
class CalibrationThresholds:
    compute_mape_percent: float = 5.0
    communication_mape_percent: float = 10.0
    joint_mape_percent: float = 10.0

    def __post_init__(self) -> None:
        for name, value in (
            ("compute_mape_percent", self.compute_mape_percent),
            ("communication_mape_percent", self.communication_mape_percent),
            ("joint_mape_percent", self.joint_mape_percent),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def parse_cost_model_reports(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract the fit report and ordered held-out reports from trainer output."""

    calibrations: dict[int, dict[str, Any]] = {}
    validations: dict[int, dict[str, Any]] = {}

    def record(target: dict[int, dict[str, Any]], report: dict[str, Any], phase: str) -> None:
        step = int(report["step"])
        previous = target.get(step)
        if previous is not None and previous != report:
            raise ModelCalibrationError(f"conflicting {phase} reports for step {step}")
        target[step] = report

    for line in text.replace("\r", "\n").splitlines():
        if _CALIBRATION_MARKER in line:
            record(
                calibrations,
                json.loads(line.split(_CALIBRATION_MARKER, 1)[1].strip()),
                "calibration",
            )
        if _VALIDATION_MARKER in line:
            record(
                validations,
                json.loads(line.split(_VALIDATION_MARKER, 1)[1].strip()),
                "validation",
            )
    if not calibrations:
        raise ModelCalibrationError("training log has no cost-model calibration report")
    if not validations:
        raise ModelCalibrationError("training log has no held-out cost-model validation report")
    if len(calibrations) != 1:
        raise ModelCalibrationError(f"training log contains calibration reports for steps {sorted(calibrations)}")
    return next(iter(calibrations.values())), [validations[step] for step in sorted(validations)]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelCalibrationError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ModelCalibrationError(f"{name} must be finite and positive, got {value!r}")
    return result


def _runtime_profile(
    payload: Mapping[str, Any],
    *,
    ep_size: int,
    ranks_per_node: int,
    hierarchy_group_sizes: Sequence[int],
) -> tuple[float, float, Mapping[str, Any]]:
    hierarchy = tuple(int(value) for value in hierarchy_group_sizes)
    if len(hierarchy) != 2:
        raise ModelCalibrationError(
            "portable model calibration currently requires a 2-level hierarchy such as [8, 16]"
        )
    runtime_hierarchy = (int(ranks_per_node), int(ep_size))
    if hierarchy != runtime_hierarchy:
        raise ModelCalibrationError(
            f"planner runtime requires hierarchy_group_sizes={runtime_hierarchy}, got {hierarchy}"
        )
    if int(payload.get("schema_version", 0) or 0) < 2:
        raise ModelCalibrationError("runtime performance model must use schema_version >= 2")
    artifact_type = payload.get("artifact_type")
    source = payload.get("source")
    if artifact_type not in {None, "hiermoe_runtime_performance_model"}:
        raise ModelCalibrationError(f"unexpected runtime performance model artifact_type {artifact_type!r}")
    if artifact_type is None and source != "bench_hiermoe_perf_model":
        raise ModelCalibrationError("runtime performance model has no recognized artifact type or source")
    if payload.get("status", "accepted") != "accepted":
        raise ModelCalibrationError(
            f"runtime performance model has status {payload.get('status')!r}, expected 'accepted'"
        )
    metadata = _mapping(payload.get("metadata"), "runtime performance model metadata")
    required_metadata = {"ep_size", "ranks_per_node", "hierarchy_group_sizes"}
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise ModelCalibrationError(f"runtime performance model metadata is missing {missing_metadata}")
    actual_ep_size = int(metadata["ep_size"])
    actual_ranks_per_node = int(metadata["ranks_per_node"])
    actual_hierarchy = tuple(int(value) for value in metadata["hierarchy_group_sizes"])
    mismatches = []
    if actual_ep_size != ep_size:
        mismatches.append(f"ep_size={actual_ep_size}, expected {ep_size}")
    if actual_ranks_per_node != ranks_per_node:
        mismatches.append(f"ranks_per_node={actual_ranks_per_node}, expected {ranks_per_node}")
    if actual_hierarchy != hierarchy:
        mismatches.append(f"hierarchy={actual_hierarchy}, expected {hierarchy}")
    if mismatches:
        raise ModelCalibrationError("runtime performance model scope mismatch: " + "; ".join(mismatches))
    inter_rows = payload.get("inter")
    if not isinstance(inter_rows, list) or not inter_rows:
        raise ModelCalibrationError("runtime performance model has no inter-node coefficients")
    inter_beta = _positive(_mapping(inter_rows[0], "inter[0]").get("beta"), "inter[0].beta")
    intra_beta = _positive(_mapping(payload.get("intra"), "intra").get("beta"), "intra.beta")
    return inter_beta, intra_beta, metadata


def validate_runtime_performance_model(
    payload: Mapping[str, Any],
    *,
    ep_size: int,
    ranks_per_node: int,
    hierarchy_group_sizes: Sequence[int],
) -> None:
    """Validate that topology calibration matches the planner runtime scope."""

    _runtime_profile(
        payload,
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
        hierarchy_group_sizes=hierarchy_group_sizes,
    )


def _fit_nonnegative(features: Sequence[Sequence[float]], targets: Sequence[float]) -> tuple[list[float], float]:
    """Fit non-negative feature weights and intercept by active-set enumeration."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ModelCalibrationError("calibration regression requires at least 2 paired samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ModelCalibrationError("calibration regression contains non-finite samples")
    columns = x.shape[1] + 1
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    scales = np.maximum(np.max(np.abs(design), axis=0), 1.0e-30)
    normalized = design / scales
    best: tuple[float, np.ndarray] | None = None
    for count in range(1, columns + 1):
        for active in itertools.combinations(range(columns), count):
            fitted, *_ = np.linalg.lstsq(normalized[:, active], y, rcond=None)
            coefficients = np.zeros(columns, dtype=np.float64)
            coefficients[list(active)] = fitted / scales[list(active)]
            if np.any(coefficients < -1.0e-12):
                continue
            coefficients = np.maximum(coefficients, 0.0)
            residual = design @ coefficients - y
            squared_error = float(residual @ residual)
            if best is None or squared_error < best[0]:
                best = squared_error, coefficients
    if best is None:
        raise ModelCalibrationError("non-negative calibration regression has no feasible solution")
    coefficients = best[1]
    return coefficients[:-1].tolist(), float(coefficients[-1])


def _diagnostics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(actual, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != estimate.shape:
        raise ModelCalibrationError("diagnostics require non-empty paired samples")
    residual = estimate - truth
    denominator = np.maximum(np.abs(truth), 1.0e-6)
    centered = truth - truth.mean()
    total_variance = float(centered @ centered)
    squared_error = float(residual @ residual)
    return {
        "sample_count": int(truth.size),
        "r_squared": 1.0 - squared_error / total_variance if total_variance > 0.0 else float(squared_error == 0.0),
        "mape_percent": float(np.mean(np.abs(residual) / denominator)) * 100.0,
        "rmse_ms": float(np.sqrt(np.mean(np.square(residual)))),
        "max_abs_error_ms": float(np.max(np.abs(residual))),
    }


def _offline_samples(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(report.get("offline_scorer_samples"), "offline_scorer_samples")


def _forward_communication_rows(
    report: Mapping[str, Any], inter_beta: float, intra_beta: float
) -> tuple[list[list[float]], list[float]]:
    samples = _offline_samples(report)
    stage1_bytes = list(samples.get("stage1_payload_endpoint_bytes", ()))
    stage2_bytes = list(samples.get("stage2_payload_endpoint_bytes", ()))
    assignments = list(samples.get("peak_assignments", ()))
    stage1_ms = list(samples.get("actual_stage1_a2a_ms", ()))
    stage2_ms = list(samples.get("actual_stage2_a2a_ms", ()))
    lengths = {len(stage1_bytes), len(stage2_bytes), len(assignments), len(stage1_ms), len(stage2_ms)}
    if len(lengths) != 1 or not stage1_bytes:
        raise ModelCalibrationError("incomplete 2-level communication samples in calibration report")
    features = [
        [inter_beta * float(inter_bytes) + intra_beta * float(intra_bytes), float(assignment)]
        for inter_bytes, intra_bytes, assignment in zip(stage1_bytes, stage2_bytes, assignments, strict=True)
    ]
    actual = [float(inter_ms) + float(intra_ms) for inter_ms, intra_ms in zip(stage1_ms, stage2_ms, strict=True)]
    return features, actual


def _forward_compute_rows(report: Mapping[str, Any]) -> tuple[list[list[float]], list[float]]:
    samples = _offline_samples(report)
    assignments = list(samples.get("paired_assignments", ()))
    measured = list(samples.get("paired_compute_ms", ()))
    if not assignments or len(assignments) != len(measured):
        raise ModelCalibrationError("incomplete expert-compute samples in calibration report")
    return [[float(value)] for value in assignments], [float(value) for value in measured]


def load_local_phase_timing_summary(
    timing_directory: Path,
    *,
    expected_ranks: Sequence[int],
    expected_steps: Sequence[int],
) -> dict[str, Any]:
    """Load and validate one node's forward/backward timing matrix.

    Each node produces one summary. The CLI exchanges these summaries after
    torchrun exits so rank 0 can recover the critical path across all ranks.
    """

    expected_rank_set = {int(value) for value in expected_ranks}
    expected_step_set = {int(value) for value in expected_steps}
    if not expected_rank_set or not expected_step_set:
        raise ModelCalibrationError("phase timing requires non-empty expected ranks and steps")
    paths = sorted(timing_directory.glob("moe_timing_rank*.jsonl"))
    if len(paths) != len(expected_rank_set):
        raise ModelCalibrationError(
            f"expected {len(expected_rank_set)} local timing files, "
            f"found {len(paths)} in {timing_directory}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    observed_ranks = {int(row["rank"]) for row in rows}
    if observed_ranks != expected_rank_set:
        raise ModelCalibrationError(
            f"timing ranks are {sorted(observed_ranks)}, expected {sorted(expected_rank_set)}"
        )
    by_key: dict[tuple[int, Any, str, str, int], float] = {}
    for payload in rows:
        step = int(payload["step"])
        if step not in expected_step_set:
            continue
        rank = int(payload["rank"])
        for span in payload.get("span_layers", ()):
            component = str(span.get("component"))
            direction = str(span.get("direction"))
            if component not in {"all_to_all", "expert_compute"} or direction not in {"forward", "backward"}:
                continue
            key = (step, span.get("layer"), direction, component, rank)
            by_key[key] = by_key.get(key, 0.0) + float(span["cuda_ms_sum"])

    required_components = {
        ("forward", "all_to_all"),
        ("backward", "all_to_all"),
        ("forward", "expert_compute"),
        ("backward", "expert_compute"),
    }
    required_pairs = {(rank, step) for rank in expected_rank_set for step in expected_step_set}
    layers_by_pair: dict[tuple[int, int], set[str]] = {pair: set() for pair in required_pairs}
    components_by_cell: dict[tuple[int, int, str], set[tuple[str, str]]] = {}
    for step, layer, direction, component, rank in by_key:
        pair = (rank, step)
        if pair not in layers_by_pair:
            continue
        layer_name = str(layer)
        layers_by_pair[pair].add(layer_name)
        components_by_cell.setdefault((rank, step, layer_name), set()).add((direction, component))
    empty_pairs = [f"rank={rank},step={step}" for (rank, step), layers in layers_by_pair.items() if not layers]
    if empty_pairs:
        raise ModelCalibrationError(f"incomplete phase timing matrix: no measured MoE layers for {empty_pairs}")
    expected_layers = next(iter(layers_by_pair.values()))
    layer_mismatches = {
        f"rank={rank},step={step}": sorted(layers ^ expected_layers)
        for (rank, step), layers in layers_by_pair.items()
        if layers != expected_layers
    }
    if layer_mismatches:
        raise ModelCalibrationError(f"phase timing layer sets differ: {layer_mismatches}")
    incomplete = {
        f"rank={rank},step={step},layer={layer}": sorted(required_components - components)
        for (rank, step, layer), components in components_by_cell.items()
        if components != required_components
    }
    if incomplete:
        raise ModelCalibrationError(f"incomplete phase timing matrix: {incomplete}")

    critical_rows = []
    step_layers = sorted({(step, str(layer)) for step, layer, _direction, _component, _rank in by_key})
    for step, layer in step_layers:
        for direction, component in sorted(required_components):
            values = [
                value
                for (
                    candidate_step,
                    candidate_layer,
                    candidate_direction,
                    candidate_component,
                    _rank,
                ), value in by_key.items()
                if candidate_step == step
                and str(candidate_layer) == layer
                and candidate_direction == direction
                and candidate_component == component
            ]
            if values:
                critical_rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "direction": direction,
                        "component": component,
                        "milliseconds": max(values),
                    }
                )
    return {
        "expected_ranks": sorted(expected_rank_set),
        "expected_steps": sorted(expected_step_set),
        "timing_file_count": len(paths),
        "critical_rows": critical_rows,
    }


def merge_phase_timing_summaries(summaries: Sequence[Mapping[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    """Combine node-local summaries into topology-wide critical-path ratios."""

    if not summaries:
        raise ModelCalibrationError("phase timing has no node summaries")
    expected_steps: tuple[int, ...] | None = None
    all_ranks: set[int] = set()
    by_key: dict[tuple[int, str, str, str], float] = {}
    timing_file_count = 0
    for index, raw_summary in enumerate(summaries):
        summary = _mapping(raw_summary, f"phase timing summary {index}")
        summary_steps = tuple(int(value) for value in summary.get("expected_steps", ()))
        if expected_steps is None:
            expected_steps = summary_steps
        elif summary_steps != expected_steps:
            raise ModelCalibrationError("phase timing summaries disagree on measured steps")
        ranks = {int(value) for value in summary.get("expected_ranks", ())}
        overlap = all_ranks & ranks
        if overlap:
            raise ModelCalibrationError(f"phase timing summaries repeat ranks {sorted(overlap)}")
        all_ranks.update(ranks)
        timing_file_count += int(summary.get("timing_file_count", 0))
        for raw_row in summary.get("critical_rows", ()):
            row = _mapping(raw_row, "phase timing critical row")
            key = (int(row["step"]), str(row["layer"]), str(row["direction"]), str(row["component"]))
            value = _positive(row["milliseconds"], "phase timing milliseconds")
            by_key[key] = max(by_key.get(key, 0.0), value)

    totals = {
        "forward_all_to_all": 0.0,
        "backward_all_to_all": 0.0,
        "forward_compute": 0.0,
        "backward_compute": 0.0,
    }
    for (_step, _layer, direction, component), value in by_key.items():
        suffix = "all_to_all" if component == "all_to_all" else "compute"
        totals[f"{direction}_{suffix}"] += value
    forward_communication = _positive(totals["forward_all_to_all"], "forward A2A timing")
    backward_communication = _positive(totals["backward_all_to_all"], "backward A2A timing")
    forward_compute = _positive(totals["forward_compute"], "forward expert-compute timing")
    backward_compute = _positive(totals["backward_compute"], "backward expert-compute timing")
    communication_multiplier = (forward_communication + backward_communication) / forward_communication
    compute_multiplier = (forward_compute + backward_compute) / forward_compute
    return communication_multiplier, compute_multiplier, {
        "selected_steps": list(expected_steps or ()),
        "timing_file_count": timing_file_count,
        "rank_count": len(all_ranks),
        **totals,
    }


def _model_identifier(root: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    model = _mapping(root.get("model"), "model")
    raw = str(model.get("model_path") or model.get("config_path") or "").rstrip("/")
    if not raw:
        raise ModelCalibrationError("model.model_path is required to derive calibration scope")
    return Path(raw).name


def build_planner_calibration_artifact(
    *,
    training_config: Mapping[str, Any],
    runtime_perf_model: Mapping[str, Any],
    runtime_perf_model_sha256: str,
    training_log_text: str,
    training_log_sha256: str,
    phase_timing_summaries: Sequence[Mapping[str, Any]],
    ranks_per_node: int,
    schedule: ModelCalibrationSchedule,
    thresholds: CalibrationThresholds | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Fit, validate, and serialize a complete PlaceMoE planner artifact."""

    thresholds = thresholds or CalibrationThresholds()
    root = _mapping(training_config, "training config")
    train = _mapping(root.get("train"), "train")
    accelerator = _mapping(train.get("accelerator"), "train.accelerator")
    hiermoe = _mapping(train.get("hiermoe"), "train.hiermoe")
    ep_size = int(accelerator.get("ep_size", 0) or 0)
    hierarchy = tuple(int(value) for value in hiermoe.get("hierarchy_group_sizes", ()) or ())
    if ep_size <= 1:
        raise ModelCalibrationError("train.accelerator.ep_size must be greater than 1")
    if not hierarchy:
        raise ModelCalibrationError("train.hiermoe.hierarchy_group_sizes must be explicit during calibration")
    inter_beta, intra_beta, runtime_metadata = _runtime_profile(
        runtime_perf_model,
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
        hierarchy_group_sizes=hierarchy,
    )
    calibration, validations = parse_cost_model_reports(training_log_text)
    if int(calibration.get("step", -1)) != schedule.calibration_step:
        raise ModelCalibrationError(
            f"calibration report step is {calibration.get('step')}, expected {schedule.calibration_step}"
        )
    if len(validations) != schedule.validation_steps:
        raise ModelCalibrationError(
            f"expected {schedule.validation_steps} held-out reports, found {len(validations)}"
        )
    expected_validation_steps = list(range(schedule.calibration_step + 1, schedule.max_steps))
    actual_validation_steps = [int(report.get("step", -1)) for report in validations]
    if actual_validation_steps != expected_validation_steps:
        raise ModelCalibrationError(
            f"held-out report steps are {actual_validation_steps}, expected {expected_validation_steps}"
        )

    communication_features, communication_actual = _forward_communication_rows(calibration, inter_beta, intra_beta)
    (network_scale, route_forward), communication_intercept = _fit_nonnegative(
        communication_features, communication_actual
    )
    compute_features, compute_actual = _forward_compute_rows(calibration)
    (compute_forward,), compute_intercept = _fit_nonnegative(compute_features, compute_actual)
    communication_phase, compute_phase, phase_diagnostics = merge_phase_timing_summaries(phase_timing_summaries)
    expected_rank_count = ep_size
    if int(phase_diagnostics["rank_count"]) != expected_rank_count:
        raise ModelCalibrationError(
            f"phase timing covers {phase_diagnostics['rank_count']} ranks, expected {expected_rank_count}"
        )

    communication_truth: list[float] = []
    communication_prediction: list[float] = []
    compute_truth: list[float] = []
    compute_prediction: list[float] = []
    joint_truth: list[float] = []
    joint_prediction: list[float] = []
    for report in validations:
        features, actual = _forward_communication_rows(report, inter_beta, intra_beta)
        predicted_communication = [
            network_scale * network + route_forward * assignments
            for network, assignments in features
        ]
        variable_communication = [max(0.0, value - communication_intercept) for value in actual]
        communication_truth.extend(variable_communication)
        communication_prediction.extend(predicted_communication)

        validation_compute_features, validation_compute_actual = _forward_compute_rows(report)
        predicted_compute = [compute_forward * row[0] for row in validation_compute_features]
        variable_compute_samples = [max(0.0, value - compute_intercept) for value in validation_compute_actual]
        compute_truth.extend(variable_compute_samples)
        compute_prediction.extend(predicted_compute)

        sample_data = _mapping(report.get("sample_data"), "validation sample_data")
        network_joint = _mapping(sample_data.get("network_joint"), "validation sample_data.network_joint")
        measured_joint = [float(value) for value in network_joint.get("measured_ms", ())]
        if len(measured_joint) != len(actual):
            raise ModelCalibrationError("network-joint and communication validation samples differ in length")
        peak_assignments = [float(row[1]) for row in features]
        actual_compute = [joint - network for joint, network in zip(measured_joint, actual, strict=True)]
        variable_joint_compute = [max(0.0, value - compute_intercept) for value in actual_compute]
        predicted_peak_compute = [compute_forward * value for value in peak_assignments]
        joint_truth.extend(
            communication_phase * network + compute_phase * compute
            for network, compute in zip(variable_communication, variable_joint_compute, strict=True)
        )
        joint_prediction.extend(
            communication_phase * network + compute_phase * compute
            for network, compute in zip(predicted_communication, predicted_peak_compute, strict=True)
        )

    diagnostics = {
        "communication": _diagnostics(communication_truth, communication_prediction),
        "compute": _diagnostics(compute_truth, compute_prediction),
        "joint": _diagnostics(joint_truth, joint_prediction),
    }
    checks = {
        "communication": diagnostics["communication"]["mape_percent"] <= thresholds.communication_mape_percent,
        "compute": diagnostics["compute"]["mape_percent"] <= thresholds.compute_mape_percent,
        "joint": diagnostics["joint"]["mape_percent"] <= thresholds.joint_mape_percent,
    }
    status = "accepted" if all(checks.values()) else "rejected"
    return {
        "schema_version": 1,
        "artifact_type": "placemoe_planner_calibration",
        "status": status,
        "scope": {
            "model_id": _model_identifier(root, model_id),
            "ep_size": ep_size,
            "ranks_per_node": int(ranks_per_node),
            "hierarchy_group_sizes": list(hierarchy),
        },
        "coefficients": {
            "inter_ms_per_byte": max(_COEFFICIENT_FLOOR, network_scale * inter_beta),
            "intra_ms_per_byte": max(_COEFFICIENT_FLOOR, network_scale * intra_beta),
            "route_ms_per_assignment": max(_COEFFICIENT_FLOOR, route_forward),
            "communication_multiplier": communication_phase,
            "compute_ms_per_assignment": max(_COEFFICIENT_FLOOR, compute_forward),
            "compute_multiplier": compute_phase,
        },
        "fit": {
            "warmup_steps": schedule.warmup_steps,
            "calibration_step": schedule.calibration_step,
            "validation_steps": [int(report["step"]) for report in validations],
            "total_training_steps": schedule.max_steps,
            "forward_communication": {
                "runtime_network_scale": network_scale,
                "route_ms_per_assignment": route_forward,
                "intercept_ms_not_used": communication_intercept,
            },
            "forward_compute": {
                "ms_per_assignment": compute_forward,
                "intercept_ms_not_used": compute_intercept,
            },
            "phase_multipliers": {
                "communication": communication_phase,
                "compute": compute_phase,
            },
            "phase_timing": phase_diagnostics,
        },
        "held_out_validation": {
            "basis": "serialized variable-cost coefficients without constant intercepts",
            **diagnostics,
            "thresholds": {
                "communication_mape_percent": thresholds.communication_mape_percent,
                "compute_mape_percent": thresholds.compute_mape_percent,
                "joint_mape_percent": thresholds.joint_mape_percent,
            },
            "checks": checks,
        },
        "provenance": {
            "runtime_perf_model_sha256": runtime_perf_model_sha256,
            "training_log_sha256": training_log_sha256,
            "runtime_perf_model_metadata": dict(runtime_metadata),
        },
    }


def materialize_model_calibration_config(
    source: Mapping[str, Any],
    *,
    runtime_perf_model: Path,
    work_directory: Path,
    schedule: ModelCalibrationSchedule,
) -> dict[str, Any]:
    """Create a side-effect-free short default-layout calibration config."""

    root = copy.deepcopy(dict(source))
    train = root.setdefault("train", {})
    if not isinstance(train, dict):
        raise ModelCalibrationError("train must be a mapping")
    accelerator = train.setdefault("accelerator", {})
    if not isinstance(accelerator, dict) or int(accelerator.get("ep_size", 0) or 0) <= 1:
        raise ModelCalibrationError("train.accelerator.ep_size must be greater than 1")
    train["max_steps"] = schedule.max_steps
    train["num_train_epochs"] = 1
    train["moe_load_balance_monitor_interval"] = 1
    optimizer = train.setdefault("optimizer", {})
    if not isinstance(optimizer, dict):
        raise ModelCalibrationError("train.optimizer must be a mapping")
    optimizer["lr"] = 0.0
    wandb = train.setdefault("wandb", {})
    if isinstance(wandb, dict):
        wandb["enable"] = False
    profile = train.setdefault("profile", {})
    if isinstance(profile, dict):
        profile["enable"] = False
    checkpoint = train.setdefault("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ModelCalibrationError("train.checkpoint must be a mapping")
    checkpoint.update(
        {
            "load_path": None,
            "output_dir": str(work_directory / "checkpoint"),
            "save_steps": 0,
            "save_epochs": 0,
            "hf_save_steps": 0,
            "hf_save_epochs": 0,
            "save_hf_weights": False,
        }
    )
    hiermoe = train.setdefault("hiermoe", {})
    if not isinstance(hiermoe, dict):
        raise ModelCalibrationError("train.hiermoe must be a mapping")
    if not hiermoe.get("hierarchy_group_sizes"):
        raise ModelCalibrationError("train.hiermoe.hierarchy_group_sizes must be explicit during calibration")
    hiermoe.update(
        {
            "enable": True,
            "token_dedup": True,
            "communication_mode": "hierarchical",
            "expert_swap": True,
            "expert_swap_interval": 1,
            "expert_swap_max_pairs_per_layer": 0,
            "expert_swap_mode": "step",
            "expert_swap_selector": "hiermoe_greedy_cover_p1",
            "max_slot_op_search_rounds": 0,
        }
    )
    hiermoe["perf_model_path"] = str(runtime_perf_model)
    hiermoe["redundant_slot_increment_per_device"] = max(
        1, int(hiermoe.get("redundant_slot_increment_per_device", 0) or 0)
    )
    hiermoe["log_interval"] = 1
    existing_placemoe = hiermoe.get("placemoe")
    resources = existing_placemoe.get("resources", {}) if isinstance(existing_placemoe, Mapping) else {}
    hiermoe["placemoe"] = {
        "enabled": True,
        "base_directory": str(work_directory),
        "initial_artifact": "",
        "runtime_perf_model": str(runtime_perf_model),
        "calibration": {
            "inter_ms_per_byte": 1.0e-12,
            "intra_ms_per_byte": 1.0e-12,
            "route_ms_per_assignment": 1.0e-12,
            "communication_multiplier": 1.0,
            "compute_ms_per_assignment": 1.0e-12,
            "compute_multiplier": 1.0,
        },
        "hot_update": {
            "enabled": False,
            "layout_interval_steps": 0,
            "mapping_interval_steps": 0,
            "work_root": str(work_directory / "planner"),
            "failure_policy": "raise",
        },
        "resources": dict(resources),
    }
    return root


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CalibrationThresholds",
    "ModelCalibrationError",
    "ModelCalibrationSchedule",
    "build_planner_calibration_artifact",
    "load_local_phase_timing_summary",
    "materialize_model_calibration_config",
    "merge_phase_timing_summaries",
    "parse_cost_model_reports",
    "sha256_path",
    "validate_runtime_performance_model",
]
