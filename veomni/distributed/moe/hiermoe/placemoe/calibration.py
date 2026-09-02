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

from ..topology import expected_hierarchy_group_sizes


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


def _json_safe(value: Any) -> Any:
    """Normalize diagnostics to values accepted by strict RFC-compliant JSON encoders."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _runtime_profile(
    payload: Mapping[str, Any],
    *,
    ep_size: int,
    ranks_per_node: int,
    hierarchy_group_sizes: Sequence[int],
) -> tuple[float, float, Mapping[str, Any]]:
    hierarchy = tuple(int(value) for value in hierarchy_group_sizes)
    runtime_hierarchy = expected_hierarchy_group_sizes(ep_size, ranks_per_node)
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
    intra_beta = _positive(_mapping(payload.get("intra"), "intra").get("beta"), "intra.beta")
    inter_rows = payload.get("inter")
    if not isinstance(inter_rows, list):
        raise ModelCalibrationError("runtime performance model inter coefficients must be a list")
    if len(hierarchy) == 1:
        if inter_rows:
            raise ModelCalibrationError("single-node runtime performance model must not contain inter-node stages")
        inter_beta = intra_beta
    else:
        if not inter_rows:
            raise ModelCalibrationError("runtime performance model has no inter-node coefficients")
        inter_beta = _positive(_mapping(inter_rows[0], "inter[0]").get("beta"), "inter[0].beta")
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


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ModelCalibrationError("diagnostic distributions require finite non-empty samples")
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _pearson_correlation(lhs: Sequence[float], rhs: Sequence[float]) -> float | None:
    x = np.asarray(lhs, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.shape != y.shape:
        raise ModelCalibrationError("correlation diagnostics require non-empty paired samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ModelCalibrationError("correlation diagnostics contain non-finite samples")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    if denominator <= 0.0:
        return None
    return float((x @ y) / denominator)


def _extended_diagnostics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(actual, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    result: dict[str, Any] = dict(_diagnostics(truth, estimate))
    absolute_error = np.abs(estimate - truth)
    truth_sum = float(np.abs(truth).sum())
    result.update(
        {
            "mae_ms": float(absolute_error.mean()),
            "median_absolute_error_ms": float(np.median(absolute_error)),
            "wape_percent": float(absolute_error.sum() / truth_sum) * 100.0 if truth_sum > 0.0 else None,
            "actual_ms": _distribution_summary(truth),
            "predicted_ms": _distribution_summary(estimate),
        }
    )
    return result


def _variable_cost_diagnostics(
    raw_actual: Sequence[float],
    variable_prediction: Sequence[float],
    *,
    intercept_ms: float,
) -> dict[str, Any]:
    raw = np.asarray(raw_actual, dtype=np.float64)
    predicted = np.asarray(variable_prediction, dtype=np.float64)
    variable_truth = np.maximum(0.0, raw - float(intercept_ms))
    result = _extended_diagnostics(variable_truth, predicted)
    denominator = np.maximum(np.abs(variable_truth), 1.0e-6)
    absolute_percentage_error = np.abs(predicted - variable_truth) / denominator * 100.0
    result.update(
        {
            "zero_truth_sample_count": int(np.count_nonzero(variable_truth == 0.0)),
            "zero_truth_fraction": float(np.mean(variable_truth == 0.0)),
            "truth_below_0_1_ms_sample_count": int(np.count_nonzero(variable_truth < 0.1)),
            "truth_below_1_ms_sample_count": int(np.count_nonzero(variable_truth < 1.0)),
            "absolute_percentage_error_percent": _distribution_summary(absolute_percentage_error),
        }
    )
    return result


def _component_diagnostic_context(
    raw_actual: Sequence[float],
    variable_prediction: Sequence[float],
    *,
    intercept_ms: float,
    features: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    raw_prediction = [float(value) + float(intercept_ms) for value in variable_prediction]
    feature_arrays = {str(name): np.asarray(values, dtype=np.float64) for name, values in features.items()}
    feature_names = list(feature_arrays)
    feature_matrix = np.column_stack([feature_arrays[name] for name in feature_names])
    centered = feature_matrix - feature_matrix.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0.0)
    matrix_rank = int(np.linalg.matrix_rank(normalized))
    feature_count = len(feature_names)
    return {
        "fit_intercept_ms": float(intercept_ms),
        "raw_affine": _extended_diagnostics(raw_actual, raw_prediction),
        "serialized_variable": _variable_cost_diagnostics(
            raw_actual,
            variable_prediction,
            intercept_ms=intercept_ms,
        ),
        "feature_target_pearson": {
            str(name): _pearson_correlation(values, raw_actual) for name, values in features.items()
        },
        "feature_ranges": {str(name): _distribution_summary(values) for name, values in features.items()},
        "feature_distinct_value_counts": {
            name: int(np.unique(values).size) for name, values in feature_arrays.items()
        },
        "feature_pair_pearson": {
            f"{lhs}__vs__{rhs}": _pearson_correlation(feature_arrays[lhs], feature_arrays[rhs])
            for lhs, rhs in itertools.combinations(feature_names, 2)
        },
        "feature_matrix": {
            "feature_count": feature_count,
            "centered_rank": matrix_rank,
            "full_rank": matrix_rank == feature_count,
        },
    }


def _feature_distribution_shift(
    reference: Mapping[str, Sequence[float]], observed: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    if set(reference) != set(observed):
        raise ModelCalibrationError("feature-shift diagnostics require matching feature names")
    result: dict[str, Any] = {}
    for name in reference:
        fit_values = np.asarray(reference[name], dtype=np.float64)
        validation_values = np.asarray(observed[name], dtype=np.float64)
        if fit_values.size == 0 or validation_values.size == 0:
            raise ModelCalibrationError("feature-shift diagnostics require non-empty samples")
        fit_min = float(fit_values.min())
        fit_max = float(fit_values.max())
        below = int(np.count_nonzero(validation_values < fit_min))
        above = int(np.count_nonzero(validation_values > fit_max))
        result[str(name)] = {
            "fit_min": fit_min,
            "fit_max": fit_max,
            "validation_sample_count": int(validation_values.size),
            "below_fit_range_sample_count": below,
            "above_fit_range_sample_count": above,
            "outside_fit_range_sample_count": below + above,
            "outside_fit_range_fraction": float((below + above) / validation_values.size),
        }
    return result


def _runtime_message_coverage(
    runtime_metadata: Mapping[str, Any], validation_reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    requested = [float(value) for value in runtime_metadata.get("message_bytes_requested", ())]
    if not requested:
        return {
            "available": False,
            "reason": "runtime metadata has no message_bytes_requested",
        }
    requested_min = min(requested)
    requested_max = max(requested)
    result: dict[str, Any] = {
        "available": True,
        "comparison_basis": "endpoint payload bytes versus runtime requested message bytes (heuristic)",
        "requested_bytes": _distribution_summary(requested),
    }
    for stage, key in (
        ("stage1", "stage1_payload_endpoint_bytes"),
        ("stage2", "stage2_payload_endpoint_bytes"),
    ):
        observed = [float(value) for report in validation_reports for value in _offline_samples(report).get(key, ())]
        if not observed:
            result[stage] = {"available": False, "reason": f"validation reports have no {key}"}
            continue
        below = sum(value < requested_min for value in observed)
        above = sum(value > requested_max for value in observed)
        result[stage] = {
            "available": True,
            "observed_bytes": _distribution_summary(observed),
            "below_requested_range_sample_count": int(below),
            "above_requested_range_sample_count": int(above),
            "outside_requested_range_sample_count": int(below + above),
            "outside_requested_range_fraction": float((below + above) / len(observed)),
        }
    return result


def _sample_alignment_context(validation_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expert_count_samples = 0
    expert_count_mismatches = 0
    expert_count_max_delta = 0.0
    expert_count_missing_steps: list[int] = []
    route_samples = 0
    route_mismatches = 0
    route_max_delta = 0.0
    route_missing_steps: list[int] = []
    destination_rank_samples = 0
    destination_rank_mismatch_samples = 0
    destination_rank_mismatches = 0
    destination_rank_max_delta = 0.0
    destination_rank_missing_steps: list[int] = []
    per_step: list[dict[str, Any]] = []
    for report in validation_reports:
        step = int(report.get("step", -1))
        samples = _offline_samples(report)
        assignments = [float(value) for value in samples.get("paired_assignments", ())]
        expert_counts = samples.get("paired_expert_token_counts")
        if isinstance(expert_counts, list) and len(expert_counts) == len(assignments):
            deltas = [
                abs(sum(float(value) for value in row) - assignment)
                for row, assignment in zip(expert_counts, assignments, strict=True)
            ]
            expert_count_samples += len(deltas)
            expert_count_mismatches += sum(delta > 0.5 for delta in deltas)
            expert_count_max_delta = max(expert_count_max_delta, max(deltas, default=0.0))
        else:
            expert_count_missing_steps.append(step)

        raw_alignment = report.get("sample_alignment")
        step_alignment: dict[str, Any] = {"step": step, "available": False}
        if isinstance(raw_alignment, Mapping):
            source = [float(value) for value in raw_alignment.get("source_assignment_totals", ())]
            destination = [float(value) for value in raw_alignment.get("destination_assignment_totals", ())]
            if source and len(source) == len(destination):
                deltas = [abs(lhs - rhs) for lhs, rhs in zip(source, destination, strict=True)]
                mismatches = sum(delta > 0.5 for delta in deltas)
                route_samples += len(deltas)
                route_mismatches += mismatches
                route_max_delta = max(route_max_delta, max(deltas, default=0.0))
                step_alignment = {
                    "step": step,
                    "available": True,
                    "ep_size": int(raw_alignment.get("ep_size", 0) or 0),
                    "layer_count": len(raw_alignment.get("layer_keys", ())),
                    "row_count_per_rank": int(raw_alignment.get("row_count_per_rank", 0) or 0),
                    "sample_count": len(deltas),
                    "mismatch_sample_count": int(mismatches),
                    "max_abs_delta": float(max(deltas, default=0.0)),
                }
            else:
                route_missing_steps.append(step)

            rank_mismatch_counts = [int(value) for value in raw_alignment.get("destination_rank_mismatch_counts", ())]
            rank_max_deltas = [float(value) for value in raw_alignment.get("destination_rank_max_abs_deltas", ())]
            if rank_mismatch_counts and len(rank_mismatch_counts) == len(rank_max_deltas):
                mismatch_row_indices = [index for index, value in enumerate(rank_mismatch_counts) if value > 0]
                mismatch_rows = len(mismatch_row_indices)
                row_layer_indices = list(raw_alignment.get("row_layer_indices", ()))
                row_call_indices = list(raw_alignment.get("row_call_indices", ()))
                layer_keys = list(raw_alignment.get("layer_keys", ()))
                mismatch_examples = []
                for row_index in mismatch_row_indices[:20]:
                    layer_index = int(row_layer_indices[row_index]) if row_index < len(row_layer_indices) else None
                    layer_key = (
                        str(layer_keys[layer_index])
                        if layer_index is not None and 0 <= layer_index < len(layer_keys)
                        else None
                    )
                    mismatch_examples.append(
                        {
                            "row_index": row_index,
                            "layer_index": layer_index,
                            "layer_key": layer_key,
                            "call_index": int(row_call_indices[row_index])
                            if row_index < len(row_call_indices)
                            else None,
                            "mismatched_rank_count": rank_mismatch_counts[row_index],
                            "max_abs_delta": rank_max_deltas[row_index],
                        }
                    )
                destination_rank_samples += len(rank_mismatch_counts)
                destination_rank_mismatch_samples += mismatch_rows
                destination_rank_mismatches += sum(rank_mismatch_counts)
                destination_rank_max_delta = max(destination_rank_max_delta, max(rank_max_deltas, default=0.0))
                step_alignment["destination_rank_alignment"] = {
                    "available": True,
                    "sample_count": len(rank_mismatch_counts),
                    "mismatch_sample_count": int(mismatch_rows),
                    "mismatched_rank_count": int(sum(rank_mismatch_counts)),
                    "max_abs_delta": float(max(rank_max_deltas, default=0.0)),
                    "mismatch_examples": mismatch_examples,
                    "mismatch_examples_truncated": len(mismatch_row_indices) > len(mismatch_examples),
                }
            else:
                destination_rank_missing_steps.append(step)
        else:
            route_missing_steps.append(step)
            destination_rank_missing_steps.append(step)
        per_step.append(step_alignment)
    return {
        "expert_count_sums": {
            "available": not expert_count_missing_steps,
            "sample_count": int(expert_count_samples),
            "mismatch_sample_count": int(expert_count_mismatches),
            "max_abs_delta": float(expert_count_max_delta),
            "missing_steps": expert_count_missing_steps,
        },
        "route_assignment_conservation": {
            "available": not route_missing_steps,
            "sample_count": int(route_samples),
            "mismatch_sample_count": int(route_mismatches),
            "max_abs_delta": float(route_max_delta),
            "missing_steps": route_missing_steps,
        },
        "destination_rank_assignment_alignment": {
            "available": not destination_rank_missing_steps,
            "sample_count": int(destination_rank_samples),
            "mismatch_sample_count": int(destination_rank_mismatch_samples),
            "mismatched_rank_count": int(destination_rank_mismatches),
            "max_abs_delta": float(destination_rank_max_delta),
            "missing_steps": destination_rank_missing_steps,
        },
        "per_step": per_step,
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


def summarize_phase_timing_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_ranks: Sequence[int],
    expected_steps: Sequence[int],
    timing_file_count: int | None = None,
) -> dict[str, Any]:
    """Validate in-memory forward/backward spans and build a critical-path summary."""

    expected_rank_set = {int(value) for value in expected_ranks}
    expected_step_set = {int(value) for value in expected_steps}
    if not expected_rank_set or not expected_step_set:
        raise ModelCalibrationError("phase timing requires non-empty expected ranks and steps")
    observed_ranks = {int(row["rank"]) for row in rows}
    if observed_ranks != expected_rank_set:
        raise ModelCalibrationError(f"timing ranks are {sorted(observed_ranks)}, expected {sorted(expected_rank_set)}")
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
        "timing_file_count": len(expected_rank_set) if timing_file_count is None else int(timing_file_count),
        "critical_rows": critical_rows,
    }


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

    paths = sorted(timing_directory.glob("moe_timing_rank*.jsonl"))
    expected_rank_set = {int(value) for value in expected_ranks}
    if len(paths) != len(expected_rank_set):
        raise ModelCalibrationError(
            f"expected {len(expected_rank_set)} local timing files, found {len(paths)} in {timing_directory}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return summarize_phase_timing_rows(
        rows,
        expected_ranks=expected_ranks,
        expected_steps=expected_steps,
        timing_file_count=len(paths),
    )


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
    return (
        communication_multiplier,
        compute_multiplier,
        {
            "selected_steps": list(expected_steps or ()),
            "timing_file_count": timing_file_count,
            "rank_count": len(all_ranks),
            **totals,
        },
    )


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
    model_id: str | None = None,
) -> dict[str, Any]:
    """Fit, validate, and serialize a complete PlaceMoE planner artifact."""

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
        raise ModelCalibrationError(f"expected {schedule.validation_steps} held-out reports, found {len(validations)}")
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
    fit_communication_feature_values = {
        "runtime_network_units": [float(row[0]) for row in communication_features],
        "peak_assignments": [float(row[1]) for row in communication_features],
    }
    fit_compute_feature_values = {
        "paired_assignments": [float(row[0]) for row in compute_features],
    }
    fit_diagnostic_context = {
        "step": int(calibration["step"]),
        "communication": _component_diagnostic_context(
            communication_actual,
            [network_scale * network + route_forward * assignments for network, assignments in communication_features],
            intercept_ms=communication_intercept,
            features=fit_communication_feature_values,
        ),
        "compute": _component_diagnostic_context(
            compute_actual,
            [compute_forward * row[0] for row in compute_features],
            intercept_ms=compute_intercept,
            features=fit_compute_feature_values,
        ),
        "sample_alignment": _sample_alignment_context([calibration]),
    }
    trainer_calibration_report = {
        "step": int(calibration["step"]),
        **{
            name: dict(calibration[name]) if isinstance(calibration.get(name), Mapping) else {"available": False}
            for name in ("communication", "compute", "joint")
        },
    }
    communication_phase, compute_phase, phase_diagnostics = merge_phase_timing_summaries(phase_timing_summaries)
    expected_rank_count = ep_size
    if int(phase_diagnostics["rank_count"]) != expected_rank_count:
        raise ModelCalibrationError(
            f"phase timing covers {phase_diagnostics['rank_count']} ranks, expected {expected_rank_count}"
        )

    communication_truth: list[float] = []
    communication_prediction: list[float] = []
    communication_raw_actual: list[float] = []
    communication_network_feature: list[float] = []
    communication_assignment_feature: list[float] = []
    compute_truth: list[float] = []
    compute_prediction: list[float] = []
    compute_raw_actual: list[float] = []
    compute_assignment_feature: list[float] = []
    joint_truth: list[float] = []
    joint_prediction: list[float] = []
    per_step_diagnostics: list[dict[str, Any]] = []
    trainer_validation_reports: list[dict[str, Any]] = []
    for report in validations:
        features, actual = _forward_communication_rows(report, inter_beta, intra_beta)
        predicted_communication = [
            network_scale * network + route_forward * assignments for network, assignments in features
        ]
        variable_communication = [max(0.0, value - communication_intercept) for value in actual]
        communication_truth.extend(variable_communication)
        communication_prediction.extend(predicted_communication)
        communication_raw_actual.extend(actual)
        communication_network_feature.extend(float(row[0]) for row in features)
        communication_assignment_feature.extend(float(row[1]) for row in features)

        validation_compute_features, validation_compute_actual = _forward_compute_rows(report)
        predicted_compute = [compute_forward * row[0] for row in validation_compute_features]
        variable_compute_samples = [max(0.0, value - compute_intercept) for value in validation_compute_actual]
        compute_truth.extend(variable_compute_samples)
        compute_prediction.extend(predicted_compute)
        compute_raw_actual.extend(validation_compute_actual)
        compute_assignment_feature.extend(float(row[0]) for row in validation_compute_features)

        per_step_diagnostics.append(
            {
                "step": int(report["step"]),
                "communication": _component_diagnostic_context(
                    actual,
                    predicted_communication,
                    intercept_ms=communication_intercept,
                    features={
                        "runtime_network_units": [float(row[0]) for row in features],
                        "peak_assignments": [float(row[1]) for row in features],
                    },
                ),
                "compute": _component_diagnostic_context(
                    validation_compute_actual,
                    predicted_compute,
                    intercept_ms=compute_intercept,
                    features={
                        "paired_assignments": [float(row[0]) for row in validation_compute_features],
                    },
                ),
            }
        )
        trainer_validation_reports.append(
            {
                "step": int(report["step"]),
                **{
                    name: dict(report[name]) if isinstance(report.get(name), Mapping) else {"available": False}
                    for name in ("communication", "compute", "joint")
                },
            }
        )

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
    sample_alignment = _sample_alignment_context(validations)
    runtime_message_coverage = _runtime_message_coverage(runtime_metadata, validations)
    communication_context = _component_diagnostic_context(
        communication_raw_actual,
        communication_prediction,
        intercept_ms=communication_intercept,
        features={
            "runtime_network_units": communication_network_feature,
            "peak_assignments": communication_assignment_feature,
        },
    )
    compute_context = _component_diagnostic_context(
        compute_raw_actual,
        compute_prediction,
        intercept_ms=compute_intercept,
        features={"paired_assignments": compute_assignment_feature},
    )
    feature_distribution_shift = {
        "communication": _feature_distribution_shift(
            fit_communication_feature_values,
            {
                "runtime_network_units": communication_network_feature,
                "peak_assignments": communication_assignment_feature,
            },
        ),
        "compute": _feature_distribution_shift(
            fit_compute_feature_values,
            {"paired_assignments": compute_assignment_feature},
        ),
    }
    outside_runtime_range = sum(
        int(_mapping(runtime_message_coverage.get(stage), stage).get("outside_requested_range_sample_count", 0))
        for stage in ("stage1", "stage2")
        if isinstance(runtime_message_coverage.get(stage), Mapping)
    )
    sample_alignment_mismatches = (
        int(_mapping(sample_alignment["expert_count_sums"], "expert count alignment")["mismatch_sample_count"])
        + int(
            _mapping(sample_alignment["route_assignment_conservation"], "route assignment alignment")[
                "mismatch_sample_count"
            ]
        )
        + int(
            _mapping(sample_alignment["destination_rank_assignment_alignment"], "destination rank alignment")[
                "mismatch_sample_count"
            ]
        )
    )
    outside_fit_range = sum(
        int(feature["outside_fit_range_sample_count"])
        for component in feature_distribution_shift.values()
        for feature in component.values()
    )
    diagnostic_context = {
        "gate_effect": "none",
        "fit_step": fit_diagnostic_context,
        "communication": communication_context,
        "compute": compute_context,
        "per_step": per_step_diagnostics,
        "trainer_calibration_report": trainer_calibration_report,
        "trainer_validation_reports": trainer_validation_reports,
        "feature_distribution_shift": feature_distribution_shift,
        "runtime_message_coverage": runtime_message_coverage,
        "sample_alignment": sample_alignment,
        "signals": {
            "communication_raw_affine_r_squared_negative": communication_context["raw_affine"]["r_squared"] < 0.0,
            "compute_raw_affine_r_squared_negative": compute_context["raw_affine"]["r_squared"] < 0.0,
            "communication_zero_truth_sample_count": communication_context["serialized_variable"][
                "zero_truth_sample_count"
            ],
            "compute_zero_truth_sample_count": compute_context["serialized_variable"]["zero_truth_sample_count"],
            "runtime_payload_outside_requested_range_sample_count": outside_runtime_range,
            "validation_feature_outside_fit_range_sample_count": outside_fit_range,
            "sample_alignment_mismatch": sample_alignment_mismatches > 0,
            "fit_communication_feature_matrix_full_rank": fit_diagnostic_context["communication"]["feature_matrix"][
                "full_rank"
            ],
            "runtime_network_scale_is_zero": network_scale == 0.0,
            "route_coefficient_is_zero": route_forward == 0.0,
            "compute_coefficient_is_zero": compute_forward == 0.0,
        },
    }
    return _json_safe(
        {
            "schema_version": 1,
            "artifact_type": "placemoe_planner_calibration",
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
                "diagnostic_context": diagnostic_context,
            },
            "provenance": {
                "runtime_perf_model_sha256": runtime_perf_model_sha256,
                "training_log_sha256": training_log_sha256,
                "runtime_perf_model_metadata": dict(runtime_metadata),
            },
        }
    )


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
    optimizer["lr_min"] = 0.0
    optimizer["lr_warmup_ratio"] = 0.0
    optimizer["lr_decay_style"] = "constant"
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
    hierarchy_levels = len(hiermoe["hierarchy_group_sizes"])
    redundant_slots = max(0, int(hiermoe.get("redundant_slot_increment_per_device", 0) or 0))
    hiermoe.update(
        {
            "enable": True,
            "token_dedup": True,
            "communication_mode": "hierarchical",
            "expert_swap": True,
            "expert_swap_interval": 1,
            "expert_swap_max_pairs_per_layer": 0,
            "expert_swap_mode": "step",
            "expert_swap_selector": "hiermoe_greedy_cover_p1" if redundant_slots > 0 else "current_joint",
            "max_slot_op_search_rounds": 0,
            "fixed_pipeline_overlap": redundant_slots > 0 and hierarchy_levels == 2,
        }
    )
    hiermoe["perf_model_path"] = str(runtime_perf_model)
    hiermoe["redundant_slot_increment_per_device"] = redundant_slots
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
    "ModelCalibrationError",
    "ModelCalibrationSchedule",
    "build_planner_calibration_artifact",
    "load_local_phase_timing_summary",
    "materialize_model_calibration_config",
    "merge_phase_timing_summaries",
    "summarize_phase_timing_rows",
    "parse_cost_model_reports",
    "sha256_path",
    "validate_runtime_performance_model",
]
