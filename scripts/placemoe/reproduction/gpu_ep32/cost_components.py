#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Composable calibration helpers for the EP32 offline HierMoE scorer."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_communication_calibration(
    path: Path,
    *,
    ep_size: int,
    ranks_per_node: int,
    hidden_size: int,
    bytes_per_element: int,
    preflight_report: Path | None = None,
    communication_source_sha256: str | None = None,
    max_validation_mape_percent: float = 10.0,
) -> tuple[float, float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {3, 4}:
        raise ValueError(f"{path} must use communication calibration schema v3 or v4")
    if payload.get("source") != "gpu32-a6000-ep32-communication-calibration":
        raise ValueError(f"{path} is not an EP32 A6000 communication calibration")
    topology = payload.get("topology")
    if not isinstance(topology, dict):
        raise ValueError(f"{path} has no topology object")
    hierarchy = [ranks_per_node, ep_size] if schema_version == 3 else [2, ranks_per_node, ep_size]
    expected = {
        "accelerator": "NVIDIA RTX A6000",
        "nodes": ep_size // ranks_per_node,
        "gpus_per_node": ranks_per_node,
        "ep_size": ep_size,
        "ranks_per_node": ranks_per_node,
        "hierarchy_group_sizes": hierarchy,
        "hidden_size": hidden_size,
        "bytes_per_element": bytes_per_element,
    }
    mismatches = {key: (topology.get(key), value) for key, value in expected.items() if topology.get(key) != value}
    if mismatches:
        raise ValueError(f"{path} topology mismatch: {mismatches}")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, dict):
        raise ValueError(f"{path} has no coefficients object")
    if schema_version == 3:
        level_values = [
            float(coefficients["inter_ms_per_byte"]),
            float(coefficients["intra_ms_per_byte"]),
        ]
        feature_names = ("inter", "intra")
        model_names = ("inter", "intra")
        validation_names = ("stage1_inter", "stage2_intra")
        raw_names = ("inter_stage_group", "intra_stage_group")
    else:
        raw_levels = coefficients.get("level_ms_per_byte")
        if not isinstance(raw_levels, list) or len(raw_levels) != 3:
            raise ValueError(f"{path} must export three level_ms_per_byte coefficients")
        level_values = [float(value) for value in raw_levels]
        feature_names = ("levels",)
        model_names = ("stage1_inter", "stage2_mid", "stage3_intra")
        validation_names = model_names
        raw_names = ("stage1_inter_group", "stage2_mid_group", "stage3_intra_group")
        legacy_values = (
            float(coefficients.get("inter_ms_per_byte", math.nan)),
            float(coefficients.get("mid_ms_per_byte", math.nan)),
            float(coefficients.get("intra_ms_per_byte", math.nan)),
        )
        if any(
            not math.isclose(actual, expected_value, rel_tol=1.0e-12, abs_tol=0.0)
            for actual, expected_value in zip(legacy_values, level_values, strict=True)
        ):
            raise ValueError(f"{path} named coefficients do not match level_ms_per_byte")
    if any(not math.isfinite(value) or value <= 0.0 for value in level_values):
        raise ValueError(f"{path} has invalid level coefficients {level_values!r}")
    coefficient_features = payload.get("coefficient_features")
    feature = "raw_balanced_a2a_payload_bytes_per_source"
    expected_features = (
        {"inter": feature, "intra": feature} if schema_version == 3 else {"levels": [feature, feature, feature]}
    )
    if coefficient_features != expected_features:
        raise ValueError(f"{path} has unsupported coefficient features: {coefficient_features!r}")
    if tuple(coefficient_features) != feature_names:
        raise ValueError(f"{path} coefficient feature order is invalid")
    link_models = payload.get("link_models")
    if not isinstance(link_models, dict):
        raise ValueError(f"{path} has no physical link models")
    for index, stage in enumerate(model_names):
        model = link_models.get(stage)
        if not isinstance(model, dict) or model.get("kind") != "local_alpha_beta_bracketing_workload_payload":
            raise ValueError(f"{path} has invalid {stage} link model")
        alpha = float(model.get("alpha_ms", math.nan))
        beta = float(model.get("beta_ms_per_byte", math.nan))
        if not math.isfinite(alpha) or alpha < 0.0 or not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"{path} has invalid {stage} alpha-beta coefficients")
        if not math.isclose(beta, level_values[index], rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError(f"{path} {stage} model beta does not match exported coefficient")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{path} has no held-out validation")
    for stage in validation_names:
        diagnostics = validation.get(stage)
        if not isinstance(diagnostics, dict) or int(diagnostics.get("count", 0)) <= 0:
            raise ValueError(f"{path} has no held-out samples for {stage}")
        if diagnostics.get("kind") != "raw_balanced_a2a_local_alpha_beta_holdout":
            raise ValueError(f"{path} has unsupported held-out validation for {stage}")
        mape = float(diagnostics.get("mape_percent", math.inf))
        if not math.isfinite(mape) or mape > max_validation_mape_percent:
            raise ValueError(f"{path} held-out {stage} MAPE {mape:.6f}% exceeds {max_validation_mape_percent:.6f}%")
    scope = payload.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") != "gpu32-cluster-software-v1":
        raise ValueError(f"{path} has no supported cluster/software scope")
    if preflight_report is not None:
        expected_preflight_sha256 = sha256(preflight_report)
        artifact_preflight_sha256 = scope.get("preflight_sha256")
        if artifact_preflight_sha256 != expected_preflight_sha256:
            raise ValueError(
                f"{path} preflight mismatch: artifact={artifact_preflight_sha256!r}, "
                f"current={expected_preflight_sha256!r}"
            )
    if (
        communication_source_sha256 is not None
        and scope.get("communication_source_sha256") != communication_source_sha256
    ):
        raise ValueError(f"{path} communication source fingerprint mismatch")
    raw_coverage = payload.get("coverage", {}).get("raw_all_to_all", {})
    return (
        level_values[0],
        level_values[-1],
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "source": payload["source"],
            "run_name": payload.get("run_name"),
            "schema_version": schema_version,
            "scope": scope,
            "hierarchy_group_sizes": hierarchy,
            "level_ms_per_byte": level_values,
            "validation": validation,
            "coverage": {
                "route_patterns": payload.get("coverage", {}).get("route_patterns"),
                "tokens_per_rank": payload.get("coverage", {}).get("tokens_per_rank"),
                "raw_samples_by_level": [len(raw_coverage.get(name, [])) for name in raw_names],
            },
        },
    )


def _curve_rows(report: dict[str, Any], phase: str) -> tuple[np.ndarray, np.ndarray]:
    samples = report.get("offline_scorer_samples")
    if not isinstance(samples, dict):
        raise ValueError(f"{phase} report has no offline_scorer_samples")
    token_rows = np.asarray(samples.get("paired_expert_token_counts"), dtype=np.float64)
    compute_ms = np.asarray(samples.get("paired_compute_ms"), dtype=np.float64)
    assignments = np.asarray(samples.get("paired_assignments"), dtype=np.float64)
    if token_rows.ndim != 2 or compute_ms.ndim != 1 or assignments.ndim != 1:
        raise ValueError(f"{phase} compute-curve arrays have invalid dimensions")
    if len(token_rows) == 0 or len(token_rows) != len(compute_ms) or len(token_rows) != len(assignments):
        raise ValueError(f"{phase} compute-curve arrays are empty or misaligned")
    if not np.isfinite(token_rows).all() or not np.isfinite(compute_ms).all():
        raise ValueError(f"{phase} compute-curve arrays contain non-finite values")
    if (token_rows < 0).any() or (compute_ms <= 0).any():
        raise ValueError(f"{phase} compute-curve arrays contain invalid values")
    if not np.allclose(token_rows.sum(axis=1), assignments, rtol=0.0, atol=0.5):
        raise ValueError(f"{phase} expert-token rows do not sum to paired assignments")
    return token_rows, compute_ms


def _knots(token_rows: np.ndarray) -> list[float]:
    positive = token_rows[token_rows > 0]
    if positive.size == 0:
        raise ValueError("compute calibration has no positive expert token bins")
    candidates = [0.0]
    candidates.extend(float(np.quantile(positive, quantile, method="nearest")) for quantile in (0.5, 0.8, 0.95))
    return sorted(set(candidates))


def _design(token_rows: np.ndarray, knots: list[float]) -> np.ndarray:
    columns = []
    for index, lower in enumerate(knots):
        values = np.maximum(token_rows - lower, 0.0)
        if index + 1 < len(knots):
            values = np.minimum(values, knots[index + 1] - lower)
        columns.append(values.sum(axis=1))
    columns.append(np.ones((len(token_rows),), dtype=np.float64))
    return np.stack(columns, axis=1)


def _diagnostics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residual = actual - predicted
    mean = float(actual.mean())
    variance = float(np.square(actual - mean).sum())
    squared = float(np.square(residual).sum())
    return {
        "count": int(len(actual)),
        "actual_mean_ms": mean,
        "prediction_mean_ms": float(predicted.mean()),
        "rmse_ms": math.sqrt(squared / len(actual)),
        "mape_percent": float(np.mean(np.abs(residual) / actual) * 100.0),
        "r_squared": None if variance <= 0.0 else 1.0 - squared / variance,
        "actual_min_ms": float(actual.min()),
        "actual_max_ms": float(actual.max()),
        "prediction_min_ms": float(predicted.min()),
        "prediction_max_ms": float(predicted.max()),
    }


def fit_compute_curve(
    calibration: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_tokens, calibration_ms = _curve_rows(calibration, "calibration")
    validation_tokens, validation_ms = _curve_rows(validation, "validation")
    if calibration_tokens.shape[1] != validation_tokens.shape[1]:
        raise ValueError("calibration/validation local expert widths differ")
    knots = _knots(calibration_tokens)
    calibration_design = _design(calibration_tokens, knots)
    coefficients, residual_norm = nnls(calibration_design, calibration_ms)
    slopes = coefficients[:-1]
    constant = float(coefficients[-1])
    calibration_predicted = calibration_design @ coefficients
    validation_predicted = _design(validation_tokens, knots) @ coefficients
    curve = {
        "kind": "sum_piecewise_linear_per_local_expert",
        "knots_tokens": knots,
        "segment_ms_per_token": [float(value) for value in slopes],
        "constant_ms": constant,
        "local_experts_per_rank": int(calibration_tokens.shape[1]),
        "nonnegative_slopes": True,
    }
    diagnostics = {
        "fit": _diagnostics(calibration_ms, calibration_predicted),
        "validation": _diagnostics(validation_ms, validation_predicted),
        "nnls_residual_norm": float(residual_norm),
        "positive_token_bin_summary": {
            "count": int((calibration_tokens > 0).sum()),
            "min": float(calibration_tokens[calibration_tokens > 0].min()),
            "median": float(statistics.median(calibration_tokens[calibration_tokens > 0].tolist())),
            "max": float(calibration_tokens.max()),
        },
    }
    return curve, diagnostics
