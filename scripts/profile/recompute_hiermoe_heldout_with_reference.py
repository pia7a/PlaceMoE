#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Recompute held-out cost predictions with coefficients from a reference workload."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-dataset", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--feature-model", default="stage_payload_inter_intra")
    parser.add_argument("--compute-max-mape", type=float, default=5.0)
    parser.add_argument("--compute-min-r2", type=float, default=0.5)
    parser.add_argument("--communication-max-mape", type=float, default=10.0)
    parser.add_argument("--communication-min-r2", type=float, default=0.0)
    parser.add_argument("--joint-max-mape", type=float, default=5.0)
    parser.add_argument("--joint-min-r2", type=float, default=0.5)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_calibration(artifact: dict[str, Any]) -> dict[str, Any]:
    source_log = Path(artifact["source_log"])
    calibration: dict[str, Any] | None = None
    marker = "HierMoE cost model calibration report: "
    with source_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker in line:
                calibration = json.loads(line.split(marker, 1)[1])
    if calibration is None:
        raise RuntimeError(f"calibration report not found in {source_log}")
    return calibration


def _predict(
    feature_names: list[str],
    coefficients: list[float],
    intercept: float,
    feature_values: dict[str, list[float]],
) -> list[float]:
    if len(feature_names) != len(coefficients):
        raise RuntimeError("feature names and coefficients are not aligned")
    lengths = {len(feature_values[name]) for name in feature_names}
    if len(lengths) != 1:
        raise RuntimeError(f"feature arrays are not aligned: {sorted(lengths)}")
    sample_count = next(iter(lengths))
    return [
        float(intercept)
        + sum(
            float(coefficient) * float(feature_values[name][index])
            for name, coefficient in zip(feature_names, coefficients)
        )
        for index in range(sample_count)
    ]


def _diagnostics(measured: list[float], predicted: list[float]) -> dict[str, float]:
    if not measured or len(measured) != len(predicted):
        raise RuntimeError("measured and predicted samples must be nonempty and aligned")
    actual = [float(value) for value in measured]
    estimate = [float(value) for value in predicted]
    residuals = [truth - guess for truth, guess in zip(actual, estimate)]
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    error = sum(value**2 for value in residuals)
    return {
        "actual_max_ms": max(actual),
        "actual_mean_ms": mean_actual,
        "actual_min_ms": min(actual),
        "mape_percent": 100.0
        * sum(abs(delta) / max(abs(truth), 1e-12) for truth, delta in zip(actual, residuals))
        / len(actual),
        "max_abs_error_ms": max(abs(value) for value in residuals),
        "predicted_max_ms": max(estimate),
        "predicted_mean_ms": sum(estimate) / len(estimate),
        "predicted_min_ms": min(estimate),
        "r_squared": 1.0 - error / total if total > 0.0 else math.nan,
        "rmse_ms": math.sqrt(error / len(actual)),
    }


def _metric_row(
    template: dict[str, Any],
    measured: list[float],
    predicted: list[float],
) -> dict[str, Any]:
    return {
        "name": template["name"],
        "feature_names": template["feature_names"],
        "coefficients": template["coefficients"],
        "intercept_ms": template["intercept_ms"],
        **_diagnostics(measured, predicted),
    }


def _passes(row: dict[str, Any], *, max_mape: float, min_r2: float) -> bool:
    return (
        math.isfinite(float(row["mape_percent"]))
        and math.isfinite(float(row["r_squared"]))
        and float(row["mape_percent"]) <= max_mape
        and float(row["r_squared"]) >= min_r2
    )


def main() -> None:
    args = _args()
    measurement = _load(args.measurement)
    reference = _load(args.reference)
    for field in (
        "model",
        "ep_size",
        "micro_batch_size",
        "global_batch_size",
        "maximum_sequence_length",
    ):
        if measurement.get(field) != reference.get(field):
            raise RuntimeError(
                f"measurement/reference mismatch for {field}: "
                f"{measurement.get(field)!r} != {reference.get(field)!r}"
            )
    if reference.get("status") != "accepted":
        raise RuntimeError(
            f"reference artifact must be accepted, got {reference.get('status')!r}"
        )

    calibration = _reference_calibration(reference)
    model = args.feature_model
    communication_template = {
        "name": model,
        **calibration["traffic_feature_models"]["communication"][model],
    }
    joint_template = {
        "name": model,
        **calibration["traffic_feature_models"]["joint"][model],
    }
    network_template = (
        {
            "name": model,
            **calibration["traffic_feature_models"]["network_joint"][model],
        }
        if model in calibration["traffic_feature_models"].get("network_joint", {})
        else None
    )

    artifact = copy.deepcopy(measurement)
    heldout_samples = artifact["sample_data"]["held_out_validation"]
    features = heldout_samples.get("feature_values")
    if not features:
        raise RuntimeError("measurement artifact has no held-out feature_values")

    compute = heldout_samples["compute"]
    compute_slope = float(reference["coefficients"]["compute_ms_per_assignment"])
    compute_constant = float(reference["coefficients"]["compute_constant_ms"])
    compute_predicted = [
        compute_constant + compute_slope * float(value)
        for value in compute["assignments"]
    ]
    communication = heldout_samples["communication_region"]
    communication_predicted = _predict(
        communication_template["feature_names"],
        communication_template["coefficients"],
        communication_template["intercept_ms"],
        features,
    )
    joint = heldout_samples["joint_moe_region"]
    joint_predicted = _predict(
        joint_template["feature_names"],
        joint_template["coefficients"],
        joint_template["intercept_ms"],
        features,
    )

    compute["predicted_ms"] = compute_predicted
    communication["predicted_ms"] = communication_predicted
    communication["feature_model"] = model
    joint["predicted_ms"] = joint_predicted
    joint["feature_model"] = model

    compute_metric = _diagnostics(compute["measured_ms"], compute_predicted)
    communication_metric = _metric_row(
        communication_template,
        communication["measured_ms"],
        communication_predicted,
    )
    joint_metric = _metric_row(
        joint_template,
        joint["measured_ms"],
        joint_predicted,
    )
    network_metric = None
    if network_template and "network_joint" in heldout_samples:
        network = heldout_samples["network_joint"]
        network_predicted = _predict(
            network_template["feature_names"],
            network_template["coefficients"],
            network_template["intercept_ms"],
            features,
        )
        network["predicted_ms"] = network_predicted
        network["feature_model"] = model
        network_metric = _metric_row(
            network_template,
            network["measured_ms"],
            network_predicted,
        )

    checks = {
        "compute": {
            "passed": _passes(
                compute_metric,
                max_mape=args.compute_max_mape,
                min_r2=args.compute_min_r2,
            ),
            "max_mape_percent": args.compute_max_mape,
            "min_r_squared": args.compute_min_r2,
        },
        "communication": {
            "passed": _passes(
                communication_metric,
                max_mape=args.communication_max_mape,
                min_r2=args.communication_min_r2,
            ),
            "max_mape_percent": args.communication_max_mape,
            "min_r_squared": args.communication_min_r2,
        },
        "joint": {
            "passed": _passes(
                joint_metric,
                max_mape=args.joint_max_mape,
                min_r2=args.joint_min_r2,
            ),
            "max_mape_percent": args.joint_max_mape,
            "min_r_squared": args.joint_min_r2,
        },
    }
    accepted = all(row["passed"] for row in checks.values())
    original_calibration = artifact.pop("calibration", None)
    artifact.update(
        {
            "schema_version": 5,
            "artifact_type": "hiermoe_cross_workload_cost_model_validation",
            "reference_dataset": args.reference_dataset,
            "measurement_dataset": args.measurement_dataset,
            "reference_artifact": {
                "path": str(args.reference.resolve()),
                "sha256": _sha256(args.reference),
                "source_run": reference["source_run"],
                "source_log": reference["source_log"],
                "source_log_sha256": reference["source_log_sha256"],
            },
            "measurement_artifact": {
                "path": str(args.measurement.resolve()),
                "sha256": _sha256(args.measurement),
                "source_run": measurement["source_run"],
                "source_log": measurement["source_log"],
                "source_log_sha256": measurement["source_log_sha256"],
            },
            "coefficients": {
                "compute_constant_ms": compute_constant,
                "compute_ms_per_assignment": compute_slope,
            },
            "reference_calibration": reference["calibration"],
            "measurement_run_calibration": original_calibration,
            "acceptance_joint_target": "region",
            "acceptance_checks": checks,
            "status": "accepted" if accepted else "rejected",
            "acceptance_reason": (
                "Frozen reference-workload coefficients passed all held-out "
                "cross-workload thresholds."
                if accepted
                else "Frozen reference-workload coefficients failed: "
                + ", ".join(name for name, row in checks.items() if not row["passed"])
                + "."
            ),
        }
    )
    heldout = artifact["held_out_validation"]
    heldout.update(
        {
            "compute": compute_metric,
            "communication": communication_metric,
            "joint": joint_metric,
            "region_joint": joint_metric,
            "network_joint": network_metric,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "reference_dataset": args.reference_dataset,
                "measurement_dataset": args.measurement_dataset,
                "compute": compute_metric,
                "communication": communication_metric,
                "joint": joint_metric,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
