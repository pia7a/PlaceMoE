#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Extract and audit HierMoE cost-model calibration and validation logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--maximum-sequence-length", type=int, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--compute-max-mape", type=float, default=5.0)
    parser.add_argument("--compute-min-r2", type=float, default=0.5)
    parser.add_argument("--communication-max-mape", type=float, default=10.0)
    parser.add_argument("--communication-min-r2", type=float, default=0.0)
    parser.add_argument("--joint-max-mape", type=float, default=5.0)
    parser.add_argument("--joint-min-r2", type=float, default=0.5)
    parser.add_argument(
        "--exclude-validation-step",
        type=int,
        action="append",
        default=[],
        help=(
            "Exclude an entire held-out step from pooled acceptance and sample_data. "
            "The raw source log remains immutable and its SHA256 is recorded."
        ),
    )
    parser.add_argument(
        "--exclusion-reason",
        default="",
        help="Auditable reason for any whole-step held-out exclusions.",
    )
    parser.add_argument(
        "--joint-target",
        choices=("network", "region"),
        default="network",
        help=(
            "Joint target used for acceptance. 'network' audits raw A2A plus "
            "compute; 'region' matches the communication-region plus compute "
            "objective consumed by the offline layout builder."
        ),
    )
    return parser.parse_args()


def _reports(text: str) -> dict[str, Any]:
    calibration_marker = "HierMoE cost model calibration report: "
    validation_marker = "HierMoE cost model validation report: "
    calibration: dict[str, Any] | None = None
    validations: list[dict[str, Any]] = []
    for line in text.replace("\r", "\n").splitlines():
        if calibration_marker in line:
            candidate = line.split(calibration_marker, 1)[1].strip()
            calibration = json.loads(candidate)
        if validation_marker in line:
            candidate = line.split(validation_marker, 1)[1].strip()
            validations.append(json.loads(candidate))
    if calibration is None:
        raise RuntimeError("missing cost-model calibration report in log")
    if not validations:
        raise RuntimeError("missing cost-model validation report in log")
    validations.sort(key=lambda report: int(report["step"]))
    return {"calibration": calibration, "validations": validations}


def _diagnostics(measured: list[float], predicted: list[float]) -> dict[str, float]:
    if not measured or len(measured) != len(predicted):
        raise RuntimeError(
            "measured/predicted samples must be nonempty and have equal lengths: "
            f"{len(measured)} != {len(predicted)}"
        )
    actual = [float(value) for value in measured]
    estimate = [float(value) for value in predicted]
    residuals = [truth - guess for truth, guess in zip(actual, estimate)]
    mean_actual = sum(actual) / len(actual)
    sum_squared_total = sum((value - mean_actual) ** 2 for value in actual)
    sum_squared_error = sum(value**2 for value in residuals)
    r_squared = (
        1.0 - sum_squared_error / sum_squared_total
        if sum_squared_total > 0.0
        else (1.0 if sum_squared_error == 0.0 else 0.0)
    )
    absolute_percentage_errors = [
        abs(error) / abs(truth)
        for truth, error in zip(actual, residuals)
        if truth != 0.0
    ]
    mape = (
        100.0 * sum(absolute_percentage_errors) / len(absolute_percentage_errors)
        if absolute_percentage_errors
        else math.inf
    )
    return {
        "actual_max_ms": max(actual),
        "actual_mean_ms": mean_actual,
        "actual_min_ms": min(actual),
        "mape_percent": mape,
        "max_abs_error_ms": max(abs(value) for value in residuals),
        "predicted_max_ms": max(estimate),
        "predicted_mean_ms": sum(estimate) / len(estimate),
        "predicted_min_ms": min(estimate),
        "r_squared": r_squared,
        "rmse_ms": math.sqrt(sum_squared_error / len(residuals)),
    }


def _pooled_sample_data(validations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sample_keys = (
        "compute",
        "communication_region",
        "joint_moe_region",
        "network_joint",
    )
    pooled: dict[str, dict[str, Any]] = {}
    for sample_key in sample_keys:
        rows = []
        for validation in validations:
            sample_data = validation.get("sample_data")
            if not sample_data or sample_key not in sample_data:
                raise RuntimeError(
                    f"validation step {validation.get('step')} has no sample_data.{sample_key}"
                )
            rows.append(sample_data[sample_key])
        feature_models = {
            str(row["feature_model"]) for row in rows if row.get("feature_model") is not None
        }
        if len(feature_models) > 1:
            raise RuntimeError(
                f"validation sample_data.{sample_key} changed feature model: "
                f"{sorted(feature_models)}"
            )
        combined: dict[str, Any] = {}
        if feature_models:
            combined["feature_model"] = next(iter(feature_models))
        list_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, list)
        }
        for field in sorted(list_fields):
            combined[field] = [
                value for row in rows for value in row.get(field, [])
            ]
        pooled[sample_key] = combined
    feature_rows = [
        validation["sample_data"].get("feature_values")
        for validation in validations
    ]
    if any(row is not None for row in feature_rows):
        if any(row is None for row in feature_rows):
            raise RuntimeError("validation sample_data.feature_values is incomplete")
        assert all(row is not None for row in feature_rows)
        feature_names = {name for row in feature_rows for name in row}
        if any(set(row) != feature_names for row in feature_rows):
            raise RuntimeError("validation sample_data.feature_values changed fields")
        pooled["feature_values"] = {
            name: [value for row in feature_rows for value in row[name]]
            for name in sorted(feature_names)
        }
    return pooled


def _pooled_metric(
    pooled: dict[str, dict[str, Any]],
    sample_key: str,
    *,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = pooled[sample_key]
    result: dict[str, Any] = {}
    if template:
        for key in ("name", "feature_names", "coefficients", "intercept_ms"):
            if key in template:
                result[key] = template[key]
    if samples.get("feature_model") is not None:
        result.setdefault("name", samples["feature_model"])
    result.update(_diagnostics(samples["measured_ms"], samples["predicted_ms"]))
    return result


def _best_feature(report: dict[str, Any], target: str) -> dict[str, Any]:
    models = report.get("traffic_feature_models", {}).get(target, {})
    if not models:
        raise RuntimeError(f"validation report has no traffic feature models for {target}")
    name, row = min(
        models.items(),
        key=lambda item: (
            float(item[1].get("mape_percent", math.inf)),
            -float(item[1].get("r_squared", -math.inf)),
        ),
    )
    return {"name": name, **row}


def _named_feature(report: dict[str, Any], target: str, name: str) -> dict[str, Any]:
    models = report.get("traffic_feature_models", {}).get(target, {})
    if name not in models:
        raise RuntimeError(
            f"validation report has no traffic feature model {name!r} for {target}"
        )
    return {"name": name, **models[name]}


def _passes(row: dict[str, Any], *, max_mape: float, min_r2: float) -> bool:
    mape = float(row.get("mape_percent", math.inf))
    r2 = float(row.get("r_squared", -math.inf))
    return math.isfinite(mape) and math.isfinite(r2) and mape <= max_mape and r2 >= min_r2


def main() -> None:
    args = _args()
    raw = args.log.read_bytes()
    reports = _reports(raw.decode("utf-8", errors="replace"))
    calibration = reports["calibration"]
    all_validations = reports["validations"]
    excluded_steps = sorted(set(args.exclude_validation_step))
    observed_steps = {int(report["step"]) for report in all_validations}
    unknown_exclusions = sorted(set(excluded_steps) - observed_steps)
    if unknown_exclusions:
        raise RuntimeError(
            f"excluded validation steps are not present in the log: {unknown_exclusions}"
        )
    if excluded_steps and not args.exclusion_reason.strip():
        raise RuntimeError("--exclusion-reason is required when excluding validation steps")
    validations = [
        report for report in all_validations if int(report["step"]) not in excluded_steps
    ]
    if not validations:
        raise RuntimeError("validation-step exclusions removed every held-out report")
    validation = validations[-1]
    pooled_sample_data: dict[str, dict[str, Any]] | None = None
    validation_reports: list[dict[str, Any]] | None = None
    if len(validations) == 1:
        communication = _best_feature(validation, "communication")
        region_joint = _best_feature(validation, "joint")
        network_joint = _best_feature(validation, "network_joint")
        compute = dict(validation["compute"])
    else:
        pooled_sample_data = _pooled_sample_data(validations)
        communication_name = str(
            pooled_sample_data["communication_region"]["feature_model"]
        )
        region_joint_name = str(pooled_sample_data["joint_moe_region"]["feature_model"])
        network_joint_name = str(pooled_sample_data["network_joint"]["feature_model"])
        communication = _pooled_metric(
            pooled_sample_data,
            "communication_region",
            template=_named_feature(
                validation, "communication", communication_name
            ),
        )
        region_joint = _pooled_metric(
            pooled_sample_data,
            "joint_moe_region",
            template=_named_feature(validation, "joint", region_joint_name),
        )
        network_joint = _pooled_metric(
            pooled_sample_data,
            "network_joint",
            template=_named_feature(
                validation, "network_joint", network_joint_name
            ),
        )
        compute = _pooled_metric(pooled_sample_data, "compute")
        validation_reports = []
        for report in validations:
            report_samples = report["sample_data"]
            validation_reports.append(
                {
                    "step": int(report["step"]),
                    "sample_count": int(report["sample_count"]),
                    "compute_fit_sample_count": int(
                        report["compute_fit_sample_count"]
                    ),
                    "compute": _diagnostics(
                        report_samples["compute"]["measured_ms"],
                        report_samples["compute"]["predicted_ms"],
                    ),
                    "communication": _diagnostics(
                        report_samples["communication_region"]["measured_ms"],
                        report_samples["communication_region"]["predicted_ms"],
                    ),
                    "region_joint": _diagnostics(
                        report_samples["joint_moe_region"]["measured_ms"],
                        report_samples["joint_moe_region"]["predicted_ms"],
                    ),
                    "network_joint": _diagnostics(
                        report_samples["network_joint"]["measured_ms"],
                        report_samples["network_joint"]["predicted_ms"],
                    ),
                }
            )
    acceptance_joint = network_joint if args.joint_target == "network" else region_joint
    raw_a2a = (
        _best_feature(validation, "raw_a2a")
        if validation.get("traffic_feature_models", {}).get("raw_a2a")
        else None
    )
    checks = {
        "compute": {
            "passed": _passes(
                compute,
                max_mape=args.compute_max_mape,
                min_r2=args.compute_min_r2,
            ),
            "max_mape_percent": args.compute_max_mape,
            "min_r_squared": args.compute_min_r2,
        },
        "communication": {
            "passed": _passes(
                communication,
                max_mape=args.communication_max_mape,
                min_r2=args.communication_min_r2,
            ),
            "max_mape_percent": args.communication_max_mape,
            "min_r_squared": args.communication_min_r2,
        },
        "joint": {
            "passed": _passes(
                acceptance_joint,
                max_mape=args.joint_max_mape,
                min_r2=args.joint_min_r2,
            ),
            "max_mape_percent": args.joint_max_mape,
            "min_r_squared": args.joint_min_r2,
        },
    }
    accepted = all(row["passed"] for row in checks.values())

    coefficients = calibration["coefficients"]
    artifact: dict[str, Any] = {
        "schema_version": 4 if len(validations) > 1 else 3,
        "artifact_type": "hiermoe_model_compute_calibration",
        "model": args.model,
        "ep_size": args.ep_size,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "maximum_sequence_length": args.maximum_sequence_length,
        "acceptance_joint_target": args.joint_target,
        "source_run": args.source_run,
        "source_log": str(args.log.resolve()),
        "source_log_sha256": hashlib.sha256(raw).hexdigest(),
        "all_validation_steps": [
            int(report["step"]) for report in all_validations
        ],
        "excluded_validation_steps": excluded_steps,
        "validation_exclusion_reason": args.exclusion_reason.strip() or None,
        "coefficients": {
            "compute_constant_ms": float(coefficients["compute_constant_ms"]),
            "compute_ms_per_assignment": float(coefficients["compute_ms_per_assignment"]),
        },
        "calibration": {
            "step": int(calibration["step"]),
            "sample_count": int(calibration["sample_count"]),
            "compute_fit_sample_count": int(calibration["compute_fit_sample_count"]),
            "compute": calibration["compute"],
            "communication": calibration["communication"],
            "joint": calibration["joint"],
            "network_joint": _best_feature(calibration, "network_joint"),
        },
        "held_out_validation": {
            "step": int(validation["step"]),
            "steps": [int(report["step"]) for report in validations],
            "validation_step_count": len(validations),
            "sample_count": sum(int(report["sample_count"]) for report in validations),
            "compute_fit_sample_count": sum(
                int(report["compute_fit_sample_count"]) for report in validations
            ),
            "compute": compute,
            "communication": communication,
            "joint": acceptance_joint,
            "network_joint": network_joint,
            "region_joint": region_joint,
            "raw_a2a": raw_a2a,
        },
        "acceptance_checks": checks,
        "status": "accepted" if accepted else "rejected",
    }
    if validation_reports is not None:
        artifact["held_out_validation_reports"] = validation_reports
    failed = [name for name, row in checks.items() if not row["passed"]]
    artifact["acceptance_reason"] = (
        "All held-out compute, communication-region, and "
        f"{args.joint_target}-joint thresholds passed."
        if accepted
        else f"Held-out thresholds failed: {', '.join(failed)}."
    )
    if calibration.get("sample_data") and validation.get("sample_data"):
        artifact["sample_data"] = {
            "calibration": calibration["sample_data"],
            "held_out_validation": (
                pooled_sample_data
                if pooled_sample_data is not None
                else validation["sample_data"]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
