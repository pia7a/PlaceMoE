#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Export the offline HierMoE scorer from an Ours cost-model verification run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.placemoe.reproduction.gpu_ep32.cost_components import (
    fit_compute_curve,
    load_communication_calibration,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--ours-log",
        type=Path,
        required=True,
        help="Rank-0 log from E2E_VARIANT=cost_model_verify with startup fitting enabled.",
    )
    parser.add_argument(
        "--r2-summary",
        type=Path,
        help="Optional Fixed-R2 summary used only for held-out eligibility reporting, never for fitting.",
    )
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument(
        "--communication-calibration",
        type=Path,
        required=True,
        help="One shared EP32 topology calibration used for inter/intra coefficients.",
    )
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--communication-source-sha256", required=True)
    parser.add_argument("--cost-scope-sha256")
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--data-source-name", required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--moe-impl", required=True)
    parser.add_argument("--freeze-vit", choices=("true", "false"), required=True)
    parser.add_argument("--phase-timing-root", type=Path, required=True)
    parser.add_argument("--phase-step", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marked_payload(path: Path, marker: str, *, python_literal: bool) -> dict[str, Any]:
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker in line:
            encoded = line.split(marker, 1)[1]
            matches.append(ast.literal_eval(encoded) if python_literal else json.loads(encoded))
    if not matches:
        raise RuntimeError(f"No {marker!r} payload found in {path}.")
    return dict(matches[-1])


def _require_finite(name: str, value: Any, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        qualifier = "positive " if positive else "non-negative "
        raise RuntimeError(f"{name} must be a finite {qualifier}number, got {value!r}.")
    return number


def _route_alignment(
    startup: dict[str, Any],
    calibration: dict[str, Any],
    validation: dict[str, Any],
    *,
    level_coefficients: tuple[float, ...],
) -> tuple[tuple[float, ...], float, dict[str, Any]]:
    if len(level_coefficients) not in {2, 3}:
        raise RuntimeError(f"Unsupported communication level count: {len(level_coefficients)}")
    for index, value in enumerate(level_coefficients):
        _require_finite(f"shared.level_ms_per_byte[{index}]", value, positive=True)

    def samples(
        report: dict[str, Any],
        phase: str,
    ) -> tuple[list[list[float]], list[float], list[float], list[list[float]]]:
        payload = report.get("offline_scorer_samples")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Ours {phase} report is missing offline_scorer_samples.")
        stage_byte_names = tuple(
            f"stage{index}_payload_endpoint_bytes" for index in range(1, len(level_coefficients) + 1)
        )
        stage_actual_names = tuple(f"actual_stage{index}_a2a_ms" for index in range(1, len(level_coefficients) + 1))
        names = (
            *stage_byte_names,
            "peak_assignments",
            "actual_communication_ms",
            *stage_actual_names,
        )
        missing = [name for name in names if not isinstance(payload.get(name), list)]
        if missing:
            raise RuntimeError(f"Ours {phase} report has invalid sample arrays: {missing}.")
        counts = {len(payload[name]) for name in names}
        if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
            raise RuntimeError(f"Ours {phase} report sample arrays are empty or misaligned: {counts}.")

        def values(name: str, *, positive: bool = True) -> list[float]:
            return [
                _require_finite(f"{phase}.{name}[{index}]", value, positive=positive)
                for index, value in enumerate(payload[name])
            ]

        return (
            [values(name) for name in stage_byte_names],
            values("peak_assignments"),
            values("actual_communication_ms"),
            [values(name) for name in stage_actual_names],
        )

    def through_origin(feature: list[float], target: list[float], name: str) -> tuple[float, float, float]:
        numerator = sum(lhs * rhs for lhs, rhs in zip(feature, target, strict=True))
        denominator = sum(value**2 for value in feature)
        if denominator <= 0.0:
            raise RuntimeError(f"{name} has no positive feature energy.")
        return max(0.0, numerator / denominator), numerator, denominator

    def cost_diagnostics(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
        residuals = [truth - estimate for truth, estimate in zip(actual, predicted, strict=True)]
        actual_mean = statistics.mean(actual)
        total_variance = sum((value - actual_mean) ** 2 for value in actual)
        squared_error = sum(value**2 for value in residuals)
        return {
            "count": len(actual),
            "actual_mean_ms": actual_mean,
            "prediction_mean_ms": statistics.mean(predicted),
            "residual_mean_ms": statistics.mean(residuals),
            "rmse_ms": math.sqrt(squared_error / len(residuals)),
            "mape_percent": statistics.mean(
                abs(residual) / truth for residual, truth in zip(residuals, actual, strict=True)
            )
            * 100.0,
            "r_squared": 1.0 - squared_error / total_variance if total_variance > 0.0 else None,
        }

    calibration_rows = samples(calibration, "calibration")
    validation_rows = samples(validation, "validation")
    calibration_stage_bytes, calibration_assignments, calibration_actual, calibration_stage_actual = calibration_rows
    workload_levels = []
    stage_fit_rows = []
    for index, (features, targets, shared) in enumerate(
        zip(calibration_stage_bytes, calibration_stage_actual, level_coefficients, strict=True),
        start=1,
    ):
        coefficient, numerator, denominator = through_origin(features, targets, f"stage {index}")
        workload_levels.append(coefficient)
        stage_fit_rows.append(
            {
                "stage": index,
                "shared_ms_per_byte": shared,
                "workload_reference_ms_per_byte": coefficient,
                "numerator": numerator,
                "denominator": denominator,
                "shared_scale": coefficient / shared,
            }
        )

    calibration_links = [
        sum(coefficient * stage[index] for index, coefficient in enumerate(workload_levels))
        for stage in zip(*calibration_stage_bytes, strict=True)
    ]
    route_targets = [actual - link for actual, link in zip(calibration_actual, calibration_links, strict=True)]
    route_samples = [
        target / assignments for assignments, target in zip(calibration_assignments, route_targets, strict=True)
    ]
    route_median = statistics.median(route_samples)
    route_mad = statistics.median(abs(value - route_median) for value in route_samples)
    route_threshold = max(6.0 * route_mad, abs(route_median) * 1.0e-6, 1.0e-12)
    route_inliers = [
        index for index, value in enumerate(route_samples) if abs(value - route_median) <= route_threshold
    ]
    if len(route_inliers) < max(1, (len(route_samples) + 1) // 2):
        raise RuntimeError("Route calibration retained fewer than half of its samples after robust outlier filtering.")
    route_numerator = sum(calibration_assignments[index] * route_targets[index] for index in route_inliers)
    route_denominator = sum(calibration_assignments[index] ** 2 for index in route_inliers)
    if route_denominator <= 0.0:
        raise RuntimeError("Route calibration has no positive assignment energy.")
    route = max(0.0, route_numerator / route_denominator)

    def diagnostics(
        rows: tuple[list[list[float]], list[float], list[float], list[list[float]]],
    ) -> dict[str, Any]:
        stage_bytes, assignments, actual, stage_actual = rows
        stage_predicted = [
            [coefficient * value for value in features]
            for coefficient, features in zip(workload_levels, stage_bytes, strict=True)
        ]
        link_predicted = [sum(values) for values in zip(*stage_predicted, strict=True)]
        predicted = [link + route * count for link, count in zip(link_predicted, assignments, strict=True)]
        return {
            **cost_diagnostics(actual, predicted),
            "link_prediction_mean_ms": statistics.mean(link_predicted),
            "route_prediction_mean_ms": statistics.mean(route * value for value in assignments),
            "stages": {
                f"stage{index}": cost_diagnostics(targets, estimates)
                for index, (targets, estimates) in enumerate(
                    zip(stage_actual, stage_predicted, strict=True),
                    start=1,
                )
            },
        }

    startup_reference = {
        "inter_ms_per_byte": _require_finite("startup.inter_ms_per_byte", startup["inter"][-1]["beta"], positive=True),
        "intra_ms_per_byte": _require_finite("startup.intra_ms_per_byte", startup["intra"]["beta"], positive=True),
        "used_for_offline_scorer": False,
    }
    return (
        tuple(workload_levels),
        route,
        {
            "fit": diagnostics(calibration_rows),
            "validation": diagnostics(validation_rows),
            "stage_link_fit": {
                "levels": stage_fit_rows,
                "shared_calibration_used_as_topology_anchor": True,
                "production_dispatch_fit_used_for_offline_scorer": True,
            },
            "route_nonnegative_least_squares_numerator": route_numerator,
            "route_nonnegative_least_squares_denominator": route_denominator,
            "route_nonnegative_boundary": float(route == 0.0),
            "route_robust_filter": {
                "kind": "median_mad_6x_then_nonnegative_least_squares",
                "sample_count": len(route_samples),
                "inlier_count": len(route_inliers),
                "excluded_indices": [index for index in range(len(route_samples)) if index not in route_inliers],
                "median_ms_per_assignment": route_median,
                "mad_ms_per_assignment": route_mad,
                "threshold_ms_per_assignment": route_threshold,
            },
            "startup_probe_reference": startup_reference,
        },
    )


def _phase_multipliers(
    root: Path,
    *,
    step: int,
    layers: int,
    ep_size: int,
) -> tuple[float, float, dict[str, Any]]:
    timing_root = root / "moe_timing"
    paths = sorted(timing_root.glob("moe_timing_rank*.jsonl"))
    if len(paths) != ep_size:
        raise RuntimeError(f"Expected {ep_size} Ours timing files under {timing_root}, found {len(paths)}.")
    values: dict[tuple[int, str, str, int, str, int, int], float] = {}
    for path in paths:
        step_rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row.get("step", -1)) != step:
                continue
            step_rows += 1
            rank = int(row["rank"])
            for item in row.get("span_invocations", []):
                component = str(item["component"])
                if component not in {"all_to_all", "expert_compute"}:
                    continue
                if int(item.get("calls", 0)) != 1:
                    raise RuntimeError(f"Expected one timing span per invocation in {path}, got {item!r}.")
                call_index = item.get("call_index")
                if call_index is None:
                    raise RuntimeError(f"Missing call_index in Ours phase timing row: {item!r}.")
                key = (
                    int(item["layer"]),
                    str(item["direction"]),
                    component,
                    int(call_index),
                    str(item["section"]),
                    int(item["invocation"]),
                    rank,
                )
                if key in values:
                    raise RuntimeError(f"Duplicate Ours phase timing invocation: {key!r}.")
                values[key] = _require_finite("cuda_ms_sum", item["cuda_ms_sum"], positive=True)
        if step_rows != 1:
            raise RuntimeError(f"Expected one step={step} timing row in {path}, found {step_rows}.")

    def component_ratios(component: str) -> tuple[list[float], list[int], list[int]]:
        ratios = []
        forward_counts = []
        backward_counts = []
        for layer in range(layers):
            direction_totals: dict[str, dict[int, float]] = {}
            for direction in ("forward", "backward"):
                invocation_keys = sorted(
                    {
                        (call_index, section, invocation)
                        for (
                            row_layer,
                            row_direction,
                            row_component,
                            call_index,
                            section,
                            invocation,
                            _rank,
                        ) in values
                        if row_layer == layer and row_direction == direction and row_component == component
                    }
                )
                if not invocation_keys:
                    raise RuntimeError(
                        f"Incomplete {component} {direction} phase timing for step={step}, layer={layer}."
                    )
                totals: dict[int, float] = defaultdict(float)
                for call_index, section, invocation in invocation_keys:
                    rank_values = [
                        values.get((layer, direction, component, call_index, section, invocation, rank))
                        for rank in range(ep_size)
                    ]
                    if any(value is None for value in rank_values):
                        raise RuntimeError(
                            "Incomplete cross-rank timing for "
                            f"step={step}, layer={layer}, direction={direction}, component={component}, "
                            f"call={call_index}, section={section}, invocation={invocation}."
                        )
                    totals[call_index] += max(float(value) for value in rank_values if value is not None)
                direction_totals[direction] = totals
            forward = direction_totals["forward"]
            backward = direction_totals["backward"]
            single_forward_critical = statistics.mean(forward.values())
            full_step_critical = sum(forward.values()) + sum(backward.values())
            ratios.append(full_step_critical / single_forward_critical)
            forward_counts.append(len(forward))
            backward_counts.append(len(backward))
        return ratios, forward_counts, backward_counts

    communication, communication_forward_calls, communication_backward_calls = component_ratios("all_to_all")
    compute, compute_forward_calls, compute_backward_calls = component_ratios("expert_compute")

    def summary(rows: list[float]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            "mean": statistics.mean(rows),
            "median": statistics.median(rows),
            "stdev": statistics.stdev(rows) if len(rows) > 1 else 0.0,
            "min": min(rows),
            "max": max(rows),
        }

    manifest = hashlib.sha256()
    for path in paths:
        manifest.update(path.name.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(_sha256(path).encode("ascii"))
        manifest.update(b"\n")
    diagnostics = {
        "step": step,
        "communication": {
            **summary(communication),
            "forward_calls_per_layer": sorted(set(communication_forward_calls)),
            "backward_calls_per_layer": sorted(set(communication_backward_calls)),
        },
        "compute": {
            **summary(compute),
            "forward_calls_per_layer": sorted(set(compute_forward_calls)),
            "backward_calls_per_layer": sorted(set(compute_backward_calls)),
        },
        "timing_manifest": {
            "root": str(root),
            "files": len(paths),
            "sha256": manifest.hexdigest(),
        },
    }
    return statistics.median(communication), statistics.median(compute), diagnostics


def main() -> None:
    args = _args()
    expected_log_name = f"{args.run_name}_rank0.host.log"
    if args.ours_log.name != expected_log_name:
        raise RuntimeError(f"Ours log identity mismatch: expected {expected_log_name!r}, got {args.ours_log.name!r}.")
    if args.phase_timing_root.name != args.run_name:
        raise RuntimeError(
            "Ours timing identity mismatch: "
            f"expected root named {args.run_name!r}, got {args.phase_timing_root.name!r}."
        )

    startup = _marked_payload(
        args.ours_log,
        "HierMoE startup performance fitting completed: ",
        python_literal=True,
    )
    calibration = _marked_payload(
        args.ours_log,
        "HierMoE cost model calibration report: ",
        python_literal=False,
    )
    validation = _marked_payload(
        args.ours_log,
        "HierMoE cost model validation report: ",
        python_literal=False,
    )
    if int(validation["step"]) != int(calibration["step"]) + 1:
        raise RuntimeError("Ours cost-model validation must immediately follow its calibration step.")
    if args.phase_step != int(validation["step"]):
        raise RuntimeError(
            "Ours phase timing must use the validation step: "
            f"phase_step={args.phase_step}, validation_step={validation['step']}."
        )

    _inter, _intra, communication_provenance = load_communication_calibration(
        args.communication_calibration,
        ep_size=args.ep_size,
        ranks_per_node=args.ranks_per_node,
        hidden_size=args.hidden_size,
        bytes_per_element=args.bytes_per_element,
        preflight_report=args.preflight_report,
        communication_source_sha256=args.communication_source_sha256,
    )
    shared_levels = tuple(float(value) for value in communication_provenance["level_ms_per_byte"])
    level_coefficients, route, route_diagnostics = _route_alignment(
        startup, calibration, validation, level_coefficients=shared_levels
    )
    communication_validation_mape = float(route_diagnostics["validation"]["mape_percent"])
    if not math.isfinite(communication_validation_mape) or communication_validation_mape > 10.0:
        raise RuntimeError(
            "Held-out production communication/route alignment error exceeds 10%: "
            f"MAPE={communication_validation_mape:.6f}%"
        )
    compute_curve, compute_curve_diagnostics = fit_compute_curve(calibration, validation)
    compute_curve_validation_mape = float(compute_curve_diagnostics["validation"]["mape_percent"])
    if not math.isfinite(compute_curve_validation_mape) or compute_curve_validation_mape > 5.0:
        raise RuntimeError(
            f"Held-out per-local-expert compute curve error exceeds 5%: MAPE={compute_curve_validation_mape:.6f}%"
        )
    communication_multiplier, compute_multiplier, phase_diagnostics = _phase_multipliers(
        args.phase_timing_root,
        step=args.phase_step,
        layers=args.layers,
        ep_size=args.ep_size,
    )
    compute = _require_finite(
        "compute_ms_per_assignment",
        validation["coefficients"]["compute_ms_per_assignment"],
    )

    comparison_ms = None
    if args.r2_summary is not None:
        r2_summary = json.loads(args.r2_summary.read_text(encoding="utf-8"))
        comparison_ms = _require_finite(
            "comparison_validation_ms",
            float(r2_summary["moe_communication_region_ms"]["mean"]) + float(r2_summary["expert_compute_ms"]["mean"]),
            positive=True,
        )

    payload = {
        **startup,
        "schema_version": 4,
        "source": "gpu32-a6000-composed-cost-model",
        "topology": {
            "accelerator": "NVIDIA RTX A6000",
            "nodes": args.ep_size // args.ranks_per_node,
            "gpus_per_node": args.ranks_per_node,
            "ep_size": args.ep_size,
            "ranks_per_node": args.ranks_per_node,
            "hierarchy_group_sizes": [2, args.ranks_per_node, args.ep_size],
            "num_experts": args.num_experts,
            "slots_per_rank": args.slots_per_rank,
            "hidden_size": args.hidden_size,
            "bytes_per_element": args.bytes_per_element,
            "layers": args.layers,
        },
        "model_scope": {
            "model_id": args.model_id,
            "checkpoint_sha256": args.checkpoint_sha256,
            "cost_scope_sha256": args.cost_scope_sha256 or args.checkpoint_sha256,
            "moe_impl": args.moe_impl,
            "hidden_size": args.hidden_size,
            "bytes_per_element": args.bytes_per_element,
        },
        "calibration_workload": {
            "dataset_id": args.dataset_id,
            "dataset_sha256": args.dataset_sha256,
            "data_source_name": args.data_source_name,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "max_seq_len": args.max_seq_len,
            "freeze_vit": args.freeze_vit == "true",
        },
        "offline_scorer": {
            "level_ms_per_byte": list(level_coefficients),
            "inter_ms_per_byte": level_coefficients[0],
            "mid_ms_per_byte": level_coefficients[1] if len(level_coefficients) == 3 else level_coefficients[0],
            "intra_ms_per_byte": level_coefficients[-1],
            "route_ms_per_assignment": route,
            "compute_curve": compute_curve,
            "communication_phase_multiplier": communication_multiplier,
            "compute_ms_per_assignment": compute,
            "compute_phase_multiplier": compute_multiplier,
            **({"comparison_validation_ms": comparison_ms} if comparison_ms is not None else {}),
        },
        "ours_cost_model_verify": {
            "calibration_step": int(calibration["step"]),
            "validation_step": int(validation["step"]),
            "route_alignment": route_diagnostics,
            "phase_alignment": phase_diagnostics,
            "compute_curve_alignment": compute_curve_diagnostics,
            "compute": validation["compute"],
            "joint": validation["joint"],
            "compute_constant_ms_not_used": _require_finite(
                "compute_constant_ms",
                validation["coefficients"]["compute_constant_ms"],
            ),
        },
        "provenance": {
            "fit_inputs": ["ep32_communication_calibration", "model_cost_verify"],
            "communication_calibration": communication_provenance,
            "model_checkpoint_sha256": args.checkpoint_sha256,
            "model_calibration_dataset": args.dataset_id,
            "execution_layout": "fixed_r2_mirrored_without_fixed_pipeline_for_primitive_excitation",
            "external_method_timing_used": False,
            "excluded_fit_timing": ["baseline", "fixed_r2_e2e", "eplb", "ours_e2e"],
            "files": {str(path): _sha256(path) for path in (args.ours_log, args.r2_summary) if path is not None},
            "r2_usage": ("held-out eligibility comparison only" if args.r2_summary is not None else "not provided"),
            "gradient_cost_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(json.dumps({"output": str(args.output), "offline_scorer": payload["offline_scorer"]}, indent=2))


if __name__ == "__main__":
    main()
