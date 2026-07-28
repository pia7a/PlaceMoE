#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Select a planned HierMoE layout using measured communication plus calibrated compute cost."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", default="mirrored_r2")
    parser.add_argument("--candidate", default="external")
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--forward-compute-ms-per-assignment", type=float, required=True)
    parser.add_argument(
        "--minimum-gain-ms",
        type=float,
        default=0.0,
        help="Minimum predicted full-step gain required before measurement uncertainty is applied.",
    )
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="Robust aggregate communication uncertainty multiplier.",
    )
    parser.add_argument(
        "--guard-mode",
        choices=("worst_case", "mad"),
        default="worst_case",
        help=(
            "Acceptance guard. 'worst_case' requires positive full-step gain in every paired measured "
            "iteration; 'mad' requires the median prediction to exceed a scaled MAD margin."
        ),
    )
    parser.add_argument("--candidate-layout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sum_feature(rows: list[dict[str, object]], name: str) -> float:
    return sum(float(row[name]) for row in rows)


def _communication_summary(
    payload: dict[str, object],
    *,
    variant: str,
) -> tuple[float, list[float]]:
    aggregate = payload.get("aggregate_timings")
    if isinstance(aggregate, dict):
        variant_aggregate = aggregate.get(variant)
        if isinstance(variant_aggregate, dict):
            region = variant_aggregate.get("communication_region_ms")
            if isinstance(region, dict):
                totals = region.get("iteration_totals_ms")
                if isinstance(totals, list) and totals:
                    values = [float(value) for value in totals]
                    return float(statistics.median(values)), values

    samples = payload["samples"]
    if not isinstance(samples, dict) or not isinstance(samples.get(variant), list):
        raise ValueError(f"Traffic benchmark has no variant {variant!r}.")
    value = _sum_feature(samples[variant], "communication_region_ms")
    return value, [value]


def _paired_communication_uncertainty(
    baseline_iterations: list[float],
    candidate_iterations: list[float],
    *,
    multiplier: float,
) -> tuple[float, list[float]]:
    if len(baseline_iterations) != len(candidate_iterations) or len(baseline_iterations) < 2:
        return 0.0, []
    deltas = [candidate - baseline for baseline, candidate in zip(baseline_iterations, candidate_iterations)]
    median = statistics.median(deltas)
    mad = statistics.median(abs(value - median) for value in deltas)
    # 1.4826 converts MAD to a Gaussian-equivalent standard deviation.
    return max(0.0, float(multiplier) * 1.4826 * mad), deltas


def main() -> None:
    args = _parse_args()
    if (
        min(
            args.communication_phase_multiplier,
            args.compute_phase_multiplier,
            args.forward_compute_ms_per_assignment,
            args.minimum_gain_ms,
            args.mad_multiplier,
        )
        < 0.0
    ):
        raise ValueError("Cost multipliers, slope, gain margin, and MAD multiplier must be non-negative.")

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"Traffic benchmark has no samples: {args.benchmark}.")
    for variant in (args.baseline, args.candidate):
        if not isinstance(samples.get(variant), list):
            raise ValueError(f"Traffic benchmark has no variant {variant!r}.")

    baseline_communication, baseline_iterations = _communication_summary(payload, variant=args.baseline)
    candidate_communication, candidate_iterations = _communication_summary(payload, variant=args.candidate)
    baseline_assignments = _sum_feature(samples[args.baseline], "peak_assignments")
    candidate_assignments = _sum_feature(samples[args.candidate], "peak_assignments")

    communication_delta = args.communication_phase_multiplier * (candidate_communication - baseline_communication)
    compute_delta = (
        args.compute_phase_multiplier
        * args.forward_compute_ms_per_assignment
        * (candidate_assignments - baseline_assignments)
    )
    total_delta = communication_delta + compute_delta
    predicted_gain = -total_delta
    communication_uncertainty, paired_deltas = _paired_communication_uncertainty(
        baseline_iterations,
        candidate_iterations,
        multiplier=args.mad_multiplier,
    )
    full_step_uncertainty = args.communication_phase_multiplier * communication_uncertainty
    paired_full_step_deltas = [args.communication_phase_multiplier * delta + compute_delta for delta in paired_deltas]
    worst_case_gain = -max(paired_full_step_deltas) if paired_full_step_deltas else predicted_gain
    if args.guard_mode == "mad":
        required_gain = max(float(args.minimum_gain_ms), full_step_uncertainty)
        accepted = predicted_gain > required_gain
    else:
        required_gain = float(args.minimum_gain_ms)
        accepted = predicted_gain > required_gain and worst_case_gain > required_gain

    result = {
        "schema_version": 1,
        "benchmark": str(args.benchmark),
        "baseline_variant": args.baseline,
        "candidate_variant": args.candidate,
        "candidate_layout": None if args.candidate_layout is None else str(args.candidate_layout),
        "decision": args.candidate if accepted else "none",
        "accepted": accepted,
        "cost_model": {
            "formula": ("k_comm_phase*T_forward_dispatch_combine + k_compute_phase*k_compute_assignment*A_max"),
            "communication_phase_multiplier": args.communication_phase_multiplier,
            "compute_phase_multiplier": args.compute_phase_multiplier,
            "forward_compute_ms_per_assignment": args.forward_compute_ms_per_assignment,
        },
        "baseline": {
            "variant": args.baseline,
            "forward_communication_ms": baseline_communication,
            "peak_assignments_sum": baseline_assignments,
        },
        "candidate": {
            "variant": args.candidate,
            "forward_communication_ms": candidate_communication,
            "peak_assignments_sum": candidate_assignments,
        },
        "delta_candidate_minus_baseline_ms": {
            "communication_full_step": communication_delta,
            "compute_full_step": compute_delta,
            "total": total_delta,
        },
        "predicted_gain_ms": predicted_gain,
        "worst_case_measured_gain_ms": worst_case_gain,
        "required_gain_ms": required_gain,
        "communication_uncertainty": {
            "paired_iteration_deltas_ms": paired_deltas,
            "paired_full_step_deltas_ms": paired_full_step_deltas,
            "robust_forward_margin_ms": communication_uncertainty,
            "full_step_margin_ms": full_step_uncertainty,
            "mad_multiplier": args.mad_multiplier,
            "guard_mode": args.guard_mode,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
