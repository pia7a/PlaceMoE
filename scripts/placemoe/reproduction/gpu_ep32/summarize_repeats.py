#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Aggregate independent EP32 run summaries using the median run mean."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


_PROVENANCE_KEYS = (
    "layout_sha256",
    "layout_report_sha256",
    "layout_bundle_sha256",
    "cost_model_sha256",
    "communication_calibration_sha256",
    "preflight_report_sha256",
    "communication_source_sha256",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--grad-protocol", choices=("paper", "blocking"), required=True)
    parser.add_argument("--grad-mode", choices=("blocking", "hidden"), required=True)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _aggregate_metric(payloads: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = []
    for payload in payloads:
        metric = payload.get(key)
        if not isinstance(metric, dict) or not isinstance(metric.get("mean"), (int, float)):
            return None
        values.append(float(metric["mean"]))
    return {
        "count": len(values),
        "mean": statistics.median(values),
        "arithmetic_mean": statistics.mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "run_means": values,
    }


def _constant_value(payloads: list[dict[str, Any]], key: str) -> Any:
    values = [payload.get(key) for payload in payloads]
    canonical = {json.dumps(value, sort_keys=True) for value in values}
    if len(canonical) != 1:
        raise ValueError(f"{key} differs across independent repeats: {values!r}")
    return values[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _args()
    if args.expected_repeats <= 0 or len(args.summary) != args.expected_repeats:
        raise ValueError(f"expected {args.expected_repeats} summaries, received {len(args.summary)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    for path, payload in zip(args.summary, payloads, strict=True):
        if payload.get("steady_steps") != [3, 5]:
            raise ValueError(f"{path} does not cover steady steps 3-5")
        if payload.get("expected_moe_ranks") != 32:
            raise ValueError(f"{path} does not describe an EP32 run")
        if payload.get("observed_moe_ranks") != 0 or payload.get("moe_timing_enabled") is not False:
            raise ValueError(f"{path} contains profiling instrumentation in a formal timing run")
        if payload.get("hiermoe_ablation_grad_mode") != args.grad_mode:
            raise ValueError(f"{path} grad mode does not match {args.grad_mode!r}")
        if payload.get("e2e_source") != "env_step_time_s":
            raise ValueError(f"{path} does not use synchronized env step_time_s timing")

    provenance = {key: _constant_value(payloads, key) for key in _PROVENANCE_KEYS}
    for key in (
        "cost_model_sha256",
        "communication_calibration_sha256",
        "preflight_report_sha256",
        "communication_source_sha256",
    ):
        if provenance[key] is None:
            raise ValueError(f"formal summaries must contain {key}")
    if args.method in {"eplb", "ours"}:
        for key in ("layout_sha256", "layout_report_sha256", "layout_bundle_sha256"):
            if provenance[key] is None:
                raise ValueError(f"{args.method} summaries must contain {key}")

    repeat_indices = [payload.get("repeat_index") for payload in payloads]
    if any(not isinstance(value, int) or value <= 0 for value in repeat_indices):
        raise ValueError(f"invalid repeat indices: {repeat_indices!r}")
    if sorted(repeat_indices) != list(range(1, args.expected_repeats + 1)):
        raise ValueError(f"invalid repeat indices: {repeat_indices!r}")
    execution_indices = [payload.get("execution_index") for payload in payloads]
    if any(not isinstance(value, int) or value <= 0 for value in execution_indices):
        raise ValueError(f"invalid execution indices: {execution_indices!r}")
    if len(set(execution_indices)) != len(execution_indices):
        raise ValueError(f"duplicate execution indices: {execution_indices!r}")
    execution_policy = _constant_value(payloads, "execution_policy")
    if execution_policy != "repeat-major-fixed-order":
        raise ValueError(f"unsupported execution policy: {execution_policy!r}")

    result: dict[str, Any] = {
        "schema_version": 3,
        "status": "accepted",
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "grad_protocol": args.grad_protocol,
        "hiermoe_ablation_grad_mode": args.grad_mode,
        "repeat_count": len(payloads),
        "aggregation": "median_of_independent_run_means",
        "steady_steps": [3, 5],
        "expected_moe_ranks": 32,
        "observed_moe_ranks": 0,
        "moe_timing_enabled": False,
        "e2e_source": "env_step_time_s",
        "run_names": [payload.get("run_name") for payload in payloads],
        "input_summaries": [str(path) for path in args.summary],
        "input_summary_sha256": [_sha256(path) for path in args.summary],
        "provenance": provenance,
        "execution_policy": execution_policy,
        "execution_order": [
            {
                "method": args.method,
                "repeat_index": payload["repeat_index"],
                "execution_index": payload["execution_index"],
                "run_name": payload.get("run_name"),
            }
            for payload in payloads
        ],
    }
    for key in (
        "e2e_step_ms",
        "tokens_per_second_millions",
        "forward_a2a_ms",
        "backward_a2a_ms",
        "moe_communication_region_ms",
        "expert_compute_ms",
        "dedup_ratio_dispatch",
        "physical_assignment_rank_cv",
        "physical_assignment_rank_max_over_mean",
    ):
        aggregate = _aggregate_metric(payloads, key)
        if aggregate is not None:
            result[key] = aggregate

    for key in (
        "peak_accelerator_allocated_gib",
        "peak_accelerator_reserved_gib",
    ):
        values = [payload.get(key) for payload in payloads]
        if all(isinstance(value, (int, float)) for value in values):
            result[key] = max(float(value) for value in values)

    if "e2e_step_ms" not in result:
        raise ValueError("summaries contain no e2e_step_ms means")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
