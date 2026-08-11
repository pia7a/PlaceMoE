#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Summarize one frozen-layout paper run over an inclusive steady-step range."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("profile/runs/pretrain"))
    parser.add_argument("--start-step", type=int, default=3)
    parser.add_argument("--end-step", type=int, default=5)
    parser.add_argument("--expected-ranks", type=int, default=32)
    parser.add_argument("--skip-moe-timing", action="store_true")
    parser.add_argument("--grad-mode")
    parser.add_argument("--layout-report", type=Path)
    parser.add_argument("--layout-path", type=Path)
    parser.add_argument("--layout-bundle", type=Path)
    parser.add_argument("--cost-model", type=Path)
    parser.add_argument("--communication-calibration", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--communication-source-sha256")
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--execution-index", type=int)
    parser.add_argument("--execution-policy")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest record for each step when a run directory was reused."""

    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row.get("step", -1))
        if step >= 0:
            latest[step] = row
    return [latest[step] for step in sorted(latest)]


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]


def _constant_text(rows: list[dict[str, Any]], key: str) -> str | None:
    values = {str(row[key]) for row in rows if key in row and row[key] is not None}
    if len(values) > 1:
        raise RuntimeError(f"{key} changed inside the steady measurement range: {sorted(values)}")
    return next(iter(values), None)


def _layout_seconds(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wall_ms", "build_ms", "layout_build_ms", "elapsed_ms", "total_elapsed_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 1000.0
    timing = payload.get("timing")
    if isinstance(timing, dict):
        for key in ("wall_ms", "build_ms", "elapsed_ms"):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                return float(value) / 1000.0
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        for key in ("planner_total_ms", "layout_build_ms", "elapsed_ms"):
            value = aggregate.get(key)
            if isinstance(value, (int, float)):
                return float(value) / 1000.0
    return None


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _e2e_from_env(
    rows_by_step: dict[int, dict[str, Any]],
    selected_steps: list[int],
) -> list[float]:
    """Read synchronized critical-rank step wall time from environment metrics."""

    values: list[float] = []
    for step in selected_steps:
        row = rows_by_step.get(step)
        if row is None:
            continue
        step_time_s = float(row["step_time_s"])
        if step_time_s > 0:
            values.append(step_time_s * 1000.0)
    return values


def main() -> None:
    args = _args()
    run_root = args.profile_root / args.run_name
    selected = set(range(args.start_step, args.end_step + 1))

    all_env_rows = _latest_by_step(_jsonl(run_root / "env_metrics" / "env_metrics_rank0.jsonl"))
    env_by_step = {int(row["step"]): row for row in all_env_rows}
    env_rows = [env_by_step[step] for step in sorted(selected) if step in env_by_step]
    full_rows = _latest_by_step(
        [
            row
            for row in _jsonl(run_root / "full_timing" / "step_timing_rank0.jsonl")
            if int(row.get("step", -1)) in selected and row.get("section") == "train_step_total"
        ]
    )

    component_by_step: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    rank_assignments_by_step: dict[int, list[float]] = defaultdict(list)
    timing_files = [] if args.skip_moe_timing else sorted((run_root / "moe_timing").glob("moe_timing_rank*.jsonl"))
    expected_steps = args.end_step - args.start_step + 1
    if not args.skip_moe_timing and len(timing_files) != args.expected_ranks:
        raise RuntimeError(
            f"{args.run_name}: expected {args.expected_ranks} MoE timing ranks, found {len(timing_files)}."
        )
    if len(env_rows) != expected_steps:
        raise RuntimeError(
            f"{args.run_name}: expected {expected_steps} steady records; "
            f"found full_timing={len(full_rows)} env_metrics={len(env_rows)}."
        )
    if full_rows:
        if len(full_rows) != expected_steps:
            raise RuntimeError(
                f"{args.run_name}: partial Full Profile data is not usable: "
                f"found {len(full_rows)} of {expected_steps} steady records."
            )
        e2e_step_ms = [float(row["wall_ms"]) for row in full_rows]
        e2e_source = "full_timing"
    else:
        e2e_step_ms = _e2e_from_env(env_by_step, sorted(selected))
        if len(e2e_step_ms) != expected_steps:
            raise RuntimeError(
                f"{args.run_name}: lightweight E2E recovery found "
                f"{len(e2e_step_ms)} of {expected_steps} steady records."
            )
        e2e_source = "env_step_time_s"
    for path in timing_files:
        for row in _latest_by_step(_jsonl(path)):
            step = int(row.get("step", -1))
            if step not in selected:
                continue
            for component in row.get("span_components", []):
                key = (step, str(component.get("direction")), str(component.get("component")))
                component_by_step[key].append(float(component.get("cuda_ms_sum", 0.0)))
            assignment = 0.0
            found = False
            for layer in row.get("span_layers", []):
                if layer.get("direction") != "forward" or layer.get("component") != "expert_compute":
                    continue
                calls = max(1, int(layer.get("calls", 1)))
                assignment += float(layer.get("token_expert_assignments", 0.0)) / calls
                found = True
            if found:
                rank_assignments_by_step[step].append(assignment)

    def critical(direction: str, component: str) -> list[float]:
        return [
            max(component_by_step[(step, direction, component)])
            for step in sorted(selected)
            if component_by_step[(step, direction, component)]
        ]

    forward_a2a = critical("forward", "all_to_all")
    backward_a2a = critical("backward", "all_to_all")
    forward_region = critical("forward", "moe_comm_region")
    expert_compute = [
        sum(values)
        for step in sorted(selected)
        if (
            values := [
                max(component_by_step[(step, direction, "expert_compute")])
                for direction in ("forward", "backward")
                if component_by_step[(step, direction, "expert_compute")]
            ]
        )
    ]
    complete_region = [
        forward_region[index] + backward_a2a[index] for index in range(min(len(forward_region), len(backward_a2a)))
    ]

    load_cv: list[float] = []
    load_max_ratio: list[float] = []
    for step in sorted(selected):
        values = rank_assignments_by_step.get(step, [])
        if len(values) < 2:
            continue
        mean = statistics.mean(values)
        load_cv.append(statistics.pstdev(values) / mean if mean else math.nan)
        load_max_ratio.append(max(values) / mean if mean else math.nan)

    source_sha256 = args.communication_source_sha256
    if source_sha256 is not None and (
        len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256.lower())
    ):
        raise ValueError("communication source SHA-256 is invalid")
    execution_values = (args.repeat_index, args.execution_index, args.execution_policy)
    if any(value is not None for value in execution_values) and any(value is None for value in execution_values):
        raise ValueError("repeat-index, execution-index, and execution-policy must be provided together")

    peak_accelerator_allocated_gib = max(_metric(env_rows, "max_memory_allocated(GB)"), default=None)
    peak_accelerator_reserved_gib = max(_metric(env_rows, "max_memory_reserved(GB)"), default=None)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run_name": args.run_name,
        "steady_steps": [args.start_step, args.end_step],
        "expected_moe_ranks": args.expected_ranks,
        "observed_moe_ranks": len(timing_files),
        "moe_timing_enabled": not args.skip_moe_timing,
        "e2e_step_ms": _mean_std(e2e_step_ms),
        "e2e_source": e2e_source,
        "tokens_per_second_millions": _mean_std(_metric(env_rows, "tokens_per_second(M)")),
        "forward_a2a_ms": _mean_std(forward_a2a),
        "backward_a2a_ms": _mean_std(backward_a2a),
        "moe_communication_region_ms": _mean_std(complete_region),
        "expert_compute_ms": _mean_std(expert_compute),
        "physical_assignment_rank_cv": _mean_std(load_cv),
        "physical_assignment_rank_max_over_mean": _mean_std(load_max_ratio),
        "dedup_ratio_dispatch": _mean_std(_metric(env_rows, "hiermoe/dedup_ratio_dispatch")),
        "hiermoe_ablation_grad_mode": (_constant_text(env_rows, "hiermoe/ablation_grad_mode") or args.grad_mode),
        "peak_accelerator_allocated_gib": peak_accelerator_allocated_gib,
        "peak_accelerator_reserved_gib": peak_accelerator_reserved_gib,
        "offline_layout_seconds": _layout_seconds(args.layout_report),
        "layout_sha256": _sha256(args.layout_path),
        "layout_report_sha256": _sha256(args.layout_report),
        "layout_bundle_sha256": _sha256(args.layout_bundle),
        "cost_model_sha256": _sha256(args.cost_model),
        "communication_calibration_sha256": _sha256(args.communication_calibration),
        "preflight_report_sha256": _sha256(args.preflight_report),
        "communication_source_sha256": source_sha256.lower() if source_sha256 is not None else None,
        "repeat_index": args.repeat_index,
        "execution_index": args.execution_index,
        "execution_policy": args.execution_policy,
    }
    output = args.output or run_root / "paper_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
