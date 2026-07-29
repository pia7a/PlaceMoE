#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Summarize one frozen-layout paper run over an inclusive steady-step range."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--start-step", type=int, default=11)
    parser.add_argument("--end-step", type=int, default=20)
    parser.add_argument("--expected-ranks", type=int, default=32)
    parser.add_argument("--layout-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def main() -> None:
    args = _args()
    run_root = args.profile_root / args.run_name
    selected = set(range(args.start_step, args.end_step + 1))

    env_rows = [
        row
        for row in _jsonl(run_root / "env_metrics" / "env_metrics_rank0.jsonl")
        if int(row.get("step", -1)) in selected
    ]
    full_rows = [
        row
        for row in _jsonl(run_root / "full_timing" / "step_timing_rank0.jsonl")
        if int(row.get("step", -1)) in selected and row.get("section") == "train_step_total"
    ]

    component_by_step: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    rank_assignments_by_step: dict[int, list[float]] = defaultdict(list)
    timing_files = sorted((run_root / "moe_timing").glob("moe_timing_rank*.jsonl"))
    expected_steps = args.end_step - args.start_step + 1
    if len(timing_files) != args.expected_ranks:
        raise RuntimeError(
            f"{args.run_name}: expected {args.expected_ranks} MoE timing ranks, found {len(timing_files)}."
        )
    if len(full_rows) != expected_steps or len(env_rows) != expected_steps:
        raise RuntimeError(
            f"{args.run_name}: expected {expected_steps} steady records; "
            f"found full_timing={len(full_rows)} env_metrics={len(env_rows)}."
        )
    for path in timing_files:
        for row in _jsonl(path):
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
        forward_region[index] + backward_a2a[index]
        for index in range(min(len(forward_region), len(backward_a2a)))
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

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_name": args.run_name,
        "steady_steps": [args.start_step, args.end_step],
        "observed_moe_ranks": len(timing_files),
        "e2e_step_ms": _mean_std([float(row["wall_ms"]) for row in full_rows]),
        "tokens_per_second_millions": _mean_std(_metric(env_rows, "tokens_per_second(M)")),
        "forward_a2a_ms": _mean_std(forward_a2a),
        "backward_a2a_ms": _mean_std(backward_a2a),
        "moe_communication_region_ms": _mean_std(complete_region),
        "expert_compute_ms": _mean_std(expert_compute),
        "physical_assignment_rank_cv": _mean_std(load_cv),
        "physical_assignment_rank_max_over_mean": _mean_std(load_max_ratio),
        "dedup_ratio_dispatch": _mean_std(_metric(env_rows, "hiermoe/dedup_ratio_dispatch")),
        "peak_npu_allocated_gib": max(_metric(env_rows, "max_memory_allocated(GB)"), default=None),
        "peak_npu_reserved_gib": max(_metric(env_rows, "max_memory_reserved(GB)"), default=None),
        "offline_layout_seconds": _layout_seconds(args.layout_report),
    }
    output = args.output or run_root / "paper_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
