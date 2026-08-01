#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Strictly compare hidden and blocking redundant-gradient synchronization runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TIME_METRICS = (
    "e2e_step_ms",
    "forward_a2a_ms",
    "backward_a2a_ms",
    "moe_communication_region_ms",
    "expert_compute_ms",
)
THROUGHPUT_METRIC = "tokens_per_second_millions"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-summary", type=Path, required=True)
    parser.add_argument("--blocking-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(payload: dict[str, Any], metric: str, path: Path) -> float:
    metric_stats = payload.get(metric)
    if not isinstance(metric_stats, dict):
        raise ValueError(f"{path}: missing {metric}")
    count = metric_stats.get("count")
    mean = metric_stats.get("mean")
    if not isinstance(count, int) or not isinstance(mean, (int, float)):
        raise ValueError(f"{path}: invalid {metric} statistics")
    expected = int(payload["steady_steps"][1]) - int(payload["steady_steps"][0]) + 1
    if count != expected:
        raise ValueError(f"{path}: {metric}.count={count}, expected {expected}")
    return float(mean)


def _validate_pair(
    hidden: dict[str, Any],
    blocking: dict[str, Any],
    hidden_path: Path,
    blocking_path: Path,
) -> None:
    if hidden.get("steady_steps") != [11, 20] or blocking.get("steady_steps") != [11, 20]:
        raise ValueError("both summaries must cover steady steps 11-20")
    if hidden.get("observed_moe_ranks") != blocking.get("observed_moe_ranks"):
        raise ValueError("observed MoE rank counts differ")
    if hidden.get("e2e_source") != blocking.get("e2e_source"):
        raise ValueError("E2E timing sources differ")
    if hidden.get("hiermoe_ablation_grad_mode") != "hidden":
        raise ValueError(f"{hidden_path}: expected grad mode 'hidden'")
    if blocking.get("hiermoe_ablation_grad_mode") != "blocking":
        raise ValueError(f"{blocking_path}: expected grad mode 'blocking'")
    hidden_layout = hidden.get("layout_sha256")
    blocking_layout = blocking.get("layout_sha256")
    if not hidden_layout or not blocking_layout:
        raise ValueError("both summaries must contain layout_sha256")
    if hidden_layout != blocking_layout:
        raise ValueError("hidden and blocking runs used different static layouts")
    for metric in (*TIME_METRICS, THROUGHPUT_METRIC):
        _mean(hidden, metric, hidden_path)
        _mean(blocking, metric, blocking_path)


def _time_row(
    metric: str,
    hidden: dict[str, Any],
    blocking: dict[str, Any],
    hidden_path: Path,
    blocking_path: Path,
) -> dict[str, Any]:
    hidden_mean = _mean(hidden, metric, hidden_path)
    blocking_mean = _mean(blocking, metric, blocking_path)
    return {
        "metric": metric,
        "hidden_mean": hidden_mean,
        "blocking_mean": blocking_mean,
        "blocking_minus_hidden": blocking_mean - hidden_mean,
        "blocking_change_pct": (blocking_mean / hidden_mean - 1.0) * 100.0,
        "blocking_vs_hidden_speedup": hidden_mean / blocking_mean,
    }


def compare(
    hidden: dict[str, Any],
    blocking: dict[str, Any],
    hidden_path: Path,
    blocking_path: Path,
) -> dict[str, Any]:
    _validate_pair(hidden, blocking, hidden_path, blocking_path)
    rows = [_time_row(metric, hidden, blocking, hidden_path, blocking_path) for metric in TIME_METRICS]
    hidden_total_a2a = _mean(hidden, "forward_a2a_ms", hidden_path) + _mean(hidden, "backward_a2a_ms", hidden_path)
    blocking_total_a2a = _mean(blocking, "forward_a2a_ms", blocking_path) + _mean(
        blocking, "backward_a2a_ms", blocking_path
    )
    rows.append(
        {
            "metric": "total_a2a_ms",
            "hidden_mean": hidden_total_a2a,
            "blocking_mean": blocking_total_a2a,
            "blocking_minus_hidden": blocking_total_a2a - hidden_total_a2a,
            "blocking_change_pct": (blocking_total_a2a / hidden_total_a2a - 1.0) * 100.0,
            "blocking_vs_hidden_speedup": hidden_total_a2a / blocking_total_a2a,
        }
    )
    hidden_throughput = _mean(hidden, THROUGHPUT_METRIC, hidden_path)
    blocking_throughput = _mean(blocking, THROUGHPUT_METRIC, blocking_path)
    return {
        "schema_version": 1,
        "pair_validated": True,
        "steady_steps": [11, 20],
        "observed_moe_ranks": hidden["observed_moe_ranks"],
        "layout_sha256": hidden["layout_sha256"],
        "hidden_run": hidden.get("run_name"),
        "blocking_run": blocking.get("run_name"),
        "time_metrics": rows,
        "throughput": {
            "metric": THROUGHPUT_METRIC,
            "hidden_mean": hidden_throughput,
            "blocking_mean": blocking_throughput,
            "blocking_over_hidden": blocking_throughput / hidden_throughput,
            "blocking_change_pct": (blocking_throughput / hidden_throughput - 1.0) * 100.0,
        },
    }


def main() -> None:
    args = _args()
    hidden = _load(args.hidden_summary)
    blocking = _load(args.blocking_summary)
    report = compare(hidden, blocking, args.hidden_summary, args.blocking_summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "metric",
            "hidden_mean",
            "blocking_mean",
            "blocking_minus_hidden",
            "blocking_change_pct",
            "blocking_vs_hidden_speedup",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["time_metrics"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
