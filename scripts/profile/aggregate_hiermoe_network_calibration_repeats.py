#!/usr/bin/env python3
"""Aggregate repeated HierMoE communication calibrations robustly.

Each input is one independent distributed benchmark run.  For every payload
and communication scope, the source artifact already reports the maximum
latency across participating groups/ranks.  This script takes the median of
those run maxima, fits alpha/beta on an explicit training subset, and stores
an independent validation subset in the usual ``fit_points`` field so the
paper plotting tools can consume it without fitting on validation data.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--fit-source-count",
        type=int,
        default=3,
        help="number of leading independent runs used only for fitting",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0.0:
        raise ValueError("Communication fit requires at least two payloads.")
    beta = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    ) / denominator
    return max(0.0, mean_y - beta * mean_x), max(0.0, beta)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(
        ordered[low] * (high - position)
        + ordered[high] * (position - low)
    )


def point_map(points: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(point["bytes"]): float(point["latency_ms_max"])
        for point in points
    }


def robust_points(
    point_sets: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    mapped = [point_map(points) for points in point_sets]
    payloads = sorted(mapped[0])
    if any(sorted(values) != payloads for values in mapped[1:]):
        raise ValueError("Repeated runs do not contain identical payloads.")
    result = []
    for payload in payloads:
        maxima = [values[payload] for values in mapped]
        result.append(
            {
                "bytes": float(payload),
                "latency_ms_mean": statistics.fmean(maxima),
                "latency_ms_median": statistics.median(maxima),
                "latency_ms_p25": percentile(maxima, 0.25),
                "latency_ms_p75": percentile(maxima, 0.75),
                "latency_ms_p90": percentile(maxima, 0.90),
                # Compatibility with the paper plotting bundle: this value is
                # a robust estimate of the critical-path maximum, not a mean.
                "latency_ms_max": statistics.median(maxima),
                "run_maxima_ms": maxima,
                "sample_count": float(len(maxima)),
            }
        )
    return result


def aggregate_scope(
    fit_sets: list[list[dict[str, Any]]],
    validation_sets: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    training_points = robust_points(fit_sets)
    validation_points = robust_points(validation_sets)
    xs = [float(point["bytes"]) for point in training_points]
    ys = [float(point["latency_ms_median"]) for point in training_points]
    alpha, beta = linear_fit(xs, ys)
    return {
        "coefficients": {"alpha": alpha, "beta": beta},
        "training_points": training_points,
        "validation_points": validation_points,
    }


def validate_metadata(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    first = artifacts[0]["metadata"]
    keys = (
        "world_size",
        "ranks_per_node",
        "hierarchy_group_sizes",
        "device_type",
        "backend",
        "dtype",
        "message_bytes_requested",
    )
    for index, artifact in enumerate(artifacts[1:], start=1):
        for key in keys:
            if artifact["metadata"].get(key) != first.get(key):
                raise ValueError(f"run {index} metadata mismatch for {key}")
    return first


def main() -> None:
    args = parse_args()
    if args.fit_source_count < 2:
        raise ValueError("At least two independent fit runs are required.")
    if len(args.source) - args.fit_source_count < 2:
        raise ValueError("At least two independent validation runs are required.")

    artifacts = [load(path) for path in args.source]
    metadata = validate_metadata(artifacts)
    fit = artifacts[: args.fit_source_count]
    validation = artifacts[args.fit_source_count :]

    a2a = aggregate_scope(
        [artifact["fit_points"]["a2a"] for artifact in fit],
        [artifact["fit_points"]["a2a"] for artifact in validation],
    )
    intra = aggregate_scope(
        [artifact["fit_points"]["intra"] for artifact in fit],
        [artifact["fit_points"]["intra"] for artifact in validation],
    )
    inter_count = len(artifacts[0]["fit_points"]["inter"])
    if any(len(artifact["fit_points"]["inter"]) != inter_count for artifact in artifacts):
        raise ValueError("Repeated runs disagree on hierarchy depth.")
    inter = [
        aggregate_scope(
            [artifact["fit_points"]["inter"][index] for artifact in fit],
            [artifact["fit_points"]["inter"][index] for artifact in validation],
        )
        for index in range(inter_count)
    ]

    output = {
        "a2a": a2a["coefficients"],
        "inter": [scope["coefficients"] for scope in inter],
        "intra": intra["coefficients"],
        "source": "aggregate_hiermoe_network_calibration_repeats",
        "metadata": {
            **metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fit_stat": "heldout_median_of_independent_run_maxima",
            "fit_source_count": len(fit),
            "validation_source_count": len(validation),
            "source_artifacts": [str(path) for path in args.source],
        },
        # These are intentionally independent validation points.  Existing
        # consumers predict them with coefficients fitted above.
        "fit_points": {
            "a2a": a2a["validation_points"],
            "inter": [scope["validation_points"] for scope in inter],
            "intra": intra["validation_points"],
        },
        "training_fit_points": {
            "a2a": a2a["training_points"],
            "inter": [scope["training_points"] for scope in inter],
            "intra": intra["training_points"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
