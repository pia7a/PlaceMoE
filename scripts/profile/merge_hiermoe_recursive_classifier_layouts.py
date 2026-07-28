#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Merge independently generated recursive-classifier layer artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--comparison-report", type=Path)
    parser.add_argument("--comparison-validation-ms", type=float, default=6116.241273880005)
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    layer_payloads: list[dict[str, object]] = []
    report_algorithms: list[str] = []
    rows: list[dict[str, object]] = []
    for layer in range(args.layers):
        layout_path = args.input_dir / f"layer{layer}_layout.json"
        report_path = args.input_dir / f"layer{layer}_report.json"
        layer_payloads.append(json.loads(layout_path.read_text(encoding="utf-8")))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_algorithms.append(str(report.get("algorithm", "")))
        report_rows = report.get("layers")
        if not isinstance(report_rows, list) or len(report_rows) != 1:
            raise ValueError(f"Expected one report row in {report_path}.")
        row = report_rows[0]
        if int(row["layer"]) != layer:
            raise ValueError(f"Unexpected layer index in {report_path}.")
        rows.append(row)

    topology = layer_payloads[0]["topology"]
    source = layer_payloads[0].get("source")
    if not isinstance(source, dict):
        raise ValueError("Layer artifact has no source metadata.")
    algorithm = str(source.get("algorithm", "capacity-general-recursive-classifier-v1"))
    if any(value != report_algorithms[0] for value in report_algorithms):
        raise ValueError("Layer reports use different algorithms.")
    layers: dict[str, object] = {}
    actions: list[dict[str, str]] = []
    for payload in layer_payloads:
        if payload["topology"] != topology:
            raise ValueError("Layer artifacts have different topology metadata.")
        payload_layers = payload.get("layers")
        if not isinstance(payload_layers, dict) or len(payload_layers) != 1:
            raise ValueError("Each layout artifact must contain exactly one layer.")
        layers.update(payload_layers)
        replay = payload.get("replay")
        if not isinstance(replay, dict):
            raise ValueError("A layout artifact has no replay payload.")
        actions_by_step = replay.get("actions_by_step")
        if not isinstance(actions_by_step, dict):
            raise ValueError("A layout artifact has no replay action table.")
        layer_actions = actions_by_step.get("1")
        if not isinstance(layer_actions, list):
            raise ValueError("A layout artifact has no step-1 replay actions.")
        actions.extend(layer_actions)

    layout = {
        "schema_version": 2,
        "source": {
            **source,
            "algorithm": algorithm,
            "merged_layer_artifacts": str(args.input_dir.resolve()),
        },
        "topology": topology,
        "replay": {"actions_by_step": {"1": actions}},
        "layers": layers,
    }

    comparison_rows: dict[int, dict[str, float]] = {}
    if args.comparison_report is not None:
        comparison = json.loads(args.comparison_report.read_text(encoding="utf-8"))
        validation = comparison.get("validation")
        if not isinstance(validation, list):
            raise ValueError("Comparison report does not contain per-layer validation rows.")
        comparison_rows = {int(row["layer"]): row["full"] for row in validation}

    optimize_total = sum(float(row["optimize"]["total_ms"]) for row in rows)
    validation_total = sum(float(row["validation"]["total_ms"]) for row in rows)
    validation_communication = sum(float(row["validation"]["communication_ms"]) for row in rows)
    validation_compute = sum(float(row["validation"]["compute_ms"]) for row in rows)
    comparison_total = float(args.comparison_validation_ms)
    comparison_communication = None
    comparison_compute = None
    better_layers = None
    if comparison_rows:
        comparison_total = sum(float(comparison_rows[layer]["total_ms"]) for layer in range(args.layers))
        comparison_communication = sum(
            float(comparison_rows[layer]["communication_ms"]) for layer in range(args.layers)
        )
        comparison_compute = sum(float(comparison_rows[layer]["compute_ms"]) for layer in range(args.layers))
        better_layers = sum(
            float(rows[layer]["validation"]["total_ms"]) < float(comparison_rows[layer]["total_ms"])
            for layer in range(args.layers)
        )

    report = {
        "schema_version": 1,
        "algorithm": report_algorithms[0] or algorithm,
        "layers": rows,
        "aggregate": {
            "layers": args.layers,
            "optimize_total_ms": optimize_total,
            "validation_total_ms": validation_total,
            "validation_communication_ms": validation_communication,
            "validation_compute_ms": validation_compute,
            "planner_total_ms": sum(float(row["planner_ms"]) for row in rows),
            "planner_mean_ms_per_layer": (sum(float(row["planner_ms"]) for row in rows) / args.layers),
            "comparison_validation_ms": comparison_total,
            "comparison_communication_ms": comparison_communication,
            "comparison_compute_ms": comparison_compute,
            "validation_gain_ms": comparison_total - validation_total,
            "validation_speedup": comparison_total / validation_total,
            "better_layers": better_layers,
            "worse_or_equal_layers": (None if better_layers is None else args.layers - better_layers),
            "e2e_eligible": validation_total <= comparison_total,
        },
    }
    for path, payload in ((args.output_layout, layout), (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
