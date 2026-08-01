#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Summarize Qwen3-VL EP32 redundancy-ratio experiments for a paper table."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_FIGURES_DIR = Path(
    os.environ.get("HIERMOE_PAPER_FIGURES_DIR", "/home/tzq/infocom2027-paper/figures")
)
BASELINE_FILE = (
    "paper32_qwen3vl30b_sharegpt4v_baseline_full_"
    "20260729_anchor10_r2_summary.json"
)
RHO_CASES = (
    (0.00, 0, "paper32_qwen3vl30b_sharegpt4v_hyper_rho000_full_20260729_anchor10_r2_summary.json"),
    (0.25, 1, "paper32_qwen3vl30b_sharegpt4v_hyper_rho025_full_20260729_anchor10_r2_summary.json"),
    (0.50, 2, "paper32_qwen3vl30b_sharegpt4v_hyper_rho050_full_20260729_anchor10_r2_summary.json"),
    (0.75, 3, "paper32_qwen3vl30b_sharegpt4v_hyper_rho075_full_20260729_anchor10_r2_summary.json"),
    (1.00, 4, "paper32_qwen3vl30b_sharegpt4v_hyper_rho100_full_20260729_anchor10_r2_summary.json"),
)
REQUIRED_METRICS = (
    "e2e_step_ms",
    "forward_a2a_ms",
    "backward_a2a_ms",
    "moe_communication_region_ms",
    "expert_compute_ms",
)


def default_results_dir() -> Path:
    bundled = Path(__file__).resolve().parent / "source_data"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Directory containing the source summary JSON files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=(
            DEFAULT_FIGURES_DIR
            / "paper32_qwen3vl30b_sharegpt4v_hyperparameter_table"
        ),
        help="Output path without an extension.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("steady_steps") != [11, 20]:
        raise ValueError(f"{path}: expected steady_steps=[11, 20]")
    if payload.get("observed_moe_ranks") != 32:
        raise ValueError(f"{path}: expected observed_moe_ranks=32")
    for metric in REQUIRED_METRICS:
        value = payload.get(metric)
        if not isinstance(value, dict) or value.get("count") != 10:
            raise ValueError(f"{path}: expected 10 samples for {metric}")
        if not isinstance(value.get("mean"), (int, float)) or value["mean"] <= 0:
            raise ValueError(f"{path}: invalid {metric}.mean")
        if not isinstance(value.get("std"), (int, float)) or value["std"] < 0:
            raise ValueError(f"{path}: invalid {metric}.std")
    return payload


def metric(payload: dict[str, Any], name: str, field: str = "mean") -> float:
    return float(payload[name][field])


def optional_metric(payload: dict[str, Any], name: str, field: str = "mean") -> float | None:
    value = payload.get(name)
    if not isinstance(value, dict) or not isinstance(value.get(field), (int, float)):
        return None
    return float(value[field])


def make_row(
    *,
    payload: dict[str, Any],
    summary_path: Path,
    label: str,
    rho: float | None,
    redundant_slots: int,
    baseline_e2e_ms: float,
    baseline_a2a_ms: float,
) -> dict[str, Any]:
    e2e_ms = metric(payload, "e2e_step_ms")
    forward_a2a_ms = metric(payload, "forward_a2a_ms")
    backward_a2a_ms = metric(payload, "backward_a2a_ms")
    total_a2a_ms = forward_a2a_ms + backward_a2a_ms
    return {
        "label": label,
        "rho": rho,
        "redundant_slots_per_rank": redundant_slots,
        "total_expert_capacity_ratio": 1.0 + (rho or 0.0),
        "e2e_step_ms": e2e_ms,
        "e2e_step_std_ms": metric(payload, "e2e_step_ms", "std"),
        "e2e_speedup": baseline_e2e_ms / e2e_ms,
        "forward_a2a_ms": forward_a2a_ms,
        "forward_a2a_std_ms": metric(payload, "forward_a2a_ms", "std"),
        "backward_a2a_ms": backward_a2a_ms,
        "backward_a2a_std_ms": metric(payload, "backward_a2a_ms", "std"),
        "total_a2a_ms": total_a2a_ms,
        "a2a_speedup": baseline_a2a_ms / total_a2a_ms,
        "moe_communication_region_ms": metric(payload, "moe_communication_region_ms"),
        "moe_communication_region_std_ms": metric(
            payload,
            "moe_communication_region_ms",
            "std",
        ),
        "expert_compute_ms": metric(payload, "expert_compute_ms"),
        "expert_compute_std_ms": metric(payload, "expert_compute_ms", "std"),
        "dedup_ratio": optional_metric(payload, "dedup_ratio_dispatch"),
        "dedup_ratio_std": optional_metric(payload, "dedup_ratio_dispatch", "std"),
        "peak_npu_allocated_gib": payload.get("peak_npu_allocated_gib"),
        "peak_npu_reserved_gib": payload.get("peak_npu_reserved_gib"),
        "gradient_sync_mode": payload.get("hiermoe_ablation_grad_mode"),
        "run_name": payload.get("run_name"),
        "summary_path": str(summary_path.resolve()),
    }


def build_report(results_dir: Path) -> dict[str, Any]:
    baseline_path = results_dir / BASELINE_FILE
    baseline = load_summary(baseline_path)
    baseline_e2e_ms = metric(baseline, "e2e_step_ms")
    baseline_a2a_ms = metric(baseline, "forward_a2a_ms") + metric(
        baseline,
        "backward_a2a_ms",
    )
    rows = [
        make_row(
            payload=baseline,
            summary_path=baseline_path,
            label="VeOmni",
            rho=None,
            redundant_slots=0,
            baseline_e2e_ms=baseline_e2e_ms,
            baseline_a2a_ms=baseline_a2a_ms,
        )
    ]
    for rho, redundant_slots, filename in RHO_CASES:
        path = results_dir / filename
        payload = load_summary(path)
        rows.append(
            make_row(
                payload=payload,
                summary_path=path,
                label=f"Ours ($\\rho={rho:g}$)",
                rho=rho,
                redundant_slots=redundant_slots,
                baseline_e2e_ms=baseline_e2e_ms,
                baseline_a2a_ms=baseline_a2a_ms,
            )
        )
    return {
        "schema_version": 1,
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "parallelism": "EP32",
        "primary_experts_per_rank": 4,
        "steady_steps": [11, 20],
        "e2e_speedup_definition": "VeOmni e2e_step_ms / method e2e_step_ms",
        "a2a_total_definition": "forward_a2a_ms + backward_a2a_ms",
        "a2a_speedup_definition": "VeOmni total_a2a_ms / method total_a2a_ms",
        "note": (
            "Total A2A standard deviation is not inferred from separate forward "
            "and backward summary deviations because their covariance is unavailable."
        ),
        "rows": rows,
    }


def format_optional(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{100.0 * value:.2f}%"
    return f"{value:.3f}"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL EP32 Hyperparameter Results",
        "",
        (
            "All values use optimizer steps 11--20. E2E and A2A speedups are "
            "normalized to the VeOmni row."
        ),
        "",
        (
            "| Configuration | $\\rho$ | Redundant slots/rank | Capacity | "
            "E2E (s/step) | E2E speedup | Fwd A2A (s) | Bwd A2A (s) | "
            "Total A2A (s) | A2A speedup | Dedup ratio |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        rho = "--" if row["rho"] is None else f"{row['rho']:.2f}"
        lines.append(
            "| "
            f"{row['label']} | {rho} | {row['redundant_slots_per_rank']} | "
            f"{row['total_expert_capacity_ratio']:.2f}$\\times$ | "
            f"{row['e2e_step_ms'] / 1000.0:.3f} $\\pm$ "
            f"{row['e2e_step_std_ms'] / 1000.0:.3f} | "
            f"{row['e2e_speedup']:.3f}$\\times$ | "
            f"{row['forward_a2a_ms'] / 1000.0:.3f} | "
            f"{row['backward_a2a_ms'] / 1000.0:.3f} | "
            f"{row['total_a2a_ms'] / 1000.0:.3f} | "
            f"{row['a2a_speedup']:.3f}$\\times$ | "
            f"{format_optional(row['dedup_ratio'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- Total A2A = Forward A2A + Backward A2A.",
            "- E2E speedup = VeOmni E2E / configuration E2E.",
            "- A2A speedup = VeOmni total A2A / configuration total A2A.",
            (
                "- Total A2A standard deviation is omitted because the source "
                "summaries do not retain Forward/Backward covariance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    fieldnames = tuple(report["rows"][0])
    with output_prefix.with_suffix(".csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def main() -> None:
    args = parse_args()
    report = build_report(args.results_dir)
    write_outputs(report, args.output_prefix)
    print(f"Wrote {args.output_prefix}.{{json,md,csv}}")


if __name__ == "__main__":
    main()
