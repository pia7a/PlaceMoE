#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Build the final paper-8h result matrix from completed per-run summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

RUNS = (
    {
        "suite": "P0-main",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "R2",
        "repeat": 1,
        "file": "paper8h_p0_sharegpt4v_b4_r2_20260729_summary.json",
    },
    {
        "suite": "P0-main",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "HireMoE",
        "repeat": 1,
        "file": "paper8h_p0_sharegpt4v_b4_hiremoe_20260729_summary.json",
    },
    {
        "suite": "P0-main",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "EPLB",
        "repeat": 1,
        "file": "paper8h_p0_sharegpt4v_b4_eplb_20260729_summary.json",
    },
    {
        "suite": "P0-main",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "Ours",
        "repeat": 1,
        "file": "paper8h_p0_sharegpt4v_b4_ours_20260729_summary.json",
    },
    {
        "suite": "P1-data",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "Tulu-3",
        "budget": 4,
        "method": "EPLB",
        "repeat": 1,
        "file": "paper8h_p1_tulu3_b4_eplb_retry2_20260729_summary.json",
    },
    {
        "suite": "P1-data",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "Tulu-3",
        "budget": 4,
        "method": "Ours",
        "repeat": 1,
        "file": "paper8h_p1_tulu3_b4_ours_retry2_20260729_summary.json",
    },
    {
        "suite": "P0-repeat",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "R2",
        "repeat": 2,
        "file": "paper8h_p0_sharegpt4v_b4_r2_repeat_20260729_summary.json",
    },
    {
        "suite": "P0-repeat",
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "budget": 4,
        "method": "Ours",
        "repeat": 2,
        "file": "paper8h_p0_sharegpt4v_b4_ours_repeat_20260729_summary.json",
    },
)

METRICS = (
    "e2e_step_ms",
    "tokens_per_second_millions",
    "forward_a2a_ms",
    "backward_a2a_ms",
    "moe_communication_region_ms",
    "expert_compute_ms",
    "physical_assignment_rank_cv",
    "physical_assignment_rank_max_over_mean",
    "dedup_ratio_dispatch",
)

PROFILE_TRAIN_SECONDS = {
    # Training-only wall time to obtain the four shared route-profile steps.
    # Model loading and route-file synchronization are excluded.
    "ShareGPT4V": 96.0,
    "Tulu-3": 74.0,
}


def _read_run(spec: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / spec["file"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {key: value for key, value in spec.items() if key != "file"}
    row["summary_file"] = str(path)
    row["run_name"] = payload["run_name"]
    row["steady_steps"] = payload["steady_steps"]
    for metric in METRICS:
        values = payload.get(metric, {})
        row[f"{metric}_mean"] = values.get("mean")
        row[f"{metric}_std"] = values.get("std")
    row["peak_npu_allocated_gib"] = payload.get("peak_npu_allocated_gib")
    row["peak_npu_reserved_gib"] = payload.get("peak_npu_reserved_gib")
    row["shared_profile_train_seconds"] = PROFILE_TRAIN_SECONDS.get(row["dataset"])
    row["offline_layout_seconds"] = payload.get("offline_layout_seconds")
    return row


def _fraction(baseline: float, contender: float) -> float:
    return (baseline - contender) / baseline


def _comparison(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    baseline: str,
    contender: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["suite"] == suite]
    base = next(row for row in selected if row["method"] == baseline)
    other = next(row for row in selected if row["method"] == contender)
    return {
        "suite": suite,
        "baseline": baseline,
        "contender": contender,
        "e2e_reduction_fraction": _fraction(
            base["e2e_step_ms_mean"],
            other["e2e_step_ms_mean"],
        ),
        "throughput_increase_fraction": (
            other["tokens_per_second_millions_mean"]
            / base["tokens_per_second_millions_mean"]
            - 1.0
        ),
        "forward_a2a_reduction_fraction": _fraction(
            base["forward_a2a_ms_mean"],
            other["forward_a2a_ms_mean"],
        ),
        "backward_a2a_reduction_fraction": _fraction(
            base["backward_a2a_ms_mean"],
            other["backward_a2a_ms_mean"],
        ),
        "moe_region_reduction_fraction": _fraction(
            base["moe_communication_region_ms_mean"],
            other["moe_communication_region_ms_mean"],
        ),
    }


def _paired_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == "ShareGPT4V" and row["method"] in {"R2", "Ours"}
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_method.setdefault(row["method"], []).append(row)
    result: dict[str, Any] = {}
    for method, method_rows in by_method.items():
        values = [row["e2e_step_ms_mean"] for row in method_rows]
        result[method] = {
            "runs": len(values),
            "mean_of_run_means_ms": sum(values) / len(values),
            "minimum_run_mean_ms": min(values),
            "maximum_run_mean_ms": max(values),
        }
    result["Ours_vs_R2_e2e_reduction_fraction"] = _fraction(
        result["R2"]["mean_of_run_means_ms"],
        result["Ours"]["mean_of_run_means_ms"],
    )
    return result


def _diagnostics() -> list[dict[str, Any]]:
    qwen_report = RESULTS / "eplb_qwen35_122b_a10b_12l_profile4_b4_ep32_report_20260729.json"
    b1_report = RESULTS / "eplb_sharegpt4v_profile4_b1_ep32_48layers_report_20260729.json"
    qwen = json.loads(qwen_report.read_text(encoding="utf-8"))
    b1 = json.loads(b1_report.read_text(encoding="utf-8"))
    return [
        {
            "suite": "P1-model",
            "status": "unsupported-pair",
            "model": "Qwen3.5-122B-A10B-12L",
            "dataset": "ShareGPT4V",
            "budget": 4,
            "evidence": {
                "32_card_smoke": "passed forward, backward, and first Adam.step",
                "route_captures": 4 * 12 * 32,
                "eplb_layout": str(qwen_report),
                "eplb_layout_build_seconds": qwen["layout_build_ms"] / 1000.0,
                "ours_layout": "no feasible recursive-classifier placement for 12/12 layers",
            },
            "reason": (
                "The current Ours initializer cannot materialize a feasible placement "
                "for 256 experts with 8 primary and 4 redundant slots per rank. "
                "No EPLB-only E2E was run because it would not be a paired comparison."
            ),
        },
        {
            "suite": "P2-budget",
            "status": "unsupported-pair",
            "model": "Qwen3-VL-30B-A3B",
            "dataset": "ShareGPT4V",
            "budget": 1,
            "evidence": {
                "eplb_layout": str(b1_report),
                "eplb_layout_build_seconds": b1["layout_build_ms"] / 1000.0,
                "ours_layout": (
                    "no feasible recursive-classifier placement under both "
                    "structured and generic-instance seeds"
                ),
            },
            "reason": (
                "The current Ours initializer cannot produce the required B=1 "
                "placement. No EPLB-only E2E was run because it would not be a "
                "paired comparison."
            ),
        },
    ]


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | E2E ms | Tokens/s (M) | Fwd A2A ms | Bwd A2A ms | MoE region ms | "
        "Expert compute ms | Dedup | Peak alloc/reserved GiB | Shared profile s | Layout s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["method"],
                    f"{_fmt(row['e2e_step_ms_mean'])} ± {_fmt(row['e2e_step_ms_std'])}",
                    f"{_fmt(row['tokens_per_second_millions_mean'], 5)} ± "
                    f"{_fmt(row['tokens_per_second_millions_std'], 5)}",
                    _fmt(row["forward_a2a_ms_mean"]),
                    _fmt(row["backward_a2a_ms_mean"]),
                    _fmt(row["moe_communication_region_ms_mean"]),
                    _fmt(row["expert_compute_ms_mean"]),
                    _fmt(row["dedup_ratio_dispatch_mean"], 5),
                    f"{_fmt(row['peak_npu_allocated_gib'])}/"
                    f"{_fmt(row['peak_npu_reserved_gib'])}",
                    _fmt(row["shared_profile_train_seconds"]),
                    _fmt(row["offline_layout_seconds"]),
                )
            )
            + " |"
        )
    return lines


def _write_markdown(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    repeats: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    output: Path,
) -> None:
    main = [row for row in rows if row["suite"] == "P0-main"]
    data = [row for row in rows if row["suite"] == "P1-data"]
    repeat = [row for row in rows if row["suite"] == "P0-repeat"]
    comp = {
        (row["suite"], row["baseline"], row["contender"]): row for row in comparisons
    }
    ours_r2 = comp[("P0-main", "R2", "Ours")]
    ours_hire = comp[("P0-main", "HireMoE", "Ours")]
    tulu = comp[("P1-data", "EPLB", "Ours")]
    lines = [
        "# HierMoE paper-8h 32-card results",
        "",
        "Protocol: 32 NPUs, 20 steps per successful E2E case, statistics over steps "
        "11–20 (mean ± population std), four shared profiling steps, offline frozen "
        "layouts, and no online Cover.",
        "",
        "## P0: Qwen3-VL-30B-A3B + ShareGPT4V, B=4",
        "",
        *_table(main),
        "",
        f"Ours reduces E2E by **{ours_r2['e2e_reduction_fraction']:.2%}** versus R2 "
        f"and by **{ours_hire['e2e_reduction_fraction']:.2%}** versus HireMoE. "
        f"Throughput increases by {ours_r2['throughput_increase_fraction']:.2%} "
        "versus R2.",
        "",
        "## P1 data generalization: Qwen3-VL-30B-A3B + Tulu-3, B=4",
        "",
        *_table(data),
        "",
        f"Ours reduces E2E by **{tulu['e2e_reduction_fraction']:.2%}** and increases "
        f"throughput by **{tulu['throughput_increase_fraction']:.2%}** versus EPLB.",
        "",
        "## P0 repeat runs",
        "",
        *_table(repeat),
        "",
        f"Across two independent runs, mean-of-run-means E2E is "
        f"{repeats['R2']['mean_of_run_means_ms']:.2f} ms for R2 and "
        f"{repeats['Ours']['mean_of_run_means_ms']:.2f} ms for Ours, an Ours "
        f"reduction of **{repeats['Ours_vs_R2_e2e_reduction_fraction']:.2%}**.",
        "",
        "## Unsupported paired cases discovered within the stop-loss budget",
        "",
    ]
    for row in diagnostics:
        lines.extend(
            [
                f"- **{row['suite']} ({row['model']}, B={row['budget']})**: "
                f"{row['reason']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Measurement caveat",
            "",
            "The per-run `physical_assignment_rank_cv` and "
            "`physical_assignment_rank_max_over_mean` fields are retained in JSON/CSV "
            "for traceability, but the current timing hook records "
            "`token_expert_assignments` before a method-comparable physical-load "
            "aggregation. They must not be presented as physical load-balance results. "
            "Dedup, A2A, full MoE communication region, expert compute, memory, and E2E "
            "are directly measured and comparable.",
            "",
            "`Shared profile s` is the training-only wall time required to collect "
            "the four route-profile steps and is amortized across methods in the same "
            "dataset suite. It excludes common model loading and route-file synchronization.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = [_read_run(spec) for spec in RUNS]
    comparisons = [
        _comparison(rows, suite="P0-main", baseline="R2", contender="HireMoE"),
        _comparison(rows, suite="P0-main", baseline="R2", contender="EPLB"),
        _comparison(rows, suite="P0-main", baseline="R2", contender="Ours"),
        _comparison(rows, suite="P0-main", baseline="HireMoE", contender="Ours"),
        _comparison(rows, suite="P1-data", baseline="EPLB", contender="Ours"),
    ]
    repeats = _paired_repeats(rows)
    diagnostics = _diagnostics()
    payload = {
        "schema_version": 1,
        "protocol": {
            "npu_count": 32,
            "steps": 20,
            "steady_steps": [11, 20],
            "profile_steps": 4,
            "layout": "offline-frozen",
            "online_cover": False,
        },
        "successful_e2e_runs": rows,
        "comparisons": comparisons,
        "repeat_aggregate": repeats,
        "unsupported_paired_cases": diagnostics,
        "measurement_caveat": (
            "physical_assignment_rank_cv/max_over_mean are retained only for "
            "traceability; the current hook is not a comparable physical-load metric"
        ),
    }
    json_output = RESULTS / "paper8h_matrix_final_20260729.json"
    csv_output = RESULTS / "paper8h_matrix_final_20260729.csv"
    markdown_output = RESULTS / "paper8h_matrix_final_20260729.md"
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, comparisons, repeats, diagnostics, markdown_output)
    print(
        json.dumps(
            {
                "json": str(json_output),
                "csv": str(csv_output),
                "markdown": str(markdown_output),
                "successful_e2e_runs": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
