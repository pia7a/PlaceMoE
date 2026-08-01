#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Plot the cumulative Qwen3-VL EP32 component ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

# Keep Matplotlib cache writes out of the training repository and read-only home.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hiermoe-matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FIGURES_DIR = Path(
    os.environ.get("HIERMOE_PAPER_FIGURES_DIR", "/home/tzq/infocom2027-paper/figures")
)

# Edit these constants to adjust the publication style.
STAGES = (
    "baseline",
    "dedup",
    "static_r2",
    "static_r2_grad_overlap",
    "comm_aware",
    "compute_aware",
)
STAGE_LABELS = {
    "baseline": "VeOmni",
    "dedup": "+Hier. Dedup.",
    "static_r2": "+Replication",
    "static_r2_grad_overlap": "+Grad. Overlap",
    "comm_aware": "+Comm.-Aware",
    "compute_aware": "+Compute-Aware",
}
STAGE_COLORS = {
    "baseline": "#149BD7",
    "dedup": "#70ADD0",
    "static_r2": "#F28E00",
    "static_r2_grad_overlap": "#386CB0",
    "comm_aware": "#16A637",
    "compute_aware": "#ED1C24",
}
STAGE_HATCHES = {
    "baseline": "///",
    "dedup": "...",
    "static_r2": "ooo",
    "static_r2_grad_overlap": "++",
    "comm_aware": "xxx",
    "compute_aware": "\\\\\\",
}
STAGE_GRAD_MODES = {
    "baseline": "not_applicable",
    "dedup": "not_applicable",
    "static_r2": "blocking",
    "static_r2_grad_overlap": "hidden",
    "comm_aware": "hidden",
    "compute_aware": "hidden",
}
SUMMARY_FILES = {
    "baseline": "paper32_qwen3vl30b_sharegpt4v_baseline_full_20260729_anchor10_r2_summary.json",
    "dedup": "paper32_qwen3vl30b_sharegpt4v_ablation_dedup_full_20260729_anchor10_r2_summary.json",
    "static_r2": (
        "paper32_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_dedup_full_"
        "huawei2_main20_v1_summary.json"
    ),
    "static_r2_grad_overlap": (
        "paper32_qwen3vl30b_sharegpt4v_ablation_static_r2_full_"
        "20260729_anchor10_r2_summary.json"
    ),
    "comm_aware": (
        "paper32_qwen3vl30b_sharegpt4v_ablation_comm_full_"
        "20260729_anchor10_r2_summary.json"
    ),
    "compute_aware": (
        "paper32_qwen3vl30b_sharegpt4v_ablation_joint_full_"
        "20260729_anchor10_r2_summary.json"
    ),
}

FIGURE_SIZE = (12.2, 4.45)
FIGURE_DPI = 240
BAR_WIDTH = 0.72
Y_LIMITS = (0.0, 2.65)


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
        help="Directory containing paper summary JSON files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_FIGURES_DIR / "paper32_qwen3vl30b_sharegpt4v_component_ablation",
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. The publication-style default has no title.",
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
    metric = payload.get("e2e_step_ms")
    if not isinstance(metric, dict) or metric.get("count") != 10:
        raise ValueError(f"{path}: expected 10 e2e_step_ms samples")
    if not isinstance(metric.get("mean"), (int, float)) or metric["mean"] <= 0:
        raise ValueError(f"{path}: invalid e2e_step_ms.mean")
    if not isinstance(metric.get("std"), (int, float)) or metric["std"] < 0:
        raise ValueError(f"{path}: invalid e2e_step_ms.std")
    return payload


def build_report(results_dir: Path) -> dict[str, Any]:
    summaries = {
        stage: load_summary(results_dir / filename)
        for stage, filename in SUMMARY_FILES.items()
    }
    baseline_ms = float(summaries["baseline"]["e2e_step_ms"]["mean"])
    rows = []
    for stage in STAGES:
        payload = summaries[stage]
        e2e_ms = float(payload["e2e_step_ms"]["mean"])
        rows.append(
            {
                "stage": stage,
                "label": STAGE_LABELS[stage],
                "run_name": str(payload["run_name"]),
                "summary_path": str((results_dir / SUMMARY_FILES[stage]).resolve()),
                "e2e_step_ms": e2e_ms,
                "e2e_step_std_ms": float(payload["e2e_step_ms"]["std"]),
                "gradient_sync_mode": STAGE_GRAD_MODES[stage],
                "normalized_throughput": baseline_ms / e2e_ms,
            }
        )
    return {
        "schema_version": 1,
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "parallelism": "EP32",
        "steady_steps": [11, 20],
        "metric": "baseline_e2e_step_ms / stage_e2e_step_ms",
        "note": (
            "The six bars form a cumulative static initialization ablation. "
            "Replication uses blocking redundant-gradient synchronization; "
            "Grad. Overlap enables hidden gradient synchronization. "
            "No online LUT, swap, or cover update is attributed to the final bar."
        ),
        "stages": rows,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 15,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#A6A6A6",
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "hatch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_report(
    report: dict[str, Any],
    output_prefix: Path,
    title: str | None,
) -> None:
    configure_matplotlib()
    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    positions = np.arange(len(STAGES), dtype=float)
    values = [row["normalized_throughput"] for row in report["stages"]]
    containers = []
    for index, stage in enumerate(STAGES):
        container = ax.bar(
            positions[index],
            values[index],
            width=BAR_WIDTH,
            color=STAGE_COLORS[stage],
            edgecolor="#252525",
            linewidth=0.65,
            hatch=STAGE_HATCHES[stage],
            label=STAGE_LABELS[stage],
            zorder=3,
        )
        containers.append(container)

    ax.set_ylim(*Y_LIMITS)
    ax.set_ylabel("Normalized Throughput")
    ax.set_xticks([positions.mean()])
    ax.set_xticklabels(["Qwen3-VL / ShareGPT4V"])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", width=0.8)
    ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    for spine in ax.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.9)
    offset = 0.035 * (Y_LIMITS[1] - Y_LIMITS[0])
    for container in containers:
        bar = container[0]
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )

    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(STAGES),
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="#D7D7D7",
        facecolor="white",
        columnspacing=1.25,
        handlelength=1.8,
        handletextpad=0.5,
    )
    if title:
        figure.suptitle(title, y=1.065, fontsize=15)
    figure.subplots_adjust(left=0.105, right=0.99, top=0.79, bottom=0.17)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def write_data_files(report: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        "stage",
        "label",
        "run_name",
        "e2e_step_ms",
        "e2e_step_std_ms",
        "gradient_sync_mode",
        "normalized_throughput",
        "summary_path",
    )
    with output_prefix.with_suffix(".csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["stages"])


def main() -> None:
    args = parse_args()
    report = build_report(args.results_dir)
    write_data_files(report, args.output_prefix)
    plot_report(report, args.output_prefix, args.title)
    print(f"Wrote {args.output_prefix}.{{svg,pdf,png,json,csv}}")


if __name__ == "__main__":
    main()
