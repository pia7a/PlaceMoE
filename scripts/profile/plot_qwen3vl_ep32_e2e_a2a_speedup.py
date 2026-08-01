#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Plot Qwen3-VL EP32 E2E and All-to-All speedups for two datasets."""

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

# Edit these constants to adjust the figure style.
METHODS = ("baseline", "r2", "eplb", "hiermoe", "ours")
METHOD_LABELS = {
    "baseline": "VeOmni",
    "r2": "Replication",
    "eplb": "EPLB",
    "hiermoe": "HierMoE",
    "ours": "Ours",
}
METHOD_COLORS = {
    "baseline": "#149BD7",
    "r2": "#F28E00",
    "eplb": "#16A637",
    "hiermoe": "#8E5AB7",
    "ours": "#ED1C24",
}
METHOD_HATCHES = {
    "baseline": "///",
    "r2": "ooo",
    "eplb": "xxx",
    "hiermoe": "////",
    "ours": "\\\\\\",
}
DATASET_LABELS = {
    "sharegpt4v": "ShareGPT4V",
    "tulu3": "Tulu-3",
}
SUMMARY_FILES = {
    "sharegpt4v": {
        "baseline": "paper32_qwen3vl30b_sharegpt4v_veomni_baseline_full_huawei2_main20_v1_summary.json",
        "r2": "paper32_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_dedup_full_huawei2_main20_v1_summary.json",
        "eplb": "paper32_qwen3vl30b_sharegpt4v_eplb_static_hierarchical_dedup_full_huawei2_main20_v1_summary.json",
        "hiermoe": "paper32_qwen3vl30b_sharegpt4v_hiermoe_exact_p1_full_huawei2_main20_v1_summary.json",
        "ours": "paper32_qwen3vl30b_sharegpt4v_ours_static_hierarchical_dedup_full_huawei2_main20_v1_summary.json",
    },
    "tulu3": {
        "baseline": "paper32_qwen3vl30b_tulu3_veomni_baseline_full_huawei2_main20_v1_summary.json",
        "r2": "paper32_qwen3vl30b_tulu3_fixed_r2_hierarchical_dedup_full_huawei2_main20_v1_summary.json",
        "eplb": "paper32_qwen3vl30b_tulu3_eplb_static_hierarchical_dedup_full_huawei1_main20_v1_summary.json",
        "hiermoe": "paper32_qwen3vl30b_tulu3_hiermoe_exact_p1_full_huawei1_main20_v1_summary.json",
        "ours": "paper32_qwen3vl30b_tulu3_ours_static_hierarchical_dedup_full_huawei1_main20_v1_summary.json",
    },
}

FIGURE_SIZE = (12.0, 4.55)
FIGURE_DPI = 240
BAR_WIDTH = 0.145
GROUP_WIDTH = 0.82
E2E_Y_LIMITS = (0.8, 2.52)
A2A_Y_LIMITS = (0.8, 6.65)


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
        default=DEFAULT_FIGURES_DIR / "paper32_qwen3vl30b_e2e_a2a_speedup_1x2",
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
    for metric in (
        "e2e_step_ms",
        "forward_a2a_ms",
        "backward_a2a_ms",
    ):
        value = payload.get(metric)
        if not isinstance(value, dict) or value.get("count") != 10:
            raise ValueError(f"{path}: expected 10 samples for {metric}")
        if not isinstance(value.get("mean"), (int, float)) or value["mean"] <= 0:
            raise ValueError(f"{path}: invalid {metric}.mean")
    return payload


def cluster_from_run_name(run_name: str) -> str:
    if "huawei1" in run_name:
        return "huawei1"
    if "huawei2" in run_name:
        return "huawei2"
    return "unknown"


def build_report(results_dir: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for dataset, method_files in SUMMARY_FILES.items():
        summaries = {
            method: load_summary(results_dir / filename)
            for method, filename in method_files.items()
        }
        baseline = summaries["baseline"]
        baseline_e2e_ms = float(baseline["e2e_step_ms"]["mean"])
        baseline_a2a_ms = float(baseline["forward_a2a_ms"]["mean"]) + float(
            baseline["backward_a2a_ms"]["mean"]
        )
        methods: list[dict[str, Any]] = []
        for method in METHODS:
            payload = summaries[method]
            e2e_ms = float(payload["e2e_step_ms"]["mean"])
            forward_a2a_ms = float(payload["forward_a2a_ms"]["mean"])
            backward_a2a_ms = float(payload["backward_a2a_ms"]["mean"])
            a2a_ms = forward_a2a_ms + backward_a2a_ms
            run_name = str(payload["run_name"])
            methods.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "run_name": run_name,
                    "source_cluster": cluster_from_run_name(run_name),
                    "summary_path": str((results_dir / method_files[method]).resolve()),
                    "e2e_step_ms": e2e_ms,
                    "e2e_step_std_ms": float(payload["e2e_step_ms"]["std"]),
                    "e2e_speedup": baseline_e2e_ms / e2e_ms,
                    "forward_a2a_ms": forward_a2a_ms,
                    "backward_a2a_ms": backward_a2a_ms,
                    "a2a_total_ms": a2a_ms,
                    "a2a_speedup": baseline_a2a_ms / a2a_ms,
                }
            )
        datasets.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS[dataset],
                "baseline_e2e_ms": baseline_e2e_ms,
                "baseline_a2a_ms": baseline_a2a_ms,
                "methods": methods,
            }
        )
    return {
        "schema_version": 1,
        "model": "Qwen3-VL-30B-A3B",
        "parallelism": "EP32",
        "steady_steps": [11, 20],
        "e2e_metric": "baseline_e2e_step_ms / method_e2e_step_ms",
        "a2a_metric": "baseline_(forward+backward)_a2a_ms / method_(forward+backward)_a2a_ms",
        "datasets": datasets,
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
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "hatch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_value_labels(ax: plt.Axes, containers: list[Any], y_span: float) -> None:
    offset = 0.018 * y_span
    for container in containers:
        for bar in container:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )


def draw_panel(
    ax: plt.Axes,
    report: dict[str, Any],
    metric: str,
    y_limits: tuple[float, float],
    panel_label: str,
) -> None:
    dataset_positions = np.arange(len(report["datasets"]), dtype=float)
    method_offsets = np.linspace(-GROUP_WIDTH / 2, GROUP_WIDTH / 2, len(METHODS))
    containers = []
    for method_index, method in enumerate(METHODS):
        values = [
            next(row[metric] for row in dataset["methods"] if row["method"] == method)
            for dataset in report["datasets"]
        ]
        container = ax.bar(
            dataset_positions + method_offsets[method_index],
            values,
            width=BAR_WIDTH,
            color=METHOD_COLORS[method],
            edgecolor="#252525",
            linewidth=0.65,
            hatch=METHOD_HATCHES[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )
        containers.append(container)

    ax.set_ylim(*y_limits)
    ax.set_ylabel("Speedup")
    ax.set_xticks(dataset_positions)
    ax.set_xticklabels([dataset["label"] for dataset in report["datasets"]])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", width=0.8)
    ax.grid(False)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)
    for spine in ax.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.9)
    add_value_labels(ax, containers, y_limits[1] - y_limits[0])
    ax.text(
        0.5,
        -0.23,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
    )


def plot_report(report: dict[str, Any], output_prefix: Path, title: str | None) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    draw_panel(
        axes[0],
        report,
        metric="e2e_speedup",
        y_limits=E2E_Y_LIMITS,
        panel_label="(a) End-to-end speedup.",
    )
    draw_panel(
        axes[1],
        report,
        metric="a2a_speedup",
        y_limits=A2A_Y_LIMITS,
        panel_label="(b) AlltoAll speedup.",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="#D7D7D7",
        facecolor="white",
        columnspacing=1.35,
        handlelength=1.9,
        handletextpad=0.55,
    )
    if title:
        figure.suptitle(title, y=1.065, fontsize=15)
    figure.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.23, wspace=0.24)
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
        "dataset",
        "method",
        "label",
        "source_cluster",
        "run_name",
        "e2e_step_ms",
        "e2e_step_std_ms",
        "e2e_speedup",
        "forward_a2a_ms",
        "backward_a2a_ms",
        "a2a_total_ms",
        "a2a_speedup",
        "summary_path",
    )
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in report["datasets"]:
            for method in dataset["methods"]:
                writer.writerow({"dataset": dataset["label"], **method})


def main() -> None:
    args = parse_args()
    report = build_report(args.results_dir)
    write_data_files(report, args.output_prefix)
    plot_report(report, args.output_prefix, args.title)
    print(f"Wrote {args.output_prefix}.{{svg,pdf,png,json,csv}}")


if __name__ == "__main__":
    main()
