#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Plot Qwen3-VL E2E and All-to-All speedups at EP32 and EP64."""

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
METHODS = ("baseline", "replication", "eplb", "hiermoe", "ours")
METHOD_LABELS = {
    "baseline": "VeOmni",
    "replication": "Replication",
    "eplb": "EPLB",
    "hiermoe": "HierMoE",
    "ours": "Ours",
}
METHOD_COLORS = {
    "baseline": "#149BD7",
    "replication": "#F28E00",
    "eplb": "#16A637",
    "hiermoe": "#8E5AB7",
    "ours": "#ED1C24",
}
METHOD_HATCHES = {
    "baseline": "///",
    "replication": "ooo",
    "eplb": "xxx",
    "hiermoe": "////",
    "ours": "\\\\\\",
}
DATASET_LABELS = {
    "sharegpt4v": "ShareGPT4V",
    "tulu3": "Tulu-3",
}
SUMMARY_FILES = {
    "ep32": {
        "expected_ranks": 32,
        "label": "EP=32",
        "datasets": {
            "sharegpt4v": {
                "baseline": (
                    "paper32_qwen3vl30b_sharegpt4v_veomni_baseline_full_"
                    "huawei2_main20_v1_summary.json"
                ),
                "replication": (
                    "paper32_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_"
                    "dedup_full_huawei2_main20_v1_summary.json"
                ),
                "eplb": (
                    "paper32_qwen3vl30b_sharegpt4v_eplb_static_hierarchical_"
                    "dedup_full_huawei2_main20_v1_summary.json"
                ),
                "hiermoe": (
                    "paper32_qwen3vl30b_sharegpt4v_hiermoe_exact_p1_full_"
                    "huawei2_main20_v1_summary.json"
                ),
                "ours": (
                    "paper32_qwen3vl30b_sharegpt4v_ours_static_hierarchical_"
                    "dedup_full_huawei2_main20_v1_summary.json"
                ),
            },
            "tulu3": {
                "baseline": (
                    "paper32_qwen3vl30b_tulu3_veomni_baseline_full_"
                    "huawei2_main20_v1_summary.json"
                ),
                "replication": (
                    "paper32_qwen3vl30b_tulu3_fixed_r2_hierarchical_dedup_"
                    "full_huawei2_main20_v1_summary.json"
                ),
                "eplb": (
                    "paper32_qwen3vl30b_tulu3_eplb_static_hierarchical_dedup_"
                    "full_huawei1_main20_v1_summary.json"
                ),
                "hiermoe": (
                    "paper32_qwen3vl30b_tulu3_hiermoe_exact_p1_full_"
                    "huawei1_main20_v1_summary.json"
                ),
                "ours": (
                    "paper32_qwen3vl30b_tulu3_ours_static_hierarchical_dedup_"
                    "full_huawei1_main20_v1_summary.json"
                ),
            },
        },
    },
    "ep64": {
        "expected_ranks": 64,
        "label": "EP=64",
        "datasets": {
            "sharegpt4v": {
                "baseline": (
                    "paper64_qwen3vl30b_sharegpt4v_veomni_baseline_full_"
                    "20260730_summary.json"
                ),
                "replication": (
                    "paper64_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_"
                    "dedup_full_20260730_summary.json"
                ),
                "eplb": (
                    "paper64_qwen3vl30b_sharegpt4v_eplb_static_hierarchical_"
                    "dedup_full_20260730_summary.json"
                ),
                "hiermoe": (
                    "paper64_qwen3vl30b_sharegpt4v_hiermoe_exact_p1_full_"
                    "20260730_summary.json"
                ),
                "ours": (
                    "paper64_qwen3vl30b_sharegpt4v_ours_static_hierarchical_"
                    "dedup_full_20260730_summary.json"
                ),
            },
            "tulu3": {
                "baseline": (
                    "paper64_qwen3vl30b_tulu3_veomni_baseline_full_"
                    "20260730_summary.json"
                ),
                "replication": (
                    "paper64_qwen3vl30b_tulu3_fixed_r2_hierarchical_dedup_"
                    "full_20260730_summary.json"
                ),
                "eplb": (
                    "paper64_qwen3vl30b_tulu3_eplb_static_hierarchical_dedup_"
                    "full_20260730_summary.json"
                ),
                "hiermoe": (
                    "paper64_qwen3vl30b_tulu3_hiermoe_exact_p1_full_"
                    "20260730_summary.json"
                ),
                "ours": (
                    "paper64_qwen3vl30b_tulu3_ours_static_hierarchical_dedup_"
                    "full_20260730_summary.json"
                ),
            },
        },
    },
}

FIGURE_SIZE = (12.0, 4.15)
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
        default=DEFAULT_FIGURES_DIR / "paper_qwen3vl30b_ep32_ep64",
        help="Base output path used for both figures and the data files.",
    )
    return parser.parse_args()


def load_summary(path: Path, expected_ranks: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("steady_steps") != [11, 20]:
        raise ValueError(f"{path}: expected steady_steps=[11, 20]")
    if payload.get("observed_moe_ranks") != expected_ranks:
        raise ValueError(f"{path}: expected observed_moe_ranks={expected_ranks}")
    for metric in ("e2e_step_ms", "forward_a2a_ms", "backward_a2a_ms"):
        value = payload.get(metric)
        if not isinstance(value, dict) or value.get("count") != 10:
            raise ValueError(f"{path}: expected 10 samples for {metric}")
        if not isinstance(value.get("mean"), (int, float)) or value["mean"] <= 0:
            raise ValueError(f"{path}: invalid {metric}.mean")
    return payload


def build_report(results_dir: Path) -> dict[str, Any]:
    topologies = []
    for topology, topology_config in SUMMARY_FILES.items():
        expected_ranks = int(topology_config["expected_ranks"])
        datasets = []
        for dataset, method_files in topology_config["datasets"].items():
            summaries = {
                method: load_summary(results_dir / filename, expected_ranks)
                for method, filename in method_files.items()
            }
            baseline = summaries["baseline"]
            baseline_e2e_ms = float(baseline["e2e_step_ms"]["mean"])
            baseline_a2a_ms = float(baseline["forward_a2a_ms"]["mean"]) + float(
                baseline["backward_a2a_ms"]["mean"]
            )
            methods = []
            for method in METHODS:
                payload = summaries[method]
                e2e_ms = float(payload["e2e_step_ms"]["mean"])
                forward_a2a_ms = float(payload["forward_a2a_ms"]["mean"])
                backward_a2a_ms = float(payload["backward_a2a_ms"]["mean"])
                a2a_ms = forward_a2a_ms + backward_a2a_ms
                methods.append(
                    {
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "run_name": str(payload["run_name"]),
                        "summary_path": str(
                            (results_dir / method_files[method]).resolve()
                        ),
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
        topologies.append(
            {
                "topology": topology,
                "label": str(topology_config["label"]),
                "expected_ranks": expected_ranks,
                "datasets": datasets,
            }
        )
    return {
        "schema_version": 1,
        "model": "Qwen3-VL-30B-A3B",
        "steady_steps": [11, 20],
        "e2e_metric": "baseline_e2e_step_ms / method_e2e_step_ms",
        "a2a_metric": (
            "baseline_(forward+backward)_a2a_ms / "
            "method_(forward+backward)_a2a_ms"
        ),
        "topologies": topologies,
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


def draw_topology_panel(
    ax: plt.Axes,
    topology: dict[str, Any],
    metric: str,
    y_limits: tuple[float, float],
) -> None:
    dataset_positions = np.arange(len(topology["datasets"]), dtype=float)
    method_offsets = np.linspace(-GROUP_WIDTH / 2, GROUP_WIDTH / 2, len(METHODS))
    containers = []
    for method_index, method in enumerate(METHODS):
        values = [
            next(row[metric] for row in dataset["methods"] if row["method"] == method)
            for dataset in topology["datasets"]
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
    ax.set_xlabel(topology["label"], labelpad=8)
    ax.set_xticks(dataset_positions)
    ax.set_xticklabels([dataset["label"] for dataset in topology["datasets"]])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", width=0.8)
    ax.grid(False)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)
    for spine in ax.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.9)
    add_value_labels(ax, containers, y_limits[1] - y_limits[0])


def plot_metric(
    report: dict[str, Any],
    metric: str,
    y_limits: tuple[float, float],
    output_prefix: Path,
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    for ax, topology in zip(axes, report["topologies"]):
        draw_topology_panel(ax, topology, metric, y_limits)
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
    figure.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.24)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def write_data_files(report: dict[str, Any], output_prefix: Path) -> None:
    data_prefix = output_prefix.parent / f"{output_prefix.name}_speedup_data"
    data_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        "parallelism",
        "dataset",
        "method",
        "label",
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
    with data_prefix.with_suffix(".csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for topology in report["topologies"]:
            for dataset in topology["datasets"]:
                for method in dataset["methods"]:
                    writer.writerow(
                        {
                            "parallelism": topology["label"],
                            "dataset": dataset["label"],
                            **method,
                        }
                    )


def main() -> None:
    args = parse_args()
    report = build_report(args.results_dir)
    write_data_files(report, args.output_prefix)
    e2e_prefix = args.output_prefix.parent / (
        f"{args.output_prefix.name}_e2e_speedup_1x2"
    )
    a2a_prefix = args.output_prefix.parent / (
        f"{args.output_prefix.name}_a2a_speedup_1x2"
    )
    plot_metric(report, "e2e_speedup", E2E_Y_LIMITS, e2e_prefix)
    plot_metric(report, "a2a_speedup", A2A_Y_LIMITS, a2a_prefix)
    print(f"Wrote {e2e_prefix}.{{svg,pdf,png}}")
    print(f"Wrote {a2a_prefix}.{{svg,pdf,png}}")
    print(
        "Wrote "
        f"{args.output_prefix.parent / f'{args.output_prefix.name}_speedup_data'}"
        ".{json,csv}"
    )


if __name__ == "__main__":
    main()
