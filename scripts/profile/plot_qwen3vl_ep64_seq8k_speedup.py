#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Plot Qwen3-VL EP64 32K-tokens/rank E2E and All-to-All speedups."""

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


DEFAULT_FIGURES_DIR = Path(
    os.environ.get("HIERMOE_PAPER_FIGURES_DIR", "/home/tzq/infocom2027-paper/figures")
)
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
SUMMARY_FILES = {
    "baseline": (
        "paper64_qwen3vl30b_sharegpt4v_veomni_baseline_full_"
        "20260730_seq8k_mb4_alloc_steps3_6_summary.json"
    ),
    "replication": (
        "paper64_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_dedup_full_"
        "20260730_seq8k_mb4_6step_alloc_summary.json"
    ),
    "eplb": (
        "paper64_qwen3vl30b_sharegpt4v_eplb_static_hierarchical_dedup_full_"
        "20260730_seq8k_mb4_6step_alloc_summary.json"
    ),
    "hiermoe": (
        "paper64_qwen3vl30b_sharegpt4v_hiermoe_exact_p1_full_"
        "20260730_seq8k_mb4_6step_alloc_summary.json"
    ),
    "ours": (
        "paper64_qwen3vl30b_sharegpt4v_ours_static_hierarchical_dedup_full_"
        "20260730_seq8k_mb4_alloc_steps3_6_summary.json"
    ),
}
FIGURE_SIZE = (8.2, 4.45)
FIGURE_DPI = 240
BAR_WIDTH = 0.72
E2E_Y_LIMITS = (0.8, 2.42)
A2A_Y_LIMITS = (0.8, 4.22)


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
            / "paper64_qwen3vl30b_sharegpt4v_seq8k_mb4"
        ),
        help="Base output path used for both figures and data files.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("steady_steps") != [3, 6]:
        raise ValueError(f"{path}: expected steady_steps=[3, 6]")
    if payload.get("observed_moe_ranks") != 64:
        raise ValueError(f"{path}: expected observed_moe_ranks=64")
    for metric in ("e2e_step_ms", "forward_a2a_ms", "backward_a2a_ms"):
        value = payload.get(metric)
        if not isinstance(value, dict) or value.get("count") != 4:
            raise ValueError(f"{path}: expected 4 samples for {metric}")
        if not isinstance(value.get("mean"), (int, float)) or value["mean"] <= 0:
            raise ValueError(f"{path}: invalid {metric}.mean")
        if not isinstance(value.get("std"), (int, float)) or value["std"] < 0:
            raise ValueError(f"{path}: invalid {metric}.std")
    return payload


def build_report(results_dir: Path) -> dict[str, Any]:
    summaries = {
        method: load_summary(results_dir / filename)
        for method, filename in SUMMARY_FILES.items()
    }
    baseline = summaries["baseline"]
    baseline_e2e_ms = float(baseline["e2e_step_ms"]["mean"])
    baseline_a2a_ms = float(baseline["forward_a2a_ms"]["mean"]) + float(
        baseline["backward_a2a_ms"]["mean"]
    )
    rows = []
    for method in METHODS:
        payload = summaries[method]
        e2e_ms = float(payload["e2e_step_ms"]["mean"])
        forward_a2a_ms = float(payload["forward_a2a_ms"]["mean"])
        backward_a2a_ms = float(payload["backward_a2a_ms"]["mean"])
        total_a2a_ms = forward_a2a_ms + backward_a2a_ms
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "e2e_step_ms": e2e_ms,
                "e2e_step_std_ms": float(payload["e2e_step_ms"]["std"]),
                "e2e_speedup": baseline_e2e_ms / e2e_ms,
                "forward_a2a_ms": forward_a2a_ms,
                "forward_a2a_std_ms": float(payload["forward_a2a_ms"]["std"]),
                "backward_a2a_ms": backward_a2a_ms,
                "backward_a2a_std_ms": float(payload["backward_a2a_ms"]["std"]),
                "total_a2a_ms": total_a2a_ms,
                "a2a_speedup": baseline_a2a_ms / total_a2a_ms,
                "run_name": payload["run_name"],
                "summary_path": str(
                    (results_dir / SUMMARY_FILES[method]).resolve()
                ),
            }
        )
    return {
        "schema_version": 1,
        "model": "Qwen3-VL-30B-A3B",
        "dataset": "ShareGPT4V",
        "parallelism": "EP64",
        "micro_batch_size_per_rank": 4,
        "sequence_length": 8192,
        "tokens_per_rank": 32768,
        "steady_steps": [3, 6],
        "steady_sample_count": 4,
        "e2e_speedup_definition": "VeOmni e2e_step_ms / method e2e_step_ms",
        "a2a_total_definition": "forward_a2a_ms + backward_a2a_ms",
        "a2a_speedup_definition": "VeOmni total_a2a_ms / method total_a2a_ms",
        "note": (
            "Total A2A standard deviation is not inferred from separate forward "
            "and backward deviations because their covariance is unavailable."
        ),
        "rows": rows,
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


def plot_metric(
    report: dict[str, Any],
    *,
    metric: str,
    output_prefix: Path,
    y_limits: tuple[float, float],
) -> None:
    configure_matplotlib()
    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    positions = list(range(len(METHODS)))
    containers = []
    by_method = {row["method"]: row for row in report["rows"]}
    for index, method in enumerate(METHODS):
        value = float(by_method[method][metric])
        container = ax.bar(
            positions[index],
            value,
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
    ax.set_xticks([sum(positions) / len(positions)])
    ax.set_xticklabels(["ShareGPT4V (32K tokens/rank)"])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", width=0.8)
    ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=2)
    for spine in ax.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.9)

    offset = 0.03 * (y_limits[1] - y_limits[0])
    for container in containers:
        bar = container[0]
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
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
        ncol=len(METHODS),
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="#D7D7D7",
        facecolor="white",
        columnspacing=1.25,
        handlelength=1.8,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.79, bottom=0.17)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL EP64 ShareGPT4V 32K-tokens/rank results",
        "",
        (
            "MB=4, sequence length=8192, 32768 tokens/rank. All values use "
            "steps 3--6 (four samples)."
        ),
        "",
        (
            "| Method | E2E (s/step) | Total A2A (s) | "
            "E2E speedup | A2A speedup |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['e2e_step_ms'] / 1000.0:.3f} "
            f"$\\pm$ {row['e2e_step_std_ms'] / 1000.0:.3f} | "
            f"{row['total_a2a_ms'] / 1000.0:.3f} | "
            f"{row['e2e_speedup']:.3f}$\\times$ | "
            f"{row['a2a_speedup']:.3f}$\\times$ |"
        )
    return "\n".join(lines) + "\n"


def write_data(report: dict[str, Any], output_prefix: Path) -> None:
    data_prefix = output_prefix.parent / f"{output_prefix.name}_speedup_data"
    data_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    fieldnames = tuple(report["rows"][0])
    with data_prefix.with_suffix(".csv").open(
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
    write_data(report, args.output_prefix)
    e2e_prefix = args.output_prefix.parent / f"{args.output_prefix.name}_e2e_speedup"
    a2a_prefix = args.output_prefix.parent / f"{args.output_prefix.name}_a2a_speedup"
    plot_metric(
        report,
        metric="e2e_speedup",
        output_prefix=e2e_prefix,
        y_limits=E2E_Y_LIMITS,
    )
    plot_metric(
        report,
        metric="a2a_speedup",
        output_prefix=a2a_prefix,
        y_limits=A2A_Y_LIMITS,
    )
    print(f"Wrote {e2e_prefix}.{{svg,pdf,png}}")
    print(f"Wrote {a2a_prefix}.{{svg,pdf,png}}")
    print(
        "Wrote "
        f"{args.output_prefix.parent / f'{args.output_prefix.name}_speedup_data'}"
        ".{json,md,csv}"
    )


if __name__ == "__main__":
    main()
