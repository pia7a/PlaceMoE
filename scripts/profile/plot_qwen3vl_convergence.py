#!/usr/bin/env python3
"""Plot Qwen3-VL convergence and per-layer physical load balance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path)
    parser.add_argument("--dynamic-events", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=600)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--last-n", type=int, default=100)
    return parser.parse_args()


def _load(path: Path, expected_steps: int) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                row = json.loads(line)
                latest[int(row["step"])] = row
    expected = list(range(1, expected_steps + 1))
    missing = [step for step in expected if step not in latest]
    if missing:
        raise RuntimeError(f"{path}: missing {len(missing)} step(s), first missing={missing[:10]}")
    return [latest[step] for step in expected]


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for index in range(values.size):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _configure_infocom_style() -> None:
    """Use IEEE-compatible dimensions, embedded fonts, and final-size text."""
    termes_dir = Path("/home/tzq/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
    for name in (
        "texgyretermes-regular.otf",
        "texgyretermes-bold.otf",
        "texgyretermes-italic.otf",
        "texgyretermes-bolditalic.otf",
    ):
        path = termes_dir / name
        if path.is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def _save_infocom(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    # Keep the exact IEEE two-column width; bbox_inches="tight" would change it.
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _applied_steps(path: Path | None) -> list[int]:
    if path is None or not path.is_file():
        return []
    steps = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "applied":
                steps.append(int(row["apply_step"]))
    return steps


def _mark_updates(ax: plt.Axes, applied_steps: list[int]) -> None:
    for index, step in enumerate(applied_steps):
        ax.axvline(
            step,
            color="#009E73",
            linestyle=(0, (1, 2)),
            linewidth=0.65,
            alpha=0.55,
            label="Layout update" if index == 0 else None,
            zorder=1,
        )


def main() -> None:
    args = _parse_args()
    baseline = _load(args.baseline, args.expected_steps)
    ours = _load(args.ours, args.expected_steps)
    dynamic = _load(args.dynamic, args.expected_steps) if args.dynamic is not None else None
    applied_steps = _applied_steps(args.dynamic_events)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, args.expected_steps + 1)
    baseline_loss = np.asarray([float(row["loss"]) for row in baseline])
    ours_loss = np.asarray([float(row["loss"]) for row in ours])
    dynamic_loss = None if dynamic is None else np.asarray([float(row["loss"]) for row in dynamic])
    baseline_smooth = _rolling(baseline_loss, args.window)
    ours_smooth = _rolling(ours_loss, args.window)

    fig, (ax, tail_ax) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.8),
        gridspec_kw={"width_ratios": [2.1, 1.0]},
    )
    ax.plot(steps, baseline_loss, color="#4C78A8", alpha=0.18, linewidth=0.7)
    ax.plot(steps, ours_loss, color="#E45756", alpha=0.18, linewidth=0.7)
    ax.plot(steps, baseline_smooth, color="#4C78A8", linewidth=2.0, label=f"VeOmni ({args.window}-step mean)")
    ax.plot(steps, ours_smooth, color="#E45756", linewidth=2.0, label=f"Ours ({args.window}-step mean)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Full 600-step trajectory")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    tail_start = max(0, args.expected_steps - args.last_n)
    tail_ax.plot(steps[tail_start:], baseline_loss[tail_start:], color="#4C78A8", alpha=0.20, linewidth=0.7)
    tail_ax.plot(steps[tail_start:], ours_loss[tail_start:], color="#E45756", alpha=0.20, linewidth=0.7)
    tail_ax.plot(steps[tail_start:], baseline_smooth[tail_start:], color="#4C78A8", linewidth=2.0)
    tail_ax.plot(steps[tail_start:], ours_smooth[tail_start:], color="#E45756", linewidth=2.0)
    tail_ax.set_xlabel("Training step")
    tail_ax.set_title(f"Final {args.last_n} steps")
    tail_ax.grid(alpha=0.25)
    fig.suptitle("Qwen3-VL-30B-A3B convergence on ShareGPT4V (EP=32)")
    _save(fig, args.output_dir, "qwen3vl_convergence_loss")

    layer_count = len(baseline[0]["layer_max_rank_tokens_over_mean"])
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"--layer must be in [0, {layer_count - 1}], got {args.layer}")
    baseline_load = np.asarray(
        [float(row["layer_max_rank_tokens_over_mean"][args.layer]) for row in baseline]
    )
    ours_load = np.asarray([float(row["layer_max_rank_tokens_over_mean"][args.layer]) for row in ours])
    baseline_load_smooth = _rolling(baseline_load, args.window)
    ours_load_smooth = _rolling(ours_load, args.window)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(steps, baseline_load, color="#4C78A8", alpha=0.20, linewidth=0.7)
    ax.plot(steps, ours_load, color="#E45756", alpha=0.20, linewidth=0.7)
    ax.plot(steps, baseline_load_smooth, color="#4C78A8", linewidth=2.0, label="VeOmni")
    ax.plot(steps, ours_load_smooth, color="#E45756", linewidth=2.0, label="Ours")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Max physical-NPU tokens / mean")
    ax.set_title(f"Physical load balance at MoE layer {args.layer}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, args.output_dir, f"qwen3vl_physical_load_layer_{args.layer:02d}")

    # Publication-ready raw loss plot: no smoothing or aggregation. IEEE
    # recommends 7.16 in for two-column figures, 9--10 pt text at final size,
    # and redundant color plus line-style encoding.
    _configure_infocom_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.plot(
        steps,
        baseline_loss,
        color="#0072B2",
        linestyle="-",
        linewidth=1.05,
        label="Default VeOmni",
        zorder=2,
    )
    ax.plot(
        steps,
        ours_loss,
        color="#D55E00",
        linestyle=(0, (5, 2)),
        linewidth=1.10,
        label="Ours",
        zorder=3,
    )
    if dynamic_loss is not None:
        ax.plot(
            steps,
            dynamic_loss,
            color="#009E73",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=1.05,
            label="Ours + periodic update",
            zorder=4,
        )
    ax.set_xlim(1, args.expected_steps)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.grid(color="#B0B0B0", linewidth=0.45, alpha=0.45)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.8, columnspacing=1.4)
    _save_infocom(fig, args.output_dir, "qwen3vl_convergence_loss_raw_infocom")

    dynamic_load = (
        None
        if dynamic is None
        else np.asarray([float(row["layer_max_rank_tokens_over_mean"][args.layer]) for row in dynamic])
    )
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.plot(steps, baseline_load, color="#0072B2", linestyle="-", linewidth=0.9, label="Default VeOmni")
    ax.plot(steps, ours_load, color="#D55E00", linestyle=(0, (5, 2)), linewidth=0.95, label="Ours (static)")
    if dynamic_load is not None:
        ax.plot(
            steps,
            dynamic_load,
            color="#009E73",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=0.95,
            label="Ours + periodic update",
        )
        _mark_updates(ax, applied_steps)
    ax.axhline(1.0, color="#333333", linestyle=(0, (2, 2)), linewidth=0.7, alpha=0.7)
    ax.set_xlim(1, args.expected_steps)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Max/Mean Physical-NPU Tokens")
    ax.grid(color="#B0B0B0", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.6, columnspacing=1.0)
    _save_infocom(fig, args.output_dir, f"qwen3vl_physical_load_layer_{args.layer:02d}_raw_infocom")

    tail = slice(max(0, args.expected_steps - args.last_n), args.expected_steps)
    baseline_tail_loss = _mean(baseline_loss[tail])
    ours_tail_loss = _mean(ours_loss[tail])
    loss_gap = abs(baseline_tail_loss - ours_tail_loss)
    mean_loss = (abs(baseline_tail_loss) + abs(ours_tail_loss)) / 2
    baseline_times = np.asarray([float(row["step_time_s"]) for row in baseline])
    ours_times = np.asarray([float(row["step_time_s"]) for row in ours])
    dynamic_times = None if dynamic is None else np.asarray([float(row["step_time_s"]) for row in dynamic])
    periodic_spike_ranges = (
        (88, 89),
        (134, 135),
        (199, 200),
        (245, 247),
        (291, 294),
        (352, 353),
        (398, 400),
        (444, 446),
        (505, 506),
        (551, 553),
        (597, 599),
    )
    periodic_spike_steps = {
        step
        for start, stop in periodic_spike_ranges
        for step in range(start, stop + 1)
    }
    keep_step = np.asarray([int(step) not in periodic_spike_steps for step in steps])
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.plot(
        steps[keep_step],
        baseline_times[keep_step],
        color="#0072B2",
        linestyle="-",
        linewidth=1.05,
        label="Default VeOmni",
    )
    ax.plot(
        steps[keep_step],
        ours_times[keep_step],
        color="#D55E00",
        linestyle=(0, (5, 2)),
        linewidth=1.10,
        label="PlaceMoE",
    )
    ax.set_xlim(1, args.expected_steps)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Step Time (s)")
    ax.grid(color="#B0B0B0", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.8, columnspacing=1.4)
    _save_infocom(fig, args.output_dir, "qwen3vl_step_time_raw_infocom")
    step_time_filter_audit = {
        "schema_version": 1,
        "filter": "exclude only the recurring prominent spike clusters",
        "action": "omit the listed steps from both plotted series; do not average or alter other steps",
        "excluded_ranges_inclusive": [list(item) for item in periodic_spike_ranges],
        "excluded_steps": sorted(periodic_spike_steps),
        "excluded_fraction": len(periodic_spike_steps) / args.expected_steps,
        "default_veomni_raw_values": {
            str(step): float(baseline_times[step - 1]) for step in sorted(periodic_spike_steps)
        },
        "placemoe_raw_values": {
            str(step): float(ours_times[step - 1]) for step in sorted(periodic_spike_steps)
        },
        "post_filter_plot": "all remaining per-step values without smoothing or replacement",
    }
    (args.output_dir / "qwen3vl_step_time_filter_audit.json").write_text(
        json.dumps(step_time_filter_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_global_load = np.asarray([float(row["max_rank_tokens_over_mean"]) for row in baseline])
    ours_global_load = np.asarray([float(row["max_rank_tokens_over_mean"]) for row in ours])
    dynamic_global_load = (
        None
        if dynamic is None
        else np.asarray([float(row["max_rank_tokens_over_mean"]) for row in dynamic])
    )
    warm = slice(min(10, args.expected_steps), args.expected_steps)
    baseline_time_mean = _mean(baseline_times[warm])
    ours_time_mean = _mean(ours_times[warm])
    summary = {
        "schema_version": 1,
        "expected_steps": args.expected_steps,
        "rolling_window": args.window,
        "last_n": args.last_n,
        "loss": {
            "veomni_last_n_mean": baseline_tail_loss,
            "ours_last_n_mean": ours_tail_loss,
            "absolute_gap": loss_gap,
            "relative_gap": loss_gap / mean_loss if mean_loss > 0 else math.nan,
            "veomni_relative_gap": loss_gap / abs(baseline_tail_loss) if baseline_tail_loss else math.nan,
            "veomni_final": float(baseline_loss[-1]),
            "ours_final": float(ours_loss[-1]),
            "final_absolute_gap": float(abs(baseline_loss[-1] - ours_loss[-1])),
            "pearson_correlation": float(np.corrcoef(baseline_loss, ours_loss)[0, 1]),
            "last_n_pearson_correlation": float(
                np.corrcoef(baseline_loss[tail], ours_loss[tail])[0, 1]
            ),
        },
        "step_time_s_after_warmup": {
            "veomni_mean": baseline_time_mean,
            "ours_mean": ours_time_mean,
            "ours_speedup": baseline_time_mean / ours_time_mean,
            "veomni_total_hours": float(np.sum(baseline_times) / 3600),
            "ours_total_hours": float(np.sum(ours_times) / 3600),
        },
        "physical_load": {
            "layer": args.layer,
            "veomni_max_over_mean_average": _mean(baseline_load),
            "ours_max_over_mean_average": _mean(ours_load),
            "veomni_last_n_average": _mean(baseline_load[tail]),
            "ours_last_n_average": _mean(ours_load[tail]),
            "global_veomni_average": _mean(baseline_global_load),
            "global_ours_average": _mean(ours_global_load),
            "global_veomni_last_n_average": _mean(baseline_global_load[tail]),
            "global_ours_last_n_average": _mean(ours_global_load[tail]),
        },
    }
    if dynamic is not None and dynamic_loss is not None and dynamic_times is not None and dynamic_load is not None:
        dynamic_tail_loss = _mean(dynamic_loss[tail])
        dynamic_time_mean = _mean(dynamic_times[warm])
        summary["loss"].update(
            {
                "periodic_last_n_mean": dynamic_tail_loss,
                "periodic_vs_veomni_relative_gap": (
                    abs(dynamic_tail_loss - baseline_tail_loss) / abs(baseline_tail_loss)
                    if baseline_tail_loss
                    else math.nan
                ),
                "periodic_final": float(dynamic_loss[-1]),
            }
        )
        summary["step_time_s_after_warmup"].update(
            {
                "periodic_mean": dynamic_time_mean,
                "periodic_speedup": baseline_time_mean / dynamic_time_mean,
                "periodic_total_hours": float(np.sum(dynamic_times) / 3600),
                "layout_apply_steps": applied_steps,
            }
        )
        summary["physical_load"].update(
            {
                "periodic_max_over_mean_average": _mean(dynamic_load),
                "periodic_last_n_average": _mean(dynamic_load[tail]),
                "global_periodic_average": _mean(dynamic_global_load),
                "global_periodic_last_n_average": _mean(dynamic_global_load[tail]),
            }
        )
    (args.output_dir / "qwen3vl_convergence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
