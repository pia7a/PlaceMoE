#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Plot per-workload E2E speedup normalized to the VeOmni baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "baseline": "VeOmni",
    "r2": "R2",
    "eplb": "EPLB",
    "hiermoe": "HierMoE",
    "ours": "Ours",
    "hyper_rho000": "Ours ρ=0",
    "hyper_rho025": "Ours ρ=0.25",
    "hyper_rho050": "Ours ρ=0.5",
    "hyper_rho075": "Ours ρ=0.75",
    "hyper_rho100": "Ours ρ=1",
    "dedup": "Hierarchical Dedup",
    "static_r2": "Static R2",
    "static_r2_grad_overlap": "Static R2 + Grad Overlap",
    "comm_only": "Comm-only",
    "compute_only": "Compute-only",
    "comm_assignment": "Comm + Assignment",
    "full_ours": "Full Ours",
    "online_lut": "Full Ours + Online LUT",
    "full_ours_grad_blocking": "Full Ours (Grad Blocking)",
}
METHOD_COLORS = {
    "baseline": "#8A94A6",
    "r2": "#5B8FF9",
    "eplb": "#61DDAA",
    "hiermoe": "#F6BD16",
    "ours": "#E8684A",
    "hyper_rho000": "#A7C7E7",
    "hyper_rho025": "#7FB3D5",
    "hyper_rho050": "#5B8FF9",
    "hyper_rho075": "#386CB0",
    "hyper_rho100": "#1F4E79",
    "dedup": "#8A94A6",
    "static_r2": "#5B8FF9",
    "static_r2_grad_overlap": "#386CB0",
    "comm_only": "#F6BD16",
    "compute_only": "#8B5CF6",
    "comm_assignment": "#61DDAA",
    "full_ours": "#E8684A",
    "online_lut": "#8B5CF6",
    "full_ours_grad_blocking": "#D97706",
}
METHOD_GRAD_MODES = {
    "static_r2": "blocking",
    "static_r2_grad_overlap": "hidden",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--steady-steps-label",
        default="steps 3–5",
        help="Human-readable sampling-window label shown in the SVG subtitle.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Paper summary JSON; baseline must be present.",
    )
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def _parse_summary(spec: str) -> tuple[str, Path]:
    method, separator, path = spec.partition("=")
    if not separator or not method or not path:
        raise ValueError(f"invalid --summary {spec!r}; expected METHOD=PATH")
    return method, Path(path)


def _step_ms(payload: dict[str, Any], path: Path) -> tuple[float, float]:
    timing = payload.get("e2e_step_ms")
    if not isinstance(timing, dict):
        raise ValueError(f"{path}: missing e2e_step_ms")
    mean = timing.get("mean")
    std = timing.get("std")
    if not isinstance(mean, (int, float)) or mean <= 0:
        raise ValueError(f"{path}: invalid e2e_step_ms.mean={mean!r}")
    if not isinstance(std, (int, float)):
        std = 0.0
    return float(mean), float(std)


def _mean(payload: dict[str, Any], key: str) -> float | None:
    metric = payload.get(key)
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_svg(report: dict[str, Any]) -> str:
    rows = report["methods"]
    width = 1000
    height = 650
    left = 105
    right = 45
    top = 100
    bottom = 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(1.0, *(float(row["speedup"]) for row in rows))
    y_max = max(1.1, maximum * 1.16)
    tick_count = 5
    slot_width = plot_width / len(rows)
    bar_width = min(115.0, slot_width * 0.62)

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}"
        ".title{font-size:27px;font-weight:700}"
        ".subtitle{font-size:16px;fill:#52606d}"
        ".tick{font-size:14px;fill:#52606d}"
        ".label{font-size:17px;font-weight:600}"
        ".value{font-size:16px;font-weight:700}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" class="title">E2E Training Speedup vs. VeOmni</text>',
        f'<text x="{width / 2:.1f}" y="70" text-anchor="middle" class="subtitle">'
        f"{escape(report['model'])} · {escape(report['dataset'])} · "
        f"{escape(report['steady_steps_label'])}</text>",
    ]
    for index in range(tick_count + 1):
        value = y_max * index / tick_count
        yy = y(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{yy:.2f}" x2="{width - right}" y2="{yy:.2f}" '
                'stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="tick">{value:.2f}×</text>',
            ]
        )
    baseline_y = y(1.0)
    parts.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{width - right}" '
        f'y2="{baseline_y:.2f}" stroke="#374151" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    for index, row in enumerate(rows):
        center = left + slot_width * (index + 0.5)
        speedup = float(row["speedup"])
        yy = y(speedup)
        bar_height = top + plot_height - yy
        method = str(row["method"])
        color = METHOD_COLORS.get(method, "#64748b")
        label = METHOD_LABELS.get(method, method)
        parts.extend(
            [
                f'<rect x="{center - bar_width / 2:.2f}" y="{yy:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" rx="5" fill="{color}"/>',
                f'<text x="{center:.2f}" y="{max(yy - 11, 88):.2f}" '
                f'text-anchor="middle" class="value">{speedup:.3f}×</text>',
                f'<text x="{center:.2f}" y="{top + plot_height + 32:.2f}" '
                f'text-anchor="middle" class="label">{escape(label)}</text>',
                f'<text x="{center:.2f}" y="{top + plot_height + 55:.2f}" '
                f'text-anchor="middle" class="tick">{float(row["e2e_step_ms"]):.1f} ms</text>',
            ]
        )
    label_y = top + plot_height / 2
    parts.extend(
        [
            f'<text x="25" y="{label_y:.2f}" text-anchor="middle" class="label" '
            f'transform="rotate(-90 25 {label_y:.2f})">Speedup (higher is better)</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    args = _args()
    summaries: list[tuple[str, Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for spec in args.summary:
        method, path = _parse_summary(spec)
        if method in seen:
            raise ValueError(f"duplicate summary method {method!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(method)
        summaries.append((method, path, json.loads(path.read_text(encoding="utf-8"))))
    if "baseline" not in seen:
        raise ValueError("a baseline=PATH summary is required")
    grad_protocols = {payload.get("grad_protocol") for _, _, payload in summaries}
    if len(grad_protocols) != 1 or None in grad_protocols:
        raise ValueError(f"summaries must use one explicit grad protocol, got {grad_protocols}")
    grad_protocol = grad_protocols.pop()

    provenance_payloads = [payload.get("provenance") for _, _, payload in summaries]
    has_provenance = [isinstance(value, dict) for value in provenance_payloads]
    if any(has_provenance) and not all(has_provenance):
        raise ValueError("summaries mix provenance-aware and legacy schemas")
    shared_provenance: dict[str, Any] = {}
    shared_execution_policy = None
    if all(has_provenance):
        for key in (
            "cost_model_sha256",
            "communication_calibration_sha256",
            "preflight_report_sha256",
            "communication_source_sha256",
        ):
            values = {payload[key] for payload in provenance_payloads if isinstance(payload, dict)}
            if len(values) != 1 or None in values:
                raise ValueError(f"summaries do not share one {key}: {values!r}")
            shared_provenance[key] = values.pop()
        execution_policies = {payload.get("execution_policy") for _, _, payload in summaries}
        if len(execution_policies) != 1 or None in execution_policies:
            raise ValueError(f"summaries do not share one execution policy: {execution_policies!r}")
        shared_execution_policy = execution_policies.pop()

    baseline_path, baseline_payload = next(
        (path, payload) for method, path, payload in summaries if method == "baseline"
    )
    baseline_ms, _baseline_std = _step_ms(baseline_payload, baseline_path)
    rows: list[dict[str, Any]] = []
    for method, path, payload in summaries:
        mean, std = _step_ms(payload, path)
        forward_a2a_ms = _mean(payload, "forward_a2a_ms")
        backward_a2a_ms = _mean(payload, "backward_a2a_ms")
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        peak_accelerator_allocated_gib = payload.get("peak_accelerator_allocated_gib")
        peak_accelerator_reserved_gib = payload.get("peak_accelerator_reserved_gib")
        rows.append(
            {
                "method": method,
                "grad_protocol": grad_protocol,
                "label": METHOD_LABELS.get(method, method),
                "run_name": payload.get("run_name"),
                "summary_path": str(path.resolve()),
                "summary_sha256": _sha256(path),
                "input_summary_sha256": payload.get("input_summary_sha256"),
                "execution_policy": payload.get("execution_policy"),
                "execution_order": payload.get("execution_order"),
                "e2e_source": payload.get("e2e_source"),
                "e2e_step_ms": mean,
                "e2e_step_std_ms": std,
                "sample_count": payload.get("e2e_step_ms", {}).get("count"),
                "steady_steps": payload.get("steady_steps"),
                "forward_a2a_ms": forward_a2a_ms,
                "backward_a2a_ms": backward_a2a_ms,
                "total_a2a_ms": (
                    forward_a2a_ms + backward_a2a_ms
                    if forward_a2a_ms is not None and backward_a2a_ms is not None
                    else None
                ),
                "moe_communication_region_ms": _mean(payload, "moe_communication_region_ms"),
                "dedup_ratio_dispatch": _mean(payload, "dedup_ratio_dispatch"),
                "peak_accelerator_allocated_gib": peak_accelerator_allocated_gib,
                "peak_accelerator_reserved_gib": peak_accelerator_reserved_gib,
                "hiermoe_ablation_grad_mode": (
                    payload.get("hiermoe_ablation_grad_mode") or METHOD_GRAD_MODES.get(method)
                ),
                **{
                    key: provenance.get(key)
                    for key in (
                        "layout_sha256",
                        "layout_report_sha256",
                        "layout_bundle_sha256",
                        "cost_model_sha256",
                        "communication_calibration_sha256",
                        "preflight_report_sha256",
                        "communication_source_sha256",
                    )
                },
                "speedup": baseline_ms / mean,
            }
        )
    report = {
        "schema_version": 2,
        "metric": "baseline_e2e_step_ms / method_e2e_step_ms",
        "baseline": "VeOmni",
        "baseline_e2e_step_ms": baseline_ms,
        "grad_protocol": grad_protocol,
        "model": args.model,
        "dataset": args.dataset,
        "steady_steps_label": args.steady_steps_label,
        "provenance": shared_provenance,
        "execution_policy": shared_execution_policy,
        "methods": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_svg.write_text(_render_svg(report), encoding="utf-8")
    output_csv = args.output_csv or args.output_json.with_suffix(".csv")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "method",
            "grad_protocol",
            "label",
            "run_name",
            "summary_path",
            "summary_sha256",
            "input_summary_sha256",
            "execution_policy",
            "execution_order",
            "e2e_source",
            "e2e_step_ms",
            "e2e_step_std_ms",
            "sample_count",
            "steady_steps",
            "forward_a2a_ms",
            "backward_a2a_ms",
            "total_a2a_ms",
            "moe_communication_region_ms",
            "dedup_ratio_dispatch",
            "peak_accelerator_allocated_gib",
            "peak_accelerator_reserved_gib",
            "hiermoe_ablation_grad_mode",
            "layout_sha256",
            "layout_report_sha256",
            "layout_bundle_sha256",
            "cost_model_sha256",
            "communication_calibration_sha256",
            "preflight_report_sha256",
            "communication_source_sha256",
            "speedup",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
