#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Audit paper-case summaries and emit one machine-readable matrix manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("baseline", "r2", "eplb", "hiermoe", "ours")
EXPECTED_GRAD_MODES = {
    "baseline": "not_applicable",
    "r2": "blocking",
    "eplb": "blocking",
    "hiermoe": "not_applicable",
    "ours": "hidden",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=10)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _method_slug(method: str) -> str:
    return {
        "baseline": "veomni_baseline",
        "r2": "fixed_r2_hierarchical_dedup",
        "eplb": "eplb_static_hierarchical_dedup",
        "hiermoe": "hiermoe_exact_p1",
        "ours": "ours_static_hierarchical_dedup",
    }[method]


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def _summary_path(args: argparse.Namespace, dataset: str, method: str) -> Path:
    run_name = (
        f"{args.artifact_prefix}_{args.model_slug}_{dataset}_{_method_slug(method)}"
        f"_full_{args.run_tag}"
    )
    return args.results_root / f"{run_name}_summary.json"


def _audit_summary(
    payload: dict[str, Any],
    *,
    path: Path,
    method: str,
    expected_ranks: int,
    expected_samples: int,
) -> list[str]:
    errors: list[str] = []
    if payload.get("observed_moe_ranks") != expected_ranks:
        errors.append(
            f"{path}: observed_moe_ranks={payload.get('observed_moe_ranks')!r}, "
            f"expected {expected_ranks}"
        )
    if payload.get("steady_steps") != [11, 20]:
        errors.append(f"{path}: steady_steps={payload.get('steady_steps')!r}, expected [11, 20]")
    if payload.get("hiermoe_ablation_grad_mode") != EXPECTED_GRAD_MODES[method]:
        errors.append(
            f"{path}: grad_mode={payload.get('hiermoe_ablation_grad_mode')!r}, "
            f"expected {EXPECTED_GRAD_MODES[method]!r}"
        )
    for metric in (
        "e2e_step_ms",
        "tokens_per_second_millions",
        "forward_a2a_ms",
        "backward_a2a_ms",
    ):
        row = payload.get(metric)
        if not isinstance(row, dict):
            errors.append(f"{path}: missing metric {metric}")
            continue
        if row.get("count") != expected_samples:
            errors.append(
                f"{path}: {metric}.count={row.get('count')!r}, expected {expected_samples}"
            )
        if not _finite_positive(row.get("mean")):
            errors.append(f"{path}: invalid {metric}.mean={row.get('mean')!r}")
    return errors


def main() -> None:
    args = _args()
    errors: list[str] = []
    workloads: list[dict[str, Any]] = []
    for dataset in args.dataset:
        summaries: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        for method in METHODS:
            path = _summary_path(args, dataset, method)
            paths[method] = path
            if not path.is_file():
                errors.append(f"missing summary: {path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries[method] = payload
            errors.extend(
                _audit_summary(
                    payload,
                    path=path,
                    method=method,
                    expected_ranks=args.expected_ranks,
                    expected_samples=args.expected_samples,
                )
            )
        if "baseline" not in summaries:
            continue
        baseline_ms = float(summaries["baseline"]["e2e_step_ms"]["mean"])
        methods: list[dict[str, Any]] = []
        for method in METHODS:
            if method not in summaries:
                continue
            payload = summaries[method]
            e2e_ms = float(payload["e2e_step_ms"]["mean"])
            methods.append(
                {
                    "method": method,
                    "run_name": payload.get("run_name"),
                    "summary_path": str(paths[method].resolve()),
                    "e2e_step_ms": e2e_ms,
                    "e2e_step_std_ms": float(payload["e2e_step_ms"]["std"]),
                    "speedup_vs_veomni": baseline_ms / e2e_ms,
                    "grad_mode": payload.get("hiermoe_ablation_grad_mode"),
                    "observed_moe_ranks": payload.get("observed_moe_ranks"),
                    "steady_samples": payload["e2e_step_ms"]["count"],
                    "offline_layout_seconds": payload.get("offline_layout_seconds"),
                }
            )
        chart_stem = (
            f"{args.artifact_prefix}_{args.model_slug}_{dataset}_speedup_vs_veomni_{args.run_tag}"
        )
        chart_paths = {
            suffix: args.results_root / f"{chart_stem}.{suffix}"
            for suffix in ("svg", "json", "csv")
        }
        for path in chart_paths.values():
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing chart artifact: {path}")
        workloads.append(
            {
                "dataset": dataset,
                "methods": methods,
                "chart_paths": {
                    suffix: str(path.resolve()) for suffix, path in chart_paths.items()
                },
            }
        )

    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "artifact_prefix": args.artifact_prefix,
        "model": args.model_slug,
        "expected_ranks": args.expected_ranks,
        "expected_samples": args.expected_samples,
        "steady_steps": [11, 20],
        "methods": list(METHODS),
        "datasets": list(args.dataset),
        "workloads": workloads,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
