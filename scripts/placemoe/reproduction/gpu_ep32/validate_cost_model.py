#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Validate and expose a scoped EP32 cost model for the canonical planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.placemoe.reproduction.gpu_ep32.cost_components import load_communication_calibration


_COST_KEYS = (
    "inter_ms_per_byte",
    "intra_ms_per_byte",
    "route_ms_per_assignment",
    "communication_phase_multiplier",
    "compute_ms_per_assignment",
    "compute_phase_multiplier",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--communication-calibration", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--communication-source-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--data-source-name", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cost-scope-sha256", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--slots-per-rank", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--moe-impl", required=True)
    parser.add_argument("--freeze-vit", choices=("true", "false"), required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _match(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if type(actual.get(key)) is not type(value) or actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} mismatch: {mismatches}")


def _validate_three_level_alignment(
    payload: dict[str, Any],
    workload_levels: list[float],
    shared_levels: list[float],
) -> None:
    verification = _mapping(payload.get("ours_cost_model_verify"), "ours_cost_model_verify")
    route_alignment = _mapping(verification.get("route_alignment"), "route_alignment")
    stage_link_fit = _mapping(route_alignment.get("stage_link_fit"), "stage_link_fit")
    if stage_link_fit.get("shared_calibration_used_as_topology_anchor") is not True:
        raise ValueError("cost model does not bind the shared communication calibration")
    if stage_link_fit.get("production_dispatch_fit_used_for_offline_scorer") is not True:
        raise ValueError("cost model does not use the production dispatch fit")
    rows = stage_link_fit.get("levels")
    if not isinstance(rows, list) or len(rows) != len(workload_levels):
        raise ValueError("cost-model stage alignment has an invalid level count")
    for index, (raw, workload, shared) in enumerate(
        zip(rows, workload_levels, shared_levels, strict=True),
        start=1,
    ):
        row = _mapping(raw, f"stage_link_fit.levels[{index - 1}]")
        if row.get("stage") != index:
            raise ValueError(f"cost-model stage alignment has invalid stage {row.get('stage')!r}")
        expected = {
            "shared_ms_per_byte": shared,
            "workload_reference_ms_per_byte": workload,
            "shared_scale": workload / shared,
        }
        for name, value in expected.items():
            actual = float(row.get(name, math.nan))
            if not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=1.0e-12, abs_tol=0.0):
                raise ValueError(f"cost-model stage {index} {name} does not match its calibrated value")


def main() -> None:
    request = _args()
    _, _, current_communication = load_communication_calibration(
        request.communication_calibration,
        ep_size=32,
        ranks_per_node=8,
        hidden_size=request.hidden_size,
        bytes_per_element=2,
        preflight_report=request.preflight_report,
        communication_source_sha256=request.communication_source_sha256,
    )
    payload = _mapping(json.loads(request.cost_model.read_text(encoding="utf-8")), "cost model")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {3, 4}:
        raise ValueError("EP32 cost model schema_version must be 3 or 4")
    if payload.get("source") != "gpu32-a6000-composed-cost-model":
        raise ValueError("EP32 cost model has an unsupported source")

    expected_topology = {
        "accelerator": "NVIDIA RTX A6000",
        "nodes": 4,
        "gpus_per_node": 8,
        "ep_size": 32,
        "ranks_per_node": 8,
        "num_experts": request.num_experts,
        "slots_per_rank": request.slots_per_rank,
        "hidden_size": request.hidden_size,
        "bytes_per_element": 2,
        "layers": request.layers,
    }
    if schema_version == 4:
        expected_topology["hierarchy_group_sizes"] = [2, 8, 32]
    _match(
        _mapping(payload.get("topology"), "topology"),
        expected_topology,
        "topology",
    )
    _match(
        _mapping(payload.get("model_scope"), "model_scope"),
        {
            "model_id": request.model_id,
            "checkpoint_sha256": request.checkpoint_sha256,
            "cost_scope_sha256": request.cost_scope_sha256,
            "moe_impl": request.moe_impl,
            "hidden_size": request.hidden_size,
            "bytes_per_element": 2,
        },
        "model_scope",
    )
    _match(
        _mapping(payload.get("calibration_workload"), "calibration_workload"),
        {
            "dataset_id": request.dataset_id,
            "dataset_sha256": request.dataset_sha256,
            "data_source_name": request.data_source_name,
            "micro_batch_size": request.micro_batch_size,
            "global_batch_size": request.global_batch_size,
            "max_seq_len": request.max_seq_len,
            "freeze_vit": request.freeze_vit == "true",
        },
        "calibration_workload",
    )

    offline = _mapping(payload.get("offline_scorer"), "offline_scorer")
    coefficients = {}
    cost_keys = _COST_KEYS if schema_version == 3 else ("inter_ms_per_byte", "mid_ms_per_byte", *_COST_KEYS[1:])
    for key in cost_keys:
        raw = offline.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"offline_scorer.{key} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"offline_scorer.{key} is invalid: {value!r}")
        if key.endswith("phase_multiplier") and value <= 0.0:
            raise ValueError(f"offline_scorer.{key} must be positive")
        coefficients[key] = value

    if schema_version == 4:
        raw_levels = offline.get("level_ms_per_byte")
        if not isinstance(raw_levels, list) or len(raw_levels) != 3:
            raise ValueError("offline_scorer.level_ms_per_byte must contain three coefficients")
        levels = [float(value) for value in raw_levels]
        if not all(math.isfinite(value) and value > 0.0 for value in levels):
            raise ValueError("offline_scorer.level_ms_per_byte contains invalid values")
        named_levels = [
            coefficients["inter_ms_per_byte"],
            coefficients["mid_ms_per_byte"],
            coefficients["intra_ms_per_byte"],
        ]
        if any(
            not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=0.0)
            for actual, expected in zip(named_levels, levels, strict=True)
        ):
            raise ValueError("named offline scorer coefficients do not match level_ms_per_byte")
        current_levels = [float(value) for value in current_communication["level_ms_per_byte"]]
        _validate_three_level_alignment(payload, levels, current_levels)
        coefficients["level_ms_per_byte"] = levels
        coefficients["hierarchy_group_sizes"] = [2, 8, 32]

    curve = _mapping(offline.get("compute_curve"), "offline_scorer.compute_curve")
    if curve.get("kind") != "sum_piecewise_linear_per_local_expert":
        raise ValueError("unsupported compute curve kind")
    if curve.get("local_experts_per_rank") != request.slots_per_rank:
        raise ValueError("compute curve does not match slots_per_rank")
    knots = curve.get("knots_tokens")
    slopes = curve.get("segment_ms_per_token")
    if not isinstance(knots, list) or not isinstance(slopes, list) or len(knots) != len(slopes):
        raise ValueError("invalid compute curve segments")
    normalized = [float(value) for value in [*knots, *slopes, curve.get("constant_ms")]]
    if not normalized or not all(math.isfinite(value) and value >= 0.0 for value in normalized):
        raise ValueError("compute curve contains invalid values")

    provenance = _mapping(payload.get("provenance"), "provenance")
    communication = _mapping(provenance.get("communication_calibration"), "communication provenance")
    communication_sha = _sha256(request.communication_calibration)
    if communication.get("sha256") != communication_sha:
        raise ValueError(
            "cost model communication provenance mismatch: "
            f"artifact={communication.get('sha256')!r}, current={communication_sha!r}"
        )

    print(
        json.dumps(
            {
                "status": "accepted",
                "cost_model": str(request.cost_model),
                "cost_model_sha256": _sha256(request.cost_model),
                "communication_calibration_sha256": communication_sha,
                "communication_scope": current_communication["scope"],
                "checkpoint_sha256": request.checkpoint_sha256,
                "coefficients": coefficients,
                "compute_curve": curve,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
