#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare a PlaceMoE plan with the canonical no-replica layout on routes."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from veomni.distributed.moe.hiermoe.placemoe.artifacts import validate_placemoe_artifact
from veomni.distributed.moe.hiermoe.placemoe.route_replay import HybridCost, HybridEvaluator, load_routes
from veomni.distributed.moe.hiermoe.topology import expected_hierarchy_group_sizes


_LAYER_CAPTURE_PATTERN = re.compile(r"^layer(?P<layer>\d+)_call(?P<call>\d+)_(?:all_ranks|rank00)\.pt$")
_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_REQUIRED_COEFFICIENTS = (
    "inter_ms_per_byte",
    "intra_ms_per_byte",
    "route_ms_per_assignment",
    "communication_multiplier",
    "compute_ms_per_assignment",
    "compute_multiplier",
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated integer list.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay captured routes to compare an existing PlaceMoE plan with the canonical "
            "no-replica, no-optimization layout under the same hierarchical A2A runtime."
        )
    )
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True, help="PlaceMoE layout artifact to evaluate.")
    parser.add_argument("--calibration", type=Path, required=True, help="Planner calibration artifact.")
    parser.add_argument(
        "--steps",
        type=_parse_int_list,
        default=None,
        help="Captured steps to replay, for example 0 or 0,1. Defaults to every captured step.",
    )
    parser.add_argument("--call-indices", type=_parse_int_list, default=(0,))
    hidden_group = parser.add_mutually_exclusive_group(required=True)
    hidden_group.add_argument("--hidden-size", type=int)
    hidden_group.add_argument(
        "--model-config",
        type=Path,
        help="Model config from which the MoE hidden size is inferred using the artifact expert count.",
    )
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. The comparison summary is always printed.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _infer_steps(route_root: Path) -> tuple[int, ...]:
    steps = []
    for path in route_root.glob("step[0-9][0-9][0-9][0-9]"):
        if path.is_dir():
            steps.append(int(path.name.removeprefix("step")))
    if not steps:
        raise ValueError(f"No stepXXXX route directories found under {route_root}.")
    return tuple(sorted(set(steps)))


def _infer_hidden_size(model_config: Path, num_experts: int) -> int:
    root = _read_json(model_config)
    values: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            expert_count = next((value.get(key) for key in _EXPERT_COUNT_KEYS if value.get(key) is not None), None)
            hidden_size = value.get("hidden_size")
            if expert_count is not None and hidden_size is not None and int(expert_count) == num_experts:
                values.add(int(hidden_size))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(root)
    if len(values) != 1:
        raise ValueError(
            f"Cannot uniquely infer the hidden size of the {num_experts}-expert MoE from {model_config}; "
            f"found {sorted(values)}. Pass --hidden-size explicitly."
        )
    return values.pop()


def _load_coefficients(calibration: Path) -> dict[str, float]:
    payload = _read_json(calibration)
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, dict):
        raise ValueError(f"Calibration artifact {calibration} has no coefficient table.")
    result: dict[str, float] = {}
    for key in _REQUIRED_COEFFICIENTS:
        try:
            value = float(coefficients[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Calibration artifact {calibration} has no valid {key} coefficient.") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Calibration coefficient {key} must be finite and non-negative, got {value}.")
        result[key] = value
    mid = coefficients.get("mid_ms_per_byte")
    if mid is not None:
        result["mid_ms_per_byte"] = float(mid)
    return result


def _discover_layers(
    route_root: Path,
    *,
    step: int,
    call_index: int,
    artifact_layer_keys: tuple[str, ...],
) -> tuple[tuple[int, str], ...]:
    step_root = route_root / f"step{step:04d}"
    captures: dict[int, Path] = {}
    for path in sorted(step_root.glob(f"layer*_call{call_index}_*.pt")):
        match = _LAYER_CAPTURE_PATTERN.match(path.name)
        if match is None or int(match.group("call")) != call_index:
            continue
        layer = int(match.group("layer"))
        if path.name.endswith("_all_ranks.pt") or layer not in captures:
            captures[layer] = path
    if not captures:
        raise ValueError(f"No layer captures for call {call_index} found under {step_root}.")

    discovered: list[tuple[int, str | None]] = []
    for layer, path in sorted(captures.items()):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = payload.get("layer_key") if isinstance(payload, dict) else None
        discovered.append((layer, str(key) if key is not None else None))

    if any(key is None for _, key in discovered):
        if len(discovered) != len(artifact_layer_keys):
            raise ValueError(
                "Route captures omit layer_key and their count does not match the layout artifact; "
                "the layer correspondence is ambiguous."
            )
        discovered = [(layer, artifact_layer_keys[offset]) for offset, (layer, _) in enumerate(discovered)]

    result = tuple((layer, str(key)) for layer, key in discovered)
    route_keys = tuple(key for _, key in result)
    if len(set(route_keys)) != len(route_keys):
        raise ValueError("Route captures contain duplicate layer keys.")
    missing = set(artifact_layer_keys) - set(route_keys)
    extra = set(route_keys) - set(artifact_layer_keys)
    if missing or extra:
        raise ValueError(f"Route/layout layer mismatch: missing={sorted(missing)}, extra={sorted(extra)}.")
    return result


def _canonical_source_lut(*, ep_size: int, num_experts: int, slots_per_rank: int) -> np.ndarray:
    if num_experts % ep_size:
        raise ValueError("Canonical comparison requires num_experts to be divisible by ep_size.")
    primary_slots_per_rank = num_experts // ep_size
    if slots_per_rank < primary_slots_per_rank:
        raise ValueError("The artifact does not reserve enough slots for the canonical layout.")
    experts = np.arange(num_experts, dtype=np.int64)
    owner_slots = (experts // primary_slots_per_rank) * slots_per_rank + experts % primary_slots_per_rank
    return np.broadcast_to(owner_slots, (ep_size, num_experts)).copy()


def _cost_summary(cost: HybridCost) -> dict[str, Any]:
    return asdict(cost)


def _aggregate(costs: list[HybridCost]) -> dict[str, float]:
    return {
        "communication_ms": sum(cost.communication_ms for cost in costs),
        "compute_ms": sum(cost.compute_ms for cost in costs),
        "total_ms": sum(cost.total_ms for cost in costs),
    }


def _ratio(baseline: float, optimized: float) -> float:
    if optimized == 0.0:
        return math.inf if baseline > 0.0 else 1.0
    return baseline / optimized


def _evaluation_scope(selected_steps: tuple[int, ...], layout_source: dict[str, Any]) -> str:
    optimize_steps = {int(step) for step in layout_source.get("optimize_steps", [])}
    selected = set(selected_steps)
    if not optimize_steps:
        return "unknown"
    if selected.isdisjoint(optimize_steps):
        return "held_out"
    if selected.issubset(optimize_steps):
        return "in_sample"
    return "mixed"


def compare(args: argparse.Namespace) -> dict[str, Any]:
    layout_payload = _read_json(args.layout)
    plans = validate_placemoe_artifact(layout_payload)
    topology = layout_payload["topology"]
    source = layout_payload.get("source", {})
    if not isinstance(source, dict):
        source = {}
    ep_size = int(topology["ep_size"])
    ranks_per_node = int(topology.get("ranks_per_node", ep_size))
    num_experts = int(topology["num_experts"])
    slots_per_rank = int(topology["slots_per_rank"])
    hierarchy_values = source.get("hierarchy_group_sizes") or expected_hierarchy_group_sizes(ep_size, ranks_per_node)
    hierarchy_group_sizes = tuple(int(size) for size in hierarchy_values)
    steps = args.steps or _infer_steps(args.route_root)
    hidden_size = (
        args.hidden_size if args.hidden_size is not None else _infer_hidden_size(args.model_config, num_experts)
    )
    if hidden_size <= 0 or args.bytes_per_element <= 0:
        raise ValueError("hidden-size and bytes-per-element must be positive.")
    coefficients = _load_coefficients(args.calibration)
    evaluator_args = SimpleNamespace(
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
        hierarchy_group_sizes=hierarchy_group_sizes,
        hidden_size=hidden_size,
        bytes_per_element=args.bytes_per_element,
        slots_per_rank=slots_per_rank,
        inter_ms_per_byte=coefficients["inter_ms_per_byte"],
        mid_ms_per_byte=coefficients.get("mid_ms_per_byte"),
        intra_ms_per_byte=coefficients["intra_ms_per_byte"],
        route_ms_per_assignment=coefficients["route_ms_per_assignment"],
        communication_phase_multiplier=coefficients["communication_multiplier"],
        compute_ms_per_assignment=coefficients["compute_ms_per_assignment"],
        compute_phase_multiplier=coefficients["compute_multiplier"],
    )
    evaluator = HybridEvaluator(evaluator_args)
    canonical_lut = _canonical_source_lut(
        ep_size=ep_size,
        num_experts=num_experts,
        slots_per_rank=slots_per_rank,
    )
    source_layer_keys = source.get("layer_keys")
    if isinstance(source_layer_keys, list) and set(source_layer_keys) == set(plans):
        artifact_layer_keys = tuple(str(key) for key in source_layer_keys)
    else:
        artifact_layer_keys = tuple(plans)
    layers = _discover_layers(
        args.route_root,
        step=steps[0],
        call_index=args.call_indices[0],
        artifact_layer_keys=artifact_layer_keys,
    )

    canonical_costs: list[HybridCost] = []
    optimized_costs: list[HybridCost] = []
    layer_rows: dict[str, Any] = {}
    for capture_layer, layer_key in layers:
        samples = load_routes(
            args.route_root,
            steps=steps,
            layer=capture_layer,
            ep_size=ep_size,
            call_indices=args.call_indices,
        )
        canonical_cost = evaluator.evaluate(samples, canonical_lut)
        optimized_cost = evaluator.evaluate(samples, plans[layer_key].source_logical_to_physical)
        canonical_costs.append(canonical_cost)
        optimized_costs.append(optimized_cost)
        layer_rows[layer_key] = {
            "capture_layer": capture_layer,
            "canonical": _cost_summary(canonical_cost),
            "placemoe": _cost_summary(optimized_cost),
            "speedup": {
                "communication": _ratio(canonical_cost.communication_ms, optimized_cost.communication_ms),
                "compute": _ratio(canonical_cost.compute_ms, optimized_cost.compute_ms),
                "total": _ratio(canonical_cost.total_ms, optimized_cost.total_ms),
            },
        }

    canonical = _aggregate(canonical_costs)
    optimized = _aggregate(optimized_costs)
    if canonical["total_ms"] <= 0.0:
        raise ValueError("Canonical replay cost is zero; captured routes or calibration coefficients are invalid.")
    speedup = {
        "communication": _ratio(canonical["communication_ms"], optimized["communication_ms"]),
        "compute": _ratio(canonical["compute_ms"], optimized["compute_ms"]),
        "total": _ratio(canonical["total_ms"], optimized["total_ms"]),
    }
    report = {
        "schema_version": 1,
        "comparison": "canonical-no-replica-vs-placemoe",
        "basis": "complete-route replay under the same hierarchical token-deduplicated A2A runtime",
        "note": "The canonical comparison is not VeOmni's conventional A2A runtime or an end-to-end speedup.",
        "evaluation_scope": _evaluation_scope(steps, source),
        "inputs": {
            "route_root": str(args.route_root.resolve()),
            "layout": str(args.layout.resolve()),
            "calibration": str(args.calibration.resolve()),
            "steps": list(steps),
            "call_indices": list(args.call_indices),
        },
        "topology": {
            **topology,
            "hierarchy_group_sizes": list(hierarchy_group_sizes),
            "hidden_size": hidden_size,
            "bytes_per_element": args.bytes_per_element,
        },
        "coefficients": coefficients,
        "layers": layer_rows,
        "aggregate": {
            "canonical": canonical,
            "placemoe": optimized,
            "speedup": speedup,
            "total_reduction_percent": 100.0 * (1.0 - optimized["total_ms"] / canonical["total_ms"]),
        },
    }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    canonical = aggregate["canonical"]
    optimized = aggregate["placemoe"]
    speedup = aggregate["speedup"]
    print(
        f"layers={len(report['layers'])} steps={report['inputs']['steps']} "
        f"evaluation_scope={report['evaluation_scope']}"
    )
    print(
        "canonical: "
        f"communication_ms={canonical['communication_ms']:.6f} "
        f"compute_ms={canonical['compute_ms']:.6f} total_ms={canonical['total_ms']:.6f}"
    )
    print(
        "placemoe: "
        f"communication_ms={optimized['communication_ms']:.6f} "
        f"compute_ms={optimized['compute_ms']:.6f} total_ms={optimized['total_ms']:.6f}"
    )
    print(
        "predicted_speedup: "
        f"communication={speedup['communication']:.6f} "
        f"compute={speedup['compute']:.6f} total={speedup['total']:.6f}"
    )
    print("note: canonical uses the same hierarchical A2A runtime; this is not a VeOmni E2E comparison.")


def main() -> None:
    args = _parse_args()
    report = compare(args)
    _print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
