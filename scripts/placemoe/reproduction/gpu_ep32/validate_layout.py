#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Commit or validate one hash-bound EP32 layout bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_NAMES = ("eplb_layout", "eplb_report", "ours_layout", "ours_report")
COST_KEYS = (
    "inter_ms_per_byte",
    "intra_ms_per_byte",
    "route_ms_per_assignment",
    "communication_phase_multiplier",
    "compute_ms_per_assignment",
    "compute_phase_multiplier",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--eplb-layout", type=Path, required=True)
    parser.add_argument("--eplb-report", type=Path, required=True)
    parser.add_argument("--ours-layout", type=Path, required=True)
    parser.add_argument("--ours-report", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--cost-model-sha256", required=True)
    parser.add_argument("--route-manifest-sha256", required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--planner-source", type=Path, action="append", required=True)
    parser.add_argument("--eplb-source", type=Path, action="append", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--hierarchy-group-sizes", default="2,8,32")
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--primary-slots-per-rank", type=int, required=True)
    parser.add_argument("--redundant-slots-per-rank", type=int, required=True)
    parser.add_argument("--slots-per-rank", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--moe-impl", required=True)
    parser.add_argument("--freeze-vit", choices=("true", "false"), required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty layout-bundle input: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be an object.")
    return payload


def _match(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if type(actual.get(key)) is not type(value) or actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} mismatch: {mismatches}")


def _source_files(paths: Iterable[Path]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for index, path in enumerate(paths):
        if path.is_file():
            files.append((f"{index}:{path.name}", path))
        elif path.is_dir():
            children = sorted(child for child in path.rglob("*.py") if child.is_file())
            if not children:
                raise RuntimeError(f"Source directory contains no Python files: {path}.")
            files.extend((f"{index}:{path.name}/{child.relative_to(path)}", child) for child in children)
        else:
            raise RuntimeError(f"Missing source fingerprint input: {path}.")
    return files


def _source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    files = _source_files(paths)
    if not files:
        raise RuntimeError("A source fingerprint requires at least one file.")
    for label, path in files:
        encoded_label = label.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _identity(
    *,
    accelerator: str,
    model_id: str,
    dataset_id: str,
    micro_batch_size: int,
    global_batch_size: int,
    max_seq_len: int,
    moe_impl: str,
    freeze_vit: str,
) -> dict[str, Any]:
    return {
        "accelerator": accelerator,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "max_seq_len": max_seq_len,
        "moe_impl": moe_impl,
        "freeze_vit": freeze_vit,
    }


def _validate_ours_report(
    report_path: Path,
    *,
    cost_model: Path,
    route_root: Path,
    topology: dict[str, int],
) -> None:
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "Ours report")
    _match(report, {"schema_version": 1, "algorithm": "placemoe-v1"}, "Ours report")
    configuration = _mapping(report.get("configuration"), "Ours report configuration")
    offline = _mapping(
        _mapping(json.loads(cost_model.read_text(encoding="utf-8")), "cost model").get("offline_scorer"),
        "cost-model offline scorer",
    )
    expected: dict[str, Any] = {
        "route_root": str(route_root.resolve()),
        "optimize_steps": [0, 1, 2],
        "validation_steps": [3],
        "layer_start": 0,
        "layers": topology["layers"],
        "expected_total_layers": topology["layers"],
        "ep_size": topology["ep_size"],
        "ranks_per_node": topology["ranks_per_node"],
        "num_experts": topology["num_experts"],
        "primary_slots_per_rank": topology["primary_slots_per_rank"],
        "redundant_slots_per_rank": topology["redundant_slots_per_rank"],
        "slots_per_rank": topology["slots_per_rank"],
        "hidden_size": topology["hidden_size"],
        "bytes_per_element": 2,
        "comparison_layout": "mirrored-r2",
        "update_mode": "full",
    }
    cost_keys = COST_KEYS
    if "mid_ms_per_byte" in offline:
        cost_keys = ("inter_ms_per_byte", "mid_ms_per_byte", *COST_KEYS[1:])
        expected["hierarchy_group_sizes"] = topology["hierarchy_group_sizes"]
    expected.update({key: float(offline[key]) for key in cost_keys})
    _match(configuration, expected, "Ours layout report configuration")


def _validate_layout(
    layout_path: Path,
    *,
    algorithm: str,
    route_root: Path,
    topology: dict[str, int],
    step_key: str,
    steps: list[int],
) -> None:
    payload = _mapping(json.loads(layout_path.read_text(encoding="utf-8")), f"{algorithm} layout")
    _match(payload, {"schema_version": 2}, f"{algorithm} layout")
    source = _mapping(payload.get("source"), f"{algorithm} layout source")
    _match(
        source,
        {
            "algorithm": algorithm,
            "route_root": str(route_root.resolve()),
            step_key: steps,
        },
        f"{algorithm} layout source",
    )
    expected_topology = {
        "ep_size": topology["ep_size"],
        "num_experts": topology["num_experts"],
        "num_physical_slots": topology["ep_size"] * topology["slots_per_rank"],
        "slots_per_rank": topology["slots_per_rank"],
    }
    if algorithm == "placemoe-v1":
        expected_topology["ranks_per_node"] = topology["ranks_per_node"]
    _match(_mapping(payload.get("topology"), f"{algorithm} topology"), expected_topology, f"{algorithm} topology")
    layers = _mapping(payload.get("layers"), f"{algorithm} layers")
    if len(layers) != topology["layers"]:
        raise RuntimeError(f"{algorithm} layout has {len(layers)} layers; expected {topology['layers']}.")


def _validate_eplb_report(report_path: Path, *, topology: dict[str, int]) -> None:
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "EPLB report")
    _match(
        report,
        {
            "schema_version": 1,
            "algorithm": "deepseek-eplb-global-v1-source-lut-compiled",
            "layers": topology["layers"],
            "profile_steps": [0, 1, 2, 3],
            "redundant_slots_per_rank": topology["redundant_slots_per_rank"],
        },
        "EPLB report",
    )


def _validate_artifact_contracts(
    artifacts: dict[str, Path],
    *,
    cost_model: Path,
    route_root: Path,
    topology: dict[str, int],
) -> None:
    _validate_ours_report(artifacts["ours_report"], cost_model=cost_model, route_root=route_root, topology=topology)
    _validate_layout(
        artifacts["ours_layout"],
        algorithm="placemoe-v1",
        route_root=route_root,
        topology=topology,
        step_key="optimize_steps",
        steps=[0, 1, 2],
    )
    _validate_eplb_report(artifacts["eplb_report"], topology=topology)
    _validate_layout(
        artifacts["eplb_layout"],
        algorithm="deepseek-eplb-global-v1-source-lut-compiled",
        route_root=route_root,
        topology=topology,
        step_key="profile_steps",
        steps=[0, 1, 2, 3],
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_layout_bundle(
    bundle: Path,
    artifacts: dict[str, Path],
    *,
    cost_model: Path,
    cost_model_sha256: str,
    route_manifest_sha256: str,
    identity: dict[str, Any],
    source_fingerprints: dict[str, str],
    route_root: Path,
    topology: dict[str, int],
) -> dict[str, Any]:
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise RuntimeError(f"Expected artifacts {ARTIFACT_NAMES}, got {sorted(artifacts)}.")
    actual_cost_sha256 = _sha256(cost_model)
    if actual_cost_sha256 != cost_model_sha256:
        raise RuntimeError(f"Cost-model SHA mismatch: expected {cost_model_sha256}, got {actual_cost_sha256}.")
    _validate_artifact_contracts(
        artifacts,
        cost_model=cost_model,
        route_root=route_root,
        topology=topology,
    )
    payload = {
        "schema_version": 2,
        "source": "gpu32-layout-bundle",
        "source_fingerprints": source_fingerprints,
        "experiment": identity,
        "cost_model": {"path": str(cost_model), "sha256": cost_model_sha256},
        "route_manifest_sha256": route_manifest_sha256,
        "artifacts": {
            name: {"path": str(artifacts[name]), "sha256": _sha256(artifacts[name])} for name in ARTIFACT_NAMES
        },
    }
    _atomic_json(bundle, payload)
    return payload


def validate_layout_bundle(
    bundle: Path,
    artifacts: dict[str, Path],
    *,
    cost_model: Path,
    cost_model_sha256: str,
    route_manifest_sha256: str,
    identity: dict[str, Any],
    source_fingerprints: dict[str, str],
    route_root: Path,
    topology: dict[str, int],
) -> None:
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_version",
        "source",
        "experiment",
        "cost_model",
        "route_manifest_sha256",
        "artifacts",
        "source_fingerprints",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise RuntimeError("Layout bundle has an invalid top-level schema.")
    expected_scalar = {
        "schema_version": 2,
        "source": "gpu32-layout-bundle",
        "source_fingerprints": source_fingerprints,
        "experiment": identity,
        "cost_model": {"path": str(cost_model), "sha256": cost_model_sha256},
        "route_manifest_sha256": route_manifest_sha256,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_scalar.items()
        if type(payload.get(key)) is not type(value) or payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Layout bundle identity mismatch: {mismatches}")
    if _sha256(cost_model) != cost_model_sha256:
        raise RuntimeError("The current cost model does not match the committed SHA.")
    committed_artifacts = payload.get("artifacts")
    if not isinstance(committed_artifacts, dict) or set(committed_artifacts) != set(ARTIFACT_NAMES):
        raise RuntimeError("Layout bundle has an invalid artifact set.")
    artifact_mismatches = {}
    for name in ARTIFACT_NAMES:
        expected = {"path": str(artifacts[name]), "sha256": _sha256(artifacts[name])}
        if committed_artifacts.get(name) != expected:
            artifact_mismatches[name] = (committed_artifacts.get(name), expected)
    if artifact_mismatches:
        raise RuntimeError(f"Layout bundle artifact mismatch: {artifact_mismatches}")
    _validate_artifact_contracts(
        artifacts,
        cost_model=cost_model,
        route_root=route_root,
        topology=topology,
    )


def main() -> None:
    args = _args()
    artifacts = {
        "eplb_layout": args.eplb_layout,
        "eplb_report": args.eplb_report,
        "ours_layout": args.ours_layout,
        "ours_report": args.ours_report,
    }
    identity = _identity(
        accelerator=args.accelerator,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        max_seq_len=args.max_seq_len,
        moe_impl=args.moe_impl,
        freeze_vit=args.freeze_vit,
    )
    source_fingerprints = {
        "placemoe": _source_fingerprint(args.planner_source),
        "eplb": _source_fingerprint(args.eplb_source),
    }
    topology = {
        "layers": args.layers,
        "ep_size": args.ep_size,
        "ranks_per_node": args.ranks_per_node,
        "hierarchy_group_sizes": [int(value) for value in args.hierarchy_group_sizes.split(",") if value.strip()],
        "num_experts": args.num_experts,
        "primary_slots_per_rank": args.primary_slots_per_rank,
        "redundant_slots_per_rank": args.redundant_slots_per_rank,
        "slots_per_rank": args.slots_per_rank,
        "hidden_size": args.hidden_size,
    }
    function = write_layout_bundle if args.write else validate_layout_bundle
    function(
        args.bundle,
        artifacts,
        cost_model=args.cost_model,
        cost_model_sha256=args.cost_model_sha256,
        route_manifest_sha256=args.route_manifest_sha256,
        identity=identity,
        source_fingerprints=source_fingerprints,
        route_root=args.route_root,
        topology=topology,
    )


if __name__ == "__main__":
    main()
