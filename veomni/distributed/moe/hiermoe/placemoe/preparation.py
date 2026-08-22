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

"""Reusable artifact validation for one-command PlaceMoE preparation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import ModelCalibrationError, sha256_path, validate_runtime_performance_model


_PLANNER_COEFFICIENTS = {
    "inter_ms_per_byte",
    "intra_ms_per_byte",
    "route_ms_per_assignment",
    "communication_multiplier",
    "compute_ms_per_assignment",
    "compute_multiplier",
}
_PLANNER_VALIDATION_CHECKS = {"communication", "compute", "joint"}


@dataclass(frozen=True)
class PreparationSpec:
    """Resolved inputs shared by runtime and model calibration."""

    config_path: Path
    entrypoint: Path
    runtime_artifact: Path
    planner_artifact: Path
    model_id: str
    ep_size: int
    ranks_per_node: int
    hierarchy_group_sizes: tuple[int, ...]
    runtime_device_type: str
    runtime_backend: str
    runtime_dtype: str
    calibration_input_sha256: str


@dataclass(frozen=True)
class CacheInspection:
    """Local state of one cached calibration artifact."""

    state: str
    detail: str
    digest: str = ""

    def __post_init__(self) -> None:
        if self.state not in {"valid", "missing", "invalid"}:
            raise ValueError(f"unsupported cache state {self.state!r}")
        if self.state == "valid" and not self.digest:
            raise ValueError("a valid cache inspection requires an artifact digest")


@dataclass(frozen=True)
class CacheDecision:
    """Cluster-wide action selected from node-local cache states."""

    action: str
    detail: str

    def __post_init__(self) -> None:
        if self.action not in {"reuse", "run", "error"}:
            raise ValueError(f"unsupported cache action {self.action!r}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelCalibrationError(f"{name} must be a mapping")
    return value


def _resolve_path(value: Any, *, base_directory: Path, name: str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        raise ModelCalibrationError(f"{name} must be configured in the training YAML")
    path = Path(raw)
    return (path if path.is_absolute() else base_directory / path).resolve()


def fingerprint_calibration_inputs(source: Mapping[str, Any], entrypoint: Path) -> str:
    """Hash model, workload, and execution inputs that affect planner coefficients."""

    normalized = copy.deepcopy(dict(source))
    train = normalized.get("train")
    if isinstance(train, dict):
        for key in ("checkpoint", "max_steps", "moe_load_balance_monitor_interval", "num_train_epochs", "optimizer"):
            train.pop(key, None)
        for key in ("profile", "wandb"):
            train.pop(key, None)
        hiermoe = train.get("hiermoe")
        if isinstance(hiermoe, dict):
            hiermoe.pop("placemoe", None)
    payload = {
        "schema_version": 1,
        "training": normalized,
        "entrypoint_sha256": sha256_path(entrypoint),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preparation_spec(
    source: Mapping[str, Any],
    *,
    config_path: Path,
    entrypoint: Path,
    nnodes: int,
    nproc_per_node: int,
    runtime_device_type: str,
    runtime_backend: str,
    runtime_dtype: str,
    model_id: str | None = None,
) -> PreparationSpec:
    """Resolve and validate the topology and artifact paths in a training YAML."""

    root = _mapping(source, "training config")
    train = _mapping(root.get("train"), "train")
    accelerator = _mapping(train.get("accelerator"), "train.accelerator")
    hiermoe = _mapping(train.get("hiermoe"), "train.hiermoe")
    placemoe = _mapping(hiermoe.get("placemoe"), "train.hiermoe.placemoe")
    if not bool(placemoe.get("enabled", False)):
        raise ModelCalibrationError("train.hiermoe.placemoe.enabled must be true")

    ep_size = int(accelerator.get("ep_size", 0) or 0)
    world_size = int(nnodes) * int(nproc_per_node)
    if ep_size <= 1 or ep_size != world_size:
        raise ModelCalibrationError(
            "PlaceMoE preparation requires exactly one EP group with ep_size > 1; "
            f"config ep_size={ep_size}, world_size={world_size}"
        )
    hierarchy = tuple(int(value) for value in hiermoe.get("hierarchy_group_sizes", ()) or ())
    expected_hierarchy = (int(nproc_per_node), ep_size)
    if hierarchy != expected_hierarchy:
        raise ModelCalibrationError(
            f"PlaceMoE preparation requires hierarchy_group_sizes={expected_hierarchy}, got {hierarchy}"
        )

    raw_base = os.path.expandvars(os.path.expanduser(str(placemoe.get("base_directory") or "").strip()))
    base_directory = Path(raw_base).resolve() if raw_base else Path.cwd()
    runtime_value = placemoe.get("runtime_perf_model") or hiermoe.get("perf_model_path")
    calibration = _mapping(placemoe.get("calibration"), "train.hiermoe.placemoe.calibration")
    resolved_entrypoint = entrypoint.expanduser().resolve()
    if not resolved_entrypoint.is_file():
        raise ModelCalibrationError(f"training entrypoint not found: {resolved_entrypoint}")

    explicit_model_id = str(model_id or "").strip()
    if explicit_model_id:
        resolved_model_id = explicit_model_id
    else:
        model = _mapping(root.get("model"), "model")
        model_path = str(model.get("model_path") or model.get("config_path") or "").rstrip("/")
        if not model_path:
            raise ModelCalibrationError("model.model_path is required to derive calibration scope")
        resolved_model_id = Path(model_path).name

    return PreparationSpec(
        config_path=config_path.expanduser().resolve(),
        entrypoint=resolved_entrypoint,
        runtime_artifact=_resolve_path(
            runtime_value,
            base_directory=base_directory,
            name="train.hiermoe.placemoe.runtime_perf_model",
        ),
        planner_artifact=_resolve_path(
            calibration.get("artifact"),
            base_directory=base_directory,
            name="train.hiermoe.placemoe.calibration.artifact",
        ),
        model_id=resolved_model_id,
        ep_size=ep_size,
        ranks_per_node=int(nproc_per_node),
        hierarchy_group_sizes=hierarchy,
        runtime_device_type=str(runtime_device_type),
        runtime_backend=str(runtime_backend),
        runtime_dtype=str(runtime_dtype),
        calibration_input_sha256=fingerprint_calibration_inputs(root, resolved_entrypoint),
    )


def _validate_link_cost(payload: Any, name: str) -> None:
    link = _mapping(payload, name)
    alpha = float(link.get("alpha"))
    beta = float(link.get("beta"))
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha < 0.0 or beta < 0.0:
        raise ModelCalibrationError(f"{name} must contain finite non-negative alpha and beta")
    if alpha == 0.0 and beta == 0.0:
        raise ModelCalibrationError(f"{name} cannot have zero alpha and beta")


def _validate_runtime_costs(payload: Mapping[str, Any]) -> None:
    _validate_link_cost(payload.get("a2a"), "a2a")
    state_move = _mapping(payload.get("state_move"), "state_move")
    _validate_link_cost(state_move.get("intra"), "state_move.intra")
    _validate_link_cost(state_move.get("inter"), "state_move.inter")
    gradient_sync = _mapping(payload.get("gradient_sync"), "gradient_sync")
    for phase in ("gather", "scatter"):
        phase_cost = _mapping(gradient_sync.get(phase), f"gradient_sync.{phase}")
        _validate_link_cost(phase_cost.get("intra"), f"gradient_sync.{phase}.intra")
        _validate_link_cost(phase_cost.get("inter"), f"gradient_sync.{phase}.inter")


def inspect_runtime_cache(path: Path, spec: PreparationSpec) -> CacheInspection:
    """Check whether a topology calibration can be reused."""

    if not path.exists():
        return CacheInspection("missing", str(path))
    if not path.is_file():
        return CacheInspection("invalid", f"not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_runtime_performance_model(
            _mapping(payload, "runtime performance model"),
            ep_size=spec.ep_size,
            ranks_per_node=spec.ranks_per_node,
            hierarchy_group_sizes=spec.hierarchy_group_sizes,
        )
        payload_mapping = _mapping(payload, "runtime performance model")
        _validate_runtime_costs(payload_mapping)
        metadata = _mapping(payload_mapping.get("metadata"), "runtime performance model metadata")
        expected_metadata = {
            "device_type": spec.runtime_device_type,
            "backend": spec.runtime_backend,
            "dtype": spec.runtime_dtype,
        }
        mismatches = [
            f"{key}={metadata.get(key)!r}, expected {value!r}"
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        ]
        if mismatches:
            raise ModelCalibrationError("runtime environment mismatch: " + "; ".join(mismatches))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, ModelCalibrationError) as error:
        return CacheInspection("invalid", str(error))
    digest = sha256_path(path)
    return CacheInspection("valid", "scope matches", digest)


def _scope_mismatches(scope: Mapping[str, Any], spec: PreparationSpec) -> list[str]:
    expected: dict[str, Any] = {
        "model_id": spec.model_id,
        "ep_size": spec.ep_size,
        "ranks_per_node": spec.ranks_per_node,
        "hierarchy_group_sizes": list(spec.hierarchy_group_sizes),
    }
    mismatches = []
    for key, expected_value in expected.items():
        if key not in scope:
            mismatches.append(f"missing {key}")
        elif scope[key] != expected_value:
            mismatches.append(f"{key}={scope[key]!r}, expected {expected_value!r}")
    return mismatches


def inspect_planner_cache(
    path: Path,
    spec: PreparationSpec,
    *,
    runtime_artifact_sha256: str,
) -> CacheInspection:
    """Check a planner artifact's status, scope, coefficients, and dependency hash."""

    if not path.exists():
        return CacheInspection("missing", str(path))
    if not path.is_file():
        return CacheInspection("invalid", f"not a file: {path}")
    try:
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "planner calibration")
        if payload.get("artifact_type") != "placemoe_planner_calibration":
            raise ModelCalibrationError("artifact_type must be 'placemoe_planner_calibration'")
        if int(payload.get("schema_version", 0) or 0) < 1:
            raise ModelCalibrationError("schema_version must be at least 1")
        if payload.get("status") != "accepted":
            raise ModelCalibrationError(f"status is {payload.get('status')!r}, expected 'accepted'")
        scope_mismatches = _scope_mismatches(_mapping(payload.get("scope"), "planner scope"), spec)
        if scope_mismatches:
            raise ModelCalibrationError("scope mismatch: " + "; ".join(scope_mismatches))
        coefficients = _mapping(payload.get("coefficients"), "planner coefficients")
        missing_coefficients = sorted(_PLANNER_COEFFICIENTS - set(coefficients))
        if missing_coefficients:
            raise ModelCalibrationError(f"missing coefficients {missing_coefficients}")
        invalid_coefficients = [
            name
            for name in sorted(_PLANNER_COEFFICIENTS)
            if not math.isfinite(float(coefficients[name])) or float(coefficients[name]) <= 0.0
        ]
        if invalid_coefficients:
            raise ModelCalibrationError(f"non-positive or non-finite coefficients {invalid_coefficients}")
        checks = _mapping(
            _mapping(payload.get("held_out_validation"), "held_out_validation").get("checks"),
            "held_out_validation.checks",
        )
        missing_checks = sorted(_PLANNER_VALIDATION_CHECKS - set(checks))
        if missing_checks:
            raise ModelCalibrationError(f"held-out validation is missing checks {missing_checks}")
        if not all(bool(checks[name]) for name in _PLANNER_VALIDATION_CHECKS):
            raise ModelCalibrationError("held-out validation checks did not all pass")
        provenance = _mapping(payload.get("provenance"), "planner provenance")
        actual_runtime_sha256 = str(provenance.get("runtime_perf_model_sha256") or "")
        if actual_runtime_sha256 != runtime_artifact_sha256:
            raise ModelCalibrationError(
                "runtime performance model changed: "
                f"artifact sha256={actual_runtime_sha256 or '<missing>'}, "
                f"current sha256={runtime_artifact_sha256}"
            )
        actual_input_sha256 = str(provenance.get("calibration_input_sha256") or "")
        if actual_input_sha256 != spec.calibration_input_sha256:
            raise ModelCalibrationError(
                "model or execution inputs changed: "
                f"artifact sha256={actual_input_sha256 or '<missing>'}, "
                f"current sha256={spec.calibration_input_sha256}"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, ModelCalibrationError) as error:
        return CacheInspection("invalid", str(error))
    digest = sha256_path(path)
    return CacheInspection("valid", "scope and dependency match", digest)


def decide_cache_action(
    inspections: Sequence[CacheInspection],
    *,
    force: bool,
) -> CacheDecision:
    """Choose one action shared by all nodes to avoid divergent distributed launches."""

    if not inspections:
        raise ValueError("at least one cache inspection is required")
    if force:
        return CacheDecision("run", "forced by command line")
    invalid = [(rank, result.detail) for rank, result in enumerate(inspections) if result.state == "invalid"]
    if invalid:
        detail = "; ".join(f"node {rank}: {message}" for rank, message in invalid)
        return CacheDecision("error", detail)
    if all(result.state == "valid" for result in inspections):
        digests = {result.digest for result in inspections}
        if "" in digests or len(digests) != 1:
            return CacheDecision("error", "valid artifacts differ across nodes")
        return CacheDecision("reuse", "valid on every node")
    return CacheDecision("run", "missing on at least one node")


__all__ = [
    "CacheDecision",
    "CacheInspection",
    "PreparationSpec",
    "build_preparation_spec",
    "decide_cache_action",
    "fingerprint_calibration_inputs",
    "inspect_planner_cache",
    "inspect_runtime_cache",
]
