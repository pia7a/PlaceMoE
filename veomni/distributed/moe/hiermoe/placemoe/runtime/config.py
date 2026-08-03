"""Typed configuration for PlaceMoE training-time updates.

The canonical interface is a YAML or JSON file referenced by
``VEOMNI_PLACEMOE_CONFIG``.  The historical ``VEOMNI_HIERMOE_*`` variables
remain supported as a compatibility adapter for the paper launchers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEFAULT_INTER_MS_PER_BYTE = 6.765449326279194e-08
_DEFAULT_INTRA_MS_PER_BYTE = 5.02482606728045e-09
_DEFAULT_ROUTE_MS_PER_ASSIGNMENT = 8.746548178958447e-05
_DEFAULT_COMMUNICATION_MULTIPLIER = 3.1
_DEFAULT_COMPUTE_MS_PER_ASSIGNMENT = 2.82807e-05
_DEFAULT_COMPUTE_MULTIPLIER = 4.19


class PlaceMoEConfigurationError(ValueError):
    """Raised before training when a PlaceMoE configuration is invalid."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PlaceMoEConfigurationError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    raise PlaceMoEConfigurationError(f"{name} must be a boolean, got {value!r}.")


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PlaceMoEConfigurationError(f"{name} must be a non-negative integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise PlaceMoEConfigurationError(f"{name} must be a non-negative integer, got {value!r}.") from error
    if parsed < 0:
        raise PlaceMoEConfigurationError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise PlaceMoEConfigurationError(f"{name} must be positive.")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PlaceMoEConfigurationError(f"{name} must be positive, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise PlaceMoEConfigurationError(f"{name} must be positive, got {value!r}.") from error
    if not parsed > 0:
        raise PlaceMoEConfigurationError(f"{name} must be positive, got {parsed}.")
    return parsed


def _legacy_flag(environment: Mapping[str, str], name: str) -> bool:
    raw = environment.get(name)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _legacy_nonnegative_int(environment: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _legacy_positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _legacy_positive_float(environment: Mapping[str, str], name: str, default: float) -> float:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _resolve_path(value: Any, base_directory: Path, name: str) -> str:
    if value is None:
        return ""
    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = base_directory / path
    return str(path.resolve())


@dataclass(frozen=True)
class PlaceMoECalibration:
    """Calibrated coefficients passed unchanged to every planner invocation."""

    inter_ms_per_byte: float = _DEFAULT_INTER_MS_PER_BYTE
    intra_ms_per_byte: float = _DEFAULT_INTRA_MS_PER_BYTE
    route_ms_per_assignment: float = _DEFAULT_ROUTE_MS_PER_ASSIGNMENT
    communication_multiplier: float = _DEFAULT_COMMUNICATION_MULTIPLIER
    compute_ms_per_assignment: float = _DEFAULT_COMPUTE_MS_PER_ASSIGNMENT
    compute_multiplier: float = _DEFAULT_COMPUTE_MULTIPLIER
    artifact: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_directory: Path) -> "PlaceMoECalibration":
        data = dict(payload)
        artifact = _resolve_path(data.pop("artifact", ""), base_directory, "calibration.artifact")
        if artifact:
            artifact_payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
            if artifact_payload.get("status", "accepted") != "accepted":
                raise PlaceMoEConfigurationError(
                    f"calibration artifact {artifact} has status "
                    f"{artifact_payload.get('status')!r}, expected 'accepted'."
                )
            coefficients = artifact_payload.get("coefficients", artifact_payload)
            if not isinstance(coefficients, Mapping):
                raise PlaceMoEConfigurationError(f"calibration artifact {artifact} has no coefficient mapping.")
            merged = dict(coefficients)
            merged.update(data)
            data = merged
        allowed = {
            "inter_ms_per_byte",
            "intra_ms_per_byte",
            "route_ms_per_assignment",
            "communication_multiplier",
            "compute_ms_per_assignment",
            "compute_multiplier",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise PlaceMoEConfigurationError(f"unknown calibration fields: {', '.join(unknown)}.")
        defaults = cls()
        return cls(
            inter_ms_per_byte=_positive_float(
                data.get("inter_ms_per_byte", defaults.inter_ms_per_byte), "calibration.inter_ms_per_byte"
            ),
            intra_ms_per_byte=_positive_float(
                data.get("intra_ms_per_byte", defaults.intra_ms_per_byte), "calibration.intra_ms_per_byte"
            ),
            route_ms_per_assignment=_positive_float(
                data.get("route_ms_per_assignment", defaults.route_ms_per_assignment),
                "calibration.route_ms_per_assignment",
            ),
            communication_multiplier=_positive_float(
                data.get("communication_multiplier", defaults.communication_multiplier),
                "calibration.communication_multiplier",
            ),
            compute_ms_per_assignment=_positive_float(
                data.get("compute_ms_per_assignment", defaults.compute_ms_per_assignment),
                "calibration.compute_ms_per_assignment",
            ),
            compute_multiplier=_positive_float(
                data.get("compute_multiplier", defaults.compute_multiplier), "calibration.compute_multiplier"
            ),
            artifact=artifact,
        )


@dataclass(frozen=True)
class PlaceMoEPlannerResources:
    workers: int = 48
    candidate_workers: int = 4
    worker_threads: int = 1
    planner_cpu_ids: str = ""
    training_cpu_ids: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PlaceMoEPlannerResources":
        allowed = {"workers", "candidate_workers", "worker_threads", "planner_cpu_ids", "training_cpu_ids"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PlaceMoEConfigurationError(f"unknown resource fields: {', '.join(unknown)}.")
        defaults = cls()
        return cls(
            workers=_positive_int(payload.get("workers", defaults.workers), "resources.workers"),
            candidate_workers=_positive_int(
                payload.get("candidate_workers", defaults.candidate_workers), "resources.candidate_workers"
            ),
            worker_threads=_positive_int(
                payload.get("worker_threads", defaults.worker_threads), "resources.worker_threads"
            ),
            planner_cpu_ids=str(payload.get("planner_cpu_ids", defaults.planner_cpu_ids)).strip(),
            training_cpu_ids=str(payload.get("training_cpu_ids", defaults.training_cpu_ids)).strip(),
        )


@dataclass(frozen=True)
class HotUpdateConfig:
    enabled: bool = False
    layout_interval_steps: int | None = None
    mapping_interval_steps: int = 0
    last_update_step: int = 2**31 - 1
    work_root: str = "/tmp/veomni_placemoe_hot_update"
    planner_path: str = ""
    failure_policy: str = "continue"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_directory: Path) -> "HotUpdateConfig":
        allowed = {
            "enabled",
            "layout_interval_steps",
            "mapping_interval_steps",
            "last_update_step",
            "work_root",
            "planner_path",
            "failure_policy",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PlaceMoEConfigurationError(f"unknown hot_update fields: {', '.join(unknown)}.")
        defaults = cls()
        layout_raw = payload.get("layout_interval_steps", defaults.layout_interval_steps)
        layout_interval = (
            None
            if layout_raw is None
            else _nonnegative_int(layout_raw, "hot_update.layout_interval_steps")
        )
        failure_policy = str(payload.get("failure_policy", defaults.failure_policy)).strip().lower()
        if failure_policy not in {"continue", "raise"}:
            raise PlaceMoEConfigurationError(
                "hot_update.failure_policy must be 'continue' or 'raise'."
            )
        return cls(
            enabled=_strict_bool(payload.get("enabled", defaults.enabled), "hot_update.enabled"),
            layout_interval_steps=layout_interval,
            mapping_interval_steps=_nonnegative_int(
                payload.get("mapping_interval_steps", defaults.mapping_interval_steps),
                "hot_update.mapping_interval_steps",
            ),
            last_update_step=_nonnegative_int(
                payload.get("last_update_step", defaults.last_update_step), "hot_update.last_update_step"
            ),
            work_root=_resolve_path(
                payload.get("work_root", defaults.work_root), base_directory, "hot_update.work_root"
            ),
            planner_path=_resolve_path(
                payload.get("planner_path", defaults.planner_path), base_directory, "hot_update.planner_path"
            ),
            failure_policy=failure_policy,
        )


@dataclass(frozen=True)
class PlaceMoERuntimeConfig:
    """Complete runtime configuration with a strict file interface and legacy adapter."""

    initial_artifact: str = ""
    runtime_perf_model: str = ""
    hot_update: HotUpdateConfig = field(default_factory=HotUpdateConfig)
    calibration: PlaceMoECalibration = field(default_factory=PlaceMoECalibration)
    resources: PlaceMoEPlannerResources = field(default_factory=PlaceMoEPlannerResources)
    source_path: str = ""

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "PlaceMoERuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise PlaceMoEConfigurationError(f"PlaceMoE config does not exist: {config_path}.")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = _mapping(payload, "PlaceMoE config")
        if "placemoe" in root:
            root = _mapping(root["placemoe"], "placemoe")
        allowed = {"initial_artifact", "runtime_perf_model", "hot_update", "calibration", "resources"}
        unknown = sorted(set(root) - allowed)
        if unknown:
            raise PlaceMoEConfigurationError(f"unknown PlaceMoE fields: {', '.join(unknown)}.")
        base_directory = config_path.parent
        result = cls(
            initial_artifact=_resolve_path(root.get("initial_artifact", ""), base_directory, "initial_artifact"),
            runtime_perf_model=_resolve_path(
                root.get("runtime_perf_model", ""), base_directory, "runtime_perf_model"
            ),
            hot_update=HotUpdateConfig.from_mapping(
                _mapping(root.get("hot_update"), "hot_update"), base_directory=base_directory
            ),
            calibration=PlaceMoECalibration.from_mapping(
                _mapping(root.get("calibration"), "calibration"), base_directory=base_directory
            ),
            resources=PlaceMoEPlannerResources.from_mapping(_mapping(root.get("resources"), "resources")),
            source_path=str(config_path),
        )
        result.validate()
        return result

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "PlaceMoERuntimeConfig":
        env = os.environ if environment is None else environment
        config_path = env.get("VEOMNI_PLACEMOE_CONFIG", "").strip()
        if config_path:
            return cls.from_file(config_path)
        result = cls(
            initial_artifact=env.get("VEOMNI_HIERMOE_STATIC_PRELOAD_LAYOUT_PATH", "").strip(),
            runtime_perf_model=env.get("VEOMNI_HIERMOE_PERF_MODEL_PATH", "").strip(),
            hot_update=HotUpdateConfig(
                enabled=_legacy_flag(env, "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN"),
                layout_interval_steps=_legacy_nonnegative_int(
                    env, "VEOMNI_HIERMOE_LAYOUT_REFRESH_INTERVAL", None
                ),
                mapping_interval_steps=_legacy_nonnegative_int(
                    env, "VEOMNI_HIERMOE_MAPPING_REFRESH_INTERVAL", 0
                )
                or 0,
                last_update_step=_legacy_nonnegative_int(
                    env, "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_LAST_STEP", 2**31 - 1
                )
                or 0,
                work_root=env.get(
                    "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_WORK_ROOT",
                    "/tmp/veomni_hiermoe_periodic_full_replan",
                ).strip(),
                planner_path=env.get("VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_BUILDER", "").strip(),
                # Preserve the historical fail-fast behavior for legacy
                # launchers. Canonical configs default to fail-open.
                failure_policy="raise",
            ),
            calibration=PlaceMoECalibration(
                inter_ms_per_byte=_legacy_positive_float(
                    env, "VEOMNI_HIERMOE_PLACEMOE_INTER_MS_PER_BYTE", _DEFAULT_INTER_MS_PER_BYTE
                ),
                intra_ms_per_byte=_legacy_positive_float(
                    env, "VEOMNI_HIERMOE_PLACEMOE_INTRA_MS_PER_BYTE", _DEFAULT_INTRA_MS_PER_BYTE
                ),
                route_ms_per_assignment=_legacy_positive_float(
                    env,
                    "VEOMNI_HIERMOE_PLACEMOE_ROUTE_MS_PER_ASSIGNMENT",
                    _DEFAULT_ROUTE_MS_PER_ASSIGNMENT,
                ),
                communication_multiplier=_legacy_positive_float(
                    env,
                    "VEOMNI_HIERMOE_PLACEMOE_COMMUNICATION_MULTIPLIER",
                    _DEFAULT_COMMUNICATION_MULTIPLIER,
                ),
                compute_ms_per_assignment=_legacy_positive_float(
                    env,
                    "VEOMNI_HIERMOE_PLACEMOE_COMPUTE_MS_PER_ASSIGNMENT",
                    _DEFAULT_COMPUTE_MS_PER_ASSIGNMENT,
                ),
                compute_multiplier=_legacy_positive_float(
                    env,
                    "VEOMNI_HIERMOE_PLACEMOE_COMPUTE_MULTIPLIER",
                    _DEFAULT_COMPUTE_MULTIPLIER,
                ),
            ),
            resources=PlaceMoEPlannerResources(
                workers=_legacy_positive_int(env, "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_WORKERS", 48),
                candidate_workers=_legacy_positive_int(
                    env, "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_CANDIDATE_WORKERS", 4
                ),
                worker_threads=_legacy_positive_int(
                    env, "VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_WORKER_THREADS", 1
                ),
                planner_cpu_ids=env.get("VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_CPU_IDS", "").strip(),
                training_cpu_ids=env.get("VEOMNI_HIERMOE_PERIODIC_FULL_REPLAN_TRAIN_CPU_IDS", "").strip(),
            ),
        )
        # Legacy inputs deliberately preserve their historical permissive
        # parsing. Strict validation is applied to the canonical file path.
        return result

    def validate(self) -> None:
        if self.hot_update.enabled:
            if not self.initial_artifact:
                raise PlaceMoEConfigurationError(
                    "initial_artifact is required when PlaceMoE hot updates are enabled."
                )
            if not self.hot_update.work_root:
                raise PlaceMoEConfigurationError("hot_update.work_root must not be empty.")


__all__ = [
    "HotUpdateConfig",
    "PlaceMoECalibration",
    "PlaceMoEConfigurationError",
    "PlaceMoEPlannerResources",
    "PlaceMoERuntimeConfig",
]
