"""Typed configuration for PlaceMoE training-time updates.

The canonical interface is the nested ``train.hiermoe.placemoe`` block in the
VeOmni training YAML. A standalone YAML/JSON file and
``VEOMNI_PLACEMOE_CONFIG`` remain compatibility inputs for archived launchers.
"""

from __future__ import annotations

import dataclasses
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
    require_scope: bool = False
    expected_scope: Mapping[str, Any] = field(default_factory=dict)
    artifact_scope: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_directory: Path) -> "PlaceMoECalibration":
        data = dict(payload)
        artifact = _resolve_path(data.pop("artifact", ""), base_directory, "calibration.artifact")
        require_scope = _strict_bool(data.pop("require_scope", False), "calibration.require_scope")
        expected_scope = dict(_mapping(data.pop("expected_scope", None), "calibration.expected_scope"))
        data = {key: value for key, value in data.items() if value is not None}
        if require_scope and not artifact:
            raise PlaceMoEConfigurationError("calibration.artifact is required when require_scope is true.")
        if require_scope and not expected_scope:
            raise PlaceMoEConfigurationError(
                "calibration.expected_scope must not be empty when require_scope is true."
            )

        artifact_scope: Mapping[str, Any] = {}
        if artifact:
            artifact_payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
            if not isinstance(artifact_payload, Mapping):
                raise PlaceMoEConfigurationError(f"calibration artifact {artifact} must contain a mapping.")
            if artifact_payload.get("status", "accepted") != "accepted":
                raise PlaceMoEConfigurationError(
                    f"calibration artifact {artifact} has status "
                    f"{artifact_payload.get('status')!r}, expected 'accepted'."
                )
            raw_artifact_scope = artifact_payload.get("scope")
            if isinstance(raw_artifact_scope, Mapping):
                artifact_scope = dict(raw_artifact_scope)
            elif require_scope:
                _mapping(raw_artifact_scope, "calibration artifact scope")
            else:
                artifact_scope = {}
            if require_scope:
                missing = sorted(set(expected_scope) - set(artifact_scope))
                mismatched = sorted(
                    key
                    for key, expected in expected_scope.items()
                    if key in artifact_scope and artifact_scope[key] != expected
                )
                if missing or mismatched:
                    details = []
                    if missing:
                        details.append(f"missing keys {missing}")
                    if mismatched:
                        mismatch_values = {
                            key: {"expected": expected_scope[key], "actual": artifact_scope[key]} for key in mismatched
                        }
                        details.append(f"mismatched values {mismatch_values}")
                    raise PlaceMoEConfigurationError(
                        f"calibration artifact {artifact} does not match expected scope: {'; '.join(details)}."
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
            require_scope=require_scope,
            expected_scope=expected_scope,
            artifact_scope=artifact_scope,
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
        result = cls(
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
        if bool(result.planner_cpu_ids) != bool(result.training_cpu_ids):
            raise PlaceMoEConfigurationError(
                "resources.planner_cpu_ids and resources.training_cpu_ids must be configured together."
            )
        return result


@dataclass(frozen=True)
class HotUpdateConfig:
    enabled: bool = False
    layout_interval_steps: int = 0
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
        layout_interval = _nonnegative_int(
            payload.get("layout_interval_steps", defaults.layout_interval_steps),
            "hot_update.layout_interval_steps",
        )
        failure_policy = str(payload.get("failure_policy", defaults.failure_policy)).strip().lower()
        if failure_policy not in {"continue", "raise"}:
            raise PlaceMoEConfigurationError("hot_update.failure_policy must be 'continue' or 'raise'.")
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
    """Complete runtime configuration with a strict file interface."""

    enabled: bool = False
    initial_artifact: str = ""
    runtime_perf_model: str = ""
    hot_update: HotUpdateConfig = field(default_factory=HotUpdateConfig)
    calibration: PlaceMoECalibration = field(default_factory=PlaceMoECalibration)
    resources: PlaceMoEPlannerResources = field(default_factory=PlaceMoEPlannerResources)
    source_path: str = ""

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_directory: str | os.PathLike[str] | None = None,
        source_path: str = "",
    ) -> "PlaceMoERuntimeConfig":
        root = _mapping(payload, "PlaceMoE config")
        if "placemoe" in root:
            root = _mapping(root["placemoe"], "placemoe")
        allowed = {"enabled", "initial_artifact", "runtime_perf_model", "hot_update", "calibration", "resources"}
        unknown = sorted(set(root) - allowed)
        if unknown:
            raise PlaceMoEConfigurationError(f"unknown PlaceMoE fields: {', '.join(unknown)}.")
        resolved_base = Path.cwd() if base_directory is None else Path(base_directory).expanduser().resolve()
        result = cls(
            enabled=_strict_bool(root.get("enabled", False), "enabled"),
            initial_artifact=_resolve_path(root.get("initial_artifact", ""), resolved_base, "initial_artifact"),
            runtime_perf_model=_resolve_path(root.get("runtime_perf_model", ""), resolved_base, "runtime_perf_model"),
            hot_update=HotUpdateConfig.from_mapping(
                _mapping(root.get("hot_update"), "hot_update"), base_directory=resolved_base
            ),
            calibration=PlaceMoECalibration.from_mapping(
                _mapping(root.get("calibration"), "calibration"), base_directory=resolved_base
            ),
            resources=PlaceMoEPlannerResources.from_mapping(_mapping(root.get("resources"), "resources")),
            source_path=source_path,
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "PlaceMoERuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise PlaceMoEConfigurationError(f"PlaceMoE config does not exist: {config_path}.")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls.from_mapping(payload, base_directory=config_path.parent, source_path=str(config_path))

    @classmethod
    def from_training_config(cls, config: Any) -> "PlaceMoERuntimeConfig":
        """Build the runtime config from ``train.hiermoe.placemoe``.

        ``config_path`` keeps existing standalone files working.  Otherwise
        the nested training YAML is the single source of truth.
        """

        if config is None:
            return cls()
        payload = (
            dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(_mapping(config, "placemoe"))
        )
        config_path = str(payload.pop("config_path", "") or "").strip()
        base_directory = str(payload.pop("base_directory", "") or "").strip()
        if config_path:
            if _training_payload_has_runtime_values(payload, config):
                raise PlaceMoEConfigurationError(
                    "train.hiermoe.placemoe.config_path is a legacy, exclusive input; "
                    "remove the inline PlaceMoE fields or remove config_path."
                )
            path = Path(os.path.expandvars(os.path.expanduser(config_path)))
            if not path.is_absolute() and base_directory:
                path = Path(base_directory).expanduser() / path
            return cls.from_file(path)
        return cls.from_mapping(
            payload,
            base_directory=base_directory or Path.cwd(),
            source_path="train.hiermoe.placemoe",
        )

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "PlaceMoERuntimeConfig":
        env = os.environ if environment is None else environment
        config_path = env.get("VEOMNI_PLACEMOE_CONFIG", "").strip()
        if config_path:
            return cls.from_file(config_path)
        return cls()

    def validate(self) -> None:
        if self.hot_update.enabled:
            if not self.hot_update.work_root:
                raise PlaceMoEConfigurationError("hot_update.work_root must not be empty.")


def _training_payload_has_runtime_values(payload: Mapping[str, Any], config: Any) -> bool:
    """Return whether a training config changes any field beside file location.

    Dataclass instances always contain every default, so comparing them with a
    fresh instance distinguishes a real inline configuration from parser
    defaults.  Mapping inputs are normally hand-written and therefore treat
    every remaining key as explicit.
    """

    if dataclasses.is_dataclass(config):
        try:
            defaults = dataclasses.asdict(type(config)())
        except TypeError:
            return True
        defaults.pop("config_path", None)
        defaults.pop("base_directory", None)
        return dict(payload) != defaults
    return bool(payload)


def training_config_is_explicit(config: Any) -> bool:
    """Whether ``train.hiermoe.placemoe`` contains non-default input."""

    if config is None:
        return False
    if dataclasses.is_dataclass(config):
        try:
            return dataclasses.asdict(config) != dataclasses.asdict(type(config)())
        except TypeError:
            return True
    return bool(_mapping(config, "placemoe"))


_CURRENT_RUNTIME_CONFIG = PlaceMoERuntimeConfig()


def set_current_runtime_config(config: PlaceMoERuntimeConfig) -> None:
    global _CURRENT_RUNTIME_CONFIG
    config.validate()
    _CURRENT_RUNTIME_CONFIG = config


def get_current_runtime_config() -> PlaceMoERuntimeConfig:
    return _CURRENT_RUNTIME_CONFIG


__all__ = [
    "HotUpdateConfig",
    "PlaceMoECalibration",
    "PlaceMoEConfigurationError",
    "PlaceMoEPlannerResources",
    "PlaceMoERuntimeConfig",
    "get_current_runtime_config",
    "set_current_runtime_config",
    "training_config_is_explicit",
]
