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

"""Preflight command for portable PlaceMoE deployments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.utils.device import get_device_type, get_torch_device

from .runtime import PlaceMoERuntimeConfig


_SUPPORTED_PYTHON = (3, 11)
_VALIDATED_TORCH = "2.9.0"
_VALIDATED_TORCH_NPU = "2.9.0.post2"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_user_path(value: Any, base_directory: Path) -> Path | None:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        return None
    path = Path(raw)
    return (path if path.is_absolute() else base_directory / path).resolve()


def _load_training_config(path: Path) -> tuple[Mapping[str, Any], PlaceMoERuntimeConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(payload)
    if "train" not in root:
        return root, PlaceMoERuntimeConfig.from_file(path)
    placemoe = _mapping(_mapping(_mapping(root.get("train")).get("hiermoe")).get("placemoe"))
    return root, PlaceMoERuntimeConfig.from_training_config(placemoe)


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return None
    return str(getattr(module, "__version__", "unknown"))


def _matches_validated_version(actual: str | None, expected: str) -> bool:
    """Accept platform-local wheel tags without weakening the pinned release."""

    return actual is not None and actual.split("+", 1)[0] == expected


def _path_check(name: str, path: Path | None, *, required: bool) -> CheckResult:
    if path is None:
        status = "FAIL" if required else "SKIP"
        return CheckResult(name, status, "not configured")
    if path.exists():
        return CheckResult(name, "PASS", str(path))
    return CheckResult(name, "FAIL", f"not found: {path}")


def _performance_model_check(
    path: Path | None,
    *,
    required: bool,
    expected_ep_size: int = 0,
    expected_hierarchy: tuple[int, ...] = (),
) -> CheckResult:
    if path is None:
        return CheckResult("performance_model_schema", "FAIL" if required else "SKIP", "not configured")
    if not path.is_file():
        return CheckResult("performance_model_schema", "FAIL", f"not found: {path}")
    try:
        model = HierMoEPerfModel.from_path(str(path))
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return CheckResult("performance_model_schema", "FAIL", f"invalid artifact: {error}")
    metadata = _mapping(payload.get("metadata"))
    artifact_ep_size = int(metadata.get("ep_size", metadata.get("world_size", 0)) or 0)
    artifact_hierarchy = tuple(int(value) for value in metadata.get("hierarchy_group_sizes", ()) or ())
    mismatches = []
    if expected_ep_size and artifact_ep_size and expected_ep_size != artifact_ep_size:
        mismatches.append(f"ep_size={artifact_ep_size}, expected {expected_ep_size}")
    if expected_hierarchy and artifact_hierarchy and expected_hierarchy != artifact_hierarchy:
        mismatches.append(f"hierarchy={artifact_hierarchy}, expected {expected_hierarchy}")
    if mismatches:
        return CheckResult("performance_model_schema", "FAIL", "; ".join(mismatches))
    status = "PASS" if model.runtime_cost_status == "complete" else "WARN"
    return CheckResult(
        "performance_model_schema",
        status,
        f"schema={model.schema_version}, runtime placement costs={model.runtime_cost_status}",
    )


def run_doctor(config_path: Path, *, require_npu: bool) -> list[CheckResult]:
    root, runtime = _load_training_config(config_path)
    is_training_config = "train" in root
    base_directory = config_path.parent.resolve()
    train = _mapping(root.get("train"))
    accelerator = _mapping(train.get("accelerator"))
    hiermoe = _mapping(train.get("hiermoe"))
    placemoe = _mapping(hiermoe.get("placemoe"))
    model = _mapping(root.get("model"))
    data = _mapping(root.get("data"))
    results = [
        CheckResult(
            "python",
            "PASS" if sys.version_info[:2] == _SUPPORTED_PYTHON else "FAIL",
            f"{platform.python_version()} (validated: 3.11)",
        ),
        CheckResult("architecture", "PASS", platform.machine()),
    ]

    torch_version = _version("torch")
    results.append(
        CheckResult(
            "torch",
            "PASS" if _matches_validated_version(torch_version, _VALIDATED_TORCH) else "FAIL",
            f"{torch_version or 'not installed'} (validated: {_VALIDATED_TORCH})",
        )
    )
    torch_npu_version = _version("torch_npu")
    npu_status = "PASS" if torch_npu_version == _VALIDATED_TORCH_NPU else ("FAIL" if require_npu else "WARN")
    results.append(
        CheckResult(
            "torch_npu",
            npu_status,
            f"{torch_npu_version or 'not installed'} (validated: {_VALIDATED_TORCH_NPU})",
        )
    )

    if torch_npu_version is not None:
        namespace = get_torch_device()
        device_count = int(namespace.device_count()) if get_device_type() == "npu" else 0
        available = get_device_type() == "npu" and bool(namespace.is_available()) and device_count > 0
        results.append(CheckResult("npu_devices", "PASS" if available else "FAIL", f"visible devices: {device_count}"))
    elif require_npu:
        results.append(CheckResult("npu_devices", "FAIL", "torch_npu is unavailable"))

    cann_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    results.append(_path_check("cann", cann_home, required=require_npu))

    model_path = _resolve_user_path(model.get("model_path") or model.get("config_path"), base_directory)
    data_path = _resolve_user_path(data.get("train_path"), base_directory)
    performance_model_path = (
        Path(runtime.runtime_perf_model)
        if runtime.runtime_perf_model
        else _resolve_user_path(hiermoe.get("perf_model_path"), base_directory)
    )
    performance_model_required = bool(is_training_config and placemoe.get("enabled", False))
    results.extend(
        (
            _path_check("model", model_path, required=is_training_config),
            _path_check("dataset", data_path, required=is_training_config),
            _path_check(
                "performance_model",
                performance_model_path,
                required=performance_model_required,
            ),
            _path_check(
                "initial_artifact",
                Path(runtime.initial_artifact) if runtime.initial_artifact else None,
                required=False,
            ),
        )
    )
    results.append(
        _performance_model_check(
            performance_model_path,
            required=performance_model_required,
            expected_ep_size=int(accelerator.get("ep_size", 0) or 0),
            expected_hierarchy=tuple(int(value) for value in hiermoe.get("hierarchy_group_sizes", ()) or ()),
        )
    )
    if is_training_config:
        try:
            OpsImplementationConfig(**dict(_mapping(model.get("ops_implementation"))))
        except (TypeError, ValueError) as error:
            results.append(CheckResult("model_ops", "FAIL", str(error)))
        else:
            results.append(CheckResult("model_ops", "PASS", "kernel backends are valid on this platform"))
        enabled = bool(placemoe.get("enabled", False))
        results.append(
            CheckResult(
                "placemoe_runtime",
                "PASS" if enabled else "FAIL",
                "canonical preset enabled" if enabled else "set train.hiermoe.placemoe.enabled: true",
            )
        )
        redundant_slots = int(hiermoe.get("redundant_slot_increment_per_device", 0) or 0)
        results.append(
            CheckResult(
                "replica_slots",
                "PASS" if redundant_slots > 0 else "FAIL",
                f"{redundant_slots} additional slots per EP rank",
            )
        )
    raw_calibration = _mapping(placemoe.get("calibration"))
    coefficient_names = {
        "inter_ms_per_byte",
        "intra_ms_per_byte",
        "route_ms_per_assignment",
        "communication_multiplier",
        "compute_ms_per_assignment",
        "compute_multiplier",
    }
    if runtime.calibration.artifact:
        calibration_path = Path(runtime.calibration.artifact)
        calibration_path_result = _path_check("calibration", calibration_path, required=True)
        results.append(calibration_path_result)
        if calibration_path_result.status == "PASS":
            try:
                artifact_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                results.append(CheckResult("calibration_schema", "FAIL", f"invalid artifact: {error}"))
            else:
                artifact_coefficients = _mapping(_mapping(artifact_payload).get("coefficients", artifact_payload))
                missing_coefficients = sorted(coefficient_names - set(artifact_coefficients))
                results.append(
                    CheckResult(
                        "calibration_schema",
                        "PASS" if not missing_coefficients else "FAIL",
                        "complete coefficients"
                        if not missing_coefficients
                        else "missing " + ", ".join(missing_coefficients),
                    )
                )
        else:
            results.append(CheckResult("calibration_schema", "FAIL", "artifact is unavailable"))
    elif is_training_config and runtime.enabled:
        missing_coefficients = sorted(coefficient_names - set(raw_calibration))
        results.append(
            CheckResult(
                "calibration",
                "PASS" if not missing_coefficients else "FAIL",
                "explicit coefficients"
                if not missing_coefficients
                else "configure calibration.artifact or all coefficients; missing " + ", ".join(missing_coefficients),
            )
        )
    else:
        results.append(CheckResult("calibration", "SKIP", "PlaceMoE training is not enabled"))
    static_updates = not (runtime.hot_update.layout_interval_steps or runtime.hot_update.mapping_interval_steps)
    results.append(
        CheckResult(
            "hot_update",
            "PASS",
            "static"
            if not runtime.hot_update.enabled or static_updates
            else (
                f"layout={runtime.hot_update.layout_interval_steps or 0}, "
                f"mapping={runtime.hot_update.mapping_interval_steps} steps"
            ),
        )
    )
    if (
        is_training_config
        and runtime.enabled
        and not runtime.initial_artifact
        and not runtime.hot_update.layout_interval_steps
    ):
        results.append(
            CheckResult(
                "startup_plan",
                "FAIL",
                "no initial artifact: configure a positive layout update interval",
            )
        )
    return results


def _doctor_command(args: argparse.Namespace) -> int:
    try:
        results = run_doctor(Path(args.config).expanduser().resolve(), require_npu=not args.allow_cpu)
    except (OSError, ValueError, yaml.YAMLError) as error:
        results = [CheckResult("configuration", "FAIL", str(error))]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        width = max(len(result.name) for result in results)
        for result in results:
            print(f"{result.status:4}  {result.name:<{width}}  {result.detail}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placemoe", description="PlaceMoE deployment and validation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="validate a training configuration and local NPU environment")
    doctor.add_argument("--config", required=True, help="VeOmni training YAML or standalone PlaceMoE YAML/JSON")
    doctor.add_argument("--allow-cpu", action="store_true", help="do not require torch_npu or visible NPUs")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable results")
    doctor.set_defaults(handler=_doctor_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
