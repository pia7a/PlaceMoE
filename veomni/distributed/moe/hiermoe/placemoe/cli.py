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
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.runtime_bridge import MOE_RUNTIME_BRIDGE_API_VERSION, load_moe_runtime_bridge
from veomni.utils.device import get_device_type, get_torch_device

from .calibration import (
    CalibrationThresholds,
    ModelCalibrationError,
    ModelCalibrationSchedule,
    build_planner_calibration_artifact,
    load_local_phase_timing_summary,
    materialize_model_calibration_config,
    sha256_path,
    validate_runtime_performance_model,
)
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
    train = _mapping(root.get("train"))
    hiermoe = _mapping(train.get("hiermoe"))
    placemoe = _mapping(hiermoe.get("placemoe"))
    runtime = PlaceMoERuntimeConfig.from_training_config(placemoe)
    hierarchy = tuple(int(value) for value in hiermoe.get("hierarchy_group_sizes", ()) or ())
    model = _mapping(root.get("model"))
    model_path = str(model.get("model_path") or model.get("config_path") or "").rstrip("/")
    if runtime.calibration.artifact:
        ranks_per_node = int(os.environ.get("NPROC_PER_NODE", hierarchy[0] if hierarchy else 0) or 0)
        runtime.calibration.validate_artifact_scope(
            {
                "model_id": Path(model_path).name,
                "ep_size": int(_mapping(train.get("accelerator")).get("ep_size", 0) or 0),
                "ranks_per_node": ranks_per_node,
                "hierarchy_group_sizes": list(hierarchy),
            }
        )
    return root, runtime


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
    try:
        runtime_bridge = load_moe_runtime_bridge("placemoe")
    except (ImportError, LookupError, RuntimeError, TypeError, ValueError) as error:
        results.append(CheckResult("runtime_bridge", "FAIL", str(error)))
    else:
        results.append(
            CheckResult(
                "runtime_bridge",
                "PASS",
                f"provider={runtime_bridge.name}, API={MOE_RUNTIME_BRIDGE_API_VERSION}",
            )
        )

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
        detail = str(error)
        if "calibration artifact" in detail:
            detail += "; generate it with `placemoe calibrate-model`"
        results = [CheckResult("configuration", "FAIL", detail)]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        width = max(len(result.name) for result in results)
        for result in results:
            print(f"{result.status:4}  {result.name:<{width}}  {result.detail}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


def _distributed_environment() -> tuple[int, int, int, str, int]:
    try:
        nnodes = int(os.environ.get("NNODES", "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        nproc_per_node = int(os.environ.get("NPROC_PER_NODE", "8"))
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
    except ValueError as error:
        raise ModelCalibrationError("NNODES, NODE_RANK, NPROC_PER_NODE, and MASTER_PORT must be integers") from error
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1").strip()
    if nnodes < 1 or nproc_per_node < 1 or not 0 <= node_rank < nnodes:
        raise ModelCalibrationError(
            f"invalid distributed launch: NNODES={nnodes}, NODE_RANK={node_rank}, "
            f"NPROC_PER_NODE={nproc_per_node}"
        )
    if not master_addr or not 0 < master_port < 65536:
        raise ModelCalibrationError("MASTER_ADDR and MASTER_PORT must identify a valid rendezvous endpoint")
    return nnodes, node_rank, nproc_per_node, master_addr, master_port


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _stream_training(command: list[str], *, environment: Mapping[str, str], log_path: Path) -> int:
    """Run torchrun while preserving a complete log and live terminal output."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as writer:
        process = subprocess.Popen(
            command,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                writer.write(line)
                writer.flush()
                print(line, end="", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
        return int(process.wait())


def _exchange_phase_timing_summaries(
    summary: Mapping[str, Any],
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> tuple[list[dict[str, Any]], Any | None]:
    """Exchange one timing summary per node after the training workers exit."""

    if nnodes == 1:
        return [dict(summary)], None
    if master_port >= 65535:
        raise ModelCalibrationError("MASTER_PORT must be below 65535 for calibration summary exchange")
    import torch.distributed as dist

    store = dist.TCPStore(
        master_addr,
        master_port + 1,
        nnodes,
        node_rank == 0,
        timeout=timedelta(minutes=5),
        wait_for_workers=True,
    )
    prefix = "placemoe-model-calibration-phase-summary"
    store.set(f"{prefix}/{node_rank}", json.dumps(dict(summary), sort_keys=True))
    keys = [f"{prefix}/{rank}" for rank in range(nnodes)]
    store.wait(keys)
    return [json.loads(store.get(key).decode("utf-8")) for key in keys], store


def _publish_calibration_result(store: Any | None, result: Mapping[str, Any]) -> None:
    if store is not None:
        store.set("placemoe-model-calibration-result", json.dumps(dict(result), sort_keys=True))


def _wait_for_calibration_result(store: Any) -> dict[str, Any]:
    key = "placemoe-model-calibration-result"
    store.wait([key])
    return json.loads(store.get(key).decode("utf-8"))


def _calibrate_model_command(args: argparse.Namespace) -> int:
    try:
        config_path = Path(args.config).expanduser().resolve()
        entrypoint = Path(args.entrypoint).expanduser().resolve()
        runtime_perf_model_path = Path(args.runtime_perf_model).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        if not config_path.is_file():
            raise ModelCalibrationError(f"training config not found: {config_path}")
        if not entrypoint.is_file():
            raise ModelCalibrationError(f"training entrypoint not found: {entrypoint}")
        if not runtime_perf_model_path.is_file():
            raise ModelCalibrationError(f"runtime performance model not found: {runtime_perf_model_path}")

        source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(source, Mapping):
            raise ModelCalibrationError("training config must contain a mapping")
        OpsImplementationConfig(**dict(_mapping(_mapping(source.get("model")).get("ops_implementation"))))
        nnodes, node_rank, nproc_per_node, master_addr, master_port = _distributed_environment()
        ep_size = int(_mapping(_mapping(source.get("train")).get("accelerator")).get("ep_size", 0) or 0)
        world_size = nnodes * nproc_per_node
        if ep_size <= 1 or world_size != ep_size:
            raise ModelCalibrationError(
                f"model calibration requires exactly one EP group with ep_size > 1; "
                f"config ep_size={ep_size}, world_size={world_size}"
            )
        hierarchy = tuple(
            int(value)
            for value in _mapping(_mapping(source.get("train")).get("hiermoe")).get(
                "hierarchy_group_sizes", ()
            )
        )
        expected_hierarchy = (nproc_per_node, ep_size)
        if hierarchy != expected_hierarchy:
            raise ModelCalibrationError(
                f"model calibration requires hierarchy_group_sizes={expected_hierarchy}, got {hierarchy}"
            )
        runtime_payload = json.loads(runtime_perf_model_path.read_text(encoding="utf-8"))
        validate_runtime_performance_model(
            _mapping(runtime_payload),
            ep_size=ep_size,
            ranks_per_node=nproc_per_node,
            hierarchy_group_sizes=hierarchy,
        )

        schedule = ModelCalibrationSchedule(
            warmup_steps=int(args.warmup_steps),
            validation_steps=int(args.validation_steps),
        )
        thresholds = CalibrationThresholds(
            compute_mape_percent=float(args.compute_mape_threshold),
            communication_mape_percent=float(args.communication_mape_threshold),
            joint_mape_percent=float(args.joint_mape_threshold),
        )
        work_root = (
            Path(args.work_directory).expanduser().resolve()
            if args.work_directory
            else output_path.parent / f".{output_path.stem}.work"
        )
        node_work = work_root / f"node-{node_rank}"
        timing_directory = node_work / "timing"
        timing_directory.mkdir(parents=True, exist_ok=True)
        for path in timing_directory.glob("moe_timing_rank*.jsonl"):
            path.unlink()
        derived_config = materialize_model_calibration_config(
            source,
            runtime_perf_model=runtime_perf_model_path,
            work_directory=node_work,
            schedule=schedule,
        )
        derived_config_path = node_work / "training.yaml"
        _atomic_write(derived_config_path, yaml.safe_dump(derived_config, sort_keys=False))

        python = os.environ.get("PLACEMOE_PYTHON", sys.executable)
        command = [
            python,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
            str(entrypoint),
            str(derived_config_path),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "VERL_MOE_TIMING_DIR": str(timing_directory),
                "VEOMNI_HIERMOE_COST_MODEL_VERIFY": "1",
                "VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES": "1",
                "VEOMNI_HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP": str(schedule.calibration_step),
                "VEOMNI_HIERMOE_COST_MODEL_VALIDATION_STEPS": str(schedule.validation_steps),
                "VEOMNI_HIERMOE_INTERNAL_TIMING": "1",
                "VEOMNI_HIERMOE_FIXED_R2_LAYOUT": "0",
                "VEOMNI_HIERMOE_CPU_PLANNER_MODE": "off",
                "VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE": "off",
                "VEOMNI_HIERMOE_FORWARD_REUSE_COVER": "0",
                "VEOMNI_HIERMOE_ONLINE_LUT_UPDATE": "0",
                "VEOMNI_HIERMOE_NPU_LAYER_OWNER_BLOCKING": "0",
                "VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS": "0",
            }
        )
        environment.pop("VEOMNI_PLACEMOE_CONFIG", None)
        log_path = node_work / "training.log"
        print(
            f"PlaceMoE model calibration: {schedule.max_steps} steps "
            f"({schedule.warmup_steps} warm-up, 1 fit, {schedule.validation_steps} held-out)"
        )
        return_code = _stream_training(command, environment=environment, log_path=log_path)
        if return_code:
            raise ModelCalibrationError(f"calibration training failed with exit code {return_code}; see {log_path}")
        expected_timing_steps = range(schedule.calibration_step + 1, schedule.max_steps + 1)
        local_summary = load_local_phase_timing_summary(
            timing_directory,
            expected_ranks=range(node_rank * nproc_per_node, (node_rank + 1) * nproc_per_node),
            expected_steps=expected_timing_steps,
        )
        phase_summaries, coordination_store = _exchange_phase_timing_summaries(
            local_summary,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        if node_rank != 0:
            if coordination_store is None:
                raise ModelCalibrationError("distributed calibration coordination store is unavailable")
            result = _wait_for_calibration_result(coordination_store)
            if int(result["return_code"]):
                print(f"PlaceMoE model calibration failed: {result['message']}", file=sys.stderr)
            else:
                print(str(result["message"]))
            return int(result["return_code"])

        try:
            artifact = build_planner_calibration_artifact(
                training_config=source,
                runtime_perf_model=_mapping(runtime_payload),
                runtime_perf_model_sha256=sha256_path(runtime_perf_model_path),
                training_log_text=log_path.read_text(encoding="utf-8"),
                training_log_sha256=sha256_path(log_path),
                phase_timing_summaries=phase_summaries,
                ranks_per_node=nproc_per_node,
                schedule=schedule,
                thresholds=thresholds,
                model_id=args.model_id,
            )
            _atomic_write(output_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
            status = str(artifact["status"])
            return_code = 0 if status == "accepted" else 2
            message = f"PlaceMoE planner calibration {status}: {output_path}"
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, ModelCalibrationError) as error:
            return_code = 2
            message = str(error)
        _publish_calibration_result(
            coordination_store,
            {"return_code": return_code, "message": message},
        )
        if return_code:
            print(f"PlaceMoE model calibration failed: {message}", file=sys.stderr)
        else:
            print(message)
        return return_code
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError, ModelCalibrationError) as error:
        print(f"PlaceMoE model calibration failed: {error}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placemoe", description="PlaceMoE deployment and validation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="validate a training configuration and local NPU environment")
    doctor.add_argument("--config", required=True, help="VeOmni training YAML or standalone PlaceMoE YAML/JSON")
    doctor.add_argument("--allow-cpu", action="store_true", help="do not require torch_npu or visible NPUs")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable results")
    doctor.set_defaults(handler=_doctor_command)
    calibrate = subparsers.add_parser(
        "calibrate-model",
        help="fit and validate planner coefficients with a short default-layout training run",
    )
    calibrate.add_argument("--config", required=True, help="normal VeOmni training YAML")
    calibrate.add_argument("--entrypoint", required=True, help="VeOmni training entrypoint for the model type")
    calibrate.add_argument("--runtime-perf-model", required=True, help="topology calibration JSON")
    calibrate.add_argument("--output", required=True, help="planner calibration JSON written by NODE_RANK=0")
    calibrate.add_argument("--work-directory", help="directory for the derived YAML, logs, and timing samples")
    calibrate.add_argument("--model-id", help="scope identifier; defaults to the model directory name")
    calibrate.add_argument("--warmup-steps", type=int, default=2, help="warm-up steps before fitting (default: 2)")
    calibrate.add_argument(
        "--validation-steps", type=int, default=2, help="held-out validation steps after fitting (default: 2)"
    )
    calibrate.add_argument("--compute-mape-threshold", type=float, default=5.0)
    calibrate.add_argument("--communication-mape-threshold", type=float, default=10.0)
    calibrate.add_argument("--joint-mape-threshold", type=float, default=10.0)
    calibrate.set_defaults(handler=_calibrate_model_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
