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

"""Calibration, preparation, and preflight commands for PlaceMoE deployments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import expected_hierarchy_group_sizes
from veomni.distributed.moe.runtime_bridge import MOE_RUNTIME_BRIDGE_API_VERSION, load_moe_runtime_bridge
from veomni.utils.device import get_device_type, get_torch_device

from .calibration import (
    ModelCalibrationError,
    ModelCalibrationSchedule,
    build_planner_calibration_artifact,
    load_local_phase_timing_summary,
    materialize_model_calibration_config,
    sha256_path,
    validate_runtime_performance_model,
)
from .preparation import (
    CacheDecision,
    CacheInspection,
    build_preparation_spec,
    decide_cache_action,
    fingerprint_calibration_inputs,
    inspect_planner_cache,
    inspect_runtime_cache,
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
    elif is_training_config and runtime.enabled and runtime.calibration.auto_generate:
        accelerator = _mapping(train.get("accelerator"))
        ep_size = int(accelerator.get("ep_size", 0) or 0)
        hierarchy = tuple(int(value) for value in hiermoe.get("hierarchy_group_sizes", ()) or ())
        issues = []
        if not bool(hiermoe.get("enable", False)):
            issues.append("train.hiermoe.enable must be true")
        if not bool(hiermoe.get("token_dedup", True)):
            issues.append("train.hiermoe.token_dedup must be true")
        if not bool(hiermoe.get("expert_swap", True)):
            issues.append("train.hiermoe.expert_swap must be true")
        if ep_size <= 1:
            issues.append("train.accelerator.ep_size must be greater than 1")
        if not hierarchy:
            issues.append("train.hiermoe.hierarchy_group_sizes must be explicit")
        world_size_text = os.environ.get("WORLD_SIZE", "").strip()
        if not world_size_text:
            nnodes = os.environ.get("NNODES", "").strip()
            nproc_per_node = os.environ.get("NPROC_PER_NODE", "").strip()
            if nnodes and nproc_per_node:
                try:
                    world_size_text = str(int(nnodes) * int(nproc_per_node))
                except ValueError:
                    issues.append("NNODES and NPROC_PER_NODE must be integers")
        try:
            world_size = int(world_size_text) if world_size_text else None
        except ValueError:
            issues.append("WORLD_SIZE must be an integer")
            world_size = None
        if world_size is not None and world_size != ep_size:
            issues.append(f"exactly one EP group is required (world_size={world_size}, ep_size={ep_size})")
        status = "FAIL" if issues else ("PASS" if world_size is not None else "WARN")
        detail = (
            "; ".join(issues)
            if issues
            else (
                f"in-training report-only calibration -> {runtime.calibration.output}"
                if world_size is not None
                else "static prerequisites pass; set WORLD_SIZE or NNODES/NPROC_PER_NODE to verify one EP group"
            )
        )
        results.append(
            CheckResult(
                "calibration",
                status,
                detail,
            )
        )
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
            detail += "; generate it with `placemoe prepare` or `placemoe calibrate-model`"
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
            f"invalid distributed launch: NNODES={nnodes}, NODE_RANK={node_rank}, NPROC_PER_NODE={nproc_per_node}"
        )
    if not master_addr or not 0 < master_port < 65536:
        raise ModelCalibrationError("MASTER_ADDR and MASTER_PORT must identify a valid rendezvous endpoint")
    return nnodes, node_rank, nproc_per_node, master_addr, master_port


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
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


def _parse_hierarchy_csv(value: str) -> tuple[int, ...]:
    try:
        hierarchy = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ModelCalibrationError("hierarchy group sizes must be comma-separated integers") from error
    if not hierarchy or any(group_size <= 0 for group_size in hierarchy):
        raise ModelCalibrationError("hierarchy group sizes must be positive")
    return hierarchy


def _resolve_runtime_backend(requested: str, device_type: str) -> str:
    if requested != "auto":
        return requested
    return {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}.get(device_type, "gloo")


def _calibrate_runtime_command(args: argparse.Namespace) -> int:
    """Run standalone topology calibration through the canonical Python CLI."""

    try:
        output_path = Path(args.output).expanduser().resolve()
        hierarchy = _parse_hierarchy_csv(args.hierarchy_group_sizes_csv)
        nnodes, node_rank, nproc_per_node, master_addr, master_port = _distributed_environment()
        ep_size = nnodes * nproc_per_node
        expected_hierarchy = expected_hierarchy_group_sizes(ep_size, nproc_per_node)
        if hierarchy != expected_hierarchy:
            raise ModelCalibrationError(
                f"portable PlaceMoE calibration requires hierarchy_group_sizes={expected_hierarchy}, got {hierarchy}"
            )
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
            "-m",
            "placemoe.calibrate_network",
            "--output-json",
            str(output_path),
            "--ep-size",
            str(ep_size),
            "--ranks-per-node",
            str(nproc_per_node),
            "--hierarchy-group-sizes-csv",
            ",".join(str(value) for value in hierarchy),
            "--backend",
            str(args.backend),
            "--dtype",
            str(args.dtype),
            "--message-bytes-csv",
            str(args.message_bytes_csv),
            "--warmup",
            str(args.runtime_warmup),
            "--iters",
            str(args.runtime_iters),
            "--measure-last-n",
            str(args.runtime_measure_last_n),
        ]
        if args.details_json:
            command.extend(("--details-json", str(Path(args.details_json).expanduser().resolve())))
        work_root = (
            Path(args.work_directory).expanduser().resolve()
            if args.work_directory
            else output_path.parent / f".{output_path.stem}.work"
        )
        log_path = work_root / f"node-{node_rank}" / "runtime-calibration.log"
        return_code = _stream_training(command, environment=os.environ, log_path=log_path)
        if return_code:
            raise ModelCalibrationError(f"runtime calibration failed with exit code {return_code}; see {log_path}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        validate_runtime_performance_model(
            _mapping(payload),
            ep_size=ep_size,
            ranks_per_node=nproc_per_node,
            hierarchy_group_sizes=hierarchy,
        )
        print(f"PlaceMoE runtime calibration accepted: {output_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, ModelCalibrationError) as error:
        print(f"PlaceMoE runtime calibration failed: {error}", file=sys.stderr)
        return 2


def _open_preparation_store(
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    port_offset: int,
    timeout: timedelta | None = None,
) -> Any | None:
    if nnodes == 1:
        return None
    coordination_port = master_port + port_offset
    if coordination_port >= 65536:
        raise ModelCalibrationError(f"MASTER_PORT must be below {65536 - port_offset} for PlaceMoE preparation")
    import torch.distributed as dist

    return dist.TCPStore(
        master_addr,
        coordination_port,
        nnodes,
        node_rank == 0,
        timeout=timeout or timedelta(hours=3),
        wait_for_workers=True,
    )


def _read_coordinated_payload(
    store: Any,
    key: str,
    *,
    acknowledgment_prefix: str,
    nnodes: int,
    node_rank: int,
) -> bytes:
    """Keep the server store alive until every client has read its payload."""

    payload = store.get(key)
    if node_rank == 0:
        store.wait([f"{acknowledgment_prefix}/{rank}" for rank in range(1, nnodes)])
    else:
        store.set(f"{acknowledgment_prefix}/{node_rank}", "1")
    return payload


def _coordinate_cache_decision(
    inspection: CacheInspection,
    *,
    force: bool,
    stage: str,
    nnodes: int,
    node_rank: int,
    store: Any | None,
) -> CacheDecision:
    if store is None:
        return decide_cache_action((inspection,), force=force)
    prefix = f"placemoe-prepare/{stage}"
    local_key = f"{prefix}/inspection/{node_rank}"
    store.set(local_key, json.dumps(asdict(inspection), sort_keys=True))
    inspection_keys = [f"{prefix}/inspection/{rank}" for rank in range(nnodes)]
    store.wait(inspection_keys)
    decision_key = f"{prefix}/decision"
    if node_rank == 0:
        inspections = [CacheInspection(**json.loads(store.get(key).decode("utf-8"))) for key in inspection_keys]
        store.set(decision_key, json.dumps(asdict(decide_cache_action(inspections, force=force)), sort_keys=True))
    store.wait([decision_key])
    payload = _read_coordinated_payload(
        store,
        decision_key,
        acknowledgment_prefix=f"{prefix}/decision-read",
        nnodes=nnodes,
        node_rank=node_rank,
    )
    return CacheDecision(**json.loads(payload.decode("utf-8")))


def _coordinate_stage_result(
    inspection: CacheInspection,
    *,
    return_code: int,
    stage: str,
    nnodes: int,
    node_rank: int,
    store: Any | None,
) -> tuple[int, str]:
    if store is None:
        if return_code == 0 and inspection.state == "valid":
            return 0, inspection.detail
        return 2, inspection.detail
    prefix = f"placemoe-prepare/{stage}"
    result_key = f"{prefix}/result/{node_rank}"
    store.set(
        result_key,
        json.dumps(
            {"return_code": int(return_code), "inspection": asdict(inspection)},
            sort_keys=True,
        ),
    )
    result_keys = [f"{prefix}/result/{rank}" for rank in range(nnodes)]
    store.wait(result_keys)
    final_key = f"{prefix}/final"
    if node_rank == 0:
        results = [json.loads(store.get(key).decode("utf-8")) for key in result_keys]
        failures = [
            f"node {rank}: {row['inspection']['detail']}"
            for rank, row in enumerate(results)
            if int(row["return_code"]) != 0 or row["inspection"]["state"] != "valid"
        ]
        digests = {
            str(row["inspection"].get("digest") or "")
            for row in results
            if int(row["return_code"]) == 0 and row["inspection"]["state"] == "valid"
        }
        if not failures and ("" in digests or len(digests) != 1):
            failures.append("generated artifacts differ across nodes")
        final = {
            "return_code": 2 if failures else 0,
            "detail": "; ".join(failures) if failures else "valid on every node",
        }
        store.set(final_key, json.dumps(final, sort_keys=True))
    store.wait([final_key])
    payload = _read_coordinated_payload(
        store,
        final_key,
        acknowledgment_prefix=f"{prefix}/final-read",
        nnodes=nnodes,
        node_rank=node_rank,
    )
    final = json.loads(payload.decode("utf-8"))
    return int(final["return_code"]), str(final["detail"])


def _coordinate_preparation_preflight(
    local: Mapping[str, Any],
    *,
    nnodes: int,
    node_rank: int,
    store: Any | None,
) -> tuple[int, str]:
    if store is None:
        return (0, "local inputs validated") if local.get("ok") else (2, str(local.get("error")))
    prefix = "placemoe-prepare/preflight"
    local_key = f"{prefix}/{node_rank}"
    store.set(local_key, json.dumps(dict(local), sort_keys=True))
    keys = [f"{prefix}/{rank}" for rank in range(nnodes)]
    store.wait(keys)
    final_key = f"{prefix}/final"
    if node_rank == 0:
        rows = [json.loads(store.get(key).decode("utf-8")) for key in keys]
        failures = [f"node {rank}: {row.get('error')}" for rank, row in enumerate(rows) if not row.get("ok")]
        identities = {json.dumps(row.get("identity"), sort_keys=True) for row in rows if row.get("ok")}
        if not failures and len(identities) != 1:
            failures.append("preparation inputs differ across nodes")
        final = {
            "return_code": 2 if failures else 0,
            "detail": "; ".join(failures) if failures else "inputs match on every node",
        }
        store.set(final_key, json.dumps(final, sort_keys=True))
    store.wait([final_key])
    final = json.loads(store.get(final_key).decode("utf-8"))
    return int(final["return_code"]), str(final["detail"])


def _run_preparation_stage(
    *,
    stage: str,
    force: bool,
    inspect: Callable[[], CacheInspection],
    run: Callable[[], int],
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    port_offset: int,
) -> tuple[int, bool]:
    store = _open_preparation_store(
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        port_offset=port_offset,
    )
    try:
        initial_inspection = inspect()
    except Exception as error:
        initial_inspection = CacheInspection("invalid", f"unexpected {stage} cache failure: {error}")
    decision = _coordinate_cache_decision(
        initial_inspection,
        force=force,
        stage=stage,
        nnodes=nnodes,
        node_rank=node_rank,
        store=store,
    )
    if decision.action == "error":
        print(f"PlaceMoE {stage} cache is invalid: {decision.detail}", file=sys.stderr)
        return 2, False
    if decision.action == "reuse":
        print(f"PlaceMoE {stage} calibration reused: {decision.detail}")
        return 0, False
    print(f"PlaceMoE {stage} calibration running: {decision.detail}")
    try:
        return_code = run()
        post_inspection = inspect()
    except Exception as error:
        return_code = 2
        post_inspection = CacheInspection("invalid", f"unexpected {stage} failure: {error}")
    final_code, detail = _coordinate_stage_result(
        post_inspection,
        return_code=return_code,
        stage=stage,
        nnodes=nnodes,
        node_rank=node_rank,
        store=store,
    )
    if final_code:
        print(f"PlaceMoE {stage} calibration failed: {detail}", file=sys.stderr)
    else:
        print(f"PlaceMoE {stage} calibration ready: {detail}")
    return final_code, True


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
            for value in _mapping(_mapping(source.get("train")).get("hiermoe")).get("hierarchy_group_sizes", ())
        )
        expected_hierarchy = expected_hierarchy_group_sizes(ep_size, nproc_per_node)
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
            artifact_content = result.get("artifact_content")
            if isinstance(artifact_content, str):
                _atomic_write(output_path, artifact_content)
            if int(result["return_code"]):
                print(f"PlaceMoE model calibration failed: {result['message']}", file=sys.stderr)
            else:
                print(str(result["message"]))
            return int(result["return_code"])

        artifact_content: str | None = None
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
                model_id=args.model_id,
            )
            artifact["provenance"]["calibration_input_sha256"] = fingerprint_calibration_inputs(source, entrypoint)
            artifact_content = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
            _atomic_write(output_path, artifact_content)
            return_code = 0
            message = f"PlaceMoE planner calibration generated: {output_path}"
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, ModelCalibrationError) as error:
            return_code = 2
            message = str(error)
        _publish_calibration_result(
            coordination_store,
            {
                "return_code": return_code,
                "message": message,
                "artifact_content": artifact_content,
            },
        )
        if return_code:
            print(f"PlaceMoE model calibration failed: {message}", file=sys.stderr)
        else:
            print(message)
        return return_code
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError, ModelCalibrationError) as error:
        print(f"PlaceMoE model calibration failed: {error}", file=sys.stderr)
        return 2


def _prepare_command(args: argparse.Namespace) -> int:
    """Reuse valid calibration artifacts and create only missing or forced stages."""

    try:
        nnodes, node_rank, nproc_per_node, master_addr, master_port = _distributed_environment()
        source = None
        spec = None
        local_error = ""
        identity: dict[str, Any] = {}
        try:
            config_path = Path(args.config).expanduser().resolve()
            if not config_path.is_file():
                raise ModelCalibrationError(f"training config not found: {config_path}")
            source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(source, Mapping):
                raise ModelCalibrationError("training config must contain a mapping")
            OpsImplementationConfig(**dict(_mapping(_mapping(source.get("model")).get("ops_implementation"))))
            runtime_device_type = get_device_type()
            runtime_backend = _resolve_runtime_backend(args.runtime_backend, runtime_device_type)
            spec = build_preparation_spec(
                source,
                config_path=config_path,
                entrypoint=Path(args.entrypoint),
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                runtime_device_type=runtime_device_type,
                runtime_backend=runtime_backend,
                runtime_dtype=args.runtime_dtype,
                model_id=args.model_id,
            )
            identity = {
                "config_sha256": sha256_path(spec.config_path),
                "entrypoint_sha256": sha256_path(spec.entrypoint),
                "model_id": spec.model_id,
                "ep_size": spec.ep_size,
                "ranks_per_node": spec.ranks_per_node,
                "hierarchy_group_sizes": list(spec.hierarchy_group_sizes),
                "runtime_artifact": str(spec.runtime_artifact),
                "planner_artifact": str(spec.planner_artifact),
                "runtime_device_type": spec.runtime_device_type,
                "runtime_backend": spec.runtime_backend,
                "runtime_dtype": spec.runtime_dtype,
                "calibration_input_sha256": spec.calibration_input_sha256,
                "runtime_options": {
                    "force": bool(args.force_runtime),
                    "message_bytes_csv": args.runtime_message_bytes_csv,
                    "warmup": args.runtime_warmup,
                    "iters": args.runtime_iters,
                    "measure_last_n": args.runtime_measure_last_n,
                },
                "model_options": {
                    "force": bool(args.force_model),
                    "warmup_steps": args.warmup_steps,
                    "validation_steps": args.validation_steps,
                },
            }
        except Exception as error:
            local_error = str(error)
        preflight_store = _open_preparation_store(
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            port_offset=4,
            timeout=timedelta(minutes=5),
        )
        preflight_code, preflight_detail = _coordinate_preparation_preflight(
            {"ok": not local_error, "error": local_error, "identity": identity},
            nnodes=nnodes,
            node_rank=node_rank,
            store=preflight_store,
        )
        if preflight_code:
            print(f"PlaceMoE preparation preflight failed: {preflight_detail}", file=sys.stderr)
            return preflight_code
        if source is None or spec is None:
            raise ModelCalibrationError("preparation inputs were not materialized after preflight")
        work_root = (
            Path(args.work_directory).expanduser().resolve()
            if args.work_directory
            else spec.planner_artifact.parent / ".placemoe-prepare.work"
        )
        runtime_args = argparse.Namespace(
            output=str(spec.runtime_artifact),
            hierarchy_group_sizes_csv=",".join(str(value) for value in spec.hierarchy_group_sizes),
            backend=args.runtime_backend,
            dtype=args.runtime_dtype,
            message_bytes_csv=args.runtime_message_bytes_csv,
            runtime_warmup=args.runtime_warmup,
            runtime_iters=args.runtime_iters,
            runtime_measure_last_n=args.runtime_measure_last_n,
            details_json=None,
            work_directory=str(work_root / "runtime"),
        )
        runtime_code, runtime_ran = _run_preparation_stage(
            stage="runtime",
            force=bool(args.force_runtime),
            inspect=lambda: inspect_runtime_cache(spec.runtime_artifact, spec),
            run=lambda: _calibrate_runtime_command(runtime_args),
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            port_offset=2,
        )
        if runtime_code:
            return runtime_code

        model_args = argparse.Namespace(
            config=str(spec.config_path),
            entrypoint=str(spec.entrypoint),
            runtime_perf_model=str(spec.runtime_artifact),
            output=str(spec.planner_artifact),
            work_directory=str(work_root / "model"),
            model_id=spec.model_id,
            warmup_steps=args.warmup_steps,
            validation_steps=args.validation_steps,
        )
        runtime_sha256 = sha256_path(spec.runtime_artifact)
        model_code, _model_ran = _run_preparation_stage(
            stage="model",
            force=bool(args.force_model or runtime_ran),
            inspect=lambda: inspect_planner_cache(
                spec.planner_artifact,
                spec,
                runtime_artifact_sha256=runtime_sha256,
            ),
            run=lambda: _calibrate_model_command(model_args),
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            port_offset=3,
        )
        if model_code:
            return model_code
        print("PlaceMoE calibration artifacts are ready; running deployment doctor.")
        return _doctor_command(
            argparse.Namespace(
                config=str(spec.config_path),
                allow_cpu=bool(args.allow_cpu),
                json=False,
            )
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError, ModelCalibrationError) as error:
        print(f"PlaceMoE preparation failed: {error}", file=sys.stderr)
        return 2


def _add_model_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", help="scope identifier; defaults to the model directory name")
    parser.add_argument("--warmup-steps", type=int, default=2, help="warm-up steps before fitting (default: 2)")
    parser.add_argument(
        "--validation-steps", type=int, default=2, help="held-out validation steps after fitting (default: 2)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placemoe", description="PlaceMoE deployment and validation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="validate a training configuration and local NPU environment")
    doctor.add_argument("--config", required=True, help="VeOmni training YAML or standalone PlaceMoE YAML/JSON")
    doctor.add_argument("--allow-cpu", action="store_true", help="do not require torch_npu or visible NPUs")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable results")
    doctor.set_defaults(handler=_doctor_command)
    runtime = subparsers.add_parser(
        "calibrate-runtime",
        help="measure and fit topology-specific PlaceMoE runtime costs",
    )
    runtime.add_argument("--output", required=True, help="runtime performance model JSON")
    runtime.add_argument("--hierarchy-group-sizes-csv", required=True, help="for example 8,16")
    runtime.add_argument("--backend", default="auto")
    runtime.add_argument("--dtype", default="bf16")
    runtime.add_argument(
        "--message-bytes-csv",
        default="67108864,134217728,268435456,536870912",
    )
    runtime.add_argument("--warmup", dest="runtime_warmup", type=int, default=2)
    runtime.add_argument("--iters", dest="runtime_iters", type=int, default=5)
    runtime.add_argument("--measure-last-n", dest="runtime_measure_last_n", type=int, default=3)
    runtime.add_argument("--details-json")
    runtime.add_argument("--work-directory")
    runtime.set_defaults(handler=_calibrate_runtime_command)
    calibrate = subparsers.add_parser(
        "calibrate-model",
        help="fit and validate planner coefficients with a short default-layout training run",
    )
    calibrate.add_argument("--config", required=True, help="normal VeOmni training YAML")
    calibrate.add_argument("--entrypoint", required=True, help="VeOmni training entrypoint for the model type")
    calibrate.add_argument("--runtime-perf-model", required=True, help="topology calibration JSON")
    calibrate.add_argument("--output", required=True, help="planner calibration JSON written on every node")
    calibrate.add_argument("--work-directory", help="directory for the derived YAML, logs, and timing samples")
    _add_model_fit_arguments(calibrate)
    calibrate.set_defaults(handler=_calibrate_model_command)
    prepare = subparsers.add_parser(
        "prepare",
        help="reuse valid calibration artifacts and create missing ones",
    )
    prepare.add_argument("--config", required=True, help="normal VeOmni training YAML")
    prepare.add_argument("--entrypoint", required=True, help="VeOmni training entrypoint for the model type")
    prepare.add_argument("--work-directory", help="directory for calibration logs and derived files")
    prepare.add_argument("--force-runtime", action="store_true", help="rerun topology calibration")
    prepare.add_argument("--force-model", action="store_true", help="rerun the 5-step model calibration")
    prepare.add_argument("--allow-cpu", action="store_true", help="allow CPU-only deployment doctor checks")
    prepare.add_argument("--runtime-backend", default="auto")
    prepare.add_argument("--runtime-dtype", default="bf16")
    prepare.add_argument(
        "--runtime-message-bytes-csv",
        default="67108864,134217728,268435456,536870912",
    )
    prepare.add_argument("--runtime-warmup", type=int, default=2)
    prepare.add_argument("--runtime-iters", type=int, default=5)
    prepare.add_argument("--runtime-measure-last-n", type=int, default=3)
    _add_model_fit_arguments(prepare)
    prepare.set_defaults(handler=_prepare_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
