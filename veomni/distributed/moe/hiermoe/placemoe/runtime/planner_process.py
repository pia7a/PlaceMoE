"""CPU planner process specification for PlaceMoE hot updates."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import PlaceMoECalibration, PlaceMoEPlannerResources
from .scheduler import UpdateKind


@dataclass(frozen=True)
class HotUpdateJob:
    kind: UpdateKind
    source_step: int
    placement_versions: tuple[int, ...]
    submitted_at: float
    snapshot_ms: float
    job_dir: str
    layout_path: str
    report_path: str
    planner_log_path: str
    process: subprocess.Popen[bytes] | None = None

    @property
    def update_mode(self) -> str:
        """Compatibility spelling used by existing runtime metrics."""

        return self.kind.value


@dataclass(frozen=True)
class PlannerCommandSpec:
    python: str
    planner_path: str
    route_root: str
    kind: UpdateKind
    layer_keys: Sequence[str]
    ep_size: int
    ranks_per_node: int
    num_experts: int
    slots_per_rank: int
    primary_slots_per_rank: int
    redundant_slots_per_rank: int
    hidden_size: int
    bytes_per_element: int
    output_layout: str
    output_report: str
    input_layout: str = ""


def build_planner_command(
    spec: PlannerCommandSpec,
    calibration: PlaceMoECalibration,
    resources: PlaceMoEPlannerResources,
) -> list[str]:
    """Build the canonical planner command without launching a subprocess."""

    command = [
        spec.python,
        spec.planner_path,
        "--route-root",
        spec.route_root,
        "--optimize-steps",
        "0",
        "--validation-steps",
        "0",
        "--update-mode",
        spec.kind.value,
        "--layers",
        str(len(spec.layer_keys)),
        "--layer-keys",
        ",".join(spec.layer_keys),
        "--expected-total-layers",
        str(len(spec.layer_keys)),
        "--workers",
        str(resources.workers),
        "--candidate-workers",
        str(resources.candidate_workers),
        "--worker-threads",
        str(resources.worker_threads),
        "--ep-size",
        str(spec.ep_size),
        "--ranks-per-node",
        str(spec.ranks_per_node),
        "--num-experts",
        str(spec.num_experts),
        "--slots-per-rank",
        str(spec.slots_per_rank),
        "--primary-slots-per-rank",
        str(spec.primary_slots_per_rank),
        "--redundant-slots-per-rank",
        str(spec.redundant_slots_per_rank),
        "--active-redundant-slots",
        str(spec.redundant_slots_per_rank * spec.ep_size),
        "--hidden-size",
        str(spec.hidden_size),
        "--bytes-per-element",
        str(spec.bytes_per_element),
        "--inter-ms-per-byte",
        str(calibration.inter_ms_per_byte),
        "--intra-ms-per-byte",
        str(calibration.intra_ms_per_byte),
        "--route-ms-per-assignment",
        str(calibration.route_ms_per_assignment),
        "--communication-phase-multiplier",
        str(calibration.communication_multiplier),
        "--compute-ms-per-assignment",
        str(calibration.compute_ms_per_assignment),
        "--compute-phase-multiplier",
        str(calibration.compute_multiplier),
        "--comparison-layout",
        "none",
        "--output-layout",
        spec.output_layout,
        "--output-report",
        spec.output_report,
    ]
    if spec.kind is UpdateKind.MAPPING_ONLY:
        if not spec.input_layout:
            raise ValueError("mapping-only planning requires input_layout.")
        command.extend(("--input-layout", spec.input_layout))
    if resources.planner_cpu_ids:
        command = ["taskset", "-c", resources.planner_cpu_ids, *command]
    return command


def planner_environment(
    resources: PlaceMoEPlannerResources,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    threads = str(resources.worker_threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def launch_planner_process(
    command: Sequence[str],
    *,
    stdout: Any,
    environment: Mapping[str, str],
) -> subprocess.Popen[bytes]:
    """Launch the planner in a supervised session shared by all workers."""

    supervised_command = [
        sys.executable,
        "-m",
        "veomni.distributed.moe.hiermoe.placemoe.runtime.planner_supervisor",
        "--",
        *command,
    ]
    supervisor_environment = dict(environment)
    supervisor_environment["PLACEMOE_SUPERVISOR_PARENT_PID"] = str(os.getpid())
    return subprocess.Popen(
        supervised_command,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=supervisor_environment,
        start_new_session=True,
    )


def terminate_planner_process(process: subprocess.Popen[bytes], *, timeout: float = 10.0) -> None:
    """Terminate the complete planner session and reap its leader.

    A planner worker can outlive a leader that failed or exited first.  The
    process group must therefore be signalled even when ``Popen.poll()``
    already reports completion.
    """

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait()
        return
    except OSError:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                process.kill()

    if process.poll() is not None:
        process.wait()
        return
    try:
        process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        process.wait()


__all__ = [
    "HotUpdateJob",
    "PlannerCommandSpec",
    "build_planner_command",
    "launch_planner_process",
    "planner_environment",
    "terminate_planner_process",
]
