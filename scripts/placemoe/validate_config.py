#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Validate a production PlaceMoE config before reserving accelerators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veomni.distributed.moe.hiermoe.placemoe.artifacts import validate_placemoe_artifact
from veomni.distributed.moe.hiermoe.placemoe.runtime import PlaceMoERuntimeConfig
from veomni.distributed.moe.hiermoe.placemoe.runtime.config import PlaceMoEConfigurationError
from veomni.distributed.moe.hiermoe.placemoe.runtime.cpu_affinity import resolve_cpu_affinity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="PlaceMoE YAML or JSON configuration.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PlaceMoERuntimeConfig.from_file(args.config)
    artifact_path = Path(config.initial_artifact)
    if not artifact_path.is_file():
        raise PlaceMoEConfigurationError(f"initial PlaceMoE artifact does not exist: {artifact_path}.")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    plans = validate_placemoe_artifact(payload)
    topology = payload["topology"]
    runtime_perf_model = Path(config.runtime_perf_model)
    if not runtime_perf_model.is_file():
        raise PlaceMoEConfigurationError(f"runtime performance model does not exist: {runtime_perf_model}.")

    cpu_plan = resolve_cpu_affinity(config.resources)
    planner_resources = cpu_plan.planner_resources()

    calibration_metadata: dict[str, object] = {}
    if config.calibration.artifact:
        calibration_metadata = json.loads(Path(config.calibration.artifact).read_text(encoding="utf-8"))
        calibrated_ep_size = calibration_metadata.get("ep_size")
        if calibrated_ep_size is not None and int(calibrated_ep_size) != int(topology["ep_size"]):
            raise PlaceMoEConfigurationError(
                "calibration EP size does not match the initial artifact: "
                f"{calibrated_ep_size} != {topology['ep_size']}."
            )

    summary = {
        "status": "valid",
        "config": config.source_path,
        "initial_artifact": str(artifact_path),
        "runtime_perf_model": str(runtime_perf_model),
        "layers": len(plans),
        "topology": topology,
        "calibration_artifact": config.calibration.artifact or None,
        "calibration_model": calibration_metadata.get("model_id"),
        "layout_interval_steps": config.hot_update.layout_interval_steps,
        "mapping_interval_steps": config.hot_update.mapping_interval_steps,
        "last_update_step": config.hot_update.last_update_step,
        "cpu_affinity": {
            "mode": "auto" if cpu_plan.automatic else "explicit",
            "training_cpu_ids": planner_resources.training_cpu_ids,
            "planner_cpu_ids": planner_resources.planner_cpu_ids,
            "planner_physical_cores": cpu_plan.planner_physical_cores,
            "workers": cpu_plan.workers,
            "candidate_workers": cpu_plan.candidate_workers,
            "worker_threads": cpu_plan.worker_threads,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
