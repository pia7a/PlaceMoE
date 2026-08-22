# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
from pathlib import Path

from veomni.distributed.moe.hiermoe.placemoe.calibration import sha256_path
from veomni.distributed.moe.hiermoe.placemoe.preparation import (
    CacheInspection,
    build_preparation_spec,
    decide_cache_action,
    inspect_planner_cache,
    inspect_runtime_cache,
)


def _source(tmp_path: Path, *, ep_size: int = 16, ranks_per_node: int = 8) -> dict:
    return {
        "model": {"model_path": "/models/Qwen3-VL-30B-A3B-Instruct"},
        "train": {
            "accelerator": {"ep_size": ep_size},
            "hiermoe": {
                "hierarchy_group_sizes": [ranks_per_node, ep_size],
                "placemoe": {
                    "enabled": True,
                    "base_directory": str(tmp_path),
                    "runtime_perf_model": "calibration/runtime.json",
                    "calibration": {"artifact": "calibration/planner.json"},
                },
            },
        },
    }


def _spec(tmp_path: Path, *, ep_size: int = 16, ranks_per_node: int = 8):
    config = tmp_path / "train.yaml"
    config.touch()
    entrypoint = tmp_path / "train.py"
    entrypoint.touch()
    return build_preparation_spec(
        _source(tmp_path, ep_size=ep_size, ranks_per_node=ranks_per_node),
        config_path=config,
        entrypoint=entrypoint,
        nnodes=ep_size // ranks_per_node,
        nproc_per_node=ranks_per_node,
        runtime_device_type="npu",
        runtime_backend="hccl",
        runtime_dtype="bf16",
    )


def _runtime_payload(*, ep_size: int = 16, ranks_per_node: int = 8) -> dict:
    return {
        "schema_version": 2,
        "source": "bench_hiermoe_perf_model",
        "a2a": {"alpha": 1.0, "beta": 1.0e-8},
        "inter": [{"alpha": 1.0, "beta": 2.0e-8}],
        "intra": {"alpha": 1.0, "beta": 2.0e-9},
        "state_move": {
            "intra": {"alpha": 1.0, "beta": 2.0e-9},
            "inter": {"alpha": 1.0, "beta": 2.0e-8},
        },
        "gradient_sync": {
            "gather": {
                "intra": {"alpha": 1.0, "beta": 2.0e-9},
                "inter": {"alpha": 1.0, "beta": 2.0e-8},
            },
            "scatter": {
                "intra": {"alpha": 1.0, "beta": 2.0e-9},
                "inter": {"alpha": 1.0, "beta": 2.0e-8},
            },
        },
        "metadata": {
            "ep_size": ep_size,
            "ranks_per_node": ranks_per_node,
            "hierarchy_group_sizes": [ranks_per_node, ep_size],
            "device_type": "npu",
            "backend": "hccl",
            "dtype": "bf16",
        },
    }


def _write_runtime(spec) -> None:
    spec.runtime_artifact.parent.mkdir(parents=True, exist_ok=True)
    spec.runtime_artifact.write_text(
        json.dumps(_runtime_payload(ep_size=spec.ep_size, ranks_per_node=spec.ranks_per_node)),
        encoding="utf-8",
    )


def _write_planner(spec, *, runtime_sha256: str | None = None) -> None:
    spec.planner_artifact.parent.mkdir(parents=True, exist_ok=True)
    spec.planner_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "placemoe_planner_calibration",
                "status": "accepted",
                "scope": {
                    "model_id": spec.model_id,
                    "ep_size": spec.ep_size,
                    "ranks_per_node": spec.ranks_per_node,
                    "hierarchy_group_sizes": list(spec.hierarchy_group_sizes),
                },
                "coefficients": {
                    "inter_ms_per_byte": 1.0e-8,
                    "intra_ms_per_byte": 2.0e-9,
                    "route_ms_per_assignment": 3.0e-5,
                    "communication_multiplier": 2.0,
                    "compute_ms_per_assignment": 4.0e-5,
                    "compute_multiplier": 2.0,
                },
                "held_out_validation": {"checks": {"communication": True, "compute": True, "joint": True}},
                "provenance": {
                    "runtime_perf_model_sha256": runtime_sha256 or sha256_path(spec.runtime_artifact),
                    "calibration_input_sha256": spec.calibration_input_sha256,
                },
            }
        ),
        encoding="utf-8",
    )


def test_preparation_spec_resolves_artifacts_from_training_yaml(tmp_path) -> None:
    spec = _spec(tmp_path)

    assert spec.model_id == "Qwen3-VL-30B-A3B-Instruct"
    assert spec.runtime_artifact == tmp_path / "calibration/runtime.json"
    assert spec.planner_artifact == tmp_path / "calibration/planner.json"
    assert spec.hierarchy_group_sizes == (8, 16)


def test_runtime_cache_requires_matching_topology(tmp_path) -> None:
    spec = _spec(tmp_path)
    assert inspect_runtime_cache(spec.runtime_artifact, spec).state == "missing"
    _write_runtime(spec)
    assert inspect_runtime_cache(spec.runtime_artifact, spec).state == "valid"

    payload = _runtime_payload(ep_size=32, ranks_per_node=8)
    spec.runtime_artifact.write_text(json.dumps(payload), encoding="utf-8")
    result = inspect_runtime_cache(spec.runtime_artifact, spec)

    assert result.state == "invalid"
    assert "scope mismatch" in result.detail


def test_runtime_cache_requires_complete_matching_runtime_scope(tmp_path) -> None:
    spec = _spec(tmp_path)
    payload = _runtime_payload()
    payload["metadata"]["backend"] = "nccl"
    spec.runtime_artifact.parent.mkdir(parents=True, exist_ok=True)
    spec.runtime_artifact.write_text(json.dumps(payload), encoding="utf-8")

    wrong_backend = inspect_runtime_cache(spec.runtime_artifact, spec)
    assert wrong_backend.state == "invalid"
    assert "runtime environment mismatch" in wrong_backend.detail

    payload = _runtime_payload()
    del payload["gradient_sync"]
    spec.runtime_artifact.write_text(json.dumps(payload), encoding="utf-8")
    incomplete = inspect_runtime_cache(spec.runtime_artifact, spec)
    assert incomplete.state == "invalid"
    assert "gradient_sync" in incomplete.detail


def test_planner_cache_requires_scope_validation_and_runtime_hash(tmp_path) -> None:
    spec = _spec(tmp_path)
    _write_runtime(spec)
    _write_planner(spec)

    result = inspect_planner_cache(
        spec.planner_artifact,
        spec,
        runtime_artifact_sha256=sha256_path(spec.runtime_artifact),
    )
    assert result.state == "valid"

    stale = inspect_planner_cache(
        spec.planner_artifact,
        spec,
        runtime_artifact_sha256="0" * 64,
    )
    assert stale.state == "invalid"
    assert "runtime performance model changed" in stale.detail


def test_planner_cache_requires_all_held_out_checks(tmp_path) -> None:
    spec = _spec(tmp_path)
    _write_runtime(spec)
    _write_planner(spec)
    payload = json.loads(spec.planner_artifact.read_text(encoding="utf-8"))
    del payload["held_out_validation"]["checks"]["compute"]
    spec.planner_artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = inspect_planner_cache(
        spec.planner_artifact,
        spec,
        runtime_artifact_sha256=sha256_path(spec.runtime_artifact),
    )

    assert result.state == "invalid"
    assert "missing checks" in result.detail


def test_planner_cache_requires_matching_execution_inputs(tmp_path) -> None:
    spec = _spec(tmp_path)
    _write_runtime(spec)
    _write_planner(spec)
    changed_source = _source(tmp_path)
    changed_source["data"] = {"train_path": "/datasets/other", "max_seq_len": 8192}
    changed_spec = build_preparation_spec(
        changed_source,
        config_path=spec.config_path,
        entrypoint=spec.entrypoint,
        nnodes=2,
        nproc_per_node=8,
        runtime_device_type="npu",
        runtime_backend="hccl",
        runtime_dtype="bf16",
    )

    result = inspect_planner_cache(
        spec.planner_artifact,
        changed_spec,
        runtime_artifact_sha256=sha256_path(spec.runtime_artifact),
    )

    assert result.state == "invalid"
    assert "model or execution inputs changed" in result.detail


def test_cache_decision_reuses_only_when_every_node_is_valid() -> None:
    valid = CacheInspection("valid", "scope matches", "abc")
    different = CacheInspection("valid", "scope matches", "def")
    missing = CacheInspection("missing", "missing")
    invalid = CacheInspection("invalid", "wrong scope")

    assert decide_cache_action((valid, valid), force=False).action == "reuse"
    assert decide_cache_action((valid, different), force=False).action == "error"
    assert decide_cache_action((valid, missing), force=False).action == "run"
    assert decide_cache_action((valid, invalid), force=False).action == "error"
    assert decide_cache_action((invalid, invalid), force=True).action == "run"
