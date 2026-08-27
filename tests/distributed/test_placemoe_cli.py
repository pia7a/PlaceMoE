# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import argparse
import json

import yaml

from veomni.distributed.moe.hiermoe.placemoe import cli
from veomni.distributed.moe.hiermoe.placemoe.calibration import sha256_path
from veomni.distributed.moe.hiermoe.placemoe.preparation import (
    CacheInspection,
    fingerprint_calibration_inputs,
)


class _FakeStore:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})
        self.wait_calls = []

    def set(self, key, value) -> None:
        self.values[key] = value.encode("utf-8") if isinstance(value, str) else value

    def get(self, key):
        return self.values[key]

    def wait(self, keys) -> None:
        self.wait_calls.append(tuple(keys))
        assert all(key in self.values for key in keys)


def test_doctor_accepts_platform_local_torch_wheel_suffix() -> None:
    assert cli._matches_validated_version("2.9.0+cpu", "2.9.0")
    assert cli._matches_validated_version("2.9.0+cu129", "2.9.0")
    assert not cli._matches_validated_version("2.9.1+cpu", "2.9.0")


def test_cli_exposes_model_calibration_command() -> None:
    args = cli.build_parser().parse_args(
        [
            "calibrate-model",
            "--config",
            "train.yaml",
            "--entrypoint",
            "tasks/train_vlm.py",
            "--runtime-perf-model",
            "runtime.json",
            "--output",
            "planner.json",
        ]
    )

    assert args.handler is cli._calibrate_model_command
    assert args.warmup_steps == 2
    assert args.validation_steps == 2


def test_cli_exposes_runtime_calibration_and_prepare_commands() -> None:
    runtime = cli.build_parser().parse_args(
        [
            "calibrate-runtime",
            "--output",
            "runtime.json",
            "--hierarchy-group-sizes-csv",
            "8,16",
        ]
    )
    prepare = cli.build_parser().parse_args(
        ["prepare", "--config", "train.yaml", "--entrypoint", "tasks/train_vlm.py"]
    )

    assert runtime.handler is cli._calibrate_runtime_command
    assert runtime.runtime_warmup == 2
    assert prepare.handler is cli._prepare_command
    assert not prepare.force_runtime
    assert not prepare.force_model


def test_runtime_calibration_launches_single_node_rank_hierarchy(tmp_path, monkeypatch) -> None:
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(cli, "_distributed_environment", lambda: (1, 0, 4, "127.0.0.1", 29500))

    def run(command, *, environment, log_path) -> int:
        assert "--nnodes=1" in command
        assert "--nproc-per-node=4" in command
        assert "--hierarchy-group-sizes-csv" in command
        assert command[command.index("--hierarchy-group-sizes-csv") + 1] == "4"
        output.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source": "bench_hiermoe_perf_model",
                    "status": "accepted",
                    "inter": [],
                    "intra": {"alpha": 1.0, "beta": 2.0e-9},
                    "metadata": {
                        "ep_size": 4,
                        "ranks_per_node": 4,
                        "hierarchy_group_sizes": [4],
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(cli, "_stream_training", run)
    args = cli.build_parser().parse_args(
        [
            "calibrate-runtime",
            "--output",
            str(output),
            "--hierarchy-group-sizes-csv",
            "4",
        ]
    )

    assert cli._calibrate_runtime_command(args) == 0


def test_stage_result_rejects_different_generated_artifacts() -> None:
    peer_result = json.dumps(
        {
            "return_code": 0,
            "inspection": {"state": "valid", "detail": "scope matches", "digest": "def"},
        }
    ).encode("utf-8")
    store = _FakeStore(
        {
            "placemoe-prepare/runtime/result/1": peer_result,
            "placemoe-prepare/runtime/final-read/1": b"1",
        }
    )

    return_code, detail = cli._coordinate_stage_result(
        CacheInspection("valid", "scope matches", "abc"),
        return_code=0,
        stage="runtime",
        nnodes=2,
        node_rank=0,
        store=store,
    )

    assert return_code == 2
    assert "differ across nodes" in detail
    assert ("placemoe-prepare/runtime/final-read/1",) in store.wait_calls


def test_cache_decision_client_acknowledges_payload_read() -> None:
    prefix = "placemoe-prepare/runtime"
    peer_inspection = json.dumps({"state": "valid", "detail": "scope matches", "digest": "abc"}).encode("utf-8")
    decision = json.dumps({"action": "reuse", "detail": "valid on every node"}).encode("utf-8")
    store = _FakeStore(
        {
            f"{prefix}/inspection/0": peer_inspection,
            f"{prefix}/decision": decision,
        }
    )

    result = cli._coordinate_cache_decision(
        CacheInspection("valid", "scope matches", "abc"),
        force=False,
        stage="runtime",
        nnodes=2,
        node_rank=1,
        store=store,
    )

    assert result.action == "reuse"
    assert store.values[f"{prefix}/decision-read/1"] == b"1"


def test_stage_result_client_acknowledges_payload_read() -> None:
    prefix = "placemoe-prepare/runtime"
    peer_result = json.dumps(
        {
            "return_code": 0,
            "inspection": {"state": "valid", "detail": "scope matches", "digest": "abc"},
        }
    ).encode("utf-8")
    final = json.dumps({"return_code": 0, "detail": "valid on every node"}).encode("utf-8")
    store = _FakeStore(
        {
            f"{prefix}/result/0": peer_result,
            f"{prefix}/final": final,
        }
    )

    return_code, detail = cli._coordinate_stage_result(
        CacheInspection("valid", "scope matches", "abc"),
        return_code=0,
        stage="runtime",
        nnodes=2,
        node_rank=1,
        store=store,
    )

    assert return_code == 0
    assert detail == "valid on every node"
    assert store.values[f"{prefix}/final-read/1"] == b"1"


def test_preflight_rejects_different_node_inputs() -> None:
    peer = json.dumps({"ok": True, "error": "", "identity": {"config_sha256": "def"}}).encode("utf-8")
    store = _FakeStore({"placemoe-prepare/preflight/1": peer})

    return_code, detail = cli._coordinate_preparation_preflight(
        {"ok": True, "error": "", "identity": {"config_sha256": "abc"}},
        nnodes=2,
        node_rank=0,
        store=store,
    )

    assert return_code == 2
    assert "inputs differ across nodes" in detail


def test_preparation_stage_converts_unexpected_exception_to_failure(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_open_preparation_store", lambda **_kwargs: None)

    return_code, ran = cli._run_preparation_stage(
        stage="runtime",
        force=False,
        inspect=lambda: CacheInspection("missing", "not generated"),
        run=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        nnodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        port_offset=2,
    )

    assert return_code == 2
    assert ran


def test_prepare_reuses_valid_artifacts_without_running_calibrators(tmp_path, monkeypatch, capsys) -> None:
    model = tmp_path / "Qwen"
    model.mkdir()
    entrypoint = tmp_path / "train.py"
    entrypoint.touch()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    runtime = calibration / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": "bench_hiermoe_perf_model",
                "a2a": {"alpha": 1.0, "beta": 1.0e-8},
                "inter": [],
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
                    "ep_size": 2,
                    "ranks_per_node": 2,
                    "hierarchy_group_sizes": [2],
                    "device_type": "npu",
                    "backend": "hccl",
                    "dtype": "bf16",
                },
            }
        ),
        encoding="utf-8",
    )
    planner = calibration / "planner.json"
    planner.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "placemoe_planner_calibration",
                "status": "accepted",
                "scope": {
                    "model_id": "Qwen",
                    "ep_size": 2,
                    "ranks_per_node": 2,
                    "hierarchy_group_sizes": [2],
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
                "provenance": {"runtime_perf_model_sha256": sha256_path(runtime)},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
model:
  model_path: {model}
  ops_implementation:
    moe_implementation: fused_npu
    cross_entropy_loss_implementation: npu
    rms_norm_implementation: npu
    swiglu_mlp_implementation: eager
    rotary_pos_emb_implementation: npu
    load_balancing_loss_implementation: eager
train:
  accelerator:
    ep_size: 2
  hiermoe:
    hierarchy_group_sizes: [2]
    placemoe:
      enabled: true
      base_directory: {tmp_path}
      runtime_perf_model: calibration/runtime.json
      calibration:
        artifact: calibration/planner.json
""",
        encoding="utf-8",
    )
    planner_payload = json.loads(planner.read_text(encoding="utf-8"))
    planner_payload["provenance"]["calibration_input_sha256"] = fingerprint_calibration_inputs(
        yaml.safe_load(config.read_text(encoding="utf-8")), entrypoint
    )
    planner.write_text(json.dumps(planner_payload), encoding="utf-8")
    monkeypatch.setattr(cli, "_distributed_environment", lambda: (1, 0, 2, "127.0.0.1", 29500))
    monkeypatch.setattr(cli, "get_device_type", lambda: "npu")
    monkeypatch.setattr(cli, "_calibrate_runtime_command", lambda _args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(cli, "_calibrate_model_command", lambda _args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(cli, "_doctor_command", lambda _args: 0)
    args = cli.build_parser().parse_args(
        ["prepare", "--config", str(config), "--entrypoint", str(entrypoint), "--allow-cpu"]
    )

    assert cli._prepare_command(args) == 0
    output = capsys.readouterr().out
    assert "runtime calibration reused" in output
    assert "model calibration reused" in output


def test_prepare_generates_missing_runtime_and_model_artifacts(tmp_path, monkeypatch) -> None:
    model = tmp_path / "Qwen"
    model.mkdir()
    entrypoint = tmp_path / "train.py"
    entrypoint.touch()
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
model:
  model_path: {model}
  ops_implementation:
    moe_implementation: fused_npu
    cross_entropy_loss_implementation: npu
    rms_norm_implementation: npu
    swiglu_mlp_implementation: eager
    rotary_pos_emb_implementation: npu
    load_balancing_loss_implementation: eager
train:
  accelerator:
    ep_size: 2
  hiermoe:
    hierarchy_group_sizes: [2]
    placemoe:
      enabled: true
      base_directory: {tmp_path}
      runtime_perf_model: calibration/runtime.json
      calibration:
        artifact: calibration/planner.json
""",
        encoding="utf-8",
    )
    calls = []

    def calibrate_runtime(args) -> int:
        calls.append("runtime")
        runtime_path = tmp_path / "calibration" / "runtime.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source": "bench_hiermoe_perf_model",
                    "a2a": {"alpha": 1.0, "beta": 1.0e-8},
                    "inter": [],
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
                        "ep_size": 2,
                        "ranks_per_node": 2,
                        "hierarchy_group_sizes": [2],
                        "device_type": "npu",
                        "backend": "hccl",
                        "dtype": "bf16",
                    },
                }
            ),
            encoding="utf-8",
        )
        assert args.output == str(runtime_path)
        return 0

    def calibrate_model(args) -> int:
        calls.append("model")
        runtime_path = tmp_path / "calibration" / "runtime.json"
        planner_path = tmp_path / "calibration" / "planner.json"
        planner_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "placemoe_planner_calibration",
                    "status": "accepted",
                    "scope": {
                        "model_id": "Qwen",
                        "ep_size": 2,
                        "ranks_per_node": 2,
                        "hierarchy_group_sizes": [2],
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
                        "runtime_perf_model_sha256": sha256_path(runtime_path),
                        "calibration_input_sha256": fingerprint_calibration_inputs(
                            yaml.safe_load(config.read_text(encoding="utf-8")), entrypoint
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        assert args.output == str(planner_path)
        return 0

    monkeypatch.setattr(cli, "_distributed_environment", lambda: (1, 0, 2, "127.0.0.1", 29500))
    monkeypatch.setattr(cli, "get_device_type", lambda: "npu")
    monkeypatch.setattr(cli, "_calibrate_runtime_command", calibrate_runtime)
    monkeypatch.setattr(cli, "_calibrate_model_command", calibrate_model)
    monkeypatch.setattr(cli, "_doctor_command", lambda _args: 0)
    args = cli.build_parser().parse_args(
        ["prepare", "--config", str(config), "--entrypoint", str(entrypoint), "--allow-cpu"]
    )

    assert cli._prepare_command(args) == 0
    assert calls == ["runtime", "model"]
    assert (tmp_path / "calibration" / "runtime.json").is_file()
    assert (tmp_path / "calibration" / "planner.json").is_file()


def test_doctor_validates_inline_training_configuration(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "data.parquet"
    dataset.touch()
    perf_model = tmp_path / "perf.json"
    perf_model.write_text(
        '{"schema_version":2,"a2a":{"alpha":1,"beta":1},'
        '"inter":[{"alpha":1,"beta":1}],"intra":{"alpha":1,"beta":1},'
        '"state_move":{"intra":{"alpha":1,"beta":1},"inter":{"alpha":1,"beta":1}},'
        '"gradient_sync":{"gather":{"intra":{"alpha":1,"beta":1},"inter":{"alpha":1,"beta":1}},'
        '"scatter":{"intra":{"alpha":1,"beta":1},"inter":{"alpha":1,"beta":1}}}}',
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"coefficients":{"inter_ms_per_byte":1e-8,"intra_ms_per_byte":1e-9,'
        '"route_ms_per_assignment":1e-5,"communication_multiplier":1.0,'
        '"compute_ms_per_assignment":2e-5,"compute_multiplier":1.0}}',
        encoding="utf-8",
    )
    config = tmp_path / "train.yaml"
    config.write_text(
        """
model:
  model_path: model
data:
  train_path: data.parquet
train:
  hiermoe:
    redundant_slot_increment_per_device: 2
    perf_model_path: perf.json
    placemoe:
      enabled: true
      base_directory: .
      calibration:
        artifact: calibration.json
      hot_update:
        enabled: true
        layout_interval_steps: 100
        mapping_interval_steps: 20
        work_root: runtime
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_version", lambda name: "2.9.0" if name == "torch" else None)

    results = cli.run_doctor(config, require_npu=False)
    by_name = {result.name: result for result in results}

    assert by_name["model"].status == "PASS"
    assert by_name["runtime_bridge"].status == "PASS"
    assert by_name["dataset"].status == "PASS"
    assert by_name["performance_model"].status == "PASS"
    assert by_name["performance_model_schema"].status == "PASS"
    assert by_name["calibration"].status == "PASS"
    assert by_name["initial_artifact"].status == "SKIP"
    assert by_name["placemoe_runtime"].status == "PASS"
    assert by_name["replica_slots"].status == "PASS"
    assert by_name["hot_update"].status == "PASS"
    assert "startup_plan" not in by_name
    assert "layout=100, mapping=20" in by_name["hot_update"].detail


def test_doctor_treats_zero_update_intervals_as_static(tmp_path, monkeypatch) -> None:
    config = tmp_path / "placemoe.yaml"
    config.write_text("placemoe:\n  hot_update:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_version", lambda name: "2.9.0" if name == "torch" else None)

    results = cli.run_doctor(config, require_npu=False)

    hot_update = next(result for result in results if result.name == "hot_update")
    assert hot_update.status == "PASS"
    assert hot_update.detail == "static"


def test_doctor_requires_production_calibration_for_enabled_training(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "data"
    dataset.mkdir()
    perf_model = tmp_path / "perf.json"
    perf_model.write_text("{}", encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
model:
  model_path: {model}
data:
  train_path: {dataset}
train:
  hiermoe:
    redundant_slot_increment_per_device: 1
    perf_model_path: {perf_model}
    placemoe:
      enabled: true
      hot_update:
        enabled: true
        layout_interval_steps: 10
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_version", lambda name: "2.9.0" if name == "torch" else None)

    results = cli.run_doctor(config, require_npu=False)

    calibration = next(result for result in results if result.name == "calibration")
    assert calibration.status == "FAIL"
    assert "calibration.artifact" in calibration.detail


def test_doctor_command_reports_missing_calibration_artifact(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "placemoe.yaml"
    config.write_text(
        "placemoe:\n  calibration:\n    artifact: missing.json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_version", lambda name: "2.9.0" if name == "torch" else None)

    exit_code = cli._doctor_command(argparse.Namespace(config=str(config), allow_cpu=True, json=False))
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "FAIL" in output
    assert "configuration" in output
    assert "missing.json" in output
    assert "placemoe prepare" in output
