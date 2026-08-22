# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import argparse

from veomni.distributed.moe.hiermoe.placemoe import cli


def test_doctor_accepts_platform_local_torch_wheel_suffix() -> None:
    assert cli._matches_validated_version("2.9.0+cpu", "2.9.0")
    assert cli._matches_validated_version("2.9.0+cu129", "2.9.0")
    assert not cli._matches_validated_version("2.9.1+cpu", "2.9.0")


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
