# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.placemoe.reproduction.gpu_ep32 import summarize_repeats
from scripts.placemoe.reproduction.gpu_ep32.calibrate_communication import _cluster_scope, _fit_raw_balanced_link
from scripts.placemoe.reproduction.gpu_ep32.cost_components import load_communication_calibration
from scripts.placemoe.reproduction.gpu_ep32.validate_cost_model import _validate_three_level_alignment
from scripts.placemoe.reproduction.gpu_ep32.validate_layout import (
    COST_KEYS,
    _identity,
    _sha256,
    validate_layout_bundle,
    write_layout_bundle,
)
from scripts.profile import plot_hiermoe_paper_speedup, summarize_hiermoe_paper_case
from veomni.distributed.moe import timing
from veomni.ops.kernels.moe._kernels.kernel import group_gemm
from veomni.trainer.callbacks.trace_callback import _true_step_time_enabled


@dataclass
class _FakeEvent:
    wall_time: float

    def elapsed_time(self, end: "_FakeEvent") -> float:
        return (end.wall_time - self.wall_time) * 1000.0


def _span(start: float, end: float) -> dict:
    return {
        "phase": "forward",
        "direction": "forward",
        "component": "all_to_all",
        "section": "pre",
        "start_event": _FakeEvent(start),
        "end_event": _FakeEvent(end),
        "call_index": 0,
        "micro_batch": 0,
        "layer": 0,
        "num_layers": 1,
        "num_experts": 8,
        "ep_size": 2,
        "tokens": 4,
        "token_expert_assignments": 8,
        "top_k": 2,
    }


def test_individual_moe_spans_are_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS", raising=False)
    monkeypatch.setattr(timing, "synchronize_accelerator", lambda: None)
    monkeypatch.setattr(timing, "_MOE_TIMING_SPANS", [_span(0.0, 0.001)])

    payload = timing.flush_moe_timing_spans()

    assert "span_calls_by_phase" in payload
    assert "span_invocations" not in payload


def test_individual_moe_spans_preserve_invocation_order(monkeypatch) -> None:
    monkeypatch.setenv("VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS", "1")
    monkeypatch.setattr(timing, "synchronize_accelerator", lambda: None)
    monkeypatch.setattr(
        timing,
        "_MOE_TIMING_SPANS",
        [_span(0.0, 0.001), _span(0.002, 0.005)],
    )

    payload = timing.flush_moe_timing_spans()

    assert [row["invocation"] for row in payload["span_invocations"]] == [0, 1]
    assert [row["cuda_ms_sum"] for row in payload["span_invocations"]] == [1.0, 3.0]


def test_true_step_time_has_an_independent_override(monkeypatch) -> None:
    monkeypatch.delenv("VEOMNI_TRUE_STEP_TIME", raising=False)
    monkeypatch.delenv("VEOMNI_CONVERGENCE_METRICS_DIR", raising=False)
    assert not _true_step_time_enabled()

    monkeypatch.setenv("VEOMNI_CONVERGENCE_METRICS_DIR", "/tmp/metrics")
    assert _true_step_time_enabled()

    monkeypatch.setenv("VEOMNI_TRUE_STEP_TIME", "0")
    assert not _true_step_time_enabled()

    monkeypatch.delenv("VEOMNI_CONVERGENCE_METRICS_DIR")
    monkeypatch.setenv("VEOMNI_TRUE_STEP_TIME", "1")
    assert _true_step_time_enabled()


def test_ep32_cost_calibration_exports_offline_scorer_samples() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/launch.sh").read_text(encoding="utf-8")
    calibration = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/calibrate_cost_model.sh").read_text(
        encoding="utf-8"
    )

    assert "VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES=${VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES_OVERRIDE" in launcher
    assert "VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES_OVERRIDE=1" in calibration


def test_ep32_launcher_uses_cuda_malloc_async_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/launch.sh").read_text(encoding="utf-8")

    assert "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF_OVERRIDE:-backend:cudaMallocAsync}" in launcher


def test_ep32_formal_matrix_uses_bounded_async_allocator_cache() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    matrix = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/matrix.sh").read_text(encoding="utf-8")
    launcher = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/launch.sh").read_text(encoding="utf-8")
    node = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/run_training_node.sh").read_text(encoding="utf-8")

    assert "EMPTY_CACHE_STEPS_OVERRIDE=${PLACEMOE_REPRO_EMPTY_CACHE_STEPS:-500}" in matrix
    assert "EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS_OVERRIDE:-500}" in launcher
    assert '--train.empty_cache_steps "${empty_cache_steps}"' in node


def test_ep32_dry_run_previews_the_five_step_formal_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    matrix = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/matrix.sh").read_text(encoding="utf-8")

    assert "max_steps=5" in matrix
    assert 'if [[ "${mode}" == smoke ]]' in matrix


def test_group_gemm_cuda_output_allocation_retries_once_after_cache_reclaim(monkeypatch) -> None:
    output = object()
    allocation_calls = 0
    cache_reclaims = 0

    def fake_empty(*args, **kwargs):
        nonlocal allocation_calls
        allocation_calls += 1
        if allocation_calls == 1:
            raise torch.OutOfMemoryError("fragmented CUDA cache")
        return output

    def fake_empty_cache() -> None:
        nonlocal cache_reclaims
        cache_reclaims += 1

    accelerator = SimpleNamespace(device=lambda _device: nullcontext(), empty_cache=fake_empty_cache)
    monkeypatch.setattr(group_gemm.torch, "empty", fake_empty)
    monkeypatch.setattr(group_gemm, "get_torch_device", lambda: accelerator)

    actual = group_gemm._empty_with_cuda_oom_retry((4, 8), dtype=torch.bfloat16, device=torch.device("cuda:0"))

    assert actual is output
    assert allocation_calls == 2
    assert cache_reclaims == 1


def test_representative_ep32_profile_uses_formal_nonpipeline_variants() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    profile = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/profile_representative.sh").read_text(
        encoding="utf-8"
    )

    assert "variant=replica" in profile
    assert "variant=static_layout" in profile
    assert "fixed_r2_mirrored_pipeline_grad" not in profile
    assert "hierarchical_full_static" not in profile
    assert "HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE" not in profile


def test_ep32_matrix_reuses_cost_scoped_dataset_layouts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    matrix = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/matrix.sh").read_text(encoding="utf-8")

    assert "model_layout_tag=${PLACEMOE_REPRO_LAYOUT_TAG_OVERRIDE:-${model_cost_scope:0:12}}" in matrix
    assert "${run_tag}_${model_cost_scope:0:12}" not in matrix
    assert "PLACEMOE_REPRO_LAYOUT_PROFILE_TAG" in matrix
    assert 'PLACEMOE_REPRO_REUSE_PROFILE="${layout_reuse_profile}"' in matrix
    assert 'PLACEMOE_REPRO_ROUTE_ROOT="${layout_route_root}"' in matrix
    assert "HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE" not in matrix


def test_ep32_common_supports_scoped_micro_batch_override_and_hashes_group_gemm_kernel() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    common = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/common.sh").read_text(encoding="utf-8")

    assert "PLACEMOE_REPRO_MICRO_BATCH_SIZE_OVERRIDE" in common
    assert ('veomni/ops/kernels/moe/_kernels/kernel/group_gemm.py"') in common


def test_ep32_entrypoint_uses_one_portable_cluster_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    suite_dir = repo_root / "scripts/placemoe/reproduction/gpu_ep32"
    paths = (
        suite_dir / "common.sh",
        suite_dir / "ssh.sh",
        suite_dir / "launch.sh",
        suite_dir / "matrix.sh",
        repo_root / "configs/placemoe/gpu_ep32.env.example",
    )
    contract = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert not any("gpu32" in path.name.lower() for path in suite_dir.iterdir())
    assert "PLACEMOE_REPRO_CONFIG" in contract
    assert "--config PATH" in contract
    assert "2.9.1+cu129" in contract
    assert "PLACEMOE_REPRO_HIERARCHY_GROUP_SIZES" in contract
    assert "GPU32_" not in contract
    assert "paper32_" not in contract
    for forbidden in ("/workspace/task3", "10.249.40.11", "31504", "30963", "32298"):
        assert forbidden not in contract


def test_ep32_external_helpers_are_repo_root_relative() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ep32_dir = repo_root / "scripts/placemoe/reproduction/gpu_ep32"
    entrypoints = "\n".join(
        (ep32_dir / name).read_text(encoding="utf-8") for name in ("prepare_layouts.sh", "matrix.sh")
    )
    helper_paths = (
        repo_root / "scripts/profile/build_hiermoe_eplb_layout.py",
        repo_root / "scripts/profile/plan_placemoe.py",
        repo_root / "scripts/profile/summarize_hiermoe_paper_case.py",
        repo_root / "scripts/profile/plot_hiermoe_paper_speedup.py",
    )

    for helper_path in helper_paths:
        assert helper_path.is_file()
        relative_path = helper_path.relative_to(repo_root).as_posix()
        assert f"${{repro_source_root}}/{relative_path}" in entrypoints


def test_ep32_route_capture_binds_moe_layer_count() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = (repo_root / "scripts/placemoe/reproduction/gpu_ep32/run_training_node.sh").read_text(encoding="utf-8")

    assert "VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL=0" in launcher
    assert "VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS" in launcher


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _layout_bundle_fixture(tmp_path: Path) -> tuple[dict, dict]:
    route_root = tmp_path / "routes"
    route_root.mkdir()
    topology = {
        "layers": 1,
        "ep_size": 2,
        "ranks_per_node": 1,
        "num_experts": 2,
        "primary_slots_per_rank": 1,
        "redundant_slots_per_rank": 1,
        "slots_per_rank": 2,
        "hidden_size": 4,
    }
    coefficients = {key: float(index + 1) for index, key in enumerate(COST_KEYS)}
    cost_model = tmp_path / "cost.json"
    _write_json(cost_model, {"offline_scorer": coefficients})
    ours_layout = tmp_path / "ours_layout.json"
    _write_json(
        ours_layout,
        {
            "schema_version": 2,
            "source": {
                "algorithm": "placemoe-v1",
                "route_root": str(route_root.resolve()),
                "optimize_steps": [0, 1, 2],
            },
            "topology": {
                "ep_size": 2,
                "ranks_per_node": 1,
                "num_experts": 2,
                "num_physical_slots": 4,
                "slots_per_rank": 2,
            },
            "layers": {"layer.0": {}},
        },
    )
    ours_report = tmp_path / "ours_report.json"
    _write_json(
        ours_report,
        {
            "schema_version": 1,
            "algorithm": "placemoe-v1",
            "configuration": {
                "route_root": str(route_root.resolve()),
                "optimize_steps": [0, 1, 2],
                "validation_steps": [3],
                "layer_start": 0,
                "layers": 1,
                "expected_total_layers": 1,
                "ep_size": 2,
                "ranks_per_node": 1,
                "num_experts": 2,
                "primary_slots_per_rank": 1,
                "redundant_slots_per_rank": 1,
                "slots_per_rank": 2,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "comparison_layout": "mirrored-r2",
                "update_mode": "full",
                **coefficients,
            },
        },
    )
    eplb_layout = tmp_path / "eplb_layout.json"
    _write_json(
        eplb_layout,
        {
            "schema_version": 2,
            "source": {
                "algorithm": "deepseek-eplb-global-v1-source-lut-compiled",
                "route_root": str(route_root.resolve()),
                "profile_steps": [0, 1, 2, 3],
            },
            "topology": {
                "ep_size": 2,
                "num_experts": 2,
                "num_physical_slots": 4,
                "slots_per_rank": 2,
            },
            "layers": {"layer.0": {}},
        },
    )
    eplb_report = tmp_path / "eplb_report.json"
    _write_json(
        eplb_report,
        {
            "schema_version": 1,
            "algorithm": "deepseek-eplb-global-v1-source-lut-compiled",
            "layers": 1,
            "profile_steps": [0, 1, 2, 3],
            "redundant_slots_per_rank": 1,
        },
    )
    artifacts = {
        "eplb_layout": eplb_layout,
        "eplb_report": eplb_report,
        "ours_layout": ours_layout,
        "ours_report": ours_report,
    }
    shared = {
        "cost_model": cost_model,
        "cost_model_sha256": _sha256(cost_model),
        "route_manifest_sha256": "a" * 64,
        "identity": _identity(
            accelerator="NVIDIA RTX A6000",
            model_id="qwen3vl30b",
            dataset_id="sharegpt4v",
            micro_batch_size=4,
            global_batch_size=128,
            max_seq_len=4096,
            moe_impl="fused_triton",
            freeze_vit="true",
        ),
        "source_fingerprints": {"placemoe": "b" * 64, "eplb": "c" * 64},
        "route_root": route_root,
        "topology": topology,
    }
    return artifacts, shared


def test_layout_bundle_reuse_binds_artifacts_and_source(tmp_path: Path) -> None:
    artifacts, shared = _layout_bundle_fixture(tmp_path)
    bundle = tmp_path / "bundle.json"

    payload = write_layout_bundle(bundle, artifacts, **shared)
    validate_layout_bundle(bundle, artifacts, **shared)

    assert payload["schema_version"] == 2
    changed_source = {**shared, "source_fingerprints": {"placemoe": "d" * 64, "eplb": "c" * 64}}
    with pytest.raises(RuntimeError, match="source_fingerprints"):
        validate_layout_bundle(bundle, artifacts, **changed_source)


def test_formal_summary_aggregate_and_plot_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_root = tmp_path / "profiles"
    cost_model = tmp_path / "cost_model.json"
    communication = tmp_path / "communication.json"
    preflight = tmp_path / "preflight.json"
    for path in (cost_model, communication, preflight):
        path.write_text(path.name, encoding="utf-8")
    communication_source_sha256 = "a" * 64
    summaries = []
    for repeat in range(1, 4):
        run_name = f"formal_r{repeat}"
        env_path = profile_root / run_name / "env_metrics" / "env_metrics_rank0.jsonl"
        env_path.parent.mkdir(parents=True)
        rows = [
            {
                "step": step,
                "step_time_s": 1.0 + repeat * 0.1 + step * 0.001,
                "tokens_per_second(M)": 0.5,
                "max_memory_allocated(GB)": 20.0 + repeat,
                "max_memory_reserved(GB)": 24.0 + repeat,
            }
            for step in range(3, 6)
        ]
        env_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        summary = tmp_path / f"{run_name}.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "summarize_hiermoe_paper_case.py",
                "--run-name",
                run_name,
                "--profile-root",
                str(profile_root),
                "--start-step",
                "3",
                "--end-step",
                "5",
                "--skip-moe-timing",
                "--grad-mode",
                "blocking",
                "--cost-model",
                str(cost_model),
                "--communication-calibration",
                str(communication),
                "--preflight-report",
                str(preflight),
                "--communication-source-sha256",
                communication_source_sha256,
                "--repeat-index",
                str(repeat),
                "--execution-index",
                str(repeat),
                "--execution-policy",
                "repeat-major-fixed-order",
                "--output",
                str(summary),
            ],
        )
        summarize_hiermoe_paper_case.main()
        summaries.append(summary)

    aggregate = tmp_path / "aggregate.json"
    aggregate_argv = [
        "summarize_repeats.py",
        "--method",
        "baseline",
        "--model",
        "qwen3vl30b",
        "--dataset",
        "sharegpt4v",
        "--grad-protocol",
        "paper",
        "--grad-mode",
        "blocking",
        "--expected-repeats",
        "3",
        "--output",
        str(aggregate),
    ]
    for summary in summaries:
        aggregate_argv.extend(("--summary", str(summary)))
    monkeypatch.setattr(sys, "argv", aggregate_argv)
    summarize_repeats.main()

    output_json = tmp_path / "speedup.json"
    output_svg = tmp_path / "speedup.svg"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_hiermoe_paper_speedup.py",
            "--model",
            "Qwen3-VL-30B-A3B",
            "--dataset",
            "sharegpt4v",
            *(
                item
                for method in ("baseline", "r2", "eplb", "ours")
                for item in ("--summary", f"{method}={aggregate}")
            ),
            "--output-json",
            str(output_json),
            "--output-svg",
            str(output_svg),
        ],
    )
    plot_hiermoe_paper_speedup.main()

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["grad_protocol"] == "paper"
    assert report["methods"][0]["e2e_source"] == "env_step_time_s"
    assert report["provenance"]["communication_source_sha256"] == communication_source_sha256
    assert report["execution_policy"] == "repeat-major-fixed-order"
    assert report["methods"][0]["preflight_report_sha256"]
    assert report["methods"][0]["peak_accelerator_allocated_gib"] == 23.0
    assert report["methods"][0]["peak_accelerator_reserved_gib"] == 27.0


def _preflight_payload() -> dict:
    nodes = []
    for rank in range(4):
        nodes.append(
            {
                "status": "accepted",
                "node_rank": rank,
                "devices": 8,
                "accelerator": "NVIDIA RTX A6000",
                "torch": "2.9.1+cu129",
                "cuda": "12.9",
                "nccl": [2, 27, 5],
                "triton": "3.5.1",
                "nccl_socket_ifname": "ibs0",
                "hostname": f"node-{rank}",
                "network_interface_address": f"00:00:00:00:00:0{rank}",
                "gpu_pci_bus_ids": [f"0000:{index:02x}:00.0" for index in range(8)],
            }
        )
    return {
        "schema_version": 1,
        "status": "accepted",
        "world_size": 32,
        "ep_size": 32,
        "ranks_per_node": 8,
        "nodes": nodes,
    }


def test_cluster_scope_rejects_mixed_software_environments(tmp_path: Path) -> None:
    payload = _preflight_payload()
    payload["nodes"][3]["cuda"] = "12.8"
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, payload)

    with pytest.raises(RuntimeError, match="one software and hardware scope"):
        _cluster_scope(SimpleNamespace(preflight_report=preflight, source_sha256="e" * 64))


def test_communication_calibration_binds_preflight_source_and_validation(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, _preflight_payload())
    source_sha256 = "e" * 64
    scope = _cluster_scope(SimpleNamespace(preflight_report=preflight, source_sha256=source_sha256))
    calibration = tmp_path / "communication.json"
    payload = {
        "schema_version": 3,
        "source": "gpu32-a6000-ep32-communication-calibration",
        "run_name": "test",
        "scope": scope,
        "topology": {
            "accelerator": "NVIDIA RTX A6000",
            "nodes": 4,
            "gpus_per_node": 8,
            "ep_size": 32,
            "ranks_per_node": 8,
            "hierarchy_group_sizes": [8, 32],
            "hidden_size": 2048,
            "bytes_per_element": 2,
        },
        "coefficients": {"inter_ms_per_byte": 1.0e-7, "intra_ms_per_byte": 2.0e-8},
        "coefficient_features": {
            "inter": "raw_balanced_a2a_payload_bytes_per_source",
            "intra": "raw_balanced_a2a_payload_bytes_per_source",
        },
        "link_models": {
            "inter": {
                "kind": "local_alpha_beta_bracketing_workload_payload",
                "alpha_ms": 0.1,
                "beta_ms_per_byte": 1.0e-7,
            },
            "intra": {
                "kind": "local_alpha_beta_bracketing_workload_payload",
                "alpha_ms": 0.05,
                "beta_ms_per_byte": 2.0e-8,
            },
        },
        "validation": {
            "stage1_inter": {
                "kind": "raw_balanced_a2a_local_alpha_beta_holdout",
                "count": 4,
                "mape_percent": 3.0,
            },
            "stage2_intra": {
                "kind": "raw_balanced_a2a_local_alpha_beta_holdout",
                "count": 4,
                "mape_percent": 4.0,
            },
        },
        "coverage": {
            "route_patterns": ["uniform", "skew"],
            "tokens_per_rank": [256, 1024],
            "raw_all_to_all": {"inter_stage_group": [{}], "intra_stage_group": [{}]},
        },
    }
    _write_json(calibration, payload)

    inter, intra, provenance = load_communication_calibration(
        calibration,
        ep_size=32,
        ranks_per_node=8,
        hidden_size=2048,
        bytes_per_element=2,
        preflight_report=preflight,
        communication_source_sha256=source_sha256,
    )

    assert inter == pytest.approx(1.0e-7)
    assert intra == pytest.approx(2.0e-8)
    assert provenance["scope"]["preflight_sha256"] == scope["preflight_sha256"]

    payload["validation"]["stage1_inter"]["mape_percent"] = 10.1
    _write_json(calibration, payload)
    with pytest.raises(ValueError, match="MAPE"):
        load_communication_calibration(
            calibration,
            ep_size=32,
            ranks_per_node=8,
            hidden_size=2048,
            bytes_per_element=2,
        )


def test_three_level_communication_calibration_contract(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, _preflight_payload())
    source_sha256 = "f" * 64
    scope = _cluster_scope(SimpleNamespace(preflight_report=preflight, source_sha256=source_sha256))
    calibration = tmp_path / "communication_v4.json"
    levels = [1.0e-7, 4.0e-8, 2.0e-8]
    stage_names = ("stage1_inter", "stage2_mid", "stage3_intra")
    raw_names = ("stage1_inter_group", "stage2_mid_group", "stage3_intra_group")
    payload = {
        "schema_version": 4,
        "source": "gpu32-a6000-ep32-communication-calibration",
        "run_name": "test-v4",
        "scope": scope,
        "topology": {
            "accelerator": "NVIDIA RTX A6000",
            "nodes": 4,
            "gpus_per_node": 8,
            "ep_size": 32,
            "ranks_per_node": 8,
            "hierarchy_group_sizes": [2, 8, 32],
            "hidden_size": 2048,
            "bytes_per_element": 2,
        },
        "coefficients": {
            "level_ms_per_byte": levels,
            "inter_ms_per_byte": levels[0],
            "mid_ms_per_byte": levels[1],
            "intra_ms_per_byte": levels[2],
        },
        "coefficient_features": {
            "levels": ["raw_balanced_a2a_payload_bytes_per_source"] * 3,
        },
        "link_models": {
            name: {
                "kind": "local_alpha_beta_bracketing_workload_payload",
                "alpha_ms": 0.1,
                "beta_ms_per_byte": coefficient,
            }
            for name, coefficient in zip(stage_names, levels, strict=True)
        },
        "validation": {
            name: {
                "kind": "raw_balanced_a2a_local_alpha_beta_holdout",
                "count": 4,
                "mape_percent": 3.0,
            }
            for name in stage_names
        },
        "coverage": {
            "route_patterns": ["uniform", "skew"],
            "tokens_per_rank": [256, 1024],
            "raw_all_to_all": {name: [{}] for name in raw_names},
        },
    }
    _write_json(calibration, payload)

    inter, intra, provenance = load_communication_calibration(
        calibration,
        ep_size=32,
        ranks_per_node=8,
        hidden_size=2048,
        bytes_per_element=2,
        preflight_report=preflight,
        communication_source_sha256=source_sha256,
    )

    assert inter == pytest.approx(levels[0])
    assert intra == pytest.approx(levels[-1])
    assert provenance["hierarchy_group_sizes"] == [2, 8, 32]
    assert provenance["level_ms_per_byte"] == pytest.approx(levels)
    assert provenance["coverage"]["raw_samples_by_level"] == [1, 1, 1]


def test_three_level_cost_alignment_preserves_shared_anchor_and_workload_fit() -> None:
    shared = [3.4e-7, 1.2e-7, 1.4e-8]
    workload = [3.1e-7, 1.1e-7, 2.3e-8]
    payload = {
        "ours_cost_model_verify": {
            "route_alignment": {
                "stage_link_fit": {
                    "shared_calibration_used_as_topology_anchor": True,
                    "production_dispatch_fit_used_for_offline_scorer": True,
                    "levels": [
                        {
                            "stage": index,
                            "shared_ms_per_byte": shared_value,
                            "workload_reference_ms_per_byte": workload_value,
                            "shared_scale": workload_value / shared_value,
                        }
                        for index, (shared_value, workload_value) in enumerate(
                            zip(shared, workload, strict=True),
                            start=1,
                        )
                    ],
                }
            }
        }
    }

    _validate_three_level_alignment(payload, workload, shared)

    payload["ours_cost_model_verify"]["route_alignment"]["stage_link_fit"]["levels"][1]["shared_ms_per_byte"] *= 2.0
    with pytest.raises(ValueError, match="shared_ms_per_byte"):
        _validate_three_level_alignment(payload, workload, shared)


def test_raw_balanced_link_uses_bracketing_workload_payloads() -> None:
    rows = [
        {
            "pattern": "uniform",
            "payload_bytes_per_source": payload,
            "median_ms": median,
            "samples_ms": samples,
        }
        for payload, median, samples in (
            (4 * 1024**2, 1.0, [0.99, 1.0, 1.01]),
            (8 * 1024**2, 1.5, [1.49, 1.5, 1.51]),
            (32 * 1024**2, 4.0, [3.99, 4.0, 4.01]),
        )
    ]

    beta, model, validation = _fit_raw_balanced_link(
        rows,
        validation_payload_bytes=8 * 1024**2,
        label="test link",
    )

    assert beta > 0.0
    assert model["lower_calibration_payload_bytes"] == 4 * 1024**2
    assert model["upper_calibration_payload_bytes"] == 32 * 1024**2
    assert validation["held_out_payload_bytes_per_source"] == 8 * 1024**2
    assert validation["mape_percent"] < 10.0
