# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.profile import placemoe_planner
from veomni.arguments.arguments_types import HierMoEConfig, PlaceMoEArguments
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.placemoe import PlacementConfig, place_instances
from veomni.distributed.moe.hiermoe.placemoe.calibration import (
    ModelCalibrationSchedule,
    materialize_model_calibration_config,
)
from veomni.distributed.moe.hiermoe.placemoe.preparation import (
    build_preparation_spec,
    inspect_runtime_cache,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy, expected_hierarchy_group_sizes


def test_single_node_hierarchy_and_zero_replica_preset(tmp_path) -> None:
    assert expected_hierarchy_group_sizes(8, 8) == (8,)
    assert expected_hierarchy_group_sizes(16, 8) == (8, 16)

    config = HierMoEConfig(
        hierarchy_group_sizes=[8],
        placemoe=PlaceMoEArguments(enabled=True),
        redundant_slot_increment_per_device=0,
    )
    assert config.expert_swap_selector == "current_joint"
    assert not config.fixed_pipeline_overlap

    source = {
        "train": {
            "accelerator": {"ep_size": 8},
            "optimizer": {"lr": 1.0e-5, "lr_min": 1.0e-7},
            "hiermoe": {
                "hierarchy_group_sizes": [8],
                "redundant_slot_increment_per_device": 0,
            },
        }
    }
    materialized = materialize_model_calibration_config(
        source,
        runtime_perf_model=tmp_path / "runtime.json",
        work_directory=tmp_path,
        schedule=ModelCalibrationSchedule(),
    )
    hiermoe = materialized["train"]["hiermoe"]
    assert hiermoe["redundant_slot_increment_per_device"] == 0
    assert hiermoe["expert_swap_selector"] == "current_joint"
    assert not hiermoe["fixed_pipeline_overlap"]


def test_single_node_planner_and_placement_use_rank_level_only() -> None:
    args = SimpleNamespace(
        hierarchy_group_sizes=(4,),
        ep_size=4,
        ranks_per_node=4,
        inter_ms_per_byte=9.0,
        intra_ms_per_byte=2.0,
        mid_ms_per_byte=None,
        hidden_size=8,
        bytes_per_element=2,
        communication_phase_multiplier=3.0,
        compute_phase_multiplier=4.0,
        compute_ms_per_assignment=5.0,
    )
    hierarchy, omegas, gamma = placemoe_planner._hierarchy_coefficients(args)
    assert hierarchy == (4,)
    assert omegas == (96.0,)
    assert gamma == 20.0

    placement = place_instances(
        np.eye(4, dtype=np.float64) * 10.0,
        np.zeros((4, 4, 4), dtype=np.float64),
        np.arange(4, dtype=np.int64),
        PlacementConfig(
            ep_size=4,
            ranks_per_node=4,
            slots_per_rank=1,
            node_omega=0.25,
            rank_omega=0.25,
            gamma=1.0,
            hierarchy_group_sizes=(4,),
            level_omegas=(0.25,),
            node_exchange_limit=4,
            rank_exchange_limit=2,
            seed=7,
        ),
    )
    np.testing.assert_array_equal(np.bincount(placement.instance_ranks, minlength=4), [1, 1, 1, 1])
    assert placement.node_objective == 0.0
    assert placement.rank_objective == placement.level_objectives[0]


def test_single_node_traffic_features_and_zero_replica_hot_capture() -> None:
    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=8,
        bytes_per_element=2,
        slots_per_rank=1,
    )
    unique = torch.eye(4, dtype=torch.float32).view(4, 1, 4)
    features = planner._hierarchical_traffic_features(unique, 2.0 * unique)
    assert features["stage1_payload_endpoint_bytes"].item() == 0.0
    assert features["stage2_payload_endpoint_bytes"].item() > 0.0

    manager = object.__new__(ExpertSwapManager)
    manager._hot_update = True
    manager._cost_model_verify = False
    manager._online_lut_update = False
    manager._initial_layout_path = ""
    manager._ablation_replay_mode = "off"
    manager.expert_swap_max_pairs_per_layer = 0
    manager.redundant_slot_increment_per_device = 0
    assert manager.placement_planning_enabled()


def test_single_node_planner_builds_zero_and_redundant_layouts(tmp_path: Path) -> None:
    route_root = tmp_path / "routes"
    samples = (
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int64),
        torch.tensor([[1, 2], [1, 3]], dtype=torch.int64),
        torch.tensor([[2, 3], [0, 2]], dtype=torch.int64),
        torch.tensor([[0, 3], [1, 3]], dtype=torch.int64),
    )
    for step in (1, 2):
        step_root = route_root / f"step{step:04d}"
        step_root.mkdir(parents=True)
        for rank, routes in enumerate(samples):
            torch.save({"ep_size": 4, "routes": routes}, step_root / f"layer00_call0_rank{rank:02d}.pt")

    for redundant_slots, copies_per_expert in ((0, 1), (1, 2)):
        output_layout = tmp_path / f"layout-r{redundant_slots}.json"
        output_report = tmp_path / f"report-r{redundant_slots}.json"
        command = [
            sys.executable,
            "scripts/profile/plan_placemoe.py",
            "--route-root",
            str(route_root),
            "--optimize-steps",
            "1",
            "--validation-steps",
            "2",
            "--layers",
            "1",
            "--expected-total-layers",
            "1",
            "--workers",
            "1",
            "--candidate-workers",
            "1",
            "--ep-size",
            "4",
            "--ranks-per-node",
            "4",
            "--hierarchy-group-sizes",
            "4",
            "--num-experts",
            "4",
            "--redundant-slots-per-rank",
            str(redundant_slots),
            "--replica-candidate-limit",
            "1",
            "--partition-restarts",
            "1",
            "--alternations",
            "1",
            "--disable-community-block-candidates",
            "--output-layout",
            str(output_layout),
            "--output-report",
            str(output_report),
        ]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])

        payload = json.loads(output_layout.read_text(encoding="utf-8"))
        layer = next(iter(payload["layers"].values()))
        assert sorted(layer["slot_to_logical"]) == sorted(list(range(4)) * copies_per_expert)
        assert all(len(set(row)) == 4 for row in layer["source_logical_to_physical"])
        report = json.loads(output_report.read_text(encoding="utf-8"))
        assert report["aggregate"]["active_replica_slots"] == 4 * redundant_slots


def test_single_node_preparation_accepts_rank_only_runtime_model(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.touch()
    entrypoint = tmp_path / "train.py"
    entrypoint.touch()
    source = {
        "model": {"model_path": "/models/demo"},
        "train": {
            "accelerator": {"ep_size": 4},
            "hiermoe": {
                "hierarchy_group_sizes": [4],
                "placemoe": {
                    "enabled": True,
                    "base_directory": str(tmp_path),
                    "runtime_perf_model": "runtime.json",
                    "calibration": {"artifact": "planner.json"},
                },
            },
        },
    }
    spec = build_preparation_spec(
        source,
        config_path=config_path,
        entrypoint=entrypoint,
        nnodes=1,
        nproc_per_node=4,
        runtime_device_type="npu",
        runtime_backend="hccl",
        runtime_dtype="bf16",
    )
    spec.runtime_artifact.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": "bench_hiermoe_perf_model",
                "status": "accepted",
                "a2a": {"alpha": 1.0, "beta": 2.0e-9},
                "inter": [],
                "intra": {"alpha": 1.0, "beta": 2.0e-9},
                "state_move": {
                    "intra": {"alpha": 1.0, "beta": 2.0e-9},
                    "inter": {"alpha": 1.0, "beta": 2.0e-8},
                },
                "gradient_sync": {
                    phase: {
                        "intra": {"alpha": 1.0, "beta": 2.0e-9},
                        "inter": {"alpha": 1.0, "beta": 2.0e-8},
                    }
                    for phase in ("gather", "scatter")
                },
                "metadata": {
                    "ep_size": 4,
                    "ranks_per_node": 4,
                    "hierarchy_group_sizes": [4],
                    "device_type": "npu",
                    "backend": "hccl",
                    "dtype": "bf16",
                },
            }
        ),
        encoding="utf-8",
    )

    assert spec.hierarchy_group_sizes == (4,)
    assert inspect_runtime_cache(spec.runtime_artifact, spec).state == "valid"
