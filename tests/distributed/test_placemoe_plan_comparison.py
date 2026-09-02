# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Regression tests for the complete-route PlaceMoE plan comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from veomni.distributed.moe.hiermoe.placemoe.artifacts import build_placemoe_artifact
from veomni.distributed.moe.hiermoe.placemoe.types import EMPTY_EXPERT, LayerPlan, PlaceMoETopology


def test_compare_plan_replays_canonical_layout(tmp_path: Path) -> None:
    route_root = tmp_path / "routes"
    step_root = route_root / "step0000"
    step_root.mkdir(parents=True)
    routes = [
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int64),
        torch.tensor([[1, 2], [1, 3]], dtype=torch.int64),
        torch.tensor([[2, 3], [0, 2]], dtype=torch.int64),
        torch.tensor([[0, 3], [1, 3]], dtype=torch.int64),
    ]
    layer_key = "layers.0.experts"
    torch.save(
        {
            "format": "hiermoe-local-route-bundle-v1",
            "ep_size": 4,
            "layer_key": layer_key,
            "routes_by_rank": routes,
        },
        step_root / "layer00_call0_all_ranks.pt",
    )

    topology = PlaceMoETopology(ep_size=4, ranks_per_node=4, num_experts=4, slots_per_rank=2)
    owners = np.array([0, 2, 4, 6], dtype=np.int64)
    layout = np.full((topology.total_slots,), EMPTY_EXPERT, dtype=np.int64)
    layout[owners] = np.arange(topology.num_experts, dtype=np.int64)
    source_lut = np.broadcast_to(owners, (topology.ep_size, topology.num_experts)).copy()
    artifact = build_placemoe_artifact(
        {
            layer_key: LayerPlan(
                slot_to_logical=layout,
                source_logical_to_physical=source_lut,
                owner_slots=owners,
            )
        },
        topology,
        source={
            "optimize_steps": [0],
            "validation_steps": [0],
            "layer_keys": [layer_key],
            "hierarchy_group_sizes": [4],
        },
    )
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(artifact), encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "coefficients": {
                    "inter_ms_per_byte": 2.0e-8,
                    "intra_ms_per_byte": 1.0e-8,
                    "route_ms_per_assignment": 1.0e-4,
                    "communication_multiplier": 2.0,
                    "compute_ms_per_assignment": 1.0e-3,
                    "compute_multiplier": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "comparison.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/profile/compare_plan.py",
            "--route-root",
            str(route_root),
            "--layout",
            str(layout_path),
            "--calibration",
            str(calibration_path),
            "--hidden-size",
            "8",
            "--steps",
            "0",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evaluation_scope"] == "in_sample"
    assert report["aggregate"]["speedup"]["communication"] == pytest.approx(1.0)
    assert report["aggregate"]["speedup"]["compute"] == pytest.approx(1.0)
    assert report["aggregate"]["speedup"]["total"] == pytest.approx(1.0)
