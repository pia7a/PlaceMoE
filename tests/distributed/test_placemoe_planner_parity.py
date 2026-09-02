# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Optional golden regressions for the paper's full EP32 and EP64 plans."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from veomni.distributed.moe.hiermoe.placemoe import (
    LayerPlan,
    PlaceMoETopology,
    build_placemoe_artifact,
)


_PLAN_FIELDS = ("slot_to_logical", "owner_slots", "source_logical_to_physical")


@pytest.mark.parametrize(("redundant_slots", "copies_per_expert"), [(0, 1), (1, 2)])
@pytest.mark.parametrize("fast_approx", [False, True])
def test_single_node_planner_builds_valid_layout(
    tmp_path: Path,
    redundant_slots: int,
    copies_per_expert: int,
    fast_approx: bool,
) -> None:
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
            torch.save(
                {"ep_size": 4, "routes": routes},
                step_root / f"layer00_call0_rank{rank:02d}.pt",
            )
    output_layout = tmp_path / "layout.json"
    output_report = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "placemoe.planner",
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
    if fast_approx:
        command.append("--fast-approx")

    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])

    layout = json.loads(output_layout.read_text(encoding="utf-8"))
    report = json.loads(output_report.read_text(encoding="utf-8"))
    layer = next(iter(layout["layers"].values()))
    assert sorted(layer["slot_to_logical"]) == sorted(list(range(4)) * copies_per_expert)
    assert len(set(layer["owner_slots"])) == 4
    assert all(len(set(row)) == 4 for row in layer["source_logical_to_physical"])
    assert report["aggregate"]["active_replica_slots"] == 4 * redundant_slots
    assert report["aggregate"]["empty_slots"] == 0
    selected_active = sum(value >= 0 for value in layer["slot_to_logical"]) - 4
    assert report["aggregate"]["selected_active_replica_slots_by_layer"] == [selected_active]
    assert report["aggregate"]["selected_empty_slots_by_layer"] == [4 * redundant_slots - selected_active]
    budget = report["aggregate"]["search_budget"]
    assert budget["mode"] == ("fast_approx" if fast_approx else "full")
    assert budget["calibrated_proposals"] is not fast_approx
    assert budget["normalized_proposals"]
    assert not budget["community_proposals"]
    strategies = {candidate["strategy"] for candidate in report["layers"][0]["candidates"]}
    assert any(strategy.startswith("placemoe_normalized_") for strategy in strategies)
    assert any(strategy.startswith("placemoe_p") for strategy in strategies) is not fast_approx

    if fast_approx and redundant_slots == 1:
        incumbent_layout = tmp_path / "incumbent_layout.json"
        incumbent_report = tmp_path / "incumbent_report.json"
        incumbent_command = command.copy()
        incumbent_command.extend(("--input-layout", str(output_layout)))
        incumbent_command[incumbent_command.index("--output-layout") + 1] = str(incumbent_layout)
        incumbent_command[incumbent_command.index("--output-report") + 1] = str(incumbent_report)

        subprocess.run(incumbent_command, check=True, cwd=Path(__file__).resolve().parents[2])

        report = json.loads(incumbent_report.read_text(encoding="utf-8"))
        strategies = {candidate["strategy"] for candidate in report["layers"][0]["candidates"]}
        assert "placemoe_current_incumbent" in strategies
        assert report["layers"][0]["comparison_validation"] is not None
        assert report["aggregate"]["comparison_validation_ms"] == pytest.approx(
            report["layers"][0]["comparison_validation"]["total_ms"]
        )
        assert report["aggregate"]["validation_total_ms"] <= report["aggregate"]["comparison_validation_ms"]

        empty_payload = json.loads(output_layout.read_text(encoding="utf-8"))
        empty_layer = next(iter(empty_payload["layers"].values()))
        empty_layer["slot_to_logical"] = [0, -1, 1, -1, 2, -1, 3, -1]
        empty_layer["owner_slots"] = [0, 2, 4, 6]
        empty_layer["source_logical_to_physical"] = [[0, 2, 4, 6] for _ in range(4)]
        empty_input_layout = tmp_path / "empty_input_layout.json"
        empty_input_layout.write_text(json.dumps(empty_payload), encoding="utf-8")
        empty_output_layout = tmp_path / "empty_incumbent_layout.json"
        empty_output_report = tmp_path / "empty_incumbent_report.json"
        empty_incumbent_command = command.copy()
        empty_incumbent_command.extend(("--input-layout", str(empty_input_layout)))
        empty_incumbent_command[empty_incumbent_command.index("--output-layout") + 1] = str(empty_output_layout)
        empty_incumbent_command[empty_incumbent_command.index("--output-report") + 1] = str(empty_output_report)

        subprocess.run(empty_incumbent_command, check=True, cwd=Path(__file__).resolve().parents[2])

        report = json.loads(empty_output_report.read_text(encoding="utf-8"))
        emitted = json.loads(empty_output_layout.read_text(encoding="utf-8"))
        emitted_layer = next(iter(emitted["layers"].values()))
        selected_active = sum(value >= 0 for value in emitted_layer["slot_to_logical"]) - 4
        strategies = {candidate["strategy"] for candidate in report["layers"][0]["candidates"]}
        assert "placemoe_current_incumbent" in strategies
        assert report["aggregate"]["active_replica_slots"] == selected_active
        assert report["aggregate"]["selected_active_replica_slots_by_layer"] == [selected_active]


def test_mixed_replica_incumbent_is_accepted_by_full_and_mapping_search(tmp_path: Path) -> None:
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
        for layer in range(2):
            for rank, routes in enumerate(samples):
                torch.save(
                    {"ep_size": 4, "routes": routes},
                    step_root / f"layer{layer:02d}_call0_rank{rank:02d}.pt",
                )

    topology = PlaceMoETopology(ep_size=4, ranks_per_node=4, num_experts=4, slots_per_rank=2)
    empty_mapping = [[0, 2, 4, 6] for _ in range(4)]
    replicated_mapping = [[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]
    mixed_artifact = build_placemoe_artifact(
        {
            "model.language_model.layers.0.mlp.experts": LayerPlan(
                slot_to_logical=[0, -1, 1, -1, 2, -1, 3, -1],
                owner_slots=[0, 2, 4, 6],
                source_logical_to_physical=empty_mapping,
            ),
            "model.language_model.layers.1.mlp.experts": LayerPlan(
                slot_to_logical=[0, 1, 2, 3, 0, 1, 2, 3],
                owner_slots=[0, 1, 2, 3],
                source_logical_to_physical=replicated_mapping,
            ),
        },
        topology,
    )
    input_layout = tmp_path / "mixed_input_layout.json"
    input_layout.write_text(json.dumps(mixed_artifact), encoding="utf-8")

    base_command = [
        sys.executable,
        "-m",
        "placemoe.planner",
        "--route-root",
        str(route_root),
        "--optimize-steps",
        "1",
        "--validation-steps",
        "2",
        "--layers",
        "2",
        "--expected-total-layers",
        "2",
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
        "1",
        "--replica-candidate-limit",
        "1",
        "--partition-restarts",
        "1",
        "--alternations",
        "1",
        "--disable-community-block-candidates",
        "--fast-approx",
        "--input-layout",
        str(input_layout),
    ]
    root = Path(__file__).resolve().parents[2]
    for update_mode in ("full", "mapping"):
        output_layout = tmp_path / f"{update_mode}_layout.json"
        output_report = tmp_path / f"{update_mode}_report.json"
        command = [
            *base_command,
            "--update-mode",
            update_mode,
            "--output-layout",
            str(output_layout),
            "--output-report",
            str(output_report),
        ]

        subprocess.run(command, check=True, cwd=root)

        report = json.loads(output_report.read_text(encoding="utf-8"))
        if update_mode == "mapping":
            assert report["aggregate"]["active_replica_slots"] is None
            assert report["aggregate"]["empty_slots"] is None
            assert report["aggregate"]["selected_active_replica_slots_by_layer"] == [0, 4]
            assert report["aggregate"]["selected_empty_slots_by_layer"] == [4, 0]


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} is required for the PlaceMoE golden planner regression")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _planner_command(configuration: dict, route_root: Path, output_layout: Path, output_report: Path) -> list[str]:
    scalar_options = {
        "layer-start": "layer_start",
        "layer-name-template": "layer_name_template",
        "layers": "layers",
        "forward-repeats": "forward_repeats",
        "expected-total-layers": "expected_total_layers",
        "workers": "workers",
        "candidate-workers": "candidate_workers",
        "worker-threads": "worker_threads",
        "ep-size": "ep_size",
        "ranks-per-node": "ranks_per_node",
        "num-experts": "num_experts",
        "slots-per-rank": "slots_per_rank",
        "redundant-slots-per-rank": "redundant_slots_per_rank",
        "primary-slots-per-rank": "primary_slots_per_rank",
        "replica-candidate-limit": "replica_candidate_limit",
        "partition-restarts": "partition_restarts",
        "alternations": "alternations",
        "lut-iterations": "lut_iterations",
        "partition-iterations": "partition_iterations",
        "assignment-iterations": "assignment_iterations",
        "hyperedge-token-sample": "hyperedge_token_sample",
        "structured-shortlist": "structured_shortlist",
        "community-shortlist": "community_shortlist",
        "community-sweeps": "community_sweeps",
        "seed": "seed",
        "hidden-size": "hidden_size",
        "bytes-per-element": "bytes_per_element",
        "inter-ms-per-byte": "inter_ms_per_byte",
        "intra-ms-per-byte": "intra_ms_per_byte",
        "route-ms-per-assignment": "route_ms_per_assignment",
        "communication-phase-multiplier": "communication_phase_multiplier",
        "compute-ms-per-assignment": "compute_ms_per_assignment",
        "compute-phase-multiplier": "compute_phase_multiplier",
        "comparison-validation-ms": "comparison_validation_ms",
        "comparison-layout": "comparison_layout",
    }
    command = [
        sys.executable,
        "-m",
        "placemoe.planner",
        "--route-root",
        str(route_root),
        "--optimize-steps",
        ",".join(str(value) for value in configuration["optimize_steps"]),
        "--validation-steps",
        ",".join(str(value) for value in configuration["validation_steps"]),
        "--call-indices",
        ",".join(str(value) for value in configuration["call_indices"]),
        "--output-layout",
        str(output_layout),
        "--output-report",
        str(output_report),
    ]
    for option, key in scalar_options.items():
        value = configuration.get(key)
        if value is not None:
            command.extend((f"--{option}", str(value)))
    if configuration.get("disable_structured_overlap_candidates", False):
        command.append("--disable-structured-overlap-candidates")
    if configuration.get("fast_approx", False):
        command.append("--fast-approx")
    return command


@pytest.mark.parametrize("ep_size", [32, 64])
def test_paper_plan_matches_historical_layout_and_mapping(ep_size: int, tmp_path: Path) -> None:
    prefix = f"PLACEMOE_EP{ep_size}"
    route_root = _required_path(f"{prefix}_ROUTE_ROOT")
    reference_layout_path = _required_path(f"{prefix}_REFERENCE_LAYOUT")
    reference_report_path = _required_path(f"{prefix}_REFERENCE_REPORT")
    reference_layout = json.loads(reference_layout_path.read_text(encoding="utf-8"))
    configuration = json.loads(reference_report_path.read_text(encoding="utf-8"))["configuration"]
    output_layout = tmp_path / f"ep{ep_size}_layout.json"
    output_report = tmp_path / f"ep{ep_size}_report.json"

    subprocess.run(
        _planner_command(configuration, route_root, output_layout, output_report),
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    actual_layout = json.loads(output_layout.read_text(encoding="utf-8"))
    assert actual_layout["topology"] == reference_layout["topology"]
    assert actual_layout["layers"].keys() == reference_layout["layers"].keys()
    for layer_key, reference_plan in reference_layout["layers"].items():
        actual_plan = actual_layout["layers"][layer_key]
        for field in _PLAN_FIELDS:
            assert actual_plan[field] == reference_plan[field], f"EP{ep_size} {layer_key} differs in {field}"
