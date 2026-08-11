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

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from tests.tools.launch_utils import torchrun
from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import oracle as oracle_module
from veomni.distributed.moe.hiermoe.all_to_all import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.oracle import (
    RouteSnapshot,
    best_found_replica_curve,
    best_found_swap_curve,
    communication_cost_from_targets,
    load_route_snapshot,
    maybe_capture_route_snapshot,
    oracle_replica_curve,
    oracle_swap_curve,
    route_count_greedy_replica_curve,
    route_count_greedy_swap_curve,
    save_route_snapshot,
)
from veomni.distributed.moe.hiermoe.state import (
    configure_hiermoe,
    disable_hiermoe_placement,
    get_hiermoe_state,
    set_hiermoe_route_capture_forward_enabled,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


_PROFILED_PERF_MODEL_PATH = str(Path(__file__).with_name("fixtures") / "hiermoe_profile.json")


def _profiled_hiermoe_config(**kwargs) -> HierMoEConfig:
    return HierMoEConfig(perf_model_path=_PROFILED_PERF_MODEL_PATH, **kwargs)


def _toy_snapshot() -> RouteSnapshot:
    return RouteSnapshot(
        routes_by_rank=(
            torch.tensor([[2], [2], [2], [2]]),
            torch.tensor([[1], [1]]),
            torch.tensor([[0], [0], [0], [0]]),
            torch.tensor([[3], [3]]),
        ),
        num_experts=4,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        logical_to_physical=torch.arange(4),
        layer_key="model.layers.1.mlp.experts",
        step=3,
        call_index=0,
        selected_dim=2,
    )


def _capture_route_worker(output_path: str) -> None:
    rank = dist.get_rank()
    selected_experts = torch.full((rank + 1, 1), rank, dtype=torch.long)
    maybe_capture_route_snapshot(
        selected_experts=selected_experts,
        num_experts=2,
        hidden_size=4,
        bytes_per_element=2,
        ep_group=dist.group.WORLD,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        layer_key="model.layers.0.mlp.experts",
        step=0,
        selected_dim=1,
    )


def _capture_local_route_worker(_output_path: str) -> None:
    rank = dist.get_rank()
    selected_experts = torch.full((rank + 1, 2), rank, dtype=torch.long)
    slot_to_logical = torch.tensor([0, 1, rank], dtype=torch.long)

    def forbidden_collective(*_args, **_kwargs):
        raise AssertionError("Local route capture must not issue a route collective.")

    original_all_gather = dist.all_gather
    original_all_gather_into_tensor = dist.all_gather_into_tensor
    dist.all_gather = forbidden_collective
    dist.all_gather_into_tensor = forbidden_collective
    try:
        maybe_capture_route_snapshot(
            selected_experts=selected_experts,
            num_experts=2,
            hidden_size=4,
            bytes_per_element=2,
            ep_group=dist.group.WORLD,
            hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
            layer_key="model.layers.7.mlp.experts",
            step=3,
            logical_to_physical=torch.arange(2),
            slot_to_logical=slot_to_logical,
            selected_dim=1,
        )
    finally:
        dist.all_gather = original_all_gather
        dist.all_gather_into_tensor = original_all_gather_into_tensor


def test_distributed_capture_handles_unequal_route_lengths_and_single_writer(tmp_path, monkeypatch):
    output = tmp_path / "route.pt"
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", str(output))
    torchrun(_capture_route_worker, 2, str(output), backend="gloo")

    snapshot = load_route_snapshot(output)
    assert snapshot.communication_dimension == 1
    assert [routes.shape[0] for routes in snapshot.routes_by_rank] == [1, 2]
    assert torch.equal(snapshot.routes_by_rank[0], torch.tensor([[0]]))
    assert torch.equal(snapshot.routes_by_rank[1], torch.tensor([[1], [1]]))


def test_distributed_local_capture_writes_each_rank_without_route_collective(tmp_path, monkeypatch):
    output = tmp_path / "step{step:04d}" / "layer{layer_index:02d}_rank{rank:02d}.pt"
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", str(output))
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE", "local")
    torchrun(_capture_local_route_worker, 2, str(output), backend="gloo")

    for rank in range(2):
        path = tmp_path / "step0003" / f"layer07_rank{rank:02d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format"] == "veomni.hiermoe.local_route"
        assert payload["version"] == 1
        assert payload["global_rank"] == rank
        assert payload["ep_rank"] == rank
        assert payload["ep_size"] == 2
        assert payload["step"] == 3
        assert payload["layer"] == 7
        assert payload["call_index"] == 0
        assert payload["layer_key"] == "model.layers.7.mlp.experts"
        assert torch.equal(payload["routes"], torch.full((rank + 1, 2), rank, dtype=torch.int32))
        assert torch.equal(payload["logical_to_physical"], torch.arange(2))
        assert torch.equal(payload["slot_to_logical"], torch.tensor([0, 1, rank]))


def test_ordinal_route_capture_wraps_gradient_accumulation_for_call_filter(tmp_path, monkeypatch):
    output = tmp_path / "step{step:04d}" / "layer{layer_index:02d}_call{call}_rank{rank:02d}.pt"
    monkeypatch.setattr(oracle_module, "_CAPTURE_PATH_TEMPLATE", str(output))
    monkeypatch.setattr(oracle_module, "_CAPTURE_CALLS", {})
    monkeypatch.setattr(oracle_module, "_CAPTURED", set())
    monkeypatch.setattr(oracle_module, "_CAPTURE_LAYER_ORDINALS", {})
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE", "local")
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL", "0")
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS", "2")

    outputs = []
    for value in range(4):
        outputs.append(
            maybe_capture_route_snapshot(
                selected_experts=torch.tensor([[value % 2]], dtype=torch.long),
                num_experts=2,
                hidden_size=4,
                bytes_per_element=2,
                ep_group=None,
                hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
                layer_key=None,
                step=0,
                selected_dim=1,
            )
        )

    assert outputs[:2] == [
        tmp_path / "step0000" / "layer00_call0_rank00.pt",
        tmp_path / "step0000" / "layer01_call0_rank00.pt",
    ]
    assert outputs[2:] == [None, None]
    assert sorted(path.name for path in (tmp_path / "step0000").glob("*.pt")) == [
        "layer00_call0_rank00.pt",
        "layer01_call0_rank00.pt",
    ]
    assert oracle_module._CAPTURE_CALLS[(0, "model.layers.0.mlp.experts")] == 2
    assert oracle_module._CAPTURE_CALLS[(0, "model.layers.1.mlp.experts")] == 2


class _FakeExperts(torch.nn.Module):
    def __init__(self, num_experts: int, num_local_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = torch.nn.Parameter(torch.ones(num_local_experts, 2, 2))
        self.down_proj = torch.nn.Parameter(torch.ones(num_local_experts, 2, 2))


def _capture_skips_placement_disabled_reference_worker(_output_path: str) -> None:
    rank = dist.get_rank()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            redundant_slot_increment_per_device=0,
            hierarchy_group_sizes=[dist.get_world_size()],
        ),
        dist.group.WORLD,
    )
    hidden = torch.ones((3, 2), dtype=torch.float32) * (rank + 1)
    weights = torch.ones((3, 1), dtype=torch.float32)
    with disable_hiermoe_placement():
        rank_dedup_dispatch(
            hidden,
            torch.zeros((3, 1), dtype=torch.long),
            weights,
            num_experts=2,
            ep_group=dist.group.WORLD,
            layer_key=None,
        )

    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None
    layer_key = "model.layers.0.mlp.experts"
    state.expert_swap_manager.register_layer(layer_key, _FakeExperts(num_experts=2, num_local_experts=1))
    previous_capture = set_hiermoe_route_capture_forward_enabled(True)
    try:
        rank_dedup_dispatch(
            hidden,
            torch.ones((3, 1), dtype=torch.long),
            weights,
            num_experts=2,
            ep_group=dist.group.WORLD,
            layer_key=layer_key,
        )
    finally:
        set_hiermoe_route_capture_forward_enabled(previous_capture)


def test_capture_skips_placement_disabled_reference_forward(tmp_path, monkeypatch):
    output = tmp_path / "policy_route.pt"
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", str(output))
    torchrun(_capture_skips_placement_disabled_reference_worker, 2, str(output), backend="gloo")

    snapshot = load_route_snapshot(output)
    assert snapshot.call_index == 0
    assert snapshot.layer_key == "model.layers.0.mlp.experts"
    for routes in snapshot.routes_by_rank:
        torch.testing.assert_close(routes, torch.ones((3, 1), dtype=torch.long))


def _checkpoint_route_capture_worker(_output_path: str) -> None:
    rank = dist.get_rank()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            hierarchy_group_sizes=[dist.get_world_size()],
        ),
        dist.group.WORLD,
    )
    layer_key = "model.layers.0.mlp.experts"
    selected = torch.full((4, 1), rank, dtype=torch.long)
    routing_weights = torch.ones((4, 1), dtype=torch.float64)

    def moe_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        dispatched, context, _counts = rank_dedup_dispatch(
            hidden_states,
            selected,
            routing_weights,
            num_experts=dist.get_world_size(),
            ep_group=dist.group.WORLD,
            layer_key=layer_key,
        )
        return rank_dedup_combine(dispatched * 2.0, context)

    baseline_hidden = torch.arange(8, dtype=torch.float64).view(4, 2).requires_grad_()
    baseline_output = moe_forward(baseline_hidden)
    baseline_output.sum().backward()

    checkpoint_hidden = baseline_hidden.detach().clone().requires_grad_()
    previous_capture = set_hiermoe_route_capture_forward_enabled(True)
    try:
        checkpoint_output = checkpoint(moe_forward, checkpoint_hidden, use_reentrant=True)
    finally:
        set_hiermoe_route_capture_forward_enabled(previous_capture)
    checkpoint_output.sum().backward()

    torch.testing.assert_close(checkpoint_output, baseline_output)
    torch.testing.assert_close(checkpoint_hidden.grad, baseline_hidden.grad)
    assert oracle_module._CAPTURE_CALLS[(0, layer_key)] == 1


def test_checkpoint_recompute_does_not_recapture_routes_and_preserves_gradients(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint_route.pt"
    monkeypatch.setenv("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", str(output))
    torchrun(_checkpoint_route_capture_worker, 2, str(output), backend="gloo")

    snapshot = load_route_snapshot(output)
    assert snapshot.call_index == 0
    assert snapshot.layer_key == "model.layers.0.mlp.experts"
    for rank, routes in enumerate(snapshot.routes_by_rank):
        torch.testing.assert_close(routes, torch.full((4, 1), rank, dtype=torch.long))


def test_route_snapshot_round_trip(tmp_path):
    source = _toy_snapshot()
    path = save_route_snapshot(source, tmp_path / "route.pt")
    loaded = load_route_snapshot(path)

    assert loaded.num_experts == source.num_experts
    assert loaded.hierarchy == source.hierarchy
    assert loaded.layer_key == source.layer_key
    assert loaded.step == source.step
    assert loaded.communication_dimension == source.communication_dimension
    for actual, expected in zip(loaded.routes_by_rank, source.routes_by_rank, strict=True):
        torch.testing.assert_close(actual, expected)


def test_2d_cost_deduplicates_topk_at_node_and_rank_stages():
    snapshot = RouteSnapshot(
        routes_by_rank=(
            torch.tensor([[2, 3]]),
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
        ),
        num_experts=4,
        hidden_size=1,
        bytes_per_element=1,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        logical_to_physical=torch.arange(4),
        layer_key="model.layers.0.mlp.experts",
        step=0,
        call_index=0,
        selected_dim=2,
    )
    targets = snapshot.routes_by_rank
    cost = communication_cost_from_targets(snapshot, targets)

    assert cost.inter_peak_tokens == 1
    assert cost.intra_peak_tokens == 1


def test_communication_cost_rejects_unsupported_3d_hierarchy():
    source = _toy_snapshot()
    snapshot = RouteSnapshot(
        routes_by_rank=source.routes_by_rank,
        num_experts=source.num_experts,
        hidden_size=source.hidden_size,
        bytes_per_element=source.bytes_per_element,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(1, 2, 4), source="test"),
        logical_to_physical=source.logical_to_physical,
        layer_key=source.layer_key,
        step=source.step,
        call_index=source.call_index,
        selected_dim=3,
    )

    with pytest.raises(NotImplementedError, match="flat and 2D hierarchies only"):
        communication_cost_from_targets(snapshot, snapshot.routes_by_rank)


def test_best_found_swap_curve_finds_zero_communication_locality_swap():
    curve = best_found_swap_curve(_toy_snapshot(), max_pairs=2)

    assert curve.points[0].speedup == 1.0
    assert curve.points[1].cost == 0.0
    assert math.isinf(curve.points[1].speedup)
    assert [point.cost for point in curve.points] == sorted(
        (point.cost for point in curve.points),
        reverse=True,
    )


def test_route_count_swap_curve_uses_disjoint_pairs():
    curve = route_count_greedy_swap_curve(_toy_snapshot(), max_pairs=2)

    for point in curve.points:
        experts = [int(value) for action in point.actions for value in action.split("<->")]
        assert len(experts) == len(set(experts))


def test_replica_curves_reach_zero_at_full_local_replication():
    snapshot = _toy_snapshot()
    route_count = route_count_greedy_replica_curve(snapshot, snapshot.full_replica_slots_per_rank)
    best_found = best_found_replica_curve(snapshot, snapshot.full_replica_slots_per_rank)

    for curve in (route_count, best_found):
        assert curve.points[0].speedup == 1.0
        assert curve.points[-1].budget == snapshot.full_replica_slots_per_rank
        assert curve.points[-1].cost == 0.0
        assert math.isinf(curve.points[-1].speedup)


def test_best_found_replica_curve_is_monotonic():
    curve = best_found_replica_curve(_toy_snapshot(), _toy_snapshot().full_replica_slots_per_rank)
    costs = [point.cost for point in curve.points]
    assert costs == sorted(costs, reverse=True)


def test_legacy_oracle_curve_names_warn_that_results_are_heuristics():
    snapshot = _toy_snapshot()
    with pytest.warns(DeprecationWarning, match="not an oracle"):
        oracle_swap_curve(snapshot, max_pairs=1)
    with pytest.warns(DeprecationWarning, match="not an oracle"):
        oracle_replica_curve(snapshot, max_slots_per_rank=1)


def test_zero_communication_baseline_has_unit_speedup():
    snapshot = RouteSnapshot(
        routes_by_rank=(torch.tensor([[0]]), torch.tensor([[1]])),
        num_experts=2,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        logical_to_physical=torch.arange(2),
        layer_key="model.layers.0.mlp.experts",
        step=0,
        call_index=0,
        selected_dim=1,
    )

    curve = best_found_swap_curve(snapshot, max_pairs=0)
    assert curve.points[0].cost == 0.0
    assert curve.points[0].speedup == 1.0
    assert curve.points[0].remaining_fraction == 1.0
