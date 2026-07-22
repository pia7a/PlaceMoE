# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import importlib
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.checkpoint import noop_context_fn

from tests.tools.launch_utils import torchrun
from veomni.arguments import HierMoEConfig, TrainingArguments, VeOmniArguments, parse_args
from veomni.distributed import torch_parallelize as torch_parallelize_module
from veomni.distributed.checkpoint import CheckpointFunction as VeOmniCheckpointFunction
from veomni.distributed.fsdp2.clip_grad_norm import (
    _combine_reduced_norm_totals,
    _finalize_total_norm,
    _fsdp2_reduce_group,
)
from veomni.distributed.moe.hiermoe import all_to_all as hiermoe_all_to_all
from veomni.distributed.moe.hiermoe import (
    assert_hiermoe_trainable_only_checkpoint_safe,
    hiermoe_has_non_identity_placement,
    hiermoe_state_dict,
    load_hiermoe_state_dict,
)
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe import state as hiermoe_state_module
from veomni.distributed.moe.hiermoe.all_to_all import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.legacy_batched_selector import LegacyBatchedSelector
from veomni.distributed.moe.hiermoe.perf_model import (
    GradientSyncCost,
    HierMoEPerfModel,
    LinkCost,
    PeerTransferCost,
    fit_link_cost,
)
from veomni.distributed.moe.hiermoe.planner import PlacementAction
from veomni.distributed.moe.hiermoe.routing import duplicate_free_counts_by_expert_group
from veomni.distributed.moe.hiermoe.state import HierMoEState, configure_hiermoe, get_hiermoe_state
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_plan import _is_hiermoe_redundant_slot_expert_param as plan_allows_slot_padding
from veomni.models.module_utils import _is_hiermoe_redundant_slot_expert_param as loader_allows_slot_padding
from veomni.trainer import base as trainer_base_module
from veomni.trainer.base import BaseTrainer


_PROFILED_PERF_MODEL_PATH = str(Path(__file__).with_name("fixtures") / "hiermoe_profile.json")


def _profiled_hiermoe_config(**kwargs) -> HierMoEConfig:
    return HierMoEConfig(perf_model_path=_PROFILED_PERF_MODEL_PATH, **kwargs)


def test_hiermoe_config_defaults_and_cli_override(tmp_path, monkeypatch):
    assert TrainingArguments().hiermoe.enable is False
    assert TrainingArguments().hiermoe.communication_mode == "hierarchical"
    assert TrainingArguments().hiermoe.expert_swap is True
    assert TrainingArguments().hiermoe.expert_swap_max_pairs_per_layer == 1
    assert TrainingArguments().hiermoe.expert_swap_selector == "current_joint"
    assert TrainingArguments().hiermoe.redundant_slot_increment_per_device == 0
    assert TrainingArguments().hiermoe.max_slot_op_search_rounds is None
    assert TrainingArguments().hiermoe.planner_route_sample_size == 1024
    assert TrainingArguments().hiermoe.expert_swap_mode == "step"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  config_path: Qwen/Qwen3-VL-30B-A3B-Instruct
  ops_implementation:
    moe_implementation: fused_npu
    cross_entropy_loss_implementation: npu
    rms_norm_implementation: npu
    rotary_pos_emb_implementation: npu
    rotary_pos_emb_vision_implementation: npu
    swiglu_mlp_implementation: eager
    load_balancing_loss_implementation: eager
train:
  hiermoe:
    enable: false
data:
  train_path: /tmp/hiermoe_dummy.jsonl
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            str(config_path),
            "--train.hiermoe.enable",
            "true",
            "--train.hiermoe.communication_mode",
            "direct",
            "--train.hiermoe.expert_swap",
            "false",
            "--train.hiermoe.expert_swap_max_pairs_per_layer",
            "2",
            "--train.hiermoe.expert_swap_selector",
            "current_joint",
            "--train.hiermoe.redundant_slot_increment_per_device",
            "1",
            "--train.hiermoe.max_slot_op_search_rounds",
            "3",
            "--train.hiermoe.planner_route_sample_size",
            "512",
            "--train.hiermoe.expert_swap_mode",
            "layer",
        ],
    )
    args = parse_args(VeOmniArguments)
    assert args.train.hiermoe.enable is True
    assert args.train.hiermoe.communication_mode == "direct"
    assert args.train.hiermoe.expert_swap is False
    assert args.train.hiermoe.expert_swap_max_pairs_per_layer == 2
    assert args.train.hiermoe.expert_swap_selector == "current_joint"
    assert args.train.hiermoe.redundant_slot_increment_per_device == 1
    assert args.train.hiermoe.max_slot_op_search_rounds == 3
    assert args.train.hiermoe.planner_route_sample_size == 512
    assert args.train.hiermoe.expert_swap_mode == "layer"


def test_hiermoe_route_sample_size_must_be_positive():
    with pytest.raises(ValueError, match="planner_route_sample_size must be > 0"):
        HierMoEConfig(planner_route_sample_size=0)


def test_hiermoe_communication_mode_must_be_supported():
    with pytest.raises(ValueError, match="communication_mode must be auto, direct, or hierarchical"):
        HierMoEConfig(communication_mode="unsupported")


def test_perf_model_runtime_cost_round_trip_and_legacy_fallback(tmp_path):
    intra = LinkCost(alpha=0.2, beta=0.3)
    inter = LinkCost(alpha=0.8, beta=0.9)
    state_move = PeerTransferCost(intra=intra, inter=inter)
    gradient_sync = GradientSyncCost(
        gather=state_move,
        scatter=PeerTransferCost(
            intra=LinkCost(alpha=0.4, beta=0.5),
            inter=LinkCost(alpha=1.0, beta=1.1),
        ),
    )
    model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=2.0),
        inter=(inter,),
        intra=intra,
        source="bench_hiermoe_perf_model",
        state_move=state_move,
        gradient_sync=gradient_sync,
        schema_version=2,
    )
    model_path = tmp_path / "perf.json"
    model_path.write_text(json.dumps(model.to_payload()), encoding="utf-8")

    loaded = HierMoEPerfModel.from_path(str(model_path))

    assert loaded.has_runtime_placement_costs
    assert loaded.is_profiled
    assert loaded.profile_source == "bench_hiermoe_perf_model"
    assert loaded.schema_version == 2
    assert loaded.runtime_cost_status == "complete"
    assert loaded.resolved_state_move() == state_move
    assert loaded.resolved_gradient_sync() == gradient_sync

    legacy = HierMoEPerfModel.default()
    assert not legacy.is_profiled
    assert not legacy.has_runtime_placement_costs
    assert legacy.runtime_cost_status == "fallback"
    assert legacy.resolved_state_move() == PeerTransferCost(intra=legacy.intra, inter=legacy.inter[-1])
    assert legacy.resolved_gradient_sync().gather == legacy.resolved_state_move()


def test_hiermoe_placement_rejects_missing_or_unverified_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _group: 0)
    config_kwargs = {
        "enable": True,
        "expert_swap": True,
        "hierarchy_group_sizes": [2],
    }

    with pytest.raises(ValueError, match="requires profiled alpha/beta coefficients"):
        configure_hiermoe(HierMoEConfig(**config_kwargs), object())

    unverified_path = tmp_path / "unverified.json"
    payload = HierMoEPerfModel.default().to_payload()
    payload["source"] = "manual"
    unverified_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires profiled alpha/beta coefficients"):
        configure_hiermoe(HierMoEConfig(perf_model_path=str(unverified_path), **config_kwargs), object())


def test_hiermoe_profile_guard_only_applies_to_active_planning(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _group: 0)

    fixed_layout_state = configure_hiermoe(
        HierMoEConfig(
            enable=True,
            expert_swap=True,
            expert_swap_max_pairs_per_layer=0,
            redundant_slot_increment_per_device=1,
            max_slot_op_search_rounds=0,
            hierarchy_group_sizes=[2],
        ),
        object(),
    )
    assert fixed_layout_state.expert_swap_manager is not None
    assert fixed_layout_state.max_replica_rounds == 0

    clamped_state = configure_hiermoe(
        HierMoEConfig(
            enable=True,
            expert_swap=True,
            expert_swap_max_pairs_per_layer=0,
            redundant_slot_increment_per_device=0,
            max_slot_op_search_rounds=3,
            hierarchy_group_sizes=[2],
        ),
        object(),
    )
    assert clamped_state.expert_swap_manager is None
    assert clamped_state.max_replica_rounds == 0


def test_hiermoe_placement_accepts_benchmark_profile(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _group: 0)

    state = configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            expert_swap=True,
            hierarchy_group_sizes=[2],
        ),
        object(),
    )

    assert state.perf_model.is_profiled
    assert state.perf_model.profile_source == "bench_hiermoe_perf_model"


def test_fit_link_cost_recovers_linear_model_and_clamps_noise():
    fitted = fit_link_cost(((0, 2.0), (100, 3.0), (200, 4.0)))

    assert fitted.alpha == pytest.approx(2.0)
    assert fitted.beta == pytest.approx(0.01)

    noisy = fit_link_cost(((0, 1.0), (100, 0.9), (200, 0.8)))
    assert noisy.alpha == pytest.approx(0.9)
    assert noisy.beta == 0.0


@pytest.mark.parametrize(
    ("configured", "slots_per_rank", "ep_size", "expected"),
    [
        (None, 1, 16, 16),
        (None, 2, 16, 32),
        (None, 0, 16, 0),
        (0, 1, 16, 0),
        (3, 1, 16, 3),
        (32, 1, 16, 16),
    ],
)
def test_hiermoe_replica_round_budget_resolution(configured, slots_per_rank, ep_size, expected):
    assert hiermoe_state_module._resolve_max_replica_rounds(configured, slots_per_rank, ep_size) == expected


def test_placement_metrics_report_configured_capacity_and_effective_replica_rounds():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=4,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        configured_max_replica_rounds=None,
        replica_slot_capacity=4,
        planner_route_sample_size=512,
    )

    manager._begin_metrics_step(1)

    assert manager.placement_metrics() == {
        "hiermoe/placement_replica_rounds_configured": "auto",
        "hiermoe/placement_replica_slot_capacity": 4,
        "hiermoe/placement_replica_rounds_effective": 4,
        "hiermoe/placement_route_sample_size": 512,
        "hiermoe/placement_runtime_cost_model": "fallback",
    }


def test_current_planner_disables_collective_digest(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
    )
    monkeypatch.setattr(dist, "get_backend", lambda _group: "gloo")
    manager.register_layer(
        "layers.0.mlp.experts",
        _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2),
    )
    layer = manager.layers["layers.0.mlp.experts"]
    layer.latest_hidden_size = 2
    layer.latest_bytes_per_element = 2

    planner = manager._planner_for_layer(
        layer,
        communication_scale=1.0,
        forward_compute_per_assignment=1.0,
    )

    assert isinstance(planner, expert_swap_module.CoReMoEPlanner)
    assert not planner.verify_collective_digest
    assert not hasattr(manager, "_verify_plan_consistency")
    assert not hasattr(manager, "_layer_has_accumulated_grad")


def test_hiermoe_env_int_allows_negative_minimum(monkeypatch):
    monkeypatch.setenv("VEOMNI_TEST_ENV_INT", "-1")
    assert expert_swap_module._env_int("VEOMNI_TEST_ENV_INT", 1, minimum=-1) == -1
    monkeypatch.setenv("VEOMNI_TEST_ENV_INT", "0")
    assert expert_swap_module._env_int("VEOMNI_TEST_ENV_INT", 1, minimum=-1) == 0
    monkeypatch.setenv("VEOMNI_TEST_ENV_INT", "3")
    assert expert_swap_module._env_int("VEOMNI_TEST_ENV_INT", 1, minimum=-1) == 3


@pytest.mark.parametrize("route_mode", ["step", "layer"])
def test_trainer_finishes_hiermoe_placement_lifecycle_for_both_route_modes(monkeypatch, route_mode):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            hiermoe=SimpleNamespace(
                enable=True,
                expert_swap=True,
                expert_swap_mode=route_mode,
            )
        )
    )
    trainer.state = SimpleNamespace(global_step=3)
    trainer._full_profile_range = lambda *_args, **_kwargs: nullcontext()
    calls = []
    monkeypatch.setattr(trainer_base_module, "maybe_run_hiermoe_expert_swap", calls.append)

    trainer.run_hiermoe_expert_swap()
    trainer.run_hiermoe_expert_swap()

    assert calls == [2]


def test_hiermoe_trainable_forward_context_restores_nested_state(monkeypatch):
    trainer = object.__new__(BaseTrainer)
    state = SimpleNamespace(layer_swap_forward_enabled=False)
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)

    with trainer.hiermoe_layer_swap_forward():
        assert state.layer_swap_forward_enabled is True
        with trainer.hiermoe_layer_swap_forward():
            assert state.layer_swap_forward_enabled is True
        assert state.layer_swap_forward_enabled is True

    assert state.layer_swap_forward_enabled is False


@pytest.mark.parametrize(("active", "mode"), ((False, "layer"), (True, "step")))
def test_ineligible_hiermoe_checkpoint_preserves_noop_context(monkeypatch, active, mode):
    state = SimpleNamespace(
        active=active,
        expert_swap=True,
        expert_swap_mode=mode,
        activation_checkpointing_enabled=True,
        placement_mapping_enabled=True,
        expert_swap_manager=object(),
    )
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)

    resolved = torch_parallelize_module._resolve_checkpoint_context_fn(noop_context_fn)

    assert resolved is noop_context_fn


def test_hiermoe_checkpoint_composes_external_context_inside_replay(monkeypatch):
    state = SimpleNamespace(
        active=True,
        expert_swap=True,
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
        placement_mapping_enabled=True,
        expert_swap_manager=object(),
        checkpoint_recompute_enabled=False,
        checkpoint_route_replay=None,
    )
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)
    events = []
    factory_calls = 0

    def external_context_fn():
        nonlocal factory_calls
        factory_calls += 1

        @contextmanager
        def external_forward():
            events.append(("forward_enter", state.checkpoint_route_replay is not None))
            yield
            events.append(("forward_exit", state.checkpoint_route_replay is not None))

        @contextmanager
        def external_recompute():
            events.append(
                (
                    "recompute_enter",
                    state.checkpoint_route_replay is not None and state.checkpoint_recompute_enabled,
                )
            )
            yield
            events.append(
                (
                    "recompute_exit",
                    state.checkpoint_route_replay is not None and state.checkpoint_recompute_enabled,
                )
            )

        return external_forward(), external_recompute()

    resolved = torch_parallelize_module._resolve_checkpoint_context_fn(external_context_fn)
    forward_context, recompute_context = resolved()
    with forward_context:
        events.append(("forward_body", state.checkpoint_route_replay is not None))
    with recompute_context:
        events.append(
            (
                "recompute_body",
                state.checkpoint_route_replay is not None and state.checkpoint_recompute_enabled,
            )
        )

    assert factory_calls == 1
    assert events == [
        ("forward_enter", True),
        ("forward_body", True),
        ("forward_exit", True),
        ("recompute_enter", True),
        ("recompute_body", True),
        ("recompute_exit", True),
    ]
    assert state.checkpoint_route_replay is None
    assert state.checkpoint_recompute_enabled is False


class _ForbiddenRecomputePlacementCallbacks:
    def __init__(self):
        self.pending_timing = ["original-forward"]

    def placement_planning_enabled(self):
        return True

    def __getattr__(self, name):
        if name in {
            "maybe_swap_layer_on_routing",
            "placement_timing_event",
            "record_dispatch_statistics",
            "record_layer_timing",
        }:
            raise AssertionError(f"recomputation invoked placement callback {name}")
        raise AttributeError(name)


def _disabled_recompute_state(manager):
    return SimpleNamespace(
        current_step=1,
        debug_validate=False,
        expert_swap_manager=manager,
        expert_swap_mode="layer",
        layer_swap_forward_enabled=False,
        log_interval=1,
        placement_mapping_enabled=True,
    )


def test_group_gemm_recompute_skips_placement_timing_callbacks(monkeypatch):
    pytest.importorskip("triton")
    group_gemm = importlib.import_module("veomni.ops.kernels.moe.group_gemm")
    manager = _ForbiddenRecomputePlacementCallbacks()
    state = _disabled_recompute_state(manager)
    parallel_state = SimpleNamespace(ep_enabled=True, ep_group=None)
    dispatch_context = SimpleNamespace()

    monkeypatch.setattr(group_gemm, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(group_gemm, "hiermoe_active", lambda: True)
    monkeypatch.setattr(group_gemm, "get_hiermoe_state", lambda: state)
    monkeypatch.setattr(group_gemm, "_start_moe_timing", lambda *_args: None)
    monkeypatch.setattr(group_gemm, "record_moe_validation_routing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(group_gemm, "record_moe_timing_span", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(group_gemm, "_finish_moe_timing", lambda *_args: None)
    monkeypatch.setattr(
        group_gemm,
        "rank_dedup_dispatch",
        lambda **kwargs: (kwargs["hidden_states"], dispatch_context, torch.tensor([2])),
    )
    monkeypatch.setattr(group_gemm, "rank_dedup_combine", lambda values, _ctx: values)
    monkeypatch.setattr(group_gemm, "EPGroupGemm", SimpleNamespace(apply=lambda values, *_args: values))

    hidden = torch.ones((2, 1))
    output = group_gemm.group_gemm_fused_moe_forward(
        num_experts=2,
        routing_weights=torch.ones((2, 1)),
        selected_experts=torch.zeros((2, 1), dtype=torch.long),
        hidden_states=hidden,
        fc1_1_weight=torch.ones((1, 1, 1)),
        fc1_2_weight=torch.ones((1, 1, 1)),
        fc2_weight=torch.ones((1, 1, 1)),
        layer_key="layers.0.mlp.experts",
    )

    torch.testing.assert_close(output, hidden)
    assert manager.pending_timing == ["original-forward"]


def test_npu_group_gemm_recompute_skips_placement_timing_callbacks(monkeypatch):
    pytest.importorskip("torch_npu")
    npu_group_gemm = importlib.import_module("veomni.ops.kernels.moe.npu_group_gemm")
    manager = _ForbiddenRecomputePlacementCallbacks()
    state = _disabled_recompute_state(manager)
    dispatch_context = SimpleNamespace()

    monkeypatch.setattr(npu_group_gemm, "hiermoe_active", lambda: True)
    monkeypatch.setattr(npu_group_gemm, "get_hiermoe_state", lambda: state)
    monkeypatch.setattr(npu_group_gemm, "_start_npu_moe_timing", lambda *_args: None)
    monkeypatch.setattr(npu_group_gemm, "record_moe_validation_routing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(npu_group_gemm, "record_moe_timing_span", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        npu_group_gemm,
        "rank_dedup_dispatch",
        lambda **kwargs: (kwargs["hidden_states"], dispatch_context, torch.tensor([2])),
    )
    monkeypatch.setattr(npu_group_gemm, "rank_dedup_combine", lambda values, _ctx: values)
    monkeypatch.setattr(npu_group_gemm, "npu_group_gemm", lambda values, *_args: values)
    monkeypatch.setattr(npu_group_gemm.torch_npu, "npu_swiglu", lambda values, dim: values)

    hidden = torch.ones((2, 1))
    output = npu_group_gemm.npu_ep_fused_moe_forward(
        num_experts=2,
        routing_weights=torch.ones((2, 1)),
        selected_experts=torch.zeros((2, 1), dtype=torch.long),
        hidden_states=hidden,
        fc1_1_weight=torch.ones((1, 1, 1)),
        fc1_2_weight=torch.ones((1, 1, 1)),
        fc2_weight=torch.ones((1, 1, 1)),
        layer_key="layers.0.mlp.experts",
    )

    torch.testing.assert_close(output, hidden)
    assert manager.pending_timing == ["original-forward"]


def _eager_linear_moe(hidden_states, selected_experts, routing_weights, expert_weight):
    output = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        for slot_idx in range(selected_experts.shape[1]):
            expert_idx = int(selected_experts[token_idx, slot_idx].item())
            y = hidden_states[token_idx] @ expert_weight[expert_idx]
            output[token_idx] = output[token_idx] + y * routing_weights[token_idx, slot_idx]
    return output


def _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, local_weight):
    chunks = []
    offset = 0
    for local_expert, count in enumerate(tokens_per_local_expert.tolist()):
        count = int(count)
        chunk = permuted_tokens[offset : offset + count]
        chunks.append(chunk @ local_weight[local_expert])
        offset += count
    return torch.cat(chunks, dim=0) if chunks else torch.empty_like(permuted_tokens)


def _rank_dedup_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
    )
    torch.manual_seed(1234 + rank)
    num_experts = 8
    hidden_size = 3
    routing_patterns = [
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        [[1, 0], [3, 2], [5, 4], [7, 6]],
        [[0, 2], [1, 3], [4, 6], [5, 7]],
        [[2, 0], [3, 1], [6, 4], [7, 5]],
    ]
    selected_experts = torch.tensor(routing_patterns[rank], dtype=torch.long)

    hidden = torch.randn(4, hidden_size, dtype=torch.double, requires_grad=True)
    routing_logits = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
    routing_weights = torch.softmax(routing_logits, dim=-1)
    full_weight_seed = torch.Generator().manual_seed(17)
    full_weight = torch.randn(num_experts, hidden_size, hidden_size, dtype=torch.double, generator=full_weight_seed)

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    baseline_logits = routing_logits.detach().clone().requires_grad_(True)
    baseline_weights = torch.softmax(baseline_logits, dim=-1)
    baseline_expert_weight = full_weight.detach().clone().requires_grad_(True)
    baseline_output = _eager_linear_moe(
        baseline_hidden,
        selected_experts,
        baseline_weights,
        baseline_expert_weight,
    )
    baseline_loss = baseline_output.square().sum()
    baseline_loss.backward()
    dist.all_reduce(baseline_expert_weight.grad)

    local_experts = num_experts // world_size
    local_start = rank * local_experts
    local_weight = full_weight[local_start : local_start + local_experts].detach().clone().requires_grad_(True)
    permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
    )
    expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, local_weight)
    output = rank_dedup_combine(expert_outputs, ctx)
    loss = output.square().sum()
    loss.backward()

    assert output.shape == baseline_output.shape
    assert output.dtype == baseline_output.dtype
    assert output.device == baseline_output.device
    torch.testing.assert_close(output, baseline_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(routing_logits.grad, baseline_logits.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        local_weight.grad,
        baseline_expert_weight.grad[local_start : local_start + local_experts],
        atol=1e-5,
        rtol=1e-5,
    )
    assert ctx.dedup_ratio_dispatch > 0.0
    assert ctx.dedup_ratio_combine > 0.0


def test_rank_dedup_dispatch_combine_forward_backward():
    torchrun(_rank_dedup_worker, world_size=4, backend="gloo")


def _rank_dedup_3d_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            hierarchy_group_sizes=[2, 4, world_size],
        ),
        dist.group.WORLD,
    )
    state = get_hiermoe_state()
    assert state is not None
    state.perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1_000.0, beta=1_000.0),
        inter=(LinkCost(alpha=0.0, beta=0.0), LinkCost(alpha=0.0, beta=0.0)),
        intra=LinkCost(alpha=0.0, beta=1.0),
        source="test:force-3d",
    )
    torch.manual_seed(2345 + rank)
    num_experts = 16
    hidden_size = 4
    base_routes = torch.tensor(
        [
            [0, 2, 8, 10],
            [1, 3, 9, 11],
            [4, 6, 12, 14],
            [5, 7, 13, 15],
            [0, 5, 10, 15],
        ],
        dtype=torch.long,
    )
    selected_experts = (base_routes + rank) % num_experts

    hidden = torch.randn(5, hidden_size, dtype=torch.double, requires_grad=True)
    routing_logits = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    routing_weights = torch.softmax(routing_logits, dim=-1)
    full_weight_seed = torch.Generator().manual_seed(23)
    full_weight = torch.randn(num_experts, hidden_size, hidden_size, dtype=torch.double, generator=full_weight_seed)

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    baseline_logits = routing_logits.detach().clone().requires_grad_(True)
    baseline_weights = torch.softmax(baseline_logits, dim=-1)
    baseline_expert_weight = full_weight.detach().clone().requires_grad_(True)
    baseline_output = _eager_linear_moe(
        baseline_hidden,
        selected_experts,
        baseline_weights,
        baseline_expert_weight,
    )
    baseline_loss = baseline_output.square().sum()
    baseline_loss.backward()
    dist.all_reduce(baseline_expert_weight.grad)

    local_experts = num_experts // world_size
    local_start = rank * local_experts
    local_weight = full_weight[local_start : local_start + local_experts].detach().clone().requires_grad_(True)
    permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
    )
    expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, local_weight)
    output = rank_dedup_combine(expert_outputs, ctx)
    loss = output.square().sum()
    loss.backward()

    assert ctx.mode == "hierarchical3d"
    assert output.shape == baseline_output.shape
    assert output.dtype == baseline_output.dtype
    assert output.device == baseline_output.device
    torch.testing.assert_close(output, baseline_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(routing_logits.grad, baseline_logits.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        local_weight.grad,
        baseline_expert_weight.grad[local_start : local_start + local_experts],
        atol=1e-5,
        rtol=1e-5,
    )
    assert ctx.dedup_ratio_dispatch > 0.0
    assert ctx.dedup_ratio_combine > 0.0


def test_rank_dedup_3d_dispatch_combine_forward_backward():
    torchrun(_rank_dedup_3d_worker, world_size=8, backend="gloo")


def test_duplicate_free_counts_keeps_or_semantics():
    selected_experts = torch.tensor(
        [
            [0, 1, 2],
            [0, 0, 3],
            [4, 5, 7],
        ],
        dtype=torch.long,
    )
    counts = duplicate_free_counts_by_expert_group(selected_experts, num_experts=8, group_size=2)
    torch.testing.assert_close(counts, torch.tensor([2, 2, 1, 1], dtype=torch.int64))


def test_repeat_ranks_and_combine_key_stride_keep_rank_token_keys_distinct():
    ranks = hiermoe_all_to_all._repeat_ranks([2, 0, 3], torch.device("cpu"))
    torch.testing.assert_close(ranks, torch.tensor([0, 0, 2, 2, 2], dtype=torch.long))

    weighted_outputs = torch.tensor([[1.0], [2.0]])
    source_token_indices = torch.tensor([3, 0], dtype=torch.long)
    combined, splits = hiermoe_all_to_all._aggregate_weighted_outputs(
        weighted_outputs,
        source_token_indices,
        [1, 1],
        key_stride=4,
    )
    assert splits == [1, 1]
    torch.testing.assert_close(combined, weighted_outputs)


def test_local_expert_sort_indices_is_a_permutation_and_tracks_counts():
    local_expert_ids = torch.tensor([3, 1, 0, 3, 2, 1, 2, 0, 3], dtype=torch.long)
    sort_indices, unsort_indices, tokens_per_local_expert = hiermoe_all_to_all._local_expert_sort_indices(
        local_expert_ids,
        num_local_experts=4,
        device=torch.device("cpu"),
    )

    sorted_ids = local_expert_ids.index_select(0, sort_indices)
    torch.testing.assert_close(sorted_ids, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=torch.long))
    torch.testing.assert_close(tokens_per_local_expert, torch.tensor([2, 2, 2, 3], dtype=torch.long))
    torch.testing.assert_close(
        sorted_ids.index_select(0, unsort_indices),
        local_expert_ids,
    )


@pytest.mark.parametrize(
    ("communication_mode", "expected_dim"),
    (("direct", 1), ("hierarchical", 3), ("auto", 2)),
)
def test_hiermoe_communication_mode_selects_dimension(monkeypatch, communication_mode, expected_dim):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    hierarchy = Hierarchy(ep_size=64, group_sizes=(8, 16, 64), source="test")
    monkeypatch.setattr(
        hiermoe_state_module,
        "_STATE",
        HierMoEState(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=1,
            redundant_slot_increment_per_device=0,
            max_slot_op_search_rounds=1,
            max_replica_rounds=0,
            expert_swap_mode="step",
            debug_validate=False,
            current_step=0,
            log_interval=1,
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            active=True,
            communication_mode=communication_mode,
        ),
    )

    selected_experts = torch.tensor([[0, 63], [8, 16], [32, 40]], dtype=torch.long)
    selected_dim = hiermoe_all_to_all._select_dimension(
        selected_experts=selected_experts,
        num_experts=64,
        hidden_size=8,
        bytes_per_element=2,
        group=None,
    )
    assert selected_dim == expected_dim


def test_vectorized_expert_swap_cost_matches_scalar_estimator():
    selected_experts = torch.tensor(
        [
            [0, 3],
            [1, 4],
            [2, 5],
            [6, 7],
            [0, 7],
        ],
        dtype=torch.long,
    )
    placement = torch.tensor([2, 1, 0, 3, 5, 4, 7, 6], dtype=torch.long)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=10.0),
        inter=(LinkCost(alpha=0.1, beta=0.1),),
        intra=LinkCost(alpha=0.1, beta=0.1),
        source="test",
    )
    candidate_pairs = [(0, 6), (2, 4), (3, 7), (1, 5)]

    pair, cost = expert_swap_module.estimate_best_swap_pair(
        selected_experts=selected_experts,
        num_experts=8,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
        placement=placement,
        candidate_pairs=candidate_pairs,
    )

    current_cost = expert_swap_module._estimate_mapping_cost(
        selected_experts,
        placement,
        8,
        8,
        2,
        hierarchy,
        perf_model,
        10.0,
    )
    expected_pair = None
    expected_cost = current_cost
    for lhs, rhs in candidate_pairs:
        swapped = placement.clone()
        swapped[lhs], swapped[rhs] = swapped[rhs].clone(), swapped[lhs].clone()
        candidate_cost = expert_swap_module._estimate_mapping_cost(
            selected_experts,
            swapped,
            8,
            8,
            2,
            hierarchy,
            perf_model,
            10.0,
        )
        if candidate_cost < expected_cost:
            expected_pair = (lhs, rhs)
            expected_cost = candidate_cost

    assert pair == expected_pair
    assert cost == pytest.approx(expected_cost)


def test_fast_2d_expert_swap_cost_matches_generic_estimator():
    selected_experts = torch.tensor(
        [
            [0, 3],
            [1, 4],
            [2, 5],
            [6, 7],
            [0, 7],
        ],
        dtype=torch.long,
    )
    placement = torch.tensor([2, 1, 0, 3, 5, 4, 7, 6], dtype=torch.long)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=10.0),
        inter=(LinkCost(alpha=0.1, beta=0.1),),
        intra=LinkCost(alpha=0.1, beta=0.1),
        source="test",
    )
    candidate_pairs = torch.tensor([(0, 6), (2, 4), (3, 7), (1, 5)], dtype=torch.long)

    result = expert_swap_module._estimate_swap_pair_costs_fast_2d(
        selected_experts=selected_experts,
        num_experts=8,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
        logical_to_physical=placement,
        candidate_pairs=candidate_pairs,
    )
    assert result is not None
    current_cost, pairs, costs = result

    expected_current = expert_swap_module._estimate_mapping_cost(
        selected_experts,
        placement,
        8,
        8,
        2,
        hierarchy,
        perf_model,
        10.0,
    )
    assert float(current_cost.item()) == pytest.approx(expected_current)
    torch.testing.assert_close(pairs.cpu(), candidate_pairs)

    expected_costs = []
    for lhs, rhs in candidate_pairs.tolist():
        swapped = placement.clone()
        swapped[lhs], swapped[rhs] = swapped[rhs].clone(), swapped[lhs].clone()
        expected_costs.append(
            expert_swap_module._estimate_mapping_cost(
                selected_experts,
                swapped,
                8,
                8,
                2,
                hierarchy,
                perf_model,
                10.0,
            )
        )
    torch.testing.assert_close(costs.cpu(), torch.tensor(expected_costs, dtype=torch.float32))


def test_fast_2d_expert_swap_cost_uses_rank_max_for_intra_stage():
    selected_experts = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1], [4], [4], [4], [4]])
    placement = torch.arange(8, dtype=torch.long)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=0.0, beta=0.0),
        inter=(LinkCost(alpha=0.0, beta=0.0),),
        intra=LinkCost(alpha=0.0, beta=1.0),
        source="test",
    )
    candidate_pairs = torch.tensor([(1, 2)], dtype=torch.long)

    result = expert_swap_module._estimate_swap_pair_costs_fast_2d(
        selected_experts=selected_experts,
        num_experts=8,
        hidden_size=1,
        bytes_per_element=1,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
        logical_to_physical=placement,
        candidate_pairs=candidate_pairs,
    )
    assert result is not None
    current_cost, _pairs, costs = result

    assert float(current_cost.item()) == pytest.approx(16.0)
    assert float(costs[0].item()) == pytest.approx(8.0)


def test_global_2d_expert_swap_cost_uses_rank_max_for_intra_stage():
    selected_experts = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1], [4], [4], [4], [4]])
    placement = torch.arange(8, dtype=torch.long)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=0.0, beta=0.0),
        inter=(LinkCost(alpha=0.0, beta=0.0),),
        intra=LinkCost(alpha=0.0, beta=1.0),
        source="test",
    )
    candidate_pairs = torch.tensor([(1, 2)], dtype=torch.long)
    rank_by_logical = torch.div(placement, 2, rounding_mode="floor")
    group_by_logical = torch.div(placement, 4, rounding_mode="floor")
    expert_counts, base_group_counts, expert_group_counts = expert_swap_module._selector_stats_2d(
        selected_experts,
        num_experts=8,
        group_by_logical=group_by_logical,
        num_groups=2,
    )
    base_rank_counts, expert_rank_counts = expert_swap_module._selector_group_stats(
        selected_experts,
        num_experts=8,
        group_by_logical=rank_by_logical,
        num_groups=4,
    )
    pair_counts = expert_swap_module._candidate_pair_token_counts(selected_experts, candidate_pairs, num_experts=8)

    current_cost, costs = expert_swap_module._costs_from_global_2d_stats(
        expert_counts=expert_counts,
        base_group_counts=base_group_counts,
        expert_group_counts=expert_group_counts,
        base_rank_counts=base_rank_counts,
        expert_rank_counts=expert_rank_counts,
        pair_counts=pair_counts,
        pairs=candidate_pairs,
        num_experts=8,
        hidden_size=1,
        bytes_per_element=1,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
        logical_to_physical=placement,
    )

    assert float(current_cost.item()) == pytest.approx(16.0)
    assert float(costs[0].item()) == pytest.approx(8.0)


def test_fast_hierarchy_expert_swap_cost_matches_generic_estimator():
    selected_experts = torch.tensor(
        [
            [12, 15],
            [5, 0],
            [3, 11],
            [3, 7],
            [9, 3],
            [5, 2],
            [4, 7],
            [6, 8],
            [8, 12],
            [10, 1],
            [6, 7],
            [7, 14],
        ],
        dtype=torch.long,
    )
    placement = torch.arange(16, dtype=torch.long)
    hierarchy = Hierarchy(ep_size=8, group_sizes=(2, 4, 8), source="test-3d")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=10.0),
        inter=(LinkCost(alpha=0.1, beta=0.1), LinkCost(alpha=0.2, beta=0.05)),
        intra=LinkCost(alpha=0.1, beta=0.1),
        source="test",
    )
    candidate_pairs = torch.tensor([(0, 15), (4, 6), (4, 12), (1, 14), (2, 8), (5, 10)], dtype=torch.long)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=16).to(torch.float32)
    level_base_group_counts = []
    level_expert_group_counts = []
    for u_i, num_groups in expert_swap_module._hierarchy_level_group_shapes(hierarchy, 16):
        expert_group_size = max(1, 16 // max(1, hierarchy.ep_size // u_i))
        group_by_logical = torch.div(placement, expert_group_size, rounding_mode="floor")
        base_counts, expert_group_counts = expert_swap_module._selector_group_stats(
            selected_experts,
            16,
            group_by_logical,
            num_groups,
        )
        level_base_group_counts.append(base_counts)
        level_expert_group_counts.append(expert_group_counts)
    rank_by_logical = torch.div(placement, 2, rounding_mode="floor")
    base_rank_counts, expert_rank_counts = expert_swap_module._selector_group_stats(
        selected_experts,
        16,
        rank_by_logical,
        hierarchy.ep_size,
    )
    pair_counts = expert_swap_module._candidate_pair_token_counts(selected_experts, candidate_pairs, 16)

    current_cost, costs = expert_swap_module._costs_from_global_hierarchy_stats(
        expert_counts=expert_counts,
        base_rank_counts=base_rank_counts,
        expert_rank_counts=expert_rank_counts,
        level_base_group_counts=level_base_group_counts,
        level_expert_group_counts=level_expert_group_counts,
        pair_counts=pair_counts,
        pairs=candidate_pairs,
        num_experts=16,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
        logical_to_physical=placement,
    )

    expected_current = expert_swap_module._estimate_mapping_cost(
        selected_experts,
        placement,
        16,
        8,
        2,
        hierarchy,
        perf_model,
        10.0,
    )
    assert float(current_cost.item()) == pytest.approx(expected_current)
    expected_costs = []
    for lhs, rhs in candidate_pairs.tolist():
        swapped = placement.clone()
        swapped[lhs], swapped[rhs] = swapped[rhs].clone(), swapped[lhs].clone()
        expected_costs.append(
            expert_swap_module._estimate_mapping_cost(
                selected_experts,
                swapped,
                16,
                8,
                2,
                hierarchy,
                perf_model,
                10.0,
            )
        )
    torch.testing.assert_close(costs.cpu(), torch.tensor(expected_costs, dtype=torch.float32))
    assert costs[1] < current_cost


def test_expert_swap_candidates_cover_all_128_expert_pairs():
    selected_experts = torch.tensor([[0, 127]], dtype=torch.long)

    pairs = expert_swap_module._candidate_pairs(selected_experts, num_experts=128)

    expected = torch.triu_indices(128, 128, offset=1).t().contiguous()
    assert pairs.shape == (8128, 2)
    torch.testing.assert_close(pairs.cpu(), expected)


def test_cross_rank_candidate_pairs_follow_current_placement():
    pairs = expert_swap_module._all_candidate_pairs(8, torch.device("cpu"))
    identity = torch.arange(8, dtype=torch.long)

    filtered = expert_swap_module._cross_rank_candidate_pairs(pairs, identity, num_local_experts=2)

    assert filtered.shape == (24, 2)
    assert [0, 1] not in filtered.tolist()
    placement = identity.clone()
    placement[0], placement[2] = placement[2].clone(), placement[0].clone()

    updated = expert_swap_module._cross_rank_candidate_pairs(pairs, placement, num_local_experts=2)

    assert updated.shape == (24, 2)
    assert [0, 1] in updated.tolist()
    assert [0, 3] not in updated.tolist()


def test_candidate_pair_token_counts_match_naive_topk_hits():
    selected_experts = torch.tensor(
        [
            [0, 1, 3, 5],
            [1, 2, 3, 7],
            [0, 4, 6, 15],
            [2, 3, 4, 12],
            [0, 3, 5, 11],
            [1, 1, 3, 3],
        ],
        dtype=torch.long,
    )
    pairs = torch.triu_indices(16, 16, offset=1).t().contiguous()[:40]
    pairs = torch.cat((pairs, torch.tensor([[3, 1]], dtype=torch.long)), dim=0)
    counts = expert_swap_module._candidate_pair_token_counts(selected_experts, pairs, num_experts=16)
    expected = []
    for lhs, rhs in pairs.tolist():
        expected.append(sum(lhs in row.tolist() and rhs in row.tolist() for row in selected_experts))
    torch.testing.assert_close(counts.cpu(), torch.tensor(expected, dtype=torch.float32))


def _rank_dedup_call_count_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
    )

    num_experts = 8
    hidden_size = 3
    routing_patterns = [
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        [[1, 0], [3, 2], [5, 4], [7, 6]],
        [[0, 2], [1, 3], [4, 6], [5, 7]],
        [[2, 0], [3, 1], [6, 4], [7, 5]],
    ]
    selected_experts = torch.tensor(routing_patterns[rank], dtype=torch.long)
    hidden = torch.randn(4, hidden_size, dtype=torch.double)
    routing_weights = torch.full((4, 2), 0.5, dtype=torch.double)
    local_experts = num_experts // world_size
    local_weight = torch.randn(local_experts, hidden_size, hidden_size, dtype=torch.double)

    original_all_to_all_single = dist.all_to_all_single
    all_to_all_call_count = 0

    def _count_all_to_all_single(*args, **kwargs):
        nonlocal all_to_all_call_count
        all_to_all_call_count += 1
        return original_all_to_all_single(*args, **kwargs)

    dist.all_to_all_single = _count_all_to_all_single
    try:
        permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
            hidden,
            selected_experts,
            routing_weights,
            num_experts,
            dist.group.WORLD,
        )
        expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, local_weight)
        rank_dedup_combine(expert_outputs, ctx)
    finally:
        dist.all_to_all_single = original_all_to_all_single

    assert all_to_all_call_count <= 8


def test_rank_dedup_packs_metadata_weights_to_reduce_collectives():
    torchrun(_rank_dedup_call_count_worker, world_size=4, backend="gloo")


def test_dimension_selection_prefers_expected_dimension():
    selected_experts = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.long)
    hierarchy = Hierarchy(ep_size=4, group_sizes=(2, 4), source="test")
    fast_hierarchy = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=10.0),
        inter=(LinkCost(alpha=0.1, beta=0.1),),
        intra=LinkCost(alpha=0.1, beta=0.1),
        source="test-fast-hierarchy",
    )
    assert fast_hierarchy.select_dimension(selected_experts, 4, 8, 2, hierarchy) == 2

    fast_a2a = HierMoEPerfModel(
        a2a=LinkCost(alpha=0.1, beta=0.01),
        inter=(LinkCost(alpha=100.0, beta=100.0),),
        intra=LinkCost(alpha=100.0, beta=100.0),
        source="test-fast-a2a",
    )
    assert fast_a2a.select_dimension(selected_experts, 4, 8, 2, hierarchy) == 1


class _FakeExperts(torch.nn.Module):
    def __init__(self, num_experts, local_start, local_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = torch.nn.Parameter(torch.empty(local_experts, hidden_size, hidden_size))
        self.down_proj = torch.nn.Parameter(torch.empty(local_experts, hidden_size, hidden_size))
        self.reset_values(local_start)

    @torch.no_grad()
    def reset_values(self, local_start):
        for local_idx in range(self.gate_up_proj.shape[0]):
            expert_id = local_start + local_idx
            self.gate_up_proj[local_idx].fill_(float(expert_id + 1))
            self.down_proj[local_idx].fill_(float(expert_id + 101))


class _FakeAsymmetricExperts(torch.nn.Module):
    def __init__(self, num_experts, local_start, local_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = torch.nn.Parameter(torch.empty(local_experts, hidden_size + 1, hidden_size))
        self.down_proj = torch.nn.Parameter(torch.empty(local_experts, hidden_size, hidden_size + 1))
        self.reset_values(local_start)

    @torch.no_grad()
    def reset_values(self, local_start):
        for local_idx in range(self.gate_up_proj.shape[0]):
            expert_id = local_start + local_idx
            self.gate_up_proj[local_idx].fill_(float(expert_id + 1))
            self.down_proj[local_idx].fill_(float(expert_id + 101))


def test_redundant_slot_expansion_preserves_existing_experts_and_zeros_tail():
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=2, hidden_size=2)

    expanded = expert_swap_module.expand_redundant_expert_slots(
        module,
        ep_size=2,
        redundant_slot_increment_per_device=1,
    )

    assert expanded == 1
    assert tuple(module.gate_up_proj.shape) == (3, 2, 2)
    assert tuple(module.down_proj.shape) == (3, 2, 2)
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor([1.0, 2.0, 0.0]))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor([101.0, 102.0, 0.0]))


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    (("step", expert_swap_module.CurrentRoutePlanner), ("layer", expert_swap_module.CoReMoEPlanner)),
)
def test_expert_swap_planner_selection_preserves_step_compatibility(mode, expected_type):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode=mode,
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.latest_hidden_size = 2
    layer.latest_bytes_per_element = 2

    planner = manager._planner_for_layer(
        layer,
        communication_scale=1.0,
        forward_compute_per_assignment=1.0,
    )

    assert type(planner) is expected_type


def test_redundant_slot_identity_layout_uses_compact_dispatch_until_copy_exists():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]

    assert layer.slot_to_logical is not None
    torch.testing.assert_close(layer.slot_to_logical, torch.tensor([0, 1, -1, 2, 3, -1]))
    selected = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    mapped = manager.map_logical_to_physical("layers.0.mlp.experts", selected)

    assert manager.num_physical_slots("layers.0.mlp.experts", 4) == 4
    torch.testing.assert_close(mapped, selected)

    manager._commit_layer_slot_ops("layers.0.mlp.experts", [expert_swap_module._SlotOpCandidate("cover", 4, 2)])

    assert manager.num_physical_slots("layers.0.mlp.experts", 4) == 6
    mapped_with_copy = manager.map_logical_to_physical("layers.0.mlp.experts", selected)
    torch.testing.assert_close(mapped_with_copy, torch.tensor([[0, 3], [1, 2]], dtype=torch.long))


def test_planner_physical_routes_are_consumed_exactly_once():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    selected = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    planned = torch.tensor([[0, 2], [4, 0]], dtype=torch.long)
    layer.pending_physical_routes = planned
    layer.pending_route_data_ptr = selected.data_ptr()

    first = manager.map_logical_to_physical(key, selected)
    second = manager.map_logical_to_physical(key, selected)

    assert first is planned
    assert second is not planned
    assert layer.pending_physical_routes is None
    assert layer.pending_route_data_ptr == 0


def test_checkpoint_replay_is_invocation_local_and_preserves_occurrence_order(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    first_selected = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    first_planned = torch.tensor([[0, 2], [4, 0]], dtype=torch.long)
    second_selected = torch.tensor([[3, 0], [0, 3]], dtype=torch.long)
    layer.pending_physical_routes = first_planned
    layer.pending_route_data_ptr = first_selected.data_ptr()

    state = SimpleNamespace(
        active=True,
        expert_swap=True,
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
        placement_mapping_enabled=True,
        expert_swap_manager=manager,
        checkpoint_recompute_enabled=False,
        checkpoint_route_replay=None,
    )
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)
    forward_context, recompute_context = hiermoe_state_module.hiermoe_checkpoint_context_fn()
    with forward_context:
        first_forward = manager.map_logical_to_physical(
            key,
            first_selected,
            checkpoint_recompute=state.checkpoint_recompute_enabled,
            checkpoint_replay=state.checkpoint_route_replay,
        )
        second_forward = manager.map_logical_to_physical(
            key,
            second_selected,
            checkpoint_recompute=state.checkpoint_recompute_enabled,
            checkpoint_replay=state.checkpoint_route_replay,
        )
    assert state.checkpoint_route_replay is None
    assert state.checkpoint_recompute_enabled is False

    with recompute_context:
        first_recompute = manager.map_logical_to_physical(
            key,
            first_selected.clone(),
            checkpoint_recompute=state.checkpoint_recompute_enabled,
            checkpoint_replay=state.checkpoint_route_replay,
        )
        second_recompute = manager.map_logical_to_physical(
            key,
            second_selected.clone(),
            checkpoint_recompute=state.checkpoint_recompute_enabled,
            checkpoint_replay=state.checkpoint_route_replay,
        )

    assert first_forward is first_planned
    torch.testing.assert_close(first_recompute, first_forward)
    torch.testing.assert_close(second_recompute, second_forward)
    assert state.checkpoint_route_replay is None
    assert state.checkpoint_recompute_enabled is False


def test_checkpoint_replay_falls_back_to_current_owner_for_invalid_slots():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    selected = torch.tensor([[3, 0], [3, 0]], dtype=torch.long)
    replayed = torch.tensor([[2, 4], [9, 0]], dtype=torch.long)

    validated = manager._validate_checkpoint_replay(layer, selected, replayed)

    torch.testing.assert_close(validated, torch.tensor([[2, 0], [4, 0]], dtype=torch.long))


def test_checkpoint_replay_preserves_compact_identity_coordinates():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    selected = torch.tensor([[0, 3], [2, 1]], dtype=torch.long)
    assert manager._uses_compact_identity_dispatch(layer)
    dispatch_num_experts = manager.num_physical_slots(key, 4)
    assert dispatch_num_experts == 4
    padded_owners = layer.mapping_for_device(selected.device).index_select(0, selected.reshape(-1)).view_as(selected)
    torch.testing.assert_close(padded_owners, torch.tensor([[0, 4], [3, 1]], dtype=torch.long))
    assert int(padded_owners.max().item()) >= dispatch_num_experts
    layer.pending_physical_routes = padded_owners
    layer.pending_route_data_ptr = selected.data_ptr()
    replay = hiermoe_state_module._HierMoECheckpointReplay()

    forward = manager.map_logical_to_physical(key, selected, checkpoint_replay=replay)
    replay.reset_reader()
    recomputed = manager.map_logical_to_physical(
        key,
        selected.clone(),
        checkpoint_recompute=True,
        checkpoint_replay=replay,
    )

    torch.testing.assert_close(forward, selected)
    torch.testing.assert_close(replay.planned_routes[key][0], selected)
    torch.testing.assert_close(recomputed, selected)
    assert int(forward.max().item()) < dispatch_num_experts
    assert int(recomputed.max().item()) < dispatch_num_experts


def test_non_reentrant_checkpoint_replays_the_forward_physical_route(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    selected = torch.tensor([[3, 0], [3, 0]], dtype=torch.long)
    fallback = manager._map_logical_to_slot(layer, selected)
    planned = torch.where(selected == 3, 6 - fallback, fallback)
    assert not torch.equal(planned, fallback)

    state = SimpleNamespace(
        active=True,
        expert_swap=True,
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
        placement_mapping_enabled=True,
        expert_swap_manager=manager,
        checkpoint_recompute_enabled=False,
        checkpoint_route_replay=None,
    )
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)
    observed_routes = []

    def checkpointed_fn(inputs):
        local_selected = selected.clone()
        if not state.checkpoint_recompute_enabled:
            layer.pending_physical_routes = planned
            layer.pending_route_data_ptr = local_selected.data_ptr()
        physical = manager.map_logical_to_physical(
            key,
            local_selected,
            checkpoint_recompute=state.checkpoint_recompute_enabled,
            checkpoint_replay=state.checkpoint_route_replay,
        )
        observed_routes.append(physical.detach().clone())
        return torch.sin(inputs * (physical.reshape(-1).to(inputs.dtype) + 1)).sum()

    inputs = torch.randn(4, dtype=torch.double, requires_grad=True)
    output = activation_checkpoint(
        checkpointed_fn,
        inputs,
        use_reentrant=False,
        context_fn=hiermoe_state_module.hiermoe_checkpoint_context_fn,
    )
    output.backward()

    assert len(observed_routes) == 2
    torch.testing.assert_close(observed_routes[0], planned)
    torch.testing.assert_close(observed_routes[1], planned)
    assert torch.isfinite(inputs.grad).all()


def test_reentrant_checkpoint_replays_the_forward_physical_route(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    selected = torch.tensor([[3, 0], [3, 0]], dtype=torch.long)
    fallback = manager._map_logical_to_slot(layer, selected)
    planned = torch.where(selected == 3, 6 - fallback, fallback)
    assert not torch.equal(planned, fallback)

    state = SimpleNamespace(
        active=True,
        expert_swap=True,
        expert_swap_mode="layer",
        activation_checkpointing_enabled=True,
        placement_mapping_enabled=True,
        expert_swap_manager=manager,
        checkpoint_recompute_enabled=False,
        checkpoint_route_replay=None,
    )
    monkeypatch.setattr(hiermoe_state_module, "_STATE", state)
    observed_routes = []

    class ReplayModule(torch.nn.Module):
        def forward(self, inputs):
            local_selected = selected.clone()
            if not state.checkpoint_recompute_enabled:
                layer.pending_physical_routes = planned
                layer.pending_route_data_ptr = local_selected.data_ptr()
            physical = manager.map_logical_to_physical(
                key,
                local_selected,
                checkpoint_recompute=state.checkpoint_recompute_enabled,
                checkpoint_replay=state.checkpoint_route_replay,
            )
            observed_routes.append(physical.detach().clone())
            return torch.sin(inputs * (physical.reshape(-1).to(inputs.dtype) + 1)).sum()

    inputs = torch.randn(4, dtype=torch.double, requires_grad=True)
    output = VeOmniCheckpointFunction.apply(ReplayModule(), True, inputs)
    output.backward()

    assert len(observed_routes) == 2
    torch.testing.assert_close(observed_routes[0], planned)
    torch.testing.assert_close(observed_routes[1], planned)
    assert torch.isfinite(inputs.grad).all()


def test_redundant_slot_mapping_prefers_duplicate_copy_that_reuses_token_groups():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]

    assert layer.slot_to_logical is not None
    torch.testing.assert_close(layer.slot_to_logical, torch.tensor([0, 1, -1, 2, 3, -1]))
    layer.slot_to_logical[2] = 3
    manager._refresh_layer_mapping_from_slots(layer)

    selected = torch.tensor([[3, 0], [0, 3]], dtype=torch.long)
    mapped = manager.map_logical_to_physical("layers.0.mlp.experts", selected)

    torch.testing.assert_close(mapped, torch.tensor([[2, 0], [0, 2]], dtype=torch.long))


def test_redundant_slot_mapping_uses_parallel_owner_rank_priority():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]

    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, 0], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer)

    mapped = manager.map_logical_to_physical("layers.0.mlp.experts", torch.tensor([[3, 0]], dtype=torch.long))

    torch.testing.assert_close(mapped, torch.tensor([[2, 5]], dtype=torch.long))


def test_redundant_copy_groups_cache_invalidates_with_layout():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.slot_to_logical is not None

    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    layer.invalidate_cache()
    groups = layer.redundant_copy_groups()
    assert groups == ((3, (2, 4)),)
    assert layer.redundant_copy_groups() is groups

    layer.slot_to_logical = torch.tensor([0, 1, 0, 2, 3, -1], dtype=torch.long)
    layer.invalidate_cache()
    assert layer.redundant_copy_groups() == ((0, (0, 2)),)


def test_placement_executor_uses_planner_recorded_replica_source(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=3,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=3, group_sizes=(3,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=6, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    # Expert 5 already has a lower-numbered redundant copy in slot 2, while
    # its canonical owner (and the source priced by the planner) is slot 7.
    layer.slot_to_logical = torch.tensor([0, 1, 5, 2, 3, -1, 4, 5, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer)

    recorded: list[tuple[int, int]] = []

    def capture_grouped(grouped_entries, _ep_rank, _ep_size, _ep_group, **_kwargs):
        for (src_rank, dst_rank), entries in grouped_entries.items():
            for entry in entries[:1]:
                recorded.append(
                    (
                        src_rank * layer.num_local_experts + entry.src_slot,
                        dst_rank * layer.num_local_experts + entry.dst_slot,
                    )
                )

    monkeypatch.setattr(expert_swap_module, "_cover_grouped_slot_entries_atomic", capture_grouped)
    action = PlacementAction("replica", 7, 5, 5, -1)
    plan = SimpleNamespace(
        quota_policy=(),
        actions=(action,),
        swaps=(),
        final_layout=(0, 1, 5, 2, 3, 5, 4, 5, -1),
        final_owner_slots=(0, 1, 3, 4, 6, 7),
    )

    manager._execute_placement_plan(layer, plan)

    assert recorded == [(7, 5)]
    assert tuple(layer.slot_to_logical.tolist()) == plan.final_layout
    assert tuple(layer.logical_to_physical.tolist()) == plan.final_owner_slots


def test_placement_executor_replays_swap_without_dropping_existing_replicas(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=2, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 2, 3))

    monkeypatch.setattr(expert_swap_module, "_cover_grouped_slot_entries_atomic", lambda *_args, **_kwargs: None)
    action = PlacementAction("swap", 0, 2, 0, 2)
    plan = SimpleNamespace(
        algorithm_version="hiermoe-v1",
        quota_policy=(),
        actions=(action,),
        swaps=(action,),
        final_layout=(2, 1, 0, 3, 0, 1, 2, 3),
        final_owner_slots=(2, 1, 0, 3),
    )

    manager._execute_placement_plan(layer, plan)

    assert tuple(layer.slot_to_logical.tolist()) == plan.final_layout
    assert tuple(layer.logical_to_physical.tolist()) == plan.final_owner_slots


def test_redundant_gradient_sync_uses_explicit_owner_not_lowest_copy():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))

    assert manager._owner_rank_for_copy_group(layer, 3, (2, 4)) == 1


def test_core_moe_v3_executor_requires_complete_owner_mapping():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    plan = SimpleNamespace(
        algorithm_version=expert_swap_module.CORE_MOE_ALGORITHM_VERSION,
        quota_policy=(),
        actions=(),
        swaps=(),
        final_layout=(0, 1, -1, 2, 3, -1),
        final_owner_slots=(),
    )

    with pytest.raises(RuntimeError, match="must provide exactly 4 owner slots"):
        manager._execute_placement_plan(manager.layers["layers.0.mlp.experts"], plan)


def test_placement_executor_normalizes_interleaved_swap_and_empty(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=2,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    swap = PlacementAction("swap", 0, 3, 0, 2)
    plan = SimpleNamespace(
        algorithm_version=expert_swap_module.CORE_MOE_ALGORITHM_VERSION,
        quota_policy=(),
        actions=(swap, PlacementAction("empty", 0, 0, 2, -1)),
        swaps=(swap,),
        final_layout=(-1, 1, 2, 0, 3, -1),
        final_owner_slots=(3, 1, 2, 4),
    )
    calls = 0

    def capture_transfer(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        expert_swap_module,
        "_cover_grouped_slot_entries_atomic",
        capture_transfer,
    )

    committed = manager._execute_placement_plan(layer, plan)

    assert calls == 1
    assert committed == ["layers.0.mlp.experts:swap(0<->2)", "layers.0.mlp.experts:empty(2@0)"]
    assert tuple(layer.slot_to_logical.tolist()) == plan.final_layout
    assert tuple(layer.logical_to_physical.tolist()) == plan.final_owner_slots


def test_placement_executor_replays_cover_before_swap_and_promotes_owner(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=3,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=3, group_sizes=(3,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=6, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, -1, 2, 3, 0, 4, 5, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4, 6, 7))
    cover_owner = PlacementAction("replica", 6, 0, 4, 0)
    swap = PlacementAction("swap", 3, 7, 2, 5)
    plan = SimpleNamespace(
        algorithm_version=expert_swap_module.CORE_MOE_ALGORITHM_VERSION,
        quota_policy=(),
        actions=(cover_owner, swap),
        swaps=(swap,),
        final_layout=(4, 1, -1, 5, 3, 0, 4, 2, -1),
        final_owner_slots=(5, 1, 7, 4, 6, 3),
    )
    recorded: list[tuple[int, int]] = []

    def capture_grouped(grouped_entries, _ep_rank, _ep_size, _ep_group, **_kwargs):
        for (src_rank, dst_rank), entries in grouped_entries.items():
            if entries:
                entry = entries[0]
                recorded.append(
                    (
                        src_rank * layer.num_local_experts + entry.src_slot,
                        dst_rank * layer.num_local_experts + entry.dst_slot,
                    )
                )

    monkeypatch.setattr(expert_swap_module, "_cover_grouped_slot_entries_atomic", capture_grouped)

    committed = manager._execute_placement_plan(layer, plan)

    assert committed == [
        "layers.0.mlp.experts:replica(4->0)",
        "layers.0.mlp.experts:swap(2<->5)",
    ]
    assert sorted(recorded) == [(3, 7), (6, 0), (7, 3)]
    assert tuple(layer.slot_to_logical.tolist()) == plan.final_layout
    assert tuple(layer.logical_to_physical.tolist()) == plan.final_owner_slots


def test_placement_executor_keeps_metadata_on_staged_transfer_failure(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    old_policy = (expert_swap_module.QuotaPolicyEntry(0, 3, (0, 1), (1, 1)),)
    layer.active_quota_policy = old_policy
    before_gate = module.gate_up_proj.detach().clone()
    before_down = module.down_proj.detach().clone()
    action = PlacementAction("replica", 4, 2, 3, -1)
    plan = SimpleNamespace(
        algorithm_version=expert_swap_module.CORE_MOE_ALGORITHM_VERSION,
        quota_policy=(),
        actions=(action,),
        swaps=(),
        final_layout=(0, 1, 3, 2, 3, -1),
        final_owner_slots=(0, 1, 3, 4),
    )

    def fail_transfer(*_args, **_kwargs):
        raise RuntimeError("injected state-transfer failure")

    monkeypatch.setattr(expert_swap_module, "_cover_grouped_slot_entries_atomic", fail_transfer)
    manager.ep_group = object()

    def materialize_missing_grad(state, *_args, **_kwargs):
        if state.numel() == 1:
            state.fill_(1)

    monkeypatch.setattr(expert_swap_module.dist, "all_reduce", materialize_missing_grad)

    with pytest.raises(RuntimeError, match="injected state-transfer failure"):
        manager._execute_placement_plan(layer, plan)

    torch.testing.assert_close(module.gate_up_proj.detach(), before_gate)
    torch.testing.assert_close(module.down_proj.detach(), before_down)
    assert tuple(layer.slot_to_logical.tolist()) == (0, 1, -1, 2, 3, -1)
    assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 3, 4)
    assert layer.active_quota_policy == old_policy
    assert module.gate_up_proj.grad is None
    assert module.down_proj.grad is None


def test_placement_executor_rejects_same_rank_replica_before_metadata_commit():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    module.gate_up_proj.detach()[2].fill_(-17)
    module.down_proj.detach()[2].fill_(-23)
    before_gate = module.gate_up_proj.detach().clone()
    before_down = module.down_proj.detach().clone()
    action = PlacementAction("replica", 0, 2, 0, 3)
    plan = SimpleNamespace(
        quota_policy=(),
        actions=(action,),
        swaps=(),
        final_layout=(0, 1, 0, 2, 3, -1),
        final_owner_slots=(0, 1, 3, 4),
    )

    with pytest.raises(ValueError, match="duplicates an expert on one device"):
        manager._execute_placement_plan(layer, plan)

    torch.testing.assert_close(module.gate_up_proj.detach(), before_gate)
    torch.testing.assert_close(module.down_proj.detach(), before_down)
    assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 3, 2, 3, -1)
    assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 3, 4)


def test_redundant_slot_checkpoint_round_trips_quota_policy():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer)
    layer.active_quota_policy = (expert_swap_module.QuotaPolicyEntry(0, 3, (0, 1), (3, 5)),)

    saved = manager.state_dict()
    layer.active_quota_policy = ()
    layer.pending_physical_routes = torch.zeros((1, 1), dtype=torch.long)
    manager.load_state_dict(saved)

    assert saved["version"] == 3
    assert layer.active_quota_policy == (expert_swap_module.QuotaPolicyEntry(0, 3, (0, 1), (3, 5)),)
    assert layer.pending_physical_routes is None


def test_redundant_slot_checkpoint_keeps_layout_and_clears_incompatible_quota(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        expert_swap_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    keys = ("layers.0.mlp.experts", "layers.1.mlp.experts")
    for key in keys:
        module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
        manager.register_layer(key, module)
        layer = manager.layers[key]
        layer.slot_to_logical = torch.tensor([0, 1, 3, 2, 3, -1], dtype=torch.long)
        manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
        layer.active_quota_policy = (expert_swap_module.QuotaPolicyEntry(0, 3, (0, 1), (3, 5)),)
    saved = manager.state_dict()
    for key in keys:
        saved["layers"][key]["quota_algorithm_version"] = "coremoe-current-v1"

    for key in keys:
        layer = manager.layers[key]
        layer.slot_to_logical = torch.tensor([0, 1, -1, 2, 3, -1], dtype=torch.long)
        manager._refresh_layer_mapping_from_slots(layer)
    manager.load_state_dict(saved)

    for key in keys:
        layer = manager.layers[key]
        assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 3, 2, 3, -1)
        assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 3, 4)
        assert layer.active_quota_policy == ()
    assert len(warnings) == 1
    assert "clearing all incompatible quota policies" in warnings[0]


def test_redundant_slot_checkpoint_validation_is_atomic_across_layers():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    keys = ("layers.0.mlp.experts", "layers.1.mlp.experts")
    for key in keys:
        manager.register_layer(key, _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2))
    before = {
        key: (
            manager.layers[key].slot_to_logical.clone(),
            manager.layers[key].logical_to_physical.clone(),
        )
        for key in keys
    }
    saved = manager.state_dict()
    saved["layers"][keys[0]]["slot_to_logical"] = [0, 1, 3, 2, 3, -1]
    saved["layers"][keys[0]]["logical_to_physical"] = [0, 1, 3, 4]
    saved["layers"][keys[0]].pop("quota_layout_crc32")
    saved["layers"][keys[1]]["slot_to_logical"] = [0, 1, 2, 2, -1, -1]
    saved["layers"][keys[1]].pop("quota_layout_crc32")

    with pytest.raises(ValueError, match="drops at least one logical expert"):
        manager.load_state_dict(saved)

    for key in keys:
        torch.testing.assert_close(manager.layers[key].slot_to_logical, before[key][0])
        torch.testing.assert_close(manager.layers[key].logical_to_physical, before[key][1])


def test_redundant_slot_padding_is_scoped_to_ep_expert_params():
    tensor_shape = torch.Size((128, 2048, 768))
    target_shape = torch.Size((9, 2048, 768))
    name = "model.layers.0.mlp.experts.down_proj"

    assert plan_allows_slot_padding("ep", name, tensor_shape, target_shape, para_size=16)
    assert loader_allows_slot_padding("ep", name, tensor_shape, target_shape, para_size=16)
    assert not plan_allows_slot_padding("tp", name, tensor_shape, target_shape, para_size=16)
    assert not loader_allows_slot_padding("tp", name, tensor_shape, target_shape, para_size=16)
    assert not plan_allows_slot_padding("ep", "model.embed_tokens.weight", torch.Size((128, 2048)), (9, 2048), 16)
    assert not loader_allows_slot_padding("ep", "model.embed_tokens.weight", torch.Size((128, 2048)), target_shape, 16)


def test_redundant_slot_padding_takes_precedence_over_generic_slice(monkeypatch):
    class _FakeParallelState:
        extra_parallel_sizes = {"ep": 16}

        @staticmethod
        def extra_parallel_enabled(_name):
            return True

        @staticmethod
        def extra_parallel_rank(_name):
            return 3

    monkeypatch.setattr("veomni.distributed.parallel_state.get_parallel_state", lambda: _FakeParallelState())
    plan = ParallelPlan({"ep": {"*.mlp.experts.gate_up_proj": Shard(0)}})
    tensor = torch.arange(128 * 2, dtype=torch.float32).view(128, 2)
    sliced = plan._slice_shard_tensor(
        tensor,
        "model.layers.0.mlp.experts.gate_up_proj",
        (16, 2),
        "ep",
    )

    assert tuple(sliced.shape) == (16, 2)
    torch.testing.assert_close(sliced[:8], tensor[24:32])
    torch.testing.assert_close(sliced[8:], torch.zeros((8, 2)))


def test_redundant_slot_state_collection_skips_ep_symmetry_checks(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: pytest.fail("unexpected symmetry check"))

    entries = manager._slot_op_cover_entries(
        manager.layers["layers.0.mlp.experts"],
        src_slot=0,
        dst_slot=2,
    )

    assert len(entries) == 6


def _placement_executor_atomic_state_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=world_size,
        ep_rank=rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=world_size, group_sizes=(world_size,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
    )
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=3, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(rank * 2)
    for param in (module.gate_up_proj, module.down_proj):
        optimizer.state[param]["exp_avg"].copy_(param.detach() + 1000)
        optimizer.state[param]["exp_avg_sq"].copy_(param.detach() + 2000)
    module.gate_up_proj.grad = module.gate_up_proj.detach().clone() + 3000
    module.down_proj.grad = module.down_proj.detach().clone() + 4000

    swap = PlacementAction("swap", 0, 3, 0, 2)
    replica = PlacementAction("replica", 4, 2, 3, -1)
    plan = SimpleNamespace(
        algorithm_version=expert_swap_module.CORE_MOE_ALGORITHM_VERSION,
        quota_policy=(),
        actions=(swap, replica),
        swaps=(swap,),
        final_layout=(2, 1, 3, 0, 3, -1),
        final_owner_slots=(3, 1, 0, 4),
    )

    committed = manager._execute_placement_plan(manager.layers[key], plan)

    assert committed == [f"{key}:swap(0<->2)", f"{key}:replica(3->2)"]
    layer = manager.layers[key]
    assert tuple(layer.slot_to_logical.tolist()) == plan.final_layout
    assert tuple(layer.logical_to_physical.tolist()) == plan.final_owner_slots
    expected_gate = {0: [3.0, 2.0, 4.0], 1: [1.0, 4.0, 5.0]}[rank]
    expected_down = [value + 100 for value in expected_gate]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down))
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
        torch.tensor([value + 1000 for value in expected_gate]),
    )
    torch.testing.assert_close(
        optimizer.state[module.down_proj]["exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 2100 for value in expected_gate]),
    )
    torch.testing.assert_close(
        module.gate_up_proj.grad[:, 0, 0],
        torch.tensor([value + 3000 for value in expected_gate]),
    )
    torch.testing.assert_close(
        module.down_proj.grad[:, 0, 0],
        torch.tensor([value + 4100 for value in expected_gate]),
    )


def test_placement_executor_batches_parameter_gradient_and_optimizer_state():
    torchrun(_placement_executor_atomic_state_worker, world_size=2, backend="gloo")


def _placement_executor_pure_swap_p2p_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=world_size,
        ep_rank=rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=world_size, group_sizes=(world_size,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=2, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01, amsgrad=True)
    manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(rank * 2)
    for param in (module.gate_up_proj, module.down_proj):
        optimizer.state[param]["exp_avg"].copy_(param.detach() + 1000)
        optimizer.state[param]["exp_avg_sq"].copy_(param.detach() + 2000)
        optimizer.state[param]["max_exp_avg_sq"].copy_(param.detach() + 3000)

    action = PlacementAction("swap", 0, 3, 0, 3)
    plan = SimpleNamespace(
        algorithm_version=None,
        quota_policy=(),
        actions=(action,),
        final_layout=(3, 1, 2, 0),
        final_owner_slots=(),
    )
    original_all_to_all_single = dist.all_to_all_single
    original_all_reduce = dist.all_reduce
    original_batch_isend_irecv = dist.batch_isend_irecv
    p2p_call_count = 0

    def _fail_collective(*_args, **_kwargs):
        raise AssertionError("pure expert swap must not use a collective")

    def _count_batch_isend_irecv(ops):
        nonlocal p2p_call_count
        p2p_call_count += 1
        return original_batch_isend_irecv(ops)

    dist.all_to_all_single = _fail_collective
    dist.all_reduce = _fail_collective
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        committed = manager._execute_placement_plan(manager.layers[key], plan)
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.all_reduce = original_all_reduce
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert committed == [f"{key}:swap(0<->3)"]
    assert p2p_call_count == 1
    assert manager.layers[key].logical_to_physical.tolist() == [3, 1, 2, 0]
    expected_gate = {0: [4.0, 2.0], 1: [3.0, 1.0]}[rank]
    expected_down = [value + 100 for value in expected_gate]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down))
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
        torch.tensor([value + 1000 for value in expected_gate]),
    )
    torch.testing.assert_close(
        optimizer.state[module.down_proj]["exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 2100 for value in expected_gate]),
    )
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["max_exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 3000 for value in expected_gate]),
    )


def test_placement_executor_pure_swap_uses_p2p_without_status_collectives():
    torchrun(_placement_executor_pure_swap_p2p_worker, world_size=2, backend="gloo")


def _placement_executor_layer_swap_sparse_all_to_all_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=world_size,
        ep_rank=rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=world_size, group_sizes=(world_size,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="layer",
    )
    module = _FakeAsymmetricExperts(num_experts=4, local_start=rank * 2, local_experts=2, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01, amsgrad=True)
    manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(rank * 2)
    for param in (module.gate_up_proj, module.down_proj):
        optimizer.state[param]["exp_avg"].copy_(param.detach() + 1000)
        optimizer.state[param]["exp_avg_sq"].copy_(param.detach() + 2000)
        optimizer.state[param]["max_exp_avg_sq"].copy_(param.detach() + 3000)

    action = PlacementAction("swap", 0, 3, 0, 3)
    plan = SimpleNamespace(
        algorithm_version=None,
        quota_policy=(),
        actions=(action,),
        final_layout=(3, 1, 2, 0),
        final_owner_slots=(),
    )
    original_all_to_all_single = dist.all_to_all_single
    original_all_reduce = dist.all_reduce
    original_batch_isend_irecv = dist.batch_isend_irecv
    original_cat = torch.cat
    all_to_all_call_count = 0

    def _count_all_to_all_single(*args, **kwargs):
        nonlocal all_to_all_call_count
        all_to_all_call_count += 1
        if kwargs.get("input_split_sizes") is None or kwargs.get("output_split_sizes") is None:
            raise AssertionError("pure swap must derive split sizes from the shared plan")
        return original_all_to_all_single(*args, **kwargs)

    def _fail_all_reduce(*_args, **_kwargs):
        raise AssertionError("production sparse expert swap must not use a status All-Reduce")

    def _fail_batch_isend_irecv(*_args, **_kwargs):
        raise AssertionError("layer expert swap must not launch an asynchronous P2P transfer")

    def _fail_cat(*_args, **_kwargs):
        raise AssertionError("pure swap must pack directly into the reusable send buffer")

    dist.all_to_all_single = _count_all_to_all_single
    dist.all_reduce = _fail_all_reduce
    dist.batch_isend_irecv = _fail_batch_isend_irecv
    torch.cat = _fail_cat
    try:
        committed = manager._execute_placement_plan(manager.layers[key], plan)
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.all_reduce = original_all_reduce
        dist.batch_isend_irecv = original_batch_isend_irecv
        torch.cat = original_cat

    assert committed == [f"{key}:swap(0<->3)"]
    assert all_to_all_call_count == 1
    assert not manager._pending_layer_swaps
    assert len(manager._swap_staging_buffers) == 1
    assert manager.layers[key].logical_to_physical.tolist() == [3, 1, 2, 0]
    expected_gate = {0: [4.0, 2.0], 1: [3.0, 1.0]}[rank]
    expected_down = [value + 100 for value in expected_gate]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down))
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
        torch.tensor([value + 1000 for value in expected_gate]),
    )
    torch.testing.assert_close(
        optimizer.state[module.down_proj]["exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 2100 for value in expected_gate]),
    )
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["max_exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 3000 for value in expected_gate]),
    )


def test_placement_executor_layer_swap_uses_sparse_full_group_all_to_all():
    torchrun(_placement_executor_layer_swap_sparse_all_to_all_worker, world_size=2, backend="gloo")


def _placement_executor_debug_payload_mismatch_worker():
    rank = dist.get_rank()
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=dist.get_world_size(),
        ep_rank=rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        debug_validate=True,
    )
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=2, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    if rank == 0:
        module.gate_up_proj.grad = torch.ones_like(module.gate_up_proj)

    action = PlacementAction("swap", 0, 3, 0, 3)
    plan = SimpleNamespace(
        algorithm_version=None,
        quota_policy=(),
        actions=(action,),
        final_layout=(3, 1, 2, 0),
        final_owner_slots=(),
    )

    with pytest.raises(RuntimeError, match="asymmetric swap payload"):
        manager._execute_placement_plan(manager.layers[key], plan)

    assert manager.layers[key].logical_to_physical.tolist() == [0, 1, 2, 3]
    assert manager._pending_layer_swaps == {}


def test_placement_executor_debug_rejects_optional_gradient_mismatch():
    torchrun(_placement_executor_debug_payload_mismatch_worker, world_size=2, backend="gloo")


def _redundant_slot_cover_without_optimizer_state_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            hierarchy_group_sizes=[world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    local_start = rank * base_local_experts
    module = _FakeExperts(num_experts, local_start, slot_capacity, hidden_size)
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)

    assert module.gate_up_proj not in optimizer.state
    assert module.down_proj not in optimizer.state
    manager._commit_layer_slot_ops(
        "layers.0.mlp.experts",
        [expert_swap_module._SlotOpCandidate("cover", 4, 2)],
    )
    assert module.gate_up_proj not in optimizer.state
    assert module.down_proj not in optimizer.state


def test_redundant_slot_cover_does_not_initialize_optimizer_state():
    torchrun(_redundant_slot_cover_without_optimizer_state_worker, world_size=2, backend="gloo")


def _redundant_slot_cover_and_grad_sync_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    local_start = rank * base_local_experts
    module = _FakeExperts(num_experts, local_start, slot_capacity, hidden_size)
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(local_start)
    for param in (module.gate_up_proj, module.down_proj):
        opt_state = optimizer.state[param]
        opt_state["exp_avg"].copy_(param.detach() + 1000)
        opt_state["exp_avg_sq"].copy_(param.detach() + 2000)
        param.grad = param.detach().clone() + 3000

    committed = manager._commit_layer_slot_ops(
        "layers.0.mlp.experts",
        [expert_swap_module._SlotOpCandidate("cover", 4, 2)],
    )
    assert committed == ["layers.0.mlp.experts:COVER(4->2)[1:1,0:2]"]
    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.slot_to_logical is not None
    torch.testing.assert_close(layer.slot_to_logical, torch.tensor([0, 1, 3, 2, 3, -1]))

    if rank == 0:
        torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor([1.0, 2.0, 4.0]))
        torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor([101.0, 102.0, 104.0]))
        torch.testing.assert_close(
            optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
            torch.tensor([1001.0, 1002.0, 1004.0]),
        )
        torch.testing.assert_close(module.gate_up_proj.grad[:, 0, 0], torch.tensor([3001.0, 3002.0, 3004.0]))

    hidden = torch.randn(4, hidden_size, dtype=torch.double)
    selected_experts = torch.tensor([[3, 0], [2, 3], [1, 2], [2, 1]], dtype=torch.long)
    routing_weights = torch.full((4, 2), 0.5, dtype=torch.double)
    full_weight = torch.stack(
        [
            torch.full((hidden_size, hidden_size), float(expert + 1), dtype=torch.double)
            for expert in range(num_experts)
        ]
    )
    permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    assert len(tokens_per_local_expert) == slot_capacity
    expert_outputs = _apply_local_linear_experts(
        permuted_tokens,
        tokens_per_local_expert,
        module.gate_up_proj.detach().to(torch.double),
    )
    output = rank_dedup_combine(expert_outputs, ctx)
    baseline_output = _eager_linear_moe(hidden, selected_experts, routing_weights, full_weight)
    torch.testing.assert_close(output, baseline_output)

    module.gate_up_proj.grad.zero_()
    module.down_proj.grad.zero_()
    if rank == 0:
        module.gate_up_proj.grad[2].fill_(10.0)
        module.down_proj.grad[2].fill_(20.0)
    else:
        module.gate_up_proj.grad[1].fill_(2.0)
        module.down_proj.grad[1].fill_(3.0)
    original_send = dist.send
    original_recv = dist.recv
    original_batch_isend_irecv = dist.batch_isend_irecv
    send_call_count = 0
    recv_call_count = 0
    batch_call_count = 0
    batch_op_counts = []

    def _count_send(*args, **kwargs):
        nonlocal send_call_count
        send_call_count += 1
        return original_send(*args, **kwargs)

    def _count_recv(*args, **kwargs):
        nonlocal recv_call_count
        recv_call_count += 1
        return original_recv(*args, **kwargs)

    def _count_batch_isend_irecv(ops):
        nonlocal batch_call_count
        batch_call_count += 1
        batch_op_counts.append(len(ops))
        return original_batch_isend_irecv(ops)

    dist.send = _count_send
    dist.recv = _count_recv
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        manager.sync_redundant_gradients()
    finally:
        dist.send = original_send
        dist.recv = original_recv
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert send_call_count == 0
    assert recv_call_count == 0
    assert batch_call_count == 1
    assert batch_op_counts == [2]

    if rank == 0:
        torch.testing.assert_close(module.gate_up_proj.grad[2], torch.full_like(module.gate_up_proj.grad[2], 12.0))
        torch.testing.assert_close(module.down_proj.grad[2], torch.full_like(module.down_proj.grad[2], 23.0))
    else:
        torch.testing.assert_close(module.gate_up_proj.grad[1], torch.full_like(module.gate_up_proj.grad[1], 12.0))
        torch.testing.assert_close(module.down_proj.grad[1], torch.full_like(module.down_proj.grad[1], 23.0))


def test_redundant_slot_cover_preserves_semantics_and_syncs_copy_gradients():
    torchrun(_redundant_slot_cover_and_grad_sync_worker, world_size=2, backend="gloo")


def _fixed_r2_parameter_initialization_worker():
    rank = dist.get_rank()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=0,
            hierarchy_group_sizes=[2],
            redundant_slot_increment_per_device=2,
            max_slot_op_search_rounds=0,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None
    manager = state.expert_swap_manager
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=4, hidden_size=2)
    with torch.no_grad():
        module.gate_up_proj[2:].zero_()
        module.down_proj[2:].zero_()
    manager.register_layer("layers.0.mlp.experts", module)

    original_all_to_all_single = dist.all_to_all_single
    collective_calls = 0

    def _count_all_to_all_single(*args, **kwargs):
        nonlocal collective_calls
        collective_calls += 1
        return original_all_to_all_single(*args, **kwargs)

    dist.all_to_all_single = _count_all_to_all_single
    try:
        manager.install_fixed_r2_layout()
    finally:
        dist.all_to_all_single = original_all_to_all_single

    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.slot_to_logical is not None
    torch.testing.assert_close(layer.slot_to_logical, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    torch.testing.assert_close(layer.logical_to_physical, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor([101.0, 102.0, 103.0, 104.0]))
    assert layer.fixed_r2_layout is True
    selected = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    torch.testing.assert_close(manager._map_logical_to_slot(layer, selected), selected + rank * 4)
    assert collective_calls == 2
    manager.prepare_calibrations = lambda _step: pytest.fail("static fixed-R2 must not run the planner")
    assert manager.maybe_swap(1) == "none"


def test_fixed_r2_parameter_initialization_packs_all_layer_tensors():
    torchrun(_fixed_r2_parameter_initialization_worker, world_size=2, backend="gloo")


def _fixed_r2_grad_sync_batches_all_experts_worker():
    rank = dist.get_rank()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[2],
            redundant_slot_increment_per_device=2,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None
    manager = state.expert_swap_manager
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=4, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 6, 7))
    module2 = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=4, hidden_size=2)
    manager.register_layer("layers.1.mlp.experts", module2)
    layer2 = manager.layers["layers.1.mlp.experts"]
    layer2.slot_to_logical = layer.slot_to_logical.clone()
    manager._refresh_layer_mapping_from_slots(layer2, (0, 1, 6, 7))

    def _set_grads():
        for current_module, current_layer in ((module, layer), (module2, layer2)):
            current_module.gate_up_proj.grad = torch.zeros_like(current_module.gate_up_proj)
            current_module.down_proj.grad = torch.zeros_like(current_module.down_proj)
            for local_slot, logical_expert in enumerate(
                current_layer.slot_to_logical[rank * 4 : (rank + 1) * 4].tolist()
            ):
                current_module.gate_up_proj.grad[local_slot].fill_(10 * (rank + 1) + logical_expert)
                current_module.down_proj.grad[local_slot].fill_(100 * (rank + 1) + logical_expert)

    original_batch_isend_irecv = dist.batch_isend_irecv
    batch_op_counts = []

    def _count_batch_isend_irecv(ops):
        batch_op_counts.append(len(ops))
        return original_batch_isend_irecv(ops)

    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        _set_grads()
        manager.sync_redundant_gradients()
        first_buffer_ptrs = {key: value.data_ptr() for key, value in manager._replica_grad_buffers.items()}
        _set_grads()
        manager.sync_redundant_gradients()
    finally:
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert batch_op_counts == [2, 2, 2, 2]
    assert len(first_buffer_ptrs) == 2
    assert {key: value.data_ptr() for key, value in manager._replica_grad_buffers.items()} == first_buffer_ptrs
    for current_module, current_layer in ((module, layer), (module2, layer2)):
        for local_slot, logical_expert in enumerate(current_layer.slot_to_logical[rank * 4 : (rank + 1) * 4].tolist()):
            torch.testing.assert_close(
                current_module.gate_up_proj.grad[local_slot],
                torch.full_like(current_module.gate_up_proj.grad[local_slot], 30 + 2 * logical_expert),
            )
            torch.testing.assert_close(
                current_module.down_proj.grad[local_slot],
                torch.full_like(current_module.down_proj.grad[local_slot], 300 + 2 * logical_expert),
            )


def test_fixed_r2_grad_sync_batches_all_experts_in_one_wave():
    torchrun(_fixed_r2_grad_sync_batches_all_experts_worker, world_size=2, backend="gloo")


def _three_copy_owner_grad_sync_uses_two_waves_worker():
    rank = dist.get_rank()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[3],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None
    manager = state.expert_swap_manager
    module = _FakeExperts(num_experts=3, local_start=rank, local_experts=2, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    layer = manager.layers["layers.0.mlp.experts"]
    layer.slot_to_logical = torch.tensor([0, -1, 0, 1, 0, 2], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (2, 3, 5))

    module.gate_up_proj.grad = torch.zeros_like(module.gate_up_proj)
    module.down_proj.grad = torch.zeros_like(module.down_proj)
    module.gate_up_proj.grad[0].fill_(rank + 1)
    module.down_proj.grad[0].fill_(10 * (rank + 1))

    original_batch_isend_irecv = dist.batch_isend_irecv
    batch_op_counts = []

    def _count_batch_isend_irecv(ops):
        batch_op_counts.append(len(ops))
        return original_batch_isend_irecv(ops)

    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        manager.sync_redundant_gradients()
    finally:
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert len(batch_op_counts) == 2
    torch.testing.assert_close(module.gate_up_proj.grad[0], torch.full_like(module.gate_up_proj.grad[0], 6.0))
    torch.testing.assert_close(module.down_proj.grad[0], torch.full_like(module.down_proj.grad[0], 60.0))


def test_three_copy_nonlowest_owner_grad_sync_uses_two_waves():
    torchrun(_three_copy_owner_grad_sync_uses_two_waves_worker, world_size=3, backend="gloo")


def test_infinity_norm_combines_multiple_extra_parallel_groups():
    total = _combine_reduced_norm_totals(
        torch.tensor(2.0),
        {"emb": torch.tensor(5.0), "ep": torch.tensor(4.0)},
        math.inf,
    )
    torch.testing.assert_close(total, torch.tensor(5.0))


def test_extra_parallel_infinity_norm_uses_max_across_all_groups(monkeypatch):
    clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
    hiermoe_module = importlib.import_module("veomni.distributed.moe.hiermoe")

    class _FakeMesh:
        def __getitem__(self, _name):
            return self

        def get_group(self):
            return None

    non_extra = torch.nn.Parameter(torch.ones(1))
    embedding = torch.nn.Parameter(torch.ones(1))
    expert = torch.nn.Parameter(torch.ones(1))
    for parameter in (non_extra, embedding, expert):
        parameter.grad = torch.ones_like(parameter)
    model = SimpleNamespace(
        _extra_parallel_param_groups={
            "non_extra_parallel": [non_extra],
            "emb": [embedding],
            "ep": [expert],
        }
    )
    parallel_state = SimpleNamespace(
        fsdp_group=None,
        extra_parallel_names=("emb", "ep"),
        extra_parallel_enabled=lambda _name: True,
        extra_parallel_group=lambda _name: None,
        extra_parallel_fsdp_device_mesh={"emb": _FakeMesh(), "ep": _FakeMesh()},
    )
    reduced = {id(non_extra): 2.0, id(embedding): 5.0, id(expert): 4.0}
    clipped_totals = []

    monkeypatch.setattr(clip_grad_norm_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(clip_grad_norm_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(hiermoe_module, "get_hiermoe_redundant_grad_norm_masks", lambda: {})
    monkeypatch.setattr(
        clip_grad_norm_module,
        "_fsdp2_reduce_group",
        lambda params, **_kwargs: torch.tensor(reduced[id(params[0])] if params else 0.0),
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grads_with_norm_",
        lambda _params, _max_norm, total_norm, **_kwargs: clipped_totals.append(float(total_norm)),
    )

    total = clip_grad_norm_module.extra_parallel_fsdp2_clip_grad_norm(
        model,
        max_norm=1.0,
        norm_type=math.inf,
        foreach=False,
    )

    torch.testing.assert_close(total, torch.tensor(5.0))
    assert clipped_totals == [5.0, 5.0, 5.0]


def _redundant_grad_norm_dtensor_worker():
    clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
    clip_grad_norm_module.get_device_type = lambda: "cpu"
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep", "fsdp"))
    fsdp_mesh = mesh["fsdp"]
    global_grad = torch.tensor([[3.0, 3.0], [4.0, 4.0]])
    param = torch.nn.Parameter(distribute_tensor(torch.zeros_like(global_grad), fsdp_mesh, [Shard(1)]))
    param.grad = distribute_tensor(global_grad, fsdp_mesh, [Shard(1)])

    ep_rank = mesh.get_local_rank("ep")
    primary_rows = torch.tensor([ep_rank == 0, ep_rank == 1], dtype=torch.bool)
    masks = {id(param): primary_rows}
    reduce_groups = [("fsdp", mesh.get_group("fsdp")), ("ep", mesh.get_group("ep"))]

    pth_sum = _fsdp2_reduce_group([param], 2.0, reduce_groups, grad_row_masks=masks)
    total_norm = _finalize_total_norm(pth_sum, 2.0)
    torch.testing.assert_close(total_norm, torch.tensor(math.sqrt(50.0)))

    infinity_norm = _fsdp2_reduce_group([param], math.inf, reduce_groups, grad_row_masks=masks)
    torch.testing.assert_close(infinity_norm, torch.tensor(4.0))

    local_before = param.grad.to_local().clone()
    torch.nn.utils.clip_grads_with_norm_([param], 1.0, total_norm, foreach=False)
    expected_coefficient = 1.0 / (math.sqrt(50.0) + 1e-6)
    torch.testing.assert_close(
        param.grad.to_local(),
        local_before * expected_coefficient,
        atol=1e-6,
        rtol=1e-6,
    )


def test_redundant_grad_norm_masks_dtensor_copies_once_and_clips_all_copies():
    torchrun(_redundant_grad_norm_dtensor_worker, world_size=4, backend="gloo")


def _redundant_slot_cover_backward_matches_logical_expert_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    module = _FakeExperts(
        num_experts,
        local_start=rank * base_local_experts,
        local_experts=slot_capacity,
        hidden_size=hidden_size,
    ).double()
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    manager._commit_layer_slot_ops("layers.0.mlp.experts", [expert_swap_module._SlotOpCandidate("cover", 4, 2)])

    hidden = torch.randn(5, hidden_size, dtype=torch.double, requires_grad=True)
    selected_experts = torch.tensor([[3, 0], [0, 3], [1, 2], [2, 1], [3, 1]], dtype=torch.long)
    routing_weights = torch.full((5, 2), 0.5, dtype=torch.double)

    permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, module.gate_up_proj)
    output = rank_dedup_combine(expert_outputs, ctx)
    output.square().sum().backward()
    manager.sync_redundant_gradients()

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    full_weight = torch.stack(
        [
            torch.full((hidden_size, hidden_size), float(expert + 1), dtype=torch.double)
            for expert in range(num_experts)
        ]
    ).requires_grad_(True)
    baseline_output = _eager_linear_moe(baseline_hidden, selected_experts, routing_weights, full_weight)
    baseline_output.square().sum().backward()
    dist.all_reduce(full_weight.grad, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(output, baseline_output)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad)

    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.canonical_physical_slots is not None
    for logical_expert in range(num_experts):
        canonical_slot = int(layer.canonical_physical_slots[logical_expert].item())
        owner_rank, local_slot = divmod(canonical_slot, slot_capacity)
        if rank == owner_rank:
            torch.testing.assert_close(
                module.gate_up_proj.grad[local_slot],
                full_weight.grad[logical_expert],
                atol=1e-6,
                rtol=1e-6,
            )
    if rank == 0:
        torch.testing.assert_close(module.gate_up_proj.grad[2], full_weight.grad[3], atol=1e-6, rtol=1e-6)


def test_redundant_slot_cover_backward_matches_logical_expert():
    torchrun(_redundant_slot_cover_backward_matches_logical_expert_worker, world_size=2, backend="gloo")


def _redundant_slot_cover_hierarchical_backward_matches_logical_expert_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[2, world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 8
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    module = _FakeExperts(
        num_experts,
        local_start=rank * base_local_experts,
        local_experts=slot_capacity,
        hidden_size=hidden_size,
    ).double()
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    manager._commit_layer_slot_ops("layers.0.mlp.experts", [expert_swap_module._SlotOpCandidate("cover", 7, 2)])

    hidden = torch.randn(6, hidden_size, dtype=torch.double, requires_grad=True)
    selected_experts = torch.tensor([[5, 0], [0, 5], [1, 6], [2, 7], [5, 1], [3, 4]], dtype=torch.long)
    routing_weights = torch.full((6, 2), 0.5, dtype=torch.double)

    permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    assert ctx.mode == "hierarchical"
    expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, module.gate_up_proj)
    output = rank_dedup_combine(expert_outputs, ctx)
    output.square().sum().backward()
    manager.sync_redundant_gradients()

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    full_weight = torch.stack(
        [
            torch.full((hidden_size, hidden_size), float(expert + 1), dtype=torch.double)
            for expert in range(num_experts)
        ]
    ).requires_grad_(True)
    baseline_output = _eager_linear_moe(baseline_hidden, selected_experts, routing_weights, full_weight)
    baseline_output.square().sum().backward()
    dist.all_reduce(full_weight.grad, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(output, baseline_output)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad)

    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.canonical_physical_slots is not None
    for logical_expert in range(num_experts):
        canonical_slot = int(layer.canonical_physical_slots[logical_expert].item())
        owner_rank, local_slot = divmod(canonical_slot, slot_capacity)
        if rank == owner_rank:
            torch.testing.assert_close(
                module.gate_up_proj.grad[local_slot],
                full_weight.grad[logical_expert],
                atol=1e-6,
                rtol=1e-6,
            )
    if rank == 0:
        torch.testing.assert_close(module.gate_up_proj.grad[2], full_weight.grad[5], atol=1e-6, rtol=1e-6)


def test_redundant_slot_cover_hierarchical_backward_matches_logical_expert():
    torchrun(_redundant_slot_cover_hierarchical_backward_matches_logical_expert_worker, world_size=4, backend="gloo")


def _redundant_slot_cover_optimizer_step_keeps_copies_equal_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    local_start = rank * base_local_experts
    module = _FakeExperts(num_experts, local_start, slot_capacity, hidden_size)
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)

    committed = manager._commit_layer_slot_ops(
        "layers.0.mlp.experts",
        [expert_swap_module._SlotOpCandidate("cover", 4, 2)],
    )
    assert committed == ["layers.0.mlp.experts:COVER(4->2)[1:1,0:2]"]

    module.gate_up_proj.grad = torch.zeros_like(module.gate_up_proj)
    module.down_proj.grad = torch.zeros_like(module.down_proj)
    if rank == 0:
        module.gate_up_proj.grad[2].fill_(7.0)
        module.down_proj.grad[2].fill_(11.0)
    else:
        module.gate_up_proj.grad[1].fill_(13.0)
        module.down_proj.grad[1].fill_(17.0)

    manager.sync_redundant_gradients()
    optimizer.step()

    copy_slot = 2 if rank == 0 else 1
    tensors_to_compare = (
        module.gate_up_proj.detach()[copy_slot].clone(),
        module.down_proj.detach()[copy_slot].clone(),
        optimizer.state[module.gate_up_proj]["exp_avg"][copy_slot].detach().clone(),
        optimizer.state[module.gate_up_proj]["exp_avg_sq"][copy_slot].detach().clone(),
        optimizer.state[module.down_proj]["exp_avg"][copy_slot].detach().clone(),
        optimizer.state[module.down_proj]["exp_avg_sq"][copy_slot].detach().clone(),
    )
    for local_tensor in tensors_to_compare:
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)
        torch.testing.assert_close(gathered[0], gathered[1])


def test_redundant_slot_cover_optimizer_step_keeps_copies_equal():
    torchrun(_redundant_slot_cover_optimizer_step_keeps_copies_equal_worker, world_size=2, backend="gloo")


def _redundant_slot_sync_materializes_missing_copy_grad_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[world_size],
            redundant_slot_increment_per_device=1,
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    base_local_experts = num_experts // world_size
    slot_capacity = base_local_experts + 1
    local_start = rank * base_local_experts
    module = _FakeExperts(num_experts, local_start, slot_capacity, hidden_size)
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    manager._commit_layer_slot_ops(
        "layers.0.mlp.experts",
        [expert_swap_module._SlotOpCandidate("cover", 4, 2)],
    )

    if rank == 0:
        module.gate_up_proj.grad = torch.zeros_like(module.gate_up_proj)
        module.gate_up_proj.grad[2].fill_(5.0)
    else:
        module.gate_up_proj.grad = None
    module.down_proj.grad = None

    manager.sync_redundant_gradients()

    copy_slot = 2 if rank == 0 else 1
    assert module.gate_up_proj.grad is not None
    assert module.down_proj.grad is not None
    torch.testing.assert_close(
        module.gate_up_proj.grad[copy_slot], torch.full_like(module.gate_up_proj[copy_slot], 5.0)
    )
    torch.testing.assert_close(module.down_proj.grad[copy_slot], torch.zeros_like(module.down_proj[copy_slot]))


def test_redundant_slot_sync_materializes_missing_copy_grad():
    torchrun(_redundant_slot_sync_materializes_missing_copy_grad_worker, world_size=2, backend="gloo")


def test_redundant_slot_sync_zeros_only_accumulated_inactive_slot_gradients():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=1,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=0, local_experts=3, hidden_size=2)
    manager.register_layer("layers.0.mlp.experts", module)
    manager.record_local_expert_token_counts("layers.0.mlp.experts", torch.tensor([0, 1, 0]))
    manager.record_local_expert_token_counts("layers.0.mlp.experts", torch.tensor([2, 0, 0]))
    module.gate_up_proj.grad = torch.full_like(module.gate_up_proj, 3.0)
    module.down_proj.grad = torch.full_like(module.down_proj, 5.0)

    manager.sync_redundant_gradients()

    torch.testing.assert_close(module.gate_up_proj.grad[0], torch.full_like(module.gate_up_proj.grad[0], 3.0))
    torch.testing.assert_close(module.gate_up_proj.grad[1], torch.full_like(module.gate_up_proj.grad[1], 3.0))
    torch.testing.assert_close(module.gate_up_proj.grad[2], torch.zeros_like(module.gate_up_proj.grad[2]))
    torch.testing.assert_close(module.down_proj.grad[0], torch.full_like(module.down_proj.grad[0], 5.0))
    torch.testing.assert_close(module.down_proj.grad[1], torch.full_like(module.down_proj.grad[1], 5.0))
    torch.testing.assert_close(module.down_proj.grad[2], torch.zeros_like(module.down_proj.grad[2]))
    assert manager.layers["layers.0.mlp.experts"].accumulated_tokens_per_local_expert is None


def _expert_swap_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    local_experts = num_experts // world_size
    local_start = rank * local_experts
    module = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    state.expert_swap_manager.register_layer("layers.0.mlp.experts", module)
    assert state.expert_swap_manager.get_layer_key_from_params(module.gate_up_proj) == "layers.0.mlp.experts"
    assert state.expert_swap_manager.get_layer_key_from_params(module.down_proj) == "layers.0.mlp.experts"
    identity_selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    assert (
        state.expert_swap_manager.map_logical_to_physical("layers.0.mlp.experts", identity_selected)
        is identity_selected
    )
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    state.expert_swap_manager.bind_optimizer(optimizer)

    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(local_start)
    for param in (module.gate_up_proj, module.down_proj):
        opt_state = optimizer.state[param]
        opt_state["exp_avg"].copy_(param.detach() + 1000)
        opt_state["exp_avg_sq"].copy_(param.detach() + 2000)
    module.gate_up_proj.grad = module.gate_up_proj.detach().clone() + 3000
    module.down_proj.grad = module.down_proj.detach().clone() + 4000

    hidden = torch.randn(4, hidden_size, dtype=torch.double)
    selected_experts = torch.tensor([[0, 3], [3, 0], [1, 2], [2, 1]], dtype=torch.long)
    routing_weights = torch.full((4, 2), 0.5, dtype=torch.double)
    with pytest.raises(RuntimeError, match="registered layer_key"):
        rank_dedup_dispatch(
            hidden,
            selected_experts,
            routing_weights,
            num_experts,
            dist.group.WORLD,
        )

    full_weight = torch.stack(
        [
            torch.full((hidden_size, hidden_size), float(expert + 1), dtype=torch.double)
            for expert in range(num_experts)
        ]
    )
    before_tokens, before_ctx, before_counts = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    before_expert_outputs = _apply_local_linear_experts(
        before_tokens,
        before_counts,
        module.gate_up_proj.detach().to(torch.double),
    )
    before_output = rank_dedup_combine(before_expert_outputs, before_ctx)
    baseline_output = _eager_linear_moe(hidden, selected_experts, routing_weights, full_weight)
    torch.testing.assert_close(before_output, baseline_output)

    state.expert_swap_manager.swap_layer_pair("layers.0.mlp.experts", (0, 3))
    layer = state.expert_swap_manager.layers["layers.0.mlp.experts"]
    assert layer.logical_to_physical.tolist() == [3, 1, 2, 0]
    mapped_selected = state.expert_swap_manager.map_logical_to_physical("layers.0.mlp.experts", identity_selected)
    assert mapped_selected is not identity_selected
    torch.testing.assert_close(mapped_selected, torch.tensor([[3, 1], [2, 0]], dtype=torch.long))
    assert hiermoe_has_non_identity_placement()

    expected_gate_values = {0: [4.0, 2.0], 1: [3.0, 1.0]}[rank]
    expected_down_values = {0: [104.0, 102.0], 1: [103.0, 101.0]}[rank]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate_values))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down_values))
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
        torch.tensor([value + 1000 for value in expected_gate_values]),
    )
    torch.testing.assert_close(
        optimizer.state[module.down_proj]["exp_avg_sq"][:, 0, 0],
        torch.tensor([value + 2000 for value in expected_down_values]),
    )
    torch.testing.assert_close(
        module.gate_up_proj.grad[:, 0, 0],
        torch.tensor([value + 3000 for value in expected_gate_values]),
    )
    torch.testing.assert_close(
        module.down_proj.grad[:, 0, 0],
        torch.tensor([value + 4000 for value in expected_down_values]),
    )

    after_tokens, after_ctx, after_counts = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    after_expert_outputs = _apply_local_linear_experts(
        after_tokens,
        after_counts,
        module.gate_up_proj.detach().to(torch.double),
    )
    after_output = rank_dedup_combine(after_expert_outputs, after_ctx)
    torch.testing.assert_close(after_output, before_output)

    saved = hiermoe_state_dict()
    layer.logical_to_physical = torch.arange(num_experts, dtype=torch.long)
    layer.invalidate_cache()
    load_hiermoe_state_dict(saved)
    assert layer.logical_to_physical.tolist() == [3, 1, 2, 0]
    with pytest.raises(RuntimeError, match="unknown layer"):
        state.expert_swap_manager.load_state_dict(
            {
                "version": 1,
                "ep_size": world_size,
                "layers": {
                    "layers.9.mlp.experts": {
                        "num_experts": num_experts,
                        "logical_to_physical": list(range(num_experts)),
                    }
                },
            }
        )


def test_expert_swap_preserves_semantics_and_optimizer_state():
    torchrun(_expert_swap_worker, world_size=2, backend="gloo")


def _expert_swap_layer_immediate_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            expert_swap_mode="layer",
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None
    hiermoe_state_module.set_hiermoe_step(0)

    num_experts = 8
    hidden_size = 3
    local_experts = num_experts // world_size
    local_start = rank * local_experts
    module = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    manager = state.expert_swap_manager
    manager.register_layer("layers.0.mlp.experts", module)
    original_all_to_all_single = dist.all_to_all_single
    original_batch_isend_irecv = dist.batch_isend_irecv
    all_to_all_call_count = 0

    def _count_all_to_all_single(*args, **kwargs):
        nonlocal all_to_all_call_count
        all_to_all_call_count += 1
        return original_all_to_all_single(*args, **kwargs)

    def _fail_batch_isend_irecv(*_args, **_kwargs):
        raise AssertionError("layer expert swap must not launch an asynchronous P2P transfer")

    hidden = torch.randn(8, hidden_size, dtype=torch.double)
    selected_experts = torch.tensor([[0, 1]] * 8, dtype=torch.long)
    routing_weights = torch.full((8, 2), 0.5, dtype=torch.double)
    full_weight = torch.stack(
        [
            torch.full((hidden_size, hidden_size), float(expert + 1), dtype=torch.double)
            for expert in range(num_experts)
        ]
    )

    hiermoe_state_module.set_hiermoe_layer_swap_forward_enabled(False)
    rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    layer = manager.layers["layers.0.mlp.experts"]
    assert layer.logical_to_physical.tolist() == list(range(num_experts))
    assert layer.latest_selected_experts is None
    assert layer.latest_route_step == -1
    layer.planner_calibration = expert_swap_module._PlannerCalibration(
        source_step=0,
        communication_scale=0.0,
        forward_compute_per_assignment=1.0,
    )

    hiermoe_state_module.set_hiermoe_step(1)
    hiermoe_state_module.set_hiermoe_layer_swap_forward_enabled(True)
    dist.all_to_all_single = _count_all_to_all_single
    dist.batch_isend_irecv = _fail_batch_isend_irecv
    try:
        permuted_tokens, ctx, tokens_per_local_expert = rank_dedup_dispatch(
            hidden,
            selected_experts,
            routing_weights,
            num_experts,
            dist.group.WORLD,
            layer_key="layers.0.mlp.experts",
        )
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv
        hiermoe_state_module.set_hiermoe_layer_swap_forward_enabled(False)

    assert layer.logical_to_physical.tolist() != list(range(num_experts))
    assert "swap(" in state.expert_swap_pair
    gathered_counts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_counts, all_to_all_call_count)
    assert min(gathered_counts) > 0
    assert len(set(gathered_counts)) == 1
    expert_outputs = _apply_local_linear_experts(
        permuted_tokens,
        tokens_per_local_expert,
        module.gate_up_proj.detach().to(torch.double),
    )
    output = rank_dedup_combine(expert_outputs, ctx)
    baseline_output = _eager_linear_moe(hidden, selected_experts, routing_weights, full_weight)
    torch.testing.assert_close(output, baseline_output)

    captured_route = layer.latest_selected_experts.clone()
    captured_route_step = layer.latest_route_step
    captured_plan_step = layer.last_planned_step
    captured_layout = layer.logical_to_physical.clone()
    recompute_selected = torch.tensor([[6, 7]] * 8, dtype=torch.long)
    hiermoe_state_module.set_hiermoe_layer_swap_forward_enabled(False)
    recompute_tokens, recompute_ctx, recompute_counts = rank_dedup_dispatch(
        hidden,
        recompute_selected,
        routing_weights,
        num_experts,
        dist.group.WORLD,
        layer_key="layers.0.mlp.experts",
    )
    recompute_expert_outputs = _apply_local_linear_experts(
        recompute_tokens,
        recompute_counts,
        module.gate_up_proj.detach().to(torch.double),
    )
    recompute_output = rank_dedup_combine(recompute_expert_outputs, recompute_ctx)
    recompute_baseline = _eager_linear_moe(hidden, recompute_selected, routing_weights, full_weight)
    torch.testing.assert_close(recompute_output, recompute_baseline)
    torch.testing.assert_close(layer.latest_selected_experts, captured_route)
    torch.testing.assert_close(layer.logical_to_physical, captured_layout)
    assert layer.latest_route_step == captured_route_step
    assert layer.last_planned_step == captured_plan_step


def test_expert_swap_layer_mode_swaps_current_routing_before_dispatch():
    torchrun(_expert_swap_layer_immediate_worker, world_size=4, backend="gloo")


def _execute_test_swap_pairs(manager, pairs_by_layer):
    plans = []
    for layer_key, pairs in pairs_by_layer.items():
        for pair in pairs:
            plan = manager._build_layer_swap_plan(layer_key, pair)
            assert plan is not None
            plans.append(plan)
    manager._execute_swap_plans(plans)
    for plan in plans:
        layer = manager.layers[plan.layer_key]
        lhs, rhs = plan.logical_lhs, plan.logical_rhs
        layer.logical_to_physical[lhs], layer.logical_to_physical[rhs] = (
            layer.logical_to_physical[rhs].clone(),
            layer.logical_to_physical[lhs].clone(),
        )
        layer.refresh_identity()
        layer.invalidate_cache()


def _expert_swap_p2p_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 3
    local_experts = num_experts // world_size
    local_start = rank * local_experts
    module0 = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    module1 = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    state.expert_swap_manager.register_layer("layers.0.mlp.experts", module0)
    state.expert_swap_manager.register_layer("layers.1.mlp.experts", module1)
    optimizer = torch.optim.AdamW(
        [module0.gate_up_proj, module0.down_proj, module1.gate_up_proj, module1.down_proj],
        lr=0.01,
    )
    state.expert_swap_manager.bind_optimizer(optimizer)
    for param in (module0.gate_up_proj, module0.down_proj, module1.gate_up_proj, module1.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()

    original_all_to_all_single = dist.all_to_all_single
    original_batch_isend_irecv = dist.batch_isend_irecv
    p2p_call_count = 0

    def _fail_all_to_all_single(*args, **kwargs):
        raise AssertionError("Expert Swap must not use all_to_all_single for a two-rank exchange")

    def _count_batch_isend_irecv(ops):
        nonlocal p2p_call_count
        p2p_call_count += 1
        return original_batch_isend_irecv(ops)

    dist.all_to_all_single = _fail_all_to_all_single
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        _execute_test_swap_pairs(
            state.expert_swap_manager,
            {layer.key: [(0, 3)] for layer in state.expert_swap_manager.layers.values()},
        )
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert p2p_call_count == 1
    assert state.expert_swap_manager.layers["layers.0.mlp.experts"].logical_to_physical.tolist() == [3, 1, 2, 0]
    assert state.expert_swap_manager.layers["layers.1.mlp.experts"].logical_to_physical.tolist() == [3, 1, 2, 0]
    staging_ptrs = {
        key: (buffer.send.data_ptr(), buffer.recv.data_ptr())
        for key, buffer in state.expert_swap_manager._swap_staging_buffers.items()
    }

    p2p_call_count = 0
    original_bucket_bytes = expert_swap_module._MAX_SWAP_BUCKET_BYTES
    expert_swap_module._MAX_SWAP_BUCKET_BYTES = 1
    dist.all_to_all_single = _fail_all_to_all_single
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        _execute_test_swap_pairs(
            state.expert_swap_manager,
            {layer.key: [(0, 3)] for layer in state.expert_swap_manager.layers.values()},
        )
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv
        expert_swap_module._MAX_SWAP_BUCKET_BYTES = original_bucket_bytes

    assert p2p_call_count == 1
    assert state.expert_swap_manager.layers["layers.0.mlp.experts"].logical_to_physical.tolist() == [0, 1, 2, 3]
    assert state.expert_swap_manager.layers["layers.1.mlp.experts"].logical_to_physical.tolist() == [0, 1, 2, 3]
    assert staging_ptrs == {
        key: (buffer.send.data_ptr(), buffer.recv.data_ptr())
        for key, buffer in state.expert_swap_manager._swap_staging_buffers.items()
    }


def test_expert_swap_uses_p2p_not_all_to_all():
    torchrun(_expert_swap_p2p_worker, world_size=2, backend="gloo")


def _expert_swap_mixed_dtype_without_state_worker():
    rank = dist.get_rank()
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=dist.get_world_size(),
        ep_rank=rank,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=4, local_start=rank * 2, local_experts=2, hidden_size=2)
    module.down_proj = torch.nn.Parameter(module.down_proj.detach().to(torch.float64))
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    assert module.gate_up_proj.grad is None
    assert module.down_proj.grad is None
    assert manager.optimizer is None

    plan = manager._build_layer_swap_plan(key, (0, 3))
    assert plan is not None
    original_all_to_all_single = dist.all_to_all_single
    original_batch_isend_irecv = dist.batch_isend_irecv
    p2p_call_count = 0

    def _fail_all_to_all_single(*_args, **_kwargs):
        raise AssertionError("Expert Swap fast transport must not use all_to_all_single")

    def _count_batch_isend_irecv(ops):
        nonlocal p2p_call_count
        p2p_call_count += 1
        return original_batch_isend_irecv(ops)

    dist.all_to_all_single = _fail_all_to_all_single
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        manager._execute_swap_plans((plan,))
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert p2p_call_count == 1
    assert {key[1] for key in manager._swap_staging_buffers} == {torch.float32, torch.float64}
    expected_gate = {0: [4.0, 2.0], 1: [3.0, 1.0]}[rank]
    expected_down = {0: [104.0, 102.0], 1: [103.0, 101.0]}[rank]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down, dtype=torch.float64))


def test_expert_swap_batches_dtypes_without_initializing_missing_state():
    torchrun(_expert_swap_mixed_dtype_without_state_worker, world_size=2, backend="gloo")


def test_expert_swap_rejects_same_rank_pair():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    manager.register_layer("layers.0.mlp.experts", _FakeExperts(4, 0, 2, 2))

    with pytest.raises(RuntimeError, match="same-rank swap"):
        manager._build_layer_swap_plan("layers.0.mlp.experts", (0, 1))

    tensor = torch.zeros((2, 1))
    invalid_plan = expert_swap_module._LayerSwapPlan(
        layer_key="layers.0.mlp.experts",
        logical_lhs=0,
        logical_rhs=1,
        lhs_rank=0,
        rhs_rank=0,
        entries=(expert_swap_module._SwapTensorEntry(tensor, lhs_slot=0, rhs_slot=1),),
    )
    with pytest.raises(RuntimeError, match="same-rank swap"):
        manager._execute_swap_plans((invalid_plan,))


def test_expert_swap_staging_buffer_reuses_and_grows():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    device = torch.device("cpu")

    initial = manager._ensure_swap_staging_buffer(device, torch.float32, 8)
    reused = manager._ensure_swap_staging_buffer(device, torch.float32, 4)
    grown = manager._ensure_swap_staging_buffer(device, torch.float32, 16)

    assert initial.send.data_ptr() == reused.send.data_ptr()
    assert initial.recv.data_ptr() == reused.recv.data_ptr()
    assert grown.send.numel() == 16
    assert grown.recv.numel() == 16
    assert grown.send.data_ptr() != initial.send.data_ptr()


def test_expert_swap_orders_all_matching_optimizer_state_keys():
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    module = _FakeExperts(num_experts=2, local_start=0, local_experts=2, hidden_size=2)
    key = "layers.0.mlp.experts"
    manager.register_layer(key, module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    manager.bind_optimizer(optimizer)
    param = module.gate_up_proj
    state = optimizer.state[param]
    state["step"] = torch.zeros(())
    for offset, state_name in enumerate(
        ("momentum_buffer", "custom_z", "exp_avg_sq", "compensation", "exp_avg", "max_exp_avg_sq", "custom_a"),
        start=1,
    ):
        state[state_name] = torch.full_like(param, float(offset))

    items = manager._optimizer_state_slot_items_for_slot_op(param)

    assert [descriptor for descriptor, _tensor in items] == [
        "optimizer[0].exp_avg",
        "optimizer[0].exp_avg_sq",
        "optimizer[0].max_exp_avg_sq",
        "optimizer[0].compensation",
        "optimizer[0].momentum_buffer",
        "optimizer[0].custom_a",
        "optimizer[0].custom_z",
    ]
    assert [tensor.data_ptr() for _descriptor, tensor in items] == [
        state[state_name].data_ptr()
        for state_name in (
            "exp_avg",
            "exp_avg_sq",
            "max_exp_avg_sq",
            "compensation",
            "momentum_buffer",
            "custom_a",
            "custom_z",
        )
    ]


def test_pending_layer_swap_waits_for_p2p_before_publish(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
    )
    events = []

    class _Work:
        def wait(self):
            events.append("work_wait")

    class _Event:
        def record(self, _stream):
            events.append("event_record")

    class _CurrentStream:
        def wait_event(self, _event):
            events.append("main_wait")

    class _DeviceApi:
        @staticmethod
        @contextmanager
        def stream(_stream):
            events.append("stream_enter")
            yield
            events.append("stream_exit")

        @staticmethod
        def Event():
            return _Event()

        @staticmethod
        def current_stream(_device=None):
            return _CurrentStream()

    timing_context = nullcontext()
    timing_context.__enter__()
    manager._pending_layer_swaps["layers.0.mlp.experts"] = expert_swap_module._PendingLayerSwap(
        layer_key="layers.0.mlp.experts",
        works=(_Work(),),
        unpack=(),
        device=torch.device("cpu"),
        timing_context=timing_context,
    )
    monkeypatch.setattr(expert_swap_module, "get_torch_device", lambda: _DeviceApi())
    monkeypatch.setattr(manager, "_swap_comm_stream", lambda _device: object())
    monkeypatch.setattr(manager, "_publish_swap_wave", lambda _unpack: events.append("publish"))

    manager.wait_pending_layer_swap("layers.0.mlp.experts")

    assert events == ["stream_enter", "work_wait", "publish", "event_record", "stream_exit", "main_wait"]


def test_placement_transaction_skips_status_collectives_by_default(monkeypatch):
    tensor = torch.tensor([[1.0], [2.0]])
    entry = expert_swap_module._CoverTensorEntry(tensor=tensor, src_slot=0, dst_slot=1)
    monkeypatch.setattr(
        expert_swap_module,
        "_placement_group_succeeded",
        lambda *_args, **_kwargs: pytest.fail("unexpected placement status collective"),
    )

    expert_swap_module._cover_grouped_slot_entries_atomic({(0, 0): [entry]}, 0, 2, object())

    torch.testing.assert_close(tensor, torch.tensor([[1.0], [1.0]]))


def _expert_swap_multi_pair_same_layer_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=2,
            hierarchy_group_sizes=[world_size],
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 2
    local_experts = num_experts // world_size
    local_start = rank * local_experts
    module = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    state.expert_swap_manager.register_layer("layers.0.mlp.experts", module)
    optimizer = torch.optim.AdamW([module.gate_up_proj, module.down_proj], lr=0.01)
    state.expert_swap_manager.bind_optimizer(optimizer)
    for param in (module.gate_up_proj, module.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    module.reset_values(local_start)
    for param in (module.gate_up_proj, module.down_proj):
        optimizer.state[param]["exp_avg"].copy_(param.detach() + 1000)

    original_all_to_all_single = dist.all_to_all_single
    original_batch_isend_irecv = dist.batch_isend_irecv
    p2p_call_count = 0

    def _fail_all_to_all_single(*args, **kwargs):
        raise AssertionError("Expert Swap must not use all_to_all_single for expert migration")

    def _count_batch_isend_irecv(ops):
        nonlocal p2p_call_count
        p2p_call_count += 1
        return original_batch_isend_irecv(ops)

    dist.all_to_all_single = _fail_all_to_all_single
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        _execute_test_swap_pairs(
            state.expert_swap_manager,
            {"layers.0.mlp.experts": [(0, 2), (1, 3)]},
        )
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv

    assert p2p_call_count == 1
    assert state.expert_swap_manager.layers["layers.0.mlp.experts"].logical_to_physical.tolist() == [2, 3, 0, 1]
    expected_gate_values = {0: [3.0, 4.0], 1: [1.0, 2.0]}[rank]
    expected_down_values = {0: [103.0, 104.0], 1: [101.0, 102.0]}[rank]
    torch.testing.assert_close(module.gate_up_proj[:, 0, 0], torch.tensor(expected_gate_values))
    torch.testing.assert_close(module.down_proj[:, 0, 0], torch.tensor(expected_down_values))
    torch.testing.assert_close(
        optimizer.state[module.gate_up_proj]["exp_avg"][:, 0, 0],
        torch.tensor([value + 1000 for value in expected_gate_values]),
    )


def test_expert_swap_batches_multiple_pairs_in_one_p2p_wave():
    torchrun(_expert_swap_multi_pair_same_layer_worker, world_size=2, backend="gloo")


def _expert_swap_multi_peer_p2p_worker():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_hiermoe(
        _profiled_hiermoe_config(
            enable=True,
            token_dedup=True,
            expert_swap=True,
            expert_swap_interval=1,
            hierarchy_group_sizes=[2, world_size],
        ),
        dist.group.WORLD,
        ep_fsdp_size=1,
    )
    state = get_hiermoe_state()
    assert state is not None and state.expert_swap_manager is not None

    num_experts = 4
    hidden_size = 2
    local_experts = num_experts // world_size
    local_start = rank * local_experts
    module0 = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    module1 = _FakeExperts(num_experts, local_start, local_experts, hidden_size)
    state.expert_swap_manager.register_layer("layers.0.mlp.experts", module0)
    state.expert_swap_manager.register_layer("layers.1.mlp.experts", module1)
    optimizer = torch.optim.AdamW(
        [module0.gate_up_proj, module0.down_proj, module1.gate_up_proj, module1.down_proj],
        lr=0.01,
    )
    state.expert_swap_manager.bind_optimizer(optimizer)
    for param in (module0.gate_up_proj, module0.down_proj, module1.gate_up_proj, module1.down_proj):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()

    original_all_to_all_single = dist.all_to_all_single
    original_batch_isend_irecv = dist.batch_isend_irecv
    p2p_call_count = 0

    def _fail_all_to_all_single(*args, **kwargs):
        raise AssertionError("Expert Swap must not use all_to_all_single for expert migration")

    def _count_batch_isend_irecv(ops):
        nonlocal p2p_call_count
        p2p_call_count += 1
        return original_batch_isend_irecv(ops)

    dist.all_to_all_single = _fail_all_to_all_single
    dist.batch_isend_irecv = _count_batch_isend_irecv
    try:
        _execute_test_swap_pairs(
            state.expert_swap_manager,
            {
                "layers.0.mlp.experts": [(0, 1)],
                "layers.1.mlp.experts": [(0, 2)],
            },
        )
    finally:
        dist.all_to_all_single = original_all_to_all_single
        dist.batch_isend_irecv = original_batch_isend_irecv

    gathered_counts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_counts, p2p_call_count)
    if rank == 0:
        assert gathered_counts == [1, 1, 1, 0]
    assert state.expert_swap_manager.layers["layers.0.mlp.experts"].logical_to_physical.tolist() == [1, 0, 2, 3]
    assert state.expert_swap_manager.layers["layers.1.mlp.experts"].logical_to_physical.tolist() == [2, 1, 0, 3]


def test_expert_swap_batches_multiple_peers_in_one_p2p_wave():
    torchrun(_expert_swap_multi_peer_p2p_worker, world_size=4, backend="gloo")


def test_expert_swap_checkpoint_state_requires_active_manager_for_non_identity():
    configure_hiermoe(_profiled_hiermoe_config(enable=False), None)
    load_hiermoe_state_dict(
        {
            "version": 1,
            "ep_size": 2,
            "layers": {
                "layers.0.mlp.experts": {
                    "num_experts": 4,
                    "logical_to_physical": [0, 1, 2, 3],
                }
            },
        }
    )
    with pytest.raises(RuntimeError):
        load_hiermoe_state_dict(
            {
                "version": 1,
                "ep_size": 2,
                "layers": {
                    "layers.0.mlp.experts": {
                        "num_experts": 4,
                        "logical_to_physical": [3, 1, 2, 0],
                    }
                },
            }
        )


def test_expert_swap_trainable_only_checkpoint_rejects_non_identity_placement():
    identity_state = {
        "version": 1,
        "ep_size": 2,
        "layers": {
            "layers.0.mlp.experts": {
                "num_experts": 4,
                "logical_to_physical": [0, 1, 2, 3],
            }
        },
    }
    assert_hiermoe_trainable_only_checkpoint_safe(True, identity_state, "load")

    non_identity_state = {
        "version": 1,
        "ep_size": 2,
        "layers": {
            "layers.0.mlp.experts": {
                "num_experts": 4,
                "logical_to_physical": [3, 1, 2, 0],
            }
        },
    }
    with pytest.raises(RuntimeError, match="trainable-only"):
        assert_hiermoe_trainable_only_checkpoint_safe(True, non_identity_state, "load")


def _expert_swap_failfast_worker():
    with pytest.raises(NotImplementedError):
        configure_hiermoe(_profiled_hiermoe_config(enable=True, expert_swap=True), dist.group.WORLD, ep_fsdp_size=2)


def test_expert_swap_failfast_when_ep_fsdp_sharded():
    torchrun(_expert_swap_failfast_worker, world_size=2, backend="gloo")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expert_swap_max_pairs_per_layer": 2}, "requires expert_swap_max_pairs_per_layer=1"),
        ({"redundant_slot_increment_per_device": 1}, "does not support redundant slots"),
    ],
)
def test_hiermoe_exact_p1_rejects_incompatible_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HierMoEConfig(expert_swap_selector="hiermoe_exact_p1", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"redundant_slot_increment_per_device": 1}, "does not support redundant slots"),
        ({"expert_swap_mode": "layer"}, "requires expert_swap_mode=step"),
    ],
)
def test_legacy_batched_rejects_incompatible_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HierMoEConfig(expert_swap_selector="legacy_batched", **kwargs)


def test_maybe_swap_dispatches_to_legacy_batched_without_current_calibration(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=4,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    calls = []

    def _legacy(layers):
        calls.append(layers)
        return ["legacy"]

    monkeypatch.setattr(manager, "_plan_legacy_batched_layers", _legacy)
    monkeypatch.setattr(
        manager,
        "prepare_calibrations",
        lambda _step: (_ for _ in ()).throw(AssertionError("legacy_batched must not calibrate current_joint")),
    )

    assert manager.maybe_swap(0) == "none"
    assert manager.maybe_swap(1) == "legacy"
    assert calls == [[]]


def test_legacy_batched_zero_pairs_is_a_noop(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    monkeypatch.setattr(
        manager,
        "_plan_legacy_batched_layers",
        lambda _layers: (_ for _ in ()).throw(AssertionError("zero pairs must disable legacy swaps")),
    )

    assert LegacyBatchedSelector(manager).select([]) == {}
    assert manager.maybe_swap(1) == "none"


def test_legacy_batched_full_candidate_shards_include_empty_route_layers(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(4,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    for layer_idx in range(2):
        key = f"layers.{layer_idx}.mlp.experts"
        manager.register_layer(
            key,
            _FakeExperts(num_experts=128, local_start=0, local_experts=32, hidden_size=2),
        )
        layer = manager.layers[key]
        layer.latest_selected_experts = (
            torch.empty((0, 4), dtype=torch.long)
            if layer_idx == 0
            else torch.tensor([[0, 64, 1, 65]], dtype=torch.long)
        )

    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("full candidate generation must not reduce route counts"),
    )

    pairs = LegacyBatchedSelector(manager)._candidate_pairs_by_layer(list(manager.layers.values()))

    assert len(pairs) == 2
    assert all(layer_pairs is not None and layer_pairs.shape == (1536, 2) for layer_pairs in pairs)
    full_pairs = torch.triu_indices(128, 128, offset=1).t().contiguous()
    owner_ranks = torch.div(torch.arange(128), 32, rounding_mode="floor")
    cross_rank = owner_ranks.index_select(0, full_pairs[:, 0]) != owner_ranks.index_select(0, full_pairs[:, 1])
    expected = full_pairs[cross_rank][::4]
    torch.testing.assert_close(pairs[0].cpu(), expected)
    torch.testing.assert_close(pairs[1].cpu(), expected)


def test_legacy_batched_p1_reduces_all_layers_in_batched_collectives(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    for layer_idx in range(3):
        key = f"layers.{layer_idx}.mlp.experts"
        manager.register_layer(key, _FakeExperts(num_experts=8, local_start=0, local_experts=2, hidden_size=4))
        layer = manager.layers[key]
        layer.latest_selected_experts = torch.tensor(
            [[layer_idx, 4], [1, 5], [2, 6], [3, 7]],
            dtype=torch.long,
        )
        layer.latest_hidden_size = 4
        layer.latest_bytes_per_element = 2

    collective_shapes = []

    def _record_all_reduce(tensor, op, group):
        collective_shapes.append(tuple(tensor.shape))

    monkeypatch.setattr(dist, "all_reduce", _record_all_reduce)

    LegacyBatchedSelector(manager).select(list(manager.layers.values()))

    assert len(collective_shapes) == 2
    assert all(shape[0] == 3 for shape in collective_shapes)


def test_legacy_batched_p4_fast_selector_selects_disjoint_pairs(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=4,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(
        key,
        _FakeExperts(num_experts=8, local_start=0, local_experts=2, hidden_size=4),
    )
    layer = manager.layers[key]
    layer.latest_selected_experts = torch.randint(
        0,
        8,
        (128, 4),
        generator=torch.Generator().manual_seed(20260721),
        dtype=torch.long,
    )
    layer.latest_hidden_size = 4
    layer.latest_bytes_per_element = 2

    monkeypatch.setattr(
        LegacyBatchedSelector,
        "_select_global_pair_lists_from_gathered_routes",
        None,
    )

    pairs = LegacyBatchedSelector(manager).select([layer]).get(key, [])

    flattened = [expert for pair in pairs for expert in pair]
    assert 0 < len(pairs) <= 4
    assert len(flattened) == len(set(flattened))


def test_legacy_batched_plan_rejects_same_rank_pairs(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=4,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(
        key,
        _FakeExperts(num_experts=4, local_start=0, local_experts=4, hidden_size=2),
    )
    monkeypatch.setattr(
        LegacyBatchedSelector,
        "select",
        lambda _selector, _layers: {key: [(0, 1), (2, 3)]},
    )
    executed_plans = []
    monkeypatch.setattr(manager, "_execute_swap_plans", lambda plans: executed_plans.extend(plans))

    with pytest.raises(RuntimeError, match="same-rank swap"):
        manager._plan_legacy_batched_layers([manager.layers[key]])

    assert executed_plans == []
    assert manager.layers[key].logical_to_physical.tolist() == [0, 1, 2, 3]


def test_legacy_batched_plan_batches_optimizer_state_validation(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=4,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="legacy_batched",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(
        key,
        _FakeExperts(num_experts=8, local_start=0, local_experts=2, hidden_size=2),
    )
    monkeypatch.setattr(
        LegacyBatchedSelector,
        "select",
        lambda _selector, _layers: {key: [(0, 2), (1, 3)]},
    )
    optimizer_state_collectives = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *_args, **_kwargs: optimizer_state_collectives.append(True),
    )
    monkeypatch.setattr(manager, "_execute_swap_plans", lambda _plans: None)

    manager._plan_legacy_batched_layers([manager.layers[key]])

    assert optimizer_state_collectives == []

    manager.debug_validate = True
    manager._plan_legacy_batched_layers([manager.layers[key]])

    assert len(optimizer_state_collectives) == 1


def test_exact_single_swap_stats_preserve_third_expert_group_hits():
    selected_experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    placement = torch.arange(4, dtype=torch.long)
    group_by_logical = torch.div(placement, 2, rounding_mode="floor")
    pairs = torch.tensor([[0, 2]], dtype=torch.long)

    token_expert_hits = expert_swap_module._token_expert_hit_matrix(selected_experts, 4)
    exact_stats = expert_swap_module._exact_single_swap_group_stats(
        token_expert_hits,
        group_by_logical,
        num_groups=2,
    )
    exact_max = expert_swap_module._exact_candidate_group_max_from_stats(
        stats=exact_stats,
        pairs=pairs,
        group_by_logical=group_by_logical,
    )

    swapped = placement.clone()
    swapped[0], swapped[2] = swapped[2].clone(), swapped[0].clone()
    physical_experts = swapped.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
    brute_force_max = expert_swap_module._duplicate_free_counts_by_expert_group_batched(
        physical_experts,
        num_experts=4,
        group_size=2,
    ).max()

    assert float(exact_max.item()) == 2.0
    assert float(exact_max.item()) == float(brute_force_max.item())


def test_exact_single_swap_costs_match_brute_force_for_topk8_hierarchy():
    generator = torch.Generator().manual_seed(20260719)
    num_experts = 16
    selected_experts = torch.randint(num_experts, (48, 8), generator=generator, dtype=torch.long)
    placement = torch.randperm(num_experts, generator=generator)
    hierarchy = Hierarchy(ep_size=8, group_sizes=(2, 4, 8), source="test-exact-3d")
    perf_model = HierMoEPerfModel(
        a2a=LinkCost(alpha=1.0, beta=0.7),
        inter=(LinkCost(alpha=0.8, beta=0.5), LinkCost(alpha=0.6, beta=0.3)),
        intra=LinkCost(alpha=0.2, beta=0.1),
        source="test",
    )
    pairs = expert_swap_module._all_candidate_pairs(num_experts, selected_experts.device)
    token_expert_hits = expert_swap_module._token_expert_hit_matrix(selected_experts, num_experts)

    num_local_experts = num_experts // hierarchy.ep_size
    group_by_logical = [torch.div(placement, num_local_experts, rounding_mode="floor")]
    num_groups = [hierarchy.ep_size]
    for u_i, level_num_groups in expert_swap_module._hierarchy_level_group_shapes(hierarchy, num_experts):
        expert_group_size = num_experts // (hierarchy.ep_size // u_i)
        group_by_logical.append(torch.div(placement, expert_group_size, rounding_mode="floor"))
        num_groups.append(level_num_groups)

    group_stats = [
        expert_swap_module._exact_single_swap_group_stats(token_expert_hits, mapping, level_num_groups)
        for mapping, level_num_groups in zip(group_by_logical, num_groups, strict=True)
    ]
    current_cost, candidate_costs = expert_swap_module._exact_single_swap_costs_from_group_stats(
        group_stats=group_stats,
        group_by_logical=group_by_logical,
        pairs=pairs,
        num_experts=num_experts,
        hidden_size=8,
        bytes_per_element=2,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=10.0,
    )

    expected_current = expert_swap_module._estimate_mapping_cost(
        selected_experts,
        placement,
        num_experts,
        8,
        2,
        hierarchy,
        perf_model,
        10.0,
    )
    expected_costs = []
    for lhs, rhs in pairs.tolist():
        swapped = placement.clone()
        swapped[lhs], swapped[rhs] = swapped[rhs].clone(), swapped[lhs].clone()
        expected_costs.append(
            expert_swap_module._estimate_mapping_cost(
                selected_experts,
                swapped,
                num_experts,
                8,
                2,
                hierarchy,
                perf_model,
                10.0,
            )
        )

    assert float(current_cost.item()) == pytest.approx(expected_current)
    torch.testing.assert_close(candidate_costs.cpu(), torch.tensor(expected_costs, dtype=torch.float32))


def test_maybe_swap_dispatches_to_exact_p1_without_current_calibration(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=1,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=1, group_sizes=(1,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="hiermoe_exact_p1",
    )
    calls = []

    def _exact(layers, step):
        calls.append((layers, step))
        return [f"exact(step={step})"]

    monkeypatch.setattr(manager, "_plan_exact_single_swap_layers", _exact)
    monkeypatch.setattr(
        manager,
        "prepare_calibrations",
        lambda _step: (_ for _ in ()).throw(AssertionError("exact P1 must not calibrate current_joint")),
    )

    assert manager.maybe_swap(0) == "none"
    assert manager.maybe_swap(1) == "exact(step=1)"
    assert calls == [([], 1)]


def test_exact_p1_direct_executor_bypasses_generic_placement(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="hiermoe_exact_p1",
        expert_swap_mode="layer",
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeAsymmetricExperts(num_experts=4, local_start=0, local_experts=2, hidden_size=2))
    layer = manager.layers[key]
    captured_plans = []

    monkeypatch.setattr(
        manager,
        "_execute_placement_plan",
        lambda *_args, **_kwargs: pytest.fail("exact P1 must not use the generic placement executor"),
    )
    monkeypatch.setattr(
        manager,
        "_record_plan_metrics",
        lambda *_args, **_kwargs: pytest.fail("exact P1 must not record a generic PlacementPlan"),
    )
    monkeypatch.setattr(
        manager,
        "_layer_layout",
        lambda *_args, **_kwargs: pytest.fail("production exact P1 must not scan the full layout"),
    )
    monkeypatch.setattr(
        manager,
        "_execute_sparse_group_swap_plans",
        lambda plans: captured_plans.extend(plans),
        raising=False,
    )

    committed = manager._execute_exact_single_swap(layer, (0, 3), timing_prefix=None)

    assert committed == f"{key}:swap(0<->3)"
    assert len(captured_plans) == 1
    assert captured_plans[0].logical_lhs == 0
    assert captured_plans[0].logical_rhs == 3
    assert layer.logical_to_physical.tolist() == [3, 1, 2, 0]
    assert layer.last_plan is None

    committed = manager._execute_exact_single_swap(layer, (0, 3), timing_prefix=None)

    assert committed == f"{key}:swap(3<->0)"
    assert len(captured_plans) == 2
    assert layer.logical_to_physical.tolist() == list(range(4))
    assert layer.is_identity


def test_exact_p1_direct_executor_validates_full_layout_only_in_debug(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="hiermoe_exact_p1",
        expert_swap_mode="layer",
        debug_validate=True,
    )
    key = "layers.0.mlp.experts"
    manager.register_layer(key, _FakeExperts(num_experts=4, local_start=0, local_experts=2, hidden_size=2))
    layer = manager.layers[key]
    layer.logical_to_physical = torch.tensor([0, 0, 2, 3], dtype=torch.long)
    monkeypatch.setattr(
        manager,
        "_execute_sparse_group_swap_plans",
        lambda *_args, **_kwargs: pytest.fail("invalid debug layout must fail before transfer"),
    )

    with pytest.raises(RuntimeError, match="invalid compact layout"):
        manager._execute_exact_single_swap(layer, (0, 3), timing_prefix=None)


def test_exact_p1_batches_layer_statistics_into_one_collective(monkeypatch):
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=object(),
        ep_size=4,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=1,
        redundant_slot_increment_per_device=0,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_selector="hiermoe_exact_p1",
    )
    for layer_idx in range(3):
        key = f"layers.{layer_idx}.mlp.experts"
        manager.register_layer(key, _FakeExperts(num_experts=8, local_start=0, local_experts=2, hidden_size=4))
        layer = manager.layers[key]
        layer.latest_selected_experts = torch.tensor(
            [[layer_idx, 4], [1, 5], [2, 6], [3, 7]],
            dtype=torch.long,
        )
        layer.latest_hidden_size = 4
        layer.latest_bytes_per_element = 2

    collective_shapes = []

    def _record_all_reduce(tensor, op, group):
        collective_shapes.append(tuple(tensor.shape))

    direct_calls = []

    monkeypatch.setattr(dist, "all_reduce", _record_all_reduce)
    monkeypatch.setattr(
        expert_swap_module,
        "PlacementPlan",
        lambda *_args, **_kwargs: pytest.fail("exact P1 must not construct a generic PlacementPlan"),
    )
    monkeypatch.setattr(
        manager,
        "_execute_placement_plan",
        lambda *_args, **_kwargs: pytest.fail("exact P1 must not use the generic placement executor"),
    )
    monkeypatch.setattr(
        manager,
        "_execute_exact_single_swap",
        lambda layer, pair, **_kwargs: direct_calls.append((layer.key, pair)),
        raising=False,
    )

    manager._plan_exact_single_swap_layers(list(manager.layers.values()), step=1)

    assert len(collective_shapes) == 1
    assert collective_shapes[0][0] == 3
    assert len(direct_calls) == 3
    assert manager.placement_metrics()["hiermoe/exact_p1_candidate_count"] == 3 * 24
