import json

import torch
import torch.distributed as dist
from torch import nn

from tests.tools.launch_utils import torchrun
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


class _FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.arange(6, dtype=torch.float32).view(3, 2, 1))
        self.down_proj = nn.Parameter(torch.arange(6, dtype=torch.float32).view(3, 1, 2))


def _manager(monkeypatch, replay_path, *, migration_mode="blocking", grad_mode="blocking"):
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_PATH", str(replay_path))
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_MODE", "step")
    monkeypatch.setattr(expert_swap_module, "_ABLATION_MIGRATION_MODE", migration_mode)
    monkeypatch.setattr(expert_swap_module, "_ABLATION_GRAD_MODE", grad_mode)
    return expert_swap_module.ExpertSwapManager(
        ep_group=None,
        ep_size=2,
        ep_rank=0,
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=1,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
        fixed_pipeline_overlap=True,
    )


def test_ablation_replay_queues_exact_action_and_blocking_migration(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "replica", "body": "2->2"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [0, 1, 2, 2, 3, -1],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    module = _FakeExperts()
    manager.register_layer(key, module)

    assert manager.maybe_swap(0) == f"{key}:replica(2->2)"
    assert tuple(manager._pipeline_pending_plans) == (key,)
    pending = manager._pipeline_pending_plans[key]
    assert pending.plan.actions[0].src_slot == 3
    assert pending.plan.actions[0].dst_slot == 2
    assert manager.placement_metrics()["hiermoe/ablation_replay_logged_step"] == 1

    def execute(layer, plan, **_kwargs):
        layer.slot_to_logical = torch.tensor(plan.final_layout, dtype=torch.long)
        manager._refresh_layer_mapping_from_slots(layer, plan.final_owner_slots)
        layer.placement_version += 1
        return [f"{layer.key}:{plan.actions[0].format()}"]

    monkeypatch.setattr(manager, "_execute_placement_plan", execute)
    manager.configure_pipeline_microstep(step=1, micro_step=0, num_micro_steps=1)
    manager.wait_pipeline_migration_before_layer(key)

    assert manager.layers[key].slot_to_logical.tolist() == [0, 1, 2, 2, 3, -1]
    assert manager._pipeline_pending_plans == {}
    metrics = manager.placement_metrics()
    assert metrics["hiermoe/pipeline_migration_jobs"] == 1
    assert metrics["hiermoe/pipeline_migration_raw_ms"] == metrics["hiermoe/pipeline_migration_exposed_ms"]
    assert metrics["hiermoe/pipeline_migration_hidden_ratio"] == 0.0
    manager.shutdown_pipeline()


def test_ablation_replay_builds_swap_from_current_owner_slots(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "swap", "body": "0<->1"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [1, 0, -1, 2, 3, -1],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]

    plan = manager._build_ablation_replay_plan(layer, (("swap", "0<->1"),))

    assert plan.actions[0].src_slot == 0
    assert plan.actions[0].dst_slot == 1
    assert plan.final_layout == (1, 0, -1, 2, 3, -1)
    assert plan.final_owner_slots == (1, 0, 3, 4)
    manager.shutdown_pipeline()


def _blocking_gradient_sync_worker():
    expert_swap_module._ABLATION_REPLAY_MODE = "off"
    expert_swap_module._ABLATION_GRAD_MODE = "blocking"
    manager = expert_swap_module.ExpertSwapManager(
        ep_group=dist.group.WORLD,
        ep_size=2,
        ep_rank=dist.get_rank(),
        expert_swap_interval=1,
        expert_swap_max_pairs_per_layer=0,
        redundant_slot_increment_per_device=2,
        max_replica_rounds=0,
        smooth_max_gamma=10.0,
        hierarchy=Hierarchy(ep_size=2, group_sizes=(2,), source="test"),
        perf_model=HierMoEPerfModel.default(),
        expert_swap_mode="step",
        expert_swap_selector="hiermoe_greedy_cover_p1",
        fixed_pipeline_overlap=True,
    )
    key = "layers.0.mlp.experts"
    module = nn.Module()
    module.num_experts = 4
    module.gate_up_proj = nn.Parameter(torch.ones((4, 2, 1)))
    module.down_proj = nn.Parameter(torch.ones((4, 1, 2)))
    manager.register_layer(key, module)
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 6, 7))

    manager.configure_pipeline_microstep(step=0, micro_step=0, num_micro_steps=1)
    (module.gate_up_proj.sum() + module.down_proj.sum()).backward()
    manager.sync_redundant_gradients()

    torch.testing.assert_close(module.gate_up_proj.grad, torch.full_like(module.gate_up_proj.grad, 2.0))
    torch.testing.assert_close(module.down_proj.grad, torch.full_like(module.down_proj.grad, 2.0))
    metrics = manager.placement_metrics()
    assert metrics["hiermoe/pipeline_grad_sync_jobs"] == 1
    assert metrics["hiermoe/pipeline_grad_sync_raw_ms"] == metrics["hiermoe/pipeline_grad_sync_exposed_ms"]
    assert metrics["hiermoe/pipeline_grad_sync_hidden_ratio"] == 0.0
    manager.shutdown_pipeline()


def test_ablation_blocking_gradient_sync_is_fully_exposed():
    torchrun(_blocking_gradient_sync_worker, world_size=2, backend="gloo")
