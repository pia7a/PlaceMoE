import json

import torch
import torch.distributed as dist
from torch import nn

from tests.tools.launch_utils import torchrun
from veomni.distributed.moe.hiermoe import expert_swap as expert_swap_module
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.distributed.parallel_plan import (
    _hiermoe_initial_layout_shard,
    _hiermoe_initial_layouts,
)


class _FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.arange(6, dtype=torch.float32).view(3, 2, 1))
        self.down_proj = nn.Parameter(torch.arange(6, dtype=torch.float32).view(3, 1, 2))


def test_initial_layout_shards_checkpoint_by_final_slot_layout(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "layout.json"
    replay_path.write_text(
        json.dumps(
            {
                "layers": {
                    key: {
                        "slot_to_logical": [3, 0, 1, 3, 2, 0],
                    }
                }
            }
        )
    )
    _hiermoe_initial_layouts.cache_clear()
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(replay_path))
    tensor = torch.arange(8, dtype=torch.float32).view(4, 2)

    rank0 = _hiermoe_initial_layout_shard(
        tensor,
        f"{key}.down_proj",
        (3, 2),
        para_rank=0,
    )
    rank1 = _hiermoe_initial_layout_shard(
        tensor,
        f"{key}.down_proj",
        (3, 2),
        para_rank=1,
    )

    torch.testing.assert_close(rank0, tensor.index_select(0, torch.tensor([3, 0, 1])))
    torch.testing.assert_close(rank1, tensor.index_select(0, torch.tensor([3, 2, 0])))


def test_initial_layout_resolves_checkpoint_wrapper_prefix(tmp_path, monkeypatch):
    layout_key = "model.language_model.layers.0.mlp.experts"
    replay_path = tmp_path / "wrapped_layout.json"
    replay_path.write_text(
        json.dumps(
            {
                "layers": {
                    layout_key: {
                        "slot_to_logical": [3, 0, 1, 3, 2, 0],
                    }
                }
            }
        )
    )
    _hiermoe_initial_layouts.cache_clear()
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(replay_path))
    tensor = torch.arange(8, dtype=torch.float32).view(4, 2)

    shard = _hiermoe_initial_layout_shard(
        tensor,
        "model.layers.0.mlp.experts.down_proj",
        (3, 2),
        para_rank=1,
    )

    torch.testing.assert_close(shard, tensor.index_select(0, torch.tensor([3, 2, 0])))


def test_initial_layout_zero_fills_inactive_slots(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "layout_with_empty.json"
    replay_path.write_text(
        json.dumps(
            {
                "layers": {
                    key: {
                        "slot_to_logical": [3, -1, 1, 2],
                    }
                }
            }
        )
    )
    _hiermoe_initial_layouts.cache_clear()
    monkeypatch.setenv("VEOMNI_HIERMOE_INITIAL_LAYOUT", str(replay_path))
    tensor = torch.arange(8, dtype=torch.float32).view(4, 2)

    shard = _hiermoe_initial_layout_shard(
        tensor,
        f"{key}.down_proj",
        (4, 2),
        para_rank=0,
    )

    torch.testing.assert_close(shard[0], tensor[3])
    torch.testing.assert_close(shard[1], torch.zeros_like(shard[1]))
    torch.testing.assert_close(shard[2:], tensor.index_select(0, torch.tensor([1, 2])))


def test_static_placement_loads_without_fixed_pipeline(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    layout_path = tmp_path / "static_layout.json"
    layout_path.write_text(
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
                        "slot_to_logical": [0, 1, 2, 2, 3, 0],
                        "owner_slots": [0, 1, 2, 4],
                        "source_logical_to_physical": [
                            [5, 1, 3, 4],
                            [0, 1, 2, 4],
                        ],
                    }
                },
            }
        )
    )
    monkeypatch.setattr(expert_swap_module, "_INITIAL_LAYOUT_PATH", str(layout_path))
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_PATH", "")
    monkeypatch.setattr(expert_swap_module, "_ABLATION_REPLAY_MODE", "off")
    monkeypatch.setattr(expert_swap_module, "_ABLATION_GRAD_MODE", "blocking")
    manager = expert_swap_module.ExpertSwapManager(
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
        fixed_pipeline_overlap=False,
    )
    manager.register_layer(key, _FakeExperts())
    manager._normalize_ablation_layer_keys()
    manager._install_static_ablation_layout()

    layer = manager.layers[key]
    assert tuple(layer.slot_to_logical.tolist()) == (0, 1, 2, 2, 3, 0)
    assert tuple(layer.logical_to_physical.tolist()) == (0, 1, 2, 4)
    assert tuple(tuple(row) for row in layer.source_logical_to_physical.tolist()) == (
        (5, 1, 3, 4),
        (0, 1, 2, 4),
    )
    assert manager._forward_reuse_cover_patch_remap is False
    torch.testing.assert_close(
        manager._map_logical_to_slot(layer, torch.tensor([[0, 2]])),
        torch.tensor([[5, 3]]),
    )
    assert manager.fixed_pipeline_overlap is False
    assert manager.placement_planning_enabled() is False


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


def test_ablation_replay_cover_promotes_remaining_victim_copy(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "replica", "body": "2->0"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [2, 1, 0, 2, 3, 2],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))

    plan = manager._build_ablation_replay_plan(layer, (("replica", "2->0"),))

    assert plan.actions[0].src_slot == 3
    assert plan.actions[0].dst_slot == 0
    assert plan.actions[0].dst_logical == 0
    assert plan.final_layout == (2, 1, 0, 2, 3, 2)
    assert plan.final_owner_slots == (2, 1, 3, 4)
    manager.shutdown_pipeline()


def test_ablation_replay_installs_owner_and_source_route_metadata(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "replica", "body": "2->5"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [1, 0, -1, 2, 3, 2],
                        "owner_slots": [1, 0, 3, 4],
                        "source_logical_to_physical": [
                            [1, 0, 3, 4],
                            [1, 0, 5, 4],
                        ],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([1, 0, -1, 2, 3, 2], dtype=torch.long)

    manager._install_static_ablation_route_metadata()
    manager._validate_ablation_final_layout()

    assert layer.logical_to_physical.tolist() == [1, 0, 3, 4]
    assert layer.source_logical_to_physical.tolist() == [
        [1, 0, 3, 4],
        [1, 0, 5, 4],
    ]
    manager.shutdown_pipeline()


def test_static_ablation_collapses_all_steps_into_one_plan_per_layer(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "replica", "body": "0->5"}],
                        "2": [{"layer": key, "kind": "replica", "body": "3->2"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [0, 1, 3, 2, 3, 0],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(key, _FakeExperts())
    layer = manager.layers[key]
    layer.slot_to_logical = torch.tensor([0, 1, 1, 2, 3, -1], dtype=torch.long)
    manager._refresh_layer_mapping_from_slots(layer, (0, 1, 3, 4))
    manager._ablation_initial_layout = "fixed_r2"
    monkeypatch.setattr(expert_swap_module, "_FIXED_R2_LAYOUT", True)
    monkeypatch.setattr(expert_swap_module, "synchronize", lambda: None)

    executed: list[tuple[str, int]] = []

    def execute(current_layer, plan, **_kwargs):
        executed.append((current_layer.key, len(plan.actions)))
        current_layer.slot_to_logical = torch.tensor(plan.final_layout, dtype=torch.long)
        manager._refresh_layer_mapping_from_slots(current_layer, plan.final_owner_slots)
        return [f"{current_layer.key}:{action.format()}" for action in plan.actions]

    monkeypatch.setattr(manager, "_execute_placement_plan", execute)
    manager._install_static_ablation_layout()

    assert executed == [(key, 2)]
    assert layer.slot_to_logical.tolist() == [0, 1, 3, 2, 3, 0]
    manager.shutdown_pipeline()


def test_initial_layout_installs_metadata_without_expert_transfer(tmp_path, monkeypatch):
    key = "layers.0.mlp.experts"
    replay_path = tmp_path / "preloaded.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": key, "kind": "replica", "body": "0->5"}],
                        "2": [{"layer": key, "kind": "replica", "body": "3->2"}],
                    }
                },
                "layers": {
                    key: {
                        "slot_to_logical": [0, 1, 3, 2, 3, 0],
                        "owner_slots": [0, 1, 3, 4],
                        "source_logical_to_physical": [
                            [0, 1, 3, 4],
                            [5, 1, 3, 4],
                        ],
                    }
                },
            }
        )
    )
    monkeypatch.setattr(expert_swap_module, "_INITIAL_LAYOUT_PATH", str(replay_path))
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(key, _FakeExperts())
    monkeypatch.setattr(
        manager,
        "_execute_placement_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected transfer")),
    )

    manager._install_static_ablation_layout()
    layer = manager.layers[key]

    assert layer.slot_to_logical.tolist() == [0, 1, 3, 2, 3, 0]
    assert layer.logical_to_physical.tolist() == [0, 1, 3, 4]
    assert layer.source_logical_to_physical.tolist() == [
        [0, 1, 3, 4],
        [5, 1, 3, 4],
    ]
    manager.shutdown_pipeline()


def test_ablation_replay_resolves_model_wrapper_prefix(tmp_path, monkeypatch):
    replay_key = "model.language_model.layers.0.mlp.experts"
    model_key = "model.layers.0.mlp.experts"
    replay_path = tmp_path / "wrapped_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "topology": {"ep_size": 2},
                "replay": {
                    "actions_by_step": {
                        "1": [{"layer": replay_key, "kind": "replica", "body": "0->5"}],
                    }
                },
                "layers": {
                    replay_key: {
                        "slot_to_logical": [0, 1, 3, 2, 3, 0],
                        "owner_slots": [0, 1, 3, 4],
                    }
                },
            }
        )
    )
    manager = _manager(monkeypatch, replay_path)
    manager.register_layer(model_key, _FakeExperts())

    manager._normalize_ablation_layer_keys()

    assert set(manager._ablation_expected_layouts) == {model_key}
    assert set(manager._ablation_expected_owner_slots) == {model_key}
    assert set(manager._ablation_actions_by_step[1]) == {model_key}
    manager.shutdown_pipeline()


def test_dedicated_gradient_group_does_not_wait_before_dispatch():
    manager = object.__new__(expert_swap_module.ExpertSwapManager)
    manager.fixed_pipeline_overlap = False
    manager.gradient_overlap_enabled = True
    manager._ablation_grad_mode = "hidden"
    manager._pipeline_micro_step = 1
    manager._pipeline_num_micro_steps = 2
    manager._owns_pipeline_grad_group = True

    manager.close_pipeline_gradient_window_before_dispatch("layers.0.mlp.experts")


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
