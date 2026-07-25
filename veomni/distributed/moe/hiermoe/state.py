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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from ....utils import logging
from .expert_swap import ExpertSwapManager, expand_redundant_expert_slots
from .metrics import peek_hiermoe_metrics
from .perf_model import HierMoEPerfModel, fit_perf_model_on_startup
from .topology import Hierarchy, infer_hierarchy


logger = logging.get_logger(__name__)


@dataclass
class HierMoEState:
    enable: bool
    token_dedup: bool
    expert_swap: bool
    expert_swap_interval: int
    expert_swap_max_pairs_per_layer: int
    redundant_slot_increment_per_device: int
    max_slot_op_search_rounds: int | None
    max_replica_rounds: int
    expert_swap_mode: str
    debug_validate: bool
    current_step: int
    log_interval: int
    hierarchy: Hierarchy
    perf_model: HierMoEPerfModel
    active: bool
    communication_mode: str = "hierarchical"
    planner_route_sample_size: int = 1024
    fixed_pipeline_overlap: bool = False
    greedy_max_copies_per_expert: int = 4
    expert_swap_selector: str = "current_joint"
    activation_checkpointing_enabled: bool = False
    placement_mapping_enabled: bool = True
    expert_swap_manager: ExpertSwapManager | None = None
    expert_swap_pair: str = "not_implemented"
    layer_swap_forward_enabled: bool = False
    route_capture_forward_enabled: bool = False


_STATE: HierMoEState | None = None


def _resolve_max_replica_rounds(configured: int | None, redundant_slots_per_rank: int, ep_size: int) -> int:
    capacity = max(0, int(redundant_slots_per_rank)) * max(0, int(ep_size))
    if configured is None:
        return capacity
    return min(max(0, int(configured)), capacity)


def configure_hiermoe(
    config: Any,
    ep_group: dist.ProcessGroup | None,
    ep_fsdp_size: int = 1,
    activation_checkpointing_enabled: bool = False,
    fsdp_offload_enabled: bool = False,
    gradient_bytes_per_element: int = 4,
) -> HierMoEState:
    global _STATE

    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
    configured_replica_rounds = config.max_slot_op_search_rounds
    max_replica_rounds = _resolve_max_replica_rounds(
        configured_replica_rounds,
        config.redundant_slot_increment_per_device,
        ep_size,
    )
    hierarchy = infer_hierarchy(
        ep_size=ep_size,
        topology=str(config.topology),
        hierarchy_group_sizes=tuple(config.hierarchy_group_sizes),
    )
    perf_model = HierMoEPerfModel.from_path(config.perf_model_path)
    active = bool(config.enable and config.token_dedup and ep_size > 1)
    placement_enabled = bool(
        config.expert_swap
        and (int(config.expert_swap_max_pairs_per_layer) > 0 or int(config.redundant_slot_increment_per_device) > 0)
    )
    placement_planning_enabled = bool(
        config.expert_swap and (int(config.expert_swap_max_pairs_per_layer) > 0 or max_replica_rounds > 0)
    )

    if config.enable and ep_size <= 1:
        logger.warning_rank0("HierMoE is enabled but EP size is <= 1; falling back to the original MoE path.")

    if config.enable and config.fit_perf_model_on_startup and ep_size > 1:
        if ep_group is None:
            raise RuntimeError("HierMoE startup performance fitting requires the EP process group.")
        perf_model = fit_perf_model_on_startup(
            perf_model,
            group=ep_group,
            local_world_size=hierarchy.local_world_size,
        )
        logger.info_rank0("HierMoE startup performance fitting completed: %s", perf_model.to_payload())

    if active and placement_enabled and int(ep_fsdp_size) != 1:
        raise NotImplementedError(
            "train.hiermoe.expert_swap=true currently requires ep_fsdp_size=1 so each EP rank owns complete "
            f"local experts; got ep_fsdp_size={ep_fsdp_size}. Set --train.hiermoe.expert_swap false or use "
            "DP_SHARD_SIZE == EP_SIZE for the current implementation."
        )

    if active and placement_enabled and fsdp_offload_enabled:
        raise NotImplementedError(
            "HierMoE placement is incompatible with FSDP2 CPU offload because planning and expert migration use "
            "the accelerator EP process group and move live parameter/optimizer payloads. Disable "
            "train.accelerator.fsdp_config.offload or disable HierMoE swap/replica placement."
        )

    if active and placement_planning_enabled and not perf_model.is_profiled:
        raise ValueError(
            "HierMoE placement planning requires profiled alpha/beta coefficients. Set "
            "--train.hiermoe.perf_model_path to JSON generated by "
            "profile/scripts/bench_hiermoe_perf_model.py, or enable "
            "--train.hiermoe.fit_perf_model_on_startup true. Default or unverified coefficients are not allowed."
        )

    expert_swap_manager = None
    if active and placement_enabled:
        expert_swap_manager = ExpertSwapManager(
            ep_group=ep_group,
            ep_size=ep_size,
            ep_rank=ep_rank,
            expert_swap_interval=int(config.expert_swap_interval),
            expert_swap_max_pairs_per_layer=int(config.expert_swap_max_pairs_per_layer),
            redundant_slot_increment_per_device=int(config.redundant_slot_increment_per_device),
            max_replica_rounds=max_replica_rounds,
            smooth_max_gamma=float(config.smooth_max_gamma),
            fixed_pipeline_overlap=bool(config.fixed_pipeline_overlap),
            hierarchy=hierarchy,
            perf_model=perf_model,
            expert_swap_mode=str(config.expert_swap_mode),
            expert_swap_selector=str(config.expert_swap_selector),
            activation_checkpointing_enabled=activation_checkpointing_enabled,
            gradient_bytes_per_element=gradient_bytes_per_element,
            configured_max_replica_rounds=configured_replica_rounds,
            replica_slot_capacity=int(config.redundant_slot_increment_per_device) * ep_size,
            planner_route_sample_size=int(config.planner_route_sample_size),
            greedy_max_copies_per_expert=int(config.greedy_max_copies_per_expert),
            debug_validate=bool(config.debug_validate),
        )

    _STATE = HierMoEState(
        enable=bool(config.enable),
        token_dedup=bool(config.token_dedup),
        expert_swap=placement_enabled,
        expert_swap_interval=int(config.expert_swap_interval),
        expert_swap_max_pairs_per_layer=int(config.expert_swap_max_pairs_per_layer),
        redundant_slot_increment_per_device=max(0, int(config.redundant_slot_increment_per_device)),
        max_slot_op_search_rounds=(
            None if configured_replica_rounds is None else max(0, int(configured_replica_rounds))
        ),
        max_replica_rounds=max_replica_rounds,
        expert_swap_mode=str(config.expert_swap_mode),
        debug_validate=bool(config.debug_validate),
        current_step=0,
        log_interval=max(1, int(config.log_interval)),
        hierarchy=hierarchy,
        perf_model=perf_model,
        fixed_pipeline_overlap=bool(config.fixed_pipeline_overlap),
        activation_checkpointing_enabled=bool(activation_checkpointing_enabled),
        active=active,
        communication_mode=str(config.communication_mode),
        planner_route_sample_size=int(config.planner_route_sample_size),
        greedy_max_copies_per_expert=int(config.greedy_max_copies_per_expert),
        expert_swap_selector=str(config.expert_swap_selector),
        expert_swap_manager=expert_swap_manager,
    )
    if config.enable:
        logger.info_rank0(
            "HierMoE configured: active=%s ep_size=%s hierarchy=%s communication_mode=%s "
            "perf_model_source=%s expert_swap=%s "
            "expert_swap_max_pairs_per_layer=%s expert_swap_mode=%s expert_swap_selector=%s "
            "fixed_pipeline_overlap=%s "
            "redundant_slot_increment_per_device=%s "
            "max_slot_op_search_rounds=%s replica_slot_capacity=%s max_replica_rounds=%s "
            "planner_route_sample_size=%s greedy_max_copies_per_expert=%s runtime_cost_model=%s",
            active,
            ep_size,
            hierarchy.group_sizes,
            config.communication_mode,
            perf_model.source,
            placement_enabled,
            config.expert_swap_max_pairs_per_layer,
            config.expert_swap_mode,
            config.expert_swap_selector,
            config.fixed_pipeline_overlap,
            config.redundant_slot_increment_per_device,
            config.max_slot_op_search_rounds,
            int(config.redundant_slot_increment_per_device) * ep_size,
            max_replica_rounds,
            config.planner_route_sample_size,
            config.greedy_max_copies_per_expert,
            perf_model.runtime_cost_status,
        )
        if placement_enabled and not perf_model.has_runtime_placement_costs:
            logger.warning_rank0(
                "HierMoE performance model %s does not contain state-migration and redundant-gradient-sync "
                "coefficients; CoRe-MoE placement will use explicitly marked fallback coefficients. Formal "
                "performance validation requires a schema-v2 model with both sections.",
                perf_model.source,
            )
    return _STATE


def get_hiermoe_state() -> HierMoEState | None:
    return _STATE


def set_hiermoe_step(step: int) -> None:
    if _STATE is not None:
        _STATE.current_step = max(0, int(step))


def set_hiermoe_layer_swap_forward_enabled(enabled: bool) -> bool:
    previous = bool(_STATE is not None and _STATE.layer_swap_forward_enabled)
    if _STATE is not None:
        _STATE.layer_swap_forward_enabled = bool(enabled)
    return previous


def set_hiermoe_route_capture_forward_enabled(enabled: bool) -> bool:
    previous = bool(_STATE is not None and getattr(_STATE, "route_capture_forward_enabled", False))
    if _STATE is not None:
        _STATE.route_capture_forward_enabled = bool(enabled)
    return previous


@contextmanager
def disable_hiermoe_placement():
    state = _STATE
    if state is None:
        yield
        return
    previous = state.placement_mapping_enabled
    state.placement_mapping_enabled = False
    try:
        yield
    finally:
        state.placement_mapping_enabled = previous


def hiermoe_active() -> bool:
    state = _STATE
    return bool(state is not None and state.active and state.current_step >= 0)


def bind_hiermoe_model(model: Any) -> None:
    state = _STATE
    if state is not None and state.placement_mapping_enabled and state.expert_swap_manager is not None:
        state.expert_swap_manager.register_model(model)


def maybe_expand_hiermoe_expert_slots(model: Any, ep_size: int) -> None:
    state = _STATE
    if (
        state is None
        or not state.placement_mapping_enabled
        or not state.active
        or not state.expert_swap
        or state.redundant_slot_increment_per_device <= 0
    ):
        return
    expanded = expand_redundant_expert_slots(
        model,
        ep_size=int(ep_size),
        redundant_slot_increment_per_device=state.redundant_slot_increment_per_device,
    )
    if expanded:
        logger.info_rank0(
            "HierMoE reserved %s redundant expert slot(s) per EP rank across %s layer(s).",
            state.redundant_slot_increment_per_device,
            expanded,
        )


def bind_hiermoe_optimizer(optimizer: Any) -> None:
    state = _STATE
    if state is not None and state.placement_mapping_enabled and state.expert_swap_manager is not None:
        state.expert_swap_manager.bind_optimizer(optimizer)


def sync_hiermoe_redundant_gradients() -> None:
    state = _STATE
    if state is not None and state.expert_swap_manager is not None:
        state.expert_swap_manager.sync_redundant_gradients()


def configure_hiermoe_pipeline_microstep(micro_step: int, num_micro_steps: int) -> None:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return
    state.expert_swap_manager.configure_pipeline_microstep(
        state.current_step,
        int(micro_step),
        int(num_micro_steps),
    )


def wait_hiermoe_pipeline_migration(layer_key: str) -> None:
    state = _STATE
    if state is not None and state.expert_swap_manager is not None:
        state.expert_swap_manager.wait_pipeline_migration_before_layer(layer_key)


def advance_hiermoe_pipeline_after_combine(layer_key: str) -> None:
    state = _STATE
    if state is not None and state.expert_swap_manager is not None:
        state.expert_swap_manager.advance_pipeline_after_combine(layer_key)


def shutdown_hiermoe_pipeline() -> None:
    state = _STATE
    if state is not None and state.expert_swap_manager is not None:
        state.expert_swap_manager.shutdown_pipeline()


def get_hiermoe_redundant_grad_norm_masks() -> dict[int, Any]:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return {}
    return state.expert_swap_manager.redundant_grad_norm_masks()


def get_hiermoe_expert_layer_key(module: Any) -> str | None:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return None
    return state.expert_swap_manager.get_layer_key(module)


def get_hiermoe_expert_layer_key_from_params(*params: Any) -> str | None:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return None
    return state.expert_swap_manager.get_layer_key_from_params(*params)


def maybe_run_hiermoe_expert_swap(step: int) -> str | None:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return None
    if state.expert_swap_mode == "layer":
        state.expert_swap_manager.prepare_calibrations(step)
        return state.expert_swap_pair
    state.expert_swap_pair = state.expert_swap_manager.maybe_swap(step)
    return state.expert_swap_pair


def maybe_log_hiermoe_metrics(step: int) -> None:
    state = _STATE
    if state is None or not state.active or int(step) % state.log_interval != 0:
        return

    metrics = peek_hiermoe_metrics()
    if state.expert_swap_manager is not None:
        metrics.update(state.expert_swap_manager.placement_metrics())
    if not metrics:
        return
    metrics["hiermoe/expert_swap_pair"] = state.expert_swap_pair

    formatted = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            formatted.append(f"{key}={value:.6g}")
        else:
            formatted.append(f"{key}={value}")
    logger.info_rank0("HierMoE metrics step=%s %s", step, " ".join(formatted))


def hiermoe_state_dict() -> dict[str, Any]:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return {}
    return state.expert_swap_manager.state_dict()


def hiermoe_has_non_identity_placement() -> bool:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        return False
    return state.expert_swap_manager.has_non_identity_placement()


def _state_dict_has_non_identity_placement(state_dict: dict[str, Any] | None) -> bool:
    if not state_dict:
        return False
    for payload in state_dict.get("layers", {}).values():
        if payload.get("slot_to_logical") is not None:
            slot_to_logical = list(payload.get("slot_to_logical", []))
            num_experts = int(payload.get("num_experts", 0))
            base_num_local_experts = int(payload.get("base_num_local_experts", 0))
            num_local_experts = int(payload.get("num_local_experts", 0))
            if num_experts <= 0 or base_num_local_experts <= 0 or num_local_experts <= 0:
                return True
            expected = [-1 for _ in slot_to_logical]
            for logical in range(num_experts):
                rank, local_slot = divmod(logical, base_num_local_experts)
                physical = rank * num_local_experts + local_slot
                if physical >= len(expected):
                    return True
                expected[physical] = logical
            if slot_to_logical != expected:
                return True
            continue
        mapping = payload.get("logical_to_physical", [])
        if list(mapping) != list(range(len(mapping))):
            return True
    return False


def assert_hiermoe_trainable_only_checkpoint_safe(
    trainable_only: bool,
    state_dict: dict[str, Any] | None = None,
    action: str = "save",
) -> None:
    if not trainable_only:
        return

    has_non_identity = (
        _state_dict_has_non_identity_placement(state_dict)
        if state_dict is not None
        else hiermoe_has_non_identity_placement()
    )
    if has_non_identity:
        raise RuntimeError(
            "HierMoE Expert Swap with trainable-only checkpoints is unsafe when expert placement is non-identity. "
            f"Refusing to {action} this checkpoint because frozen base expert weights are not included, so restoring "
            "only trainable parameters would mismatch logical expert placement and physical expert weights. Use a "
            "full DCP checkpoint, disable train.hiermoe.expert_swap, or resume with code that replays placement onto "
            "the base expert weights before loading the saved placement state."
        )


def assert_hiermoe_checkpoint_layout_compatible(state_dict: dict[str, Any] | None) -> None:
    state = _STATE
    manager = None if state is None else state.expert_swap_manager
    checkpoint_layers = {} if not state_dict else state_dict.get("layers", {})
    if not isinstance(checkpoint_layers, dict):
        raise RuntimeError("HierMoE checkpoint placement metadata has a non-mapping 'layers' payload.")
    if any(not isinstance(payload, dict) for payload in checkpoint_layers.values()):
        raise RuntimeError("HierMoE checkpoint placement metadata contains a non-mapping layer payload.")

    if manager is None:
        has_slot_layout = any(payload.get("slot_to_logical") is not None for payload in checkpoint_layers.values())
        if has_slot_layout or _state_dict_has_non_identity_placement(state_dict):
            raise RuntimeError(
                "Checkpoint contains active HierMoE placement or redundant expert slots, but the current run has "
                "no placement manager. Resume with the same HierMoE placement budgets used to save the checkpoint."
            )
        return

    if not state_dict:
        if any(layer.slot_to_logical is not None for layer in manager.layers.values()):
            raise RuntimeError(
                "Checkpoint has no HierMoE slot-layout metadata, but the current run reserves redundant expert "
                "slots. DCP model and optimizer tensor shapes require the same redundant-slot budget."
            )
        return

    checkpoint_ep_size = state_dict.get("ep_size")
    if checkpoint_ep_size is None or int(checkpoint_ep_size) != manager.ep_size:
        raise RuntimeError(
            "HierMoE checkpoint EP layout is incompatible with the current run: "
            f"checkpoint_ep_size={checkpoint_ep_size}, current_ep_size={manager.ep_size}."
        )

    current_keys = set(manager.layers)
    checkpoint_keys = set(checkpoint_layers)
    if checkpoint_keys != current_keys:
        missing = sorted(current_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - current_keys)
        raise RuntimeError(
            "HierMoE checkpoint layer set does not match the current model: "
            f"missing={missing}, unexpected={unexpected}."
        )

    for key, layer in manager.layers.items():
        checkpoint_layer = checkpoint_layers.get(key)
        assert checkpoint_layer is not None
        checkpoint_num_experts = int(checkpoint_layer.get("num_experts", -1))
        if checkpoint_num_experts != layer.num_experts:
            raise RuntimeError(
                f"HierMoE checkpoint expert count for {key} does not match the current model: "
                f"checkpoint_num_experts={checkpoint_num_experts}, current_num_experts={layer.num_experts}."
            )

        current_uses_slots = layer.slot_to_logical is not None
        checkpoint_uses_slots = checkpoint_layer.get("slot_to_logical") is not None
        if current_uses_slots != checkpoint_uses_slots:
            raise RuntimeError(
                f"HierMoE checkpoint layout for {key} is incompatible with the current redundant-slot budget: "
                f"checkpoint_uses_slots={checkpoint_uses_slots}, current_uses_slots={current_uses_slots}. "
                "DCP model and optimizer tensor shapes require identical placement budgets when saving and resuming."
            )
        if not current_uses_slots:
            mapping = list(checkpoint_layer.get("logical_to_physical", ()))
            if len(mapping) != layer.num_experts or sorted(mapping) != list(range(layer.num_experts)):
                raise RuntimeError(
                    f"HierMoE checkpoint compact placement for {key} is not a permutation of "
                    f"{layer.num_experts} physical experts."
                )
            continue

        checkpoint_base_num_local = int(checkpoint_layer.get("base_num_local_experts", -1))
        checkpoint_num_local = int(checkpoint_layer.get("num_local_experts", -1))
        checkpoint_layout = tuple(int(value) for value in checkpoint_layer.get("slot_to_logical", ()))
        if (
            checkpoint_base_num_local != layer.base_num_local_experts
            or checkpoint_num_local != layer.num_local_experts
            or len(checkpoint_layout) != layer.num_physical_slots
        ):
            raise RuntimeError(
                f"HierMoE checkpoint slot capacity for {key} does not match the current model: "
                f"checkpoint_base_num_local_experts={checkpoint_base_num_local}, "
                f"current_base_num_local_experts={layer.base_num_local_experts}, "
                f"checkpoint_num_local_experts={checkpoint_num_local}, "
                f"current_num_local_experts={layer.num_local_experts}."
            )
        manager._validate_placement_layout(layer, checkpoint_layout)

        expected_mapping = []
        for logical_expert in range(layer.num_experts):
            slots = [slot for slot, logical in enumerate(checkpoint_layout) if logical == logical_expert]
            canonical_slot = (
                None
                if layer.canonical_physical_slots is None
                else int(layer.canonical_physical_slots[logical_expert].item())
            )
            expected_mapping.append(canonical_slot if canonical_slot in slots else slots[0])
        checkpoint_mapping = list(checkpoint_layer.get("logical_to_physical", ()))
        if checkpoint_mapping != expected_mapping:
            raise RuntimeError(
                f"HierMoE checkpoint logical-to-physical mapping for {key} is inconsistent with its slot layout."
            )


def load_hiermoe_state_dict(state_dict: dict[str, Any] | None) -> None:
    state = _STATE
    if state is None or state.expert_swap_manager is None:
        if _state_dict_has_non_identity_placement(state_dict):
            raise RuntimeError(
                "Checkpoint contains non-identity HierMoE Expert Swap placement state, but "
                "train.hiermoe.expert_swap is not active. Loading it with identity expert placement would change "
                "logical expert semantics."
            )
        return
    state.expert_swap_manager.load_state_dict(state_dict)
