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

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_device_id, get_device_type
from ....utils.import_utils import is_torch_npu_available
from ..comm import all_to_all, all_to_all_pair
from ..timing import current_moe_timing_context, moe_timing_event, record_moe_timing_span
from .oracle import maybe_capture_route_snapshot, route_capture_enabled, route_capture_mode
from .state import get_hiermoe_state


@dataclass
class RankDedupDispatchContext:
    ep_group: dist.ProcessGroup | None
    ep_size: int
    ep_rank: int
    num_local_tokens: int
    num_local_experts: int
    hidden_size: int
    unique_send_splits: list[int]
    unique_recv_splits: list[int]
    assignment_send_splits: list[int]
    assignment_recv_splits: list[int]
    local_unique_token_indices: torch.Tensor
    recv_source_token_indices: torch.Tensor
    recv_assignment_weights: torch.Tensor
    unsort_indices: torch.Tensor
    selected_dim: int
    dedup_ratio_dispatch: float
    dedup_ratio_combine: float
    mode: str = "rank"
    stage1_group: dist.ProcessGroup | None = None
    stage2_group: dist.ProcessGroup | None = None
    recv_unique_indices: torch.Tensor | None = None
    stage1_unique_send_splits: list[int] | None = None
    stage1_unique_recv_splits: list[int] | None = None
    stage1_assignment_send_splits: list[int] | None = None
    stage1_assignment_recv_splits: list[int] | None = None
    stage2_unique_send_splits: list[int] | None = None
    stage2_unique_recv_splits: list[int] | None = None
    stage2_assignment_send_splits: list[int] | None = None
    stage2_assignment_recv_splits: list[int] | None = None
    stage2_send_stage1_unique_indices: torch.Tensor | None = None
    stage3_group: dist.ProcessGroup | None = None
    stage3_unique_send_splits: list[int] | None = None
    stage3_unique_recv_splits: list[int] | None = None
    stage3_send_stage2_unique_indices: torch.Tensor | None = None
    internal_timing_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None = None
    backward_internal_timing_events: dict[str, AcceleratorEvent] | None = None
    layer_key: str | None = None


@dataclass(frozen=True)
class HierarchicalProcessGroups:
    stage1_group: dist.ProcessGroup | None
    stage2_group: dist.ProcessGroup | None
    stage1_ep_ranks: tuple[int, ...]
    stage2_ep_ranks: tuple[int, ...]


@dataclass(frozen=True)
class Hierarchical3DProcessGroups:
    stage1_group: dist.ProcessGroup | None
    stage2_group: dist.ProcessGroup | None
    stage3_group: dist.ProcessGroup | None
    stage1_ep_ranks: tuple[int, ...]
    stage2_ep_ranks: tuple[int, ...]
    stage3_ep_ranks: tuple[int, ...]


@dataclass
class _PendingSplitSizeExchange:
    local: torch.Tensor | None
    exchanged: torch.Tensor | None
    work: object | None
    immediate: list[list[int]] | None

    def wait(self) -> list[list[int]]:
        if self.immediate is not None:
            return self.immediate
        if self.work is not None:
            self.work.wait()
        assert self.exchanged is not None
        return _split_matrix_columns_to_lists(self.exchanged)


_HIERARCHICAL_GROUP_CACHE: dict[tuple[tuple[int, ...], int, int], HierarchicalProcessGroups] = {}
_HIERARCHICAL_3D_GROUP_CACHE: dict[tuple[tuple[int, ...], int, int, int], Hierarchical3DProcessGroups] = {}


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.lower() in {"1", "true", "yes", "on", "y"}


_HIERMOE_INTERNAL_TIMING = _env_flag("VEOMNI_HIERMOE_INTERNAL_TIMING")


def configure_hiermoe_internal_timing(enabled: bool) -> None:
    """Enable accelerator events before the first training forward."""

    global _HIERMOE_INTERNAL_TIMING
    _HIERMOE_INTERNAL_TIMING = bool(enabled)


_BACKWARD_PREPARE_WINDOWS = {
    "backward_combine_stage1_a2a": 4,
    "backward_combine_stage2_a2a": 5,
}
_BACKWARD_SCORE_WINDOW = "backward_dispatch_stage1_a2a"


def _hiermoe_internal_event() -> AcceleratorEvent | None:
    return record_accelerator_event() if _HIERMOE_INTERNAL_TIMING else None


def _finish_hiermoe_internal_event(
    events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None,
    name: str,
    start: AcceleratorEvent | None,
) -> None:
    end = _hiermoe_internal_event()
    if events is not None and start is not None and end is not None:
        events[name] = (start, end)


class _BackwardA2AStart(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        events: dict[str, AcceleratorEvent] | None,
        key: str,
        layer_key: str | None,
    ) -> torch.Tensor:
        ctx.events = events
        ctx.key = key
        ctx.layer_key = layer_key
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        event = _hiermoe_internal_event()
        if ctx.events is not None and event is not None:
            ctx.events[f"{ctx.key}_start"] = event
        manager = _fixed_pipeline_manager(ctx.layer_key)
        if manager is not None:
            prepare_window = _BACKWARD_PREPARE_WINDOWS.get(ctx.key)
            if prepare_window is not None:
                manager.open_pipeline_planner_prepare_window(ctx.layer_key, prepare_window)
            elif ctx.key == _BACKWARD_SCORE_WINDOW:
                manager.open_pipeline_planner_score_window(ctx.layer_key)
        return grad, None, None, None


class _BackwardA2AEnd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        events: dict[str, AcceleratorEvent] | None,
        key: str,
        layer_key: str | None,
    ) -> torch.Tensor:
        ctx.events = events
        ctx.key = key
        ctx.layer_key = layer_key
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        event = _hiermoe_internal_event()
        if ctx.events is not None and event is not None:
            ctx.events[f"{ctx.key}_end"] = event
        manager = _fixed_pipeline_manager(ctx.layer_key)
        if manager is not None:
            prepare_window = _BACKWARD_PREPARE_WINDOWS.get(ctx.key)
            if prepare_window is not None:
                manager.close_pipeline_planner_prepare_window(ctx.layer_key, prepare_window)
            elif ctx.key == _BACKWARD_SCORE_WINDOW:
                manager.close_pipeline_planner_score_window(ctx.layer_key)
        return grad, None, None, None


def _mark_backward_a2a_input(
    tensor: torch.Tensor,
    events: dict[str, AcceleratorEvent] | None,
    key: str,
    layer_key: str | None = None,
) -> torch.Tensor:
    active = events is not None or _fixed_pipeline_manager(layer_key) is not None
    return _BackwardA2AEnd.apply(tensor, events, key, layer_key) if active and tensor.requires_grad else tensor


def _mark_backward_a2a_output(
    tensor: torch.Tensor,
    events: dict[str, AcceleratorEvent] | None,
    key: str,
    layer_key: str | None = None,
) -> torch.Tensor:
    active = events is not None or _fixed_pipeline_manager(layer_key) is not None
    return _BackwardA2AStart.apply(tensor, events, key, layer_key) if active and tensor.requires_grad else tensor


def _fixed_pipeline_manager(layer_key: str | None):
    if layer_key is None:
        return None
    state = get_hiermoe_state()
    if (
        state is None
        or not state.fixed_pipeline_overlap
        or state.expert_swap_manager is None
        or not state.expert_swap_manager.has_layer(layer_key)
    ):
        return None
    return state.expert_swap_manager


def _gradient_overlap_manager(layer_key: str | None):
    if layer_key is None:
        return None
    state = get_hiermoe_state()
    manager = None if state is None else state.expert_swap_manager
    if manager is None or not manager.gradient_overlap_enabled or not manager.has_layer(layer_key):
        return None
    return manager


class _BeforeExpertBackward(torch.autograd.Function):
    """Open the planner collective after combine backward and before expert GEMM."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, layer_key: str) -> torch.Tensor:
        ctx.layer_key = layer_key
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        manager = _fixed_pipeline_manager(ctx.layer_key)
        if manager is not None:
            manager.open_pipeline_planner_collective_window(ctx.layer_key)
        return grad, None


def _mark_fixed_pipeline_expert_output(expert_outputs: torch.Tensor, layer_key: str | None) -> torch.Tensor:
    if expert_outputs.requires_grad and _fixed_pipeline_manager(layer_key) is not None:
        return _BeforeExpertBackward.apply(expert_outputs, layer_key)
    return expert_outputs


class _BeforeDispatchBackward(torch.autograd.Function):
    """Close the background-gradient window before dispatch backward A2A."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, layer_key: str) -> torch.Tensor:
        ctx.layer_key = layer_key
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        planner_manager = _fixed_pipeline_manager(ctx.layer_key)
        if planner_manager is not None:
            planner_manager.close_pipeline_planner_collective_window(ctx.layer_key)
        gradient_manager = _gradient_overlap_manager(ctx.layer_key)
        if gradient_manager is not None:
            gradient_manager.close_pipeline_gradient_window_before_dispatch(ctx.layer_key)
        return grad, None


class _AfterDispatchBackward(torch.autograd.Function):
    """Open the layer gradient-sync window after dispatch backward A2A."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, layer_key: str) -> torch.Tensor:
        ctx.layer_key = layer_key
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        manager = _gradient_overlap_manager(ctx.layer_key)
        if manager is not None:
            manager.open_pipeline_gradient_window_after_dispatch(ctx.layer_key)
        return grad, None


def _mark_fixed_pipeline_dispatch_input(hidden_states: torch.Tensor, layer_key: str | None) -> torch.Tensor:
    if hidden_states.requires_grad and (
        _fixed_pipeline_manager(layer_key) is not None or _gradient_overlap_manager(layer_key) is not None
    ):
        return _AfterDispatchBackward.apply(hidden_states, layer_key)
    return hidden_states


def _mark_fixed_pipeline_dispatch_output(
    result: tuple[torch.Tensor, RankDedupDispatchContext, torch.Tensor],
    layer_key: str | None,
) -> tuple[torch.Tensor, RankDedupDispatchContext, torch.Tensor]:
    hidden_states, ctx, tokens_per_local_expert = result
    if hidden_states.requires_grad and (
        _fixed_pipeline_manager(layer_key) is not None or _gradient_overlap_manager(layer_key) is not None
    ):
        hidden_states = _BeforeDispatchBackward.apply(hidden_states, layer_key)
    return hidden_states, ctx, tokens_per_local_expert


def _begin_internal_span(section: str) -> tuple[dict, str, object] | None:
    if not _HIERMOE_INTERNAL_TIMING:
        return None
    meta = current_moe_timing_context()
    if meta is None:
        return None
    start_event = moe_timing_event()
    if start_event is None:
        return None
    return meta, section, start_event


def _end_internal_span(token: tuple[dict, str, object] | None) -> None:
    if token is None:
        return
    meta, section, start_event = token
    end_event = moe_timing_event()
    record_moe_timing_span(
        meta,
        direction="forward",
        component="hiermoe_internal",
        section=section,
        start_event=start_event,
        end_event=end_event,
    )


def _get_ep_global_ranks(group: dist.ProcessGroup | None, ep_size: int) -> tuple[int, ...]:
    if group is None or not dist.is_initialized():
        return tuple(range(ep_size))
    ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    if len(ranks) != ep_size:
        raise RuntimeError(f"HierMoE EP group rank mismatch: expected {ep_size} ranks, got {len(ranks)}.")
    return ranks


def _control_collective_device() -> torch.device:
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_device_id())


def _collect_ep_global_rank_groups(ep_global_ranks: tuple[int, ...], ep_size: int) -> tuple[tuple[int, ...], ...]:
    world_size = dist.get_world_size()
    if world_size == ep_size:
        return (ep_global_ranks,)

    device = _control_collective_device()
    local = torch.tensor(ep_global_ranks, dtype=torch.int64, device=device)
    gathered = torch.empty((world_size, ep_size), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local)
    gathered_cpu = gathered.detach().to(torch.device("cpu"))
    return tuple(sorted({tuple(int(rank) for rank in row.tolist()) for row in gathered_cpu}))


def _stage1_rank_lists(
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
) -> list[tuple[int, ...]]:
    num_nodes = ep_size // intra_size
    return [
        tuple(ep_global_ranks[local_offset + node_idx * intra_size] for node_idx in range(num_nodes))
        for ep_global_ranks in ep_global_rank_groups
        for local_offset in range(intra_size)
    ]


def _stage2_rank_lists(
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
) -> list[tuple[int, ...]]:
    num_nodes = ep_size // intra_size
    return [
        tuple(ep_global_ranks[node_idx * intra_size + local_offset] for local_offset in range(intra_size))
        for ep_global_ranks in ep_global_rank_groups
        for node_idx in range(num_nodes)
    ]


def _stage1_3d_rank_lists(
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    mid_size: int,
) -> list[tuple[int, ...]]:
    num_mid_groups = ep_size // mid_size
    return [
        tuple(ep_global_ranks[mid_offset + mid_idx * mid_size] for mid_idx in range(num_mid_groups))
        for ep_global_ranks in ep_global_rank_groups
        for mid_offset in range(mid_size)
    ]


def _stage2_3d_rank_lists(
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
    mid_size: int,
) -> list[tuple[int, ...]]:
    num_mid_groups = ep_size // mid_size
    nodes_per_mid = mid_size // intra_size
    return [
        tuple(
            ep_global_ranks[mid_idx * mid_size + node_idx * intra_size + local_offset]
            for node_idx in range(nodes_per_mid)
        )
        for ep_global_ranks in ep_global_rank_groups
        for mid_idx in range(num_mid_groups)
        for local_offset in range(intra_size)
    ]


def _stage3_3d_rank_lists(
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
) -> list[tuple[int, ...]]:
    num_nodes = ep_size // intra_size
    return [
        tuple(ep_global_ranks[node_idx * intra_size + local_offset] for local_offset in range(intra_size))
        for ep_global_ranks in ep_global_rank_groups
        for node_idx in range(num_nodes)
    ]


def _new_hiermoe_subgroup_by_enumeration(
    rank_lists: list[tuple[int, ...]],
    group_desc: str,
) -> dist.ProcessGroup | None:
    if not rank_lists or len(rank_lists[0]) <= 1:
        return None
    rank_lists_list = [list(ranks) for ranks in rank_lists]
    new_subgroups = getattr(dist, "new_subgroups_by_enumeration", None)
    if new_subgroups is not None:
        cur_group, _ = new_subgroups(rank_lists_list, group_desc=group_desc)
        return cur_group

    current_global_rank = dist.get_rank()
    cur_group = None
    for ranks in rank_lists_list:
        group = dist.new_group(ranks=ranks)
        if current_global_rank in ranks:
            cur_group = group
    return cur_group


def _create_hierarchical_process_groups_for_ep(
    ep_global_ranks: tuple[int, ...],
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
    current_global_rank: int,
    current_ep_rank: int | None = None,
) -> None:
    if len(ep_global_ranks) != ep_size:
        raise RuntimeError(f"HierMoE EP group rank mismatch: expected {ep_size} ranks, got {len(ep_global_ranks)}.")

    num_nodes = ep_size // intra_size
    rank_is_member = current_global_rank in ep_global_ranks
    if not rank_is_member:
        return

    ep_rank = ep_global_ranks.index(current_global_rank) if current_ep_rank is None else int(current_ep_rank)
    stage1_group = _new_hiermoe_subgroup_by_enumeration(
        _stage1_rank_lists(ep_global_rank_groups, ep_size, intra_size),
        group_desc="hiermoe_stage1",
    )
    stage2_group = _new_hiermoe_subgroup_by_enumeration(
        _stage2_rank_lists(ep_global_rank_groups, ep_size, intra_size),
        group_desc="hiermoe_stage2",
    )
    stage1_ep_ranks = tuple((ep_rank % intra_size) + node_idx * intra_size for node_idx in range(num_nodes))
    stage2_ep_ranks = tuple((ep_rank // intra_size) * intra_size + local_offset for local_offset in range(intra_size))
    if stage1_group is None or stage2_group is None:
        raise RuntimeError("HierMoE failed to assign the current rank to hierarchical process groups.")
    _HIERARCHICAL_GROUP_CACHE[(ep_global_ranks, ep_rank, intra_size)] = HierarchicalProcessGroups(
        stage1_group=stage1_group,
        stage2_group=stage2_group,
        stage1_ep_ranks=stage1_ep_ranks,
        stage2_ep_ranks=stage2_ep_ranks,
    )


def _create_hierarchical3d_process_groups_for_ep(
    ep_global_ranks: tuple[int, ...],
    ep_global_rank_groups: tuple[tuple[int, ...], ...],
    ep_size: int,
    intra_size: int,
    mid_size: int,
    current_global_rank: int,
    current_ep_rank: int | None = None,
) -> None:
    if len(ep_global_ranks) != ep_size:
        raise RuntimeError(f"HierMoE EP group rank mismatch: expected {ep_size} ranks, got {len(ep_global_ranks)}.")
    if mid_size <= intra_size or ep_size % mid_size != 0 or mid_size % intra_size != 0:
        raise RuntimeError(
            f"Invalid HierMoE 3D sizes: intra_size={intra_size}, mid_size={mid_size}, ep_size={ep_size}."
        )
    if current_global_rank not in ep_global_ranks:
        return

    ep_rank = ep_global_ranks.index(current_global_rank) if current_ep_rank is None else int(current_ep_rank)
    stage1_group = _new_hiermoe_subgroup_by_enumeration(
        _stage1_3d_rank_lists(ep_global_rank_groups, ep_size, mid_size),
        group_desc="hiermoe_stage1_3d",
    )
    stage2_group = _new_hiermoe_subgroup_by_enumeration(
        _stage2_3d_rank_lists(ep_global_rank_groups, ep_size, intra_size, mid_size),
        group_desc="hiermoe_stage2_3d",
    )
    stage3_group = _new_hiermoe_subgroup_by_enumeration(
        _stage3_3d_rank_lists(ep_global_rank_groups, ep_size, intra_size),
        group_desc="hiermoe_stage3_3d",
    )
    num_mid_groups = ep_size // mid_size
    nodes_per_mid = mid_size // intra_size
    mid_idx = ep_rank // mid_size
    mid_start = mid_idx * mid_size
    local_offset = ep_rank % intra_size
    node_start = (ep_rank // intra_size) * intra_size
    stage1_ep_ranks = tuple((ep_rank % mid_size) + group_idx * mid_size for group_idx in range(num_mid_groups))
    stage2_ep_ranks = tuple(mid_start + node_idx * intra_size + local_offset for node_idx in range(nodes_per_mid))
    stage3_ep_ranks = tuple(node_start + offset for offset in range(intra_size))
    if stage1_group is None or stage2_group is None or stage3_group is None:
        raise RuntimeError("HierMoE failed to assign the current rank to 3D hierarchical process groups.")
    _HIERARCHICAL_3D_GROUP_CACHE[(ep_global_ranks, ep_rank, intra_size, mid_size)] = Hierarchical3DProcessGroups(
        stage1_group=stage1_group,
        stage2_group=stage2_group,
        stage3_group=stage3_group,
        stage1_ep_ranks=stage1_ep_ranks,
        stage2_ep_ranks=stage2_ep_ranks,
        stage3_ep_ranks=stage3_ep_ranks,
    )


def _get_hierarchical_process_groups(
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    ep_rank: int,
    intra_size: int,
) -> HierarchicalProcessGroups:
    if ep_group is None or not dist.is_initialized():
        return HierarchicalProcessGroups(None, None, tuple(range(ep_size)), tuple(range(ep_size)))

    ep_global_ranks = _get_ep_global_ranks(ep_group, ep_size)
    cache_key = (ep_global_ranks, ep_rank, intra_size)
    cached = _HIERARCHICAL_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _create_hierarchical_process_groups_for_ep(
        ep_global_ranks,
        _collect_ep_global_rank_groups(ep_global_ranks, ep_size),
        ep_size,
        intra_size,
        dist.get_rank(),
        current_ep_rank=ep_rank,
    )
    cached = _HIERARCHICAL_GROUP_CACHE.get(cache_key)
    if cached is None:
        raise RuntimeError("HierMoE failed to lazily initialize hierarchical process groups before forward.")
    return cached


def _get_hierarchical3d_process_groups(
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    ep_rank: int,
    intra_size: int,
    mid_size: int,
) -> Hierarchical3DProcessGroups:
    if ep_group is None or not dist.is_initialized():
        return Hierarchical3DProcessGroups(
            None,
            None,
            None,
            tuple(range(ep_size)),
            tuple(range(ep_size)),
            tuple(range(ep_size)),
        )

    ep_global_ranks = _get_ep_global_ranks(ep_group, ep_size)
    cache_key = (ep_global_ranks, ep_rank, intra_size, mid_size)
    cached = _HIERARCHICAL_3D_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _create_hierarchical3d_process_groups_for_ep(
        ep_global_ranks,
        _collect_ep_global_rank_groups(ep_global_ranks, ep_size),
        ep_size,
        intra_size,
        mid_size,
        dist.get_rank(),
        current_ep_rank=ep_rank,
    )
    cached = _HIERARCHICAL_3D_GROUP_CACHE.get(cache_key)
    if cached is None:
        raise RuntimeError("HierMoE failed to lazily initialize 3D hierarchical process groups before forward.")
    return cached


def _select_splits(split_sizes: list[int], ranks: tuple[int, ...]) -> list[int]:
    return [int(split_sizes[rank]) for rank in ranks]


def warmup_hierarchical_process_groups(
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    intra_size: int,
) -> None:
    if ep_group is None or not dist.is_initialized() or intra_size <= 1 or intra_size >= ep_size:
        return
    if ep_size % intra_size != 0:
        raise RuntimeError(f"Invalid HierMoE 2D intra_size={intra_size} for ep_size={ep_size}.")
    ep_global_ranks = _get_ep_global_ranks(ep_group, ep_size)
    ep_rank = dist.get_rank(ep_group)
    current_global_rank = dist.get_rank()
    ep_global_rank_groups = _collect_ep_global_rank_groups(ep_global_ranks, ep_size)
    _create_hierarchical_process_groups_for_ep(
        ep_global_ranks,
        ep_global_rank_groups,
        ep_size,
        intra_size,
        current_global_rank,
        current_ep_rank=ep_rank,
    )
    _get_hierarchical_process_groups(ep_group, ep_size, ep_rank, intra_size)


def _empty_2d(rows: int, cols: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, cols), dtype=dtype, device=device)


def _metadata_payload_dtype(weight_dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if weight_dtype is torch.float64 else torch.float32


def _pack_meta_weights(meta: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    payload_dtype = _metadata_payload_dtype(weights.dtype)
    return torch.cat((meta.to(payload_dtype), weights.to(payload_dtype)), dim=1)


def _unpack_meta_weights(
    payload: torch.Tensor,
    *,
    meta_cols: int,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    meta = payload[:, :meta_cols].to(torch.int32)
    weights = payload[:, meta_cols:].to(weight_dtype)
    return meta, weights


def _pack_nonnegative_pair(lhs: torch.Tensor, rhs: torch.Tensor, rhs_base: int) -> torch.Tensor:
    return lhs.to(torch.long) * int(rhs_base) + rhs.to(torch.long)


def _unpack_nonnegative_pair(code: torch.Tensor, rhs_base: int) -> tuple[torch.Tensor, torch.Tensor]:
    code = code.to(torch.long)
    lhs = torch.div(code, int(rhs_base), rounding_mode="floor")
    rhs = code - lhs * int(rhs_base)
    return lhs, rhs


def _call_all_to_all(
    group: dist.ProcessGroup | None,
    tensor: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
) -> torch.Tensor:
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    input_total = sum(int(size) for size in input_splits)
    output_total = sum(int(size) for size in output_splits)
    if input_total != int(tensor.shape[0]):
        rank = dist.get_rank(group)
        raise RuntimeError(
            "HierMoE all_to_all input split mismatch: "
            f"rank={rank} tensor_rows={int(tensor.shape[0])} input_total={input_total} "
            f"output_total={output_total} input_splits={input_splits} output_splits={output_splits}"
        )
    if output_total < 0:
        rank = dist.get_rank(group)
        raise RuntimeError(
            "HierMoE all_to_all output split mismatch: "
            f"rank={rank} output_total={output_total} output_splits={output_splits}"
        )
    try:
        return all_to_all(group, tensor, output_splits, input_splits)
    except Exception as error:
        rank = dist.get_rank(group)
        raise RuntimeError(
            "HierMoE all_to_all failed: "
            f"rank={rank} tensor_shape={tuple(tensor.shape)} "
            f"input_splits={input_splits} output_splits={output_splits}"
        ) from error


def _validate_all_to_all_splits(
    tensor: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    *,
    name: str,
) -> None:
    input_total = sum(int(size) for size in input_splits)
    output_total = sum(int(size) for size in output_splits)
    if input_total != int(tensor.shape[0]):
        raise RuntimeError(
            f"HierMoE {name} all_to_all input split mismatch: "
            f"tensor_rows={int(tensor.shape[0])} input_total={input_total} "
            f"output_total={output_total} input_splits={input_splits} output_splits={output_splits}"
        )
    if output_total < 0:
        raise RuntimeError(f"HierMoE {name} all_to_all output split mismatch: output_splits={output_splits}")


def _call_all_to_all_pair(
    group: dist.ProcessGroup | None,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    output_splits_a: list[int],
    input_splits_a: list[int],
    output_splits_b: list[int],
    input_splits_b: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor_a, tensor_b
    _validate_all_to_all_splits(tensor_a, output_splits_a, input_splits_a, name="payload_a")
    _validate_all_to_all_splits(tensor_b, output_splits_b, input_splits_b, name="payload_b")
    return all_to_all_pair(
        group,
        tensor_a,
        tensor_b,
        output_splits_a,
        input_splits_a,
        output_splits_b,
        input_splits_b,
    )


def _split_tensor_to_list(split_sizes: torch.Tensor) -> list[int]:
    # all_to_all_single consumes Python split sizes; the CPU copy must be complete before returning.
    cpu_sizes = split_sizes.detach().to(torch.device("cpu"))
    return [int(size) for size in cpu_sizes.tolist()]


def _split_matrix_columns_to_lists(split_sizes: torch.Tensor) -> list[list[int]]:
    rows = split_sizes.detach().to(torch.device("cpu")).tolist()
    if not rows:
        return []
    return [[int(row[idx]) for row in rows] for idx in range(len(rows[0]))]


def _exchange_split_sizes(local_splits: list[int], group: dist.ProcessGroup | None, device: torch.device) -> list[int]:
    return _exchange_split_sizes_many([local_splits], group, device)[0]


def _exchange_split_sizes_many(
    local_splits_list: list[list[int]], group: dist.ProcessGroup | None, device: torch.device
) -> list[list[int]]:
    return _start_exchange_split_sizes_many(local_splits_list, group, device).wait()


def _start_exchange_split_sizes_many(
    local_splits_list: list[list[int]], group: dist.ProcessGroup | None, device: torch.device
) -> _PendingSplitSizeExchange:
    if not local_splits_list:
        return _PendingSplitSizeExchange(local=None, exchanged=None, work=None, immediate=[])
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return _PendingSplitSizeExchange(
            local=None,
            exchanged=None,
            work=None,
            immediate=[list(local_splits) for local_splits in local_splits_list],
        )

    ep_size = dist.get_world_size(group)
    local = torch.tensor(local_splits_list, dtype=torch.int64, device=device).t().contiguous()
    if local.shape[0] != ep_size:
        raise RuntimeError(
            f"HierMoE split-size exchange expected {ep_size} splits per list, got shape={tuple(local.shape)}."
        )
    exchanged = torch.empty_like(local)
    work = dist.all_to_all_single(exchanged, local, group=group, async_op=True)
    return _PendingSplitSizeExchange(local=local, exchanged=exchanged, work=work, immediate=None)


def _repeat_ranks(split_sizes: list[int], device: torch.device) -> torch.Tensor:
    pieces = [
        torch.full((int(size),), rank, dtype=torch.long, device=device)
        for rank, size in enumerate(split_sizes)
        if int(size) > 0
    ]
    if not pieces:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(pieces, dim=0)


def _sort_assignments_by_local_expert(
    assignment_tokens: torch.Tensor,
    local_expert_ids: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sort_indices, unsort_indices, tokens_per_local_expert = _local_expert_sort_indices(
        local_expert_ids,
        num_local_experts,
        assignment_tokens.device,
    )
    return assignment_tokens.index_select(0, sort_indices), unsort_indices, tokens_per_local_expert, sort_indices


def _local_expert_sort_indices(
    local_expert_ids: torch.Tensor,
    num_local_experts: int,
    device: torch.device,
    *,
    build_unsort: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens_per_local_expert = torch.bincount(local_expert_ids, minlength=num_local_experts)
    if local_expert_ids.numel() == 0:
        sort_indices = torch.empty((0,), dtype=torch.long, device=device)
        unsort_indices = torch.empty((0,), dtype=torch.long, device=device)
        return sort_indices, unsort_indices, tokens_per_local_expert

    sort_chunks = []
    for expert_idx in range(num_local_experts):
        indices = torch.nonzero(local_expert_ids == expert_idx, as_tuple=False).flatten()
        if indices.numel() > 0:
            sort_chunks.append(indices)
    sort_indices = torch.cat(sort_chunks, dim=0) if sort_chunks else torch.empty((0,), dtype=torch.long, device=device)
    if not build_unsort:
        return sort_indices, torch.empty((0,), dtype=torch.long, device=device), tokens_per_local_expert
    unsort_indices = torch.empty_like(sort_indices)
    unsort_indices.scatter_(
        0,
        sort_indices,
        torch.arange(sort_indices.numel(), dtype=torch.long, device=sort_indices.device),
    )
    return sort_indices, unsort_indices, tokens_per_local_expert


def _sort_key_indices(reference_tensor: torch.Tensor, sort_keys: torch.Tensor) -> torch.Tensor:
    if reference_tensor.device.type == "npu" and is_torch_npu_available():
        import torch_npu

        dummy_dtype = reference_tensor.dtype if torch.is_floating_point(reference_tensor) else torch.float32
        dummy_hidden = torch.empty((sort_keys.shape[0], 1), dtype=dummy_dtype, device=reference_tensor.device)
        _, row_ids_map = torch_npu.npu_moe_token_permute(dummy_hidden, sort_keys.to(torch.int32))
        flat_positions = torch.arange(sort_keys.numel(), dtype=torch.long, device=reference_tensor.device)
        sort_indices = torch.empty_like(flat_positions)
        sort_indices.scatter_(0, row_ids_map.to(torch.long), flat_positions)
        return sort_indices

    return torch.argsort(sort_keys)


class _NpuIndexAddDim0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output: torch.Tensor, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        import torch_npu

        return torch_npu._npu_index_add_(output, index, source)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        (index,) = ctx.saved_tensors
        grad_source = grad_output.index_select(0, index.to(torch.long))
        return grad_output, None, grad_source


def _index_add_dim0(output: torch.Tensor, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if output.device.type == "npu" and is_torch_npu_available():
        return _NpuIndexAddDim0.apply(output, index, source)
    output.index_add_(0, index, source)
    return output


def _index_add_dim0_fp32(output: torch.Tensor, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Accumulate into FP32 without materializing a full-size FP32 source.

    Hierarchical combine payloads can contain hundreds of thousands of hidden
    rows at EP64. Converting the complete BF16 payload with ``source.float()``
    creates a multi-GiB temporary even though index-add can consume independent
    row chunks. Chunking preserves the exact FP32 accumulation semantics and
    autograd graph while bounding the conversion peak.
    """

    if output.dtype != torch.float32:
        raise ValueError(f"FP32 index-add requires a float32 output, got {output.dtype}.")
    if int(index.numel()) != int(source.shape[0]):
        raise ValueError(f"index/source row mismatch: index={index.numel()} source={source.shape[0]}.")
    if source.numel() == 0:
        return output

    chunk_rows = max(
        1,
        int(os.getenv("VEOMNI_HIERMOE_INDEX_ADD_FP32_CHUNK_ROWS", "65536")),
    )
    if source.dtype == torch.float32 or int(source.shape[0]) <= chunk_rows:
        return _index_add_dim0(output, index, source.to(torch.float32))

    for start in range(0, int(source.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(source.shape[0]))
        output = _index_add_dim0(
            output,
            index[start:end],
            source[start:end].to(torch.float32),
        )
    return output


class _CudaIndexAddDim0CastOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source: torch.Tensor, index: torch.Tensor, num_rows: int) -> torch.Tensor:
        from .triton_segment_sum import segment_sum_dim0_fp32_to_source_dtype

        normalized_index = index.to(device=source.device, dtype=torch.long)
        ctx.save_for_backward(normalized_index)
        return segment_sum_dim0_fp32_to_source_dtype(source, normalized_index, int(num_rows))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (index,) = ctx.saved_tensors
        return grad_output.index_select(0, index.to(torch.long)), None, None


def _index_add_dim0_cast_output(
    source: torch.Tensor,
    index: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Reduce rows in FP32 and return a memory-bounded source-dtype tensor."""

    num_rows = int(num_rows)
    if int(index.numel()) != int(source.shape[0]):
        raise ValueError(f"index/source row mismatch: index={index.numel()} source={source.shape[0]}.")
    use_cuda_segment_sum = os.getenv("VEOMNI_HIERMOE_CUDA_SEGMENT_SUM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if (
        source.device.type == "cuda"
        and source.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and use_cuda_segment_sum
    ):
        return _CudaIndexAddDim0CastOutput.apply(source.contiguous(), index.contiguous(), num_rows)

    output = torch.zeros(
        (num_rows, int(source.shape[1])),
        dtype=torch.float32,
        device=source.device,
    )
    output = _index_add_dim0_fp32(output, index, source)
    return output.to(source.dtype)


def _sort_flat_assignments_by_rank_token(
    hidden_states: torch.Tensor,
    target_ranks: torch.Tensor,
    flat_ranks: torch.Tensor,
    flat_tokens: torch.Tensor,
    key_stride: int,
) -> torch.Tensor:
    del target_ranks
    return _sort_key_indices(hidden_states, flat_ranks * key_stride + flat_tokens)


def _build_local_payload(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    physical_experts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
    device = hidden_states.device
    num_local_experts = num_experts // ep_size
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
        routing_weights = routing_weights.unsqueeze(-1)
    routed_experts = selected_experts if physical_experts is None else physical_experts
    if routed_experts.ndim == 1:
        routed_experts = routed_experts.unsqueeze(-1)

    num_tokens, top_k = routed_experts.shape
    target_ranks = torch.div(routed_experts, num_local_experts, rounding_mode="floor")
    flat_ranks = target_ranks.reshape(-1).to(torch.long)
    flat_tokens = (
        torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    )
    flat_experts = routed_experts.reshape(-1).to(torch.long)
    flat_weights = routing_weights.reshape(-1)

    assignment_send_splits_tensor = torch.bincount(flat_ranks, minlength=ep_size)
    assignment_send_splits = _split_tensor_to_list(assignment_send_splits_tensor)
    if flat_ranks.numel() == 0:
        return (
            _empty_2d(0, hidden_states.shape[-1], dtype=hidden_states.dtype, device=device),
            _empty_2d(0, 3, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=routing_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(ep_size)],
            assignment_send_splits,
            target_ranks,
        )

    key_stride = max(1, num_tokens)
    sort_indices = _sort_flat_assignments_by_rank_token(
        hidden_states,
        target_ranks,
        flat_ranks,
        flat_tokens,
        key_stride,
    )
    sorted_keys = (flat_ranks * key_stride + flat_tokens).index_select(0, sort_indices)
    sorted_ranks = flat_ranks.index_select(0, sort_indices)
    sorted_tokens = flat_tokens.index_select(0, sort_indices)
    sorted_experts = flat_experts.index_select(0, sort_indices)
    sorted_weights = flat_weights.index_select(0, sort_indices)

    unique_keys, unique_inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    unique_ranks = torch.div(unique_keys, key_stride, rounding_mode="floor")
    unique_tokens = unique_keys - unique_ranks * key_stride
    unique_send_splits_tensor = torch.bincount(unique_ranks, minlength=ep_size)
    unique_send_splits = _split_tensor_to_list(unique_send_splits_tensor)

    unique_offsets = _split_offsets(unique_send_splits, device)
    unique_positions = torch.arange(unique_keys.numel(), dtype=torch.long, device=device)
    unique_ordinal_by_unique = unique_positions - unique_offsets.index_select(0, unique_ranks)
    unique_ordinal = unique_ordinal_by_unique.index_select(0, unique_inverse)
    local_expert = sorted_experts - sorted_ranks * num_local_experts

    send_hidden = hidden_states.index_select(0, unique_tokens)
    send_meta = torch.stack((unique_ordinal, local_expert, sorted_tokens), dim=1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return (
        send_hidden,
        send_meta,
        send_weights,
        unique_tokens,
        unique_send_splits,
        assignment_send_splits,
        target_ranks,
    )


def _split_offsets(split_sizes: list[int], device: torch.device) -> torch.Tensor:
    return torch.cumsum(
        torch.tensor([0] + split_sizes[:-1], dtype=torch.long, device=device),
        dim=0,
    )


def _counts_to_ep_splits(counts: torch.Tensor, ep_ranks: torch.Tensor, ep_size: int) -> list[int]:
    full = torch.zeros((ep_size,), dtype=counts.dtype, device=counts.device)
    full.scatter_(0, ep_ranks.to(torch.long), counts)
    return _split_tensor_to_list(full)


def _counts_to_ep_splits_many(
    counts_list: list[torch.Tensor],
    ep_ranks: torch.Tensor,
    ep_size: int,
) -> list[list[int]]:
    if not counts_list:
        return []
    full = torch.zeros((ep_size, len(counts_list)), dtype=counts_list[0].dtype, device=counts_list[0].device)
    scatter_ranks = ep_ranks.to(torch.long)
    for idx, counts in enumerate(counts_list):
        full[:, idx].scatter_(0, scatter_ranks, counts)
    return _split_matrix_columns_to_lists(full)


def _counts_to_splits_many(counts_list: list[torch.Tensor]) -> list[list[int]]:
    if not counts_list:
        return []
    stacked = torch.stack(counts_list, dim=1)
    return _split_matrix_columns_to_lists(stacked)


def _select_dimension(
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    group: dist.ProcessGroup | None,
) -> int:
    state = get_hiermoe_state()
    if state is None:
        return 1
    if state.communication_mode == "direct":
        return 1
    if state.communication_mode == "hierarchical":
        return max(1, int(state.hierarchy.selected_dim))
    if state.perf_model.source == "default":
        return min(2, max(1, int(state.hierarchy.selected_dim)))
    local_dim = state.perf_model.select_dimension(
        selected_experts=selected_experts,
        num_experts=num_experts,
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        hierarchy=state.hierarchy,
    )
    if (
        state.hierarchy.selected_dim <= 1
        or group is None
        or not dist.is_initialized()
        or dist.get_world_size(group) == 1
    ):
        return local_dim

    dim_tensor = torch.tensor([local_dim], dtype=torch.int64, device=selected_experts.device)
    dist.all_reduce(dim_tensor, op=dist.ReduceOp.MIN, group=group)
    return int(dim_tensor.item())


def _hierarchical_intra_size(ep_size: int, selected_dim: int) -> int | None:
    state = get_hiermoe_state()
    if state is None or selected_dim < 2 or len(state.hierarchy.group_sizes) < 2:
        return None
    intra_size = int(state.hierarchy.group_sizes[0])
    if intra_size <= 1 or intra_size >= ep_size or ep_size % intra_size != 0:
        return None
    return intra_size


def _hierarchical3d_sizes(ep_size: int, selected_dim: int) -> tuple[int, int] | None:
    state = get_hiermoe_state()
    if state is None or selected_dim < 3 or len(state.hierarchy.group_sizes) < 3:
        return None
    intra_size = int(state.hierarchy.group_sizes[0])
    mid_size = int(state.hierarchy.group_sizes[1])
    if (
        intra_size <= 1
        or mid_size <= intra_size
        or mid_size >= ep_size
        or mid_size % intra_size != 0
        or ep_size % mid_size != 0
    ):
        return None
    return intra_size, mid_size


def _build_stage1_payload(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    intra_size: int,
    physical_experts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    device = hidden_states.device
    num_local_experts = num_experts // ep_size
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
        routing_weights = routing_weights.unsqueeze(-1)
    routed_experts = selected_experts if physical_experts is None else physical_experts
    if routed_experts.ndim == 1:
        routed_experts = routed_experts.unsqueeze(-1)

    num_tokens, top_k = routed_experts.shape
    target_ranks = torch.div(routed_experts, num_local_experts, rounding_mode="floor")
    target_nodes = torch.div(target_ranks, intra_size, rounding_mode="floor")
    num_nodes = ep_size // intra_size
    flat_nodes = target_nodes.reshape(-1).to(torch.long)
    flat_target_local_ranks = target_ranks.remainder(intra_size).reshape(-1).to(torch.long)
    flat_tokens = (
        torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    )
    flat_experts = routed_experts.reshape(-1).to(torch.long)
    flat_weights = routing_weights.reshape(-1)

    assignment_counts_by_node = torch.bincount(flat_nodes, minlength=num_nodes)
    if flat_nodes.numel() == 0:
        assignment_send_splits = _split_tensor_to_list(assignment_counts_by_node)
        return (
            _empty_2d(0, hidden_states.shape[-1], dtype=hidden_states.dtype, device=device),
            _empty_2d(0, 2, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=routing_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(num_nodes)],
            assignment_send_splits,
        )

    token_node_hits = torch.zeros((num_tokens, num_nodes), dtype=torch.bool, device=device)
    token_node_hits.scatter_(dim=1, index=target_nodes.to(torch.long), value=True)
    unique_counts_by_node = token_node_hits.sum(dim=0, dtype=torch.int32)
    unique_send_splits, assignment_send_splits = _counts_to_splits_many(
        [unique_counts_by_node, assignment_counts_by_node],
    )

    unique_node_tokens = token_node_hits.t().nonzero(as_tuple=False)
    unique_tokens = unique_node_tokens[:, 1].to(torch.long)
    node_unique_ordinal_by_token = torch.cumsum(token_node_hits, dim=0, dtype=torch.int32) - 1
    node_unique_ordinal = node_unique_ordinal_by_token[flat_tokens, flat_nodes]

    sort_indices = _sort_key_indices(hidden_states, flat_nodes)
    sorted_node_unique_ordinal = node_unique_ordinal.index_select(0, sort_indices)
    sorted_experts = flat_experts.index_select(0, sort_indices)
    sorted_target_local_ranks = flat_target_local_ranks.index_select(0, sort_indices)
    sorted_weights = flat_weights.index_select(0, sort_indices)

    target_ranks_for_assignments = torch.div(sorted_experts, num_local_experts, rounding_mode="floor")
    local_experts = sorted_experts - target_ranks_for_assignments * num_local_experts

    send_hidden = hidden_states.index_select(0, unique_tokens)
    rank_expert_code = _pack_nonnegative_pair(sorted_target_local_ranks, local_experts, num_local_experts)
    send_meta = torch.stack((sorted_node_unique_ordinal, rank_expert_code), dim=1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return send_hidden, send_meta, send_weights, unique_tokens, unique_send_splits, assignment_send_splits


def _build_stage2_payload(
    stage1_hidden: torch.Tensor,
    stage1_meta: torch.Tensor,
    stage1_weights: torch.Tensor,
    stage1_unique_recv_splits: list[int],
    stage1_assignment_recv_splits: list[int],
    ep_size: int,
    ep_rank: int,
    intra_size: int,
    num_local_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    device = stage1_hidden.device
    hidden_size = stage1_hidden.shape[-1]
    source_group_ranks = _repeat_ranks(stage1_assignment_recv_splits, device)
    if stage1_meta.numel() == 0:
        return (
            _empty_2d(0, hidden_size, dtype=stage1_hidden.dtype, device=device),
            _empty_2d(0, 1, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=stage1_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(intra_size)],
            [0 for _ in range(intra_size)],
        )

    stage1_offsets = _split_offsets(stage1_unique_recv_splits, device)
    stage1_unique_indices = stage1_offsets.index_select(0, source_group_ranks) + stage1_meta[:, 0].to(torch.long)
    target_local_ranks, local_experts = _unpack_nonnegative_pair(stage1_meta[:, 1], num_local_experts)

    stage1_rank_hits = torch.zeros((stage1_hidden.shape[0], intra_size), dtype=torch.bool, device=device)
    stage1_rank_hits[stage1_unique_indices, target_local_ranks] = True
    unique_counts_by_local = stage1_rank_hits.sum(dim=0, dtype=torch.int32)
    stage1_rank_unique_ordinal_by_token = torch.cumsum(stage1_rank_hits, dim=0, dtype=torch.int32) - 1
    rank_unique_ordinal = stage1_rank_unique_ordinal_by_token[stage1_unique_indices, target_local_ranks]

    rank_major_unique_tokens = torch.nonzero(stage1_rank_hits.t(), as_tuple=False)
    unique_stage1_indices = rank_major_unique_tokens[:, 1].to(torch.long)
    assignment_counts_by_local = torch.bincount(target_local_ranks, minlength=intra_size)
    unique_send_splits, assignment_send_splits = _counts_to_splits_many(
        [unique_counts_by_local, assignment_counts_by_local],
    )

    sort_indices = _sort_key_indices(stage1_hidden, target_local_ranks)
    sorted_rank_unique_ordinal = rank_unique_ordinal.index_select(0, sort_indices)
    sorted_weights = stage1_weights.reshape(-1).index_select(0, sort_indices)
    sorted_local_experts = local_experts.index_select(0, sort_indices)

    send_hidden = stage1_hidden.index_select(0, unique_stage1_indices)
    rank_expert_code = _pack_nonnegative_pair(sorted_rank_unique_ordinal, sorted_local_experts, num_local_experts)
    send_meta = rank_expert_code.unsqueeze(1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return (
        send_hidden,
        send_meta,
        send_weights,
        unique_stage1_indices,
        unique_send_splits,
        assignment_send_splits,
    )


def _build_stage1_3d_payload(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    intra_size: int,
    mid_size: int,
    physical_experts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    device = hidden_states.device
    num_local_experts = num_experts // ep_size
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
        routing_weights = routing_weights.unsqueeze(-1)
    routed_experts = selected_experts if physical_experts is None else physical_experts
    if routed_experts.ndim == 1:
        routed_experts = routed_experts.unsqueeze(-1)

    num_tokens, top_k = routed_experts.shape
    target_ranks = torch.div(routed_experts, num_local_experts, rounding_mode="floor")
    target_mid_groups = torch.div(target_ranks, mid_size, rounding_mode="floor")
    num_mid_groups = ep_size // mid_size
    flat_mid_groups = target_mid_groups.reshape(-1).to(torch.long)
    flat_tokens = (
        torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    )
    flat_experts = routed_experts.reshape(-1).to(torch.long)
    flat_weights = routing_weights.reshape(-1)

    assignment_counts_by_mid = torch.bincount(flat_mid_groups, minlength=num_mid_groups)
    if flat_mid_groups.numel() == 0:
        assignment_send_splits = _split_tensor_to_list(assignment_counts_by_mid)
        return (
            _empty_2d(0, hidden_states.shape[-1], dtype=hidden_states.dtype, device=device),
            _empty_2d(0, 2, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=routing_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(num_mid_groups)],
            assignment_send_splits,
        )

    token_mid_hits = torch.zeros((num_tokens, num_mid_groups), dtype=torch.bool, device=device)
    token_mid_hits.scatter_(dim=1, index=target_mid_groups.to(torch.long), value=True)
    unique_counts_by_mid = token_mid_hits.sum(dim=0, dtype=torch.int32)
    unique_send_splits, assignment_send_splits = _counts_to_splits_many(
        [unique_counts_by_mid, assignment_counts_by_mid]
    )

    unique_mid_tokens = token_mid_hits.t().nonzero(as_tuple=False)
    unique_tokens = unique_mid_tokens[:, 1].to(torch.long)
    mid_unique_ordinal_by_token = torch.cumsum(token_mid_hits, dim=0, dtype=torch.int32) - 1
    mid_unique_ordinal = mid_unique_ordinal_by_token[flat_tokens, flat_mid_groups]

    target_rank_in_mid = target_ranks.remainder(mid_size)
    target_node_in_mid = torch.div(target_rank_in_mid, intra_size, rounding_mode="floor")
    target_local_rank = target_ranks.remainder(intra_size)
    flat_target_node_in_mid = target_node_in_mid.reshape(-1).to(torch.long)
    flat_target_local_ranks = target_local_rank.reshape(-1).to(torch.long)

    sort_indices = _sort_key_indices(hidden_states, flat_mid_groups)
    sorted_mid_unique_ordinal = mid_unique_ordinal.index_select(0, sort_indices)
    sorted_experts = flat_experts.index_select(0, sort_indices)
    sorted_target_nodes = flat_target_node_in_mid.index_select(0, sort_indices)
    sorted_target_local_ranks = flat_target_local_ranks.index_select(0, sort_indices)
    sorted_weights = flat_weights.index_select(0, sort_indices)

    target_ranks_for_assignments = torch.div(sorted_experts, num_local_experts, rounding_mode="floor")
    local_experts = sorted_experts - target_ranks_for_assignments * num_local_experts
    local_code = _pack_nonnegative_pair(sorted_target_local_ranks, local_experts, num_local_experts)
    node_local_code = _pack_nonnegative_pair(sorted_target_nodes, local_code, intra_size * num_local_experts)

    send_hidden = hidden_states.index_select(0, unique_tokens)
    send_meta = torch.stack((sorted_mid_unique_ordinal, node_local_code), dim=1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return send_hidden, send_meta, send_weights, unique_tokens, unique_send_splits, assignment_send_splits


def _build_stage2_3d_payload(
    stage1_hidden: torch.Tensor,
    stage1_meta: torch.Tensor,
    stage1_weights: torch.Tensor,
    stage1_unique_recv_splits: list[int],
    stage1_assignment_recv_splits: list[int],
    intra_size: int,
    mid_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    device = stage1_hidden.device
    hidden_size = stage1_hidden.shape[-1]
    nodes_per_mid = mid_size // intra_size
    source_mid_ranks = _repeat_ranks(stage1_assignment_recv_splits, device)
    if stage1_meta.numel() == 0:
        return (
            _empty_2d(0, hidden_size, dtype=stage1_hidden.dtype, device=device),
            _empty_2d(0, 1, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=stage1_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(nodes_per_mid)],
            [0 for _ in range(nodes_per_mid)],
        )

    stage1_offsets = _split_offsets(stage1_unique_recv_splits, device)
    stage1_unique_indices = stage1_offsets.index_select(0, source_mid_ranks) + stage1_meta[:, 0].to(torch.long)
    target_nodes, local_code = _unpack_nonnegative_pair(stage1_meta[:, 1], intra_size * num_local_experts)

    stage1_node_hits = torch.zeros((stage1_hidden.shape[0], nodes_per_mid), dtype=torch.bool, device=device)
    stage1_node_hits[stage1_unique_indices, target_nodes] = True
    unique_counts_by_node = stage1_node_hits.sum(dim=0, dtype=torch.int32)
    stage1_node_unique_ordinal_by_token = torch.cumsum(stage1_node_hits, dim=0, dtype=torch.int32) - 1
    node_unique_ordinal = stage1_node_unique_ordinal_by_token[stage1_unique_indices, target_nodes]

    node_major_unique_tokens = torch.nonzero(stage1_node_hits.t(), as_tuple=False)
    unique_stage1_indices = node_major_unique_tokens[:, 1].to(torch.long)
    assignment_counts_by_node = torch.bincount(target_nodes, minlength=nodes_per_mid)
    unique_send_splits, assignment_send_splits = _counts_to_splits_many(
        [unique_counts_by_node, assignment_counts_by_node]
    )

    sort_indices = _sort_key_indices(stage1_hidden, target_nodes)
    sorted_node_unique_ordinal = node_unique_ordinal.index_select(0, sort_indices)
    sorted_local_code = local_code.index_select(0, sort_indices)
    sorted_weights = stage1_weights.reshape(-1).index_select(0, sort_indices)

    rank_expert_code = _pack_nonnegative_pair(
        sorted_node_unique_ordinal,
        sorted_local_code,
        intra_size * num_local_experts,
    )
    send_hidden = stage1_hidden.index_select(0, unique_stage1_indices)
    send_meta = rank_expert_code.unsqueeze(1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return (
        send_hidden,
        send_meta,
        send_weights,
        unique_stage1_indices,
        unique_send_splits,
        assignment_send_splits,
    )


def _build_stage3_3d_payload(
    stage2_hidden: torch.Tensor,
    stage2_meta: torch.Tensor,
    stage2_weights: torch.Tensor,
    stage2_unique_recv_splits: list[int],
    stage2_assignment_recv_splits: list[int],
    intra_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    device = stage2_hidden.device
    hidden_size = stage2_hidden.shape[-1]
    source_node_ranks = _repeat_ranks(stage2_assignment_recv_splits, device)
    if stage2_meta.numel() == 0:
        return (
            _empty_2d(0, hidden_size, dtype=stage2_hidden.dtype, device=device),
            _empty_2d(0, 1, dtype=torch.int32, device=device),
            _empty_2d(0, 1, dtype=stage2_weights.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [0 for _ in range(intra_size)],
            [0 for _ in range(intra_size)],
        )

    stage2_offsets = _split_offsets(stage2_unique_recv_splits, device)
    stage2_unique_ordinal, local_code = _unpack_nonnegative_pair(stage2_meta[:, 0], intra_size * num_local_experts)
    stage2_unique_indices = stage2_offsets.index_select(0, source_node_ranks) + stage2_unique_ordinal
    target_local_ranks, local_experts = _unpack_nonnegative_pair(local_code, num_local_experts)

    stage2_rank_hits = torch.zeros((stage2_hidden.shape[0], intra_size), dtype=torch.bool, device=device)
    stage2_rank_hits[stage2_unique_indices, target_local_ranks] = True
    unique_counts_by_local = stage2_rank_hits.sum(dim=0, dtype=torch.int32)
    stage2_rank_unique_ordinal_by_token = torch.cumsum(stage2_rank_hits, dim=0, dtype=torch.int32) - 1
    rank_unique_ordinal = stage2_rank_unique_ordinal_by_token[stage2_unique_indices, target_local_ranks]

    rank_major_unique_tokens = torch.nonzero(stage2_rank_hits.t(), as_tuple=False)
    unique_stage2_indices = rank_major_unique_tokens[:, 1].to(torch.long)
    assignment_counts_by_local = torch.bincount(target_local_ranks, minlength=intra_size)
    unique_send_splits, assignment_send_splits = _counts_to_splits_many(
        [unique_counts_by_local, assignment_counts_by_local]
    )

    sort_indices = _sort_key_indices(stage2_hidden, target_local_ranks)
    sorted_rank_unique_ordinal = rank_unique_ordinal.index_select(0, sort_indices)
    sorted_weights = stage2_weights.reshape(-1).index_select(0, sort_indices)
    sorted_local_experts = local_experts.index_select(0, sort_indices)

    send_hidden = stage2_hidden.index_select(0, unique_stage2_indices)
    rank_expert_code = _pack_nonnegative_pair(sorted_rank_unique_ordinal, sorted_local_experts, num_local_experts)
    send_meta = rank_expert_code.unsqueeze(1).to(torch.int32)
    send_weights = sorted_weights.unsqueeze(1)
    return (
        send_hidden,
        send_meta,
        send_weights,
        unique_stage2_indices,
        unique_send_splits,
        assignment_send_splits,
    )


def _hierarchical_dedup_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    selected_dim: int,
    intra_size: int,
    physical_experts: torch.Tensor | None = None,
    layer_key: str | None = None,
) -> tuple[torch.Tensor, RankDedupDispatchContext, torch.Tensor]:
    groups = _get_hierarchical_process_groups(ep_group, ep_size, ep_rank, intra_size)
    pipeline_manager = _fixed_pipeline_manager(layer_key)
    internal_timing_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None = (
        {} if _HIERMOE_INTERNAL_TIMING else None
    )
    backward_internal_timing_events = {} if _HIERMOE_INTERNAL_TIMING and hidden_states.requires_grad else None
    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_stage1_payload_build")
    try:
        (
            stage1_send_hidden,
            stage1_send_meta,
            stage1_send_weights,
            local_unique_token_indices,
            stage1_unique_send_splits_full,
            stage1_assignment_send_splits_full,
        ) = _build_stage1_payload(
            hidden_states,
            selected_experts,
            routing_weights,
            num_experts,
            ep_size,
            ep_rank,
            intra_size,
            physical_experts=physical_experts,
        )
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            internal_timing_events,
            "stage1_payload_build",
            internal_start,
        )
    stage1_unique_send_splits = stage1_unique_send_splits_full
    stage1_assignment_send_splits = stage1_assignment_send_splits_full
    span = _begin_internal_span("hiermoe_stage1_split_sizes_start")
    try:
        stage1_split_exchange = _start_exchange_split_sizes_many(
            [stage1_unique_send_splits, stage1_assignment_send_splits],
            groups.stage1_group,
            hidden_states.device,
        )
    finally:
        _end_internal_span(span)
    internal_start = _hiermoe_internal_event()
    stage1_send_meta_weights = _pack_meta_weights(stage1_send_meta, stage1_send_weights)
    _finish_hiermoe_internal_event(
        internal_timing_events,
        "stage1_meta_pack",
        internal_start,
    )
    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_stage1_split_sizes")
    try:
        stage1_unique_recv_splits, stage1_assignment_recv_splits = stage1_split_exchange.wait()
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            internal_timing_events,
            "stage1_split_wait",
            internal_start,
        )
    stage1_send_hidden = _mark_backward_a2a_input(
        stage1_send_hidden,
        backward_internal_timing_events,
        "backward_dispatch_stage1_a2a",
        layer_key,
    )
    if pipeline_manager is not None:
        pipeline_manager.open_pipeline_planner_prepare_window(layer_key, 0)
    stage1_a2a_start = _hiermoe_internal_event()
    stage1_recv_hidden, stage1_recv_meta_weights = _call_all_to_all_pair(
        groups.stage1_group,
        stage1_send_hidden,
        stage1_send_meta_weights,
        stage1_unique_recv_splits,
        stage1_unique_send_splits,
        stage1_assignment_recv_splits,
        stage1_assignment_send_splits,
    )
    stage1_a2a_end = _hiermoe_internal_event()
    if stage1_a2a_start is not None and stage1_a2a_end is not None and internal_timing_events is not None:
        internal_timing_events["stage1_a2a"] = (stage1_a2a_start, stage1_a2a_end)
    if pipeline_manager is not None:
        pipeline_manager.close_pipeline_planner_prepare_window(layer_key, 0)
    stage1_recv_hidden = _mark_backward_a2a_output(
        stage1_recv_hidden,
        backward_internal_timing_events,
        "backward_dispatch_stage1_a2a",
        layer_key,
    )
    internal_start = _hiermoe_internal_event()
    stage1_recv_meta, stage1_recv_weights = _unpack_meta_weights(
        stage1_recv_meta_weights, meta_cols=2, weight_dtype=routing_weights.dtype
    )
    _finish_hiermoe_internal_event(
        internal_timing_events,
        "stage1_meta_unpack",
        internal_start,
    )

    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_stage2_payload_build")
    try:
        (
            stage2_send_hidden,
            stage2_send_meta,
            stage2_send_weights,
            stage2_send_stage1_unique_indices,
            stage2_unique_send_splits,
            stage2_assignment_send_splits,
        ) = _build_stage2_payload(
            stage1_recv_hidden,
            stage1_recv_meta,
            stage1_recv_weights,
            stage1_unique_recv_splits,
            stage1_assignment_recv_splits,
            ep_size,
            ep_rank,
            intra_size,
            num_local_experts,
        )
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            internal_timing_events,
            "stage2_payload_build",
            internal_start,
        )
    span = _begin_internal_span("hiermoe_stage2_split_sizes_start")
    try:
        stage2_split_exchange = _start_exchange_split_sizes_many(
            [stage2_unique_send_splits, stage2_assignment_send_splits],
            groups.stage2_group,
            hidden_states.device,
        )
    finally:
        _end_internal_span(span)
    internal_start = _hiermoe_internal_event()
    stage2_send_meta_weights = _pack_meta_weights(stage2_send_meta, stage2_send_weights)
    _finish_hiermoe_internal_event(
        internal_timing_events,
        "stage2_meta_pack",
        internal_start,
    )
    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_stage2_split_sizes")
    try:
        stage2_unique_recv_splits, stage2_assignment_recv_splits = stage2_split_exchange.wait()
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            internal_timing_events,
            "stage2_split_wait",
            internal_start,
        )
    stage2_send_hidden = _mark_backward_a2a_input(
        stage2_send_hidden,
        backward_internal_timing_events,
        "backward_dispatch_stage2_a2a",
        layer_key,
    )
    if pipeline_manager is not None:
        pipeline_manager.open_pipeline_planner_prepare_window(layer_key, 1)
    stage2_a2a_start = _hiermoe_internal_event()
    recv_hidden, recv_meta_weights = _call_all_to_all_pair(
        groups.stage2_group,
        stage2_send_hidden,
        stage2_send_meta_weights,
        stage2_unique_recv_splits,
        stage2_unique_send_splits,
        stage2_assignment_recv_splits,
        stage2_assignment_send_splits,
    )
    stage2_a2a_end = _hiermoe_internal_event()
    if stage2_a2a_start is not None and stage2_a2a_end is not None and internal_timing_events is not None:
        internal_timing_events["stage2_a2a"] = (stage2_a2a_start, stage2_a2a_end)
    if pipeline_manager is not None:
        pipeline_manager.close_pipeline_planner_prepare_window(layer_key, 1)
    recv_hidden = _mark_backward_a2a_output(
        recv_hidden,
        backward_internal_timing_events,
        "backward_dispatch_stage2_a2a",
        layer_key,
    )
    internal_start = _hiermoe_internal_event()
    recv_meta, recv_weights = _unpack_meta_weights(recv_meta_weights, meta_cols=1, weight_dtype=routing_weights.dtype)
    _finish_hiermoe_internal_event(
        internal_timing_events,
        "stage2_meta_unpack",
        internal_start,
    )

    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_dispatch_finalize")
    try:
        relay_group_ranks = _repeat_ranks(stage2_assignment_recv_splits, hidden_states.device)
        if recv_meta.numel() == 0:
            local_expert_ids = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            permuted_tokens = _empty_2d(
                0,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            source_token_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            recv_unique_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            unsort_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            tokens_per_local_expert = torch.zeros(
                (num_local_experts,),
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            recv_offsets = _split_offsets(stage2_unique_recv_splits, hidden_states.device)
            recv_unique_ordinal, local_expert_ids = _unpack_nonnegative_pair(recv_meta[:, 0], num_local_experts)
            recv_unique_indices = recv_offsets.index_select(0, relay_group_ranks) + recv_unique_ordinal
            source_token_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            expert_sort_indices, unsort_indices, tokens_per_local_expert = _local_expert_sort_indices(
                local_expert_ids,
                num_local_experts,
                hidden_states.device,
                build_unsort=False,
            )
            recv_unique_indices = recv_unique_indices.index_select(0, expert_sort_indices)
            recv_weights = recv_weights.index_select(0, expert_sort_indices)
            permuted_tokens = recv_hidden.index_select(0, recv_unique_indices)
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            internal_timing_events,
            "dispatch_finalize",
            internal_start,
        )
    original_assignments = int(selected_experts.numel())
    dedup_tokens = int(sum(stage1_unique_send_splits))
    dedup_ratio = 1.0 - (dedup_tokens / max(1, original_assignments))
    ctx = RankDedupDispatchContext(
        ep_group=ep_group,
        stage1_group=groups.stage1_group,
        stage2_group=groups.stage2_group,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_tokens=int(hidden_states.shape[0]),
        num_local_experts=num_local_experts,
        hidden_size=int(hidden_states.shape[-1]),
        unique_send_splits=stage1_unique_send_splits,
        unique_recv_splits=stage1_unique_recv_splits,
        assignment_send_splits=stage2_assignment_send_splits,
        assignment_recv_splits=stage2_assignment_recv_splits,
        local_unique_token_indices=local_unique_token_indices,
        recv_source_token_indices=source_token_indices,
        recv_assignment_weights=recv_weights,
        unsort_indices=unsort_indices,
        selected_dim=selected_dim,
        dedup_ratio_dispatch=dedup_ratio,
        dedup_ratio_combine=dedup_ratio,
        mode="hierarchical",
        recv_unique_indices=recv_unique_indices,
        stage1_unique_send_splits=stage1_unique_send_splits,
        stage1_unique_recv_splits=stage1_unique_recv_splits,
        stage1_assignment_send_splits=stage1_assignment_send_splits,
        stage1_assignment_recv_splits=stage1_assignment_recv_splits,
        stage2_unique_send_splits=stage2_unique_send_splits,
        stage2_unique_recv_splits=stage2_unique_recv_splits,
        stage2_assignment_send_splits=stage2_assignment_send_splits,
        stage2_assignment_recv_splits=stage2_assignment_recv_splits,
        stage2_send_stage1_unique_indices=stage2_send_stage1_unique_indices,
        internal_timing_events=internal_timing_events,
        backward_internal_timing_events=backward_internal_timing_events,
        layer_key=layer_key,
    )
    return permuted_tokens, ctx, tokens_per_local_expert


def _hierarchical3d_dedup_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    selected_dim: int,
    intra_size: int,
    mid_size: int,
    physical_experts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RankDedupDispatchContext, torch.Tensor]:
    groups = _get_hierarchical3d_process_groups(ep_group, ep_size, ep_rank, intra_size, mid_size)
    internal_timing_events = {} if _HIERMOE_INTERNAL_TIMING else None
    span = _begin_internal_span("hiermoe_stage1_3d_payload_build")
    try:
        (
            stage1_send_hidden,
            stage1_send_meta,
            stage1_send_weights,
            local_unique_token_indices,
            stage1_unique_send_splits,
            stage1_assignment_send_splits,
        ) = _build_stage1_3d_payload(
            hidden_states,
            selected_experts,
            routing_weights,
            num_experts,
            ep_size,
            intra_size,
            mid_size,
            physical_experts=physical_experts,
        )
    finally:
        _end_internal_span(span)

    span = _begin_internal_span("hiermoe_stage1_3d_split_sizes_start")
    try:
        stage1_split_exchange = _start_exchange_split_sizes_many(
            [stage1_unique_send_splits, stage1_assignment_send_splits],
            groups.stage1_group,
            hidden_states.device,
        )
    finally:
        _end_internal_span(span)
    stage1_send_meta_weights = _pack_meta_weights(stage1_send_meta, stage1_send_weights)
    span = _begin_internal_span("hiermoe_stage1_3d_split_sizes")
    try:
        stage1_unique_recv_splits, stage1_assignment_recv_splits = stage1_split_exchange.wait()
    finally:
        _end_internal_span(span)
    stage1_a2a_start = _hiermoe_internal_event()
    stage1_recv_hidden, stage1_recv_meta_weights = _call_all_to_all_pair(
        groups.stage1_group,
        stage1_send_hidden,
        stage1_send_meta_weights,
        stage1_unique_recv_splits,
        stage1_unique_send_splits,
        stage1_assignment_recv_splits,
        stage1_assignment_send_splits,
    )
    _finish_hiermoe_internal_event(internal_timing_events, "stage1_a2a", stage1_a2a_start)
    stage1_recv_meta, stage1_recv_weights = _unpack_meta_weights(
        stage1_recv_meta_weights, meta_cols=2, weight_dtype=routing_weights.dtype
    )

    span = _begin_internal_span("hiermoe_stage2_3d_payload_build")
    try:
        (
            stage2_send_hidden,
            stage2_send_meta,
            stage2_send_weights,
            stage2_send_stage1_unique_indices,
            stage2_unique_send_splits,
            stage2_assignment_send_splits,
        ) = _build_stage2_3d_payload(
            stage1_recv_hidden,
            stage1_recv_meta,
            stage1_recv_weights,
            stage1_unique_recv_splits,
            stage1_assignment_recv_splits,
            intra_size,
            mid_size,
            num_local_experts,
        )
    finally:
        _end_internal_span(span)

    span = _begin_internal_span("hiermoe_stage2_3d_split_sizes_start")
    try:
        stage2_split_exchange = _start_exchange_split_sizes_many(
            [stage2_unique_send_splits, stage2_assignment_send_splits],
            groups.stage2_group,
            hidden_states.device,
        )
    finally:
        _end_internal_span(span)
    stage2_send_meta_weights = _pack_meta_weights(stage2_send_meta, stage2_send_weights)
    span = _begin_internal_span("hiermoe_stage2_3d_split_sizes")
    try:
        stage2_unique_recv_splits, stage2_assignment_recv_splits = stage2_split_exchange.wait()
    finally:
        _end_internal_span(span)
    stage2_a2a_start = _hiermoe_internal_event()
    stage2_recv_hidden, stage2_recv_meta_weights = _call_all_to_all_pair(
        groups.stage2_group,
        stage2_send_hidden,
        stage2_send_meta_weights,
        stage2_unique_recv_splits,
        stage2_unique_send_splits,
        stage2_assignment_recv_splits,
        stage2_assignment_send_splits,
    )
    _finish_hiermoe_internal_event(internal_timing_events, "stage2_a2a", stage2_a2a_start)
    stage2_recv_meta, stage2_recv_weights = _unpack_meta_weights(
        stage2_recv_meta_weights, meta_cols=1, weight_dtype=routing_weights.dtype
    )

    span = _begin_internal_span("hiermoe_stage3_3d_payload_build")
    try:
        (
            stage3_send_hidden,
            stage3_send_meta,
            stage3_send_weights,
            stage3_send_stage2_unique_indices,
            stage3_unique_send_splits,
            stage3_assignment_send_splits,
        ) = _build_stage3_3d_payload(
            stage2_recv_hidden,
            stage2_recv_meta,
            stage2_recv_weights,
            stage2_unique_recv_splits,
            stage2_assignment_recv_splits,
            intra_size,
            num_local_experts,
        )
    finally:
        _end_internal_span(span)

    span = _begin_internal_span("hiermoe_stage3_3d_split_sizes_start")
    try:
        stage3_split_exchange = _start_exchange_split_sizes_many(
            [stage3_unique_send_splits, stage3_assignment_send_splits],
            groups.stage3_group,
            hidden_states.device,
        )
    finally:
        _end_internal_span(span)
    stage3_send_meta_weights = _pack_meta_weights(stage3_send_meta, stage3_send_weights)
    span = _begin_internal_span("hiermoe_stage3_3d_split_sizes")
    try:
        stage3_unique_recv_splits, stage3_assignment_recv_splits = stage3_split_exchange.wait()
    finally:
        _end_internal_span(span)
    stage3_a2a_start = _hiermoe_internal_event()
    recv_hidden, recv_meta_weights = _call_all_to_all_pair(
        groups.stage3_group,
        stage3_send_hidden,
        stage3_send_meta_weights,
        stage3_unique_recv_splits,
        stage3_unique_send_splits,
        stage3_assignment_recv_splits,
        stage3_assignment_send_splits,
    )
    _finish_hiermoe_internal_event(internal_timing_events, "stage3_a2a", stage3_a2a_start)
    recv_meta, recv_weights = _unpack_meta_weights(recv_meta_weights, meta_cols=1, weight_dtype=routing_weights.dtype)

    span = _begin_internal_span("hiermoe_dispatch_3d_finalize")
    try:
        relay_local_ranks = _repeat_ranks(stage3_assignment_recv_splits, hidden_states.device)
        if recv_meta.numel() == 0:
            local_expert_ids = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            permuted_tokens = _empty_2d(
                0,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            recv_unique_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            unsort_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            tokens_per_local_expert = torch.zeros(
                (num_local_experts,),
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            recv_offsets = _split_offsets(stage3_unique_recv_splits, hidden_states.device)
            recv_unique_ordinal, local_expert_ids = _unpack_nonnegative_pair(recv_meta[:, 0], num_local_experts)
            recv_unique_indices = recv_offsets.index_select(0, relay_local_ranks) + recv_unique_ordinal
            expert_sort_indices, unsort_indices, tokens_per_local_expert = _local_expert_sort_indices(
                local_expert_ids,
                num_local_experts,
                hidden_states.device,
                build_unsort=False,
            )
            recv_unique_indices = recv_unique_indices.index_select(0, expert_sort_indices)
            recv_weights = recv_weights.index_select(0, expert_sort_indices)
            permuted_tokens = recv_hidden.index_select(0, recv_unique_indices)
    finally:
        _end_internal_span(span)

    original_assignments = int(selected_experts.numel())
    dedup_tokens = int(sum(stage1_unique_send_splits))
    dedup_ratio = 1.0 - (dedup_tokens / max(1, original_assignments))
    ctx = RankDedupDispatchContext(
        ep_group=ep_group,
        stage1_group=groups.stage1_group,
        stage2_group=groups.stage2_group,
        stage3_group=groups.stage3_group,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_tokens=int(hidden_states.shape[0]),
        num_local_experts=num_local_experts,
        hidden_size=int(hidden_states.shape[-1]),
        unique_send_splits=stage1_unique_send_splits,
        unique_recv_splits=stage1_unique_recv_splits,
        assignment_send_splits=stage3_assignment_send_splits,
        assignment_recv_splits=stage3_assignment_recv_splits,
        local_unique_token_indices=local_unique_token_indices,
        recv_source_token_indices=torch.empty((0,), dtype=torch.long, device=hidden_states.device),
        recv_assignment_weights=recv_weights,
        unsort_indices=unsort_indices,
        selected_dim=selected_dim,
        dedup_ratio_dispatch=dedup_ratio,
        dedup_ratio_combine=dedup_ratio,
        mode="hierarchical3d",
        recv_unique_indices=recv_unique_indices,
        stage1_unique_send_splits=stage1_unique_send_splits,
        stage1_unique_recv_splits=stage1_unique_recv_splits,
        stage1_assignment_send_splits=stage1_assignment_send_splits,
        stage1_assignment_recv_splits=stage1_assignment_recv_splits,
        stage2_unique_send_splits=stage2_unique_send_splits,
        stage2_unique_recv_splits=stage2_unique_recv_splits,
        stage2_assignment_send_splits=stage2_assignment_send_splits,
        stage2_assignment_recv_splits=stage2_assignment_recv_splits,
        stage2_send_stage1_unique_indices=stage2_send_stage1_unique_indices,
        stage3_unique_send_splits=stage3_unique_send_splits,
        stage3_unique_recv_splits=stage3_unique_recv_splits,
        stage3_send_stage2_unique_indices=stage3_send_stage2_unique_indices,
        internal_timing_events=internal_timing_events,
    )
    return permuted_tokens, ctx, tokens_per_local_expert


def rank_dedup_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
    layer_key: str | None = None,
    placement_already_applied: bool = False,
) -> tuple[torch.Tensor, RankDedupDispatchContext, torch.Tensor]:
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    hidden_states = _mark_fixed_pipeline_dispatch_input(hidden_states, layer_key)
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
    if num_experts % ep_size != 0:
        raise ValueError(f"Number of experts ({num_experts}) must be divisible by EP size ({ep_size}).")
    num_local_experts = num_experts // ep_size
    state = get_hiermoe_state()
    physical_experts = selected_experts
    dispatch_num_experts = num_experts
    if (
        state is not None
        and state.placement_mapping_enabled
        and state.expert_swap
        and state.expert_swap_manager is not None
    ):
        if layer_key is None:
            raise RuntimeError(
                "train.hiermoe.expert_swap=true requires fused MoE dispatch to pass a registered layer_key. "
                "Use the OpSlot MoE experts adapter or disable train.hiermoe.expert_swap for direct fused_moe_forward "
                "callers that do not expose their expert module identity."
            )
        if not state.expert_swap_manager.has_layer(layer_key):
            raise RuntimeError(
                f"train.hiermoe.expert_swap=true received unregistered MoE layer_key={layer_key!r}. "
                "Expert Swap cannot safely map logical experts to physical placement for this layer."
            )
        if state.fixed_pipeline_overlap:
            state.expert_swap_manager.wait_pipeline_migration_before_layer(layer_key)
        if state.expert_swap_mode == "layer" and state.layer_swap_forward_enabled and not placement_already_applied:
            state.expert_swap_pair = state.expert_swap_manager.maybe_swap_layer_on_routing(
                layer_key=layer_key,
                selected_experts=selected_experts,
                hidden_size=hidden_states.shape[-1],
                bytes_per_element=hidden_states.element_size(),
                step=state.current_step,
            )
        elif state.layer_swap_forward_enabled and not placement_already_applied:
            state.expert_swap_manager.record_routing(
                layer_key=layer_key,
                selected_experts=selected_experts,
                hidden_size=hidden_states.shape[-1],
                bytes_per_element=hidden_states.element_size(),
                step=state.current_step,
            )
            state.expert_swap_manager.mark_route_step(layer_key, state.current_step)
        mapping_span = _begin_internal_span("hiermoe_logical_to_physical_mapping")
        try:
            physical_experts = state.expert_swap_manager.map_logical_to_physical(
                layer_key,
                selected_experts,
                checkpoint_recompute=bool(getattr(state, "checkpoint_recompute_enabled", False)),
                checkpoint_replay=getattr(state, "checkpoint_route_replay", None),
            )
        finally:
            _end_internal_span(mapping_span)
        if state.layer_swap_forward_enabled and not placement_already_applied:
            state.expert_swap_manager.record_forward_physical_routes(layer_key, physical_experts)
        dispatch_num_experts = state.expert_swap_manager.num_physical_slots(layer_key, num_experts)
    num_local_experts = dispatch_num_experts // ep_size
    selected_dim = _select_dimension(
        selected_experts=physical_experts,
        num_experts=dispatch_num_experts,
        hidden_size=hidden_states.shape[-1],
        bytes_per_element=hidden_states.element_size(),
        group=ep_group,
    )
    if (
        state is not None
        and state.placement_mapping_enabled
        and state.route_capture_forward_enabled
        and route_capture_enabled()
    ):
        capture_mapping = None
        capture_slot_layout = None
        if state.expert_swap_manager is not None and layer_key is not None:
            layer = state.expert_swap_manager.layers.get(layer_key)
            if layer is not None and layer.slot_layout_enabled and route_capture_mode() != "local":
                raise RuntimeError(
                    "Global HierMoE route-oracle capture requires a baseline run without redundant expert slots."
                )
            if layer is not None:
                capture_mapping = layer.logical_to_physical
                capture_slot_layout = layer.slot_to_logical
        maybe_capture_route_snapshot(
            selected_experts=selected_experts,
            num_experts=num_experts,
            hidden_size=hidden_states.shape[-1],
            bytes_per_element=hidden_states.element_size(),
            ep_group=ep_group,
            hierarchy=state.hierarchy,
            layer_key=layer_key,
            step=state.current_step,
            logical_to_physical=capture_mapping,
            slot_to_logical=capture_slot_layout,
            smooth_max_gamma=(
                state.expert_swap_manager.smooth_max_gamma if state.expert_swap_manager is not None else 10.0
            ),
            selected_dim=selected_dim,
        )
    if state is not None and state.fixed_pipeline_overlap and selected_dim != 2:
        raise RuntimeError(
            "HierMoE fixed six-window pipeline requires two-level rank/node communication. "
            f"The runtime selected {selected_dim} communication levels."
        )
    hierarchical3d_sizes = _hierarchical3d_sizes(ep_size, selected_dim)
    if hierarchical3d_sizes is not None:
        intra_size, mid_size = hierarchical3d_sizes
        result = _hierarchical3d_dedup_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            num_experts=dispatch_num_experts,
            ep_group=ep_group,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_local_experts=dispatch_num_experts // ep_size,
            selected_dim=selected_dim,
            intra_size=intra_size,
            mid_size=mid_size,
            physical_experts=physical_experts,
        )
        if state is not None and state.expert_swap_manager is not None and layer_key is not None:
            state.expert_swap_manager.record_local_expert_token_counts(layer_key, result[2])
        return _mark_fixed_pipeline_dispatch_output(result, layer_key)

    intra_size = _hierarchical_intra_size(ep_size, selected_dim)
    if intra_size is not None:
        result = _hierarchical_dedup_dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            num_experts=dispatch_num_experts,
            ep_group=ep_group,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_local_experts=dispatch_num_experts // ep_size,
            selected_dim=selected_dim,
            intra_size=intra_size,
            physical_experts=physical_experts,
            layer_key=layer_key,
        )
        if state is not None and state.expert_swap_manager is not None and layer_key is not None:
            state.expert_swap_manager.record_local_expert_token_counts(layer_key, result[2])
        return _mark_fixed_pipeline_dispatch_output(result, layer_key)

    (
        send_hidden,
        send_meta,
        send_weights,
        local_unique_token_indices,
        unique_send_splits,
        assignment_send_splits,
        _target_ranks,
    ) = _build_local_payload(
        hidden_states,
        selected_experts,
        routing_weights,
        dispatch_num_experts,
        ep_size,
        physical_experts=physical_experts,
    )
    internal_timing_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None = (
        {} if _HIERMOE_INTERNAL_TIMING else None
    )

    split_exchange = _start_exchange_split_sizes_many(
        [unique_send_splits, assignment_send_splits], ep_group, hidden_states.device
    )

    send_meta_weights = _pack_meta_weights(send_meta, send_weights)
    unique_recv_splits, assignment_recv_splits = split_exchange.wait()
    rank_a2a_start = _hiermoe_internal_event()
    recv_hidden, recv_meta_weights = _call_all_to_all_pair(
        ep_group,
        send_hidden,
        send_meta_weights,
        unique_recv_splits,
        unique_send_splits,
        assignment_recv_splits,
        assignment_send_splits,
    )
    _finish_hiermoe_internal_event(internal_timing_events, "stage2_a2a", rank_a2a_start)
    recv_meta, recv_weights = _unpack_meta_weights(recv_meta_weights, meta_cols=3, weight_dtype=routing_weights.dtype)

    source_ranks = _repeat_ranks(assignment_recv_splits, hidden_states.device)
    if recv_meta.numel() == 0:
        local_expert_ids = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
        assignment_tokens = _empty_2d(
            0,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        source_token_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
    else:
        recv_offsets = _split_offsets(unique_recv_splits, hidden_states.device)
        unique_indices = recv_offsets.index_select(0, source_ranks) + recv_meta[:, 0].to(torch.long)
        assignment_tokens = recv_hidden.index_select(0, unique_indices)
        local_expert_ids = recv_meta[:, 1].to(torch.long)
        source_token_indices = recv_meta[:, 2].to(torch.long)

    permuted_tokens, unsort_indices, tokens_per_local_expert, _expert_sort_indices = _sort_assignments_by_local_expert(
        assignment_tokens, local_expert_ids, num_local_experts
    )
    original_assignments = int(selected_experts.numel())
    dedup_tokens = int(sum(unique_send_splits))
    dedup_ratio = 1.0 - (dedup_tokens / max(1, original_assignments))

    ctx = RankDedupDispatchContext(
        ep_group=ep_group,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_tokens=int(hidden_states.shape[0]),
        num_local_experts=num_local_experts,
        hidden_size=int(hidden_states.shape[-1]),
        unique_send_splits=unique_send_splits,
        unique_recv_splits=unique_recv_splits,
        assignment_send_splits=assignment_send_splits,
        assignment_recv_splits=assignment_recv_splits,
        local_unique_token_indices=local_unique_token_indices,
        recv_source_token_indices=source_token_indices,
        recv_assignment_weights=recv_weights,
        unsort_indices=unsort_indices,
        selected_dim=selected_dim,
        dedup_ratio_dispatch=dedup_ratio,
        dedup_ratio_combine=dedup_ratio,
        internal_timing_events=internal_timing_events,
        layer_key=layer_key,
    )
    if state is not None and state.expert_swap_manager is not None and layer_key is not None:
        state.expert_swap_manager.record_local_expert_token_counts(layer_key, tokens_per_local_expert)
    return _mark_fixed_pipeline_dispatch_output((permuted_tokens, ctx, tokens_per_local_expert), layer_key)


def _aggregate_weighted_outputs(
    weighted_outputs: torch.Tensor,
    source_token_indices: torch.Tensor,
    split_sizes: list[int],
    key_stride: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    if weighted_outputs.numel() == 0:
        return (
            _empty_2d(
                0,
                weighted_outputs.shape[-1],
                dtype=weighted_outputs.dtype,
                device=weighted_outputs.device,
            ),
            [],
        )

    source_ranks = _repeat_ranks(split_sizes, weighted_outputs.device)
    # Source ranks may have different token counts (especially for multimodal
    # batches), so the receiving rank's local token count is not a safe radix.
    # Source-token metadata is int32; reserving its full unsigned range keeps
    # rank-token keys collision-free without synchronizing the accelerator.
    key_stride = max(1 << 32, int(key_stride) if key_stride is not None else 0)
    combine_keys = source_ranks * key_stride + source_token_indices.to(torch.long)
    unique_keys, inverse = torch.unique_consecutive(combine_keys, return_inverse=True)
    unique_source_ranks = torch.div(unique_keys, key_stride, rounding_mode="floor")
    output_splits = _split_tensor_to_list(torch.bincount(unique_source_ranks, minlength=len(split_sizes)))
    accum = _index_add_dim0_cast_output(
        weighted_outputs,
        inverse,
        int(unique_keys.numel()),
    )
    return accum, output_splits


def _hierarchical_dedup_combine(expert_outputs: torch.Tensor, ctx: RankDedupDispatchContext) -> torch.Tensor:
    pipeline_manager = _fixed_pipeline_manager(ctx.layer_key)
    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_combine_stage2_accum")
    try:
        weighted_outputs = expert_outputs * ctx.recv_assignment_weights.to(expert_outputs.dtype)
        stage2_recv_count = sum(ctx.stage2_unique_recv_splits or [])
        if ctx.recv_unique_indices is None:
            stage2_accum = torch.zeros(
                (stage2_recv_count, ctx.hidden_size),
                dtype=weighted_outputs.dtype,
                device=weighted_outputs.device,
            )
        else:
            stage2_accum = _index_add_dim0_cast_output(
                weighted_outputs,
                ctx.recv_unique_indices,
                stage2_recv_count,
            )
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            ctx.internal_timing_events,
            "combine_stage2_accum",
            internal_start,
        )
    # The weighted expert output is no longer consumed after the stage-2
    # accumulation.  Drop its Python reference before materializing the BF16
    # communication buffer; for long-sequence workloads both tensors can be
    # hundreds of MiB and otherwise overlap until this function returns.
    del weighted_outputs
    stage2_send = _mark_backward_a2a_input(
        stage2_accum,
        ctx.backward_internal_timing_events,
        "backward_combine_stage2_a2a",
        ctx.layer_key,
    )
    del stage2_accum
    if pipeline_manager is not None:
        pipeline_manager.open_pipeline_planner_prepare_window(ctx.layer_key, 2)
    combine_stage2_a2a_start = _hiermoe_internal_event()
    relay_partials = _call_all_to_all(
        ctx.stage2_group,
        stage2_send,
        ctx.stage2_unique_send_splits or [],
        ctx.stage2_unique_recv_splits or [],
    )
    del stage2_send
    combine_stage2_a2a_end = _hiermoe_internal_event()
    if pipeline_manager is not None:
        pipeline_manager.close_pipeline_planner_prepare_window(ctx.layer_key, 2)
    if (
        ctx.internal_timing_events is not None
        and combine_stage2_a2a_start is not None
        and combine_stage2_a2a_end is not None
    ):
        ctx.internal_timing_events["combine_stage2_a2a"] = (
            combine_stage2_a2a_start,
            combine_stage2_a2a_end,
        )
    relay_partials = _mark_backward_a2a_output(
        relay_partials,
        ctx.backward_internal_timing_events,
        "backward_combine_stage2_a2a",
        ctx.layer_key,
    )

    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_combine_stage1_accum")
    try:
        stage1_recv_count = sum(ctx.stage1_unique_recv_splits or [])
        if ctx.stage2_send_stage1_unique_indices is None:
            stage1_accum = torch.zeros(
                (stage1_recv_count, ctx.hidden_size),
                dtype=relay_partials.dtype,
                device=relay_partials.device,
            )
        else:
            stage1_accum = _index_add_dim0_cast_output(
                relay_partials,
                ctx.stage2_send_stage1_unique_indices,
                stage1_recv_count,
            )
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            ctx.internal_timing_events,
            "combine_stage1_accum",
            internal_start,
        )

    # relay_partials has been fully reduced into stage1_accum.  Releasing it
    # before the FP32->payload cast avoids a full-size relay buffer overlapping
    # the stage-1 send allocation at the peak of checkpoint recomputation.
    del relay_partials
    stage1_send = _mark_backward_a2a_input(
        stage1_accum,
        ctx.backward_internal_timing_events,
        "backward_combine_stage1_a2a",
        ctx.layer_key,
    )
    del stage1_accum
    if pipeline_manager is not None:
        pipeline_manager.open_pipeline_planner_prepare_window(ctx.layer_key, 3)
    combine_stage1_a2a_start = _hiermoe_internal_event()
    partial_outputs = _call_all_to_all(
        ctx.stage1_group,
        stage1_send,
        ctx.stage1_unique_send_splits or [],
        ctx.stage1_unique_recv_splits or [],
    )
    del stage1_send
    combine_stage1_a2a_end = _hiermoe_internal_event()
    if pipeline_manager is not None:
        pipeline_manager.close_pipeline_planner_prepare_window(ctx.layer_key, 3)
    if (
        ctx.internal_timing_events is not None
        and combine_stage1_a2a_start is not None
        and combine_stage1_a2a_end is not None
    ):
        ctx.internal_timing_events["combine_stage1_a2a"] = (
            combine_stage1_a2a_start,
            combine_stage1_a2a_end,
        )
    partial_outputs = _mark_backward_a2a_output(
        partial_outputs,
        ctx.backward_internal_timing_events,
        "backward_combine_stage1_a2a",
        ctx.layer_key,
    )
    internal_start = _hiermoe_internal_event()
    span = _begin_internal_span("hiermoe_combine_final_accum")
    try:
        final_hidden_states = _index_add_dim0_cast_output(
            partial_outputs,
            ctx.local_unique_token_indices,
            ctx.num_local_tokens,
        )
    finally:
        _end_internal_span(span)
        _finish_hiermoe_internal_event(
            ctx.internal_timing_events,
            "combine_final_accum",
            internal_start,
        )
    output_dtype = partial_outputs.dtype
    del partial_outputs
    return final_hidden_states.to(output_dtype)


def _hierarchical3d_dedup_combine(expert_outputs: torch.Tensor, ctx: RankDedupDispatchContext) -> torch.Tensor:
    span = _begin_internal_span("hiermoe_combine_stage3_3d_accum")
    try:
        weighted_outputs = expert_outputs * ctx.recv_assignment_weights.to(expert_outputs.dtype)
        stage3_recv_count = sum(ctx.stage3_unique_recv_splits or [])
        if ctx.recv_unique_indices is None:
            stage3_accum = torch.zeros(
                (stage3_recv_count, ctx.hidden_size),
                dtype=weighted_outputs.dtype,
                device=weighted_outputs.device,
            )
        else:
            stage3_accum = _index_add_dim0_cast_output(
                weighted_outputs,
                ctx.recv_unique_indices,
                stage3_recv_count,
            )
    finally:
        _end_internal_span(span)
    combine_stage3_start = _hiermoe_internal_event()
    stage3_partials = _call_all_to_all(
        ctx.stage3_group,
        stage3_accum,
        ctx.stage3_unique_send_splits or [],
        ctx.stage3_unique_recv_splits or [],
    )

    _finish_hiermoe_internal_event(ctx.internal_timing_events, "combine_stage3_a2a", combine_stage3_start)
    span = _begin_internal_span("hiermoe_combine_stage2_3d_accum")
    try:
        stage2_recv_count = sum(ctx.stage2_unique_recv_splits or [])
        if ctx.stage3_send_stage2_unique_indices is None:
            stage2_accum = torch.zeros(
                (stage2_recv_count, ctx.hidden_size),
                dtype=stage3_partials.dtype,
                device=stage3_partials.device,
            )
        else:
            stage2_accum = _index_add_dim0_cast_output(
                stage3_partials,
                ctx.stage3_send_stage2_unique_indices,
                stage2_recv_count,
            )
    finally:
        _end_internal_span(span)
    combine_stage2_start = _hiermoe_internal_event()
    stage2_partials = _call_all_to_all(
        ctx.stage2_group,
        stage2_accum,
        ctx.stage2_unique_send_splits or [],
        ctx.stage2_unique_recv_splits or [],
    )

    _finish_hiermoe_internal_event(ctx.internal_timing_events, "combine_stage2_a2a", combine_stage2_start)
    span = _begin_internal_span("hiermoe_combine_stage1_3d_accum")
    try:
        stage1_recv_count = sum(ctx.stage1_unique_recv_splits or [])
        if ctx.stage2_send_stage1_unique_indices is None:
            stage1_accum = torch.zeros(
                (stage1_recv_count, ctx.hidden_size),
                dtype=stage2_partials.dtype,
                device=stage2_partials.device,
            )
        else:
            stage1_accum = _index_add_dim0_cast_output(
                stage2_partials,
                ctx.stage2_send_stage1_unique_indices,
                stage1_recv_count,
            )
    finally:
        _end_internal_span(span)
    combine_stage1_start = _hiermoe_internal_event()
    partial_outputs = _call_all_to_all(
        ctx.stage1_group,
        stage1_accum,
        ctx.stage1_unique_send_splits or [],
        ctx.stage1_unique_recv_splits or [],
    )

    _finish_hiermoe_internal_event(ctx.internal_timing_events, "combine_stage1_a2a", combine_stage1_start)
    span = _begin_internal_span("hiermoe_combine_final_3d_accum")
    try:
        final_hidden_states = _index_add_dim0_cast_output(
            partial_outputs,
            ctx.local_unique_token_indices,
            ctx.num_local_tokens,
        )
    finally:
        _end_internal_span(span)
    return final_hidden_states.to(partial_outputs.dtype)


def rank_dedup_combine(expert_outputs: torch.Tensor, ctx: RankDedupDispatchContext) -> torch.Tensor:
    expert_outputs = _mark_fixed_pipeline_expert_output(expert_outputs, ctx.layer_key)
    if ctx.mode == "hierarchical3d":
        return _hierarchical3d_dedup_combine(expert_outputs, ctx)
    if ctx.mode == "hierarchical":
        return _hierarchical_dedup_combine(expert_outputs, ctx)

    if expert_outputs.numel() == 0:
        assignment_order_outputs = expert_outputs
    else:
        assignment_order_outputs = expert_outputs.index_select(0, ctx.unsort_indices)

    weighted_outputs = assignment_order_outputs * ctx.recv_assignment_weights.to(assignment_order_outputs.dtype)
    combine_input, combine_input_splits = _aggregate_weighted_outputs(
        weighted_outputs,
        ctx.recv_source_token_indices,
        ctx.assignment_recv_splits,
    )
    rank_a2a_start = _hiermoe_internal_event()
    partial_outputs = _call_all_to_all(ctx.ep_group, combine_input, ctx.unique_send_splits, combine_input_splits)
    _finish_hiermoe_internal_event(ctx.internal_timing_events, "combine_stage2_a2a", rank_a2a_start)

    return _index_add_dim0_cast_output(
        partial_outputs,
        ctx.local_unique_token_indices,
        ctx.num_local_tokens,
    )
