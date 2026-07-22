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

from __future__ import annotations

import math
import os
import time
import zlib
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - older torch fallback
    DTensor = ()  # type: ignore[assignment]

from ....utils import logging
from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_device_type, get_torch_device
from .core_planner import (
    CORE_MOE_ALGORITHM_VERSION,
    CoReMoEPlanner,
    QuotaPolicyEntry,
    assign_tokens_to_copies_with_quota,
)
from .perf_model import HierMoEPerfModel
from .planner import (
    CurrentRoutePlanner,
    PlacementPlan,
    assign_tokens_to_copies,
    assign_tokens_to_mirrored_r2,
)
from .topology import Hierarchy


logger = logging.get_logger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default


def _env_candidate_shards(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.lower() == "auto":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using automatic candidate sharding.", name, raw)
        return 0


_MAX_SWAP_BUCKET_BYTES = _env_int("VEOMNI_HIERMOE_SWAP_BUCKET_MIB", 1024) * 1024 * 1024
_MAX_SWAP_WAVE_BYTES = _env_int("VEOMNI_HIERMOE_SWAP_WAVE_MIB", 2048) * 1024 * 1024
_EXACT_SINGLE_SWAP_MAX_EXPERTS = 256
_EXACT_SINGLE_SWAP_MAX_STATS_BYTES = 64 * 1024 * 1024
_SWAP_COST_CHUNK_CANDIDATES = _env_int("VEOMNI_HIERMOE_SWAP_COST_CHUNK_CANDIDATES", 96)
_ALL_CANDIDATE_PAIR_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.lower() in {"1", "true", "yes", "on", "y"}


_USE_FAST_2D_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_FAST_2D")
_USE_GLOBAL_2D_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_GLOBAL_2D")
_USE_GLOBAL_HIERARCHY_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_GLOBAL_HIERARCHY")
_SWAP_CANDIDATE_SHARDS = _env_candidate_shards("VEOMNI_HIERMOE_SWAP_CANDIDATE_SHARDS")
_DEBUG_REDUNDANT_COPY_STATS = _env_flag("VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS")
_DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS = _env_int("VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS", 2)
_FIXED_R2_LAYOUT = _env_flag("VEOMNI_HIERMOE_FIXED_R2_LAYOUT")


def _full_timing_range(section: str):
    if not _env_flag("VEOMNI_FULL_PROFILE_ENABLE"):
        return nullcontext()
    try:
        from ....utils.full_timing_profiler import get_active_full_timing_profiler

        profiler = get_active_full_timing_profiler()
    except Exception:
        profiler = None
    if profiler is None:
        return nullcontext()
    return profiler.cuda_range(section)


def _placement_timing_range(prefix: str | None, phase: str):
    if prefix is None:
        return nullcontext()
    return _full_timing_range(f"{prefix}_{phase}")


@dataclass
class ExpertPlacement:
    logical_to_physical: tuple[int, ...]

    @classmethod
    def identity(cls, num_experts: int) -> "ExpertPlacement":
        return cls(logical_to_physical=tuple(range(num_experts)))

    def swapped(self, lhs: int, rhs: int) -> "ExpertPlacement":
        mapping = list(self.logical_to_physical)
        mapping[lhs], mapping[rhs] = mapping[rhs], mapping[lhs]
        return ExpertPlacement(logical_to_physical=tuple(mapping))


def _initial_slot_to_logical(
    num_experts: int,
    base_num_local_experts: int,
    slot_capacity_per_rank: int,
    ep_size: int,
) -> torch.Tensor:
    layout = torch.full((int(ep_size) * int(slot_capacity_per_rank),), -1, dtype=torch.long)
    for logical_expert in range(int(num_experts)):
        rank, local_slot = divmod(logical_expert, int(base_num_local_experts))
        layout[int(rank) * int(slot_capacity_per_rank) + int(local_slot)] = int(logical_expert)
    return layout


def _canonical_physical_slots(
    num_experts: int,
    base_num_local_experts: int,
    slot_capacity_per_rank: int,
) -> torch.Tensor:
    canonical = torch.empty((int(num_experts),), dtype=torch.long)
    for logical_expert in range(int(num_experts)):
        rank, local_slot = divmod(logical_expert, int(base_num_local_experts))
        canonical[logical_expert] = int(rank) * int(slot_capacity_per_rank) + int(local_slot)
    return canonical


@dataclass(frozen=True)
class _PendingLayerTiming:
    step: int
    selected_experts: torch.Tensor
    slot_to_logical: torch.Tensor
    dispatch_start: AcceleratorEvent
    dispatch_end: AcceleratorEvent
    compute_start: AcceleratorEvent
    compute_end: AcceleratorEvent
    combine_start: AcceleratorEvent
    combine_end: AcceleratorEvent


@dataclass(frozen=True)
class _PlannerCalibration:
    source_step: int
    communication_scale: float
    forward_compute_per_assignment: float


@dataclass
class ExpertLayerState:
    key: str
    module_id: int
    num_experts: int
    base_num_local_experts: int
    num_local_experts: int
    gate_up_proj: torch.nn.Parameter
    down_proj: torch.nn.Parameter
    logical_to_physical: torch.Tensor
    slot_to_logical: torch.Tensor | None = None
    canonical_physical_slots: torch.Tensor | None = None
    latest_selected_experts: torch.Tensor | None = None
    latest_route_step: int = -1
    last_planned_step: int = -1
    accumulated_tokens_per_local_expert: torch.Tensor | None = None
    latest_hidden_size: int = 0
    latest_bytes_per_element: int = 0
    is_identity: bool = True
    _device_mapping_cache: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    _device_slot_layout_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    _device_redundant_groups_cache: dict[torch.device, tuple[tuple[int, torch.Tensor], ...]] = field(
        default_factory=dict
    )
    _redundant_copy_groups_cache: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    _replica_grad_schedule_cache: _ReplicaGradSchedule | None = None
    pending_timing: _PendingLayerTiming | None = None
    planner_calibration: _PlannerCalibration | None = None
    last_plan: PlacementPlan | None = None
    pending_physical_routes: torch.Tensor | None = None
    pending_route_data_ptr: int = 0
    active_quota_policy: tuple[QuotaPolicyEntry, ...] = ()
    fixed_r2_layout: bool = False

    def invalidate_cache(self) -> None:
        self._device_mapping_cache.clear()
        self._device_slot_layout_cache.clear()
        self._device_redundant_groups_cache.clear()
        self._redundant_copy_groups_cache = None
        self._replica_grad_schedule_cache = None

    def refresh_identity(self) -> None:
        if self.slot_to_logical is not None:
            expected = _initial_slot_to_logical(
                self.num_experts,
                self.base_num_local_experts,
                self.num_local_experts,
                self.num_experts // self.base_num_local_experts,
            )
            self.is_identity = torch.equal(self.slot_to_logical.cpu(), expected)
        else:
            identity = torch.arange(self.num_experts, dtype=torch.long)
            self.is_identity = torch.equal(self.logical_to_physical.cpu(), identity)

    def mapping_for_device(self, device: torch.device) -> torch.Tensor:
        cached = self._device_mapping_cache.get(device)
        if cached is None:
            cached = self.logical_to_physical.to(device=device, non_blocking=True)
            self._device_mapping_cache[device] = cached
        return cached

    @property
    def slot_layout_enabled(self) -> bool:
        return self.slot_to_logical is not None

    @property
    def num_physical_slots(self) -> int:
        return self.num_local_experts * (self.num_experts // self.base_num_local_experts)

    def copy_slots_for_device(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._device_slot_layout_cache.get(device)
        if cached is not None:
            return cached
        if self.slot_to_logical is None:
            mapping = self.mapping_for_device(device)
            copy_slots = mapping.view(self.num_experts, 1)
            copy_mask = torch.ones_like(copy_slots, dtype=torch.bool)
            cached = (copy_slots, copy_mask)
            self._device_slot_layout_cache[device] = cached
            return cached

        slot_to_logical_cpu = self.slot_to_logical.detach().cpu()
        counts = torch.bincount(slot_to_logical_cpu[slot_to_logical_cpu >= 0], minlength=self.num_experts)
        max_copies = max(1, int(counts.max().item()))
        copy_slots_cpu = torch.full((self.num_experts, max_copies), -1, dtype=torch.long)
        copy_mask_cpu = torch.zeros((self.num_experts, max_copies), dtype=torch.bool)
        offsets = torch.zeros((self.num_experts,), dtype=torch.long)
        for physical_slot, logical_expert in enumerate(slot_to_logical_cpu.tolist()):
            if logical_expert < 0:
                continue
            offset = int(offsets[logical_expert].item())
            copy_slots_cpu[logical_expert, offset] = int(physical_slot)
            copy_mask_cpu[logical_expert, offset] = True
            offsets[logical_expert] += 1
        cached = (
            copy_slots_cpu.to(device=device, non_blocking=True),
            copy_mask_cpu.to(device=device, non_blocking=True),
        )
        self._device_slot_layout_cache[device] = cached
        return cached

    def redundant_copy_groups(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        cached = self._redundant_copy_groups_cache
        if cached is not None:
            return cached
        if self.slot_to_logical is None:
            self._redundant_copy_groups_cache = ()
            return ()

        layout = self.slot_to_logical.detach().cpu()
        active = layout[layout >= 0]
        if active.numel() == 0:
            self._redundant_copy_groups_cache = ()
            return ()

        counts = torch.bincount(active, minlength=self.num_experts)
        groups: list[tuple[int, tuple[int, ...]]] = []
        for logical_expert in torch.nonzero(counts > 1, as_tuple=False).flatten().tolist():
            slots = torch.nonzero(layout == int(logical_expert), as_tuple=False).flatten().tolist()
            groups.append((int(logical_expert), tuple(int(slot) for slot in slots)))
        self._redundant_copy_groups_cache = tuple(groups)
        return self._redundant_copy_groups_cache

    def redundant_copy_groups_for_device(self, device: torch.device) -> tuple[tuple[int, torch.Tensor], ...]:
        cached = self._device_redundant_groups_cache.get(device)
        if cached is not None:
            return cached
        cached = tuple(
            (
                int(logical_expert),
                torch.tensor(slots, dtype=torch.long, device=device),
            )
            for logical_expert, slots in self.redundant_copy_groups()
        )
        self._device_redundant_groups_cache[device] = cached
        return cached


@dataclass(frozen=True)
class _SwapTensorEntry:
    tensor: torch.Tensor
    lhs_slot: int
    rhs_slot: int


@dataclass(frozen=True)
class _CoverTensorEntry:
    tensor: torch.Tensor
    src_slot: int
    dst_slot: int


@dataclass(frozen=True)
class _SlotOpCandidate:
    kind: str
    src_slot: int
    dst_slot: int

    def format(self) -> str:
        arrow = "<->" if self.kind == "swap" else "->"
        return f"{self.kind.upper()}({self.src_slot}{arrow}{self.dst_slot})"


@dataclass(frozen=True)
class _LayerSwapPlan:
    layer_key: str
    logical_lhs: int
    logical_rhs: int
    lhs_rank: int
    rhs_rank: int
    entries: tuple[_SwapTensorEntry, ...]


@dataclass
class _SwapStagingBuffer:
    send: torch.Tensor
    recv: torch.Tensor


@dataclass
class _PendingLayerSwap:
    layer_key: str
    works: tuple[Any, ...]
    unpack: tuple[Any, ...]
    device: torch.device
    timing_context: Any


@dataclass(frozen=True)
class _OptimizerParamBinding:
    optimizer: Any
    group: dict[str, Any]


@dataclass(frozen=True)
class _RedundantGradBucketItem:
    local_grad: torch.Tensor
    local_slots: tuple[int, ...]
    shape: torch.Size
    numel: int


@dataclass
class _RedundantGradBucket:
    owner_rank: int
    copy_ranks: tuple[int, ...]
    items: tuple[_RedundantGradBucketItem, ...]
    send_buffer: torch.Tensor
    accum_buffer: torch.Tensor | None = None


@dataclass(frozen=True)
class _ReplicaGradGroup:
    logical_expert: int
    owner_rank: int
    copy_ranks: tuple[int, ...]
    local_slots: tuple[int, ...]


@dataclass
class _ReplicaGradSchedule:
    groups: tuple[_ReplicaGradGroup, ...]
    pairwise: bool


@dataclass(frozen=True)
class _ReplicaGradContribution:
    logical_expert: int
    param_index: int
    local_grad: torch.Tensor
    local_slots: tuple[int, ...]
    local_sum: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.local_sum.numel())


_SwapBucketItem = tuple[torch.Tensor, int, torch.Tensor, int, int]
_SlotStateItem = tuple[str, torch.Tensor]


def _is_dtensor(tensor: torch.Tensor) -> bool:
    return bool(DTensor) and isinstance(tensor, DTensor)


def _local_tensor_view(tensor: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(tensor):
        return tensor
    local = tensor.to_local()
    if tuple(local.shape) != tuple(tensor.shape):
        raise NotImplementedError(
            "HierMoE expert swap requires complete local expert tensors. "
            f"Got DTensor global shape {tuple(tensor.shape)} and local shape {tuple(local.shape)}."
        )
    return local


def _copy_tensor_attrs(src: torch.Tensor, dst: torch.Tensor, attrs: tuple[str, ...]) -> None:
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _expanded_local_parameter(param: torch.nn.Parameter, target_slots: int) -> torch.nn.Parameter:
    local = _local_tensor_view(param)
    if int(local.shape[0]) == int(target_slots):
        return param
    if int(local.shape[0]) > int(target_slots):
        raise ValueError(f"Cannot shrink HierMoE expert parameter from {tuple(local.shape)} to {target_slots} slots.")

    expanded = torch.empty(
        (int(target_slots), *tuple(local.shape[1:])),
        dtype=local.dtype,
        device=local.device,
    )
    if local.device.type != "meta":
        expanded[: local.shape[0]].copy_(local.detach())
        expanded[local.shape[0] :].zero_()
    new_param = torch.nn.Parameter(expanded, requires_grad=param.requires_grad)
    _copy_tensor_attrs(param, new_param, ("spec_info",))
    return new_param


def expand_redundant_expert_slots(model: nn.Module, *, ep_size: int, redundant_slot_increment_per_device: int) -> int:
    """Reserve empty local expert slots after EP slicing and before FSDP wrapping."""

    increment = max(0, int(redundant_slot_increment_per_device))
    if increment == 0 or int(ep_size) <= 1:
        return 0

    expanded_layers = 0
    for _key, module in model.named_modules():
        if not (
            hasattr(module, "num_experts")
            and hasattr(module, "gate_up_proj")
            and hasattr(module, "down_proj")
            and isinstance(module.gate_up_proj, torch.Tensor)
            and isinstance(module.down_proj, torch.Tensor)
        ):
            continue
        num_experts = int(module.num_experts)
        if num_experts % int(ep_size) != 0:
            raise ValueError(
                f"HierMoE redundant slots require num_experts={num_experts} divisible by ep_size={ep_size}."
            )
        base_slots = num_experts // int(ep_size)
        target_slots = base_slots + increment
        gate_local = _local_tensor_view(module.gate_up_proj)
        down_local = _local_tensor_view(module.down_proj)
        if int(gate_local.shape[0]) == target_slots and int(down_local.shape[0]) == target_slots:
            continue
        if int(gate_local.shape[0]) != base_slots or int(down_local.shape[0]) != base_slots:
            raise ValueError(
                "HierMoE redundant slot expansion must run immediately after EP slicing. "
                f"Expected {base_slots} local experts, got gate_up_proj={tuple(gate_local.shape)} "
                f"down_proj={tuple(down_local.shape)}."
            )
        module.gate_up_proj = _expanded_local_parameter(module.gate_up_proj, target_slots)
        module.down_proj = _expanded_local_parameter(module.down_proj, target_slots)
        expanded_layers += 1
    return expanded_layers


def _slot_tensor(tensor: torch.Tensor, slot: int) -> torch.Tensor:
    local = _local_tensor_view(tensor)
    return local.detach()[slot : slot + 1].contiguous()


def _empty_slot_like(tensor: torch.Tensor) -> torch.Tensor:
    local = _local_tensor_view(tensor)
    return torch.empty((0, *local.shape[1:]), dtype=local.dtype, device=local.device)


def _ep_global_rank(ep_group: dist.ProcessGroup | None, ep_rank: int) -> int:
    if ep_group is None:
        return int(ep_rank)
    try:
        return int(dist.get_global_rank(ep_group, int(ep_rank)))
    except (AttributeError, RuntimeError, ValueError):
        return int(ep_rank)


def _create_expert_swap_process_group(
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
) -> dist.ProcessGroup | None:
    if ep_group is None or ep_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    global_ranks = [_ep_global_rank(ep_group, rank) for rank in range(ep_size)]
    try:
        swap_group = dist.new_group(
            ranks=global_ranks,
            backend=dist.get_backend(ep_group),
            use_local_synchronization=True,
            group_desc="hiermoe_expert_swap",
        )
    except TypeError as error:
        raise RuntimeError(
            "HierMoE asynchronous expert swap requires PyTorch new_group support for "
            "use_local_synchronization and group_desc."
        ) from error
    if swap_group is None or swap_group == dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError("HierMoE failed to create the dedicated expert-swap process group.")
    # Eager initialization avoids the first P2P batch requiring every group rank.
    dist.barrier(group=swap_group)
    return swap_group


def _swap_local_slot(tensor: torch.Tensor, lhs_slot: int, rhs_slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    tmp = local_tensor.detach()[lhs_slot].clone()
    local_tensor.detach()[lhs_slot].copy_(local_tensor.detach()[rhs_slot])
    local_tensor.detach()[rhs_slot].copy_(tmp)


def _copy_local_slot(tensor: torch.Tensor, src_slot: int, dst_slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    local_tensor.detach()[dst_slot].copy_(local_tensor.detach()[src_slot])


def _zero_local_slot(tensor: torch.Tensor, slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    local_tensor.detach()[slot].zero_()


def _chunk_swap_bucket(bucket: list[_SwapBucketItem]) -> list[list[_SwapBucketItem]]:
    chunks: list[list[_SwapBucketItem]] = []
    current: list[_SwapBucketItem] = []
    current_nbytes = 0
    for item in bucket:
        item_nbytes = item[4]
        if current and current_nbytes + item_nbytes > _MAX_SWAP_BUCKET_BYTES:
            chunks.append(current)
            current = []
            current_nbytes = 0
        current.append(item)
        current_nbytes += item_nbytes
    if current:
        chunks.append(current)
    return chunks


def _pack_swap_chunk(chunk: list[_SwapBucketItem]) -> torch.Tensor:
    send_parts = [entry[2] for entry in chunk]
    return torch.cat(send_parts, dim=0) if len(send_parts) > 1 else send_parts[0]


def _swap_chunk_nbytes(chunk: list[_SwapBucketItem]) -> int:
    return sum(item[4] for item in chunk)


def _unpack_swap_chunk(recv_buffer: torch.Tensor, chunk: list[_SwapBucketItem]) -> None:
    offset = 0
    for local_tensor, local_slot, _send_view, numel, _nbytes in chunk:
        recv_view = recv_buffer[offset : offset + numel].view_as(local_tensor.detach()[local_slot])
        local_tensor.detach()[local_slot].copy_(recv_view)
        offset += numel


@torch.no_grad()
def _exchange_or_swap_slot_entries(
    entries: Iterable[_SwapTensorEntry],
    lhs_rank: int,
    rhs_rank: int,
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    entry_list = list(entries)
    if lhs_rank == rhs_rank:
        if ep_rank == lhs_rank:
            for entry in entry_list:
                _swap_local_slot(entry.tensor, entry.lhs_slot, entry.rhs_slot)
        return

    if ep_group is None or ep_size <= 1 or ep_rank not in (lhs_rank, rhs_rank):
        return

    peer_rank = rhs_rank if ep_rank == lhs_rank else lhs_rank
    peer_global_rank = _ep_global_rank(ep_group, peer_rank)

    buckets: dict[tuple[torch.device, torch.dtype], list[_SwapBucketItem]] = defaultdict(list)
    for entry in entry_list:
        local_slot = entry.lhs_slot if ep_rank == lhs_rank else entry.rhs_slot
        local_tensor = _local_tensor_view(entry.tensor)
        slot_view = local_tensor.detach()[local_slot]
        send_view = slot_view.contiguous().view(-1)
        numel = int(send_view.numel())
        nbytes = numel * int(send_view.element_size())
        buckets[(send_view.device, send_view.dtype)].append((local_tensor, int(local_slot), send_view, numel, nbytes))

    for bucket in buckets.values():
        for chunk in _chunk_swap_bucket(bucket):
            send_buffer = _pack_swap_chunk(chunk)
            recv_buffer = torch.empty_like(send_buffer)
            works = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send_buffer, peer_global_rank),
                    dist.P2POp(dist.irecv, recv_buffer, peer_global_rank),
                ]
            )
            for work in works:
                work.wait()

            _unpack_swap_chunk(recv_buffer, chunk)


@torch.no_grad()
def _exchange_or_swap_grouped_slot_entries(
    grouped_entries: dict[tuple[int, int], list[_SwapTensorEntry]],
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    remote_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for (lhs_rank, rhs_rank), entry_list in sorted(grouped_entries.items()):
        if lhs_rank == rhs_rank:
            if ep_rank == lhs_rank:
                for entry in entry_list:
                    _swap_local_slot(entry.tensor, entry.lhs_slot, entry.rhs_slot)
            continue

        if ep_group is None or ep_size <= 1 or ep_rank not in (lhs_rank, rhs_rank):
            continue

        peer_rank = rhs_rank if ep_rank == lhs_rank else lhs_rank
        peer_global_rank = _ep_global_rank(ep_group, peer_rank)
        for entry in entry_list:
            local_slot = entry.lhs_slot if ep_rank == lhs_rank else entry.rhs_slot
            local_tensor = _local_tensor_view(entry.tensor)
            slot_view = local_tensor.detach()[local_slot]
            send_view = slot_view.contiguous().view(-1)
            numel = int(send_view.numel())
            nbytes = numel * int(send_view.element_size())
            remote_buckets[(send_view.device, send_view.dtype)][peer_global_rank].append(
                (local_tensor, int(local_slot), send_view, numel, nbytes)
            )

    for peer_buckets in remote_buckets.values():
        peer_chunks = [
            (peer_global_rank, _chunk_swap_bucket(bucket))
            for peer_global_rank, bucket in sorted(peer_buckets.items())
            if bucket
        ]
        chunk_indices = [0 for _ in peer_chunks]
        while any(index < len(chunks) for index, (_peer_global_rank, chunks) in zip(chunk_indices, peer_chunks)):
            ops: list[dist.P2POp] = []
            descriptors: list[tuple[torch.Tensor, torch.Tensor, list[_SwapBucketItem]]] = []
            wave_nbytes = 0
            for idx, (peer_global_rank, chunks) in enumerate(peer_chunks):
                if chunk_indices[idx] >= len(chunks):
                    continue
                chunk = chunks[chunk_indices[idx]]
                chunk_nbytes = _swap_chunk_nbytes(chunk)
                if ops and wave_nbytes + 2 * chunk_nbytes > _MAX_SWAP_WAVE_BYTES:
                    continue
                chunk_indices[idx] += 1
                wave_nbytes += 2 * chunk_nbytes
                send_buffer = _pack_swap_chunk(chunk)
                recv_buffer = torch.empty_like(send_buffer)
                ops.extend(
                    [
                        dist.P2POp(dist.isend, send_buffer, peer_global_rank),
                        dist.P2POp(dist.irecv, recv_buffer, peer_global_rank),
                    ]
                )
                descriptors.append((send_buffer, recv_buffer, chunk))

            if not ops:
                continue
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()
            for _send_buffer, recv_buffer, chunk in descriptors:
                _unpack_swap_chunk(recv_buffer, chunk)


@torch.no_grad()
def _exchange_or_swap_grouped_slot_entries_collective(
    grouped_entries: dict[tuple[int, int], list[_SwapTensorEntry]],
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    remote_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
        lambda: defaultdict(list)
    )
    active_bucket_keys: set[tuple[torch.device, torch.dtype]] = set()

    for (lhs_rank, rhs_rank), entry_list in sorted(grouped_entries.items()):
        if lhs_rank == rhs_rank:
            if ep_rank == lhs_rank:
                for entry in entry_list:
                    _swap_local_slot(entry.tensor, entry.lhs_slot, entry.rhs_slot)
            continue

        for entry in entry_list:
            local_tensor = _local_tensor_view(entry.tensor)
            active_bucket_keys.add((local_tensor.device, local_tensor.dtype))

        if ep_group is None or ep_size <= 1 or ep_rank not in (lhs_rank, rhs_rank):
            continue

        peer_rank = rhs_rank if ep_rank == lhs_rank else lhs_rank
        for entry in entry_list:
            local_slot = entry.lhs_slot if ep_rank == lhs_rank else entry.rhs_slot
            local_tensor = _local_tensor_view(entry.tensor)
            slot_view = local_tensor.detach()[local_slot]
            send_view = slot_view.contiguous().view(-1)
            numel = int(send_view.numel())
            nbytes = numel * int(send_view.element_size())
            remote_buckets[(send_view.device, send_view.dtype)][peer_rank].append(
                (local_tensor, int(local_slot), send_view, numel, nbytes)
            )

    if ep_group is None or ep_size <= 1:
        return

    def _bucket_sort_key(key: tuple[torch.device, torch.dtype]) -> tuple[str, int, str]:
        device, dtype = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype))

    for device, dtype in sorted(active_bucket_keys, key=_bucket_sort_key):
        peer_buckets = remote_buckets.get((device, dtype), {})
        send_split_tensor = torch.zeros((ep_size,), dtype=torch.long, device=device)
        for peer_rank, bucket in peer_buckets.items():
            send_split_tensor[int(peer_rank)] = sum(item[3] for item in bucket)

        recv_split_tensor = torch.empty_like(send_split_tensor)
        dist.all_to_all_single(recv_split_tensor, send_split_tensor, group=ep_group)
        input_splits = [int(value) for value in send_split_tensor.detach().cpu().tolist()]
        output_splits = [int(value) for value in recv_split_tensor.detach().cpu().tolist()]

        send_buffer = torch.empty((sum(input_splits),), dtype=dtype, device=device)
        offset = 0
        for peer_rank in range(ep_size):
            bucket = peer_buckets.get(peer_rank, ())
            if not bucket:
                continue
            packed = _pack_swap_chunk(bucket)
            send_buffer[offset : offset + packed.numel()].copy_(packed)
            offset += packed.numel()

        recv_buffer = torch.empty((sum(output_splits),), dtype=dtype, device=device)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
        )

        offset = 0
        for peer_rank, split_size in enumerate(output_splits):
            if split_size <= 0:
                continue
            bucket = peer_buckets.get(peer_rank)
            if not bucket:
                raise RuntimeError(
                    f"Expert Swap collective exchange received {split_size} values from rank {peer_rank} "
                    "without a matching local swap plan."
                )
            expected = sum(item[3] for item in bucket)
            if split_size != expected:
                raise RuntimeError(
                    f"Expert Swap collective exchange split mismatch from rank {peer_rank}: "
                    f"expected {expected}, got {split_size}."
                )
            _unpack_swap_chunk(recv_buffer[offset : offset + split_size], bucket)
            offset += split_size


@torch.no_grad()
def _cover_slot_entries(
    entries: Iterable[_CoverTensorEntry],
    src_rank: int,
    dst_rank: int,
    ep_rank: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    entry_list = list(entries)
    if not entry_list:
        return
    if src_rank == dst_rank:
        if ep_rank == src_rank:
            for entry in entry_list:
                _copy_local_slot(entry.tensor, entry.src_slot, entry.dst_slot)
        return
    if ep_group is None or ep_rank not in (src_rank, dst_rank):
        return

    peer_global_rank = _ep_global_rank(ep_group, dst_rank if ep_rank == src_rank else src_rank)
    buckets: dict[tuple[torch.device, torch.dtype], list[tuple[torch.Tensor, int, torch.Tensor, int]]] = defaultdict(
        list
    )
    for entry in entry_list:
        local_tensor = _local_tensor_view(entry.tensor)
        if ep_rank == src_rank:
            view = local_tensor.detach()[entry.src_slot].contiguous().view(-1)
            buckets[(view.device, view.dtype)].append((local_tensor, -1, view, int(view.numel())))
        else:
            view = local_tensor.detach()[entry.dst_slot].view(-1)
            buckets[(view.device, view.dtype)].append((local_tensor, int(entry.dst_slot), view, int(view.numel())))

    for bucket in buckets.values():
        if ep_rank == src_rank:
            send_buffer = torch.cat([item[2] for item in bucket], dim=0) if len(bucket) > 1 else bucket[0][2]
            dist.send(send_buffer, dst=peer_global_rank)
        else:
            total_numel = sum(item[3] for item in bucket)
            recv_buffer = torch.empty((total_numel,), dtype=bucket[0][2].dtype, device=bucket[0][2].device)
            dist.recv(recv_buffer, src=peer_global_rank)
            offset = 0
            for _local_tensor, _dst_slot, view, numel in bucket:
                view.copy_(recv_buffer[offset : offset + numel].view_as(view))
                offset += numel


@torch.no_grad()
def _placement_group_succeeded(
    local_success: bool,
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> bool:
    if ep_size <= 1 or ep_group is None:
        return bool(local_success)
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor([int(local_success)], dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=ep_group)
    return bool(status.item())


@torch.no_grad()
def _cover_grouped_slot_entries_atomic(
    grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]],
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
    *,
    zero_entry_groups: Iterable[tuple[int, Iterable[_CoverTensorEntry]]] = (),
    debug_validate: bool = False,
) -> None:
    """Stage all directed slot copies, then publish their destination tensors.

    Placement combines bidirectional swaps and one-way replica copies.  Treating
    both as directed copies lets us batch by peer, dtype, and slot shape while
    keeping destination tensors untouched until every communication succeeds.
    """

    zero_groups = tuple((int(rank), tuple(entries)) for rank, entries in zero_entry_groups)
    all_entries = tuple(entry for entries in grouped_entries.values() for entry in entries) + tuple(
        entry for _rank, entries in zero_groups for entry in entries
    )
    if not all_entries:
        return
    if ep_size > 1 and ep_group is None and any(src_rank != dst_rank for src_rank, dst_rank in grouped_entries):
        raise RuntimeError("HierMoE placement state migration requires an EP process group.")
    status_device = _local_tensor_view(all_entries[0].tensor).device

    def group_succeeded(local_success: bool) -> bool:
        if not debug_validate:
            return bool(local_success)
        return _placement_group_succeeded(
            local_success,
            device=status_device,
            ep_size=ep_size,
            ep_group=ep_group,
        )

    bucket_keys: set[tuple[torch.device, torch.dtype, tuple[int, ...]]] = set()
    send_buckets: dict[
        tuple[torch.device, torch.dtype, tuple[int, ...]], dict[int, list[tuple[torch.Tensor, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    recv_buckets: dict[
        tuple[torch.device, torch.dtype, tuple[int, ...]], dict[int, list[tuple[torch.Tensor, int, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    local_copies: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []
    has_remote = False
    stage_error: Exception | None = None

    try:
        for (src_rank, dst_rank), entries in sorted(grouped_entries.items()):
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                src_view = local_tensor.detach()[entry.src_slot]
                dst_view = local_tensor.detach()[entry.dst_slot]
                if tuple(src_view.shape) != tuple(dst_view.shape):
                    raise RuntimeError("HierMoE placement tried to copy between incompatible expert slot shapes.")
                key = (src_view.device, src_view.dtype, tuple(int(value) for value in src_view.shape))
                bucket_keys.add(key)
                if src_rank == dst_rank:
                    if ep_rank == src_rank:
                        local_copies.append((local_tensor, int(entry.dst_slot), src_view.clone()))
                    continue
                has_remote = True
                if ep_rank == src_rank:
                    send_buckets[key][int(dst_rank)].append((src_view.contiguous().view(-1), int(src_view.numel())))
                elif ep_rank == dst_rank:
                    recv_buckets[key][int(src_rank)].append((local_tensor, int(entry.dst_slot), int(dst_view.numel())))
        for dst_rank, entries in zero_groups:
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                dst_view = local_tensor.detach()[entry.dst_slot]
                bucket_keys.add((dst_view.device, dst_view.dtype, tuple(int(value) for value in dst_view.shape)))
                if ep_rank == dst_rank:
                    # A ``None`` staged value below represents transactional zeroing.
                    local_copies.append((local_tensor, int(entry.dst_slot), None))
    except Exception as error:
        stage_error = error

    if not group_succeeded(stage_error is None):
        if stage_error is not None:
            raise RuntimeError("HierMoE placement transaction preflight failed.") from stage_error
        raise RuntimeError("Another EP rank rejected the HierMoE placement transaction preflight.")

    def bucket_sort_key(
        key: tuple[torch.device, torch.dtype, tuple[int, ...]],
    ) -> tuple[str, int, str, tuple[int, ...]]:
        device, dtype, shape = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype), shape)

    remote_commits: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    try:
        if has_remote:
            for key in sorted(bucket_keys, key=bucket_sort_key):
                device, dtype, _shape = key
                peer_sends = send_buckets.get(key, {})
                peer_recvs = recv_buckets.get(key, {})
                send_splits_tensor = torch.zeros((ep_size,), dtype=torch.long, device=device)
                for peer_rank, items in peer_sends.items():
                    send_splits_tensor[int(peer_rank)] = sum(item[1] for item in items)
                recv_splits_tensor = torch.empty_like(send_splits_tensor)
                if ep_size > 1:
                    dist.all_to_all_single(recv_splits_tensor, send_splits_tensor, group=ep_group)
                else:
                    recv_splits_tensor.copy_(send_splits_tensor)
                input_splits = [int(value) for value in send_splits_tensor.detach().cpu().tolist()]
                output_splits = [int(value) for value in recv_splits_tensor.detach().cpu().tolist()]

                send_buffer = torch.empty((sum(input_splits),), dtype=dtype, device=device)
                offset = 0
                for peer_rank in range(ep_size):
                    items = peer_sends.get(peer_rank, ())
                    if not items:
                        continue
                    packed = torch.cat([item[0] for item in items], dim=0) if len(items) > 1 else items[0][0]
                    send_buffer[offset : offset + packed.numel()].copy_(packed)
                    offset += int(packed.numel())
                recv_buffer = torch.empty((sum(output_splits),), dtype=dtype, device=device)
                if ep_size > 1:
                    dist.all_to_all_single(
                        recv_buffer,
                        send_buffer,
                        output_split_sizes=output_splits,
                        input_split_sizes=input_splits,
                        group=ep_group,
                    )
                elif send_buffer.numel():
                    recv_buffer.copy_(send_buffer)

                offset = 0
                for peer_rank, split_size in enumerate(output_splits):
                    items = peer_recvs.get(peer_rank, ())
                    expected = sum(item[2] for item in items)
                    if int(split_size) != int(expected):
                        raise RuntimeError(
                            f"HierMoE placement state migration from rank {peer_rank} expected {expected} values, "
                            f"received {split_size}."
                        )
                    inner_offset = offset
                    for local_tensor, dst_slot, numel in items:
                        staged = recv_buffer[inner_offset : inner_offset + numel].view_as(
                            local_tensor.detach()[dst_slot]
                        )
                        remote_commits.append((local_tensor, dst_slot, staged))
                        inner_offset += numel
                    offset += split_size
    except Exception as error:
        stage_error = error

    if not group_succeeded(stage_error is None):
        if stage_error is not None:
            raise RuntimeError("HierMoE placement state migration staging failed.") from stage_error
        raise RuntimeError("Another EP rank failed to stage the HierMoE placement state migration.")

    publish_ops: list[tuple[torch.Tensor, int, torch.Tensor | None]] = [*local_copies, *remote_commits]
    destinations: set[tuple[int, int]] = set()
    publish_preflight_error: Exception | None = None
    try:
        for local_tensor, dst_slot, _staged in publish_ops:
            key = (id(local_tensor), int(dst_slot))
            if key in destinations:
                raise RuntimeError("HierMoE placement transaction writes one tensor slot more than once.")
            destinations.add(key)
    except Exception as error:
        publish_preflight_error = error
    if not group_succeeded(publish_preflight_error is None):
        if publish_preflight_error is not None:
            raise RuntimeError("HierMoE placement publish preflight failed.") from publish_preflight_error
        raise RuntimeError("Another EP rank rejected the HierMoE placement publish preflight.")

    publish_error: Exception | None = None
    try:
        for local_tensor, dst_slot, staged in publish_ops:
            if staged is None:
                local_tensor.detach()[dst_slot].zero_()
            else:
                local_tensor.detach()[dst_slot].copy_(staged)
    except Exception as error:
        publish_error = error

    publish_succeeded = group_succeeded(publish_error is None)
    if not publish_succeeded:
        if publish_error is not None:
            raise RuntimeError("HierMoE placement state migration publish failed.") from publish_error
        raise RuntimeError("Another EP rank failed to publish the HierMoE placement state migration.")


@torch.no_grad()
def _zero_slot_entries(entries: Iterable[_CoverTensorEntry], dst_rank: int, ep_rank: int) -> None:
    if ep_rank != dst_rank:
        return
    for entry in entries:
        _zero_local_slot(entry.tensor, entry.dst_slot)


@torch.no_grad()
def _exchange_or_swap_slots(
    tensors: Iterable[torch.Tensor],
    lhs_rank: int,
    lhs_slot: int,
    rhs_rank: int,
    rhs_slot: int,
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    entries = tuple(_SwapTensorEntry(tensor=tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot) for tensor in tensors)
    _exchange_or_swap_slot_entries(entries, lhs_rank, rhs_rank, ep_rank, ep_size, ep_group)


@torch.no_grad()
def _exchange_or_swap_slot(
    tensor: torch.Tensor,
    lhs_rank: int,
    lhs_slot: int,
    rhs_rank: int,
    rhs_slot: int,
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    _exchange_or_swap_slots((tensor,), lhs_rank, lhs_slot, rhs_rank, rhs_slot, ep_rank, ep_size, ep_group)


def _iter_leaf_optimizers(optimizer: Any) -> Iterable[Any]:
    if optimizer is None:
        return ()
    if hasattr(optimizer, "optimizers_dict"):
        return tuple(optimizer.optimizers_dict.values())
    return (optimizer,)


def _optimizer_has_param(optimizer: Any, param: torch.nn.Parameter) -> bool:
    return any(any(group_param is param for group_param in group["params"]) for group in optimizer.param_groups)


def _param_group_for_param(optimizer: Any, param: torch.nn.Parameter) -> dict[str, Any] | None:
    for group in optimizer.param_groups:
        if any(group_param is param for group_param in group["params"]):
            return group
    return None


def _step_device(param: torch.nn.Parameter, group: dict[str, Any]) -> torch.device:
    if bool(group.get("capturable", False)) or bool(group.get("fused", False)):
        return param.device
    return torch.device("cpu")


def _ensure_optimizer_state_for_group(
    optimizer: Any, param: torch.nn.Parameter, group: dict[str, Any]
) -> dict[str, Any] | None:
    state = optimizer.state[param]
    if state:
        return state

    opt_name = type(optimizer).__name__
    if opt_name == "AdamW":
        state["step"] = torch.zeros((), dtype=torch.float32, device=_step_device(param, group))
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if bool(group.get("amsgrad", False)):
            state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    elif opt_name == "AnyPrecisionAdamW":
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(param, dtype=group["momentum_dtype"])
        state["exp_avg_sq"] = torch.zeros_like(param, dtype=group["variance_dtype"])
        if bool(group.get("use_kahan_summation", False)):
            state["compensation"] = torch.zeros_like(param, dtype=group["compensation_buffer_dtype"])
    elif opt_name == "DistributedMuon":
        state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    else:
        raise NotImplementedError(
            f"HierMoE expert swap does not know how to initialize optimizer state for {opt_name}."
        )
    return state


def _ensure_optimizer_state(optimizer: Any, param: torch.nn.Parameter) -> dict[str, Any] | None:
    if not _optimizer_has_param(optimizer, param):
        return None

    group = _param_group_for_param(optimizer, param)
    if group is None:
        return None

    return _ensure_optimizer_state_for_group(optimizer, param, group)


def _existing_optimizer_state(optimizer: Any, param: torch.nn.Parameter) -> dict[str, Any] | None:
    state = getattr(optimizer, "state", None)
    if state is None:
        return None
    existing = state.get(param)
    return existing if existing else None


def _build_optimizer_param_bindings(optimizer: Any) -> dict[int, tuple[_OptimizerParamBinding, ...]]:
    bindings: dict[int, list[_OptimizerParamBinding]] = defaultdict(list)
    for opt in _iter_leaf_optimizers(optimizer):
        for group in opt.param_groups:
            for param in group["params"]:
                bindings[id(param)].append(_OptimizerParamBinding(opt, group))
    return {param_id: tuple(items) for param_id, items in bindings.items()}


@torch.no_grad()
def _swap_optimizer_state_slots(
    optimizer: Any,
    param: torch.nn.Parameter,
    lhs_rank: int,
    lhs_slot: int,
    rhs_rank: int,
    rhs_slot: int,
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    for opt in _iter_leaf_optimizers(optimizer):
        state = _existing_optimizer_state(opt, param)
        if not state:
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if tuple(_local_tensor_view(value).shape) != tuple(_local_tensor_view(param).shape):
                continue
            _exchange_or_swap_slot(value, lhs_rank, lhs_slot, rhs_rank, rhs_slot, ep_rank, ep_size, ep_group)


def _optimizer_state_slot_tensors(optimizer: Any, param: torch.nn.Parameter) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for opt in _iter_leaf_optimizers(optimizer):
        state = _existing_optimizer_state(opt, param)
        if not state:
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if tuple(_local_tensor_view(value).shape) != tuple(_local_tensor_view(param).shape):
                continue
            tensors.append(value)
    return tensors


def _optimizer_state_slot_tensors_from_bindings(
    bindings: Iterable[_OptimizerParamBinding], param: torch.nn.Parameter
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for binding in bindings:
        state = _existing_optimizer_state(binding.optimizer, param)
        if not state:
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if tuple(_local_tensor_view(value).shape) != tuple(_local_tensor_view(param).shape):
                continue
            tensors.append(value)
    return tensors


def _smooth_cost(per_dim_costs: list[float], gamma: float) -> float:
    if not per_dim_costs:
        return float("inf")
    cost_tensor = torch.tensor(per_dim_costs, dtype=torch.float64)
    return float(torch.logsumexp(cost_tensor * gamma, dim=0).item() / gamma)


def _smooth_cost_tensor(per_dim_costs: torch.Tensor, gamma: float) -> torch.Tensor:
    if per_dim_costs.numel() == 0:
        return torch.full((0,), float("inf"), dtype=torch.float32, device=per_dim_costs.device)
    return torch.logsumexp(per_dim_costs * float(gamma), dim=-1) / float(gamma)


def _as_batched_experts(selected_experts: torch.Tensor) -> torch.Tensor:
    selected_experts = selected_experts.to(torch.long)
    if selected_experts.ndim == 1:
        return selected_experts.view(1, -1, 1)
    if selected_experts.ndim == 2:
        return selected_experts.unsqueeze(0)
    return selected_experts


def _duplicate_free_counts_by_rank_batched(
    physical_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
) -> torch.Tensor:
    num_local_experts = max(1, num_experts // ep_size)
    return _duplicate_free_counts_by_expert_group_batched(physical_experts, num_experts, num_local_experts)


def _duplicate_free_counts_by_expert_group_batched(
    physical_experts: torch.Tensor,
    num_experts: int,
    group_size: int,
) -> torch.Tensor:
    if num_experts % group_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by group_size={group_size}.")
    physical_experts = _as_batched_experts(physical_experts)
    num_groups = num_experts // group_size
    target_groups = torch.div(physical_experts, group_size, rounding_mode="floor")
    token_group_hits = torch.zeros(
        (*target_groups.shape[:-1], num_groups),
        dtype=torch.float32,
        device=physical_experts.device,
    )
    token_group_hits.scatter_(dim=-1, index=target_groups, value=1.0)
    return token_group_hits.sum(dim=-2)


def _estimate_physical_costs(
    physical_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
) -> torch.Tensor:
    physical_experts = _as_batched_experts(physical_experts)
    batch_size = physical_experts.shape[0]
    per_dim_costs: list[torch.Tensor] = []
    max_dim = max(1, int(hierarchy.selected_dim))
    for dim in range(1, max_dim + 1):
        if dim <= 1:
            counts = _duplicate_free_counts_by_rank_batched(physical_experts, num_experts, hierarchy.ep_size)
            n_a2a = float(hierarchy.ep_size * hidden_size * bytes_per_element) * counts.max(dim=1).values
            per_dim_costs.append(perf_model.a2a.alpha + n_a2a * perf_model.a2a.beta)
            continue

        group_sizes = hierarchy.group_sizes[: dim - 1]
        total = torch.zeros((batch_size,), dtype=torch.float32, device=physical_experts.device)
        previous_u = 1
        for idx, u_i in enumerate(group_sizes):
            expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // u_i))
            counts = _duplicate_free_counts_by_expert_group_batched(
                physical_experts,
                num_experts,
                expert_group_size,
            )
            n_inter = float((u_i / previous_u) * hidden_size * bytes_per_element) * counts.max(dim=1).values
            link = perf_model.inter[min(idx, len(perf_model.inter) - 1)]
            total = total + link.alpha + n_inter * link.beta
            previous_u = u_i

        intra_group_size = max(1, num_experts // hierarchy.ep_size)
        intra_counts = _duplicate_free_counts_by_expert_group_batched(physical_experts, num_experts, intra_group_size)
        n_intra = float((hierarchy.ep_size / previous_u) * hidden_size * bytes_per_element) * (
            intra_counts.max(dim=1).values
        )
        total = total + perf_model.intra.alpha + n_intra * perf_model.intra.beta
        per_dim_costs.append(total)

    return _smooth_cost_tensor(torch.stack(per_dim_costs, dim=1), gamma)


def _estimate_swap_pair_costs_fast_2d(
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
    logical_to_physical: torch.Tensor,
    candidate_pairs: list[tuple[int, int]] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    max_dim = max(1, int(hierarchy.selected_dim))
    if max_dim > 2:
        return None

    selected_experts = selected_experts.to(torch.long)
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    logical_to_physical = logical_to_physical.to(device=selected_experts.device, dtype=torch.long, non_blocking=True)
    pairs = _candidate_pairs_tensor(selected_experts, num_experts, candidate_pairs)

    num_local_experts = max(1, num_experts // hierarchy.ep_size)
    rank_by_logical = torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")
    token_expert_hits = _token_expert_hit_matrix(selected_experts, num_experts)
    base_rank_counts, token_rank_counts = _token_group_counts_from_hits(
        token_expert_hits,
        rank_by_logical,
        hierarchy.ep_size,
    )
    current_dim_costs: list[torch.Tensor] = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * base_rank_counts.max() * perf_model.a2a.beta
    ]

    candidate_dim_costs: list[torch.Tensor] = []
    if pairs.numel() > 0:
        candidate_rank_max = _candidate_group_max_from_token_counts(
            token_expert_hits=token_expert_hits,
            token_group_counts=token_rank_counts,
            base_group_counts=base_rank_counts,
            pairs=pairs,
            group_by_logical=rank_by_logical,
        )
        candidate_dim_costs.append(
            perf_model.a2a.alpha
            + float(hierarchy.ep_size * hidden_size * bytes_per_element) * candidate_rank_max * perf_model.a2a.beta
        )
    else:
        candidate_rank_max = torch.full((0,), float("inf"), dtype=torch.float32, device=selected_experts.device)

    if max_dim >= 2:
        if not hierarchy.group_sizes or not perf_model.inter:
            return None
        u_i = int(hierarchy.group_sizes[0])
        expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // u_i))
        if num_experts % expert_group_size != 0:
            return None
        num_groups = num_experts // expert_group_size
        group_by_logical = torch.div(logical_to_physical, expert_group_size, rounding_mode="floor")
        base_group_counts, token_group_counts = _token_group_counts_from_hits(
            token_expert_hits,
            group_by_logical,
            num_groups,
        )
        group_max = base_group_counts.max()
        link = perf_model.inter[0]
        current_dim_costs.append(
            link.alpha
            + float(u_i * hidden_size * bytes_per_element) * group_max * link.beta
            + perf_model.intra.alpha
            + float((hierarchy.ep_size / u_i) * hidden_size * bytes_per_element)
            * base_rank_counts.max()
            * perf_model.intra.beta
        )

        if pairs.numel() > 0:
            candidate_group_max = _candidate_group_max_from_token_counts(
                token_expert_hits=token_expert_hits,
                token_group_counts=token_group_counts,
                base_group_counts=base_group_counts,
                pairs=pairs,
                group_by_logical=group_by_logical,
            )
            candidate_dim_costs.append(
                link.alpha
                + float(u_i * hidden_size * bytes_per_element) * candidate_group_max * link.beta
                + perf_model.intra.alpha
                + float((hierarchy.ep_size / u_i) * hidden_size * bytes_per_element)
                * candidate_rank_max
                * perf_model.intra.beta
            )

    current_cost = _smooth_cost_tensor(torch.stack(current_dim_costs, dim=0).unsqueeze(0), gamma)[0]
    if pairs.numel() == 0:
        empty_costs = torch.full((0,), float("inf"), dtype=torch.float32, device=selected_experts.device)
        return current_cost.to(torch.float32), pairs, empty_costs

    costs = _smooth_cost_tensor(torch.stack(candidate_dim_costs, dim=1), gamma)
    return current_cost.to(torch.float32), pairs, costs.to(torch.float32)


def _estimate_best_swap_pair_row_fast_2d(
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
    logical_to_physical: torch.Tensor,
    candidate_pairs: list[tuple[int, int]] | torch.Tensor | None = None,
) -> torch.Tensor | None:
    result = _estimate_swap_pair_costs_fast_2d(
        selected_experts=selected_experts,
        num_experts=num_experts,
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        hierarchy=hierarchy,
        perf_model=perf_model,
        gamma=gamma,
        logical_to_physical=logical_to_physical,
        candidate_pairs=candidate_pairs,
    )
    if result is None:
        return None

    current_cost, pairs, costs = result
    if pairs.numel() == 0:
        return torch.stack(
            (
                current_cost.to(torch.float32),
                selected_experts.new_tensor(-1, dtype=torch.float32),
                selected_experts.new_tensor(-1, dtype=torch.float32),
            )
        )

    best_idx = torch.argmin(costs)
    best_cost = costs.index_select(0, best_idx.view(1))[0]
    improved = best_cost < current_cost
    best_pair = pairs.index_select(0, best_idx.view(1))[0]
    no_pair = torch.full((2,), -1, dtype=torch.long, device=selected_experts.device)
    chosen_pair = torch.where(improved, best_pair, no_pair)
    chosen_cost = torch.where(improved, best_cost, current_cost)
    return torch.stack(
        (chosen_cost.to(torch.float32), chosen_pair[0].to(torch.float32), chosen_pair[1].to(torch.float32))
    )


def _selector_stats_2d(
    selected_experts: torch.Tensor,
    num_experts: int,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = selected_experts.device
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    selected_experts = selected_experts.to(torch.long)
    if selected_experts.numel() == 0:
        return (
            torch.zeros((num_experts,), dtype=torch.float32, device=device),
            torch.zeros((num_groups,), dtype=torch.float32, device=device),
            torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device),
        )

    selected_groups = group_by_logical.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
    token_group_hits = torch.zeros(
        (selected_experts.shape[0], num_groups),
        dtype=torch.float32,
        device=device,
    )
    token_group_hits.scatter_(dim=1, index=selected_groups, value=1.0)
    base_group_counts = token_group_hits.sum(dim=0)

    flat_experts = selected_experts.reshape(-1)
    expert_counts = torch.bincount(flat_experts, minlength=num_experts).to(torch.float32)
    flat_tokens = (
        torch.arange(selected_experts.shape[0], device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(selected_experts)
        .reshape(-1)
    )
    expert_group_counts = torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device)
    expert_group_counts.index_add_(0, flat_experts, token_group_hits.index_select(0, flat_tokens))
    return expert_counts, base_group_counts, expert_group_counts


def _selector_group_stats(
    selected_experts: torch.Tensor,
    num_experts: int,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = selected_experts.device
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    selected_experts = selected_experts.to(torch.long)
    if selected_experts.numel() == 0:
        return (
            torch.zeros((num_groups,), dtype=torch.float32, device=device),
            torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device),
        )

    selected_groups = group_by_logical.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
    token_group_hits = torch.zeros(
        (selected_experts.shape[0], num_groups),
        dtype=torch.float32,
        device=device,
    )
    token_group_hits.scatter_(dim=1, index=selected_groups, value=1.0)
    base_group_counts = token_group_hits.sum(dim=0)

    flat_experts = selected_experts.reshape(-1)
    flat_tokens = (
        torch.arange(selected_experts.shape[0], device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(selected_experts)
        .reshape(-1)
    )
    expert_group_counts = torch.zeros((num_experts, num_groups), dtype=torch.float32, device=device)
    expert_group_counts.index_add_(0, flat_experts, token_group_hits.index_select(0, flat_tokens))
    return base_group_counts, expert_group_counts


def _token_expert_hit_matrix(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    selected_experts = selected_experts.to(torch.long)
    hits = torch.zeros((selected_experts.shape[0], num_experts), dtype=torch.bool, device=selected_experts.device)
    if selected_experts.numel() > 0:
        hits.scatter_(dim=1, index=selected_experts, value=True)
    return hits


def _token_group_counts_from_hits(
    token_expert_hits: torch.Tensor,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_index = group_by_logical.view(1, -1).expand(token_expert_hits.shape[0], -1)
    token_group_counts = torch.zeros(
        (token_expert_hits.shape[0], num_groups),
        dtype=torch.float32,
        device=token_expert_hits.device,
    )
    token_group_counts.scatter_add_(1, group_index, token_expert_hits.to(torch.float32))
    base_group_counts = (token_group_counts > 0).sum(dim=0).to(torch.float32)
    return base_group_counts, token_group_counts


@dataclass(frozen=True)
class _ExactSingleSwapGroupStats:
    """Additive statistics that exactly score any one expert swap."""

    base_group_counts: torch.Tensor
    zero_group_expert_hits: torch.Tensor
    sole_expert_hits: torch.Tensor
    sole_expert_cohits: torch.Tensor


def _exact_single_swap_group_stats(
    token_expert_hits: torch.Tensor,
    group_by_logical: torch.Tensor,
    num_groups: int,
) -> _ExactSingleSwapGroupStats:
    """Build exact, token-order-independent statistics for one group layout."""

    num_experts = int(token_expert_hits.shape[1])
    base_group_counts, token_group_counts = _token_group_counts_from_hits(
        token_expert_hits,
        group_by_logical,
        num_groups,
    )
    hit_values = token_expert_hits.to(torch.float32)
    zero_group_expert_hits = (token_group_counts == 0).to(torch.float32).transpose(0, 1).matmul(hit_values)

    if num_experts == 0:
        sole_expert_hits = torch.zeros((0,), dtype=torch.float32, device=token_expert_hits.device)
        sole_expert_cohits = torch.zeros((0, 0), dtype=torch.float32, device=token_expert_hits.device)
    else:
        expert_group_counts = token_group_counts.index_select(1, group_by_logical)
        sole_hit_matrix = hit_values * (expert_group_counts == 1).to(torch.float32)
        sole_expert_hits = sole_hit_matrix.sum(dim=0)
        sole_expert_cohits = sole_hit_matrix.transpose(0, 1).matmul(hit_values)

    return _ExactSingleSwapGroupStats(
        base_group_counts=base_group_counts,
        zero_group_expert_hits=zero_group_expert_hits,
        sole_expert_hits=sole_expert_hits,
        sole_expert_cohits=sole_expert_cohits,
    )


def _flatten_exact_single_swap_group_stats(stats: _ExactSingleSwapGroupStats) -> torch.Tensor:
    return torch.cat(
        (
            stats.base_group_counts.reshape(-1),
            stats.zero_group_expert_hits.reshape(-1),
            stats.sole_expert_hits.reshape(-1),
            stats.sole_expert_cohits.reshape(-1),
        ),
        dim=0,
    )


def _unpack_exact_single_swap_group_stats(
    flat_stats: torch.Tensor,
    *,
    offset: int,
    num_experts: int,
    num_groups: int,
) -> tuple[_ExactSingleSwapGroupStats, int]:
    base_end = offset + num_groups
    zero_end = base_end + num_groups * num_experts
    sole_end = zero_end + num_experts
    cohit_end = sole_end + num_experts * num_experts
    return (
        _ExactSingleSwapGroupStats(
            base_group_counts=flat_stats[offset:base_end],
            zero_group_expert_hits=flat_stats[base_end:zero_end].view(num_groups, num_experts),
            sole_expert_hits=flat_stats[zero_end:sole_end],
            sole_expert_cohits=flat_stats[sole_end:cohit_end].view(num_experts, num_experts),
        ),
        cohit_end,
    )


def _exact_candidate_group_max_from_stats(
    *,
    stats: _ExactSingleSwapGroupStats,
    pairs: torch.Tensor,
    group_by_logical: torch.Tensor,
) -> torch.Tensor:
    """Return the exact post-swap maximum duplicate-free count for every pair."""

    if pairs.numel() == 0:
        return torch.full(
            (0,),
            float("inf"),
            dtype=torch.float32,
            device=stats.base_group_counts.device,
        )

    lhs = pairs[:, 0]
    rhs = pairs[:, 1]
    lhs_group = group_by_logical.index_select(0, lhs)
    rhs_group = group_by_logical.index_select(0, rhs)
    same_group = lhs_group == rhs_group

    lhs_loss = stats.sole_expert_hits.index_select(0, lhs) - stats.sole_expert_cohits[lhs, rhs]
    rhs_loss = stats.sole_expert_hits.index_select(0, rhs) - stats.sole_expert_cohits[rhs, lhs]
    lhs_gain = stats.zero_group_expert_hits[lhs_group, rhs]
    rhs_gain = stats.zero_group_expert_hits[rhs_group, lhs]
    lhs_delta = torch.where(same_group, torch.zeros_like(lhs_gain), lhs_gain - lhs_loss)
    rhs_delta = torch.where(same_group, torch.zeros_like(rhs_gain), rhs_gain - rhs_loss)

    candidate_group_counts = stats.base_group_counts.unsqueeze(0).expand(pairs.shape[0], -1).clone()
    candidate_group_counts.scatter_add_(1, lhs_group.unsqueeze(1), lhs_delta.unsqueeze(1))
    candidate_group_counts.scatter_add_(1, rhs_group.unsqueeze(1), rhs_delta.unsqueeze(1))
    return candidate_group_counts.max(dim=1).values


def _exact_single_swap_costs_from_group_stats(
    *,
    group_stats: list[_ExactSingleSwapGroupStats],
    group_by_logical: list[torch.Tensor],
    pairs: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the existing cost model to exact post-swap group counts."""

    if not group_stats or len(group_stats) != len(group_by_logical):
        raise ValueError("Exact HierMoE single-swap statistics are incomplete.")

    level_shapes = _hierarchy_level_group_shapes(hierarchy, num_experts)
    max_dim = max(1, int(hierarchy.selected_dim))
    if len(group_stats) != 1 + len(level_shapes):
        raise ValueError("Exact HierMoE single-swap hierarchy statistics have an invalid shape.")

    current_group_maxes = [stats.base_group_counts.max() for stats in group_stats]
    candidate_group_maxes = [
        _exact_candidate_group_max_from_stats(
            stats=stats,
            pairs=pairs,
            group_by_logical=mapping,
        )
        for stats, mapping in zip(group_stats, group_by_logical, strict=True)
    ]
    current_rank_max = current_group_maxes[0]
    candidate_rank_max = candidate_group_maxes[0]
    current_dim_costs: list[torch.Tensor] = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * current_rank_max * perf_model.a2a.beta
    ]
    candidate_dim_costs: list[torch.Tensor] = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * candidate_rank_max * perf_model.a2a.beta
    ]

    for dim in range(2, max_dim + 1):
        current_total = torch.zeros((), dtype=torch.float32, device=pairs.device)
        candidate_total = torch.zeros((pairs.shape[0],), dtype=torch.float32, device=pairs.device)
        previous_u = 1
        for level_idx, (u_i, _num_groups) in enumerate(level_shapes[: dim - 1]):
            link = perf_model.inter[min(level_idx, len(perf_model.inter) - 1)]
            scale = float((u_i / previous_u) * hidden_size * bytes_per_element)
            current_total = current_total + link.alpha + scale * current_group_maxes[level_idx + 1] * link.beta
            candidate_level_max = candidate_group_maxes[level_idx + 1]
            candidate_total = candidate_total + link.alpha + scale * candidate_level_max * link.beta
            previous_u = u_i

        intra_scale = float((hierarchy.ep_size / previous_u) * hidden_size * bytes_per_element)
        current_total = current_total + perf_model.intra.alpha + intra_scale * current_rank_max * perf_model.intra.beta
        candidate_total = (
            candidate_total + perf_model.intra.alpha + intra_scale * candidate_rank_max * perf_model.intra.beta
        )
        current_dim_costs.append(current_total)
        candidate_dim_costs.append(candidate_total)

    current_cost = _smooth_cost_tensor(torch.stack(current_dim_costs, dim=0).unsqueeze(0), gamma)[0]
    if pairs.numel() == 0:
        return current_cost.to(torch.float32), torch.full(
            (0,),
            float("inf"),
            dtype=torch.float32,
            device=current_cost.device,
        )
    candidate_costs = _smooth_cost_tensor(torch.stack(candidate_dim_costs, dim=1), gamma)
    return current_cost.to(torch.float32), candidate_costs.to(torch.float32)


def _candidate_group_max_from_token_counts(
    *,
    token_expert_hits: torch.Tensor,
    token_group_counts: torch.Tensor,
    base_group_counts: torch.Tensor,
    pairs: torch.Tensor,
    group_by_logical: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.full((0,), float("inf"), dtype=torch.float32, device=base_group_counts.device)

    results: list[torch.Tensor] = []
    for start in range(0, pairs.shape[0], chunk_size):
        chunk = pairs[start : start + chunk_size]
        lhs = chunk[:, 0]
        rhs = chunk[:, 1]
        lhs_group = group_by_logical.index_select(0, lhs)
        rhs_group = group_by_logical.index_select(0, rhs)
        same_group = lhs_group == rhs_group

        has_lhs = token_expert_hits.index_select(1, lhs).transpose(0, 1).to(torch.float32)
        has_rhs = token_expert_hits.index_select(1, rhs).transpose(0, 1).to(torch.float32)
        lhs_group_counts = token_group_counts.index_select(1, lhs_group).transpose(0, 1)
        rhs_group_counts = token_group_counts.index_select(1, rhs_group).transpose(0, 1)

        before_lhs = lhs_group_counts > 0
        before_rhs = rhs_group_counts > 0
        after_lhs = (lhs_group_counts - has_lhs + has_rhs) > 0
        after_rhs = (rhs_group_counts - has_rhs + has_lhs) > 0
        delta_lhs_group = after_lhs.sum(dim=1).to(torch.float32) - before_lhs.sum(dim=1).to(torch.float32)
        delta_rhs_group = after_rhs.sum(dim=1).to(torch.float32) - before_rhs.sum(dim=1).to(torch.float32)
        delta_lhs_group = torch.where(same_group, torch.zeros_like(delta_lhs_group), delta_lhs_group)
        delta_rhs_group = torch.where(same_group, torch.zeros_like(delta_rhs_group), delta_rhs_group)

        group_counts = base_group_counts.unsqueeze(0).expand(chunk.shape[0], -1).clone()
        group_counts.scatter_add_(1, lhs_group.unsqueeze(1), delta_lhs_group.unsqueeze(1))
        group_counts.scatter_add_(1, rhs_group.unsqueeze(1), delta_rhs_group.unsqueeze(1))
        results.append(group_counts.max(dim=1).values)

    return torch.cat(results, dim=0)


def _hierarchy_level_group_shapes(hierarchy: Hierarchy, num_experts: int) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    max_dim = max(1, int(hierarchy.selected_dim))
    for u_i in hierarchy.group_sizes[: max(0, max_dim - 1)]:
        expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // int(u_i)))
        if num_experts % expert_group_size != 0:
            return []
        levels.append((int(u_i), int(num_experts // expert_group_size)))
    return levels


def _candidate_group_max_from_global_stats(
    *,
    expert_counts: torch.Tensor,
    base_group_counts: torch.Tensor,
    expert_group_counts: torch.Tensor,
    pair_counts: torch.Tensor,
    pairs: torch.Tensor,
    group_by_logical: torch.Tensor,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.full((0,), float("inf"), dtype=torch.float32, device=expert_counts.device)

    lhs = pairs[:, 0]
    rhs = pairs[:, 1]
    lhs_group = group_by_logical.index_select(0, lhs)
    rhs_group = group_by_logical.index_select(0, rhs)
    same_group = lhs_group == rhs_group
    lhs_count = expert_counts.index_select(0, lhs)
    rhs_count = expert_counts.index_select(0, rhs)
    rhs_hits_lhs_group = expert_group_counts[rhs, lhs_group]
    lhs_hits_rhs_group = expert_group_counts[lhs, rhs_group]
    delta_lhs_group = rhs_count - rhs_hits_lhs_group - lhs_count + pair_counts
    delta_rhs_group = lhs_count - lhs_hits_rhs_group - rhs_count + pair_counts
    delta_lhs_group = torch.where(same_group, torch.zeros_like(delta_lhs_group), delta_lhs_group)
    delta_rhs_group = torch.where(same_group, torch.zeros_like(delta_rhs_group), delta_rhs_group)

    group_counts = base_group_counts.unsqueeze(0).expand(pairs.shape[0], -1).clone()
    group_counts.scatter_add_(1, lhs_group.unsqueeze(1), delta_lhs_group.unsqueeze(1))
    group_counts.scatter_add_(1, rhs_group.unsqueeze(1), delta_rhs_group.unsqueeze(1))
    return group_counts.max(dim=1).values


def _costs_from_global_hierarchy_stats(
    *,
    expert_counts: torch.Tensor,
    base_rank_counts: torch.Tensor,
    expert_rank_counts: torch.Tensor,
    level_base_group_counts: list[torch.Tensor],
    level_expert_group_counts: list[torch.Tensor],
    pair_counts: torch.Tensor,
    pairs: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
    logical_to_physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_local_experts = max(1, num_experts // hierarchy.ep_size)
    rank_by_logical = torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")
    current_dim_costs: list[torch.Tensor] = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * base_rank_counts.max() * perf_model.a2a.beta
    ]

    candidate_dim_costs: list[torch.Tensor] = []
    if pairs.numel() > 0:
        candidate_rank_max = _candidate_group_max_from_global_stats(
            expert_counts=expert_counts,
            base_group_counts=base_rank_counts,
            expert_group_counts=expert_rank_counts,
            pair_counts=pair_counts,
            pairs=pairs,
            group_by_logical=rank_by_logical,
        )
        candidate_dim_costs.append(
            perf_model.a2a.alpha
            + float(hierarchy.ep_size * hidden_size * bytes_per_element) * candidate_rank_max * perf_model.a2a.beta
        )
    else:
        candidate_rank_max = torch.full((0,), float("inf"), dtype=torch.float32, device=expert_counts.device)

    level_shapes = _hierarchy_level_group_shapes(hierarchy, num_experts)
    level_group_by_logical: list[torch.Tensor] = []
    for u_i, _num_groups in level_shapes:
        expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // u_i))
        level_group_by_logical.append(torch.div(logical_to_physical, expert_group_size, rounding_mode="floor"))

    max_dim = max(1, int(hierarchy.selected_dim))
    for dim in range(2, max_dim + 1):
        if len(level_shapes) < dim - 1 or not perf_model.inter:
            break
        current_total = torch.zeros((), dtype=torch.float32, device=expert_counts.device)
        candidate_total = (
            torch.zeros((pairs.shape[0],), dtype=torch.float32, device=expert_counts.device)
            if pairs.numel() > 0
            else None
        )
        previous_u = 1
        for level_idx, (u_i, _num_groups) in enumerate(level_shapes[: dim - 1]):
            base_group_counts = level_base_group_counts[level_idx]
            link = perf_model.inter[min(level_idx, len(perf_model.inter) - 1)]
            scale = float((u_i / previous_u) * hidden_size * bytes_per_element)
            current_total = current_total + link.alpha + scale * base_group_counts.max() * link.beta
            if candidate_total is not None:
                candidate_max = _candidate_group_max_from_global_stats(
                    expert_counts=expert_counts,
                    base_group_counts=base_group_counts,
                    expert_group_counts=level_expert_group_counts[level_idx],
                    pair_counts=pair_counts,
                    pairs=pairs,
                    group_by_logical=level_group_by_logical[level_idx],
                )
                candidate_total = candidate_total + link.alpha + scale * candidate_max * link.beta
            previous_u = u_i

        scale = float((hierarchy.ep_size / previous_u) * hidden_size * bytes_per_element)
        current_total = current_total + perf_model.intra.alpha + scale * base_rank_counts.max() * perf_model.intra.beta
        current_dim_costs.append(current_total)
        if candidate_total is not None:
            candidate_total = (
                candidate_total + perf_model.intra.alpha + scale * candidate_rank_max * perf_model.intra.beta
            )
            candidate_dim_costs.append(candidate_total)

    current_cost = _smooth_cost_tensor(torch.stack(current_dim_costs, dim=0).unsqueeze(0), gamma)[0]
    if pairs.numel() == 0:
        return current_cost.to(torch.float32), torch.full(
            (0,), float("inf"), dtype=torch.float32, device=expert_counts.device
        )

    costs = _smooth_cost_tensor(torch.stack(candidate_dim_costs, dim=1), gamma)
    return current_cost.to(torch.float32), costs.to(torch.float32)


def _candidate_pair_token_counts(
    selected_experts: torch.Tensor,
    pairs: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=selected_experts.device)
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    if selected_experts.numel() == 0:
        return torch.zeros((pairs.shape[0],), dtype=torch.float32, device=pairs.device)

    selected_experts = selected_experts.to(device=pairs.device, dtype=torch.long, non_blocking=True)
    hits = torch.zeros(
        (selected_experts.shape[0], num_experts),
        dtype=torch.float32,
        device=pairs.device,
    )
    hits.scatter_(dim=1, index=selected_experts, value=1.0)
    pair_matrix = hits.transpose(0, 1).matmul(hits)
    return pair_matrix[pairs[:, 0], pairs[:, 1]].to(torch.float32)


def _costs_from_global_2d_stats(
    *,
    expert_counts: torch.Tensor,
    base_group_counts: torch.Tensor,
    expert_group_counts: torch.Tensor,
    base_rank_counts: torch.Tensor,
    expert_rank_counts: torch.Tensor,
    pair_counts: torch.Tensor,
    pairs: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
    logical_to_physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_local_experts = max(1, num_experts // hierarchy.ep_size)
    rank_by_logical = torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")
    current_dim_costs: list[torch.Tensor] = [
        perf_model.a2a.alpha
        + float(hierarchy.ep_size * hidden_size * bytes_per_element) * base_rank_counts.max() * perf_model.a2a.beta
    ]

    candidate_dim_costs: list[torch.Tensor] = []
    lhs = pairs[:, 0]
    rhs = pairs[:, 1]
    if pairs.numel() > 0:
        candidate_rank_max = _candidate_group_max_from_global_stats(
            expert_counts=expert_counts,
            base_group_counts=base_rank_counts,
            expert_group_counts=expert_rank_counts,
            pair_counts=pair_counts,
            pairs=pairs,
            group_by_logical=rank_by_logical,
        )
        candidate_dim_costs.append(
            perf_model.a2a.alpha
            + float(hierarchy.ep_size * hidden_size * bytes_per_element) * candidate_rank_max * perf_model.a2a.beta
        )
    else:
        candidate_rank_max = torch.full((0,), float("inf"), dtype=torch.float32, device=expert_counts.device)

    if max(1, int(hierarchy.selected_dim)) >= 2:
        u_i = int(hierarchy.group_sizes[0])
        link = perf_model.inter[0]
        group_max = base_group_counts.max()
        current_dim_costs.append(
            link.alpha
            + float(u_i * hidden_size * bytes_per_element) * group_max * link.beta
            + perf_model.intra.alpha
            + float((hierarchy.ep_size / u_i) * hidden_size * bytes_per_element)
            * base_rank_counts.max()
            * perf_model.intra.beta
        )
        if pairs.numel() > 0:
            expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // u_i))
            group_by_logical = torch.div(logical_to_physical, expert_group_size, rounding_mode="floor")
            lhs_group = group_by_logical.index_select(0, lhs)
            rhs_group = group_by_logical.index_select(0, rhs)
            same_group = lhs_group == rhs_group
            lhs_count = expert_counts.index_select(0, lhs)
            rhs_count = expert_counts.index_select(0, rhs)
            rhs_hits_lhs_group = expert_group_counts[rhs, lhs_group]
            lhs_hits_rhs_group = expert_group_counts[lhs, rhs_group]
            delta_lhs_group = rhs_count - rhs_hits_lhs_group - lhs_count + pair_counts
            delta_rhs_group = lhs_count - lhs_hits_rhs_group - rhs_count + pair_counts
            delta_lhs_group = torch.where(same_group, torch.zeros_like(delta_lhs_group), delta_lhs_group)
            delta_rhs_group = torch.where(same_group, torch.zeros_like(delta_rhs_group), delta_rhs_group)

            group_counts = base_group_counts.unsqueeze(0).expand(pairs.shape[0], -1).clone()
            group_counts.scatter_add_(1, lhs_group.unsqueeze(1), delta_lhs_group.unsqueeze(1))
            group_counts.scatter_add_(1, rhs_group.unsqueeze(1), delta_rhs_group.unsqueeze(1))
            candidate_group_max = group_counts.max(dim=1).values
            candidate_dim_costs.append(
                link.alpha
                + float(u_i * hidden_size * bytes_per_element) * candidate_group_max * link.beta
                + perf_model.intra.alpha
                + float((hierarchy.ep_size / u_i) * hidden_size * bytes_per_element)
                * candidate_rank_max
                * perf_model.intra.beta
            )

    current_cost = _smooth_cost_tensor(torch.stack(current_dim_costs, dim=0).unsqueeze(0), gamma)[0]
    if pairs.numel() == 0:
        return current_cost.to(torch.float32), torch.full(
            (0,),
            float("inf"),
            dtype=torch.float32,
            device=expert_counts.device,
        )

    costs = _smooth_cost_tensor(torch.stack(candidate_dim_costs, dim=1), gamma)
    return current_cost.to(torch.float32), costs.to(torch.float32)


def _estimate_mapping_cost(
    selected_experts: torch.Tensor,
    logical_to_physical: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
) -> float:
    mapping = logical_to_physical.to(device=selected_experts.device, non_blocking=True)
    physical_experts = mapping.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
    per_dim = [
        perf_model.estimate_hierarchical_time(
            physical_experts,
            num_experts,
            hidden_size,
            bytes_per_element,
            hierarchy,
            dim,
        )
        for dim in range(1, hierarchy.selected_dim + 1)
    ]
    return _smooth_cost(per_dim, gamma)


def _all_candidate_pairs(num_experts: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(num_experts))
    pairs = _ALL_CANDIDATE_PAIR_CACHE.get(key)
    if pairs is None:
        pairs = torch.triu_indices(num_experts, num_experts, offset=1).t().contiguous()
        pairs = pairs.to(device=device, non_blocking=True)
        _ALL_CANDIDATE_PAIR_CACHE[key] = pairs
    return pairs


def _cross_rank_candidate_pairs(
    pairs: torch.Tensor,
    logical_to_physical: torch.Tensor,
    num_local_experts: int,
) -> torch.Tensor:
    """Keep only swaps whose current physical owners are different ranks."""

    if pairs.numel() == 0:
        return pairs
    mapping = logical_to_physical.to(device=pairs.device, dtype=torch.long, non_blocking=True)
    owner_ranks = torch.div(mapping, int(num_local_experts), rounding_mode="floor")
    lhs_ranks = owner_ranks.index_select(0, pairs[:, 0])
    rhs_ranks = owner_ranks.index_select(0, pairs[:, 1])
    return pairs[lhs_ranks != rhs_ranks]


def _candidate_pairs(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    return _all_candidate_pairs(num_experts, selected_experts.device)


def _shard_candidate_pairs(
    candidate_pairs: torch.Tensor | None,
    *,
    shard_idx: int,
    num_shards: int,
) -> torch.Tensor | None:
    if candidate_pairs is None or num_shards <= 1:
        return candidate_pairs
    return candidate_pairs[int(shard_idx) :: int(num_shards)].contiguous()


def _resolve_candidate_shards(ep_size: int, num_experts: int) -> int:
    candidate_count = max(1, num_experts * (num_experts - 1) // 2)
    if num_experts <= 64:
        return 1
    if _SWAP_CANDIDATE_SHARDS > 0:
        return min(_SWAP_CANDIDATE_SHARDS, max(1, ep_size), candidate_count)
    return min(max(1, ep_size), candidate_count)


def _candidate_pairs_tensor(
    selected_experts: torch.Tensor,
    num_experts: int,
    candidate_pairs: list[tuple[int, int]] | torch.Tensor | None,
) -> torch.Tensor:
    if candidate_pairs is None:
        return _candidate_pairs(selected_experts, num_experts)
    if torch.is_tensor(candidate_pairs):
        return candidate_pairs.to(device=selected_experts.device, dtype=torch.long, non_blocking=True)
    return torch.tensor(candidate_pairs, dtype=torch.long, device=selected_experts.device)


def _estimate_best_swap_pair_row(
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float,
    logical_to_physical: torch.Tensor,
    candidate_pairs: list[tuple[int, int]] | torch.Tensor | None = None,
) -> torch.Tensor:
    selected_experts = selected_experts.to(torch.long)
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
    logical_to_physical = logical_to_physical.to(device=selected_experts.device, dtype=torch.long, non_blocking=True)
    if _USE_FAST_2D_SELECTOR:
        fast_row = _estimate_best_swap_pair_row_fast_2d(
            selected_experts=selected_experts,
            num_experts=num_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            hierarchy=hierarchy,
            perf_model=perf_model,
            gamma=gamma,
            logical_to_physical=logical_to_physical,
            candidate_pairs=candidate_pairs,
        )
        if fast_row is not None:
            return fast_row

    base_physical = logical_to_physical.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
    current_cost = _estimate_physical_costs(
        base_physical,
        num_experts,
        hidden_size,
        bytes_per_element,
        hierarchy,
        perf_model,
        gamma,
    )[0]

    pairs = _candidate_pairs_tensor(selected_experts, num_experts, candidate_pairs)
    if pairs.numel() == 0:
        return torch.stack(
            (
                current_cost.to(torch.float32),
                selected_experts.new_tensor(-1, dtype=torch.float32),
                selected_experts.new_tensor(-1, dtype=torch.float32),
            )
        )

    cost_chunks = []
    pair_chunks = []
    selected_for_compare = selected_experts.unsqueeze(0)
    base_physical = base_physical.unsqueeze(0)
    for chunk in pairs.split(_SWAP_COST_CHUNK_CANDIDATES, dim=0):
        lhs = chunk[:, 0]
        rhs = chunk[:, 1]
        lhs_physical = logical_to_physical.index_select(0, lhs).view(-1, 1, 1)
        rhs_physical = logical_to_physical.index_select(0, rhs).view(-1, 1, 1)
        lhs_logical = lhs.view(-1, 1, 1)
        rhs_logical = rhs.view(-1, 1, 1)
        physical = torch.where(
            selected_for_compare == lhs_logical,
            rhs_physical,
            torch.where(selected_for_compare == rhs_logical, lhs_physical, base_physical),
        )
        cost_chunks.append(
            _estimate_physical_costs(
                physical,
                num_experts,
                hidden_size,
                bytes_per_element,
                hierarchy,
                perf_model,
                gamma,
            )
        )
        pair_chunks.append(chunk)

    costs = torch.cat(cost_chunks, dim=0)
    all_pairs = torch.cat(pair_chunks, dim=0)
    best_idx = torch.argmin(costs)
    best_cost = costs.index_select(0, best_idx.view(1))[0]
    improved = best_cost < current_cost
    best_pair = all_pairs.index_select(0, best_idx.view(1))[0]
    no_pair = torch.full((2,), -1, dtype=torch.long, device=selected_experts.device)
    chosen_pair = torch.where(improved, best_pair, no_pair)
    chosen_cost = torch.where(improved, best_cost, current_cost)
    return torch.stack(
        (chosen_cost.to(torch.float32), chosen_pair[0].to(torch.float32), chosen_pair[1].to(torch.float32))
    )


def estimate_best_swap_pair(
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    perf_model: HierMoEPerfModel,
    gamma: float = 10.0,
    placement: ExpertPlacement | torch.Tensor | None = None,
    candidate_pairs: list[tuple[int, int]] | torch.Tensor | None = None,
) -> tuple[tuple[int, int] | None, float]:
    """Estimate the best logical expert pair without mutating placement state."""

    if num_experts < 2:
        return None, float("inf")

    if placement is None:
        logical_to_physical = torch.arange(num_experts, dtype=torch.long)
    elif isinstance(placement, ExpertPlacement):
        logical_to_physical = torch.tensor(placement.logical_to_physical, dtype=torch.long)
    else:
        logical_to_physical = placement.detach().to(torch.long)

    row = (
        _estimate_best_swap_pair_row(
            selected_experts=selected_experts,
            num_experts=num_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            hierarchy=hierarchy,
            perf_model=perf_model,
            gamma=gamma,
            logical_to_physical=logical_to_physical,
            candidate_pairs=candidate_pairs,
        )
        .detach()
        .to(torch.device("cpu"))
    )
    lhs = int(row[1].item())
    rhs = int(row[2].item())
    if lhs < 0 or rhs < 0:
        return None, float(row[0].item())
    return (lhs, rhs), float(row[0].item())


class ExpertSwapManager:
    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        expert_swap_interval: int,
        expert_swap_max_pairs_per_layer: int,
        redundant_slot_increment_per_device: int,
        max_replica_rounds: int,
        smooth_max_gamma: float,
        hierarchy: Hierarchy,
        perf_model: HierMoEPerfModel,
        expert_swap_mode: str = "step",
        expert_swap_selector: str = "current_joint",
        activation_checkpointing_enabled: bool = False,
        gradient_bytes_per_element: int = 4,
        configured_max_replica_rounds: int | None = None,
        replica_slot_capacity: int | None = None,
        planner_route_sample_size: int = 1024,
        debug_validate: bool = False,
    ) -> None:
        self.ep_group = ep_group
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.expert_swap_interval = int(expert_swap_interval)
        self.expert_swap_max_pairs_per_layer = max(0, int(expert_swap_max_pairs_per_layer))
        self.redundant_slot_increment_per_device = max(0, int(redundant_slot_increment_per_device))
        self.max_replica_rounds = max(0, int(max_replica_rounds))
        self.configured_max_replica_rounds = (
            None if configured_max_replica_rounds is None else max(0, int(configured_max_replica_rounds))
        )
        self.replica_slot_capacity = (
            self.redundant_slot_increment_per_device * self.ep_size
            if replica_slot_capacity is None
            else max(0, int(replica_slot_capacity))
        )
        if planner_route_sample_size <= 0:
            raise ValueError("planner_route_sample_size must be positive.")
        self.planner_route_sample_size = int(planner_route_sample_size)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.expert_swap_mode = str(expert_swap_mode)
        self.expert_swap_selector = str(expert_swap_selector)
        if self.expert_swap_selector not in {"current_joint", "hiermoe_exact_p1", "legacy_batched"}:
            raise ValueError("expert_swap_selector must be current_joint, hiermoe_exact_p1, or legacy_batched.")
        if self.expert_swap_selector == "hiermoe_exact_p1":
            if self.expert_swap_max_pairs_per_layer != 1:
                raise ValueError("hiermoe_exact_p1 requires expert_swap_max_pairs_per_layer=1.")
            if self.redundant_slot_increment_per_device != 0 or self.max_replica_rounds != 0:
                raise ValueError("hiermoe_exact_p1 does not support redundant expert slots.")
        if self.expert_swap_selector == "legacy_batched":
            if self.redundant_slot_increment_per_device != 0 or self.max_replica_rounds != 0:
                raise ValueError("legacy_batched does not support redundant expert slots.")
            if self.expert_swap_mode != "step":
                raise ValueError("legacy_batched requires expert_swap_mode=step.")
        self.activation_checkpointing_enabled = bool(activation_checkpointing_enabled)
        self._swap_group = (
            _create_expert_swap_process_group(ep_group, self.ep_size)
            if self.expert_swap_max_pairs_per_layer > 0
            else None
        )
        self.gradient_bytes_per_element = max(1, int(gradient_bytes_per_element))
        self.debug_validate = bool(debug_validate)
        self.layers: dict[str, ExpertLayerState] = {}
        self.module_id_to_key: dict[int, str] = {}
        self.param_id_to_key: dict[int, str] = {}
        self.optimizer: Any = None
        self._optimizer_param_bindings: dict[int, tuple[_OptimizerParamBinding, ...]] = {}
        # Replica-gradient waves are executed layer by layer and synchronously
        # waited. Reuse one manager-wide staging pool instead of retaining a
        # send/receive pair for every layer.
        self._replica_grad_buffers: dict[tuple[str, int, str, str, int], torch.Tensor] = {}
        self._swap_staging_buffers: dict[tuple[torch.device, torch.dtype], _SwapStagingBuffer] = {}
        self._swap_comm_streams: dict[torch.device, Any] = {}
        self._pending_layer_swaps: dict[str, _PendingLayerSwap] = {}
        self._exact_candidate_pair_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.latest_pair: str = "none"
        self._pending_state: dict[str, Any] | None = None
        self._placement_metrics: dict[str, float | int | str] = {}
        self._metrics_step = -1

    def placement_planning_enabled(self) -> bool:
        return self.expert_swap_max_pairs_per_layer > 0 or self.redundant_slot_increment_per_device > 0

    def layer_calibration_enabled(self) -> bool:
        return self.expert_swap_selector == "current_joint"

    def placement_metrics(self) -> dict[str, float | int | str]:
        return dict(self._placement_metrics)

    def _begin_metrics_step(self, step: int) -> None:
        if self._metrics_step == int(step):
            return
        self._metrics_step = int(step)
        self._placement_metrics = {
            "hiermoe/placement_replica_rounds_configured": (
                "auto" if self.configured_max_replica_rounds is None else self.configured_max_replica_rounds
            ),
            "hiermoe/placement_replica_slot_capacity": self.replica_slot_capacity,
            "hiermoe/placement_replica_rounds_effective": self.max_replica_rounds,
            "hiermoe/placement_route_sample_size": self.planner_route_sample_size,
            "hiermoe/placement_runtime_cost_model": self.perf_model.runtime_cost_status,
            "hiermoe/expert_swap_selector": self.expert_swap_selector,
        }

    def _accumulate_metric(self, key: str, value: float | int | str) -> None:
        if isinstance(value, str):
            self._placement_metrics[key] = value
        elif isinstance(value, int):
            self._placement_metrics[key] = int(self._placement_metrics.get(key, 0)) + value
        else:
            self._placement_metrics[key] = float(self._placement_metrics.get(key, 0.0)) + float(value)

    def bind_optimizer(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self._optimizer_param_bindings = _build_optimizer_param_bindings(optimizer)

    def _debug_should_log_copy_stats(self, layer_key: str) -> bool:
        if not _DEBUG_REDUNDANT_COPY_STATS:
            return False
        layer_keys = sorted(key for key, layer in self.layers.items() if layer.slot_layout_enabled)
        return layer_key in set(layer_keys[:_DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS])

    @staticmethod
    def _debug_slot_stats(tensor: torch.Tensor, local_slot: int) -> torch.Tensor:
        local = _local_tensor_view(tensor).detach()
        values = local[int(local_slot)].to(dtype=torch.float32)
        if values.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=values.device)
            return torch.stack((zero, zero, zero, zero))
        return torch.stack((values.sum(), values.square().sum(), values.abs().max(), values.mean()))

    def _debug_global_slot_stats(self, tensor: torch.Tensor, layer: ExpertLayerState, slot: int) -> torch.Tensor:
        local = _local_tensor_view(tensor)
        stats = torch.zeros((4,), dtype=torch.float32, device=local.device)
        slot_rank, local_slot = divmod(int(slot), layer.num_local_experts)
        if self.ep_rank == slot_rank:
            stats = self._debug_slot_stats(tensor, local_slot)
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.ep_group)
        return stats

    def _debug_copy_group_stat_delta(
        self,
        tensor: torch.Tensor,
        layer: ExpertLayerState,
        slots: tuple[int, ...],
    ) -> float:
        ref_stats: torch.Tensor | None = None
        max_delta = 0.0
        for slot in slots:
            stats = self._debug_global_slot_stats(tensor, layer, int(slot))
            if ref_stats is None:
                ref_stats = stats
                continue
            delta = float((stats - ref_stats).abs().max().detach().cpu().item())
            max_delta = max(max_delta, delta)
        return max_delta

    def _debug_global_accumulated_counts(self, layer: ExpertLayerState) -> torch.Tensor:
        local_device = _local_tensor_view(layer.gate_up_proj).device
        counts = layer.accumulated_tokens_per_local_expert
        if counts is None or counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            local_counts = torch.zeros((layer.num_local_experts,), dtype=torch.float32, device=local_device)
        else:
            local_counts = counts.detach().to(device=local_device, dtype=torch.float32)
        if self.ep_group is None or self.ep_size <= 1:
            return local_counts.detach().cpu()
        gathered = torch.empty(
            (self.ep_size * layer.num_local_experts,),
            dtype=torch.float32,
            device=local_device,
        )
        dist.all_gather_into_tensor(gathered, local_counts.contiguous(), group=self.ep_group)
        return gathered.detach().cpu()

    def _debug_log_redundant_copy_stats(
        self,
        phase: str,
        *,
        layer_key: str | None = None,
        layer: ExpertLayerState | None = None,
        include_grads: bool = False,
    ) -> None:
        if not _DEBUG_REDUNDANT_COPY_STATS:
            return
        if layer_key is not None and layer is not None:
            items = [(layer_key, layer)] if self._debug_should_log_copy_stats(layer_key) else []
        else:
            items = [
                (key, candidate)
                for key, candidate in sorted(self.layers.items())
                if self._debug_should_log_copy_stats(key)
            ]

        for key, candidate in items:
            groups = candidate.redundant_copy_groups()
            if not groups:
                continue
            param_delta = 0.0
            grad_delta = 0.0
            grad_groups = 0
            global_counts = self._debug_global_accumulated_counts(candidate) if include_grads else None
            worst_grad_logical = -1
            worst_grad_slots: tuple[int, ...] = ()
            worst_grad_counts: tuple[float, ...] = ()
            for _logical_expert, slots in groups:
                for _param_name, param in (
                    ("gate_up_proj", candidate.gate_up_proj),
                    ("down_proj", candidate.down_proj),
                ):
                    param_delta = max(param_delta, self._debug_copy_group_stat_delta(param, candidate, slots))
                    grad = getattr(param, "grad", None)
                    if not include_grads or not torch.is_tensor(grad):
                        continue
                    if tuple(_local_tensor_view(grad).shape) != tuple(_local_tensor_view(param).shape):
                        continue
                    group_grad_delta = self._debug_copy_group_stat_delta(grad, candidate, slots)
                    grad_delta = max(grad_delta, group_grad_delta)
                    grad_groups += 1
                    if group_grad_delta >= grad_delta and global_counts is not None:
                        worst_grad_logical = int(_logical_expert)
                        worst_grad_slots = tuple(int(slot) for slot in slots)
                        worst_grad_counts = tuple(float(global_counts[int(slot)].item()) for slot in slots)
            if self.ep_rank == 0:
                logger.warning(
                    "HierMoE redundant copy stats phase=%s layer=%s groups=%s "
                    "param_stat_delta=%.6g grad_stat_delta=%.6g grad_groups=%s "
                    "worst_grad_logical=%s worst_grad_slots=%s worst_grad_counts=%s",
                    phase,
                    key,
                    len(groups),
                    param_delta,
                    grad_delta,
                    grad_groups,
                    worst_grad_logical,
                    worst_grad_slots,
                    worst_grad_counts,
                )

    def register_model(self, model: nn.Module) -> None:
        for key, module in model.named_modules():
            if self._is_expert_module(module):
                self.register_layer(key, module)
        if _FIXED_R2_LAYOUT:
            self.install_fixed_r2_layout()
        if self._pending_state is not None:
            self.load_state_dict(self._pending_state)
            self._pending_state = None

    @torch.no_grad()
    def install_fixed_r2_layout(self) -> None:
        """Install the static two-copy experiment layout before the first forward."""

        if self.ep_size <= 1 or self.ep_size % 2 != 0:
            raise ValueError(f"Fixed R2 requires a positive even EP size, got {self.ep_size}.")
        half_ep = self.ep_size // 2
        incompatible_groups = tuple(
            int(group_size)
            for group_size in self.hierarchy.group_sizes
            if 1 < int(group_size) < self.ep_size and half_ep % int(group_size) != 0
        )
        if incompatible_groups:
            raise ValueError(
                f"Fixed R2 requires every proper hierarchy group size to divide half EP={half_ep}, "
                f"got {incompatible_groups}."
            )
        for key, layer in self.layers.items():
            if not layer.slot_layout_enabled or layer.slot_to_logical is None:
                raise ValueError(f"Fixed R2 requires reserved redundant slots for layer {key}.")
            if layer.num_experts % half_ep != 0:
                raise ValueError(f"Fixed R2 requires num_experts={layer.num_experts} divisible by half EP={half_ep}.")
            expected_slots_per_rank = layer.num_experts // half_ep
            if layer.num_local_experts != expected_slots_per_rank:
                raise ValueError(
                    f"Fixed R2 layer {key} requires {expected_slots_per_rank} slots per rank, "
                    f"got {layer.num_local_experts}."
                )

            logical = torch.arange(layer.num_experts, dtype=torch.long)
            rank_in_half = torch.div(logical, layer.num_local_experts, rounding_mode="floor")
            local_slot = torch.remainder(logical, layer.num_local_experts)
            first_slots = rank_in_half * layer.num_local_experts + local_slot
            second_slots = (half_ep + rank_in_half) * layer.num_local_experts + local_slot
            target_layout = torch.full((layer.num_physical_slots,), -1, dtype=torch.long)
            target_layout[first_slots] = logical
            target_layout[second_slots] = logical

            current_layout = layer.slot_to_logical.detach().cpu()
            if torch.equal(current_layout, target_layout):
                self._refresh_layer_mapping_from_slots(layer, tuple(int(slot) for slot in first_slots.tolist()))
                layer.fixed_r2_layout = True
                continue

            state_tensors = (
                [layer.gate_up_proj, layer.down_proj] if self.optimizer is None else self._slot_op_state_tensors(layer)
            )
            grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
            for dst_slot, logical_expert in enumerate(target_layout.tolist()):
                if int(current_layout[dst_slot].item()) == logical_expert:
                    continue
                source_slots = torch.nonzero(current_layout == logical_expert, as_tuple=False).flatten()
                if source_slots.numel() == 0:
                    raise RuntimeError(
                        f"Fixed R2 cannot find source state for logical expert {logical_expert} in layer {key}."
                    )
                src_slot = int(source_slots[0].item())
                src_rank = src_slot // layer.num_local_experts
                dst_rank = dst_slot // layer.num_local_experts
                grouped_entries[(src_rank, dst_rank)].extend(
                    self._slot_op_cover_entries_from_tensors(
                        state_tensors,
                        num_local_experts=layer.num_local_experts,
                        src_slot=src_slot,
                        dst_slot=dst_slot,
                    )
                )

            _cover_grouped_slot_entries_atomic(
                grouped_entries,
                self.ep_rank,
                self.ep_size,
                self.ep_group,
                debug_validate=self.debug_validate,
            )
            layer.slot_to_logical = target_layout
            self._refresh_layer_mapping_from_slots(layer, tuple(int(slot) for slot in first_slots.tolist()))
            layer.active_quota_policy = ()
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            layer.fixed_r2_layout = True

        logger.info_rank0("HierMoE installed the fixed R2 layout for %s layer(s).", len(self.layers))

    def _is_expert_module(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "num_experts")
            and hasattr(module, "gate_up_proj")
            and hasattr(module, "down_proj")
            and isinstance(module.gate_up_proj, torch.Tensor)
            and isinstance(module.down_proj, torch.Tensor)
        )

    def register_layer(self, key: str, module: nn.Module) -> None:
        gate_up_proj = module.gate_up_proj
        down_proj = module.down_proj
        local_gate_up_proj = _local_tensor_view(gate_up_proj)
        _local_tensor_view(down_proj)

        num_experts = int(module.num_experts)
        num_local_experts = int(local_gate_up_proj.shape[0])
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"HierMoE layer {key} has {num_local_experts=} and {num_experts=} with ep_size={self.ep_size}."
            )
        base_num_local_experts = num_experts // self.ep_size
        if num_local_experts not in {
            base_num_local_experts,
            base_num_local_experts + self.redundant_slot_increment_per_device,
        }:
            raise ValueError(
                f"HierMoE layer {key} has {num_local_experts=} and {num_experts=} with ep_size={self.ep_size}."
            )
        slot_layout_enabled = (
            self.redundant_slot_increment_per_device > 0 and num_local_experts > base_num_local_experts
        )
        canonical_slots = (
            _canonical_physical_slots(num_experts, base_num_local_experts, num_local_experts)
            if slot_layout_enabled
            else None
        )
        layer = self.layers.get(key)
        if slot_layout_enabled:
            mapping = canonical_slots.clone() if layer is None else layer.logical_to_physical.detach().cpu().clone()
            slot_to_logical = (
                _initial_slot_to_logical(num_experts, base_num_local_experts, num_local_experts, self.ep_size)
                if layer is None or layer.slot_to_logical is None
                else layer.slot_to_logical.detach().cpu().clone()
            )
            is_identity = torch.equal(
                slot_to_logical,
                _initial_slot_to_logical(num_experts, base_num_local_experts, num_local_experts, self.ep_size),
            )
        else:
            mapping = (
                torch.arange(num_experts, dtype=torch.long)
                if layer is None
                else layer.logical_to_physical.detach().cpu().clone()
            )
            slot_to_logical = None
            is_identity = torch.equal(mapping, torch.arange(num_experts, dtype=torch.long))
        self.layers[key] = ExpertLayerState(
            key=key,
            module_id=id(module),
            num_experts=num_experts,
            base_num_local_experts=base_num_local_experts,
            num_local_experts=num_local_experts,
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            logical_to_physical=mapping,
            slot_to_logical=slot_to_logical,
            canonical_physical_slots=canonical_slots,
            is_identity=bool(is_identity),
        )
        self.module_id_to_key[id(module)] = key
        self.param_id_to_key[id(gate_up_proj)] = key
        self.param_id_to_key[id(down_proj)] = key

    def get_layer_key(self, module: nn.Module) -> str | None:
        return self.module_id_to_key.get(id(module))

    def get_layer_key_from_params(self, *params: torch.Tensor | None) -> str | None:
        for param in params:
            if param is None:
                continue
            key = self.param_id_to_key.get(id(param))
            if key is not None:
                return key
        return None

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    @staticmethod
    def _uses_compact_identity_dispatch(layer: ExpertLayerState) -> bool:
        return layer.slot_layout_enabled and layer.is_identity and not layer.redundant_copy_groups()

    @staticmethod
    def _validate_checkpoint_replay(
        layer: ExpertLayerState,
        selected_experts: torch.Tensor,
        planned_routes: torch.Tensor,
    ) -> torch.Tensor:
        if ExpertSwapManager._uses_compact_identity_dispatch(layer):
            return selected_experts
        selected = selected_experts.to(torch.long)
        owner = layer.mapping_for_device(selected.device).index_select(0, selected.reshape(-1)).view_as(selected)
        if not layer.slot_layout_enabled:
            return owner
        copy_slots, copy_mask = layer.copy_slots_for_device(selected.device)
        selected_copy_slots = copy_slots.index_select(0, selected.reshape(-1))
        selected_copy_mask = copy_mask.index_select(0, selected.reshape(-1))
        planned = planned_routes.to(dtype=torch.long)
        valid = ((selected_copy_slots == planned.reshape(-1, 1)) & selected_copy_mask).any(dim=-1)
        return torch.where(valid.view_as(selected), planned, owner)

    def map_logical_to_physical(
        self,
        layer_key: str,
        selected_experts: torch.Tensor,
        *,
        checkpoint_recompute: bool = False,
        checkpoint_replay: Any | None = None,
    ) -> torch.Tensor:
        layer = self.layers.get(layer_key)
        if layer is None:
            return selected_experts
        if checkpoint_recompute and checkpoint_replay is not None:
            replay = checkpoint_replay.next(layer_key)
            if (
                replay is not None
                and replay.device == selected_experts.device
                and replay.shape == selected_experts.shape
            ):
                return self._validate_checkpoint_replay(layer, selected_experts, replay)
        pending = layer.pending_physical_routes
        if (
            pending is not None
            and pending.device == selected_experts.device
            and pending.shape == selected_experts.shape
            and layer.pending_route_data_ptr == selected_experts.data_ptr()
        ):
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            dispatched_routes = selected_experts if self._uses_compact_identity_dispatch(layer) else pending
            if checkpoint_replay is not None and not checkpoint_recompute:
                checkpoint_replay.record(layer_key, dispatched_routes)
            return dispatched_routes
        if checkpoint_replay is not None and not checkpoint_recompute:
            checkpoint_replay.record(layer_key, None)
        if layer.slot_layout_enabled:
            if self._uses_compact_identity_dispatch(layer):
                return selected_experts
            return self._map_logical_to_slot(layer, selected_experts)
        if layer.is_identity:
            return selected_experts
        mapping = layer.mapping_for_device(selected_experts.device)
        return mapping.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)

    def num_physical_slots(self, layer_key: str, fallback_num_experts: int) -> int:
        layer = self.layers.get(layer_key)
        if layer is None or not layer.slot_layout_enabled:
            return int(fallback_num_experts)
        if self._uses_compact_identity_dispatch(layer):
            return int(fallback_num_experts)
        return int(layer.num_physical_slots)

    def _map_logical_to_slot(self, layer: ExpertLayerState, selected_experts: torch.Tensor) -> torch.Tensor:
        original_ndim = selected_experts.ndim
        selected = selected_experts.to(torch.long)
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)

        mapping = layer.mapping_for_device(selected.device)
        chosen = mapping.index_select(0, selected.reshape(-1)).view_as(selected)
        redundant_groups = layer.redundant_copy_groups_for_device(selected.device)
        if not redundant_groups:
            return chosen.squeeze(-1) if original_ndim == 1 else chosen

        return self._map_logical_to_slot_dedup_aware(layer, selected, chosen, redundant_groups, original_ndim)

    def _map_logical_to_slot_dedup_aware(
        self,
        layer: ExpertLayerState,
        selected: torch.Tensor,
        chosen: torch.Tensor,
        redundant_groups: tuple[tuple[int, torch.Tensor], ...],
        original_ndim: int,
    ) -> torch.Tensor:
        del chosen, redundant_groups
        if layer.slot_to_logical is None:
            raise RuntimeError(f"HierMoE layer {layer.key} has no physical slot layout.")
        if layer.active_quota_policy:
            mapping = assign_tokens_to_copies_with_quota(
                selected,
                layer.slot_to_logical,
                slots_per_rank=layer.num_local_experts,
                source_ranks=self.ep_rank,
                hierarchy=self.hierarchy,
                owner_slots=layer.logical_to_physical,
                quota_policy=layer.active_quota_policy,
                step=max(0, int(layer.latest_route_step)),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            )
            physical = mapping.physical_slots
            return physical.squeeze(-1) if original_ndim == 1 else physical
        copy_slots, copy_mask = layer.copy_slots_for_device(selected.device)
        if layer.fixed_r2_layout:
            physical = assign_tokens_to_mirrored_r2(
                selected,
                copy_slots,
                source_ranks=self.ep_rank,
                num_ranks=self.ep_size,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
        physical = assign_tokens_to_copies(
            selected,
            layer.slot_to_logical,
            slots_per_rank=layer.num_local_experts,
            source_ranks=self.ep_rank,
            hierarchy_group_sizes=self.hierarchy.group_sizes,
            owner_slots=layer.logical_to_physical,
            step=max(0, int(layer.latest_route_step)),
            layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            copy_slots=copy_slots,
            copy_mask=copy_mask,
            validate_copy_table=False,
        )
        return physical.squeeze(-1) if original_ndim == 1 else physical

    def record_routing(
        self,
        *,
        layer_key: str,
        selected_experts: torch.Tensor,
        hidden_size: int,
        bytes_per_element: int,
        step: int | None = None,
    ) -> None:
        layer = self.layers.get(layer_key)
        if layer is None:
            return
        layer.latest_selected_experts = selected_experts.detach()
        if step is not None:
            layer.latest_route_step = int(step)
        layer.latest_hidden_size = int(hidden_size)
        layer.latest_bytes_per_element = int(bytes_per_element)

    def mark_route_step(self, layer_key: str, step: int) -> None:
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_route_step = int(step)

    @staticmethod
    def _timing_event() -> AcceleratorEvent:
        event = record_accelerator_event()
        if event is not None:
            return event
        return AcceleratorEvent(device_type=get_device_type(), event=None, wall_time=time.perf_counter())

    def placement_timing_event(self) -> AcceleratorEvent:
        return self._timing_event()

    def record_dispatch_statistics(self, **_kwargs: Any) -> None:
        # The planner derives exact duplicate-free statistics from the raw
        # token-top-k routes. Runtime split summaries are intentionally unused.
        return

    def record_layer_timing(
        self,
        *,
        layer_key: str,
        step: int,
        selected_experts: torch.Tensor,
        tokens_per_local_expert: torch.Tensor,
        dispatch_start: AcceleratorEvent,
        dispatch_end: AcceleratorEvent,
        compute_start: AcceleratorEvent,
        compute_end: AcceleratorEvent,
        combine_start: AcceleratorEvent,
        combine_end: AcceleratorEvent,
        selected_dim: int | None = None,
    ) -> None:
        del selected_dim, tokens_per_local_expert
        layer = self.layers.get(layer_key)
        if layer is None or not self.placement_planning_enabled():
            return
        if layer.slot_to_logical is None:
            layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
            logical = torch.arange(layer.num_experts, dtype=torch.long)
            layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        else:
            layout = layer.slot_to_logical.detach().cpu().clone()
        layer.pending_timing = _PendingLayerTiming(
            step=int(step),
            selected_experts=selected_experts.detach(),
            slot_to_logical=layout,
            dispatch_start=dispatch_start,
            dispatch_end=dispatch_end,
            compute_start=compute_start,
            compute_end=compute_end,
            combine_start=combine_start,
            combine_end=combine_end,
        )

    def record_local_expert_token_counts(self, layer_key: str, tokens_per_local_expert: torch.Tensor) -> None:
        layer = self.layers.get(layer_key)
        if layer is None or not layer.slot_layout_enabled:
            return
        counts = tokens_per_local_expert.detach()
        if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            return
        if layer.accumulated_tokens_per_local_expert is None:
            layer.accumulated_tokens_per_local_expert = counts.clone()
        else:
            if tuple(layer.accumulated_tokens_per_local_expert.shape) != tuple(counts.shape):
                layer.accumulated_tokens_per_local_expert = counts.clone()
            else:
                layer.accumulated_tokens_per_local_expert = layer.accumulated_tokens_per_local_expert.to(
                    device=counts.device, dtype=counts.dtype
                )
                layer.accumulated_tokens_per_local_expert.add_(counts)

    @staticmethod
    def _zero_grad_slots(param: torch.nn.Parameter, zero_slots: torch.Tensor) -> None:
        grad = getattr(param, "grad", None)
        if not torch.is_tensor(grad):
            return
        local_grad = _local_tensor_view(grad)
        if tuple(local_grad.shape) != tuple(_local_tensor_view(param).shape):
            return
        local_zero_slots = zero_slots.to(device=local_grad.device, dtype=torch.bool)
        slot_mask_shape = (int(local_zero_slots.numel()),) + (1,) * (local_grad.ndim - 1)
        local_grad.detach().masked_fill_(local_zero_slots.view(slot_mask_shape), 0)

    @staticmethod
    def _local_grad_for_redundant_sync(param: torch.nn.Parameter) -> torch.Tensor | None:
        grad = getattr(param, "grad", None)
        if not torch.is_tensor(grad):
            # Copy ranks must all participate in redundant-gradient sync. A missing
            # grad is an explicit zero contribution, not a reason to skip P2P.
            param.grad = torch.zeros_like(param)
            grad = param.grad
        local_grad = _local_tensor_view(grad)
        if tuple(local_grad.shape) != tuple(_local_tensor_view(param).shape):
            return None
        return local_grad

    @torch.no_grad()
    def _zero_inactive_slot_grads(self) -> None:
        for layer in self.layers.values():
            counts = layer.accumulated_tokens_per_local_expert
            if counts is None or not layer.slot_layout_enabled:
                continue
            if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
                layer.accumulated_tokens_per_local_expert = None
                continue
            zero_slots = counts <= 0
            # NPU grouped-matmul backward may leave undefined weight gradients
            # for zero-token groups. Those slots are mathematically inactive.
            self._zero_grad_slots(layer.gate_up_proj, zero_slots)
            self._zero_grad_slots(layer.down_proj, zero_slots)

    def _clear_accumulated_token_counts(self) -> None:
        for layer in self.layers.values():
            layer.accumulated_tokens_per_local_expert = None

    def _slot_layout_is_device_unique(self, layer: ExpertLayerState, slot_to_logical: torch.Tensor) -> bool:
        for rank in range(self.ep_size):
            seen: set[int] = set()
            start = rank * layer.num_local_experts
            for slot in range(start, start + layer.num_local_experts):
                logical = int(slot_to_logical[slot].item())
                if logical < 0:
                    continue
                if logical in seen:
                    return False
                seen.add(logical)
        return True

    def _validate_placement_layout(
        self,
        layer: ExpertLayerState,
        slot_to_logical: torch.Tensor | Iterable[int],
        owner_slots: torch.Tensor | Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layout = torch.as_tensor(slot_to_logical, dtype=torch.long).detach().cpu().reshape(-1).clone()
        if layout.numel() != layer.num_physical_slots:
            raise ValueError(
                f"HierMoE slot layout for {layer.key} has {layout.numel()} slots, expected {layer.num_physical_slots}."
            )
        if bool(((layout < -1) | (layout >= layer.num_experts)).any().item()):
            raise ValueError(f"HierMoE slot layout for {layer.key} contains an invalid logical expert.")
        active = layout[layout >= 0]
        counts = (
            torch.bincount(active, minlength=layer.num_experts)
            if active.numel()
            else torch.zeros((layer.num_experts,), dtype=torch.long)
        )
        if bool((counts <= 0).any().item()):
            raise ValueError(f"HierMoE slot layout for {layer.key} drops at least one logical expert.")
        if not self._slot_layout_is_device_unique(layer, layout):
            raise ValueError(f"HierMoE slot layout for {layer.key} duplicates an expert on one device.")

        if owner_slots is None:
            return layout, None
        owners = torch.as_tensor(owner_slots, dtype=torch.long).detach().cpu().reshape(-1).clone()
        if owners.numel() != layer.num_experts:
            raise ValueError(
                f"HierMoE owner mapping for {layer.key} has {owners.numel()} entries, expected {layer.num_experts}."
            )
        if bool(((owners < 0) | (owners >= layout.numel())).any().item()):
            raise ValueError(f"HierMoE owner mapping for {layer.key} contains an invalid slot.")
        if len({int(value) for value in owners.tolist()}) != layer.num_experts:
            raise ValueError(f"HierMoE owner mapping for {layer.key} contains duplicate owner slots.")
        for logical_expert, physical_slot in enumerate(owners.tolist()):
            if int(layout[int(physical_slot)].item()) != logical_expert:
                raise ValueError(
                    f"HierMoE owner slot {physical_slot} for logical expert {logical_expert} "
                    "does not contain that expert."
                )
        return layout, owners

    def _validate_quota_policy(
        self,
        layer: ExpertLayerState,
        slot_to_logical: torch.Tensor | Iterable[int],
        quota_policy: Iterable[QuotaPolicyEntry],
    ) -> tuple[QuotaPolicyEntry, ...]:
        layout = torch.as_tensor(slot_to_logical, dtype=torch.long).detach().cpu().reshape(-1)
        entries = tuple(quota_policy)
        policy_keys: set[tuple[int, int, tuple[int, ...], int | None]] = set()
        for entry in entries:
            if not 0 <= entry.source_rank < self.ep_size:
                raise ValueError(f"HierMoE quota policy for {layer.key} has an invalid source rank.")
            if not 0 <= entry.logical_expert < layer.num_experts:
                raise ValueError(f"HierMoE quota policy for {layer.key} has an invalid expert.")
            if not entry.destination_ranks or len(entry.destination_ranks) != len(entry.quotas):
                raise ValueError(f"HierMoE quota policy for {layer.key} has inconsistent quota widths.")
            if tuple(sorted(set(entry.destination_ranks))) != entry.destination_ranks or any(
                not 0 <= rank < self.ep_size for rank in entry.destination_ranks
            ):
                raise ValueError(f"HierMoE quota policy for {layer.key} has invalid destination ranks.")
            if any(quota < 0 for quota in entry.quotas):
                raise ValueError(f"HierMoE quota policy for {layer.key} has a negative quota.")
            if entry.multiplicity is not None and entry.multiplicity <= 0:
                raise ValueError(f"HierMoE quota policy for {layer.key} has an invalid multiplicity.")
            policy_key = (
                entry.source_rank,
                entry.logical_expert,
                entry.destination_ranks,
                entry.multiplicity,
            )
            if policy_key in policy_keys:
                raise ValueError(f"HierMoE quota policy for {layer.key} contains duplicate rows.")
            policy_keys.add(policy_key)
            copy_ranks = {
                slot // layer.num_local_experts
                for slot in torch.nonzero(layout == entry.logical_expert, as_tuple=False).flatten().tolist()
            }
            if any(rank not in copy_ranks for rank in entry.destination_ranks):
                raise ValueError(f"HierMoE quota policy for {layer.key} references a rank without a copy.")
        return entries

    def _validate_optimizer_state_slot_tensors_across_ep(
        self,
        rows: Iterable[tuple[torch.nn.Parameter, list[_SlotStateItem]]],
    ) -> None:
        state_rows = list(rows)
        if not state_rows or self.ep_group is None or self.ep_size <= 1:
            return

        bounds_rows = []
        for _param, items in state_rows:
            count = len(items)
            signature_rows = tuple(
                (
                    descriptor,
                    str(_local_tensor_view(tensor).dtype),
                    tuple(int(value) for value in _local_tensor_view(tensor).shape),
                )
                for descriptor, tensor in items
            )
            signature = zlib.crc32(repr(signature_rows).encode("utf-8"))
            bounds_rows.append((count, -count, signature, -signature))

        local_bounds = torch.tensor(
            bounds_rows,
            dtype=torch.long,
            device=_local_tensor_view(state_rows[0][0]).device,
        )
        global_bounds = local_bounds.clone()
        dist.all_reduce(global_bounds, op=dist.ReduceOp.MIN, group=self.ep_group)
        if not torch.equal(global_bounds, local_bounds):
            raise RuntimeError(
                "HierMoE cannot migrate an asymmetric swap payload across the EP group; "
                "ordered parameter, gradient, and optimizer-state descriptors must match on every rank."
            )

    def _optimizer_state_slot_items_for_slot_op(
        self,
        param: torch.nn.Parameter,
    ) -> list[_SlotStateItem]:
        known_state_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "compensation", "momentum_buffer")
        items: list[_SlotStateItem] = []
        bindings = self._optimizer_param_bindings.get(id(param), ())
        optimizers = tuple(binding.optimizer for binding in bindings) or tuple(_iter_leaf_optimizers(self.optimizer))
        for optimizer_index, optimizer in enumerate(optimizers):
            state = _existing_optimizer_state(optimizer, param)
            if not state:
                continue
            matching = [
                (state_name, value)
                for state_name, value in state.items()
                if torch.is_tensor(value)
                and tuple(_local_tensor_view(value).shape) == tuple(_local_tensor_view(param).shape)
            ]
            ordered = [item for known_name in known_state_names for item in matching if item[0] == known_name]
            ordered.extend(
                sorted(
                    (item for item in matching if item[0] not in known_state_names),
                    key=lambda item: str(item[0]),
                )
            )
            items.extend((f"optimizer[{optimizer_index}].{state_name}", tensor) for state_name, tensor in ordered)
        return items

    def _optimizer_state_slot_tensors_for_slot_op(
        self,
        param: torch.nn.Parameter,
    ) -> list[torch.Tensor]:
        return [tensor for _descriptor, tensor in self._optimizer_state_slot_items_for_slot_op(param)]

    def _slot_op_state_rows(
        self,
        layer: ExpertLayerState,
    ) -> list[tuple[torch.nn.Parameter, list[_SlotStateItem]]]:
        rows: list[tuple[torch.nn.Parameter, list[_SlotStateItem]]] = []
        for param in (layer.gate_up_proj, layer.down_proj):
            items: list[_SlotStateItem] = [("parameter", param)]
            grad = getattr(param, "grad", None)
            if torch.is_tensor(grad):
                if tuple(_local_tensor_view(grad).shape) != tuple(_local_tensor_view(param).shape):
                    raise RuntimeError(
                        f"HierMoE cannot migrate a gradient whose shape differs from parameter {tuple(param.shape)}."
                    )
                items.append(("gradient", grad))
            items.extend(self._optimizer_state_slot_items_for_slot_op(param))
            rows.append((param, items))
        return rows

    def _slot_op_state_tensors(self, layer: ExpertLayerState) -> list[torch.Tensor]:
        return [tensor for _param, items in self._slot_op_state_rows(layer) for _descriptor, tensor in items]

    @staticmethod
    def _slot_op_cover_entries_from_tensors(
        tensors: Iterable[torch.Tensor],
        *,
        num_local_experts: int,
        src_slot: int,
        dst_slot: int,
    ) -> list[_CoverTensorEntry]:
        src_local = int(src_slot) % int(num_local_experts)
        dst_local = int(dst_slot) % int(num_local_experts)
        return [_CoverTensorEntry(tensor, src_slot=src_local, dst_slot=dst_local) for tensor in tensors]

    def _apply_slot_op_to_layout(self, slot_to_logical: torch.Tensor, op: _SlotOpCandidate) -> torch.Tensor:
        updated = slot_to_logical.clone()
        if op.kind == "swap":
            updated[op.src_slot], updated[op.dst_slot] = updated[op.dst_slot].clone(), updated[op.src_slot].clone()
        elif op.kind == "cover":
            updated[op.dst_slot] = updated[op.src_slot].clone()
        else:
            raise ValueError(f"Unknown HierMoE slot op kind: {op.kind}")
        return updated

    def _slot_op_cover_entries(self, layer: ExpertLayerState, src_slot: int, dst_slot: int) -> list[_CoverTensorEntry]:
        return self._slot_op_cover_entries_from_tensors(
            self._slot_op_state_tensors(layer),
            num_local_experts=layer.num_local_experts,
            src_slot=src_slot,
            dst_slot=dst_slot,
        )

    def _slot_op_swap_plan(self, layer_key: str, op: _SlotOpCandidate) -> _LayerSwapPlan:
        layer = self.layers[layer_key]
        lhs_rank, lhs_slot = divmod(int(op.src_slot), layer.num_local_experts)
        rhs_rank, rhs_slot = divmod(int(op.dst_slot), layer.num_local_experts)
        entries = [
            _SwapTensorEntry(tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot)
            for tensor in self._slot_op_state_tensors(layer)
        ]
        return _LayerSwapPlan(
            layer_key=layer_key,
            logical_lhs=int(op.src_slot),
            logical_rhs=int(op.dst_slot),
            lhs_rank=int(lhs_rank),
            rhs_rank=int(rhs_rank),
            entries=tuple(entries),
        )

    def _refresh_layer_mapping_from_slots(
        self,
        layer: ExpertLayerState,
        owner_slots: torch.Tensor | Iterable[int] | None = None,
    ) -> None:
        if layer.slot_to_logical is None:
            return
        if owner_slots is not None:
            mapping = torch.as_tensor(owner_slots, dtype=torch.long).detach().cpu().reshape(-1).clone()
            if mapping.numel() != layer.num_experts:
                raise RuntimeError(
                    f"HierMoE owner mapping for {layer.key} has {mapping.numel()} entries, "
                    f"expected {layer.num_experts}."
                )
            for logical_expert, physical_slot in enumerate(mapping.tolist()):
                if not 0 <= int(physical_slot) < layer.num_physical_slots:
                    raise RuntimeError(
                        f"HierMoE owner slot {physical_slot} for logical expert {logical_expert} is out of range."
                    )
                if int(layer.slot_to_logical[int(physical_slot)].item()) != logical_expert:
                    raise RuntimeError(
                        f"HierMoE owner slot {physical_slot} does not contain logical expert {logical_expert}."
                    )
            layer.logical_to_physical = mapping
            layer.refresh_identity()
            layer.invalidate_cache()
            return
        mapping = torch.empty((layer.num_experts,), dtype=torch.long)
        for logical_expert in range(layer.num_experts):
            slots = torch.nonzero(layer.slot_to_logical == logical_expert, as_tuple=False).flatten()
            if slots.numel() == 0:
                raise RuntimeError(f"HierMoE slot layout lost all copies of logical expert {logical_expert}.")
            canonical = layer.canonical_physical_slots
            if canonical is not None and int(canonical[logical_expert].item()) in set(slots.tolist()):
                mapping[logical_expert] = int(canonical[logical_expert].item())
            else:
                mapping[logical_expert] = int(slots[0].item())
        layer.logical_to_physical = mapping
        layer.refresh_identity()
        layer.invalidate_cache()

    @staticmethod
    def _owner_rank_for_copy_group(
        layer: ExpertLayerState,
        logical_expert: int,
        slots: Iterable[int],
    ) -> int:
        logical = int(logical_expert)
        if not 0 <= logical < layer.num_experts:
            raise RuntimeError(f"HierMoE redundant gradient sync received invalid logical expert {logical}.")
        slot_values = tuple(int(slot) for slot in slots)
        owner_slot = int(layer.logical_to_physical[logical].item())
        if owner_slot not in slot_values:
            raise RuntimeError(
                f"HierMoE owner slot {owner_slot} for logical expert {logical} is not in its copy group."
            )
        return owner_slot // layer.num_local_experts

    @torch.no_grad()
    def _commit_layer_slot_ops(
        self,
        layer_key: str,
        ops: Iterable[_SlotOpCandidate],
        *,
        force_collective: bool = False,
    ) -> list[str]:
        layer = self.layers[layer_key]
        if layer.slot_to_logical is None:
            return []
        committed: list[str] = []
        for op in ops:
            src_rank, src_local = divmod(int(op.src_slot), layer.num_local_experts)
            dst_rank, dst_local = divmod(int(op.dst_slot), layer.num_local_experts)
            if op.kind == "swap":
                self._execute_swap_plans((self._slot_op_swap_plan(layer_key, op),), force_collective=force_collective)
            elif op.kind == "cover":
                entries = self._slot_op_cover_entries(layer, op.src_slot, op.dst_slot)
                if int(layer.slot_to_logical[op.src_slot].item()) < 0:
                    _zero_slot_entries(entries, dst_rank, self.ep_rank)
                else:
                    _cover_slot_entries(entries, src_rank, dst_rank, self.ep_rank, self.ep_group)
            else:
                raise ValueError(f"Unknown HierMoE slot op kind: {op.kind}")
            layer.slot_to_logical = self._apply_slot_op_to_layout(layer.slot_to_logical, op)
            layer.fixed_r2_layout = False
            self._refresh_layer_mapping_from_slots(layer)
            committed.append(f"{layer_key}:{op.format()}[{src_rank}:{src_local},{dst_rank}:{dst_local}]")
        return committed

    @torch.no_grad()
    def _redundant_grad_buckets_for_group(
        self,
        params: Iterable[torch.nn.Parameter],
        layer: ExpertLayerState,
        logical_expert: int,
        slots: list[int],
    ) -> list[_RedundantGradBucket]:
        owner_rank = self._owner_rank_for_copy_group(layer, logical_expert, slots)
        local_slots = [
            int(slot) % layer.num_local_experts
            for slot in slots
            if int(slot) // layer.num_local_experts == self.ep_rank
        ]
        if not local_slots:
            return []

        buckets: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[torch.Tensor, tuple[int, ...], torch.Size, int, torch.Tensor]],
        ] = defaultdict(list)
        for param in params:
            local_grad = self._local_grad_for_redundant_sync(param)
            if local_grad is None:
                continue
            local_sum = local_grad.detach()[local_slots[0]].clone()
            for local_slot in local_slots[1:]:
                local_sum.add_(local_grad.detach()[local_slot])
            flat_sum = local_sum.contiguous().view(-1)
            buckets[(flat_sum.device, flat_sum.dtype)].append(
                (local_grad, tuple(local_slots), local_sum.shape, int(flat_sum.numel()), flat_sum)
            )
        if not buckets:
            return []

        copy_ranks = tuple(sorted({int(slot) // layer.num_local_experts for slot in slots}))
        grad_buckets: list[_RedundantGradBucket] = []
        for bucket in buckets.values():
            send_buffer = torch.cat([item[4] for item in bucket], dim=0) if len(bucket) > 1 else bucket[0][4]
            items = tuple(
                _RedundantGradBucketItem(
                    local_grad=item[0],
                    local_slots=item[1],
                    shape=item[2],
                    numel=item[3],
                )
                for item in bucket
            )
            grad_buckets.append(
                _RedundantGradBucket(
                    owner_rank=owner_rank,
                    copy_ranks=copy_ranks,
                    items=items,
                    send_buffer=send_buffer,
                )
            )
        return grad_buckets

    @staticmethod
    def _unpack_redundant_grad_bucket(buffer: torch.Tensor, bucket: _RedundantGradBucket) -> None:
        offset = 0
        for item in bucket.items:
            synced = buffer[offset : offset + item.numel].view(item.shape)
            for local_slot in item.local_slots:
                item.local_grad.detach()[local_slot].copy_(synced)
            offset += item.numel

    def _sync_redundant_grad_bucket_wave(self, buckets: list[_RedundantGradBucket]) -> None:
        if not buckets:
            return
        if self.ep_group is None or self.ep_size <= 1:
            for bucket in buckets:
                self._unpack_redundant_grad_bucket(bucket.send_buffer, bucket)
            return

        for bucket in buckets:
            if self.ep_rank not in bucket.copy_ranks:
                continue
            if self.ep_rank == bucket.owner_rank:
                accum_buffer = bucket.send_buffer.clone()
                for src_rank in bucket.copy_ranks:
                    if src_rank == bucket.owner_rank:
                        continue
                    recv_buffer = torch.empty_like(bucket.send_buffer)
                    works = dist.batch_isend_irecv(
                        [dist.P2POp(dist.irecv, recv_buffer, _ep_global_rank(self.ep_group, src_rank))]
                    )
                    for work in works:
                        work.wait()
                    accum_buffer.add_(recv_buffer)
                bucket.accum_buffer = accum_buffer
            else:
                works = dist.batch_isend_irecv(
                    [
                        dist.P2POp(
                            dist.isend,
                            bucket.send_buffer,
                            _ep_global_rank(self.ep_group, bucket.owner_rank),
                        )
                    ]
                )
                for work in works:
                    work.wait()

            if self.ep_rank == bucket.owner_rank:
                if bucket.accum_buffer is None:
                    raise RuntimeError("HierMoE redundant gradient sync owner has no accumulated buffer.")
                scatter_ops: list[dist.P2POp] = []
                for dst_rank in bucket.copy_ranks:
                    if dst_rank == bucket.owner_rank:
                        continue
                    scatter_ops.append(
                        dist.P2POp(dist.isend, bucket.accum_buffer, _ep_global_rank(self.ep_group, dst_rank))
                    )
                if scatter_ops:
                    works = dist.batch_isend_irecv(scatter_ops)
                    for work in works:
                        work.wait()
                self._unpack_redundant_grad_bucket(bucket.accum_buffer, bucket)
            else:
                recv_buffer = torch.empty_like(bucket.send_buffer)
                works = dist.batch_isend_irecv(
                    [
                        dist.P2POp(
                            dist.irecv,
                            recv_buffer,
                            _ep_global_rank(self.ep_group, bucket.owner_rank),
                        )
                    ]
                )
                for work in works:
                    work.wait()
                self._unpack_redundant_grad_bucket(recv_buffer, bucket)

    @torch.no_grad()
    def _sync_redundant_grads_for_params_blocking(
        self,
        params: Iterable[torch.nn.Parameter],
        layer: ExpertLayerState,
        logical_expert: int,
        slots: list[int],
    ) -> None:
        owner_rank = self._owner_rank_for_copy_group(layer, logical_expert, slots)
        local_slots = [
            int(slot) % layer.num_local_experts
            for slot in slots
            if int(slot) // layer.num_local_experts == self.ep_rank
        ]
        if not local_slots:
            return

        buckets: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[torch.Tensor, tuple[int, ...], torch.Size, int, torch.Tensor]],
        ] = defaultdict(list)
        for param in params:
            local_grad = self._local_grad_for_redundant_sync(param)
            if local_grad is None:
                continue
            local_sum = local_grad.detach()[local_slots[0]].clone()
            for local_slot in local_slots[1:]:
                local_sum.add_(local_grad.detach()[local_slot])
            flat_sum = local_sum.contiguous().view(-1)
            buckets[(flat_sum.device, flat_sum.dtype)].append(
                (local_grad, tuple(local_slots), local_sum.shape, int(flat_sum.numel()), flat_sum)
            )
        if not buckets:
            return

        def _unpack_synced(
            buffer: torch.Tensor,
            bucket: list[tuple[torch.Tensor, tuple[int, ...], torch.Size, int, torch.Tensor]],
        ) -> None:
            offset = 0
            for local_grad, bucket_local_slots, shape, numel, _flat_sum in bucket:
                synced = buffer[offset : offset + numel].view(shape)
                for local_slot in bucket_local_slots:
                    local_grad.detach()[local_slot].copy_(synced)
                offset += numel

        if self.ep_group is None or self.ep_size <= 1:
            for bucket in buckets.values():
                send_buffer = torch.cat([item[4] for item in bucket], dim=0) if len(bucket) > 1 else bucket[0][4]
                _unpack_synced(send_buffer, bucket)
            return

        copy_ranks = sorted({int(slot) // layer.num_local_experts for slot in slots})
        owner_global_rank = _ep_global_rank(self.ep_group, owner_rank)
        for bucket in buckets.values():
            send_buffer = torch.cat([item[4] for item in bucket], dim=0) if len(bucket) > 1 else bucket[0][4]
            if self.ep_rank == owner_rank:
                accum = send_buffer.clone()
                for src_rank in copy_ranks:
                    if src_rank == owner_rank:
                        continue
                    recv_buffer = torch.empty_like(accum)
                    dist.recv(recv_buffer, src=_ep_global_rank(self.ep_group, src_rank))
                    accum.add_(recv_buffer)
                for dst_rank in copy_ranks:
                    if dst_rank == owner_rank:
                        continue
                    dist.send(accum, dst=_ep_global_rank(self.ep_group, dst_rank))
                _unpack_synced(accum, bucket)
            else:
                dist.send(send_buffer, dst=owner_global_rank)
                synced = torch.empty_like(send_buffer)
                dist.recv(synced, src=owner_global_rank)
                _unpack_synced(synced, bucket)

    def _replica_grad_schedule_for_layer(self, layer: ExpertLayerState) -> _ReplicaGradSchedule:
        cached = layer._replica_grad_schedule_cache
        if cached is not None:
            return cached

        groups = []
        for logical_expert, slots in layer.redundant_copy_groups():
            copy_ranks = tuple(sorted({int(slot) // layer.num_local_experts for slot in slots}))
            local_slots = tuple(
                int(slot) % layer.num_local_experts
                for slot in slots
                if int(slot) // layer.num_local_experts == self.ep_rank
            )
            groups.append(
                _ReplicaGradGroup(
                    logical_expert=int(logical_expert),
                    owner_rank=self._owner_rank_for_copy_group(layer, int(logical_expert), list(slots)),
                    copy_ranks=copy_ranks,
                    local_slots=local_slots,
                )
            )
        cached = _ReplicaGradSchedule(
            groups=tuple(groups),
            pairwise=all(len(group.copy_ranks) <= 2 for group in groups),
        )
        layer._replica_grad_schedule_cache = cached
        return cached

    def _replica_grad_contributions(
        self,
        layer: ExpertLayerState,
        schedule: _ReplicaGradSchedule,
    ) -> dict[tuple[torch.device, torch.dtype], dict[tuple[int, int], _ReplicaGradContribution]]:
        params = (layer.gate_up_proj, layer.down_proj)
        local_grads = tuple(self._local_grad_for_redundant_sync(param) for param in params)
        contributions: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ] = defaultdict(dict)
        for group in schedule.groups:
            if not group.local_slots:
                continue
            for param_index, local_grad in enumerate(local_grads):
                if local_grad is None:
                    raise RuntimeError(
                        f"HierMoE redundant gradient for layer {layer.key} does not match its local parameter shape."
                    )
                local_sum = local_grad.detach()[group.local_slots[0]].clone()
                for local_slot in group.local_slots[1:]:
                    local_sum.add_(local_grad.detach()[local_slot])
                contributions[(local_sum.device, local_sum.dtype)][(group.logical_expert, param_index)] = (
                    _ReplicaGradContribution(
                        logical_expert=group.logical_expert,
                        param_index=param_index,
                        local_grad=local_grad,
                        local_slots=group.local_slots,
                        local_sum=local_sum,
                    )
                )
        return contributions

    @staticmethod
    def _replica_grad_bucket_sort_key(
        key: tuple[torch.device, torch.dtype],
    ) -> tuple[str, int, str]:
        device, dtype = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype))

    @staticmethod
    def _unpack_replica_grad(total: torch.Tensor, contribution: _ReplicaGradContribution) -> None:
        synced = total.view_as(contribution.local_sum)
        for local_slot in contribution.local_slots:
            contribution.local_grad.detach()[local_slot].copy_(synced)

    def _replica_grad_buffer(
        self,
        *,
        kind: str,
        peer_rank: int,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (kind, int(peer_rank), str(device), str(dtype), int(numel))
        cached = self._replica_grad_buffers.get(key)
        if cached is None:
            cached = torch.empty((numel,), dtype=dtype, device=device)
            self._replica_grad_buffers[key] = cached
        return cached

    def _pack_replica_grad_items(
        self,
        *,
        kind: str,
        peer_rank: int,
        items: Iterable[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        parts = tuple(item.reshape(-1) for item in items)
        numel = sum(int(part.numel()) for part in parts)
        buffer = self._replica_grad_buffer(
            kind=kind,
            peer_rank=peer_rank,
            numel=numel,
            dtype=dtype,
            device=device,
        )
        offset = 0
        for part in parts:
            buffer[offset : offset + part.numel()].copy_(part)
            offset += int(part.numel())
        return buffer

    def _run_replica_grad_p2p_wave(
        self,
        *,
        phase: str,
        send_buffers: dict[int, torch.Tensor],
        recv_specs: dict[int, tuple[int, torch.dtype, torch.device]],
    ) -> dict[int, torch.Tensor]:
        if self.ep_group is None or self.ep_size <= 1:
            if send_buffers or recv_specs:
                raise RuntimeError("HierMoE redundant gradient synchronization requires an EP process group.")
            return {}

        recv_buffers = {
            peer_rank: self._replica_grad_buffer(
                kind=f"{phase}_recv",
                peer_rank=peer_rank,
                numel=numel,
                dtype=dtype,
                device=device,
            )
            for peer_rank, (numel, dtype, device) in recv_specs.items()
        }
        ops: list[dist.P2POp] = []
        for peer_rank in sorted(set(send_buffers) | set(recv_buffers)):
            peer_global_rank = _ep_global_rank(self.ep_group, peer_rank)
            send_buffer = send_buffers.get(peer_rank)
            if send_buffer is not None:
                ops.append(dist.P2POp(dist.isend, send_buffer, peer_global_rank))
            recv_buffer = recv_buffers.get(peer_rank)
            if recv_buffer is not None:
                ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_global_rank))
        if ops:
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()
        return recv_buffers

    def _sync_pairwise_replica_gradients(
        self,
        schedule: _ReplicaGradSchedule,
        contributions_by_bucket: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ],
    ) -> None:
        for bucket_key in sorted(contributions_by_bucket, key=self._replica_grad_bucket_sort_key):
            device, dtype = bucket_key
            contributions = contributions_by_bucket[bucket_key]
            peer_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            for group in schedule.groups:
                if not group.local_slots:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    remote_ranks = tuple(rank for rank in group.copy_ranks if rank != self.ep_rank)
                    if not remote_ranks:
                        self._unpack_replica_grad(contribution.local_sum, contribution)
                        continue
                    if len(remote_ranks) != 1:
                        raise RuntimeError("Pairwise redundant gradient schedule contains more than one remote copy.")
                    peer_items[remote_ranks[0]].append(contribution)

            send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="pairwise_send",
                    peer_rank=peer_rank,
                    items=(item.local_sum for item in items),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, items in peer_items.items()
            }
            recv_specs = {
                peer_rank: (int(send_buffer.numel()), dtype, device) for peer_rank, send_buffer in send_buffers.items()
            }
            recv_buffers = self._run_replica_grad_p2p_wave(
                phase="pairwise",
                send_buffers=send_buffers,
                recv_specs=recv_specs,
            )
            for peer_rank, items in peer_items.items():
                recv_buffer = recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    remote = recv_buffer[offset : offset + item.numel].view_as(item.local_sum)
                    total = item.local_sum + remote
                    self._unpack_replica_grad(total, item)
                    offset += item.numel

    def _sync_owner_replica_gradients(
        self,
        schedule: _ReplicaGradSchedule,
        contributions_by_bucket: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ],
    ) -> None:
        for bucket_key in sorted(contributions_by_bucket, key=self._replica_grad_bucket_sort_key):
            device, dtype = bucket_key
            contributions = contributions_by_bucket[bucket_key]
            reduce_send_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            reduce_recv_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            owner_totals: dict[tuple[int, int], torch.Tensor] = {}

            for group in schedule.groups:
                if self.ep_rank not in group.copy_ranks:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    item_key = (group.logical_expert, param_index)
                    if self.ep_rank == group.owner_rank:
                        owner_totals[item_key] = contribution.local_sum.clone()
                        for source_rank in group.copy_ranks:
                            if source_rank != group.owner_rank:
                                reduce_recv_items[source_rank].append(contribution)
                    else:
                        reduce_send_items[group.owner_rank].append(contribution)

            reduce_send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="reduce_send",
                    peer_rank=peer_rank,
                    items=(item.local_sum for item in items),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, items in reduce_send_items.items()
            }
            reduce_recv_specs = {
                peer_rank: (sum(item.numel for item in items), dtype, device)
                for peer_rank, items in reduce_recv_items.items()
            }
            reduce_recv_buffers = self._run_replica_grad_p2p_wave(
                phase="reduce",
                send_buffers=reduce_send_buffers,
                recv_specs=reduce_recv_specs,
            )
            for peer_rank, items in reduce_recv_items.items():
                recv_buffer = reduce_recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    remote = recv_buffer[offset : offset + item.numel].view_as(item.local_sum)
                    owner_totals[(item.logical_expert, item.param_index)].add_(remote)
                    offset += item.numel

            broadcast_send_keys: dict[int, list[tuple[int, int]]] = defaultdict(list)
            broadcast_recv_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            for group in schedule.groups:
                if self.ep_rank not in group.copy_ranks:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    item_key = (group.logical_expert, param_index)
                    if self.ep_rank == group.owner_rank:
                        for destination_rank in group.copy_ranks:
                            if destination_rank != group.owner_rank:
                                broadcast_send_keys[destination_rank].append(item_key)
                    else:
                        broadcast_recv_items[group.owner_rank].append(contribution)

            broadcast_send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="broadcast_send",
                    peer_rank=peer_rank,
                    items=(owner_totals[item_key] for item_key in item_keys),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, item_keys in broadcast_send_keys.items()
            }
            broadcast_recv_specs = {
                peer_rank: (sum(item.numel for item in items), dtype, device)
                for peer_rank, items in broadcast_recv_items.items()
            }
            broadcast_recv_buffers = self._run_replica_grad_p2p_wave(
                phase="broadcast",
                send_buffers=broadcast_send_buffers,
                recv_specs=broadcast_recv_specs,
            )

            for item_key, total in owner_totals.items():
                self._unpack_replica_grad(total, contributions[item_key])
            for peer_rank, items in broadcast_recv_items.items():
                recv_buffer = broadcast_recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    total = recv_buffer[offset : offset + item.numel]
                    self._unpack_replica_grad(total, item)
                    offset += item.numel

    @torch.no_grad()
    def sync_redundant_gradients(self) -> None:
        with _full_timing_range("hiermoe_redundant_grad_sync"):
            self._zero_inactive_slot_grads()
            self._debug_log_redundant_copy_stats("before_grad_sync", include_grads=True)
            for layer in self.layers.values():
                if layer.slot_to_logical is None:
                    continue
                schedule = self._replica_grad_schedule_for_layer(layer)
                if not schedule.groups:
                    continue
                contributions = self._replica_grad_contributions(layer, schedule)
                if schedule.pairwise:
                    self._sync_pairwise_replica_gradients(schedule, contributions)
                else:
                    self._sync_owner_replica_gradients(schedule, contributions)
            self._debug_log_redundant_copy_stats("after_grad_sync", include_grads=True)
            self._clear_accumulated_token_counts()

    def redundant_grad_norm_masks(self) -> dict[int, torch.Tensor]:
        masks: dict[int, torch.Tensor] = {}
        for layer in self.layers.values():
            if not layer.slot_layout_enabled or not layer.redundant_copy_groups():
                continue
            mask = torch.zeros((layer.num_local_experts,), dtype=torch.bool)
            for physical_slot in layer.logical_to_physical.tolist():
                rank, local_slot = divmod(int(physical_slot), layer.num_local_experts)
                if rank == self.ep_rank:
                    mask[local_slot] = True
            for param in (layer.gate_up_proj, layer.down_proj):
                masks[id(param)] = mask
        return masks

    def _planner_collective_backend(self) -> str | None:
        if self.ep_group is None or self.ep_size <= 1:
            return None
        return str(dist.get_backend(self.ep_group)).lower().rsplit(".", maxsplit=1)[-1]

    def _planner_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.ep_group is not None and self.ep_size > 1:
            backend = self._planner_collective_backend()
            if backend == "gloo" and tensor.device.type != "cpu":
                reduced = tensor.detach().to(device="cpu")
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.ep_group)
                tensor.copy_(reduced.to(device=tensor.device))
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.ep_group)
        return tensor

    def _planner_gather_fixed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.ep_group is None or self.ep_size <= 1:
            return tensor.unsqueeze(0)
        backend = self._planner_collective_backend()
        local = tensor.contiguous()
        if backend == "gloo" and local.device.type != "cpu":
            local = local.to(device="cpu")
        gathered = torch.empty(
            (self.ep_size * local.numel(),),
            dtype=local.dtype,
            device=local.device,
        )
        dist.all_gather_into_tensor(gathered, local, group=self.ep_group)
        result = gathered.view(self.ep_size, local.numel())
        return result if result.device == tensor.device else result.to(device=tensor.device)

    def _expert_payload_bytes(self, layer: ExpertLayerState) -> tuple[tuple[int, ...], tuple[int, ...]]:
        state_bytes = 0
        gradient_bytes = 0
        for param in (layer.gate_up_proj, layer.down_proj):
            local = _local_tensor_view(param)
            slot_numel = int(local[0].numel())
            state_bytes += slot_numel * int(local.element_size())
            gradient_bytes += slot_numel * self.gradient_bytes_per_element
            state_tensors = _optimizer_state_slot_tensors_from_bindings(
                self._optimizer_param_bindings.get(id(param), ()), param
            )
            if not state_tensors:
                state_tensors = _optimizer_state_slot_tensors(self.optimizer, param)
            for tensor in state_tensors:
                local_state = _local_tensor_view(tensor)
                state_bytes += int(local_state[0].numel()) * int(local_state.element_size())
        return (
            (state_bytes,) * layer.num_experts,
            (gradient_bytes,) * layer.num_experts,
        )

    def _planner_for_layer(
        self,
        layer: ExpertLayerState,
        *,
        communication_scale: float,
        forward_compute_per_assignment: float,
    ) -> CurrentRoutePlanner:
        if self.expert_swap_mode != "layer":
            return CurrentRoutePlanner(
                hierarchy=self.hierarchy,
                perf_model=self.perf_model,
                hidden_size=layer.latest_hidden_size,
                bytes_per_element=layer.latest_bytes_per_element,
                slots_per_rank=layer.num_local_experts,
                communication_scale=communication_scale,
                forward_compute_per_assignment=forward_compute_per_assignment,
                reducer=self._planner_reduce_sum,
                candidate_chunk_size=_SWAP_COST_CHUNK_CANDIDATES,
            )
        expert_state_bytes, expert_gradient_bytes = self._expert_payload_bytes(layer)
        return CoReMoEPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layer.latest_hidden_size,
            bytes_per_element=layer.latest_bytes_per_element,
            slots_per_rank=layer.num_local_experts,
            communication_scale=communication_scale,
            forward_compute_per_assignment=forward_compute_per_assignment,
            reducer=self._planner_reduce_sum,
            gather_fixed=self._planner_gather_fixed,
            collective_backend=self._planner_collective_backend(),
            route_sample_size=self.planner_route_sample_size,
            expert_state_bytes=expert_state_bytes,
            expert_gradient_bytes=expert_gradient_bytes,
        )

    @staticmethod
    def _events_ready(timing: _PendingLayerTiming) -> bool:
        for accelerator_event in (
            timing.dispatch_end,
            timing.compute_end,
            timing.combine_end,
        ):
            event = accelerator_event.event
            query = getattr(event, "query", None)
            if callable(query):
                try:
                    if not query():
                        return False
                except Exception:
                    return False
        return True

    @torch.no_grad()
    def prepare_calibrations(self, step: int) -> None:
        if not self.layer_calibration_enabled():
            return
        started = time.perf_counter()
        updated = 0
        for layer_key in sorted(self.layers):
            layer = self.layers[layer_key]
            timing = layer.pending_timing
            if timing is None or timing.step > int(step) or not self._events_ready(timing):
                continue
            selected = timing.selected_experts
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=1.0,
            )
            copy_slots, _copy_mask = layer.copy_slots_for_device(selected.device)
            reference = planner.score_layout(
                selected,
                timing.slot_to_logical,
                source_ranks=self.ep_rank,
                owner_slots=layer.logical_to_physical,
                step=timing.step,
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=int(copy_slots.shape[1]),
            )
            values = torch.tensor(
                [
                    timing.dispatch_start.elapsed_time(timing.dispatch_end)
                    + timing.combine_start.elapsed_time(timing.combine_end),
                    timing.compute_start.elapsed_time(timing.compute_end),
                ],
                dtype=torch.float32,
                device=selected.device,
            )
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.ep_group)
            communication_units = reference.communication_model_units
            peak_assignments = reference.compute / 3.0
            forward_communication_ms = float(values[0].item())
            forward_compute_ms = float(values[1].item())
            if communication_units <= 0.0 or peak_assignments <= 0.0:
                continue
            communication_scale = (2.0 * forward_communication_ms) / communication_units
            compute_scale = forward_compute_ms / peak_assignments
            if not math.isfinite(communication_scale) or not math.isfinite(compute_scale):
                continue
            layer.planner_calibration = _PlannerCalibration(
                source_step=timing.step,
                communication_scale=communication_scale,
                forward_compute_per_assignment=compute_scale,
            )
            layer.pending_timing = None
            updated += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_calibration_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_calibrated_layers", updated)

    def _record_plan_metrics(self, plan: PlacementPlan) -> None:
        values: dict[str, float | int] = {
            "hiermoe/placement_planning_ms": plan.planning_ms,
            "hiermoe/placement_route_stats_ms": plan.route_stats_ms,
            "hiermoe/placement_swap_ms": plan.swap_ms,
            "hiermoe/placement_replica_ms": plan.replica_ms,
            "hiermoe/placement_swap_score_ms": plan.swap_score_ms,
            "hiermoe/placement_swap_update_ms": plan.swap_update_ms,
            "hiermoe/placement_swap_collective_ms": plan.swap_collective_ms,
            "hiermoe/placement_replica_score_ms": plan.replica_score_ms,
            "hiermoe/placement_replica_update_ms": plan.replica_update_ms,
            "hiermoe/placement_replica_collective_ms": plan.replica_collective_ms,
            "hiermoe/placement_decision_sync_ms": plan.decision_sync_ms,
            "hiermoe/placement_finalization_ms": plan.finalization_ms,
            "hiermoe/placement_swap_count": plan.swap_rounds,
            "hiermoe/placement_replica_count": plan.replica_rounds,
            "hiermoe/placement_predicted_communication_ms": plan.final_cost.communication,
            "hiermoe/placement_predicted_compute_ms": plan.final_cost.compute,
            "hiermoe/placement_predicted_state_move_ms": plan.final_cost.state_move_exposed,
            "hiermoe/placement_predicted_gradient_sync_ms": plan.final_cost.gradient_sync,
            "hiermoe/placement_predicted_total_ms": plan.final_cost.total,
            "hiermoe/placement_baseline_total_ms": plan.baseline_cost.total,
        }
        for key, value in values.items():
            self._accumulate_metric(key, value)

    @torch.no_grad()
    def _execute_placement_plan(
        self,
        layer: ExpertLayerState,
        plan: PlacementPlan,
        *,
        timing_prefix: str | None = None,
    ) -> list[str]:
        quota_policy = tuple(QuotaPolicyEntry.from_tuple(row) for row in plan.quota_policy)
        final_owner_slots = tuple(int(value) for value in plan.final_owner_slots)
        algorithm_version = getattr(plan, "algorithm_version", None)
        if algorithm_version == CORE_MOE_ALGORITHM_VERSION and len(final_owner_slots) != layer.num_experts:
            raise RuntimeError(
                f"CoRe-MoE placement plan for {layer.key} must provide exactly {layer.num_experts} owner slots."
            )
        if final_owner_slots and len(final_owner_slots) != layer.num_experts:
            raise RuntimeError(
                f"HierMoE placement plan for {layer.key} has {len(final_owner_slots)} owners, "
                f"expected {layer.num_experts}."
            )
        current_layout = self._layer_layout(layer)
        if len(plan.final_layout) != int(current_layout.numel()):
            raise RuntimeError(
                f"HierMoE placement plan for {layer.key} has {len(plan.final_layout)} physical slots, "
                f"expected {current_layout.numel()}."
            )
        if not plan.actions and tuple(int(value) for value in current_layout.tolist()) == plan.final_layout:
            with _placement_timing_range(timing_prefix, "apply"):
                if layer.slot_layout_enabled:
                    validated_layout, validated_owners = self._validate_placement_layout(
                        layer,
                        current_layout,
                        final_owner_slots or None,
                    )
                    quota_policy = self._validate_quota_policy(layer, validated_layout, quota_policy)
                    if validated_owners is not None:
                        self._refresh_layer_mapping_from_slots(layer, validated_owners)
                elif quota_policy:
                    raise RuntimeError(f"HierMoE compact placement for {layer.key} cannot install replica quota rows.")
                layer.active_quota_policy = tuple(quota_policy)
            return []
        working = current_layout.clone()
        origin_by_slot = torch.arange(int(working.numel()), dtype=torch.long)
        committed: list[str] = []
        for action in plan.actions:
            src_slot = int(action.src_slot)
            dst_slot = int(action.dst_slot)
            if not 0 <= src_slot < int(working.numel()) or not 0 <= dst_slot < int(working.numel()):
                raise RuntimeError(
                    f"HierMoE placement action contains an out-of-range physical slot: {action.format()}."
                )
            if action.kind == "swap":
                if (
                    int(working[src_slot].item()) != action.src_logical
                    or int(working[dst_slot].item()) != action.dst_logical
                ):
                    raise RuntimeError(
                        f"HierMoE placement swap does not match the ordered working layout: {action.format()}."
                    )
                working[src_slot], working[dst_slot] = (
                    working[dst_slot].clone(),
                    working[src_slot].clone(),
                )
                origin_by_slot[src_slot], origin_by_slot[dst_slot] = (
                    origin_by_slot[dst_slot].clone(),
                    origin_by_slot[src_slot].clone(),
                )
            elif action.kind == "replica":
                if (
                    action.src_logical < 0
                    or int(working[src_slot].item()) != action.src_logical
                    or int(working[dst_slot].item()) != action.dst_logical
                ):
                    raise RuntimeError(
                        f"HierMoE placement cover does not match the ordered working layout: {action.format()}."
                    )
                working[dst_slot] = action.src_logical
                origin_by_slot[dst_slot] = origin_by_slot[src_slot]
            elif action.kind == "empty":
                if (
                    src_slot != dst_slot
                    or int(working[dst_slot].item()) != action.src_logical
                    or action.dst_logical != -1
                ):
                    raise RuntimeError(
                        f"HierMoE placement empty action does not match the ordered working layout: {action.format()}."
                    )
                working[dst_slot] = -1
                origin_by_slot[dst_slot] = -1
            else:
                raise RuntimeError(f"HierMoE placement plan contains an unknown action kind: {action.kind!r}.")
            committed.append(f"{layer.key}:{action.format()}")

        final_layout = torch.tensor(plan.final_layout, dtype=torch.long)
        if tuple(int(value) for value in working.tolist()) != plan.final_layout:
            raise RuntimeError(f"HierMoE executor diverged from the planner for layer {layer.key}.")

        directed_transfers: list[tuple[int, int]] = []
        zero_slots: list[int] = []
        for dst_slot in range(int(final_layout.numel())):
            desired = int(final_layout[dst_slot].item())
            initial = int(current_layout[dst_slot].item())
            if desired == initial:
                continue
            if desired < 0:
                zero_slots.append(dst_slot)
                continue
            source_origin = int(origin_by_slot[dst_slot].item())
            if (
                source_origin < 0
                or source_origin >= int(current_layout.numel())
                or int(current_layout[source_origin].item()) != desired
            ):
                raise RuntimeError(
                    f"HierMoE placement plan has no original state source for logical expert {desired} "
                    f"at physical slot {dst_slot}."
                )
            directed_transfers.append((source_origin, dst_slot))
        if final_owner_slots:
            for logical_expert, physical_slot in enumerate(final_owner_slots):
                if not 0 <= physical_slot < int(working.numel()):
                    raise RuntimeError(
                        f"HierMoE placement owner slot {physical_slot} for logical expert {logical_expert} "
                        "is out of range."
                    )
                if int(working[physical_slot].item()) != logical_expert:
                    raise RuntimeError(
                        f"HierMoE placement owner slot {physical_slot} does not contain logical expert "
                        f"{logical_expert}."
                    )
        if layer.slot_layout_enabled:
            working, _ = self._validate_placement_layout(layer, working, final_owner_slots or None)
        elif quota_policy:
            raise RuntimeError(f"HierMoE compact placement for {layer.key} cannot install replica quota rows.")
        quota_policy = self._validate_quota_policy(layer, working, quota_policy)

        grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
        written_slots: set[int] = set()
        for _src_slot, dst_slot in directed_transfers:
            if dst_slot in written_slots:
                raise RuntimeError(f"HierMoE placement plan writes physical slot {dst_slot} more than once.")
            written_slots.add(dst_slot)
        overlapping_zero_slots = written_slots.intersection(zero_slots)
        if overlapping_zero_slots:
            slot = min(overlapping_zero_slots)
            raise RuntimeError(f"HierMoE placement plan both copies into and clears physical slot {slot}.")

        originally_missing_grads = tuple(
            param for param in (layer.gate_up_proj, layer.down_proj) if getattr(param, "grad", None) is None
        )
        swap_actions = tuple(action for action in plan.actions if action.kind == "swap")
        for action in swap_actions:
            lhs_rank = int(action.src_slot) // layer.num_local_experts
            rhs_rank = int(action.dst_slot) // layer.num_local_experts
            if lhs_rank == rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {layer.key}: "
                    f"rank={lhs_rank}, experts=({action.src_logical}, {action.dst_logical})."
                )
        swap_slots = tuple(slot for action in swap_actions for slot in (int(action.src_slot), int(action.dst_slot)))
        use_pure_swap_transport = (
            bool(swap_actions)
            and len(swap_actions) == len(plan.actions)
            and len(set(swap_slots)) == len(swap_slots)
            and not zero_slots
            and not quota_policy
            and self.ep_group is not None
        )

        if use_pure_swap_transport:
            try:
                state_rows = self._slot_op_state_rows(layer)
                state_tensors = [tensor for _param, items in state_rows for _descriptor, tensor in items]
                if self.debug_validate and self.ep_size > 1:
                    self._validate_optimizer_state_slot_tensors_across_ep(state_rows)
                swap_plans: list[_LayerSwapPlan] = []
                for action in swap_actions:
                    lhs_rank, lhs_slot = divmod(int(action.src_slot), layer.num_local_experts)
                    rhs_rank, rhs_slot = divmod(int(action.dst_slot), layer.num_local_experts)
                    swap_plans.append(
                        _LayerSwapPlan(
                            layer_key=layer.key,
                            logical_lhs=int(action.src_logical),
                            logical_rhs=int(action.dst_logical),
                            lhs_rank=int(lhs_rank),
                            rhs_rank=int(rhs_rank),
                            entries=tuple(
                                _SwapTensorEntry(tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot)
                                for tensor in state_tensors
                            ),
                        )
                    )
                if self.expert_swap_mode == "layer":
                    with _placement_timing_range(timing_prefix, "transfer"):
                        self._execute_sparse_group_swap_plans(swap_plans)
                else:
                    self.launch_pending_layer_swap(layer.key, swap_plans, timing_prefix=timing_prefix)
            except Exception:
                for param in originally_missing_grads:
                    param.grad = None
                raise
        else:
            with _placement_timing_range(timing_prefix, "transfer"):
                try:
                    state_tensors = self._slot_op_state_tensors(layer) if directed_transfers or zero_slots else []
                    for src_slot, dst_slot in directed_transfers:
                        src_rank = src_slot // layer.num_local_experts
                        dst_rank = dst_slot // layer.num_local_experts
                        grouped_entries[(src_rank, dst_rank)].extend(
                            self._slot_op_cover_entries_from_tensors(
                                state_tensors,
                                num_local_experts=layer.num_local_experts,
                                src_slot=src_slot,
                                dst_slot=dst_slot,
                            )
                        )
                    zero_entries = {
                        dst_slot: self._slot_op_cover_entries_from_tensors(
                            state_tensors,
                            num_local_experts=layer.num_local_experts,
                            src_slot=dst_slot,
                            dst_slot=dst_slot,
                        )
                        for dst_slot in zero_slots
                    }
                    _cover_grouped_slot_entries_atomic(
                        grouped_entries,
                        self.ep_rank,
                        self.ep_size,
                        self.ep_group,
                        zero_entry_groups=(
                            (dst_slot // layer.num_local_experts, zero_entries[dst_slot]) for dst_slot in zero_slots
                        ),
                        debug_validate=self.debug_validate,
                    )
                except Exception:
                    for param in originally_missing_grads:
                        param.grad = None
                    raise

        with _placement_timing_range(timing_prefix, "apply"):
            if layer.slot_layout_enabled:
                layer.slot_to_logical = working
                layer.fixed_r2_layout = False
                self._refresh_layer_mapping_from_slots(layer, final_owner_slots or None)
            else:
                mapping = torch.empty((layer.num_experts,), dtype=torch.long)
                for physical_slot, logical in enumerate(working.tolist()):
                    mapping[int(logical)] = int(physical_slot)
                layer.logical_to_physical = mapping
                layer.refresh_identity()
                layer.invalidate_cache()
            layer.active_quota_policy = quota_policy
        return committed

    @staticmethod
    def _layer_layout(layer: ExpertLayerState) -> torch.Tensor:
        if layer.slot_to_logical is not None:
            return layer.slot_to_logical.detach().cpu().clone()
        layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
        logical = torch.arange(layer.num_experts, dtype=torch.long)
        layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        return layout

    @torch.no_grad()
    def _execute_exact_single_swap(
        self,
        layer: ExpertLayerState,
        chosen_pair: tuple[int, int] | None,
        *,
        timing_prefix: str | None,
    ) -> str | None:
        """Execute one exact compact-layout swap without constructing a PlacementPlan."""

        if chosen_pair is None:
            with _placement_timing_range(timing_prefix, "apply"):
                pass
            return None
        if layer.slot_layout_enabled:
            raise RuntimeError(f"hiermoe_exact_p1 requires a compact expert layout for layer {layer.key}.")

        lhs, rhs = (int(value) for value in chosen_pair)
        if not 0 <= lhs < layer.num_experts or not 0 <= rhs < layer.num_experts or lhs == rhs:
            raise RuntimeError(
                f"hiermoe_exact_p1 produced an invalid expert pair for layer {layer.key}: ({lhs}, {rhs})."
            )

        initial_layout: torch.Tensor | None = None
        if self.debug_validate:
            initial_layout = self._layer_layout(layer)
            expected_logical = torch.arange(layer.num_experts, dtype=torch.long)
            if not torch.equal(torch.sort(initial_layout).values, expected_logical):
                raise RuntimeError(f"hiermoe_exact_p1 found an invalid compact layout for layer {layer.key}.")

        plan = self._build_layer_swap_plan(layer.key, (lhs, rhs))
        if plan is None:
            raise RuntimeError(f"hiermoe_exact_p1 mapped distinct experts to one physical slot for layer {layer.key}.")
        physical_lhs = plan.lhs_rank * layer.num_local_experts + plan.entries[0].lhs_slot
        physical_rhs = plan.rhs_rank * layer.num_local_experts + plan.entries[0].rhs_slot
        may_restore_identity = not layer.is_identity and physical_lhs == rhs and physical_rhs == lhs

        if self.debug_validate:
            assert initial_layout is not None
            if int(initial_layout[physical_lhs].item()) != lhs or int(initial_layout[physical_rhs].item()) != rhs:
                raise RuntimeError(f"hiermoe_exact_p1 layout mismatch for layer {layer.key}.")

        if self.expert_swap_mode == "layer":
            with _placement_timing_range(timing_prefix, "transfer"):
                self._execute_sparse_group_swap_plans((plan,))
        else:
            self.launch_pending_layer_swap(layer.key, (plan,), timing_prefix=timing_prefix)

        with _placement_timing_range(timing_prefix, "apply"):
            layer.logical_to_physical[lhs], layer.logical_to_physical[rhs] = (
                layer.logical_to_physical[rhs].clone(),
                layer.logical_to_physical[lhs].clone(),
            )
            layer.is_identity = False
            layer.invalidate_cache()
            if self.debug_validate:
                assert initial_layout is not None
                expected_layout = initial_layout.clone()
                expected_layout[physical_lhs], expected_layout[physical_rhs] = (
                    expected_layout[physical_rhs].clone(),
                    expected_layout[physical_lhs].clone(),
                )
                if not torch.equal(self._layer_layout(layer), expected_layout):
                    raise RuntimeError(f"hiermoe_exact_p1 mapping update failed for layer {layer.key}.")
            if may_restore_identity or self.debug_validate:
                layer.refresh_identity()

        return f"{layer.key}:swap({physical_lhs}<->{physical_rhs})"

    @torch.no_grad()
    def _plan_exact_single_swap_layers(self, layers: list[ExpertLayerState], step: int) -> list[str]:
        """Plan paper-exact P1 swaps and execute them through the direct swap path."""

        if not layers:
            return []

        with _full_timing_range("hiermoe_exact_p1_stats"):
            stats_started = time.perf_counter()
            local_rows: list[torch.Tensor] = []
            row_lengths: list[int] = []
            group_rows: list[list[int]] = []
            group_mappings: list[list[torch.Tensor]] = []
            common_device: torch.device | None = None

            for layer in layers:
                if layer.num_experts > _EXACT_SINGLE_SWAP_MAX_EXPERTS:
                    raise ValueError(
                        "hiermoe_exact_p1 supports at most "
                        f"{_EXACT_SINGLE_SWAP_MAX_EXPERTS} experts per layer, got {layer.num_experts}."
                    )
                selected = layer.latest_selected_experts
                layer_device = (
                    selected.device
                    if selected is not None and selected.numel() > 0
                    else _local_tensor_view(layer.gate_up_proj).device
                )
                if common_device is None:
                    common_device = layer_device
                elif layer_device != common_device:
                    raise ValueError("hiermoe_exact_p1 requires all planned layers to reside on one device.")

                if selected is None or selected.numel() == 0:
                    selected = torch.empty((0, 1), dtype=torch.long, device=layer_device)
                selected = selected.to(device=layer_device, dtype=torch.long, non_blocking=True)
                token_expert_hits = _token_expert_hit_matrix(selected, layer.num_experts)
                logical_to_physical = layer.mapping_for_device(layer_device)

                num_local_experts = max(1, layer.num_experts // self.hierarchy.ep_size)
                mappings = [torch.div(logical_to_physical, num_local_experts, rounding_mode="floor")]
                num_group_rows = [self.hierarchy.ep_size]
                level_shapes = _hierarchy_level_group_shapes(self.hierarchy, layer.num_experts)
                expected_levels = max(0, max(1, int(self.hierarchy.selected_dim)) - 1)
                if len(level_shapes) != expected_levels:
                    raise ValueError("hiermoe_exact_p1 hierarchy is incompatible with the expert count.")
                for u_i, num_groups in level_shapes:
                    expert_group_size = max(
                        1,
                        layer.num_experts // max(1, self.hierarchy.ep_size // u_i),
                    )
                    mappings.append(torch.div(logical_to_physical, expert_group_size, rounding_mode="floor"))
                    num_group_rows.append(num_groups)

                flat_row = torch.cat(
                    [
                        _flatten_exact_single_swap_group_stats(
                            _exact_single_swap_group_stats(token_expert_hits, group_by_logical, num_groups)
                        )
                        for group_by_logical, num_groups in zip(mappings, num_group_rows, strict=True)
                    ],
                    dim=0,
                )
                local_rows.append(flat_row)
                row_lengths.append(int(flat_row.numel()))
                group_rows.append(num_group_rows)
                group_mappings.append(mappings)

            if common_device is None:
                return []
            max_stats = max(row_lengths)
            stats_bytes = len(layers) * max_stats * torch.tensor([], dtype=torch.float32).element_size()
            if stats_bytes > _EXACT_SINGLE_SWAP_MAX_STATS_BYTES:
                raise ValueError(
                    "hiermoe_exact_p1 statistics exceed the fixed payload limit: "
                    f"{stats_bytes} > {_EXACT_SINGLE_SWAP_MAX_STATS_BYTES} bytes."
                )
            global_stats = torch.zeros((len(layers), max_stats), dtype=torch.float32, device=common_device)
            for layer_idx, row in enumerate(local_rows):
                global_stats[layer_idx, : row.numel()] = row
            stats_ms = (time.perf_counter() - stats_started) * 1000.0

        with _full_timing_range("hiermoe_exact_p1_collective"):
            collective_started = time.perf_counter()
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(global_stats, op=dist.ReduceOp.SUM, group=self.ep_group)
            collective_ms = (time.perf_counter() - collective_started) * 1000.0

        with _full_timing_range("hiermoe_exact_p1_score"):
            score_started = time.perf_counter()
            decision_pairs: list[torch.Tensor] = []
            candidate_total = 0
            for layer_idx, layer in enumerate(layers):
                flat_row = global_stats[layer_idx, : row_lengths[layer_idx]]
                offset = 0
                exact_stats: list[_ExactSingleSwapGroupStats] = []
                for num_groups in group_rows[layer_idx]:
                    stats, offset = _unpack_exact_single_swap_group_stats(
                        flat_row,
                        offset=offset,
                        num_experts=layer.num_experts,
                        num_groups=num_groups,
                    )
                    exact_stats.append(stats)
                if offset != row_lengths[layer_idx]:
                    raise RuntimeError("hiermoe_exact_p1 statistics were unpacked incorrectly.")

                pair_cache_key = (layer.num_experts, common_device)
                all_pairs = self._exact_candidate_pair_cache.get(pair_cache_key)
                if all_pairs is None:
                    all_pairs = _all_candidate_pairs(layer.num_experts, common_device)
                    self._exact_candidate_pair_cache[pair_cache_key] = all_pairs
                pairs = _cross_rank_candidate_pairs(
                    all_pairs,
                    layer.mapping_for_device(common_device),
                    layer.num_local_experts,
                )
                current_cost, candidate_costs = _exact_single_swap_costs_from_group_stats(
                    group_stats=exact_stats,
                    group_by_logical=group_mappings[layer_idx],
                    pairs=pairs,
                    num_experts=layer.num_experts,
                    hidden_size=layer.latest_hidden_size,
                    bytes_per_element=layer.latest_bytes_per_element,
                    hierarchy=self.hierarchy,
                    perf_model=self.perf_model,
                    gamma=self.smooth_max_gamma,
                )
                candidate_total += int(pairs.shape[0])
                if candidate_costs.numel() > 0:
                    candidate_value, best_idx = torch.min(candidate_costs, dim=0)
                    improved = candidate_value < current_cost
                    chosen_pair = torch.where(
                        improved,
                        pairs[best_idx],
                        torch.full((2,), -1, dtype=torch.long, device=common_device),
                    )
                else:
                    chosen_pair = torch.full((2,), -1, dtype=torch.long, device=common_device)
                decision_pairs.append(chosen_pair)

            decisions_cpu = torch.stack(decision_pairs, dim=0).detach().to(torch.device("cpu"))
            selections: list[tuple[ExpertLayerState, tuple[int, int] | None]] = []
            accepted = 0
            for layer, row in zip(layers, decisions_cpu.tolist(), strict=True):
                lhs, rhs = (int(value) for value in row)
                chosen_pair = None if lhs < 0 or rhs < 0 else (lhs, rhs)
                accepted += int(chosen_pair is not None)
                selections.append((layer, chosen_pair))
            score_ms = (time.perf_counter() - score_started) * 1000.0

        self._accumulate_metric("hiermoe/exact_p1_stats_ms", stats_ms)
        self._accumulate_metric("hiermoe/exact_p1_collective_ms", collective_ms)
        self._accumulate_metric("hiermoe/exact_p1_score_ms", score_ms)
        self._accumulate_metric("hiermoe/exact_p1_candidate_count", candidate_total)
        self._accumulate_metric("hiermoe/exact_p1_accepted_count", accepted)

        committed: list[str] = []
        for layer, chosen_pair in selections:
            result = self._execute_exact_single_swap(layer, chosen_pair, timing_prefix="hiermoe_exact_p1")
            if result is not None:
                committed.append(result)
        return committed

    @torch.no_grad()
    def _plan_current_layer(self, layer: ExpertLayerState, step: int) -> list[str]:
        calibration = layer.planner_calibration
        selected = layer.latest_selected_experts
        if calibration is None or selected is None or selected.numel() == 0:
            return []
        planner = self._planner_for_layer(
            layer,
            communication_scale=calibration.communication_scale,
            forward_compute_per_assignment=calibration.forward_compute_per_assignment,
        )
        with _full_timing_range("hiermoe_placement_planning"):
            plan = planner.plan(
                selected,
                self._layer_layout(layer),
                layer.logical_to_physical,
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds if layer.slot_layout_enabled else 0,
                step=step,
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            )
            layer.last_plan = plan
            self._record_plan_metrics(plan)
        committed = self._execute_placement_plan(layer, plan, timing_prefix="hiermoe_placement")
        if self.expert_swap_mode == "layer" and plan.local_physical_routes is not None:
            layer.pending_physical_routes = plan.local_physical_routes
            layer.pending_route_data_ptr = selected.data_ptr()
        return committed

    @torch.no_grad()
    def _plan_legacy_batched_layers(self, layers: list[ExpertLayerState]) -> list[str]:
        from .legacy_batched_selector import LegacyBatchedSelector

        with _full_timing_range("hiermoe_expert_swap_select"):
            pair_lists = LegacyBatchedSelector(self).select(layers)

        plans: list[_LayerSwapPlan] = []
        with _full_timing_range("hiermoe_expert_swap_plan"):
            state_rows = []
            if self.debug_validate and self.ep_group is not None and self.ep_size > 1:
                for layer in layers:
                    if not pair_lists.get(layer.key):
                        continue
                    state_rows.extend(self._slot_op_state_rows(layer))
                self._validate_optimizer_state_slot_tensors_across_ep(state_rows)

            for layer in layers:
                for pair in pair_lists.get(layer.key, ()):
                    plan = self._build_layer_swap_plan(
                        layer.key,
                        pair,
                    )
                    if plan is not None:
                        plans.append(plan)

        with _full_timing_range("hiermoe_expert_swap_exchange"):
            self._execute_swap_plans(plans)

        committed: list[str] = []
        with _full_timing_range("hiermoe_expert_swap_apply"):
            for plan in plans:
                layer = self.layers[plan.layer_key]
                lhs, rhs = plan.logical_lhs, plan.logical_rhs
                layer.logical_to_physical[lhs], layer.logical_to_physical[rhs] = (
                    layer.logical_to_physical[rhs].clone(),
                    layer.logical_to_physical[lhs].clone(),
                )
                layer.refresh_identity()
                layer.invalidate_cache()
                committed.append(f"{plan.layer_key}:{lhs}<->{rhs}")
        return committed

    @torch.no_grad()
    def maybe_swap(self, step: int) -> str:
        self._begin_metrics_step(step)
        if (
            self.layers
            and self.expert_swap_max_pairs_per_layer == 0
            and self.max_replica_rounds == 0
            and all(layer.fixed_r2_layout for layer in self.layers.values())
        ):
            self.latest_pair = "none"
            return self.latest_pair

        if self.expert_swap_selector == "legacy_batched" and self.expert_swap_max_pairs_per_layer <= 0:
            self.latest_pair = "none"
            return self.latest_pair

        if self.expert_swap_selector == "hiermoe_exact_p1":
            if int(step) <= 0 or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
                self.latest_pair = "none"
                return self.latest_pair
            with _full_timing_range("hiermoe_exact_p1_plan"):
                layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
                committed = self._plan_exact_single_swap_layers(layers, int(step))
        elif self.expert_swap_selector == "legacy_batched":
            if int(step) <= 0 or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
                self.latest_pair = "none"
                return self.latest_pair
            with _full_timing_range("hiermoe_legacy_batched_plan"):
                layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
                committed = self._plan_legacy_batched_layers(layers)
        else:
            self.prepare_calibrations(step)
            if int(step) <= 0 or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
                self.latest_pair = "none"
                return self.latest_pair
            committed = []
            with _full_timing_range("hiermoe_current_route_plan"):
                for layer_key in sorted(self.layers):
                    committed.extend(self._plan_current_layer(self.layers[layer_key], int(step)))
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    @torch.no_grad()
    def maybe_swap_layer_on_routing(
        self,
        *,
        layer_key: str,
        selected_experts: torch.Tensor,
        hidden_size: int,
        bytes_per_element: int,
        step: int,
    ) -> str:
        self.record_routing(
            layer_key=layer_key,
            selected_experts=selected_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            step=step,
        )
        self._begin_metrics_step(step)
        if int(step) <= 0 or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            self.latest_pair = "none"
            return self.latest_pair

        layer = self.layers.get(layer_key)
        if layer is None:
            return self.latest_pair
        if layer.last_planned_step == int(step):
            return self.latest_pair
        layer.last_planned_step = int(step)
        if self.expert_swap_selector == "hiermoe_exact_p1":
            with _full_timing_range("hiermoe_exact_p1_layer_plan"):
                committed = self._plan_exact_single_swap_layers([layer], int(step))
        else:
            with _full_timing_range("hiermoe_current_route_layer_plan"):
                committed = self._plan_current_layer(layer, int(step))
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    def _ensure_swap_staging_buffer(
        self,
        device: torch.device,
        dtype: torch.dtype,
        required_numel: int,
    ) -> _SwapStagingBuffer:
        key = (device, dtype)
        cached = self._swap_staging_buffers.get(key)
        if cached is None or int(cached.send.numel()) < int(required_numel):
            cached = _SwapStagingBuffer(
                send=torch.empty((required_numel,), dtype=dtype, device=device),
                recv=torch.empty((required_numel,), dtype=dtype, device=device),
            )
            self._swap_staging_buffers[key] = cached
        return cached

    @torch.no_grad()
    def _execute_sparse_group_swap_plans(self, plans: Iterable[_LayerSwapPlan]) -> None:
        """Synchronously exchange pure swaps with one full-group All-to-All per dtype.

        The training lifecycle creates gradients and optimizer state consistently on every
        EP rank. Debug validation checks that descriptor invariant without adding a production
        split-size collective to this path.
        """

        plan_list = tuple(plans)
        if not plan_list:
            return
        if self._pending_layer_swaps:
            pending = next(iter(self._pending_layer_swaps))
            raise RuntimeError(f"HierMoE tried to execute a collective swap while layer {pending} is still pending.")
        if self.ep_group is None or self.ep_size <= 1:
            raise RuntimeError("HierMoE cross-rank expert swap requires an EP process group.")

        bucket_keys: set[tuple[torch.device, torch.dtype]] = set()
        peer_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        occupied_slots: set[tuple[str, int, int]] = set()
        for plan in plan_list:
            if not 0 <= int(plan.lhs_rank) < self.ep_size or not 0 <= int(plan.rhs_rank) < self.ep_size:
                raise RuntimeError(
                    f"HierMoE planner produced an out-of-range swap rank for layer {plan.layer_key}: "
                    f"({plan.lhs_rank}, {plan.rhs_rank})."
                )
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            if not plan.entries:
                raise RuntimeError(f"HierMoE pure swap plan for layer {plan.layer_key} has no tensor entries.")
            for rank, slot in ((plan.lhs_rank, plan.entries[0].lhs_slot), (plan.rhs_rank, plan.entries[0].rhs_slot)):
                occupied = (plan.layer_key, int(rank), int(slot))
                if occupied in occupied_slots:
                    raise RuntimeError(
                        f"HierMoE pure swap plan writes layer {plan.layer_key} rank {rank} slot {slot} more than once."
                    )
                occupied_slots.add(occupied)

            for entry in plan.entries:
                local_tensor = _local_tensor_view(entry.tensor)
                lhs_view = local_tensor.detach()[entry.lhs_slot]
                rhs_view = local_tensor.detach()[entry.rhs_slot]
                if tuple(lhs_view.shape) != tuple(rhs_view.shape):
                    raise RuntimeError("HierMoE pure swap tried to exchange incompatible expert slot shapes.")
                key = (lhs_view.device, lhs_view.dtype)
                bucket_keys.add(key)
                if self.ep_rank == plan.lhs_rank:
                    local_slot = int(entry.lhs_slot)
                    peer_rank = int(plan.rhs_rank)
                elif self.ep_rank == plan.rhs_rank:
                    local_slot = int(entry.rhs_slot)
                    peer_rank = int(plan.lhs_rank)
                else:
                    continue
                send_view = local_tensor.detach()[local_slot]
                numel = int(send_view.numel())
                peer_buckets[key][peer_rank].append(
                    (local_tensor, local_slot, send_view, numel, numel * int(send_view.element_size()))
                )

        def bucket_sort_key(key: tuple[torch.device, torch.dtype]) -> tuple[str, int, str]:
            device, dtype = key
            return (device.type, -1 if device.index is None else int(device.index), str(dtype))

        pending_publish: list[tuple[torch.Tensor, dict[int, list[_SwapBucketItem]]]] = []
        for device, dtype in sorted(bucket_keys, key=bucket_sort_key):
            buckets = peer_buckets.get((device, dtype), {})
            input_splits = [0] * self.ep_size
            output_splits = [0] * self.ep_size
            for peer_rank, items in buckets.items():
                payload_numel = sum(item[3] for item in items)
                input_splits[int(peer_rank)] = payload_numel
                output_splits[int(peer_rank)] = payload_numel

            send_numel = sum(input_splits)
            recv_numel = sum(output_splits)
            staging = self._ensure_swap_staging_buffer(device, dtype, max(send_numel, recv_numel))
            send_buffer = staging.send[:send_numel]
            recv_buffer = staging.recv[:recv_numel]
            offset = 0
            for peer_rank in range(self.ep_size):
                for _local_tensor, _local_slot, send_view, numel, _nbytes in buckets.get(peer_rank, ()):
                    send_buffer[offset : offset + numel].view_as(send_view).copy_(send_view)
                    offset += numel

            dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=self.ep_group,
            )

            pending_publish.append((recv_buffer, buckets))

        for recv_buffer, buckets in pending_publish:
            offset = 0
            for peer_rank in range(self.ep_size):
                for local_tensor, local_slot, _send_view, numel, _nbytes in buckets.get(peer_rank, ()):
                    staged = recv_buffer[offset : offset + numel].view_as(local_tensor.detach()[local_slot])
                    local_tensor.detach()[local_slot].copy_(staged)
                    offset += numel

    def _compile_swap_waves(
        self,
        plans: Iterable[_LayerSwapPlan],
    ) -> list[list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]]]:
        remote_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for plan in plans:
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            if self.ep_group is None or self.ep_size <= 1:
                raise RuntimeError("HierMoE cross-rank expert swap requires an EP process group.")
            if self.ep_rank not in (plan.lhs_rank, plan.rhs_rank):
                continue

            peer_rank = plan.rhs_rank if self.ep_rank == plan.lhs_rank else plan.lhs_rank
            peer_global_rank = _ep_global_rank(self.ep_group, peer_rank)
            for entry in plan.entries:
                local_slot = entry.lhs_slot if self.ep_rank == plan.lhs_rank else entry.rhs_slot
                local_tensor = _local_tensor_view(entry.tensor)
                slot_view = local_tensor.detach()[local_slot]
                send_view = slot_view.contiguous().view(-1)
                numel = int(send_view.numel())
                nbytes = numel * int(send_view.element_size())
                remote_buckets[(send_view.device, send_view.dtype)][peer_global_rank].append(
                    (local_tensor, int(local_slot), send_view, numel, nbytes)
                )

        items: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]] = []
        for key in sorted(
            remote_buckets,
            key=lambda item: (
                item[0].type,
                -1 if item[0].index is None else int(item[0].index),
                str(item[1]),
            ),
        ):
            for peer_global_rank, bucket in sorted(remote_buckets[key].items()):
                items.extend((key, peer_global_rank, chunk) for chunk in _chunk_swap_bucket(bucket))

        waves: list[list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]]] = []
        current: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]] = []
        current_nbytes = 0
        for item in items:
            item_nbytes = 2 * _swap_chunk_nbytes(item[2])
            if current and current_nbytes + item_nbytes > _MAX_SWAP_WAVE_BYTES:
                waves.append(current)
                current = []
                current_nbytes = 0
            current.append(item)
            current_nbytes += item_nbytes
        if current:
            waves.append(current)
        return waves

    def _stage_swap_wave(
        self,
        wave: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]],
    ) -> tuple[tuple[Any, ...], list[tuple[torch.Tensor, list[_SwapBucketItem]]]]:
        required: dict[tuple[torch.device, torch.dtype], int] = defaultdict(int)
        for key, _peer_global_rank, chunk in wave:
            required[key] += sum(item[3] for item in chunk)
        buffers = {key: self._ensure_swap_staging_buffer(key[0], key[1], numel) for key, numel in required.items()}
        offsets: dict[tuple[torch.device, torch.dtype], int] = defaultdict(int)
        ops: list[dist.P2POp] = []
        unpack: list[tuple[torch.Tensor, list[_SwapBucketItem]]] = []
        for key, peer_global_rank, chunk in wave:
            numel = sum(item[3] for item in chunk)
            start = offsets[key]
            offsets[key] += numel
            send_segment = buffers[key].send[start : start + numel]
            recv_segment = buffers[key].recv[start : start + numel]
            inner_offset = 0
            for _local_tensor, _local_slot, send_view, item_numel, _nbytes in chunk:
                send_segment[inner_offset : inner_offset + item_numel].copy_(send_view)
                inner_offset += item_numel
            ops.extend(
                (
                    dist.P2POp(dist.isend, send_segment, peer_global_rank, self._swap_group),
                    dist.P2POp(dist.irecv, recv_segment, peer_global_rank, self._swap_group),
                )
            )
            unpack.append((recv_segment, chunk))
        works = tuple(dist.batch_isend_irecv(ops)) if ops else ()
        return works, unpack

    @staticmethod
    def _publish_swap_wave(unpack: Iterable[tuple[torch.Tensor, list[_SwapBucketItem]]]) -> None:
        for recv_segment, chunk in unpack:
            _unpack_swap_chunk(recv_segment, chunk)

    def _swap_comm_stream(self, device: torch.device) -> Any:
        cached = self._swap_comm_streams.get(device)
        if cached is not None:
            return cached
        device_api = get_torch_device()
        try:
            cached = device_api.Stream(device=device)
        except TypeError:
            cached = device_api.Stream()
        self._swap_comm_streams[device] = cached
        return cached

    def _execute_swap_plan_batch(
        self,
        plans: Iterable[_LayerSwapPlan],
        *,
        pending_layer_key: str | None = None,
        timing_prefix: str | None = None,
    ) -> None:
        plan_list = tuple(plans)
        if not plan_list:
            return
        if self._pending_layer_swaps:
            pending = next(iter(self._pending_layer_swaps))
            raise RuntimeError(f"HierMoE tried to launch a swap while layer {pending} is still pending.")

        timing_context = _placement_timing_range(timing_prefix, "transfer")
        timing_context.__enter__()
        try:
            waves = self._compile_swap_waves(plan_list)
            if not waves:
                timing_context.__exit__(None, None, None)
                return

            devices = {key[0] for wave in waves for key, _peer, _chunk in wave}
            asynchronous = pending_layer_key is not None and len(waves) == 1 and len(devices) == 1
            device = next(iter(devices))
            if self._swap_group is None and self.ep_group is dist.group.WORLD:
                asynchronous = False
            asynchronous = asynchronous and device.type != "cpu"
            if not asynchronous:
                for wave in waves:
                    works, unpack = self._stage_swap_wave(wave)
                    for work in works:
                        work.wait()
                    self._publish_swap_wave(unpack)
                timing_context.__exit__(None, None, None)
                return

            device_api = get_torch_device()
            comm_stream = self._swap_comm_stream(device)
            try:
                current_stream = device_api.current_stream(device)
            except TypeError:
                current_stream = device_api.current_stream()
            comm_stream.wait_stream(current_stream)
            with device_api.stream(comm_stream):
                works, unpack = self._stage_swap_wave(waves[0])
            self._pending_layer_swaps[pending_layer_key] = _PendingLayerSwap(
                layer_key=pending_layer_key,
                works=works,
                unpack=tuple(unpack),
                device=device,
                timing_context=timing_context,
            )
        except Exception as error:
            timing_context.__exit__(type(error), error, error.__traceback__)
            raise

    def launch_pending_layer_swap(
        self,
        layer_key: str,
        plans: Iterable[_LayerSwapPlan],
        *,
        timing_prefix: str | None = None,
    ) -> None:
        self._execute_swap_plan_batch(
            plans,
            pending_layer_key=layer_key if self.expert_swap_mode == "layer" else None,
            timing_prefix=timing_prefix,
        )

    def wait_pending_layer_swap(self, layer_key: str) -> None:
        pending = self._pending_layer_swaps.pop(layer_key, None)
        if pending is None:
            return
        try:
            device_api = get_torch_device()
            comm_stream = self._swap_comm_stream(pending.device)
            with device_api.stream(comm_stream):
                for work in pending.works:
                    work.wait()
                self._publish_swap_wave(pending.unpack)
                done_event = device_api.Event()
                done_event.record(comm_stream)
            try:
                current_stream = device_api.current_stream(pending.device)
            except TypeError:
                current_stream = device_api.current_stream()
            current_stream.wait_event(done_event)
        except Exception as error:
            pending.timing_context.__exit__(type(error), error, error.__traceback__)
            raise
        pending.timing_context.__exit__(None, None, None)

    def _build_layer_swap_plan(
        self,
        layer_key: str,
        pair: tuple[int, int],
    ) -> _LayerSwapPlan | None:
        layer = self.layers[layer_key]
        lhs, rhs = pair
        physical_lhs = int(layer.logical_to_physical[lhs].item())
        physical_rhs = int(layer.logical_to_physical[rhs].item())
        if physical_lhs == physical_rhs:
            return None

        lhs_rank, lhs_slot = divmod(physical_lhs, layer.num_local_experts)
        rhs_rank, rhs_slot = divmod(physical_rhs, layer.num_local_experts)
        if lhs_rank == rhs_rank:
            raise RuntimeError(
                f"HierMoE planner produced a same-rank swap for layer {layer_key}: "
                f"rank={lhs_rank}, experts=({lhs}, {rhs})."
            )
        state_rows = self._slot_op_state_rows(layer)
        if self.debug_validate and self.ep_size > 1:
            self._validate_optimizer_state_slot_tensors_across_ep(state_rows)
        entries = [
            _SwapTensorEntry(tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot)
            for _param, items in state_rows
            for _descriptor, tensor in items
        ]
        return _LayerSwapPlan(
            layer_key=layer_key,
            logical_lhs=int(lhs),
            logical_rhs=int(rhs),
            lhs_rank=int(lhs_rank),
            rhs_rank=int(rhs_rank),
            entries=tuple(entries),
        )

    @torch.no_grad()
    def _execute_swap_plans(self, plans: Iterable[_LayerSwapPlan], *, force_collective: bool = False) -> None:
        plan_list = tuple(plans)
        grouped: dict[tuple[int, int], list[_SwapTensorEntry]] = defaultdict(list)
        for plan in plan_list:
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            lhs_rank = min(plan.lhs_rank, plan.rhs_rank)
            rhs_rank = max(plan.lhs_rank, plan.rhs_rank)
            if plan.lhs_rank == lhs_rank:
                grouped[(lhs_rank, rhs_rank)].extend(plan.entries)
            else:
                grouped[(lhs_rank, rhs_rank)].extend(
                    _SwapTensorEntry(entry.tensor, lhs_slot=entry.rhs_slot, rhs_slot=entry.lhs_slot)
                    for entry in plan.entries
                )

        if force_collective:
            _exchange_or_swap_grouped_slot_entries_collective(grouped, self.ep_rank, self.ep_size, self.ep_group)
        else:
            self._execute_swap_plan_batch(plan_list)

    @torch.no_grad()
    def swap_layer_pair(self, layer_key: str, pair: tuple[int, int]) -> None:
        plan = self._build_layer_swap_plan(layer_key, pair)
        if plan is None:
            return
        self._execute_swap_plans((plan,))
        layer = self.layers[layer_key]
        lhs, rhs = plan.logical_lhs, plan.logical_rhs
        layer.logical_to_physical[lhs], layer.logical_to_physical[rhs] = (
            layer.logical_to_physical[rhs].clone(),
            layer.logical_to_physical[lhs].clone(),
        )
        layer.refresh_identity()
        layer.invalidate_cache()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "ep_size": self.ep_size,
            "layers": {
                key: (
                    {
                        "num_experts": layer.num_experts,
                        "base_num_local_experts": layer.base_num_local_experts,
                        "num_local_experts": layer.num_local_experts,
                        "logical_to_physical": layer.logical_to_physical.tolist(),
                        "slot_to_logical": layer.slot_to_logical.tolist(),
                        "quota_algorithm_version": CORE_MOE_ALGORITHM_VERSION,
                        "quota_layout_crc32": zlib.crc32(repr(tuple(layer.slot_to_logical.tolist())).encode()),
                        "quota_policy": [entry.as_tuple() for entry in layer.active_quota_policy],
                    }
                    if layer.slot_to_logical is not None
                    else {
                        "num_experts": layer.num_experts,
                        "logical_to_physical": layer.logical_to_physical.tolist(),
                    }
                )
                for key, layer in sorted(self.layers.items())
            },
        }

    def has_non_identity_placement(self) -> bool:
        if self._pending_state:
            for payload in self._pending_state.get("layers", {}).values():
                mapping = payload.get("logical_to_physical", [])
                if list(mapping) != list(range(len(mapping))):
                    return True
        for layer in self.layers.values():
            if layer.slot_to_logical is not None:
                expected = _initial_slot_to_logical(
                    layer.num_experts,
                    layer.base_num_local_experts,
                    layer.num_local_experts,
                    self.ep_size,
                )
                if not torch.equal(layer.slot_to_logical.cpu(), expected):
                    return True
                continue
            identity = torch.arange(layer.num_experts, dtype=torch.long)
            if not torch.equal(layer.logical_to_physical.cpu(), identity):
                return True
        return False

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        if int(state_dict.get("ep_size", self.ep_size)) != self.ep_size:
            raise ValueError(
                f"HierMoE checkpoint placement was saved with ep_size={state_dict.get('ep_size')}, "
                f"but the current run uses ep_size={self.ep_size}."
            )
        if not self.layers:
            self._pending_state = state_dict
            return
        staged: dict[str, tuple[torch.Tensor | None, torch.Tensor, tuple[QuotaPolicyEntry, ...]]] = {}
        incompatible_quota_key: str | None = None
        for key, payload in state_dict.get("layers", {}).items():
            layer = self.layers.get(key)
            if layer is None:
                raise RuntimeError(
                    f"Checkpoint contains HierMoE placement for unknown layer {key!r}. "
                    "Loading without that placement would change logical expert semantics."
                )
            if payload.get("slot_to_logical") is not None:
                slot_to_logical = torch.tensor(payload["slot_to_logical"], dtype=torch.long)
                if layer.slot_to_logical is None:
                    raise RuntimeError(
                        f"Checkpoint contains HierMoE slot layout for {key}, but current layer has no redundant slots."
                    )
                slot_to_logical, _ = self._validate_placement_layout(layer, slot_to_logical)
                checkpoint_owners = payload.get("logical_to_physical")
                if checkpoint_owners is None:
                    derived_owners: list[int] = []
                    for logical_expert in range(layer.num_experts):
                        slots = torch.nonzero(slot_to_logical == logical_expert, as_tuple=False).flatten().tolist()
                        canonical = (
                            -1
                            if layer.canonical_physical_slots is None
                            else int(layer.canonical_physical_slots[logical_expert].item())
                        )
                        derived_owners.append(canonical if canonical in slots else int(slots[0]))
                    checkpoint_owners = derived_owners
                slot_to_logical, owner_mapping = self._validate_placement_layout(
                    layer,
                    slot_to_logical,
                    checkpoint_owners,
                )
                assert owner_mapping is not None
                policy_version = payload.get("quota_algorithm_version")
                raw_policy = payload.get("quota_policy", ())
                if raw_policy and policy_version != CORE_MOE_ALGORITHM_VERSION:
                    incompatible_quota_key = incompatible_quota_key or key
                    quota_policy = ()
                else:
                    quota_policy = tuple(QuotaPolicyEntry.from_tuple(row) for row in raw_policy)
                expected_crc = payload.get("quota_layout_crc32")
                actual_crc = zlib.crc32(repr(tuple(slot_to_logical.tolist())).encode())
                if expected_crc is not None and int(expected_crc) != actual_crc:
                    raise ValueError(f"HierMoE checkpoint quota policy for {key} does not match its slot layout.")
                quota_policy = self._validate_quota_policy(layer, slot_to_logical, quota_policy)
                staged[key] = (slot_to_logical, owner_mapping, quota_policy)
                continue

            if layer.slot_to_logical is not None:
                raise RuntimeError(
                    f"Checkpoint contains a compact HierMoE layout for {key}, "
                    "but the current layer reserves redundant slots."
                )
            mapping = torch.tensor(payload["logical_to_physical"], dtype=torch.long)
            if mapping.numel() != layer.num_experts:
                raise ValueError(
                    f"HierMoE checkpoint placement for {key} has {mapping.numel()} entries, "
                    f"expected {layer.num_experts}."
                )
            if sorted(mapping.tolist()) != list(range(layer.num_experts)):
                raise ValueError(f"HierMoE checkpoint placement for {key} is not a valid permutation.")
            staged[key] = (None, mapping, ())

        snapshots = {
            key: (
                None if self.layers[key].slot_to_logical is None else self.layers[key].slot_to_logical.clone(),
                self.layers[key].logical_to_physical.clone(),
                self.layers[key].active_quota_policy,
                self.layers[key].pending_physical_routes,
                self.layers[key].pending_route_data_ptr,
                self.layers[key].fixed_r2_layout,
            )
            for key in staged
        }
        try:
            for key, (slot_to_logical, owner_mapping, quota_policy) in staged.items():
                layer = self.layers[key]
                layer.slot_to_logical = None if slot_to_logical is None else slot_to_logical.clone()
                layer.fixed_r2_layout = False
                layer.logical_to_physical = owner_mapping.clone()
                layer.active_quota_policy = quota_policy
                layer.pending_physical_routes = None
                layer.pending_route_data_ptr = 0
                layer.refresh_identity()
                layer.invalidate_cache()
        except Exception:
            for key, (
                slot_to_logical,
                owner_mapping,
                quota_policy,
                pending_routes,
                pending_data_ptr,
                fixed_r2_layout,
            ) in snapshots.items():
                layer = self.layers[key]
                layer.slot_to_logical = slot_to_logical
                layer.logical_to_physical = owner_mapping
                layer.active_quota_policy = quota_policy
                layer.pending_physical_routes = pending_routes
                layer.pending_route_data_ptr = pending_data_ptr
                layer.fixed_r2_layout = fixed_r2_layout
                layer.refresh_identity()
                layer.invalidate_cache()
            raise
        if incompatible_quota_key is not None:
            logger.warning(
                "HierMoE checkpoint quota policy for %s uses a version other than %r; "
                "restoring expert layouts and clearing all incompatible quota policies.",
                incompatible_quota_key,
                CORE_MOE_ALGORITHM_VERSION,
            )
