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

"""CPU implementations of the exact redundant-expert placement planner.

The module deliberately reuses the exact sufficient-statistic implementation
from :mod:`greedy_planner`.  It changes only where the work is executed:

* local route statistics are built by a bounded CPU thread pool;
* every layer is assigned to exactly one EP rank;
* one packed CPU all-to-all sends local statistics to the layer owners;
* owners score their layers and one small all-reduce publishes the decisions.

All candidate rows and cost-model arithmetic remain identical to the full
exact planner.  Empty-slot initialization is intentionally not handled here:
it must use the current route and complete before this steady-state backend is
started.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .greedy_planner import (
    GREEDY_COVER_ALGORITHM_VERSION,
    GreedyCommunicationPlanner,
    _PreparedActionCounts,
)
from .planner import PlacementAction, PlacementCost, PlacementPlan


CPU_LAYER_OWNER_ALGORITHM_VERSION = f"{GREEDY_COVER_ALGORITHM_VERSION}-cpu-layer-owner-v1"
CPU_HCCL_BATCHED_ALGORITHM_VERSION = f"{GREEDY_COVER_ALGORITHM_VERSION}-cpu-hccl-batched-v1"
_DECISION_WIDTH = 13


@dataclass(frozen=True)
class CPUPlannerResources:
    """Resolved per-process CPU resources for exact planning."""

    visible_cpu_cores: int
    cpu_cores_per_rank: int
    reserved_cpu_cores: int
    usable_cpu_cores: int
    layer_workers: int
    intraop_threads: int
    local_process_count: int


@dataclass(frozen=True)
class CPULayerOwnerTiming:
    """Host wall-clock breakdown for one layer-owner planning pass."""

    context_ms: float
    local_prepare_ms: float
    statistic_pack_ms: float
    statistic_collective_ms: float
    owner_score_ms: float
    decision_collective_ms: float
    finalization_ms: float
    total_ms: float
    local_payload_bytes: int
    received_payload_bytes: int
    owned_layer_count: int
    layer_workers: int
    intraop_threads: int


@dataclass(frozen=True)
class CPULayerOwnerResult:
    """Plans and timing returned by :class:`CPULayerOwnerPlanner`."""

    plans: tuple[PlacementPlan, ...]
    timing: CPULayerOwnerTiming
    owner_ranks: tuple[int, ...]


@dataclass(frozen=True)
class AsyncCPUPlanCompletion:
    """One completed asynchronous plan and its placement-version validity."""

    source_step: int
    result: CPULayerOwnerResult
    valid: bool
    stale_reason: str
    placement_versions: tuple[int, ...]


@dataclass(frozen=True)
class SharedMemoryCPUPlanResult:
    """One result returned by the isolated CPU planner process."""

    source_step: int
    result: CPULayerOwnerResult | None
    error: str


@dataclass(frozen=True)
class _LayerContext:
    selected: torch.Tensor
    layout: torch.Tensor
    owners: torch.Tensor
    rows: torch.Tensor
    copy_slots: torch.Tensor
    sources: torch.Tensor
    uniform_source_rank: int | None
    ordinals: torch.Tensor
    layer_seed: int
    original_selected_ndim: int


@dataclass(frozen=True)
class _PreparedLayer:
    context: _LayerContext
    counts: _PreparedActionCounts
    packed_local: torch.Tensor
    communication_width: int
    has_assignments: bool

    @property
    def row_count(self) -> int:
        return int(self.packed_local.shape[0])

    @property
    def flat_size(self) -> int:
        return int(self.packed_local.numel())


@dataclass(frozen=True)
class _PackedStatistics:
    send_buffer: torch.Tensor
    send_splits: tuple[int, ...]
    layers_by_owner: tuple[tuple[int, ...], ...]
    metadata_signature: tuple[int, ...]


@dataclass(frozen=True)
class _AsyncSubmission:
    source_step: int
    placement_versions: tuple[int, ...]
    layout_digest: str
    future: Future[CPULayerOwnerResult]


def _visible_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _runtime_local_process_count() -> int:
    for name in ("LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"):
        raw_value = os.environ.get(name)
        if raw_value is not None and int(raw_value) > 0:
            return int(raw_value)
    return 1


def resolve_cpu_planner_resources(
    *,
    layer_count: int,
    local_process_count: int,
    visible_cpu_cores: int | None = None,
    cpu_cores_per_rank: int | None = None,
    reserve_cpu_cores: int = 2,
    layer_workers: int | None = None,
    intraop_threads: int | None = None,
) -> CPUPlannerResources:
    """Resolve a non-oversubscribed CPU configuration.

    ``visible_cpu_cores`` means the cores in the current process affinity mask.
    If the process is already pinned to roughly one local-rank share, the full
    mask is used.  Otherwise the mask is divided by ``local_process_count``.
    Every value can be overridden for launchers with explicit CPU binding.
    """

    layers = max(1, int(layer_count))
    local_processes = max(1, int(local_process_count))
    visible = max(1, int(_visible_cpu_count() if visible_cpu_cores is None else visible_cpu_cores))
    system_cores = max(1, int(os.cpu_count() or visible))
    expected_affinity_share = max(1, math.ceil(system_cores / local_processes))
    if cpu_cores_per_rank is None or int(cpu_cores_per_rank) <= 0:
        per_rank = visible if visible <= expected_affinity_share else max(1, visible // local_processes)
    else:
        per_rank = max(1, int(cpu_cores_per_rank))
    reserved = min(max(0, int(reserve_cpu_cores)), max(0, per_rank - 1))
    usable = max(1, per_rank - reserved)

    requested_workers = 0 if layer_workers is None else int(layer_workers)
    workers = min(layers, usable) if requested_workers <= 0 else min(layers, usable, requested_workers)
    requested_intraop = 0 if intraop_threads is None else int(intraop_threads)
    native_threads = max(1, usable // workers) if requested_intraop <= 0 else max(1, requested_intraop)
    if workers * native_threads > usable:
        native_threads = max(1, usable // workers)
    return CPUPlannerResources(
        visible_cpu_cores=visible,
        cpu_cores_per_rank=per_rank,
        reserved_cpu_cores=reserved,
        usable_cpu_cores=usable,
        layer_workers=workers,
        intraop_threads=native_threads,
        local_process_count=local_processes,
    )


def balanced_layer_owner_ranks(
    layer_count: int,
    ep_size: int,
    *,
    owner_offset: int = 0,
) -> tuple[int, ...]:
    """Assign layers evenly across arbitrary EP sizes.

    Multiplicative partitioning also spreads owners across the EP group when
    ``layer_count < ep_size`` instead of concentrating all work on low ranks.
    """

    layers = max(0, int(layer_count))
    ranks = int(ep_size)
    if ranks <= 0:
        raise ValueError("ep_size must be positive.")
    offset = int(owner_offset) % ranks
    if layers == 0:
        return ()
    return tuple(((layer * ranks) // layers + offset) % ranks for layer in range(layers))


def _layout_digest(layouts: Sequence[torch.Tensor], owners: Sequence[torch.Tensor]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for layout, owner in zip(layouts, owners, strict=True):
        for tensor in (layout, owner):
            host = tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
            digest.update(host.numpy().tobytes())
            digest.update(int(host.numel()).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _cost_values(cost: PlacementCost) -> tuple[float, ...]:
    return (
        cost.communication,
        cost.compute,
        cost.communication_model_units,
        float(cost.peak_communication_rank),
        float(cost.peak_compute_rank),
        float(cost.selected_dim),
    )


def assert_exact_plan_match(reference: PlacementPlan, actual: PlacementPlan) -> None:
    """Raise with a precise field name if two exact plans diverge."""

    if reference.actions != actual.actions:
        raise AssertionError(f"action mismatch: reference={reference.actions}, actual={actual.actions}")
    if reference.initial_layout != actual.initial_layout:
        raise AssertionError("initial_layout mismatch")
    if reference.final_layout != actual.final_layout:
        raise AssertionError("final_layout mismatch")
    if reference.final_owner_slots != actual.final_owner_slots:
        raise AssertionError("final_owner_slots mismatch")
    for name in ("baseline_cost", "final_cost"):
        expected = getattr(reference, name)
        observed = getattr(actual, name)
        if _cost_values(expected) != _cost_values(observed):
            raise AssertionError(f"{name} mismatch: reference={expected}, actual={observed}")


class CPUExactPlanner:
    """Single-process CPU facade over the existing full-exact implementation."""

    def __init__(self, planner: GreedyCommunicationPlanner) -> None:
        self.planner = planner

    @torch.no_grad()
    def plan_layers(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        **kwargs,
    ) -> list[PlacementPlan]:
        """Run the unmodified exact algorithm on CPU tensors."""

        selected = [value.detach().to(device="cpu", dtype=torch.long).contiguous() for value in selected_experts]
        layouts = [value.detach().to(device="cpu", dtype=torch.long).contiguous() for value in slot_to_logical]
        owners = [value.detach().to(device="cpu", dtype=torch.long).contiguous() for value in owner_slots]
        source_ranks = kwargs.get("source_ranks")
        if not isinstance(source_ranks, int):
            kwargs["source_ranks"] = [
                value if isinstance(value, int) else value.detach().to(device="cpu", dtype=torch.long).contiguous()
                for value in source_ranks
            ]
        return self.planner.plan_layers(selected, layouts, owners, **kwargs)


class CPULayerOwnerPlanner:
    """Distributed exact planner that scores each layer on one CPU rank."""

    def __init__(
        self,
        planner: GreedyCommunicationPlanner,
        *,
        process_group: dist.ProcessGroup | None = None,
        local_process_count: int | None = None,
        resources: CPUPlannerResources | None = None,
        visible_cpu_cores: int | None = None,
        cpu_cores_per_rank: int | None = None,
        reserve_cpu_cores: int = 2,
        layer_workers: int | None = None,
        intraop_threads: int | None = None,
        validate_metadata: bool = True,
        configure_torch_threads: bool = True,
    ) -> None:
        if planner.candidate_scorer != "statistics":
            raise ValueError("CPU layer-owner planning currently requires candidate_scorer='statistics'.")
        if planner.adaptive_topk or planner.early_proxy_topk or planner.exact_primitive_topk:
            raise ValueError("CPU layer-owner planning implements full exact scoring; approximate modes must be off.")
        self.planner = planner
        self.process_group = process_group
        self.local_process_count = max(
            1,
            int(_runtime_local_process_count() if local_process_count is None else local_process_count),
        )
        self._configured_resources = resources
        self._resource_overrides = {
            "visible_cpu_cores": visible_cpu_cores,
            "cpu_cores_per_rank": cpu_cores_per_rank,
            "reserve_cpu_cores": reserve_cpu_cores,
            "layer_workers": layer_workers,
            "intraop_threads": intraop_threads,
        }
        self.validate_metadata = bool(validate_metadata)
        self.configure_torch_threads = bool(configure_torch_threads)
        self._validated_metadata: set[tuple[int, ...]] = set()
        self._metadata_lock = Lock()
        self.last_timing: CPULayerOwnerTiming | None = None

    def _group_info(self) -> tuple[int, int]:
        if not dist.is_available() or not dist.is_initialized():
            if self.planner.ep_size != 1:
                raise RuntimeError(
                    "Distributed CPU layer-owner planning requires an initialized CPU process group "
                    f"for ep_size={self.planner.ep_size}."
                )
            return 0, 1
        group = self.process_group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if world_size != self.planner.ep_size:
            raise ValueError(
                f"CPU layer-owner process group has world_size={world_size}, expected ep_size={self.planner.ep_size}."
            )
        backend = str(dist.get_backend(group)).lower().rsplit(".", maxsplit=1)[-1]
        if backend != "gloo":
            raise ValueError(f"CPU layer-owner collectives require a Gloo process group, got backend={backend!r}.")
        return int(rank), int(world_size)

    def _resources(self, layer_count: int) -> CPUPlannerResources:
        resources = self._configured_resources
        if resources is None:
            resources = resolve_cpu_planner_resources(
                layer_count=layer_count,
                local_process_count=self.local_process_count,
                **self._resource_overrides,
            )
        if self.configure_torch_threads and torch.get_num_threads() != resources.intraop_threads:
            torch.set_num_threads(resources.intraop_threads)
        return resources

    def _candidate_metadata(
        self,
        layout: torch.Tensor,
        owners: torch.Tensor,
        *,
        max_swaps: int,
        max_replicas: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        host_layout = layout.detach().to(device="cpu", dtype=torch.long).contiguous().clone()
        host_owners = owners.detach().to(device="cpu", dtype=torch.long).reshape(-1).contiguous().clone()
        if host_layout.numel() != self.planner.ep_size * self.planner.slots_per_rank:
            raise ValueError("slot_to_logical does not match ep_size * slots_per_rank.")
        if bool((host_layout < 0).any().item()):
            raise ValueError(
                "CPU layer-owner planning is a steady-state backend; empty slots must be initialized first."
            )
        all_slots = torch.arange(host_layout.numel(), dtype=torch.long)
        owner_mask = torch.zeros((host_layout.numel(),), dtype=torch.bool)
        owner_mask.scatter_(0, host_owners, True)
        rows_by_kind = []
        if max(0, int(max_swaps)) > 0:
            rows_by_kind.append(self.planner._swap_rows(host_layout, host_owners))
        if max(0, int(max_replicas)) > 0:
            cover_slots = all_slots[(~owner_mask) & (host_layout >= 0)]
            rows_by_kind.append(self.planner._cover_rows(host_layout, host_owners, cover_slots))
        nonempty = [rows for rows in rows_by_kind if rows.numel()]
        rows = torch.cat(nonempty, dim=0) if nonempty else torch.empty((0, 5), dtype=torch.long)
        return host_layout, host_owners, rows

    def _build_contexts(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_values: Sequence[int | torch.Tensor],
        layer_seeds: Sequence[int],
        max_swaps: int,
        max_replicas: int,
    ) -> list[_LayerContext]:
        contexts = []
        metadata_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer_index, (selected_raw, layout_raw, owners_raw, source_raw, layer_seed) in enumerate(
            zip(
                selected_experts,
                slot_to_logical,
                owner_slots,
                source_values,
                layer_seeds,
                strict=True,
            )
        ):
            if selected_raw.device.type != "cpu":
                raise ValueError(
                    "CPULayerOwnerPlanner requires CPU route snapshots; copy them asynchronously before submit."
                )
            selected = selected_raw.detach().to(dtype=torch.long).contiguous()
            original_ndim = selected.ndim
            if selected.ndim == 1:
                selected = selected.unsqueeze(-1)
            if selected.ndim != 2:
                raise ValueError(
                    f"selected_experts[{layer_index}] must have rank 1 or 2, got shape={tuple(selected.shape)}."
                )
            metadata_key = (id(layout_raw), id(owners_raw))
            metadata = metadata_cache.get(metadata_key)
            if metadata is None:
                layout, owners, rows = self._candidate_metadata(
                    layout_raw,
                    owners_raw,
                    max_swaps=max_swaps,
                    max_replicas=max_replicas,
                )
                copy_slots = self.planner._copy_table(layout, int(owners.numel()))
                metadata = (layout, owners, rows, copy_slots)
                metadata_cache[metadata_key] = metadata
            layout, owners, rows, copy_slots = metadata
            if isinstance(source_raw, int):
                uniform_source_rank = int(source_raw)
                sources = torch.full((selected.shape[0],), uniform_source_rank, dtype=torch.long)
            else:
                if source_raw.device.type != "cpu":
                    raise ValueError("CPU source-rank snapshots must reside on CPU.")
                uniform_source_rank = None
                sources = source_raw.detach().to(dtype=torch.long).reshape(-1).contiguous()
            if sources.numel() != selected.shape[0]:
                raise ValueError("A source_ranks tensor does not match its local token count.")
            contexts.append(
                _LayerContext(
                    selected=selected,
                    layout=layout,
                    owners=owners,
                    rows=rows,
                    copy_slots=copy_slots,
                    sources=sources,
                    uniform_source_rank=uniform_source_rank,
                    ordinals=torch.arange(selected.shape[0], dtype=torch.long),
                    layer_seed=int(layer_seed),
                    original_selected_ndim=original_ndim,
                )
            )
        return contexts

    def _prepare_layer(
        self,
        context: _LayerContext,
        *,
        step: int,
        include_assignments: bool,
    ) -> _PreparedLayer:
        prepared = self.planner._prepare_action_counts(
            context.selected,
            context.layout,
            context.rows,
            source_ranks=context.sources,
            uniform_source_rank=context.uniform_source_rank,
            copy_slots=context.copy_slots,
            affected_groups=None,
            token_ordinals=context.ordinals,
            step=step,
            layer_seed=context.layer_seed,
            num_experts=int(context.owners.numel()),
            include_assignment_counts=include_assignments,
            include_pair_interactions=True,
            include_pair_bounds=False,
        )
        communication = torch.cat((prepared.baseline_local, prepared.candidate_local), dim=0)
        communication_width = int(communication.shape[1])
        if include_assignments:
            if prepared.baseline_assignment_local is None or prepared.candidate_assignment_local is None:
                raise RuntimeError("Exact CPU planning requested missing assignment statistics.")
            assignments = torch.cat(
                (prepared.baseline_assignment_local, prepared.candidate_assignment_local),
                dim=0,
            )
            packed = torch.cat((communication, assignments), dim=1)
        else:
            packed = communication
        return _PreparedLayer(
            context=context,
            counts=prepared,
            packed_local=packed.contiguous(),
            communication_width=communication_width,
            has_assignments=include_assignments,
        )

    def _prepare_layers(
        self,
        contexts: Sequence[_LayerContext],
        *,
        step: int,
        include_assignments: bool,
        resources: CPUPlannerResources,
    ) -> list[_PreparedLayer]:
        if resources.layer_workers <= 1 or len(contexts) <= 1:
            return [
                self._prepare_layer(context, step=step, include_assignments=include_assignments)
                for context in contexts
            ]
        with ThreadPoolExecutor(
            max_workers=resources.layer_workers,
            thread_name_prefix="hiermoe-cpu-stat",
        ) as executor:
            futures = [
                executor.submit(
                    self._prepare_layer,
                    context,
                    step=step,
                    include_assignments=include_assignments,
                )
                for context in contexts
            ]
            return [future.result() for future in futures]

    def _metadata_signature(
        self,
        prepared: Sequence[_PreparedLayer],
        owner_ranks: Sequence[int],
        send_splits: Sequence[int],
    ) -> tuple[int, ...]:
        values = [len(prepared), *send_splits]
        for layer, owner_rank in zip(prepared, owner_ranks, strict=True):
            layout_bytes = layer.context.layout.numpy().tobytes()
            owner_bytes = layer.context.owners.numpy().tobytes()
            digest = hashlib.blake2b(layout_bytes + owner_bytes, digest_size=8).digest()
            values.extend(
                (
                    int(owner_rank),
                    layer.row_count,
                    int(layer.packed_local.shape[1]),
                    int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1),
                )
            )
        return tuple(values)

    def _validate_metadata_once(self, signature: tuple[int, ...], world_size: int) -> None:
        if not self.validate_metadata or world_size <= 1:
            return
        with self._metadata_lock:
            if signature in self._validated_metadata:
                return
        local = torch.tensor(signature, dtype=torch.long)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local, group=self.process_group)
        if any(not torch.equal(local, value) for value in gathered):
            raise RuntimeError("CPU layer-owner candidate metadata differs across EP ranks.")
        with self._metadata_lock:
            self._validated_metadata.add(signature)

    def _pack_statistics(
        self,
        prepared: Sequence[_PreparedLayer],
        owner_ranks: Sequence[int],
        *,
        world_size: int,
    ) -> _PackedStatistics:
        layers_by_owner = [
            [index for index, owner in enumerate(owner_ranks) if owner == destination]
            for destination in range(world_size)
        ]
        send_splits = [
            sum(prepared[layer_index].flat_size for layer_index in indices) for indices in layers_by_owner
        ]
        send_chunks = [
            torch.cat([prepared[index].packed_local.reshape(-1) for index in indices])
            if indices
            else torch.empty((0,), dtype=torch.float32)
            for indices in layers_by_owner
        ]
        send_buffer = torch.cat(send_chunks).contiguous()
        signature = self._metadata_signature(prepared, owner_ranks, send_splits)
        return _PackedStatistics(
            send_buffer=send_buffer,
            send_splits=tuple(send_splits),
            layers_by_owner=tuple(tuple(indices) for indices in layers_by_owner),
            metadata_signature=signature,
        )

    def _exchange_statistics(
        self,
        prepared: Sequence[_PreparedLayer],
        packed: _PackedStatistics,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[dict[int, torch.Tensor], int, int]:
        self._validate_metadata_once(packed.metadata_signature, world_size)
        owned_indices = packed.layers_by_owner[rank]
        owned_flat_size = packed.send_splits[rank]
        if world_size == 1:
            reduced = packed.send_buffer
        else:
            receive_buffer = torch.empty(
                (world_size * owned_flat_size,),
                dtype=packed.send_buffer.dtype,
            )
            dist.all_to_all_single(
                receive_buffer,
                packed.send_buffer,
                output_split_sizes=[owned_flat_size] * world_size,
                input_split_sizes=list(packed.send_splits),
                group=self.process_group,
            )
            reduced = receive_buffer.view(world_size, owned_flat_size).sum(dim=0)

        global_by_layer: dict[int, torch.Tensor] = {}
        offset = 0
        for layer_index in owned_indices:
            size = prepared[layer_index].flat_size
            global_by_layer[layer_index] = reduced[offset : offset + size].view_as(
                prepared[layer_index].packed_local
            )
            offset += size
        if offset != owned_flat_size:
            raise RuntimeError("CPU layer-owner statistic unpack consumed an unexpected number of values.")
        return global_by_layer, int(packed.send_buffer.numel() * packed.send_buffer.element_size()), int(
            world_size * owned_flat_size * packed.send_buffer.element_size()
        )

    def _owner_decisions(
        self,
        prepared: Sequence[_PreparedLayer],
        global_by_layer: dict[int, torch.Tensor],
        *,
        communication_scales: Sequence[float],
        compute_slopes: Sequence[float],
        compute_constants: Sequence[float],
    ) -> torch.Tensor:
        decision = torch.zeros((len(prepared), _DECISION_WIDTH), dtype=torch.float32)
        empty_long = torch.empty((0,), dtype=torch.long)
        for layer_index, global_rows in global_by_layer.items():
            layer = prepared[layer_index]
            communication = global_rows[:, : layer.communication_width]
            assignments = global_rows[:, layer.communication_width :] if layer.has_assignments else None
            scored = self.planner._score_global_count_rows(
                communication,
                assignments,
                communication_scale=communication_scales[layer_index],
                forward_compute_per_assignment=compute_slopes[layer_index],
                forward_compute_constant=compute_constants[layer_index],
                baseline_physical_routes=empty_long,
                route_hashes=empty_long,
            )
            candidate_count = int(scored.total.numel()) - 1
            winner = int(scored.total[1:].argmin().item()) if candidate_count else -1
            candidate_row = winner + 1 if winner >= 0 else 0
            baseline_values = torch.stack(
                (
                    scored.communication[0],
                    scored.compute[0],
                    scored.communication_model_units[0],
                    scored.peak_rank[0].to(torch.float32),
                    scored.peak_compute_rank[0].to(torch.float32),
                    scored.selected_dim[0].to(torch.float32),
                )
            )
            candidate_values = torch.stack(
                (
                    scored.communication[candidate_row],
                    scored.compute[candidate_row],
                    scored.communication_model_units[candidate_row],
                    scored.peak_rank[candidate_row].to(torch.float32),
                    scored.peak_compute_rank[candidate_row].to(torch.float32),
                    scored.selected_dim[candidate_row].to(torch.float32),
                )
            )
            decision[layer_index, 0] = float(winner + 1)
            decision[layer_index, 1:7] = baseline_values
            decision[layer_index, 7:13] = candidate_values
        return decision

    def _publish_decisions(self, decision: torch.Tensor, world_size: int) -> torch.Tensor:
        if world_size > 1:
            dist.all_reduce(decision, op=dist.ReduceOp.SUM, group=self.process_group)
        return decision

    def _finalize(
        self,
        prepared: Sequence[_PreparedLayer],
        decision: torch.Tensor,
        *,
        timing: CPULayerOwnerTiming,
    ) -> list[PlacementPlan]:
        plans = []
        layer_count = max(1, len(prepared))
        per_layer_total = timing.total_ms / layer_count
        per_layer_prepare = (timing.context_ms + timing.local_prepare_ms) / layer_count
        per_layer_decision = timing.decision_collective_ms / layer_count
        per_layer_score = timing.owner_score_ms / max(1, timing.owned_layer_count)
        for layer_index, layer in enumerate(prepared):
            row = decision[layer_index].tolist()
            winner = int(row[0]) - 1
            baseline_cost = self.planner._placement_cost_from_values(row[1:7])
            candidate_cost = self.planner._placement_cost_from_values(row[7:13])
            context = layer.context
            final_layout = context.layout.clone()
            final_owners = context.owners.clone()
            actions: tuple[PlacementAction, ...] = ()
            final_cost = baseline_cost
            if winner >= 0 and candidate_cost.total < baseline_cost.total:
                action = self.planner._placement_action(context.rows[winner].tolist())
                actions = (action,)
                final_cost = candidate_cost
                if action.kind == "swap":
                    final_layout[action.src_slot] = action.dst_logical
                    final_layout[action.dst_slot] = action.src_logical
                    final_owners[action.src_logical], final_owners[action.dst_logical] = (
                        context.owners[action.dst_logical],
                        context.owners[action.src_logical],
                    )
                else:
                    final_layout[action.dst_slot] = action.src_logical
            chose_swap = bool(actions) and actions[0].kind == "swap"
            chose_cover = bool(actions) and not chose_swap
            plans.append(
                PlacementPlan(
                    actions=actions,
                    initial_layout=tuple(int(value) for value in context.layout.tolist()),
                    final_layout=tuple(int(value) for value in final_layout.tolist()),
                    baseline_cost=baseline_cost,
                    final_cost=final_cost,
                    swap_rounds=int(chose_swap),
                    replica_rounds=int(chose_cover),
                    planning_ms=per_layer_total,
                    route_stats_ms=per_layer_prepare,
                    swap_ms=per_layer_score if chose_swap else 0.0,
                    replica_ms=per_layer_score if chose_cover else 0.0,
                    swap_score_ms=per_layer_score,
                    swap_update_ms=0.0,
                    swap_collective_ms=timing.statistic_collective_ms / layer_count,
                    replica_score_ms=0.0,
                    replica_update_ms=0.0,
                    replica_collective_ms=0.0,
                    decision_sync_ms=per_layer_decision,
                    finalization_ms=timing.finalization_ms / layer_count,
                    algorithm_version=CPU_LAYER_OWNER_ALGORITHM_VERSION,
                    local_physical_routes=None,
                    final_owner_slots=tuple(int(value) for value in final_owners.tolist()),
                )
            )
        return plans

    @torch.no_grad()
    def plan_layers(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_ranks: int | Sequence[int | torch.Tensor],
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
        step: int = 0,
        communication_scales: Sequence[float] | None = None,
        forward_compute_per_assignment: Sequence[float] | None = None,
        forward_compute_constant: Sequence[float] | None = None,
        owner_offset: int = 0,
    ) -> CPULayerOwnerResult:
        """Plan all steady-state layers exactly on their assigned CPU ranks."""

        started = time.perf_counter()
        layer_count = len(selected_experts)
        if not (len(slot_to_logical) == len(owner_slots) == len(layer_seeds) == layer_count):
            raise ValueError("CPU layer-owner inputs must have identical layer counts.")
        if layer_count == 0:
            raise ValueError("CPU layer-owner planning requires at least one layer.")
        rank, world_size = self._group_info()
        resources = self._resources(layer_count)
        scales = (
            [self.planner.communication_scale] * layer_count
            if communication_scales is None
            else [float(value) for value in communication_scales]
        )
        compute_slopes = (
            [self.planner.forward_compute_per_assignment] * layer_count
            if forward_compute_per_assignment is None
            else [float(value) for value in forward_compute_per_assignment]
        )
        compute_constants = (
            [self.planner.forward_compute_constant] * layer_count
            if forward_compute_constant is None
            else [float(value) for value in forward_compute_constant]
        )
        if not (len(scales) == len(compute_slopes) == len(compute_constants) == layer_count):
            raise ValueError("CPU layer-owner cost-model arrays must match the number of layers.")
        source_values = (
            [int(source_ranks)] * layer_count if isinstance(source_ranks, int) else list(source_ranks)
        )
        if len(source_values) != layer_count:
            raise ValueError("CPU layer-owner source_ranks must match the number of layers.")

        context_started = time.perf_counter()
        contexts = self._build_contexts(
            selected_experts,
            slot_to_logical,
            owner_slots,
            source_values=source_values,
            layer_seeds=layer_seeds,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
        )
        context_ms = (time.perf_counter() - context_started) * 1000.0

        prepare_started = time.perf_counter()
        include_assignments = any(value > 0.0 for value in compute_slopes)
        prepared = self._prepare_layers(
            contexts,
            step=step,
            include_assignments=include_assignments,
            resources=resources,
        )
        local_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        owner_ranks = balanced_layer_owner_ranks(layer_count, world_size, owner_offset=owner_offset)
        pack_started = time.perf_counter()
        packed = self._pack_statistics(prepared, owner_ranks, world_size=world_size)
        statistic_pack_ms = (time.perf_counter() - pack_started) * 1000.0

        collective_started = time.perf_counter()
        global_by_layer, sent_bytes, received_bytes = self._exchange_statistics(
            prepared,
            packed,
            rank=rank,
            world_size=world_size,
        )
        statistic_collective_ms = (time.perf_counter() - collective_started) * 1000.0

        owner_score_started = time.perf_counter()
        decision = self._owner_decisions(
            prepared,
            global_by_layer,
            communication_scales=scales,
            compute_slopes=compute_slopes,
            compute_constants=compute_constants,
        )
        owner_score_ms = (time.perf_counter() - owner_score_started) * 1000.0

        decision_started = time.perf_counter()
        decision = self._publish_decisions(decision, world_size)
        decision_collective_ms = (time.perf_counter() - decision_started) * 1000.0

        finalization_started = time.perf_counter()
        timing_without_finalization = CPULayerOwnerTiming(
            context_ms=context_ms,
            local_prepare_ms=local_prepare_ms,
            statistic_pack_ms=statistic_pack_ms,
            statistic_collective_ms=statistic_collective_ms,
            owner_score_ms=owner_score_ms,
            decision_collective_ms=decision_collective_ms,
            finalization_ms=0.0,
            total_ms=(time.perf_counter() - started) * 1000.0,
            local_payload_bytes=sent_bytes,
            received_payload_bytes=received_bytes,
            owned_layer_count=sum(owner == rank for owner in owner_ranks),
            layer_workers=resources.layer_workers,
            intraop_threads=resources.intraop_threads,
        )
        plans = self._finalize(prepared, decision, timing=timing_without_finalization)
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = CPULayerOwnerTiming(
            **{
                **vars(timing_without_finalization),
                "finalization_ms": finalization_ms,
                "total_ms": total_ms,
            }
        )
        per_layer_total = total_ms / layer_count
        per_layer_finalization = finalization_ms / layer_count
        plans = [
            PlacementPlan(
                **{
                    **vars(plan),
                    "planning_ms": per_layer_total,
                    "finalization_ms": per_layer_finalization,
                }
            )
            for plan in plans
        ]
        result = CPULayerOwnerResult(tuple(plans), timing, owner_ranks)
        self.last_timing = timing
        return result


class CPUHCCLBatchedPlanner:
    """Build all layer statistics on CPU and reduce one packed payload.

    The supplied reducer owns CPU-to-device staging, the HCCL collective, and
    device-to-CPU copy-back. Keeping that boundary injectable lets the runtime
    place the one collective in a deterministic backward window while this
    class remains independently testable on CPU.
    """

    def __init__(
        self,
        planner: GreedyCommunicationPlanner,
        *,
        reducer: Callable[[torch.Tensor], torch.Tensor],
        local_process_count: int | None = None,
        resources: CPUPlannerResources | None = None,
        cpu_cores_per_rank: int | None = None,
        reserve_cpu_cores: int = 2,
        layer_workers: int | None = None,
        intraop_threads: int | None = None,
    ) -> None:
        self.planner = planner
        self.reducer = reducer
        self.local_process_count = max(
            1,
            int(_runtime_local_process_count() if local_process_count is None else local_process_count),
        )
        self._configured_resources = resources
        self._resource_overrides = {
            "cpu_cores_per_rank": cpu_cores_per_rank,
            "reserve_cpu_cores": reserve_cpu_cores,
            "layer_workers": layer_workers,
            "intraop_threads": intraop_threads,
        }
        self._local = CPULayerOwnerPlanner(
            planner,
            process_group=None,
            local_process_count=self.local_process_count,
            resources=resources,
            validate_metadata=False,
        )
        self.last_timing: CPULayerOwnerTiming | None = None

    def _resources(self, layer_count: int) -> CPUPlannerResources:
        resources = self._configured_resources
        if resources is None:
            resources = resolve_cpu_planner_resources(
                layer_count=layer_count,
                local_process_count=self.local_process_count,
                **self._resource_overrides,
            )
        if torch.get_num_threads() != resources.intraop_threads:
            torch.set_num_threads(resources.intraop_threads)
        return resources

    def _score_all_layers(
        self,
        prepared: Sequence[_PreparedLayer],
        global_rows: Sequence[torch.Tensor],
        *,
        communication_scales: Sequence[float],
        compute_slopes: Sequence[float],
        compute_constants: Sequence[float],
        resources: CPUPlannerResources,
    ) -> torch.Tensor:
        def score_one(layer_index: int) -> torch.Tensor:
            decision = self._local._owner_decisions(
                prepared,
                {layer_index: global_rows[layer_index]},
                communication_scales=communication_scales,
                compute_slopes=compute_slopes,
                compute_constants=compute_constants,
            )
            return decision[layer_index]

        if resources.layer_workers <= 1 or len(prepared) <= 1:
            return torch.stack([score_one(index) for index in range(len(prepared))])
        with ThreadPoolExecutor(
            max_workers=resources.layer_workers,
            thread_name_prefix="hiermoe-cpu-score",
        ) as executor:
            return torch.stack(list(executor.map(score_one, range(len(prepared)))))

    @torch.no_grad()
    def plan_layers(
        self,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        source_ranks: int | Sequence[int | torch.Tensor],
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
        step: int = 0,
        communication_scales: Sequence[float] | None = None,
        forward_compute_per_assignment: Sequence[float] | None = None,
        forward_compute_constant: Sequence[float] | None = None,
    ) -> CPULayerOwnerResult:
        """Plan steady-state layers exactly with one injected reduction."""

        started = time.perf_counter()
        layer_count = len(selected_experts)
        if not (len(slot_to_logical) == len(owner_slots) == len(layer_seeds) == layer_count):
            raise ValueError("CPU HCCL batched planner inputs must have identical layer counts.")
        if layer_count == 0:
            raise ValueError("CPU HCCL batched planning requires at least one layer.")
        resources = self._resources(layer_count)
        scales = (
            [self.planner.communication_scale] * layer_count
            if communication_scales is None
            else [float(value) for value in communication_scales]
        )
        compute_slopes = (
            [self.planner.forward_compute_per_assignment] * layer_count
            if forward_compute_per_assignment is None
            else [float(value) for value in forward_compute_per_assignment]
        )
        compute_constants = (
            [self.planner.forward_compute_constant] * layer_count
            if forward_compute_constant is None
            else [float(value) for value in forward_compute_constant]
        )
        if not (len(scales) == len(compute_slopes) == len(compute_constants) == layer_count):
            raise ValueError("CPU HCCL batched cost-model arrays must match the number of layers.")
        source_values = (
            [int(source_ranks)] * layer_count if isinstance(source_ranks, int) else list(source_ranks)
        )
        if len(source_values) != layer_count:
            raise ValueError("CPU HCCL batched source_ranks must match the number of layers.")

        context_started = time.perf_counter()
        contexts = self._local._build_contexts(
            selected_experts,
            slot_to_logical,
            owner_slots,
            source_values=source_values,
            layer_seeds=layer_seeds,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
        )
        context_ms = (time.perf_counter() - context_started) * 1000.0

        prepare_started = time.perf_counter()
        prepared = self._local._prepare_layers(
            contexts,
            step=step,
            include_assignments=any(value > 0.0 for value in compute_slopes),
            resources=resources,
        )
        local_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        pack_started = time.perf_counter()
        flat_sizes = [layer.flat_size for layer in prepared]
        local_payload = torch.cat([layer.packed_local.reshape(-1) for layer in prepared]).contiguous()
        statistic_pack_ms = (time.perf_counter() - pack_started) * 1000.0

        collective_started = time.perf_counter()
        global_payload = self.reducer(local_payload)
        statistic_collective_ms = (time.perf_counter() - collective_started) * 1000.0
        if global_payload.device.type != "cpu" or global_payload.numel() != local_payload.numel():
            raise RuntimeError("CPU HCCL reducer must return one same-sized CPU tensor.")

        global_by_layer = []
        offset = 0
        for layer, flat_size in zip(prepared, flat_sizes, strict=True):
            global_by_layer.append(global_payload[offset : offset + flat_size].view_as(layer.packed_local))
            offset += flat_size

        score_started = time.perf_counter()
        decision = self._score_all_layers(
            prepared,
            global_by_layer,
            communication_scales=scales,
            compute_slopes=compute_slopes,
            compute_constants=compute_constants,
            resources=resources,
        )
        owner_score_ms = (time.perf_counter() - score_started) * 1000.0

        finalization_started = time.perf_counter()
        timing_without_finalization = CPULayerOwnerTiming(
            context_ms=context_ms,
            local_prepare_ms=local_prepare_ms,
            statistic_pack_ms=statistic_pack_ms,
            statistic_collective_ms=statistic_collective_ms,
            owner_score_ms=owner_score_ms,
            decision_collective_ms=0.0,
            finalization_ms=0.0,
            total_ms=(time.perf_counter() - started) * 1000.0,
            local_payload_bytes=int(local_payload.numel() * local_payload.element_size()),
            received_payload_bytes=int(global_payload.numel() * global_payload.element_size()),
            owned_layer_count=layer_count,
            layer_workers=resources.layer_workers,
            intraop_threads=resources.intraop_threads,
        )
        plans = self._local._finalize(prepared, decision, timing=timing_without_finalization)
        finalization_ms = (time.perf_counter() - finalization_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = CPULayerOwnerTiming(
            **{
                **vars(timing_without_finalization),
                "finalization_ms": finalization_ms,
                "total_ms": total_ms,
            }
        )
        per_layer_total = total_ms / layer_count
        per_layer_finalization = finalization_ms / layer_count
        plans = [
            PlacementPlan(
                **{
                    **vars(plan),
                    "planning_ms": per_layer_total,
                    "finalization_ms": per_layer_finalization,
                    "algorithm_version": CPU_HCCL_BATCHED_ALGORITHM_VERSION,
                }
            )
            for plan in plans
        ]
        result = CPULayerOwnerResult(
            tuple(plans),
            timing,
            tuple(0 for _ in range(layer_count)),
        )
        self.last_timing = timing
        return result


def _bind_current_process_to_cpus(cpu_ids: Sequence[int]) -> None:
    """Bind the current process before it creates planner worker threads."""

    cpus = {int(cpu_id) for cpu_id in cpu_ids}
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


def _shared_memory_cpu_planner_main(
    command_queue,
    collective_queue,
    result_queue,
    request_events,
    response_events,
    result_events,
    stop_event,
    planner_cpu_ids: tuple[int, ...],
) -> None:
    """Child entrypoint. It never initializes a device or distributed group."""

    _bind_current_process_to_cpus(planner_cpu_ids)
    torch.set_num_threads(1)
    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if command is None:
            break
        slot = int(command["slot"])
        source_step = int(command["source_step"])
        try:
            planner = CPUHCCLBatchedPlanner(
                command["planner"],
                reducer=lambda tensor: _shared_memory_collective_exchange(
                    tensor,
                    slot=slot,
                    source_step=source_step,
                    collective_queue=collective_queue,
                    request_event=request_events[slot],
                    response_event=response_events[slot],
                    stop_event=stop_event,
                ),
                local_process_count=1,
                reserve_cpu_cores=0,
                layer_workers=max(1, len(planner_cpu_ids)),
                intraop_threads=1,
            )
            result = planner.plan_layers(
                command["selected_experts"],
                command["slot_to_logical"],
                command["owner_slots"],
                source_ranks=int(command["source_rank"]),
                max_swaps=int(command["max_swaps"]),
                max_replicas=int(command["max_replicas"]),
                layer_seeds=command["layer_seeds"],
                step=source_step,
                communication_scales=command["communication_scales"],
                forward_compute_per_assignment=command["compute_slopes"],
                forward_compute_constant=command["compute_constants"],
            )
            payload = SharedMemoryCPUPlanResult(source_step, result, "")
        except BaseException:
            error = traceback.format_exc()
            # Wake a parent waiting for the collective window even when the
            # failure happened before the reducer was reached.
            collective_queue.put(("error", slot, source_step, error))
            request_events[slot].set()
            payload = SharedMemoryCPUPlanResult(source_step, None, error)
        result_queue.put((slot, payload))
        result_events[slot].set()


def _shared_memory_collective_exchange(
    tensor: torch.Tensor,
    *,
    slot: int,
    source_step: int,
    collective_queue,
    request_event,
    response_event,
    stop_event,
) -> torch.Tensor:
    shared = tensor.detach().contiguous()
    shared.share_memory_()
    collective_queue.put(("collective", int(slot), int(source_step), shared))
    request_event.set()
    while not response_event.wait(timeout=0.1):
        if stop_event.is_set():
            raise RuntimeError("CPU planner process stopped while waiting for its collective response.")
    return shared


class SharedMemoryCPUPlannerProcess:
    """Spawn-isolated CPU exact planner with shared-tensor IPC.

    The child builds and scores statistics. The training parent services the
    one device collective between those stages, so the child never touches
    HCCL or imports a training process group.
    """

    def __init__(self, *, planner_cpu_ids: Sequence[int], buffer_count: int = 2) -> None:
        if int(buffer_count) != 2:
            raise ValueError("Shared-memory CPU planning currently uses exactly two slots.")
        context = mp.get_context("spawn")
        self._context = context
        self._command_queue = context.Queue(maxsize=buffer_count)
        self._collective_queue = context.Queue(maxsize=buffer_count)
        self._result_queue = context.Queue(maxsize=buffer_count)
        self._request_events = tuple(context.Event() for _ in range(buffer_count))
        self._response_events = tuple(context.Event() for _ in range(buffer_count))
        self._result_events = tuple(context.Event() for _ in range(buffer_count))
        self._stop_event = context.Event()
        self._collective_by_slot: dict[int, tuple[str, int, object]] = {}
        self._result_by_slot: dict[int, SharedMemoryCPUPlanResult] = {}
        self._closed = False
        self._process = context.Process(
            target=_shared_memory_cpu_planner_main,
            args=(
                self._command_queue,
                self._collective_queue,
                self._result_queue,
                self._request_events,
                self._response_events,
                self._result_events,
                self._stop_event,
                tuple(int(cpu_id) for cpu_id in planner_cpu_ids),
            ),
            name="hiermoe-cpu-planner",
            daemon=True,
        )
        self._process.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @staticmethod
    def share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        shared = tensor.detach().to(device="cpu").contiguous()
        shared.share_memory_()
        return shared

    def submit(
        self,
        *,
        slot: int,
        source_step: int,
        planner: GreedyCommunicationPlanner,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        source_rank: int,
        max_swaps: int,
        max_replicas: int,
        layer_seeds: Sequence[int],
        communication_scales: Sequence[float],
        compute_slopes: Sequence[float],
        compute_constants: Sequence[float],
    ) -> None:
        if self._closed:
            raise RuntimeError("Shared-memory CPU planner is closed.")
        index = int(slot)
        for event in (
            self._request_events[index],
            self._response_events[index],
            self._result_events[index],
        ):
            event.clear()
        self._collective_by_slot.pop(index, None)
        self._result_by_slot.pop(index, None)
        if any(tensor.device.type != "cpu" or not tensor.is_shared() for tensor in selected_experts):
            raise ValueError("CPU planner routes must be shared-memory CPU tensors.")
        self._command_queue.put(
            {
                "slot": index,
                "source_step": int(source_step),
                "planner": planner,
                "selected_experts": tuple(selected_experts),
                "slot_to_logical": tuple(slot_to_logical),
                "owner_slots": tuple(owner_slots),
                "source_rank": int(source_rank),
                "max_swaps": int(max_swaps),
                "max_replicas": int(max_replicas),
                "layer_seeds": tuple(int(value) for value in layer_seeds),
                "communication_scales": tuple(float(value) for value in communication_scales),
                "compute_slopes": tuple(float(value) for value in compute_slopes),
                "compute_constants": tuple(float(value) for value in compute_constants),
            }
        )

    def _check_process(self) -> None:
        if not self._process.is_alive():
            raise RuntimeError(f"CPU planner process exited unexpectedly with code {self._process.exitcode}.")

    def wait_collective(
        self,
        slot: int,
        *,
        poll_seconds: float = 0.05,
    ) -> tuple[int, torch.Tensor]:
        index = int(slot)
        while not self._request_events[index].wait(timeout=poll_seconds):
            self._check_process()
        cached = self._collective_by_slot.pop(index, None)
        if cached is None:
            while True:
                kind, request_slot, source_step, payload = self._collective_queue.get()
                request_slot = int(request_slot)
                row = (str(kind), int(source_step), payload)
                if request_slot == index:
                    cached = row
                    break
                self._collective_by_slot[request_slot] = row
        kind, source_step, payload = cached
        if kind == "error":
            raise RuntimeError(f"CPU planner process failed before collective:\n{payload}")
        if not isinstance(payload, torch.Tensor) or payload.device.type != "cpu" or not payload.is_shared():
            raise RuntimeError("CPU planner collective request did not contain a shared CPU tensor.")
        return source_step, payload

    def complete_collective(self, slot: int) -> None:
        self._response_events[int(slot)].set()

    def wait_result(
        self,
        slot: int,
        *,
        poll_seconds: float = 0.05,
    ) -> SharedMemoryCPUPlanResult:
        index = int(slot)
        while not self._result_events[index].wait(timeout=poll_seconds):
            self._check_process()
        cached = self._result_by_slot.pop(index, None)
        if cached is None:
            while True:
                result_slot, payload = self._result_queue.get()
                result_slot = int(result_slot)
                if result_slot == index:
                    cached = payload
                    break
                self._result_by_slot[result_slot] = payload
        if cached.error:
            raise RuntimeError(f"CPU planner process failed:\n{cached.error}")
        if cached.result is None:
            raise RuntimeError("CPU planner process returned neither a result nor an error.")
        return cached

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        for event in self._response_events:
            event.set()
        self._command_queue.put(None)
        self._process.join(timeout=30.0 if wait else 0.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        for ipc_queue in (self._command_queue, self._collective_queue, self._result_queue):
            ipc_queue.close()
            ipc_queue.join_thread()


class AsyncCPULayerOwnerPlanner:
    """Two-slot, non-blocking frontend for :class:`CPULayerOwnerPlanner`.

    All EP ranks must submit source steps in identical order because the worker
    executes Gloo collectives.  ``poll`` never blocks.  A completed result is
    marked stale if the placement version or layout changed while it ran.
    """

    def __init__(self, planner: CPULayerOwnerPlanner, *, buffer_count: int = 2) -> None:
        if int(buffer_count) != 2:
            raise ValueError("Async CPU planning currently uses exactly two buffers.")
        self.planner = planner
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-cpu-plan")
        self._submissions: deque[_AsyncSubmission] = deque()
        self._lock = Lock()
        self._closed = False

    def submit(
        self,
        source_step: int,
        selected_experts: Sequence[torch.Tensor],
        slot_to_logical: Sequence[torch.Tensor],
        owner_slots: Sequence[torch.Tensor],
        *,
        placement_versions: Sequence[int],
        **kwargs,
    ) -> bool:
        """Submit without waiting; return ``False`` when both buffers are busy."""

        versions = tuple(int(value) for value in placement_versions)
        if len(versions) != len(selected_experts):
            raise ValueError("placement_versions must match the number of submitted layers.")
        if any(value.device.type != "cpu" for value in selected_experts):
            raise ValueError("Asynchronous CPU planning requires owned CPU route buffers.")
        requested_step = kwargs.get("step")
        if requested_step is not None and int(requested_step) != int(source_step):
            raise ValueError(
                f"Planner step={requested_step} must match the asynchronous source_step={source_step}."
            )
        kwargs["step"] = int(source_step)
        digest = _layout_digest(slot_to_logical, owner_slots)
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncCPULayerOwnerPlanner is closed.")
            if len(self._submissions) >= 2:
                return False
            future = self._executor.submit(
                self.planner.plan_layers,
                selected_experts,
                slot_to_logical,
                owner_slots,
                **kwargs,
            )
            self._submissions.append(
                _AsyncSubmission(
                    source_step=int(source_step),
                    placement_versions=versions,
                    layout_digest=digest,
                    future=future,
                )
            )
        return True

    def poll(
        self,
        *,
        current_placement_versions: Sequence[int],
        current_layouts: Sequence[torch.Tensor],
        current_owner_slots: Sequence[torch.Tensor],
    ) -> AsyncCPUPlanCompletion | None:
        """Return the oldest ready result, or ``None`` without synchronizing."""

        with self._lock:
            if not self._submissions or not self._submissions[0].future.done():
                return None
            submission = self._submissions.popleft()
        result = submission.future.result()
        current_versions = tuple(int(value) for value in current_placement_versions)
        current_digest = _layout_digest(current_layouts, current_owner_slots)
        reasons = []
        if current_versions != submission.placement_versions:
            reasons.append("placement_version_changed")
        if current_digest != submission.layout_digest:
            reasons.append("layout_changed")
        return AsyncCPUPlanCompletion(
            source_step=submission.source_step,
            result=result,
            valid=not reasons,
            stale_reason=",".join(reasons),
            placement_versions=submission.placement_versions,
        )

    def wait_next(self, timeout: float | None = None) -> CPULayerOwnerResult | None:
        """Wait for the oldest submission; intended for tests and shutdown."""

        with self._lock:
            if not self._submissions:
                return None
            submission = self._submissions.popleft()
        return submission.future.result(timeout=timeout)

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> AsyncCPULayerOwnerPlanner:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
