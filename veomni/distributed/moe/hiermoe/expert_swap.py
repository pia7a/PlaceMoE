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

import json
import math
import os
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - older torch fallback
    DTensor = ()  # type: ignore[assignment]

from ....utils import logging
from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_device_type, get_torch_device, synchronize
from .core_planner import (
    CORE_MOE_ALGORITHM_VERSION,
    CoReMoEPlanner,
    QuotaPolicyEntry,
    _deterministic_sample_indices,
    assign_tokens_to_copies_with_quota,
)
from .forward_cover_planner import (
    ForwardCoverHeuristicStatistics,
    forward_cover_local_heuristic_statistics,
    forward_cover_local_heuristic_statistics_batched,
    forward_cover_local_validation_stats,
    forward_cover_patch_source_rank_relevant,
    forward_cover_patch_validation_stats_batched,
    patch_forward_cover_routes,
    propose_forward_reuse_covers,
    rotating_service_target_rank,
)
from .greedy_planner import GreedyCommunicationPlanner, assign_tokens_to_copies_greedy
from .online_lut_planner import propose_online_lut_move
from .perf_model import HierMoEPerfModel
from .placemoe.artifacts import build_placemoe_artifact, validate_placemoe_artifact
from .placemoe.model_adapter import MoEModelAdapter, resolve_moe_model_adapter
from .placemoe.runtime import (
    HotUpdateController,
    HotUpdateJob,
    PlaceMoECalibration,
    PlaceMoERuntimeConfig,
    PlannerCommandSpec,
    UpdateKind,
    build_planner_command,
    launch_planner_process,
    planner_environment,
    terminate_planner_process,
)
from .placemoe.runtime.cpu_affinity import resolve_cpu_affinity
from .placemoe.types import LayerPlan, PlaceMoETopology
from .planner import (
    CurrentRoutePlanner,
    PlacementAction,
    PlacementCost,
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


def _env_optional_nonnegative_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring the override.", name, raw)
        return None


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
_EXACT_P1_ROUTE_SAMPLE_SIZE = _env_int(
    "VEOMNI_HIERMOE_EXACT_P1_ROUTE_SAMPLE_SIZE",
    0,
    minimum=0,
)
_SWAP_COST_CHUNK_CANDIDATES = _env_int("VEOMNI_HIERMOE_SWAP_COST_CHUNK_CANDIDATES", 96)
_GREEDY_LAYER_PARALLEL_STREAMS = _env_int("VEOMNI_HIERMOE_GREEDY_LAYER_STREAMS", 8)
_GREEDY_ADAPTIVE_TOPK_INITIAL = _env_int("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL", 32)
_GREEDY_EXACT_PRIMITIVE_TOPK = _env_int(
    "VEOMNI_HIERMOE_GREEDY_EXACT_PRIMITIVE_TOPK",
    0,
    minimum=0,
)
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
_DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS = _env_int("VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS", 4)
_FIXED_R2_LAYOUT = _env_flag("VEOMNI_HIERMOE_FIXED_R2_LAYOUT")
_FORCE_FIXED_R2_MIRRORED_REMAP = _env_flag("VEOMNI_HIERMOE_FORCE_FIXED_R2_MIRRORED_REMAP")
_GREEDY_ADAPTIVE_TOPK = _env_flag("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK")
_GREEDY_ADAPTIVE_TOPK_STRICT = _env_flag("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT")
_GREEDY_POST_SHORTLIST_COMPACT_PAIR = _env_flag("VEOMNI_HIERMOE_GREEDY_POST_SHORTLIST_COMPACT_PAIR")
_GREEDY_EXACT_PRIMITIVE_MAX_ONLY = _env_flag("VEOMNI_HIERMOE_GREEDY_EXACT_PRIMITIVE_MAX_ONLY")
_PIPELINE_STAGE_TIMING = _env_flag("VEOMNI_HIERMOE_PIPELINE_STAGE_TIMING")
_HIERMOE_DIAG_PHASES = _env_flag("VEOMNI_HIERMOE_DIAG_PHASES")
_PIPELINE_PREPARE_SUBSTAGES = (
    "planner_setup",
    "context",
    "route_hash",
    "baseline_route",
    "occupancy",
    "candidate_routes",
    "pair_events",
    "unary_statistics",
    "unary_scoring",
    "pair_statistics",
    "pair_interaction",
    "candidate_pack",
    "collective_pack",
)
_PIPELINE_PREPARE_CUT_POINTS = (2, 2, 5, 7, 10, 13)
# Event.wait() still wakes immediately on the normal path.  The timeout only
# controls how often a blocked planner checks for an exceptional future or
# shutdown.  A 1 ms timeout makes 48 layer workers contend for the GIL and the
# manager lock tens of thousands of times per second while they are supposed
# to be dormant between fixed pipeline windows.
_PIPELINE_HOST_EVENT_POLL_SECONDS = 0.05
_PIPELINE_PLAN_WORKERS = _env_int("VEOMNI_HIERMOE_PIPELINE_PLAN_WORKERS", 64)
_ABLATION_REPLAY_PATH = os.environ.get("VEOMNI_HIERMOE_ABLATION_REPLAY_PATH", "").strip()
_ABLATION_REPLAY_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_REPLAY_MODE", "off").strip().lower()
_ABLATION_MIGRATION_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE", "hidden").strip().lower()
_ABLATION_GRAD_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_GRAD_MODE", "hidden").strip().lower()
_INITIAL_LAYOUT_PATH = os.environ.get("VEOMNI_HIERMOE_INITIAL_LAYOUT", "").strip()
_PLACEMOE_RUNTIME_CONFIG = PlaceMoERuntimeConfig.from_environment()
if _PLACEMOE_RUNTIME_CONFIG.source_path:
    _INITIAL_LAYOUT_PATH = _PLACEMOE_RUNTIME_CONFIG.initial_artifact
    _ABLATION_REPLAY_PATH = _PLACEMOE_RUNTIME_CONFIG.initial_artifact
_CPU_PLANNER_MODE = os.environ.get("VEOMNI_HIERMOE_CPU_PLANNER_MODE", "off").strip().lower()
_CPU_TRAIN_CORES_PER_RANK = _env_int("VEOMNI_HIERMOE_CPU_TRAIN_CORES_PER_RANK", 8)
_NPU_LAYER_OWNER_BLOCKING = _env_flag("VEOMNI_HIERMOE_NPU_LAYER_OWNER_BLOCKING")
_NPU_LAYER_OWNER_COLLECTIVE = (
    os.environ.get(
        "VEOMNI_HIERMOE_NPU_LAYER_OWNER_COLLECTIVE",
        "reduce_scatter",
    )
    .strip()
    .lower()
)
_HOT_UPDATE = _PLACEMOE_RUNTIME_CONFIG.hot_update.enabled
_HOT_UPDATE_WORK_ROOT = _PLACEMOE_RUNTIME_CONFIG.hot_update.work_root
_HOT_UPDATE_BUILDER = _PLACEMOE_RUNTIME_CONFIG.hot_update.planner_path
_HOT_UPDATE_RESOURCES = _PLACEMOE_RUNTIME_CONFIG.resources
_HOT_UPDATE_LAST_STEP = _PLACEMOE_RUNTIME_CONFIG.hot_update.last_update_step
_HOT_UPDATE_LAYOUT_INTERVAL = _PLACEMOE_RUNTIME_CONFIG.hot_update.layout_interval_steps
_HOT_UPDATE_MAPPING_INTERVAL = _PLACEMOE_RUNTIME_CONFIG.hot_update.mapping_interval_steps
_HOT_UPDATE_INTER_MS_PER_BYTE = _PLACEMOE_RUNTIME_CONFIG.calibration.inter_ms_per_byte
_HOT_UPDATE_INTRA_MS_PER_BYTE = _PLACEMOE_RUNTIME_CONFIG.calibration.intra_ms_per_byte
_HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT = _PLACEMOE_RUNTIME_CONFIG.calibration.route_ms_per_assignment
_HOT_UPDATE_COMMUNICATION_MULTIPLIER = _PLACEMOE_RUNTIME_CONFIG.calibration.communication_multiplier
_HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT = _PLACEMOE_RUNTIME_CONFIG.calibration.compute_ms_per_assignment
_HOT_UPDATE_COMPUTE_MULTIPLIER = _PLACEMOE_RUNTIME_CONFIG.calibration.compute_multiplier


def configure_placemoe_runtime(config: PlaceMoERuntimeConfig) -> None:
    """Install the canonical runtime config before creating a manager.

    Environment variables remain a compatibility input at import time, but
    production training passes the nested VeOmni configuration explicitly.
    """

    global _PLACEMOE_RUNTIME_CONFIG
    global _INITIAL_LAYOUT_PATH, _ABLATION_REPLAY_PATH
    global _HOT_UPDATE, _HOT_UPDATE_WORK_ROOT, _HOT_UPDATE_BUILDER, _HOT_UPDATE_RESOURCES
    global _HOT_UPDATE_LAST_STEP, _HOT_UPDATE_LAYOUT_INTERVAL, _HOT_UPDATE_MAPPING_INTERVAL
    global _HOT_UPDATE_INTER_MS_PER_BYTE, _HOT_UPDATE_INTRA_MS_PER_BYTE
    global _HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT, _HOT_UPDATE_COMMUNICATION_MULTIPLIER
    global _HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT, _HOT_UPDATE_COMPUTE_MULTIPLIER

    config.validate()
    _PLACEMOE_RUNTIME_CONFIG = config
    _INITIAL_LAYOUT_PATH = config.initial_artifact
    if config.initial_artifact:
        _ABLATION_REPLAY_PATH = config.initial_artifact
    _HOT_UPDATE = config.hot_update.enabled
    _HOT_UPDATE_WORK_ROOT = config.hot_update.work_root
    _HOT_UPDATE_BUILDER = config.hot_update.planner_path
    _HOT_UPDATE_RESOURCES = config.resources
    _HOT_UPDATE_LAST_STEP = config.hot_update.last_update_step
    _HOT_UPDATE_LAYOUT_INTERVAL = config.hot_update.layout_interval_steps
    _HOT_UPDATE_MAPPING_INTERVAL = config.hot_update.mapping_interval_steps
    _HOT_UPDATE_INTER_MS_PER_BYTE = config.calibration.inter_ms_per_byte
    _HOT_UPDATE_INTRA_MS_PER_BYTE = config.calibration.intra_ms_per_byte
    _HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT = config.calibration.route_ms_per_assignment
    _HOT_UPDATE_COMMUNICATION_MULTIPLIER = config.calibration.communication_multiplier
    _HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT = config.calibration.compute_ms_per_assignment
    _HOT_UPDATE_COMPUTE_MULTIPLIER = config.calibration.compute_multiplier


def _expert_swap_diag_phase(phase: str) -> None:
    if not _HIERMOE_DIAG_PHASES:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(
        f"HIERMOE_EXPERT_SWAP_DIAG rank={rank} phase={phase} monotonic={time.monotonic():.6f}",
        flush=True,
    )


_ONLINE_FREEZE_COST_MODE = os.environ.get("VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE", "off").strip().lower()
_ONLINE_FREEZE_CALIBRATION_STEP = _env_int(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP",
    1,
    minimum=0,
)
_ONLINE_FREEZE_COMMUNICATION_RATIO = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_COMMUNICATION_RATIO",
    3.1,
)
_ONLINE_FREEZE_COMPUTE_RATIO = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_COMPUTE_RATIO",
    4.19,
)
_ONLINE_FREEZE_INTER_MS_PER_BYTE = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE",
    6.765449326279194e-08,
)
_ONLINE_FREEZE_INTRA_MS_PER_BYTE = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE",
    5.02482606728045e-09,
)
_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT",
    8.746548178958447e-05,
)
_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS",
    16.771503695343263,
)
_COST_MODEL_VERIFY = _env_flag("VEOMNI_HIERMOE_COST_MODEL_VERIFY")
_EXPORT_COST_MODEL_SAMPLES = _env_flag("VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES")
_COST_MODEL_VALIDATION_STEPS = _env_int(
    "VEOMNI_HIERMOE_COST_MODEL_VALIDATION_STEPS",
    1,
    minimum=1,
)
_ONLINE_LUT_UPDATE = _env_flag("VEOMNI_HIERMOE_ONLINE_LUT_UPDATE")
_ONLINE_LUT_START_STEP = _env_int(
    "VEOMNI_HIERMOE_ONLINE_LUT_START_STEP",
    1,
    minimum=0,
)
_ONLINE_LUT_MIN_GAIN = _env_float(
    "VEOMNI_HIERMOE_ONLINE_LUT_MIN_GAIN",
    0.0,
)
_FORWARD_REUSE_COVER = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER")
_FORWARD_REUSE_COVER_COMPUTE_WEIGHT = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_WEIGHT",
    1.0,
)
_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT",
    2.82807e-05,
)
_FORWARD_REUSE_COVER_MIN_GAIN = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_MIN_GAIN",
    0.0,
)
_FORWARD_REUSE_COVER_PATCH_REMAP = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP")
_FORWARD_REUSE_COVER_FAST = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_FAST")
_FORWARD_REUSE_COVER_ROUNDS = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ROUNDS",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_ONLY_STEP = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ONLY_STEP",
    -1,
    minimum=-1,
)
_FORWARD_REUSE_COVER_VICTIM_MODE = (
    os.environ.get("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_VICTIM_MODE", "minimum").strip().lower()
)
_FORWARD_REUSE_COVER_SERVICE_SCOPE = (
    os.environ.get("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE", "rank").strip().lower()
)
_FORWARD_REUSE_COVER_CONFIRM_SAMPLES = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_CONFIRM_SAMPLES",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP")
_FORWARD_REUSE_COVER_PROPOSAL_TOPK = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PROPOSAL_TOPK",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_EMPTY_SEEDING = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_EMPTY_SEEDING")


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
    local_assignment_count: torch.Tensor
    dispatch_start: AcceleratorEvent
    dispatch_end: AcceleratorEvent
    compute_start: AcceleratorEvent
    compute_end: AcceleratorEvent
    combine_start: AcceleratorEvent
    combine_end: AcceleratorEvent


@dataclass
class _CostModelTiming:
    step: int
    physical_routes: torch.Tensor
    local_expert_token_counts: torch.Tensor
    local_assignment_count: torch.Tensor
    communication_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None
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
    forward_compute_constant: float = 0.0


@dataclass
class ExpertLayerState:
    key: str
    module_id: int
    num_experts: int
    base_num_local_experts: int
    num_local_experts: int
    expert_parameter_names: tuple[str, ...]
    expert_parameters: tuple[torch.nn.Parameter, ...]
    model_adapter: MoEModelAdapter
    logical_to_physical: torch.Tensor
    slot_to_logical: torch.Tensor | None = None
    canonical_physical_slots: torch.Tensor | None = None
    latest_selected_experts: torch.Tensor | None = None
    latest_physical_routes: torch.Tensor | None = None
    latest_forward_baseline_communication_counts: torch.Tensor | None = None
    latest_forward_traffic_endpoint_statistics: torch.Tensor | None = None
    latest_route_step: int = -1
    last_planned_step: int = -1
    accumulated_tokens_per_local_expert: torch.Tensor | None = None
    latest_tokens_per_local_expert: torch.Tensor | None = None
    latest_hidden_size: int = 0
    latest_bytes_per_element: int = 0
    is_identity: bool = True
    _device_mapping_cache: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    source_logical_to_physical: torch.Tensor | None = None
    _device_source_mapping_cache: dict[tuple[torch.device, int], torch.Tensor] = field(default_factory=dict)
    _device_slot_layout_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    _device_redundant_groups_cache: dict[torch.device, tuple[tuple[int, torch.Tensor], ...]] = field(
        default_factory=dict
    )
    _redundant_copy_groups_cache: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    _replica_grad_schedule_cache: _ReplicaGradSchedule | None = None
    placement_version: int = 0
    pending_timing: _PendingLayerTiming | None = None
    cost_model_timings: list[_CostModelTiming] = field(default_factory=list)
    planner_calibration: _PlannerCalibration | None = None
    last_plan: PlacementPlan | None = None
    pending_physical_routes: torch.Tensor | None = None
    pending_route_data_ptr: int = 0
    active_quota_policy: tuple[QuotaPolicyEntry, ...] = ()
    fixed_r2_layout: bool = False

    @property
    def primary_parameter(self) -> torch.nn.Parameter:
        return self.expert_parameters[0]

    def named_expert_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        return tuple(zip(self.expert_parameter_names, self.expert_parameters, strict=True))

    def invalidate_cache(self) -> None:
        self._device_mapping_cache.clear()
        self._device_source_mapping_cache.clear()
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

    def source_mapping_for_device(self, device: torch.device, source_rank: int) -> torch.Tensor:
        if self.source_logical_to_physical is None:
            raise RuntimeError(f"HierMoE layer {self.key} has no source-rank route LUT.")
        key = (device, int(source_rank))
        cached = self._device_source_mapping_cache.get(key)
        if cached is None:
            cached = self.source_logical_to_physical[int(source_rank)].to(device=device, non_blocking=True)
            self._device_source_mapping_cache[key] = cached
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
class _PipelinePlanResult:
    layer_key: str
    source_step: int
    placement_version: int
    plan: PlacementPlan
    raw_ms: float
    latency_ms: float
    prepare_device_ms: float = 0.0
    collective_device_ms: float = 0.0
    score_device_ms: float = 0.0
    prepare_substage_device_ms: dict[str, float] = field(default_factory=dict)
    prepare_substage_host_ms: dict[str, float] = field(default_factory=dict)
    prepare_substage_thread_cpu_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class _PipelinePrepareSubstageTiming:
    start_event: AcceleratorEvent | None = None
    started_at: float = 0.0
    started_thread_at: float = 0.0
    event_ranges: dict[str, list[tuple[AcceleratorEvent | None, AcceleratorEvent | None]]] = field(
        default_factory=dict
    )
    host_ms: dict[str, float] = field(default_factory=dict)
    thread_cpu_ms: dict[str, float] = field(default_factory=dict)

    def begin(self, event: AcceleratorEvent | None, started_at: float, started_thread_at: float) -> None:
        self.start_event = event
        self.started_at = started_at
        self.started_thread_at = started_thread_at

    def checkpoint(
        self,
        stage: str,
        ended_at: float,
        ended_thread_at: float,
        end_event: AcceleratorEvent | None,
    ) -> None:
        if stage not in _PIPELINE_PREPARE_SUBSTAGES:
            raise ValueError(f"Unknown pipeline Prepare substage: {stage}")
        self.event_ranges.setdefault(stage, []).append((self.start_event, end_event))
        self.host_ms[stage] = self.host_ms.get(stage, 0.0) + (ended_at - self.started_at) * 1000.0
        self.thread_cpu_ms[stage] = (
            self.thread_cpu_ms.get(stage, 0.0) + (ended_thread_at - self.started_thread_at) * 1000.0
        )
        self.start_event = end_event
        self.started_at = time.perf_counter()
        self.started_thread_at = time.thread_time()

    def durations_ms(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        device_ms = {
            stage: sum(0.0 if start is None or end is None else start.elapsed_time(end) for start, end in ranges)
            for stage, ranges in self.event_ranges.items()
        }
        return device_ms, dict(self.host_ms), dict(self.thread_cpu_ms)


@dataclass
class _PipelinePlannerStageTiming:
    prepare_start: AcceleratorEvent | None = None
    prepare_end: AcceleratorEvent | None = None
    collective_start: AcceleratorEvent | None = None
    collective_end: AcceleratorEvent | None = None
    score_start: AcceleratorEvent | None = None
    score_end: AcceleratorEvent | None = None

    @staticmethod
    def _elapsed(start: AcceleratorEvent | None, end: AcceleratorEvent | None) -> float:
        return 0.0 if start is None or end is None else start.elapsed_time(end)

    def durations_ms(self) -> tuple[float, float, float]:
        return (
            self._elapsed(self.prepare_start, self.prepare_end),
            self._elapsed(self.collective_start, self.collective_end),
            self._elapsed(self.score_start, self.score_end),
        )


@dataclass(frozen=True)
class _PipelineMigrationResult:
    layer_key: str
    source_step: int
    committed: tuple[str, ...]
    raw_ms: float


@dataclass(frozen=True)
class _PipelineGradResult:
    layer_key: str
    raw_ms: float
    start_event: AcceleratorEvent | None = None
    completion_event: AcceleratorEvent | None = None


@dataclass(frozen=True)
class _CPUBatchedPlanResult:
    source_step: int
    placement_versions: tuple[int, ...]
    plans: tuple[PlacementPlan, ...]
    route_copy_ms: float
    active_ms: float
    latency_ms: float
    timing: Any


@dataclass
class _CPUBatchedPlanState:
    source_step: int
    placement_versions: tuple[int, ...]
    submitted_at: float
    background: bool
    collective_ready: Event = field(default_factory=Event)
    collective_gate: Event = field(default_factory=Event)
    collective_enqueued: Event = field(default_factory=Event)
    collective_done_event: AcceleratorEvent | None = None
    collective_error: BaseException | None = None
    collective_gate_wait_ms: float = 0.0
    collective_ready_host_wait_ms: float = 0.0
    collective_close_host_wait_ms: float = 0.0
    process_slot: int = -1
    route_share_ms: float = 0.0
    process_collective_active_ms: float = 0.0
    process_collective_future: Future[Any] | None = None
    future: Future[_CPUBatchedPlanResult] | None = None


@dataclass(frozen=True)
class _PendingPipelinePlan:
    plan: PlacementPlan
    source_step: int
    placement_version: int


@dataclass
class _PipelinePlannerWindows:
    prepare_gates: tuple[Event, ...] = field(
        default_factory=lambda: tuple(Event() for _ in _PIPELINE_PREPARE_CUT_POINTS)
    )
    prepare_enqueued: tuple[Event, ...] = field(
        default_factory=lambda: tuple(Event() for _ in _PIPELINE_PREPARE_CUT_POINTS)
    )
    prepare_done_events: list[AcceleratorEvent | None] = field(
        default_factory=lambda: [None] * len(_PIPELINE_PREPARE_CUT_POINTS)
    )
    prepare_a2a_end_events: list[AcceleratorEvent | None] = field(
        default_factory=lambda: [None] * len(_PIPELINE_PREPARE_CUT_POINTS)
    )
    prepare_next_window: int = 0
    collective_gate: Event = field(default_factory=Event)
    collective_tensor_ready: Event = field(default_factory=Event)
    collective_result_ready: Event = field(default_factory=Event)
    collective_done: Event = field(default_factory=Event)
    collective_tensor: torch.Tensor | None = None
    collective_device: torch.device | None = None
    collective_timing: _PipelinePlannerStageTiming | None = None
    collective_error: BaseException | None = None
    collective_future: Future[Any] | None = None
    collective_done_event: AcceleratorEvent | None = None
    collective_deadline_event: AcceleratorEvent | None = None
    score_gate: Event = field(default_factory=Event)
    score_done: Event = field(default_factory=Event)
    score_done_event: AcceleratorEvent | None = None
    score_deadline_event: AcceleratorEvent | None = None


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
    globally_ordered_pairs: bool


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
        adapter = resolve_moe_model_adapter(module)
        if adapter is None:
            continue
        num_experts = adapter.num_experts(module)
        if num_experts % int(ep_size) != 0:
            raise ValueError(
                f"HierMoE redundant slots require num_experts={num_experts} divisible by ep_size={ep_size}."
            )
        base_slots = num_experts // int(ep_size)
        target_slots = base_slots + increment
        expert_parameters = adapter.expert_parameters(module)
        local_parameters = tuple(_local_tensor_view(item.parameter) for item in expert_parameters)
        if all(int(parameter.shape[0]) == target_slots for parameter in local_parameters):
            continue
        invalid = [
            f"{item.name}={tuple(parameter.shape)}"
            for item, parameter in zip(expert_parameters, local_parameters, strict=True)
            if int(parameter.shape[0]) != base_slots
        ]
        if invalid:
            raise ValueError(
                "HierMoE redundant slot expansion must run immediately after EP slicing. "
                f"Expected {base_slots} local experts, got {', '.join(invalid)}."
            )
        for item in expert_parameters:
            adapter.replace_expert_parameter(
                module, item.name, _expanded_local_parameter(item.parameter, target_slots)
            )
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
    group_desc: str = "hiermoe_expert_swap",
) -> dist.ProcessGroup | None:
    if ep_group is None or ep_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    global_ranks = [_ep_global_rank(ep_group, rank) for rank in range(ep_size)]
    try:
        swap_group = dist.new_group(
            ranks=global_ranks,
            backend=dist.get_backend(ep_group),
            use_local_synchronization=True,
            group_desc=group_desc,
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
def _placement_group_boolean_consensus(
    local_value: bool,
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple[bool, bool]:
    """Return whether every rank is true and whether all ranks agree."""

    if ep_size <= 1 or ep_group is None:
        return bool(local_value), True
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor([int(local_value)], dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.SUM, group=ep_group)
    true_count = int(status.item())
    return true_count == ep_size, true_count in (0, ep_size)


@torch.no_grad()
def _placement_group_all_true_mask(
    local_values: Sequence[bool],
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple[bool, ...]:
    """Return a rank-consistent mask that is true only when every rank is ready."""

    if ep_size <= 1 or ep_group is None:
        return tuple(bool(value) for value in local_values)
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor(local_values, dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=ep_group)
    return tuple(bool(value) for value in status.to(device="cpu").tolist())


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
        fixed_pipeline_overlap: bool = False,
        greedy_max_copies_per_expert: int = 4,
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
        if not 1 <= greedy_max_copies_per_expert <= 8:
            raise ValueError("greedy_max_copies_per_expert must be between 1 and 8.")
        self.greedy_max_copies_per_expert = int(greedy_max_copies_per_expert)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.expert_swap_mode = str(expert_swap_mode)
        self.expert_swap_selector = str(expert_swap_selector)
        if self.expert_swap_selector not in {
            "current_joint",
            "hiermoe_exact_p1",
            "hiermoe_greedy_cover_p1",
            "legacy_batched",
        }:
            raise ValueError(
                "expert_swap_selector must be current_joint, hiermoe_exact_p1, "
                "hiermoe_greedy_cover_p1, or legacy_batched."
            )
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
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            if self.expert_swap_max_pairs_per_layer > 1:
                raise ValueError("hiermoe_greedy_cover_p1 supports at most one steady-state swap per layer.")
            if self.redundant_slot_increment_per_device <= 0:
                raise ValueError("hiermoe_greedy_cover_p1 requires redundant expert slots.")
        self.fixed_pipeline_overlap = bool(fixed_pipeline_overlap)
        if self.fixed_pipeline_overlap and (
            self.expert_swap_mode != "step" or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
        ):
            raise ValueError("fixed_pipeline_overlap requires step mode with the hiermoe_greedy_cover_p1 selector.")
        if _ABLATION_REPLAY_MODE not in {"off", "static", "step"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_REPLAY_MODE must be off, static, or step, got {_ABLATION_REPLAY_MODE!r}."
            )
        if _ABLATION_MIGRATION_MODE not in {"hidden", "blocking"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE must be hidden or blocking, got {_ABLATION_MIGRATION_MODE!r}."
            )
        if _ABLATION_GRAD_MODE not in {"hidden", "blocking"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_GRAD_MODE must be hidden or blocking, got {_ABLATION_GRAD_MODE!r}."
            )
        if _CPU_PLANNER_MODE not in {
            "off",
            "blocking",
            "background",
            "process_blocking",
            "process_background",
        }:
            raise ValueError(
                "VEOMNI_HIERMOE_CPU_PLANNER_MODE must be off, blocking, background, "
                "process_blocking, or process_background, "
                f"got {_CPU_PLANNER_MODE!r}."
            )
        if _CPU_PLANNER_MODE != "off" and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
        ):
            raise ValueError(
                "The experimental CPU planner requires fixed-pipeline step mode with "
                "the hiermoe_greedy_cover_p1 selector."
            )
        if _NPU_LAYER_OWNER_BLOCKING and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or _CPU_PLANNER_MODE != "off"
        ):
            raise ValueError(
                "Blocking NPU layer-owner planning requires fixed-pipeline step mode, "
                "the hiermoe_greedy_cover_p1 selector, and CPU planner mode off."
            )
        if _NPU_LAYER_OWNER_COLLECTIVE not in {"reduce_scatter", "all_to_all"}:
            raise ValueError(
                "VEOMNI_HIERMOE_NPU_LAYER_OWNER_COLLECTIVE must be reduce_scatter or all_to_all, "
                f"got {_NPU_LAYER_OWNER_COLLECTIVE!r}."
            )
        if _ONLINE_FREEZE_COST_MODE not in {"off", "communication", "joint"}:
            raise ValueError(
                "VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE must be off, communication, or joint, "
                f"got {_ONLINE_FREEZE_COST_MODE!r}."
            )
        if _ONLINE_FREEZE_COST_MODE != "off" and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or not _FIXED_R2_LAYOUT
            or self.expert_swap_max_pairs_per_layer != 0
            or self.max_replica_rounds != self.replica_slot_capacity
        ):
            raise ValueError(
                "The online freeze experiment requires fixed R2, fixed-pipeline step mode, "
                "the hiermoe_greedy_cover_p1 selector, zero swaps, and one initialization "
                "round for every redundant slot."
            )
        if _COST_MODEL_VERIFY and (
            self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or not _FIXED_R2_LAYOUT
            or self.expert_swap_max_pairs_per_layer != 0
            or _ONLINE_FREEZE_COST_MODE != "off"
            or _FORWARD_REUSE_COVER
        ):
            raise ValueError(
                "Cost-model verification requires fixed R2 step mode, "
                "the hiermoe_greedy_cover_p1 selector, zero swaps, and all placement "
                "experiments disabled."
            )
        if _ONLINE_LUT_UPDATE and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or _ABLATION_REPLAY_MODE != "static"
            or not _FORWARD_REUSE_COVER
            or not _FORWARD_REUSE_COVER_PATCH_REMAP
            or _CPU_PLANNER_MODE != "off"
            or _NPU_LAYER_OWNER_BLOCKING
            or _ONLINE_FREEZE_COST_MODE != "off"
            or _COST_MODEL_VERIFY
        ):
            raise ValueError(
                "Online LUT correction requires a preloaded static layout, fixed-pipeline "
                "step mode, the hiermoe_greedy_cover_p1 selector, source-LUT Forward "
                "routing, and all other online planners disabled."
            )
        if _HOT_UPDATE and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or _ABLATION_REPLAY_MODE not in {"off", "static"}
            or _ONLINE_LUT_UPDATE
            or _CPU_PLANNER_MODE != "off"
            or _NPU_LAYER_OWNER_BLOCKING
            or _ONLINE_FREEZE_COST_MODE != "off"
            or _COST_MODEL_VERIFY
        ):
            raise ValueError(
                "PlaceMoE hot updates require fixed-pipeline step mode and all other online planners disabled."
            )
        if _HOT_UPDATE and not _HOT_UPDATE_WORK_ROOT:
            raise ValueError("PlaceMoE hot-update work root must not be empty.")
        if _FORWARD_REUSE_COVER and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or (not _FIXED_R2_LAYOUT and not _FORWARD_REUSE_COVER_EMPTY_SEEDING)
            or _CPU_PLANNER_MODE != "off"
            or _NPU_LAYER_OWNER_BLOCKING
            or _ONLINE_FREEZE_COST_MODE != "off"
            or _ABLATION_REPLAY_MODE not in {"off", "static"}
        ):
            raise ValueError(
                "Forward-reuse cover planning requires fixed R2 or explicit empty seeding, "
                "fixed-pipeline step mode, the hiermoe_greedy_cover_p1 selector, and all "
                "other online planners disabled."
            )
        if _FORWARD_REUSE_COVER_EMPTY_SEEDING and _FIXED_R2_LAYOUT:
            raise ValueError("Empty-seeding Forward Cover requires VEOMNI_HIERMOE_FIXED_R2_LAYOUT=0.")
        if _FORWARD_REUSE_COVER_PATCH_REMAP and not _FORWARD_REUSE_COVER:
            raise ValueError("Forward-reuse patch remapping requires VEOMNI_HIERMOE_FORWARD_REUSE_COVER=1.")
        if _FORWARD_REUSE_COVER_FAST and not _FORWARD_REUSE_COVER_PATCH_REMAP:
            raise ValueError("Fast Forward-reuse Cover requires VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP=1.")
        if _FORWARD_REUSE_COVER_FAST and _FORWARD_REUSE_COVER_CONFIRM_SAMPLES > 1:
            raise ValueError("Multi-sample Cover confirmation requires exact global validation.")
        if _FORWARD_REUSE_COVER_FAST and _FORWARD_REUSE_COVER_PROPOSAL_TOPK > 1:
            raise ValueError("Top-K Cover proposals require exact global validation.")
        if _FORWARD_REUSE_COVER_SERVICE_SCOPE not in {"rank", "node"}:
            raise ValueError(
                "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE must be rank or node, "
                f"got {_FORWARD_REUSE_COVER_SERVICE_SCOPE!r}."
            )
        if _ABLATION_REPLAY_MODE == "step" and not self.fixed_pipeline_overlap:
            raise ValueError("Step-by-step HierMoE ablation replay requires fixed_pipeline_overlap=true.")
        if _ABLATION_REPLAY_MODE != "off" and not _ABLATION_REPLAY_PATH:
            raise ValueError("VEOMNI_HIERMOE_ABLATION_REPLAY_PATH is required when ablation replay is enabled.")
        self._ablation_replay_mode = _ABLATION_REPLAY_MODE
        self._initial_layout_path = _INITIAL_LAYOUT_PATH
        self._ablation_migration_mode = _ABLATION_MIGRATION_MODE
        self._ablation_grad_mode = _ABLATION_GRAD_MODE
        self._cpu_planner_mode = _CPU_PLANNER_MODE
        self._npu_layer_owner_blocking = _NPU_LAYER_OWNER_BLOCKING
        self._npu_layer_owner_collective = _NPU_LAYER_OWNER_COLLECTIVE
        self._online_freeze_cost_mode = _ONLINE_FREEZE_COST_MODE
        self._online_freeze_calibration_step = _ONLINE_FREEZE_CALIBRATION_STEP
        self._online_freeze_communication_ratio = _ONLINE_FREEZE_COMMUNICATION_RATIO
        self._online_freeze_compute_ratio = _ONLINE_FREEZE_COMPUTE_RATIO
        self._online_freeze_inter_ms_per_byte = _ONLINE_FREEZE_INTER_MS_PER_BYTE
        self._online_freeze_intra_ms_per_byte = _ONLINE_FREEZE_INTRA_MS_PER_BYTE
        self._online_freeze_route_ms_per_assignment = _ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT
        self._online_freeze_traffic_intercept_ms = _ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS
        self._cost_model_verify = _COST_MODEL_VERIFY
        self._cost_model_validation_steps = _COST_MODEL_VALIDATION_STEPS
        self._cost_model_verify_coefficients: tuple[float, float, float, float] | None = None
        self._cost_model_verify_receive_only_coefficients: tuple[float, float] | None = None
        self._cost_model_verify_feature_coefficients: (
            dict[
                str,
                dict[str, tuple[tuple[float, ...], float]],
            ]
            | None
        ) = None
        self._cost_model_verify_complete = False
        self._online_lut_update = _ONLINE_LUT_UPDATE
        self._online_lut_start_step = _ONLINE_LUT_START_STEP
        self._online_lut_min_gain = _ONLINE_LUT_MIN_GAIN
        self._forward_reuse_cover = _FORWARD_REUSE_COVER
        self._forward_reuse_cover_patch_remap = _FORWARD_REUSE_COVER_PATCH_REMAP
        self._forward_reuse_cover_fast = _FORWARD_REUSE_COVER_FAST
        self._forward_reuse_cover_compute_weight = _FORWARD_REUSE_COVER_COMPUTE_WEIGHT
        self._forward_reuse_cover_compute_ms_per_assignment = _FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT
        self._forward_reuse_cover_min_gain = _FORWARD_REUSE_COVER_MIN_GAIN
        self._forward_reuse_cover_rounds = _FORWARD_REUSE_COVER_ROUNDS
        self._forward_reuse_cover_only_step = _FORWARD_REUSE_COVER_ONLY_STEP
        self._forward_reuse_cover_victim_mode = _FORWARD_REUSE_COVER_VICTIM_MODE
        self._forward_reuse_cover_service_scope = _FORWARD_REUSE_COVER_SERVICE_SCOPE
        self._forward_reuse_cover_confirm_samples = _FORWARD_REUSE_COVER_CONFIRM_SAMPLES
        self._forward_reuse_cover_aggregate_service_group = _FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP
        self._forward_reuse_cover_proposal_topk = _FORWARD_REUSE_COVER_PROPOSAL_TOPK
        self._forward_reuse_cover_empty_seeding = _FORWARD_REUSE_COVER_EMPTY_SEEDING
        self._forward_reuse_cover_pending: dict[str, tuple[PlacementAction, int]] = {}
        self._hot_update = _HOT_UPDATE
        self._hot_update_layout_interval = int(_HOT_UPDATE_LAYOUT_INTERVAL)
        self._hot_update_mapping_interval = int(_HOT_UPDATE_MAPPING_INTERVAL)
        self._hot_update_controller = HotUpdateController(
            layout_interval_steps=self._hot_update_layout_interval,
            mapping_interval_steps=self._hot_update_mapping_interval,
            last_update_step=_HOT_UPDATE_LAST_STEP,
            failure_policy=_PLACEMOE_RUNTIME_CONFIG.hot_update.failure_policy,
        )
        self._hot_update_updates = 0
        self._hot_update_layout_updates = 0
        self._hot_update_mapping_updates = 0
        self._hot_update_last_source_step = -1
        self._hot_update_last_apply_step = -1
        self._hot_update_last_staleness_steps = -1
        self._hot_update_last_snapshot_ms = 0.0
        self._hot_update_last_planner_ms = 0.0
        self._hot_update_last_migration_ms = 0.0
        self._hot_update_last_moved_slots = 0
        if self._forward_reuse_cover_service_scope == "rank":
            self._forward_reuse_cover_service_group_size = 1
        else:
            proper_group_sizes = [
                int(group_size)
                for group_size in self.hierarchy.group_sizes
                if 1 < int(group_size) < self.ep_size and self.ep_size % int(group_size) == 0
            ]
            if not proper_group_sizes:
                raise ValueError("Node-scoped Forward-reuse Cover requires a proper hierarchy group size.")
            self._forward_reuse_cover_service_group_size = min(proper_group_sizes)
        self._ablation_actions_by_step: dict[int, dict[str, tuple[tuple[str, str], ...]]] = {}
        self._ablation_expected_layouts: dict[str, tuple[int, ...]] = {}
        self._ablation_expected_owner_slots: dict[str, tuple[int, ...]] = {}
        self._ablation_expected_source_luts: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._ablation_initial_layout = ""
        layout_metadata_path = self._initial_layout_path or (
            _ABLATION_REPLAY_PATH if self._ablation_replay_mode != "off" else ""
        )
        if layout_metadata_path:
            self._load_ablation_replay(layout_metadata_path)

        self.activation_checkpointing_enabled = bool(activation_checkpointing_enabled)
        self.gradient_overlap_enabled = bool(
            self._ablation_grad_mode == "hidden"
            and self.redundant_slot_increment_per_device > 0
            and ep_group is not None
        )
        self._swap_group = (
            _create_expert_swap_process_group(ep_group, self.ep_size)
            if self.expert_swap_max_pairs_per_layer > 0 and not self.fixed_pipeline_overlap
            else None
        )
        # Planner and migration collectives are serialized and can reuse the
        # training EP group. Hidden replica-gradient P2P is launched on a
        # separate accelerator stream while backward/FSDP collectives are active.
        # Arbitrary partial-capacity layouts require multiple peer waves, so
        # sharing the training group can violate cross-stream HCCL ordering.
        # Give only that path a dedicated group.
        self._pipeline_background_group = ep_group if self.fixed_pipeline_overlap else None
        self._pipeline_planner_group = self._pipeline_background_group
        self._pipeline_migration_group = self._pipeline_background_group
        self._owns_pipeline_grad_group = self.gradient_overlap_enabled
        self._pipeline_grad_group = (
            _create_expert_swap_process_group(
                ep_group,
                self.ep_size,
                group_desc="hiermoe_pipeline_grad",
            )
            if self._owns_pipeline_grad_group
            else self._pipeline_background_group
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
        self._replica_grad_buffers: dict[tuple[str, int, str, str], torch.Tensor] = {}
        self._swap_staging_buffers: dict[tuple[torch.device, torch.dtype], _SwapStagingBuffer] = {}
        self._swap_comm_streams: dict[torch.device, Any] = {}
        self._pending_layer_swaps: dict[str, _PendingLayerSwap] = {}
        self._exact_candidate_pair_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.latest_pair: str = "none"
        self._pending_state: dict[str, Any] | None = None
        self._placement_metrics: dict[str, float | int | str] = {}
        self._metrics_step = -1

        self._pipeline_lock = Lock()
        self._pipeline_grad_submit_lock = Lock()
        self._pipeline_plan_worker_capacity = max(1, _PIPELINE_PLAN_WORKERS)
        self._pipeline_plan_executor = (
            ThreadPoolExecutor(
                max_workers=self._pipeline_plan_worker_capacity,
                thread_name_prefix="hiermoe-plan",
            )
            if self.fixed_pipeline_overlap
            else None
        )
        self._pipeline_collective_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-collective")
            if self.fixed_pipeline_overlap
            else None
        )
        self._pipeline_migration_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-move")
            if self.fixed_pipeline_overlap
            else None
        )
        # NCCL host launches stay on the autograd thread; GPU work overlaps on its dedicated stream.
        self._pipeline_grad_executor = None
        self._cpu_plan_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-cpu-batch")
            if self._cpu_planner_mode == "background"
            else None
        )
        self._pipeline_streams: dict[tuple[str, torch.device], Any] = {}
        self._pipeline_plan_futures: dict[str, Future[_PipelinePlanResult]] = {}
        self._pipeline_planner_windows: dict[str, _PipelinePlannerWindows] = {}
        self._pipeline_planner_dispatch_events: dict[str, Any] = {}
        self._pipeline_planner_compute_events: dict[str, Any] = {}
        self._pipeline_pending_plans: dict[str, _PendingPipelinePlan] = {}
        self._pipeline_migration_futures: dict[str, Future[_PipelineMigrationResult]] = {}
        self._pipeline_grad_futures: dict[str, Future[_PipelineGradResult]] = {}
        self._pipeline_grad_ready: dict[str, set[int]] = defaultdict(set)
        self._pipeline_grad_ready_events: dict[str, dict[int, Any]] = defaultdict(dict)
        self._pipeline_grad_dispatch_complete: set[str] = set()
        self._pipeline_grad_window_waited: set[str] = set()
        self._pipeline_grad_comm_blocked = False
        self._pipeline_grad_window_exposed_ms = 0.0
        self._pipeline_grad_hook_handles: list[Any] = []
        self._pipeline_grad_hook_params: set[int] = set()
        self._cpu_batch_state: _CPUBatchedPlanState | None = None
        self._cpu_process_runtime: Any | None = None
        self._cpu_training_affinity: tuple[int, ...] = ()
        self._cpu_planner_affinity: tuple[int, ...] = ()
        self._hot_update_resources = _HOT_UPDATE_RESOURCES
        self._hot_update_affinity_automatic = False
        self._hot_update_planner_physical_cores = 0
        self._pipeline_step = -1
        self._pipeline_micro_step = 0
        self._pipeline_num_micro_steps = 1
        self._pipeline_next_migration_index = 0
        self._pipeline_next_grad_index = 0
        self._pipeline_layer_order: tuple[str, ...] = ()
        self._pipeline_shutdown = False

    def _load_ablation_replay(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load HierMoE ablation replay from {path!r}.") from error

        topology = payload.get("topology", {})
        replay_ep_size = int(topology.get("ep_size", -1))
        if replay_ep_size != self.ep_size:
            raise ValueError(f"HierMoE ablation replay uses ep_size={replay_ep_size}, current ep_size={self.ep_size}.")
        raw_steps = payload.get("replay", {}).get("actions_by_step", {})
        if not isinstance(raw_steps, dict) or not raw_steps:
            raise ValueError("HierMoE ablation replay contains no actions_by_step.")

        actions_by_step: dict[int, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for raw_step, rows in raw_steps.items():
            step = int(raw_step)
            if not isinstance(rows, list):
                raise ValueError(f"HierMoE ablation replay step {step} is not a list.")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"HierMoE ablation replay step {step} contains a non-mapping action.")
                layer = str(row.get("layer", ""))
                kind = str(row.get("kind", ""))
                body = str(row.get("body", ""))
                if not layer or kind not in {"swap", "replica", "empty"} or not body:
                    raise ValueError(f"HierMoE ablation replay step {step} contains an invalid action: {row!r}.")
                actions_by_step[step][layer].append((kind, body))
        self._ablation_actions_by_step = {
            step: {layer: tuple(actions) for layer, actions in by_layer.items()}
            for step, by_layer in actions_by_step.items()
        }

        raw_layers = payload.get("layers", {})
        if not isinstance(raw_layers, dict) or not raw_layers:
            raise ValueError("HierMoE ablation replay contains no final layer layouts.")
        expected_layouts: dict[str, tuple[int, ...]] = {}
        expected_owner_slots: dict[str, tuple[int, ...]] = {}
        expected_source_luts: dict[str, tuple[tuple[int, ...], ...]] = {}
        for raw_layer, raw_layer_payload in raw_layers.items():
            layer = str(raw_layer)
            if not isinstance(raw_layer_payload, dict):
                raise ValueError(f"HierMoE ablation layer {layer!r} is not a mapping.")
            expected_layouts[layer] = tuple(int(value) for value in raw_layer_payload["slot_to_logical"])
            raw_owners = raw_layer_payload.get("owner_slots")
            if raw_owners is not None:
                expected_owner_slots[layer] = tuple(int(value) for value in raw_owners)
            raw_source_lut = raw_layer_payload.get("source_logical_to_physical")
            if raw_source_lut is not None:
                expected_source_luts[layer] = tuple(tuple(int(value) for value in row) for row in raw_source_lut)
        self._ablation_expected_layouts = expected_layouts
        self._ablation_expected_owner_slots = expected_owner_slots
        self._ablation_expected_source_luts = expected_source_luts
        self._ablation_initial_layout = str(payload.get("source", {}).get("initial_layout", "fixed_r2"))

    def _normalize_ablation_layer_keys(self) -> None:
        """Match replay keys to the model after checkpoint wrapper conversion."""

        if not self._ablation_expected_layouts:
            return

        replay_keys = set(self._ablation_expected_layouts)
        replay_keys.update(self._ablation_expected_owner_slots)
        replay_keys.update(self._ablation_expected_source_luts)
        for layers_by_step in self._ablation_actions_by_step.values():
            replay_keys.update(layers_by_step)

        key_mapping: dict[str, str] = {}
        for replay_key in replay_keys:
            if replay_key in self.layers:
                key_mapping[replay_key] = replay_key
                continue
            marker = ".layers."
            suffix = replay_key[replay_key.index(marker) :] if marker in replay_key else ""
            matches = [model_key for model_key in self.layers if suffix and model_key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"HierMoE ablation layer {replay_key!r} has no unambiguous model layer; "
                    f"suffix={suffix!r}, matches={matches}."
                )
            key_mapping[replay_key] = matches[0]

        def remap_values(values: dict[str, Any]) -> dict[str, Any]:
            remapped: dict[str, Any] = {}
            for replay_key, value in values.items():
                model_key = key_mapping[replay_key]
                if model_key in remapped:
                    raise RuntimeError(f"Multiple HierMoE replay layers resolve to model layer {model_key!r}.")
                remapped[model_key] = value
            return remapped

        self._ablation_expected_layouts = remap_values(self._ablation_expected_layouts)
        self._ablation_expected_owner_slots = remap_values(self._ablation_expected_owner_slots)
        self._ablation_expected_source_luts = remap_values(self._ablation_expected_source_luts)
        self._ablation_actions_by_step = {
            step: remap_values(layers_by_step) for step, layers_by_step in self._ablation_actions_by_step.items()
        }

    @staticmethod
    def _zero_placement_cost() -> PlacementCost:
        return PlacementCost(
            communication=0.0,
            compute=0.0,
            communication_model_units=0.0,
            peak_communication_rank=-1,
            peak_compute_rank=-1,
            selected_dim=0,
        )

    def _build_ablation_replay_plan(
        self,
        layer: ExpertLayerState,
        specs: Sequence[tuple[str, str]],
    ) -> PlacementPlan:
        initial = self._layer_layout(layer)
        working = initial.clone()
        owners = layer.logical_to_physical.detach().cpu().clone()
        actions: list[PlacementAction] = []
        for kind, body in specs:
            if kind == "swap":
                lhs_text, rhs_text = body.split("<->", maxsplit=1)
                lhs, rhs = int(lhs_text), int(rhs_text)
                lhs_slot, rhs_slot = int(owners[lhs].item()), int(owners[rhs].item())
                if int(working[lhs_slot].item()) != lhs or int(working[rhs_slot].item()) != rhs:
                    raise RuntimeError(f"HierMoE ablation swap {body} does not match layer {layer.key}.")
                actions.append(PlacementAction("swap", lhs_slot, rhs_slot, lhs, rhs))
                working[lhs_slot], working[rhs_slot] = working[rhs_slot].clone(), working[lhs_slot].clone()
                owners[lhs], owners[rhs] = owners[rhs].clone(), owners[lhs].clone()
                continue
            if kind == "replica":
                logical_text, dst_text = body.split("->", maxsplit=1)
                logical, dst_slot = int(logical_text), int(dst_text)
                src_slot = int(owners[logical].item())
                previous = int(working[dst_slot].item())
                if src_slot == dst_slot or int(working[src_slot].item()) != logical:
                    raise RuntimeError(f"HierMoE ablation replica {body} has no valid owner source in {layer.key}.")
                actions.append(PlacementAction("replica", src_slot, dst_slot, logical, previous))
                working[dst_slot] = logical
                if previous >= 0 and int(owners[previous].item()) == dst_slot:
                    remaining = torch.nonzero(working == previous, as_tuple=False).flatten()
                    if remaining.numel() == 0:
                        raise RuntimeError(
                            f"HierMoE ablation replica {body} removes the final copy of victim "
                            f"expert {previous} in {layer.key}."
                        )
                    owners[previous] = int(remaining.min().item())
                continue
            logical_text, slot_text = body.split("@", maxsplit=1)
            logical, slot = int(logical_text), int(slot_text)
            if int(working[slot].item()) != logical or slot in {int(value) for value in owners.tolist()}:
                raise RuntimeError(f"HierMoE ablation empty action {body} is invalid for layer {layer.key}.")
            actions.append(PlacementAction("empty", slot, slot, logical, -1))
            working[slot] = -1

        zero_cost = self._zero_placement_cost()
        final_layout = tuple(int(value) for value in working.tolist())
        return PlacementPlan(
            actions=tuple(actions),
            initial_layout=tuple(int(value) for value in initial.tolist()),
            final_layout=final_layout,
            baseline_cost=zero_cost,
            final_cost=zero_cost,
            swap_rounds=sum(action.kind == "swap" for action in actions),
            replica_rounds=sum(action.kind == "replica" for action in actions),
            planning_ms=0.0,
            route_stats_ms=0.0,
            swap_ms=0.0,
            replica_ms=0.0,
            swap_score_ms=0.0,
            swap_update_ms=0.0,
            swap_collective_ms=0.0,
            replica_score_ms=0.0,
            replica_update_ms=0.0,
            replica_collective_ms=0.0,
            decision_sync_ms=0.0,
            finalization_ms=0.0,
            algorithm_version="hiermoe-ablation-replay-v1",
            layout_digest=f"{zlib.crc32(repr(final_layout).encode()):08x}",
            final_owner_slots=tuple(int(value) for value in owners.tolist()),
        )

    def _validate_ablation_final_layout(self) -> None:
        if set(self._ablation_expected_layouts) != set(self.layers):
            missing = sorted(set(self.layers) - set(self._ablation_expected_layouts))
            unexpected = sorted(set(self._ablation_expected_layouts) - set(self.layers))
            raise RuntimeError(f"HierMoE ablation layer mismatch: missing={missing}, unexpected={unexpected}.")
        for layer_key, layer in self.layers.items():
            actual = tuple(int(value) for value in self._layer_layout(layer).tolist())
            expected = self._ablation_expected_layouts[layer_key]
            if actual != expected:
                raise RuntimeError(f"HierMoE ablation final layout does not match replay output for {layer_key}.")
            expected_owners = self._ablation_expected_owner_slots.get(layer_key)
            if expected_owners is not None:
                actual_owners = tuple(int(value) for value in layer.logical_to_physical.tolist())
                if actual_owners != expected_owners:
                    raise RuntimeError(f"HierMoE ablation owner mapping does not match replay output for {layer_key}.")
            expected_source_lut = self._ablation_expected_source_luts.get(layer_key)
            if expected_source_lut is not None:
                if layer.source_logical_to_physical is None:
                    raise RuntimeError(f"HierMoE ablation source route LUT is missing for {layer_key}.")
                actual_source_lut = tuple(
                    tuple(int(value) for value in row) for row in layer.source_logical_to_physical.tolist()
                )
                if actual_source_lut != expected_source_lut:
                    raise RuntimeError(
                        f"HierMoE ablation source route LUT does not match replay output for {layer_key}."
                    )

    def _install_static_ablation_route_metadata(self) -> None:
        for layer_key, layer in self.layers.items():
            expected_owners = self._ablation_expected_owner_slots.get(layer_key)
            if expected_owners is not None:
                self._refresh_layer_mapping_from_slots(layer, expected_owners)
            expected_source_lut = self._ablation_expected_source_luts.get(layer_key)
            if expected_source_lut is None:
                continue
            source_lut = torch.tensor(expected_source_lut, dtype=torch.long)
            expected_shape = (self.ep_size, layer.num_experts)
            if tuple(source_lut.shape) != expected_shape:
                raise RuntimeError(
                    f"HierMoE ablation source route LUT for {layer_key} has shape "
                    f"{tuple(source_lut.shape)}, expected {expected_shape}."
                )
            if bool(((source_lut < 0) | (source_lut >= layer.num_physical_slots)).any().item()):
                raise RuntimeError(f"HierMoE ablation source route LUT references an invalid slot for {layer_key}.")
            layout = self._layer_layout(layer)
            logical = torch.arange(layer.num_experts, dtype=torch.long).view(1, -1)
            routed_logical = layout.index_select(0, source_lut.reshape(-1)).view_as(source_lut)
            if not torch.equal(routed_logical, logical.expand_as(routed_logical)):
                raise RuntimeError(f"HierMoE ablation source route LUT references the wrong expert for {layer_key}.")
            layer.source_logical_to_physical = source_lut
            layer._device_source_mapping_cache.clear()

    @torch.no_grad()
    def _install_static_ablation_layout(self) -> None:
        if self._initial_layout_path:
            for layer_key, layer in self.layers.items():
                expected_layout = self._ablation_expected_layouts[layer_key]
                validated_layout, _owners = self._validate_placement_layout(
                    layer,
                    expected_layout,
                    self._ablation_expected_owner_slots.get(layer_key),
                )
                layer.slot_to_logical = validated_layout
                layer.fixed_r2_layout = False
                layer.active_quota_policy = ()
                layer.pending_physical_routes = None
                layer.pending_route_data_ptr = 0
                layer.placement_version += 1
            self._install_static_ablation_route_metadata()
            self._validate_ablation_final_layout()
            logger.info_rank0(
                "HierMoE installed preloaded static placement metadata from %s without expert P2P.",
                self._initial_layout_path,
            )
            return

        canonical_empty = self._ablation_initial_layout == "canonical_empty"
        if canonical_empty:
            if _FIXED_R2_LAYOUT:
                raise RuntimeError("Canonical-empty static replay must not install the fixed R2 layout.")
            for layer in self.layers.values():
                expected = _initial_slot_to_logical(
                    layer.num_experts,
                    layer.base_num_local_experts,
                    layer.num_local_experts,
                    self.ep_size,
                )
                if not torch.equal(self._layer_layout(layer), expected):
                    raise RuntimeError(
                        f"Canonical-empty static replay has a non-canonical initial {layer.key} layout."
                    )
        elif not _FIXED_R2_LAYOUT:
            raise RuntimeError("Static HierMoE ablation replay requires VEOMNI_HIERMOE_FIXED_R2_LAYOUT=1.")
        # A static replay already knows the final layout before training
        # starts.  Replaying one action wave at a time needlessly serializes
        # hundreds of expert transfers and can leave the first FSDP
        # collective queued behind minutes of device copies.  Concatenate all
        # recorded actions per layer, let ``_build_ablation_replay_plan``
        # resolve every final slot back to an original state source, and
        # materialize that layer with one sparse transfer plan.  The final
        # owner/source-LUT metadata is installed below, so static replay does
        # not need the online Cover path's incremental LUT patches.
        specs_by_layer: dict[str, list[tuple[str, str]]] = {layer_key: [] for layer_key in self.layers}
        for step in sorted(self._ablation_actions_by_step):
            by_layer = self._ablation_actions_by_step[step]
            unexpected = set(by_layer) - set(self.layers)
            if unexpected:
                raise RuntimeError(
                    f"HierMoE ablation replay step {step} contains unknown layers: {sorted(unexpected)}."
                )
            for layer_key in self.layers:
                specs_by_layer[layer_key].extend(by_layer.get(layer_key, ()))

        action_count = 0
        for layer_key, layer in self.layers.items():
            plan = self._build_ablation_replay_plan(layer, specs_by_layer[layer_key])
            self._execute_placement_plan(
                layer,
                plan,
                timing_prefix=None,
                transfer_group=self.ep_group,
                force_staged_transfer=False,
                fast_sparse_transfer=True,
            )
            # ``ProcessGroupHCCL::Work.wait`` guarantees that the P2P work was
            # submitted, but the destination-slot copies consuming the shared
            # receive staging buffer may still be queued on the default
            # stream.  Static installation immediately reuses that staging
            # buffer for the next layer; without a device fence dozens of
            # recv/copy waves can accumulate ahead of the first FSDP
            # collective and eventually hit HCCL's dispatch timeout.  This is
            # startup-only work, so finish each layer before reusing the
            # manager-wide staging pool.
            synchronize()
            action_count += len(plan.actions)
        self._install_static_ablation_route_metadata()
        self._validate_ablation_final_layout()
        logger.info_rank0(
            "HierMoE installed static ablation layout from %s using %s replayed actions.",
            _ABLATION_REPLAY_PATH,
            action_count,
        )

    def _queue_ablation_replay_step(self, step: int) -> str:
        if self._ablation_replay_mode == "static":
            self.latest_pair = "none"
            return self.latest_pair
        # ``maybe_swap`` receives the zero-based optimizer-step index while
        # action logs use the one-based training-step number shown in metrics.
        logged_step = int(step) + 1
        by_layer = self._ablation_actions_by_step.get(logged_step)
        if by_layer is None:
            if logged_step > max(self._ablation_actions_by_step):
                self._validate_ablation_final_layout()
            self.latest_pair = "none"
            return self.latest_pair
        if set(by_layer) != set(self.layers):
            raise RuntimeError(f"HierMoE ablation replay step {logged_step} does not contain every registered layer.")

        committed: list[str] = []
        for layer_key in self.layers:
            if layer_key in self._pipeline_pending_plans:
                raise RuntimeError(f"HierMoE ablation replay has an unconsumed plan for {layer_key}.")
            layer = self.layers[layer_key]
            plan = self._build_ablation_replay_plan(layer, by_layer[layer_key])
            self._pipeline_pending_plans[layer_key] = _PendingPipelinePlan(
                plan=plan,
                source_step=int(step),
                placement_version=int(layer.placement_version),
            )
            committed.extend(f"{layer_key}:{action.format()}" for action in plan.actions)
        self._accumulate_metric("hiermoe/ablation_replay_actions", len(committed))
        self._accumulate_metric("hiermoe/ablation_replay_logged_step", logged_step)
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    def _ensure_pipeline_plan_worker_capacity(self) -> None:
        """Guarantee that every gate-blocked layer owns a planner worker.

        Prepare spans forward and backward windows. A bounded executor smaller
        than the number of MoE layers can otherwise fill with forward-order
        tasks while the reverse-order task needed by backward remains queued.
        """

        if not self.fixed_pipeline_overlap or self._pipeline_shutdown:
            return
        required = max(1, _PIPELINE_PLAN_WORKERS, len(self.layers))
        with self._pipeline_lock:
            executor = self._pipeline_plan_executor
            if executor is None or self._pipeline_plan_worker_capacity >= required:
                return
            if self._pipeline_plan_futures:
                raise RuntimeError("Cannot resize the HierMoE planner executor while planner jobs are active.")
            self._pipeline_plan_executor = ThreadPoolExecutor(
                max_workers=required,
                thread_name_prefix="hiermoe-plan",
            )
            self._pipeline_plan_worker_capacity = required
        executor.shutdown(wait=True, cancel_futures=False)

    def placement_planning_enabled(self) -> bool:
        return self._online_lut_update or (
            not self._initial_layout_path
            and self._ablation_replay_mode != "static"
            and (self.expert_swap_max_pairs_per_layer > 0 or self.redundant_slot_increment_per_device > 0)
        )

    def layer_calibration_enabled(self) -> bool:
        if self._cost_model_verify:
            return not self._cost_model_verify_complete
        if self._forward_reuse_cover:
            return False
        if self.expert_swap_selector == "current_joint":
            return True
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            return any(layer.planner_calibration is None for layer in self.layers.values())
        return False

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
            "hiermoe/cost_model_verify": int(self._cost_model_verify),
            "hiermoe/expert_swap_selector": self.expert_swap_selector,
            "hiermoe/fixed_pipeline_overlap": int(self.fixed_pipeline_overlap),
            "hiermoe/gradient_overlap_enabled": int(self.gradient_overlap_enabled),
            "hiermoe/cpu_planner_mode": self._cpu_planner_mode,
            "hiermoe/cpu_training_affinity_cores": len(self._cpu_training_affinity),
            "hiermoe/cpu_planner_affinity_cores": len(self._cpu_planner_affinity),
            "hiermoe/ablation_replay_mode": self._ablation_replay_mode,
            "hiermoe/ablation_migration_mode": self._ablation_migration_mode,
            "hiermoe/ablation_grad_mode": self._ablation_grad_mode,
            "hiermoe/online_freeze_cost_mode": self._online_freeze_cost_mode,
            "hiermoe/online_freeze_inter_ms_per_byte": self._online_freeze_inter_ms_per_byte,
            "hiermoe/online_freeze_intra_ms_per_byte": self._online_freeze_intra_ms_per_byte,
            "hiermoe/online_freeze_route_ms_per_assignment": self._online_freeze_route_ms_per_assignment,
            "hiermoe/online_freeze_traffic_intercept_ms": self._online_freeze_traffic_intercept_ms,
            "hiermoe/fixed_r2_mirrored_remap": int(_FORCE_FIXED_R2_MIRRORED_REMAP),
            "hiermoe/online_lut_update": int(self._online_lut_update),
            "hiermoe/online_lut_start_step": self._online_lut_start_step,
            "hiermoe/online_lut_min_gain": self._online_lut_min_gain,
            "placemoe/hot_update_enabled": int(self._hot_update),
            "placemoe/cpu_affinity_automatic": int(self._hot_update_affinity_automatic),
            "placemoe/planner_physical_cores": self._hot_update_planner_physical_cores,
            "placemoe/planner_workers": self._hot_update_resources.workers,
            "placemoe/planner_candidate_workers": self._hot_update_resources.candidate_workers,
            "placemoe/planner_worker_threads": self._hot_update_resources.worker_threads,
            "placemoe/layout_interval_steps": self._hot_update_layout_interval,
            "placemoe/mapping_interval_steps": self._hot_update_mapping_interval,
            "placemoe/hot_update_running": int(self._hot_update_controller.active_job is not None),
            "placemoe/layout_updates": self._hot_update_layout_updates,
            "placemoe/mapping_updates": self._hot_update_mapping_updates,
            "placemoe/last_source_step": self._hot_update_last_source_step,
            "placemoe/last_apply_step": self._hot_update_last_apply_step,
            "placemoe/last_staleness_steps": self._hot_update_last_staleness_steps,
            "placemoe/last_snapshot_ms": self._hot_update_last_snapshot_ms,
            "placemoe/last_planner_ms": self._hot_update_last_planner_ms,
            "placemoe/last_migration_ms": self._hot_update_last_migration_ms,
            "placemoe/last_moved_slots": self._hot_update_last_moved_slots,
            "hiermoe/forward_reuse_cover": int(self._forward_reuse_cover),
            "hiermoe/forward_reuse_cover_patch_remap": int(self._forward_reuse_cover_patch_remap),
            "hiermoe/forward_reuse_cover_fast": int(self._forward_reuse_cover_fast),
            "hiermoe/forward_reuse_cover_compute_weight": self._forward_reuse_cover_compute_weight,
            "hiermoe/forward_reuse_cover_compute_ms_per_assignment": (
                self._forward_reuse_cover_compute_ms_per_assignment
            ),
            "hiermoe/forward_reuse_cover_min_gain": self._forward_reuse_cover_min_gain,
            "hiermoe/forward_reuse_cover_rounds": self._forward_reuse_cover_rounds,
            "hiermoe/forward_reuse_cover_only_step": self._forward_reuse_cover_only_step,
            "hiermoe/forward_reuse_cover_victim_mode": self._forward_reuse_cover_victim_mode,
            "hiermoe/forward_reuse_cover_service_scope": self._forward_reuse_cover_service_scope,
            "hiermoe/forward_reuse_cover_service_group_size": self._forward_reuse_cover_service_group_size,
            "hiermoe/forward_reuse_cover_aggregate_service_group": int(
                self._forward_reuse_cover_aggregate_service_group
            ),
            "hiermoe/forward_reuse_cover_proposal_topk": self._forward_reuse_cover_proposal_topk,
            "hiermoe/forward_reuse_cover_empty_seeding": int(self._forward_reuse_cover_empty_seeding),
            "hiermoe/forward_reuse_cover_confirm_samples": self._forward_reuse_cover_confirm_samples,
            "hiermoe/forward_reuse_cover_pending": len(self._forward_reuse_cover_pending),
            "hiermoe/pipeline_planner_backend": (
                self._planner_collective_backend(self._pipeline_planner_group)
                if self._pipeline_planner_group is not None
                else "none"
            ),
        }

    def _accumulate_metric(self, key: str, value: float | int | str) -> None:
        if isinstance(value, str):
            self._placement_metrics[key] = value
        elif isinstance(value, int):
            self._placement_metrics[key] = int(self._placement_metrics.get(key, 0)) + value
        else:
            self._placement_metrics[key] = float(self._placement_metrics.get(key, 0.0)) + float(value)

    def _pipeline_device(self, layer: ExpertLayerState) -> torch.device:
        return _local_tensor_view(layer.primary_parameter).device

    def _pipeline_stream(self, kind: str, device: torch.device) -> Any | None:
        if device.type == "cpu":
            return None
        key = (str(kind), device)
        cached = self._pipeline_streams.get(key)
        if cached is not None:
            return cached
        device_api = get_torch_device()
        device_api.set_device(device)
        try:
            cached = device_api.Stream(device=device)
        except TypeError:
            cached = device_api.Stream()
        self._pipeline_streams[key] = cached
        return cached

    @staticmethod
    def _pipeline_ready_event(device: torch.device) -> Any | None:
        if device.type == "cpu":
            return None
        device_api = get_torch_device()
        try:
            current_stream = device_api.current_stream(device)
        except TypeError:
            current_stream = device_api.current_stream()
        event = device_api.Event()
        event.record(current_stream)
        return event

    def _pipeline_stage_event(self) -> AcceleratorEvent | None:
        return record_accelerator_event() if (_PIPELINE_STAGE_TIMING or self.fixed_pipeline_overlap) else None

    def _run_pipeline_stream_task(
        self,
        kind: str,
        device: torch.device,
        ready_event: Any | None,
        task: Any,
    ) -> Any:
        if device.type == "cpu":
            return task()
        device_api = get_torch_device()
        device_api.set_device(device)
        stream = self._pipeline_stream(kind, device)
        assert stream is not None
        with device_api.stream(stream):
            if ready_event is not None:
                ready_events = ready_event if isinstance(ready_event, tuple) else (ready_event,)
                for event in ready_events:
                    if event is not None:
                        stream.wait_event(event)
            result = task()
        stream.synchronize()
        return result

    def configure_pipeline_microstep(self, step: int, micro_step: int, num_micro_steps: int) -> None:
        """Advance placement and gradient-overlap state at a microbatch boundary."""

        if not self.fixed_pipeline_overlap and not self.gradient_overlap_enabled:
            return
        self._pipeline_step = int(step)
        self._pipeline_micro_step = int(micro_step)
        self._pipeline_num_micro_steps = max(1, int(num_micro_steps))
        if int(micro_step) != 0:
            return
        self._debug_log_redundant_copy_stats("step_begin", include_grads=False)
        self._begin_metrics_step(step)
        with self._pipeline_lock:
            if self._pipeline_grad_futures:
                raise RuntimeError("HierMoE started a new step before redundant gradient synchronization completed.")
            self._pipeline_grad_ready.clear()
            self._pipeline_grad_ready_events.clear()
            self._pipeline_grad_dispatch_complete.clear()
            self._pipeline_grad_window_waited.clear()
            self._pipeline_grad_comm_blocked = False
            self._pipeline_grad_window_exposed_ms = 0.0
            self._pipeline_layer_order = tuple(self.layers)
            self._pipeline_next_migration_index = 0
            self._pipeline_next_grad_index = 0
        if not self.fixed_pipeline_overlap:
            return
        if self._uses_cpu_process_planner():
            self._ensure_cpu_process_runtime()
        if self._ablation_replay_mode == "off" and not self._npu_layer_owner_blocking:
            self._ensure_pipeline_plan_worker_capacity()
        if self._ablation_migration_mode == "hidden":
            self._launch_next_pipeline_migration()

    def _pipeline_is_final_microstep(self) -> bool:
        return self._pipeline_micro_step + 1 >= self._pipeline_num_micro_steps

    def _wait_pipeline_host_event(self, layer_key: str, event: Event) -> None:
        while not event.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
            with self._pipeline_lock:
                future = self._pipeline_plan_futures.get(layer_key)
            if future is not None and future.done():
                future.result()
            if self._pipeline_shutdown:
                return

    def _wait_pipeline_prepare_start(self, layer_key: str) -> None:
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
        self._wait_pipeline_host_event(layer_key, windows.prepare_gates[0])

    def _complete_pipeline_prepare_stage(
        self,
        layer_key: str,
        completed_stages: int,
    ) -> bool:
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
            first_window = windows.prepare_next_window
            next_window = first_window
            while next_window < len(_PIPELINE_PREPARE_CUT_POINTS) and _PIPELINE_PREPARE_CUT_POINTS[next_window] <= int(
                completed_stages
            ):
                next_window += 1
            if next_window == first_window:
                return False
            done_event = record_accelerator_event()
            for window_index in range(first_window, next_window):
                windows.prepare_done_events[window_index] = done_event
            windows.prepare_next_window = next_window
            enqueued = windows.prepare_enqueued[first_window:next_window]
            next_gate = windows.prepare_gates[next_window] if next_window < len(_PIPELINE_PREPARE_CUT_POINTS) else None
        for event in enqueued:
            event.set()
        if next_gate is None:
            return False
        self._wait_pipeline_host_event(layer_key, next_gate)
        return True

    def open_pipeline_planner_prepare_window(self, layer_key: str, window_index: int) -> None:
        if not self.fixed_pipeline_overlap:
            return
        if not 0 <= int(window_index) < len(_PIPELINE_PREPARE_CUT_POINTS):
            raise ValueError(f"Invalid pipeline Prepare window index: {window_index}")
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is not None:
            windows.prepare_gates[int(window_index)].set()

    def release_pipeline_planner_prepare(self, layer_key: str) -> None:
        """Release all Prepare gates for isolated planner tests and shutdown fallbacks."""

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is not None:
            for gate in windows.prepare_gates:
                gate.set()

    def close_pipeline_planner_prepare_window(self, layer_key: str, window_index: int) -> None:
        if not self.fixed_pipeline_overlap:
            return
        index = int(window_index)
        if not 0 <= index < len(_PIPELINE_PREPARE_CUT_POINTS):
            raise ValueError(f"Invalid pipeline Prepare window index: {window_index}")
        a2a_end_event = record_accelerator_event()
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        host_wait_started = time.perf_counter()
        self._wait_pipeline_host_event(layer_key, windows.prepare_enqueued[index])
        host_wait_ms = (time.perf_counter() - host_wait_started) * 1000.0
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_host_gate_wait_ms", host_wait_ms)
        self._accumulate_metric(
            f"hiermoe/pipeline_planner_prepare_window_{index}_host_gate_wait_ms",
            host_wait_ms,
        )
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            windows.prepare_a2a_end_events[index] = a2a_end_event
            planner_done_event = windows.prepare_done_events[index]
        if planner_done_event is None or planner_done_event.event is None:
            return
        layer = self.layers.get(layer_key)
        if layer is None:
            return
        device = self._pipeline_device(layer)
        if device.type == "cpu":
            return
        device_api = get_torch_device()
        try:
            current_stream = device_api.current_stream(device)
        except TypeError:
            current_stream = device_api.current_stream()
        current_stream.wait_event(planner_done_event.event)

    @staticmethod
    def _pipeline_prepare_exposure_ms(windows: _PipelinePlannerWindows) -> tuple[float, tuple[float, ...]]:
        per_window = []
        for a2a_end, planner_done in zip(
            windows.prepare_a2a_end_events,
            windows.prepare_done_events,
            strict=True,
        ):
            if a2a_end is None or planner_done is None:
                per_window.append(0.0)
            else:
                per_window.append(max(0.0, a2a_end.elapsed_time(planner_done)))
        return sum(per_window), tuple(per_window)

    @staticmethod
    def _pipeline_stage_exposure_ms(
        deadline: AcceleratorEvent | None,
        done: AcceleratorEvent | None,
    ) -> float:
        if deadline is None or done is None:
            return 0.0
        return max(0.0, deadline.elapsed_time(done))

    def _uses_cpu_process_planner(self) -> bool:
        return self._cpu_planner_mode in {"process_blocking", "process_background"}

    @staticmethod
    def _bind_all_process_threads(cpu_ids: Sequence[int]) -> None:
        cpus = {int(cpu_id) for cpu_id in cpu_ids}
        if not cpus or not hasattr(os, "sched_setaffinity"):
            return
        task_root = "/proc/self/task"
        try:
            task_ids = tuple(int(name) for name in os.listdir(task_root) if name.isdigit())
        except OSError:
            task_ids = (0,)
        for task_id in task_ids:
            try:
                os.sched_setaffinity(task_id, cpus)
            except (OSError, ProcessLookupError):
                continue
        os.sched_setaffinity(0, cpus)

    def _configure_hot_update_training_affinity(self) -> None:
        if not self._hot_update:
            return
        node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
        if node_rank != 0:
            return
        plan = resolve_cpu_affinity(_HOT_UPDATE_RESOURCES)
        self._bind_all_process_threads(plan.training_cpu_ids)
        self._cpu_training_affinity = plan.training_cpu_ids
        self._cpu_planner_affinity = plan.planner_cpu_ids
        self._hot_update_resources = plan.planner_resources()
        self._hot_update_affinity_automatic = plan.automatic
        self._hot_update_planner_physical_cores = plan.planner_physical_cores
        logger.info_rank0(
            "PlaceMoE isolated hot-update CPU planner mode=%s training_cpus=%s planner_cpus=%s "
            "planner_physical_cores=%s workers=%s candidate_workers=%s worker_threads=%s.",
            "auto" if plan.automatic else "explicit",
            self._hot_update_resources.training_cpu_ids,
            self._hot_update_resources.planner_cpu_ids,
            plan.planner_physical_cores,
            plan.workers,
            plan.candidate_workers,
            plan.worker_threads,
        )

    def _cpu_process_affinity_masks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        visible = sorted(os.sched_getaffinity(0))
        local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.ep_rank % local_world_size)))
        start = (len(visible) * local_rank) // local_world_size
        end = (len(visible) * (local_rank + 1)) // local_world_size
        rank_cpus = tuple(visible[start:end])
        if len(rank_cpus) < 2:
            return rank_cpus, rank_cpus
        training_count = min(max(1, _CPU_TRAIN_CORES_PER_RANK), len(rank_cpus) - 1)
        return rank_cpus[:training_count], rank_cpus[training_count:]

    def _ensure_cpu_process_runtime(self) -> Any:
        if not self._uses_cpu_process_planner():
            raise RuntimeError("CPU planner process requested for a non-process planner mode.")
        if self._cpu_process_runtime is not None:
            return self._cpu_process_runtime
        from .cpu_planner import SharedMemoryCPUPlannerProcess

        training_cpus, planner_cpus = self._cpu_process_affinity_masks()
        if not training_cpus or not planner_cpus:
            raise RuntimeError("CPU planner process isolation requires at least two visible CPU cores per local rank.")
        # Bind every currently existing training/HCCL/PyTorch thread. New
        # threads inherit the calling thread's training mask. The spawned
        # planner immediately switches to the disjoint planner mask.
        self._bind_all_process_threads(training_cpus)
        runtime = SharedMemoryCPUPlannerProcess(planner_cpu_ids=planner_cpus)
        self._cpu_training_affinity = training_cpus
        self._cpu_planner_affinity = planner_cpus
        self._cpu_process_runtime = runtime
        self._placement_metrics["hiermoe/cpu_training_affinity_cores"] = len(training_cpus)
        self._placement_metrics["hiermoe/cpu_planner_affinity_cores"] = len(planner_cpus)
        logger.info_rank0(
            "HierMoE isolated CPU planner process pid=%s with training cores=%s planner cores=%s.",
            runtime.pid,
            training_cpus,
            planner_cpus,
        )
        return runtime

    def _cpu_exact_planner_for_layer(self, layer: ExpertLayerState) -> GreedyCommunicationPlanner:
        """Build the same exact scorer as the fixed-pipeline NPU backend on CPU."""

        return GreedyCommunicationPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layer.latest_hidden_size,
            bytes_per_element=layer.latest_bytes_per_element,
            slots_per_rank=layer.num_local_experts,
            communication_scale=1.0,
            forward_compute_per_assignment=0.0,
            forward_compute_constant=0.0,
            smooth_max_gamma=self.smooth_max_gamma,
            reducer=None,
            candidate_chunk_size=_SWAP_COST_CHUNK_CANDIDATES,
            process_group=None,
            max_copies=self.greedy_max_copies_per_expert,
            assume_unique_routes=True,
            layer_parallel_streams=_GREEDY_LAYER_PARALLEL_STREAMS,
            adaptive_topk=False,
            adaptive_topk_initial=_GREEDY_ADAPTIVE_TOPK_INITIAL,
            adaptive_topk_strict_certificate=False,
            exact_primitive_topk=0,
            post_shortlist_compact_pair=False,
            exact_primitive_max_only=False,
        )

    def _cpu_batched_plan_layers(self, step: int) -> tuple[ExpertLayerState, ...]:
        order = self._pipeline_layer_order or tuple(self.layers)
        layers = tuple(self.layers[key] for key in order)
        if not layers:
            return ()
        if any(layer.latest_selected_experts is None or layer.latest_route_step != int(step) for layer in layers):
            return ()
        if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in layers):
            return ()
        signature = {
            (
                layer.latest_hidden_size,
                layer.latest_bytes_per_element,
                layer.num_local_experts,
                layer.num_experts,
            )
            for layer in layers
        }
        if len(signature) != 1:
            raise RuntimeError(
                "The 48-layer CPU/HCCL planner requires identical route and expert shapes across planned layers."
            )
        return layers

    @staticmethod
    def _synchronize_pipeline_event(event: Any | None) -> None:
        if event is None:
            return
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def _cpu_hccl_reduce(
        self,
        state: _CPUBatchedPlanState,
        local_payload: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Stage one packed CPU payload through the existing EP/HCCL group."""

        state.collective_ready.set()
        if state.background:
            gate_started = time.perf_counter()
            while not state.collective_gate.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
                if self._pipeline_shutdown:
                    raise RuntimeError("CPU planner was shut down before its HCCL window opened.")
            state.collective_gate_wait_ms = (time.perf_counter() - gate_started) * 1000.0

        if device.type == "cpu" or self._pipeline_planner_group is None or self.ep_size <= 1:
            state.collective_enqueued.set()
            return local_payload

        device_api = get_torch_device()
        device_api.set_device(device)
        stream = self._pipeline_stream("planner", device)
        if stream is None:
            raise RuntimeError("CPU planner HCCL reduction requires a device planner stream.")
        try:
            with device_api.stream(stream):
                device_payload = local_payload.to(device=device, non_blocking=True)
                dist.all_reduce(
                    device_payload,
                    op=dist.ReduceOp.SUM,
                    group=self._pipeline_planner_group,
                )
                done_event = self._pipeline_stage_event()
                state.collective_done_event = done_event
            state.collective_enqueued.set()
            if done_event is not None:
                self._synchronize_pipeline_event(done_event.event)
            else:
                stream.synchronize()
            return device_payload.to(device="cpu")
        except BaseException as error:
            state.collective_error = error
            state.collective_enqueued.set()
            raise

    @torch.no_grad()
    def _cpu_batched_plan_worker(
        self,
        state: _CPUBatchedPlanState,
        layers: tuple[ExpertLayerState, ...],
        routes: tuple[torch.Tensor, ...],
        layouts: tuple[torch.Tensor, ...],
        owners: tuple[torch.Tensor, ...],
        route_ready_event: Any | None,
    ) -> _CPUBatchedPlanResult:
        from .cpu_planner import CPUHCCLBatchedPlanner

        self._synchronize_pipeline_event(route_ready_event)
        route_copy_started = time.perf_counter()
        cpu_routes = tuple(route.detach().to(device="cpu") for route in routes)
        route_copy_ms = (time.perf_counter() - route_copy_started) * 1000.0
        device = routes[0].device
        planner = CPUHCCLBatchedPlanner(
            self._cpu_exact_planner_for_layer(layers[0]),
            reducer=lambda tensor: self._cpu_hccl_reduce(state, tensor, device),
        )
        result = planner.plan_layers(
            cpu_routes,
            layouts,
            owners,
            source_ranks=self.ep_rank,
            max_swaps=self.expert_swap_max_pairs_per_layer,
            max_replicas=self.max_replica_rounds,
            layer_seeds=[zlib.crc32(layer.key.encode("utf-8")) for layer in layers],
            step=state.source_step,
            communication_scales=[1.0] * len(layers),
            forward_compute_per_assignment=[0.0] * len(layers),
            forward_compute_constant=[0.0] * len(layers),
        )
        latency_ms = (time.perf_counter() - state.submitted_at) * 1000.0
        active_ms = max(
            0.0,
            route_copy_ms + result.timing.total_ms - state.collective_gate_wait_ms,
        )
        return _CPUBatchedPlanResult(
            source_step=state.source_step,
            placement_versions=state.placement_versions,
            plans=result.plans,
            route_copy_ms=route_copy_ms,
            active_ms=active_ms,
            latency_ms=latency_ms,
            timing=result.timing,
        )

    def _submit_cpu_batched_plan(self, step: int) -> None:
        """Start the 48-layer host job after the final forward route is available."""

        if self._cpu_planner_mode != "background" or self._pipeline_shutdown:
            return
        if not self._pipeline_is_final_microstep():
            return
        if self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            return
        layers = self._cpu_batched_plan_layers(step)
        if not layers:
            return
        executor = self._cpu_plan_executor
        if executor is None:
            raise RuntimeError("Background CPU planner executor is unavailable.")
        with self._pipeline_lock:
            if self._cpu_batch_state is not None:
                if self._cpu_batch_state.source_step == int(step):
                    return
                raise RuntimeError("A previous CPU planner job was not consumed before the next submission.")
            if any(layer.last_planned_step == int(step) for layer in layers):
                return
            for layer in layers:
                layer.last_planned_step = int(step)
            state = _CPUBatchedPlanState(
                source_step=int(step),
                placement_versions=tuple(int(layer.placement_version) for layer in layers),
                submitted_at=time.perf_counter(),
                background=True,
            )
            self._cpu_batch_state = state
        routes = tuple(layer.latest_selected_experts.detach() for layer in layers)  # type: ignore[union-attr]
        layouts = tuple(self._layer_layout(layer).clone() for layer in layers)
        owners = tuple(layer.logical_to_physical.detach().cpu().clone() for layer in layers)
        route_ready_event = self._pipeline_ready_event(routes[-1].device)
        future = executor.submit(
            self._cpu_batched_plan_worker,
            state,
            layers,
            routes,
            layouts,
            owners,
            route_ready_event,
        )
        state.future = future

    def _submit_cpu_process_plan(self, step: int, *, background: bool) -> _CPUBatchedPlanState | None:
        """Copy routes once into shared memory and enqueue the isolated process."""

        if self._pipeline_shutdown:
            return None
        if not self._pipeline_is_final_microstep():
            return None
        if self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            return None
        layers = self._cpu_batched_plan_layers(step)
        if not layers:
            return None
        runtime = self._ensure_cpu_process_runtime()
        with self._pipeline_lock:
            if self._cpu_batch_state is not None:
                if self._cpu_batch_state.source_step == int(step):
                    return self._cpu_batch_state
                raise RuntimeError("A previous CPU planner process job was not consumed.")
            if any(layer.last_planned_step == int(step) for layer in layers):
                return None
            for layer in layers:
                layer.last_planned_step = int(step)
            state = _CPUBatchedPlanState(
                source_step=int(step),
                placement_versions=tuple(int(layer.placement_version) for layer in layers),
                submitted_at=time.perf_counter(),
                background=bool(background),
                process_slot=int(step) % 2,
            )
            self._cpu_batch_state = state

        share_started = time.perf_counter()
        shared_routes = tuple(
            runtime.share_cpu_tensor(layer.latest_selected_experts)  # type: ignore[arg-type]
            for layer in layers
        )
        state.route_share_ms = (time.perf_counter() - share_started) * 1000.0
        runtime.submit(
            slot=state.process_slot,
            source_step=state.source_step,
            planner=self._cpu_exact_planner_for_layer(layers[0]),
            selected_experts=shared_routes,
            slot_to_logical=tuple(self._layer_layout(layer).clone() for layer in layers),
            owner_slots=tuple(layer.logical_to_physical.detach().cpu().clone() for layer in layers),
            source_rank=self.ep_rank,
            max_swaps=self.expert_swap_max_pairs_per_layer,
            max_replicas=self.max_replica_rounds,
            layer_seeds=tuple(zlib.crc32(layer.key.encode("utf-8")) for layer in layers),
            communication_scales=(1.0,) * len(layers),
            compute_slopes=(0.0,) * len(layers),
            compute_constants=(0.0,) * len(layers),
        )
        return state

    def _service_cpu_process_collective(self, state: _CPUBatchedPlanState) -> None:
        """Execute only the isolated child's packed reduction in the parent."""

        runtime = self._ensure_cpu_process_runtime()
        request_wait_started = time.perf_counter()
        collective_started: float | None = None
        try:
            source_step, local_payload = runtime.wait_collective(state.process_slot)
            state.collective_ready_host_wait_ms += (time.perf_counter() - request_wait_started) * 1000.0
            state.collective_ready.set()
            collective_started = time.perf_counter()
            if source_step != state.source_step:
                raise RuntimeError(
                    f"CPU planner collective source step {source_step} does not match {state.source_step}."
                )
            order = self._pipeline_layer_order or tuple(self.layers)
            device = self._pipeline_device(self.layers[order[0]])
            if device.type == "cpu" or self._pipeline_planner_group is None or self.ep_size <= 1:
                state.collective_enqueued.set()
            else:
                device_api = get_torch_device()
                device_api.set_device(device)
                stream = self._pipeline_stream("planner", device)
                if stream is None:
                    raise RuntimeError("CPU planner process reduction requires a device planner stream.")
                with device_api.stream(stream):
                    device_payload = local_payload.to(device=device, non_blocking=True)
                    dist.all_reduce(
                        device_payload,
                        op=dist.ReduceOp.SUM,
                        group=self._pipeline_planner_group,
                    )
                    done_event = self._pipeline_stage_event()
                    state.collective_done_event = done_event
                state.collective_enqueued.set()
                if done_event is not None:
                    self._synchronize_pipeline_event(done_event.event)
                else:
                    stream.synchronize()
                local_payload.copy_(device_payload.to(device="cpu"))
            runtime.complete_collective(state.process_slot)
        except BaseException as error:
            state.collective_error = error
            state.collective_enqueued.set()
            runtime.complete_collective(state.process_slot)
            raise
        finally:
            if collective_started is not None:
                state.process_collective_active_ms = (time.perf_counter() - collective_started) * 1000.0

    def _cpu_process_result(self, state: _CPUBatchedPlanState) -> _CPUBatchedPlanResult:
        runtime = self._ensure_cpu_process_runtime()
        completion = runtime.wait_result(state.process_slot)
        if completion.source_step != state.source_step or completion.result is None:
            raise RuntimeError("CPU planner process returned a stale or empty result.")
        timing = completion.result.timing
        active_ms = max(
            0.0,
            state.route_share_ms
            + timing.total_ms
            - timing.statistic_collective_ms
            + state.process_collective_active_ms,
        )
        return _CPUBatchedPlanResult(
            source_step=state.source_step,
            placement_versions=state.placement_versions,
            plans=completion.result.plans,
            route_copy_ms=state.route_share_ms,
            active_ms=active_ms,
            latency_ms=(time.perf_counter() - state.submitted_at) * 1000.0,
            timing=timing,
        )

    def _collect_cpu_process_plan(self, step: int) -> str:
        blocking = self._cpu_planner_mode == "process_blocking"
        blocking_started = time.perf_counter()
        with self._pipeline_lock:
            state = self._cpu_batch_state
        if state is None and blocking:
            state = self._submit_cpu_process_plan(step, background=False)
        if state is None or state.source_step != int(step):
            self.latest_pair = "none"
            return self.latest_pair

        wait_started = time.perf_counter()
        if state.process_collective_future is None:
            self._service_cpu_process_collective(state)
        else:
            state.process_collective_future.result()
        result = self._cpu_process_result(state)
        deadline_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        exposed_ms = (
            (time.perf_counter() - blocking_started) * 1000.0
            if blocking
            else state.route_share_ms + state.collective_close_host_wait_ms + deadline_wait_ms
        )
        with self._pipeline_lock:
            if self._cpu_batch_state is state:
                self._cpu_batch_state = None
        return self._accept_cpu_batched_result(state, result, exposed_ms=exposed_ms)

    def _run_blocking_cpu_batched_plan(
        self,
        step: int,
        layers: tuple[ExpertLayerState, ...],
    ) -> tuple[_CPUBatchedPlanState, _CPUBatchedPlanResult]:
        state = _CPUBatchedPlanState(
            source_step=int(step),
            placement_versions=tuple(int(layer.placement_version) for layer in layers),
            submitted_at=time.perf_counter(),
            background=False,
        )
        state.collective_gate.set()
        routes = tuple(layer.latest_selected_experts.detach() for layer in layers)  # type: ignore[union-attr]
        result = self._cpu_batched_plan_worker(
            state,
            layers,
            routes,
            tuple(self._layer_layout(layer).clone() for layer in layers),
            tuple(layer.logical_to_physical.detach().cpu().clone() for layer in layers),
            self._pipeline_ready_event(routes[-1].device),
        )
        return state, result

    def _accept_cpu_batched_result(
        self,
        state: _CPUBatchedPlanState,
        result: _CPUBatchedPlanResult,
        *,
        exposed_ms: float,
    ) -> str:
        order = self._pipeline_layer_order or tuple(self.layers)
        if len(order) != len(result.plans):
            raise RuntimeError("CPU planner returned a different number of layer plans.")
        committed: list[str] = []
        accepted = 0
        for index, (layer_key, plan) in enumerate(zip(order, result.plans, strict=True)):
            layer = self.layers[layer_key]
            layer.last_plan = plan
            self._record_plan_metrics(plan)
            if result.source_step != state.source_step or result.placement_versions[index] != int(
                layer.placement_version
            ):
                self._accumulate_metric("hiermoe/cpu_planner_stale", 1)
            elif plan.actions:
                self._pipeline_pending_plans[layer_key] = _PendingPipelinePlan(
                    source_step=result.source_step,
                    placement_version=result.placement_versions[index],
                    plan=plan,
                )
                committed.extend(f"{layer_key}:{action.format()}" for action in plan.actions)
                accepted += 1

        timing = result.timing
        collective_active_ms = (
            state.process_collective_active_ms
            if self._uses_cpu_process_planner()
            else max(
                0.0,
                float(timing.statistic_collective_ms) - state.collective_gate_wait_ms,
            )
        )
        self._accumulate_metric("hiermoe/cpu_planner_jobs", 1)
        self._accumulate_metric("hiermoe/cpu_planner_layers", len(result.plans))
        self._accumulate_metric("hiermoe/cpu_planner_accepted", accepted)
        self._accumulate_metric("hiermoe/cpu_planner_route_copy_ms", result.route_copy_ms)
        self._accumulate_metric("hiermoe/cpu_planner_active_ms", result.active_ms)
        self._accumulate_metric("hiermoe/cpu_planner_latency_ms", result.latency_ms)
        self._accumulate_metric("hiermoe/cpu_planner_exposed_ms", exposed_ms)
        self._accumulate_metric("hiermoe/cpu_planner_context_ms", timing.context_ms)
        self._accumulate_metric("hiermoe/cpu_planner_local_prepare_ms", timing.local_prepare_ms)
        self._accumulate_metric("hiermoe/cpu_planner_statistic_pack_ms", timing.statistic_pack_ms)
        self._accumulate_metric("hiermoe/cpu_planner_collective_ms", collective_active_ms)
        self._accumulate_metric("hiermoe/cpu_planner_score_ms", timing.owner_score_ms)
        self._accumulate_metric("hiermoe/cpu_planner_finalization_ms", timing.finalization_ms)
        self._accumulate_metric("hiermoe/cpu_planner_payload_bytes", timing.local_payload_bytes)
        self._accumulate_metric("hiermoe/cpu_planner_route_share_ms", state.route_share_ms)
        self._accumulate_metric(
            "hiermoe/cpu_planner_collective_ready_host_wait_ms",
            state.collective_ready_host_wait_ms,
        )
        self._accumulate_metric(
            "hiermoe/cpu_planner_collective_close_host_wait_ms",
            state.collective_close_host_wait_ms,
        )
        if result.active_ms > 0.0:
            self._placement_metrics["hiermoe/cpu_planner_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_ms / result.active_ms),
            )
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    def _collect_cpu_batched_plan(self, step: int) -> str:
        if self._cpu_planner_mode == "blocking":
            layers = self._cpu_batched_plan_layers(step)
            if not layers:
                self.latest_pair = "none"
                return self.latest_pair
            wait_started = time.perf_counter()
            state, result = self._run_blocking_cpu_batched_plan(step, layers)
            exposed_ms = (time.perf_counter() - wait_started) * 1000.0
            return self._accept_cpu_batched_result(state, result, exposed_ms=exposed_ms)

        with self._pipeline_lock:
            state = self._cpu_batch_state
        if state is None or state.source_step != int(step) or state.future is None:
            self.latest_pair = "none"
            return self.latest_pair
        # Fallback for short/atypical backward graphs that did not reach the
        # designated collective window.
        state.collective_gate.set()
        wait_started = time.perf_counter()
        result = state.future.result()
        deadline_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        exposed_ms = state.collective_ready_host_wait_ms + state.collective_close_host_wait_ms + deadline_wait_ms
        with self._pipeline_lock:
            if self._cpu_batch_state is state:
                self._cpu_batch_state = None
        return self._accept_cpu_batched_result(state, result, exposed_ms=exposed_ms)

    def _submit_pipeline_plan(
        self,
        layer: ExpertLayerState,
        selected_experts: torch.Tensor,
        step: int,
    ) -> None:
        executor = self._pipeline_plan_executor
        if executor is None or self._pipeline_shutdown:
            return
        if self.expert_swap_max_pairs_per_layer <= 0 and self.max_replica_rounds <= 0:
            return
        if not self._pipeline_is_final_microstep():
            return
        if self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            return
        layout = self._layer_layout(layer)
        if bool((layout < 0).any().item()):
            return
        with self._pipeline_lock:
            if layer.key in self._pipeline_plan_futures or layer.last_planned_step == int(step):
                return
            layer.last_planned_step = int(step)
            windows = _PipelinePlannerWindows()
            self._pipeline_planner_windows[layer.key] = windows
        device = selected_experts.device
        ready_event = self._pipeline_ready_event(device)
        submitted_at = time.perf_counter()
        route = selected_experts.detach()
        owners = layer.logical_to_physical.detach().cpu().clone()
        placement_version = int(layer.placement_version)
        future = executor.submit(
            self._pipeline_plan_worker,
            layer.key,
            route,
            layout,
            owners,
            placement_version,
            int(step),
            ready_event,
            submitted_at,
        )
        with self._pipeline_lock:
            self._pipeline_plan_futures[layer.key] = future

    @torch.no_grad()
    def _pipeline_plan_worker(
        self,
        layer_key: str,
        selected_experts: torch.Tensor,
        layout: torch.Tensor,
        owners: torch.Tensor,
        placement_version: int,
        source_step: int,
        ready_event: Any | None,
        submitted_at: float,
    ) -> _PipelinePlanResult:
        layer = self.layers[layer_key]
        device = selected_experts.device
        started = time.perf_counter()
        timing = _PipelinePlannerStageTiming()
        prepare_timing = _PipelinePrepareSubstageTiming()
        prepare_stage_index = 0

        def prepare_checkpoint(stage: str) -> None:
            nonlocal prepare_stage_index
            try:
                stage_index = _PIPELINE_PREPARE_SUBSTAGES.index(stage)
            except ValueError as exc:
                raise RuntimeError(f"Unknown pipeline Prepare stage for {layer_key}: {stage}.") from exc
            if stage_index < prepare_stage_index:
                expected_stage = _PIPELINE_PREPARE_SUBSTAGES[prepare_stage_index]
                raise RuntimeError(
                    f"Pipeline Prepare stage order diverged for {layer_key}: expected {expected_stage}, got {stage}."
                )
            # Some exact fallback paths do not materialize the compact
            # statistical substages. Treat their missing checkpoints as empty
            # stages while preserving the same six-window cut boundaries.
            prepare_stage_index = stage_index
            if prepare_timing is not None:
                ended_at = time.perf_counter()
                ended_thread_at = time.thread_time()
                prepare_timing.checkpoint(
                    stage,
                    ended_at,
                    ended_thread_at,
                    self._pipeline_stage_event(),
                )
            prepare_stage_index += 1
            paused = self._complete_pipeline_prepare_stage(layer_key, prepare_stage_index)
            if paused and prepare_timing is not None:
                resumed = self._pipeline_stage_event()
                prepare_timing.begin(resumed, time.perf_counter(), time.thread_time())

        def run() -> PlacementPlan:
            self._wait_pipeline_prepare_start(layer_key)
            timing.prepare_start = self._pipeline_stage_event()
            if prepare_timing is not None:
                prepare_timing.begin(timing.prepare_start, time.perf_counter(), time.thread_time())
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=0.0,
                forward_compute_constant=0.0,
                process_group=self._pipeline_planner_group,
            )
            if not isinstance(planner, GreedyCommunicationPlanner):
                raise RuntimeError("The fixed pipeline requires GreedyCommunicationPlanner.")
            prepare_checkpoint("planner_setup")
            planner.reducer = lambda tensor: self._pipeline_planner_reduce_sum(layer_key, tensor, device, timing)
            plan = planner.plan_layers(
                [selected_experts],
                [layout],
                [owners],
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds,
                layer_seeds=[zlib.crc32(layer_key.encode("utf-8"))],
                step=source_step,
                communication_scales=[1.0],
                forward_compute_per_assignment=[0.0],
                forward_compute_constant=[0.0],
                skip_final_route_update=True,
                prepare_stage_callback=prepare_checkpoint,
            )[0]
            timing.score_end = self._pipeline_stage_event()
            with self._pipeline_lock:
                windows = self._pipeline_planner_windows.get(layer_key)
                if windows is not None:
                    windows.score_done_event = timing.score_end
                    windows.score_done.set()
            return plan

        plan = self._run_pipeline_stream_task("planner", device, ready_event, run)
        prepare_ms, collective_ms, score_ms = timing.durations_ms()
        prepare_substage_device_ms, prepare_substage_host_ms, prepare_substage_thread_cpu_ms = (
            ({}, {}, {}) if prepare_timing is None else prepare_timing.durations_ms()
        )
        if prepare_substage_device_ms:
            prepare_ms = sum(prepare_substage_device_ms.values())
        finished = time.perf_counter()
        active_ms = prepare_ms + collective_ms + score_ms
        if active_ms <= 0.0:
            active_ms = (finished - started) * 1000.0
        return _PipelinePlanResult(
            layer_key=layer_key,
            source_step=source_step,
            placement_version=placement_version,
            plan=plan,
            raw_ms=active_ms,
            latency_ms=(finished - submitted_at) * 1000.0,
            prepare_device_ms=prepare_ms,
            collective_device_ms=collective_ms,
            score_device_ms=score_ms,
            prepare_substage_device_ms=prepare_substage_device_ms,
            prepare_substage_host_ms=prepare_substage_host_ms,
            prepare_substage_thread_cpu_ms=prepare_substage_thread_cpu_ms,
        )

    def _pipeline_planner_reduce_sum(
        self,
        layer_key: str,
        tensor: torch.Tensor,
        device: torch.device,
        timing: _PipelinePlannerStageTiming | None = None,
    ) -> torch.Tensor:
        """Hand the reduction to the single ordered collective launcher."""

        if timing is not None:
            timing.prepare_end = self._pipeline_stage_event()
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
            windows.collective_tensor = tensor
            windows.collective_device = device
            windows.collective_timing = timing
            windows.collective_tensor_ready.set()
        windows.collective_result_ready.wait()
        if windows.collective_error is not None:
            raise windows.collective_error
        stream = self._pipeline_stream("planner", device)
        windows.score_gate.wait()
        with self._pipeline_lock:
            compute_done = self._pipeline_planner_compute_events.get(layer_key)
        if stream is not None and compute_done is not None:
            stream.wait_event(compute_done)
        if timing is not None:
            timing.score_start = self._pipeline_stage_event()
        return tensor

    def _launch_pipeline_planner_collective(
        self,
        layer_key: str,
        windows: _PipelinePlannerWindows,
    ) -> None:
        """Launch one layer's HCCL reduction from the globally ordered thread."""

        self._wait_pipeline_host_event(layer_key, windows.collective_tensor_ready)
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            tensor = windows.collective_tensor
            device = windows.collective_device
            timing = windows.collective_timing
            dispatch_done = self._pipeline_planner_dispatch_events.get(layer_key)
        if tensor is None or device is None:
            windows.collective_done.set()
            windows.collective_result_ready.set()
            return
        if device.type == "cpu":
            stream = None
            stream_context = nullcontext()
        else:
            device_api = get_torch_device()
            device_api.set_device(device)
            stream = self._pipeline_stream("planner", device)
            stream_context = device_api.stream(stream)
        try:
            with stream_context:
                if stream is not None and dispatch_done is not None:
                    stream.wait_event(dispatch_done)
                if timing is not None:
                    timing.collective_start = self._pipeline_stage_event()
                self._planner_reduce_sum(tensor, self._pipeline_planner_group)
                if timing is not None:
                    timing.collective_end = self._pipeline_stage_event()
            with self._pipeline_lock:
                if self._pipeline_planner_windows.get(layer_key) is windows:
                    windows.collective_done_event = None if timing is None else timing.collective_end
        except BaseException as error:
            windows.collective_error = error
            raise
        finally:
            windows.collective_done.set()
            windows.collective_result_ready.set()

    @staticmethod
    def _wait_pipeline_stage(event: Event, future: Future[Any]) -> None:
        while not event.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
            if future.done():
                future.result()

    def open_pipeline_planner_collective_window(self, layer_key: str) -> None:
        """Run the planner collective after combine backward and during expert GEMM."""

        if not self.fixed_pipeline_overlap:
            return
        if self._cpu_planner_mode == "process_background":
            order = self._pipeline_layer_order or tuple(self.layers)
            if not order or layer_key != order[0]:
                return
            with self._pipeline_lock:
                state = self._cpu_batch_state
            if state is None or state.source_step != self._pipeline_step:
                return
            executor = self._pipeline_collective_executor
            if executor is None:
                raise RuntimeError("CPU planner process collective executor is unavailable.")
            with self._pipeline_lock:
                if state.process_collective_future is None:
                    state.process_collective_future = executor.submit(
                        self._service_cpu_process_collective,
                        state,
                    )
            return
        if self._cpu_planner_mode == "background":
            order = self._pipeline_layer_order or tuple(self.layers)
            if not order or layer_key != order[0]:
                return
            with self._pipeline_lock:
                state = self._cpu_batch_state
            if state is None or state.source_step != self._pipeline_step or state.future is None:
                return
            wait_started = time.perf_counter()
            while not state.collective_ready.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
                if state.future.done():
                    state.future.result()
                if self._pipeline_shutdown:
                    return
            state.collective_ready_host_wait_ms += (time.perf_counter() - wait_started) * 1000.0
            state.collective_gate.set()
            return
        layer = self.layers.get(layer_key)
        combine_done = None if layer is None else self._pipeline_ready_event(self._pipeline_device(layer))
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
            if windows is not None:
                self._pipeline_planner_dispatch_events[layer_key] = combine_done
        if windows is not None:
            windows.collective_gate.set()
            executor = self._pipeline_collective_executor
            if executor is None:
                raise RuntimeError("HierMoE pipeline collective executor is unavailable.")
            with self._pipeline_lock:
                if windows.collective_future is None:
                    windows.collective_future = executor.submit(
                        self._launch_pipeline_planner_collective,
                        layer_key,
                        windows,
                    )

    def close_pipeline_planner_collective_window(self, layer_key: str) -> None:
        """Enforce collective completion before dispatch-backward uses the EP communicator."""

        if not self.fixed_pipeline_overlap:
            return
        if self._cpu_planner_mode == "process_background":
            order = self._pipeline_layer_order or tuple(self.layers)
            if not order or layer_key != order[0]:
                return
            with self._pipeline_lock:
                state = self._cpu_batch_state
            if state is None or state.source_step != self._pipeline_step:
                return
            wait_started = time.perf_counter()
            while not state.collective_enqueued.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
                future = state.process_collective_future
                if future is not None and future.done():
                    future.result()
                if self._pipeline_shutdown:
                    return
            state.collective_close_host_wait_ms += (time.perf_counter() - wait_started) * 1000.0
            if state.collective_error is not None:
                raise state.collective_error
            done_event = state.collective_done_event
            layer = self.layers.get(layer_key)
            if layer is not None and done_event is not None and done_event.event is not None:
                device = self._pipeline_device(layer)
                if device.type != "cpu":
                    device_api = get_torch_device()
                    try:
                        current_stream = device_api.current_stream(device)
                    except TypeError:
                        current_stream = device_api.current_stream()
                    current_stream.wait_event(done_event.event)
            return
        if self._cpu_planner_mode == "background":
            order = self._pipeline_layer_order or tuple(self.layers)
            if not order or layer_key != order[0]:
                return
            with self._pipeline_lock:
                state = self._cpu_batch_state
            if state is None or state.source_step != self._pipeline_step or state.future is None:
                return
            wait_started = time.perf_counter()
            while not state.collective_enqueued.wait(timeout=_PIPELINE_HOST_EVENT_POLL_SECONDS):
                if state.future.done():
                    state.future.result()
                if self._pipeline_shutdown:
                    return
            state.collective_close_host_wait_ms += (time.perf_counter() - wait_started) * 1000.0
            if state.collective_error is not None:
                raise state.collective_error
            done_event = state.collective_done_event
            layer = self.layers.get(layer_key)
            if layer is not None and done_event is not None and done_event.event is not None:
                device = self._pipeline_device(layer)
                if device.type != "cpu":
                    device_api = get_torch_device()
                    try:
                        current_stream = device_api.current_stream(device)
                    except TypeError:
                        current_stream = device_api.current_stream()
                    current_stream.wait_event(done_event.event)
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
            future = self._pipeline_plan_futures.get(layer_key)
        if windows is None or future is None:
            return
        deadline_event = self._pipeline_stage_event()
        wait_started = time.perf_counter()
        self._wait_pipeline_stage(windows.collective_done, future)
        host_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is windows:
                windows.collective_deadline_event = deadline_event
                done_event = windows.collective_done_event
            else:
                done_event = None
        layer = self.layers.get(layer_key)
        if layer is not None and done_event is not None and done_event.event is not None:
            device = self._pipeline_device(layer)
            device_api = get_torch_device()
            try:
                current_stream = device_api.current_stream(device)
            except TypeError:
                current_stream = device_api.current_stream()
            current_stream.wait_event(done_event.event)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_host_gate_wait_ms", host_wait_ms)
        if host_wait_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_planner_collective_host_gate_miss", 1)

    def open_pipeline_planner_score_window(self, layer_key: str) -> None:
        """Run candidate scoring during dispatch-backward Stage1 payload A2A."""

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        layer = self.layers.get(layer_key)
        score_ready = None if layer is None else self._pipeline_ready_event(self._pipeline_device(layer))
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            self._pipeline_planner_compute_events[layer_key] = score_ready
        windows.score_gate.set()

    def close_pipeline_planner_score_window(self, layer_key: str) -> None:
        """Record the preferred score window without adding a layer barrier.

        Candidate scoring only consumes the previous step's route and its
        correctness deadline is the next-step plan collection.  Waiting here
        would turn every layer's short A2A window into a host barrier and
        prevent the planner queue from carrying unfinished C4 work diagonally
        across later layers.
        """

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        deadline_event = self._pipeline_stage_event()
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is windows:
                windows.score_deadline_event = deadline_event

    def _collect_pipeline_plans(self, step: int) -> str:
        with self._pipeline_lock:
            futures = [
                (key, self._pipeline_plan_futures[key])
                for key in self._pipeline_layer_order
                if key in self._pipeline_plan_futures
            ]
        committed: list[str] = []
        raw_ms = 0.0
        deadline_exposed_ms = 0.0
        prepare_exposed_ms = 0.0
        collective_exposed_ms = 0.0
        score_exposed_ms = 0.0
        score_window_overrun_ms = 0.0
        prepare_window_exposed_ms = [0.0] * len(_PIPELINE_PREPARE_CUT_POINTS)
        collective_window_misses = 0
        score_window_misses = 0
        deadline_misses = 0
        latency_ms = 0.0
        prepare_device_ms = 0.0
        collective_device_ms = 0.0
        score_device_ms = 0.0
        prepare_substage_device_ms = defaultdict(float)
        prepare_substage_host_ms = defaultdict(float)
        prepare_substage_thread_cpu_ms = defaultdict(float)
        accepted = 0
        for layer_key, future in futures:
            with self._pipeline_lock:
                windows = self._pipeline_planner_windows.get(layer_key)
            wait_started = time.perf_counter()
            result = future.result()
            layer_exposed_ms = (time.perf_counter() - wait_started) * 1000.0
            deadline_exposed_ms += layer_exposed_ms
            if layer_exposed_ms > 0.01:
                deadline_misses += 1
            if windows is not None:
                layer_prepare_exposed_ms, per_window = self._pipeline_prepare_exposure_ms(windows)
                prepare_exposed_ms += layer_prepare_exposed_ms
                for window_index, value in enumerate(per_window):
                    prepare_window_exposed_ms[window_index] += value
                layer_collective_exposed_ms = self._pipeline_stage_exposure_ms(
                    windows.collective_deadline_event,
                    windows.collective_done_event,
                )
                layer_score_window_overrun_ms = self._pipeline_stage_exposure_ms(
                    windows.score_deadline_event,
                    windows.score_done_event,
                )
                collective_exposed_ms += layer_collective_exposed_ms
                score_window_overrun_ms += layer_score_window_overrun_ms
                collective_window_misses += int(layer_collective_exposed_ms > 0.01)
                score_window_misses += int(layer_score_window_overrun_ms > 0.01)
            raw_ms += result.raw_ms
            latency_ms += result.latency_ms
            prepare_device_ms += result.prepare_device_ms
            collective_device_ms += result.collective_device_ms
            score_device_ms += result.score_device_ms
            for stage, value in result.prepare_substage_device_ms.items():
                prepare_substage_device_ms[stage] += value
            for stage, value in result.prepare_substage_host_ms.items():
                prepare_substage_host_ms[stage] += value
            for stage, value in result.prepare_substage_thread_cpu_ms.items():
                prepare_substage_thread_cpu_ms[stage] += value
            layer = self.layers[layer_key]
            layer.last_plan = result.plan
            self._record_plan_metrics(result.plan)
            if result.source_step != int(step) or result.placement_version != int(layer.placement_version):
                self._accumulate_metric("hiermoe/pipeline_planner_stale", 1)
            elif result.plan.actions:
                self._pipeline_pending_plans[layer_key] = _PendingPipelinePlan(
                    source_step=result.source_step,
                    placement_version=result.placement_version,
                    plan=result.plan,
                )
                committed.extend(f"{layer_key}:{action.format()}" for action in result.plan.actions)
                accepted += 1
            with self._pipeline_lock:
                self._pipeline_plan_futures.pop(layer_key, None)
                self._pipeline_planner_windows.pop(layer_key, None)
                self._pipeline_planner_dispatch_events.pop(layer_key, None)
                self._pipeline_planner_compute_events.pop(layer_key, None)
        self._accumulate_metric("hiermoe/pipeline_planner_jobs", len(futures))
        self._accumulate_metric("hiermoe/pipeline_planner_accepted", accepted)
        self._accumulate_metric("hiermoe/pipeline_planner_raw_ms", raw_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_latency_ms", latency_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_device_ms", prepare_device_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_device_ms", collective_device_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_device_ms", score_device_ms)
        for stage in _PIPELINE_PREPARE_SUBSTAGES:
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_device_ms",
                prepare_substage_device_ms[stage],
            )
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_host_ms",
                prepare_substage_host_ms[stage],
            )
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_thread_cpu_ms",
                prepare_substage_thread_cpu_ms[stage],
            )
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_exposed_ms", prepare_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_exposed_ms", collective_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_window_overrun_ms", score_window_overrun_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_exposed_ms", score_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_window_miss", collective_window_misses)
        self._accumulate_metric("hiermoe/pipeline_planner_score_window_miss", score_window_misses)
        for window_index, value in enumerate(prepare_window_exposed_ms):
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_window_{window_index}_exposed_ms",
                value,
            )
        self._accumulate_metric("hiermoe/pipeline_planner_deadline_exposed_ms", deadline_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_deadline_miss", deadline_misses)
        self._accumulate_metric(
            "hiermoe/pipeline_planner_exposed_ms",
            prepare_exposed_ms + collective_exposed_ms + score_exposed_ms + deadline_exposed_ms,
        )
        exposed_total = float(self._placement_metrics.get("hiermoe/pipeline_planner_exposed_ms", 0.0))
        if raw_ms > 0.0:
            self._placement_metrics["hiermoe/pipeline_planner_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_total / raw_ms),
            )
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    def _launch_next_pipeline_migration(self) -> None:
        executor = self._pipeline_migration_executor
        if executor is None or self._pipeline_shutdown:
            return
        with self._pipeline_lock:
            if self._pipeline_migration_futures:
                return
            while self._pipeline_next_migration_index < len(self._pipeline_layer_order):
                layer_key = self._pipeline_layer_order[self._pipeline_next_migration_index]
                self._pipeline_next_migration_index += 1
                pending = self._pipeline_pending_plans.get(layer_key)
                if pending is None:
                    continue
                layer = self.layers[layer_key]
                if int(layer.placement_version) != pending.placement_version:
                    self._pipeline_pending_plans.pop(layer_key, None)
                    self._accumulate_metric("hiermoe/pipeline_migration_stale", 1)
                    continue
                device = self._pipeline_device(layer)
                ready_event = self._pipeline_ready_event(device)
                future = executor.submit(
                    self._pipeline_migration_worker,
                    layer_key,
                    pending,
                    ready_event,
                )
                self._pipeline_migration_futures[layer_key] = future
                return

    @torch.no_grad()
    def _pipeline_migration_worker(
        self,
        layer_key: str,
        pending: _PendingPipelinePlan,
        ready_event: Any | None,
    ) -> _PipelineMigrationResult:
        layer = self.layers[layer_key]
        device = self._pipeline_device(layer)
        started = time.perf_counter()

        def run() -> tuple[str, ...]:
            committed = self._execute_placement_plan(
                layer,
                pending.plan,
                timing_prefix="hiermoe_pipeline_migration",
                transfer_group=self._pipeline_migration_group,
                force_staged_transfer=False,
                fast_sparse_transfer=True,
            )
            return tuple(committed)

        committed = self._run_pipeline_stream_task("migration", device, ready_event, run)
        return _PipelineMigrationResult(
            layer_key=layer_key,
            source_step=pending.source_step,
            committed=committed,
            raw_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _finish_pipeline_migration(self, layer_key: str, *, wait: bool) -> bool:
        with self._pipeline_lock:
            future = self._pipeline_migration_futures.get(layer_key)
        if future is None or (not wait and not future.done()):
            return False
        wait_started = time.perf_counter()
        result = future.result()
        exposed_ms = (time.perf_counter() - wait_started) * 1000.0 if wait else 0.0
        with self._pipeline_lock:
            self._pipeline_migration_futures.pop(layer_key, None)
            self._pipeline_pending_plans.pop(layer_key, None)
        self._accumulate_metric("hiermoe/pipeline_migration_jobs", 1)
        self._accumulate_metric("hiermoe/pipeline_migration_raw_ms", result.raw_ms)
        self._accumulate_metric("hiermoe/pipeline_migration_exposed_ms", exposed_ms)
        raw_total = float(self._placement_metrics.get("hiermoe/pipeline_migration_raw_ms", 0.0))
        exposed_total = float(self._placement_metrics.get("hiermoe/pipeline_migration_exposed_ms", 0.0))
        if raw_total > 0.0:
            self._placement_metrics["hiermoe/pipeline_migration_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_total / raw_total),
            )
        if exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_migration_deadline_miss", 1)
        return True

    def advance_pipeline_after_combine(self, _layer_key: str) -> None:
        if not self.fixed_pipeline_overlap or self._ablation_migration_mode != "hidden":
            return
        # Candidate scoring is a compute-only stage whose deadline is the next
        # step, not the end of the current layer. Let the single planner queue
        # carry it diagonally across later layers and collect completed plans at
        # the optimizer boundary.
        with self._pipeline_lock:
            active = tuple(self._pipeline_migration_futures)
        if active and self._finish_pipeline_migration(active[0], wait=False):
            self._launch_next_pipeline_migration()
        elif not active:
            self._launch_next_pipeline_migration()

    def wait_pipeline_migration_before_layer(self, layer_key: str) -> None:
        if not self.fixed_pipeline_overlap or layer_key not in self._pipeline_pending_plans:
            return
        if self._ablation_migration_mode == "blocking":
            pending = self._pipeline_pending_plans[layer_key]
            result = self._pipeline_migration_worker(layer_key, pending, None)
            with self._pipeline_lock:
                self._pipeline_pending_plans.pop(layer_key, None)
            self._accumulate_metric("hiermoe/pipeline_migration_jobs", 1)
            self._accumulate_metric("hiermoe/pipeline_migration_raw_ms", result.raw_ms)
            self._accumulate_metric("hiermoe/pipeline_migration_exposed_ms", result.raw_ms)
            self._placement_metrics["hiermoe/pipeline_migration_hidden_ratio"] = 0.0
            if result.raw_ms > 0.01:
                self._accumulate_metric("hiermoe/pipeline_migration_deadline_miss", 1)
            return
        while layer_key in self._pipeline_pending_plans:
            with self._pipeline_lock:
                active = tuple(self._pipeline_migration_futures)
            if not active:
                self._launch_next_pipeline_migration()
                continue
            self._finish_pipeline_migration(active[0], wait=True)

    def _register_pipeline_gradient_hooks(self, layer: ExpertLayerState) -> None:
        if not self.gradient_overlap_enabled:
            return
        unsupported = [
            index
            for index, param in enumerate(layer.expert_parameters)
            if not callable(getattr(param, "register_post_accumulate_grad_hook", None))
        ]
        if unsupported:
            raise RuntimeError(
                "PlaceMoE requires replica-gradient overlap, but layer "
                f"{layer.key!r} cannot register post-accumulate hooks for expert parameter indices "
                f"{unsupported}. Disable PlaceMoE explicitly or use a supported PyTorch parameter implementation; "
                "blocking synchronization is not selected automatically."
            )

        registered_handles: list[Any] = []
        registered_param_ids: list[int] = []
        for param_index, param in enumerate(layer.expert_parameters):
            if id(param) in self._pipeline_grad_hook_params:
                continue
            register = getattr(param, "register_post_accumulate_grad_hook", None)

            def hook(_param: torch.Tensor, *, key: str = layer.key, index: int = param_index) -> None:
                device = _local_tensor_view(_param).device
                self._pipeline_on_gradient_ready(key, index, self._pipeline_ready_event(device))

            try:
                handle = register(hook)
            except (RuntimeError, TypeError) as error:
                for registered_handle in registered_handles:
                    registered_handle.remove()
                    self._pipeline_grad_hook_handles.remove(registered_handle)
                for param_id in registered_param_ids:
                    self._pipeline_grad_hook_params.discard(param_id)
                raise RuntimeError(
                    "PlaceMoE requires replica-gradient overlap, but failed to register a gradient hook for "
                    f"layer {layer.key!r}, parameter index {param_index}: {error}. Blocking synchronization is not "
                    "selected automatically."
                ) from error
            self._pipeline_grad_hook_handles.append(handle)
            self._pipeline_grad_hook_params.add(id(param))
            registered_handles.append(handle)
            registered_param_ids.append(id(param))

    def _pipeline_on_gradient_ready(
        self,
        layer_key: str,
        param_index: int,
        ready_event: Any | None = None,
    ) -> None:
        if not self.gradient_overlap_enabled or self._pipeline_shutdown or not self._pipeline_is_final_microstep():
            return
        with self._pipeline_lock:
            index = int(param_index)
            self._pipeline_grad_ready[layer_key].add(index)
            if ready_event is not None:
                self._pipeline_grad_ready_events[layer_key][index] = ready_event
        self._advance_pipeline_gradient_queue()

    def _advance_pipeline_gradient_queue(self) -> None:
        """Submit ready layers after dispatch backward in deterministic order."""

        order = tuple(reversed(self._pipeline_layer_order or tuple(self.layers)))
        with self._pipeline_grad_submit_lock:
            while True:
                with self._pipeline_lock:
                    if self._pipeline_grad_comm_blocked:
                        return
                    index = self._pipeline_next_grad_index
                    if index >= len(order):
                        return
                    layer_key = order[index]
                    if (
                        len(self._pipeline_grad_ready[layer_key]) < 2
                        or layer_key not in self._pipeline_grad_dispatch_complete
                    ):
                        return
                    self._pipeline_next_grad_index += 1
                self._submit_pipeline_gradient_sync(layer_key)

    def close_pipeline_gradient_window_before_dispatch(self, _layer_key: str) -> None:
        """Drain background gradient communication before training dispatch backward."""

        if not self.gradient_overlap_enabled or not self._pipeline_is_final_microstep():
            return
        if self._owns_pipeline_grad_group:
            # A dedicated gradient process group is allowed to overlap the
            # training group's dispatch collectives. Waiting here can form a
            # cross-group cycle when ranks reach backward layers at different
            # times: one rank waits for gradient P2P while its peer is already
            # waiting for that rank in the next dispatch collective.
            return
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_comm_blocked = True
                futures = [
                    (layer_key, future)
                    for layer_key, future in self._pipeline_grad_futures.items()
                    if layer_key not in self._pipeline_grad_window_waited
                ]
        exposed_ms = 0.0
        for layer_key, future in futures:
            wait_started = time.perf_counter()
            self._wait_pipeline_gradient_result(future)
            exposed_ms += (time.perf_counter() - wait_started) * 1000.0
            with self._pipeline_lock:
                self._pipeline_grad_window_waited.add(layer_key)
        self._pipeline_grad_window_exposed_ms += exposed_ms
        self._accumulate_metric("hiermoe/pipeline_grad_sync_window_exposed_ms", exposed_ms)
        if exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_grad_sync_window_miss", 1)

    def open_pipeline_gradient_window_after_dispatch(self, layer_key: str) -> None:
        """Admit this layer's redundant-gradient sync after dispatch backward."""

        if not self.gradient_overlap_enabled or not self._pipeline_is_final_microstep():
            return
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_dispatch_complete.add(layer_key)
                self._pipeline_grad_comm_blocked = False
        self._advance_pipeline_gradient_queue()

    @torch.no_grad()
    def _submit_pipeline_gradient_sync(self, layer_key: str) -> None:
        if not self.gradient_overlap_enabled or self._pipeline_shutdown:
            return
        layer = self.layers[layer_key]
        schedule = self._replica_grad_schedule_for_layer(layer)
        if not schedule.groups:
            return
        with self._pipeline_lock:
            if layer_key in self._pipeline_grad_futures:
                return
            previous_layer_key = next(reversed(self._pipeline_grad_futures), None)
            previous_future = None if previous_layer_key is None else self._pipeline_grad_futures[previous_layer_key]
        if previous_future is not None:
            wait_started = time.perf_counter()
            self._wait_pipeline_gradient_result(previous_future)
            exposed_ms = (time.perf_counter() - wait_started) * 1000.0
            self._pipeline_grad_window_exposed_ms += exposed_ms
            self._accumulate_metric("hiermoe/pipeline_grad_sync_backpressure_exposed_ms", exposed_ms)
        device = self._pipeline_device(layer)
        dispatch_event = self._pipeline_ready_event(device)
        with self._pipeline_lock:
            parameter_events = tuple(
                self._pipeline_grad_ready_events[layer_key][index]
                for index in sorted(self._pipeline_grad_ready_events[layer_key])
            )
        ready_events = parameter_events + ((dispatch_event,) if dispatch_event is not None else ())
        future: Future[_PipelineGradResult] = Future()
        result = self._pipeline_gradient_worker(layer_key, schedule, ready_events or None)
        future.set_result(result)
        with self._pipeline_lock:
            self._pipeline_grad_futures[layer_key] = future

    @torch.no_grad()
    def _pipeline_gradient_worker(
        self,
        layer_key: str,
        schedule: _ReplicaGradSchedule,
        ready_event: Any | None,
    ) -> _PipelineGradResult:
        layer = self.layers[layer_key]
        device = self._pipeline_device(layer)
        started = time.perf_counter()
        start_event: AcceleratorEvent | None = None
        completion_event: AcceleratorEvent | None = None

        def run() -> None:
            self._zero_inactive_slot_grads_for_layer(layer)
            contributions = self._replica_grad_contributions(layer, schedule)
            if schedule.pairwise:
                self._sync_pairwise_replica_gradients(
                    schedule,
                    contributions,
                    process_group=self._pipeline_grad_group,
                )
            else:
                self._sync_owner_replica_gradients(
                    schedule,
                    contributions,
                    process_group=self._pipeline_grad_group,
                )

        if device.type == "cpu":
            run()
        else:
            device_api = get_torch_device()
            device_api.set_device(device)
            stream = self._pipeline_stream("gradient", device)
            assert stream is not None
            with device_api.stream(stream):
                ready_events = ready_event if isinstance(ready_event, tuple) else (ready_event,)
                for event in ready_events:
                    if event is not None:
                        stream.wait_event(event)
                start_event = record_accelerator_event()
                run()
                completion_event = record_accelerator_event()
        return _PipelineGradResult(
            layer_key=layer_key,
            raw_ms=(time.perf_counter() - started) * 1000.0,
            start_event=start_event,
            completion_event=completion_event,
        )

    @staticmethod
    def _wait_pipeline_gradient_result(
        future: Future[_PipelineGradResult],
    ) -> tuple[_PipelineGradResult, float]:
        result = future.result()
        completion_event = result.completion_event
        if completion_event is not None and completion_event.event is not None:
            completion_event.event.synchronize()
        raw_ms = result.raw_ms
        if result.start_event is not None and completion_event is not None:
            raw_ms = result.start_event.elapsed_time(completion_event)
        return result, raw_ms

    @torch.no_grad()
    def _finish_pipeline_gradient_sync(self) -> None:
        order = self._pipeline_layer_order or tuple(self.layers)
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_comm_blocked = False
                self._pipeline_grad_dispatch_complete.update(order)
            for layer_key in reversed(order):
                self._submit_pipeline_gradient_sync(layer_key)
        with self._pipeline_lock:
            futures = [
                (layer_key, self._pipeline_grad_futures[layer_key])
                for layer_key in reversed(order)
                if layer_key in self._pipeline_grad_futures
            ]
        raw_ms = 0.0
        deadline_exposed_ms = 0.0
        for layer_key, future in futures:
            if layer_key in self._pipeline_grad_window_waited:
                _result, layer_raw_ms = self._wait_pipeline_gradient_result(future)
            else:
                wait_started = time.perf_counter()
                _result, layer_raw_ms = self._wait_pipeline_gradient_result(future)
                deadline_exposed_ms += (time.perf_counter() - wait_started) * 1000.0
            raw_ms += layer_raw_ms
            with self._pipeline_lock:
                self._pipeline_grad_futures.pop(layer_key, None)
        exposed_ms = self._pipeline_grad_window_exposed_ms + deadline_exposed_ms
        self._accumulate_metric("hiermoe/pipeline_grad_sync_jobs", len(futures))
        self._accumulate_metric("hiermoe/pipeline_grad_sync_raw_ms", raw_ms)
        self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_exposed_ms", deadline_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_grad_sync_exposed_ms", exposed_ms)
        if raw_ms > 0.0:
            self._placement_metrics["hiermoe/pipeline_grad_sync_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_ms / raw_ms),
            )
        if deadline_exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_miss", 1)
        with self._pipeline_lock:
            self._pipeline_grad_window_waited.clear()
            self._pipeline_grad_window_exposed_ms = 0.0
        self._debug_log_redundant_copy_stats("after_grad_sync", include_grads=True)
        self._clear_accumulated_token_counts()

    def shutdown_pipeline(self) -> None:
        if (
            not self.fixed_pipeline_overlap and not self.gradient_overlap_enabled and not self._hot_update
        ) or self._pipeline_shutdown:
            return
        self._pipeline_shutdown = True
        hot_update_state = self._hot_update_controller.active_job
        if hot_update_state is not None and hot_update_state.process is not None:
            terminate_planner_process(hot_update_state.process)
            self._hot_update_event(
                "terminated_on_shutdown",
                update_mode=hot_update_state.update_mode,
                source_step=hot_update_state.source_step,
            )
        self._hot_update_controller.finish()
        with self._pipeline_lock:
            windows = tuple(self._pipeline_planner_windows.values())
            cpu_state = self._cpu_batch_state
        if cpu_state is not None:
            cpu_state.collective_gate.set()
        for window in windows:
            for gate in window.prepare_gates:
                gate.set()
            for enqueued in window.prepare_enqueued:
                enqueued.set()
            window.collective_gate.set()
            if window.collective_future is None:
                window.collective_done.set()
                window.collective_result_ready.set()
            window.score_gate.set()
        for future in tuple(self._pipeline_plan_futures.values()):
            future.result()
        for future in tuple(self._pipeline_migration_futures.values()):
            future.result()
        for future in tuple(self._pipeline_grad_futures.values()):
            future.result()
        if cpu_state is not None and cpu_state.future is not None:
            cpu_state.future.result()
        if self._cpu_process_runtime is not None:
            self._cpu_process_runtime.close()
            self._cpu_process_runtime = None
        for handle in self._pipeline_grad_hook_handles:
            handle.remove()
        for executor in (
            self._pipeline_plan_executor,
            self._pipeline_collective_executor,
            self._pipeline_migration_executor,
            self._pipeline_grad_executor,
            self._cpu_plan_executor,
        ):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    def destroy_pipeline_process_groups(self) -> None:
        if not self._owns_pipeline_grad_group:
            return
        group = self._pipeline_grad_group
        self._pipeline_grad_group = None
        self._owns_pipeline_grad_group = False
        if (
            group is not None
            and dist.is_available()
            and dist.is_initialized()
            and group != dist.GroupMember.NON_GROUP_MEMBER
        ):
            dist.destroy_process_group(group)

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
        local_device = _local_tensor_view(layer.primary_parameter).device
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
            groups = candidate.redundant_copy_groups()[:_DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS]
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
                for _param_name, param in candidate.named_expert_parameters():
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
        self._normalize_ablation_layer_keys()
        if _FIXED_R2_LAYOUT:
            self.install_fixed_r2_layout()
        if self._initial_layout_path or self._ablation_replay_mode == "static":
            self._install_static_ablation_layout()
        if self._pending_state is not None:
            self.load_state_dict(self._pending_state)
            self._pending_state = None
        self._configure_hot_update_training_affinity()

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
            source_route_lut = first_slots.view(1, -1).expand(self.ep_size, -1).clone()
            source_route_lut[half_ep:] = second_slots

            current_layout = layer.slot_to_logical.detach().cpu()
            if torch.equal(current_layout, target_layout):
                self._refresh_layer_mapping_from_slots(layer, tuple(int(slot) for slot in first_slots.tolist()))
                if self._forward_reuse_cover_patch_remap:
                    layer.source_logical_to_physical = source_route_lut
                    layer._device_source_mapping_cache.clear()
                layer.fixed_r2_layout = True
                continue

            state_tensors = (
                list(layer.expert_parameters) if self.optimizer is None else self._slot_op_state_tensors(layer)
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
            if self._forward_reuse_cover_patch_remap:
                layer.source_logical_to_physical = source_route_lut
                layer._device_source_mapping_cache.clear()
            layer.fixed_r2_layout = True

        logger.info_rank0("HierMoE installed the fixed R2 layout for %s layer(s).", len(self.layers))

    def _is_expert_module(self, module: nn.Module) -> bool:
        return resolve_moe_model_adapter(module) is not None

    def register_layer(self, key: str, module: nn.Module) -> None:
        adapter = resolve_moe_model_adapter(module)
        if adapter is None:
            raise TypeError(f"No PlaceMoE model adapter supports expert layer {key!r}.")
        named_parameters = adapter.expert_parameters(module)
        if not named_parameters:
            raise ValueError(f"PlaceMoE model adapter {adapter.name!r} exposes no parameters for {key!r}.")
        local_parameters = tuple(_local_tensor_view(item.parameter) for item in named_parameters)
        num_local_experts = int(local_parameters[0].shape[0])
        if any(int(parameter.shape[0]) != num_local_experts for parameter in local_parameters):
            shapes = {
                item.name: tuple(parameter.shape)
                for item, parameter in zip(named_parameters, local_parameters, strict=True)
            }
            raise ValueError(f"PlaceMoE expert parameters for {key!r} have inconsistent slot dimensions: {shapes}.")

        num_experts = adapter.num_experts(module)
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
        previous_source_lut = None if layer is None else layer.source_logical_to_physical
        registered_layer = ExpertLayerState(
            key=key,
            module_id=id(module),
            num_experts=num_experts,
            base_num_local_experts=base_num_local_experts,
            num_local_experts=num_local_experts,
            expert_parameter_names=tuple(item.name for item in named_parameters),
            expert_parameters=tuple(item.parameter for item in named_parameters),
            model_adapter=adapter,
            logical_to_physical=mapping,
            slot_to_logical=slot_to_logical,
            canonical_physical_slots=canonical_slots,
            is_identity=bool(is_identity),
        )
        if (self._hot_update or self._forward_reuse_cover_patch_remap) and slot_layout_enabled:
            if previous_source_lut is not None and tuple(previous_source_lut.shape) == (self.ep_size, num_experts):
                registered_layer.source_logical_to_physical = previous_source_lut.detach().cpu().clone()
            else:
                # Before the first planner result, every source rank routes to
                # the canonical owner. This gives hot updates a valid M even
                # when training starts without a precomputed PlaceMoE artifact.
                registered_layer.source_logical_to_physical = mapping.view(1, -1).expand(self.ep_size, -1).clone()
        self.layers[key] = registered_layer
        self.module_id_to_key[id(module)] = key
        for parameter in registered_layer.expert_parameters:
            self.param_id_to_key[id(parameter)] = key
        self._register_pipeline_gradient_hooks(self.layers[key])

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
            if replay is None:
                raise RuntimeError(f"Checkpoint route replay is missing an occurrence for {layer_key}.")
            if replay.shape != selected_experts.shape:
                raise RuntimeError(
                    f"Checkpoint route replay for {layer_key} does not match recompute input: "
                    f"replay_shape={tuple(replay.shape)}, input_shape={tuple(selected_experts.shape)}."
                )
            replay = replay.to(device=selected_experts.device, dtype=torch.long)
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
        if layer.slot_layout_enabled:
            dispatched_routes = (
                selected_experts
                if self._uses_compact_identity_dispatch(layer)
                else self._map_logical_to_slot(layer, selected_experts)
            )
        elif layer.is_identity:
            dispatched_routes = selected_experts
        else:
            mapping = layer.mapping_for_device(selected_experts.device)
            dispatched_routes = mapping.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
        if checkpoint_replay is not None and not checkpoint_recompute:
            checkpoint_replay.record(layer_key, dispatched_routes)
        return dispatched_routes

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
        # A source-conditioned LUT is the authoritative routing policy emitted
        # by PlaceMoE. Static preload and hot-update paths install it without
        # enabling Forward-reuse Cover, so gating it on that optimization
        # silently replaces the scored mapping with the generic greedy mapper.
        if layer.source_logical_to_physical is not None:
            mapping = layer.source_mapping_for_device(selected.device, self.ep_rank)
            physical = mapping.index_select(0, selected.reshape(-1)).view_as(selected)
            return physical.squeeze(-1) if original_ndim == 1 else physical
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
        if layer.fixed_r2_layout and _FORCE_FIXED_R2_MIRRORED_REMAP:
            physical = assign_tokens_to_mirrored_r2(
                selected,
                copy_slots,
                source_ranks=self.ep_rank,
                num_ranks=self.ep_size,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            physical = assign_tokens_to_copies_greedy(
                selected,
                layer.slot_to_logical,
                slots_per_rank=layer.num_local_experts,
                source_ranks=self.ep_rank,
                hierarchy_group_sizes=self.hierarchy.group_sizes,
                num_experts=layer.num_experts,
                step=max(0, int(layer.latest_route_step)),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=self.greedy_max_copies_per_expert,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
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
        if (
            self.fixed_pipeline_overlap
            and self._ablation_replay_mode == "off"
            and not self._npu_layer_owner_blocking
            and self._online_freeze_cost_mode == "off"
            and not self._forward_reuse_cover
            and step is not None
        ):
            if self._cpu_planner_mode == "background":
                self._submit_cpu_batched_plan(int(step))
            elif self._cpu_planner_mode == "process_background":
                self._submit_cpu_process_plan(int(step), background=True)
            elif self._cpu_planner_mode == "off":
                self._submit_pipeline_plan(layer, selected_experts, int(step))

    def record_forward_physical_routes(self, layer_key: str, physical_routes: torch.Tensor) -> None:
        """Keep the physical routes already consumed by the trainable Forward."""

        if not self._forward_reuse_cover and not self._cost_model_verify and self._online_freeze_cost_mode == "off":
            return
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_physical_routes = physical_routes.detach()

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

    def record_dispatch_statistics(
        self,
        *,
        layer_key: str,
        step: int,
        dispatch_context: Any,
    ) -> None:
        """Cache exact baseline receive counts already produced by Forward.

        The winner validator previously rescanned every token row to rebuild
        the unchanged baseline. Hierarchical dispatch already computed the
        destination receive counts: each rank contributes its rank receive
        total and its relay's node receive total to a sparse packed row. A
        later SUM collective reconstructs the same global rank/node vector as
        ``_local_packed_counts`` without another token scan.
        """

        if not self._forward_reuse_cover:
            return
        layer = self.layers.get(layer_key)
        if layer is None or int(step) != int(layer.latest_route_step):
            return
        if getattr(dispatch_context, "mode", None) != "hierarchical":
            layer.latest_forward_baseline_communication_counts = None
            layer.latest_forward_traffic_endpoint_statistics = None
            return
        stage1_send = getattr(dispatch_context, "stage1_unique_send_splits", None)
        stage1_recv = getattr(dispatch_context, "stage1_unique_recv_splits", None)
        stage1_assignment_send = getattr(
            dispatch_context,
            "stage1_assignment_send_splits",
            None,
        )
        stage2_send = getattr(dispatch_context, "stage2_unique_send_splits", None)
        stage2_recv = getattr(dispatch_context, "stage2_unique_recv_splits", None)
        stage2_assignment_send = getattr(
            dispatch_context,
            "stage2_assignment_send_splits",
            None,
        )
        valid_sizes = [
            int(size)
            for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
            if 1 < int(size) < self.ep_size and self.ep_size % int(size) == 0
        ]
        if (
            stage1_send is None
            or stage1_recv is None
            or stage1_assignment_send is None
            or stage2_send is None
            or stage2_recv is None
            or stage2_assignment_send is None
            or len(valid_sizes) != 1
        ):
            layer.latest_forward_baseline_communication_counts = None
            layer.latest_forward_traffic_endpoint_statistics = None
            return

        group_size = valid_sizes[0]
        num_nodes = self.ep_size // group_size
        if (
            len(stage1_send) != num_nodes
            or len(stage1_assignment_send) != num_nodes
            or len(stage2_send) != group_size
            or len(stage2_assignment_send) != group_size
        ):
            layer.latest_forward_baseline_communication_counts = None
            layer.latest_forward_traffic_endpoint_statistics = None
            return
        counts = torch.zeros((self.ep_size + self.ep_size // group_size,), dtype=torch.float32)
        counts[self.ep_rank] = float(sum(int(value) for value in stage2_recv))
        counts[self.ep_size + self.ep_rank // group_size] = float(sum(int(value) for value in stage1_recv))
        layer.latest_forward_baseline_communication_counts = counts

        endpoint = torch.zeros((8, self.ep_size), dtype=torch.float32)
        lane = self.ep_rank % group_size
        node = self.ep_rank // group_size
        stage1_destinations = lane * num_nodes + torch.arange(num_nodes)
        stage2_destinations = node * group_size + torch.arange(group_size)

        endpoint[0, self.ep_rank] = float(sum(int(value) for value in stage1_send))
        endpoint[1, stage1_destinations] = torch.tensor(stage1_send, dtype=torch.float32)
        endpoint[2, self.ep_rank] = float(sum(int(value) for value in stage1_assignment_send))
        endpoint[3, stage1_destinations] = torch.tensor(
            stage1_assignment_send,
            dtype=torch.float32,
        )
        endpoint[4, self.ep_rank] = float(sum(int(value) for value in stage2_send))
        endpoint[5, stage2_destinations] = torch.tensor(stage2_send, dtype=torch.float32)
        endpoint[6, self.ep_rank] = float(sum(int(value) for value in stage2_assignment_send))
        endpoint[7, stage2_destinations] = torch.tensor(
            stage2_assignment_send,
            dtype=torch.float32,
        )
        layer.latest_forward_traffic_endpoint_statistics = endpoint.reshape(-1)

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
        communication_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None = None,
    ) -> None:
        del selected_dim
        layer = self.layers.get(layer_key)
        if layer is None or not self.placement_planning_enabled() or not self.layer_calibration_enabled():
            return
        if layer.slot_to_logical is None:
            layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
            logical = torch.arange(layer.num_experts, dtype=torch.long)
            layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        else:
            layout = layer.slot_to_logical.detach().cpu().clone()
        timing = _PendingLayerTiming(
            step=int(step),
            selected_experts=selected_experts.detach(),
            slot_to_logical=layout,
            local_assignment_count=tokens_per_local_expert.detach().sum().to(dtype=torch.float32),
            dispatch_start=dispatch_start,
            dispatch_end=dispatch_end,
            compute_start=compute_start,
            compute_end=compute_end,
            combine_start=combine_start,
            combine_end=combine_end,
        )
        layer.pending_timing = timing
        capture_cost_model_sample = (
            self._cost_model_verify
            and int(self._online_freeze_calibration_step)
            <= int(step)
            <= int(self._online_freeze_calibration_step) + int(self._cost_model_validation_steps)
        ) or (self._online_freeze_cost_mode != "off" and int(step) == int(self._online_freeze_calibration_step))
        if capture_cost_model_sample:
            physical_routes = layer.latest_physical_routes
            if physical_routes is None or physical_routes.shape != selected_experts.shape:
                raise RuntimeError(
                    f"Cost-model verification did not capture the Forward physical routes for {layer_key}."
                )
            layer.cost_model_timings.append(
                _CostModelTiming(
                    step=int(step),
                    physical_routes=physical_routes.detach(),
                    local_expert_token_counts=tokens_per_local_expert.detach(),
                    local_assignment_count=tokens_per_local_expert.detach().sum().to(dtype=torch.float32),
                    communication_events=communication_events,
                    dispatch_start=dispatch_start,
                    dispatch_end=dispatch_end,
                    compute_start=compute_start,
                    compute_end=compute_end,
                    combine_start=combine_start,
                    combine_end=combine_end,
                )
            )

    def record_local_expert_token_counts(self, layer_key: str, tokens_per_local_expert: torch.Tensor) -> None:
        layer = self.layers.get(layer_key)
        if layer is None or not layer.slot_layout_enabled:
            return
        counts = tokens_per_local_expert.detach()
        if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            return
        if self._forward_reuse_cover:
            layer.latest_tokens_per_local_expert = counts
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
            self._zero_inactive_slot_grads_for_layer(layer)

    @torch.no_grad()
    def _zero_inactive_slot_grads_for_layer(self, layer: ExpertLayerState) -> None:
        counts = layer.accumulated_tokens_per_local_expert
        if counts is None or not layer.slot_layout_enabled:
            return
        if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            layer.accumulated_tokens_per_local_expert = None
            return
        zero_slots = counts <= 0
        # NPU grouped-matmul backward may leave undefined weight gradients
        # for zero-token groups. Those slots are mathematically inactive.
        for parameter in layer.expert_parameters:
            self._zero_grad_slots(parameter, zero_slots)

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
        policy_keys: set[tuple[int, int, tuple[int, ...]]] = set()
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
            policy_key = (
                entry.source_rank,
                entry.logical_expert,
                entry.destination_ranks,
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
        for param in layer.expert_parameters:
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
        pairwise = all(len(group.copy_ranks) <= 2 for group in groups)
        peer_neighbors = [set() for _ in range(int(self.ep_size))]
        if pairwise:
            for group in groups:
                if len(group.copy_ranks) != 2:
                    continue
                left_rank, right_rank = group.copy_ranks
                peer_neighbors[int(left_rank)].add(int(right_rank))
                peer_neighbors[int(right_rank)].add(int(left_rank))
        cached = _ReplicaGradSchedule(
            groups=tuple(groups),
            pairwise=pairwise,
            # Fixed mirrored R2 has one peer per rank and safely uses one
            # batched wave. Arbitrary partial-capacity layouts form a general
            # rank graph; all ranks must traverse its edges in the same order
            # or HCCL can leave the P2P work pending indefinitely.
            globally_ordered_pairs=pairwise and any(len(neighbors) > 1 for neighbors in peer_neighbors),
        )
        layer._replica_grad_schedule_cache = cached
        return cached

    def _replica_grad_contributions(
        self,
        layer: ExpertLayerState,
        schedule: _ReplicaGradSchedule,
    ) -> dict[tuple[torch.device, torch.dtype], dict[tuple[int, int], _ReplicaGradContribution]]:
        params = layer.expert_parameters
        local_grads = tuple(self._local_grad_for_redundant_sync(param) for param in params)
        contributions: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ] = defaultdict(dict)
        # Pre-seed the local parameter buckets so every rank enters the same
        # packed collective even when a partial-capacity layout gives this
        # rank no redundant expert in a particular layer.
        for local_grad in local_grads:
            if local_grad is not None:
                contributions[(local_grad.device, local_grad.dtype)]
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
        key = (kind, int(peer_rank), str(device), str(dtype))
        cached = self._replica_grad_buffers.get(key)
        if cached is None or cached.numel() < numel:
            cached = torch.empty((numel,), dtype=dtype, device=device)
            self._replica_grad_buffers[key] = cached
        return cached[:numel]

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
        process_group: dist.ProcessGroup | None = None,
        globally_ordered_pairs: bool = False,
    ) -> dict[int, torch.Tensor]:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
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

        def _run_peer(peer_rank: int) -> None:
            peer_global_rank = _ep_global_rank(process_group, peer_rank)
            ops: list[dist.P2POp] = []
            send_buffer = send_buffers.get(peer_rank)
            if send_buffer is not None:
                ops.append(dist.P2POp(dist.isend, send_buffer, peer_global_rank, process_group))
            recv_buffer = recv_buffers.get(peer_rank)
            if recv_buffer is not None:
                ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_global_rank, process_group))
            if not ops:
                return
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()

        if globally_ordered_pairs:
            # An arbitrary EPLB layout can give one expert copies on many
            # ranks.  Issuing every rank's owner-star peer list in its local
            # order can create a different HCCL P2P order on each rank and
            # deadlock.  Traverse the same undirected rank-pair schedule
            # everywhere; only the two participating ranks issue work.
            for left_rank in range(int(self.ep_size)):
                for right_rank in range(left_rank + 1, int(self.ep_size)):
                    if self.ep_rank == left_rank:
                        _run_peer(right_rank)
                    elif self.ep_rank == right_rank:
                        _run_peer(left_rank)
        else:
            ops: list[dist.P2POp] = []
            for peer_rank in sorted(set(send_buffers) | set(recv_buffers)):
                peer_global_rank = _ep_global_rank(process_group, peer_rank)
                send_buffer = send_buffers.get(peer_rank)
                if send_buffer is not None:
                    ops.append(dist.P2POp(dist.isend, send_buffer, peer_global_rank, process_group))
                recv_buffer = recv_buffers.get(peer_rank)
                if recv_buffer is not None:
                    ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_global_rank, process_group))
            if ops:
                works = dist.batch_isend_irecv(ops)
                for work in works:
                    work.wait()
        return recv_buffers

    def _run_replica_grad_all_to_all_wave(
        self,
        *,
        send_buffers: dict[int, torch.Tensor],
        process_group: dist.ProcessGroup | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Exchange an arbitrary pair graph with one globally ordered collective."""

        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
            if send_buffers:
                raise RuntimeError("HierMoE redundant gradient synchronization requires an EP process group.")
            return {}

        split_sizes = [
            0 if rank not in send_buffers else int(send_buffers[rank].numel()) for rank in range(int(self.ep_size))
        ]
        parts = [send_buffers[rank] for rank in range(int(self.ep_size)) if split_sizes[rank] > 0]
        send_buffer = torch.cat(parts, dim=0) if parts else torch.empty((0,), dtype=dtype, device=device)
        recv_buffer = torch.empty_like(send_buffer)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=process_group,
        )

        received: dict[int, torch.Tensor] = {}
        offset = 0
        for peer_rank, size in enumerate(split_sizes):
            if size > 0:
                received[peer_rank] = recv_buffer[offset : offset + size]
                offset += size
        return received

    def _sync_pairwise_replica_gradients(
        self,
        schedule: _ReplicaGradSchedule,
        contributions_by_bucket: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ],
        process_group: dist.ProcessGroup | None = None,
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
            if schedule.globally_ordered_pairs:
                # A lexicographic sequence of blocking rank-pair P2P calls is
                # not a true global order: ranks skip inactive edges and can
                # form a wait cycle on a general multi-neighbor replica graph.
                # Pack all incident edges into one collective. Each two-copy
                # edge is symmetric, so the input/output split vector is the
                # same and no split-size exchange is required.
                recv_buffers = self._run_replica_grad_all_to_all_wave(
                    send_buffers=send_buffers,
                    process_group=process_group,
                    dtype=dtype,
                    device=device,
                )
            else:
                recv_buffers = self._run_replica_grad_p2p_wave(
                    phase="pairwise",
                    send_buffers=send_buffers,
                    process_group=process_group,
                    recv_specs=recv_specs,
                    globally_ordered_pairs=False,
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
        process_group: dist.ProcessGroup | None = None,
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
                process_group=process_group,
                recv_specs=reduce_recv_specs,
                globally_ordered_pairs=True,
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
                process_group=process_group,
                recv_specs=broadcast_recv_specs,
                globally_ordered_pairs=True,
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
        if self.gradient_overlap_enabled:
            with _full_timing_range("hiermoe_redundant_grad_sync_deadline"):
                self._finish_pipeline_gradient_sync()
            return
        started = time.perf_counter()
        jobs = 0
        with _full_timing_range("hiermoe_redundant_grad_sync"):
            self._zero_inactive_slot_grads()
            self._debug_log_redundant_copy_stats("before_grad_sync", include_grads=True)
            for layer in self.layers.values():
                if layer.slot_to_logical is None:
                    continue
                schedule = self._replica_grad_schedule_for_layer(layer)
                if not schedule.groups:
                    continue
                jobs += 1
                contributions = self._replica_grad_contributions(layer, schedule)
                if schedule.pairwise:
                    self._sync_pairwise_replica_gradients(schedule, contributions)
                else:
                    self._sync_owner_replica_gradients(schedule, contributions)
            self._debug_log_redundant_copy_stats("after_grad_sync", include_grads=True)
            self._clear_accumulated_token_counts()
        if self.fixed_pipeline_overlap:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._accumulate_metric("hiermoe/pipeline_grad_sync_jobs", jobs)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_raw_ms", elapsed_ms)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_exposed_ms", elapsed_ms)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_exposed_ms", elapsed_ms)
            self._placement_metrics["hiermoe/pipeline_grad_sync_hidden_ratio"] = 0.0
            if elapsed_ms > 0.01:
                self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_miss", 1)

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
            for param in layer.expert_parameters:
                masks[id(param)] = mask
        return masks

    def _planner_collective_backend(self, process_group: dist.ProcessGroup | None = None) -> str | None:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
            return None
        return str(dist.get_backend(process_group)).lower().rsplit(".", maxsplit=1)[-1]

    def _planner_reduce_sum(
        self,
        tensor: torch.Tensor,
        process_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is not None and self.ep_size > 1:
            backend = self._planner_collective_backend(process_group)
            if backend == "gloo" and tensor.device.type != "cpu":
                reduced = tensor.detach().to(device="cpu")
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)
                tensor.copy_(reduced.to(device=tensor.device))
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
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
        for param in layer.expert_parameters:
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
        forward_compute_constant: float = 0.0,
        process_group: dist.ProcessGroup | None = None,
    ) -> CurrentRoutePlanner | GreedyCommunicationPlanner:
        planner_group = self.ep_group if process_group is None else process_group
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            return GreedyCommunicationPlanner(
                hierarchy=self.hierarchy,
                perf_model=self.perf_model,
                hidden_size=layer.latest_hidden_size,
                bytes_per_element=layer.latest_bytes_per_element,
                slots_per_rank=layer.num_local_experts,
                communication_scale=communication_scale,
                forward_compute_per_assignment=forward_compute_per_assignment,
                forward_compute_constant=forward_compute_constant,
                smooth_max_gamma=self.smooth_max_gamma,
                reducer=lambda tensor: self._planner_reduce_sum(tensor, planner_group),
                candidate_chunk_size=_SWAP_COST_CHUNK_CANDIDATES,
                process_group=planner_group,
                max_copies=self.greedy_max_copies_per_expert,
                assume_unique_routes=True,
                layer_parallel_streams=_GREEDY_LAYER_PARALLEL_STREAMS,
                adaptive_topk=(
                    _GREEDY_ADAPTIVE_TOPK and not self.fixed_pipeline_overlap and _GREEDY_EXACT_PRIMITIVE_TOPK == 0
                ),
                adaptive_topk_initial=_GREEDY_ADAPTIVE_TOPK_INITIAL,
                adaptive_topk_strict_certificate=_GREEDY_ADAPTIVE_TOPK_STRICT,
                exact_primitive_topk=(0 if self.fixed_pipeline_overlap else _GREEDY_EXACT_PRIMITIVE_TOPK),
                post_shortlist_compact_pair=(not self.fixed_pipeline_overlap and _GREEDY_POST_SHORTLIST_COMPACT_PAIR),
                exact_primitive_max_only=(not self.fixed_pipeline_overlap and _GREEDY_EXACT_PRIMITIVE_MAX_ONLY),
                traffic_inter_ms_per_byte=(
                    self._online_freeze_inter_ms_per_byte if self._online_freeze_cost_mode != "off" else None
                ),
                traffic_intra_ms_per_byte=(
                    self._online_freeze_intra_ms_per_byte if self._online_freeze_cost_mode != "off" else None
                ),
                traffic_route_ms_per_assignment=(
                    self._online_freeze_route_ms_per_assignment if self._online_freeze_cost_mode == "joint" else 0.0
                ),
                traffic_communication_phase_multiplier=(
                    self._online_freeze_communication_ratio if self._online_freeze_cost_mode != "off" else 1.0
                ),
                traffic_compute_phase_multiplier=(
                    self._online_freeze_compute_ratio if self._online_freeze_cost_mode == "joint" else 1.0
                ),
            )
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

    @staticmethod
    def _fit_nonnegative_compute_model(samples: list[tuple[float, float]]) -> tuple[float, float]:
        """Fit y = slope * x + intercept with both parameters constrained non-negative."""

        finite = [(x, y) for x, y in samples if math.isfinite(x) and math.isfinite(y) and x >= 0.0 and y >= 0.0]
        if not finite:
            return 0.0, 0.0
        x = torch.tensor([row[0] for row in finite], dtype=torch.float64)
        y = torch.tensor([row[1] for row in finite], dtype=torch.float64)
        candidates: list[tuple[float, float]] = []
        design = torch.stack((x, torch.ones_like(x)), dim=1)
        solution = torch.linalg.lstsq(design, y).solution
        unconstrained = (float(solution[0].item()), float(solution[1].item()))
        if unconstrained[0] >= 0.0 and unconstrained[1] >= 0.0:
            candidates.append(unconstrained)
        x_square_sum = float(torch.dot(x, x).item())
        if x_square_sum > 0.0:
            candidates.append((max(0.0, float(torch.dot(x, y).item()) / x_square_sum), 0.0))
        else:
            candidates.append((0.0, 0.0))
        candidates.append((0.0, max(0.0, float(y.mean().item()))))
        return min(
            candidates,
            key=lambda row: float(torch.square(row[0] * x + row[1] - y).sum().item()),
        )

    @staticmethod
    def _fit_positive_through_origin(samples: Sequence[tuple[float, float]]) -> float:
        """Fit y = slope * x through the origin with a non-negative slope."""

        finite = [(x, y) for x, y in samples if math.isfinite(x) and math.isfinite(y) and x > 0.0 and y >= 0.0]
        if not finite:
            return 0.0
        x = torch.tensor([row[0] for row in finite], dtype=torch.float64)
        y = torch.tensor([row[1] for row in finite], dtype=torch.float64)
        denominator = float(torch.dot(x, x).item())
        if denominator <= 0.0:
            return 0.0
        return max(0.0, float(torch.dot(x, y).item()) / denominator)

    @staticmethod
    def _fit_nonnegative_linear_model(
        feature_rows: Sequence[Sequence[float]],
        targets: Sequence[float],
    ) -> tuple[tuple[float, ...], float]:
        """Fit a small non-negative affine model by enumerating active sets."""

        if len(feature_rows) != len(targets) or not feature_rows:
            raise ValueError("Linear cost-model fitting requires non-empty paired samples.")
        feature_count = len(feature_rows[0])
        if feature_count <= 0 or any(len(row) != feature_count for row in feature_rows):
            raise ValueError("Linear cost-model feature rows must have one consistent non-zero width.")
        finite_rows = [
            (tuple(float(value) for value in row), float(target))
            for row, target in zip(feature_rows, targets, strict=True)
            if all(math.isfinite(float(value)) and float(value) >= 0.0 for value in row)
            and math.isfinite(float(target))
            and float(target) >= 0.0
        ]
        if not finite_rows:
            return tuple(0.0 for _ in range(feature_count)), 0.0
        x = torch.tensor([row for row, _target in finite_rows], dtype=torch.float64)
        y = torch.tensor([target for _row, target in finite_rows], dtype=torch.float64)
        best: tuple[float, tuple[float, ...], float] | None = None
        # With at most three physical features, exact active-set enumeration is
        # deterministic and avoids introducing a SciPy dependency.
        for mask in range(1 << feature_count):
            active = [index for index in range(feature_count) if mask & (1 << index)]
            for use_intercept in (False, True):
                columns = [x[:, index] for index in active]
                if use_intercept:
                    columns.append(torch.ones_like(y))
                if columns:
                    design = torch.stack(columns, dim=1)
                    solution = torch.linalg.lstsq(design, y).solution
                    if bool((solution < 0.0).any().item()):
                        continue
                    fitted = design @ solution
                else:
                    solution = torch.empty((0,), dtype=torch.float64)
                    fitted = torch.zeros_like(y)
                coefficients = [0.0 for _ in range(feature_count)]
                for position, feature_index in enumerate(active):
                    coefficients[feature_index] = float(solution[position].item())
                intercept = float(solution[-1].item()) if use_intercept else 0.0
                error = float(torch.square(fitted - y).sum().item())
                candidate = (error, tuple(coefficients), intercept)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        return best[1], best[2]

    @staticmethod
    def _predict_linear_model(
        feature_rows: Sequence[Sequence[float]],
        coefficients: Sequence[float],
        intercept: float,
    ) -> list[float]:
        return [
            float(intercept)
            + sum(float(coefficient) * float(value) for coefficient, value in zip(coefficients, row, strict=True))
            for row in feature_rows
        ]

    @staticmethod
    def _cost_model_diagnostics(
        actual_values: Sequence[float],
        predicted_values: Sequence[float],
    ) -> dict[str, float]:
        """Return deterministic regression diagnostics for one modeled phase."""

        actual = torch.tensor(tuple(actual_values), dtype=torch.float64)
        predicted = torch.tensor(tuple(predicted_values), dtype=torch.float64)
        if actual.numel() == 0 or actual.shape != predicted.shape:
            raise ValueError("Cost-model diagnostics require non-empty paired samples.")
        residual = predicted - actual
        squared_error = torch.square(residual)
        centered = actual - actual.mean()
        total_variance = float(torch.square(centered).sum().item())
        r_squared = 1.0 - float(squared_error.sum().item()) / total_variance if total_variance > 0.0 else float("nan")
        relative = residual.abs() / actual.abs().clamp_min(1.0e-6)
        return {
            "r_squared": r_squared,
            "mape_percent": float(relative.mean().item()) * 100.0,
            "rmse_ms": float(torch.sqrt(squared_error.mean()).item()),
            "max_abs_error_ms": float(residual.abs().max().item()),
            "actual_min_ms": float(actual.min().item()),
            "actual_max_ms": float(actual.max().item()),
            "actual_mean_ms": float(actual.mean().item()),
            "predicted_min_ms": float(predicted.min().item()),
            "predicted_max_ms": float(predicted.max().item()),
            "predicted_mean_ms": float(predicted.mean().item()),
        }

    @torch.no_grad()
    def _cost_model_step_observations(
        self,
        layers: Sequence[ExpertLayerState],
        *,
        step: int,
    ) -> dict[str, Any]:
        """Aggregate exact Forward-route features and measured NPU times."""

        ordered_layers = sorted(layers, key=lambda value: value.key)
        if not ordered_layers:
            raise RuntimeError("Cost-model verification requires registered expert layers.")
        common_device = _local_tensor_view(ordered_layers[0].primary_parameter).device
        local_sample_counts = torch.tensor(
            [sum(int(timing.step) == int(step) for timing in layer.cost_model_timings) for layer in ordered_layers],
            dtype=torch.int64,
            device=common_device,
        )
        if self.ep_group is not None and self.ep_size > 1:
            gathered_counts = torch.empty(
                (self.ep_size * len(ordered_layers),),
                dtype=local_sample_counts.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_counts, local_sample_counts, group=self.ep_group)
            gathered_counts = gathered_counts.view(self.ep_size, len(ordered_layers))
        else:
            gathered_counts = local_sample_counts.view(1, -1)
        if bool((gathered_counts != gathered_counts[0]).any().item()):
            raise RuntimeError(
                f"Cost-model verification sample counts differ across EP ranks at step {step}: "
                f"{gathered_counts.detach().cpu().tolist()}."
            )
        if bool((local_sample_counts <= 0).any().item()):
            raise RuntimeError(
                f"Cost-model verification has missing layer samples at step {step}: "
                f"{local_sample_counts.detach().cpu().tolist()}."
            )

        synchronize()
        local_packed_rows: list[torch.Tensor] = []
        local_assignment_packed_rows: list[torch.Tensor] = []
        local_timing_rows: list[tuple[float, ...]] = []
        local_expert_token_rows: list[torch.Tensor] = []
        layer_row_ranges: list[tuple[GreedyCommunicationPlanner, int, int]] = []
        row_start = 0
        communication_event_names = (
            (
                "stage1_a2a",
                "stage2_a2a",
                "stage3_a2a",
                "combine_stage3_a2a",
                "combine_stage2_a2a",
                "combine_stage1_a2a",
            )
            if int(self.hierarchy.selected_dim) == 3
            else ("stage1_a2a", "stage2_a2a", "combine_stage2_a2a", "combine_stage1_a2a")
        )
        for layer in ordered_layers:
            timings = [timing for timing in layer.cost_model_timings if int(timing.step) == int(step)]
            if not all(
                timing.dispatch_start.elapsed_time(timing.dispatch_end) >= 0.0
                and timing.compute_start.elapsed_time(timing.compute_end) >= 0.0
                and timing.combine_start.elapsed_time(timing.combine_end) >= 0.0
                for timing in timings
            ):
                raise RuntimeError(f"Cost-model verification found an invalid timing event in {layer.key}.")
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=0.0,
                forward_compute_constant=0.0,
            )
            if not isinstance(planner, GreedyCommunicationPlanner):
                raise RuntimeError("Cost-model verification requires GreedyCommunicationPlanner.")
            routes = [timing.physical_routes for timing in timings]
            if all(route.shape == routes[0].shape for route in routes):
                stacked_routes = torch.stack(routes, dim=0)
                packed = planner._local_packed_counts(stacked_routes)
                assignment_packed = planner._local_packed_assignment_counts(stacked_routes)
            else:
                packed = torch.cat([planner._local_packed_counts(route) for route in routes], dim=0)
                assignment_packed = torch.cat(
                    [planner._local_packed_assignment_counts(route) for route in routes],
                    dim=0,
                )
            local_packed_rows.append(packed)
            local_assignment_packed_rows.append(assignment_packed)
            for timing in timings:
                stage_times = [-1.0 for _ in communication_event_names]
                if timing.communication_events is not None and all(
                    name in timing.communication_events for name in communication_event_names
                ):
                    stage_times = [
                        timing.communication_events[name][0].elapsed_time(timing.communication_events[name][1])
                        for name in communication_event_names
                    ]
                local_timing_rows.append(
                    (
                        timing.dispatch_start.elapsed_time(timing.dispatch_end)
                        + timing.combine_start.elapsed_time(timing.combine_end),
                        timing.compute_start.elapsed_time(timing.compute_end),
                        float(timing.local_assignment_count.item()),
                        *stage_times,
                    )
                )
                local_expert_token_rows.append(timing.local_expert_token_counts.to(dtype=torch.float32))
            row_end = row_start + len(timings)
            layer_row_ranges.append((planner, row_start, row_end))
            row_start = row_end

        local_packed = torch.cat(local_packed_rows, dim=0)
        local_assignment_packed = torch.cat(local_assignment_packed_rows, dim=0)
        if self.ep_group is not None and self.ep_size > 1:
            source_packed = torch.empty(
                (self.ep_size * row_start, local_packed.shape[1]),
                dtype=local_packed.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(source_packed, local_packed.contiguous(), group=self.ep_group)
            source_packed = source_packed.view(self.ep_size, row_start, local_packed.shape[1])
            source_assignment_packed = torch.empty_like(source_packed)
            dist.all_gather_into_tensor(
                source_assignment_packed,
                local_assignment_packed.contiguous(),
                group=self.ep_group,
            )
            source_assignment_packed = source_assignment_packed.view(
                self.ep_size,
                row_start,
                local_assignment_packed.shape[1],
            )
        else:
            source_packed = local_packed.unsqueeze(0)
            source_assignment_packed = local_assignment_packed.unsqueeze(0)
        global_packed = source_packed.sum(dim=0)

        communication_units = torch.empty(
            (row_start,),
            dtype=torch.float32,
            device=common_device,
        )
        receive_only_communication_units = torch.empty_like(communication_units)
        level_count = len(layer_row_ranges[0][0]._count_widths())
        source_send_maxima = torch.empty(
            (row_start, level_count),
            dtype=torch.float32,
            device=common_device,
        )
        destination_receive_maxima = torch.empty_like(source_send_maxima)
        traffic_features: dict[str, torch.Tensor] = {}
        for planner, start, end in layer_row_ranges:
            receive_only_communication_units[start:end] = planner._communication_cost_details(
                global_packed[start:end]
            )[1]
            (
                _communication,
                communication_units[start:end],
                source_send_maxima[start:end],
                destination_receive_maxima[start:end],
                _selected_dim,
            ) = planner._source_aware_communication_cost_details(source_packed[:, start:end])
            layer_features = planner._hierarchical_traffic_features(
                source_packed[:, start:end],
                source_assignment_packed[:, start:end],
            )
            for name, values in layer_features.items():
                target = traffic_features.get(name)
                if target is None:
                    target = torch.empty((row_start,), dtype=torch.float32, device=common_device)
                    traffic_features[name] = target
                target[start:end] = values

        local_timings = torch.tensor(local_timing_rows, dtype=torch.float32, device=common_device)
        local_expert_tokens = torch.stack(local_expert_token_rows).to(device=common_device, dtype=torch.float32)
        if self.ep_group is not None and self.ep_size > 1:
            gathered_timings = torch.empty(
                (self.ep_size * row_start, local_timings.shape[1]),
                dtype=local_timings.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_timings, local_timings, group=self.ep_group)
            gathered_timings = gathered_timings.view(self.ep_size, row_start, local_timings.shape[1])
            gathered_expert_tokens = torch.empty(
                (self.ep_size * row_start, local_expert_tokens.shape[1]),
                dtype=local_expert_tokens.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_expert_tokens, local_expert_tokens.contiguous(), group=self.ep_group)
            gathered_expert_tokens = gathered_expert_tokens.view(self.ep_size, row_start, local_expert_tokens.shape[1])
        else:
            gathered_timings = local_timings.view(1, row_start, local_timings.shape[1])
            gathered_expert_tokens = local_expert_tokens.view(1, row_start, local_expert_tokens.shape[1])

        actual_communication = gathered_timings[:, :, 0].max(dim=0).values
        actual_compute = gathered_timings[:, :, 1].max(dim=0).values
        peak_assignments = gathered_timings[:, :, 2].max(dim=0).values
        stage_timings = gathered_timings[:, :, 3:]
        raw_a2a_available = bool((stage_timings >= 0.0).all().item())
        actual_stage_a2a = (
            stage_timings.max(dim=0).values if raw_a2a_available else torch.empty((0, 0), device=common_device)
        )
        actual_raw_a2a = actual_stage_a2a.sum(dim=1) if raw_a2a_available else torch.empty((0,), device=common_device)
        return {
            "sample_count": row_start,
            "compute_fit_sample_count": int(self.ep_size * row_start),
            "communication_units": communication_units.detach().cpu().tolist(),
            "receive_only_communication_units": receive_only_communication_units.detach().cpu().tolist(),
            "communication_level_names": [
                "rank",
                *[
                    f"group_{int(size)}"
                    for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
                ],
            ],
            "source_send_maxima": source_send_maxima.detach().cpu().tolist(),
            "destination_receive_maxima": destination_receive_maxima.detach().cpu().tolist(),
            "traffic_features": {name: values.detach().cpu().tolist() for name, values in traffic_features.items()},
            "peak_assignments": peak_assignments.detach().cpu().tolist(),
            "actual_communication_ms": actual_communication.detach().cpu().tolist(),
            "actual_compute_ms": actual_compute.detach().cpu().tolist(),
            "actual_raw_a2a_ms": actual_raw_a2a.detach().cpu().tolist(),
            "actual_stage_a2a_ms": actual_stage_a2a.detach().cpu().tolist(),
            "actual_stage_a2a_names": list(communication_event_names),
            "paired_expert_token_counts": gathered_expert_tokens.reshape(-1, gathered_expert_tokens.shape[-1])
            .detach()
            .cpu()
            .tolist(),
            "paired_assignments": gathered_timings[:, :, 2].reshape(-1).detach().cpu().tolist(),
            "paired_compute_ms": gathered_timings[:, :, 1].reshape(-1).detach().cpu().tolist(),
        }

    def _record_cost_model_report(self, phase: str, report: dict[str, Any]) -> None:
        prefix = f"hiermoe/cost_model_{phase}"
        self._accumulate_metric(f"{prefix}_samples", int(report["sample_count"]))
        self._accumulate_metric(f"{prefix}_communication_r2", float(report["communication"]["r_squared"]))
        self._accumulate_metric(
            f"{prefix}_communication_mape_percent",
            float(report["communication"]["mape_percent"]),
        )
        self._accumulate_metric(
            f"{prefix}_receive_only_communication_r2",
            float(report["receive_only_communication"]["r_squared"]),
        )
        self._accumulate_metric(
            f"{prefix}_receive_only_communication_mape_percent",
            float(report["receive_only_communication"]["mape_percent"]),
        )
        self._accumulate_metric(f"{prefix}_compute_r2", float(report["compute"]["r_squared"]))
        self._accumulate_metric(f"{prefix}_compute_mape_percent", float(report["compute"]["mape_percent"]))
        self._accumulate_metric(f"{prefix}_joint_r2", float(report["joint"]["r_squared"]))
        self._accumulate_metric(f"{prefix}_joint_mape_percent", float(report["joint"]["mape_percent"]))
        network_joint_models = report.get("traffic_feature_models", {}).get("network_joint", {})
        if network_joint_models:
            best_network_joint = min(
                network_joint_models.values(),
                key=lambda row: (
                    float(row["mape_percent"]),
                    -float(row["r_squared"]),
                ),
            )
            self._accumulate_metric(
                f"{prefix}_network_joint_r2",
                float(best_network_joint["r_squared"]),
            )
            self._accumulate_metric(
                f"{prefix}_network_joint_mape_percent",
                float(best_network_joint["mape_percent"]),
            )
        self._accumulate_metric(
            f"{prefix}_communication_units_min",
            float(report["communication_units"]["min"]),
        )
        self._accumulate_metric(
            f"{prefix}_communication_units_max",
            float(report["communication_units"]["max"]),
        )
        self._accumulate_metric(
            f"{prefix}_peak_assignments_min",
            float(report["peak_assignments"]["min"]),
        )
        self._accumulate_metric(
            f"{prefix}_peak_assignments_max",
            float(report["peak_assignments"]["max"]),
        )
        logger.info_rank0("HierMoE cost model %s report: %s", phase, json.dumps(report, sort_keys=True))

    @torch.no_grad()
    def _run_cost_model_verification(self, layers: Sequence[ExpertLayerState], step: int) -> str:
        calibration_step = int(self._online_freeze_calibration_step)
        validation_end_step = calibration_step + int(self._cost_model_validation_steps)
        if int(step) < calibration_step or int(step) > validation_end_step:
            self.latest_pair = "none"
            return self.latest_pair

        started = time.perf_counter()
        observations = self._cost_model_step_observations(layers, step=int(step))
        communication_units = list(observations["communication_units"])
        receive_only_communication_units = list(
            observations.get("receive_only_communication_units", communication_units)
        )
        peak_assignments = list(observations["peak_assignments"])
        actual_communication = list(observations["actual_communication_ms"])
        actual_compute = list(observations["actual_compute_ms"])
        actual_raw_a2a = list(observations.get("actual_raw_a2a_ms", ()))
        traffic_features = {
            str(name): list(values) for name, values in dict(observations.get("traffic_features", {})).items()
        }
        base_feature_models = {
            "legacy_source_aware": ("legacy_source_aware",),
            "stage_unique_endpoint": ("stage_unique_endpoint_link_units",),
            "stage_payload_endpoint": ("stage_payload_endpoint_link_units",),
            "stage_payload_endpoint_edge": (
                "stage_payload_endpoint_link_units",
                "stage_payload_edge_link_units",
            ),
            "stage_payload_inter_intra": (
                "stage1_payload_endpoint_bytes",
                "stage2_payload_endpoint_bytes",
            ),
            "stage_payload_levels": (
                "stage1_payload_endpoint_bytes",
                "stage2_payload_endpoint_bytes",
                "stage3_payload_endpoint_bytes",
            ),
            "stage_payload_lane_shared_node": (
                "stage_payload_endpoint_link_units",
                "stage_shared_node_endpoint_link_units",
            ),
            "stage_remote_payload_endpoint": ("stage_remote_payload_endpoint_link_units",),
            "stage_remote_payload_endpoint_self": (
                "stage_remote_payload_endpoint_link_units",
                "stage_self_payload_link_units",
            ),
            "stage_remote_payload_endpoint_edge": (
                "stage_remote_payload_endpoint_link_units",
                "stage_remote_payload_edge_link_units",
            ),
        }
        feature_values = {"legacy_source_aware": communication_units, **traffic_features}
        base_feature_models = {
            name: features
            for name, features in base_feature_models.items()
            if all(feature in feature_values for feature in features)
        }
        model_rows = {
            name: [
                [float(feature_values[feature][row]) for feature in features]
                for row in range(len(actual_communication))
            ]
            for name, features in base_feature_models.items()
        }
        joint_model_rows = {
            name: [[*row, float(peak_assignments[index])] for index, row in enumerate(rows)]
            for name, rows in model_rows.items()
        }

        if int(step) == calibration_step:
            communication_slope, communication_constant = self._fit_nonnegative_compute_model(
                list(zip(communication_units, actual_communication, strict=True))
            )
            compute_slope, compute_constant = self._fit_nonnegative_compute_model(
                list(
                    zip(
                        observations["paired_assignments"],
                        observations["paired_compute_ms"],
                        strict=True,
                    )
                )
            )
            self._cost_model_verify_coefficients = (
                communication_slope,
                communication_constant,
                compute_slope,
                compute_constant,
            )
            self._cost_model_verify_receive_only_coefficients = self._fit_nonnegative_compute_model(
                list(zip(receive_only_communication_units, actual_communication, strict=True))
            )
            actual_joint_targets = [
                communication + compute
                for communication, compute in zip(actual_communication, actual_compute, strict=True)
            ]
            feature_coefficients: dict[str, dict[str, tuple[tuple[float, ...], float]]] = {
                "communication": {
                    name: self._fit_nonnegative_linear_model(rows, actual_communication)
                    for name, rows in model_rows.items()
                },
                "joint": {
                    name: self._fit_nonnegative_linear_model(rows, actual_joint_targets)
                    for name, rows in joint_model_rows.items()
                },
            }
            if actual_raw_a2a:
                feature_coefficients["raw_a2a"] = {
                    name: self._fit_nonnegative_linear_model(rows, actual_raw_a2a) for name, rows in model_rows.items()
                }
                # The placement objective is network A2A plus expert compute,
                # not the wider dispatch/combine region. Compose the two
                # independently calibrated models so held-out validation
                # measures exactly the objective consumed by the layout
                # planner. Keep the wider ``joint`` target above as a
                # diagnostic for local remap/pack and rank-arrival overheads.
                feature_coefficients["network_joint"] = {
                    name: (
                        (*raw_coefficients, compute_slope),
                        raw_intercept + compute_constant,
                    )
                    for name, (raw_coefficients, raw_intercept) in feature_coefficients["raw_a2a"].items()
                }
            self._cost_model_verify_feature_coefficients = feature_coefficients
            phase = "calibration"
        else:
            if self._cost_model_verify_coefficients is None:
                raise RuntimeError("Cost-model validation has no coefficients from the calibration step.")
            if self._cost_model_verify_receive_only_coefficients is None:
                raise RuntimeError("Cost-model validation has no receive-only coefficients from the calibration step.")
            if self._cost_model_verify_feature_coefficients is None:
                raise RuntimeError("Cost-model validation has no traffic-feature coefficients.")
            communication_slope, communication_constant, compute_slope, compute_constant = (
                self._cost_model_verify_coefficients
            )
            phase = "validation"

        assert self._cost_model_verify_receive_only_coefficients is not None
        receive_only_slope, receive_only_constant = self._cost_model_verify_receive_only_coefficients
        predicted_communication = [
            communication_slope * value + communication_constant for value in communication_units
        ]
        receive_only_predicted_communication = [
            receive_only_slope * value + receive_only_constant for value in receive_only_communication_units
        ]
        predicted_compute = [compute_slope * value + compute_constant for value in peak_assignments]
        actual_joint = [
            communication + compute
            for communication, compute in zip(actual_communication, actual_compute, strict=True)
        ]
        predicted_joint = [
            communication + compute
            for communication, compute in zip(predicted_communication, predicted_compute, strict=True)
        ]
        assert self._cost_model_verify_feature_coefficients is not None
        feature_model_report: dict[str, dict[str, Any]] = {}
        target_rows: dict[str, tuple[dict[str, list[list[float]]], list[float]]] = {
            "communication": (model_rows, actual_communication),
            "joint": (
                joint_model_rows,
                [
                    communication + compute
                    for communication, compute in zip(actual_communication, actual_compute, strict=True)
                ],
            ),
        }
        if actual_raw_a2a and "raw_a2a" in self._cost_model_verify_feature_coefficients:
            target_rows["raw_a2a"] = (model_rows, actual_raw_a2a)
        if actual_raw_a2a and "network_joint" in self._cost_model_verify_feature_coefficients:
            target_rows["network_joint"] = (
                joint_model_rows,
                [raw_a2a + compute for raw_a2a, compute in zip(actual_raw_a2a, actual_compute, strict=True)],
            )
        for target_name, (rows_by_model, targets) in target_rows.items():
            target_report: dict[str, Any] = {}
            for model_name, rows in rows_by_model.items():
                coefficients, intercept = self._cost_model_verify_feature_coefficients[target_name][model_name]
                predicted = self._predict_linear_model(rows, coefficients, intercept)
                target_report[model_name] = {
                    "feature_names": [
                        *base_feature_models[model_name],
                        *(["peak_assignments"] if target_name in {"joint", "network_joint"} else []),
                    ],
                    "coefficients": list(coefficients),
                    "intercept_ms": float(intercept),
                    **self._cost_model_diagnostics(targets, predicted),
                }
            feature_model_report[target_name] = target_report
        report: dict[str, Any] = {
            "step": int(step),
            "sample_count": int(observations["sample_count"]),
            "compute_fit_sample_count": int(observations["compute_fit_sample_count"]),
            "coefficients": {
                "communication_ms_per_model_unit": communication_slope,
                "communication_constant_ms": communication_constant,
                "compute_ms_per_assignment": compute_slope,
                "compute_constant_ms": compute_constant,
                "receive_only_communication_ms_per_model_unit": receive_only_slope,
                "receive_only_communication_constant_ms": receive_only_constant,
            },
            "communication_units": {
                "min": min(communication_units),
                "max": max(communication_units),
                "mean": sum(communication_units) / len(communication_units),
            },
            "peak_assignments": {
                "min": min(peak_assignments),
                "max": max(peak_assignments),
                "mean": sum(peak_assignments) / len(peak_assignments),
            },
            "communication": self._cost_model_diagnostics(actual_communication, predicted_communication),
            "receive_only_communication": self._cost_model_diagnostics(
                actual_communication,
                receive_only_predicted_communication,
            ),
            "compute": self._cost_model_diagnostics(actual_compute, predicted_compute),
            "joint": self._cost_model_diagnostics(actual_joint, predicted_joint),
            "traffic_feature_models": feature_model_report,
            "traffic_feature_ranges": {
                name: {
                    "min": min(float(value) for value in values),
                    "max": max(float(value) for value in values),
                    "mean": sum(float(value) for value in values) / len(values),
                }
                for name, values in traffic_features.items()
            },
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        if _EXPORT_COST_MODEL_SAMPLES:
            paired_assignments = [float(value) for value in observations["paired_assignments"]]
            paired_compute = [float(value) for value in observations["paired_compute_ms"]]
            actual_stage_rows = [
                [float(value) for value in row] for row in observations.get("actual_stage_a2a_ms", ())
            ]
            offline_samples = {
                "paired_expert_token_counts": observations["paired_expert_token_counts"],
                "paired_assignments": observations["paired_assignments"],
                "paired_compute_ms": observations["paired_compute_ms"],
                "peak_assignments": observations["peak_assignments"],
                "actual_communication_ms": observations["actual_communication_ms"],
                **{
                    name: [float(value) for value in values]
                    for name, values in traffic_features.items()
                    if name.startswith("stage") and name.endswith("_payload_endpoint_bytes")
                },
            }
            if int(self.hierarchy.selected_dim) == 3:
                if any(len(row) != 6 for row in actual_stage_rows):
                    raise RuntimeError("Three-stage cost samples require six dispatch/combine A2A timings.")
                offline_samples.update(
                    {
                        "actual_stage1_a2a_ms": [row[0] + row[5] for row in actual_stage_rows],
                        "actual_stage2_a2a_ms": [row[1] + row[4] for row in actual_stage_rows],
                        "actual_stage3_a2a_ms": [row[2] + row[3] for row in actual_stage_rows],
                    }
                )
            else:
                if any(len(row) != 4 for row in actual_stage_rows):
                    raise RuntimeError("Two-stage cost samples require four dispatch/combine A2A timings.")
                offline_samples.update(
                    {
                        "actual_stage1_a2a_ms": [row[0] + row[3] for row in actual_stage_rows],
                        "actual_stage2_a2a_ms": [row[1] + row[2] for row in actual_stage_rows],
                    }
                )
            report["offline_scorer_samples"] = offline_samples
            sample_model = (
                "stage_payload_levels" if int(self.hierarchy.selected_dim) == 3 else "stage_payload_inter_intra"
            )
            report["sample_data"] = {
                "feature_values": {
                    **{name: [float(value) for value in values] for name, values in traffic_features.items()},
                    "peak_assignments": [float(value) for value in peak_assignments],
                },
                "compute": {
                    "assignments": paired_assignments,
                    "measured_ms": paired_compute,
                    "predicted_ms": [compute_slope * value + compute_constant for value in paired_assignments],
                },
                "communication_region": {
                    "feature_model": sample_model,
                    "measured_ms": [float(value) for value in actual_communication],
                    "predicted_ms": self._predict_linear_model(
                        model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["communication"][sample_model],
                    ),
                },
                "joint_moe_region": {
                    "feature_model": sample_model,
                    "measured_ms": [float(value) for value in actual_joint],
                    "predicted_ms": self._predict_linear_model(
                        joint_model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["joint"][sample_model],
                    ),
                },
            }
            if actual_raw_a2a and "network_joint" in self._cost_model_verify_feature_coefficients:
                report["sample_data"]["network_joint"] = {
                    "feature_model": sample_model,
                    "measured_ms": [
                        float(raw_a2a + compute)
                        for raw_a2a, compute in zip(
                            actual_raw_a2a,
                            actual_compute,
                            strict=True,
                        )
                    ],
                    "predicted_ms": self._predict_linear_model(
                        joint_model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["network_joint"][sample_model],
                    ),
                }
        if actual_raw_a2a:
            report["raw_a2a_ms"] = {
                "min": min(actual_raw_a2a),
                "max": max(actual_raw_a2a),
                "mean": sum(actual_raw_a2a) / len(actual_raw_a2a),
                "stage_names": list(observations.get("actual_stage_a2a_names", ())),
            }
        level_names = list(observations.get("communication_level_names", ()))
        source_send_rows = list(observations.get("source_send_maxima", ()))
        destination_receive_rows = list(observations.get("destination_receive_maxima", ()))
        if level_names and source_send_rows and destination_receive_rows:
            report["communication_bottlenecks"] = {
                level: {
                    "source_send_min": min(float(row[level_index]) for row in source_send_rows),
                    "source_send_max": max(float(row[level_index]) for row in source_send_rows),
                    "source_send_mean": sum(float(row[level_index]) for row in source_send_rows)
                    / len(source_send_rows),
                    "destination_receive_min": min(float(row[level_index]) for row in destination_receive_rows),
                    "destination_receive_max": max(float(row[level_index]) for row in destination_receive_rows),
                    "destination_receive_mean": sum(float(row[level_index]) for row in destination_receive_rows)
                    / len(destination_receive_rows),
                    "source_dominant_samples": sum(
                        float(source_row[level_index]) > float(destination_row[level_index])
                        for source_row, destination_row in zip(
                            source_send_rows,
                            destination_receive_rows,
                            strict=True,
                        )
                    ),
                }
                for level_index, level in enumerate(level_names)
            }
        self._record_cost_model_report(phase, report)
        for layer in layers:
            layer.cost_model_timings = [timing for timing in layer.cost_model_timings if int(timing.step) != int(step)]
        if int(step) == validation_end_step:
            self._cost_model_verify_complete = True
        self.latest_pair = "none"
        return self.latest_pair

    @torch.no_grad()
    def _prepare_online_freeze_calibrations(
        self,
        layers: Sequence[ExpertLayerState],
        *,
        step: int,
        started: float,
    ) -> None:
        """Validate offline traffic coefficients and fit online GEMM cost."""

        ordered_layers = sorted(layers, key=lambda value: value.key)
        records = [
            (layer, layer.pending_timing)
            for layer in ordered_layers
            if layer.pending_timing is not None
            and layer.pending_timing.step == int(step)
            and self._events_ready(layer.pending_timing)
        ]
        if len(records) != len(ordered_layers):
            self._accumulate_metric(
                "hiermoe/placement_calibration_ms",
                (time.perf_counter() - started) * 1000.0,
            )
            return

        has_full_samples = all(
            any(int(timing.step) == int(step) for timing in layer.cost_model_timings) for layer in ordered_layers
        )
        communication_samples = 0
        compute_samples: list[tuple[float, float]]
        communication_diagnostics: dict[str, float] | None = None
        compute_diagnostics: dict[str, float] | None = None
        joint_diagnostics: dict[str, float] | None = None
        traffic_scale = 1.0
        traffic_constant = self._online_freeze_traffic_intercept_ms
        traffic_predictors: list[float] = []
        if has_full_samples:
            observations = self._cost_model_step_observations(ordered_layers, step=int(step))
            traffic_features = dict(observations["traffic_features"])
            stage1 = [float(value) for value in traffic_features["stage1_payload_endpoint_bytes"]]
            stage2 = [float(value) for value in traffic_features["stage2_payload_endpoint_bytes"]]
            peak_assignments = [float(value) for value in observations["peak_assignments"]]
            actual_communication = [float(value) for value in observations["actual_communication_ms"]]
            route_coefficient = (
                self._online_freeze_route_ms_per_assignment if self._online_freeze_cost_mode == "joint" else 0.0
            )
            traffic_predictors = [
                self._online_freeze_inter_ms_per_byte * inter_bytes
                + self._online_freeze_intra_ms_per_byte * intra_bytes
                + route_coefficient * assignments
                for inter_bytes, intra_bytes, assignments in zip(
                    stage1,
                    stage2,
                    peak_assignments,
                    strict=True,
                )
            ]
            traffic_scale = self._fit_positive_through_origin(
                [
                    (predictor, max(0.0, actual - traffic_constant))
                    for predictor, actual in zip(
                        traffic_predictors,
                        actual_communication,
                        strict=True,
                    )
                ]
            )
            if traffic_scale <= 0.0:
                traffic_scale = self._fit_positive_through_origin(
                    list(zip(traffic_predictors, actual_communication, strict=True))
                )
                traffic_constant = 0.0
            predicted_communication = [
                traffic_scale * predictor + traffic_constant for predictor in traffic_predictors
            ]
            communication_diagnostics = self._cost_model_diagnostics(
                actual_communication,
                predicted_communication,
            )
            communication_samples = len(actual_communication)
            compute_samples = [
                (float(assignments), float(compute_ms))
                for assignments, compute_ms in zip(
                    observations["paired_assignments"],
                    observations["paired_compute_ms"],
                    strict=True,
                )
                if float(assignments) > 0.0
            ]
        else:
            # Unit tests and legacy callers may provide one pending sample per
            # layer without the full microbatch route capture.
            compute_samples = []
            for _layer, timing in records:
                assert timing is not None
                local_values = torch.stack(
                    (
                        timing.local_assignment_count.to(dtype=torch.float32),
                        torch.tensor(
                            timing.compute_start.elapsed_time(timing.compute_end),
                            dtype=torch.float32,
                            device=timing.local_assignment_count.device,
                        ),
                    )
                )
                if self.ep_group is not None and self.ep_size > 1:
                    gathered_flat = torch.empty(
                        (self.ep_size * int(local_values.numel()),),
                        dtype=local_values.dtype,
                        device=local_values.device,
                    )
                    dist.all_gather_into_tensor(gathered_flat, local_values, group=self.ep_group)
                    gathered = gathered_flat.view(self.ep_size, int(local_values.numel()))
                else:
                    gathered = local_values.view(1, -1)
                compute_samples.extend(
                    (float(row[0].item()), float(row[1].item())) for row in gathered if float(row[0].item()) > 0.0
                )

        compute_slope, compute_constant = self._fit_nonnegative_compute_model(compute_samples)
        if compute_slope <= 0.0:
            compute_slope = self._fit_positive_through_origin(compute_samples)
            compute_constant = 0.0
        compute_diagnostics = self._cost_model_diagnostics(
            [target for _assignments, target in compute_samples],
            [compute_slope * assignments + compute_constant for assignments, _target in compute_samples],
        )

        if has_full_samples:
            actual_communication = [float(value) for value in observations["actual_communication_ms"]]
            actual_compute = [float(value) for value in observations["actual_compute_ms"]]
            predicted_communication = [
                traffic_scale * predictor + traffic_constant for predictor in traffic_predictors
            ]
            predicted_compute = [
                compute_slope * float(assignments) + compute_constant
                for assignments in observations["peak_assignments"]
            ]
            joint_diagnostics = self._cost_model_diagnostics(
                [
                    communication + compute
                    for communication, compute in zip(actual_communication, actual_compute, strict=True)
                ],
                [
                    communication + compute
                    for communication, compute in zip(predicted_communication, predicted_compute, strict=True)
                ],
            )

        if self._online_freeze_cost_mode == "joint":
            planner_compute_slope = compute_slope
            planner_compute_constant = compute_constant
        else:
            planner_compute_slope = 0.0
            planner_compute_constant = 0.0

        for layer, timing in records:
            assert timing is not None
            layer.planner_calibration = _PlannerCalibration(
                source_step=timing.step,
                communication_scale=traffic_scale,
                forward_compute_per_assignment=planner_compute_slope,
                forward_compute_constant=planner_compute_constant,
            )
            layer.pending_timing = None
            layer.cost_model_timings = [sample for sample in layer.cost_model_timings if int(sample.step) != int(step)]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_calibration_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_calibrated_layers", len(records))
        self._accumulate_metric("hiermoe/placement_calibration_communication_samples", communication_samples)
        self._accumulate_metric("hiermoe/placement_calibration_compute_samples", len(compute_samples))
        self._accumulate_metric("hiermoe/placement_traffic_online_scale", traffic_scale)
        self._accumulate_metric("hiermoe/placement_traffic_online_constant_ms", traffic_constant)
        if traffic_predictors:
            self._accumulate_metric("hiermoe/placement_traffic_predictor_min_ms", min(traffic_predictors))
            self._accumulate_metric("hiermoe/placement_traffic_predictor_max_ms", max(traffic_predictors))
        self._accumulate_metric("hiermoe/placement_forward_compute_ms_per_assignment", compute_slope)
        self._accumulate_metric("hiermoe/placement_forward_compute_constant_ms", compute_constant)
        self._accumulate_metric(
            "hiermoe/placement_full_compute_ms_per_assignment",
            self._online_freeze_compute_ratio * compute_slope,
        )
        if communication_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_communication_r2",
                communication_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_communication_mape_percent",
                communication_diagnostics["mape_percent"],
            )
        if compute_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_compute_r2",
                compute_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_compute_mape_percent",
                compute_diagnostics["mape_percent"],
            )
        if joint_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_joint_r2",
                joint_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_joint_mape_percent",
                joint_diagnostics["mape_percent"],
            )
        if communication_diagnostics is not None and communication_diagnostics["mape_percent"] > 5.0:
            logger.warning_rank0(
                "Online-freeze per-layer E2E traffic timings are too noisy for action-level validation: "
                "MAPE=%.3f%% exceeds 5%%; "
                f"R2={communication_diagnostics['r_squared']:.6f}, "
                f"RMSE={communication_diagnostics['rmse_ms']:.3f} ms, "
                f"max_abs={communication_diagnostics['max_abs_error_ms']:.3f} ms, "
                f"online_scale={traffic_scale:.9g}, intercept={traffic_constant:.3f} ms, "
                f"predictor_range="
                f"[{min(traffic_predictors, default=0.0):.3f}, "
                f"{max(traffic_predictors, default=0.0):.3f}] ms. "
                "Keeping the offline multi-layout feature ratios and validating the frozen winner by E2E.",
                communication_diagnostics["mape_percent"],
            )

    @torch.no_grad()
    def prepare_calibrations(self, step: int) -> None:
        started = time.perf_counter()
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            uncalibrated = [layer for layer in self.layers.values() if layer.planner_calibration is None]
            if not self.layers:
                return
            consensus_device = _local_tensor_view(next(iter(self.layers.values())).primary_parameter).device
            all_need_calibration, need_state_agrees = _placement_group_boolean_consensus(
                bool(uncalibrated),
                device=consensus_device,
                ep_size=self.ep_size,
                ep_group=self.ep_group,
            )
            if not need_state_agrees:
                raise RuntimeError("HierMoE planner calibration state differs across the EP group.")
            if not all_need_calibration:
                return
            local_ready = not any(
                layer.pending_timing is None
                or layer.pending_timing.step > int(step)
                or not self._events_ready(layer.pending_timing)
                for layer in uncalibrated
            )
            all_ready, _ready_state_agrees = _placement_group_boolean_consensus(
                local_ready,
                device=consensus_device,
                ep_size=self.ep_size,
                ep_group=self.ep_group,
            )
            if not all_ready:
                self._accumulate_metric(
                    "hiermoe/placement_calibration_ms",
                    (time.perf_counter() - started) * 1000.0,
                )
                return
            if self._online_freeze_cost_mode != "off":
                self._prepare_online_freeze_calibrations(
                    uncalibrated,
                    step=int(step),
                    started=started,
                )
                return
        elif not self.layer_calibration_enabled():
            return
        updated = 0
        greedy_records: list[tuple[ExpertLayerState, _PendingLayerTiming, float, float, float]] = []
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
                forward_compute_constant=0.0,
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
                    timing.local_assignment_count,
                ],
                dtype=torch.float32,
                device=selected.device,
            )
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.ep_group)
            communication_units = reference.communication_model_units
            peak_assignments = float(values[2].item())
            forward_communication_ms = float(values[0].item())
            forward_compute_ms = float(values[1].item())
            if communication_units <= 0.0 or peak_assignments <= 0.0:
                continue
            if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
                # The greedy planner explicitly accounts for four communication
                # phases. Normalize the measured forward dispatch+combine pair
                # to the residual scale of one modeled phase.
                communication_scale = forward_communication_ms / (2.0 * communication_units)
            else:
                # CurrentRoutePlanner.communication_model_units already includes
                # all four communication phases.
                communication_scale = (2.0 * forward_communication_ms) / communication_units
            if not math.isfinite(communication_scale):
                continue
            if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
                greedy_records.append((layer, timing, communication_scale, peak_assignments, forward_compute_ms))
                continue
            compute_scale = forward_compute_ms / peak_assignments
            if not math.isfinite(compute_scale):
                continue
            layer.planner_calibration = _PlannerCalibration(
                source_step=timing.step,
                communication_scale=communication_scale,
                forward_compute_per_assignment=compute_scale,
            )
            layer.pending_timing = None
            updated += 1
        if greedy_records:
            compute_scale, compute_constant = self._fit_nonnegative_compute_model(
                [
                    (peak_assignments, forward_compute_ms)
                    for _, _, _, peak_assignments, forward_compute_ms in greedy_records
                ]
            )
            for layer, timing, communication_scale, _peak_assignments, _forward_compute_ms in greedy_records:
                layer.planner_calibration = _PlannerCalibration(
                    source_step=timing.step,
                    communication_scale=communication_scale,
                    forward_compute_per_assignment=compute_scale,
                    forward_compute_constant=compute_constant,
                )
                layer.pending_timing = None
                updated += 1
            self._accumulate_metric("hiermoe/placement_compute_ms_per_assignment", compute_scale)
            self._accumulate_metric("hiermoe/placement_compute_constant_ms", compute_constant)
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
        transfer_group: dist.ProcessGroup | None = None,
        force_staged_transfer: bool = False,
        fast_sparse_transfer: bool = False,
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
            param for param in layer.expert_parameters if getattr(param, "grad", None) is None
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
            and not force_staged_transfer
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
                if fast_sparse_transfer:
                    with _placement_timing_range(timing_prefix, "transfer"):
                        self._execute_swap_plan_batch(swap_plans)
                elif self.expert_swap_mode == "layer":
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
                    zero_entry_groups = tuple(
                        (dst_slot // layer.num_local_experts, zero_entries[dst_slot]) for dst_slot in zero_slots
                    )
                    if fast_sparse_transfer:
                        self._execute_sparse_group_slot_transfers(
                            grouped_entries,
                            zero_entry_groups=zero_entry_groups,
                            process_group=self.ep_group if transfer_group is None else transfer_group,
                        )
                    else:
                        _cover_grouped_slot_entries_atomic(
                            grouped_entries,
                            self.ep_rank,
                            self.ep_size,
                            self.ep_group if transfer_group is None else transfer_group,
                            zero_entry_groups=zero_entry_groups,
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
            layer.placement_version += int(bool(plan.actions))
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

        _expert_swap_diag_phase("exact_p1_stats_start")
        with _full_timing_range("hiermoe_exact_p1_stats"):
            stats_started = time.perf_counter()
            local_rows: list[torch.Tensor] = []
            row_lengths: list[int] = []
            group_rows: list[list[int]] = []
            group_mappings: list[list[torch.Tensor]] = []
            common_device: torch.device | None = None
            population_token_total = 0
            sampled_token_total = 0

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
                    else _local_tensor_view(layer.primary_parameter).device
                )
                if common_device is None:
                    common_device = layer_device
                elif layer_device != common_device:
                    raise ValueError("hiermoe_exact_p1 requires all planned layers to reside on one device.")

                if selected is None or selected.numel() == 0:
                    selected = torch.empty((0, 1), dtype=torch.long, device=layer_device)
                selected = selected.to(device=layer_device, dtype=torch.long, non_blocking=True)
                population_tokens = int(selected.shape[0])
                population_token_total += population_tokens
                population_scale = 1.0
                if _EXACT_P1_ROUTE_SAMPLE_SIZE > 0 and population_tokens > _EXACT_P1_ROUTE_SAMPLE_SIZE:
                    sample_indices = _deterministic_sample_indices(
                        population_tokens,
                        _EXACT_P1_ROUTE_SAMPLE_SIZE,
                        source_rank=self.ep_rank,
                        step=int(step),
                        layer_seed=zlib.crc32(layer.key.encode("utf-8")) & 0x7FFFFFFF,
                        device=layer_device,
                    )
                    selected = selected.index_select(0, sample_indices)
                    population_scale = float(population_tokens) / float(selected.shape[0])
                sampled_token_total += int(selected.shape[0])
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
                if population_scale != 1.0:
                    flat_row.mul_(population_scale)
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
        _expert_swap_diag_phase("exact_p1_stats_done")

        _expert_swap_diag_phase("exact_p1_collective_start")
        with _full_timing_range("hiermoe_exact_p1_collective"):
            collective_started = time.perf_counter()
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(global_stats, op=dist.ReduceOp.SUM, group=self.ep_group)
            collective_ms = (time.perf_counter() - collective_started) * 1000.0
        _expert_swap_diag_phase("exact_p1_collective_done")

        _expert_swap_diag_phase("exact_p1_score_start")
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
        _expert_swap_diag_phase("exact_p1_score_done")

        self._accumulate_metric("hiermoe/exact_p1_stats_ms", stats_ms)
        self._accumulate_metric("hiermoe/exact_p1_collective_ms", collective_ms)
        self._accumulate_metric("hiermoe/exact_p1_score_ms", score_ms)
        self._accumulate_metric("hiermoe/exact_p1_candidate_count", candidate_total)
        self._accumulate_metric("hiermoe/exact_p1_accepted_count", accepted)
        self._accumulate_metric("hiermoe/exact_p1_population_tokens", population_token_total)
        self._accumulate_metric("hiermoe/exact_p1_sampled_tokens", sampled_token_total)

        committed: list[str] = []
        _expert_swap_diag_phase("exact_p1_execute_start")
        for layer, chosen_pair in selections:
            result = self._execute_exact_single_swap(layer, chosen_pair, timing_prefix="hiermoe_exact_p1")
            if result is not None:
                committed.append(result)
        _expert_swap_diag_phase("exact_p1_execute_done")
        return committed

    @torch.no_grad()
    def _plan_current_layer(self, layer: ExpertLayerState, step: int) -> list[str]:
        calibration = layer.planner_calibration
        selected = layer.latest_selected_experts
        greedy_cover = self.expert_swap_selector == "hiermoe_greedy_cover_p1"
        if (
            selected is None
            or (selected.numel() == 0 and not greedy_cover)
            or (calibration is None and not greedy_cover)
        ):
            return []
        planner = self._planner_for_layer(
            layer,
            communication_scale=1.0 if calibration is None else calibration.communication_scale,
            forward_compute_per_assignment=0.0 if calibration is None else calibration.forward_compute_per_assignment,
            forward_compute_constant=0.0 if calibration is None else calibration.forward_compute_constant,
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
    def _plan_historical_layers(self, layers: list[ExpertLayerState], step: int) -> list[str]:
        """Plan all initialized layers from the previous forward routes as one batch."""

        if not layers:
            return []
        consensus_device = _local_tensor_view(layers[0].primary_parameter).device
        globally_ready = _placement_group_all_true_mask(
            [layer.latest_selected_experts is not None for layer in layers],
            device=consensus_device,
            ep_size=self.ep_size,
            ep_group=self.ep_group,
        )
        ready = [layer for layer, is_ready in zip(layers, globally_ready, strict=True) if is_ready]
        if not ready:
            return []
        if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in ready):
            # Empty-slot initialization has sequential marginal dependencies
            # within each layer. It is excluded from steady-state timing.
            committed: list[str] = []
            for layer in ready:
                committed.extend(self._plan_current_layer(layer, step))
            return committed
        structural_signature = {
            (
                layer.latest_hidden_size,
                layer.latest_bytes_per_element,
                layer.num_local_experts,
                layer.num_experts,
            )
            for layer in ready
        }
        if len(structural_signature) != 1:
            committed = []
            for layer in ready:
                committed.extend(self._plan_current_layer(layer, step))
            return committed

        calibrations = [layer.planner_calibration for layer in ready]
        communication_scales = [
            1.0 if calibration is None else calibration.communication_scale for calibration in calibrations
        ]
        compute_slopes = [
            0.0 if calibration is None else calibration.forward_compute_per_assignment for calibration in calibrations
        ]
        compute_constants = [
            0.0 if calibration is None else calibration.forward_compute_constant for calibration in calibrations
        ]
        planner = self._planner_for_layer(
            ready[0],
            communication_scale=communication_scales[0],
            forward_compute_per_assignment=compute_slopes[0],
            forward_compute_constant=compute_constants[0],
        )
        if not isinstance(planner, GreedyCommunicationPlanner):
            raise RuntimeError("Historical batched planning requires GreedyCommunicationPlanner.")

        started = time.perf_counter()
        with _full_timing_range("hiermoe_historical_route_batch_plan"):
            plans = planner.plan_layers(
                [layer.latest_selected_experts for layer in ready],
                [self._layer_layout(layer) for layer in ready],
                [layer.logical_to_physical for layer in ready],
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds,
                layer_seeds=[zlib.crc32(layer.key.encode("utf-8")) for layer in ready],
                step=step,
                communication_scales=communication_scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                skip_final_route_update=True,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_step_batch_planning_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_step_batch_layers", len(ready))

        for layer, plan in zip(ready, plans, strict=True):
            layer.last_plan = plan
            self._record_plan_metrics(plan)
        committed: list[str] = []
        for layer, plan in zip(ready, plans, strict=True):
            committed.extend(
                self._execute_placement_plan(
                    layer,
                    plan,
                    timing_prefix="hiermoe_historical_placement",
                )
            )
        return committed

    @torch.no_grad()
    def _plan_npu_layer_owner_layers(self, layers: list[ExpertLayerState], step: int) -> list[str]:
        """Run exact layer-owner planning and migration synchronously at the step boundary."""

        if not layers:
            return []
        if self.ep_group is None or self.ep_size <= 1:
            raise RuntimeError("NPU layer-owner planning requires a distributed EP process group.")
        consensus_device = _local_tensor_view(layers[0].primary_parameter).device
        globally_ready = _placement_group_all_true_mask(
            [layer.latest_selected_experts is not None for layer in layers],
            device=consensus_device,
            ep_size=self.ep_size,
            ep_group=self.ep_group,
        )
        ready = [layer for layer, is_ready in zip(layers, globally_ready, strict=True) if is_ready]
        if len(ready) != len(layers):
            return []
        if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in ready):
            raise RuntimeError("Blocking NPU layer-owner planning only supports initialized layouts.")
        structural_signature = {
            (
                layer.latest_hidden_size,
                layer.latest_bytes_per_element,
                layer.num_local_experts,
                layer.num_experts,
            )
            for layer in ready
        }
        if len(structural_signature) != 1:
            raise RuntimeError("Blocking NPU layer-owner planning requires structurally identical MoE layers.")

        calibrations = [layer.planner_calibration for layer in ready]
        communication_scales = [
            1.0 if calibration is None else calibration.communication_scale for calibration in calibrations
        ]
        compute_slopes = [
            0.0 if calibration is None else calibration.forward_compute_per_assignment for calibration in calibrations
        ]
        compute_constants = [
            0.0 if calibration is None else calibration.forward_compute_constant for calibration in calibrations
        ]
        planner = self._planner_for_layer(
            ready[0],
            communication_scale=communication_scales[0],
            forward_compute_per_assignment=compute_slopes[0],
            forward_compute_constant=compute_constants[0],
        )
        if not isinstance(planner, GreedyCommunicationPlanner):
            raise RuntimeError("NPU layer-owner planning requires GreedyCommunicationPlanner.")

        from .npu_layer_owner_planner import NPULayerOwnerPlanner

        # Keep the planner metric independent of asynchronous optimizer work
        # queued earlier in the training step. This barrier does not change the
        # blocking experiment's total exposed work; it only gives that work the
        # correct timing attribution.
        synchronize()
        started = time.perf_counter()
        with _full_timing_range("hiermoe_npu_layer_owner_plan"):
            result = NPULayerOwnerPlanner(
                planner,
                process_group=self.ep_group,
                statistic_collective=self._npu_layer_owner_collective,
            ).plan_layers(
                [layer.latest_selected_experts for layer in ready],
                [self._layer_layout(layer) for layer in ready],
                [layer.logical_to_physical for layer in ready],
                source_rank=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds,
                layer_seeds=[zlib.crc32(layer.key.encode("utf-8")) for layer in ready],
                step=step,
                communication_scales=communication_scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
            )
        synchronize()
        planning_ms = (time.perf_counter() - started) * 1000.0
        timing = result.timing
        self._accumulate_metric("hiermoe/npu_layer_owner_planning_ms", planning_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_context_ms", timing.context_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_local_prepare_ms", timing.local_prepare_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_statistic_pack_ms", timing.statistic_pack_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_collective_ms", timing.statistic_collective_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_score_ms", timing.owner_score_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_decision_ms", timing.decision_collective_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_finalization_ms", timing.finalization_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_sent_bytes", timing.sent_statistic_bytes)
        self._accumulate_metric("hiermoe/npu_layer_owner_received_bytes", timing.received_statistic_bytes)
        self._accumulate_metric("hiermoe/npu_layer_owner_layers", len(ready))
        self._accumulate_metric("hiermoe/npu_layer_owner_owned_layers", timing.owned_layer_count)

        for layer, plan in zip(ready, result.plans, strict=True):
            layer.last_plan = plan
            self._record_plan_metrics(plan)

        migration_started = time.perf_counter()
        committed: list[str] = []
        accepted = 0
        with _full_timing_range("hiermoe_npu_layer_owner_migration"):
            for layer, plan in zip(ready, result.plans, strict=True):
                committed.extend(
                    self._execute_placement_plan(
                        layer,
                        plan,
                        timing_prefix="hiermoe_npu_layer_owner_migration",
                        transfer_group=self.ep_group,
                        force_staged_transfer=False,
                        fast_sparse_transfer=True,
                    )
                )
                accepted += int(bool(plan.actions))
        synchronize()
        migration_ms = (time.perf_counter() - migration_started) * 1000.0
        self._accumulate_metric("hiermoe/npu_layer_owner_migration_ms", migration_ms)
        self._accumulate_metric("hiermoe/npu_layer_owner_accepted", accepted)
        return committed

    def _forward_cover_level_weights(self, layer: ExpertLayerState) -> tuple[float, ...]:
        """Return beta-weighted rank and hierarchy-group marginal costs."""

        payload_bytes = max(1, int(layer.latest_hidden_size) * int(layer.latest_bytes_per_element))
        valid_sizes = [
            int(size)
            for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
            if 1 < int(size) < self.ep_size and self.ep_size % int(size) == 0
        ]
        if not valid_sizes:
            weight = float(self.ep_size * payload_bytes) * float(self.perf_model.a2a.beta)
            return (weight if weight > 0.0 else 1.0,)

        group_weights: list[float] = []
        previous_size = 1
        for level_index, size in enumerate(valid_sizes):
            link = self.perf_model.inter[min(level_index, len(self.perf_model.inter) - 1)]
            group_weights.append(float((size / previous_size) * payload_bytes) * float(link.beta))
            previous_size = size
        rank_weight = float((self.ep_size / previous_size) * payload_bytes) * float(self.perf_model.intra.beta)
        weights = (rank_weight, *group_weights)
        if all(weight <= 0.0 for weight in weights):
            return (1.0,) * len(weights)
        return tuple(max(0.0, weight) for weight in weights)

    @torch.no_grad()
    def _execute_forward_cover_actions(
        self,
        selections: Sequence[tuple[ExpertLayerState, PlacementAction]],
    ) -> list[str]:
        """Batch all selected layer covers into one sparse P2P wave per dtype."""

        if not selections:
            return []
        grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
        prepared: list[tuple[ExpertLayerState, torch.Tensor, torch.Tensor, PlacementAction]] = []
        for layer, action in selections:
            if action.kind != "replica":
                raise RuntimeError("Forward-reuse planning only supports replica cover actions.")
            layout = self._layer_layout(layer)
            if int(layout[action.src_slot].item()) != int(action.src_logical) or int(
                layout[action.dst_slot].item()
            ) != int(action.dst_logical):
                raise RuntimeError(
                    f"Forward-reuse cover does not match the current layout for {layer.key}: {action.format()}."
                )
            updated = layout.clone()
            updated[int(action.dst_slot)] = int(action.src_logical)
            updated_owners = layer.logical_to_physical.clone()
            victim = int(action.dst_logical)
            if victim >= 0 and int(updated_owners[victim].item()) == int(action.dst_slot):
                remaining = torch.nonzero(updated == victim, as_tuple=False).flatten()
                if remaining.numel() == 0:
                    raise RuntimeError(
                        f"Forward-reuse cover would remove the last copy of expert {victim} in {layer.key}."
                    )
                updated_owners[victim] = int(remaining.min().item())
            validated, validated_owners = self._validate_placement_layout(
                layer,
                updated,
                updated_owners,
            )
            src_rank = int(action.src_slot) // int(layer.num_local_experts)
            dst_rank = int(action.dst_slot) // int(layer.num_local_experts)
            grouped_entries[(src_rank, dst_rank)].extend(
                self._slot_op_cover_entries_from_tensors(
                    self._slot_op_state_tensors(layer),
                    num_local_experts=layer.num_local_experts,
                    src_slot=int(action.src_slot),
                    dst_slot=int(action.dst_slot),
                )
            )
            prepared.append((layer, validated, validated_owners, action))

        self._execute_sparse_group_slot_transfers(
            grouped_entries,
            process_group=self.ep_group,
        )
        committed: list[str] = []
        for layer, updated, updated_owners, action in prepared:
            if self._forward_reuse_cover_patch_remap:
                source_lut = layer.source_logical_to_physical
                if source_lut is None:
                    raise RuntimeError(f"Forward-reuse cover has no source-rank route LUT for {layer.key}.")
                source_lut = source_lut.clone()
                victim = int(action.dst_logical)
                victim_fallback = int(action.dst_slot) if victim < 0 else int(updated_owners[victim].item())
                if victim >= 0:
                    source_lut[:, victim] = torch.where(
                        source_lut[:, victim] == int(action.dst_slot),
                        torch.full_like(source_lut[:, victim], victim_fallback),
                        source_lut[:, victim],
                    )
                destination_rank = int(action.dst_slot) // int(layer.num_local_experts)
                service_group_size = int(self._forward_reuse_cover_service_group_size)
                service_start = (destination_rank // service_group_size) * service_group_size
                source_lut[
                    service_start : service_start + service_group_size,
                    int(action.src_logical),
                ] = int(action.dst_slot)
                layer.source_logical_to_physical = source_lut
                selected = layer.latest_selected_experts
                physical = layer.latest_physical_routes
                if selected is not None and physical is not None and selected.shape == physical.shape:
                    layer.latest_physical_routes = patch_forward_cover_routes(
                        selected_experts=selected,
                        physical_routes=physical,
                        action=action,
                        source_rank=self.ep_rank,
                        slots_per_rank=layer.num_local_experts,
                        victim_fallback_slot=victim_fallback,
                        service_group_size=service_group_size,
                    )
                    # The previous endpoint cache describes the Forward route
                    # before this cover. A later offline round must reconstruct
                    # its baseline from the patched route instead of reusing it.
                    layer.latest_forward_baseline_communication_counts = None
                    layer.latest_forward_traffic_endpoint_statistics = None
            layer.slot_to_logical = updated
            layer.logical_to_physical = updated_owners
            layer.fixed_r2_layout = False
            layer.active_quota_policy = ()
            self._refresh_layer_mapping_from_slots(layer, updated_owners)
            layer.placement_version += 1
            committed.append(f"{layer.key}:{action.format()}")
        return committed

    @torch.no_grad()
    @torch.no_grad()
    def _plan_online_lut_layers(self, layers: Sequence[ExpertLayerState]) -> list[str]:
        """Apply at most one exact source-LUT correction per layer."""

        if not layers:
            return []
        missing_selected = sum(layer.latest_selected_experts is None for layer in layers)
        missing_physical = sum(layer.latest_physical_routes is None for layer in layers)
        missing_endpoint = sum(layer.latest_forward_traffic_endpoint_statistics is None for layer in layers)
        missing_source_lut = sum(layer.source_logical_to_physical is None for layer in layers)
        self._accumulate_metric("hiermoe/online_lut_missing_selected", missing_selected)
        self._accumulate_metric("hiermoe/online_lut_missing_physical", missing_physical)
        self._accumulate_metric("hiermoe/online_lut_missing_endpoint", missing_endpoint)
        self._accumulate_metric("hiermoe/online_lut_missing_source_lut", missing_source_lut)
        if missing_selected or missing_physical or missing_source_lut:
            self._accumulate_metric(
                "hiermoe/online_lut_skipped_incomplete",
                missing_selected + missing_physical + missing_source_lut,
            )
            return []

        common_device = layers[0].latest_selected_experts.device  # type: ignore[union-attr]
        if any(
            layer.latest_selected_experts.device != common_device  # type: ignore[union-attr]
            for layer in layers
        ):
            raise RuntimeError("Online LUT correction requires all layer routes on one device.")

        synchronize()
        planning_started = time.perf_counter()
        planner = GreedyCommunicationPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layers[0].latest_hidden_size,
            bytes_per_element=layers[0].latest_bytes_per_element,
            slots_per_rank=layers[0].num_local_experts,
            communication_scale=1.0,
            forward_compute_per_assignment=self._forward_reuse_cover_compute_ms_per_assignment,
            forward_compute_constant=0.0,
            smooth_max_gamma=self.smooth_max_gamma,
            process_group=self.ep_group,
            max_copies=self.greedy_max_copies_per_expert,
            assume_unique_routes=True,
            traffic_inter_ms_per_byte=self._online_freeze_inter_ms_per_byte,
            traffic_intra_ms_per_byte=self._online_freeze_intra_ms_per_byte,
            traffic_route_ms_per_assignment=self._online_freeze_route_ms_per_assignment,
            traffic_communication_phase_multiplier=self._online_freeze_communication_ratio,
            traffic_compute_phase_multiplier=self._online_freeze_compute_ratio,
        )

        def baseline_endpoint(layer: ExpertLayerState) -> torch.Tensor:
            cached = layer.latest_forward_traffic_endpoint_statistics
            if cached is not None:
                return cached.to(
                    device=common_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            physical = layer.latest_physical_routes
            assert physical is not None
            unique_counts = planner._local_packed_counts(physical)
            assignment_counts = planner._local_packed_assignment_counts(physical)
            return planner._local_traffic_endpoint_statistics(
                unique_counts,
                assignment_counts,
                source_rank=self.ep_rank,
            ).squeeze(0)

        baseline_endpoints = torch.stack(
            [baseline_endpoint(layer) for layer in layers],
            dim=0,
        )
        self._accumulate_metric("hiermoe/online_lut_endpoint_fallback_layers", missing_endpoint)
        collective_started = time.perf_counter()
        self._planner_reduce_sum(baseline_endpoints)
        synchronize()
        collective_ms = (time.perf_counter() - collective_started) * 1000.0

        baseline_communication, baseline_compute, *_baseline_details = planner._traffic_endpoint_cost_details(
            baseline_endpoints
        )
        baseline_total = baseline_communication + baseline_compute
        local_winners = torch.full(
            (len(layers), 6),
            float("inf"),
            dtype=torch.float32,
            device=common_device,
        )
        local_winners[:, 5] = float(self.ep_rank)
        for layer_index, layer in enumerate(layers):
            selected = layer.latest_selected_experts
            source_lut = layer.source_logical_to_physical
            if selected is None or source_lut is None:
                continue
            proposal = propose_online_lut_move(
                planner=planner,
                selected_experts=selected,
                slot_to_logical=self._layer_layout(layer),
                source_lut=source_lut[self.ep_rank],
                global_baseline_endpoint=baseline_endpoints[layer_index],
                source_rank=self.ep_rank,
                num_experts=layer.num_experts,
            )
            if proposal is None:
                continue
            local_winners[layer_index] = local_winners.new_tensor(
                (
                    proposal.candidate_cost,
                    proposal.candidate_communication,
                    proposal.candidate_compute,
                    proposal.expert,
                    proposal.destination_slot,
                    self.ep_rank,
                )
            )

        collective_started = time.perf_counter()
        gathered = self._planner_gather_fixed(local_winners.reshape(-1)).view(
            self.ep_size,
            len(layers),
            local_winners.shape[1],
        )
        synchronize()
        collective_ms += (time.perf_counter() - collective_started) * 1000.0
        winner_rank_indices = gathered[:, :, 0].argmin(dim=0)
        layer_indices = torch.arange(len(layers), dtype=torch.long, device=common_device)
        winners = gathered[winner_rank_indices, layer_indices].detach().to(device="cpu")
        baseline_cpu = (
            torch.stack(
                (baseline_total, baseline_communication, baseline_compute),
                dim=1,
            )
            .detach()
            .to(device="cpu")
        )

        committed: list[str] = []
        total_gain = 0.0
        communication_gain = 0.0
        compute_gain = 0.0
        for layer, winner, baseline in zip(layers, winners, baseline_cpu, strict=True):
            candidate_cost = float(winner[0].item())
            gain = float(baseline[0].item()) - candidate_cost
            if not math.isfinite(candidate_cost) or not gain > self._online_lut_min_gain:
                continue
            expert = int(winner[3].item())
            destination = int(winner[4].item())
            source_rank = int(winner[5].item())
            source_lut = layer.source_logical_to_physical
            if source_lut is None:
                raise RuntimeError(f"Online LUT correction lost the source LUT for {layer.key}.")
            old_slot = int(source_lut[source_rank, expert].item())
            layout = self._layer_layout(layer)
            if int(layout[destination].item()) != expert:
                raise RuntimeError(
                    f"Online LUT correction maps expert {expert} to invalid slot {destination} for {layer.key}."
                )
            if destination == old_slot:
                raise RuntimeError("Online LUT correction selected a no-op move.")
            source_lut[source_rank, expert] = destination
            layer._device_source_mapping_cache.clear()
            layer.placement_version += 1
            committed.append(f"{layer.key}:lut(rank={source_rank},expert={expert},{old_slot}->{destination})")
            total_gain += gain
            communication_gain += float(baseline[1].item()) - float(winner[1].item())
            compute_gain += float(baseline[2].item()) - float(winner[2].item())

        planning_ms = (time.perf_counter() - planning_started) * 1000.0
        self._accumulate_metric("hiermoe/online_lut_planning_ms", planning_ms)
        self._accumulate_metric("hiermoe/online_lut_collective_ms", collective_ms)
        self._accumulate_metric("hiermoe/online_lut_layers", len(layers))
        self._accumulate_metric("hiermoe/online_lut_accepted", len(committed))
        self._accumulate_metric("hiermoe/online_lut_estimated_gain_ms", total_gain)
        self._accumulate_metric(
            "hiermoe/online_lut_communication_gain_ms",
            communication_gain,
        )
        self._accumulate_metric("hiermoe/online_lut_compute_gain_ms", compute_gain)
        return committed

    def _plan_forward_reuse_cover_layers(self, layers: Sequence[ExpertLayerState], step: int) -> list[str]:
        """Select at most one positive-gain cover per layer from current Forward state."""

        if not layers:
            return []
        common_device = _local_tensor_view(layers[0].primary_parameter).device
        synchronize()
        planning_started = time.perf_counter()
        decision_rows = torch.zeros(
            (len(layers), self._forward_reuse_cover_proposal_topk, 8),
            dtype=torch.float32,
            device=common_device,
        )
        target_owner_ranks = [-1] * len(layers)
        hierarchy_sizes = self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        aggregate_rows: torch.Tensor | None = None
        aggregate_entries: list[tuple[int, ExpertLayerState, int]] = []
        if self._forward_reuse_cover_aggregate_service_group:
            num_experts = int(layers[0].num_experts)
            if any(int(layer.num_experts) != num_experts for layer in layers):
                raise RuntimeError("Service-group Cover aggregation requires the same expert count in every layer.")
            aggregate_rows = torch.zeros(
                (len(layers), 2 * num_experts + 1),
                dtype=torch.float32,
                device=common_device,
            )

        for layer_index, layer in enumerate(layers):
            layout = self._layer_layout(layer)
            active_layout = layout[layout >= 0]
            copy_counts = torch.bincount(active_layout, minlength=layer.num_experts)
            target_ranks = tuple(
                rank
                for rank in range(self.ep_size)
                if (
                    self._forward_reuse_cover_empty_seeding
                    and bool((layout[rank * layer.num_local_experts : (rank + 1) * layer.num_local_experts] < 0).any())
                )
                or any(
                    int(layout[slot].item()) >= 0 and int(copy_counts[int(layout[slot].item())].item()) > 1
                    for slot in range(
                        rank * layer.num_local_experts,
                        (rank + 1) * layer.num_local_experts,
                    )
                )
            )
            if not target_ranks:
                continue
            owner_rank = rotating_service_target_rank(
                target_ranks,
                layer_index=layer_index,
                step=step,
                service_group_size=self._forward_reuse_cover_service_group_size,
            )
            target_owner_ranks[layer_index] = owner_rank
            if aggregate_rows is None:
                continue
            service_group_size = int(self._forward_reuse_cover_service_group_size)
            service_start = (owner_rank // service_group_size) * service_group_size
            if not service_start <= self.ep_rank < service_start + service_group_size:
                continue
            selected = layer.latest_selected_experts
            physical = layer.latest_physical_routes
            if selected is None or physical is None or selected.shape != physical.shape:
                continue
            aggregate_entries.append((layer_index, layer, owner_rank))

        service_statistics_compute_started = time.perf_counter()
        if aggregate_rows is not None and aggregate_entries:
            entry_shapes = {
                tuple(layer.latest_selected_experts.shape)
                for _layer_index, layer, _owner_rank in aggregate_entries
                if layer.latest_selected_experts is not None
            }
            entry_weights = [
                self._forward_cover_level_weights(layer) for _layer_index, layer, _owner_rank in aggregate_entries
            ]
            can_batch_service_statistics = len(entry_shapes) == 1 and all(
                weights == entry_weights[0] for weights in entry_weights[1:]
            )
            if can_batch_service_statistics:
                selected_batch = torch.stack(
                    [
                        layer.latest_selected_experts
                        for _layer_index, layer, _owner_rank in aggregate_entries
                        if layer.latest_selected_experts is not None
                    ],
                    dim=0,
                )
                physical_batch = torch.stack(
                    [
                        layer.latest_physical_routes
                        for _layer_index, layer, _owner_rank in aggregate_entries
                        if layer.latest_physical_routes is not None
                    ],
                    dim=0,
                )
                target_rank_batch = torch.tensor(
                    [owner_rank for _layer_index, _layer, owner_rank in aggregate_entries],
                    dtype=torch.long,
                    device=common_device,
                )
                batched_statistics = forward_cover_local_heuristic_statistics_batched(
                    selected_experts=selected_batch,
                    physical_routes=physical_batch,
                    source_rank=self.ep_rank,
                    target_ranks=target_rank_batch,
                    slots_per_rank=aggregate_entries[0][1].num_local_experts,
                    ep_size=self.ep_size,
                    hierarchy_group_sizes=hierarchy_sizes,
                    num_experts=aggregate_entries[0][1].num_experts,
                    level_weights=entry_weights[0],
                )
                layer_indices = torch.tensor(
                    [layer_index for layer_index, _layer, _owner_rank in aggregate_entries],
                    dtype=torch.long,
                    device=common_device,
                )
                aggregate_rows[:, :num_experts].index_copy_(
                    0,
                    layer_indices,
                    batched_statistics.communication_benefit,
                )
                aggregate_rows[:, num_experts : 2 * num_experts].index_copy_(
                    0,
                    layer_indices,
                    batched_statistics.expert_assignments,
                )
                aggregate_rows[:, -1].index_copy_(
                    0,
                    layer_indices,
                    batched_statistics.baseline_communication_units,
                )
            else:
                for layer_index, layer, owner_rank in aggregate_entries:
                    assert layer.latest_selected_experts is not None
                    assert layer.latest_physical_routes is not None
                    local_statistics = forward_cover_local_heuristic_statistics(
                        selected_experts=layer.latest_selected_experts,
                        physical_routes=layer.latest_physical_routes,
                        source_rank=self.ep_rank,
                        target_rank=owner_rank,
                        slots_per_rank=layer.num_local_experts,
                        ep_size=self.ep_size,
                        hierarchy_group_sizes=hierarchy_sizes,
                        num_experts=layer.num_experts,
                        level_weights=self._forward_cover_level_weights(layer),
                    )
                    aggregate_rows[layer_index, : layer.num_experts] = local_statistics.communication_benefit
                    aggregate_rows[layer_index, layer.num_experts : 2 * layer.num_experts] = (
                        local_statistics.expert_assignments
                    )
                    aggregate_rows[layer_index, -1] = local_statistics.baseline_communication_units
        synchronize()
        service_statistics_compute_ms = (time.perf_counter() - service_statistics_compute_started) * 1000.0

        service_statistics_collective_ms = 0.0
        if aggregate_rows is not None:
            service_statistics_collective_started = time.perf_counter()
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(aggregate_rows, op=dist.ReduceOp.SUM, group=self.ep_group)
            synchronize()
            service_statistics_collective_ms = (time.perf_counter() - service_statistics_collective_started) * 1000.0

        owned_layers = 0
        ready_owned_layers = 0
        empty_count_on_owned_targets = 0
        for layer_index, layer in enumerate(layers):
            owner_rank = target_owner_ranks[layer_index]
            if owner_rank != self.ep_rank:
                continue
            owned_layers += 1
            layout = self._layer_layout(layer)
            selected = layer.latest_selected_experts
            physical = layer.latest_physical_routes
            local_counts = layer.latest_tokens_per_local_expert
            if selected is None or physical is None:
                continue
            if selected.shape != physical.shape:
                continue
            target_start = int(owner_rank) * int(layer.num_local_experts)
            target_end = target_start + int(layer.num_local_experts)
            target_empty_count = int((layout[target_start:target_end] < 0).sum().item())
            empty_count_on_owned_targets += target_empty_count
            if local_counts is None:
                if not self._forward_reuse_cover_empty_seeding or target_empty_count <= 0:
                    continue
                # Before any redundant copy is active, some dispatch paths do
                # not publish per-physical-slot counts.  Empty destinations
                # are nevertheless exact zero-load victims, so a zero vector
                # contains all information the empty-slot proposal needs.
                local_counts = torch.zeros(
                    (layer.num_local_experts,),
                    dtype=torch.float32,
                    device=selected.device,
                )
            ready_owned_layers += 1
            aggregated_statistics = None
            if aggregate_rows is not None:
                row = aggregate_rows[layer_index]
                aggregated_statistics = ForwardCoverHeuristicStatistics(
                    communication_benefit=row[: layer.num_experts],
                    expert_assignments=row[layer.num_experts : 2 * layer.num_experts],
                    baseline_communication_units=row[-1],
                )
            local_proposals = propose_forward_reuse_covers(
                selected_experts=selected,
                physical_routes=physical,
                slot_to_logical=layout,
                owner_slots=layer.logical_to_physical,
                local_slot_assignments=local_counts,
                source_rank=self.ep_rank,
                slots_per_rank=layer.num_local_experts,
                hierarchy_group_sizes=self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)],
                num_experts=layer.num_experts,
                max_copies=self.greedy_max_copies_per_expert,
                level_weights=self._forward_cover_level_weights(layer),
                compute_weight=self._forward_reuse_cover_compute_weight,
                minimum_gain=self._forward_reuse_cover_min_gain,
                victim_mode=self._forward_reuse_cover_victim_mode,
                service_group_size=self._forward_reuse_cover_service_group_size,
                aggregated_statistics=aggregated_statistics,
                max_proposals=self._forward_reuse_cover_proposal_topk,
            )
            for proposal_index, proposal in enumerate(local_proposals):
                if proposal_index >= self._forward_reuse_cover_proposal_topk or proposal.action is None:
                    continue
                action = proposal.action
                decision_rows[layer_index, proposal_index] = torch.tensor(
                    [
                        1.0,
                        float(action.src_slot + 1),
                        float(action.dst_slot + 1),
                        float(action.src_logical + 1),
                        float(action.dst_logical + 1),
                        float(proposal.estimated_gain),
                        float(proposal.communication_gain),
                        float(proposal.assignment_delta),
                    ],
                    dtype=torch.float32,
                    device=common_device,
                )

        proposal_collective_started = time.perf_counter()
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(decision_rows, op=dist.ReduceOp.SUM, group=self.ep_group)
        synchronize()
        proposal_collective_ms = (time.perf_counter() - proposal_collective_started) * 1000.0

        rows = decision_rows.detach().to(device="cpu")
        fresh_proposals: dict[int, list[tuple[ExpertLayerState, PlacementAction]]] = {}
        for layer_index, (layer, layer_rows) in enumerate(zip(layers, rows, strict=True)):
            for row in layer_rows:
                if int(row[0].item()) <= 0:
                    continue
                action = PlacementAction(
                    kind="replica",
                    src_slot=int(row[1].item()) - 1,
                    dst_slot=int(row[2].item()) - 1,
                    src_logical=int(row[3].item()) - 1,
                    dst_logical=int(row[4].item()) - 1,
                )
                fresh_proposals.setdefault(layer_index, []).append((layer, action))

        pending_layer_indices: set[int] = set()
        proposals: list[tuple[int, ExpertLayerState, PlacementAction]] = []
        for layer_index, layer in enumerate(layers):
            pending = self._forward_reuse_cover_pending.get(layer.key)
            if pending is not None:
                proposals.append((layer_index, layer, pending[0]))
                pending_layer_indices.add(layer_index)
                continue
            for fresh in fresh_proposals.get(layer_index, ()):
                proposals.append((layer_index, fresh[0], fresh[1]))

        if self._forward_reuse_cover_fast:
            # The layer owner selected a positive local action directly from
            # the physical routes consumed by Forward.  Synchronize only this
            # tiny action table: do not rebuild candidate-by-group statistics
            # or run the second global validation collective.
            planning_ms = (time.perf_counter() - planning_started) * 1000.0
            selections = [(layer, action) for _layer_index, layer, action in proposals]
            migration_started = time.perf_counter()
            with _full_timing_range("hiermoe_forward_reuse_cover_migration"):
                committed = self._execute_forward_cover_actions(selections)
            synchronize()
            migration_ms = (time.perf_counter() - migration_started) * 1000.0
            self._accumulate_metric("hiermoe/forward_cover_planning_ms", planning_ms)
            self._accumulate_metric(
                "hiermoe/forward_cover_decision_collective_ms",
                proposal_collective_ms,
            )
            self._accumulate_metric(
                "hiermoe/forward_cover_proposal_collective_ms",
                proposal_collective_ms,
            )
            self._accumulate_metric(
                "hiermoe/forward_cover_service_statistics_collective_ms",
                service_statistics_collective_ms,
            )
            self._accumulate_metric(
                "hiermoe/forward_cover_service_statistics_compute_ms",
                service_statistics_compute_ms,
            )
            self._accumulate_metric("hiermoe/forward_cover_validation_compute_ms", 0.0)
            self._accumulate_metric("hiermoe/forward_cover_validation_collective_ms", 0.0)
            self._accumulate_metric("hiermoe/forward_cover_migration_ms", migration_ms)
            self._accumulate_metric("hiermoe/forward_cover_owned_layers", owned_layers)
            self._accumulate_metric("hiermoe/forward_cover_proposed", len(proposals))
            self._accumulate_metric("hiermoe/forward_cover_accepted", len(committed))
            self._accumulate_metric("hiermoe/forward_cover_validation_affected_tokens", 0)
            self._accumulate_metric(
                "hiermoe/forward_cover_estimated_gain",
                float(decision_rows[..., 5].sum().item()),
            )
            self._accumulate_metric(
                "hiermoe/forward_cover_communication_gain",
                float(decision_rows[..., 6].sum().item()),
            )
            self._accumulate_metric(
                "hiermoe/forward_cover_assignment_delta",
                float(decision_rows[..., 7].sum().item()),
            )
            return committed

        endpoint_width = 8 * self.ep_size
        validation_row_count = max(1, len(proposals))
        validation_rows = torch.zeros(
            (validation_row_count, 2 * endpoint_width),
            dtype=torch.float32,
            device=common_device,
        )
        debug_endpoint_deltas = (
            torch.zeros(
                (validation_row_count, endpoint_width),
                dtype=torch.float32,
                device=common_device,
            )
            if self.debug_validate
            else None
        )
        cost_planner = GreedyCommunicationPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layers[0].latest_hidden_size,
            bytes_per_element=layers[0].latest_bytes_per_element,
            slots_per_rank=layers[0].num_local_experts,
            communication_scale=1.0,
            forward_compute_per_assignment=self._forward_reuse_cover_compute_ms_per_assignment,
            forward_compute_constant=0.0,
            smooth_max_gamma=self.smooth_max_gamma,
            process_group=self.ep_group,
            max_copies=self.greedy_max_copies_per_expert,
            assume_unique_routes=True,
            traffic_inter_ms_per_byte=self._online_freeze_inter_ms_per_byte,
            traffic_intra_ms_per_byte=self._online_freeze_intra_ms_per_byte,
            traffic_route_ms_per_assignment=self._online_freeze_route_ms_per_assignment,
            traffic_communication_phase_multiplier=self._online_freeze_communication_ratio,
            traffic_compute_phase_multiplier=self._online_freeze_compute_ratio,
        )
        validation_compute_started = time.perf_counter()
        affected_token_counts = [0] * len(proposals)
        batched_affected_tokens: torch.Tensor | None = None
        local_validation_proposal_count = len(proposals)

        def validate(proposal_index: int, layer_index: int, layer: ExpertLayerState, action: PlacementAction) -> None:
            selected = layer.latest_selected_experts
            physical = layer.latest_physical_routes
            if selected is None or physical is None:
                return
            baseline_assignments = None
            local_counts = layer.latest_tokens_per_local_expert
            cached_endpoint = layer.latest_forward_traffic_endpoint_statistics
            if cached_endpoint is not None and local_counts is not None:
                baseline_assignments = torch.zeros(
                    (self.ep_size,),
                    dtype=torch.float32,
                    device=local_counts.device,
                )
                baseline_assignments[self.ep_rank] = local_counts.to(torch.float32).sum()
            local_validation = forward_cover_local_validation_stats(
                selected_experts=selected,
                physical_routes=physical,
                slot_to_logical=self._layer_layout(layer),
                action=action,
                source_rank=self.ep_rank,
                slots_per_rank=layer.num_local_experts,
                hierarchy_group_sizes=hierarchy_sizes,
                num_experts=layer.num_experts,
                max_copies=self.greedy_max_copies_per_expert,
                step=max(0, int(layer.latest_route_step)),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                patch_remap=self._forward_reuse_cover_patch_remap,
                victim_fallback_slot=(
                    int(action.dst_slot)
                    if int(action.dst_logical) < 0
                    else int(layer.logical_to_physical[int(action.dst_logical)].item())
                ),
                service_group_size=self._forward_reuse_cover_service_group_size,
                baseline_communication_counts=(
                    layer.latest_forward_baseline_communication_counts if cached_endpoint is not None else None
                ),
                baseline_assignment_counts=baseline_assignments,
            )
            if cached_endpoint is None:
                baseline_endpoint = cost_planner._local_traffic_endpoint_statistics(
                    local_validation.baseline_communication_counts.unsqueeze(0),
                    local_validation.baseline_assignment_counts.unsqueeze(0),
                    source_rank=self.ep_rank,
                ).squeeze(0)
            else:
                baseline_endpoint = cached_endpoint.to(
                    device=common_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            endpoint_delta = cost_planner._local_traffic_endpoint_statistics(
                local_validation.communication_count_delta.unsqueeze(0),
                local_validation.assignment_count_delta.unsqueeze(0),
                source_rank=self.ep_rank,
            ).squeeze(0)
            validation_rows[proposal_index, :endpoint_width] = baseline_endpoint
            validation_rows[proposal_index, endpoint_width:] = endpoint_delta
            if self.debug_validate and cached_endpoint is not None:
                uncached_validation = forward_cover_local_validation_stats(
                    selected_experts=selected,
                    physical_routes=physical,
                    slot_to_logical=self._layer_layout(layer),
                    action=action,
                    source_rank=self.ep_rank,
                    slots_per_rank=layer.num_local_experts,
                    hierarchy_group_sizes=hierarchy_sizes,
                    num_experts=layer.num_experts,
                    max_copies=self.greedy_max_copies_per_expert,
                    step=max(0, int(layer.latest_route_step)),
                    layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                    patch_remap=self._forward_reuse_cover_patch_remap,
                    victim_fallback_slot=(
                        int(action.dst_slot)
                        if int(action.dst_logical) < 0
                        else int(layer.logical_to_physical[int(action.dst_logical)].item())
                    ),
                    service_group_size=self._forward_reuse_cover_service_group_size,
                )
                rescanned_baseline = cost_planner._local_traffic_endpoint_statistics(
                    uncached_validation.baseline_communication_counts.unsqueeze(0),
                    uncached_validation.baseline_assignment_counts.unsqueeze(0),
                    source_rank=self.ep_rank,
                ).squeeze(0)
                assert debug_endpoint_deltas is not None
                debug_endpoint_deltas[proposal_index] = baseline_endpoint - rescanned_baseline
            affected_token_counts[proposal_index] = local_validation.affected_tokens

        can_batch_patch_validation = (
            self._forward_reuse_cover_patch_remap
            and not self.debug_validate
            and bool(proposals)
            and all(
                layer.latest_selected_experts is not None
                and layer.latest_physical_routes is not None
                and layer.latest_forward_traffic_endpoint_statistics is not None
                for _layer_index, layer, _action in proposals
            )
        )
        proposal_shapes = {
            tuple(layer.latest_selected_experts.shape)
            for _layer_index, layer, _action in proposals
            if layer.latest_selected_experts is not None
        }
        can_batch_patch_validation = can_batch_patch_validation and len(proposal_shapes) == 1
        if can_batch_patch_validation:
            relevant_proposal_indices = [
                proposal_index
                for proposal_index, (_layer_index, layer, action) in enumerate(proposals)
                if forward_cover_patch_source_rank_relevant(
                    action=action,
                    source_rank=self.ep_rank,
                    slots_per_rank=layer.num_local_experts,
                    service_group_size=self._forward_reuse_cover_service_group_size,
                    source_logical_to_physical=(
                        None
                        if layer.source_logical_to_physical is None
                        else layer.source_logical_to_physical[self.ep_rank]
                    ),
                )
            ]
            local_validation_proposal_count = len(relevant_proposal_indices)
            relevant_proposals = [proposals[index] for index in relevant_proposal_indices]
            baseline_endpoint_batch = torch.stack(
                [
                    layer.latest_forward_traffic_endpoint_statistics
                    for _layer_index, layer, _action in proposals
                    if layer.latest_forward_traffic_endpoint_statistics is not None
                ],
                dim=0,
            ).to(
                device=common_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            validation_rows[: len(proposals), :endpoint_width].copy_(baseline_endpoint_batch)
        if can_batch_patch_validation and relevant_proposals:
            selected_batch = torch.stack(
                [
                    layer.latest_selected_experts
                    for _layer_index, layer, _action in relevant_proposals
                    if layer.latest_selected_experts is not None
                ],
                dim=0,
            )
            physical_batch = torch.stack(
                [
                    layer.latest_physical_routes
                    for _layer_index, layer, _action in relevant_proposals
                    if layer.latest_physical_routes is not None
                ],
                dim=0,
            )
            source_logical = torch.tensor(
                [action.src_logical for _layer_index, _layer, action in relevant_proposals],
                dtype=torch.long,
                device=common_device,
            )
            victim_logical = torch.tensor(
                [action.dst_logical for _layer_index, _layer, action in relevant_proposals],
                dtype=torch.long,
                device=common_device,
            )
            destination_slots = torch.tensor(
                [action.dst_slot for _layer_index, _layer, action in relevant_proposals],
                dtype=torch.long,
                device=common_device,
            )
            victim_fallback_slots = torch.tensor(
                [
                    (
                        int(action.dst_slot)
                        if int(action.dst_logical) < 0
                        else int(layer.logical_to_physical[int(action.dst_logical)].item())
                    )
                    for _layer_index, layer, action in relevant_proposals
                ],
                dtype=torch.long,
                device=common_device,
            )
            batched_validation = forward_cover_patch_validation_stats_batched(
                selected_experts=selected_batch,
                physical_routes=physical_batch,
                source_logical=source_logical,
                victim_logical=victim_logical,
                destination_slots=destination_slots,
                victim_fallback_slots=victim_fallback_slots,
                source_rank=self.ep_rank,
                slots_per_rank=layers[0].num_local_experts,
                ep_size=self.ep_size,
                hierarchy_group_sizes=hierarchy_sizes,
                service_group_size=self._forward_reuse_cover_service_group_size,
            )
            endpoint_delta = cost_planner._local_traffic_endpoint_statistics(
                batched_validation.communication_count_delta,
                batched_validation.assignment_count_delta,
                source_rank=self.ep_rank,
            )
            relevant_index_tensor = torch.tensor(
                relevant_proposal_indices,
                dtype=torch.long,
                device=common_device,
            )
            validation_rows[:, endpoint_width:].index_copy_(
                0,
                relevant_index_tensor,
                endpoint_delta,
            )
            batched_affected_tokens = batched_validation.affected_tokens
        elif not can_batch_patch_validation and common_device.type != "cpu" and len(proposals) > 1:
            device_api = get_torch_device()
            try:
                caller_stream = device_api.current_stream(common_device)
            except TypeError:
                caller_stream = device_api.current_stream()
            stream_count = min(_GREEDY_LAYER_PARALLEL_STREAMS, len(proposals))
            validation_streams = [
                self._pipeline_stream(f"forward_cover_validation_{index}", common_device)
                for index in range(stream_count)
            ]
            for stream in validation_streams:
                assert stream is not None
                stream.wait_stream(caller_stream)
            for proposal_index, (layer_index, layer, action) in enumerate(proposals):
                stream = validation_streams[proposal_index % stream_count]
                assert stream is not None
                with device_api.stream(stream):
                    validate(proposal_index, layer_index, layer, action)
            for stream in validation_streams:
                assert stream is not None
                caller_stream.wait_stream(stream)
        elif not can_batch_patch_validation:
            for proposal_index, (layer_index, layer, action) in enumerate(proposals):
                validate(proposal_index, layer_index, layer, action)
        synchronize()
        validation_compute_ms = (time.perf_counter() - validation_compute_started) * 1000.0
        local_affected_tokens = (
            int(batched_affected_tokens.sum().item())
            if batched_affected_tokens is not None
            else sum(affected_token_counts)
        )

        validation_collective_started = time.perf_counter()
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(validation_rows, op=dist.ReduceOp.SUM, group=self.ep_group)
            if debug_endpoint_deltas is not None:
                dist.all_reduce(
                    debug_endpoint_deltas,
                    op=dist.ReduceOp.SUM,
                    group=self.ep_group,
                )
        synchronize()
        validation_collective_ms = (time.perf_counter() - validation_collective_started) * 1000.0

        if debug_endpoint_deltas is not None:
            cached_mismatch = debug_endpoint_deltas.abs().max()
            if float(cached_mismatch.item()) != 0.0:
                raise RuntimeError(
                    "Forward cached endpoint statistics disagree with route reconstruction: "
                    f"max_abs_delta={float(cached_mismatch.item())}."
                )
        baseline_endpoint_statistics = validation_rows[:, :endpoint_width]
        candidate_endpoint_statistics = baseline_endpoint_statistics + validation_rows[:, endpoint_width:]
        (
            baseline_communication,
            baseline_compute,
            _baseline_units,
            _baseline_peak_rank,
            _baseline_peak_compute_rank,
            _baseline_dim,
        ) = cost_planner._traffic_endpoint_cost_details(baseline_endpoint_statistics)
        (
            candidate_communication,
            candidate_compute,
            _candidate_units,
            _candidate_peak_rank,
            _candidate_peak_compute_rank,
            _candidate_dim,
        ) = cost_planner._traffic_endpoint_cost_details(candidate_endpoint_statistics)
        baseline_assignment_receive = baseline_endpoint_statistics[:, 7 * self.ep_size :]
        candidate_assignment_receive = candidate_endpoint_statistics[:, 7 * self.ep_size :]
        baseline_peak_assignments = baseline_assignment_receive.max(dim=1).values
        candidate_peak_assignments = candidate_assignment_receive.max(dim=1).values
        communication_gain_rows = baseline_communication - candidate_communication
        compute_gain_rows = baseline_compute - candidate_compute
        assignment_gain_rows = baseline_peak_assignments - candidate_peak_assignments
        global_gain_rows = communication_gain_rows + compute_gain_rows
        accepted_mask = torch.isfinite(global_gain_rows) & (global_gain_rows > self._forward_reuse_cover_min_gain)
        validation_summary = (
            torch.stack(
                (
                    accepted_mask.to(torch.float32),
                    global_gain_rows,
                    communication_gain_rows,
                    assignment_gain_rows,
                    baseline_communication,
                    candidate_communication,
                    baseline_peak_assignments,
                    candidate_peak_assignments,
                ),
                dim=1,
            )[: len(proposals)]
            .detach()
            .to(device="cpu")
        )

        proposal_groups: dict[
            int,
            list[tuple[tuple[ExpertLayerState, PlacementAction], torch.Tensor]],
        ] = {}
        for (layer_index, layer, action), summary in zip(proposals, validation_summary, strict=True):
            proposal_groups.setdefault(layer_index, []).append(((layer, action), summary))
        selections: list[tuple[ExpertLayerState, PlacementAction]] = []
        estimated_gain = 0.0
        communication_gain = 0.0
        assignment_delta = 0.0
        for layer_index, candidates in proposal_groups.items():
            accepted_candidates = [
                (proposed, summary) for proposed, summary in candidates if int(summary[0].item()) > 0
            ]
            if not accepted_candidates:
                if layer_index in pending_layer_indices:
                    layer = candidates[0][0][0]
                    self._forward_reuse_cover_pending.pop(layer.key, None)
                continue
            proposed, summary = max(
                accepted_candidates,
                key=lambda item: float(item[1][1].item()),
            )
            layer, action = proposed
            if self._forward_reuse_cover_confirm_samples > 1:
                if layer_index in pending_layer_indices:
                    _pending_action, confirmed_samples = self._forward_reuse_cover_pending.pop(layer.key)
                    confirmed_samples += 1
                    if confirmed_samples >= self._forward_reuse_cover_confirm_samples:
                        selections.append(proposed)
                    else:
                        self._forward_reuse_cover_pending[layer.key] = (action, confirmed_samples)
                else:
                    self._forward_reuse_cover_pending[layer.key] = (action, 1)
            else:
                selections.append(proposed)
            estimated_gain += float(summary[1].item())
            communication_gain += float(summary[2].item())
            assignment_delta += float(summary[7].item() - summary[6].item())

        planning_ms = (time.perf_counter() - planning_started) * 1000.0
        collective_ms = proposal_collective_ms + validation_collective_ms

        migration_started = time.perf_counter()
        with _full_timing_range("hiermoe_forward_reuse_cover_migration"):
            committed = self._execute_forward_cover_actions(selections)
        synchronize()
        migration_ms = (time.perf_counter() - migration_started) * 1000.0

        self._accumulate_metric("hiermoe/forward_cover_planning_ms", planning_ms)
        self._accumulate_metric("hiermoe/forward_cover_decision_collective_ms", collective_ms)
        self._accumulate_metric("hiermoe/forward_cover_proposal_collective_ms", proposal_collective_ms)
        self._accumulate_metric(
            "hiermoe/forward_cover_service_statistics_collective_ms",
            service_statistics_collective_ms,
        )
        self._accumulate_metric(
            "hiermoe/forward_cover_service_statistics_compute_ms",
            service_statistics_compute_ms,
        )
        self._accumulate_metric("hiermoe/forward_cover_validation_compute_ms", validation_compute_ms)
        self._accumulate_metric("hiermoe/forward_cover_validation_collective_ms", validation_collective_ms)
        self._accumulate_metric("hiermoe/forward_cover_migration_ms", migration_ms)
        self._accumulate_metric("hiermoe/forward_cover_owned_layers", owned_layers)
        self._accumulate_metric("hiermoe/forward_cover_ready_owned_layers", ready_owned_layers)
        self._accumulate_metric(
            "hiermoe/forward_cover_empty_count_on_owned_targets",
            empty_count_on_owned_targets,
        )
        self._accumulate_metric("hiermoe/forward_cover_proposed", len(proposals))
        self._accumulate_metric(
            "hiermoe/forward_cover_validation_local_proposals",
            local_validation_proposal_count,
        )
        self._accumulate_metric("hiermoe/forward_cover_accepted", len(committed))
        self._accumulate_metric("hiermoe/forward_cover_validation_affected_tokens", local_affected_tokens)
        self._accumulate_metric("hiermoe/forward_cover_estimated_gain", estimated_gain)
        self._accumulate_metric("hiermoe/forward_cover_communication_gain", communication_gain)
        self._accumulate_metric("hiermoe/forward_cover_assignment_delta", assignment_delta)
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
                        validate_optimizer_state=False,
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
    def _reset_redundant_slots_for_online_freeze(self, layers: Sequence[ExpertLayerState]) -> None:
        """Keep canonical owners and expose every redundant R2 slot for greedy filling."""

        for layer in layers:
            if layer.slot_to_logical is None:
                raise RuntimeError(f"Online freeze requires a slot layout for layer {layer.key}.")
            owners = layer.logical_to_physical.detach().to(device="cpu", dtype=torch.long)
            if int(torch.unique(owners).numel()) != layer.num_experts:
                raise RuntimeError(f"Online freeze found non-unique owner slots in layer {layer.key}.")
            layout = torch.full((layer.num_physical_slots,), -1, dtype=torch.long)
            logical = torch.arange(layer.num_experts, dtype=torch.long)
            layout.scatter_(0, owners, logical)
            empty_slots = int((layout < 0).sum().item())
            if empty_slots != self.replica_slot_capacity:
                raise RuntimeError(
                    f"Online freeze expected {self.replica_slot_capacity} redundant slots in layer {layer.key}, "
                    f"found {empty_slots}."
                )
            layer.slot_to_logical = layout
            layer.fixed_r2_layout = False
            layer.active_quota_policy = ()
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            layer.invalidate_cache()

    @torch.no_grad()
    def _run_online_freeze_step(self, step: int) -> str:
        calibration_step = self._online_freeze_calibration_step
        planning_step = calibration_step + 1
        if int(step) < calibration_step:
            self.latest_pair = "none"
            return self.latest_pair
        if int(step) > planning_step:
            self.latest_pair = "none"
            return self.latest_pair

        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        if int(step) == calibration_step:
            self.prepare_calibrations(int(step))
            if any(layer.planner_calibration is None for layer in layers):
                raise RuntimeError(
                    f"Online freeze calibration did not complete at step {step}; "
                    "the profiled layer events were not ready on every EP rank."
                )
            self.latest_pair = "none"
            return self.latest_pair

        if any(layer.planner_calibration is None for layer in layers):
            raise RuntimeError(
                f"Online freeze planning at step {step} has no calibration from step {calibration_step}."
            )
        r2_costs: dict[str, float] = {}
        for layer in layers:
            calibration = layer.planner_calibration
            selected = layer.latest_selected_experts
            if calibration is None or selected is None:
                raise RuntimeError(f"Online freeze cannot score the R2 baseline for layer {layer.key}.")
            planner = self._planner_for_layer(
                layer,
                communication_scale=calibration.communication_scale,
                forward_compute_per_assignment=calibration.forward_compute_per_assignment,
                forward_compute_constant=calibration.forward_compute_constant,
            )
            copy_slots, _copy_mask = layer.copy_slots_for_device(selected.device)
            r2_costs[layer.key] = planner.score_layout(
                selected,
                self._layer_layout(layer),
                source_ranks=self.ep_rank,
                owner_slots=layer.logical_to_physical,
                step=int(step),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=int(copy_slots.shape[1]),
            ).total
        self._reset_redundant_slots_for_online_freeze(layers)
        with _full_timing_range("hiermoe_online_freeze_plan"):
            committed = self._plan_historical_layers(layers, int(step))
        expected_actions = len(layers) * self.replica_slot_capacity
        if len(committed) != expected_actions:
            raise RuntimeError(f"Online freeze committed {len(committed)} cover actions, expected {expected_actions}.")
        final_costs = {layer.key: layer.last_plan.final_cost.total for layer in layers if layer.last_plan is not None}
        if len(final_costs) != len(layers):
            raise RuntimeError("Online freeze did not retain a final predicted cost for every layer.")
        r2_total = sum(r2_costs.values())
        final_total = sum(final_costs.values())
        self._accumulate_metric("hiermoe/online_freeze_cover_count", len(committed))
        self._accumulate_metric("hiermoe/online_freeze_r2_predicted_cost_ms", r2_total)
        self._accumulate_metric("hiermoe/online_freeze_final_predicted_cost_ms", final_total)
        self._accumulate_metric("hiermoe/online_freeze_predicted_gain_ms", r2_total - final_total)
        self._accumulate_metric(
            "hiermoe/online_freeze_predicted_speedup",
            r2_total / final_total if final_total > 0.0 else 0.0,
        )
        self.latest_pair = ",".join(committed)
        return self.latest_pair

    def _hot_update_event(self, event: str, **values: Any) -> None:
        if self.ep_rank != 0:
            return
        root = os.path.abspath(_HOT_UPDATE_WORK_ROOT)
        os.makedirs(root, exist_ok=True)
        payload = {
            "event": str(event),
            "wall_time": time.time(),
            **values,
        }
        with open(os.path.join(root, "events.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _hot_update_builder_path(self) -> str:
        if _HOT_UPDATE_BUILDER:
            return os.path.abspath(_HOT_UPDATE_BUILDER)
        repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        return os.path.join(repository_root, "scripts", "profile", "plan_placemoe.py")

    def _write_hot_update_current_layout(
        self,
        layers: Sequence[ExpertLayerState],
        path: str,
    ) -> None:
        plans: dict[str, LayerPlan] = {}
        for layer in layers:
            if layer.source_logical_to_physical is None:
                raise RuntimeError(f"PlaceMoE mapping update requires a source LUT for {layer.key}.")
            plans[layer.key] = LayerPlan(
                slot_to_logical=self._layer_layout(layer).tolist(),
                owner_slots=layer.logical_to_physical.detach().cpu().tolist(),
                source_logical_to_physical=layer.source_logical_to_physical.detach().cpu().tolist(),
            )
        payload = build_placemoe_artifact(
            plans,
            PlaceMoETopology(
                ep_size=self.ep_size,
                ranks_per_node=min(self.ep_size, self.hierarchy.local_world_size),
                num_experts=layers[0].num_experts,
                slots_per_rank=layers[0].num_local_experts,
            ),
            source={"algorithm": "placemoe-v1", "update_mode": "mapping"},
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @torch.no_grad()
    def _capture_hot_update_routes(self, placement_step: int, training_step: int, job_dir: str) -> float:
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        if not layers:
            raise RuntimeError("PlaceMoE hot update has no registered expert layers.")
        if any(
            layer.latest_selected_experts is None or int(layer.latest_route_step) != int(placement_step)
            for layer in layers
        ):
            missing = [
                layer.key
                for layer in layers
                if layer.latest_selected_experts is None or int(layer.latest_route_step) != int(placement_step)
            ]
            raise RuntimeError(
                f"PlaceMoE hot-update step {training_step} has stale or missing routes for {missing[:4]}."
            )
        started = time.perf_counter()
        capture_dir = os.path.join(job_dir, "routes", "step0000")
        if self.ep_rank == 0:
            os.makedirs(capture_dir, exist_ok=True)

        for layer_index, layer in enumerate(layers):
            selected = layer.latest_selected_experts
            assert selected is not None
            if selected.ndim != 2:
                raise RuntimeError(f"PlaceMoE hot updates require 2D routes for layer {layer.key}.")
            route = selected.detach().to(dtype=torch.int32).contiguous()
            shape = torch.tensor(tuple(int(value) for value in route.shape), dtype=torch.int64, device=route.device)
            if self.ep_size > 1:
                gathered_shapes = torch.empty((self.ep_size * 2,), dtype=torch.int64, device=route.device)
                dist.all_gather_into_tensor(gathered_shapes, shape, group=self.ep_group)
                shapes = gathered_shapes.view(self.ep_size, 2)
            else:
                shapes = shape.view(1, 2)
            max_numel = int((shapes[:, 0] * shapes[:, 1]).max().item())
            padded = torch.full((max_numel,), -1, dtype=torch.int32, device=route.device)
            padded[: route.numel()].copy_(route.reshape(-1))
            if self.ep_size > 1:
                gathered_routes = torch.empty(
                    (self.ep_size * max_numel,),
                    dtype=torch.int32,
                    device=route.device,
                )
                dist.all_gather_into_tensor(gathered_routes, padded, group=self.ep_group)
            else:
                gathered_routes = padded
            if self.ep_rank == 0:
                cpu_shapes = shapes.detach().cpu()
                cpu_routes = gathered_routes.detach().cpu().view(self.ep_size, max_numel)
                routes_by_rank = []
                for rank in range(self.ep_size):
                    rows, width = (int(value) for value in cpu_shapes[rank].tolist())
                    routes_by_rank.append(cpu_routes[rank, : rows * width].view(rows, width).clone())
                torch.save(
                    {
                        "format": "hiermoe-local-route-bundle-v1",
                        "ep_size": self.ep_size,
                        "source_training_step": int(training_step),
                        "layer_key": layer.key,
                        "routes_by_rank": routes_by_rank,
                    },
                    os.path.join(capture_dir, f"layer{layer_index:02d}_call0_all_ranks.pt"),
                )
            del gathered_routes, padded, shapes
        if self.ep_group is not None and self.ep_size > 1:
            dist.barrier(group=self.ep_group)
        return (time.perf_counter() - started) * 1000.0

    def _launch_hot_update(
        self,
        *,
        placement_step: int,
        training_step: int,
        update_mode: str,
    ) -> None:
        try:
            update_kind = UpdateKind(update_mode)
        except ValueError as error:
            raise ValueError(f"Unsupported PlaceMoE update mode {update_mode!r}.") from error
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        job_dir = os.path.join(
            os.path.abspath(_HOT_UPDATE_WORK_ROOT),
            f"source_step_{int(training_step):06d}_{update_mode}",
        )
        snapshot_ms = self._capture_hot_update_routes(placement_step, training_step, job_dir)
        layout_path = os.path.join(job_dir, "layout.json")
        input_layout_path = os.path.join(job_dir, "current_layout.json")
        report_path = os.path.join(job_dir, "report.json")
        planner_log_path = os.path.join(job_dir, "planner.log")
        submitted_at = time.perf_counter()
        process: subprocess.Popen[bytes] | None = None
        if self.ep_rank == 0:
            builder = self._hot_update_builder_path()
            if not os.path.isfile(builder):
                raise RuntimeError(f"PlaceMoE planner does not exist: {builder}.")
            primary_slots = layers[0].num_experts // self.ep_size
            redundant_slots = layers[0].num_local_experts - primary_slots
            if update_mode == "mapping":
                self._write_hot_update_current_layout(layers, input_layout_path)
            resources = self._hot_update_resources
            calibration = PlaceMoECalibration(
                inter_ms_per_byte=_HOT_UPDATE_INTER_MS_PER_BYTE,
                intra_ms_per_byte=_HOT_UPDATE_INTRA_MS_PER_BYTE,
                route_ms_per_assignment=_HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT,
                communication_multiplier=_HOT_UPDATE_COMMUNICATION_MULTIPLIER,
                compute_ms_per_assignment=_HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT,
                compute_multiplier=_HOT_UPDATE_COMPUTE_MULTIPLIER,
            )
            command = build_planner_command(
                PlannerCommandSpec(
                    python=sys.executable,
                    planner_path=builder,
                    route_root=os.path.join(job_dir, "routes"),
                    kind=update_kind,
                    layer_keys=tuple(layer.key for layer in layers),
                    ep_size=self.ep_size,
                    ranks_per_node=min(self.ep_size, self.hierarchy.local_world_size),
                    num_experts=layers[0].num_experts,
                    slots_per_rank=layers[0].num_local_experts,
                    primary_slots_per_rank=primary_slots,
                    redundant_slots_per_rank=redundant_slots,
                    hidden_size=layers[0].latest_hidden_size,
                    bytes_per_element=layers[0].latest_bytes_per_element,
                    output_layout=layout_path,
                    output_report=report_path,
                    input_layout=input_layout_path,
                ),
                calibration,
                resources,
            )
            environment = planner_environment(resources)
            os.makedirs(job_dir, exist_ok=True)
            with open(planner_log_path, "wb") as planner_log:
                process = launch_planner_process(command, stdout=planner_log, environment=environment)
        self._hot_update_controller.start(
            HotUpdateJob(
                kind=update_kind,
                source_step=int(training_step),
                placement_versions=tuple(int(layer.placement_version) for layer in layers),
                submitted_at=submitted_at,
                snapshot_ms=snapshot_ms,
                job_dir=job_dir,
                layout_path=layout_path,
                report_path=report_path,
                planner_log_path=planner_log_path,
                process=process,
            )
        )
        self._hot_update_last_source_step = int(training_step)
        self._hot_update_last_snapshot_ms = snapshot_ms
        self._hot_update_event(
            "submitted",
            update_mode=update_mode,
            source_step=int(training_step),
            snapshot_ms=snapshot_ms,
            job_dir=job_dir,
            planner_pid=None if process is None else process.pid,
        )

    def _hot_update_status(self, state: HotUpdateJob, device: torch.device) -> int:
        status = 0
        if self.ep_rank == 0:
            assert state.process is not None
            return_code = state.process.poll()
            status = 0 if return_code is None else (1 if return_code == 0 else 2)
        if self.ep_size > 1:
            status_tensor = torch.tensor([status], dtype=torch.int32, device=device)
            dist.broadcast(
                status_tensor,
                src=_ep_global_rank(self.ep_group, 0),
                group=self.ep_group,
            )
            status = int(status_tensor.item())
        return status

    def _finish_hot_update_job(self, state: HotUpdateJob) -> None:
        """Reap the complete planner process group before releasing the job."""

        process = getattr(state, "process", None)
        if self.ep_rank == 0 and process is not None:
            terminate_planner_process(process)
        self._hot_update_controller.finish()

    def _broadcast_hot_update_payload(
        self,
        state: HotUpdateJob,
        device: torch.device,
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        if self.ep_rank == 0:
            with open(state.layout_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        if self.ep_size > 1:
            objects: list[Any] = [payload]
            dist.broadcast_object_list(
                objects,
                src=_ep_global_rank(self.ep_group, 0),
                group=self.ep_group,
                device=device,
            )
            payload = objects[0]
        if not isinstance(payload, dict):
            raise RuntimeError("PlaceMoE hot update broadcast an invalid layout payload.")
        try:
            validate_placemoe_artifact(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("PlaceMoE hot update produced an invalid PlaceMoE artifact.") from error
        return payload

    def _prepare_hot_update_layer(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
        update_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and materialize one layer update without mutating runtime state."""

        target, owners = self._validate_placement_layout(
            layer,
            raw_layer.get("slot_to_logical", ()),
            raw_layer.get("owner_slots"),
        )
        if owners is None:
            raise RuntimeError(f"PlaceMoE layout for {layer.key} has no owner mapping.")
        source_lut = (
            torch.as_tensor(
                raw_layer.get("source_logical_to_physical", ()),
                dtype=torch.long,
            )
            .detach()
            .cpu()
        )
        expected_shape = (self.ep_size, layer.num_experts)
        if tuple(source_lut.shape) != expected_shape:
            raise RuntimeError(
                f"PlaceMoE mapping for {layer.key} has shape {tuple(source_lut.shape)}, expected {expected_shape}."
            )
        if bool(((source_lut < 0) | (source_lut >= layer.num_physical_slots)).any().item()):
            raise RuntimeError(f"PlaceMoE mapping for {layer.key} references an invalid slot.")

        current = self._layer_layout(layer)
        if update_mode == "mapping":
            if not torch.equal(target, current) or not torch.equal(
                owners,
                layer.logical_to_physical.detach().cpu(),
            ):
                raise RuntimeError(f"PlaceMoE mapping update attempted to change layout L for {layer.key}.")
            lookup_layout = current
        elif update_mode == "full":
            desired_experts = {int(value) for value in target.tolist() if int(value) >= 0}
            available_experts = {int(value) for value in current.tolist() if int(value) >= 0}
            missing = sorted(desired_experts - available_experts)
            if missing:
                raise RuntimeError(f"PlaceMoE layout cannot find current state for experts {missing} in {layer.key}.")
            lookup_layout = target
        else:
            raise RuntimeError(f"Unknown PlaceMoE update mode: {update_mode}.")

        logical = torch.arange(layer.num_experts, dtype=torch.long).view(1, -1)
        if not torch.equal(
            lookup_layout.index_select(0, source_lut.reshape(-1)).view_as(source_lut),
            logical.expand_as(source_lut),
        ):
            raise RuntimeError(f"PlaceMoE mapping for {layer.key} references the wrong expert.")
        return target, owners, source_lut

    @torch.no_grad()
    def _install_hot_update_layout(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
    ) -> int:
        target, owners, source_lut = self._prepare_hot_update_layer(layer, raw_layer, "full")

        current = self._layer_layout(layer)
        changed_slots = [slot for slot in range(layer.num_physical_slots) if current[slot] != target[slot]]
        if changed_slots:
            state_tensors = self._slot_op_state_tensors(layer)
            grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
            zero_entry_groups: list[tuple[int, list[_CoverTensorEntry]]] = []
            for dst_slot in changed_slots:
                desired = int(target[dst_slot].item())
                dst_rank = dst_slot // layer.num_local_experts
                if desired < 0:
                    zero_entry_groups.append(
                        (
                            dst_rank,
                            self._slot_op_cover_entries_from_tensors(
                                state_tensors,
                                num_local_experts=layer.num_local_experts,
                                src_slot=dst_slot,
                                dst_slot=dst_slot,
                            ),
                        )
                    )
                    continue
                candidates = [int(value) for value in torch.nonzero(current == desired, as_tuple=False).flatten()]
                if not candidates:
                    raise RuntimeError(
                        f"PlaceMoE layout cannot find current state for expert {desired} in {layer.key}."
                    )
                same_rank = [slot for slot in candidates if slot // layer.num_local_experts == dst_rank]
                owner_slot = int(layer.logical_to_physical[desired].item())
                src_slot = min(same_rank or ([owner_slot] if owner_slot in candidates else candidates))
                src_rank = src_slot // layer.num_local_experts
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
                zero_entry_groups=zero_entry_groups,
                debug_validate=self.debug_validate,
            )
            synchronize()

        layout_changed = not torch.equal(current, target)
        lut_changed = layer.source_logical_to_physical is None or not torch.equal(
            layer.source_logical_to_physical,
            source_lut,
        )
        layer.slot_to_logical = target
        layer.fixed_r2_layout = False
        layer.active_quota_policy = ()
        layer.pending_physical_routes = None
        layer.pending_route_data_ptr = 0
        layer.latest_physical_routes = None
        layer.latest_forward_traffic_endpoint_statistics = None
        self._refresh_layer_mapping_from_slots(layer, owners)
        layer.source_logical_to_physical = source_lut.clone()
        layer._device_source_mapping_cache.clear()
        layer.placement_version += int(layout_changed or lut_changed)
        return len(changed_slots)

    @torch.no_grad()
    def _install_hot_update_mapping(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
    ) -> int:
        _target, _owners, source_lut = self._prepare_hot_update_layer(layer, raw_layer, "mapping")
        changed = layer.source_logical_to_physical is None or not torch.equal(
            layer.source_logical_to_physical,
            source_lut,
        )
        layer.pending_physical_routes = None
        layer.pending_route_data_ptr = 0
        layer.latest_physical_routes = None
        layer.latest_forward_traffic_endpoint_statistics = None
        layer.source_logical_to_physical = source_lut.clone()
        layer._device_source_mapping_cache.clear()
        layer.placement_version += int(changed)
        return int(changed)

    @torch.no_grad()
    def _apply_hot_update(self, state: HotUpdateJob, training_step: int) -> str:
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        versions = tuple(int(layer.placement_version) for layer in layers)
        if versions != state.placement_versions:
            raise RuntimeError(
                f"PlaceMoE layout from step {state.source_step} is stale: "
                f"source versions={state.placement_versions}, current versions={versions}."
            )
        device = self._pipeline_device(layers[0])
        payload = self._broadcast_hot_update_payload(state, device)
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, dict) or set(raw_layers) != set(self.layers):
            raise RuntimeError("PlaceMoE layout layer keys do not match the registered model.")
        for layer in layers:
            raw_layer = raw_layers[layer.key]
            if not isinstance(raw_layer, dict):
                raise RuntimeError(f"PlaceMoE layout for {layer.key} is not a mapping.")
            self._prepare_hot_update_layer(layer, raw_layer, state.update_mode)

        migration_started = time.perf_counter()
        moved_slots = 0
        for layer in layers:
            raw_layer = raw_layers[layer.key]
            if state.update_mode == "full":
                moved_slots += self._install_hot_update_layout(layer, raw_layer)
            else:
                self._install_hot_update_mapping(layer, raw_layer)
        if self.ep_group is not None and self.ep_size > 1:
            dist.barrier(group=self.ep_group)
        migration_ms = (time.perf_counter() - migration_started) * 1000.0
        planner_ms = (time.perf_counter() - state.submitted_at) * 1000.0
        if self.ep_rank == 0 and os.path.isfile(state.report_path):
            with open(state.report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            planner_ms = float(report.get("aggregate", {}).get("planner_wall_ms", planner_ms))
        self._hot_update_updates += 1
        if state.update_mode == "full":
            self._hot_update_layout_updates += 1
        else:
            self._hot_update_mapping_updates += 1
        self._hot_update_last_apply_step = int(training_step)
        self._hot_update_last_staleness_steps = int(training_step) - int(state.source_step)
        self._hot_update_last_planner_ms = planner_ms
        self._hot_update_last_migration_ms = migration_ms
        self._hot_update_last_moved_slots = moved_slots
        self._hot_update_event(
            "applied",
            update_mode=state.update_mode,
            source_step=state.source_step,
            apply_step=int(training_step),
            staleness_steps=int(training_step) - int(state.source_step),
            snapshot_ms=state.snapshot_ms,
            planner_ms=planner_ms,
            migration_ms=migration_ms,
            moved_slots=moved_slots,
        )
        return f"placemoe_{state.update_mode}_update:{state.source_step}->{int(training_step)}:{moved_slots}"

    @torch.no_grad()
    def _run_hot_update_step(self, placement_step: int) -> str:
        training_step = int(placement_step) + 1
        self._hot_update_controller.observe_step(training_step)
        state = self._hot_update_controller.active_job
        if state is not None:
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            status = self._hot_update_status(state, self._pipeline_device(layers[0]))
            if status == 0:
                self.latest_pair = f"placemoe_{state.update_mode}_update_running"
                return self.latest_pair
            if status == 2:
                self._hot_update_event(
                    "failed",
                    update_mode=state.update_mode,
                    source_step=state.source_step,
                    planner_log_path=state.planner_log_path,
                )
                message = f"PlaceMoE planner failed for source step {state.source_step}; see {state.planner_log_path}."
                self._finish_hot_update_job(state)
                if self._hot_update_controller.failure_policy == "continue":
                    logger.error("%s Keeping the current layout and mapping.", message)
                    self.latest_pair = f"placemoe_hot_update_failed:{state.source_step}"
                    return self.latest_pair
                raise RuntimeError(message)
            try:
                self.latest_pair = self._apply_hot_update(state, training_step)
            finally:
                self._finish_hot_update_job(state)
            return self.latest_pair

        update_kind = self._hot_update_controller.next_update()
        if update_kind is None:
            self.latest_pair = "none"
            return self.latest_pair
        update_mode = update_kind.value
        self._launch_hot_update(
            placement_step=placement_step,
            training_step=training_step,
            update_mode=update_mode,
        )
        self.latest_pair = f"placemoe_{update_mode}_update_submitted:{training_step}"
        return self.latest_pair

    @torch.no_grad()
    def maybe_swap(self, step: int) -> str:
        self._begin_metrics_step(step)
        if self._hot_update:
            return self._run_hot_update_step(int(step))
        if self._online_lut_update:
            if (
                int(step) < self._online_lut_start_step
                or self.expert_swap_interval <= 0
                or int(step) % self.expert_swap_interval != 0
            ):
                self.latest_pair = "none"
                return self.latest_pair
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            with _full_timing_range("hiermoe_online_lut_plan"):
                committed = self._plan_online_lut_layers(layers)
            self.latest_pair = ",".join(committed) if committed else "none"
            return self.latest_pair
        if self._ablation_replay_mode != "off":
            return self._queue_ablation_replay_step(int(step))
        if self._cost_model_verify:
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            return self._run_cost_model_verification(layers, int(step))
        if self._online_freeze_cost_mode != "off":
            return self._run_online_freeze_step(int(step))
        if self._forward_reuse_cover:
            if (
                int(step) <= 0
                or (self._forward_reuse_cover_only_step >= 0 and int(step) != self._forward_reuse_cover_only_step)
                or self.expert_swap_interval <= 0
                or int(step) % self.expert_swap_interval != 0
            ):
                self.latest_pair = "none"
                return self.latest_pair
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            with _full_timing_range("hiermoe_forward_reuse_cover_plan"):
                committed: list[str] = []
                for round_index in range(self._forward_reuse_cover_rounds):
                    committed.extend(
                        self._plan_forward_reuse_cover_layers(
                            layers,
                            int(step) + round_index,
                        )
                    )
            self.latest_pair = ",".join(committed) if committed else "none"
            return self.latest_pair
        if self.fixed_pipeline_overlap:
            if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in self.layers.values()):
                layers = [self.layers[layer_key] for layer_key in self.layers]
                committed = self._plan_historical_layers(layers, int(step))
                self.latest_pair = ",".join(committed) if committed else "none"
                return self.latest_pair
            if self._npu_layer_owner_blocking:
                layers = [self.layers[layer_key] for layer_key in self._pipeline_layer_order or tuple(self.layers)]
                committed = self._plan_npu_layer_owner_layers(layers, int(step))
                self.latest_pair = ",".join(committed) if committed else "none"
                return self.latest_pair
            if self._uses_cpu_process_planner():
                return self._collect_cpu_process_plan(int(step))
            if self._cpu_planner_mode != "off":
                return self._collect_cpu_batched_plan(int(step))
            return self._collect_pipeline_plans(int(step))

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
            minimum_step = 0 if self.expert_swap_selector == "hiermoe_greedy_cover_p1" else 1
            if (
                int(step) < minimum_step
                or self.expert_swap_interval <= 0
                or int(step) % self.expert_swap_interval != 0
            ):
                self.latest_pair = "none"
                return self.latest_pair
            committed = []
            with _full_timing_range("hiermoe_current_route_plan"):
                layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
                if self.expert_swap_selector == "hiermoe_greedy_cover_p1" and self.expert_swap_mode == "step":
                    committed.extend(self._plan_historical_layers(layers, int(step)))
                else:
                    for layer in layers:
                        committed.extend(self._plan_current_layer(layer, int(step)))
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
        minimum_step = 0 if self.expert_swap_selector == "hiermoe_greedy_cover_p1" else 1
        if int(step) < minimum_step or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
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
    def _execute_sparse_group_slot_transfers(
        self,
        grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]],
        *,
        zero_entry_groups: Iterable[tuple[int, Iterable[_CoverTensorEntry]]] = (),
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Execute a deterministic directed placement with batched sparse P2P.

        Every EP rank has the same exact placement plan, so peers and payload
        sizes are known without a split-size collective. Only source and
        destination ranks participate; all destination slots are published
        after the complete P2P batch succeeds.
        """

        zero_groups = tuple((int(rank), tuple(entries)) for rank, entries in zero_entry_groups)
        all_entries = tuple(entry for entries in grouped_entries.values() for entry in entries) + tuple(
            entry for _rank, entries in zero_groups for entry in entries
        )
        if not all_entries:
            return
        group = self.ep_group if process_group is None else process_group
        if self.ep_size > 1 and group is None and any(src_rank != dst_rank for src_rank, dst_rank in grouped_entries):
            raise RuntimeError("HierMoE sparse placement migration requires an EP process group.")

        bucket_keys: set[tuple[torch.device, torch.dtype]] = set()
        send_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[tuple[torch.Tensor, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        recv_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[tuple[torch.Tensor, int, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        local_commits: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []

        for (src_rank, dst_rank), entries in sorted(grouped_entries.items()):
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                src_view = local_tensor.detach()[entry.src_slot]
                dst_view = local_tensor.detach()[entry.dst_slot]
                if tuple(src_view.shape) != tuple(dst_view.shape):
                    raise RuntimeError("HierMoE sparse placement copied incompatible expert slot shapes.")
                key = (src_view.device, src_view.dtype)
                bucket_keys.add(key)
                if src_rank == dst_rank:
                    if self.ep_rank == src_rank:
                        local_commits.append((local_tensor, int(entry.dst_slot), src_view.clone()))
                elif self.ep_rank == src_rank:
                    flat = src_view.contiguous().view(-1)
                    send_buckets[key][int(dst_rank)].append((flat, int(flat.numel())))
                elif self.ep_rank == dst_rank:
                    recv_buckets[key][int(src_rank)].append((local_tensor, int(entry.dst_slot), int(dst_view.numel())))

        for dst_rank, entries in zero_groups:
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                dst_view = local_tensor.detach()[entry.dst_slot]
                bucket_keys.add((dst_view.device, dst_view.dtype))
                if self.ep_rank == dst_rank:
                    local_commits.append((local_tensor, int(entry.dst_slot), None))

        pending_remote: list[tuple[torch.Tensor, dict[int, list[tuple[torch.Tensor, int, int]]], list[int]]] = []
        for device, dtype in sorted(
            bucket_keys,
            key=lambda item: (
                item[0].type,
                -1 if item[0].index is None else int(item[0].index),
                str(item[1]),
            ),
        ):
            key = (device, dtype)
            peer_sends = send_buckets.get(key, {})
            peer_recvs = recv_buckets.get(key, {})
            input_splits = [
                sum(numel for _view, numel in peer_sends.get(peer_rank, ())) for peer_rank in range(self.ep_size)
            ]
            output_splits = [
                sum(numel for _tensor, _slot, numel in peer_recvs.get(peer_rank, ()))
                for peer_rank in range(self.ep_size)
            ]
            send_numel = sum(input_splits)
            recv_numel = sum(output_splits)
            staging = self._ensure_swap_staging_buffer(device, dtype, max(send_numel, recv_numel))
            send_buffer = staging.send[:send_numel]
            recv_buffer = staging.recv[:recv_numel]

            offset = 0
            send_offsets = [0] * self.ep_size
            for peer_rank in range(self.ep_size):
                send_offsets[peer_rank] = offset
                for view, numel in peer_sends.get(peer_rank, ()):
                    send_buffer[offset : offset + numel].view_as(view).copy_(view)
                    offset += numel
            recv_offsets = [0] * self.ep_size
            offset = 0
            for peer_rank, split_size in enumerate(output_splits):
                recv_offsets[peer_rank] = offset
                offset += int(split_size)

            ops: list[dist.P2POp] = []
            if self.ep_size > 1:
                assert group is not None
                for peer_rank in range(self.ep_size):
                    peer_global_rank = _ep_global_rank(group, peer_rank)
                    input_size = int(input_splits[peer_rank])
                    if input_size:
                        start = send_offsets[peer_rank]
                        ops.append(
                            dist.P2POp(
                                dist.isend,
                                send_buffer[start : start + input_size],
                                peer_global_rank,
                                group,
                            )
                        )
                    output_size = int(output_splits[peer_rank])
                    if output_size:
                        start = recv_offsets[peer_rank]
                        ops.append(
                            dist.P2POp(
                                dist.irecv,
                                recv_buffer[start : start + output_size],
                                peer_global_rank,
                                group,
                            )
                        )
                works = dist.batch_isend_irecv(ops) if ops else ()
                for work in works:
                    work.wait()
                if works:
                    # HCCL ``Work.wait`` only guarantees host-side
                    # submission on Ascend.  The receive buffers are consumed
                    # below on the default stream, so establish the missing
                    # communication-stream -> default-stream dependency before
                    # reading them.  Without this fence a following slot copy
                    # can be queued against an in-flight recv and the first
                    # later collective stalls behind ``aclnnInplaceCopy``.
                    synchronize()
            elif send_numel:
                recv_buffer.copy_(send_buffer)
            pending_remote.append((recv_buffer, peer_recvs, output_splits))

        destinations: set[tuple[int, int]] = set()
        for local_tensor, dst_slot, staged in local_commits:
            destination = (id(local_tensor), int(dst_slot))
            if destination in destinations:
                raise RuntimeError("HierMoE sparse placement writes one tensor slot more than once.")
            destinations.add(destination)
            if staged is None:
                local_tensor.detach()[dst_slot].zero_()
            else:
                local_tensor.detach()[dst_slot].copy_(staged)
        for recv_buffer, peer_recvs, output_splits in pending_remote:
            offset = 0
            for peer_rank, split_size in enumerate(output_splits):
                inner_offset = offset
                for local_tensor, dst_slot, numel in peer_recvs.get(peer_rank, ()):
                    destination = (id(local_tensor), int(dst_slot))
                    if destination in destinations:
                        raise RuntimeError("HierMoE sparse placement writes one tensor slot more than once.")
                    destinations.add(destination)
                    staged = recv_buffer[inner_offset : inner_offset + numel].view_as(local_tensor.detach()[dst_slot])
                    local_tensor.detach()[dst_slot].copy_(staged)
                    inner_offset += numel
                if inner_offset - offset != int(split_size):
                    raise RuntimeError("HierMoE sparse placement payload size does not match the exact plan.")
                offset += int(split_size)

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
            transfer_group = self._swap_group if self._swap_group is not None else self.ep_group
            ops.extend(
                (
                    dist.P2POp(dist.isend, send_segment, peer_global_rank, transfer_group),
                    dist.P2POp(dist.irecv, recv_segment, peer_global_rank, transfer_group),
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
        *,
        validate_optimizer_state: bool = True,
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
        if validate_optimizer_state and self.debug_validate and self.ep_size > 1:
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
