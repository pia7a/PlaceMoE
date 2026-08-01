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

"""Source-aware token-to-copy mapping for a fixed PlaceMoE layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ProfileStatistics


@dataclass(frozen=True)
class MappingConfig:
    """Calibrated communication and computation costs for mapping updates."""

    ranks_per_node: int
    node_omega: float
    rank_omega: float
    gamma: float
    sweep_limit: int = 6

    def __post_init__(self) -> None:
        if self.ranks_per_node <= 0:
            raise ValueError("ranks_per_node must be positive.")
        if self.node_omega < 0 or self.rank_omega < 0 or self.gamma < 0:
            raise ValueError("Calibrated mapping coefficients must be non-negative.")
        if self.sweep_limit < 0:
            raise ValueError("Mapping sweep limit must be non-negative.")


@dataclass(frozen=True)
class MappingResult:
    """A source-aware instance mapping and convergence diagnostics."""

    mapping: np.ndarray
    sweeps: int
    changes: int
    peak_rank_load: float

    def __post_init__(self) -> None:
        mapping = np.asarray(self.mapping, dtype=np.int64).copy()
        mapping.setflags(write=False)
        object.__setattr__(self, "mapping", mapping)


def _validate_layout(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    *,
    ep_size: int,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    instance_ranks = np.asarray(instance_ranks, dtype=np.int64)
    if logical_instances.ndim != 1 or instance_ranks.shape != logical_instances.shape:
        raise ValueError("Logical instances and instance ranks must be matching vectors.")
    if bool((instance_ranks < 0).any()) or bool((instance_ranks >= ep_size).any()):
        raise ValueError("Instance ranks must reference a source rank in the profile.")
    if bool((logical_instances < -1).any()) or bool((logical_instances >= num_experts).any()):
        raise ValueError("Logical instances contain an invalid expert ID.")
    choices = tuple(np.flatnonzero(logical_instances == expert) for expert in range(num_experts))
    if any(len(expert_choices) == 0 for expert_choices in choices):
        raise ValueError("Every logical expert must have at least one physical copy.")
    return logical_instances, instance_ranks, choices


def validate_instance_mapping(
    mapping: np.ndarray,
    logical_instances: np.ndarray,
    *,
    ep_size: int,
    num_experts: int,
) -> np.ndarray:
    """Validate and return a mutable source-to-instance mapping copy."""

    mapping = np.asarray(mapping, dtype=np.int64).copy()
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    if mapping.shape != (ep_size, num_experts):
        raise ValueError("Mapping must have shape [source_rank, logical_expert].")
    if bool((mapping < 0).any()) or bool((mapping >= len(logical_instances)).any()):
        raise ValueError("Mapping references an instance outside the layout.")
    expected = np.broadcast_to(np.arange(num_experts, dtype=np.int64), mapping.shape)
    if not np.array_equal(logical_instances[mapping], expected):
        raise ValueError("Mapping references a copy of the wrong logical expert.")
    return mapping


def mapping_rank_loads(
    mapping: np.ndarray,
    instance_ranks: np.ndarray,
    statistics: ProfileStatistics,
) -> np.ndarray:
    """Aggregate non-deduplicated assignment demand onto destination ranks."""

    mapping = np.asarray(mapping, dtype=np.int64)
    instance_ranks = np.asarray(instance_ranks, dtype=np.int64)
    destination_ranks = instance_ranks[mapping]
    loads = np.zeros((statistics.ep_size,), dtype=np.float64)
    np.add.at(loads, destination_ranks, statistics.demand)
    return loads


def initialize_mapping(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ranks_per_node: int,
    prefer_node_local: bool = True,
) -> np.ndarray:
    """Construct the paper's demand-ordered initial mapping."""

    demand_by_source = np.asarray(demand_by_source, dtype=np.float64)
    if demand_by_source.ndim != 2 or bool((demand_by_source < 0).any()):
        raise ValueError("Source demand must be a non-negative [source_rank, expert] matrix.")
    ep_size, num_experts = demand_by_source.shape
    logical_instances, instance_ranks, choices = _validate_layout(
        logical_instances,
        instance_ranks,
        ep_size=ep_size,
        num_experts=num_experts,
    )
    if ep_size % ranks_per_node:
        raise ValueError("Profile EP size must be divisible by ranks_per_node.")
    mapping = np.full((ep_size, num_experts), -1, dtype=np.int64)
    rank_loads = np.zeros((ep_size,), dtype=np.float64)
    jobs = sorted(
        (
            (float(demand_by_source[source, expert]), source, expert)
            for source in range(ep_size)
            for expert in range(num_experts)
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    for amount, source, expert in jobs:
        expert_choices = choices[expert]
        source_node = source // ranks_per_node
        node_local = expert_choices[instance_ranks[expert_choices] // ranks_per_node == source_node]
        candidates = node_local if prefer_node_local and len(node_local) else expert_choices
        selected = min(
            candidates.tolist(),
            key=lambda instance: (
                rank_loads[int(instance_ranks[instance])] + amount,
                int(instance_ranks[instance]),
                instance,
            ),
        )
        mapping[source, expert] = selected
        rank_loads[int(instance_ranks[selected])] += amount
    return mapping


def _group_affinity_cache(
    mapping: np.ndarray,
    instance_groups: np.ndarray,
    statistics: ProfileStatistics,
    *,
    num_groups: int,
) -> np.ndarray:
    cache = np.zeros((statistics.ep_size, statistics.num_experts, num_groups), dtype=np.float64)
    for source in range(statistics.ep_size):
        selected_groups = instance_groups[mapping[source]]
        for expert in range(statistics.num_experts):
            np.add.at(cache[source, expert], selected_groups, statistics.affinity[source, expert])
    return cache


def optimize_mapping(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    initial_mapping: np.ndarray,
    statistics: ProfileStatistics,
    config: MappingConfig,
) -> MappingResult:
    """Refine a mapping using the calibrated score in the PlaceMoE paper."""

    logical_instances, instance_ranks, choices = _validate_layout(
        logical_instances,
        instance_ranks,
        ep_size=statistics.ep_size,
        num_experts=statistics.num_experts,
    )
    if statistics.ep_size % config.ranks_per_node:
        raise ValueError("Profile EP size must be divisible by ranks_per_node.")
    mapping = validate_instance_mapping(
        initial_mapping,
        logical_instances,
        ep_size=statistics.ep_size,
        num_experts=statistics.num_experts,
    )
    instance_nodes = instance_ranks // config.ranks_per_node
    num_nodes = statistics.ep_size // config.ranks_per_node
    node_cache = _group_affinity_cache(mapping, instance_nodes, statistics, num_groups=num_nodes)
    rank_cache = _group_affinity_cache(mapping, instance_ranks, statistics, num_groups=statistics.ep_size)
    rank_loads = mapping_rank_loads(mapping, instance_ranks, statistics)
    order = sorted(
        (
            (float(statistics.demand[source, expert]), source, expert)
            for source in range(statistics.ep_size)
            for expert in range(statistics.num_experts)
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )

    total_changes = 0
    completed_sweeps = 0
    for _ in range(config.sweep_limit):
        sweep_changes = 0
        for amount, source, expert in order:
            if len(choices[expert]) <= 1:
                continue
            current = int(mapping[source, expert])
            current_rank = int(instance_ranks[current])
            best: tuple[float, int, int] | None = None
            for candidate in choices[expert].tolist():
                candidate_rank = int(instance_ranks[candidate])
                candidate_node = int(instance_nodes[candidate])
                communication = float(config.node_omega) * (
                    node_cache[source, expert, candidate_node]
                    + amount * float(candidate_node == source // config.ranks_per_node)
                ) + float(config.rank_omega) * (
                    rank_cache[source, expert, candidate_rank] + amount * float(candidate_rank == source)
                )
                projected = rank_loads.copy()
                projected[current_rank] -= amount
                projected[candidate_rank] += amount
                score = communication - float(config.gamma) * float(projected.max(initial=0.0))
                key = (score, -candidate_rank, -candidate)
                if best is None or key > best:
                    best = key
            if best is None:
                raise RuntimeError(f"Logical expert {expert} has no mapping candidate.")
            selected = -best[2]
            if selected == current:
                continue

            selected_rank = int(instance_ranks[selected])
            selected_node = int(instance_nodes[selected])
            current_node = int(instance_nodes[current])
            rank_loads[current_rank] -= amount
            rank_loads[selected_rank] += amount
            affinity_column = statistics.affinity[source, :, expert]
            if selected_node != current_node:
                node_cache[source, :, current_node] -= affinity_column
                node_cache[source, :, selected_node] += affinity_column
            if selected_rank != current_rank:
                rank_cache[source, :, current_rank] -= affinity_column
                rank_cache[source, :, selected_rank] += affinity_column
            mapping[source, expert] = selected
            sweep_changes += 1
        completed_sweeps += 1
        total_changes += sweep_changes
        if sweep_changes == 0:
            break

    return MappingResult(
        mapping=mapping,
        sweeps=completed_sweeps,
        changes=total_changes,
        peak_rank_load=float(rank_loads.max(initial=0.0)),
    )
