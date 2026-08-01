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

"""Calibrated capacity-constrained affinity partitioning for PlaceMoE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for one hierarchy level of the PlaceMoE partitioner.

    ``omega`` and ``gamma`` express communication reuse and expert-compute
    load in the same time unit. ``ranks_per_group`` implements the per-rank
    normalization in the paper's partition objective.
    """

    capacities: tuple[int, ...]
    ranks_per_group: tuple[int, ...]
    omega: float
    gamma: float
    restarts: int = 3
    assignment_iterations: int = 12
    exchange_limit: int = 24
    seed: int = 0
    improvement_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.capacities or any(capacity <= 0 for capacity in self.capacities):
            raise ValueError("Partition capacities must be positive.")
        if len(self.ranks_per_group) != len(self.capacities):
            raise ValueError("Each partition group must define its rank count.")
        if any(ranks <= 0 for ranks in self.ranks_per_group):
            raise ValueError("Ranks per partition group must be positive.")
        if self.omega < 0 or self.gamma < 0:
            raise ValueError("Calibrated partition coefficients must be non-negative.")
        if self.restarts <= 0:
            raise ValueError("Partition restarts must be positive.")
        if self.assignment_iterations <= 0:
            raise ValueError("Capacity-assignment iterations must be positive.")
        if self.exchange_limit < 0:
            raise ValueError("Exchange limit must be non-negative.")
        if self.improvement_tolerance < 0:
            raise ValueError("Improvement tolerance must be non-negative.")


@dataclass(frozen=True)
class PartitionResult:
    """One capacity-feasible partition candidate and its calibrated cost."""

    labels: np.ndarray
    objective: float
    within_affinity: float
    peak_assignments_per_rank: float
    exchanges: tuple[tuple[int, int], ...]
    restart: int

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).copy()
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)


def _validate_inputs(
    affinity: np.ndarray, demand: np.ndarray, config: PartitionConfig
) -> tuple[np.ndarray, np.ndarray]:
    affinity = np.asarray(affinity, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)
    size = int(demand.shape[0]) if demand.ndim == 1 else -1
    if size != sum(config.capacities):
        raise ValueError("Demand length must equal the total partition capacity.")
    if affinity.shape != (size, size):
        raise ValueError("Affinity must be a square matrix matching the demand length.")
    if bool((affinity < 0).any()) or bool((demand < 0).any()):
        raise ValueError("Partition demand and affinity must be non-negative.")
    if not np.allclose(affinity, affinity.T):
        raise ValueError("Partition affinity must be symmetric.")
    if not np.allclose(np.diag(affinity), 0.0):
        raise ValueError("Partition affinity diagonal must be zero.")
    return affinity, demand


def partition_objective(
    affinity: np.ndarray,
    demand: np.ndarray,
    labels: np.ndarray,
    config: PartitionConfig,
) -> tuple[float, float, float]:
    """Return paper objective, within-group affinity, and peak per-rank load."""

    affinity, demand = _validate_inputs(affinity, demand, config)
    labels = np.asarray(labels, dtype=np.int64)
    parts = len(config.capacities)
    if labels.shape != demand.shape or bool((labels < 0).any()) or bool((labels >= parts).any()):
        raise ValueError("Partition labels are invalid.")
    if not np.array_equal(np.bincount(labels, minlength=parts), np.asarray(config.capacities)):
        raise ValueError("Partition labels do not satisfy the configured capacities.")

    within = 0.0
    for part in range(parts):
        members = np.flatnonzero(labels == part)
        within += float(np.triu(affinity[np.ix_(members, members)], k=1).sum())
    loads = np.bincount(labels, weights=demand, minlength=parts)
    per_rank = loads / np.asarray(config.ranks_per_group, dtype=np.float64)
    peak = float(per_rank.max(initial=0.0))
    return -float(config.omega) * within + float(config.gamma) * peak, within, peak


def _spectral_embedding(affinity: np.ndarray, dimensions: int) -> np.ndarray:
    degree = np.maximum(affinity.sum(axis=1), 1.0)
    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    embedding = eigenvectors[:, np.argsort(eigenvalues)[-dimensions:]]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norms, 1e-12)


def _capacity_assignment(embedding: np.ndarray, config: PartitionConfig, *, seed: int) -> np.ndarray:
    parts = len(config.capacities)
    kmeans = KMeans(n_clusters=parts, random_state=seed, n_init=1, max_iter=100)
    kmeans.fit(embedding)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    labels = np.full((embedding.shape[0],), -1, dtype=np.int64)
    slot_groups = np.repeat(np.arange(parts, dtype=np.int64), config.capacities)

    for _ in range(config.assignment_iterations):
        repeated_centers = centers[slot_groups]
        distances = ((embedding[:, None, :] - repeated_centers[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(distances)
        labels[rows] = slot_groups[columns]
        new_centers = np.stack([embedding[labels == part].mean(axis=0) for part in range(parts)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def _refine_partition(
    affinity: np.ndarray,
    demand: np.ndarray,
    initial_labels: np.ndarray,
    config: PartitionConfig,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    labels = np.asarray(initial_labels, dtype=np.int64).copy()
    parts = len(config.capacities)
    membership = np.eye(parts, dtype=np.float64)[labels]
    affinity_to_part = affinity @ membership
    loads = np.bincount(labels, weights=demand, minlength=parts).astype(np.float64)
    ranks = np.asarray(config.ranks_per_group, dtype=np.float64)
    all_left, all_right = np.triu_indices(len(labels), k=1)
    exchanges: list[tuple[int, int]] = []

    for _ in range(config.exchange_limit):
        cross_group = labels[all_left] != labels[all_right]
        left = all_left[cross_group]
        right = all_right[cross_group]
        if not len(left):
            break
        left_parts = labels[left]
        right_parts = labels[right]
        affinity_gain = (
            affinity_to_part[left, right_parts]
            - affinity_to_part[left, left_parts]
            + affinity_to_part[right, left_parts]
            - affinity_to_part[right, right_parts]
            - 2.0 * affinity[left, right]
        )

        current_peak = float((loads / ranks).max(initial=0.0))
        candidate_loads = np.broadcast_to(loads, (len(left), parts)).copy()
        rows = np.arange(len(left))
        candidate_loads[rows, left_parts] += demand[right] - demand[left]
        candidate_loads[rows, right_parts] += demand[left] - demand[right]
        candidate_peaks = (candidate_loads / ranks[None, :]).max(axis=1)
        objective_delta = -float(config.omega) * affinity_gain + float(config.gamma) * (candidate_peaks - current_peak)
        best = int(np.argmin(objective_delta))
        if objective_delta[best] >= -float(config.improvement_tolerance):
            break

        lhs = int(left[best])
        rhs = int(right[best])
        lhs_part = int(labels[lhs])
        rhs_part = int(labels[rhs])
        affinity_to_part[:, lhs_part] += affinity[:, rhs] - affinity[:, lhs]
        affinity_to_part[:, rhs_part] += affinity[:, lhs] - affinity[:, rhs]
        loads[lhs_part] += demand[rhs] - demand[lhs]
        loads[rhs_part] += demand[lhs] - demand[rhs]
        labels[lhs], labels[rhs] = labels[rhs], labels[lhs]
        exchanges.append((lhs, rhs))
    return labels, tuple(exchanges)


def partition_items(affinity: np.ndarray, demand: np.ndarray, config: PartitionConfig) -> tuple[PartitionResult, ...]:
    """Generate unique calibrated partition candidates from spectral restarts."""

    affinity, demand = _validate_inputs(affinity, demand, config)
    embedding = _spectral_embedding(affinity, len(config.capacities))
    results: list[PartitionResult] = []
    seen: set[tuple[tuple[int, int, tuple[int, ...]], ...]] = set()
    for restart in range(config.restarts):
        initial = _capacity_assignment(embedding, config, seed=config.seed + 7919 * restart)
        labels, exchanges = _refine_partition(affinity, demand, initial, config)
        groups = tuple(
            sorted(
                (
                    config.capacities[part],
                    config.ranks_per_group[part],
                    tuple(np.flatnonzero(labels == part).tolist()),
                )
                for part in range(len(config.capacities))
            )
        )
        if groups in seen:
            continue
        seen.add(groups)
        objective, within, peak = partition_objective(affinity, demand, labels, config)
        results.append(
            PartitionResult(
                labels=labels,
                objective=objective,
                within_affinity=within,
                peak_assignments_per_rank=peak,
                exchanges=exchanges,
                restart=restart,
            )
        )
    return tuple(sorted(results, key=lambda result: (result.objective, result.restart)))


def map_groups_to_locations(
    labels: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    sources_by_location: tuple[np.ndarray, ...] | list[np.ndarray],
    group_capacities: tuple[int, ...] | None = None,
    location_capacities: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Map abstract groups to compatible physical locations by source locality."""

    labels = np.asarray(labels, dtype=np.int64)
    demand_by_source = np.asarray(demand_by_source, dtype=np.float64)
    locations = tuple(np.asarray(sources, dtype=np.int64) for sources in sources_by_location)
    parts = len(locations)
    if labels.ndim != 1 or demand_by_source.ndim != 2 or demand_by_source.shape[1] != len(labels):
        raise ValueError("Source demand must have shape [source_rank, item].")
    if parts == 0 or bool((labels < 0).any()) or bool((labels >= parts).any()):
        raise ValueError("Physical locations must match the abstract partition groups.")
    if group_capacities is None:
        group_capacities = tuple(np.bincount(labels, minlength=parts).tolist())
    if location_capacities is None:
        location_capacities = group_capacities
    if len(group_capacities) != parts or len(location_capacities) != parts:
        raise ValueError("Group and location capacities must match the number of locations.")

    benefit = np.full((parts, parts), -np.inf, dtype=np.float64)
    for group in range(parts):
        members = np.flatnonzero(labels == group)
        for location, sources in enumerate(locations):
            if group_capacities[group] == location_capacities[location]:
                benefit[group, location] = demand_by_source[np.ix_(sources, members)].sum()
    finite = np.isfinite(benefit)
    if not bool(finite.any(axis=1).all()) or not bool(finite.any(axis=0).all()):
        raise ValueError("No capacity-compatible physical mapping exists.")
    cost = np.where(finite, -benefit, 1e30)
    groups, assigned_locations = linear_sum_assignment(cost)
    if bool((cost[groups, assigned_locations] >= 1e29).any()):
        raise ValueError("No capacity-compatible one-to-one physical mapping exists.")
    group_to_location = np.full((parts,), -1, dtype=np.int64)
    group_to_location[groups] = assigned_locations
    return group_to_location[labels]
