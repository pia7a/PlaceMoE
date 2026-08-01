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

"""Bounded dynamic programming for exact-budget replica allocation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bounded_group_shortlist(
    groups: Sequence[np.ndarray],
    demand: np.ndarray,
    *,
    selected_groups: int,
    candidate_limit: int,
) -> list[tuple[int, ...]]:
    """Retain the highest-demand fixed-cardinality group subsets."""

    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    if not 0 <= selected_groups <= len(groups):
        raise ValueError("selected_groups is outside the available group range.")
    demand = np.asarray(demand, dtype=np.float64)
    if demand.ndim != 1:
        raise ValueError("demand must be one-dimensional.")
    scores: list[float] = []
    for group in groups:
        group = np.asarray(group, dtype=np.int64)
        if group.ndim != 1 or bool((group < 0).any() or (group >= len(demand)).any()):
            raise ValueError("Replica group contains an invalid logical expert ID.")
        scores.append(float(demand[group].sum()))

    states: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(selected_groups + 1)]
    states[0] = [(0.0, ())]
    for group_index, score in enumerate(scores):
        upper = min(selected_groups, group_index + 1)
        for count in range(upper, 0, -1):
            proposals = states[count] + [
                (value + score, indices + (group_index,)) for value, indices in states[count - 1]
            ]
            proposals.sort(key=lambda row: (-row[0], row[1]))
            states[count] = proposals[:candidate_limit]
    return [indices for _, indices in states[selected_groups]]


def build_replica_allocations(
    partitions: Sequence[np.ndarray],
    demand: np.ndarray,
    *,
    additional_copies: int,
    candidate_limit: int,
) -> list[np.ndarray]:
    """Build exact-budget allocations from equal-capacity affinity partitions."""

    demand = np.asarray(demand, dtype=np.float64)
    if demand.ndim != 1 or len(demand) == 0:
        raise ValueError("demand must contain one value per logical expert.")
    num_experts = len(demand)
    if additional_copies < 0:
        raise ValueError("additional_copies must be non-negative.")
    full_rounds, residual = divmod(additional_copies, num_experts)
    full = np.tile(np.arange(num_experts, dtype=np.int64), full_rounds)
    if residual == 0:
        return [full]

    group_size = int(np.gcd(num_experts, residual))
    expected_groups = num_experts // group_size
    selected_groups = residual // group_size
    rows: list[tuple[float, bytes, np.ndarray]] = []
    seen: set[bytes] = set()
    for partition in partitions:
        partition = np.asarray(partition, dtype=np.int64)
        if partition.shape != (num_experts,):
            raise ValueError("Each partition must contain one label per logical expert.")
        labels = np.unique(partition)
        if not np.array_equal(labels, np.arange(expected_groups, dtype=np.int64)):
            raise ValueError("Partition labels must be consecutive and match the required group count.")
        groups = [np.flatnonzero(partition == label) for label in labels]
        if any(len(group) != group_size for group in groups):
            raise ValueError("Replica allocation requires equal-capacity expert groups.")
        for combination in bounded_group_shortlist(
            groups,
            demand,
            selected_groups=selected_groups,
            candidate_limit=candidate_limit,
        ):
            residual_experts = np.sort(np.concatenate([groups[index] for index in combination]))
            allocation = np.concatenate([full, residual_experts]).astype(np.int64, copy=False)
            key = allocation.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append((-float(demand[residual_experts].sum()), key, allocation))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [allocation for _, _, allocation in rows[:candidate_limit]]
