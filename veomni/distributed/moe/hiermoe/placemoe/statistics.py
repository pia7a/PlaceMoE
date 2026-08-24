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

"""Token-level profile statistics used by the PlaceMoE optimizer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .types import EMPTY_EXPERT, ProfileStatistics


RouteSamples = Sequence[Sequence[torch.Tensor]]


def profile_route_statistics(samples: RouteSamples, *, num_experts: int) -> ProfileStatistics:
    """Compute the paper's source-conditioned demand and co-selection affinity."""

    if not samples or not samples[0]:
        raise ValueError("At least one routing sample and source rank are required.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    ep_size = len(samples[0])
    demand = np.zeros((ep_size, num_experts), dtype=np.float64)
    affinity = np.zeros((ep_size, num_experts, num_experts), dtype=np.float64)
    for sample in samples:
        if len(sample) != ep_size:
            raise ValueError("All routing samples must contain the same source ranks.")
        for source_rank, route in enumerate(sample):
            if route.ndim != 2 or route.shape[1] <= 0:
                raise ValueError("Each route tensor must have shape [tokens, top_k].")
            route = route.detach().to(device="cpu", dtype=torch.long)
            if route.numel() and bool(((route < 0) | (route >= num_experts)).any().item()):
                raise ValueError("Routing sample contains an invalid logical expert ID.")
            demand[source_rank] += torch.bincount(route.reshape(-1), minlength=num_experts).numpy()
            for lhs in range(int(route.shape[1])):
                for rhs in range(lhs + 1, int(route.shape[1])):
                    pair = route[:, lhs] * num_experts + route[:, rhs]
                    counts = torch.bincount(pair, minlength=num_experts * num_experts).reshape(
                        num_experts,
                        num_experts,
                    )
                    values = counts.numpy().astype(np.float64, copy=False)
                    affinity[source_rank] += values + values.T
    for matrix in affinity:
        np.fill_diagonal(matrix, 0.0)
    return ProfileStatistics(demand=demand, affinity=affinity)


def _copy_counts(copy_to_expert: np.ndarray, num_experts: int) -> np.ndarray:
    active = copy_to_expert >= 0
    if bool((copy_to_expert < EMPTY_EXPERT).any() or (copy_to_expert >= num_experts).any()):
        raise ValueError("copy_to_expert contains an invalid logical expert ID.")
    counts = np.bincount(copy_to_expert[active], minlength=num_experts)
    if bool((counts < 1).any()):
        raise ValueError("Every logical expert must retain at least one physical copy.")
    return counts


def uniform_copy_statistics(
    statistics: ProfileStatistics,
    copy_to_expert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split logical statistics uniformly before a token-to-copy mapping exists."""

    copy_to_expert = np.asarray(copy_to_expert, dtype=np.int64)
    if copy_to_expert.ndim != 1:
        raise ValueError("copy_to_expert must be one-dimensional.")
    counts = _copy_counts(copy_to_expert, statistics.num_experts)
    active = copy_to_expert >= 0
    active_experts = copy_to_expert[active]
    demand = np.zeros((statistics.ep_size, len(copy_to_expert)), dtype=np.float64)
    demand[:, active] = statistics.demand[:, active_experts] / counts[active_experts][None, :]
    affinity = np.zeros(
        (statistics.ep_size, len(copy_to_expert), len(copy_to_expert)),
        dtype=np.float64,
    )
    for lhs, expert_lhs in enumerate(copy_to_expert.tolist()):
        if expert_lhs == EMPTY_EXPERT:
            continue
        for rhs in range(lhs + 1, len(copy_to_expert)):
            expert_rhs = int(copy_to_expert[rhs])
            if expert_rhs == EMPTY_EXPERT or expert_lhs == expert_rhs:
                continue
            values = statistics.affinity[:, expert_lhs, expert_rhs] / (counts[expert_lhs] * counts[expert_rhs])
            affinity[:, lhs, rhs] = values
            affinity[:, rhs, lhs] = values
    return demand, affinity


def project_statistics_to_copies(
    statistics: ProfileStatistics,
    copy_to_expert: np.ndarray,
    mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project logical statistics onto the physical copies selected by mapping M."""

    copy_to_expert = np.asarray(copy_to_expert, dtype=np.int64)
    mapping = np.asarray(mapping, dtype=np.int64)
    _copy_counts(copy_to_expert, statistics.num_experts)
    if mapping.shape != (statistics.ep_size, statistics.num_experts):
        raise ValueError("mapping must have shape [source_rank, logical_expert].")
    if bool((mapping < 0).any() or (mapping >= len(copy_to_expert)).any()):
        raise ValueError("mapping references an unavailable physical copy.")
    requested = np.broadcast_to(np.arange(statistics.num_experts, dtype=np.int64), mapping.shape)
    if not np.array_equal(copy_to_expert[mapping], requested):
        raise ValueError("mapping references a copy of the wrong logical expert.")

    demand = np.zeros((statistics.ep_size, len(copy_to_expert)), dtype=np.float64)
    affinity = np.zeros(
        (statistics.ep_size, len(copy_to_expert), len(copy_to_expert)),
        dtype=np.float64,
    )
    for source_rank in range(statistics.ep_size):
        selected = mapping[source_rank]
        np.add.at(demand[source_rank], selected, statistics.demand[source_rank])
        for expert_lhs in range(statistics.num_experts):
            copy_lhs = int(selected[expert_lhs])
            for expert_rhs in range(expert_lhs + 1, statistics.num_experts):
                value = statistics.affinity[source_rank, expert_lhs, expert_rhs]
                if value == 0:
                    continue
                copy_rhs = int(selected[expert_rhs])
                affinity[source_rank, copy_lhs, copy_rhs] += value
                affinity[source_rank, copy_rhs, copy_lhs] += value
    return demand, affinity
