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

import itertools
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


@dataclass(frozen=True)
class CommunityMappingConfig:
    """Calibrated node-level costs for affinity-coarsened mapping updates."""

    ranks_per_node: int
    communication_ms_per_token: float
    assignment_ms_per_assignment: float
    sweep_limit: int = 4
    row_candidate_limit: int = 64
    beam_width: int = 64

    def __post_init__(self) -> None:
        if self.ranks_per_node <= 0:
            raise ValueError("ranks_per_node must be positive.")
        if self.communication_ms_per_token < 0 or self.assignment_ms_per_assignment < 0:
            raise ValueError("Community mapping coefficients must be non-negative.")
        if self.sweep_limit < 0:
            raise ValueError("Community mapping sweep limit must be non-negative.")
        if self.row_candidate_limit <= 0 or self.beam_width <= 0:
            raise ValueError("Community mapping row and beam limits must be positive.")


@dataclass(frozen=True)
class CommunityMappingResult:
    """A group-coherent mapping proposal and its node-level proxy cost."""

    mapping: np.ndarray
    destination_nodes: np.ndarray
    sweeps: int
    changes: int
    proxy_cost: float

    def __post_init__(self) -> None:
        for name in ("mapping", "destination_nodes"):
            value = np.asarray(getattr(self, name), dtype=np.int64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.proxy_cost):
            raise ValueError("Community mapping proxy cost must be finite.")


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


def optimize_mapping_normalized(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    initial_mapping: np.ndarray,
    statistics: ProfileStatistics,
    *,
    ranks_per_node: int,
    assignment_weight: float,
    sweep_limit: int = 6,
    node_weight: float = 1.0,
    rank_weight: float = 0.15,
) -> MappingResult:
    """Generate a scale-normalized mapping proposal for exact-route selection.

    This proposal is intentionally separate from the paper's calibrated
    coordinate update. Normalizing affinity and demand protects the candidate
    search from pairwise-affinity scale error; the exact route evaluator still
    decides whether the resulting complete plan is useful.
    """

    if ranks_per_node <= 0:
        raise ValueError("ranks_per_node must be positive.")
    if sweep_limit < 0:
        raise ValueError("Mapping sweep limit must be non-negative.")
    if assignment_weight < 0 or node_weight < 0 or rank_weight < 0:
        raise ValueError("Normalized mapping weights must be non-negative.")
    logical_instances, instance_ranks, choices = _validate_layout(
        logical_instances,
        instance_ranks,
        ep_size=statistics.ep_size,
        num_experts=statistics.num_experts,
    )
    if statistics.ep_size % ranks_per_node:
        raise ValueError("Profile EP size must be divisible by ranks_per_node.")
    mapping = validate_instance_mapping(
        initial_mapping,
        logical_instances,
        ep_size=statistics.ep_size,
        num_experts=statistics.num_experts,
    )
    rank_loads = mapping_rank_loads(mapping, instance_ranks, statistics)
    total_affinity = max(float(statistics.affinity.sum()), 1.0)
    total_demand = max(float(statistics.demand.sum()), 1.0)
    total_changes = 0
    completed_sweeps = 0

    for _ in range(sweep_limit):
        sweep_changes = 0
        for source in range(statistics.ep_size):
            order = np.argsort(-statistics.demand[source], kind="stable")
            selected_ranks = instance_ranks[mapping[source]].copy()
            selected_nodes = selected_ranks // ranks_per_node
            source_node = source // ranks_per_node
            for expert in order.tolist():
                if len(choices[expert]) <= 1:
                    continue
                current = int(mapping[source, expert])
                current_rank = int(instance_ranks[current])
                amount = float(statistics.demand[source, expert])
                affinity_row = statistics.affinity[source, expert]
                best: tuple[float, int, int] | None = None
                for candidate in choices[expert].tolist():
                    candidate_rank = int(instance_ranks[candidate])
                    candidate_node = candidate_rank // ranks_per_node
                    projected = rank_loads.copy()
                    projected[current_rank] -= amount
                    projected[candidate_rank] += amount
                    node_reward = float(affinity_row[selected_nodes == candidate_node].sum())
                    rank_reward = float(affinity_row[selected_ranks == candidate_rank].sum())
                    if candidate_node == source_node:
                        node_reward += amount
                    if candidate_rank == source:
                        rank_reward += amount
                    score = (
                        node_weight * node_reward / total_affinity
                        + rank_weight * rank_reward / total_affinity
                        - assignment_weight * float(projected.max(initial=0.0)) / total_demand
                    )
                    key = (score, -candidate_rank, -candidate)
                    if best is None or key > best:
                        best = key
                if best is None:
                    raise RuntimeError(f"Logical expert {expert} has no mapping candidate.")
                selected = -best[2]
                if selected == current:
                    continue
                selected_rank = int(instance_ranks[selected])
                rank_loads[current_rank] -= amount
                rank_loads[selected_rank] += amount
                mapping[source, expert] = selected
                selected_ranks[expert] = selected_rank
                selected_nodes[expert] = selected_rank // ranks_per_node
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


def community_intersection_hits(mask_histogram: np.ndarray) -> np.ndarray:
    """Precompute token counts whose community mask intersects each subset."""

    mask_histogram = np.asarray(mask_histogram, dtype=np.int64)
    if mask_histogram.ndim != 2 or mask_histogram.shape[1] <= 0:
        raise ValueError("Community-mask histogram must have shape [source_node, mask].")
    num_masks = int(mask_histogram.shape[1])
    if num_masks & (num_masks - 1):
        raise ValueError("Community-mask histogram width must be a power of two.")
    mask_ids = np.arange(num_masks, dtype=np.int64)
    result = np.zeros_like(mask_histogram)
    for subset in range(1, num_masks):
        result[:, subset] = mask_histogram[:, (mask_ids & subset) != 0].sum(axis=1)
    return result


def _remote_group_hits(intersection_hits: np.ndarray, destination_nodes: np.ndarray, source_node: int) -> float:
    """Count destination-node unions exactly from precomputed subset hits."""

    intersection_hits = np.asarray(intersection_hits, dtype=np.int64)
    destination_nodes = np.asarray(destination_nodes, dtype=np.int64)
    num_nodes = int(destination_nodes.max(initial=source_node)) + 1
    community_subsets = np.zeros((num_nodes,), dtype=np.int64)
    for community, node in enumerate(destination_nodes.tolist()):
        community_subsets[node] |= 1 << community
    return float(
        sum(
            int(intersection_hits[community_subsets[node]])
            for node in range(num_nodes)
            if node != source_node
        )
    )


def optimize_community_mapping(
    logical_instances: np.ndarray,
    instance_nodes: np.ndarray,
    logical_communities: np.ndarray,
    assignments_by_community: np.ndarray,
    community_mask_histogram: np.ndarray,
    config: CommunityMappingConfig,
    *,
    intersection_hits: np.ndarray | None = None,
) -> CommunityMappingResult:
    """Jointly map every source-node and expert-community block.

    The move score uses exact token destination-node unions summarized by
    ``community_mask_histogram``. Computation is represented by the calibrated
    projected peak assignment load per rank. The resulting mapping is a
    proposal; complete token-route replay remains the final selector.
    """

    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    instance_nodes = np.asarray(instance_nodes, dtype=np.int64)
    logical_communities = np.asarray(logical_communities, dtype=np.int64)
    assignments_by_community = np.asarray(assignments_by_community, dtype=np.float64)
    community_mask_histogram = np.asarray(community_mask_histogram, dtype=np.int64)
    if logical_instances.ndim != 1 or instance_nodes.shape != logical_instances.shape:
        raise ValueError("Logical instances and instance nodes must be matching vectors.")
    if logical_communities.ndim != 1 or bool((logical_communities < 0).any()):
        raise ValueError("Logical communities must contain one non-negative label per expert.")
    num_experts = len(logical_communities)
    num_communities = int(logical_communities.max(initial=-1)) + 1
    if not np.array_equal(np.unique(logical_communities), np.arange(num_communities)):
        raise ValueError("Logical community labels must be consecutive.")
    if bool((logical_instances < -1).any()) or bool((logical_instances >= num_experts).any()):
        raise ValueError("Logical instances contain an invalid expert ID.")
    if assignments_by_community.ndim != 2 or assignments_by_community.shape[1] != num_communities:
        raise ValueError("Community assignments must have shape [source_node, community].")
    num_nodes = int(assignments_by_community.shape[0])
    if num_nodes <= 0 or bool((instance_nodes < 0).any()) or bool((instance_nodes >= num_nodes).any()):
        raise ValueError("Instance nodes do not match the source-node topology.")
    if community_mask_histogram.shape != (num_nodes, 1 << num_communities):
        raise ValueError("Community-mask histogram does not match the logical communities.")
    if intersection_hits is None:
        intersection_hits = community_intersection_hits(community_mask_histogram)
    else:
        intersection_hits = np.asarray(intersection_hits, dtype=np.int64)
        if intersection_hits.shape != community_mask_histogram.shape:
            raise ValueError("Precomputed community intersection hits have an invalid shape.")

    instances_by_expert_node = np.full((num_experts, num_nodes), -1, dtype=np.int64)
    for instance, expert in enumerate(logical_instances.tolist()):
        if expert < 0:
            continue
        node = int(instance_nodes[instance])
        if instances_by_expert_node[expert, node] >= 0:
            raise ValueError("One expert has multiple copies on the same node.")
        instances_by_expert_node[expert, node] = instance

    nodes_by_community: list[np.ndarray] = []
    for community in range(num_communities):
        experts = np.flatnonzero(logical_communities == community)
        if not len(experts):
            raise ValueError("Every logical community must contain an expert.")
        common = np.flatnonzero((instances_by_expert_node[experts] >= 0).all(axis=0))
        if not len(common):
            raise RuntimeError("An expert community has no common serving node.")
        nodes_by_community.append(common)

    def proxy_cost(remote_hits: np.ndarray, assignments: np.ndarray) -> float:
        communication = float(config.communication_ms_per_token) * float(remote_hits.sum())
        computation = (
            float(config.assignment_ms_per_assignment)
            * float(assignments.max(initial=0.0))
            / float(config.ranks_per_node)
        )
        return communication + computation

    source_candidates: list[list[tuple[float, tuple[int, ...], float, np.ndarray]]] = []
    for source_node in range(num_nodes):
        rows: list[tuple[float, tuple[int, ...], float, np.ndarray]] = []
        row_count = 1
        for nodes in nodes_by_community:
            row_count *= len(nodes)
        enumeration_limit = config.row_candidate_limit * config.beam_width
        if row_count <= enumeration_limit:
            for destinations in itertools.product(*(nodes.tolist() for nodes in nodes_by_community)):
                destination_row = np.asarray(destinations, dtype=np.int64)
                remote_hits = _remote_group_hits(
                    intersection_hits[source_node],
                    destination_row,
                    source_node,
                )
                assignments = np.zeros((num_nodes,), dtype=np.float64)
                np.add.at(assignments, destination_row, assignments_by_community[source_node])
                local_cost = proxy_cost(np.asarray([remote_hits]), assignments)
                rows.append((local_cost, tuple(int(value) for value in destinations), remote_hits, assignments))
        else:
            partial_rows: list[tuple[float, tuple[int, ...], float, np.ndarray]] = [
                (0.0, (), 0.0, np.zeros((num_nodes,), dtype=np.float64))
            ]
            for community, nodes in enumerate(nodes_by_community):
                expanded: list[tuple[float, tuple[int, ...], float, np.ndarray]] = []
                for _, destinations, _, assignments in partial_rows:
                    for node in nodes.tolist():
                        candidate_destinations = destinations + (int(node),)
                        candidate_remote_hits = _remote_group_hits(
                            intersection_hits[source_node],
                            np.asarray(candidate_destinations, dtype=np.int64),
                            source_node,
                        )
                        candidate_assignments = assignments.copy()
                        candidate_assignments[node] += assignments_by_community[source_node, community]
                        local_cost = proxy_cost(
                            np.asarray([candidate_remote_hits]),
                            candidate_assignments,
                        )
                        expanded.append(
                            (
                                local_cost,
                                candidate_destinations,
                                candidate_remote_hits,
                                candidate_assignments,
                            )
                        )
                expanded.sort(key=lambda row: (row[0], row[1]))
                partial_rows = expanded[: config.row_candidate_limit]
            rows = partial_rows
        rows.sort(key=lambda row: (row[0], row[1]))
        source_candidates.append(rows[: config.row_candidate_limit])

    beam: list[tuple[float, float, np.ndarray, tuple[tuple[int, ...], ...]]] = [
        (0.0, 0.0, np.zeros((num_nodes,), dtype=np.float64), ())
    ]
    for source_node in range(num_nodes):
        expanded: list[tuple[float, float, np.ndarray, tuple[tuple[int, ...], ...]]] = []
        for _, remote_hits, assignments, destination_rows in beam:
            for _, destinations, row_remote_hits, row_assignments in source_candidates[source_node]:
                candidate_remote_hits = remote_hits + row_remote_hits
                candidate_assignments = assignments + row_assignments
                cost = proxy_cost(np.asarray([candidate_remote_hits]), candidate_assignments)
                expanded.append(
                    (
                        cost,
                        candidate_remote_hits,
                        candidate_assignments,
                        destination_rows + (destinations,),
                    )
                )
        expanded.sort(key=lambda row: (row[0], row[3]))
        beam = expanded[: config.beam_width]
    _, _, node_assignments, selected_rows = beam[0]
    destination_nodes = np.asarray(selected_rows, dtype=np.int64)
    remote_hits_by_source = np.asarray(
        [
            _remote_group_hits(intersection_hits[source], destination_nodes[source], source)
            for source in range(num_nodes)
        ],
        dtype=np.float64,
    )

    order = sorted(
        (
            (float(assignments_by_community[source, community]), source, community)
            for source in range(num_nodes)
            for community in range(num_communities)
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    total_changes = 0
    completed_sweeps = 0
    tolerance = 1e-12
    for _ in range(config.sweep_limit):
        sweep_changes = 0
        for amount, source_node, community in order:
            current_node = int(destination_nodes[source_node, community])
            current_cost = proxy_cost(remote_hits_by_source, node_assignments)
            best = (current_cost, current_node)
            for candidate_node in nodes_by_community[community].tolist():
                if candidate_node == current_node:
                    continue
                candidate_destinations = destination_nodes[source_node].copy()
                candidate_destinations[community] = candidate_node
                candidate_remote = _remote_group_hits(
                    intersection_hits[source_node],
                    candidate_destinations,
                    source_node,
                )
                candidate_assignments = node_assignments.copy()
                candidate_assignments[current_node] -= amount
                candidate_assignments[candidate_node] += amount
                candidate_remote_hits = remote_hits_by_source.copy()
                candidate_remote_hits[source_node] = candidate_remote
                key = (proxy_cost(candidate_remote_hits, candidate_assignments), candidate_node)
                if key < best:
                    best = key
            if best[1] == current_node or best[0] >= current_cost - tolerance:
                continue
            selected_node = best[1]
            destination_nodes[source_node, community] = selected_node
            node_assignments[current_node] -= amount
            node_assignments[selected_node] += amount
            remote_hits_by_source[source_node] = _remote_group_hits(
                intersection_hits[source_node],
                destination_nodes[source_node],
                source_node,
            )
            sweep_changes += 1
        completed_sweeps += 1
        total_changes += sweep_changes
        if sweep_changes == 0:
            break

    ep_size = num_nodes * config.ranks_per_node
    mapping = np.full((ep_size, num_experts), -1, dtype=np.int64)
    for source_node in range(num_nodes):
        source_slice = slice(
            source_node * config.ranks_per_node,
            (source_node + 1) * config.ranks_per_node,
        )
        for expert in range(num_experts):
            destination_node = int(destination_nodes[source_node, logical_communities[expert]])
            instance = int(instances_by_expert_node[expert, destination_node])
            if instance < 0:
                raise RuntimeError("Community mapping references an absent expert copy.")
            mapping[source_slice, expert] = instance
    return CommunityMappingResult(
        mapping=mapping,
        destination_nodes=destination_nodes,
        sweeps=completed_sweeps,
        changes=total_changes,
        proxy_cost=proxy_cost(remote_hits_by_source, node_assignments),
    )
