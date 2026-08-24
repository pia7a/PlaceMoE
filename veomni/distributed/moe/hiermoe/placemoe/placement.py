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

"""Hierarchical physical-copy placement for PlaceMoE."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .partition import PartitionConfig, map_groups_to_locations, partition_items


@dataclass(frozen=True)
class PlacementConfig:
    """Topology and calibrated coefficients for one hierarchical placement."""

    ep_size: int
    ranks_per_node: int
    slots_per_rank: int
    node_omega: float
    rank_omega: float
    gamma: float
    assignment_iterations: int = 12
    node_exchange_limit: int = 24
    rank_exchange_limit: int = 12
    seed: int = 0
    seed_load_weight: float | None = None
    calibrated_partition_refinement: bool = True
    hierarchy_group_sizes: tuple[int, ...] = ()
    level_omegas: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or self.ranks_per_node <= 0 or self.slots_per_rank <= 0:
            raise ValueError("Placement topology dimensions must be positive.")
        if self.ep_size % self.ranks_per_node:
            raise ValueError("ep_size must be divisible by ranks_per_node.")
        if self.node_omega < 0 or self.rank_omega < 0 or self.gamma < 0:
            raise ValueError("Calibrated placement coefficients must be non-negative.")
        if self.assignment_iterations <= 0:
            raise ValueError("Capacity-assignment iterations must be positive.")
        if self.node_exchange_limit < 0 or self.rank_exchange_limit < 0:
            raise ValueError("Placement exchange limits must be non-negative.")
        _placement_hierarchy(self)

    @property
    def num_nodes(self) -> int:
        return self.ep_size // self.ranks_per_node

    @property
    def total_slots(self) -> int:
        return self.ep_size * self.slots_per_rank


def _placement_hierarchy(config: PlacementConfig) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return coarse-to-fine physical group sizes and their paper weights."""

    if not config.hierarchy_group_sizes:
        return (config.ranks_per_node, 1), (config.node_omega, config.rank_omega)
    runtime_sizes = tuple(int(size) for size in config.hierarchy_group_sizes)
    if runtime_sizes[-1] != config.ep_size:
        raise ValueError("The final hierarchy group size must equal ep_size.")
    if any(lhs <= 0 or rhs % lhs for lhs, rhs in zip(runtime_sizes, runtime_sizes[1:], strict=False)):
        raise ValueError("Hierarchy group sizes must form an increasing divisibility chain.")
    level_sizes = (*reversed(runtime_sizes[:-1]), 1)
    if level_sizes[0] != config.ranks_per_node:
        raise ValueError("The coarsest proper hierarchy group must match ranks_per_node.")
    omegas = tuple(float(value) for value in config.level_omegas)
    if len(omegas) != len(level_sizes):
        raise ValueError("One calibrated placement coefficient is required per hierarchy stage.")
    if any(value < 0.0 for value in omegas):
        raise ValueError("Hierarchy placement coefficients must be non-negative.")
    return level_sizes, omegas


@dataclass(frozen=True)
class PlacementResult:
    """Physical rank of every copy and whether rank feasibility was repaired."""

    instance_ranks: np.ndarray
    repaired: bool
    node_objective: float
    rank_objective: float
    level_objectives: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        instance_ranks = np.asarray(self.instance_ranks, dtype=np.int64).copy()
        instance_ranks.setflags(write=False)
        object.__setattr__(self, "instance_ranks", instance_ranks)


def community_node_placements(
    logical_instances: np.ndarray,
    logical_communities: np.ndarray,
    demand_by_source: np.ndarray,
    config: PlacementConfig,
    *,
    candidate_limit: int,
) -> tuple[np.ndarray, ...]:
    """Generate balanced community-coherent node placements.

    Every copy round maps the affinity communities to nodes through a
    different balanced permutation. This gives all experts in a community a
    shared node footprint without assuming a particular node count. Empty
    instances fill the residual physical capacity. Allocations that split a
    community fall back to the generic instance partitioner rather than
    weakening copy coherence inside this proposal family.
    """

    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    logical_communities = np.asarray(logical_communities, dtype=np.int64)
    demand_by_source = np.asarray(demand_by_source, dtype=np.float64)
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    if logical_instances.shape != (config.total_slots,):
        raise ValueError("Logical instances must fill the physical capacity.")
    if logical_communities.ndim != 1 or bool((logical_communities < 0).any()):
        raise ValueError("Logical communities must contain one label per expert.")
    num_experts = len(logical_communities)
    if demand_by_source.shape != (config.ep_size, num_experts):
        raise ValueError("Source demand must have shape [source_rank, expert].")
    if bool((logical_instances < -1).any()) or bool((logical_instances >= num_experts).any()):
        raise ValueError("Logical instances contain an invalid expert ID.")
    if not np.array_equal(np.unique(logical_communities), np.arange(config.num_nodes)):
        return ()
    community_sizes = np.bincount(logical_communities, minlength=config.num_nodes)
    if not bool((community_sizes == community_sizes[0]).all()):
        return ()
    copy_counts = np.bincount(logical_instances[logical_instances >= 0], minlength=num_experts)
    if bool((copy_counts < 1).any()):
        return ()
    community_copy_counts = np.zeros((config.num_nodes,), dtype=np.int64)
    for community in range(config.num_nodes):
        members = np.flatnonzero(logical_communities == community)
        member_copy_counts = copy_counts[members]
        if not bool((member_copy_counts == member_copy_counts[0]).all()):
            return ()
        community_copy_counts[community] = int(member_copy_counts[0])
    max_copies = int(community_copy_counts.max(initial=0))
    if max_copies > config.num_nodes:
        return ()
    node_capacity = config.ranks_per_node * config.slots_per_rank

    instances_by_expert = [np.flatnonzero(logical_instances == expert).tolist() for expert in range(num_experts)]
    empty_instances = np.flatnonzero(logical_instances < 0).tolist()
    if len(empty_instances) != config.total_slots - int(copy_counts.sum()):
        return ()
    source_nodes = demand_by_source.reshape(
        config.num_nodes,
        config.ranks_per_node,
        num_experts,
    ).sum(axis=1)

    identity = np.arange(config.num_nodes, dtype=np.int64)
    copy_node_tables: list[np.ndarray] = []
    if max_copies == 1:
        copy_node_tables.append(identity[None, :])
    elif max_copies == 2:
        if config.num_nodes <= 6:
            permutations = (np.asarray(row, dtype=np.int64) for row in itertools.permutations(range(config.num_nodes)))
        else:
            rows = [np.roll(identity, -shift) for shift in range(1, config.num_nodes)]
            rng = np.random.default_rng(config.seed)
            pool_limit = max(64, 16 * candidate_limit)
            attempts = 0
            while len(rows) < pool_limit and attempts < 32 * pool_limit:
                attempts += 1
                row = rng.permutation(config.num_nodes)
                if bool((row == identity).any()):
                    continue
                rows.append(row)
            permutations = iter(rows)
        for permutation in permutations:
            if bool((permutation == identity).any()):
                continue
            copy_node_tables.append(np.stack((identity, permutation)))
    else:
        for additional in itertools.combinations(
            range(1, config.num_nodes),
            max_copies - 1,
        ):
            shifts = (0, *additional)
            copy_node_tables.append(np.stack([np.roll(identity, -shift) for shift in shifts]))
    rows: list[tuple[float, bytes, np.ndarray]] = []
    seen_abstract: set[bytes] = set()
    for copy_nodes in copy_node_tables:
        abstract_nodes = np.full((config.total_slots,), -1, dtype=np.int64)
        for expert in range(num_experts):
            community = int(logical_communities[expert])
            for copy_index, instance in enumerate(instances_by_expert[expert]):
                abstract_nodes[instance] = int(copy_nodes[copy_index, community])
        active_counts = np.bincount(
            abstract_nodes[abstract_nodes >= 0],
            minlength=config.num_nodes,
        )
        if bool((active_counts > node_capacity).any()):
            continue
        empty_per_node = node_capacity - active_counts
        if int(empty_per_node.sum()) != len(empty_instances):
            raise RuntimeError("Community placement has inconsistent empty capacity.")
        empty_offset = 0
        for node in range(config.num_nodes):
            for _ in range(int(empty_per_node[node])):
                abstract_nodes[empty_instances[empty_offset]] = node
                empty_offset += 1
        if not np.array_equal(
            np.bincount(abstract_nodes, minlength=config.num_nodes),
            np.full((config.num_nodes,), node_capacity),
        ):
            raise RuntimeError("Community placement violates abstract node capacity.")
        abstract_key = abstract_nodes.tobytes()
        if abstract_key in seen_abstract:
            continue
        seen_abstract.add(abstract_key)

        locality = np.zeros((config.num_nodes, config.num_nodes), dtype=np.float64)
        for abstract_node in range(config.num_nodes):
            active = logical_instances[abstract_nodes == abstract_node]
            experts = np.unique(active[active >= 0])
            locality[abstract_node] = source_nodes[:, experts].sum(axis=1)
        abstract_rows, physical_nodes = linear_sum_assignment(-locality)
        abstract_to_physical = np.full((config.num_nodes,), -1, dtype=np.int64)
        abstract_to_physical[abstract_rows] = physical_nodes
        instance_nodes = abstract_to_physical[abstract_nodes]
        locality_score = float(locality[abstract_rows, physical_nodes].sum())
        rows.append((-locality_score, instance_nodes.tobytes(), instance_nodes))

    rows.sort(key=lambda row: (row[0], row[1]))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _, key, instance_nodes in rows:
        if key in seen:
            continue
        seen.add(key)
        unique.append(instance_nodes.copy())
        if len(unique) == candidate_limit:
            break
    return tuple(unique)


def rank_placement_is_unique(instance_ranks: np.ndarray, logical_instances: np.ndarray, *, ep_size: int) -> bool:
    """Return whether a rank contains at most one copy of each expert."""

    instance_ranks = np.asarray(instance_ranks, dtype=np.int64)
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    if instance_ranks.shape != logical_instances.shape:
        raise ValueError("Rank and logical-instance vectors must have matching shapes.")
    for rank in range(ep_size):
        active = logical_instances[(instance_ranks == rank) & (logical_instances >= 0)]
        if len(active) != len(np.unique(active)):
            return False
    return True


def repair_rank_placement(
    instance_nodes: np.ndarray,
    demand: np.ndarray,
    affinity: np.ndarray,
    logical_instances: np.ndarray,
    config: PlacementConfig,
) -> np.ndarray:
    """Demand-ordered rank repair with fixed node membership and capacity."""

    instance_nodes = np.asarray(instance_nodes, dtype=np.int64)
    demand = np.asarray(demand, dtype=np.float64)
    affinity = np.asarray(affinity, dtype=np.float64)
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    if not (instance_nodes.shape == demand.shape == logical_instances.shape):
        raise ValueError("Node, demand, and logical-instance vectors must have matching shapes.")
    if affinity.shape != (len(instance_nodes), len(instance_nodes)):
        raise ValueError("Copy affinity must match the number of physical instances.")

    result = np.full_like(instance_nodes, -1)
    node_capacity = config.ranks_per_node * config.slots_per_rank
    for node in range(config.num_nodes):
        members = np.flatnonzero(instance_nodes == node)
        if len(members) != node_capacity:
            raise ValueError("Node membership does not match its physical capacity.")
        capacities = np.full((config.ranks_per_node,), config.slots_per_rank, dtype=np.int64)
        loads = np.zeros((config.ranks_per_node,), dtype=np.float64)
        lane_members: list[list[int]] = [[] for _ in range(config.ranks_per_node)]
        active = logical_instances[members]
        multiplicity = np.bincount(active[active >= 0], minlength=int(active.max(initial=-1)) + 1)
        order = sorted(
            members.tolist(),
            key=lambda item: (
                -int(multiplicity[int(logical_instances[item])]) if int(logical_instances[item]) >= 0 else 0,
                -float(demand[item]),
                item,
            ),
        )
        for instance in order:
            logical = int(logical_instances[instance])
            best: tuple[float, float, int] | None = None
            for lane in range(config.ranks_per_node):
                if capacities[lane] <= 0:
                    continue
                if logical >= 0 and any(int(logical_instances[other]) == logical for other in lane_members[lane]):
                    continue
                projected_peak = max(float(loads.max(initial=0.0)), loads[lane] + float(demand[instance]))
                affinity_gain = sum(float(affinity[instance, other]) for other in lane_members[lane])
                key = (projected_peak, -affinity_gain, lane)
                if best is None or key < best:
                    best = key
            if best is None:
                raise RuntimeError("Rank repair could not satisfy copy uniqueness and capacity.")
            lane = best[2]
            result[instance] = node * config.ranks_per_node + lane
            capacities[lane] -= 1
            loads[lane] += float(demand[instance])
            lane_members[lane].append(instance)
    return result


def _place_instances_multilevel(
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    logical_instances: np.ndarray,
    config: PlacementConfig,
) -> PlacementResult:
    """Apply the paper partition-and-map update recursively at every link level."""

    demand = demand_by_source.sum(axis=0)
    affinity = affinity_by_source.sum(axis=0)
    level_sizes, level_omegas = _placement_hierarchy(config)
    parents: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.arange(config.total_slots, dtype=np.int64),
            np.arange(config.ep_size, dtype=np.int64),
        )
    ]
    instance_ranks = np.full((config.total_slots,), -1, dtype=np.int64)
    objectives: list[float] = []
    for level_index, (group_size, omega) in enumerate(zip(level_sizes, level_omegas, strict=True)):
        next_parents: list[tuple[np.ndarray, np.ndarray]] = []
        level_objective = 0.0
        for parent_index, (members, parent_ranks) in enumerate(parents):
            if len(parent_ranks) % group_size:
                raise RuntimeError("A hierarchy level does not divide its parent group.")
            child_count = len(parent_ranks) // group_size
            child_capacity = group_size * config.slots_per_rank
            partition = partition_items(
                affinity[np.ix_(members, members)],
                demand[members],
                PartitionConfig(
                    capacities=(child_capacity,) * child_count,
                    ranks_per_group=(group_size,) * child_count,
                    omega=omega,
                    gamma=config.gamma,
                    restarts=1,
                    assignment_iterations=config.assignment_iterations,
                    exchange_limit=(config.node_exchange_limit if level_index == 0 else config.rank_exchange_limit),
                    seed=(config.seed + 104729 * (level_index + 1) + 1009 * (parent_index + 1)),
                    seed_load_weight=config.seed_load_weight,
                    calibrated_refinement=config.calibrated_partition_refinement,
                ),
            )[0]
            child_sources = tuple(
                parent_ranks[offset : offset + group_size] for offset in range(0, len(parent_ranks), group_size)
            )
            locations = map_groups_to_locations(
                partition.labels,
                demand_by_source[:, members],
                sources_by_location=child_sources,
            )
            level_objective += float(partition.objective)
            if group_size == 1:
                instance_ranks[members] = parent_ranks[locations]
            else:
                for child in range(child_count):
                    next_parents.append(
                        (
                            members[locations == child],
                            child_sources[child],
                        )
                    )
        objectives.append(level_objective)
        parents = next_parents

    repaired = False
    if not rank_placement_is_unique(instance_ranks, logical_instances, ep_size=config.ep_size):
        instance_nodes = instance_ranks // config.ranks_per_node
        instance_ranks = repair_rank_placement(
            instance_nodes,
            demand,
            affinity,
            logical_instances,
            config,
        )
        repaired = True
    counts = np.bincount(instance_ranks, minlength=config.ep_size)
    if not np.array_equal(counts, np.full((config.ep_size,), config.slots_per_rank)):
        raise RuntimeError("Hierarchical placement violates rank capacity.")
    if not rank_placement_is_unique(instance_ranks, logical_instances, ep_size=config.ep_size):
        raise RuntimeError("Hierarchical placement retains duplicate expert copies on one rank.")
    return PlacementResult(
        instance_ranks=instance_ranks,
        repaired=repaired,
        node_objective=objectives[0],
        rank_objective=sum(objectives[1:]),
        level_objectives=tuple(objectives),
    )


def place_instances(
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    logical_instances: np.ndarray,
    config: PlacementConfig,
) -> PlacementResult:
    """Place physical copies across nodes and then ranks using the paper objective."""

    demand_by_source = np.asarray(demand_by_source, dtype=np.float64)
    affinity_by_source = np.asarray(affinity_by_source, dtype=np.float64)
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    if demand_by_source.shape != (config.ep_size, config.total_slots):
        raise ValueError("Copy demand must have shape [source_rank, physical_instance].")
    if affinity_by_source.shape != (config.ep_size, config.total_slots, config.total_slots):
        raise ValueError("Copy affinity must have shape [source_rank, instance, instance].")
    if logical_instances.shape != (config.total_slots,):
        raise ValueError("Logical instances must fill the physical slot capacity, including empty items.")
    if len(_placement_hierarchy(config)[0]) > 2:
        return _place_instances_multilevel(demand_by_source, affinity_by_source, logical_instances, config)
    demand = demand_by_source.sum(axis=0)
    affinity = affinity_by_source.sum(axis=0)
    node_capacity = config.ranks_per_node * config.slots_per_rank
    node_partition = partition_items(
        affinity,
        demand,
        PartitionConfig(
            capacities=(node_capacity,) * config.num_nodes,
            ranks_per_group=(config.ranks_per_node,) * config.num_nodes,
            omega=config.node_omega,
            gamma=config.gamma,
            restarts=1,
            assignment_iterations=config.assignment_iterations,
            exchange_limit=config.node_exchange_limit,
            seed=config.seed,
            seed_load_weight=config.seed_load_weight,
            calibrated_refinement=config.calibrated_partition_refinement,
        ),
    )[0]
    node_sources = tuple(
        np.arange(node * config.ranks_per_node, (node + 1) * config.ranks_per_node, dtype=np.int64)
        for node in range(config.num_nodes)
    )
    instance_nodes = map_groups_to_locations(
        node_partition.labels,
        demand_by_source,
        sources_by_location=node_sources,
    )

    instance_ranks = np.full((config.total_slots,), -1, dtype=np.int64)
    rank_objective = 0.0
    for node in range(config.num_nodes):
        members = np.flatnonzero(instance_nodes == node)
        rank_partition = partition_items(
            affinity[np.ix_(members, members)],
            demand[members],
            PartitionConfig(
                capacities=(config.slots_per_rank,) * config.ranks_per_node,
                ranks_per_group=(1,) * config.ranks_per_node,
                omega=config.rank_omega,
                gamma=config.gamma,
                restarts=1,
                assignment_iterations=config.assignment_iterations,
                exchange_limit=config.rank_exchange_limit,
                seed=config.seed + 104729 * (node + 1),
                seed_load_weight=config.seed_load_weight,
                calibrated_refinement=config.calibrated_partition_refinement,
            ),
        )[0]
        rank_sources = tuple(
            np.asarray([node * config.ranks_per_node + lane], dtype=np.int64) for lane in range(config.ranks_per_node)
        )
        lanes = map_groups_to_locations(
            rank_partition.labels,
            demand_by_source[:, members],
            sources_by_location=rank_sources,
        )
        instance_ranks[members] = node * config.ranks_per_node + lanes
        rank_objective += rank_partition.objective

    repaired = False
    if not rank_placement_is_unique(instance_ranks, logical_instances, ep_size=config.ep_size):
        instance_ranks = repair_rank_placement(instance_nodes, demand, affinity, logical_instances, config)
        repaired = True
    counts = np.bincount(instance_ranks, minlength=config.ep_size)
    if not np.array_equal(counts, np.full((config.ep_size,), config.slots_per_rank)):
        raise RuntimeError("Hierarchical placement violates rank capacity.")
    if not rank_placement_is_unique(instance_ranks, logical_instances, ep_size=config.ep_size):
        raise RuntimeError("Hierarchical placement retains duplicate expert copies on one rank.")
    return PlacementResult(
        instance_ranks=instance_ranks,
        repaired=repaired,
        node_objective=node_partition.objective,
        rank_objective=rank_objective,
        level_objectives=(node_partition.objective, rank_objective),
    )
