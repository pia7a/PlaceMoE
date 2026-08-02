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

from dataclasses import dataclass

import numpy as np

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

    @property
    def num_nodes(self) -> int:
        return self.ep_size // self.ranks_per_node

    @property
    def total_slots(self) -> int:
        return self.ep_size * self.slots_per_rank


@dataclass(frozen=True)
class PlacementResult:
    """Physical rank of every copy and whether rank feasibility was repaired."""

    instance_ranks: np.ndarray
    repaired: bool
    node_objective: float
    rank_objective: float

    def __post_init__(self) -> None:
        instance_ranks = np.asarray(self.instance_ranks, dtype=np.int64).copy()
        instance_ranks.setflags(write=False)
        object.__setattr__(self, "instance_ranks", instance_ranks)


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
    )
