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

"""Bounded PlaceMoE layout--mapping alternation for one replica allocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .mapping import MappingConfig, initialize_mapping, optimize_mapping, optimize_mapping_normalized
from .materialize import materialize_plan
from .placement import PlacementConfig, place_instances
from .statistics import project_statistics_to_copies, uniform_copy_statistics
from .types import LayerPlan, PlaceMoETopology, ProfileStatistics


@dataclass(frozen=True)
class OptimizerConfig:
    """Topology, calibrated coefficients, and bounded PlaceMoE search limits."""

    topology: PlaceMoETopology
    primary_slots_per_rank: int
    node_omega: float
    rank_omega: float
    gamma: float
    rounds: int = 3
    assignment_iterations: int = 12
    node_exchange_limit: int = 24
    rank_exchange_limit: int = 12
    mapping_sweep_limit: int = 6
    prefer_node_local: bool = True
    seed: int = 0
    seed_load_weights: tuple[float, ...] = (8.0, 32.0, 128.0)
    normalized_mapping_weights: tuple[float, ...] = ()
    calibrated_mapping_refinement: bool = True
    carry_mapping_across_rounds: bool = True
    calibrated_partition_refinement: bool = True

    def __post_init__(self) -> None:
        if self.primary_slots_per_rank <= 0 or self.primary_slots_per_rank > self.topology.slots_per_rank:
            raise ValueError("Primary slots per rank must be within the physical rank capacity.")
        if self.node_omega < 0 or self.rank_omega < 0 or self.gamma < 0:
            raise ValueError("Calibrated optimizer coefficients must be non-negative.")
        if self.rounds <= 0:
            raise ValueError("Layout--mapping rounds must be positive.")
        if not self.seed_load_weights or any(weight < 0 for weight in self.seed_load_weights):
            raise ValueError("At least one non-negative seed load weight is required.")
        if any(weight < 0 for weight in self.normalized_mapping_weights):
            raise ValueError("Normalized mapping weights must be non-negative.")


@dataclass(frozen=True)
class OptimizerCandidate:
    """One complete layout--mapping pair evaluated on profiled routes."""

    plan: LayerPlan
    logical_instances: np.ndarray
    instance_ranks: np.ndarray
    instance_mapping: np.ndarray
    cost: float
    round_index: int
    placement_repaired: bool
    mapping_sweeps: int
    mapping_changes: int

    def __post_init__(self) -> None:
        for name in ("logical_instances", "instance_ranks", "instance_mapping"):
            value = np.asarray(getattr(self, name), dtype=np.int64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.cost):
            raise ValueError("Candidate cost must be finite.")


@dataclass(frozen=True)
class OptimizationResult:
    """All evaluated rounds and the candidate with minimum exact cost."""

    candidates: tuple[OptimizerCandidate, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("Optimization result must contain at least one candidate.")

    @property
    def best(self) -> OptimizerCandidate:
        return min(self.candidates, key=lambda candidate: (candidate.cost, candidate.round_index))


def optimize_replica_allocation(
    statistics: ProfileStatistics,
    logical_instances: np.ndarray,
    config: OptimizerConfig,
    evaluate: Callable[[LayerPlan], float],
) -> OptimizationResult:
    """Alternate layout and mapping, using complete-route cost through ``evaluate``."""

    topology = config.topology
    if statistics.ep_size != topology.ep_size or statistics.num_experts != topology.num_experts:
        raise ValueError("Profile statistics do not match the optimizer topology.")
    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    if logical_instances.shape != (topology.total_slots,):
        raise ValueError("Replica allocation must fill total capacity with copies and empty items.")
    if bool((logical_instances < -1).any()) or bool((logical_instances >= topology.num_experts).any()):
        raise ValueError("Replica allocation contains an invalid expert ID.")
    counts = np.bincount(logical_instances[logical_instances >= 0], minlength=topology.num_experts)
    if bool((counts < 1).any()):
        raise ValueError("Replica allocation must retain every logical expert.")

    instance_demand, instance_affinity = uniform_copy_statistics(statistics, logical_instances)
    previous_mapping: np.ndarray | None = None
    candidates: list[OptimizerCandidate] = []
    for round_index in range(config.rounds):
        try:
            placement = place_instances(
                instance_demand,
                instance_affinity,
                logical_instances,
                PlacementConfig(
                    ep_size=topology.ep_size,
                    ranks_per_node=topology.ranks_per_node,
                    slots_per_rank=topology.slots_per_rank,
                    node_omega=config.node_omega,
                    rank_omega=config.rank_omega,
                    gamma=config.gamma,
                    assignment_iterations=config.assignment_iterations,
                    node_exchange_limit=config.node_exchange_limit,
                    rank_exchange_limit=config.rank_exchange_limit,
                    seed=config.seed + 1009 * round_index,
                    seed_load_weight=config.seed_load_weights[round_index % len(config.seed_load_weights)],
                    calibrated_partition_refinement=config.calibrated_partition_refinement,
                ),
            )
        except RuntimeError:
            if candidates:
                break
            raise

        fresh_mapping = initialize_mapping(
            logical_instances,
            placement.instance_ranks,
            statistics.demand,
            ranks_per_node=topology.ranks_per_node,
            prefer_node_local=config.prefer_node_local,
        )
        mapping_seeds = [fresh_mapping]
        if (
            config.carry_mapping_across_rounds
            and previous_mapping is not None
            and not np.array_equal(previous_mapping, fresh_mapping)
        ):
            # The paper update follows movable copies when L changes. Retain
            # that candidate, but also restart M for the new layout: a carried
            # mapping can be a poor coordinate-descent seed after relocation.
            mapping_seeds.insert(0, previous_mapping)

        seen_round_mappings: set[bytes] = set()
        for initial_mapping in mapping_seeds:
            initial_key = initial_mapping.tobytes()
            if initial_key in seen_round_mappings:
                continue
            seen_round_mappings.add(initial_key)
            initial_plan = materialize_plan(
                logical_instances,
                placement.instance_ranks,
                initial_mapping,
                statistics.demand,
                topology,
                primary_slots_per_rank=config.primary_slots_per_rank,
            )
            initial_candidate = OptimizerCandidate(
                plan=initial_plan,
                logical_instances=logical_instances,
                instance_ranks=placement.instance_ranks,
                instance_mapping=initial_mapping,
                cost=float(evaluate(initial_plan)),
                round_index=round_index,
                placement_repaired=placement.repaired,
                mapping_sweeps=0,
                mapping_changes=0,
            )
            candidates.append(initial_candidate)

            if config.calibrated_mapping_refinement:
                mapping = optimize_mapping(
                    logical_instances,
                    placement.instance_ranks,
                    initial_mapping,
                    statistics,
                    MappingConfig(
                        ranks_per_node=topology.ranks_per_node,
                        node_omega=config.node_omega,
                        rank_omega=config.rank_omega,
                        gamma=config.gamma,
                        sweep_limit=config.mapping_sweep_limit,
                    ),
                )
                mapping_key = mapping.mapping.tobytes()
                if mapping_key not in seen_round_mappings:
                    seen_round_mappings.add(mapping_key)
                    plan = materialize_plan(
                        logical_instances,
                        placement.instance_ranks,
                        mapping.mapping,
                        statistics.demand,
                        topology,
                        primary_slots_per_rank=config.primary_slots_per_rank,
                    )
                    refined_candidate = OptimizerCandidate(
                        plan=plan,
                        logical_instances=logical_instances,
                        instance_ranks=placement.instance_ranks,
                        instance_mapping=mapping.mapping,
                        cost=float(evaluate(plan)),
                        round_index=round_index,
                        placement_repaired=placement.repaired,
                        mapping_sweeps=mapping.sweeps,
                        mapping_changes=mapping.changes,
                    )
                    candidates.append(refined_candidate)

            if np.array_equal(initial_mapping, fresh_mapping):
                for assignment_weight in config.normalized_mapping_weights:
                    normalized = optimize_mapping_normalized(
                        logical_instances,
                        placement.instance_ranks,
                        initial_mapping,
                        statistics,
                        ranks_per_node=topology.ranks_per_node,
                        assignment_weight=assignment_weight,
                        sweep_limit=config.mapping_sweep_limit,
                    )
                    normalized_key = normalized.mapping.tobytes()
                    if normalized_key in seen_round_mappings:
                        continue
                    seen_round_mappings.add(normalized_key)
                    normalized_plan = materialize_plan(
                        logical_instances,
                        placement.instance_ranks,
                        normalized.mapping,
                        statistics.demand,
                        topology,
                        primary_slots_per_rank=config.primary_slots_per_rank,
                    )
                    normalized_candidate = OptimizerCandidate(
                        plan=normalized_plan,
                        logical_instances=logical_instances,
                        instance_ranks=placement.instance_ranks,
                        instance_mapping=normalized.mapping,
                        cost=float(evaluate(normalized_plan)),
                        round_index=round_index,
                        placement_repaired=placement.repaired,
                        mapping_sweeps=normalized.sweeps,
                        mapping_changes=normalized.changes,
                    )
                    candidates.append(normalized_candidate)

        # Keep the exact-cost incumbent monotonic across rounds. A newly
        # generated layout may be worse than an earlier round; projecting its
        # mapping would discard the state that produced the best candidate.
        previous_mapping = min(candidates, key=lambda candidate: candidate.cost).instance_mapping.copy()
        instance_demand, instance_affinity = project_statistics_to_copies(
            statistics,
            logical_instances,
            previous_mapping,
        )
    return OptimizationResult(candidates=tuple(candidates))
