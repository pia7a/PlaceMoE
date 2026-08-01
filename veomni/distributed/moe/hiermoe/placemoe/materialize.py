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

"""Materialize PlaceMoE physical instances into runtime slots."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mapping import validate_instance_mapping
from .types import EMPTY_EXPERT, LayerPlan, PlaceMoETopology


def materialize_plan(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    instance_mapping: np.ndarray,
    demand_by_source: np.ndarray,
    topology: PlaceMoETopology,
    *,
    primary_slots_per_rank: int,
) -> LayerPlan:
    """Assign instances to slots and rewrite the mapping to follow each copy."""

    logical_instances = np.asarray(logical_instances, dtype=np.int64)
    instance_ranks = np.asarray(instance_ranks, dtype=np.int64)
    if logical_instances.shape != (topology.total_slots,) or instance_ranks.shape != logical_instances.shape:
        raise ValueError("Physical instances and ranks must match the total slot capacity.")
    demand_by_source = np.asarray(demand_by_source, dtype=np.float64)
    if demand_by_source.shape != (topology.ep_size, topology.num_experts):
        raise ValueError("Source demand does not match the target topology.")
    if primary_slots_per_rank <= 0 or primary_slots_per_rank > topology.slots_per_rank:
        raise ValueError("Primary slot capacity must be within the per-rank slot capacity.")
    if bool((instance_ranks < 0).any()) or bool((instance_ranks >= topology.ep_size).any()):
        raise ValueError("Instance placement references an invalid physical rank.")
    expected_capacity = np.full((topology.ep_size,), topology.slots_per_rank)
    if not np.array_equal(np.bincount(instance_ranks, minlength=topology.ep_size), expected_capacity):
        raise ValueError("Instance placement does not fill every rank capacity.")
    instance_mapping = validate_instance_mapping(
        instance_mapping,
        logical_instances,
        ep_size=topology.ep_size,
        num_experts=topology.num_experts,
    )

    served = np.zeros((len(logical_instances),), dtype=np.float64)
    for source in range(topology.ep_size):
        np.add.at(served, instance_mapping[source], demand_by_source[source])
    owner_instances = _select_runtime_owners(
        logical_instances,
        instance_ranks,
        served,
        topology,
        primary_slots_per_rank=primary_slots_per_rank,
    )

    slot_to_logical = np.full((topology.total_slots,), EMPTY_EXPERT, dtype=np.int64)
    owner_slots = np.full((topology.num_experts,), -1, dtype=np.int64)
    instance_to_slot = np.full((topology.total_slots,), -1, dtype=np.int64)
    owner_set = set(owner_instances.tolist())
    for rank in range(topology.ep_size):
        owners = sorted(
            (instance for instance in np.flatnonzero(instance_ranks == rank).tolist() if instance in owner_set),
            key=lambda instance: int(logical_instances[instance]),
        )
        remaining = sorted(
            (instance for instance in np.flatnonzero(instance_ranks == rank).tolist() if instance not in owner_set),
            key=lambda instance: (int(logical_instances[instance]), instance),
        )
        for local_slot, instance in enumerate(owners + remaining):
            slot = rank * topology.slots_per_rank + local_slot
            expert = int(logical_instances[instance])
            instance_to_slot[instance] = slot
            if expert == EMPTY_EXPERT:
                continue
            slot_to_logical[slot] = expert
            if instance in owner_set:
                owner_slots[expert] = slot
    if bool((instance_to_slot < 0).any()) or bool((owner_slots < 0).any()):
        raise RuntimeError("Plan materialization lost a physical instance or runtime owner.")

    plan = LayerPlan(
        slot_to_logical=slot_to_logical,
        owner_slots=owner_slots,
        source_logical_to_physical=instance_to_slot[instance_mapping],
    )
    plan.validate(
        topology,
        additional_copies=int((logical_instances != EMPTY_EXPERT).sum()) - topology.num_experts,
    )
    return plan


def _select_runtime_owners(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    served: np.ndarray,
    topology: PlaceMoETopology,
    *,
    primary_slots_per_rank: int,
) -> np.ndarray:
    """Prefer balanced owners but do not change the optimized placement."""

    owner_columns = np.repeat(np.arange(topology.ep_size, dtype=np.int64), primary_slots_per_rank)
    owner_cost = np.full((topology.num_experts, len(owner_columns)), 1e30, dtype=np.float64)
    for expert in range(topology.num_experts):
        instances = np.flatnonzero(logical_instances == expert)
        for column, rank in enumerate(owner_columns.tolist()):
            eligible = instances[instance_ranks[instances] == rank]
            if len(eligible):
                owner_cost[expert, column] = -float(served[eligible].max())
    experts, columns = linear_sum_assignment(owner_cost)
    result = np.full((topology.num_experts,), -1, dtype=np.int64)
    if len(experts) == topology.num_experts and not bool((owner_cost[experts, columns] >= 1e29).any()):
        owner_ranks = np.full((topology.num_experts,), -1, dtype=np.int64)
        owner_ranks[experts] = owner_columns[columns]
        for expert in range(topology.num_experts):
            instances = np.flatnonzero((logical_instances == expert) & (instance_ranks == owner_ranks[expert]))
            result[expert] = int(instances[np.argmax(served[instances])])
        return result

    for expert in range(topology.num_experts):
        instances = np.flatnonzero(logical_instances == expert)
        if not len(instances):
            raise ValueError(f"Logical expert {expert} has no physical copy.")
        result[expert] = int(instances[np.argmax(served[instances])])
    return result
