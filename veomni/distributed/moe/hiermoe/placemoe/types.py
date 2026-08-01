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

"""Data types and invariants shared by the PlaceMoE optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EMPTY_EXPERT = -1


def _readonly_int64(value: np.ndarray | list[int] | list[list[int]]) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlaceMoETopology:
    """Physical capacities used by one expert-parallel MoE layer."""

    ep_size: int
    ranks_per_node: int
    num_experts: int
    slots_per_rank: int

    def __post_init__(self) -> None:
        for name in ("ep_size", "ranks_per_node", "num_experts", "slots_per_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.ep_size % self.ranks_per_node:
            raise ValueError("ep_size must be divisible by ranks_per_node.")
        if self.total_slots < self.num_experts:
            raise ValueError("Physical capacity cannot hold one copy of every expert.")

    @property
    def num_nodes(self) -> int:
        return self.ep_size // self.ranks_per_node

    @property
    def total_slots(self) -> int:
        return self.ep_size * self.slots_per_rank

    def rank_of_slot(self, slot: int) -> int:
        if not 0 <= slot < self.total_slots:
            raise ValueError(f"Physical slot {slot} is outside the topology.")
        return slot // self.slots_per_rank


@dataclass(frozen=True)
class ProfileStatistics:
    """Source-conditioned assignment demand and expert co-selection affinity."""

    demand: np.ndarray
    affinity: np.ndarray

    def __post_init__(self) -> None:
        demand = _readonly_float64(self.demand)
        affinity = _readonly_float64(self.affinity)
        if demand.ndim != 2:
            raise ValueError("demand must have shape [source_rank, expert].")
        ep_size, num_experts = demand.shape
        if affinity.shape != (ep_size, num_experts, num_experts):
            raise ValueError("affinity must have shape [source_rank, expert, expert].")
        if bool((demand < 0).any()) or bool((affinity < 0).any()):
            raise ValueError("Profile statistics must be non-negative.")
        if not np.allclose(affinity, affinity.transpose(0, 2, 1)):
            raise ValueError("Expert affinity must be symmetric for every source rank.")
        if not np.allclose(np.diagonal(affinity, axis1=1, axis2=2), 0.0):
            raise ValueError("Expert affinity diagonal must be zero.")
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "affinity", affinity)

    @property
    def ep_size(self) -> int:
        return int(self.demand.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.demand.shape[1])

    @property
    def expert_demand(self) -> np.ndarray:
        return self.demand.sum(axis=0)

    @property
    def expert_affinity(self) -> np.ndarray:
        return self.affinity.sum(axis=0)


@dataclass(frozen=True)
class LayerPlan:
    """A physical expert layout and source-aware token-to-copy mapping."""

    slot_to_logical: np.ndarray
    source_logical_to_physical: np.ndarray
    owner_slots: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_to_logical", _readonly_int64(self.slot_to_logical))
        object.__setattr__(
            self,
            "source_logical_to_physical",
            _readonly_int64(self.source_logical_to_physical),
        )
        object.__setattr__(self, "owner_slots", _readonly_int64(self.owner_slots))

    @property
    def copy_counts(self) -> np.ndarray:
        active = self.slot_to_logical[self.slot_to_logical != EMPTY_EXPERT]
        minimum = int(self.owner_slots.shape[0])
        return np.bincount(active, minlength=minimum)

    def validate(self, topology: PlaceMoETopology, *, additional_copies: int | None = None) -> None:
        if self.slot_to_logical.shape != (topology.total_slots,):
            raise ValueError("Layout length does not match the physical slot capacity.")
        if self.source_logical_to_physical.shape != (topology.ep_size, topology.num_experts):
            raise ValueError("Mapping shape does not match [source_rank, logical_expert].")
        if self.owner_slots.shape != (topology.num_experts,):
            raise ValueError("owner_slots must contain one runtime owner per logical expert.")
        if bool((self.slot_to_logical < EMPTY_EXPERT).any() or (self.slot_to_logical >= topology.num_experts).any()):
            raise ValueError("Layout contains an invalid logical expert ID.")
        counts = self.copy_counts
        if bool((counts[: topology.num_experts] < 1).any()):
            raise ValueError("Every logical expert must retain at least one physical copy.")
        if additional_copies is not None and int(counts.sum()) - topology.num_experts != additional_copies:
            raise ValueError("Layout does not use the requested additional-copy budget.")
        mapping = self.source_logical_to_physical
        if bool((mapping < 0).any() or (mapping >= topology.total_slots).any()):
            raise ValueError("Mapping references a slot outside the topology.")
        requested = np.broadcast_to(np.arange(topology.num_experts, dtype=np.int64), mapping.shape)
        if not np.array_equal(self.slot_to_logical[mapping], requested):
            raise ValueError("Mapping references a copy of the wrong logical expert.")
        if bool((self.owner_slots < 0).any() or (self.owner_slots >= topology.total_slots).any()):
            raise ValueError("Runtime owner references a slot outside the topology.")
        if not np.array_equal(
            self.slot_to_logical[self.owner_slots],
            np.arange(topology.num_experts, dtype=np.int64),
        ):
            raise ValueError("Runtime owner slots do not match their logical experts.")

    def to_runtime_payload(self) -> dict[str, Any]:
        return {
            "slot_to_logical": self.slot_to_logical.tolist(),
            "owner_slots": self.owner_slots.tolist(),
            "source_logical_to_physical": self.source_logical_to_physical.tolist(),
        }

    @classmethod
    def from_runtime_payload(cls, payload: dict[str, Any]) -> LayerPlan:
        return cls(
            slot_to_logical=payload["slot_to_logical"],
            owner_slots=payload["owner_slots"],
            source_logical_to_physical=payload["source_logical_to_physical"],
        )
