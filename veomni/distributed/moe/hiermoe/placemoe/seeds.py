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

"""Deterministic feasible seeds retained by the PlaceMoE optimizer."""

from __future__ import annotations

import numpy as np

from .types import LayerPlan, PlaceMoETopology


def mirrored_r2_plan(topology: PlaceMoETopology) -> LayerPlan:
    """Return the default-order 2-copy plan for an even EP topology."""

    if topology.ep_size % 2 or topology.total_slots != 2 * topology.num_experts:
        raise ValueError("Mirrored R2 requires an even EP size and exactly two copies per expert.")
    layout = np.full((topology.total_slots,), -1, dtype=np.int64)
    owners = np.full((topology.num_experts,), -1, dtype=np.int64)
    mapping = np.full((topology.ep_size, topology.num_experts), -1, dtype=np.int64)
    half = topology.ep_size // 2
    for expert in range(topology.num_experts):
        rank_in_half, local_slot = divmod(expert, topology.slots_per_rank)
        first = rank_in_half * topology.slots_per_rank + local_slot
        second = (half + rank_in_half) * topology.slots_per_rank + local_slot
        layout[first] = expert
        layout[second] = expert
        owners[expert] = first
        mapping[:half, expert] = first
        mapping[half:, expert] = second
    plan = LayerPlan(
        slot_to_logical=layout,
        owner_slots=owners,
        source_logical_to_physical=mapping,
    )
    plan.validate(topology, additional_copies=topology.num_experts)
    return plan
