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

"""Stable PlaceMoE layout artifact construction and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import LayerPlan, PlaceMoETopology


PLACEMOE_ARTIFACT_SCHEMA_VERSION = 2


def build_placemoe_artifact(
    plans: Mapping[str, LayerPlan],
    topology: PlaceMoETopology,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the preloaded runtime artifact used by static and hot updates."""

    if not plans:
        raise ValueError("A PlaceMoE artifact must contain at least one layer plan.")
    layers: dict[str, Any] = {}
    for name, plan in plans.items():
        if not name:
            raise ValueError("Layer names must be non-empty.")
        plan.validate(topology)
        layers[name] = plan.to_runtime_payload()
    payload: dict[str, Any] = {
        "schema_version": PLACEMOE_ARTIFACT_SCHEMA_VERSION,
        "source": {
            "algorithm": "placemoe-v1",
            "initial_layout": "preloaded",
            "requires_static_preload": True,
            **dict(source or {}),
        },
        "topology": {
            "ep_size": topology.ep_size,
            "ranks_per_node": topology.ranks_per_node,
            "num_experts": topology.num_experts,
            "num_physical_slots": topology.total_slots,
            "slots_per_rank": topology.slots_per_rank,
        },
        "replay": {"actions_by_step": {"1": []}},
        "layers": layers,
    }
    validate_placemoe_artifact(payload)
    return payload


def validate_placemoe_artifact(payload: Mapping[str, Any]) -> dict[str, LayerPlan]:
    """Validate a PlaceMoE runtime artifact and return its layer plans."""

    if int(payload.get("schema_version", -1)) != PLACEMOE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported PlaceMoE artifact schema version.")
    topology_row = payload.get("topology")
    if not isinstance(topology_row, Mapping):
        raise ValueError("PlaceMoE artifact is missing its topology.")
    topology = PlaceMoETopology(
        ep_size=int(topology_row["ep_size"]),
        ranks_per_node=int(topology_row.get("ranks_per_node", topology_row["ep_size"])),
        num_experts=int(topology_row["num_experts"]),
        slots_per_rank=int(topology_row["slots_per_rank"]),
    )
    if int(topology_row.get("num_physical_slots", -1)) != topology.total_slots:
        raise ValueError("Artifact physical-slot count does not match its topology.")
    layer_rows = payload.get("layers")
    if not isinstance(layer_rows, Mapping) or not layer_rows:
        raise ValueError("PlaceMoE artifact must contain a non-empty layer table.")
    plans: dict[str, LayerPlan] = {}
    for name, row in layer_rows.items():
        if not isinstance(name, str) or not isinstance(row, dict):
            raise ValueError("PlaceMoE layer entries must map names to plan objects.")
        plan = LayerPlan.from_runtime_payload(row)
        plan.validate(topology)
        plans[name] = plan
    return plans
