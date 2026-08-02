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

"""Canonical profile-guided PlaceMoE optimizer APIs."""

from .allocation import bounded_group_shortlist, build_replica_allocations
from .artifacts import PLACEMOE_ARTIFACT_SCHEMA_VERSION, build_placemoe_artifact, validate_placemoe_artifact
from .mapping import (
    MappingConfig,
    MappingResult,
    initialize_mapping,
    mapping_rank_loads,
    optimize_mapping,
    optimize_mapping_normalized,
    validate_instance_mapping,
)
from .materialize import materialize_plan
from .optimizer import OptimizationResult, OptimizerCandidate, OptimizerConfig, optimize_replica_allocation
from .partition import PartitionConfig, PartitionResult, map_groups_to_locations, partition_items, partition_objective
from .placement import (
    PlacementConfig,
    PlacementResult,
    place_instances,
    rank_placement_is_unique,
    repair_rank_placement,
)
from .seeds import mirrored_r2_plan
from .statistics import profile_route_statistics, project_statistics_to_copies, uniform_copy_statistics
from .types import EMPTY_EXPERT, LayerPlan, PlaceMoETopology, ProfileStatistics


__all__ = [
    "EMPTY_EXPERT",
    "LayerPlan",
    "MappingConfig",
    "MappingResult",
    "OptimizationResult",
    "OptimizerCandidate",
    "OptimizerConfig",
    "PartitionConfig",
    "PartitionResult",
    "PlacementConfig",
    "PlacementResult",
    "PlaceMoETopology",
    "PLACEMOE_ARTIFACT_SCHEMA_VERSION",
    "ProfileStatistics",
    "bounded_group_shortlist",
    "build_replica_allocations",
    "build_placemoe_artifact",
    "initialize_mapping",
    "map_groups_to_locations",
    "materialize_plan",
    "mirrored_r2_plan",
    "mapping_rank_loads",
    "optimize_mapping",
    "optimize_mapping_normalized",
    "optimize_replica_allocation",
    "partition_items",
    "partition_objective",
    "place_instances",
    "profile_route_statistics",
    "project_statistics_to_copies",
    "rank_placement_is_unique",
    "repair_rank_placement",
    "uniform_copy_statistics",
    "validate_instance_mapping",
    "validate_placemoe_artifact",
]
