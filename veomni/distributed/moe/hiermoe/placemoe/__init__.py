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
from .statistics import profile_route_statistics, project_statistics_to_copies, uniform_copy_statistics
from .types import EMPTY_EXPERT, LayerPlan, PlaceMoETopology, ProfileStatistics


__all__ = [
    "EMPTY_EXPERT",
    "LayerPlan",
    "PlaceMoETopology",
    "ProfileStatistics",
    "bounded_group_shortlist",
    "build_replica_allocations",
    "profile_route_statistics",
    "project_statistics_to_copies",
    "uniform_copy_statistics",
]
