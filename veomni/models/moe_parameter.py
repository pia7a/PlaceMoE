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

"""Compatibility exports for PlaceMoE expert-parameter metadata."""

from __future__ import annotations

from placemoe.model_adapter import (
    FUSED_STACKED_EXPERT_PARAMETER_NAMES,
    SPLIT_STACKED_EXPERT_PARAMETER_NAMES,
    STACKED_EXPERT_PARAMETER_NAMES,
    is_stacked_expert_parameter_name,
)


__all__ = [
    "FUSED_STACKED_EXPERT_PARAMETER_NAMES",
    "SPLIT_STACKED_EXPERT_PARAMETER_NAMES",
    "STACKED_EXPERT_PARAMETER_NAMES",
    "is_stacked_expert_parameter_name",
]
