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

"""Public PlaceMoE integration package."""

from .model_adapter import (
    FUSED_STACKED_EXPERT_PARAMETER_NAMES,
    SPLIT_STACKED_EXPERT_PARAMETER_NAMES,
    STACKED_EXPERT_PARAMETER_NAMES,
    ExpertParameter,
    MoEKernelWeights,
    MoEModelAdapter,
    is_stacked_expert_parameter_name,
    register_moe_model_adapter,
    require_moe_model_adapter,
    resolve_moe_model_adapter,
)


def __getattr__(name: str):
    if name == "PlaceMoERuntimeBridge":
        from .bridge import PlaceMoERuntimeBridge

        return PlaceMoERuntimeBridge
    raise AttributeError(name)


__all__ = [
    "ExpertParameter",
    "FUSED_STACKED_EXPERT_PARAMETER_NAMES",
    "MoEKernelWeights",
    "MoEModelAdapter",
    "PlaceMoERuntimeBridge",
    "SPLIT_STACKED_EXPERT_PARAMETER_NAMES",
    "STACKED_EXPERT_PARAMETER_NAMES",
    "is_stacked_expert_parameter_name",
    "register_moe_model_adapter",
    "require_moe_model_adapter",
    "resolve_moe_model_adapter",
]
