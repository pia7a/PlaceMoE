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

"""Model-independent metadata for expert-stacked parameters."""

from __future__ import annotations


FUSED_STACKED_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
SPLIT_STACKED_EXPERT_PARAMETER_NAMES = ("gate_proj", "up_proj", "down_proj")
STACKED_EXPERT_PARAMETER_NAMES = frozenset(FUSED_STACKED_EXPERT_PARAMETER_NAMES + SPLIT_STACKED_EXPERT_PARAMETER_NAMES)


def is_stacked_expert_parameter_name(parameter_name: str) -> bool:
    """Whether a checkpoint key has an expert-indexed leading dimension.

    Supported VeOmni MoE implementations expose either fused ``gate_up_proj``
    or split ``gate_proj``/``up_proj`` tensors directly under ``experts``.
    ``.weight`` is accepted for adapters that wrap the tensor in a module.
    Per-expert ``ModuleList`` keys are intentionally excluded.
    """

    marker = ".experts."
    if marker not in parameter_name:
        return False
    leaf = parameter_name.rsplit(marker, 1)[1]
    if leaf.endswith(".weight"):
        leaf = leaf[: -len(".weight")]
    return leaf in STACKED_EXPERT_PARAMETER_NAMES


__all__ = [
    "FUSED_STACKED_EXPERT_PARAMETER_NAMES",
    "SPLIT_STACKED_EXPERT_PARAMETER_NAMES",
    "STACKED_EXPERT_PARAMETER_NAMES",
    "is_stacked_expert_parameter_name",
]
