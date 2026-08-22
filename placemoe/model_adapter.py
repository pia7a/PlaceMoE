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

"""Public, model-independent expert boundary for PlaceMoE.

Models using VeOmni's stacked fused or split expert projections are detected
structurally. A model with another expert representation can register one
small adapter without changing the planner, runtime, or training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


FUSED_STACKED_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
SPLIT_STACKED_EXPERT_PARAMETER_NAMES = ("gate_proj", "up_proj", "down_proj")
STACKED_EXPERT_PARAMETER_NAMES = frozenset(FUSED_STACKED_EXPERT_PARAMETER_NAMES + SPLIT_STACKED_EXPERT_PARAMETER_NAMES)


@dataclass(frozen=True)
class ExpertParameter:
    """One parameter whose leading dimension indexes local expert slots."""

    name: str
    parameter: nn.Parameter


@dataclass(frozen=True)
class MoEKernelWeights:
    """Model-independent weight view accepted by VeOmni fused MoE kernels."""

    fc1_1_weight: torch.Tensor | None
    fc1_2_weight: torch.Tensor | None
    fc2_weight: torch.Tensor
    fc1_1_2_weight: torch.Tensor | None


@runtime_checkable
class MoEModelAdapter(Protocol):
    """Describe how a model exposes expert parameters and kernel weights."""

    name: str

    def matches(self, module: nn.Module) -> bool: ...

    def num_experts(self, module: nn.Module) -> int: ...

    def expert_parameters(self, module: nn.Module) -> tuple[ExpertParameter, ...]: ...

    def replace_expert_parameter(self, module: nn.Module, name: str, parameter: nn.Parameter) -> None: ...

    def kernel_weights(self, module: nn.Module) -> MoEKernelWeights: ...


class _StackedProjectionAdapter:
    """Default adapter for stacked fused or split expert projections."""

    def __init__(self, name: str, parameter_names: tuple[str, ...], *, fused_gate_up: bool) -> None:
        self.name = name
        self._parameter_names = parameter_names
        self._fused_gate_up = fused_gate_up

    def matches(self, module: nn.Module) -> bool:
        return hasattr(module, "num_experts") and all(
            isinstance(getattr(module, name, None), nn.Parameter) for name in self._parameter_names
        )

    def num_experts(self, module: nn.Module) -> int:
        return int(module.num_experts)

    def expert_parameters(self, module: nn.Module) -> tuple[ExpertParameter, ...]:
        return tuple(ExpertParameter(name, getattr(module, name)) for name in self._parameter_names)

    def replace_expert_parameter(self, module: nn.Module, name: str, parameter: nn.Parameter) -> None:
        if name not in self._parameter_names:
            raise ValueError(f"Adapter {self.name!r} does not manage expert parameter {name!r}.")
        setattr(module, name, parameter)

    def kernel_weights(self, module: nn.Module) -> MoEKernelWeights:
        if self._fused_gate_up:
            return MoEKernelWeights(
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc2_weight=module.down_proj,
                fc1_1_2_weight=module.gate_up_proj,
            )
        return MoEKernelWeights(
            fc1_1_weight=module.gate_proj,
            fc1_2_weight=module.up_proj,
            fc2_weight=module.down_proj,
            fc1_1_2_weight=None,
        )


_ADAPTERS: list[MoEModelAdapter] = [
    _StackedProjectionAdapter("stacked-gate-up", FUSED_STACKED_EXPERT_PARAMETER_NAMES, fused_gate_up=True),
    _StackedProjectionAdapter(
        "stacked-split-gate-up",
        SPLIT_STACKED_EXPERT_PARAMETER_NAMES,
        fused_gate_up=False,
    ),
]


def is_stacked_expert_parameter_name(parameter_name: str) -> bool:
    """Whether a checkpoint key has an expert-indexed leading dimension."""

    marker = ".experts."
    if marker not in parameter_name:
        return False
    leaf = parameter_name.rsplit(marker, 1)[1]
    if leaf.endswith(".weight"):
        leaf = leaf[: -len(".weight")]
    return leaf in STACKED_EXPERT_PARAMETER_NAMES


def register_moe_model_adapter(adapter: MoEModelAdapter, *, prepend: bool = True) -> None:
    """Register an adapter for a non-standard expert representation."""

    if not isinstance(adapter, MoEModelAdapter):
        raise TypeError("PlaceMoE model adapters must implement MoEModelAdapter.")
    if any(existing.name == adapter.name for existing in _ADAPTERS):
        raise ValueError(f"A PlaceMoE model adapter named {adapter.name!r} is already registered.")
    if prepend:
        _ADAPTERS.insert(0, adapter)
    else:
        _ADAPTERS.append(adapter)


def resolve_moe_model_adapter(module: nn.Module) -> MoEModelAdapter | None:
    """Return the first adapter supporting ``module`` or ``None``."""

    explicit = getattr(module, "_veomni_placemoe_adapter", None)
    if explicit is not None:
        if not isinstance(explicit, MoEModelAdapter):
            raise TypeError("module._veomni_placemoe_adapter does not implement MoEModelAdapter.")
        return explicit
    return next((adapter for adapter in _ADAPTERS if adapter.matches(module)), None)


def require_moe_model_adapter(module: nn.Module) -> MoEModelAdapter:
    adapter = resolve_moe_model_adapter(module)
    if adapter is None:
        raise TypeError(
            f"No PlaceMoE model adapter supports {type(module).__qualname__}. "
            "Register a MoEModelAdapter that exposes expert-stacked parameters."
        )
    return adapter


__all__ = [
    "ExpertParameter",
    "FUSED_STACKED_EXPERT_PARAMETER_NAMES",
    "MoEKernelWeights",
    "MoEModelAdapter",
    "SPLIT_STACKED_EXPERT_PARAMETER_NAMES",
    "STACKED_EXPERT_PARAMETER_NAMES",
    "is_stacked_expert_parameter_name",
    "register_moe_model_adapter",
    "require_moe_model_adapter",
    "resolve_moe_model_adapter",
]
