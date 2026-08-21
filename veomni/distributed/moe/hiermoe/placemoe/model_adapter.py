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

"""Stable model boundary used by the PlaceMoE runtime.

The planner is model independent.  Runtime state migration only needs to know
which tensors are stacked along the expert dimension, while the fused MoE
kernel additionally needs a normalized view of the feed-forward weights.
Model integrations provide those two views through :class:`MoEModelAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from veomni.models.moe_parameter import (
    FUSED_STACKED_EXPERT_PARAMETER_NAMES,
    SPLIT_STACKED_EXPERT_PARAMETER_NAMES,
)


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
    """Describe how a model exposes its expert parameters and kernel weights."""

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
    _StackedProjectionAdapter(
        "stacked-gate-up",
        FUSED_STACKED_EXPERT_PARAMETER_NAMES,
        fused_gate_up=True,
    ),
    _StackedProjectionAdapter(
        "stacked-split-gate-up",
        SPLIT_STACKED_EXPERT_PARAMETER_NAMES,
        fused_gate_up=False,
    ),
]


def register_moe_model_adapter(adapter: MoEModelAdapter, *, prepend: bool = True) -> None:
    """Register a model adapter.

    Third-party model packages should call this once during model
    registration.  ``prepend=True`` lets a model-specific adapter override a
    structural default without changing PlaceMoE itself.
    """

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
    "MoEKernelWeights",
    "MoEModelAdapter",
    "register_moe_model_adapter",
    "require_moe_model_adapter",
    "resolve_moe_model_adapter",
]
