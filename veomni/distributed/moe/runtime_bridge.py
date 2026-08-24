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

"""Stable host boundary for optional MoE runtime extensions.

VeOmni owns the training loop and model construction.  A runtime extension
owns only the MoE-specific lifecycle exposed here.  Keeping these calls behind
one versioned interface lets an external PlaceMoE package integrate with a
model-modified VeOmni checkout without patching individual model files.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable


MOE_RUNTIME_BRIDGE_API_VERSION = 1
MOE_RUNTIME_BRIDGE_ENTRY_POINT = "veomni.moe_runtime_bridges"


@runtime_checkable
class MoERuntimeBridge(Protocol):
    """Lifecycle contract implemented by a VeOmni MoE runtime extension."""

    name: str
    api_version: int

    def configure(
        self,
        config: Any,
        *,
        ep_group: Any | None,
        ep_fsdp_size: int,
        activation_checkpointing_enabled: bool,
        fsdp_offload_enabled: bool,
        gradient_bytes_per_element: int,
    ) -> Any: ...

    def maybe_expand_expert_slots(self, model: Any, ep_size: int) -> None: ...

    def bind_model(self, model: Any) -> None: ...

    def bind_optimizer(self, optimizer: Any) -> None: ...

    def set_step(self, step: int) -> None: ...

    def training_forward(self) -> AbstractContextManager[None]: ...

    def placement_disabled(self) -> AbstractContextManager[None]: ...

    def configure_microstep(self, micro_step: int, num_micro_steps: int) -> None: ...

    def sync_redundant_gradients(self) -> None: ...

    def run_step_update(self, step: int) -> str | None: ...

    def log_metrics(self, step: int) -> None: ...

    def shutdown(self) -> None: ...

    def destroy_process_groups(self) -> None: ...


_ACTIVE_BRIDGE: MoERuntimeBridge | None = None


def _load_entry_point_bridge(name: str) -> MoERuntimeBridge | None:
    matches = tuple(entry_points(group=MOE_RUNTIME_BRIDGE_ENTRY_POINT, name=name))
    if not matches:
        return None
    if len(matches) != 1:
        providers = sorted(f"{item.dist.name}:{item.value}" for item in matches)
        raise RuntimeError(f"Multiple MoE runtime bridges named {name!r} are installed: {providers}.")
    factory = matches[0].load()
    return factory()


def _load_builtin_bridge(name: str) -> MoERuntimeBridge:
    if name == "placemoe":
        factory = import_module("placemoe.bridge").create_bridge
    elif name == "hiermoe":
        factory = import_module("veomni.distributed.moe.hiermoe.bridge").create_bridge
    else:
        raise LookupError(f"No built-in MoE runtime bridge named {name!r}.")
    return factory()


def _validate_bridge(bridge: MoERuntimeBridge, expected_name: str) -> None:
    if not isinstance(bridge, MoERuntimeBridge):
        raise TypeError(f"MoE runtime bridge {expected_name!r} does not implement MoERuntimeBridge.")
    if bridge.name != expected_name:
        raise ValueError(f"Requested MoE runtime bridge {expected_name!r}, but provider reports {bridge.name!r}.")
    if int(bridge.api_version) != MOE_RUNTIME_BRIDGE_API_VERSION:
        raise RuntimeError(
            f"MoE runtime bridge {expected_name!r} uses API version {bridge.api_version}; "
            f"VeOmni requires {MOE_RUNTIME_BRIDGE_API_VERSION}."
        )


def load_moe_runtime_bridge(name: str) -> MoERuntimeBridge:
    """Load and validate one installed or bundled runtime provider."""

    bridge = _load_entry_point_bridge(name) or _load_builtin_bridge(name)
    _validate_bridge(bridge, name)
    return bridge


def configure_moe_runtime_bridge(config: Any, **kwargs: Any) -> Any:
    """Select, configure, and publish the runtime bridge."""

    global _ACTIVE_BRIDGE

    placemoe_config = getattr(config, "placemoe", None)
    name = "placemoe" if bool(getattr(placemoe_config, "enabled", False)) else "hiermoe"
    bridge = load_moe_runtime_bridge(name)
    state = bridge.configure(config, **kwargs)
    _ACTIVE_BRIDGE = bridge
    return state


def get_moe_runtime_bridge() -> MoERuntimeBridge:
    if _ACTIVE_BRIDGE is None:
        raise RuntimeError("The MoE runtime bridge has not been configured.")
    return _ACTIVE_BRIDGE


def get_configured_moe_runtime_bridge() -> MoERuntimeBridge | None:
    """Return the active bridge, or ``None`` outside the trainer lifecycle."""

    return _ACTIVE_BRIDGE


def reset_moe_runtime_bridge_for_test() -> None:
    """Clear process-global bridge state for isolated unit tests."""

    global _ACTIVE_BRIDGE
    _ACTIVE_BRIDGE = None


__all__ = [
    "MOE_RUNTIME_BRIDGE_API_VERSION",
    "MOE_RUNTIME_BRIDGE_ENTRY_POINT",
    "MoERuntimeBridge",
    "configure_moe_runtime_bridge",
    "get_configured_moe_runtime_bridge",
    "get_moe_runtime_bridge",
    "load_moe_runtime_bridge",
]
