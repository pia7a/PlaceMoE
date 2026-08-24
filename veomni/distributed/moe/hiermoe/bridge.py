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

"""Compatibility bridge for the bundled HierMoE runtime."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ..runtime_bridge import MOE_RUNTIME_BRIDGE_API_VERSION
from .state import (
    bind_hiermoe_model,
    bind_hiermoe_optimizer,
    configure_hiermoe,
    configure_hiermoe_pipeline_microstep,
    destroy_hiermoe_pipeline_process_groups,
    disable_hiermoe_placement,
    maybe_expand_hiermoe_expert_slots,
    maybe_log_hiermoe_metrics,
    maybe_run_hiermoe_expert_swap,
    set_hiermoe_layer_swap_forward_enabled,
    set_hiermoe_route_capture_forward_enabled,
    set_hiermoe_step,
    shutdown_hiermoe_pipeline,
    sync_hiermoe_redundant_gradients,
)


class HierMoERuntimeBridge:
    """Expose the existing HierMoE implementation through the stable API."""

    name = "hiermoe"
    api_version = MOE_RUNTIME_BRIDGE_API_VERSION

    @staticmethod
    def configure(config: Any, **kwargs: Any) -> Any:
        return configure_hiermoe(config, **kwargs)

    @staticmethod
    def maybe_expand_expert_slots(model: Any, ep_size: int) -> None:
        maybe_expand_hiermoe_expert_slots(model, ep_size)

    @staticmethod
    def bind_model(model: Any) -> None:
        bind_hiermoe_model(model)

    @staticmethod
    def bind_optimizer(optimizer: Any) -> None:
        bind_hiermoe_optimizer(optimizer)

    @staticmethod
    def set_step(step: int) -> None:
        set_hiermoe_step(step)

    @staticmethod
    @contextmanager
    def training_forward():
        previous_swap = set_hiermoe_layer_swap_forward_enabled(True)
        previous_capture = set_hiermoe_route_capture_forward_enabled(True)
        try:
            yield
        finally:
            set_hiermoe_route_capture_forward_enabled(previous_capture)
            set_hiermoe_layer_swap_forward_enabled(previous_swap)

    @staticmethod
    def placement_disabled():
        return disable_hiermoe_placement()

    @staticmethod
    def configure_microstep(micro_step: int, num_micro_steps: int) -> None:
        configure_hiermoe_pipeline_microstep(micro_step, num_micro_steps)

    @staticmethod
    def sync_redundant_gradients() -> None:
        sync_hiermoe_redundant_gradients()

    @staticmethod
    def run_step_update(step: int) -> str | None:
        return maybe_run_hiermoe_expert_swap(step)

    @staticmethod
    def log_metrics(step: int) -> None:
        maybe_log_hiermoe_metrics(step)

    @staticmethod
    def shutdown() -> None:
        shutdown_hiermoe_pipeline()

    @staticmethod
    def destroy_process_groups() -> None:
        destroy_hiermoe_pipeline_process_groups()


def create_bridge() -> HierMoERuntimeBridge:
    return HierMoERuntimeBridge()


__all__ = ["HierMoERuntimeBridge", "create_bridge"]
