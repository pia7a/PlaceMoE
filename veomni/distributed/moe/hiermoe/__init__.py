# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from .all_to_all import RankDedupDispatchContext, rank_dedup_combine, rank_dedup_dispatch
from .metrics import flush_hiermoe_metrics, peek_hiermoe_metrics, record_hiermoe_metrics
from .state import (
    advance_hiermoe_pipeline_after_combine,
    assert_hiermoe_checkpoint_layout_compatible,
    assert_hiermoe_trainable_only_checkpoint_safe,
    bind_hiermoe_model,
    bind_hiermoe_optimizer,
    configure_hiermoe,
    configure_hiermoe_pipeline_microstep,
    destroy_hiermoe_pipeline_process_groups,
    disable_hiermoe_placement,
    get_hiermoe_expert_layer_key,
    get_hiermoe_expert_layer_key_from_params,
    get_hiermoe_redundant_grad_norm_masks,
    get_hiermoe_state,
    hiermoe_active,
    hiermoe_has_non_identity_placement,
    hiermoe_state_dict,
    load_hiermoe_state_dict,
    maybe_expand_hiermoe_expert_slots,
    maybe_log_hiermoe_metrics,
    maybe_run_hiermoe_expert_swap,
    set_hiermoe_layer_swap_forward_enabled,
    set_hiermoe_route_capture_forward_enabled,
    set_hiermoe_step,
    shutdown_hiermoe_pipeline,
    sync_hiermoe_redundant_gradients,
    wait_hiermoe_pipeline_migration,
)


__all__ = [
    "RankDedupDispatchContext",
    "advance_hiermoe_pipeline_after_combine",
    "assert_hiermoe_checkpoint_layout_compatible",
    "assert_hiermoe_trainable_only_checkpoint_safe",
    "bind_hiermoe_model",
    "bind_hiermoe_optimizer",
    "configure_hiermoe_pipeline_microstep",
    "configure_hiermoe",
    "destroy_hiermoe_pipeline_process_groups",
    "disable_hiermoe_placement",
    "flush_hiermoe_metrics",
    "get_hiermoe_expert_layer_key",
    "get_hiermoe_expert_layer_key_from_params",
    "get_hiermoe_redundant_grad_norm_masks",
    "get_hiermoe_state",
    "hiermoe_active",
    "hiermoe_has_non_identity_placement",
    "hiermoe_state_dict",
    "load_hiermoe_state_dict",
    "maybe_expand_hiermoe_expert_slots",
    "maybe_log_hiermoe_metrics",
    "maybe_run_hiermoe_expert_swap",
    "peek_hiermoe_metrics",
    "rank_dedup_combine",
    "rank_dedup_dispatch",
    "record_hiermoe_metrics",
    "set_hiermoe_layer_swap_forward_enabled",
    "set_hiermoe_route_capture_forward_enabled",
    "set_hiermoe_step",
    "shutdown_hiermoe_pipeline",
    "wait_hiermoe_pipeline_migration",
    "sync_hiermoe_redundant_gradients",
]
