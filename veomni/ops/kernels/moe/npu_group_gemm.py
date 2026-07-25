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


import os
import time
from typing import List, Optional

import torch
import torch.distributed as dist
import torch_npu

from ....distributed.moe.comm import all_to_all
from ....distributed.moe.hiermoe import (
    get_hiermoe_state,
    hiermoe_active,
    rank_dedup_combine,
    rank_dedup_dispatch,
    record_hiermoe_metrics,
)
from ....distributed.moe.moe_utils import sort_chunks_by_idxs
from ....distributed.moe.timing import (
    current_full_profile_phase,
    enter_moe_profile_range,
    exit_moe_profile_range,
    moe_timing_context,
    moe_timing_enabled,
    moe_timing_event,
    record_moe_timing_span,
)
from ....distributed.moe.validation import moe_validation_enabled, record_moe_validation_routing
from ....distributed.parallel_state import get_parallel_state
from ....utils.device import stream_synchronize
from ._kernels.kernel.npu_group_gemm import npu_group_gemm


_NPU_MOE_TIMING_CALL_INDEX = 0


def _moe_timing_num_layers() -> int:
    raw = os.environ.get("VERL_MOE_TIMING_NUM_LAYERS", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _start_npu_moe_timing(
    num_experts: int,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
) -> dict[str, object] | None:
    global _NPU_MOE_TIMING_CALL_INDEX
    if not moe_timing_enabled() and not moe_validation_enabled():
        return None

    call_index = _NPU_MOE_TIMING_CALL_INDEX
    _NPU_MOE_TIMING_CALL_INDEX += 1
    num_layers = _moe_timing_num_layers()
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    return {
        "call_index": call_index,
        "layer": call_index % num_layers if num_layers else None,
        "num_layers": num_layers,
        "num_experts": int(num_experts),
        "ep_size": int(ep_size),
        "tokens": int(hidden_states.shape[0]) if hidden_states.ndim > 1 else int(hidden_states.numel()),
        "token_expert_assignments": int(selected_experts.numel()),
        "top_k": int(selected_experts.shape[-1]) if selected_experts.ndim > 1 else 1,
        "phase": current_full_profile_phase(),
    }


def _npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """NPU single-device fused MoE forward pass (non-EP).

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Weights are merged and transposed for the NPU group-gemm kernel.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = npu_group_gemm(intermediate_activations, fc2_weight.transpose(1, 2), tokens_per_expert)
    hidden_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return hidden_states


def npu_ep_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    ep_group: Optional[dist.ProcessGroup] = None,
    layer_key: str | None = None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward pass.

    Accepts either split fc1 weights or a merged fc1_1_2_weight tensor.
    Handles alltoall dispatch/combine for expert parallelism.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    timing_record = _start_npu_moe_timing(num_experts, hidden_states, selected_experts, ep_group)
    record_moe_validation_routing(
        timing_record,
        selected_experts=selected_experts,
        num_experts=num_experts,
        ep_group=ep_group,
    )
    if hiermoe_active():
        state = get_hiermoe_state()
        placement_manager = (
            state.expert_swap_manager
            if (
                state is not None
                and state.placement_mapping_enabled
                and state.expert_swap_manager is not None
                and state.expert_swap_manager.placement_planning_enabled()
                and layer_key is not None
            )
            else None
        )
        capture_placement_timing = (
            placement_manager is not None
            and state.layer_swap_forward_enabled
            and placement_manager.layer_calibration_enabled()
        )
        record_metrics = (
            state is not None and state.layer_swap_forward_enabled and state.current_step % state.log_interval == 0
        )
        record_wall_metrics = record_metrics and state.debug_validate
        baseline_original_all_to_all_ms = None
        if state is not None and state.debug_validate:
            baseline_start = time.perf_counter()
            with torch.no_grad():
                (
                    debug_input_splits,
                    debug_output_splits,
                    debug_tokens_per_local_expert,
                    _debug_tokens_per_expert,
                ) = dispatch_preprocess(selected_experts, num_experts, ep_group)
                alltoall_dispatch(
                    hidden_states.detach(),
                    selected_experts,
                    debug_input_splits,
                    debug_output_splits,
                    num_experts,
                    debug_tokens_per_local_expert,
                    ep_group,
                )
                stream_synchronize()
            baseline_original_all_to_all_ms = (time.perf_counter() - baseline_start) * 1000.0

        placement_already_applied = False
        if placement_manager is not None and state.expert_swap_mode == "layer" and state.layer_swap_forward_enabled:
            assert layer_key is not None
            state.expert_swap_pair = placement_manager.maybe_swap_layer_on_routing(
                layer_key=layer_key,
                selected_experts=selected_experts,
                hidden_size=hidden_states.shape[-1],
                bytes_per_element=hidden_states.element_size(),
                step=state.current_step,
            )
            placement_already_applied = True

        placement_dispatch_start = placement_manager.placement_timing_event() if capture_placement_timing else None
        dispatch_start = time.perf_counter() if record_wall_metrics else None
        region_start = moe_timing_event() if timing_record is not None else None
        with moe_timing_context(timing_record, component="all_to_all", section="hiermoe_pre_all_to_all"):
            hidden_states, hiermoe_ctx, num_global_sum_tokens_per_local_expert = rank_dedup_dispatch(
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                num_experts=num_experts,
                ep_group=ep_group,
                layer_key=layer_key,
                placement_already_applied=placement_already_applied,
            )
        placement_dispatch_end = placement_manager.placement_timing_event() if capture_placement_timing else None
        dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0 if dispatch_start is not None else None
        region_end = moe_timing_event() if timing_record is not None else None
        record_moe_timing_span(
            timing_record,
            direction="forward",
            component="moe_comm_region",
            section="hiermoe_pre_all_to_all_region",
            start_event=region_start,
            end_event=region_end,
        )
        if capture_placement_timing:
            assert layer_key is not None
            assert placement_manager is not None
            placement_manager.record_dispatch_statistics(
                layer_key=layer_key,
                step=state.current_step,
                dispatch_context=hiermoe_ctx,
            )

        if placement_already_applied and placement_manager is not None and layer_key is not None:
            placement_manager.wait_pending_layer_swap(layer_key)
        if placement_manager is not None and layer_key is not None and state.layer_swap_forward_enabled:
            placement_manager.open_pipeline_planner_collective_window(layer_key)

        if fc1_1_2_weight is not None:
            fc1_weight = fc1_1_2_weight
        else:
            fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
        fc1_weight = fc1_weight.transpose(1, 2)
        active_local_experts = int(num_global_sum_tokens_per_local_expert.numel())
        if active_local_experts < int(fc1_weight.shape[0]):
            fc1_weight = fc1_weight[:active_local_experts]
            fc2_weight = fc2_weight[:active_local_experts]
        placement_compute_start = placement_manager.placement_timing_event() if capture_placement_timing else None
        expert_start = time.perf_counter() if record_wall_metrics else None
        with moe_timing_context(timing_record, component="expert_compute", section="hiermoe_expert_compute"):
            annotation = enter_moe_profile_range(
                timing_record,
                direction="forward",
                component="expert_compute",
                section="hiermoe_expert_compute",
            )
            timing_start = moe_timing_event() if timing_record is not None else None
            try:
                intermediate_hidden_states = npu_group_gemm(
                    hidden_states,
                    fc1_weight,
                    num_global_sum_tokens_per_local_expert,
                )
                intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
                hidden_states = npu_group_gemm(
                    intermediate_activations, fc2_weight.transpose(1, 2), num_global_sum_tokens_per_local_expert
                )
            finally:
                timing_end = moe_timing_event() if timing_record is not None else None
                exit_moe_profile_range(annotation)
            record_moe_timing_span(
                timing_record,
                direction="forward",
                component="expert_compute",
                section="hiermoe_expert_compute",
                start_event=timing_start,
                end_event=timing_end,
            )
        expert_ms = (time.perf_counter() - expert_start) * 1000.0 if expert_start is not None else None
        placement_compute_end = placement_manager.placement_timing_event() if capture_placement_timing else None

        if placement_manager is not None and layer_key is not None and state.layer_swap_forward_enabled:
            placement_manager.open_pipeline_planner_score_window(layer_key)
        placement_combine_start = placement_manager.placement_timing_event() if capture_placement_timing else None
        combine_start = time.perf_counter() if record_wall_metrics else None
        region_start = moe_timing_event() if timing_record is not None else None
        with moe_timing_context(timing_record, component="all_to_all", section="hiermoe_post_all_to_all"):
            hidden_states = rank_dedup_combine(hidden_states, hiermoe_ctx)
        placement_combine_end = placement_manager.placement_timing_event() if capture_placement_timing else None
        if placement_manager is not None and layer_key is not None and state.layer_swap_forward_enabled:
            placement_manager.advance_pipeline_after_combine(layer_key)
        combine_ms = (time.perf_counter() - combine_start) * 1000.0 if combine_start is not None else None
        region_end = moe_timing_event() if timing_record is not None else None
        record_moe_timing_span(
            timing_record,
            direction="forward",
            component="moe_comm_region",
            section="hiermoe_post_all_to_all_region",
            start_event=region_start,
            end_event=region_end,
        )
        if record_metrics:
            metrics = {
                "enable": True,
                "selected_dim": hiermoe_ctx.selected_dim,
                "dedup_ratio_dispatch": hiermoe_ctx.dedup_ratio_dispatch,
                "dedup_ratio_combine": hiermoe_ctx.dedup_ratio_combine,
                "expert_swap_pair": state.expert_swap_pair,
                "expert_swap_interval": state.expert_swap_interval,
                "expert_swap_max_pairs_per_layer": state.expert_swap_max_pairs_per_layer,
                "perf_model_source": state.perf_model.source,
            }
            if record_wall_metrics:
                metrics["dispatch_wall_ms"] = float(dispatch_ms or 0.0)
                metrics["combine_wall_ms"] = float(combine_ms or 0.0)
                metrics["local_expert_compute_wall_ms"] = float(expert_ms or 0.0)
                metrics["baseline_original_all_to_all_ms"] = float(baseline_original_all_to_all_ms or 0.0)
            record_hiermoe_metrics(metrics)
        if capture_placement_timing:
            assert layer_key is not None
            assert placement_manager is not None
            assert placement_dispatch_start is not None and placement_dispatch_end is not None
            assert placement_compute_start is not None and placement_compute_end is not None
            assert placement_combine_start is not None and placement_combine_end is not None
            placement_manager.record_layer_timing(
                layer_key=layer_key,
                step=state.current_step,
                selected_experts=selected_experts,
                tokens_per_local_expert=num_global_sum_tokens_per_local_expert,
                dispatch_start=placement_dispatch_start,
                dispatch_end=placement_dispatch_end,
                compute_start=placement_compute_start,
                compute_end=placement_compute_end,
                combine_start=placement_combine_start,
                combine_end=placement_combine_end,
                selected_dim=hiermoe_ctx.selected_dim,
            )
        return hidden_states

    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    region_start = moe_timing_event() if timing_record is not None else None
    with moe_timing_context(timing_record, component="all_to_all", section="pre_all_to_all"):
        hidden_states, unpermute_indices = alltoall_dispatch(
            hidden_states,
            selected_experts,
            input_splits,
            output_splits,
            num_experts,
            num_global_tokens_per_local_expert,
            ep_group,
        )
    region_end = moe_timing_event() if timing_record is not None else None
    record_moe_timing_span(
        timing_record,
        direction="forward",
        component="moe_comm_region",
        section="pre_all_to_all_region",
        start_event=region_start,
        end_event=region_end,
    )

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    with moe_timing_context(timing_record, component="expert_compute", section="expert_compute"):
        annotation = enter_moe_profile_range(
            timing_record,
            direction="forward",
            component="expert_compute",
            section="expert_compute",
        )
        timing_start = moe_timing_event() if timing_record is not None else None
        try:
            intermediate_hidden_states = npu_group_gemm(
                hidden_states,
                fc1_weight,
                num_global_sum_tokens_per_local_expert,
            )
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            hidden_states = npu_group_gemm(
                intermediate_activations, fc2_weight.transpose(1, 2), num_global_sum_tokens_per_local_expert
            )
        finally:
            timing_end = moe_timing_event() if timing_record is not None else None
            exit_moe_profile_range(annotation)
        record_moe_timing_span(
            timing_record,
            direction="forward",
            component="expert_compute",
            section="expert_compute",
            start_event=timing_start,
            end_event=timing_end,
        )

    region_start = moe_timing_event() if timing_record is not None else None
    with moe_timing_context(timing_record, component="all_to_all", section="post_all_to_all"):
        hidden_states = alltoall_combine(
            hidden_states,
            routing_weights,
            unpermute_indices,
            input_splits,
            output_splits,
            num_experts,
            num_global_tokens_per_local_expert,
            ep_group,
        )
    region_end = moe_timing_event() if timing_record is not None else None
    record_moe_timing_span(
        timing_record,
        direction="forward",
        component="moe_comm_region",
        section="post_all_to_all_region",
        start_event=region_start,
        end_event=region_end,
    )
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    if ep_group is None:
        ep_size = 1
        ep_rank = 0
    else:
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    num_local_experts = num_global_experts // ep_size

    num_local_tokens_per_expert = torch.bincount(selected_experts.view(-1), minlength=num_global_experts)

    if ep_group is None or ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    hidden_states, unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    hidden_states = all_to_all(ep_group, hidden_states, output_splits, input_splits)

    stream_synchronize()
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    permute_order = torch.arange(num_global_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )
    return hidden_states, unpermute_indices


def alltoall_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    unpermute_indices: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    assert num_global_experts % ep_size == 0, (
        f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    unpermute_order = torch.arange(num_global_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    hidden_states = all_to_all(ep_group, hidden_states, input_splits, output_splits)
    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, unpermute_indices, probs=routing_weights)
    return hidden_states


def npu_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    layer_key: str | None = None,
):
    if get_parallel_state().ep_enabled:
        final_hidden_states = npu_ep_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group=get_parallel_state().ep_group,
            layer_key=layer_key,
        )
    else:
        final_hidden_states = _npu_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
        )
    return final_hidden_states
