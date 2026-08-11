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
from collections import defaultdict
from typing import Any

import torch

from ....distributed.moe import EPGroupGemm, EPMergedFc1GroupGemm, preprocess, token_pre_all2all, tokens_post_all2all
from ....distributed.moe.hiermoe import (
    get_hiermoe_state,
    hiermoe_active,
    rank_dedup_combine,
    rank_dedup_dispatch,
    record_hiermoe_metrics,
)
from ....distributed.moe.timing import (
    current_full_profile_phase,
    flush_moe_timing_spans,
    moe_timing_context,
    record_moe_timing_span,
)
from ....distributed.moe.validation import moe_validation_enabled, record_moe_validation_routing
from ....distributed.parallel_state import get_parallel_state
from ....utils.accelerator_timing import (
    AcceleratorEvent,
    accelerator_timing_available,
    record_accelerator_event,
    synchronize_accelerator,
)
from ._kernels.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from ._kernels.kernel.moe import expert_histogram, moe_gather, moe_scatter


_MOE_TIMING_RECORDS: list[dict[str, Any]] = []
_MOE_TIMING_CALL_INDEX = 0


def _moe_timing_enabled() -> bool:
    return bool(os.environ.get("VERL_MOE_TIMING_DIR")) and accelerator_timing_available()


def _moe_timing_event() -> AcceleratorEvent | None:
    if not _moe_timing_enabled():
        return None
    return record_accelerator_event()


def _moe_timing_num_layers() -> int:
    raw = os.environ.get("VERL_MOE_TIMING_NUM_LAYERS", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _start_moe_timing(
    num_experts: int,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
) -> dict[str, Any] | None:
    global _MOE_TIMING_CALL_INDEX
    timing_enabled = _moe_timing_enabled()
    if not timing_enabled and not moe_validation_enabled():
        return None

    call_index = _MOE_TIMING_CALL_INDEX
    _MOE_TIMING_CALL_INDEX += 1
    num_layers = _moe_timing_num_layers()
    ep_group = get_parallel_state().ep_group
    ep_size = torch.distributed.get_world_size(ep_group) if ep_group is not None else 1
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
        "events": {"start": _moe_timing_event()} if timing_enabled else {},
        "timing_enabled": timing_enabled,
    }


def _mark_moe_timing(record: dict[str, Any] | None, name: str) -> None:
    if record is not None and record.get("timing_enabled"):
        record["events"][name] = _moe_timing_event()


def _finish_moe_timing(record: dict[str, Any] | None) -> None:
    if record is not None and record.get("timing_enabled"):
        record["events"]["end"] = _moe_timing_event()
        record_moe_timing_span(
            record,
            direction="forward",
            component="moe_total",
            section="moe_forward_total",
            start_event=record["events"]["start"],
            end_event=record["events"]["end"],
        )
        _MOE_TIMING_RECORDS.append(record)


def flush_moe_timing_payload(current_step: int, num_layers: int | None = None) -> dict[str, Any]:
    """Synchronize once and return aggregate MoE timing since the previous flush."""
    global _MOE_TIMING_RECORDS
    records = _MOE_TIMING_RECORDS
    _MOE_TIMING_RECORDS = []
    span_payload = flush_moe_timing_spans()
    if not records:
        if span_payload:
            return {
                "step": int(current_step),
                "num_layers_config": int(num_layers or _moe_timing_num_layers()),
                "num_records": 0,
                **span_payload,
                "note": "Accelerator-event span timings cover trainer-side EP MoE forward/backward components.",
            }
        return {}

    synchronize_accelerator()

    configured_layers = num_layers or _moe_timing_num_layers()
    aggregates: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "tokens": 0,
            "token_expert_assignments": 0,
            "preprocess_ms_sum": 0.0,
            "pre_all2all_ms_sum": 0.0,
            "expert_compute_ms_sum": 0.0,
            "post_all2all_ms_sum": 0.0,
            "all_to_all_ms_sum": 0.0,
            "other_ms_sum": 0.0,
            "total_ms_sum": 0.0,
            "total_ms_max": 0.0,
        }
    )

    for record in records:
        events = record["events"]
        preprocess_ms = events["start"].elapsed_time(events["after_preprocess"])
        pre_all2all_ms = events["after_preprocess"].elapsed_time(events["after_pre_all2all"])
        expert_compute_ms = events["after_pre_all2all"].elapsed_time(events["after_expert_compute"])
        post_all2all_ms = events["after_expert_compute"].elapsed_time(events["end"])
        total_ms = events["start"].elapsed_time(events["end"])
        all_to_all_ms = pre_all2all_ms + post_all2all_ms
        other_ms = max(0.0, total_ms - all_to_all_ms - expert_compute_ms)

        layer = record.get("layer")
        if layer is None:
            layer = record["call_index"]
        layer_stats = aggregates[int(layer)]
        layer_stats["calls"] += 1
        layer_stats["tokens"] += int(record["tokens"])
        layer_stats["token_expert_assignments"] += int(record["token_expert_assignments"])
        layer_stats["preprocess_ms_sum"] += float(preprocess_ms)
        layer_stats["pre_all2all_ms_sum"] += float(pre_all2all_ms)
        layer_stats["expert_compute_ms_sum"] += float(expert_compute_ms)
        layer_stats["post_all2all_ms_sum"] += float(post_all2all_ms)
        layer_stats["all_to_all_ms_sum"] += float(all_to_all_ms)
        layer_stats["other_ms_sum"] += float(other_ms)
        layer_stats["total_ms_sum"] += float(total_ms)
        layer_stats["total_ms_max"] = max(float(layer_stats["total_ms_max"]), float(total_ms))

    layers = []
    for layer, stats in sorted(aggregates.items()):
        calls = max(1, int(stats["calls"]))
        item = {"layer": layer, **stats}
        for key in (
            "preprocess_ms",
            "pre_all2all_ms",
            "expert_compute_ms",
            "post_all2all_ms",
            "all_to_all_ms",
            "other_ms",
            "total_ms",
        ):
            item[f"{key}_avg"] = item[f"{key}_sum"] / calls
        layers.append(item)

    return {
        "step": int(current_step),
        "num_layers_config": int(configured_layers),
        "num_records": len(records),
        "layers": layers,
        **span_payload,
        "note": (
            "layers contains legacy fused MoE forward timing. span_layers/span_components contain "
            "component spans for EP MoE forward/backward all-to-all and expert compute."
        ),
    }


class TritonFusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # MOE Step 3: dispatch input tokens to the experts
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        # MOE Step 3-1: compute the token num for each expert
        # splits shape (num_experts)
        splits = expert_histogram(expert_index, num_experts)

        # MOE Step 3-2: compute the each token's index in result
        # scatter_index shape (batch_size * sequence_len, topk)
        # TODO(wenyawei): opt it
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # MOE Step 3-3: compute the result, select tokens by scatter_index, and put them together
        # scatter_output shape (batch_size * sequence_len * topk, hidden_size)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # MOE Step 4: compute linear layer 1-1
        # Not consistent.
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 6: compute linear layer 1-2
        # fc1_2_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_2_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 5: compute the actication of linear layer 1-1
        # TODO(wenyawei): act function
        # fc1_1_activation shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7: compute final result of linear layer 1
        fc1_activation = fc1_1_activation * fc1_2_output

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # MOE Step 9: compute linear layer 2
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 10: gather the final token result by averate the the topk token results
        expert_output = moe_gather(fc2_output, scatter_index)

        # reshape the output with input shape
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # MOE Step 9
        # grad_fc1_weighted_output = torch.empty_like(fc1_weighted_output)

        # dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8
        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 6
        # grad_scatter_output_2 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=scatter_output,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # MOE Step 4
        # grad_scatter_output_1 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=scatter_output,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        # MOE Step 3-3
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)

        # MOE Step 3-2: no grad
        # MOE Step 3-1: no grad

        # reshape the result with input shape
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


class MergedFc1TritonFusedMoeExpertFunction(torch.autograd.Function):
    """Fused MoE autograd function that natively accepts a merged fc1_1_2 weight [E, 2I, H].

    Uses a single group_gemm_same_nk call for fc1 instead of two separate calls,
    avoiding the split+contiguous copy when the caller already has merged weights.
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
    ):
        splits = expert_histogram(expert_index, num_experts)
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        cumsum_t = torch.cumsum(splits, dim=0)

        # Single fc1 gemm: output shape [T, 2I]
        fc1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # chunk is a view, no copy
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        expert_output = moe_gather(fc2_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # MOE Step 9 - dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # MOE Step 9 - wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # Merge grad_fc1_1_output and grad_fc1_2_output back to [T, 2I]
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        # MOE Step 4 - single dgrad for merged fc1
        grad_scatter_output = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # MOE Step 4 - single wgrad for merged fc1
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=scatter_output,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def group_gemm_fused_moe_forward(
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
    """Triton grouped-gemm fused MoE forward pass.

    Accepts either split fc1 weights (fc1_1_weight, fc1_2_weight) or a merged
    fc1_1_2_weight tensor.

    - Non-EP path: dispatches to ``MergedFc1TritonFusedMoeExpertFunction`` when
      merged weights are provided, or ``TritonFusedMoeExpertFunction`` when split
      weights are provided.  No format conversion is performed.
    - EP path: always resolves to split format for ``EPGroupGemm``.
    """
    if get_parallel_state().ep_enabled:
        timing_record = _start_moe_timing(num_experts, hidden_states, selected_experts)
        record_moe_validation_routing(
            timing_record,
            selected_experts=selected_experts,
            num_experts=num_experts,
            ep_group=get_parallel_state().ep_group,
        )
        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
        else:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")
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
                    debug_expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(
                        2, 1, 0
                    )
                    (
                        debug_input_splits,
                        debug_output_splits,
                        debug_tokens_per_local_expert,
                        _debug_tokens_per_expert,
                    ) = preprocess(
                        expert_mask=debug_expert_mask,
                        num_experts=num_experts,
                        ep_group=get_parallel_state().ep_group,
                    )
                    token_pre_all2all(
                        hidden_states=hidden_states.detach(),
                        expert_mask=debug_expert_mask,
                        num_experts=num_experts,
                        input_splits=debug_input_splits,
                        output_splits=debug_output_splits,
                        num_global_tokens_per_local_expert=debug_tokens_per_local_expert,
                        ep_group=get_parallel_state().ep_group,
                    )
                    synchronize_accelerator()
                baseline_original_all_to_all_ms = (time.perf_counter() - baseline_start) * 1000.0

            placement_already_applied = False
            if (
                placement_manager is not None
                and state.expert_swap_mode == "layer"
                and state.layer_swap_forward_enabled
            ):
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
            region_start = _moe_timing_event() if timing_record is not None else None
            with moe_timing_context(timing_record, component="all_to_all", section="hiermoe_pre_all_to_all"):
                permute_tokens, hiermoe_ctx, tokens_per_local_expert = rank_dedup_dispatch(
                    hidden_states=hidden_states,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    num_experts=num_experts,
                    ep_group=get_parallel_state().ep_group,
                    layer_key=layer_key,
                    placement_already_applied=placement_already_applied,
                )
            placement_dispatch_end = placement_manager.placement_timing_event() if capture_placement_timing else None
            dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0 if dispatch_start is not None else None
            region_end = _moe_timing_event() if timing_record is not None else None
            record_moe_timing_span(
                timing_record,
                direction="forward",
                component="moe_comm_region",
                section="hiermoe_pre_all_to_all_region",
                start_event=region_start,
                end_event=region_end,
            )
            if placement_manager is not None and layer_key is not None and state.layer_swap_forward_enabled:
                assert layer_key is not None
                placement_manager.record_dispatch_statistics(
                    layer_key=layer_key,
                    step=state.current_step,
                    dispatch_context=hiermoe_ctx,
                )

            if placement_already_applied and placement_manager is not None and layer_key is not None:
                placement_manager.wait_pending_layer_swap(layer_key)
            cumsum = torch.cumsum(tokens_per_local_expert, dim=0).to(permute_tokens.device)
            active_local_experts = int(tokens_per_local_expert.numel())
            if active_local_experts < int(fc2_weight.shape[0]):
                fc2_weight = fc2_weight[:active_local_experts]
                if fc1_1_2_weight is not None:
                    fc1_1_2_weight = fc1_1_2_weight[:active_local_experts]
                else:
                    fc1_1_weight = fc1_1_weight[:active_local_experts]
                    fc1_2_weight = fc1_2_weight[:active_local_experts]
            placement_compute_start = placement_manager.placement_timing_event() if capture_placement_timing else None
            expert_start = time.perf_counter() if record_wall_metrics else None
            if fc1_1_2_weight is not None:
                with moe_timing_context(timing_record, component="expert_compute", section="hiermoe_expert_compute"):
                    final_permute_tokens = EPMergedFc1GroupGemm.apply(
                        permute_tokens,
                        cumsum,
                        fc1_1_2_weight,
                        fc2_weight,
                    )
            else:
                with moe_timing_context(timing_record, component="expert_compute", section="hiermoe_expert_compute"):
                    final_permute_tokens = EPGroupGemm.apply(
                        permute_tokens,
                        cumsum,
                        fc1_1_weight,
                        fc1_2_weight,
                        fc2_weight,
                    )
            expert_ms = (time.perf_counter() - expert_start) * 1000.0 if expert_start is not None else None
            placement_compute_end = placement_manager.placement_timing_event() if capture_placement_timing else None

            placement_combine_start = placement_manager.placement_timing_event() if capture_placement_timing else None
            combine_start = time.perf_counter() if record_wall_metrics else None
            region_start = _moe_timing_event() if timing_record is not None else None
            with moe_timing_context(timing_record, component="all_to_all", section="hiermoe_post_all_to_all"):
                final_hidden_states = rank_dedup_combine(final_permute_tokens, hiermoe_ctx)
            placement_combine_end = placement_manager.placement_timing_event() if capture_placement_timing else None
            if placement_manager is not None and layer_key is not None and state.layer_swap_forward_enabled:
                placement_manager.advance_pipeline_after_combine(layer_key)
            combine_ms = (time.perf_counter() - combine_start) * 1000.0 if combine_start is not None else None
            region_end = _moe_timing_event() if timing_record is not None else None
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
                    tokens_per_local_expert=tokens_per_local_expert,
                    dispatch_start=placement_dispatch_start,
                    dispatch_end=placement_dispatch_end,
                    compute_start=placement_compute_start,
                    compute_end=placement_compute_end,
                    combine_start=placement_combine_start,
                    combine_end=placement_combine_end,
                    selected_dim=hiermoe_ctx.selected_dim,
                    communication_events=hiermoe_ctx.internal_timing_events,
                )
            _finish_moe_timing(timing_record)
            return final_hidden_states
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        # preprocess, permute token for ep
        input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
            preprocess(
                expert_mask=expert_mask,
                num_experts=num_experts,
                ep_group=get_parallel_state().ep_group,
            )
        )
        _mark_moe_timing(timing_record, "after_preprocess")
        region_start = _moe_timing_event() if timing_record is not None else None
        with moe_timing_context(timing_record, component="all_to_all", section="pre_all_to_all"):
            permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
                hidden_states=hidden_states,
                expert_mask=expert_mask,
                num_experts=num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                ep_group=get_parallel_state().ep_group,
            )
        region_end = _moe_timing_event() if timing_record is not None else None
        record_moe_timing_span(
            timing_record,
            direction="forward",
            component="moe_comm_region",
            section="pre_all_to_all_region",
            start_event=region_start,
            end_event=region_end,
        )
        _mark_moe_timing(timing_record, "after_pre_all2all")

        cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)

        if fc1_1_2_weight is not None:
            with moe_timing_context(timing_record, component="expert_compute", section="expert_compute"):
                final_permute_tokens = EPMergedFc1GroupGemm.apply(
                    permute_tokens,
                    cumsum,
                    fc1_1_2_weight,
                    fc2_weight,
                )
        else:
            with moe_timing_context(timing_record, component="expert_compute", section="expert_compute"):
                final_permute_tokens = EPGroupGemm.apply(
                    permute_tokens,
                    cumsum,
                    fc1_1_weight,
                    fc1_2_weight,
                    fc2_weight,
                )
        _mark_moe_timing(timing_record, "after_expert_compute")

        # unpermute with routing_weight
        region_start = _moe_timing_event() if timing_record is not None else None
        with moe_timing_context(timing_record, component="all_to_all", section="post_all_to_all"):
            final_hidden_states = tokens_post_all2all(
                expert_outputs=final_permute_tokens,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                num_experts=num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                routing_map=routing_map,
                local_input_permutation_mapping=local_input_permutation_mapping,
                org_hidden_states_shape=org_hidden_states_shape,
                ep_group=get_parallel_state().ep_group,
            )
        region_end = _moe_timing_event() if timing_record is not None else None
        record_moe_timing_span(
            timing_record,
            direction="forward",
            component="moe_comm_region",
            section="post_all_to_all_region",
            start_event=region_start,
            end_event=region_end,
        )
        _finish_moe_timing(timing_record)
    else:
        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
            final_hidden_states = MergedFc1TritonFusedMoeExpertFunction.apply(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_2_weight,
                fc2_weight,
            )
        else:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
            final_hidden_states = TritonFusedMoeExpertFunction.apply(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )
    return final_hidden_states
