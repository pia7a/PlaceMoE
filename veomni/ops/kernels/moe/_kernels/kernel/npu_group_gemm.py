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

import torch
import torch_npu

from veomni.distributed.moe.timing import (
    current_moe_timing_context,
    enter_moe_profile_range,
    exit_moe_profile_range,
    moe_timing_event,
    record_moe_timing_span,
    with_current_full_profile_phase,
)


class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx._moe_timing_meta = current_moe_timing_context()

        fwd_output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list
        timing_meta = with_current_full_profile_phase(getattr(ctx, "_moe_timing_meta", None), force=True)
        annotation = enter_moe_profile_range(
            timing_meta,
            direction="backward",
            component="expert_compute",
            section="npu_group_gemm_backward",
        )
        timing_start = moe_timing_event() if timing_meta is not None else None

        try:
            weight = torch.transpose(weight, 1, 2)
            grad_input = torch_npu.npu_grouped_matmul(
                [grad_output],
                [weight],
                bias=None,
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]

            grad_weight = torch_npu.npu_grouped_matmul(
                [input_tensor.T],
                [grad_output],
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]
        finally:
            timing_end = moe_timing_event() if timing_meta is not None else None
            exit_moe_profile_range(annotation)
        record_moe_timing_span(
            timing_meta,
            direction="backward",
            component="expert_compute",
            section="npu_group_gemm_backward",
            start_event=timing_start,
            end_event=timing_end,
        )

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    output = GmmFunction.apply(x, weight, group_list)
    return output
