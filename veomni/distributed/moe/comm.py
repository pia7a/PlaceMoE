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
import torch.distributed as dist

from .timing import (
    current_moe_timing_context,
    enter_moe_profile_range,
    exit_moe_profile_range,
    moe_timing_event,
    record_moe_timing_span,
    with_current_full_profile_phase,
)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        timing_meta = current_moe_timing_context()
        ctx._moe_timing_meta = timing_meta

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        section = str(timing_meta.get("section") if timing_meta else "all_to_all")
        annotation = enter_moe_profile_range(
            timing_meta,
            direction="forward",
            component="all_to_all",
            section=section,
        )
        timing_start = moe_timing_event() if timing_meta is not None else None
        try:
            dist.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
        finally:
            timing_end = moe_timing_event() if timing_meta is not None else None
            exit_moe_profile_range(annotation)
        record_moe_timing_span(
            timing_meta,
            direction="forward",
            component="all_to_all",
            section=section,
            start_event=timing_start,
            end_event=timing_end,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        timing_meta = with_current_full_profile_phase(getattr(ctx, "_moe_timing_meta", None), force=True)
        section = str(timing_meta.get("section") if timing_meta else "all_to_all")
        section = f"{section}_backward"
        annotation = enter_moe_profile_range(
            timing_meta,
            direction="backward",
            component="all_to_all",
            section=section,
        )
        timing_start = moe_timing_event() if timing_meta is not None else None
        try:
            grad_input = _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes)
        finally:
            timing_end = moe_timing_event() if timing_meta is not None else None
            exit_moe_profile_range(annotation)
        record_moe_timing_span(
            timing_meta,
            direction="backward",
            component="all_to_all",
            section=section,
            start_event=timing_start,
            end_event=timing_end,
        )
        return (
            None,
            grad_input,
            None,
            None,
        )


class _AllToAll_Async(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        async_handle = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        return output, async_handle

    @staticmethod
    def backward(ctx, *grad_output, grad_async_handle):
        return (
            None,
            _AllToAll_Async.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


def all_to_all_async(group, input, output_split_size, input_split_size):
    return _AllToAll_Async.apply(group, input, output_split_size, input_split_size)


def _empty_a2a_output(input: torch.Tensor, output_split_sizes):
    if output_split_sizes is None:
        return torch.empty_like(input)
    return torch.empty(
        size=(sum(output_split_sizes), *input.shape[1:]),
        dtype=input.dtype,
        device=input.device,
    )


class _AllToAllPair(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group,
        input_a,
        input_b,
        output_split_sizes_a,
        input_split_sizes_a,
        output_split_sizes_b,
        input_split_sizes_b,
    ):
        ctx.group = group
        ctx.output_split_sizes_a = output_split_sizes_a
        ctx.input_split_sizes_a = input_split_sizes_a
        ctx.output_split_sizes_b = output_split_sizes_b
        ctx.input_split_sizes_b = input_split_sizes_b
        timing_meta = current_moe_timing_context()
        ctx._moe_timing_meta = timing_meta

        if group is None or not dist.is_initialized() or dist.get_world_size(group=group) == 1:
            return input_a, input_b

        input_a = input_a.contiguous()
        input_b = input_b.contiguous()
        output_a = _empty_a2a_output(input_a, output_split_sizes_a)
        output_b = _empty_a2a_output(input_b, output_split_sizes_b)
        section = str(timing_meta.get("section") if timing_meta else "all_to_all_pair")
        annotation = enter_moe_profile_range(
            timing_meta,
            direction="forward",
            component="all_to_all",
            section=section,
        )
        timing_start = moe_timing_event() if timing_meta is not None else None
        try:
            work_a = dist.all_to_all_single(
                output_a,
                input_a,
                output_split_sizes=output_split_sizes_a,
                input_split_sizes=input_split_sizes_a,
                group=group,
                async_op=True,
            )
            work_b = dist.all_to_all_single(
                output_b,
                input_b,
                output_split_sizes=output_split_sizes_b,
                input_split_sizes=input_split_sizes_b,
                group=group,
                async_op=True,
            )
            work_a.wait()
            work_b.wait()
        finally:
            timing_end = moe_timing_event() if timing_meta is not None else None
            exit_moe_profile_range(annotation)
        record_moe_timing_span(
            timing_meta,
            direction="forward",
            component="all_to_all",
            section=section,
            start_event=timing_start,
            end_event=timing_end,
        )
        return output_a, output_b

    @staticmethod
    def backward(ctx, grad_output_a, grad_output_b):
        timing_meta = with_current_full_profile_phase(getattr(ctx, "_moe_timing_meta", None), force=True)
        section = str(timing_meta.get("section") if timing_meta else "all_to_all_pair")
        section = f"{section}_backward"
        if ctx.group is None or not dist.is_initialized() or dist.get_world_size(group=ctx.group) == 1:
            grad_input_a = grad_output_a
            grad_input_b = grad_output_b
        else:
            annotation = enter_moe_profile_range(
                timing_meta,
                direction="backward",
                component="all_to_all",
                section=section,
            )
            timing_start = moe_timing_event() if timing_meta is not None else None
            grad_input_a = None
            grad_input_b = None
            work_a = None
            work_b = None
            try:
                if grad_output_a is not None:
                    contiguous_grad_a = grad_output_a.contiguous()
                    grad_input_a = _empty_a2a_output(contiguous_grad_a, ctx.input_split_sizes_a)
                    work_a = dist.all_to_all_single(
                        grad_input_a,
                        contiguous_grad_a,
                        output_split_sizes=ctx.input_split_sizes_a,
                        input_split_sizes=ctx.output_split_sizes_a,
                        group=ctx.group,
                        async_op=True,
                    )
                if grad_output_b is not None:
                    contiguous_grad_b = grad_output_b.contiguous()
                    grad_input_b = _empty_a2a_output(contiguous_grad_b, ctx.input_split_sizes_b)
                    work_b = dist.all_to_all_single(
                        grad_input_b,
                        contiguous_grad_b,
                        output_split_sizes=ctx.input_split_sizes_b,
                        input_split_sizes=ctx.output_split_sizes_b,
                        group=ctx.group,
                        async_op=True,
                    )
                if work_a is not None:
                    work_a.wait()
                if work_b is not None:
                    work_b.wait()
            finally:
                timing_end = moe_timing_event() if timing_meta is not None else None
                exit_moe_profile_range(annotation)
            record_moe_timing_span(
                timing_meta,
                direction="backward",
                component="all_to_all",
                section=section,
                start_event=timing_start,
                end_event=timing_end,
            )
        return (
            None,
            grad_input_a,
            grad_input_b,
            None,
            None,
            None,
            None,
        )


def all_to_all_pair(
    group,
    input_a,
    input_b,
    output_split_size_a,
    input_split_size_a,
    output_split_size_b,
    input_split_size_b,
):
    return _AllToAllPair.apply(
        group,
        input_a,
        input_b,
        output_split_size_a,
        input_split_size_a,
        output_split_size_b,
        input_split_size_b,
    )
