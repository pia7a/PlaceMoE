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

"""Memory-bounded FP32 segmented reduction for CUDA HierMoE combine."""

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_sum_dim0_kernel(
    source,
    order,
    offsets,
    output,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    column_mask = columns < hidden_size
    position = tl.load(offsets + row)
    end = tl.load(offsets + row + 1)
    accumulator = tl.zeros((block_size,), tl.float32)
    while position < end:
        source_row = tl.load(order + position)
        accumulator += tl.load(
            source + source_row * hidden_size + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        position += 1
    tl.store(
        output + row * hidden_size + columns,
        accumulator,
        mask=column_mask,
    )


def segment_sum_dim0_fp32_to_source_dtype(
    source: torch.Tensor,
    index: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Sum source rows in FP32 and write results directly in source dtype."""

    if source.device.type != "cuda":
        raise ValueError("Triton segmented reduction requires a CUDA tensor.")
    if source.ndim != 2 or index.ndim != 1:
        raise ValueError("source must be 2D and index must be 1D.")
    if int(source.shape[0]) != int(index.numel()):
        raise ValueError("index length must match source rows.")
    if source.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"unsupported source dtype for CUDA segmented reduction: {source.dtype}.")

    num_rows = int(num_rows)
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}.")
    hidden_size = int(source.shape[1])
    output = torch.empty(
        (num_rows, hidden_size),
        dtype=source.dtype,
        device=source.device,
    )
    if num_rows == 0:
        if source.shape[0] != 0:
            raise ValueError("non-empty source cannot be reduced into zero rows.")
        return output
    if hidden_size == 0:
        return output

    source = source.contiguous()
    index_long = index.to(device=source.device, dtype=torch.long).contiguous()
    _sorted_index, order = torch.sort(index_long, stable=True)
    counts = torch.bincount(index_long, minlength=num_rows)
    if int(counts.shape[0]) != num_rows:
        raise ValueError("index contains a row outside [0, num_rows).")
    offsets = torch.empty((num_rows + 1,), dtype=torch.long, device=source.device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    block_size = min(1024, triton.next_power_of_2(hidden_size))
    _segment_sum_dim0_kernel[(num_rows, triton.cdiv(hidden_size, block_size))](
        source,
        order,
        offsets,
        output,
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=8 if block_size >= 256 else 4,
    )
    return output


__all__ = ["segment_sum_dim0_fp32_to_source_dtype"]
