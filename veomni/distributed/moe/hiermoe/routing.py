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


def duplicate_free_counts_by_expert_group(
    selected_experts: torch.Tensor,
    num_experts: int,
    group_size: int,
) -> torch.Tensor:
    """Count tokens hitting each expert group after OR dedup over top-k choices."""

    if num_experts % group_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by group_size={group_size}.")
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)

    num_groups = num_experts // group_size
    target_groups = torch.div(selected_experts.to(torch.long), group_size, rounding_mode="floor")
    token_group_hits = torch.zeros(
        (*target_groups.shape[:-1], num_groups),
        dtype=torch.int32,
        device=selected_experts.device,
    )
    token_group_hits.scatter_(dim=-1, index=target_groups, value=1)
    return token_group_hits.sum(dim=-2)


def expert_assignment_counts(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
