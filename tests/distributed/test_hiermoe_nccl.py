# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe.all_to_all import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.state import configure_hiermoe


_PROFILE_PATH = str(Path(__file__).with_name("fixtures") / "hiermoe_profile.json")


def _eager_linear_moe(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for token_index in range(hidden_states.shape[0]):
        for slot_index in range(selected_experts.shape[1]):
            expert_index = int(selected_experts[token_index, slot_index])
            output[token_index] += (hidden_states[token_index] @ expert_weight[expert_index]) * routing_weights[
                token_index, slot_index
            ]
    return output


def _apply_local_linear_experts(
    permuted_tokens: torch.Tensor,
    tokens_per_local_expert: torch.Tensor,
    local_weight: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    offset = 0
    for local_expert, count_tensor in enumerate(tokens_per_local_expert):
        count = int(count_tensor)
        chunk = permuted_tokens[offset : offset + count]
        chunks.append(chunk @ local_weight[local_expert])
        offset += count
    return torch.cat(chunks, dim=0) if chunks else torch.empty_like(permuted_tokens)


def _rank_dedup_nccl_worker(communication_mode: str, hierarchy_group_sizes: list[int] | None = None) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    configure_hiermoe(
        HierMoEConfig(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            communication_mode=communication_mode,
            hierarchy_group_sizes=hierarchy_group_sizes or [2, world_size],
            perf_model_path=_PROFILE_PATH,
        ),
        dist.group.WORLD,
    )

    hidden_size = 16
    dtype = torch.bfloat16 if world_size == 4 else torch.float32
    if world_size == 4:
        num_experts = 8
        routing_patterns = [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[1, 0], [3, 2], [5, 4], [7, 6]],
            [[0, 2], [1, 3], [4, 6], [5, 7]],
            [[2, 0], [3, 1], [6, 4], [7, 5]],
        ]
        selected_experts = torch.tensor(routing_patterns[rank], dtype=torch.long, device=device)
    elif world_size == 8:
        num_experts = 16
        base_routes = torch.tensor(
            [
                [0, 2, 8, 10],
                [1, 3, 9, 11],
                [4, 6, 12, 14],
                [5, 7, 13, 15],
                [0, 5, 10, 15],
            ],
            dtype=torch.long,
            device=device,
        )
        selected_experts = (base_routes + rank) % num_experts
    else:
        raise RuntimeError(f"unsupported NCCL test world size: {world_size}")
    num_tokens, top_k = selected_experts.shape
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    hidden = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device, generator=generator)
    hidden.requires_grad_(True)
    routing_logits = torch.randn((num_tokens, top_k), dtype=dtype, device=device, generator=generator)
    routing_logits.requires_grad_(True)
    routing_weights = torch.softmax(routing_logits, dim=-1)
    weight_generator = torch.Generator(device=device).manual_seed(17)
    full_weight = torch.randn(
        (num_experts, hidden_size, hidden_size),
        dtype=dtype,
        device=device,
        generator=weight_generator,
    )

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    baseline_logits = routing_logits.detach().clone().requires_grad_(True)
    baseline_weights = torch.softmax(baseline_logits, dim=-1)
    baseline_expert_weight = full_weight.detach().clone().requires_grad_(True)
    baseline_output = _eager_linear_moe(
        baseline_hidden,
        selected_experts,
        baseline_weights,
        baseline_expert_weight,
    )
    baseline_output.float().square().sum().backward()
    dist.all_reduce(baseline_expert_weight.grad)

    local_experts = num_experts // world_size
    local_start = rank * local_experts
    local_weight = full_weight[local_start : local_start + local_experts].detach().clone().requires_grad_(True)
    permuted_tokens, context, tokens_per_local_expert = rank_dedup_dispatch(
        hidden,
        selected_experts,
        routing_weights,
        num_experts,
        dist.group.WORLD,
    )
    expert_outputs = _apply_local_linear_experts(permuted_tokens, tokens_per_local_expert, local_weight)
    output = rank_dedup_combine(expert_outputs, context)
    output.float().square().sum().backward()

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    weight_tolerance = 8e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(output, baseline_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(routing_logits.grad, baseline_logits.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        local_weight.grad,
        baseline_expert_weight.grad[local_start : local_start + local_experts],
        atol=weight_tolerance,
        rtol=weight_tolerance,
    )
    if communication_mode == "direct":
        expected_context_mode = "rank"
    elif hierarchy_group_sizes is not None and len(hierarchy_group_sizes) == 3:
        expected_context_mode = "hierarchical3d"
    else:
        expected_context_mode = "hierarchical"
    assert context.mode == expected_context_mode


@pytest.mark.parametrize("communication_mode", ["direct", "hierarchical"])
def test_rank_dedup_dispatch_combine_nccl_forward_backward(communication_mode: str) -> None:
    torchrun(_rank_dedup_nccl_worker, 4, communication_mode)


def test_rank_dedup_3d_dispatch_combine_nccl_forward_backward() -> None:
    torchrun(_rank_dedup_nccl_worker, 8, "hierarchical", [2, 4, 8])
