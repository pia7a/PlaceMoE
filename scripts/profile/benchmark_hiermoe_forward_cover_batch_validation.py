# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Compare per-layer and batched exact patch validation on saved routes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from veomni.distributed.moe.hiermoe.forward_cover_planner import (
    _local_assignment_counts,
    _local_packed_counts,
    forward_cover_local_validation_stats,
    forward_cover_patch_validation_stats_batched,
    propose_forward_reuse_cover,
)
from veomni.distributed.moe.hiermoe.planner import PlacementAction, assign_tokens_to_mirrored_r2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--route-rank", type=int, default=0)
    parser.add_argument("--source-rank", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _fixed_r2_layout(
    num_experts: int,
    ep_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    slots_per_rank = 2 * num_experts // ep_size
    half_ep = ep_size // 2
    logical = torch.arange(num_experts, dtype=torch.long, device=device)
    rank_in_half = torch.div(logical, slots_per_rank, rounding_mode="floor")
    local_slot = torch.remainder(logical, slots_per_rank)
    owners = rank_in_half * slots_per_rank + local_slot
    copies = (rank_in_half + half_ep) * slots_per_rank + local_slot
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout[owners] = logical
    layout[copies] = logical
    return layout, owners, slots_per_rank


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    records = [
        torch.load(
            args.route_dir / f"layer{layer:02d}_rank{args.route_rank:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for layer in range(args.layers)
    ]
    num_experts = int(records[0]["num_experts"])
    layout, owners, slots_per_rank = _fixed_r2_layout(num_experts, args.ep_size, device)
    selected_layers = [record["routes"].to(device=device, dtype=torch.long) for record in records]
    physical_layers = [
        assign_tokens_to_mirrored_r2(
            selected,
            torch.stack((owners, owners + num_experts), dim=1),
            source_ranks=args.source_rank,
            num_ranks=args.ep_size,
        )
        for selected in selected_layers
    ]
    actions: list[PlacementAction] = []
    fallbacks: list[int] = []
    baseline_communication: list[torch.Tensor] = []
    baseline_assignments: list[torch.Tensor] = []
    for selected, physical in zip(selected_layers, physical_layers, strict=True):
        local_mask = torch.div(physical, slots_per_rank, rounding_mode="floor") == int(args.source_rank)
        local_counts = torch.zeros((slots_per_rank,), dtype=torch.float32, device=device)
        local_counts.scatter_add_(
            0,
            torch.remainder(physical[local_mask], slots_per_rank),
            torch.ones_like(physical[local_mask], dtype=torch.float32),
        )
        proposal = propose_forward_reuse_cover(
            selected_experts=selected,
            physical_routes=physical,
            slot_to_logical=layout,
            owner_slots=owners,
            local_slot_assignments=local_counts,
            source_rank=args.source_rank,
            slots_per_rank=slots_per_rank,
            hierarchy_group_sizes=(args.group_size,),
            num_experts=num_experts,
            max_copies=4,
            level_weights=(1.0, 4.0),
            compute_weight=0.0,
        )
        if proposal.action is None:
            raise RuntimeError("Saved route did not produce a Cover proposal.")
        actions.append(proposal.action)
        fallbacks.append(int(owners[int(proposal.action.dst_logical)].item()))
        baseline_communication.append(
            _local_packed_counts(
                physical,
                slots_per_rank=slots_per_rank,
                ep_size=args.ep_size,
                hierarchy_group_sizes=(args.group_size,),
            )
        )
        baseline_assignments.append(
            _local_assignment_counts(
                physical,
                slots_per_rank=slots_per_rank,
                ep_size=args.ep_size,
            )
        )

    def individual() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        results = [
            forward_cover_local_validation_stats(
                selected_experts=selected,
                physical_routes=physical,
                slot_to_logical=layout,
                action=action,
                source_rank=args.source_rank,
                slots_per_rank=slots_per_rank,
                hierarchy_group_sizes=(args.group_size,),
                num_experts=num_experts,
                max_copies=4,
                step=0,
                layer_seed=layer,
                patch_remap=True,
                victim_fallback_slot=fallback,
                baseline_communication_counts=communication,
                baseline_assignment_counts=assignments,
            )
            for layer, (selected, physical, action, fallback, communication, assignments) in enumerate(
                zip(
                    selected_layers,
                    physical_layers,
                    actions,
                    fallbacks,
                    baseline_communication,
                    baseline_assignments,
                    strict=True,
                )
            )
        ]
        return (
            torch.stack([result.communication_count_delta for result in results]),
            torch.stack([result.assignment_count_delta for result in results]),
            torch.tensor([result.affected_tokens for result in results], device=device),
        )

    def batched() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = forward_cover_patch_validation_stats_batched(
            selected_experts=torch.stack(selected_layers),
            physical_routes=torch.stack(physical_layers),
            source_logical=torch.tensor(
                [action.src_logical for action in actions],
                dtype=torch.long,
                device=device,
            ),
            victim_logical=torch.tensor(
                [action.dst_logical for action in actions],
                dtype=torch.long,
                device=device,
            ),
            destination_slots=torch.tensor(
                [action.dst_slot for action in actions],
                dtype=torch.long,
                device=device,
            ),
            victim_fallback_slots=torch.tensor(fallbacks, dtype=torch.long, device=device),
            source_rank=args.source_rank,
            slots_per_rank=slots_per_rank,
            ep_size=args.ep_size,
            hierarchy_group_sizes=(args.group_size,),
        )
        return (
            result.communication_count_delta,
            result.assignment_count_delta,
            result.affected_tokens,
        )

    expected = individual()
    actual = batched()
    for expected_value, actual_value in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_value, expected_value)

    for _ in range(args.warmup):
        individual()
        batched()
    torch.npu.synchronize()
    individual_samples = []
    batched_samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        individual()
        torch.npu.synchronize()
        individual_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        batched()
        torch.npu.synchronize()
        batched_samples.append((time.perf_counter() - started) * 1000.0)

    payload = {
        "route_dir": str(args.route_dir),
        "layers": args.layers,
        "tokens_per_layer": int(selected_layers[0].shape[0]),
        "top_k": int(selected_layers[0].shape[1]),
        "affected_tokens": int(actual[2].sum().item()),
        "individual_median_ms": statistics.median(individual_samples),
        "batched_median_ms": statistics.median(batched_samples),
        "speedup": statistics.median(individual_samples) / statistics.median(batched_samples),
        "individual_samples_ms": individual_samples,
        "batched_samples_ms": batched_samples,
        "exact_match": True,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
