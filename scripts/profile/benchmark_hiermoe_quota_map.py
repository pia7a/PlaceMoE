# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark the exact CoRe-MoE quota-map primitive on a saved route snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from veomni.distributed.moe.hiermoe.core_planner import compress_local_route_payload
from veomni.distributed.moe.hiermoe.oracle import load_route_snapshot
from veomni.ops.platform.npu.hiermoe_planner_ops import get_hiermoe_planner_npu_ops


def _measure(function: Callable[[], object], *, warmup: int, iterations: int) -> dict[str, float]:
    event_values: list[float] = []
    external_values: list[float] = []
    for iteration in range(warmup + iterations):
        started = torch.npu.Event(enable_timing=True)
        finished = torch.npu.Event(enable_timing=True)
        external_started = time.perf_counter()
        started.record()
        output = function()
        finished.record()
        torch.npu.synchronize()
        if iteration >= warmup:
            event_values.append(float(started.elapsed_time(finished)))
            external_values.append((time.perf_counter() - external_started) * 1000.0)
        del output
    event_ordered = sorted(event_values)
    external_ordered = sorted(external_values)
    p90_index = min(len(event_ordered) - 1, max(0, math.ceil(0.9 * len(event_ordered)) - 1))
    return {
        "event_median_ms": statistics.median(event_ordered),
        "event_p90_ms": event_ordered[p90_index],
        "event_max_ms": event_ordered[-1],
        "external_median_ms": statistics.median(external_ordered),
        "external_p90_ms": external_ordered[p90_index],
        "external_max_ms": external_ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-copies", type=int, default=8)
    parser.add_argument("--custom-opp", type=Path)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()

    if not 1 <= args.max_copies <= 8:
        raise ValueError("max-copies must be between one and eight.")
    if args.custom_opp is not None:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.custom_opp.resolve())
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    snapshot = load_route_snapshot(args.snapshot)
    routes = snapshot.routes_by_rank[args.rank].to("npu:0")
    payloads = torch.stack(
        [
            compress_local_route_payload(
                rank_routes,
                num_experts=snapshot.num_experts,
                sample_size=args.sample_size,
                source_rank=source_rank,
                step=snapshot.step,
                layer_seed=24,
            )
            for source_rank, rank_routes in enumerate(snapshot.routes_by_rank)
        ],
        dim=0,
    )
    ordinal_offset = 2 + snapshot.num_experts
    sample_ordinals = payloads[:, ordinal_offset : ordinal_offset + args.sample_size].reshape(-1).to("npu:0")
    offset = ordinal_offset + 2 * args.sample_size
    sample = payloads[:, offset:].view(snapshot.ep_size, args.sample_size, -1).reshape(-1, routes.shape[1])
    sample = sample.to("npu:0")
    equal_routes = sample.unsqueeze(-1).eq(sample.unsqueeze(-2))
    route_positions = torch.arange(sample.shape[1], dtype=torch.long, device="npu:0")
    prior_positions = route_positions.view(1, -1, 1) > route_positions.view(1, 1, -1)
    sample_multiplicity = equal_routes.sum(dim=-1).to(torch.long) * (~(equal_routes & prior_positions).any(dim=-1))

    slots_per_rank = snapshot.num_experts // snapshot.ep_size + 1
    owner_slots = torch.arange(snapshot.num_experts, device="npu:0")
    owner_slots = torch.div(
        owner_slots, snapshot.num_experts // snapshot.ep_size, rounding_mode="floor"
    ) * slots_per_rank + torch.remainder(owner_slots, snapshot.num_experts // snapshot.ep_size)
    layout = torch.full((snapshot.ep_size * slots_per_rank,), -1, dtype=torch.long, device="npu:0")
    layout.scatter_(0, owner_slots, torch.arange(snapshot.num_experts, device="npu:0"))
    candidate = layout.clone()
    candidate[2 * slots_per_rank - 1] = 0
    layouts = torch.stack((layout, candidate), dim=0)
    logical_ids = torch.arange(snapshot.num_experts, device="npu:0").view(1, 1, -1)
    slot_ids = torch.arange(layouts.shape[1], device="npu:0").view(1, -1, 1)
    matches = layouts.unsqueeze(-1) == logical_ids
    copy_counts = matches.sum(dim=1)
    copy_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, layouts.shape[1]))
    copy_slots = copy_slots.sort(dim=1).values[:, : args.max_copies].transpose(1, 2).contiguous()
    copy_indices = torch.arange(args.max_copies, dtype=torch.long, device="npu:0").view(1, 1, -1)
    copy_slots = torch.where(copy_indices < copy_counts.unsqueeze(-1), copy_slots, -torch.ones_like(copy_slots))
    owner_ranks = torch.div(owner_slots, slots_per_rank, rounding_mode="floor").view(1, -1).expand(2, -1)
    owner_ranks = owner_ranks.contiguous()

    if args.extension is None:
        extension = get_hiermoe_planner_npu_ops()
    else:
        extension_path = args.extension.resolve()
        spec = importlib.util.spec_from_file_location("_hiermoe_npu_ops", extension_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load the NPU extension from {extension_path}.")
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    if extension is None or not hasattr(extension, "quota_policy") or not hasattr(extension, "quota_map"):
        raise RuntimeError("The CoRe-MoE quota-policy and quota-map NPU ops must be built.")

    assignment_offset = 2
    assignment_counts = payloads[:, assignment_offset : assignment_offset + snapshot.num_experts]
    assignment_counts = assignment_counts.to(device="npu:0", dtype=torch.long)
    sample_sources = torch.arange(snapshot.ep_size, device="npu:0").repeat_interleave(args.sample_size)
    policy_levels = tuple(snapshot.hierarchy.group_sizes[:-1])
    padded_levels = (*policy_levels, 1, 1)[:2]
    quota_weights, quota_configured, _rows, _row_counts, _digest = extension.quota_policy(
        sample.contiguous(),
        sample_multiplicity.contiguous(),
        sample_sources.contiguous(),
        sample_ordinals.contiguous(),
        assignment_counts.contiguous(),
        layouts,
        owner_slots.view(1, -1).expand(2, -1).contiguous(),
        slots_per_rank,
        args.rank,
        snapshot.ep_size,
        args.max_copies,
        args.sample_size,
        len(policy_levels),
        padded_levels[0],
        padded_levels[1],
    )
    token_ordinals = torch.arange(routes.shape[0], dtype=torch.long, device="npu:0")

    def quota_map():
        return extension.quota_map(
            routes,
            copy_slots,
            copy_counts,
            owner_ranks,
            quota_weights,
            quota_configured,
            token_ordinals,
            slots_per_rank,
            args.rank,
            snapshot.ep_size,
            len(policy_levels),
            padded_levels[0],
            padded_levels[1],
            snapshot.step,
            24,
        )

    result = _measure(quota_map, warmup=args.warmup, iterations=args.iterations)
    print(
        f"tokens={routes.shape[0]} top_k={routes.shape[1]} copies={args.max_copies} "
        f"event_median={result['event_median_ms']:.3f}ms event_p90={result['event_p90_ms']:.3f}ms "
        f"event_max={result['event_max_ms']:.3f}ms "
        f"external_median={result['external_median_ms']:.3f}ms "
        f"external_p90={result['external_p90_ms']:.3f}ms "
        f"external_max={result['external_max_ms']:.3f}ms"
    )


if __name__ == "__main__":
    main()
