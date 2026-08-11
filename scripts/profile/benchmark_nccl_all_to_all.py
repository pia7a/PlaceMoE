#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Measure critical-rank NCCL all-to-all latency and logical payload rate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-rank", default="4096,16384")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _one_repeat(input_tensor: torch.Tensor, output_tensor: torch.Tensor, iterations: int) -> list[float]:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dist.all_to_all_single(output_tensor, input_tensor)
    end.record()
    end.synchronize()
    local_ms = torch.tensor([start.elapsed_time(end) / iterations], dtype=torch.float64, device=input_tensor.device)
    rank_ms = [torch.empty_like(local_ms) for _ in range(dist.get_world_size())]
    dist.all_gather(rank_ms, local_ms)
    return [float(value.item()) for value in rank_ms]


def _benchmark(rows_per_rank: int, hidden_size: int, warmup: int, iterations: int, repeats: int) -> dict:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rows_per_rank % world_size:
        raise ValueError("rows-per-rank must be divisible by world size")
    rows_per_peer = rows_per_rank // world_size
    blocks = [
        torch.full(
            (rows_per_peer, hidden_size),
            rank * world_size + destination,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for destination in range(world_size)
    ]
    input_tensor = torch.cat(blocks, dim=0)
    output_tensor = torch.empty_like(input_tensor)
    for _ in range(warmup):
        dist.all_to_all_single(output_tensor, input_tensor)
    torch.cuda.synchronize()

    for source in range(world_size):
        block = output_tensor[source * rows_per_peer : (source + 1) * rows_per_peer]
        expected = source * world_size + rank
        if not torch.all(block == expected):
            raise AssertionError(f"all-to-all correctness failed for source={source}, rank={rank}")

    critical_samples = []
    per_repeat_rank_ms = []
    for _ in range(repeats):
        rank_ms = _one_repeat(input_tensor, output_tensor, iterations)
        per_repeat_rank_ms.append(rank_ms)
        critical_samples.append(max(rank_ms))
    payload_bytes = input_tensor.numel() * input_tensor.element_size()
    median_ms = statistics.median(critical_samples)
    return {
        "rows_per_rank": rows_per_rank,
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
        "payload_bytes_per_rank": payload_bytes,
        "critical_rank_median_ms": median_ms,
        "critical_rank_min_ms": min(critical_samples),
        "critical_rank_samples_ms": critical_samples,
        "rank_samples_ms": per_repeat_rank_ms,
        "logical_payload_gbps_per_rank": payload_bytes / median_ms / 1_000_000.0,
    }


def main() -> None:
    args = _args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        rows = [int(value) for value in args.rows_per_rank.split(",")]
        results = [_benchmark(value, args.hidden_size, args.warmup, args.iterations, args.repeats) for value in rows]
        if dist.get_rank() == 0:
            payload = {
                "schema_version": 1,
                "status": "accepted",
                "backend": "nccl",
                "world_size": dist.get_world_size(),
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "results": results,
            }
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            print(rendered)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
