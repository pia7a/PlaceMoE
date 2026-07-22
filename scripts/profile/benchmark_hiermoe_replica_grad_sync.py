#!/usr/bin/env python3
"""Benchmark blocking and layer-packed redundant-expert gradient synchronization."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _event_elapsed_ms(fn) -> tuple[float, float]:
    torch.npu.synchronize()
    start_wall = time.perf_counter()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end), (time.perf_counter() - start_wall) * 1_000.0


def _exchange(send: torch.Tensor, recv: torch.Tensor, peer: int) -> None:
    operations = [
        dist.P2POp(dist.isend, send, peer),
        dist.P2POp(dist.irecv, recv, peer),
    ]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _cross_rank_max(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--local-copy-count", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dist.init_process_group("hccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % 2:
        raise ValueError(f"mirrored R2 requires an even world size, got {world_size}")
    peer = (rank + world_size // 2) % world_size

    gate_elements = 2 * args.hidden_size * args.moe_intermediate_size
    down_elements = args.hidden_size * args.moe_intermediate_size
    sizes = [
        size
        for _ in range(args.local_copy_count)
        for size in (gate_elements, down_elements)
    ]
    gradients = [
        torch.full((size,), float(rank + 1), dtype=torch.bfloat16, device=device)
        for size in sizes
    ]
    old_recv = [torch.empty_like(gradient) for gradient in gradients]
    old_result = [torch.empty_like(gradient) for gradient in gradients]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    packed_send = torch.empty(offsets[-1], dtype=torch.bfloat16, device=device)
    packed_recv = torch.empty_like(packed_send)
    packed_result = torch.empty_like(packed_send)

    def blocking_reference() -> None:
        for gradient, recv, result in zip(gradients, old_recv, old_result, strict=True):
            _exchange(gradient, recv, peer)
            torch.add(gradient, recv, out=result)

    def packed_sync() -> None:
        for gradient, start, end in zip(gradients, offsets[:-1], offsets[1:], strict=True):
            packed_send[start:end].copy_(gradient)
        _exchange(packed_send, packed_recv, peer)
        torch.add(packed_send, packed_recv, out=packed_result)

    for _ in range(args.warmup):
        blocking_reference()
        packed_sync()
    dist.barrier()

    old_accelerator: list[float] = []
    old_external: list[float] = []
    new_accelerator: list[float] = []
    new_external: list[float] = []
    for _ in range(args.iterations):
        accelerator_ms, external_ms = _event_elapsed_ms(blocking_reference)
        old_accelerator.append(accelerator_ms)
        old_external.append(external_ms)
        accelerator_ms, external_ms = _event_elapsed_ms(packed_sync)
        new_accelerator.append(accelerator_ms)
        new_external.append(external_ms)

    old_accelerator = _cross_rank_max(old_accelerator, device)
    old_external = _cross_rank_max(old_external, device)
    new_accelerator = _cross_rank_max(new_accelerator, device)
    new_external = _cross_rank_max(new_external, device)

    expected = float(rank + peer + 2)
    if not torch.all(packed_result == expected).item():
        raise AssertionError("packed gradient sum differs from the blocking reference")
    for result in old_result:
        if not torch.all(result == expected).item():
            raise AssertionError("blocking gradient sum is incorrect")

    bytes_per_buffer = packed_send.numel() * packed_send.element_size()
    report = {
        "world_size": world_size,
        "local_copy_count": args.local_copy_count,
        "dtype": str(packed_send.dtype),
        "bytes_per_rank_per_direction": bytes_per_buffer,
        "old": {
            "waves_per_layer_per_dtype": len(gradients),
            "accelerator_ms": _summarize(old_accelerator),
            "external_ms": _summarize(old_external),
            "peak_communication_staging_bytes": max(
                gradient.numel() * gradient.element_size() for gradient in gradients
            ),
        },
        "packed": {
            "waves_per_layer_per_dtype": 1,
            "accelerator_ms": _summarize(new_accelerator),
            "external_ms": _summarize(new_external),
            "peak_communication_staging_bytes": 2 * bytes_per_buffer,
        },
        "accelerator_speedup_median": statistics.median(old_accelerator)
        / statistics.median(new_accelerator),
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    if rank == 0:
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
