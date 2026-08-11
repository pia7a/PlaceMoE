#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Compare two- and three-level HierMoE communication on the EP32 GPU cluster."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from scripts.placemoe.reproduction.gpu_ep32.calibrate_communication import _cluster_scope
from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.state import configure_hiermoe


_HIERARCHIES = {
    "two_level": [8, 32],
    "three_level_nvlink": [2, 8, 32],
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=(256, 1024, 4096))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _routes(
    *,
    rank: int,
    tokens: int,
    top_k: int,
    num_experts: int,
    pattern: str,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(tokens, dtype=torch.long, device=device).view(-1, 1)
    offsets = torch.arange(top_k, dtype=torch.long, device=device).view(1, -1)
    if pattern == "uniform":
        destination_ranks = torch.remainder(rows * top_k + offsets + rank, 32)
    elif pattern == "skew":
        destination_ranks = offsets.expand(tokens, -1)
    else:
        raise ValueError(pattern)
    experts_per_rank = num_experts // 32
    local_experts = torch.remainder(rows + 3 * offsets, experts_per_rank)
    selected = destination_ranks * experts_per_rank + local_experts
    ordered = selected.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
        raise RuntimeError(f"{pattern} routes contain duplicate experts")
    return selected.contiguous()


def _configure(name: str) -> None:
    configure_hiermoe(
        HierMoEConfig(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            communication_mode="hierarchical",
            hierarchy_group_sizes=_HIERARCHIES[name],
        ),
        dist.group.WORLD,
    )


def _run_once(
    name: str,
    *,
    hidden: torch.Tensor,
    selected: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> tuple[float, int]:
    _configure(name)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(hidden.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    permuted, context, _ = rank_dedup_dispatch(
        hidden,
        selected,
        weights,
        num_experts,
        dist.group.WORLD,
    )
    output = rank_dedup_combine(permuted, context)
    end.record()
    torch.cuda.synchronize(hidden.device)
    expected_mode = "hierarchical" if name == "two_level" else "hierarchical3d"
    if context.mode != expected_mode:
        raise RuntimeError(f"{name} selected {context.mode!r}, expected {expected_mode!r}")
    torch.testing.assert_close(output, hidden, atol=2.0e-2, rtol=2.0e-2)
    measured = torch.tensor(
        [float(start.elapsed_time(end)), float(torch.cuda.max_memory_allocated(hidden.device))],
        dtype=torch.float64,
        device=hidden.device,
    )
    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
    return float(measured[0].item()), int(measured[1].item())


def _summary(samples: list[dict[str, Any]], name: str) -> dict[str, Any]:
    wall = [float(row["wall_ms"]) for row in samples if row["hierarchy"] == name]
    memory = [int(row["peak_memory_bytes"]) for row in samples if row["hierarchy"] == name]
    ordered = sorted(wall)
    return {
        "count": len(wall),
        "mean_wall_ms": statistics.mean(wall),
        "median_wall_ms": statistics.median(wall),
        "min_wall_ms": min(wall),
        "max_wall_ms": max(wall),
        "p95_wall_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "median_peak_memory_bytes": int(statistics.median(memory)),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 32 or args.num_experts % 32:
            raise RuntimeError("the hierarchy benchmark requires EP32 and evenly sharded experts")
        if args.iterations < 2:
            raise ValueError("--iterations must be at least two for alternating order")
        device = torch.device("cuda", local_rank)
        device_orders: list[str | None] = [None] * 32
        dist.all_gather_object(device_orders, os.environ.get("CUDA_VISIBLE_DEVICES"))
        cells = []
        speedups = []
        for tokens in args.tokens:
            for pattern in ("uniform", "skew"):
                selected = _routes(
                    rank=rank,
                    tokens=tokens,
                    top_k=args.top_k,
                    num_experts=args.num_experts,
                    pattern=pattern,
                    device=device,
                )
                generator = torch.Generator(device=device).manual_seed(7319 + rank + tokens)
                hidden = torch.randn(
                    (tokens, args.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                weights = torch.full(
                    (tokens, args.top_k),
                    1.0 / args.top_k,
                    dtype=torch.float32,
                    device=device,
                )
                for name in _HIERARCHIES:
                    for _ in range(args.warmup):
                        _run_once(
                            name,
                            hidden=hidden,
                            selected=selected,
                            weights=weights,
                            num_experts=args.num_experts,
                        )
                samples = []
                for iteration in range(args.iterations):
                    order = list(_HIERARCHIES)
                    if iteration % 2:
                        order.reverse()
                    for order_index, name in enumerate(order):
                        wall_ms, peak_memory = _run_once(
                            name,
                            hidden=hidden,
                            selected=selected,
                            weights=weights,
                            num_experts=args.num_experts,
                        )
                        samples.append(
                            {
                                "iteration": iteration,
                                "order_index": order_index,
                                "hierarchy": name,
                                "wall_ms": wall_ms,
                                "peak_memory_bytes": peak_memory,
                            }
                        )
                summaries = {name: _summary(samples, name) for name in _HIERARCHIES}
                speedup = summaries["two_level"]["median_wall_ms"] / summaries["three_level_nvlink"]["median_wall_ms"]
                speedups.append(speedup)
                cells.append(
                    {
                        "tokens_per_rank": tokens,
                        "pattern": pattern,
                        "samples": samples,
                        "summary": summaries,
                        "two_level_over_three_level_speedup": speedup,
                    }
                )
        if rank == 0:
            geometric_mean = math.exp(statistics.mean(math.log(value) for value in speedups))
            minimum = min(speedups)
            payload = {
                "schema_version": 1,
                "source": "gpu32-a6000-ep32-hierarchy-benchmark",
                "run_name": args.run_name,
                "scope": _cluster_scope(args),
                "topology": {
                    "accelerator": "NVIDIA RTX A6000",
                    "nodes": 4,
                    "gpus_per_node": 8,
                    "ep_size": 32,
                    "hidden_size": args.hidden_size,
                    "num_experts": args.num_experts,
                    "top_k": args.top_k,
                    "hierarchies": _HIERARCHIES,
                    "cuda_visible_devices_by_node": {str(node): device_orders[node * 8] for node in range(4)},
                },
                "protocol": {
                    "dtype": "bfloat16",
                    "operation": "rank_dedup_dispatch_plus_combine_identity_experts",
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "execution_order": "repeat-major alternating",
                    "correctness": "output_equals_input_atol_rtol_2e-2",
                },
                "cells": cells,
                "aggregate": {
                    "geometric_mean_two_level_over_three_level_speedup": geometric_mean,
                    "minimum_two_level_over_three_level_speedup": minimum,
                },
                "performance_gate": {
                    "geometric_mean_minimum": 1.05,
                    "per_cell_minimum": 0.95,
                    "candidate_passes": geometric_mean >= 1.05 and minimum >= 0.95,
                    "note": "Passing makes three-level communication a candidate; fixed-pipeline integration still requires separate validation.",
                },
            }
            _atomic_json(args.output, payload)
            print(json.dumps({"output": str(args.output), **payload["aggregate"], **payload["performance_gate"]}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
