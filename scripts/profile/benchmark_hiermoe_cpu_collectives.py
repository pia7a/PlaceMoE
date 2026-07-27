# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark CPU/Gloo collectives for exact HierMoE planner statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--candidates-per-layer", type=int, default=22_784)
    parser.add_argument("--statistic-width", type=int, default=68)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--collectives",
        nargs="+",
        choices=("all-to-all", "reduce-scatter"),
        default=("all-to-all", "reduce-scatter"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _rank_times(local_ms: float, world_size: int) -> list[float]:
    local = torch.tensor([local_ms], dtype=torch.float64)
    gathered = torch.empty((world_size,), dtype=torch.float64)
    dist.all_gather_into_tensor(gathered, local)
    return gathered.tolist()


def _summarize(samples: list[list[float]]) -> dict[str, object]:
    maxima = [max(sample) for sample in samples]
    medians = [statistics.median(sample) for sample in samples]
    minima = [min(sample) for sample in samples]
    return {
        "max_rank_ms": {
            "median": statistics.median(maxima),
            "p90": _percentile(maxima, 0.9),
            "samples": maxima,
        },
        "median_rank_ms": {
            "median": statistics.median(medians),
            "p90": _percentile(medians, 0.9),
            "samples": medians,
        },
        "min_rank_ms": {
            "median": statistics.median(minima),
            "p90": _percentile(minima, 0.9),
            "samples": minima,
        },
    }


def _balanced_layer_owners(layer_count: int, world_size: int) -> list[int]:
    return [min(world_size - 1, layer * world_size // layer_count) for layer in range(layer_count)]


def _benchmark_all_to_all(
    *,
    rank: int,
    world_size: int,
    layers: int,
    candidates_per_layer: int,
    width: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    rows_per_layer = candidates_per_layer + 1
    elements_per_layer = rows_per_layer * width
    owner_ranks = _balanced_layer_owners(layers, world_size)
    layers_by_owner = [
        [layer for layer, owner in enumerate(owner_ranks) if owner == destination] for destination in range(world_size)
    ]
    input_splits = [len(indices) * elements_per_layer for indices in layers_by_owner]
    own_elements = input_splits[rank]
    output_splits = [own_elements] * world_size
    send = torch.full((sum(input_splits),), float(rank + 1), dtype=torch.float32)
    receive = torch.empty((world_size * own_elements,), dtype=torch.float32)
    expected = world_size * (world_size + 1) / 2

    collective_samples: list[list[float]] = []
    reduction_samples: list[list[float]] = []
    for iteration in range(warmup + iterations):
        dist.barrier()
        started = time.perf_counter()
        dist.all_to_all_single(
            receive,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        collective_ms = (time.perf_counter() - started) * 1000.0

        reduction_started = time.perf_counter()
        reduced = receive.view(world_size, own_elements).sum(dim=0)
        reduction_ms = (time.perf_counter() - reduction_started) * 1000.0
        if reduced.numel() and not torch.all(reduced == expected):
            raise AssertionError("The all-to-all owner reduction produced an unexpected value.")
        if iteration >= warmup:
            collective_samples.append(_rank_times(collective_ms, world_size))
            reduction_samples.append(_rank_times(reduction_ms, world_size))

    return {
        "collective": _summarize(collective_samples),
        "owner_cpu_sum": _summarize(reduction_samples),
        "local_input_bytes": int(send.numel() * send.element_size()),
        "local_receive_bytes": int(receive.numel() * receive.element_size()),
        "layers_owned": len(layers_by_owner[rank]),
    }


def _benchmark_reduce_scatter(
    *,
    rank: int,
    world_size: int,
    layers: int,
    candidates_per_layer: int,
    width: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    shard_rows = (candidates_per_layer + world_size - 1) // world_size
    padded_candidates = shard_rows * world_size
    output_elements = layers * shard_rows * width
    send = torch.full((world_size * output_elements,), float(rank + 1), dtype=torch.float32)
    receive = torch.empty((output_elements,), dtype=torch.float32)
    expected = world_size * (world_size + 1) / 2

    collective_samples: list[list[float]] = []
    for iteration in range(warmup + iterations):
        dist.barrier()
        started = time.perf_counter()
        dist.reduce_scatter_tensor(receive, send, op=dist.ReduceOp.SUM)
        collective_ms = (time.perf_counter() - started) * 1000.0
        if not torch.all(receive == expected):
            raise AssertionError("The reduce-scatter result produced an unexpected value.")
        if iteration >= warmup:
            collective_samples.append(_rank_times(collective_ms, world_size))

    return {
        "collective": _summarize(collective_samples),
        "local_input_bytes": int(send.numel() * send.element_size()),
        "local_output_bytes": int(receive.numel() * receive.element_size()),
        "candidate_rows_per_layer": shard_rows,
        "padded_candidates_per_layer": padded_candidates,
    }


def main() -> None:
    args = _parse_args()
    dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    torch.set_num_threads(1)

    results: dict[str, object] = {
        "backend": str(dist.get_backend()),
        "world_size": world_size,
        "layers": args.layers,
        "candidates_per_layer": args.candidates_per_layer,
        "statistic_width": args.statistic_width,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    if "all-to-all" in args.collectives:
        results["all_to_all"] = _benchmark_all_to_all(
            rank=rank,
            world_size=world_size,
            layers=args.layers,
            candidates_per_layer=args.candidates_per_layer,
            width=args.statistic_width,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    if "reduce-scatter" in args.collectives:
        results["reduce_scatter"] = _benchmark_reduce_scatter(
            rank=rank,
            world_size=world_size,
            layers=args.layers,
            candidates_per_layer=args.candidates_per_layer,
            width=args.statistic_width,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    if rank == 0:
        encoded = json.dumps(results, indent=2, sort_keys=True)
        print(encoded, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + os.linesep)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
