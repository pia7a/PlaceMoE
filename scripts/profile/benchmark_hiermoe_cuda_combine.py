#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark the CUDA HierMoE final-combine reduction against the fallback."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from veomni.distributed.moe.hiermoe.all_to_all import _index_add_dim0_cast_output


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=int, default=65_536)
    parser.add_argument("--output-rows", type=int, default=32_768)
    parser.add_argument("--hidden-size", type=int, default=2_048)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _measure(
    source: torch.Tensor,
    index: torch.Tensor,
    output_rows: int,
    *,
    enabled: bool,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    os.environ["VEOMNI_HIERMOE_CUDA_SEGMENT_SUM"] = "1" if enabled else "0"
    output = None
    for _ in range(warmup):
        output = _index_add_dim0_cast_output(source, index, output_rows)
    torch.cuda.synchronize()
    del output

    samples_ms = []
    peak_extra_bytes = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = _index_add_dim0_cast_output(source, index, output_rows)
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)) / iterations)
        peak_extra_bytes.append(max(0, torch.cuda.max_memory_allocated() - allocated_before))
        del output

    return {
        "implementation": "triton_segment_sum" if enabled else "torch_index_add_fp32",
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "samples_ms": samples_ms,
        "peak_extra_bytes": max(peak_extra_bytes),
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if min(args.source_rows, args.output_rows, args.hidden_size, args.iterations, args.repeats) <= 0:
        raise SystemExit("all dimensions, iterations, and repeats must be positive")
    if args.warmup < 0:
        raise SystemExit("warmup must be non-negative")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = _DTYPES[args.dtype]
    source = torch.randn(
        (args.source_rows, args.hidden_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    index = torch.arange(args.source_rows, dtype=torch.long, device=device).remainder(args.output_rows)
    index = index[torch.randperm(args.source_rows, device=device, generator=generator)]

    previous = os.environ.get("VEOMNI_HIERMOE_CUDA_SEGMENT_SUM")
    try:
        os.environ["VEOMNI_HIERMOE_CUDA_SEGMENT_SUM"] = "0"
        reference = _index_add_dim0_cast_output(source, index, args.output_rows)
        os.environ["VEOMNI_HIERMOE_CUDA_SEGMENT_SUM"] = "1"
        actual = _index_add_dim0_cast_output(source, index, args.output_rows)
        torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
        del actual, reference

        fallback = _measure(
            source,
            index,
            args.output_rows,
            enabled=False,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        triton_result = _measure(
            source,
            index,
            args.output_rows,
            enabled=True,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    finally:
        if previous is None:
            os.environ.pop("VEOMNI_HIERMOE_CUDA_SEGMENT_SUM", None)
        else:
            os.environ["VEOMNI_HIERMOE_CUDA_SEGMENT_SUM"] = previous

    payload = {
        "status": "accepted",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": {
            "source_rows": args.source_rows,
            "output_rows": args.output_rows,
            "hidden_size": args.hidden_size,
            "dtype": args.dtype,
        },
        "fallback": fallback,
        "triton": triton_result,
        "speedup": fallback["median_ms"] / triton_result["median_ms"],
        "peak_memory_ratio": fallback["peak_extra_bytes"] / max(1, triton_result["peak_extra_bytes"]),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
