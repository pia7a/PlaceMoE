#!/usr/bin/env python3
"""Fit HierMoE alpha/beta communication model with distributed A2A microbenchmarks.

The output JSON is consumed directly by
``veomni.distributed.moe.hiermoe.perf_model.HierMoEPerfModel.from_path``.
Run this script with the same torchrun rank layout as training, before the
training torchrun starts.  The measured time is therefore not included in
training speedup accounting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel, fit_perf_model_on_startup
from veomni.distributed.moe.hiermoe.topology import infer_hierarchy
from veomni.utils.device import get_device_type, get_torch_device, set_device, synchronize


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def init_device() -> tuple[str, torch.device]:
    local_rank = env_int("LOCAL_RANK", 0)
    device_type = get_device_type()
    if device_type == "cpu":
        return device_type, torch.device("cpu")
    namespace = get_torch_device()
    if not namespace.is_available():
        raise RuntimeError(f"The selected {device_type} backend is unavailable.")
    set_device(local_rank)
    return device_type, torch.device(device_type, local_rank)


def auto_backend(device_type: str) -> str:
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def sync_device(device_type: str) -> None:
    if device_type != "cpu":
        synchronize()


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def parse_int_csv(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - pos) + ordered[high] * (pos - low))


def summarize_latencies(latencies_ms: list[float], measure_last_n: int) -> dict[str, float | list[float] | int]:
    selected = latencies_ms[-measure_last_n:] if measure_last_n > 0 else latencies_ms
    if not selected:
        selected = latencies_ms
    return {
        "latencies_ms": latencies_ms,
        "selected_latencies_ms": selected,
        "measured_iters": len(latencies_ms),
        "measured_iters_used": len(selected),
        "latency_ms_mean": statistics.fmean(selected) if selected else 0.0,
        "latency_ms_median": statistics.median(selected) if selected else 0.0,
        "latency_ms_p90": percentile(selected, 0.90),
        "latency_ms_max": max(selected) if selected else 0.0,
    }


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if not ys:
        return 0.0, 0.0
    if len(xs) < 2:
        return 0.0, float(ys[0])
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0, float(mean_y)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = mean_y - slope * mean_x
    return float(slope), float(intercept)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    model_key: str
    level_index: int
    ranks: tuple[int, ...]


def build_group_specs(world_size: int, ranks_per_node: int, hierarchy_sizes: tuple[int, ...]) -> list[GroupSpec]:
    specs = [GroupSpec(name="a2a_world", model_key="a2a", level_index=-1, ranks=tuple(range(world_size)))]
    if world_size <= 1 or not hierarchy_sizes:
        return specs

    intra_size = hierarchy_sizes[0]
    if len(hierarchy_sizes) == 1 or intra_size <= 1:
        return specs

    if len(hierarchy_sizes) == 2:
        num_nodes = world_size // intra_size
        for local_offset in range(intra_size):
            ranks = tuple(local_offset + node_idx * intra_size for node_idx in range(num_nodes))
            if len(ranks) > 1:
                specs.append(
                    GroupSpec(
                        name=f"inter0_local{local_offset}",
                        model_key="inter_0",
                        level_index=0,
                        ranks=ranks,
                    )
                )
    else:
        mid_size = hierarchy_sizes[-2]
        num_mid_groups = world_size // mid_size
        for mid_offset in range(mid_size):
            ranks = tuple(mid_offset + mid_idx * mid_size for mid_idx in range(num_mid_groups))
            if len(ranks) > 1:
                specs.append(
                    GroupSpec(
                        name=f"inter0_mid_offset{mid_offset}",
                        model_key="inter_0",
                        level_index=0,
                        ranks=ranks,
                    )
                )
        nodes_per_mid = mid_size // intra_size
        for mid_idx in range(num_mid_groups):
            mid_start = mid_idx * mid_size
            for local_offset in range(intra_size):
                ranks = tuple(mid_start + node_idx * intra_size + local_offset for node_idx in range(nodes_per_mid))
                if len(ranks) > 1:
                    specs.append(
                        GroupSpec(
                            name=f"inter1_mid{mid_idx}_local{local_offset}",
                            model_key="inter_1",
                            level_index=1,
                            ranks=ranks,
                        )
                    )

    for start in range(0, world_size, intra_size):
        ranks = tuple(rank for rank in range(start, min(start + intra_size, world_size)))
        if len(ranks) > 1:
            specs.append(GroupSpec(name=f"intra_{start // intra_size}", model_key="intra", level_index=0, ranks=ranks))
    return specs


def bench_group(
    *,
    group: dist.ProcessGroup,
    group_size: int,
    message_bytes: int,
    dtype: torch.dtype,
    device: torch.device,
    device_type: str,
    warmup: int,
    iters: int,
    measure_last_n: int,
) -> tuple[dict[str, Any], int]:
    itemsize = torch.empty((), dtype=dtype).element_size()
    chunk_elements = max(1, math.ceil(message_bytes / max(1, group_size * itemsize)))
    input_tensor = torch.empty(group_size * chunk_elements, dtype=dtype, device=device)
    output_tensor = torch.empty_like(input_tensor)
    input_tensor.fill_(1)

    for _ in range(warmup):
        dist.all_to_all_single(output_tensor, input_tensor, group=group)
    sync_device(device_type)

    latencies_ms: list[float] = []
    for _ in range(iters):
        sync_device(device_type)
        start = time.perf_counter()
        dist.all_to_all_single(output_tensor, input_tensor, group=group)
        sync_device(device_type)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    sync_device(device_type)

    actual_message_bytes = int(input_tensor.numel() * input_tensor.element_size())
    return summarize_latencies(latencies_ms, measure_last_n), actual_message_bytes


def gather_case_result(
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    valid: bool,
    latency_ms: float,
    actual_message_bytes: int,
    group_size: int,
    spec_index: int,
    message_index: int,
) -> list[list[float]]:
    # HCCL does not reliably support float64 collectives on every Ascend
    # runtime. The payload values are all exactly representable at the tested
    # message sizes, and float32 provides ample precision for millisecond
    # latency fitting.
    row = torch.tensor(
        [
            1.0 if valid else 0.0,
            float(latency_ms),
            float(actual_message_bytes),
            float(group_size),
            float(spec_index),
            float(message_index),
            float(rank),
        ],
        dtype=torch.float32,
        device=device,
    )
    gathered = [torch.empty_like(row) for _ in range(world_size)]
    dist.all_gather(gathered, row)
    return [item.cpu().tolist() for item in gathered]


def current_stage_group(
    specs: list[GroupSpec],
    model_key: str,
    rank: int,
) -> tuple[int, GroupSpec, dist.ProcessGroup] | None:
    stage_specs = [
        (idx, spec) for idx, spec in enumerate(specs) if spec.model_key == model_key and len(spec.ranks) > 1
    ]
    current = next(((idx, spec) for idx, spec in stage_specs if rank in spec.ranks), None)
    if current is None:
        return None
    if model_key == "a2a":
        idx, spec = current
        return idx, spec, dist.group.WORLD

    rank_lists = [list(spec.ranks) for _, spec in stage_specs]
    new_subgroups = getattr(dist, "new_subgroups_by_enumeration", None)
    if new_subgroups is not None:
        group, _ = new_subgroups(rank_lists, group_desc=f"hiermoe_perf_{model_key}")
        if group is None:
            return None
        idx, spec = current
        return idx, spec, group

    current_group = None
    for _, spec in stage_specs:
        group = dist.new_group(ranks=list(spec.ranks))
        if rank in spec.ranks:
            current_group = group
    if current_group is None:
        return None
    idx, spec = current
    return idx, spec, current_group


def broadcast_text(text: str | None, device: torch.device, src: int = 0) -> str:
    rank = dist.get_rank()
    encoded = (text or "").encode("utf-8") if rank == src else b""
    size = torch.tensor([len(encoded)], dtype=torch.long, device=device)
    dist.broadcast(size, src=src)
    num_bytes = int(size.item())
    if rank == src:
        buffer = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    else:
        buffer = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    dist.broadcast(buffer, src=src)
    return bytes(buffer.cpu().tolist()).decode("utf-8")


def fit_model(rows: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    by_bytes: dict[int, list[float]] = {}
    for row in rows:
        if row["model_key"] != model_key:
            continue
        by_bytes.setdefault(int(row["message_bytes"]), []).append(float(row["latency_ms"]))

    fit_points: list[dict[str, float]] = []
    for message_bytes, latencies in sorted(by_bytes.items()):
        fit_points.append(
            {
                "bytes": float(message_bytes),
                "latency_ms_mean": statistics.fmean(latencies),
                "latency_ms_p90": percentile(latencies, 0.90),
                "latency_ms_max": max(latencies),
                "sample_count": float(len(latencies)),
            }
        )

    xs = [point["bytes"] for point in fit_points]
    ys = [point["latency_ms_max"] for point in fit_points]
    beta, alpha = linear_fit(xs, ys)
    return {
        "alpha": max(0.0, float(alpha)),
        "beta": max(0.0, float(beta)),
        "fit_stat": "latency_ms_max_across_groups",
        "fit_points": fit_points,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--message-bytes-csv",
        default="67108864,134217728,268435456,536870912",
        help="Comma-separated total payload bytes per rank for each all_to_all_single call.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--measure-last-n",
        type=int,
        default=3,
        help="Use only the last N measured iterations for each rank latency. 0 uses all measured iterations.",
    )
    parser.add_argument("--ranks-per-node", type=int, default=0)
    parser.add_argument(
        "--ep-size",
        type=int,
        default=0,
        help="EP group size. The standalone calibration job must contain exactly one EP group.",
    )
    parser.add_argument(
        "--hierarchy-group-sizes-csv",
        help="Comma-separated hierarchy cumulative group sizes, e.g. 8,16,64. Defaults to VeOmni auto topology.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--details-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_type, device = init_device()
    backend = auto_backend(device_type) if args.backend == "auto" else args.backend
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_size = args.ep_size or world_size
    if ep_size != world_size:
        raise ValueError(
            "Standalone PlaceMoE calibration must be launched with exactly one EP group: "
            f"--ep-size={ep_size}, distributed world_size={world_size}."
        )
    ranks_per_node = args.ranks_per_node or env_int("LOCAL_WORLD_SIZE", 1)
    message_bytes_list = parse_int_csv(args.message_bytes_csv)
    if not message_bytes_list:
        raise ValueError("--message-bytes-csv must contain at least one positive integer.")
    if any(message_bytes <= 0 for message_bytes in message_bytes_list):
        raise ValueError("--message-bytes-csv values must be positive.")

    os.environ["LOCAL_WORLD_SIZE"] = str(ranks_per_node)
    hierarchy = infer_hierarchy(
        ep_size=ep_size,
        topology="auto",
        hierarchy_group_sizes=parse_int_csv(args.hierarchy_group_sizes_csv),
    )
    hierarchy_sizes = hierarchy.group_sizes
    specs = build_group_specs(world_size=world_size, ranks_per_node=ranks_per_node, hierarchy_sizes=hierarchy_sizes)
    dtype = dtype_from_name(args.dtype)

    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        if args.details_json is not None:
            args.details_json.parent.mkdir(parents=True, exist_ok=True)
        print(
            "HierMoE perf bench: "
            f"world_size={world_size} ranks_per_node={ranks_per_node} hierarchy={hierarchy_sizes} "
            f"dtype={args.dtype} messages={message_bytes_list} backend={backend}",
            flush=True,
        )

    rank_rows: list[dict[str, Any]] = []
    spec_by_index = dict(enumerate(specs))
    stage_keys = ["a2a"]
    stage_keys.extend(f"inter_{idx}" for idx in range(max(0, len(hierarchy_sizes) - 1)))
    if len(hierarchy_sizes) > 1:
        stage_keys.append("intra")

    for stage_key in stage_keys:
        member = current_stage_group(specs, stage_key, rank)
        for message_idx, message_bytes in enumerate(message_bytes_list):
            stats: dict[str, Any] = {}
            actual_message_bytes = message_bytes
            if member is not None:
                spec_idx, spec, group = member
                stats, actual_message_bytes = bench_group(
                    group=group,
                    group_size=len(spec.ranks),
                    message_bytes=message_bytes,
                    dtype=dtype,
                    device=device,
                    device_type=device_type,
                    warmup=args.warmup,
                    iters=args.iters,
                    measure_last_n=args.measure_last_n,
                )
            else:
                spec_idx = -1
                spec = GroupSpec(name=f"{stage_key}_non_member", model_key=stage_key, level_index=-1, ranks=())
            gathered = gather_case_result(
                device=device,
                rank=rank,
                world_size=world_size,
                valid=member is not None,
                latency_ms=float(stats.get("latency_ms_mean", 0.0)) if member is not None else 0.0,
                actual_message_bytes=actual_message_bytes,
                group_size=len(spec.ranks) if member is not None else 0,
                spec_index=spec_idx,
                message_index=message_idx,
            )
            if rank == 0:
                valid_rows = [row for row in gathered if row[0] > 0.5]
                for row in valid_rows:
                    row_spec = spec_by_index[int(row[4])]
                    rank_rows.append(
                        {
                            "scope": row_spec.name,
                            "model_key": row_spec.model_key,
                            "level_index": row_spec.level_index,
                            "group_size": int(row[3]),
                            "rank": int(row[6]),
                            "requested_message_bytes": int(message_bytes),
                            "message_bytes": int(row[2]),
                            "latency_ms": float(row[1]),
                        }
                    )
                if valid_rows:
                    latency_values = [float(row[1]) for row in valid_rows]
                    print(
                        f"{stage_key} bytes={int(valid_rows[0][2])} "
                        f"mean={statistics.fmean(latency_values):.3f}ms max={max(latency_values):.3f}ms",
                        flush=True,
                    )

    runtime_fit = fit_perf_model_on_startup(
        HierMoEPerfModel.default(),
        group=dist.group.WORLD,
        local_world_size=ranks_per_node,
        warmup=args.warmup,
        repeats=args.iters,
    )

    output_text: str | None = None
    if rank == 0:
        a2a = fit_model(rank_rows, "a2a")
        inter = []
        for idx in range(max(0, len(hierarchy_sizes) - 1)):
            inter.append(fit_model(rank_rows, f"inter_{idx}"))
        intra = fit_model(rank_rows, "intra") if len(hierarchy_sizes) > 1 else fit_model(rank_rows, "a2a")

        output = {
            "schema_version": 2,
            "a2a": {"alpha": a2a["alpha"], "beta": a2a["beta"]},
            "inter": [{"alpha": row["alpha"], "beta": row["beta"]} for row in inter],
            "intra": {"alpha": intra["alpha"], "beta": intra["beta"]},
            "state_move": runtime_fit.state_move.to_payload(),
            "gradient_sync": runtime_fit.gradient_sync.to_payload(),
            "source": "bench_hiermoe_perf_model",
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "world_size": world_size,
                "ep_size": ep_size,
                "ranks_per_node": ranks_per_node,
                "hierarchy_group_sizes": list(hierarchy_sizes),
                "device_type": device_type,
                "backend": backend,
                "dtype": args.dtype,
                "message_bytes_requested": message_bytes_list,
                "warmup": args.warmup,
                "iters": args.iters,
                "measure_last_n": args.measure_last_n,
                "fit_stat": "latency_ms_max_across_groups",
            },
            "fit_points": {
                "a2a": a2a["fit_points"],
                "inter": [row["fit_points"] for row in inter],
                "intra": intra["fit_points"],
            },
        }
        output_text = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.details_json is not None:
            args.details_json.write_text(
                json.dumps({"rows": rank_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    output_text = broadcast_text(output_text, device=device, src=0)
    if env_int("LOCAL_RANK", 0) == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(f".{args.output_json.name}.tmp-{os.getpid()}-{rank}")
        temporary.write_text(output_text, encoding="utf-8")
        temporary.replace(args.output_json)
        print(f"Wrote HierMoE perf model to {args.output_json}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
