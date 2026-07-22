# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Benchmark exact greedy swap/cover planning on saved per-rank routes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0, help="Route rank for a non-distributed run.")
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--slot-increment", type=int, default=1)
    parser.add_argument("--phase", choices=("steady", "initialize", "initialized-steady"), default="steady")
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--max-covers", type=int, default=1)
    parser.add_argument("--max-copies", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _initialize(backend: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "hccl":
        importlib.import_module("torch_npu")
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend=backend)
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, device


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _load_route(route_dir: Path, layer: int, rank: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or "routes" not in record:
        raise ValueError(f"{path} is not a VeOmni local-route snapshot.")
    routes = record["routes"].to(device=device, dtype=torch.long, non_blocking=True)
    return routes, record


def _initial_layout(
    *,
    num_experts: int,
    ep_size: int,
    slot_increment: int,
    phase: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size.")
    base = num_experts // ep_size
    slots_per_rank = base + slot_increment
    experts = torch.arange(num_experts, dtype=torch.long, device=device)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + torch.remainder(experts, base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, experts)
    if phase == "steady":
        ranks = torch.arange(ep_size, dtype=torch.long, device=device)
        for offset in range(slot_increment):
            replica = torch.remainder(ranks + offset + 1, ep_size) * base + (offset % base)
            layout[ranks * slots_per_rank + base + offset] = replica
    return layout, owners, slots_per_rank


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _plan_digest(plan) -> int:
    payload = {
        "actions": [action.format() for action in plan.actions],
        "layout": list(plan.final_layout),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return int(digest[:15], 16)


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive.")
    if not 1 <= args.max_copies <= 8:
        raise ValueError("max-copies must be between 1 and 8.")
    rank, world_size, device = _initialize(args.backend)
    route_rank = rank if world_size > 1 else args.rank
    if world_size > 1 and world_size != args.ep_size:
        raise ValueError(f"Distributed world size {world_size} must equal ep_size {args.ep_size}.")
    routes, metadata = _load_route(args.route_dir, args.layer, route_rank, device)
    num_experts = int(metadata["num_experts"])
    layout, owners, slots_per_rank = _initial_layout(
        num_experts=num_experts,
        ep_size=args.ep_size,
        slot_increment=args.slot_increment,
        phase=args.phase,
        device=device,
    )

    def reducer(tensor: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=tuple(args.group_sizes),
            source="benchmark",
            local_world_size=args.local_world_size,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=int(metadata["hidden_size"]),
        bytes_per_element=int(metadata["bytes_per_element"]),
        slots_per_rank=slots_per_rank,
        reducer=reducer,
        process_group=dist.group.WORLD if dist.is_initialized() else None,
        max_copies=args.max_copies,
    )
    initialization_ms = None
    initialization_actions: list[str] = []
    if args.phase == "initialized-steady":
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        initialization = planner.plan(
            routes,
            layout,
            owners,
            source_ranks=route_rank,
            max_swaps=0,
            max_replicas=args.ep_size * args.slot_increment,
            step=0,
            layer_seed=args.layer,
        )
        _synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        initialization_ms = float(elapsed.item())
        initialization_actions = [action.format() for action in initialization.actions]
        layout = torch.tensor(initialization.final_layout, dtype=torch.long, device=device)
        if bool((layout < 0).any().item()):
            raise RuntimeError("Initialization did not fill every redundant slot.")
    all_slots = torch.arange(layout.numel(), dtype=torch.long, device=device)
    owner_mask = torch.zeros_like(layout, dtype=torch.bool)
    owner_mask.scatter_(0, owners, True)
    if args.phase == "initialize":
        destinations = torch.nonzero(layout < 0, as_tuple=False).flatten()
        candidate_rows = planner._cover_rows(layout, owners, destinations)
        max_covers = args.ep_size * args.slot_increment
        max_swaps = 0
    else:
        rows = []
        if args.max_swaps:
            rows.append(planner._swap_rows(layout, owners))
        if args.max_covers:
            destinations = all_slots[(~owner_mask) & (layout >= 0)]
            rows.append(planner._cover_rows(layout, owners, destinations))
        candidate_rows = torch.cat([row for row in rows if row.numel()], dim=0)
        max_covers = args.max_covers
        max_swaps = args.max_swaps
    _synchronize(device)
    candidate_count = int(candidate_rows.shape[0])

    samples: list[float] = []
    final_plan = None
    for iteration in range(args.warmup + args.iterations):
        if dist.is_initialized():
            dist.barrier()
        _synchronize(device)
        started = time.perf_counter()
        plan = planner.plan(
            routes,
            layout,
            owners,
            source_ranks=route_rank,
            max_swaps=max_swaps,
            max_replicas=max_covers,
            step=1,
            layer_seed=args.layer,
        )
        _synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            digest = torch.tensor([_plan_digest(plan)], dtype=torch.int64, device=device)
            gathered = torch.empty((world_size,), dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(gathered, digest)
            if bool((gathered != gathered[0]).any().item()):
                raise RuntimeError("Ranks selected different placement plans.")
        elapsed_ms = float(elapsed.item())
        if iteration >= args.warmup:
            samples.append(elapsed_ms)
        if args.verbose and rank == 0:
            print(json.dumps({"iteration": iteration, "elapsed_ms": elapsed_ms}))
        final_plan = plan

    assert final_plan is not None
    result = {
        "route": {
            "layer": args.layer,
            "rank": route_rank if world_size == 1 else "distributed",
            "shape": list(routes.shape),
        },
        "world_size": world_size,
        "ep_size": args.ep_size,
        "phase": args.phase,
        "max_copies": args.max_copies,
        "initialization_ms": initialization_ms,
        "initialization_actions": initialization_actions,
        "candidate_count": candidate_count,
        "timing_ms": {
            "median": statistics.median(samples),
            "p90": _percentile(samples, 0.9),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "actions": [action.format() for action in final_plan.actions],
        "baseline_communication": final_plan.baseline_cost.communication,
        "final_communication": final_plan.final_cost.communication,
        "predicted_communication_speedup": (
            final_plan.baseline_cost.communication / final_plan.final_cost.communication
            if final_plan.final_cost.communication > 0
            else math.inf
        ),
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
