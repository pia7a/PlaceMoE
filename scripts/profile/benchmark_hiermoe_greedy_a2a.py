#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Replay greedy swap/cover physical routes through production HierMoE A2A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.state import configure_hiermoe
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.utils.import_utils import is_torch_npu_available


_BASELINE0 = "baseline0_original_flat_a2a"
_BASELINE1 = "baseline1_owner_hierarchical_rank_dedup"
_INITIAL = "baseline2_initial_redundant_layout"
_GREEDY = "greedy_swap_cover"
_METRICS = ("wall_ms", "dispatch_ms", "combine_ms", "a2a_ms")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--slot-increment", type=int, default=1)
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--max-covers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _initial_layout(
    num_experts: int,
    ep_size: int,
    slot_increment: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by EP size.")
    base = num_experts // ep_size
    slots_per_rank = base + slot_increment
    experts = torch.arange(num_experts, dtype=torch.long, device=device)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + torch.remainder(experts, base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, experts)
    ranks = torch.arange(ep_size, dtype=torch.long, device=device)
    for offset in range(slot_increment):
        replica = torch.remainder(ranks + offset + 1, ep_size) * base + (offset % base)
        layout[ranks * slots_per_rank + base + offset] = replica
    return layout, owners, slots_per_rank


def _load_routes(
    route_dir: Path,
    rank: int,
    layers: range,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    routes = []
    bytes_per_element = 0
    for layer in layers:
        path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "veomni.hiermoe.local_route":
            raise ValueError(f"Unsupported route snapshot: {path}")
        if int(payload["global_rank"]) != rank or int(payload["layer"]) != layer:
            raise ValueError(f"Route metadata mismatch: {path}")
        route = payload["routes"].to(dtype=torch.long)
        if route.ndim != 2:
            raise ValueError(f"Expected rank-2 routes in {path}.")
        sorted_route = route.sort(dim=-1).values
        if route.shape[1] > 1 and bool((sorted_route[:, 1:] == sorted_route[:, :-1]).any().item()):
            raise ValueError(f"Captured gate top-k contains duplicate logical experts: {path}")
        routes.append(route.to(device=device, non_blocking=True).contiguous())
        bytes_per_element = int(payload["bytes_per_element"])
    return routes, bytes_per_element


def _plan_routes(
    logical_routes: list[torch.Tensor],
    *,
    layers: range,
    layout: torch.Tensor,
    owners: torch.Tensor,
    slots_per_rank: int,
    hierarchy: Hierarchy,
    hidden_size: int,
    bytes_per_element: int,
    source_rank: int,
    max_swaps: int,
    max_covers: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    def reducer(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    planner = GreedyCommunicationPlanner(
        hierarchy=hierarchy,
        perf_model=HierMoEPerfModel.default(),
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        slots_per_rank=slots_per_rank,
        reducer=reducer,
    )
    initial_routes = []
    greedy_routes = []
    records = []
    for layer, routes in zip(layers, logical_routes, strict=True):
        initial_routes.append(
            assign_tokens_to_copies_greedy(
                routes,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=source_rank,
                hierarchy_group_sizes=hierarchy.group_sizes,
                num_experts=int(owners.numel()),
                step=1,
                layer_seed=layer,
            ).contiguous()
        )
        torch.npu.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        plan = planner.plan(
            routes,
            layout,
            owners,
            source_ranks=source_rank,
            max_swaps=max_swaps,
            max_replicas=max_covers,
            step=1,
            layer_seed=layer,
        )
        torch.npu.synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if plan.local_physical_routes is None:
            raise RuntimeError("Greedy planner did not return physical routes.")
        greedy_routes.append(plan.local_physical_routes.contiguous())
        digest_payload = json.dumps(
            {
                "actions": [action.format() for action in plan.actions],
                "layout": list(plan.final_layout),
            },
            sort_keys=True,
        ).encode()
        digest = torch.tensor(
            [int(hashlib.sha256(digest_payload).hexdigest()[:15], 16)],
            dtype=torch.int64,
            device=device,
        )
        gathered = torch.empty((hierarchy.ep_size,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(gathered, digest)
        if bool((gathered != gathered[0]).any().item()):
            raise RuntimeError(f"Ranks selected different plans for layer {layer}.")
        records.append(
            {
                "layer": layer,
                "planning_ms": float(elapsed.item()),
                "actions": [action.format() for action in plan.actions],
                "baseline_communication": plan.baseline_cost.communication,
                "final_communication": plan.final_cost.communication,
                "predicted_speedup": (
                    plan.baseline_cost.communication / plan.final_cost.communication
                    if plan.final_cost.communication > 0
                    else math.inf
                ),
            }
        )
    return initial_routes, greedy_routes, records


def _count_sorted_remote_unique(values: torch.Tensor, source: int) -> torch.Tensor:
    remote = values != source
    first = torch.ones_like(remote)
    first[..., 1:] = values[..., 1:] != values[..., :-1]
    return (remote & first).sum(dim=-1)


def _communication_score(
    routes_by_layer: list[torch.Tensor],
    *,
    slots_per_rank: int,
    source_rank: int,
    ranks_per_node: int,
    deduplicate: bool,
) -> tuple[int, int]:
    remote_nodes = 0
    remote_ranks = 0
    for routes in routes_by_layer:
        ranks = torch.div(routes, slots_per_rank, rounding_mode="floor").sort(dim=-1).values
        nodes = torch.div(ranks, ranks_per_node, rounding_mode="floor")
        if deduplicate:
            remote_nodes += int(_count_sorted_remote_unique(nodes, source_rank // ranks_per_node).sum().item())
            remote_ranks += int(_count_sorted_remote_unique(ranks, source_rank).sum().item())
        else:
            remote_nodes += int((nodes != source_rank // ranks_per_node).sum().item())
            remote_ranks += int((ranks != source_rank).sum().item())
    return remote_nodes, remote_ranks


def _run_rank_dedup(
    routes: torch.Tensor,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    num_physical_experts: int,
) -> tuple[Any, Any, Any, Any]:
    dispatch_start = torch.npu.Event(enable_timing=True)
    dispatch_end = torch.npu.Event(enable_timing=True)
    combine_start = torch.npu.Event(enable_timing=True)
    combine_end = torch.npu.Event(enable_timing=True)
    dispatch_start.record()
    permuted, context, _ = rank_dedup_dispatch(
        hidden[: routes.shape[0]],
        routes,
        weights[: routes.shape[0]],
        num_physical_experts,
        dist.group.WORLD,
    )
    dispatch_end.record()
    combine_start.record()
    rank_dedup_combine(permuted, context)
    combine_end.record()
    return dispatch_start, dispatch_end, combine_start, combine_end


def _run_original(
    routes: torch.Tensor,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> tuple[Any, Any, Any, Any]:
    dispatch_start = torch.npu.Event(enable_timing=True)
    dispatch_end = torch.npu.Event(enable_timing=True)
    combine_start = torch.npu.Event(enable_timing=True)
    combine_end = torch.npu.Event(enable_timing=True)
    dispatch_start.record()
    expert_mask = torch.nn.functional.one_hot(routes, num_classes=num_experts).permute(2, 1, 0)
    input_splits, output_splits, tokens_per_local_expert, _ = preprocess(
        expert_mask=expert_mask,
        num_experts=num_experts,
        ep_group=dist.group.WORLD,
    )
    permuted, routing_map, local_mapping, original_shape = token_pre_all2all(
        hidden_states=hidden[: routes.shape[0]],
        expert_mask=expert_mask,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=tokens_per_local_expert,
        ep_group=dist.group.WORLD,
    )
    dispatch_end.record()
    combine_start.record()
    tokens_post_all2all(
        expert_outputs=permuted,
        routing_weights=weights[: routes.shape[0]],
        selected_experts=routes,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=tokens_per_local_expert,
        routing_map=routing_map,
        local_input_permutation_mapping=local_mapping,
        org_hidden_states_shape=original_shape,
        ep_group=dist.group.WORLD,
    )
    combine_end.record()
    return dispatch_start, dispatch_end, combine_start, combine_end


def _run_variant(
    routes_by_layer: list[torch.Tensor],
    hidden: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_physical_experts: int,
    original: bool,
) -> tuple[float, float, float, float]:
    torch.npu.synchronize()
    dist.barrier()
    started = time.perf_counter()
    events = [
        (
            _run_original(routes, hidden, weights, num_physical_experts)
            if original
            else _run_rank_dedup(routes, hidden, weights, num_physical_experts)
        )
        for routes in routes_by_layer
    ]
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    dispatch_ms = sum(float(start.elapsed_time(end)) for start, end, _, _ in events)
    combine_ms = sum(float(start.elapsed_time(end)) for _, _, start, end in events)
    values = torch.tensor(
        [wall_ms, dispatch_ms, combine_ms, dispatch_ms + combine_ms],
        dtype=torch.float32,
        device=hidden.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    dist.barrier()
    return tuple(float(value) for value in values.cpu().tolist())


def _summary(samples: list[tuple[float, ...]]) -> dict[str, dict[str, float]]:
    result = {}
    for index, name in enumerate(_METRICS):
        values = sorted(row[index] for row in samples)
        p90 = values[min(len(values) - 1, max(0, math.ceil(0.9 * len(values)) - 1))]
        result[name] = {"median": statistics.median(values), "p90": p90, "max": max(values)}
    return result


def main() -> None:
    args = _parse_args()
    if not is_torch_npu_available():
        raise RuntimeError("This benchmark requires torch_npu and Ascend devices.")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        rank = dist.get_rank()
        ep_size = dist.get_world_size()
        if ep_size % args.ranks_per_node or args.num_experts % ep_size:
            raise ValueError("EP topology is incompatible with ranks-per-node or num-experts.")
        if args.group_sizes[-1] != ep_size:
            raise ValueError("The final hierarchy group size must equal EP size.")
        device = torch.device(f"npu:{local_rank}")
        configure_hiermoe(
            HierMoEConfig(
                enable=True,
                token_dedup=True,
                expert_swap=False,
                hierarchy_group_sizes=args.group_sizes,
            ),
            dist.group.WORLD,
        )
        layers = range(args.layer_start, args.layer_start + args.layers)
        logical_routes, bytes_per_element = _load_routes(args.route_dir, rank, layers, device)
        layout, owners, slots_per_rank = _initial_layout(
            args.num_experts,
            ep_size,
            args.slot_increment,
            device,
        )
        hierarchy = Hierarchy(ep_size, tuple(args.group_sizes), "benchmark", args.ranks_per_node)
        initial_routes, greedy_routes, plans = _plan_routes(
            logical_routes,
            layers=layers,
            layout=layout,
            owners=owners,
            slots_per_rank=slots_per_rank,
            hierarchy=hierarchy,
            hidden_size=args.hidden_size,
            bytes_per_element=bytes_per_element,
            source_rank=rank,
            max_swaps=args.max_swaps,
            max_covers=args.max_covers,
            device=device,
        )
        owner_routes = [
            owners.index_select(0, routes.reshape(-1)).view_as(routes).contiguous() for routes in logical_routes
        ]
        routes_by_variant = {
            _BASELINE0: logical_routes,
            _BASELINE1: owner_routes,
            _INITIAL: initial_routes,
            _GREEDY: greedy_routes,
        }
        physical_counts = {
            _BASELINE0: args.num_experts,
            _BASELINE1: int(layout.numel()),
            _INITIAL: int(layout.numel()),
            _GREEDY: int(layout.numel()),
        }
        local_counts = []
        for name, routes in routes_by_variant.items():
            local_counts.extend(
                _communication_score(
                    routes,
                    slots_per_rank=args.num_experts // ep_size if name == _BASELINE0 else slots_per_rank,
                    source_rank=rank,
                    ranks_per_node=args.ranks_per_node,
                    deduplicate=name != _BASELINE0,
                )
            )
        counts = torch.tensor(local_counts, dtype=torch.long, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        top_k = int(logical_routes[0].shape[1])
        max_tokens = max(int(routes.shape[0]) for routes in logical_routes)
        hidden = torch.zeros((max_tokens, args.hidden_size), dtype=torch.bfloat16, device=device)
        weights = torch.full((max_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        names = list(routes_by_variant)
        samples = {name: [] for name in names}
        for iteration in range(args.warmup + args.iterations):
            rotated = names[iteration % len(names) :] + names[: iteration % len(names)]
            for name in rotated:
                values = _run_variant(
                    routes_by_variant[name],
                    hidden,
                    weights,
                    num_physical_experts=physical_counts[name],
                    original=name == _BASELINE0,
                )
                if iteration >= args.warmup:
                    samples[name].append(values)

        if rank == 0:
            summaries = {name: _summary(values) for name, values in samples.items()}
            result = {
                "metadata": {
                    "route_dir": str(args.route_dir),
                    "layers": list(layers),
                    "ep_size": ep_size,
                    "ranks_per_node": args.ranks_per_node,
                    "num_experts": args.num_experts,
                    "num_physical_slots": int(layout.numel()),
                    "top_k": top_k,
                    "timed_scope": "dispatch/preprocess plus combine; planning and remapping excluded",
                },
                "plans": plans,
                "communication_counts": {
                    name: {
                        "remote_node_token_copies": int(counts[index * 2].item()),
                        "remote_rank_token_copies": int(counts[index * 2 + 1].item()),
                    }
                    for index, name in enumerate(names)
                },
                "timings": summaries,
                "median_speedup_vs_baseline0": {
                    name: {
                        metric: summaries[_BASELINE0][metric]["median"] / summaries[name][metric]["median"]
                        for metric in _METRICS
                    }
                    for name in names
                    if name != _BASELINE0
                },
                "median_speedup_vs_baseline1": {
                    name: {
                        metric: summaries[_BASELINE1][metric]["median"] / summaries[name][metric]["median"]
                        for metric in _METRICS
                    }
                    for name in names
                    if name != _BASELINE1
                },
            }
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
