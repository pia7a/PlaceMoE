"""Replay real EP32 routes through HierMoE dedup with fixed two-copy experts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.state import configure_hiermoe


def _load_planner_functions():
    planner_source = os.getenv("HIERMOE_PLANNER_SOURCE")
    if not planner_source:
        from veomni.distributed.moe.hiermoe.planner import assign_tokens_to_mirrored_r2

        return assign_tokens_to_mirrored_r2

    module_name = "veomni.distributed.moe.hiermoe.planner"
    spec = importlib.util.spec_from_file_location(module_name, planner_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load production planner source from {planner_source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.assign_tokens_to_mirrored_r2


assign_tokens_to_mirrored_r2 = _load_planner_functions()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_routes(route_dir: Path, rank: int, layers: int, device: torch.device) -> list[torch.Tensor]:
    routes: list[torch.Tensor] = []
    for layer in range(layers):
        path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "veomni.hiermoe.local_route" or int(payload.get("version", -1)) != 1:
            raise ValueError(f"Unsupported route capture: {path}")
        if int(payload["global_rank"]) != rank or int(payload["layer"]) != layer:
            raise ValueError(f"Route metadata mismatch: {path}")
        route = payload["routes"].to(dtype=torch.long)
        if route.ndim != 2:
            raise ValueError(f"Expected rank-2 routes in {path}, got {tuple(route.shape)}")
        sorted_route = route.sort(dim=-1).values
        if route.shape[1] > 1 and bool((sorted_route[:, 1:] == sorted_route[:, :-1]).any().item()):
            raise ValueError(f"Captured gate top-k contains duplicate logical experts: {path}")
        routes.append(route.to(device=device, non_blocking=True).contiguous())
    return routes


def _fixed_r2_copy_slots(num_experts: int, ep_size: int, device: torch.device) -> tuple[torch.Tensor, int]:
    half_ep = ep_size // 2
    if ep_size % 2 != 0 or num_experts % half_ep != 0:
        raise ValueError(
            f"Fixed R2 requires even EP and experts divisible by half EP, got EP={ep_size}, E={num_experts}"
        )
    slots_per_rank = num_experts // half_ep
    logical = torch.arange(num_experts, dtype=torch.long, device=device)
    rank_in_half = torch.div(logical, slots_per_rank, rounding_mode="floor")
    local_slot = torch.remainder(logical, slots_per_rank)
    first_copy_slot = rank_in_half * slots_per_rank + local_slot
    second_copy_slot = (half_ep + rank_in_half) * slots_per_rank + local_slot
    return torch.stack((first_copy_slot, second_copy_slot), dim=-1), slots_per_rank


def _choice_masks(top_k: int, device: torch.device) -> torch.Tensor:
    combinations = 1 << top_k
    masks = torch.arange(combinations, dtype=torch.long, device=device).view(-1, 1)
    bits = torch.arange(top_k, dtype=torch.long, device=device).view(1, -1)
    return torch.bitwise_and(torch.bitwise_right_shift(masks, bits), 1).to(torch.bool)


def _count_sorted_remote_unique(values: torch.Tensor, source: int) -> torch.Tensor:
    remote = values != int(source)
    first = torch.ones_like(remote)
    first[..., 1:] = values[..., 1:] != values[..., :-1]
    return (remote & first).sum(dim=-1)


def _exact_remap(
    routes: torch.Tensor,
    copy_slots: torch.Tensor,
    *,
    slots_per_rank: int,
    source_rank: int,
    ranks_per_node: int,
    chunk_size: int,
) -> torch.Tensor:
    num_tokens, top_k = routes.shape
    masks = _choice_masks(top_k, routes.device)
    num_masks = int(masks.shape[0])
    mask_ids = torch.arange(num_masks, dtype=torch.long, device=routes.device).view(1, num_masks)
    output = torch.empty_like(routes)
    source_node = source_rank // ranks_per_node

    for start in range(0, num_tokens, chunk_size):
        stop = min(num_tokens, start + chunk_size)
        chunk = routes[start:stop]
        pairs = copy_slots.index_select(0, chunk.reshape(-1)).view(stop - start, top_k, 2)
        candidates = torch.where(
            masks.view(1, num_masks, top_k),
            pairs[:, None, :, 1],
            pairs[:, None, :, 0],
        )
        candidate_ranks = torch.div(candidates, slots_per_rank, rounding_mode="floor")
        sorted_ranks = candidate_ranks.sort(dim=-1).values
        sorted_nodes = torch.div(sorted_ranks, ranks_per_node, rounding_mode="floor")
        remote_nodes = _count_sorted_remote_unique(sorted_nodes, source_node)
        remote_ranks = _count_sorted_remote_unique(sorted_ranks, source_rank)
        score = remote_nodes * (top_k + 1) + remote_ranks

        token_ids = torch.arange(start, stop, dtype=torch.long, device=routes.device).view(-1, 1)
        tie_target = torch.remainder(token_ids * 1_000_003 + source_rank * 65_537, num_masks)
        tie_key = torch.remainder(mask_ids - tie_target, num_masks)
        best = (score * num_masks + tie_key).argmin(dim=-1)
        chosen_mask = masks.index_select(0, best)
        output[start:stop] = torch.where(chosen_mask, pairs[..., 1], pairs[..., 0])
    return output


def _communication_score(
    physical_slots: torch.Tensor,
    *,
    slots_per_rank: int,
    source_rank: int,
    ranks_per_node: int,
) -> tuple[int, int]:
    remote_nodes, remote_ranks = _per_token_communication_score(
        physical_slots,
        slots_per_rank=slots_per_rank,
        source_rank=source_rank,
        ranks_per_node=ranks_per_node,
    )
    return int(remote_nodes.sum().item()), int(remote_ranks.sum().item())


def _per_token_communication_score(
    physical_slots: torch.Tensor,
    *,
    slots_per_rank: int,
    source_rank: int,
    ranks_per_node: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = torch.div(physical_slots, slots_per_rank, rounding_mode="floor").sort(dim=-1).values
    nodes = torch.div(ranks, ranks_per_node, rounding_mode="floor")
    return (
        _count_sorted_remote_unique(nodes, source_rank // ranks_per_node),
        _count_sorted_remote_unique(ranks, source_rank),
    )


def _validate_remap(
    logical_routes: torch.Tensor,
    physical_routes: torch.Tensor,
    copy_slots: torch.Tensor,
) -> None:
    copies = copy_slots.index_select(0, logical_routes.reshape(-1)).view(*logical_routes.shape, 2)
    legal = (physical_routes == copies[..., 0]) | (physical_routes == copies[..., 1])
    if not bool(legal.all().item()):
        raise AssertionError("R2 remap selected a slot that does not hold the logical expert")


def _run_layer(
    routes: torch.Tensor,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_physical_experts: int,
) -> tuple[torch.npu.Event, torch.npu.Event, torch.npu.Event, torch.npu.Event]:
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


def _run_variant(
    routes_by_layer: list[torch.Tensor],
    hidden: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_physical_experts: int,
) -> tuple[float, float, float]:
    torch.npu.synchronize()
    dist.barrier()
    started = time.perf_counter()
    events: list[tuple[torch.npu.Event, torch.npu.Event, torch.npu.Event, torch.npu.Event]] = []
    for routes in routes_by_layer:
        events.append(
            _run_layer(
                routes,
                hidden,
                weights,
                num_physical_experts=num_physical_experts,
            )
        )
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    dispatch_ms = sum(float(dispatch_start.elapsed_time(dispatch_end)) for dispatch_start, dispatch_end, _, _ in events)
    combine_ms = sum(float(combine_start.elapsed_time(combine_end)) for _, _, combine_start, combine_end in events)
    dist.barrier()
    values = torch.tensor([wall_ms, dispatch_ms, combine_ms], dtype=torch.float32, device=hidden.device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return tuple(float(value) for value in values.cpu().tolist())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    names = ("wall_ms", "dispatch_ms", "combine_ms")
    return {
        name: {
            "median": float(statistics.median(row[index] for row in values)),
            "p90": _percentile([row[index] for row in values], 0.90),
            "max": max(row[index] for row in values),
        }
        for index, name in enumerate(names)
    }


def _run_vectorized_remap(
    routes: list[torch.Tensor],
    copy_slots: torch.Tensor,
    *,
    source_rank: int,
    ep_size: int,
) -> tuple[list[torch.Tensor], tuple[float, float]]:
    torch.npu.synchronize()
    dist.barrier()
    started = time.perf_counter()
    event_start = torch.npu.Event(enable_timing=True)
    event_end = torch.npu.Event(enable_timing=True)
    event_start.record()
    remapped = [
        assign_tokens_to_mirrored_r2(
            route,
            copy_slots,
            source_ranks=source_rank,
            num_ranks=ep_size,
        )
        for route in routes
    ]
    event_end.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    event_ms = float(event_start.elapsed_time(event_end))
    cross_rank = torch.tensor((wall_ms, event_ms), dtype=torch.float32, device=routes[0].device)
    dist.all_reduce(cross_rank, op=dist.ReduceOp.MAX)
    return remapped, (float(cross_rank[0].item()), float(cross_rank[1].item()))


def _remap_summary(values: list[tuple[float, float]], layers: int) -> dict[str, dict[str, float]]:
    names = ("external_ms", "accelerator_ms")
    result = {}
    for index, name in enumerate(names):
        samples = [row[index] for row in values]
        result[name] = {
            "median": float(statistics.median(samples)),
            "p90": _percentile(samples, 0.90),
            "max": max(samples),
            "per_layer_median": float(statistics.median(samples)) / layers,
            "per_layer_p90": _percentile(samples, 0.90) / layers,
        }
    return result


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    ep_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")
    configure_hiermoe(
        HierMoEConfig(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            hierarchy_group_sizes=[args.ranks_per_node, ep_size],
        ),
        dist.group.WORLD,
    )

    routes = _load_routes(args.route_dir, rank, args.layers, device)
    num_experts = 128
    top_k = int(routes[0].shape[1])
    if any(route.shape[1] != top_k for route in routes):
        raise ValueError("Every captured layer must use the same top-k")
    copy_slots, slots_per_rank = _fixed_r2_copy_slots(num_experts, ep_size, device)

    for _ in range(args.warmup):
        _run_vectorized_remap(
            routes,
            copy_slots,
            source_rank=rank,
            ep_size=ep_size,
        )
    remap_results = []
    r2_routes = routes
    for _ in range(args.iterations):
        r2_routes, remap_timing = _run_vectorized_remap(
            routes,
            copy_slots,
            source_rank=rank,
            ep_size=ep_size,
        )
        remap_results.append(remap_timing)
    base_slots = num_experts // ep_size
    for route, r2_route in zip(routes, r2_routes, strict=True):
        _validate_remap(route, r2_route, copy_slots)

    baseline_node_count = 0
    baseline_rank_count = 0
    r2_node_count = 0
    r2_rank_count = 0
    for baseline_route, r2_route in zip(routes, r2_routes, strict=True):
        baseline_physical = (
            torch.div(baseline_route, base_slots, rounding_mode="floor") * base_slots
            + torch.remainder(baseline_route, base_slots)
        )
        base_nodes, base_ranks = _communication_score(
            baseline_physical,
            slots_per_rank=base_slots,
            source_rank=rank,
            ranks_per_node=args.ranks_per_node,
        )
        new_nodes, new_ranks = _communication_score(
            r2_route,
            slots_per_rank=slots_per_rank,
            source_rank=rank,
            ranks_per_node=args.ranks_per_node,
        )
        baseline_node_count += base_nodes
        baseline_rank_count += base_ranks
        r2_node_count += new_nodes
        r2_rank_count += new_ranks
    counts = torch.tensor(
        [baseline_node_count, baseline_rank_count, r2_node_count, r2_rank_count],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    max_tokens = max(int(route.shape[0]) for route in routes)
    hidden_size = 2048
    hidden = torch.zeros((max_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    weights = torch.full((max_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

    for index in range(args.warmup):
        if index % 2 == 0:
            _run_variant(routes, hidden, weights, num_physical_experts=num_experts)
            _run_variant(r2_routes, hidden, weights, num_physical_experts=num_experts * 2)
        else:
            _run_variant(r2_routes, hidden, weights, num_physical_experts=num_experts * 2)
            _run_variant(routes, hidden, weights, num_physical_experts=num_experts)

    baseline_results: list[tuple[float, float, float]] = []
    r2_results: list[tuple[float, float, float]] = []
    for index in range(args.iterations):
        if index % 2 == 0:
            baseline_results.append(_run_variant(routes, hidden, weights, num_physical_experts=num_experts))
            r2_results.append(_run_variant(r2_routes, hidden, weights, num_physical_experts=num_experts * 2))
        else:
            r2_results.append(_run_variant(r2_routes, hidden, weights, num_physical_experts=num_experts * 2))
            baseline_results.append(_run_variant(routes, hidden, weights, num_physical_experts=num_experts))

    if rank == 0:
        baseline_summary = _summary(baseline_results)
        r2_summary = _summary(r2_results)
        remap_summary = _remap_summary(remap_results, args.layers)
        result = {
            "ep_size": ep_size,
            "layers": args.layers,
            "top_k": top_k,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "remap": remap_summary,
            "remap_cross_rank_max_ms": remap_summary["external_ms"]["median"],
            "remap_per_layer_ms": remap_summary["external_ms"]["per_layer_median"],
            "communication_counts": {
                "baseline_remote_nodes": int(counts[0].item()),
                "baseline_remote_ranks": int(counts[1].item()),
                "r2_remote_nodes": int(counts[2].item()),
                "r2_remote_ranks": int(counts[3].item()),
            },
            "baseline": baseline_summary,
            "static_r2": r2_summary,
            "wall_speedup_median": baseline_summary["wall_ms"]["median"] / r2_summary["wall_ms"]["median"],
            "dispatch_speedup_median": baseline_summary["dispatch_ms"]["median"]
            / r2_summary["dispatch_ms"]["median"],
            "combine_speedup_median": baseline_summary["combine_ms"]["median"] / r2_summary["combine_ms"]["median"],
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
