#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Measure traffic-matrix cost features across controlled redundant layouts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.greedy_planner import (
    GreedyCommunicationPlanner,
    assign_tokens_to_copies_greedy,
)
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.state import configure_hiermoe
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.utils.import_utils import is_torch_npu_available


_EVENT_NAMES = (
    "stage1_a2a",
    "stage2_a2a",
    "combine_stage2_a2a",
    "combine_stage1_a2a",
    "stage1_payload_build",
    "stage1_meta_pack",
    "stage1_split_wait",
    "stage1_meta_unpack",
    "stage2_payload_build",
    "stage2_meta_pack",
    "stage2_split_wait",
    "stage2_meta_unpack",
    "dispatch_finalize",
    "combine_stage2_accum",
    "combine_stage1_accum",
    "combine_final_accum",
)
_METRIC_NAMES = ("communication_region_ms", *_EVENT_NAMES)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--slot-increment", type=int, default=4)
    parser.add_argument("--max-copies", type=int, default=8)
    parser.add_argument(
        "--variants",
        default="owners,mirrored_r2,node_local,quarter_cyclic,hot,random",
        help="Comma-separated controlled layout variants; use 'external' with --layout-json.",
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        help="Optional replay JSON whose per-layer slot_to_logical layouts are measured as 'external'.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_routes(
    route_dir: Path,
    *,
    rank: int,
    layers: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    routes: list[torch.Tensor] = []
    bytes_per_element = 0
    for layer in range(layers):
        path = route_dir / f"layer{layer:02d}_rank{rank:02d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        route = payload["routes"].to(device=device, dtype=torch.long, non_blocking=True).contiguous()
        if route.ndim != 2:
            raise ValueError(f"Expected rank-2 routes in {path}.")
        routes.append(route)
        bytes_per_element = int(payload["bytes_per_element"])
    return routes, bytes_per_element


def _owner_layout(
    *,
    num_experts: int,
    ep_size: int,
    slot_increment: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    base = num_experts // ep_size
    slots_per_rank = base + slot_increment
    experts = torch.arange(num_experts, dtype=torch.long, device=device)
    owners = torch.div(experts, base, rounding_mode="floor") * slots_per_rank + experts.remainder(base)
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, experts)
    return layout, owners, slots_per_rank


def _fill_redundant_slots(
    owner_layout: torch.Tensor,
    *,
    variant: str,
    layer: int,
    global_expert_counts: torch.Tensor,
    ep_size: int,
    num_experts: int,
    slots_per_rank: int,
) -> torch.Tensor:
    layout = owner_layout.clone()
    base = num_experts // ep_size
    redundant = slots_per_rank - base
    if variant == "owners":
        return layout
    hot = torch.argsort(global_expert_counts, descending=True, stable=True).tolist()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260727 + int(layer))
    random_order = torch.randperm(num_experts, generator=generator).tolist()
    for rank in range(ep_size):
        used = {int(value) for value in layout[rank * slots_per_rank : rank * slots_per_rank + base].tolist()}
        for offset in range(redundant):
            if variant == "mirrored_r2":
                candidate = ((rank + ep_size // 2) % ep_size) * base + offset % base
            elif variant == "node_local":
                local = rank % 8
                candidate = (rank - local + (local + 1 + offset) % 8) * base + offset % base
            elif variant == "quarter_cyclic":
                candidate = ((rank + ep_size // 4) % ep_size) * base + offset % base
            elif variant in {"hot", "random"}:
                order = hot if variant == "hot" else random_order
                cursor = rank * redundant + offset
                candidate = int(order[cursor % len(order)])
                while candidate in used:
                    cursor += 1
                    candidate = int(order[cursor % len(order)])
            else:
                raise ValueError(f"Unknown layout variant {variant!r}.")
            layout[rank * slots_per_rank + base + offset] = candidate
            used.add(candidate)
    return layout


def _load_external_layouts(
    path: Path,
    *,
    layers: int,
    num_physical_slots: int,
    device: torch.device,
) -> list[torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_layers = payload.get("layers", {})
    if not isinstance(raw_layers, dict):
        raise ValueError(f"External layout JSON has no layer mapping: {path}.")

    indexed: dict[int, torch.Tensor] = {}
    for layer_name, layer_payload in raw_layers.items():
        match = re.search(r"\.layers\.(\d+)\.", str(layer_name))
        if match is None:
            continue
        layer = int(match.group(1))
        values = layer_payload.get("slot_to_logical") if isinstance(layer_payload, dict) else None
        if not isinstance(values, list) or len(values) != num_physical_slots:
            raise ValueError(
                f"External layout {layer_name!r} has {0 if values is None else len(values)} slots; "
                f"expected {num_physical_slots}."
            )
        indexed[layer] = torch.tensor(values, dtype=torch.long, device=device)
    missing = sorted(set(range(layers)) - set(indexed))
    if missing:
        raise ValueError(f"External layout JSON is missing layers {missing}: {path}.")
    return [indexed[layer] for layer in range(layers)]


def _physical_routes_by_variant(
    logical_routes: list[torch.Tensor],
    *,
    owner_layout: torch.Tensor,
    owners: torch.Tensor,
    slots_per_rank: int,
    hierarchy: Hierarchy,
    source_rank: int,
    num_experts: int,
    max_copies: int,
    variants: tuple[str, ...],
    external_layouts: list[torch.Tensor] | None = None,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]:
    supported = {
        "owners",
        "mirrored_r2",
        "node_local",
        "quarter_cyclic",
        "hot",
        "random",
        "external",
    }
    unknown = sorted(set(variants) - supported)
    if unknown:
        raise ValueError(f"Unknown traffic benchmark variants: {unknown}.")
    if "external" in variants and external_layouts is None:
        raise ValueError("The external benchmark variant requires --layout-json.")
    if external_layouts is not None and len(external_layouts) != len(logical_routes):
        raise ValueError("External layout count must match the logical route layer count.")
    layouts = {name: [] for name in variants}
    routes_by_variant = {name: [] for name in variants}
    for layer, logical in enumerate(logical_routes):
        local_counts = torch.bincount(logical.reshape(-1), minlength=num_experts).to(torch.float32)
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        for name in variants:
            layout = (
                external_layouts[layer]
                if name == "external" and external_layouts is not None
                else _fill_redundant_slots(
                    owner_layout,
                    variant=name,
                    layer=layer,
                    global_expert_counts=local_counts,
                    ep_size=hierarchy.ep_size,
                    num_experts=num_experts,
                    slots_per_rank=slots_per_rank,
                )
            )
            physical = assign_tokens_to_copies_greedy(
                logical,
                layout,
                slots_per_rank=slots_per_rank,
                source_ranks=source_rank,
                hierarchy_group_sizes=hierarchy.group_sizes,
                num_experts=num_experts,
                step=0,
                layer_seed=layer,
                max_copies=max_copies,
            ).contiguous()
            layouts[name].append(layout)
            routes_by_variant[name].append(physical)
    return layouts, routes_by_variant


def _traffic_feature_rows(
    routes_by_variant: dict[str, list[torch.Tensor]],
    *,
    planner: GreedyCommunicationPlanner,
    ep_size: int,
) -> dict[str, list[dict[str, float]]]:
    names = list(routes_by_variant)
    local_unique = torch.cat(
        [planner._local_packed_counts(route) for name in names for route in routes_by_variant[name]],
        dim=0,
    )
    local_assignments = torch.cat(
        [planner._local_packed_assignment_counts(route) for name in names for route in routes_by_variant[name]],
        dim=0,
    )
    source_unique = torch.empty(
        (ep_size * local_unique.shape[0], local_unique.shape[1]),
        dtype=local_unique.dtype,
        device=local_unique.device,
    )
    source_assignments = torch.empty_like(source_unique)
    dist.all_gather_into_tensor(source_unique, local_unique.contiguous())
    dist.all_gather_into_tensor(source_assignments, local_assignments.contiguous())
    source_unique = source_unique.view(ep_size, local_unique.shape[0], local_unique.shape[1])
    source_assignments = source_assignments.view(
        ep_size,
        local_assignments.shape[0],
        local_assignments.shape[1],
    )
    features = planner._hierarchical_traffic_features(source_unique, source_assignments)
    global_assignments = source_assignments.sum(dim=0)[:, :ep_size]
    features["peak_assignments"] = global_assignments.max(dim=1).values

    layer_count = len(next(iter(routes_by_variant.values())))
    output: dict[str, list[dict[str, float]]] = {}
    for variant_index, name in enumerate(names):
        rows = []
        for layer in range(layer_count):
            index = variant_index * layer_count + layer
            rows.append({feature: float(values[index].item()) for feature, values in features.items()})
        output[name] = rows
    return output


def _run_one(
    routes: torch.Tensor,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_physical_experts: int,
) -> tuple[object, object, dict[str, tuple[object, object]]]:
    dispatch_start = torch.npu.Event(enable_timing=True)
    dispatch_end = torch.npu.Event(enable_timing=True)
    combine_start = torch.npu.Event(enable_timing=True)
    combine_end = torch.npu.Event(enable_timing=True)
    dispatch_start.record()
    permuted, context, _counts = rank_dedup_dispatch(
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
    if context.internal_timing_events is None:
        raise RuntimeError("Traffic cost-model benchmark requires VEOMNI_HIERMOE_INTERNAL_TIMING=1.")
    return (
        (dispatch_start, dispatch_end),
        (combine_start, combine_end),
        context.internal_timing_events,
    )


def _measure(
    routes_by_variant: dict[str, list[torch.Tensor]],
    *,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    num_physical_experts: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, object]]]:
    names = list(routes_by_variant)
    samples = {
        name: [[[] for _metric in _METRIC_NAMES] for _layer in routes] for name, routes in routes_by_variant.items()
    }
    for iteration in range(warmup + iterations):
        rotated = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in rotated:
            events = [
                _run_one(
                    routes,
                    hidden,
                    weights,
                    num_physical_experts=num_physical_experts,
                )
                for routes in routes_by_variant[name]
            ]
            torch.npu.synchronize()
            values = []
            for dispatch, combine, internal in events:
                stage_values = [float(internal[event][0].elapsed_time(internal[event][1])) for event in _EVENT_NAMES]
                values.append(
                    [
                        float(dispatch[0].elapsed_time(dispatch[1])) + float(combine[0].elapsed_time(combine[1])),
                        *stage_values,
                    ]
                )
            global_values = torch.tensor(values, dtype=torch.float32, device=hidden.device)
            dist.all_reduce(global_values, op=dist.ReduceOp.MAX)
            if iteration >= warmup:
                for layer, row in enumerate(global_values.cpu().tolist()):
                    for metric, value in enumerate(row):
                        samples[name][layer][metric].append(float(value))
        dist.barrier()

    layer_medians = {
        name: [
            {
                metric: statistics.median(layer_samples[metric_index])
                for metric_index, metric in enumerate(_METRIC_NAMES)
            }
            for layer_samples in variant_samples
        ]
        for name, variant_samples in samples.items()
    }
    aggregate_timings: dict[str, dict[str, object]] = {}
    for name, variant_samples in samples.items():
        variant_summary: dict[str, object] = {}
        for metric_index, metric in enumerate(_METRIC_NAMES):
            iteration_totals = [
                sum(layer_samples[metric_index][iteration] for layer_samples in variant_samples)
                for iteration in range(iterations)
            ]
            median = statistics.median(iteration_totals)
            absolute_deviations = [abs(value - median) for value in iteration_totals]
            variant_summary[metric] = {
                "iteration_totals_ms": iteration_totals,
                "median_ms": median,
                "mad_ms": statistics.median(absolute_deviations),
                "min_ms": min(iteration_totals),
                "max_ms": max(iteration_totals),
            }
        aggregate_timings[name] = variant_summary
    return layer_medians, aggregate_timings


def main() -> None:
    args = _parse_args()
    if not is_torch_npu_available():
        raise RuntimeError("This benchmark requires torch_npu and Ascend devices.")
    import torch_npu  # noqa: F401

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        rank = dist.get_rank()
        ep_size = dist.get_world_size()
        device = torch.device(f"npu:{local_rank}")
        hierarchy = Hierarchy(ep_size=ep_size, group_sizes=(args.ranks_per_node, ep_size), source="traffic")
        configure_hiermoe(
            HierMoEConfig(
                enable=True,
                token_dedup=True,
                expert_swap=False,
                hierarchy_group_sizes=hierarchy.group_sizes,
            ),
            dist.group.WORLD,
        )
        logical_routes, bytes_per_element = _load_routes(
            args.route_dir,
            rank=rank,
            layers=args.layers,
            device=device,
        )
        owner_layout, owners, slots_per_rank = _owner_layout(
            num_experts=args.num_experts,
            ep_size=ep_size,
            slot_increment=args.slot_increment,
            device=device,
        )
        external_layouts = (
            None
            if args.layout_json is None
            else _load_external_layouts(
                args.layout_json,
                layers=args.layers,
                num_physical_slots=int(owner_layout.numel()),
                device=device,
            )
        )
        _layouts, routes_by_variant = _physical_routes_by_variant(
            logical_routes,
            owner_layout=owner_layout,
            owners=owners,
            slots_per_rank=slots_per_rank,
            hierarchy=hierarchy,
            source_rank=rank,
            num_experts=args.num_experts,
            max_copies=args.max_copies,
            variants=tuple(name.strip() for name in args.variants.split(",") if name.strip()),
            external_layouts=external_layouts,
        )
        planner = GreedyCommunicationPlanner(
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=args.hidden_size,
            bytes_per_element=bytes_per_element,
            slots_per_rank=slots_per_rank,
        )
        features = _traffic_feature_rows(routes_by_variant, planner=planner, ep_size=ep_size)
        max_tokens = max(int(routes.shape[0]) for routes in logical_routes)
        top_k = int(logical_routes[0].shape[1])
        hidden = torch.zeros((max_tokens, args.hidden_size), dtype=torch.bfloat16, device=device)
        weights = torch.full((max_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        timings, aggregate_timings = _measure(
            routes_by_variant,
            hidden=hidden,
            weights=weights,
            num_physical_experts=int(owner_layout.numel()),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if rank == 0:
            if args.output is None:
                raise ValueError("--output is required on global rank 0.")
            result = {
                "metadata": {
                    "ep_size": ep_size,
                    "layers": args.layers,
                    "route_dir": str(args.route_dir),
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "variants": list(routes_by_variant),
                    "layout_json": None if args.layout_json is None else str(args.layout_json),
                },
                "samples": {
                    name: [
                        {"layer": layer, **features[name][layer], **timings[name][layer]}
                        for layer in range(args.layers)
                    ]
                    for name in routes_by_variant
                },
                "aggregate_timings": aggregate_timings,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result["metadata"], sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
