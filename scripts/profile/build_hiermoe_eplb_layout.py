#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Compile an official EPLB placement into a static HierMoE replay layout.

The expert weights are measured from a shared set of cached Forward routes.
DeepSeek's EPLB chooses replica counts and packs physical replicas onto EP
ranks.  This adapter preserves those replica counts, repairs same-rank
duplicates (which are not useful to HierMoE), assigns one durable owner to
every logical expert, and compiles a source-rank LUT using the same profile.

The emitted JSON is consumable by ``VEOMNI_HIERMOE_ABLATION_REPLAY_PATH`` and
does not invoke a planner or expert migration in the measured steady state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    _layer_payload,
    _normalize_layout_for_replay,
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eplb-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--profile-steps", type=_parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument(
        "--layer-name-template",
        default="model.language_model.layers.{layer}.mlp.experts",
        help="Python format template for the runtime expert-module key.",
    )
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--primary-slots-per-rank", type=int, default=4)
    parser.add_argument("--redundant-slots-per-rank", type=int, required=True)
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument(
        "--call-indices",
        type=_parse_int_list,
        default=None,
        help=(
            "Captured Forward call indices to include for every optimizer step. "
            "Defaults to the legacy single --call-index value."
        ),
    )
    parser.add_argument(
        "--forward-repeats",
        type=int,
        default=1,
        help=(
            "Forward microbatches captured per optimizer step. Repeated "
            "forwards are stored in consecutive layer-index blocks."
        ),
    )
    parser.add_argument(
        "--capture-layer-stride",
        type=int,
        default=None,
        help="Layer-index stride between repeated Forward captures. Defaults to --layers.",
    )
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _load_eplb(root: Path) -> ModuleType:
    module_path = root / "eplb.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official EPLB implementation not found: {module_path}")
    spec = importlib.util.spec_from_file_location("deepseek_eplb", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load EPLB module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "rebalance_experts"):
        raise AttributeError("Official EPLB module does not export rebalance_experts.")
    return module


def _route_path(
    root: Path,
    *,
    step: int,
    layer: int,
    rank: int,
    call_index: int,
) -> Path:
    return (
        root
        / f"step{step:04d}"
        / f"layer{layer:02d}_call{call_index}_rank{rank:02d}.pt"
    )


def _profile_demand(args: argparse.Namespace) -> np.ndarray:
    if args.forward_repeats <= 0:
        raise ValueError("forward_repeats must be positive.")
    layer_stride = args.layers if args.capture_layer_stride is None else args.capture_layer_stride
    call_indices = (args.call_index,) if args.call_indices is None else args.call_indices
    if args.forward_repeats > 1 and layer_stride <= 0:
        raise ValueError("capture_layer_stride must be positive when forward_repeats > 1.")

    demand = np.zeros(
        (args.layers, args.ep_size, args.num_experts),
        dtype=np.float64,
    )
    for layer_offset in range(args.layers):
        layer = args.layer_start + layer_offset
        for step in args.profile_steps:
            for repeat in range(args.forward_repeats):
                capture_layer = layer + repeat * layer_stride
                for call_index in call_indices:
                    for rank in range(args.ep_size):
                        path = _route_path(
                            args.route_root,
                            step=step,
                            layer=capture_layer,
                            rank=rank,
                            call_index=call_index,
                        )
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        routes = payload["routes"] if isinstance(payload, dict) else payload
                        counts = torch.bincount(
                            routes.to(dtype=torch.long).reshape(-1),
                            minlength=args.num_experts,
                        )
                        demand[layer_offset, rank] += counts.numpy()
    demand /= float(len(args.profile_steps))
    return demand


def _repair_same_rank_duplicates(
    layout: np.ndarray,
    *,
    ep_size: int,
    slots_per_rank: int,
    expert_weight: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Swap replicas between ranks until every rank has distinct experts."""

    result = layout.reshape(ep_size, slots_per_rank).copy()
    repairs = 0
    max_repairs = int(result.size * result.size)
    while True:
        duplicate: tuple[int, int] | None = None
        for rank in range(ep_size):
            values, counts = np.unique(result[rank], return_counts=True)
            repeated = values[counts > 1]
            if repeated.size:
                expert = int(repeated[0])
                positions = np.flatnonzero(result[rank] == expert)
                duplicate = (rank, int(positions[-1]))
                break
        if duplicate is None:
            return result.reshape(-1), repairs
        source_rank, source_local = duplicate
        source_expert = int(result[source_rank, source_local])
        best: tuple[float, int, int] | None = None
        source_set = set(int(value) for value in result[source_rank])
        for target_rank in range(ep_size):
            if target_rank == source_rank or source_expert in result[target_rank]:
                continue
            for target_local, target_expert_value in enumerate(result[target_rank]):
                target_expert = int(target_expert_value)
                if target_expert in source_set:
                    continue
                score = abs(float(expert_weight[source_expert] - expert_weight[target_expert]))
                candidate = (score, target_rank, target_local)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError(
                f"Cannot repair duplicate expert {source_expert} on rank {source_rank}."
            )
        _score, target_rank, target_local = best
        result[source_rank, source_local], result[target_rank, target_local] = (
            result[target_rank, target_local],
            result[source_rank, source_local],
        )
        repairs += 1
        if repairs > max_repairs:
            raise RuntimeError("Same-rank duplicate repair did not converge.")


def _assign_owners_and_reorder(
    layout: np.ndarray,
    *,
    ep_size: int,
    slots_per_rank: int,
    primary_slots_per_rank: int,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign exactly one owner per expert with fixed per-rank owner capacity."""

    by_rank = layout.reshape(ep_size, slots_per_rank)
    lanes = np.repeat(np.arange(ep_size, dtype=np.int64), primary_slots_per_rank)
    if lanes.size != num_experts:
        raise ValueError("Primary owner capacity must equal the logical expert count.")
    high = 1_000_000.0
    cost = np.full((num_experts, num_experts), high, dtype=np.float64)
    canonical_rank = np.arange(num_experts, dtype=np.int64) // primary_slots_per_rank
    for expert in range(num_experts):
        present_ranks = set(int(rank) for rank in np.flatnonzero((by_rank == expert).any(axis=1)))
        for lane, rank in enumerate(lanes):
            if int(rank) in present_ranks:
                cost[expert, lane] = (
                    0.0 if int(rank) == int(canonical_rank[expert]) else 1.0
                ) + float(rank) * 1e-6 + float(lane) * 1e-9
    rows, columns = linear_sum_assignment(cost)
    if rows.size != num_experts or bool((cost[rows, columns] >= high).any()):
        raise RuntimeError("EPLB placement has no feasible one-owner-per-expert assignment.")
    owner_rank = np.full((num_experts,), -1, dtype=np.int64)
    owner_rank[rows] = lanes[columns]

    reordered = np.empty_like(by_rank)
    owners = np.full((num_experts,), -1, dtype=np.int64)
    for rank in range(ep_size):
        owner_experts = sorted(int(expert) for expert in np.flatnonzero(owner_rank == rank))
        if len(owner_experts) != primary_slots_per_rank:
            raise RuntimeError(f"Rank {rank} received {len(owner_experts)} owners.")
        remaining = [int(value) for value in by_rank[rank].tolist()]
        for expert in owner_experts:
            remaining.remove(expert)
        row = owner_experts + remaining
        if len(row) != slots_per_rank:
            raise RuntimeError("Owner reordering changed the number of physical slots.")
        reordered[rank] = row
        for local, expert in enumerate(owner_experts):
            owners[expert] = rank * slots_per_rank + local
    if bool((owners < 0).any()):
        raise RuntimeError("Owner assignment lost a logical expert.")
    return reordered.reshape(-1), owners


class _FlowEdge:
    __slots__ = ("capacity", "destination", "reverse")

    def __init__(self, destination: int, reverse: int, capacity: int) -> None:
        self.destination = destination
        self.reverse = reverse
        self.capacity = capacity


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    destination: int,
    capacity: int,
) -> _FlowEdge:
    forward = _FlowEdge(destination, len(graph[destination]), capacity)
    backward = _FlowEdge(source, len(graph[source]), 0)
    graph[source].append(forward)
    graph[destination].append(backward)
    return forward


def _owner_feasible_layout(
    raw_layout: np.ndarray,
    copy_counts: np.ndarray,
    *,
    ep_size: int,
    slots_per_rank: int,
    primary_slots_per_rank: int,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Preserve EPLB copy counts while satisfying durable-owner constraints."""

    redundant_slots_per_rank = slots_per_rank - primary_slots_per_rank
    if bool((copy_counts < 1).any()):
        raise RuntimeError("EPLB produced an expert without a physical copy.")
    remaining = copy_counts.astype(np.int64, copy=True) - 1
    if int(remaining.sum()) != ep_size * redundant_slots_per_rank:
        raise RuntimeError("EPLB copy counts do not match the redundant-slot budget.")

    raw_by_rank = raw_layout.reshape(ep_size, slots_per_rank)
    lanes = np.repeat(np.arange(ep_size, dtype=np.int64), primary_slots_per_rank)
    owner_cost = np.empty((num_experts, num_experts), dtype=np.float64)
    canonical_rank = np.arange(num_experts, dtype=np.int64) // primary_slots_per_rank
    for expert in range(num_experts):
        for lane, rank in enumerate(lanes):
            owner_cost[expert, lane] = (
                0.0 if expert in raw_by_rank[rank] else 100.0
            ) + (
                0.0 if int(rank) == int(canonical_rank[expert]) else 0.01
            ) + float(rank) * 1e-6 + float(lane) * 1e-9
    owner_rows, owner_columns = linear_sum_assignment(owner_cost)
    owner_rank = np.full((num_experts,), -1, dtype=np.int64)
    owner_rank[owner_rows] = lanes[owner_columns]
    if bool((owner_rank < 0).any()):
        raise RuntimeError("Cannot assign one durable owner to every EPLB expert.")

    source_node = 0
    expert_base = 1
    rank_base = expert_base + num_experts
    sink_node = rank_base + ep_size
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink_node + 1)]
    expert_rank_edges: dict[tuple[int, int], _FlowEdge] = {}
    for expert in range(num_experts):
        _add_flow_edge(graph, source_node, expert_base + expert, int(remaining[expert]))
        expert_owner_rank = int(owner_rank[expert])
        preferred = [
            rank
            for rank in range(ep_size)
            if expert in raw_by_rank[rank] and rank != expert_owner_rank
        ]
        fallback = [
            rank
            for rank in range(ep_size)
            if rank != expert_owner_rank and rank not in preferred
        ]
        for rank in preferred + fallback:
            edge = _add_flow_edge(
                graph,
                expert_base + expert,
                rank_base + rank,
                1,
            )
            expert_rank_edges[(expert, rank)] = edge
    for rank in range(ep_size):
        _add_flow_edge(
            graph,
            rank_base + rank,
            sink_node,
            redundant_slots_per_rank,
        )

    total_flow = 0
    required_flow = int(remaining.sum())
    while total_flow < required_flow:
        level = [-1] * len(graph)
        level[source_node] = 0
        queue = [source_node]
        for node in queue:
            for edge in graph[node]:
                if edge.capacity > 0 and level[edge.destination] < 0:
                    level[edge.destination] = level[node] + 1
                    queue.append(edge.destination)
        if level[sink_node] < 0:
            break
        cursor = [0] * len(graph)

        def push(node: int, available: int) -> int:
            if node == sink_node:
                return available
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                if edge.capacity > 0 and level[edge.destination] == level[node] + 1:
                    sent = push(edge.destination, min(available, edge.capacity))
                    if sent:
                        edge.capacity -= sent
                        graph[edge.destination][edge.reverse].capacity += sent
                        return sent
                cursor[node] += 1
            return 0

        while True:
            sent = push(source_node, required_flow - total_flow)
            if not sent:
                break
            total_flow += sent
    if total_flow != required_flow:
        raise RuntimeError(
            f"Cannot allocate EPLB replicas around durable owners: {total_flow}/{required_flow}."
        )

    replicas_by_rank: list[list[int]] = [[] for _ in range(ep_size)]
    for (expert, rank), edge in expert_rank_edges.items():
        if edge.capacity == 0:
            replicas_by_rank[rank].append(expert)
    layout = np.empty((ep_size, slots_per_rank), dtype=np.int64)
    owners = np.empty((num_experts,), dtype=np.int64)
    preserved = 0
    for rank in range(ep_size):
        owner_experts = sorted(int(expert) for expert in np.flatnonzero(owner_rank == rank))
        if len(owner_experts) != primary_slots_per_rank:
            raise RuntimeError(f"Rank {rank} received {len(owner_experts)} durable owners.")
        replica_experts = sorted(
            replicas_by_rank[rank],
            key=lambda expert: (
                0 if expert in raw_by_rank[rank] else 1,
                expert,
            ),
        )
        if len(replica_experts) != redundant_slots_per_rank:
            raise RuntimeError(f"Rank {rank} received {len(replica_experts)} EPLB replicas.")
        layout[rank] = owner_experts + replica_experts
        for local, expert in enumerate(owner_experts):
            owners[expert] = rank * slots_per_rank + local
        raw_counts = np.bincount(raw_by_rank[rank], minlength=num_experts)
        final_counts = np.bincount(layout[rank], minlength=num_experts)
        preserved += int(np.minimum(raw_counts, final_counts).sum())
    actual_counts = np.bincount(layout.reshape(-1), minlength=num_experts)
    if not np.array_equal(actual_counts, copy_counts):
        raise RuntimeError("Owner-feasible packing changed official EPLB copy counts.")
    preservation_fraction = preserved / float(layout.size)
    return layout.reshape(-1), owners, preservation_fraction


def _topology_distance(source: int, destination: int, ranks_per_node: int) -> int:
    if source == destination:
        return 0
    if source // ranks_per_node == destination // ranks_per_node:
        return 1
    return 2


def _compile_source_lut(
    layout: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ep_size: int,
    ranks_per_node: int,
    num_experts: int,
    slots_per_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance profiled source demand across EPLB replicas."""

    lut = np.empty((ep_size, num_experts), dtype=np.int64)
    rank_load = np.zeros((ep_size,), dtype=np.float64)
    work: list[tuple[float, int, int]] = []
    for source in range(ep_size):
        for expert in range(num_experts):
            work.append((float(demand_by_source[source, expert]), source, expert))
    work.sort(key=lambda item: (-item[0], item[1], item[2]))
    for weight, source, expert in work:
        copies = np.flatnonzero(layout == expert)
        if copies.size == 0:
            raise RuntimeError(f"EPLB layout lost expert {expert}.")
        best_slot = min(
            (int(slot) for slot in copies),
            key=lambda slot: (
                float(rank_load[slot // slots_per_rank] + weight),
                _topology_distance(source, slot // slots_per_rank, ranks_per_node),
                slot,
            ),
        )
        lut[source, expert] = best_slot
        rank_load[best_slot // slots_per_rank] += weight
    for source in range(ep_size):
        for expert in range(num_experts):
            slot = int(lut[source, expert])
            if int(layout[slot]) != expert:
                raise RuntimeError("Compiled EPLB source LUT points to the wrong logical expert.")
    return lut, rank_load


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(values.mean())
    return float(values.std() / mean) if mean else 0.0


def main() -> None:
    args = _parse_args()
    if "{layer}" not in args.layer_name_template:
        raise ValueError("--layer-name-template must contain the '{layer}' placeholder.")
    args.slots_per_rank = args.primary_slots_per_rank + args.redundant_slots_per_rank
    args.optimize_steps = args.profile_steps
    args.validation_steps = ()
    if args.num_experts != args.ep_size * args.primary_slots_per_rank:
        raise ValueError("Primary owner slots must contain every logical expert exactly once.")
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")

    started = time.perf_counter()
    eplb = _load_eplb(args.eplb_root)
    demand = _profile_demand(args)
    global_weight = demand.sum(axis=1)
    num_physical_slots = args.ep_size * args.slots_per_rank
    weight_tensor = torch.from_numpy(global_weight).to(dtype=torch.float32)
    # num_groups=1 selects official EPLB's global policy because 1 is not
    # divisible by the four physical nodes.
    physical_to_logical, _logical_to_physical, logical_count = eplb.rebalance_experts(
        weight_tensor,
        num_physical_slots,
        1,
        args.ep_size // args.ranks_per_node,
        args.ep_size,
    )

    layers: dict[str, object] = {}
    all_actions: list[dict[str, str]] = []
    layer_reports: list[dict[str, object]] = []
    for offset in range(args.layers):
        layer = args.layer_start + offset
        raw_layout = physical_to_logical[offset].to(dtype=torch.long).cpu().numpy()
        expected_counts = logical_count[offset].to(dtype=torch.long).cpu().numpy()
        ordered_layout, owners, preservation_fraction = _owner_feasible_layout(
            raw_layout,
            expected_counts,
            ep_size=args.ep_size,
            slots_per_rank=args.slots_per_rank,
            primary_slots_per_rank=args.primary_slots_per_rank,
            num_experts=args.num_experts,
        )
        actual_counts = np.bincount(ordered_layout, minlength=args.num_experts)
        lut, rank_load = _compile_source_lut(
            ordered_layout,
            demand[offset],
            ep_size=args.ep_size,
            ranks_per_node=args.ranks_per_node,
            num_experts=args.num_experts,
            slots_per_rank=args.slots_per_rank,
        )
        normalized_layout, normalized_owners, normalized_lut, actions = (
            _normalize_layout_for_replay(
                ordered_layout,
                owners,
                lut,
                args=args,
            )
        )
        layer_name = args.layer_name_template.format(layer=layer)
        layers[layer_name] = _layer_payload(
            layout=normalized_layout,
            owners=normalized_owners,
            lut=normalized_lut,
        )
        all_actions.extend({"layer": layer_name, **action} for action in actions)
        layer_reports.append(
            {
                "layer": layer,
                "official_rank_placement_preservation_fraction": preservation_fraction,
                "copy_count_min": int(actual_counts.min()),
                "copy_count_max": int(actual_counts.max()),
                "rank_assignment_max": float(rank_load.max()),
                "rank_assignment_mean": float(rank_load.mean()),
                "rank_assignment_cv": _coefficient_of_variation(rank_load),
                "replay_actions": len(actions),
            }
        )

    payload = {
        "schema_version": 2,
        "source": {
            "initial_layout": "canonical_empty",
            "algorithm": "deepseek-eplb-global-v1-source-lut-compiled",
            "official_repository": "https://github.com/deepseek-ai/EPLB",
            "route_root": str(args.route_root.resolve()),
            "profile_steps": list(args.profile_steps),
            "layer_name_template": args.layer_name_template,
            "source_lut_policy": "global-rank-load-greedy-locality-tiebreak",
        },
        "topology": {
            "ep_size": args.ep_size,
            "num_experts": args.num_experts,
            "num_physical_slots": num_physical_slots,
            "slots_per_rank": args.slots_per_rank,
        },
        "replay": {"actions_by_step": {"1": all_actions}},
        "layers": layers,
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "schema_version": 1,
        "algorithm": payload["source"]["algorithm"],
        "profile_steps": list(args.profile_steps),
        "redundant_slots_per_rank": args.redundant_slots_per_rank,
        "layers": args.layers,
        "layout_build_ms": elapsed_ms,
        "rank_assignment_cv": {
            "mean": statistics.mean(float(row["rank_assignment_cv"]) for row in layer_reports),
            "maximum": max(float(row["rank_assignment_cv"]) for row in layer_reports),
        },
        "rank_assignment_max": {
            "mean": statistics.mean(float(row["rank_assignment_max"]) for row in layer_reports),
            "maximum": max(float(row["rank_assignment_max"]) for row in layer_reports),
        },
        "copy_count_min": min(int(row["copy_count_min"]) for row in layer_reports),
        "copy_count_max": max(int(row["copy_count_max"]) for row in layer_reports),
        "official_rank_placement_preservation_fraction": statistics.mean(
            float(row["official_rank_placement_preservation_fraction"])
            for row in layer_reports
        ),
        "replay_actions": len(all_actions),
        "layer_results": layer_reports,
    }
    args.output_layout.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_layout.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "layer_results"}, indent=2))


if __name__ == "__main__":
    main()
