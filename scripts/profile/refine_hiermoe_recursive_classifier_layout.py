#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Refine an existing HierMoE layout with one reusable hierarchy primitive.

This is an offline, winner-only refinement.  It deliberately does not repeat
the expensive seed/replica/node-library search.  For each selected layer it:

1. replays the incumbent source LUT to construct mapped instance statistics;
2. applies the same capacity-preserving co-occurrence/peak-load partition
   primitive first across nodes and then across ranks inside each node;
3. recompiles the source LUT with the calibrated communication/assignment
   coefficients;
4. accepts a state only when the full hybrid evaluator improves the optimize
   route, while reserving the held-out route for the final 48-layer E2E gate.

Repeating this coordinate update realizes:
``layout -> LUT -> mapped statistics -> layout`` without widening the search
back to all initialization candidates.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    HybridCost,
    _HybridEvaluator,
    _load_routes,
    _replay_payload,
)
from scripts.profile.build_hiermoe_recursive_classifier_layout import (
    _mapped_instance_statistics,
    _materialize_layout,
    _optimize_lut_instances,
    _parse_int_list,
    _preloaded_replay_payload,
    _refine_balanced_partition,
    _source_statistics,
)


@dataclass(frozen=True)
class _RefinementState:
    strategy: str
    logical_instances: np.ndarray
    instance_ranks: np.ndarray
    lut_instances: np.ndarray
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray
    optimize_cost: HybridCost
    validation_cost: HybridCost


@dataclass(frozen=True)
class _GeometryProposal:
    strategy: str
    instance_ranks: np.ndarray
    node_swaps: int
    rank_swaps: int
    node_proxy_cost: float
    rank_proxy_cost: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(1,))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(2,))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--node-max-swaps", type=int, default=32)
    parser.add_argument("--rank-max-swaps", type=int, default=16)
    parser.add_argument("--lut-iterations", type=int, default=6)
    parser.add_argument("--ep-size", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--slots-per-rank", type=int, default=None)
    parser.add_argument("--primary-slots-per-rank", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _layer_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _load_incumbent(
    payload: dict[str, object],
    *,
    layer: int,
    optimize_samples: list[list[torch.Tensor]],
    validation_samples: list[list[torch.Tensor]],
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
) -> _RefinementState:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Input layout has no layer table.")
    row = layers.get(_layer_name(layer))
    if not isinstance(row, dict):
        raise ValueError(f"Input layout has no state for layer {layer}.")
    layout = np.asarray(row["slot_to_logical"], dtype=np.int64)
    owners = np.asarray(row["owner_slots"], dtype=np.int64)
    lut = np.asarray(row["source_logical_to_physical"], dtype=np.int64)
    expected_slots = args.ep_size * args.slots_per_rank
    if layout.shape != (expected_slots,):
        raise ValueError(f"Layer {layer} has an unexpected physical layout shape.")
    if lut.shape != (args.ep_size, args.num_experts):
        raise ValueError(f"Layer {layer} has an unexpected source LUT shape.")
    if not bool((layout[lut] == np.arange(args.num_experts, dtype=np.int64)[None, :]).all()):
        raise ValueError(f"Layer {layer} has an invalid source LUT.")
    return _RefinementState(
        strategy="input_winner",
        logical_instances=layout.copy(),
        instance_ranks=np.arange(expected_slots, dtype=np.int64) // args.slots_per_rank,
        lut_instances=lut.copy(),
        layout=layout,
        owners=owners,
        lut=lut,
        optimize_cost=evaluator.evaluate(optimize_samples, lut),
        validation_cost=evaluator.evaluate(validation_samples, lut),
    )


def _cost_coefficients(args: argparse.Namespace) -> tuple[float, float, float]:
    payload_bytes = float(args.hidden_size * args.bytes_per_element)
    node_affinity = float(args.communication_phase_multiplier) * payload_bytes * float(args.inter_ms_per_byte)
    rank_affinity = float(args.communication_phase_multiplier) * payload_bytes * float(args.intra_ms_per_byte)
    assignment = float(args.compute_phase_multiplier) * float(args.compute_ms_per_assignment) + float(
        args.communication_phase_multiplier
    ) * float(args.route_ms_per_assignment)
    return node_affinity, rank_affinity, assignment


def _refine_rank_geometry(
    instance_ranks: np.ndarray,
    logical_instances: np.ndarray,
    demand: np.ndarray,
    affinity: np.ndarray,
    *,
    rank_affinity_ms_per_hit: float,
    assignment_ms_per_assignment: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, float]:
    result = instance_ranks.copy()
    global_loads = np.bincount(
        result,
        weights=demand,
        minlength=args.ep_size,
    ).astype(np.float64)
    swaps = 0
    proxy_cost = 0.0
    for node in range(args.ep_size // args.ranks_per_node):
        members = np.flatnonzero(result // args.ranks_per_node == node)
        if len(members) != args.ranks_per_node * args.slots_per_rank:
            raise RuntimeError("Node membership does not match its physical capacity.")
        node_ranks = np.arange(
            node * args.ranks_per_node,
            (node + 1) * args.ranks_per_node,
            dtype=np.int64,
        )
        other_ranks = np.setdiff1d(
            np.arange(args.ep_size, dtype=np.int64),
            node_ranks,
            assume_unique=True,
        )
        peak_floor = float(global_loads[other_ranks].max(initial=0.0))
        local = _refine_balanced_partition(
            affinity[np.ix_(members, members)],
            demand[members],
            result[members] - node * args.ranks_per_node,
            parts=args.ranks_per_node,
            capacity=args.slots_per_rank,
            affinity_ms_per_hit=rank_affinity_ms_per_hit,
            assignment_ms_per_assignment=assignment_ms_per_assignment,
            peak_floor=peak_floor,
            max_swaps=args.rank_max_swaps,
            item_kinds=logical_instances[members],
            forbid_duplicate_kinds=True,
        )
        result[members] = node * args.ranks_per_node + local.labels
        swaps += len(local.swaps)
        proxy_cost += local.proxy_cost
        global_loads[node_ranks] = np.bincount(
            local.labels,
            weights=demand[members],
            minlength=args.ranks_per_node,
        )
    return result, swaps, proxy_cost


def _feasible_rank_seed(
    instance_nodes: np.ndarray,
    preferred_ranks: np.ndarray,
    logical_instances: np.ndarray,
    demand: np.ndarray,
    affinity: np.ndarray,
    *,
    args: argparse.Namespace,
) -> np.ndarray:
    """Assign moved node instances to legal rank lanes before refinement."""

    result = np.full_like(instance_nodes, -1)
    for node in range(args.ep_size // args.ranks_per_node):
        members = np.flatnonzero(instance_nodes == node)
        capacities = np.full((args.ranks_per_node,), args.slots_per_rank, dtype=np.int64)
        loads = np.zeros((args.ranks_per_node,), dtype=np.float64)
        lane_members: list[list[int]] = [[] for _ in range(args.ranks_per_node)]
        active_logical = logical_instances[members]
        active_logical = active_logical[active_logical >= 0]
        multiplicity = np.bincount(
            active_logical,
            minlength=args.num_experts,
        )
        order = sorted(
            members.tolist(),
            key=lambda item: (
                -int(multiplicity[int(logical_instances[item])]) if int(logical_instances[item]) >= 0 else 0,
                -float(demand[item]),
                item,
            ),
        )
        for instance in order:
            logical = int(logical_instances[instance])
            preferred_lane = int(preferred_ranks[instance] % args.ranks_per_node)
            best: tuple[tuple[float, int, float, int], int] | None = None
            for lane in range(args.ranks_per_node):
                if capacities[lane] <= 0:
                    continue
                if logical >= 0 and any(int(logical_instances[other]) == logical for other in lane_members[lane]):
                    continue
                projected_peak = max(
                    float(loads.max(initial=0.0)),
                    loads[lane] + float(demand[instance]),
                )
                affinity_gain = sum(float(affinity[instance, other]) for other in lane_members[lane])
                key = (
                    projected_peak,
                    int(lane != preferred_lane),
                    -affinity_gain,
                    lane,
                )
                if best is None or key < best[0]:
                    best = (key, lane)
            if best is None:
                raise RuntimeError("Could not construct a duplicate-free rank seed.")
            lane = best[1]
            result[instance] = node * args.ranks_per_node + lane
            capacities[lane] -= 1
            loads[lane] += float(demand[instance])
            lane_members[lane].append(instance)
    if bool((result < 0).any()):
        raise RuntimeError("Rank seed left an instance unassigned.")
    return result


def _geometry_proposals(
    state: _RefinementState,
    instance_demand: np.ndarray,
    instance_affinity: np.ndarray,
    *,
    args: argparse.Namespace,
) -> list[_GeometryProposal]:
    demand = instance_demand.sum(axis=0)
    affinity = instance_affinity.sum(axis=0)
    node_affinity, rank_affinity, assignment = _cost_coefficients(args)
    current_ranks = state.instance_ranks

    rank_only, rank_swaps, rank_proxy = _refine_rank_geometry(
        current_ranks,
        state.logical_instances,
        demand,
        affinity,
        rank_affinity_ms_per_hit=rank_affinity,
        assignment_ms_per_assignment=assignment,
        args=args,
    )
    proposals = [
        _GeometryProposal(
            strategy="rank_only",
            instance_ranks=rank_only,
            node_swaps=0,
            rank_swaps=rank_swaps,
            node_proxy_cost=0.0,
            rank_proxy_cost=rank_proxy,
        )
    ]

    num_nodes = args.ep_size // args.ranks_per_node
    node = _refine_balanced_partition(
        affinity,
        demand,
        current_ranks // args.ranks_per_node,
        parts=num_nodes,
        capacity=args.ranks_per_node * args.slots_per_rank,
        affinity_ms_per_hit=node_affinity,
        assignment_ms_per_assignment=assignment,
        assignment_divisor=float(args.ranks_per_node),
        max_swaps=args.node_max_swaps,
    )
    node_seed_ranks = _feasible_rank_seed(
        node.labels,
        current_ranks,
        state.logical_instances,
        demand,
        affinity,
        args=args,
    )
    if not np.array_equal(
        node_seed_ranks // args.ranks_per_node,
        node.labels,
    ):
        raise RuntimeError("Node refinement did not preserve the exchanged rank lanes.")
    hierarchical, hierarchical_rank_swaps, hierarchical_rank_proxy = _refine_rank_geometry(
        node_seed_ranks,
        state.logical_instances,
        demand,
        affinity,
        rank_affinity_ms_per_hit=rank_affinity,
        assignment_ms_per_assignment=assignment,
        args=args,
    )
    proposals.append(
        _GeometryProposal(
            strategy="node_then_rank",
            instance_ranks=hierarchical,
            node_swaps=len(node.swaps),
            rank_swaps=hierarchical_rank_swaps,
            node_proxy_cost=node.proxy_cost,
            rank_proxy_cost=hierarchical_rank_proxy,
        )
    )
    return proposals


def _calibrated_lut(
    state: _RefinementState,
    instance_ranks: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    *,
    args: argparse.Namespace,
) -> np.ndarray:
    node_affinity, rank_affinity, assignment = _cost_coefficients(args)
    total_affinity = max(float(affinity_by_source.sum()), 1.0)
    total_demand = max(float(demand_by_source.sum()), 1.0)
    return _optimize_lut_instances(
        state.logical_instances,
        instance_ranks,
        state.lut_instances,
        demand_by_source,
        affinity_by_source,
        ranks_per_node=args.ranks_per_node,
        iterations=args.lut_iterations,
        node_weight=node_affinity * total_affinity,
        rank_weight=rank_affinity * total_affinity,
        assignment_weight=assignment * total_demand,
    )


def _refine_layer(
    payload: dict[str, object],
    *,
    layer: int,
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
) -> tuple[_RefinementState, dict[str, object]]:
    started = time.perf_counter()
    optimize_samples = _load_routes(
        args.route_root,
        steps=args.optimize_steps,
        layer=layer,
        ep_size=args.ep_size,
    )
    validation_samples = _load_routes(
        args.route_root,
        steps=args.validation_steps,
        layer=layer,
        ep_size=args.ep_size,
    )
    demand_by_source, affinity_by_source = _source_statistics(
        optimize_samples,
        num_experts=args.num_experts,
    )
    incumbent = _load_incumbent(
        payload,
        layer=layer,
        optimize_samples=optimize_samples,
        validation_samples=validation_samples,
        evaluator=evaluator,
        args=args,
    )
    initial = incumbent
    statistics_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    initial_route_key = incumbent.instance_ranks[incumbent.lut_instances].tobytes()
    optimize_cache: dict[bytes, HybridCost] = {
        initial_route_key: incumbent.optimize_cost,
    }
    validation_cache: dict[bytes, HybridCost] = {
        initial_route_key: incumbent.validation_cost,
    }
    round_rows: list[dict[str, object]] = []

    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        statistics_started = time.perf_counter()
        statistics_key = incumbent.lut_instances.tobytes()
        if statistics_key not in statistics_cache:
            statistics_cache[statistics_key] = _mapped_instance_statistics(
                optimize_samples,
                incumbent.lut_instances,
                instances=len(incumbent.logical_instances),
            )
        instance_demand, instance_affinity = statistics_cache[statistics_key]
        statistics_ms = (time.perf_counter() - statistics_started) * 1000.0

        proposal_started = time.perf_counter()
        geometries = _geometry_proposals(
            incumbent,
            instance_demand,
            instance_affinity,
            args=args,
        )
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        candidates: list[tuple[_RefinementState, dict[str, object]]] = []
        seen: set[bytes] = set()
        exact_started = time.perf_counter()
        for geometry in geometries:
            lut_variants = [
                ("reuse_lut", incumbent.lut_instances),
                (
                    "calibrated_lut",
                    _calibrated_lut(
                        incumbent,
                        geometry.instance_ranks,
                        demand_by_source,
                        affinity_by_source,
                        args=args,
                    ),
                ),
            ]
            for lut_strategy, lut_instances in lut_variants:
                route_key = geometry.instance_ranks[lut_instances].tobytes()
                if route_key in seen:
                    continue
                seen.add(route_key)
                try:
                    layout, owners, lut = _materialize_layout(
                        incumbent.logical_instances,
                        geometry.instance_ranks,
                        lut_instances,
                        demand_by_source,
                        ep_size=args.ep_size,
                        slots_per_rank=args.slots_per_rank,
                        primary_slots_per_rank=args.primary_slots_per_rank,
                        num_experts=args.num_experts,
                    )
                except RuntimeError:
                    continue
                if route_key not in optimize_cache:
                    optimize_cache[route_key] = evaluator.evaluate(optimize_samples, lut)
                optimize_cost = optimize_cache[route_key]
                validation_cost = validation_cache.get(
                    route_key,
                    HybridCost(
                        communication_ms=float("inf"),
                        compute_ms=float("inf"),
                        total_ms=float("inf"),
                        peak_communication_rank=-1,
                        peak_compute_rank=-1,
                        mean_destination_nodes=float("inf"),
                        mean_destination_ranks=float("inf"),
                        peak_assignments=float("inf"),
                    ),
                )
                candidate = _RefinementState(
                    strategy=f"round{round_index + 1}_{geometry.strategy}_{lut_strategy}",
                    logical_instances=incumbent.logical_instances,
                    instance_ranks=geometry.instance_ranks.copy(),
                    lut_instances=lut_instances.copy(),
                    layout=layout,
                    owners=owners,
                    lut=lut,
                    optimize_cost=optimize_cost,
                    validation_cost=validation_cost,
                )
                candidates.append(
                    (
                        candidate,
                        {
                            "strategy": candidate.strategy,
                            "node_swaps": geometry.node_swaps,
                            "rank_swaps": geometry.rank_swaps,
                            "node_proxy_cost": geometry.node_proxy_cost,
                            "rank_proxy_cost": geometry.rank_proxy_cost,
                            "optimize": asdict(optimize_cost),
                            "validation": (
                                None if not np.isfinite(validation_cost.total_ms) else asdict(validation_cost)
                            ),
                        },
                    )
                )
        exact_ms = (time.perf_counter() - exact_started) * 1000.0
        feasible = [
            row for row in candidates if row[0].optimize_cost.total_ms < incumbent.optimize_cost.total_ms - 1e-9
        ]
        accepted = min(feasible, key=lambda row: row[0].optimize_cost.total_ms) if feasible else None
        if accepted is not None:
            accepted_state, accepted_metadata = accepted
            accepted_key = accepted_state.instance_ranks[accepted_state.lut_instances].tobytes()
            if accepted_key not in validation_cache:
                validation_cache[accepted_key] = evaluator.evaluate(
                    validation_samples,
                    accepted_state.lut,
                )
            accepted_state = replace(
                accepted_state,
                validation_cost=validation_cache[accepted_key],
            )
            accepted_metadata["validation"] = asdict(accepted_state.validation_cost)
            accepted = (accepted_state, accepted_metadata)
            incumbent = accepted_state
        round_rows.append(
            {
                "round": round_index + 1,
                "accepted": None if accepted is None else accepted[0].strategy,
                "statistics_ms": statistics_ms,
                "proposal_ms": proposal_ms,
                "exact_ms": exact_ms,
                "round_ms": (time.perf_counter() - round_started) * 1000.0,
                "candidates": [row[1] for row in candidates],
                "incumbent_optimize": asdict(incumbent.optimize_cost),
                "incumbent_validation": asdict(incumbent.validation_cost),
            }
        )
        if accepted is None:
            break

    row = {
        "layer": layer,
        "strategy": incumbent.strategy,
        "accepted_rounds": sum(item["accepted"] is not None for item in round_rows),
        "initial": {
            "optimize": asdict(initial.optimize_cost),
            "validation": asdict(initial.validation_cost),
        },
        "optimize": asdict(incumbent.optimize_cost),
        "validation": asdict(incumbent.validation_cost),
        "optimize_gain_ms": initial.optimize_cost.total_ms - incumbent.optimize_cost.total_ms,
        "validation_gain_ms": initial.validation_cost.total_ms - incumbent.validation_cost.total_ms,
        "rounds": round_rows,
        "planner_ms": (time.perf_counter() - started) * 1000.0,
        "mapped_statistics_builds": len(statistics_cache),
        "exact_route_evaluations": len(optimize_cache),
        "validation_route_evaluations": len(validation_cache),
    }
    return incumbent, row


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    topology = payload.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("Input layout has no topology metadata.")
    for argument, topology_key in (
        ("ep_size", "ep_size"),
        ("num_experts", "num_experts"),
        ("slots_per_rank", "slots_per_rank"),
    ):
        if getattr(args, argument) is None:
            setattr(args, argument, int(topology[topology_key]))
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")
    if args.num_experts % args.ep_size:
        raise ValueError("Logical experts must divide evenly across EP ranks.")
    expected_primary = args.num_experts // args.ep_size
    if args.primary_slots_per_rank is None:
        args.primary_slots_per_rank = expected_primary
    elif args.primary_slots_per_rank != expected_primary:
        raise ValueError("Primary slots per rank must equal num_experts / ep_size.")
    expected = {
        "ep_size": args.ep_size,
        "num_experts": args.num_experts,
        "num_physical_slots": args.ep_size * args.slots_per_rank,
        "slots_per_rank": args.slots_per_rank,
    }
    if any(int(topology.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("Input layout topology does not match the refinement configuration.")

    evaluator = _HybridEvaluator(args)
    layouts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    luts: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for layer in range(args.layer_start, args.layer_start + args.layers):
        state, row = _refine_layer(
            payload,
            layer=layer,
            evaluator=evaluator,
            args=args,
        )
        layouts.append(state.layout)
        owners.append(state.owners)
        luts.append(state.lut)
        rows.append(row)
        print(
            f"layer={layer:02d} accepted_rounds={row['accepted_rounds']} "
            f"validation_gain_ms={row['validation_gain_ms']:.6f} "
            f"planner_ms={row['planner_ms']:.1f}",
            flush=True,
        )

    initial_validation = sum(float(row["initial"]["validation"]["total_ms"]) for row in rows)
    final_validation = sum(float(row["validation"]["total_ms"]) for row in rows)
    report = {
        "schema_version": 1,
        "algorithm": "recursive-classifier-unified-refinement-v2",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output_layout", "output_report"}
        },
        "layers": rows,
        "aggregate": {
            "layers": len(rows),
            "accepted_layers": sum(int(row["accepted_rounds"]) > 0 for row in rows),
            "accepted_rounds": sum(int(row["accepted_rounds"]) for row in rows),
            "initial_optimize_ms": sum(float(row["initial"]["optimize"]["total_ms"]) for row in rows),
            "final_optimize_ms": sum(float(row["optimize"]["total_ms"]) for row in rows),
            "initial_validation_ms": initial_validation,
            "final_validation_ms": final_validation,
            "validation_gain_ms": initial_validation - final_validation,
            "validation_speedup": initial_validation / final_validation,
            "planner_total_ms": sum(float(row["planner_ms"]) for row in rows),
            "planner_mean_ms_per_layer": sum(float(row["planner_ms"]) for row in rows) / len(rows),
            "mapped_statistics_builds": sum(int(row["mapped_statistics_builds"]) for row in rows),
            "exact_route_evaluations": sum(int(row["exact_route_evaluations"]) for row in rows),
            "validation_route_evaluations": sum(int(row["validation_route_evaluations"]) for row in rows),
            "e2e_eligible": bool(
                args.layer_start == 0 and args.layers == 48 and final_validation <= initial_validation
            ),
        },
    }
    replay_builder = (
        _preloaded_replay_payload if any(bool((layout < 0).any()) for layout in layouts) else _replay_payload
    )
    replay = replay_builder(
        layouts=layouts,
        owners=owners,
        luts=luts,
        args=args,
        algorithm="recursive-classifier-unified-refinement-v2",
    )
    for path, value in ((args.output_layout, replay), (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
