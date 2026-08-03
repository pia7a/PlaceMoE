#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Replay low-width online Covers after a static HierMoE initialization.

The initializer already emits a physical layout and a source-rank routing
LUT. This tool leaves that initialization untouched and simulates the online
phase:

1. every destination rank emits one inexpensive Cover proposal (M=1);
2. the complete hybrid cost model exactly replays those proposals;
3. the globally best strictly positive proposal is committed per layer;
4. the physical layout, owner slots, and source LUT are updated together.

The proposal heuristic is deliberately narrow. Exact replay, rather than the
heuristic score, is the acceptance oracle. Optimization and held-out routes
are reported separately so a locally improving Cover cannot be mistaken for
a generalization result.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    HybridCost,
    _HybridEvaluator,
    _load_routes,
)
from scripts.profile.placemoe_planner import (
    _parse_int_list,
    _source_statistics,
)


@dataclass(frozen=True)
class _CoverAction:
    source_logical: int
    source_slot: int
    destination_slot: int
    victim_logical: int
    target_rank: int
    proxy_score: float

    def body(self) -> str:
        return f"{self.source_logical}->{self.destination_slot}"


@dataclass(frozen=True)
class _LayerState:
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray
    optimize_cost: HybridCost
    validation_cost: HybridCost


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(2,))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(3, 4, 5, 6, 7))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--service-group-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--minimum-gain-ms", type=float, default=0.0)
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


def _cost_coefficients(args: argparse.Namespace) -> tuple[float, float, float]:
    payload_bytes = float(args.hidden_size * args.bytes_per_element)
    node_affinity = float(args.communication_phase_multiplier) * payload_bytes * float(args.inter_ms_per_byte)
    rank_affinity = float(args.communication_phase_multiplier) * payload_bytes * float(args.intra_ms_per_byte)
    assignment = float(args.compute_phase_multiplier) * float(args.compute_ms_per_assignment) + float(
        args.communication_phase_multiplier
    ) * float(args.route_ms_per_assignment)
    return node_affinity, rank_affinity, assignment


def _validate_state(
    layout: np.ndarray,
    owners: np.ndarray,
    lut: np.ndarray,
    *,
    args: argparse.Namespace,
) -> None:
    expected_slots = args.ep_size * args.slots_per_rank
    if layout.shape != (expected_slots,):
        raise ValueError(f"Unexpected layout shape {layout.shape}, expected ({expected_slots},).")
    if owners.shape != (args.num_experts,):
        raise ValueError("owner_slots has an unexpected shape.")
    if lut.shape != (args.ep_size, args.num_experts):
        raise ValueError("source_logical_to_physical has an unexpected shape.")
    logical = np.arange(args.num_experts, dtype=np.int64)
    if np.any(owners < 0) or not np.array_equal(layout[owners], logical):
        raise ValueError("Owner slots do not reference their logical experts.")
    if np.any(lut < 0) or not np.array_equal(layout[lut], np.broadcast_to(logical, lut.shape)):
        raise ValueError("Source LUT does not reference the requested logical experts.")
    counts = np.bincount(layout[layout >= 0], minlength=args.num_experts)
    if np.any(counts < 1):
        raise ValueError("The physical layout loses at least one logical expert.")
    if np.any(counts > args.max_copies):
        raise ValueError("The physical layout exceeds max_copies.")


def _load_state(
    payload: dict[str, object],
    *,
    layer: int,
    optimize_samples: list[list[object]],
    validation_samples: list[list[object]],
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
) -> _LayerState:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Input layout has no layer table.")
    row = layers.get(_layer_name(layer))
    if not isinstance(row, dict):
        raise ValueError(f"Input layout has no state for layer {layer}.")
    layout = np.asarray(row["slot_to_logical"], dtype=np.int64)
    owners = np.asarray(row["owner_slots"], dtype=np.int64)
    lut = np.asarray(row["source_logical_to_physical"], dtype=np.int64)
    _validate_state(layout, owners, lut, args=args)
    return _LayerState(
        layout=layout,
        owners=owners,
        lut=lut,
        optimize_cost=evaluator.evaluate(optimize_samples, lut),
        validation_cost=evaluator.evaluate(validation_samples, lut),
    )


def _slot_service_loads(
    lut: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    num_slots: int,
) -> np.ndarray:
    loads = np.zeros((num_slots,), dtype=np.float64)
    np.add.at(
        loads,
        lut.reshape(-1),
        demand_by_source.reshape(-1),
    )
    return loads


def _patch_cover_state(
    state: _LayerState,
    action: _CoverAction,
    *,
    optimize_samples: list[list[object]],
    validation_samples: list[list[object]],
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
    evaluate_validation: bool = True,
) -> _LayerState:
    layout = state.layout.copy()
    owners = state.owners.copy()
    lut = state.lut.copy()
    source = int(action.source_logical)
    destination = int(action.destination_slot)
    victim = int(action.victim_logical)
    if int(layout[destination]) != victim or int(layout[int(owners[source])]) != source:
        raise ValueError("Cover action does not match the current physical layout.")

    layout[destination] = source
    if int(owners[victim]) == destination:
        remaining = np.flatnonzero(layout == victim)
        if remaining.size == 0:
            raise ValueError("Cover action removes the final victim copy.")
        owners[victim] = int(remaining.min())
    victim_fallback = int(owners[victim])

    # Both updates are vectorized. The victim column falls back only where it
    # referenced the overwritten slot; the inserted expert is retargeted for
    # every source rank in the destination service group.
    lut[:, victim] = np.where(
        lut[:, victim] == destination,
        victim_fallback,
        lut[:, victim],
    )
    destination_rank = destination // args.slots_per_rank
    service_start = (destination_rank // args.service_group_size) * args.service_group_size
    lut[service_start : service_start + args.service_group_size, source] = destination
    _validate_state(layout, owners, lut, args=args)
    return _LayerState(
        layout=layout,
        owners=owners,
        lut=lut,
        optimize_cost=evaluator.evaluate(optimize_samples, lut),
        validation_cost=(
            evaluator.evaluate(validation_samples, lut) if evaluate_validation else state.validation_cost
        ),
    )


def _rank_cover_proposals(
    state: _LayerState,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[_CoverAction, ...]:
    """Return one vectorized proxy winner for every target rank."""

    layout = state.layout
    lut = state.lut
    ep_size = args.ep_size
    experts = args.num_experts
    slots_per_rank = args.slots_per_rank
    ranks_per_node = args.ranks_per_node
    copy_counts = np.bincount(layout[layout >= 0], minlength=experts)
    slot_loads = _slot_service_loads(
        lut,
        demand_by_source,
        num_slots=layout.size,
    )
    routed_ranks = lut // slots_per_rank
    routed_nodes = routed_ranks // ranks_per_node
    node_weight, rank_weight, assignment_weight = _cost_coefficients(args)
    proposals: list[_CoverAction] = []

    for target_rank in range(ep_size):
        start = target_rank * slots_per_rank
        stop = start + slots_per_rank
        slots = np.arange(start, stop, dtype=np.int64)
        victims = layout[slots]
        eligible = (victims >= 0) & (copy_counts[victims] > 1)
        if not np.any(eligible):
            continue
        eligible_slots = slots[eligible]
        victim_slot = int(eligible_slots[np.argmin(slot_loads[eligible_slots], axis=0)])
        victim = int(layout[victim_slot])
        victim_load = float(slot_loads[victim_slot])

        service_start = (target_rank // args.service_group_size) * args.service_group_size
        service_sources = np.arange(
            service_start,
            service_start + args.service_group_size,
            dtype=np.int64,
        )
        target_node = target_rank // ranks_per_node
        node_reward = np.zeros((experts,), dtype=np.float64)
        rank_reward = np.zeros((experts,), dtype=np.float64)
        incoming = demand_by_source[service_sources].sum(axis=0)
        for source_rank in service_sources.tolist():
            affinity = affinity_by_source[source_rank]
            node_membership = routed_nodes[source_rank] == target_node
            rank_membership = routed_ranks[source_rank] == target_rank
            node_reward += affinity[:, node_membership].sum(axis=1)
            rank_reward += affinity[:, rank_membership].sum(axis=1)
            if source_rank // ranks_per_node == target_node:
                node_reward += demand_by_source[source_rank]
            if source_rank == target_rank:
                rank_reward += demand_by_source[source_rank]

        score = (
            node_weight * node_reward
            + rank_weight * rank_reward
            - assignment_weight * np.maximum(0.0, incoming - victim_load)
        )
        service_slot_start = service_start * slots_per_rank
        service_slot_stop = (service_start + args.service_group_size) * slots_per_rank
        experts_in_service_group = layout[service_slot_start:service_slot_stop]
        invalid = copy_counts >= args.max_copies
        invalid |= np.isin(
            np.arange(experts),
            experts_in_service_group,
        )
        invalid[victim] = True
        score[invalid] = -np.inf
        source = int(np.argmax(score))
        if not np.isfinite(score[source]):
            continue
        proposals.append(
            _CoverAction(
                source_logical=source,
                source_slot=int(state.owners[source]),
                destination_slot=victim_slot,
                victim_logical=victim,
                target_rank=target_rank,
                proxy_score=float(score[source]),
            )
        )
    return tuple(proposals)


def _refine_layer(
    layer: int,
    *,
    payload: dict[str, object],
    args: argparse.Namespace,
) -> tuple[int, _LayerState, dict[str, object], list[dict[str, object]]]:
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
    evaluator = _HybridEvaluator(args)
    state = _load_state(
        payload,
        layer=layer,
        optimize_samples=optimize_samples,
        validation_samples=validation_samples,
        evaluator=evaluator,
        args=args,
    )
    demand, affinity = _source_statistics(
        optimize_samples,
        num_experts=args.num_experts,
    )
    initial = state
    actions: list[dict[str, object]] = []
    rounds: list[dict[str, object]] = []
    started = time.perf_counter()
    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        proposals = _rank_cover_proposals(
            state,
            demand,
            affinity,
            args=args,
        )
        best_state: _LayerState | None = None
        best_action: _CoverAction | None = None
        for action in proposals:
            candidate = _patch_cover_state(
                state,
                action,
                optimize_samples=optimize_samples,
                validation_samples=validation_samples,
                evaluator=evaluator,
                args=args,
                evaluate_validation=False,
            )
            if best_state is None or candidate.optimize_cost.total_ms < best_state.optimize_cost.total_ms:
                best_state = candidate
                best_action = action
        baseline_optimize = state.optimize_cost.total_ms
        accepted = (
            best_state is not None
            and best_action is not None
            and best_state.optimize_cost.total_ms < baseline_optimize - float(args.minimum_gain_ms)
        )
        if accepted:
            assert best_state is not None and best_action is not None
            best_state = replace(
                best_state,
                validation_cost=evaluator.evaluate(
                    validation_samples,
                    best_state.lut,
                ),
            )
            optimize_gain = baseline_optimize - best_state.optimize_cost.total_ms
            validation_gain = state.validation_cost.total_ms - best_state.validation_cost.total_ms
            state = best_state
            action_row = {
                "body": best_action.body(),
                "kind": "replica",
                "layer": _layer_name(layer),
            }
            actions.append(action_row)
        else:
            optimize_gain = 0.0
            validation_gain = 0.0
        best_candidate_gain = 0.0 if best_state is None else baseline_optimize - best_state.optimize_cost.total_ms
        rounds.append(
            {
                "round": round_index + 1,
                "accepted": accepted,
                "action": None if best_action is None else asdict(best_action),
                "proposals": len(proposals),
                "best_candidate_gain_ms": best_candidate_gain,
                "optimize_gain_ms": optimize_gain,
                "validation_gain_ms": validation_gain,
                "optimize_total_ms": state.optimize_cost.total_ms,
                "validation_total_ms": state.validation_cost.total_ms,
                "round_ms": (time.perf_counter() - round_started) * 1000.0,
            }
        )
        if not accepted:
            break

    report = {
        "layer": layer,
        "initial_optimize": asdict(initial.optimize_cost),
        "final_optimize": asdict(state.optimize_cost),
        "initial_validation": asdict(initial.validation_cost),
        "final_validation": asdict(state.validation_cost),
        "accepted_covers": len(actions),
        "rounds": rounds,
        "planner_ms": (time.perf_counter() - started) * 1000.0,
    }
    return layer, state, report, actions


def _mean_cost(total: float, samples: int) -> float:
    return float(total) / max(1, int(samples))


def main() -> None:
    args = _parse_args()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if args.service_group_size <= 0 or args.ep_size % args.service_group_size:
        raise ValueError("service_group_size must be a positive divisor of ep_size.")
    if args.ranks_per_node <= 0 or args.ep_size % args.ranks_per_node:
        raise ValueError("ranks_per_node must be a positive divisor of ep_size.")
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    topology = payload.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("Input layout has no topology metadata.")
    expected_topology = {
        "ep_size": args.ep_size,
        "num_experts": args.num_experts,
        "slots_per_rank": args.slots_per_rank,
    }
    for key, expected in expected_topology.items():
        actual = int(topology.get(key, -1))
        if actual != expected:
            raise ValueError(f"Input layout {key}={actual}, expected {expected}.")

    layer_ids = tuple(range(args.layer_start, args.layer_start + args.layers))
    worker = partial(_refine_layer, payload=payload, args=args)
    wall_started = time.perf_counter()
    if args.workers == 1:
        results = [worker(layer) for layer in layer_ids]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(layer_ids))) as executor:
            results = list(executor.map(worker, layer_ids))
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    results.sort(key=lambda row: row[0])

    output_layers = dict(payload["layers"])
    reports: list[dict[str, object]] = []
    actions_by_round: dict[str, list[dict[str, object]]] = {}
    for layer, state, report, actions in results:
        output_layers[_layer_name(layer)] = {
            "owner_slots": state.owners.tolist(),
            "slot_to_logical": state.layout.tolist(),
            "source_logical_to_physical": state.lut.tolist(),
        }
        reports.append(report)
        for round_index, action in enumerate(actions, start=1):
            actions_by_round.setdefault(str(round_index), []).append(action)

    input_replay = payload.get("replay", {})
    input_actions = dict(input_replay.get("actions_by_step", {})) if isinstance(input_replay, dict) else {}
    numeric_steps = [int(step) for step in input_actions] if input_actions else [0]
    first_online_step = max(numeric_steps) + 1
    for round_text, actions in actions_by_round.items():
        input_actions[str(first_online_step + int(round_text) - 1)] = actions

    output_layout = {
        "schema_version": 1,
        "source": {
            "algorithm": "recursive-classifier-online-cover-m1-v1",
            "input_layout": str(args.input_layout.resolve()),
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
            "rounds": args.rounds,
            "proposal_topk_per_rank": 1,
            "service_group_size": args.service_group_size,
        },
        "topology": dict(topology),
        "replay": {
            "actions_by_step": input_actions,
        },
        "layers": output_layers,
    }
    initial_optimize = sum(
        float(report["initial_optimize"]["total_ms"])  # type: ignore[index]
        for report in reports
    )
    final_optimize = sum(
        float(report["final_optimize"]["total_ms"])  # type: ignore[index]
        for report in reports
    )
    initial_validation = sum(
        float(report["initial_validation"]["total_ms"])  # type: ignore[index]
        for report in reports
    )
    final_validation = sum(
        float(report["final_validation"]["total_ms"])  # type: ignore[index]
        for report in reports
    )
    optimize_samples = len(args.optimize_steps)
    validation_samples = len(args.validation_steps)
    initial_optimize_mean = _mean_cost(initial_optimize, optimize_samples)
    final_optimize_mean = _mean_cost(final_optimize, optimize_samples)
    initial_validation_mean = _mean_cost(initial_validation, validation_samples)
    final_validation_mean = _mean_cost(final_validation, validation_samples)
    report_payload = {
        "schema_version": 1,
        "algorithm": "recursive-classifier-online-cover-m1-v1",
        "input_layout": str(args.input_layout.resolve()),
        "route_root": str(args.route_root.resolve()),
        "optimize_steps": list(args.optimize_steps),
        "validation_steps": list(args.validation_steps),
        "rounds": args.rounds,
        "proposal_topk_per_rank": 1,
        "aggregate": {
            "layers": len(reports),
            "accepted_covers": sum(int(report["accepted_covers"]) for report in reports),
            "changed_layers": sum(int(report["accepted_covers"]) > 0 for report in reports),
            "initial_optimize_mean_ms": initial_optimize_mean,
            "final_optimize_mean_ms": final_optimize_mean,
            "optimize_gain_ms": initial_optimize_mean - final_optimize_mean,
            "optimize_speedup": initial_optimize_mean / final_optimize_mean,
            "initial_validation_mean_ms": initial_validation_mean,
            "final_validation_mean_ms": final_validation_mean,
            "validation_gain_ms": initial_validation_mean - final_validation_mean,
            "validation_speedup": initial_validation_mean / final_validation_mean,
            "planner_total_ms": sum(float(report["planner_ms"]) for report in reports),
            "planner_wall_ms": wall_ms,
        },
        "layers": reports,
    }
    args.output_layout.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_layout.write_text(
        json.dumps(output_layout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
