#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Probe exhaustive Cover-only improvement for one initialized MoE layer.

Every round evaluates all legal source-expert/destination-slot Covers against
the same cached route samples. Candidate evaluation is parallel across CPU
workers, while rounds remain sequential because committing one Cover changes
the marginal value and legality of the next round.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    _HybridEvaluator,
    _load_routes,
)
from scripts.profile.build_hiermoe_recursive_classifier_layout import _parse_int_list
from scripts.profile.refine_hiermoe_online_cover import (
    _CoverAction,
    _LayerState,
    _load_state,
    _patch_cover_state,
)


_WORKER_STATE: _LayerState | None = None
_WORKER_OPTIMIZE_SAMPLES: list[list[object]] | None = None
_WORKER_VALIDATION_SAMPLES: list[list[object]] | None = None
_WORKER_EVALUATOR: _HybridEvaluator | None = None
_WORKER_ARGS: argparse.Namespace | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(2, 3, 4, 5))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(6, 7))
    parser.add_argument("--minimum-gain-ms", type=float, default=0.0)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--service-group-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _legal_covers(state: _LayerState, args: argparse.Namespace) -> list[_CoverAction]:
    layout = state.layout
    copy_counts = np.bincount(layout[layout >= 0], minlength=args.num_experts)
    candidates: list[_CoverAction] = []
    for destination, victim in enumerate(layout.tolist()):
        if victim < 0 or int(copy_counts[victim]) <= 1:
            continue
        target_rank = destination // args.slots_per_rank
        rank_experts = layout[
            target_rank * args.slots_per_rank : (target_rank + 1) * args.slots_per_rank
        ]
        for source in range(args.num_experts):
            if source == victim or int(copy_counts[source]) >= args.max_copies:
                continue
            if bool((rank_experts == source).any()):
                continue
            candidates.append(
                _CoverAction(
                    source_logical=source,
                    source_slot=int(state.owners[source]),
                    destination_slot=destination,
                    victim_logical=victim,
                    target_rank=target_rank,
                    proxy_score=0.0,
                )
            )
    return candidates


def _score_action(action: _CoverAction) -> tuple[float, _CoverAction]:
    assert _WORKER_STATE is not None
    assert _WORKER_OPTIMIZE_SAMPLES is not None
    assert _WORKER_VALIDATION_SAMPLES is not None
    assert _WORKER_EVALUATOR is not None
    assert _WORKER_ARGS is not None
    candidate = _patch_cover_state(
        _WORKER_STATE,
        action,
        optimize_samples=_WORKER_OPTIMIZE_SAMPLES,
        validation_samples=_WORKER_VALIDATION_SAMPLES,
        evaluator=_WORKER_EVALUATOR,
        args=_WORKER_ARGS,
        evaluate_validation=False,
    )
    return candidate.optimize_cost.total_ms, action


def _install_worker_state(
    state: _LayerState,
    optimize_samples: list[list[object]],
    validation_samples: list[list[object]],
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
) -> None:
    global _WORKER_STATE
    global _WORKER_OPTIMIZE_SAMPLES
    global _WORKER_VALIDATION_SAMPLES
    global _WORKER_EVALUATOR
    global _WORKER_ARGS
    _WORKER_STATE = state
    _WORKER_OPTIMIZE_SAMPLES = optimize_samples
    _WORKER_VALIDATION_SAMPLES = validation_samples
    _WORKER_EVALUATOR = evaluator
    _WORKER_ARGS = args


def _cost_mean(cost: Any, samples: int) -> dict[str, float]:
    divisor = max(1, int(samples))
    return {
        "communication_ms": float(cost.communication_ms) / divisor,
        "compute_ms": float(cost.compute_ms) / divisor,
        "total_ms": float(cost.total_ms) / divisor,
    }


def main() -> None:
    args = _parse_args()
    if args.rounds <= 0 or args.workers <= 0 or args.chunksize <= 0:
        raise ValueError("rounds, workers, and chunksize must be positive.")
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("The exhaustive Cover probe requires the fork multiprocessing start method.")
    torch.set_num_threads(1)
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    optimize_samples = _load_routes(
        args.route_root,
        steps=args.optimize_steps,
        layer=args.layer,
        ep_size=args.ep_size,
    )
    validation_samples = _load_routes(
        args.route_root,
        steps=args.validation_steps,
        layer=args.layer,
        ep_size=args.ep_size,
    )
    evaluator = _HybridEvaluator(args)
    state = _load_state(
        payload,
        layer=args.layer,
        optimize_samples=optimize_samples,
        validation_samples=validation_samples,
        evaluator=evaluator,
        args=args,
    )
    initial = state
    rounds: list[dict[str, object]] = []
    started = time.perf_counter()
    for round_index in range(args.rounds):
        candidates = _legal_covers(state, args)
        if not candidates:
            break
        _install_worker_state(
            state,
            optimize_samples,
            validation_samples,
            evaluator,
            args,
        )
        best_cost = float("inf")
        best_action: _CoverAction | None = None
        best_key: tuple[float, int, int] | None = None
        round_started = time.perf_counter()
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.workers, len(candidates))) as pool:
            for candidate_cost, action in pool.imap_unordered(
                _score_action,
                candidates,
                chunksize=args.chunksize,
            ):
                key = (
                    float(candidate_cost),
                    int(action.destination_slot),
                    int(action.source_logical),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_cost = float(candidate_cost)
                    best_action = action

        baseline_total = float(state.optimize_cost.total_ms)
        baseline_validation = float(state.validation_cost.total_ms)
        mean_gain = (baseline_total - best_cost) / max(1, len(args.optimize_steps))
        accepted = best_action is not None and mean_gain > float(args.minimum_gain_ms)
        if accepted:
            assert best_action is not None
            state = _patch_cover_state(
                state,
                best_action,
                optimize_samples=optimize_samples,
                validation_samples=validation_samples,
                evaluator=evaluator,
                args=args,
                evaluate_validation=True,
            )
        rounds.append(
            {
                "round": round_index + 1,
                "candidate_count": len(candidates),
                "accepted": accepted,
                "action": None if best_action is None else asdict(best_action),
                "optimize_gain_mean_ms": mean_gain if accepted else 0.0,
                "validation_gain_mean_ms": (
                    0.0
                    if not accepted
                    else (
                        baseline_validation
                        - float(state.validation_cost.total_ms)
                    )
                    / max(1, len(args.validation_steps))
                ),
                "optimize_cost_mean": _cost_mean(
                    state.optimize_cost,
                    len(args.optimize_steps),
                ),
                "validation_cost_mean": _cost_mean(
                    state.validation_cost,
                    len(args.validation_steps),
                ),
                "round_wall_ms": (time.perf_counter() - round_started) * 1000.0,
            }
        )
        if not accepted:
            break

    result = {
        "schema_version": 1,
        "algorithm": "single-layer-exhaustive-cover-oracle-v1",
        "input_layout": str(args.input_layout.resolve()),
        "route_root": str(args.route_root.resolve()),
        "layer": args.layer,
        "optimize_steps": list(args.optimize_steps),
        "validation_steps": list(args.validation_steps),
        "round_limit": args.rounds,
        "workers": args.workers,
        "initial_optimize_cost_mean": _cost_mean(
            initial.optimize_cost,
            len(args.optimize_steps),
        ),
        "final_optimize_cost_mean": _cost_mean(
            state.optimize_cost,
            len(args.optimize_steps),
        ),
        "initial_validation_cost_mean": _cost_mean(
            initial.validation_cost,
            len(args.validation_steps),
        ),
        "final_validation_cost_mean": _cost_mean(
            state.validation_cost,
            len(args.validation_steps),
        ),
        "accepted_covers": sum(bool(row["accepted"]) for row in rounds),
        "rounds": rounds,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "final_layout": state.layout.tolist(),
        "final_owner_slots": state.owners.tolist(),
        "final_source_logical_to_physical": state.lut.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {
        "final_layout",
        "final_owner_slots",
        "final_source_logical_to_physical",
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
