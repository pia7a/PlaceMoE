#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Refine a static HierMoE layout with affinity-group Cover actions.

Each physical rank in the initialized layout is treated as one rank-level
expert class. A group Cover copies the complete source class over a target
class, provided every logical expert still has at least one physical copy.
Only source-rank LUT columns changed by the action are replayed. Within those
columns, only token rows containing a changed expert are recomputed, while
their complete top-k destinations remain visible so multi-expert dedup
interactions stay exact.

All group candidates are ranked on a deterministic token sample. The top
shortlist is scored with the complete optimization routes, and every accepted
winner is checked against the existing full hybrid replay before commit.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    HybridCost,
    _HybridEvaluator,
    _load_routes,
)
from scripts.profile.build_hiermoe_recursive_classifier_layout import (
    _parse_int_list,
    _preloaded_replay_payload,
)


@dataclass(frozen=True)
class _GroupCoverAction:
    source_rank: int
    target_rank: int
    service_group_size: int
    inserted_experts: tuple[int, ...]
    evicted_experts: tuple[int, ...]

    def body(self) -> str:
        return f"rank{self.source_rank}->rank{self.target_rank}/g{self.service_group_size}"


@dataclass(frozen=True)
class _GroupCoverCandidate:
    action: _GroupCoverAction
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray


@dataclass(frozen=True)
class _LayerState:
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray


@dataclass
class _SourceRouteCache:
    logical: np.ndarray
    physical_ranks: np.ndarray
    expert_rows: tuple[np.ndarray, ...]
    scale: float


@dataclass
class _SampleRouteCache:
    sources: list[_SourceRouteCache]
    endpoint: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(0, 1, 2))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(3,))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument(
        "--service-group-sizes",
        type=_parse_int_list,
        default=(1, 8),
        help="Source-rank ranges retargeted to an inserted rank class.",
    )
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--proxy-token-limit", type=int, default=2048)
    parser.add_argument("--exact-shortlist", type=int, default=32)
    parser.add_argument("--minimum-gain-ms", type=float, default=0.0)
    parser.add_argument("--anchor-atol-ms", type=float, default=2e-3)
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


def _expert_token_rows(logical: np.ndarray, num_experts: int) -> tuple[np.ndarray, ...]:
    flat = logical.reshape(-1)
    token_ids = np.repeat(np.arange(logical.shape[0], dtype=np.int64), logical.shape[1])
    order = np.argsort(flat, kind="stable")
    sorted_experts = flat[order]
    sorted_tokens = token_ids[order]
    boundaries = np.searchsorted(sorted_experts, np.arange(num_experts + 1), side="left")
    return tuple(
        np.unique(sorted_tokens[boundaries[expert] : boundaries[expert + 1]]) for expert in range(num_experts)
    )


def _endpoint_from_ranks(
    physical_ranks: np.ndarray,
    *,
    source_rank: int,
    ep_size: int,
    ranks_per_node: int,
) -> np.ndarray:
    """Return exact two-stage endpoint statistics for selected token rows."""

    result = np.zeros((8 * ep_size,), dtype=np.float64)
    if physical_ranks.size == 0:
        return result
    num_nodes = ep_size // ranks_per_node
    tokens, top_k = physical_ranks.shape
    token_ids = np.repeat(np.arange(tokens, dtype=np.int64), top_k)

    rank_hits = np.zeros((tokens, ep_size), dtype=bool)
    rank_hits[token_ids, physical_ranks.reshape(-1)] = True
    unique_rank = rank_hits.sum(axis=0, dtype=np.float64)
    assignment_rank = np.bincount(
        physical_ranks.reshape(-1),
        minlength=ep_size,
    ).astype(np.float64, copy=False)

    physical_nodes = physical_ranks // ranks_per_node
    node_hits = np.zeros((tokens, num_nodes), dtype=bool)
    node_hits[token_ids, physical_nodes.reshape(-1)] = True
    unique_node = node_hits.sum(axis=0, dtype=np.float64)
    assignment_node = np.bincount(
        physical_nodes.reshape(-1),
        minlength=num_nodes,
    ).astype(np.float64, copy=False)

    lane = source_rank % ranks_per_node
    result[source_rank] = unique_node.sum()
    result[ep_size + lane * num_nodes : ep_size + (lane + 1) * num_nodes] = unique_node
    result[2 * ep_size + source_rank] = assignment_node.sum()
    result[3 * ep_size + lane * num_nodes : 3 * ep_size + (lane + 1) * num_nodes] = assignment_node

    node_lanes = np.arange(num_nodes, dtype=np.int64) * ranks_per_node + lane
    result[4 * ep_size + node_lanes] = unique_rank.reshape(num_nodes, ranks_per_node).sum(axis=1)
    result[5 * ep_size : 6 * ep_size] = unique_rank
    result[6 * ep_size + node_lanes] = assignment_rank.reshape(num_nodes, ranks_per_node).sum(axis=1)
    result[7 * ep_size : 8 * ep_size] = assignment_rank
    return result


class _AffectedRouteEvaluator:
    """Exact hybrid scorer that replays only rows touched by LUT changes."""

    def __init__(
        self,
        samples: list[list[torch.Tensor]],
        source_lut: np.ndarray,
        *,
        evaluator: _HybridEvaluator,
        args: argparse.Namespace,
        token_limit: int | None = None,
    ) -> None:
        self.args = args
        self.evaluator = evaluator
        self.lut = source_lut.copy()
        self.samples = [self._build_sample(sample, token_limit=token_limit) for sample in samples]

    def _build_sample(
        self,
        sample: list[torch.Tensor],
        *,
        token_limit: int | None,
    ) -> _SampleRouteCache:
        sources: list[_SourceRouteCache] = []
        endpoint = np.zeros((8 * self.args.ep_size,), dtype=np.float64)
        for source_rank, logical_tensor in enumerate(sample):
            original = logical_tensor.numpy().astype(np.int64, copy=False)
            if token_limit is not None and 0 < token_limit < len(original):
                indices = np.linspace(0, len(original) - 1, num=token_limit, dtype=np.int64)
                logical = original[indices]
                scale = len(original) / len(logical)
            else:
                logical = original
                scale = 1.0
            physical_slots = self.lut[source_rank][logical]
            physical_ranks = physical_slots // self.args.slots_per_rank
            endpoint += scale * _endpoint_from_ranks(
                physical_ranks,
                source_rank=source_rank,
                ep_size=self.args.ep_size,
                ranks_per_node=self.args.ranks_per_node,
            )
            sources.append(
                _SourceRouteCache(
                    logical=logical,
                    physical_ranks=physical_ranks.copy(),
                    expert_rows=_expert_token_rows(logical, self.args.num_experts),
                    scale=scale,
                )
            )
        return _SampleRouteCache(sources=sources, endpoint=endpoint)

    def _source_patch(
        self,
        source: _SourceRouteCache,
        *,
        source_rank: int,
        new_lut: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        changed = np.flatnonzero(self.lut[source_rank] != new_lut[source_rank])
        if changed.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.int64), 0
        nonempty = [source.expert_rows[int(expert)] for expert in changed if source.expert_rows[int(expert)].size]
        if not nonempty:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.int64), 0
        rows = np.unique(np.concatenate(nonempty))
        logical = source.logical[rows]
        before = source.physical_ranks[rows]
        changed_mask = np.isin(logical, changed)
        after = np.where(
            changed_mask,
            (new_lut[source_rank] // self.args.slots_per_rank)[logical],
            before,
        )
        return rows, after, int(len(rows))

    def _candidate_endpoints(self, new_lut: np.ndarray) -> tuple[np.ndarray, int]:
        endpoints: list[np.ndarray] = []
        affected_tokens = 0
        for sample in self.samples:
            endpoint = sample.endpoint.copy()
            for source_rank, source in enumerate(sample.sources):
                rows, after, affected = self._source_patch(
                    source,
                    source_rank=source_rank,
                    new_lut=new_lut,
                )
                if not affected:
                    continue
                before = source.physical_ranks[rows]
                endpoint += source.scale * (
                    _endpoint_from_ranks(
                        after,
                        source_rank=source_rank,
                        ep_size=self.args.ep_size,
                        ranks_per_node=self.args.ranks_per_node,
                    )
                    - _endpoint_from_ranks(
                        before,
                        source_rank=source_rank,
                        ep_size=self.args.ep_size,
                        ranks_per_node=self.args.ranks_per_node,
                    )
                )
                affected_tokens += affected
            endpoints.append(endpoint)
        return np.stack(endpoints, axis=0), affected_tokens

    def cost(self, new_lut: np.ndarray | None = None) -> tuple[HybridCost, int]:
        if new_lut is None or np.array_equal(new_lut, self.lut):
            endpoints = np.stack([sample.endpoint for sample in self.samples], axis=0)
            affected_tokens = 0
        else:
            endpoints, affected_tokens = self._candidate_endpoints(new_lut)
        details = self.evaluator.planner._traffic_endpoint_cost_details(
            torch.from_numpy(endpoints.astype(np.float32, copy=False))
        )
        communication = float(details[0].sum().item())
        compute = float(details[1].sum().item())
        return (
            HybridCost(
                communication_ms=communication,
                compute_ms=compute,
                total_ms=communication + compute,
                peak_communication_rank=int(details[3][-1].item()),
                peak_compute_rank=int(details[4][-1].item()),
                mean_destination_nodes=0.0,
                mean_destination_ranks=0.0,
                peak_assignments=float(endpoints[:, 7 * self.args.ep_size :].max()),
            ),
            affected_tokens,
        )

    def commit(self, new_lut: np.ndarray) -> int:
        affected_tokens = 0
        for sample in self.samples:
            for source_rank, source in enumerate(sample.sources):
                rows, after, affected = self._source_patch(
                    source,
                    source_rank=source_rank,
                    new_lut=new_lut,
                )
                if not affected:
                    continue
                before = source.physical_ranks[rows]
                sample.endpoint += source.scale * (
                    _endpoint_from_ranks(
                        after,
                        source_rank=source_rank,
                        ep_size=self.args.ep_size,
                        ranks_per_node=self.args.ranks_per_node,
                    )
                    - _endpoint_from_ranks(
                        before,
                        source_rank=source_rank,
                        ep_size=self.args.ep_size,
                        ranks_per_node=self.args.ranks_per_node,
                    )
                )
                source.physical_ranks[rows] = after
                affected_tokens += affected
        self.lut = new_lut.copy()
        return affected_tokens


def _nearest_slot(
    source_rank: int,
    slots: np.ndarray,
    *,
    slots_per_rank: int,
    ranks_per_node: int,
) -> int:
    ranks = slots // slots_per_rank
    source_node = source_rank // ranks_per_node
    order = np.lexsort(
        (
            slots,
            np.abs(ranks - source_rank),
            ranks // ranks_per_node != source_node,
            ranks != source_rank,
        )
    )
    return int(slots[int(order[0])])


def _patch_group_cover(
    state: _LayerState,
    *,
    source_rank: int,
    target_rank: int,
    service_group_size: int,
    args: argparse.Namespace,
) -> _GroupCoverCandidate | None:
    if source_rank == target_rank:
        return None
    slots_per_rank = args.slots_per_rank
    source_slots = np.arange(
        source_rank * slots_per_rank,
        (source_rank + 1) * slots_per_rank,
        dtype=np.int64,
    )
    target_slots = np.arange(
        target_rank * slots_per_rank,
        (target_rank + 1) * slots_per_rank,
        dtype=np.int64,
    )
    inserted = state.layout[source_slots]
    evicted = state.layout[target_slots]
    if np.any(inserted < 0) or np.any(evicted < 0):
        return None
    if len(np.unique(inserted)) != slots_per_rank or np.array_equal(inserted, evicted):
        return None

    layout = state.layout.copy()
    layout[target_slots] = inserted
    counts = np.bincount(layout[layout >= 0], minlength=args.num_experts)
    if np.any(counts < 1) or np.any(counts > args.max_copies):
        return None
    for rank in range(args.ep_size):
        row = layout[rank * slots_per_rank : (rank + 1) * slots_per_rank]
        active = row[row >= 0]
        if len(active) != len(np.unique(active)):
            return None

    owners = state.owners.copy()
    for expert in np.unique(evicted):
        owner = int(owners[expert])
        if int(layout[owner]) != int(expert):
            remaining = np.flatnonzero(layout == expert)
            if remaining.size == 0:
                return None
            owners[expert] = int(remaining.min())

    lut = state.lut.copy()
    affected_experts = np.unique(np.concatenate((inserted, evicted)))
    for expert in affected_experts.tolist():
        invalid_sources = np.flatnonzero(layout[lut[:, expert]] != expert)
        if invalid_sources.size == 0:
            continue
        remaining = np.flatnonzero(layout == expert)
        for source in invalid_sources.tolist():
            lut[source, expert] = _nearest_slot(
                source,
                remaining,
                slots_per_rank=slots_per_rank,
                ranks_per_node=args.ranks_per_node,
            )

    service_start = (target_rank // service_group_size) * service_group_size
    service_stop = service_start + service_group_size
    for slot, expert in zip(target_slots.tolist(), inserted.tolist(), strict=True):
        lut[service_start:service_stop, expert] = slot
    logical = np.arange(args.num_experts, dtype=np.int64)
    if not np.array_equal(layout[lut], np.broadcast_to(logical, lut.shape)):
        raise RuntimeError("Group Cover produced an invalid source LUT.")
    return _GroupCoverCandidate(
        action=_GroupCoverAction(
            source_rank=source_rank,
            target_rank=target_rank,
            service_group_size=service_group_size,
            inserted_experts=tuple(int(value) for value in inserted.tolist()),
            evicted_experts=tuple(int(value) for value in evicted.tolist()),
        ),
        layout=layout,
        owners=owners,
        lut=lut,
    )


def _group_cover_candidates(
    state: _LayerState,
    *,
    args: argparse.Namespace,
) -> list[_GroupCoverCandidate]:
    rows: list[_GroupCoverCandidate] = []
    seen: set[bytes] = set()
    for source_rank in range(args.ep_size):
        for target_rank in range(args.ep_size):
            for group_size in args.service_group_sizes:
                candidate = _patch_group_cover(
                    state,
                    source_rank=source_rank,
                    target_rank=target_rank,
                    service_group_size=int(group_size),
                    args=args,
                )
                if candidate is None:
                    continue
                key = candidate.layout.tobytes() + candidate.lut.tobytes()
                if key not in seen:
                    seen.add(key)
                    rows.append(candidate)
    return rows


def _load_state(
    payload: dict[str, object],
    *,
    layer: int,
    args: argparse.Namespace,
) -> _LayerState:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Input layout has no layer table.")
    row = layers.get(_layer_name(layer))
    if not isinstance(row, dict):
        raise ValueError(f"Input layout has no layer {layer}.")
    state = _LayerState(
        layout=np.asarray(row["slot_to_logical"], dtype=np.int64),
        owners=np.asarray(row["owner_slots"], dtype=np.int64),
        lut=np.asarray(row["source_logical_to_physical"], dtype=np.int64),
    )
    expected_slots = args.ep_size * args.slots_per_rank
    if state.layout.shape != (expected_slots,):
        raise ValueError("Input layout slot count does not match its topology.")
    logical = np.arange(args.num_experts, dtype=np.int64)
    if not np.array_equal(state.layout[state.owners], logical):
        raise ValueError("Input owners do not reference their logical experts.")
    if not np.array_equal(state.layout[state.lut], np.broadcast_to(logical, state.lut.shape)):
        raise ValueError("Input source LUT does not reference its logical experts.")
    return state


def _refine_layer(
    layer: int,
    *,
    payload: dict[str, object],
    args: argparse.Namespace,
) -> tuple[int, _LayerState, dict[str, object]]:
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
    state = _load_state(payload, layer=layer, args=args)
    exact = _AffectedRouteEvaluator(
        optimize_samples,
        state.lut,
        evaluator=evaluator,
        args=args,
    )
    proxy = _AffectedRouteEvaluator(
        optimize_samples,
        state.lut,
        evaluator=evaluator,
        args=args,
        token_limit=args.proxy_token_limit,
    )
    initial_optimize = evaluator.evaluate(optimize_samples, state.lut)
    initial_validation = evaluator.evaluate(validation_samples, state.lut)
    incremental_initial, _ = exact.cost()
    initial_error = abs(incremental_initial.total_ms - initial_optimize.total_ms)
    if initial_error > args.anchor_atol_ms:
        raise RuntimeError(f"Layer {layer} incremental baseline differs from full replay by {initial_error:.6f} ms.")

    rounds: list[dict[str, object]] = []
    started = time.perf_counter()
    maximum_anchor_error = initial_error
    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        candidates = _group_cover_candidates(state, args=args)
        if not candidates:
            break
        proxy_rows: list[tuple[float, int]] = []
        for index, candidate in enumerate(candidates):
            cost, _ = proxy.cost(candidate.lut)
            proxy_rows.append((cost.total_ms, index))
        shortlist = [
            candidates[index]
            for _, index in sorted(proxy_rows, key=lambda row: (row[0], row[1]))[: args.exact_shortlist]
        ]

        exact_rows: list[tuple[float, int, HybridCost, int]] = []
        for index, candidate in enumerate(shortlist):
            cost, affected = exact.cost(candidate.lut)
            exact_rows.append((cost.total_ms, index, cost, affected))
        best_total, best_index, best_cost, affected_tokens = min(
            exact_rows,
            key=lambda row: (row[0], row[1]),
        )
        winner = shortlist[best_index]
        baseline, _ = exact.cost()
        accepted = best_total < baseline.total_ms - float(args.minimum_gain_ms)
        anchor_error = 0.0
        if accepted:
            full = evaluator.evaluate(optimize_samples, winner.lut)
            anchor_error = abs(full.total_ms - best_cost.total_ms)
            maximum_anchor_error = max(maximum_anchor_error, anchor_error)
            if anchor_error > args.anchor_atol_ms:
                raise RuntimeError(f"Layer {layer} group Cover differs from full replay by {anchor_error:.6f} ms.")
            exact.commit(winner.lut)
            proxy.commit(winner.lut)
            state = _LayerState(
                layout=winner.layout,
                owners=winner.owners,
                lut=winner.lut,
            )
        rounds.append(
            {
                "round": round_index + 1,
                "accepted": accepted,
                "action": asdict(winner.action),
                "legal_candidates": len(candidates),
                "exact_shortlist": len(shortlist),
                "affected_tokens": affected_tokens,
                "baseline": asdict(baseline),
                "candidate": asdict(best_cost),
                "baseline_ms": baseline.total_ms,
                "candidate_ms": best_total,
                "gain_ms": max(0.0, baseline.total_ms - best_total) if accepted else 0.0,
                "anchor_error_ms": anchor_error,
                "round_ms": (time.perf_counter() - round_started) * 1000.0,
            }
        )
        if not accepted:
            break

    final_optimize = evaluator.evaluate(optimize_samples, state.lut)
    final_validation = evaluator.evaluate(validation_samples, state.lut)
    report = {
        "layer": layer,
        "initial_optimize": asdict(initial_optimize),
        "final_optimize": asdict(final_optimize),
        "initial_validation": asdict(initial_validation),
        "final_validation": asdict(final_validation),
        "accepted_group_covers": sum(bool(row["accepted"]) for row in rounds),
        "maximum_anchor_error_ms": maximum_anchor_error,
        "rounds": rounds,
        "planner_ms": (time.perf_counter() - started) * 1000.0,
    }
    return layer, state, report


def _mean_total(reports: list[dict[str, object]], key: str, samples: int) -> float:
    return sum(float(report[key]["total_ms"]) for report in reports) / max(1, samples)  # type: ignore[index]


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    topology = payload.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("Input layout has no topology metadata.")
    args.ep_size = int(topology["ep_size"])
    args.num_experts = int(topology["num_experts"])
    args.slots_per_rank = int(topology["slots_per_rank"])
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")
    if args.rounds <= 0 or args.workers <= 0 or args.exact_shortlist <= 0:
        raise ValueError("Rounds, workers, and exact shortlist must be positive.")
    if any(size <= 0 or args.ep_size % size for size in args.service_group_sizes):
        raise ValueError("Every service group size must be a positive divisor of EP size.")

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

    layouts = [row[1].layout for row in results]
    owners = [row[1].owners for row in results]
    luts = [row[1].lut for row in results]
    reports = [row[2] for row in results]
    output = _preloaded_replay_payload(
        layouts=layouts,
        owners=owners,
        luts=luts,
        args=args,
        algorithm="recursive-classifier-group-cover-v1",
    )
    optimize_initial = _mean_total(reports, "initial_optimize", len(args.optimize_steps))
    optimize_final = _mean_total(reports, "final_optimize", len(args.optimize_steps))
    validation_initial = _mean_total(reports, "initial_validation", len(args.validation_steps))
    validation_final = _mean_total(reports, "final_validation", len(args.validation_steps))
    aggregate = {
        "layers": len(reports),
        "accepted_group_covers": sum(int(report["accepted_group_covers"]) for report in reports),
        "changed_layers": sum(int(report["accepted_group_covers"]) > 0 for report in reports),
        "initial_optimize_mean_ms": optimize_initial,
        "final_optimize_mean_ms": optimize_final,
        "optimize_gain_ms": optimize_initial - optimize_final,
        "optimize_speedup": optimize_initial / optimize_final,
        "initial_validation_mean_ms": validation_initial,
        "final_validation_mean_ms": validation_final,
        "validation_gain_ms": validation_initial - validation_final,
        "validation_speedup": validation_initial / validation_final,
        "validation_gain_fraction": (validation_initial - validation_final) / validation_initial,
        "maximum_anchor_error_ms": max(float(report["maximum_anchor_error_ms"]) for report in reports),
        "planner_total_ms": sum(float(report["planner_ms"]) for report in reports),
        "planner_wall_ms": wall_ms,
        "meets_five_percent_gate": validation_final <= 0.95 * validation_initial,
    }
    report_payload = {
        "schema_version": 1,
        "algorithm": "recursive-classifier-group-cover-v1",
        "input_layout": str(args.input_layout.resolve()),
        "route_root": str(args.route_root.resolve()),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output_layout", "output_report"}
        },
        "aggregate": aggregate,
        "layers": reports,
    }
    for path, value in ((args.output_layout, output), (args.output_report, report_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
