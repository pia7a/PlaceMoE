#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Build explainable hierarchical primary and replica layouts from captured routes.

The initializer intentionally runs outside the steady-state training path.  It
uses source-aware expert affinity only to propose balanced node/rank
partitions, then selects proposals with the calibrated HierMoE hybrid cost.
The emitted replay contains all state-migration actions plus the exact
source-rank service LUT used by Forward.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans

from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


@dataclass(frozen=True)
class HybridCost:
    communication_ms: float
    compute_ms: float
    total_ms: float
    peak_communication_rank: int
    peak_compute_rank: int
    mean_destination_nodes: float
    mean_destination_ranks: float
    peak_assignments: float


@dataclass(frozen=True)
class LayerResult:
    layer: int
    primary_seed: int
    primary_cost: HybridCost
    replica_strategy: str | None
    full_cost: HybridCost | None
    r2_cost: HybridCost
    primary_gain_over_r2_ms: float
    full_gain_over_r2_ms: float | None
    primary_node_loads: tuple[float, ...]
    primary_rank_loads: tuple[float, ...]
    copy_counts: tuple[int, ...]


@dataclass(frozen=True)
class ClosureResult:
    strategy: str
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray
    cost: HybridCost
    selected_by_node: tuple[tuple[int, ...], ...]


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(1,))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(2,))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--primary-slots-per-rank", type=int, default=4)
    parser.add_argument("--node-restarts", type=int, default=6)
    parser.add_argument("--partition-iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument(
        "--greedy-max-actions",
        type=int,
        default=0,
        help="Maximum unrestricted replica actions per layer; zero fills until no positive gain or no slot.",
    )
    parser.add_argument(
        "--greedy-force-fill",
        action="store_true",
        help=(
            "Fill the requested replica budget even across zero/negative intermediate margins. "
            "This initialization-only mode tests complementarity between co-routed experts."
        ),
    )
    parser.add_argument("--output-primary", type=Path, required=True)
    parser.add_argument("--output-full", type=Path, required=True)
    parser.add_argument(
        "--output-greedy",
        type=Path,
        help=(
            "Optional replay for unrestricted positive-marginal replica filling. "
            "When omitted, the additional greedy search is skipped."
        ),
    )
    parser.add_argument(
        "--output-closure",
        type=Path,
        help=(
            "Optional replay for independent closure-batch replica initialization. "
            "The generated layout and source LUT do not use the paired incumbent as a warm start."
        ),
    )
    parser.add_argument(
        "--closure-candidate-limit",
        type=int,
        default=256,
        help="Maximum distinct token closures considered per source node.",
    )
    parser.add_argument(
        "--closure-lut-iterations",
        type=int,
        default=12,
        help="Maximum affinity coordinate-descent passes used to compile the static source LUT.",
    )
    parser.add_argument(
        "--closure-token-sample",
        type=int,
        default=65536,
        help="Maximum token rows used by each overlapping expert-library initialization restart.",
    )
    parser.add_argument(
        "--closure-batch-size",
        type=int,
        default=8,
        help="Number of affinity-coherent experts moved by one initialization batch.",
    )
    parser.add_argument(
        "--closure-refinement-rounds",
        type=int,
        default=0,
        help=(
            "Maximum proxy-scored capacity-preserving affinity-batch replacement rounds. "
            "Disabled by default because the fast initializer already evaluates its final layouts exactly."
        ),
    )
    parser.add_argument(
        "--closure-exact-neighbors",
        type=int,
        default=0,
        help=(
            "Top proxy-ranked one-batch replacements per partition/pairing verified by full route replay. "
            "This diagnostic search is expensive and disabled by default."
        ),
    )
    parser.add_argument(
        "--closure-wide-search",
        action="store_true",
        help="Evaluate extra spectral restarts, node pairings, and locality weights.",
    )
    parser.add_argument(
        "--closure-legacy-proposals",
        action="store_true",
        help="Also evaluate the slower legacy source-closure and overlapping-library proposals.",
    )
    parser.add_argument(
        "--validation-baseline-ms",
        type=float,
        default=6116.241273880005,
        help="Held-out 48-layer cost gate for the current strongest static layout.",
    )
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _load_routes(
    root: Path,
    *,
    steps: tuple[int, ...],
    layer: int,
    ep_size: int,
    call_indices: tuple[int, ...] = (0,),
    forward_repeats: int = 1,
    layer_stride: int | None = None,
) -> list[list[torch.Tensor]]:
    if not call_indices:
        raise ValueError("call_indices must not be empty.")
    if forward_repeats <= 0:
        raise ValueError("forward_repeats must be positive.")
    if layer_stride is None:
        layer_stride = 0
    if layer_stride < 0:
        raise ValueError("layer_stride must be non-negative.")
    if forward_repeats > 1 and layer_stride == 0:
        raise ValueError("layer_stride must be positive when forward_repeats > 1.")

    samples: list[list[torch.Tensor]] = []
    for step in steps:
        for repeat in range(forward_repeats):
            capture_layer = layer + repeat * layer_stride
            for call_index in call_indices:
                bundle_path = (
                    root
                    / f"step{step:04d}"
                    / f"layer{capture_layer:02d}_call{call_index}_all_ranks.pt"
                )
                if bundle_path.is_file():
                    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
                    routes_by_rank = payload.get("routes_by_rank") if isinstance(payload, dict) else None
                    if (
                        not isinstance(payload, dict)
                        or payload.get("format") != "hiermoe-local-route-bundle-v1"
                        or int(payload.get("ep_size", -1)) != ep_size
                        or not isinstance(routes_by_rank, (list, tuple))
                        or len(routes_by_rank) != ep_size
                    ):
                        raise ValueError(f"Invalid bundled route capture: {bundle_path}.")
                    rows = []
                    for route in routes_by_rank:
                        if not torch.is_tensor(route) or route.ndim != 2:
                            raise ValueError(f"Invalid bundled route capture: {bundle_path}.")
                        rows.append(route.to(dtype=torch.long).contiguous())
                    samples.append(rows)
                    continue
                rows: list[torch.Tensor] = []
                for rank in range(ep_size):
                    path = (
                        root
                        / f"step{step:04d}"
                        / f"layer{capture_layer:02d}_call{call_index}_rank{rank:02d}.pt"
                    )
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    route = payload.get("routes") if isinstance(payload, dict) else None
                    if not torch.is_tensor(route) or route.ndim != 2:
                        raise ValueError(f"Invalid route capture: {path}.")
                    if int(payload.get("ep_size", -1)) != ep_size:
                        raise ValueError(f"Route capture has a different EP size: {path}.")
                    rows.append(route.to(dtype=torch.long).contiguous())
                samples.append(rows)
    return samples


def _route_statistics(
    samples: list[list[torch.Tensor]],
    *,
    num_experts: int,
    ranks_per_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ep_size = len(samples[0])
    num_nodes = ep_size // ranks_per_node
    demand_by_rank = np.zeros((ep_size, num_experts), dtype=np.float64)
    affinity = np.zeros((num_experts, num_experts), dtype=np.float64)
    affinity_by_source_node = np.zeros((num_nodes, num_experts, num_experts), dtype=np.float64)
    routes_by_node: list[list[torch.Tensor]] = [[] for _ in range(num_nodes)]
    all_routes: list[torch.Tensor] = []
    for sample in samples:
        for source_rank, route in enumerate(sample):
            flat = route.reshape(-1)
            demand_by_rank[source_rank] += torch.bincount(flat, minlength=num_experts).numpy()
            source_node = source_rank // ranks_per_node
            routes_by_node[source_node].append(route)
            all_routes.append(route)

    def accumulate_pair_affinity(routes: list[torch.Tensor]) -> np.ndarray:
        merged = torch.cat(routes, dim=0)
        result = np.zeros((num_experts, num_experts), dtype=np.float64)
        top_k = int(merged.shape[1])
        for lhs in range(top_k):
            for rhs in range(lhs + 1, top_k):
                pair = merged[:, lhs] * num_experts + merged[:, rhs]
                counts = torch.bincount(pair, minlength=num_experts * num_experts).reshape(
                    num_experts,
                    num_experts,
                )
                values = counts.numpy().astype(np.float64, copy=False)
                result += values + values.T
        return result

    affinity += accumulate_pair_affinity(all_routes)
    for source_node, routes in enumerate(routes_by_node):
        affinity_by_source_node[source_node] += accumulate_pair_affinity(routes)
    np.fill_diagonal(affinity, 0.0)
    for matrix in affinity_by_source_node:
        np.fill_diagonal(matrix, 0.0)
    demand_by_node = demand_by_rank.reshape(num_nodes, ranks_per_node, num_experts).sum(axis=1)
    return demand_by_rank, demand_by_node, affinity, affinity_by_source_node


def _balanced_spectral_partition(
    affinity: np.ndarray,
    demand: np.ndarray,
    *,
    parts: int,
    capacity: int,
    seed: int,
    load_weight: float,
    iterations: int,
) -> np.ndarray:
    size = int(affinity.shape[0])
    if size != parts * capacity:
        raise ValueError("Balanced partition requires parts * capacity vertices.")
    degree = np.maximum(affinity.sum(axis=1), 1.0)
    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    embedding = eigenvectors[:, np.argsort(eigenvalues)[-parts:]]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-12)

    kmeans = KMeans(n_clusters=parts, random_state=seed, n_init=1, max_iter=100)
    kmeans.fit(embedding)
    centers = kmeans.cluster_centers_
    labels = np.zeros((size,), dtype=np.int64)
    for _ in range(12):
        repeated = np.repeat(centers, capacity, axis=0)
        slot_distances = ((embedding[:, None, :] - repeated[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(slot_distances)
        labels[rows] = columns // capacity
        new_centers = np.stack(
            [embedding[labels == part].mean(axis=0) for part in range(parts)],
            axis=0,
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    total_affinity = max(float(affinity.sum()) / 2.0, 1.0)
    total_demand = max(float(demand.sum()), 1.0)

    loads = np.asarray([demand[labels == part].sum() for part in range(parts)])
    for _ in range(iterations):
        membership = np.eye(parts, dtype=np.float64)[labels]
        affinity_to_part = affinity @ membership
        best: tuple[float, int, int] | None = None
        for lhs in range(size):
            lhs_part = int(labels[lhs])
            for rhs in range(lhs + 1, size):
                rhs_part = int(labels[rhs])
                if lhs_part == rhs_part:
                    continue
                old_affinity = affinity_to_part[lhs, lhs_part] + affinity_to_part[rhs, rhs_part]
                new_affinity = (
                    affinity_to_part[lhs, rhs_part] + affinity_to_part[rhs, lhs_part] - 2.0 * affinity[lhs, rhs]
                )
                affinity_delta = float(new_affinity - old_affinity) / total_affinity

                lhs_fraction = loads[lhs_part] / total_demand
                rhs_fraction = loads[rhs_part] / total_demand
                load_delta = float(demand[rhs] - demand[lhs])
                new_lhs_fraction = (loads[lhs_part] + load_delta) / total_demand
                new_rhs_fraction = (loads[rhs_part] - load_delta) / total_demand
                imbalance_delta = (
                    (new_lhs_fraction - 1.0 / parts) ** 2
                    + (new_rhs_fraction - 1.0 / parts) ** 2
                    - (lhs_fraction - 1.0 / parts) ** 2
                    - (rhs_fraction - 1.0 / parts) ** 2
                )
                score_delta = affinity_delta - float(load_weight) * imbalance_delta
                if score_delta > 1e-12 and (best is None or score_delta > best[0]):
                    best = (score_delta, lhs, rhs)
        if best is None:
            break
        _, lhs, rhs = best
        lhs_part = int(labels[lhs])
        rhs_part = int(labels[rhs])
        load_delta = float(demand[rhs] - demand[lhs])
        loads[lhs_part] += load_delta
        loads[rhs_part] -= load_delta
        labels[lhs], labels[rhs] = labels[rhs], labels[lhs]
    return labels


def _map_node_clusters(
    labels: np.ndarray,
    demand_by_node: np.ndarray,
) -> np.ndarray:
    parts = int(labels.max()) + 1
    local_benefit = np.zeros((parts, parts), dtype=np.float64)
    for cluster in range(parts):
        experts = np.flatnonzero(labels == cluster)
        local_benefit[cluster] = demand_by_node[:, experts].sum(axis=1)
    clusters, nodes = linear_sum_assignment(-local_benefit)
    cluster_to_node = np.empty((parts,), dtype=np.int64)
    cluster_to_node[clusters] = nodes
    return cluster_to_node[labels]


def _hierarchical_expert_nodes(
    affinity: np.ndarray,
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> np.ndarray:
    num_nodes = int(demand_by_node.shape[0])
    if num_nodes != 4:
        raise ValueError("The first hierarchical initializer supports exactly four nodes.")
    num_experts = int(affinity.shape[0])
    super_labels = _balanced_spectral_partition(
        affinity,
        demand_by_rank.sum(axis=0),
        parts=2,
        capacity=num_experts // 2,
        seed=seed,
        load_weight=2.0,
        iterations=iterations,
    )
    node_labels = np.full((num_experts,), -1, dtype=np.int64)
    for super_group in range(2):
        experts = np.flatnonzero(super_labels == super_group)
        sub_labels = _balanced_spectral_partition(
            affinity[np.ix_(experts, experts)],
            demand_by_rank[:, experts].sum(axis=0),
            parts=2,
            capacity=num_experts // 4,
            seed=seed + 313 * (super_group + 1),
            load_weight=8.0,
            iterations=iterations,
        )
        node_labels[experts] = 2 * super_group + sub_labels
    return _map_node_clusters(node_labels, demand_by_node)


def _assign_primary_ranks(
    expert_nodes: np.ndarray,
    *,
    affinity: np.ndarray,
    demand_by_rank: np.ndarray,
    ranks_per_node: int,
    primary_slots_per_rank: int,
    seed: int,
    iterations: int,
) -> np.ndarray:
    expert_to_rank = np.full((expert_nodes.shape[0],), -1, dtype=np.int64)
    for node in range(int(expert_nodes.max()) + 1):
        experts = np.flatnonzero(expert_nodes == node)
        labels = _balanced_spectral_partition(
            affinity[np.ix_(experts, experts)],
            demand_by_rank[:, experts].sum(axis=0),
            parts=ranks_per_node,
            capacity=primary_slots_per_rank,
            seed=seed + 97 * node,
            load_weight=8.0,
            iterations=iterations,
        )
        local_benefit = np.zeros((ranks_per_node, ranks_per_node), dtype=np.float64)
        for cluster in range(ranks_per_node):
            members = experts[labels == cluster]
            for lane in range(ranks_per_node):
                source_rank = node * ranks_per_node + lane
                local_benefit[cluster, lane] = demand_by_rank[source_rank, members].sum()
        clusters, lanes = linear_sum_assignment(-local_benefit)
        cluster_to_lane = np.empty((ranks_per_node,), dtype=np.int64)
        cluster_to_lane[clusters] = lanes
        expert_to_rank[experts] = node * ranks_per_node + cluster_to_lane[labels]
    if bool((expert_to_rank < 0).any()):
        raise RuntimeError("Primary rank assignment is incomplete.")
    return expert_to_rank


def _primary_layout(
    expert_to_rank: np.ndarray,
    *,
    ep_size: int,
    num_experts: int,
    slots_per_rank: int,
    primary_slots_per_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    layout = np.full((ep_size * slots_per_rank,), -1, dtype=np.int64)
    owners = np.full((num_experts,), -1, dtype=np.int64)
    for rank in range(ep_size):
        experts = np.flatnonzero(expert_to_rank == rank)
        if len(experts) != primary_slots_per_rank:
            raise RuntimeError(f"Rank {rank} received {len(experts)} primary experts.")
        for local_slot, expert in enumerate(sorted(experts.tolist())):
            slot = rank * slots_per_rank + local_slot
            layout[slot] = expert
            owners[expert] = slot
    return layout, owners


def _mirrored_r2(
    *,
    ep_size: int,
    num_experts: int,
    slots_per_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = np.full((ep_size * slots_per_rank,), -1, dtype=np.int64)
    owners = np.full((num_experts,), -1, dtype=np.int64)
    lut = np.full((ep_size, num_experts), -1, dtype=np.int64)
    half = ep_size // 2
    for expert in range(num_experts):
        rank_in_half, local_slot = divmod(expert, slots_per_rank)
        first = rank_in_half * slots_per_rank + local_slot
        second = (half + rank_in_half) * slots_per_rank + local_slot
        layout[first] = expert
        layout[second] = expert
        owners[expert] = first
        lut[:half, expert] = first
        lut[half:, expert] = second
    return layout, owners, lut


class _HybridEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        hierarchy = Hierarchy(
            ep_size=args.ep_size,
            group_sizes=(args.ranks_per_node, args.ep_size),
            source="hierarchical-init",
        )
        self.planner = GreedyCommunicationPlanner(
            hierarchy=hierarchy,
            perf_model=HierMoEPerfModel.default(),
            hidden_size=args.hidden_size,
            bytes_per_element=args.bytes_per_element,
            slots_per_rank=args.slots_per_rank,
            forward_compute_per_assignment=args.compute_ms_per_assignment,
            traffic_inter_ms_per_byte=args.inter_ms_per_byte,
            traffic_intra_ms_per_byte=args.intra_ms_per_byte,
            traffic_route_ms_per_assignment=args.route_ms_per_assignment,
            traffic_communication_phase_multiplier=args.communication_phase_multiplier,
            traffic_compute_phase_multiplier=args.compute_phase_multiplier,
        )
        self.args = args

    def evaluate(
        self,
        samples: list[list[torch.Tensor]],
        source_lut: np.ndarray,
    ) -> HybridCost:
        communication = 0.0
        compute = 0.0
        node_destinations = 0.0
        rank_destinations = 0.0
        tokens = 0
        peak_assignments = 0.0
        peak_communication_rank = -1
        peak_compute_rank = -1
        lut = torch.from_numpy(source_lut).to(torch.long)
        for sample in samples:
            endpoint = torch.zeros((1, 8 * self.args.ep_size), dtype=torch.float32)
            assignment_totals = torch.zeros((self.args.ep_size,), dtype=torch.float32)
            for source_rank, logical in enumerate(sample):
                physical = lut[source_rank].index_select(0, logical.reshape(-1)).view_as(logical)
                unique = self.planner._local_packed_counts(physical)
                assignments = self.planner._local_packed_assignment_counts(physical)
                endpoint += self.planner._local_traffic_endpoint_statistics(
                    unique,
                    assignments,
                    source_rank=source_rank,
                )
                assignment_totals += assignments[0, : self.args.ep_size]
                ranks = torch.div(physical, self.args.slots_per_rank, rounding_mode="floor")
                rank_hits = torch.zeros((logical.shape[0], self.args.ep_size), dtype=torch.bool)
                rank_hits.scatter_(1, ranks, True)
                nodes = torch.div(ranks, self.args.ranks_per_node, rounding_mode="floor")
                node_hits = torch.zeros(
                    (logical.shape[0], self.args.ep_size // self.args.ranks_per_node),
                    dtype=torch.bool,
                )
                node_hits.scatter_(1, nodes, True)
                rank_destinations += float(rank_hits.sum().item())
                node_destinations += float(node_hits.sum().item())
                tokens += int(logical.shape[0])
            details = self.planner._traffic_endpoint_cost_details(endpoint)
            communication += float(details[0].item())
            compute += float(details[1].item())
            peak_communication_rank = int(details[3].item())
            peak_compute_rank = int(details[4].item())
            peak_assignments = max(peak_assignments, float(assignment_totals.max().item()))
        return HybridCost(
            communication_ms=communication,
            compute_ms=compute,
            total_ms=communication + compute,
            peak_communication_rank=peak_communication_rank,
            peak_compute_rank=peak_compute_rank,
            mean_destination_nodes=node_destinations / max(tokens, 1),
            mean_destination_ranks=rank_destinations / max(tokens, 1),
            peak_assignments=peak_assignments,
        )


class _ExpertTokenRows(NamedTuple):
    rows: np.ndarray
    multiplicities: np.ndarray


@dataclass
class _IncrementalSourceRoutes:
    logical: np.ndarray
    physical_ranks: np.ndarray
    rank_refcounts: np.ndarray
    node_refcounts: np.ndarray
    expert_rows: list[_ExpertTokenRows]


@dataclass
class _IncrementalSample:
    sources: list[_IncrementalSourceRoutes]
    endpoint: np.ndarray


@dataclass(frozen=True)
class _ReplicaCandidate:
    expert: int
    target_rank: int
    target_slot: int
    total_ms: float


@dataclass(frozen=True)
class _UnrestrictedGreedyResult:
    layout: np.ndarray
    lut: np.ndarray
    cost: HybridCost
    actions: tuple[tuple[int, int], ...]
    stopped_for_nonpositive_gain: bool


class _UnrestrictedReplicaGreedy:
    """Exact incremental positive-marginal replica filling.

    Candidate placement is unrestricted across nodes. The only redundant
    candidate removed is a second copy of the same logical expert on one rank,
    because it cannot change either hierarchical communication or rank load.

    Runtime service is topology-derived rather than learned from historical
    source demand: source ranks with a copy in their node use the nearest local
    copy, while all other source ranks retain their previous mapping.
    """

    def __init__(
        self,
        samples: list[list[torch.Tensor]],
        layout: np.ndarray,
        owners: np.ndarray,
        lut: np.ndarray,
        *,
        evaluator: _HybridEvaluator,
        args: argparse.Namespace,
    ) -> None:
        self.args = args
        self.evaluator = evaluator
        self.layout = layout.copy()
        self.owners = owners.copy()
        self.lut = lut.copy()
        self.num_nodes = args.ep_size // args.ranks_per_node
        self.samples = [self._build_sample(sample, evaluator=evaluator) for sample in samples]
        self.rank_slots: list[list[int]] = []
        for rank in range(args.ep_size):
            start = rank * args.slots_per_rank
            self.rank_slots.append(
                [slot for slot in range(start, start + args.slots_per_rank) if int(self.layout[slot]) < 0]
            )
        self.expert_slots: list[dict[int, int]] = []
        for expert in range(args.num_experts):
            slots = np.flatnonzero(self.layout == expert)
            self.expert_slots.append({int(slot) // args.slots_per_rank: int(slot) for slot in slots.tolist()})

    @staticmethod
    def _expert_rows(logical: np.ndarray, num_experts: int) -> list[_ExpertTokenRows]:
        tokens, top_k = logical.shape
        flat = logical.reshape(-1)
        token_ids = np.repeat(np.arange(tokens, dtype=np.int64), top_k)
        order = np.argsort(flat, kind="stable")
        sorted_experts = flat[order]
        sorted_tokens = token_ids[order]
        boundaries = np.searchsorted(sorted_experts, np.arange(num_experts + 1), side="left")
        result: list[_ExpertTokenRows] = []
        for expert in range(num_experts):
            values = sorted_tokens[boundaries[expert] : boundaries[expert + 1]]
            if values.size == 0:
                result.append(
                    _ExpertTokenRows(
                        np.empty((0,), dtype=np.int64),
                        np.empty((0,), dtype=np.uint8),
                    )
                )
                continue
            rows, multiplicities = np.unique(values, return_counts=True)
            result.append(
                _ExpertTokenRows(
                    rows.astype(np.int64, copy=False),
                    multiplicities.astype(np.uint8, copy=False),
                )
            )
        return result

    def _build_sample(
        self,
        sample: list[torch.Tensor],
        *,
        evaluator: _HybridEvaluator,
    ) -> _IncrementalSample:
        endpoint = torch.zeros((1, 8 * self.args.ep_size), dtype=torch.float32)
        sources: list[_IncrementalSourceRoutes] = []
        lut = torch.from_numpy(self.lut).to(torch.long)
        for source_rank, logical_tensor in enumerate(sample):
            logical = logical_tensor.to(torch.long)
            physical = lut[source_rank].index_select(0, logical.reshape(-1)).view_as(logical)
            unique = evaluator.planner._local_packed_counts(physical)
            assignments = evaluator.planner._local_packed_assignment_counts(physical)
            endpoint += evaluator.planner._local_traffic_endpoint_statistics(
                unique,
                assignments,
                source_rank=source_rank,
            )
            logical_np = logical.numpy().astype(np.int64, copy=False)
            physical_ranks = physical.numpy().astype(np.int64, copy=False) // self.args.slots_per_rank
            tokens, top_k = logical_np.shape
            token_ids = np.repeat(np.arange(tokens, dtype=np.int64), top_k)
            rank_refcounts = np.zeros(
                (tokens, self.args.ep_size),
                dtype=np.uint8,
            )
            np.add.at(
                rank_refcounts,
                (token_ids, physical_ranks.reshape(-1)),
                1,
            )
            node_refcounts = np.zeros(
                (tokens, self.num_nodes),
                dtype=np.uint8,
            )
            np.add.at(
                node_refcounts,
                (
                    token_ids,
                    (physical_ranks.reshape(-1) // self.args.ranks_per_node),
                ),
                1,
            )
            sources.append(
                _IncrementalSourceRoutes(
                    logical=logical_np,
                    physical_ranks=physical_ranks.copy(),
                    rank_refcounts=rank_refcounts,
                    node_refcounts=node_refcounts,
                    expert_rows=self._expert_rows(logical_np, self.args.num_experts),
                )
            )
        return _IncrementalSample(
            sources=sources,
            endpoint=endpoint.numpy()[0].astype(np.float64),
        )

    def _local_mapping_after_add(
        self,
        expert: int,
        target_rank: int,
    ) -> dict[int, int]:
        target_node = target_rank // self.args.ranks_per_node
        slots_by_rank = {
            rank: slot
            for rank, slot in self.expert_slots[expert].items()
            if rank // self.args.ranks_per_node == target_node
        }
        slots_by_rank[target_rank] = self.rank_slots[target_rank][0]
        local_ranks = sorted(slots_by_rank)
        result: dict[int, int] = {}
        source_start = target_node * self.args.ranks_per_node
        for source_rank in range(source_start, source_start + self.args.ranks_per_node):
            nearest = min(
                local_ranks,
                key=lambda rank: (abs(rank - source_rank), rank),
            )
            result[source_rank] = slots_by_rank[nearest]
        return result

    def _source_endpoint_delta(
        self,
        source: _IncrementalSourceRoutes,
        *,
        source_rank: int,
        expert: int,
        old_rank: int,
        new_rank: int,
    ) -> np.ndarray:
        delta = np.zeros((8 * self.args.ep_size,), dtype=np.float64)
        if old_rank == new_rank:
            return delta
        affected = source.expert_rows[expert]
        rows = affected.rows
        if rows.size == 0:
            return delta
        multiplicities = affected.multiplicities
        assignment_count = float(multiplicities.sum(dtype=np.int64))
        removed_rank = float(np.count_nonzero(source.rank_refcounts[rows, old_rank] == multiplicities))
        added_rank = float(np.count_nonzero(source.rank_refcounts[rows, new_rank] == 0))
        old_node = old_rank // self.args.ranks_per_node
        new_node = new_rank // self.args.ranks_per_node
        lane = source_rank % self.args.ranks_per_node

        # Stage 2 operates on destination ranks within each node.
        delta[4 * self.args.ep_size + old_node * self.args.ranks_per_node + lane] -= removed_rank
        delta[4 * self.args.ep_size + new_node * self.args.ranks_per_node + lane] += added_rank
        delta[5 * self.args.ep_size + old_rank] -= removed_rank
        delta[5 * self.args.ep_size + new_rank] += added_rank
        delta[6 * self.args.ep_size + old_node * self.args.ranks_per_node + lane] -= assignment_count
        delta[6 * self.args.ep_size + new_node * self.args.ranks_per_node + lane] += assignment_count
        delta[7 * self.args.ep_size + old_rank] -= assignment_count
        delta[7 * self.args.ep_size + new_rank] += assignment_count

        if old_node != new_node:
            removed_node = float(np.count_nonzero(source.node_refcounts[rows, old_node] == multiplicities))
            added_node = float(np.count_nonzero(source.node_refcounts[rows, new_node] == 0))
            delta[source_rank] += -removed_node + added_node
            delta[self.args.ep_size + lane * self.num_nodes + old_node] -= removed_node
            delta[self.args.ep_size + lane * self.num_nodes + new_node] += added_node
            delta[2 * self.args.ep_size + source_rank] += 0.0
            delta[3 * self.args.ep_size + lane * self.num_nodes + old_node] -= assignment_count
            delta[3 * self.args.ep_size + lane * self.num_nodes + new_node] += assignment_count
        return delta

    def _candidate_deltas(
        self,
        expert: int,
        target_rank: int,
    ) -> list[np.ndarray]:
        mapping = self._local_mapping_after_add(expert, target_rank)
        deltas: list[np.ndarray] = []
        for sample in self.samples:
            delta = np.zeros_like(sample.endpoint)
            for source_rank, target_slot in mapping.items():
                old_rank = int(self.lut[source_rank, expert]) // self.args.slots_per_rank
                new_rank = int(target_slot) // self.args.slots_per_rank
                delta += self._source_endpoint_delta(
                    sample.sources[source_rank],
                    source_rank=source_rank,
                    expert=expert,
                    old_rank=old_rank,
                    new_rank=new_rank,
                )
            deltas.append(delta)
        return deltas

    def _cost_rows(self, endpoints: np.ndarray) -> np.ndarray:
        original_shape = endpoints.shape[:-1]
        rows = endpoints.reshape(-1, 8 * self.args.ep_size)
        (
            unique_stage1_send,
            unique_stage1_receive,
            assignment_stage1_send,
            assignment_stage1_receive,
            unique_stage2_send,
            unique_stage2_receive,
            assignment_stage2_send,
            assignment_stage2_receive,
        ) = np.split(rows, 8, axis=1)
        hidden_bytes = float(self.evaluator.planner.payload_bytes)

        def stage_bytes(
            unique_send: np.ndarray,
            unique_receive: np.ndarray,
            assignment_send: np.ndarray,
            assignment_receive: np.ndarray,
            metadata_bytes: int,
        ) -> np.ndarray:
            dispatch = np.maximum(
                np.max(hidden_bytes * unique_send + metadata_bytes * assignment_send, axis=1),
                np.max(hidden_bytes * unique_receive + metadata_bytes * assignment_receive, axis=1),
            )
            combine = hidden_bytes * np.maximum(
                np.max(unique_send, axis=1),
                np.max(unique_receive, axis=1),
            )
            return dispatch + combine

        stage1 = stage_bytes(
            unique_stage1_send,
            unique_stage1_receive,
            assignment_stage1_send,
            assignment_stage1_receive,
            3 * 4,
        )
        stage2 = stage_bytes(
            unique_stage2_send,
            unique_stage2_receive,
            assignment_stage2_send,
            assignment_stage2_receive,
            2 * 4,
        )
        peak_assignments = np.max(assignment_stage2_receive, axis=1)
        planner = self.evaluator.planner
        communication = (
            planner.communication_scale
            * planner.traffic_communication_phase_multiplier
            * (
                planner.traffic_inter_ms_per_byte * stage1
                + planner.traffic_intra_ms_per_byte * stage2
                + planner.traffic_route_ms_per_assignment * peak_assignments
            )
        )
        compute = planner.traffic_compute_phase_multiplier * (
            planner.forward_compute_per_assignment * peak_assignments + planner.forward_compute_constant
        )
        return (communication + compute).reshape(original_shape)

    def _cost(self, endpoint: np.ndarray) -> float:
        return float(self._cost_rows(endpoint.reshape(1, -1))[0])

    def current_total_ms(self) -> float:
        return sum(self._cost(sample.endpoint) for sample in self.samples)

    def _score_candidate(self, expert: int, target_rank: int) -> _ReplicaCandidate:
        deltas = self._candidate_deltas(expert, target_rank)
        total = sum(self._cost(sample.endpoint + delta) for sample, delta in zip(self.samples, deltas, strict=True))
        return _ReplicaCandidate(
            expert=expert,
            target_rank=target_rank,
            target_slot=self.rank_slots[target_rank][0],
            total_ms=total,
        )

    def _candidate_cost_matrix(self) -> np.ndarray:
        """Score every expert/rank add with one vectorized endpoint batch."""

        experts = self.args.num_experts
        ep_size = self.args.ep_size
        ranks_per_node = self.args.ranks_per_node
        width = 8 * ep_size
        total = np.zeros((experts, ep_size), dtype=np.float64)

        # [expert, candidate target rank, source lane] -> selected local rank.
        chosen = np.empty((experts, ep_size, ranks_per_node), dtype=np.int64)
        for expert in range(experts):
            for node in range(self.num_nodes):
                start = node * ranks_per_node
                existing = [rank for rank in self.expert_slots[expert] if rank // ranks_per_node == node]
                for target_rank in range(start, start + ranks_per_node):
                    local_ranks = sorted(set(existing + [target_rank]))
                    for lane in range(ranks_per_node):
                        source_rank = start + lane
                        chosen[expert, target_rank, lane] = min(
                            local_ranks,
                            key=lambda rank: (abs(rank - source_rank), rank),
                        )

        for sample in self.samples:
            candidate_delta = np.zeros((experts, ep_size, width), dtype=np.float64)
            for source_rank, source in enumerate(sample.sources):
                node = source_rank // ranks_per_node
                lane = source_rank % ranks_per_node
                node_start = node * ranks_per_node
                target_ranks = np.arange(
                    node_start,
                    node_start + ranks_per_node,
                    dtype=np.int64,
                )
                local_ranks = target_ranks
                for expert in range(experts):
                    affected = source.expert_rows[expert]
                    rows = affected.rows
                    if rows.size == 0:
                        continue
                    multiplicities = affected.multiplicities
                    old_rank = int(self.lut[source_rank, expert]) // self.args.slots_per_rank
                    old_node = old_rank // ranks_per_node
                    assignment_count = float(multiplicities.sum(dtype=np.int64))
                    removed_rank = float(np.count_nonzero(source.rank_refcounts[rows, old_rank] == multiplicities))
                    added_ranks = np.count_nonzero(
                        source.rank_refcounts[rows][:, local_ranks] == 0,
                        axis=0,
                    ).astype(np.float64, copy=False)
                    delta_by_new = np.zeros((ranks_per_node, width), dtype=np.float64)
                    for new_lane, new_rank in enumerate(local_ranks.tolist()):
                        if new_rank == old_rank:
                            continue
                        added_rank = float(added_ranks[new_lane])
                        delta_by_new[
                            new_lane,
                            4 * ep_size + old_node * ranks_per_node + lane,
                        ] -= removed_rank
                        delta_by_new[
                            new_lane,
                            4 * ep_size + node * ranks_per_node + lane,
                        ] += added_rank
                        delta_by_new[new_lane, 5 * ep_size + old_rank] -= removed_rank
                        delta_by_new[new_lane, 5 * ep_size + new_rank] += added_rank
                        delta_by_new[
                            new_lane,
                            6 * ep_size + old_node * ranks_per_node + lane,
                        ] -= assignment_count
                        delta_by_new[
                            new_lane,
                            6 * ep_size + node * ranks_per_node + lane,
                        ] += assignment_count
                        delta_by_new[new_lane, 7 * ep_size + old_rank] -= assignment_count
                        delta_by_new[new_lane, 7 * ep_size + new_rank] += assignment_count
                        if old_node != node:
                            removed_node = float(
                                np.count_nonzero(source.node_refcounts[rows, old_node] == multiplicities)
                            )
                            added_node = float(np.count_nonzero(source.node_refcounts[rows, node] == 0))
                            delta_by_new[new_lane, source_rank] += -removed_node + added_node
                            delta_by_new[
                                new_lane,
                                ep_size + lane * self.num_nodes + old_node,
                            ] -= removed_node
                            delta_by_new[
                                new_lane,
                                ep_size + lane * self.num_nodes + node,
                            ] += added_node
                            delta_by_new[
                                new_lane,
                                3 * ep_size + lane * self.num_nodes + old_node,
                            ] -= assignment_count
                            delta_by_new[
                                new_lane,
                                3 * ep_size + lane * self.num_nodes + node,
                            ] += assignment_count
                    selected_lanes = chosen[expert, target_ranks, lane] - node_start
                    candidate_delta[expert, target_ranks] += delta_by_new[selected_lanes]
            total += self._cost_rows(sample.endpoint.reshape(1, 1, -1) + candidate_delta)

        for target_rank in range(ep_size):
            if not self.rank_slots[target_rank]:
                total[:, target_rank] = np.inf
                continue
            for expert in range(experts):
                if target_rank in self.expert_slots[expert]:
                    total[expert, target_rank] = np.inf
        return total

    def _commit(self, candidate: _ReplicaCandidate) -> None:
        expert = candidate.expert
        target_rank = candidate.target_rank
        target_slot = candidate.target_slot
        mapping = self._local_mapping_after_add(expert, target_rank)
        deltas = self._candidate_deltas(expert, target_rank)
        for sample, delta in zip(self.samples, deltas, strict=True):
            sample.endpoint += delta
            for source_rank, target_mapping_slot in mapping.items():
                old_rank = int(self.lut[source_rank, expert]) // self.args.slots_per_rank
                new_rank = int(target_mapping_slot) // self.args.slots_per_rank
                if old_rank == new_rank:
                    continue
                source = sample.sources[source_rank]
                affected = source.expert_rows[expert]
                rows = affected.rows
                if rows.size == 0:
                    continue
                multiplicities = affected.multiplicities
                source.rank_refcounts[rows, old_rank] -= multiplicities
                source.rank_refcounts[rows, new_rank] += multiplicities
                old_node = old_rank // self.args.ranks_per_node
                new_node = new_rank // self.args.ranks_per_node
                if old_node != new_node:
                    source.node_refcounts[rows, old_node] -= multiplicities
                    source.node_refcounts[rows, new_node] += multiplicities
                mask = source.logical == expert
                source.physical_ranks[mask] = new_rank
        self.layout[target_slot] = expert
        self.rank_slots[target_rank].pop(0)
        self.expert_slots[expert][target_rank] = target_slot
        for source_rank, target_mapping_slot in mapping.items():
            self.lut[source_rank, expert] = target_mapping_slot

    def run(self) -> _UnrestrictedGreedyResult:
        actions: list[tuple[int, int]] = []
        stopped = False
        max_actions = (
            sum(len(slots) for slots in self.rank_slots)
            if self.args.greedy_max_actions <= 0
            else int(self.args.greedy_max_actions)
        )

        while any(self.rank_slots) and len(actions) < max_actions:
            baseline = self.current_total_ms()
            costs = self._candidate_cost_matrix()
            flat_index = int(np.argmin(costs))
            expert, target_rank = np.unravel_index(flat_index, costs.shape)
            best = _ReplicaCandidate(
                expert=int(expert),
                target_rank=int(target_rank),
                target_slot=self.rank_slots[int(target_rank)][0],
                total_ms=float(costs[expert, target_rank]),
            )
            if not np.isfinite(best.total_ms) or (
                not self.args.greedy_force_fill and best.total_ms >= baseline - 1e-9
            ):
                stopped = True
                break
            self._commit(best)
            actions.append((best.expert, best.target_slot))
            if len(actions) == 1 or len(actions) % 16 == 0:
                print(
                    f"  greedy_replicas={len(actions):03d} "
                    f"cost_ms={best.total_ms:.3f} gain_ms={baseline - best.total_ms:.3f}",
                    flush=True,
                )
        final_cost = self.evaluator.evaluate(self._torch_samples(), self.lut)
        return _UnrestrictedGreedyResult(
            layout=self.layout.copy(),
            lut=self.lut.copy(),
            cost=final_cost,
            actions=tuple(actions),
            stopped_for_nonpositive_gain=stopped,
        )

    def _torch_samples(self) -> list[list[torch.Tensor]]:
        return [
            [torch.from_numpy(source.logical).to(torch.long) for source in sample.sources] for sample in self.samples
        ]


def _primary_lut(owners: np.ndarray, ep_size: int) -> np.ndarray:
    return np.broadcast_to(owners.reshape(1, -1), (ep_size, owners.shape[0])).copy()


def _node_sole_benefit(
    samples: list[list[torch.Tensor]],
    expert_to_node: np.ndarray,
    *,
    num_experts: int,
    ranks_per_node: int,
) -> np.ndarray:
    num_nodes = len(samples[0]) // ranks_per_node
    benefit = np.zeros((num_nodes, num_experts), dtype=np.float64)
    expert_nodes = torch.from_numpy(expert_to_node).to(torch.long)
    for sample in samples:
        for source_rank, route in enumerate(sample):
            source_node = source_rank // ranks_per_node
            destinations = expert_nodes.index_select(0, route.reshape(-1)).view_as(route)
            counts = torch.zeros((route.shape[0], num_nodes), dtype=torch.int16)
            counts.scatter_add_(1, destinations, torch.ones_like(destinations, dtype=torch.int16))
            sole = counts.gather(1, destinations) == 1
            remote = destinations != source_node
            values = torch.bincount(
                route.reshape(-1),
                weights=(sole & remote).reshape(-1).to(torch.float32),
                minlength=num_experts,
            )
            benefit[source_node] += values.numpy()
    return benefit


def _replica_layout_candidate(
    primary_layout: np.ndarray,
    primary_owners: np.ndarray,
    *,
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    sole_benefit: np.ndarray,
    strategy: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = args.ep_size // args.ranks_per_node
    primary_rank = primary_owners // args.slots_per_rank
    primary_node = primary_rank // args.ranks_per_node
    demand_by_node = demand_by_rank.reshape(num_nodes, args.ranks_per_node, args.num_experts).sum(axis=1)
    selected_by_node: list[list[int]] = []
    copy_counts = np.ones((args.num_experts,), dtype=np.int64)
    for node in range(num_nodes):
        remote = np.flatnonzero(primary_node != node)
        if strategy == "demand":
            score = demand_by_node[node, remote]
        elif strategy == "sole":
            score = sole_benefit[node, remote]
        elif strategy == "hybrid_quarter":
            score = sole_benefit[node, remote] + 0.25 * demand_by_node[node, remote]
        elif strategy == "hybrid":
            score = sole_benefit[node, remote] + demand_by_node[node, remote]
        else:
            raise ValueError(f"Unknown replica strategy {strategy!r}.")
        order = remote[np.argsort(-score, kind="stable")]
        chosen: list[int] = []
        for expert in order.tolist():
            if copy_counts[expert] >= args.max_copies:
                continue
            chosen.append(expert)
            copy_counts[expert] += 1
            if len(chosen) == args.ranks_per_node * (args.slots_per_rank - args.primary_slots_per_rank):
                break
        if len(chosen) != args.ranks_per_node * (args.slots_per_rank - args.primary_slots_per_rank):
            raise RuntimeError(f"Replica strategy {strategy} could not fill node {node}.")
        selected_by_node.append(chosen)

    return _materialize_replica_layout(
        primary_layout,
        primary_owners,
        selected_by_node=selected_by_node,
        demand_by_rank=demand_by_rank,
        affinity_by_source_node=affinity_by_source_node,
        args=args,
    )


def _materialize_replica_layout(
    primary_layout: np.ndarray,
    primary_owners: np.ndarray,
    *,
    selected_by_node: list[list[int]],
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = args.ep_size // args.ranks_per_node
    if len(selected_by_node) != num_nodes:
        raise ValueError("Replica selection must contain one expert list per node.")
    expected_per_node = args.ranks_per_node * (args.slots_per_rank - args.primary_slots_per_rank)
    if any(len(selected) != expected_per_node for selected in selected_by_node):
        raise ValueError("Replica selection does not fill every redundant slot.")
    layout = primary_layout.copy()
    lut = _primary_lut(primary_owners, args.ep_size)
    primary_rank = primary_owners // args.slots_per_rank
    demand_by_node = demand_by_rank.reshape(num_nodes, args.ranks_per_node, args.num_experts).sum(axis=1)
    global_rank_load = np.zeros((args.ep_size,), dtype=np.float64)
    for source_rank in range(args.ep_size):
        global_rank_load += np.bincount(
            primary_rank,
            weights=demand_by_rank[source_rank],
            minlength=args.ep_size,
        )

    for node, selected in enumerate(selected_by_node):
        capacities = np.full(
            (args.ranks_per_node,),
            args.slots_per_rank - args.primary_slots_per_rank,
            dtype=np.int64,
        )
        assigned: list[list[int]] = [[] for _ in range(args.ranks_per_node)]
        local_demand = demand_by_node[node]
        for expert in sorted(selected, key=lambda value: (-local_demand[value], value)):
            old_rank = int(primary_rank[expert])
            amount = float(local_demand[expert])
            best: tuple[float, float, int] | None = None
            for lane in range(args.ranks_per_node):
                if capacities[lane] <= 0:
                    continue
                target_rank = node * args.ranks_per_node + lane
                projected = global_rank_load.copy()
                projected[old_rank] -= amount
                projected[target_rank] += amount
                affinity_gain = sum(affinity_by_source_node[node, expert, other] for other in assigned[lane])
                key = (float(projected.max()), -float(affinity_gain), target_rank)
                if best is None or key < best:
                    best = key
            if best is None:
                raise RuntimeError("Replica rank assignment exhausted all slots.")
            target_rank = best[2]
            lane = target_rank - node * args.ranks_per_node
            local_slot = args.primary_slots_per_rank + len(assigned[lane])
            slot = target_rank * args.slots_per_rank + local_slot
            layout[slot] = expert
            assigned[lane].append(expert)
            capacities[lane] -= 1
            global_rank_load[old_rank] -= amount
            global_rank_load[target_rank] += amount
            source_start = node * args.ranks_per_node
            lut[source_start : source_start + args.ranks_per_node, expert] = slot
    return layout, lut


def _token_closures_by_source_node(
    samples: list[list[torch.Tensor]],
    primary_owners: np.ndarray,
    *,
    args: argparse.Namespace,
) -> list[list[tuple[frozenset[int], float]]]:
    """Extract destination-node expert closures from captured top-k routes.

    A closure contains every expert that makes one token visit one remote node
    under the unique-primary layout.  Replicating only a strict subset cannot
    remove that token-to-node edge, so initialization proposes the set as one
    capacity-aware batch instead of independent singleton Covers.
    """

    num_nodes = args.ep_size // args.ranks_per_node
    owner_nodes = primary_owners // args.slots_per_rank // args.ranks_per_node
    closures: list[dict[frozenset[int], float]] = [dict() for _ in range(num_nodes)]
    for source_node in range(num_nodes):
        source_start = source_node * args.ranks_per_node
        merged = torch.cat(
            [
                sample[source_rank]
                for sample in samples
                for source_rank in range(source_start, source_start + args.ranks_per_node)
            ],
            dim=0,
        ).numpy()
        destination_nodes = owner_nodes[merged]
        for destination_node in range(num_nodes):
            if destination_node == source_node:
                continue
            masked = np.where(destination_nodes == destination_node, merged, -1)
            masked.sort(axis=1)
            unique_rows, counts = np.unique(masked, axis=0, return_counts=True)
            for row, count in zip(unique_rows, counts, strict=True):
                closure = frozenset(int(value) for value in row if int(value) >= 0)
                if not closure:
                    continue
                closures[source_node][closure] = closures[source_node].get(closure, 0.0) + float(count)

    result: list[list[tuple[frozenset[int], float]]] = []
    for rows in closures:
        ordered = sorted(rows.items(), key=lambda item: (-item[1], len(item[0]), tuple(sorted(item[0]))))
        result.append(ordered[: max(1, int(args.closure_candidate_limit))])
    return result


def _select_closure_replicas(
    closures: list[list[tuple[frozenset[int], float]]],
    primary_owners: np.ndarray,
    demand_by_node: np.ndarray,
    *,
    batch_penalty: float,
    hot_weight: float,
    args: argparse.Namespace,
) -> list[list[int]]:
    """Choose one exact-capacity replica set per node from closure batches."""

    num_nodes = args.ep_size // args.ranks_per_node
    replicas_per_node = args.ranks_per_node * (args.slots_per_rank - args.primary_slots_per_rank)
    primary_nodes = primary_owners // args.slots_per_rank // args.ranks_per_node
    selected_by_node: list[list[int]] = []
    for source_node in range(num_nodes):
        eligible = {expert for expert in range(args.num_experts) if int(primary_nodes[expert]) != source_node}
        rows = [(closure & eligible, weight) for closure, weight in closures[source_node] if closure & eligible]
        selected: set[int] = set()
        while len(selected) < replicas_per_node:
            remaining = replicas_per_node - len(selected)
            best: tuple[float, float, tuple[int, ...]] | None = None
            for closure, _ in rows:
                missing = tuple(sorted(closure - selected))
                if not missing or len(missing) > remaining:
                    continue
                projected = selected | set(missing)
                coverage_gain = sum(
                    weight
                    for candidate, weight in rows
                    if candidate.issubset(projected) and not candidate.issubset(selected)
                )
                hot_gain = float(demand_by_node[source_node, list(missing)].sum())
                score = (coverage_gain + float(hot_weight) * hot_gain) / (float(len(missing)) ** float(batch_penalty))
                key = (score, coverage_gain, tuple(-value for value in missing))
                if best is None or key > (best[0], best[1], tuple(-value for value in best[2])):
                    best = (score, coverage_gain, missing)
            if best is None or best[0] <= 0.0:
                break
            selected.update(best[2])

        if len(selected) < replicas_per_node:
            candidates = sorted(
                eligible - selected,
                key=lambda expert: (-float(demand_by_node[source_node, expert]), expert),
            )
            selected.update(candidates[: replicas_per_node - len(selected)])
        if len(selected) != replicas_per_node:
            raise RuntimeError(
                f"Closure initialization selected {len(selected)} experts for node {source_node}, "
                f"expected {replicas_per_node}."
            )
        selected_by_node.append(sorted(selected))
    return selected_by_node


def _compile_affinity_source_lut(
    layout: np.ndarray,
    owners: np.ndarray,
    initial_lut: np.ndarray,
    *,
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    load_weight: float,
    locality_weight: float,
    iterations: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Compile a static source-node/expert policy for one fixed layout.

    The coordinate objective is only a proposal heuristic.  Every resulting
    LUT is subsequently ranked by the exact hybrid evaluator, while Forward
    consumes the selected LUT verbatim.
    """

    num_nodes = args.ep_size // args.ranks_per_node
    slot_by_expert_node = np.full((args.num_experts, num_nodes), -1, dtype=np.int64)
    for slot, expert in enumerate(layout.tolist()):
        if expert < 0:
            continue
        node = slot // args.slots_per_rank // args.ranks_per_node
        current = int(slot_by_expert_node[expert, node])
        if current < 0 or slot < current:
            slot_by_expert_node[expert, node] = slot

    labels = np.empty((num_nodes, args.num_experts), dtype=np.int64)
    for source_node in range(num_nodes):
        source_rank = source_node * args.ranks_per_node
        labels[source_node] = initial_lut[source_rank] // args.slots_per_rank // args.ranks_per_node

    demand_by_source_node = demand_by_rank.reshape(
        num_nodes,
        args.ranks_per_node,
        args.num_experts,
    ).sum(axis=1)
    rank_loads = np.zeros((args.ep_size,), dtype=np.float64)
    for source_node in range(num_nodes):
        for expert in range(args.num_experts):
            slot = int(slot_by_expert_node[expert, labels[source_node, expert]])
            rank_loads[slot // args.slots_per_rank] += demand_by_source_node[source_node, expert]
    total_demand = max(float(demand_by_source_node.sum()), 1.0)

    for _ in range(max(0, int(iterations))):
        changed = False
        for source_node in range(num_nodes):
            membership = np.eye(num_nodes, dtype=np.float64)[labels[source_node]]
            affinity_to_node = affinity_by_source_node[source_node] @ membership
            order = np.argsort(-demand_by_source_node[source_node], kind="stable")
            for expert in order.tolist():
                available_nodes = np.flatnonzero(slot_by_expert_node[expert] >= 0)
                if len(available_nodes) <= 1:
                    continue
                current_node = int(labels[source_node, expert])
                current_slot = int(slot_by_expert_node[expert, current_node])
                current_rank = current_slot // args.slots_per_rank
                amount = float(demand_by_source_node[source_node, expert])
                best: tuple[float, int] | None = None
                for candidate_node in available_nodes.tolist():
                    candidate_slot = int(slot_by_expert_node[expert, candidate_node])
                    candidate_rank = candidate_slot // args.slots_per_rank
                    projected = rank_loads.copy()
                    projected[current_rank] -= amount
                    projected[candidate_rank] += amount
                    affinity_reward = float(affinity_to_node[expert, candidate_node])
                    locality_reward = amount if candidate_node == source_node else 0.0
                    score = (
                        affinity_reward
                        + float(locality_weight) * locality_reward
                        - float(load_weight) * float(projected.max()) / total_demand
                    )
                    key = (score, -candidate_node)
                    if best is None or key > (best[0], -best[1]):
                        best = (score, candidate_node)
                if best is None or best[1] == current_node:
                    continue
                new_node = best[1]
                new_slot = int(slot_by_expert_node[expert, new_node])
                rank_loads[current_rank] -= amount
                rank_loads[new_slot // args.slots_per_rank] += amount
                labels[source_node, expert] = new_node
                changed = True
        if not changed:
            break

    lut = initial_lut.copy()
    for source_node in range(num_nodes):
        source_start = source_node * args.ranks_per_node
        for expert in range(args.num_experts):
            slot = int(slot_by_expert_node[expert, labels[source_node, expert]])
            lut[source_start : source_start + args.ranks_per_node, expert] = slot
    return lut


def _overlapping_expert_membership(
    samples: list[list[torch.Tensor]],
    *,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Cluster token hyperedges into fixed-capacity, overlapping node libraries."""

    num_nodes = args.ep_size // args.ranks_per_node
    capacity = args.ranks_per_node * args.slots_per_rank
    routes = torch.cat([route for sample in samples for route in sample], dim=0).numpy()
    sample_limit = min(int(args.closure_token_sample), int(routes.shape[0]))
    rng = np.random.default_rng(seed)
    if sample_limit < routes.shape[0]:
        indices = np.sort(rng.choice(routes.shape[0], size=sample_limit, replace=False))
        routes = routes[indices]
    features = np.zeros((routes.shape[0], args.num_experts), dtype=np.float32)
    row_ids = np.repeat(np.arange(routes.shape[0], dtype=np.int64), routes.shape[1])
    features[row_ids, routes.reshape(-1)] = 1.0
    model = MiniBatchKMeans(
        n_clusters=num_nodes,
        random_state=seed,
        batch_size=2048,
        max_iter=100,
        n_init=3,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(features)
    counts = np.zeros((num_nodes, args.num_experts), dtype=np.float64)
    for cluster in range(num_nodes):
        members = routes[labels == cluster]
        if members.size:
            counts[cluster] = np.bincount(members.reshape(-1), minlength=args.num_experts)

    membership = np.zeros((num_nodes, args.num_experts), dtype=bool)
    for cluster in range(num_nodes):
        order = np.argsort(-counts[cluster], kind="stable")
        membership[cluster, order[:capacity]] = True

    missing = list(np.flatnonzero(~membership.any(axis=0)))
    for expert in sorted(missing, key=lambda value: -float(counts[:, value].max())):
        copy_counts = membership.sum(axis=0)
        best: tuple[float, int, int] | None = None
        for cluster in range(num_nodes):
            replaceable = np.flatnonzero(membership[cluster] & (copy_counts > 1))
            for victim in replaceable.tolist():
                loss = float(counts[cluster, victim] - counts[cluster, expert])
                key = (loss, cluster, victim)
                if best is None or key < best:
                    best = key
        if best is None:
            raise RuntimeError("Cannot repair overlapping expert libraries without losing expert coverage.")
        _, cluster, victim = best
        membership[cluster, victim] = False
        membership[cluster, expert] = True

    if not bool((membership.sum(axis=1) == capacity).all()):
        raise RuntimeError("Overlapping expert libraries violate node capacity.")
    if not bool(membership.any(axis=0).all()):
        raise RuntimeError("Overlapping expert libraries lost at least one logical expert.")
    return membership


def _materialize_overlapping_layout(
    membership_by_node: np.ndarray,
    *,
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    affinity: np.ndarray,
    assignment_by_node: np.ndarray | None = None,
    affinity_by_node: np.ndarray | None = None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign variable-copy expert libraries to owner/replica rank slots."""

    num_nodes = args.ep_size // args.ranks_per_node
    owner_slots = np.repeat(np.arange(num_nodes), args.ranks_per_node * args.primary_slots_per_rank)
    owner_cost = np.full((args.num_experts, args.num_experts), 1e30, dtype=np.float64)
    placement_demand = demand_by_node if assignment_by_node is None else assignment_by_node
    for expert in range(args.num_experts):
        for column, node in enumerate(owner_slots.tolist()):
            if membership_by_node[node, expert]:
                owner_cost[expert, column] = -float(placement_demand[node, expert])
    rows, columns = linear_sum_assignment(owner_cost)
    if len(rows) != args.num_experts or bool((owner_cost[rows, columns] >= 1e29).any()):
        raise RuntimeError("Overlapping expert libraries have no feasible balanced owner assignment.")
    owner_node = np.full((args.num_experts,), -1, dtype=np.int64)
    owner_node[rows] = owner_slots[columns]

    if assignment_by_node is None:
        copy_counts = membership_by_node.sum(axis=0)
        placement_demand = np.broadcast_to(
            demand_by_rank.sum(axis=0) / np.maximum(copy_counts, 1),
            (num_nodes, args.num_experts),
        )
    layout = np.full((args.ep_size * args.slots_per_rank,), -1, dtype=np.int64)
    owners = np.full((args.num_experts,), -1, dtype=np.int64)
    slot_by_node_expert = np.full((num_nodes, args.num_experts), -1, dtype=np.int64)
    for node in range(num_nodes):
        owner_experts = [int(value) for value in np.flatnonzero(owner_node == node)]
        replica_experts = [
            int(value) for value in np.flatnonzero(membership_by_node[node]) if int(owner_node[value]) != node
        ]
        expected = args.ranks_per_node * args.primary_slots_per_rank
        if len(owner_experts) != expected or len(replica_experts) != expected:
            raise RuntimeError(
                f"Node {node} has owners/replicas={len(owner_experts)}/{len(replica_experts)}, "
                f"expected {expected}/{expected}."
            )
        lane_loads = np.zeros((args.ranks_per_node,), dtype=np.float64)
        lane_members: list[list[int]] = [[] for _ in range(args.ranks_per_node)]

        def assign(
            experts: list[int],
            slot_offset: int,
            *,
            current_node: int = node,
            current_lane_loads: np.ndarray = lane_loads,
            current_lane_members: list[list[int]] = lane_members,
            current_effective_demand: np.ndarray = placement_demand[node],
            current_affinity: np.ndarray = (affinity if affinity_by_node is None else affinity_by_node[node]),
        ) -> None:
            capacities = np.full((args.ranks_per_node,), args.primary_slots_per_rank, dtype=np.int64)
            expert_array = np.asarray(experts, dtype=np.int64)
            order = np.lexsort((expert_array, -current_effective_demand[expert_array]))
            for expert in expert_array[order].tolist():
                best: tuple[float, float, int] | None = None
                amount = float(current_effective_demand[expert])
                for lane in range(args.ranks_per_node):
                    if capacities[lane] <= 0:
                        continue
                    projected_peak = max(
                        float(current_lane_loads.max()),
                        current_lane_loads[lane] + amount,
                    )
                    affinity_gain = sum(current_affinity[expert, other] for other in current_lane_members[lane])
                    key = (projected_peak, -float(affinity_gain), lane)
                    if best is None or key < best:
                        best = key
                if best is None:
                    raise RuntimeError("Overlapping rank placement exhausted all slots.")
                lane = best[2]
                local_slot = slot_offset + (args.primary_slots_per_rank - capacities[lane])
                rank = current_node * args.ranks_per_node + lane
                slot = rank * args.slots_per_rank + local_slot
                layout[slot] = expert
                slot_by_node_expert[current_node, expert] = slot
                if slot_offset == 0:
                    owners[expert] = slot
                current_lane_loads[lane] += amount
                current_lane_members[lane].append(expert)
                capacities[lane] -= 1

        assign(owner_experts, 0)
        assign(replica_experts, args.primary_slots_per_rank)

    if bool((layout < 0).any()) or bool((owners < 0).any()):
        raise RuntimeError("Overlapping layout materialization is incomplete.")
    lut = np.empty((args.ep_size, args.num_experts), dtype=np.int64)
    for source_rank in range(args.ep_size):
        source_node = source_rank // args.ranks_per_node
        for expert in range(args.num_experts):
            local_slot = int(slot_by_node_expert[source_node, expert])
            lut[source_rank, expert] = local_slot if local_slot >= 0 else int(owners[expert])
    return layout, owners, lut


def _overlapping_library_candidates(
    samples: list[list[torch.Tensor]],
    *,
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    affinity: np.ndarray,
    affinity_by_source_node: np.ndarray,
    evaluator: _HybridEvaluator,
    layer_seed: int,
    args: argparse.Namespace,
) -> list[ClosureResult]:
    """Generate from-scratch variable-copy layouts without a paired warm start."""

    num_nodes = args.ep_size // args.ranks_per_node
    proposals: list[ClosureResult] = []
    seen: set[tuple[int, ...]] = set()
    for restart in range(max(2, args.node_restarts)):
        membership = _overlapping_expert_membership(
            samples,
            seed=layer_seed + 7919 * restart,
            args=args,
        )
        local_benefit = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for cluster in range(num_nodes):
            experts = np.flatnonzero(membership[cluster])
            local_benefit[cluster] = demand_by_node[:, experts].sum(axis=1)
        clusters, nodes = linear_sum_assignment(-local_benefit)
        cluster_to_node = np.empty((num_nodes,), dtype=np.int64)
        cluster_to_node[clusters] = nodes
        physical_membership = np.zeros_like(membership)
        for cluster in range(num_nodes):
            physical_membership[cluster_to_node[cluster]] = membership[cluster]
        key = tuple(int(value) for value in physical_membership.reshape(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        try:
            layout, owners, initial_lut = _materialize_overlapping_layout(
                physical_membership,
                demand_by_rank=demand_by_rank,
                demand_by_node=demand_by_node,
                affinity=affinity,
                args=args,
            )
        except RuntimeError:
            continue
        for load_weight, locality_weight in ((0.0, 0.5), (0.5, 1.0), (2.0, 2.0)):
            lut = _compile_affinity_source_lut(
                layout,
                owners,
                initial_lut,
                demand_by_rank=demand_by_rank,
                affinity_by_source_node=affinity_by_source_node,
                load_weight=load_weight,
                locality_weight=locality_weight,
                iterations=args.closure_lut_iterations,
                args=args,
            )
            cost = evaluator.evaluate(samples, lut)
            selected_by_node = tuple(
                tuple(int(value) for value in np.flatnonzero(physical_membership[node])) for node in range(num_nodes)
            )
            proposals.append(
                ClosureResult(
                    strategy=(f"overlap_r{restart}_l{load_weight:g}_n{locality_weight:g}"),
                    layout=layout.copy(),
                    owners=owners.copy(),
                    lut=lut,
                    cost=cost,
                    selected_by_node=selected_by_node,
                )
            )
    return proposals


def _batch_statistics(
    batch_labels: np.ndarray,
    demand_by_node: np.ndarray,
    affinity_by_source_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = demand_by_node.shape[0]
    num_batches = int(batch_labels.max()) + 1
    batch_demand = np.zeros((num_nodes, num_batches), dtype=np.float64)
    batch_affinity = np.zeros((num_nodes, num_batches, num_batches), dtype=np.float64)
    membership = np.eye(num_batches, dtype=np.float64)[batch_labels]
    for source_node in range(num_nodes):
        batch_demand[source_node] = demand_by_node[source_node] @ membership
        batch_affinity[source_node] = membership.T @ affinity_by_source_node[source_node] @ membership
        np.fill_diagonal(batch_affinity[source_node], 0.0)
    return batch_demand, batch_affinity


def _compile_batch_service_nodes(
    membership: np.ndarray,
    batch_demand: np.ndarray,
    batch_affinity: np.ndarray,
    *,
    locality_weight: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Choose one static destination node per source-node/batch pair."""

    num_batches, num_nodes = membership.shape
    labels = np.empty((num_nodes, num_batches), dtype=np.int64)
    for source_node in range(num_nodes):
        for batch in range(num_batches):
            available = np.flatnonzero(membership[batch])
            if membership[batch, source_node]:
                labels[source_node, batch] = source_node
            else:
                index = min(len(available) - 1, source_node * len(available) // num_nodes)
                labels[source_node, batch] = int(available[index])
    for _ in range(max(1, iterations)):
        changed = False
        for source_node in range(num_nodes):
            assignment = np.eye(num_nodes, dtype=np.float64)[labels[source_node]]
            affinity_to_node = batch_affinity[source_node] @ assignment
            for batch in np.argsort(-batch_demand[source_node], kind="stable").tolist():
                current_node = int(labels[source_node, batch])
                best: tuple[float, int] | None = None
                for node in np.flatnonzero(membership[batch]).tolist():
                    score = float(affinity_to_node[batch, node])
                    if node == source_node:
                        score += float(locality_weight) * float(batch_demand[source_node, batch])
                    key = (score, int(node == current_node), -node)
                    if best is None or key > (best[0], int(best[1] == current_node), -best[1]):
                        best = (score, node)
                if best is not None and best[1] != int(labels[source_node, batch]):
                    labels[source_node, batch] = best[1]
                    changed = True
        if not changed:
            break

    affinity_reward = 0.0
    locality_reward = 0.0
    target_loads = np.zeros((num_nodes,), dtype=np.float64)
    for source_node in range(num_nodes):
        same = labels[source_node, :, None] == labels[source_node, None, :]
        affinity_reward += float((batch_affinity[source_node] * same).sum()) / 2.0
        for batch in range(num_batches):
            node = int(labels[source_node, batch])
            amount = float(batch_demand[source_node, batch])
            target_loads[node] += amount
            if node == source_node:
                locality_reward += amount
    total_affinity = max(float(batch_affinity.sum()) / 2.0, 1.0)
    total_demand = max(float(batch_demand.sum()), 1.0)
    proxy = (
        affinity_reward / total_affinity
        + float(locality_weight) * locality_reward / total_demand
        - 0.5 * float(target_loads.max()) / total_demand
    )
    return labels, proxy


def _refine_batch_membership(
    initial: np.ndarray,
    batch_demand: np.ndarray,
    batch_affinity: np.ndarray,
    *,
    locality_weight: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Capacity-preserving batch replacement with variable batch copy counts."""

    membership = initial.copy()
    service, score = _compile_batch_service_nodes(
        membership,
        batch_demand,
        batch_affinity,
        locality_weight=locality_weight,
        iterations=8,
    )
    num_batches, num_nodes = membership.shape
    for _ in range(max(0, rounds)):
        copy_counts = membership.sum(axis=1)
        best: tuple[float, int, int, np.ndarray] | None = None
        for node in range(num_nodes):
            removable = np.flatnonzero(membership[:, node] & (copy_counts > 1))
            addable = np.flatnonzero(~membership[:, node])
            for victim in removable.tolist():
                for added in addable.tolist():
                    candidate = membership.copy()
                    candidate[victim, node] = False
                    candidate[added, node] = True
                    candidate_service, candidate_score = _compile_batch_service_nodes(
                        candidate,
                        batch_demand,
                        batch_affinity,
                        locality_weight=locality_weight,
                        iterations=8,
                    )
                    if candidate_score <= score + 1e-12:
                        continue
                    key = (candidate_score, -node, -victim)
                    if best is None or key > (best[0], -best[1], -best[2]):
                        best = (candidate_score, node, victim, candidate_service)
                        best_added = added
        if best is None:
            break
        score, node, victim, service = best
        membership[victim, node] = False
        membership[best_added, node] = True
    return membership, service


def _batch_lut(
    layout: np.ndarray,
    batch_labels: np.ndarray,
    service_nodes: np.ndarray,
    *,
    args: argparse.Namespace,
) -> np.ndarray:
    num_nodes = args.ep_size // args.ranks_per_node
    slot_by_node_expert = np.full((num_nodes, args.num_experts), -1, dtype=np.int64)
    for slot, expert in enumerate(layout.tolist()):
        node = slot // args.slots_per_rank // args.ranks_per_node
        slot_by_node_expert[node, expert] = slot
    lut = np.empty((args.ep_size, args.num_experts), dtype=np.int64)
    for source_rank in range(args.ep_size):
        source_node = source_rank // args.ranks_per_node
        for expert in range(args.num_experts):
            node = int(service_nodes[source_node, batch_labels[expert]])
            slot = int(slot_by_node_expert[node, expert])
            if slot < 0:
                raise RuntimeError("Batch service policy points to an absent expert copy.")
            lut[source_rank, expert] = slot
    return lut


def _top_batch_membership_neighbors(
    membership: np.ndarray,
    batch_demand: np.ndarray,
    batch_affinity: np.ndarray,
    *,
    locality_weight: float,
    limit: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    copy_counts = membership.sum(axis=1)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for node in range(membership.shape[1]):
        removable = np.flatnonzero(membership[:, node] & (copy_counts > 1))
        addable = np.flatnonzero(~membership[:, node])
        for victim in removable.tolist():
            for added in addable.tolist():
                candidate = membership.copy()
                candidate[victim, node] = False
                candidate[added, node] = True
                service, score = _compile_batch_service_nodes(
                    candidate,
                    batch_demand,
                    batch_affinity,
                    locality_weight=locality_weight,
                    iterations=8,
                )
                candidates.append((score, candidate, service))
    candidates.sort(key=lambda value: value[0], reverse=True)
    return [(membership, service, score) for score, membership, service in candidates[: max(0, limit)]]


def _materialize_batch_result(
    membership: np.ndarray,
    service_nodes: np.ndarray,
    batch_labels: np.ndarray,
    *,
    strategy: str,
    samples: list[list[torch.Tensor]],
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    affinity: np.ndarray,
    affinity_by_source_node: np.ndarray,
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
) -> ClosureResult | None:
    num_nodes = args.ep_size // args.ranks_per_node
    expert_membership = membership[batch_labels].T
    assignment_by_target_node = np.zeros(
        (num_nodes, args.num_experts),
        dtype=np.float64,
    )
    affinity_by_target_node = np.zeros(
        (num_nodes, args.num_experts, args.num_experts),
        dtype=np.float64,
    )
    for source_node in range(num_nodes):
        for expert in range(args.num_experts):
            target_node = int(service_nodes[source_node, batch_labels[expert]])
            assignment_by_target_node[target_node, expert] += demand_by_node[
                source_node,
                expert,
            ]
        expert_targets = service_nodes[source_node, batch_labels]
        for target_node in range(num_nodes):
            indices = np.flatnonzero(expert_targets == target_node)
            affinity_by_target_node[target_node][np.ix_(indices, indices)] += affinity_by_source_node[source_node][
                np.ix_(indices, indices)
            ]
    try:
        layout, owners, _ = _materialize_overlapping_layout(
            expert_membership,
            demand_by_rank=demand_by_rank,
            demand_by_node=demand_by_node,
            affinity=affinity,
            assignment_by_node=assignment_by_target_node,
            affinity_by_node=affinity_by_target_node,
            args=args,
        )
    except RuntimeError:
        return None
    lut = _batch_lut(layout, batch_labels, service_nodes, args=args)
    return ClosureResult(
        strategy=strategy,
        layout=layout,
        owners=owners,
        lut=lut,
        cost=evaluator.evaluate(samples, lut),
        selected_by_node=tuple(
            tuple(int(value) for value in np.flatnonzero(expert_membership[node])) for node in range(num_nodes)
        ),
    )


def _affinity_batch_candidates(
    samples: list[list[torch.Tensor]],
    *,
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    affinity: np.ndarray,
    affinity_by_source_node: np.ndarray,
    evaluator: _HybridEvaluator,
    layer_seed: int,
    args: argparse.Namespace,
) -> list[ClosureResult]:
    """Construct variable-copy layouts from independently clustered expert batches."""

    num_nodes = args.ep_size // args.ranks_per_node
    batch_size = min(int(args.closure_batch_size), args.num_experts // num_nodes)
    num_batches = args.num_experts // batch_size
    if batch_size * num_batches != args.num_experts:
        raise ValueError("Affinity batches must evenly partition the logical experts.")
    proposals: list[ClosureResult] = []
    node_pairings = (
        ((0, 2), (1, 3)),
        ((0, 1), (2, 3)),
        ((0, 3), (1, 2)),
    )
    partition_specs: list[tuple[str, np.ndarray, int, float]] = [
        (
            "sequential",
            np.arange(args.num_experts, dtype=np.int64) // (args.num_experts // 2),
            layer_seed + 99_999,
            0.0,
        )
    ]
    seed_and_load: list[tuple[int, float]] = [
        (layer_seed + 100_000, 0.0),
        (layer_seed + 100_001, 2.0),
    ]
    if args.closure_wide_search:
        seed_and_load.extend(
            (
                layer_seed + 3571 * restart,
                (0.0, 0.5, 2.0)[restart % 3],
            )
            for restart in range(max(0, int(args.node_restarts) - len(seed_and_load)))
        )
    for restart, (partition_seed, partition_load_weight) in enumerate(seed_and_load):
        expert_super_labels = _balanced_spectral_partition(
            affinity,
            demand_by_rank.sum(axis=0),
            parts=2,
            capacity=args.num_experts // 2,
            seed=partition_seed,
            load_weight=partition_load_weight,
            iterations=args.partition_iterations,
        )
        partition_specs.append(
            (
                f"spectral_{restart}",
                expert_super_labels,
                partition_seed,
                partition_load_weight,
            )
        )
    active_pairings = node_pairings if args.closure_wide_search else node_pairings[:1]
    locality_weights = (0.5, 1.0, 2.0) if args.closure_wide_search else (0.5,)
    refinement_rounds = max(0, int(args.closure_refinement_rounds))
    round_options = (0,) if refinement_rounds == 0 else (0, refinement_rounds)
    for partition_name, expert_super_labels, partition_seed, partition_load_weight in partition_specs:
        batches_per_super = num_batches // 2
        batch_labels = np.full((args.num_experts,), -1, dtype=np.int64)
        for super_group in range(2):
            experts = np.flatnonzero(expert_super_labels == super_group)
            sub_labels = _balanced_spectral_partition(
                affinity[np.ix_(experts, experts)],
                demand_by_rank[:, experts].sum(axis=0),
                parts=batches_per_super,
                capacity=batch_size,
                seed=partition_seed + 53 * (super_group + 1),
                load_weight=partition_load_weight,
                iterations=max(8, args.partition_iterations // 2),
            )
            batch_labels[experts] = super_group * batches_per_super + sub_labels
        batch_demand, batch_affinity = _batch_statistics(
            batch_labels,
            demand_by_node,
            affinity_by_source_node,
        )
        super_labels = np.arange(num_batches, dtype=np.int64) // batches_per_super
        for pairing_index, pair_by_super in enumerate(active_pairings):
            initial = np.zeros((num_batches, num_nodes), dtype=bool)
            for batch in range(num_batches):
                initial[batch, list(pair_by_super[int(super_labels[batch])])] = True
            for locality_weight, refinement_rounds in itertools.product(
                locality_weights,
                round_options,
            ):
                strategy = f"batch_{partition_name}_p{pairing_index}_n{locality_weight:g}_x{refinement_rounds}"
                membership, service_nodes = _refine_batch_membership(
                    initial,
                    batch_demand,
                    batch_affinity,
                    locality_weight=locality_weight,
                    rounds=refinement_rounds,
                )
                result = _materialize_batch_result(
                    membership,
                    service_nodes,
                    batch_labels,
                    strategy=strategy,
                    samples=samples,
                    demand_by_rank=demand_by_rank,
                    demand_by_node=demand_by_node,
                    affinity=affinity,
                    affinity_by_source_node=affinity_by_source_node,
                    evaluator=evaluator,
                    args=args,
                )
                if result is not None:
                    proposals.append(result)
            neighbors = _top_batch_membership_neighbors(
                initial,
                batch_demand,
                batch_affinity,
                locality_weight=0.5,
                limit=args.closure_exact_neighbors,
            )
            for neighbor_index, (membership, service_nodes, _) in enumerate(neighbors):
                result = _materialize_batch_result(
                    membership,
                    service_nodes,
                    batch_labels,
                    strategy=f"batch_exact_{partition_name}_p{pairing_index}_q{neighbor_index}",
                    samples=samples,
                    demand_by_rank=demand_by_rank,
                    demand_by_node=demand_by_node,
                    affinity=affinity,
                    affinity_by_source_node=affinity_by_source_node,
                    evaluator=evaluator,
                    args=args,
                )
                if result is not None:
                    proposals.append(result)
    return proposals


def _closure_replica_candidates(
    samples: list[list[torch.Tensor]],
    primary_layout: np.ndarray,
    primary_owners: np.ndarray,
    *,
    demand_by_rank: np.ndarray,
    demand_by_node: np.ndarray,
    affinity_by_source_node: np.ndarray,
    evaluator: _HybridEvaluator,
    layer_seed: int,
    args: argparse.Namespace,
) -> list[ClosureResult]:
    proposals: list[ClosureResult] = []
    if args.closure_legacy_proposals:
        closures = _token_closures_by_source_node(samples, primary_owners, args=args)
        seen: set[tuple[int, ...]] = set()
        selection_variants = (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.5, 0.125),
        )
        lut_variants = (
            (0.0, 0.5),
            (0.5, 1.0),
            (2.0, 2.0),
        )
        for batch_penalty, hot_weight in selection_variants:
            selected_by_node = _select_closure_replicas(
                closures,
                primary_owners,
                demand_by_node,
                batch_penalty=batch_penalty,
                hot_weight=hot_weight,
                args=args,
            )
            layout, initial_lut = _materialize_replica_layout(
                primary_layout,
                primary_owners,
                selected_by_node=selected_by_node,
                demand_by_rank=demand_by_rank,
                affinity_by_source_node=affinity_by_source_node,
                args=args,
            )
            layout_key = tuple(int(value) for value in layout.tolist())
            if layout_key in seen:
                continue
            seen.add(layout_key)
            for load_weight, locality_weight in lut_variants:
                lut = _compile_affinity_source_lut(
                    layout,
                    primary_owners,
                    initial_lut,
                    demand_by_rank=demand_by_rank,
                    affinity_by_source_node=affinity_by_source_node,
                    load_weight=load_weight,
                    locality_weight=locality_weight,
                    iterations=args.closure_lut_iterations,
                    args=args,
                )
                cost = evaluator.evaluate(samples, lut)
                proposals.append(
                    ClosureResult(
                        strategy=(f"closure_p{batch_penalty:g}_h{hot_weight:g}_l{load_weight:g}_n{locality_weight:g}"),
                        layout=layout.copy(),
                        owners=primary_owners.copy(),
                        lut=lut,
                        cost=cost,
                        selected_by_node=tuple(tuple(row) for row in selected_by_node),
                    )
                )
        overlapping = _overlapping_library_candidates(
            samples,
            demand_by_rank=demand_by_rank,
            demand_by_node=demand_by_node,
            affinity=affinity_by_source_node.sum(axis=0),
            affinity_by_source_node=affinity_by_source_node,
            evaluator=evaluator,
            layer_seed=layer_seed,
            args=args,
        )
        if overlapping:
            best_overlap = min(overlapping, key=lambda value: value.cost.total_ms)
            print(
                f"overlap_candidates={len(overlapping)} best={best_overlap.strategy} "
                f"cost_ms={best_overlap.cost.total_ms:.3f}",
                flush=True,
            )
        else:
            print("overlap_candidates=0", flush=True)
        proposals.extend(overlapping)
    batched = _affinity_batch_candidates(
        samples,
        demand_by_rank=demand_by_rank,
        demand_by_node=demand_by_node,
        affinity=affinity_by_source_node.sum(axis=0),
        affinity_by_source_node=affinity_by_source_node,
        evaluator=evaluator,
        layer_seed=layer_seed,
        args=args,
    )
    if batched:
        best_batched = min(batched, key=lambda value: value.cost.total_ms)
        print(
            f"batch_candidates={len(batched)} best={best_batched.strategy} cost_ms={best_batched.cost.total_ms:.3f}",
            flush=True,
        )
    proposals.extend(batched)
    return proposals


def _structured_replica_candidates(
    primary_layout: np.ndarray,
    primary_owners: np.ndarray,
    *,
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    num_nodes = args.ep_size // args.ranks_per_node
    primary_node = primary_owners // args.slots_per_rank // args.ranks_per_node
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for destination_by_source in itertools.permutations(range(num_nodes)):
        if any(destination_by_source[source] == source for source in range(num_nodes)):
            continue
        selected_by_node: list[list[int]] = [[] for _ in range(num_nodes)]
        for source, destination in enumerate(destination_by_source):
            selected_by_node[destination] = [int(expert) for expert in np.flatnonzero(primary_node == source).tolist()]
        layout, lut = _materialize_replica_layout(
            primary_layout,
            primary_owners,
            selected_by_node=selected_by_node,
            demand_by_rank=demand_by_rank,
            affinity_by_source_node=affinity_by_source_node,
            args=args,
        )
        label = "cluster_pair_" + "".join(str(value) for value in destination_by_source)
        results.append((label, layout, lut))
    return results


def _paired_supercluster_candidate(
    super_labels: np.ndarray,
    primary_pair_by_group: tuple[tuple[int, int], tuple[int, int]],
    *,
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = args.ep_size // args.ranks_per_node
    if num_nodes != 4:
        raise ValueError("Paired-supercluster placement requires exactly four nodes.")
    source_to_target = np.full((num_nodes, 2), -1, dtype=np.int64)
    for group, pair in enumerate(primary_pair_by_group):
        for source_node in range(num_nodes):
            if source_node in pair:
                source_to_target[source_node, group] = source_node
            else:
                source_to_target[source_node, group] = pair[source_node // 2]

    assignment_by_node = np.zeros((num_nodes, args.num_experts), dtype=np.float64)
    affinity_by_target_node = np.zeros(
        (num_nodes, args.num_experts, args.num_experts),
        dtype=np.float64,
    )
    for source_rank in range(args.ep_size):
        source_node = source_rank // args.ranks_per_node
        for group in range(2):
            experts = super_labels == group
            target_node = int(source_to_target[source_node, group])
            assignment_by_node[target_node, experts] += demand_by_rank[source_rank, experts]
            affinity_by_target_node[target_node] += affinity_by_source_node[source_node]

    owner_node = np.full((args.num_experts,), -1, dtype=np.int64)
    for group, pair in enumerate(primary_pair_by_group):
        experts = np.flatnonzero(super_labels == group)
        score = assignment_by_node[pair[0], experts] - assignment_by_node[pair[1], experts]
        order = experts[np.argsort(-score, kind="stable")]
        midpoint = len(order) // 2
        owner_node[order[:midpoint]] = pair[0]
        owner_node[order[midpoint:]] = pair[1]

    layout = np.full((args.ep_size * args.slots_per_rank,), -1, dtype=np.int64)
    owners = np.full((args.num_experts,), -1, dtype=np.int64)
    slot_by_node_expert = np.full((num_nodes, args.num_experts), -1, dtype=np.int64)
    for node in range(num_nodes):
        group = next(group for group, pair in enumerate(primary_pair_by_group) if node in pair)
        experts = np.flatnonzero(super_labels == group)
        owner_experts = [int(expert) for expert in experts if owner_node[expert] == node]
        replica_experts = [int(expert) for expert in experts if owner_node[expert] != node]
        lane_loads = np.zeros((args.ranks_per_node,), dtype=np.float64)
        lane_members: list[list[int]] = [[] for _ in range(args.ranks_per_node)]

        def assign(
            experts_to_place: list[int],
            slot_offset: int,
            *,
            current_node: int = node,
            current_lane_loads: np.ndarray = lane_loads,
            current_lane_members: list[list[int]] = lane_members,
        ) -> None:
            capacities = np.full(
                (args.ranks_per_node,),
                args.primary_slots_per_rank,
                dtype=np.int64,
            )
            for expert in sorted(
                experts_to_place,
                key=lambda value: (-assignment_by_node[current_node, value], value),
            ):
                best: tuple[float, float, int] | None = None
                amount = float(assignment_by_node[current_node, expert])
                for lane in range(args.ranks_per_node):
                    if capacities[lane] <= 0:
                        continue
                    projected_peak = max(
                        current_lane_loads[lane] + amount,
                        float(current_lane_loads.max()),
                    )
                    affinity_gain = sum(
                        affinity_by_target_node[current_node, expert, other] for other in current_lane_members[lane]
                    )
                    key = (projected_peak, -float(affinity_gain), lane)
                    if best is None or key < best:
                        best = key
                if best is None:
                    raise RuntimeError("Paired-supercluster rank placement exhausted all slots.")
                lane = best[2]
                local_slot = slot_offset + (args.primary_slots_per_rank - capacities[lane])
                rank = current_node * args.ranks_per_node + lane
                slot = rank * args.slots_per_rank + local_slot
                layout[slot] = expert
                slot_by_node_expert[current_node, expert] = slot
                if slot_offset == 0:
                    owners[expert] = slot
                current_lane_loads[lane] += amount
                current_lane_members[lane].append(expert)
                capacities[lane] -= 1

        assign(owner_experts, 0)
        assign(replica_experts, args.primary_slots_per_rank)

    if bool((owners < 0).any()) or bool((layout < 0).any()):
        raise RuntimeError("Paired-supercluster layout is incomplete.")
    lut = np.full((args.ep_size, args.num_experts), -1, dtype=np.int64)
    for source_rank in range(args.ep_size):
        source_node = source_rank // args.ranks_per_node
        for expert in range(args.num_experts):
            group = int(super_labels[expert])
            target_node = int(source_to_target[source_node, group])
            lut[source_rank, expert] = slot_by_node_expert[target_node, expert]
    return layout, owners, lut


def _paired_supercluster_candidates(
    affinity: np.ndarray,
    demand_by_rank: np.ndarray,
    affinity_by_source_node: np.ndarray,
    *,
    seed: int,
    iterations: int,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    partitions = [("sequential", np.arange(args.num_experts, dtype=np.int64) // (args.num_experts // 2))]
    for offset, load_weight in enumerate((0.0, 2.0)):
        labels = _balanced_spectral_partition(
            affinity,
            demand_by_rank.sum(axis=0),
            parts=2,
            capacity=args.num_experts // 2,
            seed=seed + offset,
            load_weight=load_weight,
            iterations=iterations,
        )
        partitions.append((f"spectral_{offset}", labels))

    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    physical_pairs = ((0, 2), (1, 3))
    for name, labels in partitions:
        for reverse in (False, True):
            pair_by_group = physical_pairs[::-1] if reverse else physical_pairs
            layout, owners, lut = _paired_supercluster_candidate(
                labels,
                pair_by_group,
                demand_by_rank=demand_by_rank,
                affinity_by_source_node=affinity_by_source_node,
                args=args,
            )
            results.append((f"paired_{name}_{int(reverse)}", layout, owners, lut))
    return results


def _normalize_layout_for_replay(
    target_layout: np.ndarray,
    target_owners: np.ndarray,
    target_lut: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    occupied_slots = [
        rank * args.slots_per_rank + local
        for rank in range(args.ep_size)
        for local in range(args.primary_slots_per_rank)
    ]
    current = np.full((args.ep_size * args.slots_per_rank,), -1, dtype=np.int64)
    for expert in range(args.num_experts):
        rank, local = divmod(expert, args.primary_slots_per_rank)
        current[rank * args.slots_per_rank + local] = expert
    actions: list[dict[str, str]] = []
    target_rank = target_owners // args.slots_per_rank
    for _ in range(args.num_experts):
        mismatched = [
            slot for slot in occupied_slots if target_rank[int(current[slot])] != slot // args.slots_per_rank
        ]
        if not mismatched:
            break
        slot = mismatched[0]
        lhs = int(current[slot])
        destination_rank = int(target_rank[lhs])
        destination_slots = [
            candidate
            for candidate in occupied_slots
            if candidate // args.slots_per_rank == destination_rank
            and target_rank[int(current[candidate])] != destination_rank
        ]
        if not destination_slots:
            raise RuntimeError(f"Cannot find a displaced expert in target rank {destination_rank}.")
        other = destination_slots[0]
        rhs = int(current[other])
        if slot // args.slots_per_rank == other // args.slots_per_rank:
            raise RuntimeError("Replay normalization produced a same-rank owner swap.")
        actions.append({"kind": "swap", "body": f"{lhs}<->{rhs}"})
        current[slot], current[other] = current[other], current[slot]
    else:
        raise RuntimeError("Owner placement did not converge.")

    for rank in range(args.ep_size):
        slots = [rank * args.slots_per_rank + local for local in range(args.primary_slots_per_rank)]
        actual = sorted(int(current[slot]) for slot in slots)
        expected = sorted(int(target_layout[slot]) for slot in slots)
        if actual != expected:
            raise RuntimeError(f"Replay normalization changed the expert set on rank {rank}.")

    for slot, expert in enumerate(target_layout.tolist()):
        if slot in occupied_slots:
            continue
        if expert < 0 or current[slot] == expert:
            continue
        source_slots = np.flatnonzero(current == expert)
        if len(source_slots) == 0:
            raise RuntimeError(f"Cannot replicate absent expert {expert}.")
        actions.append({"kind": "replica", "body": f"{expert}->{slot}"})
        current[slot] = expert

    normalized_owners = np.full((args.num_experts,), -1, dtype=np.int64)
    for slot in occupied_slots:
        normalized_owners[int(current[slot])] = slot
    if bool((normalized_owners < 0).any()):
        raise RuntimeError("Replay normalization lost a primary expert.")
    normalized_lut = target_lut.copy()
    for expert in range(args.num_experts):
        old_owner = int(target_owners[expert])
        new_owner = int(normalized_owners[expert])
        normalized_lut[normalized_lut[:, expert] == old_owner, expert] = new_owner
    for source_rank in range(args.ep_size):
        for expert in range(args.num_experts):
            if int(current[int(normalized_lut[source_rank, expert])]) != expert:
                raise RuntimeError("Replay normalization produced an invalid source route LUT.")
    return current, normalized_owners, normalized_lut, actions


def _layer_payload(
    *,
    layout: np.ndarray,
    owners: np.ndarray,
    lut: np.ndarray,
) -> dict[str, object]:
    return {
        "slot_to_logical": [int(value) for value in layout.tolist()],
        "owner_slots": [int(value) for value in owners.tolist()],
        "source_logical_to_physical": [[int(value) for value in row] for row in lut.tolist()],
    }


def _replay_payload(
    *,
    layouts: list[np.ndarray],
    owners: list[np.ndarray],
    luts: list[np.ndarray],
    args: argparse.Namespace,
    algorithm: str,
) -> dict[str, object]:
    layers: dict[str, object] = {}
    actions: list[dict[str, str]] = []
    for offset, (layout, owner, lut) in enumerate(zip(layouts, owners, luts, strict=True)):
        layer = args.layer_start + offset
        name = f"model.language_model.layers.{layer}.mlp.experts"
        normalized_layout, normalized_owner, normalized_lut, layer_actions = _normalize_layout_for_replay(
            layout,
            owner,
            lut,
            args=args,
        )
        layers[name] = _layer_payload(
            layout=normalized_layout,
            owners=normalized_owner,
            lut=normalized_lut,
        )
        for action in layer_actions:
            actions.append({"layer": name, **action})
    return {
        "schema_version": 2,
        "source": {
            "initial_layout": "canonical_empty",
            "algorithm": algorithm,
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
        },
        "topology": {
            "ep_size": args.ep_size,
            "num_experts": args.num_experts,
            "num_physical_slots": args.ep_size * args.slots_per_rank,
            "slots_per_rank": args.slots_per_rank,
        },
        "replay": {"actions_by_step": {"1": actions}},
        "layers": layers,
    }


def main() -> None:
    args = _parse_args()
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")
    if args.num_experts != args.ep_size * args.primary_slots_per_rank:
        raise ValueError("The primary slots must contain exactly one copy of every expert.")
    if args.primary_slots_per_rank >= args.slots_per_rank:
        raise ValueError("The full initialization experiment requires redundant slots.")

    evaluator = _HybridEvaluator(args)
    r2_layout, r2_owners, r2_lut = _mirrored_r2(
        ep_size=args.ep_size,
        num_experts=args.num_experts,
        slots_per_rank=args.slots_per_rank,
    )
    del r2_layout, r2_owners

    primary_layouts: list[np.ndarray] = []
    primary_owners: list[np.ndarray] = []
    primary_luts: list[np.ndarray] = []
    full_layouts: list[np.ndarray] = []
    full_owners: list[np.ndarray] = []
    full_luts: list[np.ndarray] = []
    greedy_layouts: list[np.ndarray] = []
    greedy_owners: list[np.ndarray] = []
    greedy_luts: list[np.ndarray] = []
    greedy_rows: list[dict[str, object]] = []
    closure_layouts: list[np.ndarray] = []
    closure_owners: list[np.ndarray] = []
    closure_luts: list[np.ndarray] = []
    closure_rows: list[dict[str, object]] = []
    layer_results: list[LayerResult] = []
    validation_rows: list[dict[str, object]] = []

    for layer in range(args.layer_start, args.layer_start + args.layers):
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
        demand_by_rank, demand_by_node, affinity, affinity_by_source_node = _route_statistics(
            optimize_samples,
            num_experts=args.num_experts,
            ranks_per_node=args.ranks_per_node,
        )
        best_primary: tuple[HybridCost, int, np.ndarray, np.ndarray, np.ndarray] | None = None
        load_weights = (0.0, 0.5, 2.0, 8.0)
        for restart in range(args.node_restarts):
            seed = args.seed + 1009 * layer + restart
            labels = _balanced_spectral_partition(
                affinity,
                demand_by_rank.sum(axis=0),
                parts=args.ep_size // args.ranks_per_node,
                capacity=args.num_experts // (args.ep_size // args.ranks_per_node),
                seed=seed,
                load_weight=load_weights[restart % len(load_weights)],
                iterations=args.partition_iterations,
            )
            expert_nodes = _map_node_clusters(labels, demand_by_node)
            expert_ranks = _assign_primary_ranks(
                expert_nodes,
                affinity=affinity,
                demand_by_rank=demand_by_rank,
                ranks_per_node=args.ranks_per_node,
                primary_slots_per_rank=args.primary_slots_per_rank,
                seed=seed,
                iterations=max(8, args.partition_iterations // 2),
            )
            layout, owners = _primary_layout(
                expert_ranks,
                ep_size=args.ep_size,
                num_experts=args.num_experts,
                slots_per_rank=args.slots_per_rank,
                primary_slots_per_rank=args.primary_slots_per_rank,
            )
            lut = _primary_lut(owners, args.ep_size)
            cost = evaluator.evaluate(optimize_samples, lut)
            if best_primary is None or cost.total_ms < best_primary[0].total_ms:
                best_primary = (cost, seed, layout, owners, lut)
        if best_primary is None:
            raise RuntimeError(f"No primary layout candidate for layer {layer}.")
        primary_cost, primary_seed, layout, owners, lut = best_primary
        primary_layouts.append(layout)
        primary_owners.append(owners)
        primary_luts.append(lut)

        greedy_result: _UnrestrictedGreedyResult | None = None
        if args.output_greedy is not None:
            greedy_result = _UnrestrictedReplicaGreedy(
                optimize_samples,
                layout,
                owners,
                lut,
                evaluator=evaluator,
                args=args,
            ).run()
            greedy_layouts.append(greedy_result.layout)
            greedy_owners.append(owners.copy())
            greedy_luts.append(greedy_result.lut)

        closure_result: ClosureResult | None = None
        closure_planner_ms: float | None = None
        if args.output_closure is not None:
            closure_started = time.perf_counter()
            closure_candidates = _closure_replica_candidates(
                optimize_samples,
                layout,
                owners,
                demand_by_rank=demand_by_rank,
                demand_by_node=demand_by_node,
                affinity_by_source_node=affinity_by_source_node,
                evaluator=evaluator,
                layer_seed=args.seed + 1009 * layer,
                args=args,
            )
            if not closure_candidates:
                raise RuntimeError(f"No closure-batch replica candidate for layer {layer}.")
            closure_result = min(closure_candidates, key=lambda value: value.cost.total_ms)
            closure_planner_ms = (time.perf_counter() - closure_started) * 1000.0
            closure_layouts.append(closure_result.layout)
            closure_owners.append(closure_result.owners)
            closure_luts.append(closure_result.lut)

        full_seed = args.seed + 1009 * layer + 100_000
        best_full: tuple[HybridCost, str, np.ndarray, np.ndarray, np.ndarray] | None = None
        for strategy, candidate_layout, candidate_owners, candidate_lut in _paired_supercluster_candidates(
            affinity,
            demand_by_rank,
            affinity_by_source_node,
            seed=full_seed,
            iterations=args.partition_iterations,
            args=args,
        ):
            candidate_cost = evaluator.evaluate(optimize_samples, candidate_lut)
            if best_full is None or candidate_cost.total_ms < best_full[0].total_ms:
                best_full = (
                    candidate_cost,
                    strategy,
                    candidate_layout,
                    candidate_owners,
                    candidate_lut,
                )
        if best_full is None:
            raise RuntimeError(f"No full layout candidate for layer {layer}.")
        full_cost, strategy, full_layout, full_owner, full_lut = best_full
        full_layouts.append(full_layout)
        full_owners.append(full_owner)
        full_luts.append(full_lut)

        r2_cost = evaluator.evaluate(optimize_samples, r2_lut)
        primary_rank = owners // args.slots_per_rank
        primary_node = primary_rank // args.ranks_per_node
        primary_node_loads = tuple(
            float(demand_by_rank[:, primary_node == node].sum()) for node in range(args.ep_size // args.ranks_per_node)
        )
        primary_rank_loads = tuple(
            float(demand_by_rank[:, primary_rank == rank].sum()) for rank in range(args.ep_size)
        )
        copy_counts = np.bincount(full_layout[full_layout >= 0], minlength=args.num_experts)
        result = LayerResult(
            layer=layer,
            primary_seed=primary_seed,
            primary_cost=primary_cost,
            replica_strategy=strategy,
            full_cost=full_cost,
            r2_cost=r2_cost,
            primary_gain_over_r2_ms=r2_cost.total_ms - primary_cost.total_ms,
            full_gain_over_r2_ms=r2_cost.total_ms - full_cost.total_ms,
            primary_node_loads=primary_node_loads,
            primary_rank_loads=primary_rank_loads,
            copy_counts=tuple(int(value) for value in copy_counts.tolist()),
        )
        layer_results.append(result)

        validation_row: dict[str, object] = {
            "layer": layer,
            "r2": asdict(evaluator.evaluate(validation_samples, r2_lut)),
            "primary": asdict(evaluator.evaluate(validation_samples, lut)),
            "full": asdict(evaluator.evaluate(validation_samples, full_lut)),
        }
        if greedy_result is not None:
            greedy_validation = evaluator.evaluate(validation_samples, greedy_result.lut)
            validation_row["unrestricted_greedy"] = asdict(greedy_validation)
            greedy_rows.append(
                {
                    "layer": layer,
                    "actions": len(greedy_result.actions),
                    "stopped_for_nonpositive_gain": greedy_result.stopped_for_nonpositive_gain,
                    "copy_counts": [
                        int(value)
                        for value in np.bincount(
                            greedy_result.layout[greedy_result.layout >= 0],
                            minlength=args.num_experts,
                        ).tolist()
                    ],
                    "optimize": asdict(greedy_result.cost),
                    "validation": asdict(greedy_validation),
                }
            )
        if closure_result is not None:
            closure_validation = evaluator.evaluate(validation_samples, closure_result.lut)
            validation_row["closure"] = asdict(closure_validation)
            closure_rows.append(
                {
                    "layer": layer,
                    "planner_ms": closure_planner_ms,
                    "strategy": closure_result.strategy,
                    "selected_by_node": [list(row) for row in closure_result.selected_by_node],
                    "copy_counts": [
                        int(value)
                        for value in np.bincount(
                            closure_result.layout[closure_result.layout >= 0],
                            minlength=args.num_experts,
                        ).tolist()
                    ],
                    "optimize": asdict(closure_result.cost),
                    "validation": asdict(closure_validation),
                }
            )
        validation_rows.append(validation_row)
        print(
            f"layer={layer:02d} seed={primary_seed} replica={strategy} "
            f"train_ms(r2/primary/full)="
            f"{r2_cost.total_ms:.3f}/{primary_cost.total_ms:.3f}/{full_cost.total_ms:.3f}"
            + ("" if greedy_result is None else f" unrestricted={greedy_result.cost.total_ms:.3f}")
            + ("" if closure_result is None else f" closure={closure_result.cost.total_ms:.3f}"),
            flush=True,
        )

    primary_payload = _replay_payload(
        layouts=primary_layouts,
        owners=primary_owners,
        luts=primary_luts,
        args=args,
        algorithm="hierarchical-primary-v1",
    )
    full_payload = _replay_payload(
        layouts=full_layouts,
        owners=full_owners,
        luts=full_luts,
        args=args,
        algorithm="hierarchical-primary-replica-v1",
    )
    greedy_payload = None
    if args.output_greedy is not None:
        greedy_payload = _replay_payload(
            layouts=greedy_layouts,
            owners=greedy_owners,
            luts=greedy_luts,
            args=args,
            algorithm="hierarchical-unrestricted-positive-marginal-v1",
        )
    closure_payload = None
    if args.output_closure is not None:
        closure_payload = _replay_payload(
            layouts=closure_layouts,
            owners=closure_owners,
            luts=closure_luts,
            args=args,
            algorithm="hierarchical-closure-batch-static-lut-v1",
        )
    validation_closure_ms = (
        None if not closure_rows else sum(float(row["validation"]["total_ms"]) for row in closure_rows)
    )
    full_validation_run = args.layer_start == 0 and args.layers == 48
    closure_e2e_eligible = bool(
        full_validation_run
        and validation_closure_ms is not None
        and validation_closure_ms < float(args.validation_baseline_ms)
    )
    report = {
        "schema_version": 1,
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "output_primary",
                "output_full",
                "output_greedy",
                "output_closure",
                "output_report",
            }
        },
        "layers": [asdict(row) for row in layer_results],
        "unrestricted_greedy": greedy_rows,
        "closure_batch": closure_rows,
        "validation": validation_rows,
        "aggregate": {
            "optimize_r2_ms": sum(row.r2_cost.total_ms for row in layer_results),
            "optimize_primary_ms": sum(row.primary_cost.total_ms for row in layer_results),
            "optimize_full_ms": sum(row.full_cost.total_ms for row in layer_results if row.full_cost is not None),
            "validation_r2_ms": sum(float(row["r2"]["total_ms"]) for row in validation_rows),
            "validation_primary_ms": sum(float(row["primary"]["total_ms"]) for row in validation_rows),
            "validation_full_ms": sum(float(row["full"]["total_ms"]) for row in validation_rows),
            "optimize_unrestricted_greedy_ms": (
                None if not greedy_rows else sum(float(row["optimize"]["total_ms"]) for row in greedy_rows)
            ),
            "validation_unrestricted_greedy_ms": (
                None if not greedy_rows else sum(float(row["validation"]["total_ms"]) for row in greedy_rows)
            ),
            "optimize_closure_ms": (
                None if not closure_rows else sum(float(row["optimize"]["total_ms"]) for row in closure_rows)
            ),
            "validation_closure_ms": validation_closure_ms,
            "closure_planner_mean_ms_per_layer": (
                None if not closure_rows else sum(float(row["planner_ms"]) for row in closure_rows) / len(closure_rows)
            ),
            "validation_baseline_ms": float(args.validation_baseline_ms),
            "closure_e2e_eligible": closure_e2e_eligible,
        },
    }
    outputs: list[tuple[Path, dict[str, object]]] = [
        (args.output_primary, primary_payload),
        (args.output_full, full_payload),
        (args.output_report, report),
    ]
    if args.output_greedy is not None and greedy_payload is not None:
        outputs.append((args.output_greedy, greedy_payload))
    if args.output_closure is not None and closure_payload is not None:
        outputs.append((args.output_closure, closure_payload))
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_closure is not None:
        comparison = "PASS" if closure_e2e_eligible else "FAIL"
        print(
            f"closure_validation_gate={comparison} cost_ms={validation_closure_ms} "
            f"baseline_ms={args.validation_baseline_ms}; "
            f"{'E2E is eligible' if closure_e2e_eligible else 'E2E must not run'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
