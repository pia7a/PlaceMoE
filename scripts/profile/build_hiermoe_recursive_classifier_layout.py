#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Build a capacity-general hierarchical expert layout from captured routes.

The offline initializer has three independent stages:

1. classify the unique logical experts into topology-derived balanced groups;
2. allocate an exact arbitrary replica budget from affinity classes;
3. jointly classify unique and redundant physical instances into nodes and
   ranks while alternating with the static source-rank routing LUT.

Affinity and assignment load propose complete states.  The existing exact
HierMoE hybrid evaluator selects states, so the emitted Forward LUT and the
offline cost model always use the same physical routes. Uniformly reserved
but inactive physical slots are represented as EMPTY and can be preloaded as
zero-filled expert state.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

from scripts.profile.build_hiermoe_hierarchical_init_layout import (
    HybridCost,
    _HybridEvaluator,
    _load_routes,
    _mirrored_r2,
    _replay_payload,
)
from veomni.distributed.moe.hiermoe.placemoe import (
    PartitionConfig,
    ProfileStatistics,
    build_replica_allocations,
    map_groups_to_locations,
    partition_items,
    profile_route_statistics,
    project_statistics_to_copies,
    uniform_copy_statistics,
)


@dataclass(frozen=True)
class _Candidate:
    strategy: str
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray
    lut_instances: np.ndarray
    logical_instances: np.ndarray
    instance_ranks: np.ndarray
    optimize_cost: HybridCost
    planner_ms: float
    alternations: int


@dataclass(frozen=True)
class _PartitionRefinement:
    labels: np.ndarray
    proxy_cost: float
    affinity_gain: float
    peak_load: float
    swaps: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _CapacityPlan:
    primary_slots_per_rank: int
    reserved_replicas: int
    active_replicas: int
    empty_slots: int


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
    parser.add_argument(
        "--layer-name-template",
        default="model.language_model.layers.{layer}.mlp.experts",
        help="Python format template for the runtime expert-module key.",
    )
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument(
        "--call-indices",
        type=_parse_int_list,
        default=(0,),
        help="Captured Forward call indices to include for every optimizer step.",
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
        "--expected-total-layers",
        type=int,
        default=48,
        help="Total MoE layers in the model. Used only to gate full-model E2E eligibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(48, max(1, (os.cpu_count() or 1) // 4)),
        help="Independent layer planner processes. One preserves the legacy serial execution.",
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=4,
        help="Concurrent exact candidate builders within each layer process.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads available to each layer planner process.",
    )
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument(
        "--slots-per-rank",
        type=int,
        default=None,
        help="Total reserved expert slots per rank. Defaults to primary capacity plus four redundant slots.",
    )
    parser.add_argument(
        "--redundant-slots-per-rank",
        type=int,
        default=None,
        help="Optional per-rank redundant capacity used to derive or validate slots-per-rank.",
    )
    parser.add_argument(
        "--primary-slots-per-rank",
        type=int,
        default=None,
        help="Canonical owner capacity per rank. Defaults to num_experts / ep_size.",
    )
    parser.add_argument(
        "--active-redundant-slots",
        type=int,
        default=None,
        help=(
            "Number of redundant copies to activate globally. Defaults to all reserved slots; "
            "unused uniformly reserved slots remain EMPTY."
        ),
    )
    parser.add_argument(
        "--replica-candidate-limit",
        type=int,
        default=64,
        help="Maximum exact-budget replica allocations evaluated per logical partition.",
    )
    parser.add_argument("--partition-restarts", type=int, default=3)
    parser.add_argument("--alternations", type=int, default=3)
    parser.add_argument("--lut-iterations", type=int, default=6)
    parser.add_argument("--partition-iterations", type=int, default=24)
    parser.add_argument("--hyperedge-token-sample", type=int, default=16384)
    parser.add_argument("--structured-shortlist", type=int, default=2)
    parser.add_argument(
        "--generic-instance-seed",
        action="store_true",
        help="Also evaluate the weaker uniform-copy instance-partition seed.",
    )
    parser.add_argument(
        "--hyperedge-seed",
        action="store_true",
        help="Also evaluate the denser token-KMeans node-library seed.",
    )
    parser.add_argument(
        "--communication-blind-proposals",
        action="store_true",
        help=(
            "Generate candidates and source LUTs from assignment demand only. "
            "This is required for a strict compute-only ablation: token "
            "co-occurrence, source locality, and hierarchical communication "
            "must not influence either proposal generation or exact selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument(
        "--comparison-validation-ms",
        type=float,
        default=6116.241273880005,
        help=(
            "Comparison-only held-out cost. It is never used to generate a candidate. "
            "Ignored when --comparison-layout is not 'none'."
        ),
    )
    parser.add_argument(
        "--comparison-layout",
        choices=("none", "mirrored-r2"),
        default="none",
        help=(
            "Optionally evaluate a matched comparison layout on the same held-out routes. "
            "The main paper matrix uses mirrored-r2 instead of a topology-specific constant."
        ),
    )
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _is_e2e_eligible(
    *,
    layer_start: int,
    layers: int,
    expected_total_layers: int,
    validation_total_ms: float,
    comparison_validation_ms: float,
) -> bool:
    return bool(
        layer_start == 0 and layers == expected_total_layers and validation_total_ms <= comparison_validation_ms
    )


def _source_statistics(
    samples: list[list[torch.Tensor]],
    *,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    statistics = profile_route_statistics(samples, num_experts=num_experts)
    return statistics.demand.copy(), statistics.affinity.copy()


def _partition_coefficients(args: argparse.Namespace) -> tuple[float, float, float]:
    """Return calibrated inter-node, intra-node, and compute coefficients."""

    payload_bytes = float(args.hidden_size * args.bytes_per_element)
    communication_multiplier = float(args.communication_phase_multiplier)
    node_omega = communication_multiplier * payload_bytes * float(args.inter_ms_per_byte)
    rank_omega = communication_multiplier * payload_bytes * float(args.intra_ms_per_byte)
    gamma = float(args.compute_phase_multiplier) * float(args.compute_ms_per_assignment)
    return node_omega, rank_omega, gamma


def _logical_base_partitions(
    affinity: np.ndarray,
    demand: np.ndarray,
    *,
    num_nodes: int,
    restarts: int,
    iterations: int,
    seed: int,
    omega: float = 1.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    num_experts = int(affinity.shape[0])
    if num_experts % num_nodes:
        raise ValueError("Logical experts must divide evenly across nodes.")
    config = PartitionConfig(
        capacities=(num_experts // num_nodes,) * num_nodes,
        ranks_per_group=(1,) * num_nodes,
        omega=omega,
        gamma=gamma,
        restarts=restarts,
        exchange_limit=iterations,
        seed=seed,
    )
    results: list[np.ndarray] = []
    for result in partition_items(affinity, demand, config):
        groups = sorted(
            (tuple(np.flatnonzero(result.labels == label).tolist()) for label in range(num_nodes)),
            key=lambda row: row,
        )
        canonical = np.full_like(result.labels, -1)
        for label, experts in enumerate(groups):
            canonical[list(experts)] = label
        results.append(canonical)
    return results


def _replica_sets_from_partition(
    labels: np.ndarray,
    *,
    replicas: int,
) -> list[np.ndarray]:
    groups = [np.flatnonzero(labels == value) for value in range(int(labels.max()) + 1)]
    if replicas == 0:
        return [np.empty((0,), dtype=np.int64)]
    group_size = len(groups[0])
    if any(len(group) != group_size for group in groups):
        raise ValueError("Replica group enumeration requires equal-capacity classes.")
    if replicas % group_size:
        raise ValueError("Replica capacity must be a multiple of the base-group size.")
    chosen_groups = replicas // group_size
    if chosen_groups > len(groups):
        raise ValueError("Replica capacity exceeds the logical expert count.")
    return [
        np.sort(np.concatenate([groups[index] for index in combination])).astype(np.int64, copy=False)
        for combination in itertools.combinations(range(len(groups)), chosen_groups)
    ]


def _replica_allocations(
    affinity: np.ndarray,
    demand: np.ndarray,
    *,
    replicas: int,
    restarts: int,
    iterations: int,
    seed: int,
    candidate_limit: int,
    omega: float = 1.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    """Return exact-size replica multisets for any feasible global budget.

    Full expert-library copies are peeled off first. The remaining budget is
    represented as a union of equal affinity classes whose size is
    ``gcd(num_experts, residual)``. This is the capacity-general form of the
    previous whole-node-class enumeration: for example, 96 replicas among 128
    experts becomes three classes chosen from four, while 16 replicas becomes
    one class chosen from eight.
    """

    num_experts = int(affinity.shape[0])
    if affinity.shape != (num_experts, num_experts):
        raise ValueError("Replica affinity must be square.")
    if demand.shape != (num_experts,):
        raise ValueError("Replica demand shape does not match the expert count.")
    if replicas < 0:
        raise ValueError("Replica capacity must be non-negative.")
    if candidate_limit <= 0:
        raise ValueError("Replica candidate limit must be positive.")

    _, residual = divmod(int(replicas), num_experts)
    if residual == 0:
        return build_replica_allocations(
            [],
            demand,
            additional_copies=replicas,
            candidate_limit=candidate_limit,
        )

    group_size = int(np.gcd(num_experts, residual))
    num_groups = num_experts // group_size
    partitions = (
        [np.arange(num_experts, dtype=np.int64)]
        if group_size == 1
        else _logical_base_partitions(
            affinity,
            demand,
            num_nodes=num_groups,
            restarts=restarts,
            iterations=iterations,
            seed=seed,
            omega=omega,
            gamma=gamma,
        )
    )
    return build_replica_allocations(
        partitions,
        demand,
        additional_copies=replicas,
        candidate_limit=candidate_limit,
    )


def _logical_instances(
    num_experts: int,
    replica_experts: np.ndarray,
    *,
    total_slots: int | None = None,
) -> np.ndarray:
    active = np.concatenate(
        [
            np.arange(num_experts, dtype=np.int64),
            replica_experts.astype(np.int64, copy=False),
        ]
    )
    if total_slots is None:
        return active
    if total_slots < len(active):
        raise ValueError("Physical slot capacity is smaller than the active expert instance count.")
    return np.pad(active, (0, total_slots - len(active)), constant_values=-1)


def _structured_instance_node_candidates(
    labels: np.ndarray,
    logical_instances: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ranks_per_node: int,
) -> list[tuple[str, np.ndarray]]:
    """Enumerate every degree-two base-class overlap for four full node libraries."""

    ep_size = int(demand_by_source.shape[0])
    if ep_size % ranks_per_node:
        raise ValueError("Source ranks must divide evenly into nodes.")
    num_nodes = ep_size // ranks_per_node
    if num_nodes != 4 or bool((logical_instances < 0).any()):
        return []
    if len(logical_instances) != 2 * len(labels):
        return []
    if not bool((np.bincount(logical_instances, minlength=len(labels)) == 2).all()):
        return []
    group_pairs = list(itertools.combinations(range(num_nodes), 2))
    patterns: list[tuple[tuple[int, int], ...]] = []
    for combination in itertools.combinations_with_replacement(range(len(group_pairs)), num_nodes):
        degree = np.zeros((num_nodes,), dtype=np.int64)
        rows = tuple(group_pairs[index] for index in combination)
        for row in rows:
            degree[list(row)] += 1
        if bool((degree == 2).all()):
            patterns.append(rows)

    instances_by_expert = [np.flatnonzero(logical_instances == expert).tolist() for expert in range(len(labels))]
    source_nodes = demand_by_source.reshape(
        num_nodes,
        ranks_per_node,
        len(labels),
    ).sum(axis=1)
    results: list[tuple[str, np.ndarray]] = []
    for pattern_index, pattern in enumerate(patterns):
        abstract_membership = np.zeros((num_nodes, len(labels)), dtype=bool)
        for abstract_node, group_pair in enumerate(pattern):
            abstract_membership[abstract_node] = np.isin(labels, group_pair)
        locality = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for abstract_node in range(num_nodes):
            experts = np.flatnonzero(abstract_membership[abstract_node])
            locality[abstract_node] = source_nodes[:, experts].sum(axis=1)
        abstract_rows, physical_nodes = linear_sum_assignment(-locality)
        abstract_to_physical = np.full((num_nodes,), -1, dtype=np.int64)
        abstract_to_physical[abstract_rows] = physical_nodes

        instance_nodes = np.full((len(logical_instances),), -1, dtype=np.int64)
        for expert in range(len(labels)):
            abstract_nodes = np.flatnonzero(abstract_membership[:, expert])
            if len(abstract_nodes) != 2:
                raise RuntimeError("Structured overlap did not place two copies of an expert.")
            for instance, abstract_node in zip(
                instances_by_expert[expert],
                abstract_nodes.tolist(),
                strict=True,
            ):
                instance_nodes[instance] = abstract_to_physical[abstract_node]
        expected = len(logical_instances) // num_nodes
        if not bool((np.bincount(instance_nodes, minlength=num_nodes) == expected).all()):
            raise RuntimeError("Structured overlap violates node capacity.")
        results.append((f"degree2_{pattern_index}", instance_nodes))
    return results


def _group_coherent_node_lut(
    samples: list[list[torch.Tensor]],
    logical_instances: np.ndarray,
    instance_nodes: np.ndarray,
    logical_groups: np.ndarray,
    *,
    ranks_per_node: int,
    communication_ms_per_token: float,
    assignment_ms_per_assignment: float,
    route_statistics: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Choose one shared destination node per source-node and logical class."""

    ep_size = len(samples[0])
    num_nodes = ep_size // ranks_per_node
    num_experts = len(logical_groups)
    num_groups = int(logical_groups.max()) + 1
    instances_by_expert_node = np.full(
        (num_experts, num_nodes),
        -1,
        dtype=np.int64,
    )
    for instance, expert in enumerate(logical_instances.tolist()):
        instances_by_expert_node[expert, instance_nodes[instance]] = instance

    nodes_by_group: list[tuple[int, ...]] = []
    for group in range(num_groups):
        experts = np.flatnonzero(logical_groups == group)
        common = set(np.flatnonzero(instances_by_expert_node[experts[0]] >= 0).tolist())
        for expert in experts[1:].tolist():
            common &= set(np.flatnonzero(instances_by_expert_node[expert] >= 0).tolist())
        if not common:
            raise RuntimeError("A structured logical class has no common serving node.")
        nodes_by_group.append(tuple(sorted(common)))

    if route_statistics is None:
        route_statistics = _group_route_statistics(
            samples,
            logical_groups,
            ranks_per_node=ranks_per_node,
        )
    assignments_by_group, group_mask_histogram = route_statistics
    if assignments_by_group.shape != (num_nodes, num_groups):
        raise ValueError("Group assignment statistics do not match the topology.")
    if group_mask_histogram.shape != (num_nodes, 1 << num_groups):
        raise ValueError("Group-mask statistics do not match the logical partition.")

    source_candidates: list[list[tuple[tuple[int, ...], float, np.ndarray]]] = []
    for source_node in range(num_nodes):
        candidates: list[tuple[tuple[int, ...], float, np.ndarray]] = []
        for destination_nodes in itertools.product(*nodes_by_group):
            node_by_group = torch.tensor(destination_nodes, dtype=torch.long)
            remote_hits = 0.0
            node_assignments = np.zeros((num_nodes,), dtype=np.float64)
            for group, destination_node in enumerate(destination_nodes):
                node_assignments[destination_node] += assignments_by_group[source_node, group]
            for mask, count in enumerate(group_mask_histogram[source_node].tolist()):
                if count == 0:
                    continue
                remote_nodes = {
                    int(node_by_group[group])
                    for group in range(num_groups)
                    if mask & (1 << group) and int(node_by_group[group]) != source_node
                }
                remote_hits += float(count * len(remote_nodes))
            candidates.append(
                (
                    tuple(int(value) for value in destination_nodes),
                    remote_hits,
                    node_assignments,
                )
            )
        if not candidates:
            raise RuntimeError("No group-coherent source LUT candidate was generated.")
        source_candidates.append(candidates)

    shapes = tuple(len(rows) for rows in source_candidates)
    combinations = np.indices(shapes, dtype=np.int64).reshape(num_nodes, -1).T
    remote_hits = np.zeros((len(combinations),), dtype=np.float64)
    node_assignments = np.zeros((len(combinations), num_nodes), dtype=np.float64)
    destination_tables: list[list[tuple[int, ...]]] = []
    for source_node, candidates in enumerate(source_candidates):
        indices = combinations[:, source_node]
        remote_table = np.asarray([candidate[1] for candidate in candidates])
        assignment_table = np.stack([candidate[2] for candidate in candidates])
        remote_hits += remote_table[indices]
        node_assignments += assignment_table[indices]
        destination_tables.append([candidate[0] for candidate in candidates])
    communication_ms = float(communication_ms_per_token) * remote_hits
    assignment_ms = float(assignment_ms_per_assignment) * node_assignments.max(axis=1) / ranks_per_node
    total_ms = communication_ms + assignment_ms
    best_index = int(
        np.lexsort(
            (
                np.arange(len(combinations)),
                assignment_ms,
                communication_ms,
                total_ms,
            )
        )[0]
    )
    destination_rows = tuple(
        destination_tables[source_node][int(combinations[best_index, source_node])] for source_node in range(num_nodes)
    )

    lut = np.full((ep_size, num_experts), -1, dtype=np.int64)
    for source_node, destination_nodes in enumerate(destination_rows):
        source_start = source_node * ranks_per_node
        for expert in range(num_experts):
            node = destination_nodes[int(logical_groups[expert])]
            instance = int(instances_by_expert_node[expert, node])
            if instance < 0:
                raise RuntimeError("Group-coherent LUT references an absent expert copy.")
            lut[source_start : source_start + ranks_per_node, expert] = instance
    return lut, float(total_ms[best_index])


def _group_route_statistics(
    samples: list[list[torch.Tensor]],
    logical_groups: np.ndarray,
    *,
    ranks_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize exact token-group incidence once for all structured seeds."""

    ep_size = len(samples[0])
    if ep_size % ranks_per_node:
        raise ValueError("Source ranks must divide evenly into nodes.")
    num_nodes = ep_size // ranks_per_node
    num_groups = int(logical_groups.max()) + 1
    if num_groups >= 63:
        raise ValueError("Group-mask statistics require fewer than 63 logical groups.")
    assignments_by_group = np.zeros((num_nodes, num_groups), dtype=np.float64)
    group_mask_histogram = np.zeros((num_nodes, 1 << num_groups), dtype=np.int64)
    group_lut = torch.from_numpy(logical_groups).to(torch.long)
    for sample in samples:
        for source_rank, route in enumerate(sample):
            source_node = source_rank // ranks_per_node
            groups = group_lut.index_select(0, route.reshape(-1)).view_as(route)
            assignments_by_group[source_node] += torch.bincount(
                groups.reshape(-1),
                minlength=num_groups,
            ).numpy()
            masks = torch.zeros((route.shape[0],), dtype=torch.long)
            for position in range(route.shape[1]):
                masks.bitwise_or_(
                    torch.bitwise_left_shift(
                        torch.ones_like(groups[:, position]),
                        groups[:, position],
                    )
                )
            group_mask_histogram[source_node] += torch.bincount(
                masks,
                minlength=1 << num_groups,
            ).numpy()
    return assignments_by_group, group_mask_histogram


def _node_proxy_lut(
    samples: list[list[torch.Tensor]],
    logical_instances: np.ndarray,
    instance_nodes: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    *,
    ranks_per_node: int,
    iterations: int = 4,
) -> tuple[np.ndarray, float]:
    ep_size, num_experts = demand_by_source.shape
    choices = [np.flatnonzero(logical_instances == expert) for expert in range(num_experts)]
    lut = np.full((ep_size, num_experts), -1, dtype=np.int64)
    for source_rank in range(ep_size):
        source_node = source_rank // ranks_per_node
        for expert in range(num_experts):
            local = choices[expert][instance_nodes[choices[expert]] == source_node]
            lut[source_rank, expert] = int(local[0] if len(local) else choices[expert][0])
    for _ in range(iterations):
        changed = False
        for source_rank in range(ep_size):
            selected_nodes = instance_nodes[lut[source_rank]]
            source_node = source_rank // ranks_per_node
            for expert in np.argsort(-demand_by_source[source_rank], kind="stable").tolist():
                affinity_row = affinity_by_source[source_rank, expert]
                best: tuple[float, int] | None = None
                for instance in choices[expert].tolist():
                    node = int(instance_nodes[instance])
                    reward = float(affinity_row[selected_nodes == node].sum())
                    if node == source_node:
                        reward += float(demand_by_source[source_rank, expert])
                    key = (reward, -node, -instance)
                    if best is None or key > (best[0], -int(instance_nodes[best[1]]), -best[1]):
                        best = (reward, instance)
                if best is not None and best[1] != int(lut[source_rank, expert]):
                    lut[source_rank, expert] = best[1]
                    selected_nodes[expert] = instance_nodes[best[1]]
                    changed = True
        if not changed:
            break

    score = 0.0
    torch_lut = torch.from_numpy(lut).to(torch.long)
    torch_nodes = torch.from_numpy(instance_nodes).to(torch.long)
    for sample in samples:
        for source_rank, logical in enumerate(sample):
            mapped = torch_lut[source_rank].index_select(0, logical.reshape(-1)).view_as(logical)
            nodes = torch_nodes.index_select(0, mapped.reshape(-1)).view_as(mapped)
            hits = torch.zeros((logical.shape[0], ep_size // ranks_per_node), dtype=torch.bool)
            hits.scatter_(1, nodes, True)
            score += float(hits.sum().item())
    return lut, score


def _uniform_instance_statistics(
    logical_instances: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return uniform_copy_statistics(
        ProfileStatistics(demand=demand_by_source, affinity=affinity_by_source),
        logical_instances,
    )


def _mapped_instance_statistics(
    samples: list[list[torch.Tensor]],
    lut_instances: np.ndarray,
    *,
    logical_instances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    statistics = profile_route_statistics(samples, num_experts=lut_instances.shape[1])
    return project_statistics_to_copies(statistics, logical_instances, lut_instances)


def _partition_proxy_cost(
    affinity: np.ndarray,
    demand: np.ndarray,
    labels: np.ndarray,
    *,
    parts: int,
    affinity_ms_per_hit: float,
    assignment_ms_per_assignment: float,
    assignment_divisor: float = 1.0,
    peak_floor: float = 0.0,
) -> tuple[float, float, float]:
    """Return the explainable co-occurrence and assignment-peak proxy."""

    if affinity.shape != (len(labels), len(labels)):
        raise ValueError("Affinity shape does not match the partition labels.")
    if demand.shape != (len(labels),):
        raise ValueError("Demand shape does not match the partition labels.")
    within = 0.0
    for part in range(parts):
        members = np.flatnonzero(labels == part)
        within += float(np.triu(affinity[np.ix_(members, members)], k=1).sum())
    loads = np.bincount(labels, weights=demand, minlength=parts)
    peak = max(float(loads.max(initial=0.0)), float(peak_floor))
    cost = -float(affinity_ms_per_hit) * within + float(assignment_ms_per_assignment) * peak / max(
        float(assignment_divisor), 1.0
    )
    return cost, within, peak


def _refine_balanced_partition(
    affinity: np.ndarray,
    demand: np.ndarray,
    initial_labels: np.ndarray,
    *,
    parts: int,
    capacity: int,
    affinity_ms_per_hit: float,
    assignment_ms_per_assignment: float,
    assignment_divisor: float = 1.0,
    peak_floor: float = 0.0,
    max_swaps: int = 32,
    item_kinds: np.ndarray | None = None,
    forbid_duplicate_kinds: bool = False,
    improvement_epsilon: float = 1e-9,
) -> _PartitionRefinement:
    """Improve a balanced partition with exact vectorized pair exchanges.

    The same primitive is used for node and rank classification.  It rewards
    token co-occurrence kept inside a group and penalizes the largest
    non-deduplicated assignment load.  Pair exchanges preserve every group's
    capacity, and an optional kind constraint prevents two copies of one
    logical expert from occupying the same rank.
    """

    labels = np.asarray(initial_labels, dtype=np.int64).copy()
    affinity = np.asarray(affinity, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)
    if len(labels) != parts * capacity:
        raise ValueError("Balanced partition size does not match parts times capacity.")
    if not bool((np.bincount(labels, minlength=parts) == capacity).all()):
        raise ValueError("Initial labels do not satisfy the requested capacity.")
    if not np.allclose(affinity, affinity.T):
        raise ValueError("Partition affinity must be symmetric.")
    if item_kinds is not None:
        item_kinds = np.asarray(item_kinds, dtype=np.int64)
        if item_kinds.shape != labels.shape:
            raise ValueError("Item-kind shape does not match partition labels.")

    one_hot = np.eye(parts, dtype=np.float64)[labels]
    affinity_to_part = affinity @ one_hot
    loads = np.bincount(labels, weights=demand, minlength=parts).astype(np.float64)
    kind_counts: np.ndarray | None = None
    if forbid_duplicate_kinds:
        if item_kinds is None:
            raise ValueError("Duplicate-kind filtering requires item kinds.")
        kind_counts = np.zeros((parts, int(item_kinds.max(initial=-1)) + 1), dtype=np.int64)
        active_kinds = item_kinds >= 0
        np.add.at(kind_counts, (labels[active_kinds], item_kinds[active_kinds]), 1)
        if bool((kind_counts > 1).any()):
            raise ValueError("Initial partition already contains duplicate item kinds.")

    swaps: list[tuple[int, int]] = []
    lhs_all, rhs_all = np.triu_indices(len(labels), k=1)
    for _ in range(max(0, max_swaps)):
        different = labels[lhs_all] != labels[rhs_all]
        lhs = lhs_all[different]
        rhs = rhs_all[different]
        if not len(lhs):
            break
        lhs_parts = labels[lhs]
        rhs_parts = labels[rhs]

        legal = np.ones((len(lhs),), dtype=bool)
        if kind_counts is not None and item_kinds is not None:
            lhs_kinds = item_kinds[lhs]
            rhs_kinds = item_kinds[rhs]
            lhs_available = (lhs_kinds < 0) | (kind_counts[rhs_parts, np.maximum(lhs_kinds, 0)] == 0)
            rhs_available = (rhs_kinds < 0) | (kind_counts[lhs_parts, np.maximum(rhs_kinds, 0)] == 0)
            legal &= (lhs_kinds == rhs_kinds) | (lhs_available & rhs_available)
        if not bool(legal.any()):
            break

        affinity_gain = (
            affinity_to_part[lhs, rhs_parts]
            - affinity_to_part[lhs, lhs_parts]
            + affinity_to_part[rhs, lhs_parts]
            - affinity_to_part[rhs, rhs_parts]
            - 2.0 * affinity[lhs, rhs]
        )
        current_peak = max(float(loads.max(initial=0.0)), float(peak_floor))
        candidate_loads = np.broadcast_to(loads, (len(lhs), parts)).copy()
        row_ids = np.arange(len(lhs))
        candidate_loads[row_ids, lhs_parts] += demand[rhs] - demand[lhs]
        candidate_loads[row_ids, rhs_parts] += demand[lhs] - demand[rhs]
        candidate_peaks = np.maximum(candidate_loads.max(axis=1), float(peak_floor))
        deltas = -float(affinity_ms_per_hit) * affinity_gain + float(assignment_ms_per_assignment) * (
            candidate_peaks - current_peak
        ) / max(float(assignment_divisor), 1.0)
        deltas[~legal] = np.inf
        best_index = int(np.argmin(deltas))
        if not np.isfinite(deltas[best_index]) or deltas[best_index] >= -float(improvement_epsilon):
            break

        left = int(lhs[best_index])
        right = int(rhs[best_index])
        left_part = int(labels[left])
        right_part = int(labels[right])
        left_kind = None if item_kinds is None else int(item_kinds[left])
        right_kind = None if item_kinds is None else int(item_kinds[right])
        affinity_to_part[:, left_part] += affinity[:, right] - affinity[:, left]
        affinity_to_part[:, right_part] += affinity[:, left] - affinity[:, right]
        loads[left_part] += demand[right] - demand[left]
        loads[right_part] += demand[left] - demand[right]
        labels[left], labels[right] = labels[right], labels[left]
        if kind_counts is not None and left_kind is not None and right_kind is not None:
            if left_kind >= 0:
                kind_counts[left_part, left_kind] -= 1
                kind_counts[right_part, left_kind] += 1
            if right_kind >= 0:
                kind_counts[right_part, right_kind] -= 1
                kind_counts[left_part, right_kind] += 1
        swaps.append((left, right))

    proxy_cost, within, peak = _partition_proxy_cost(
        affinity,
        demand,
        labels,
        parts=parts,
        affinity_ms_per_hit=affinity_ms_per_hit,
        assignment_ms_per_assignment=assignment_ms_per_assignment,
        assignment_divisor=assignment_divisor,
        peak_floor=peak_floor,
    )
    initial_cost, initial_within, _ = _partition_proxy_cost(
        affinity,
        demand,
        np.asarray(initial_labels, dtype=np.int64),
        parts=parts,
        affinity_ms_per_hit=affinity_ms_per_hit,
        assignment_ms_per_assignment=assignment_ms_per_assignment,
        assignment_divisor=assignment_divisor,
        peak_floor=peak_floor,
    )
    if proxy_cost > initial_cost + max(float(improvement_epsilon), 1e-8):
        raise RuntimeError("Capacity-preserving refinement increased its proxy cost.")
    return _PartitionRefinement(
        labels=labels,
        proxy_cost=proxy_cost,
        affinity_gain=within - initial_within,
        peak_load=peak,
        swaps=tuple(swaps),
    )


def _map_clusters_to_locations(
    labels: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    sources_by_location: list[np.ndarray],
) -> np.ndarray:
    return map_groups_to_locations(
        labels,
        demand_by_source,
        sources_by_location=sources_by_location,
    )


def _hyperedge_instance_nodes(
    samples: list[list[torch.Tensor]],
    logical_instances: np.ndarray,
    *,
    num_nodes: int,
    capacity: int,
    num_experts: int,
    seed: int,
    sample_limit: int = 65536,
) -> np.ndarray:
    """Build overlapping node libraries from complete token top-k hyperedges."""

    routes = torch.cat([route for sample in samples for route in sample], dim=0).numpy()
    rng = np.random.default_rng(seed)
    if routes.shape[0] > sample_limit:
        indices = np.sort(rng.choice(routes.shape[0], size=sample_limit, replace=False))
        routes = routes[indices]
    features = np.zeros((routes.shape[0], num_experts), dtype=np.float32)
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
    token_labels = model.fit_predict(features)
    counts = np.zeros((num_nodes, num_experts), dtype=np.float64)
    for node_class in range(num_nodes):
        members = routes[token_labels == node_class]
        if members.size:
            counts[node_class] = np.bincount(
                members.reshape(-1),
                minlength=num_experts,
            )

    node_columns = np.repeat(np.arange(num_nodes, dtype=np.int64), capacity)
    benefit = np.zeros((len(logical_instances), len(node_columns)), dtype=np.float64)
    active = logical_instances >= 0
    benefit[active] = counts[node_columns[:, None], logical_instances[active][None, :]].T
    instances, columns = linear_sum_assignment(-benefit)
    instance_nodes = np.full((len(logical_instances),), -1, dtype=np.int64)
    instance_nodes[instances] = node_columns[columns]
    if bool((instance_nodes < 0).any()):
        raise RuntimeError("Hyperedge node-library assignment is incomplete.")
    if not bool((np.bincount(instance_nodes, minlength=num_nodes) == capacity).all()):
        raise RuntimeError("Hyperedge node-library assignment violates node capacity.")
    return instance_nodes


def _greedy_ranks_with_fixed_nodes(
    instance_nodes: np.ndarray,
    instance_demand: np.ndarray,
    instance_affinity: np.ndarray,
    *,
    ranks_per_node: int,
    slots_per_rank: int,
    logical_instances: np.ndarray | None = None,
) -> np.ndarray:
    """Balance exact instance assignments before using affinity as a tie-break."""

    demand = instance_demand.sum(axis=0)
    affinity = instance_affinity.sum(axis=0)
    if logical_instances is None:
        logical_instances = np.arange(len(instance_nodes), dtype=np.int64)
    else:
        logical_instances = np.asarray(logical_instances, dtype=np.int64)
        if logical_instances.shape != instance_nodes.shape:
            raise ValueError("Logical instance shape does not match node assignment.")
    num_nodes = int(instance_nodes.max()) + 1
    ep_size = num_nodes * ranks_per_node
    num_experts = int(logical_instances.max(initial=-1)) + 1
    if num_experts % ep_size:
        raise ValueError("Logical experts must divide evenly across EP ranks.")
    primary_slots_per_rank = num_experts // ep_size
    max_empty_per_rank = slots_per_rank - primary_slots_per_rank
    instance_ranks = np.full_like(instance_nodes, -1)
    for node in range(num_nodes):
        members = np.flatnonzero(instance_nodes == node)
        capacities = np.full((ranks_per_node,), slots_per_rank, dtype=np.int64)
        loads = np.zeros((ranks_per_node,), dtype=np.float64)
        lane_members: list[list[int]] = [[] for _ in range(ranks_per_node)]
        active_members = logical_instances[members]
        multiplicity = np.bincount(
            active_members[active_members >= 0],
            minlength=int(active_members.max(initial=-1)) + 1,
        )
        for instance in sorted(
            members.tolist(),
            key=lambda value: (
                -int(multiplicity[int(logical_instances[value])]) if int(logical_instances[value]) >= 0 else 0,
                -float(demand[value]),
                value,
            ),
        ):
            best: tuple[float, float, int] | None = None
            amount = float(demand[instance])
            logical = int(logical_instances[instance])
            for lane in range(ranks_per_node):
                if capacities[lane] <= 0:
                    continue
                if logical < 0:
                    empty_count = sum(int(logical_instances[other]) < 0 for other in lane_members[lane])
                    if empty_count >= max_empty_per_rank:
                        continue
                if logical >= 0 and any(int(logical_instances[other]) == logical for other in lane_members[lane]):
                    continue
                projected_peak = max(loads[lane] + amount, float(loads.max()))
                affinity_gain = sum(float(affinity[instance, other]) for other in lane_members[lane])
                key = (projected_peak, -affinity_gain, lane)
                if best is None or key < best:
                    best = key
            if best is None:
                raise RuntimeError("Greedy rank classification exhausted its capacity.")
            lane = best[2]
            instance_ranks[instance] = node * ranks_per_node + lane
            capacities[lane] -= 1
            loads[lane] += amount
            lane_members[lane].append(instance)
    if bool((instance_ranks < 0).any()):
        raise RuntimeError("Greedy rank classification is incomplete.")
    return instance_ranks


def _rank_assignment_is_device_unique(
    instance_ranks: np.ndarray,
    logical_instances: np.ndarray,
    *,
    ep_size: int,
) -> bool:
    for rank in range(ep_size):
        active = logical_instances[(instance_ranks == rank) & (logical_instances >= 0)]
        if len(active) != len(np.unique(active)):
            return False
    return True


def _rank_assignment_supports_balanced_owners(
    instance_ranks: np.ndarray,
    logical_instances: np.ndarray,
    *,
    ep_size: int,
) -> bool:
    num_experts = int(logical_instances.max(initial=-1)) + 1
    if num_experts % ep_size:
        return False
    minimum_active = num_experts // ep_size
    for rank in range(ep_size):
        active = (instance_ranks == rank) & (logical_instances >= 0)
        if int(active.sum()) < minimum_active:
            return False
    return True


def _greedy_instance_nodes(
    instance_demand: np.ndarray,
    instance_affinity: np.ndarray,
    logical_instances: np.ndarray,
    *,
    num_nodes: int,
    ranks_per_node: int,
    slots_per_rank: int,
) -> np.ndarray:
    """Build a capacity-feasible node seed when spectral labels overpack a copy kind."""

    demand = instance_demand.sum(axis=0)
    affinity = instance_affinity.sum(axis=0)
    capacity = ranks_per_node * slots_per_rank
    ep_size = num_nodes * ranks_per_node
    num_experts = int(logical_instances.max(initial=-1)) + 1
    if num_experts % ep_size:
        raise ValueError("Logical experts must divide evenly across EP ranks.")
    primary_slots_per_node = (num_experts // ep_size) * ranks_per_node
    max_empty_per_node = capacity - primary_slots_per_node
    capacities = np.full((num_nodes,), capacity, dtype=np.int64)
    loads = np.zeros((num_nodes,), dtype=np.float64)
    node_members: list[list[int]] = [[] for _ in range(num_nodes)]
    active = logical_instances[logical_instances >= 0]
    multiplicity = np.bincount(active, minlength=int(active.max(initial=-1)) + 1)
    result = np.full((len(logical_instances),), -1, dtype=np.int64)
    for instance in sorted(
        range(len(logical_instances)),
        key=lambda value: (
            -int(multiplicity[int(logical_instances[value])]) if int(logical_instances[value]) >= 0 else 0,
            -float(demand[value]),
            value,
        ),
    ):
        logical = int(logical_instances[instance])
        amount = float(demand[instance])
        best: tuple[tuple[float, float, float, int], int] | None = None
        for node in range(num_nodes):
            if capacities[node] <= 0:
                continue
            if logical < 0:
                empty_count = sum(int(logical_instances[other]) < 0 for other in node_members[node])
                if empty_count >= max_empty_per_node:
                    continue
            if logical >= 0:
                same_kind = sum(int(logical_instances[other]) == logical for other in node_members[node])
                if same_kind >= ranks_per_node:
                    continue
            projected_peak = max(loads[node] + amount, float(loads.max(initial=0.0)))
            affinity_gain = sum(float(affinity[instance, other]) for other in node_members[node])
            sources = slice(node * ranks_per_node, (node + 1) * ranks_per_node)
            locality = float(instance_demand[sources, instance].sum())
            key = (projected_peak, -affinity_gain, -locality, node)
            if best is None or key < best[0]:
                best = (key, node)
        if best is None:
            raise RuntimeError("Could not construct a capacity-feasible node assignment.")
        node = best[1]
        result[instance] = node
        capacities[node] -= 1
        loads[node] += amount
        node_members[node].append(instance)
    return result


def _classify_instances(
    instance_demand: np.ndarray,
    instance_affinity: np.ndarray,
    logical_instances: np.ndarray,
    *,
    ep_size: int,
    ranks_per_node: int,
    slots_per_rank: int,
    seed: int,
    iterations: int,
    node_omega: float,
    rank_omega: float,
    gamma: float,
) -> np.ndarray:
    instances = int(instance_demand.shape[1])
    num_nodes = ep_size // ranks_per_node
    if instances != ep_size * slots_per_rank:
        raise ValueError("Instance count must equal the physical slot capacity.")
    affinity = instance_affinity.sum(axis=0)
    demand = instance_demand.sum(axis=0)
    node_config = PartitionConfig(
        capacities=(instances // num_nodes,) * num_nodes,
        ranks_per_group=(ranks_per_node,) * num_nodes,
        omega=node_omega,
        gamma=gamma,
        restarts=1,
        exchange_limit=iterations,
        seed=seed,
    )
    node_labels = partition_items(affinity, demand, node_config)[0].labels
    node_sources = [
        np.arange(node * ranks_per_node, (node + 1) * ranks_per_node, dtype=np.int64) for node in range(num_nodes)
    ]
    instance_nodes = _map_clusters_to_locations(
        node_labels,
        instance_demand,
        sources_by_location=node_sources,
    )

    instance_ranks = np.full((instances,), -1, dtype=np.int64)
    for node in range(num_nodes):
        members = np.flatnonzero(instance_nodes == node)
        rank_config = PartitionConfig(
            capacities=(slots_per_rank,) * ranks_per_node,
            ranks_per_group=(1,) * ranks_per_node,
            omega=rank_omega,
            gamma=gamma,
            restarts=1,
            exchange_limit=max(1, iterations // 2),
            seed=seed + 104729 * (node + 1),
        )
        local_labels = partition_items(
            affinity[np.ix_(members, members)],
            demand[members],
            rank_config,
        )[0].labels
        rank_sources = [np.asarray([node * ranks_per_node + lane], dtype=np.int64) for lane in range(ranks_per_node)]
        mapped_lanes = _map_clusters_to_locations(
            local_labels,
            instance_demand[:, members],
            sources_by_location=rank_sources,
        )
        instance_ranks[members] = node * ranks_per_node + mapped_lanes
    if bool((instance_ranks < 0).any()):
        raise RuntimeError("Hierarchical instance classification is incomplete.")
    counts = np.bincount(instance_ranks, minlength=ep_size)
    if not bool((counts == slots_per_rank).all()):
        raise RuntimeError("Hierarchical instance classification violates rank capacity.")
    valid = _rank_assignment_is_device_unique(
        instance_ranks,
        logical_instances,
        ep_size=ep_size,
    ) and _rank_assignment_supports_balanced_owners(
        instance_ranks,
        logical_instances,
        ep_size=ep_size,
    )
    if not valid:
        try:
            instance_ranks = _greedy_ranks_with_fixed_nodes(
                instance_nodes,
                instance_demand,
                instance_affinity,
                ranks_per_node=ranks_per_node,
                slots_per_rank=slots_per_rank,
                logical_instances=logical_instances,
            )
        except RuntimeError:
            pass
        valid = _rank_assignment_is_device_unique(
            instance_ranks,
            logical_instances,
            ep_size=ep_size,
        ) and _rank_assignment_supports_balanced_owners(
            instance_ranks,
            logical_instances,
            ep_size=ep_size,
        )
        if not valid:
            instance_nodes = _greedy_instance_nodes(
                instance_demand,
                instance_affinity,
                logical_instances,
                num_nodes=num_nodes,
                ranks_per_node=ranks_per_node,
                slots_per_rank=slots_per_rank,
            )
            instance_ranks = _greedy_ranks_with_fixed_nodes(
                instance_nodes,
                instance_demand,
                instance_affinity,
                ranks_per_node=ranks_per_node,
                slots_per_rank=slots_per_rank,
                logical_instances=logical_instances,
            )
    if not (
        _rank_assignment_is_device_unique(
            instance_ranks,
            logical_instances,
            ep_size=ep_size,
        )
        and _rank_assignment_supports_balanced_owners(
            instance_ranks,
            logical_instances,
            ep_size=ep_size,
        )
    ):
        raise RuntimeError("Hierarchical instance classification violates rank ownership or duplicate constraints.")
    return instance_ranks


def _initial_lut_instances(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ranks_per_node: int,
    prefer_local: bool = True,
) -> np.ndarray:
    ep_size, num_experts = demand_by_source.shape
    choices = [np.flatnonzero(logical_instances == expert) for expert in range(num_experts)]
    lut = np.full((ep_size, num_experts), -1, dtype=np.int64)
    rank_loads = np.zeros((ep_size,), dtype=np.float64)
    jobs = [
        (float(demand_by_source[source_rank, expert]), source_rank, expert)
        for source_rank in range(ep_size)
        for expert in range(num_experts)
    ]
    for amount, source_rank, expert in sorted(
        jobs,
        key=lambda row: (-row[0], row[1], row[2]),
    ):
        source_node = source_rank // ranks_per_node
        local = choices[expert][instance_ranks[choices[expert]] // ranks_per_node == source_node]
        candidates = local if prefer_local and len(local) else choices[expert]
        best: tuple[tuple[float, ...], int] | None = None
        for instance in candidates.tolist():
            destination_rank = int(instance_ranks[instance])
            key = (
                rank_loads[destination_rank] + amount,
                float(destination_rank),
                float(instance),
            )
            if best is None or key < best[0]:
                best = (key, instance)
        if best is None:
            raise RuntimeError(f"Logical expert {expert} has no physical instance.")
        lut[source_rank, expert] = best[1]
        rank_loads[int(instance_ranks[best[1]])] += amount
    return lut


def _optimize_lut_instances(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    initial_lut: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    *,
    ranks_per_node: int,
    iterations: int,
    node_weight: float,
    rank_weight: float,
    assignment_weight: float,
) -> np.ndarray:
    ep_size, num_experts = demand_by_source.shape
    choices = [np.flatnonzero(logical_instances == expert) for expert in range(num_experts)]
    lut = initial_lut.copy()
    rank_loads = np.zeros((ep_size,), dtype=np.float64)
    for source_rank in range(ep_size):
        for expert in range(num_experts):
            rank = int(instance_ranks[lut[source_rank, expert]])
            rank_loads[rank] += demand_by_source[source_rank, expert]
    total_affinity = max(float(affinity_by_source.sum()), 1.0)
    total_demand = max(float(demand_by_source.sum()), 1.0)

    for _ in range(max(0, iterations)):
        changed = False
        for source_rank in range(ep_size):
            order = np.argsort(-demand_by_source[source_rank], kind="stable")
            selected_ranks = instance_ranks[lut[source_rank]]
            selected_nodes = selected_ranks // ranks_per_node
            source_node = source_rank // ranks_per_node
            for expert in order.tolist():
                if len(choices[expert]) <= 1:
                    continue
                current_instance = int(lut[source_rank, expert])
                current_rank = int(instance_ranks[current_instance])
                amount = float(demand_by_source[source_rank, expert])
                affinity_row = affinity_by_source[source_rank, expert]
                best: tuple[float, int] | None = None
                for candidate in choices[expert].tolist():
                    candidate_rank = int(instance_ranks[candidate])
                    candidate_node = candidate_rank // ranks_per_node
                    projected = rank_loads.copy()
                    projected[current_rank] -= amount
                    projected[candidate_rank] += amount
                    node_reward = float(affinity_row[selected_nodes == candidate_node].sum())
                    rank_reward = float(affinity_row[selected_ranks == candidate_rank].sum())
                    if candidate_node == source_node:
                        node_reward += amount
                    if candidate_rank == source_rank:
                        rank_reward += amount
                    score = (
                        float(node_weight) * node_reward / total_affinity
                        + float(rank_weight) * rank_reward / total_affinity
                        - float(assignment_weight) * float(projected.max()) / total_demand
                    )
                    key = (score, -candidate_rank, -candidate)
                    if best is None or key > (best[0], -int(instance_ranks[best[1]]), -best[1]):
                        best = (score, candidate)
                if best is None or best[1] == current_instance:
                    continue
                new_instance = best[1]
                new_rank = int(instance_ranks[new_instance])
                rank_loads[current_rank] -= amount
                rank_loads[new_rank] += amount
                lut[source_rank, expert] = new_instance
                selected_ranks[expert] = new_rank
                selected_nodes[expert] = new_rank // ranks_per_node
                changed = True
        if not changed:
            break
    return lut


def _materialize_layout(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    lut_instances: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ep_size: int,
    slots_per_rank: int,
    primary_slots_per_rank: int,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner_columns = np.repeat(np.arange(ep_size, dtype=np.int64), primary_slots_per_rank)
    owner_cost = np.full((num_experts, len(owner_columns)), 1e30, dtype=np.float64)
    served = np.zeros((len(logical_instances),), dtype=np.float64)
    for source_rank in range(ep_size):
        for expert in range(num_experts):
            served[int(lut_instances[source_rank, expert])] += demand_by_source[source_rank, expert]
    for expert in range(num_experts):
        instances = np.flatnonzero(logical_instances == expert)
        for column, rank in enumerate(owner_columns.tolist()):
            eligible = instances[instance_ranks[instances] == rank]
            if len(eligible):
                owner_cost[expert, column] = -float(served[eligible].max())
    experts, columns = linear_sum_assignment(owner_cost)
    owner_instance = np.full((num_experts,), -1, dtype=np.int64)
    if len(experts) == num_experts and not bool((owner_cost[experts, columns] >= 1e29).any()):
        owner_rank = np.full((num_experts,), -1, dtype=np.int64)
        owner_rank[experts] = owner_columns[columns]
        for expert in range(num_experts):
            instances = np.flatnonzero((logical_instances == expert) & (instance_ranks == owner_rank[expert]))
            owner_instance[expert] = int(instances[np.argmax(served[instances])])
    else:
        # Canonical ownership is a gradient/checkpoint identity, not a
        # placement capacity. Fall back to the most-served copy when an
        # arbitrary active/empty budget makes balanced owners infeasible.
        for expert in range(num_experts):
            instances = np.flatnonzero(logical_instances == expert)
            owner_instance[expert] = int(instances[np.argmax(served[instances])])

    layout = np.full((ep_size * slots_per_rank,), -1, dtype=np.int64)
    owners = np.full((num_experts,), -1, dtype=np.int64)
    instance_to_slot = np.full((len(logical_instances),), -1, dtype=np.int64)
    owner_set = {int(instance) for instance in owner_instance.tolist()}
    for rank in range(ep_size):
        owner_instances = sorted(
            np.flatnonzero(
                (instance_ranks == rank) & np.isin(np.arange(len(logical_instances)), owner_instance)
            ).tolist(),
            key=lambda instance: int(logical_instances[instance]),
        )
        remaining = sorted(
            [int(instance) for instance in np.flatnonzero(instance_ranks == rank) if int(instance) not in owner_set],
            key=lambda instance: (int(logical_instances[instance]), instance),
        )
        ordered = owner_instances + remaining
        if len(ordered) != slots_per_rank:
            raise RuntimeError(f"Rank {rank} received {len(ordered)} physical instances.")
        for local_slot, instance in enumerate(ordered):
            slot = rank * slots_per_rank + local_slot
            expert = int(logical_instances[instance])
            instance_to_slot[instance] = slot
            if expert < 0:
                continue
            layout[slot] = expert
            if instance in owner_set:
                owners[expert] = slot
    if bool((owners < 0).any()) or bool((instance_to_slot < 0).any()):
        raise RuntimeError("Layout materialization lost an owner or physical instance.")
    lut = instance_to_slot[lut_instances]
    if not bool((layout[lut] == np.arange(num_experts, dtype=np.int64)[None, :]).all()):
        raise RuntimeError("Materialized source LUT does not reference the requested expert.")
    return layout, owners, lut


def _build_candidate(
    samples: list[list[torch.Tensor]],
    *,
    logical_instances: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    evaluator: _HybridEvaluator,
    args: argparse.Namespace,
    seed: int,
    strategy: str,
    fixed_instance_nodes: np.ndarray | None = None,
    fixed_initial_lut: np.ndarray | None = None,
    initial_instance_statistics: tuple[np.ndarray, np.ndarray] | None = None,
    cost_cache: dict[bytes, HybridCost] | None = None,
) -> _Candidate | None:
    started = time.perf_counter()
    communication_blind = bool(args.communication_blind_proposals)
    if fixed_instance_nodes is None:
        if initial_instance_statistics is None:
            instance_demand, instance_affinity = _uniform_instance_statistics(
                logical_instances,
                demand_by_source,
                affinity_by_source,
            )
        else:
            instance_demand, instance_affinity = initial_instance_statistics
    else:
        if fixed_initial_lut is None:
            node_lut, _ = _node_proxy_lut(
                samples,
                logical_instances,
                fixed_instance_nodes,
                demand_by_source,
                affinity_by_source,
                ranks_per_node=args.ranks_per_node,
            )
        else:
            node_lut = fixed_initial_lut
        instance_demand, instance_affinity = _mapped_instance_statistics(
            samples,
            node_lut,
            logical_instances=logical_instances,
        )
    best: _Candidate | None = None
    node_omega, rank_omega, gamma = _partition_coefficients(args)
    if communication_blind:
        node_omega = 0.0
        rank_omega = 0.0
    for alternation in range(args.alternations):
        if fixed_instance_nodes is None:
            instance_ranks = _classify_instances(
                instance_demand,
                instance_affinity,
                logical_instances,
                ep_size=args.ep_size,
                ranks_per_node=args.ranks_per_node,
                slots_per_rank=args.slots_per_rank,
                seed=seed + 1009 * alternation,
                iterations=args.partition_iterations,
                node_omega=node_omega,
                rank_omega=rank_omega,
                gamma=gamma,
            )
        else:
            instance_ranks = _greedy_ranks_with_fixed_nodes(
                fixed_instance_nodes,
                instance_demand,
                instance_affinity,
                ranks_per_node=args.ranks_per_node,
                slots_per_rank=args.slots_per_rank,
                logical_instances=logical_instances,
            )
        if fixed_initial_lut is None:
            initial_lut = _initial_lut_instances(
                logical_instances,
                instance_ranks,
                demand_by_source,
                ranks_per_node=args.ranks_per_node,
                prefer_local=not communication_blind,
            )
        else:
            initial_lut = fixed_initial_lut
        lut_variants = [initial_lut]
        if fixed_initial_lut is None:
            for assignment_weight in (8.0, 32.0, 128.0):
                lut_variants.append(
                    _optimize_lut_instances(
                        logical_instances,
                        instance_ranks,
                        initial_lut,
                        demand_by_source,
                        affinity_by_source,
                        ranks_per_node=args.ranks_per_node,
                        iterations=args.lut_iterations,
                        node_weight=0.0 if communication_blind else 1.0,
                        rank_weight=0.0 if communication_blind else 0.15,
                        assignment_weight=assignment_weight,
                    )
                )
        for lut_instances in lut_variants:
            try:
                layout, owners, lut = _materialize_layout(
                    logical_instances,
                    instance_ranks,
                    lut_instances,
                    demand_by_source,
                    ep_size=args.ep_size,
                    slots_per_rank=args.slots_per_rank,
                    primary_slots_per_rank=args.primary_slots_per_rank,
                    num_experts=args.num_experts,
                )
            except RuntimeError:
                continue
            cache_key = lut.tobytes()
            cost = None if cost_cache is None else cost_cache.get(cache_key)
            if cost is None:
                cost = evaluator.evaluate(samples, lut)
                if cost_cache is not None:
                    cost_cache[cache_key] = cost
            candidate = _Candidate(
                strategy=strategy,
                layout=layout,
                owners=owners,
                lut=lut,
                lut_instances=lut_instances.copy(),
                logical_instances=logical_instances.copy(),
                instance_ranks=instance_ranks.copy(),
                optimize_cost=cost,
                planner_ms=(time.perf_counter() - started) * 1000.0,
                alternations=alternation + 1,
            )
            if best is None or cost.total_ms < best.optimize_cost.total_ms:
                best = candidate
        if best is None:
            continue
        instance_demand, instance_affinity = _mapped_instance_statistics(
            samples,
            best.lut_instances,
            logical_instances=logical_instances,
        )
        if communication_blind:
            instance_affinity.fill(0.0)
    if best is None:
        return None
    return _Candidate(
        **{
            **best.__dict__,
            "planner_ms": (time.perf_counter() - started) * 1000.0,
        }
    )


def _validate_configuration(args: argparse.Namespace) -> _CapacityPlan:
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")
    if args.num_experts <= 0 or args.ep_size <= 0:
        raise ValueError("Expert and EP sizes must be positive.")
    if args.num_experts % args.ep_size:
        raise ValueError("This initializer requires logical experts to divide evenly across EP ranks.")

    primary_slots = args.num_experts // args.ep_size
    if args.primary_slots_per_rank is None:
        args.primary_slots_per_rank = primary_slots
    elif int(args.primary_slots_per_rank) != primary_slots:
        raise ValueError(
            "Primary slots per rank must equal num_experts / ep_size; "
            f"expected {primary_slots}, got {args.primary_slots_per_rank}."
        )
    configured_redundant = getattr(args, "redundant_slots_per_rank", None)
    if args.slots_per_rank is None:
        redundant_per_rank = 4 if configured_redundant is None else int(configured_redundant)
        if redundant_per_rank < 0:
            raise ValueError("Redundant slots per rank must be non-negative.")
        args.slots_per_rank = primary_slots + redundant_per_rank
    elif configured_redundant is not None and int(args.slots_per_rank) != primary_slots + int(configured_redundant):
        raise ValueError("slots-per-rank does not match primary plus redundant slots per rank.")
    if args.slots_per_rank < primary_slots:
        raise ValueError("Physical slots per rank cannot be smaller than the canonical owner capacity.")

    reserved = args.ep_size * (args.slots_per_rank - primary_slots)
    active = reserved if args.active_redundant_slots is None else int(args.active_redundant_slots)
    if active < 0 or active > reserved:
        raise ValueError(f"Active redundant slots must be between zero and reserved capacity {reserved}.")
    # A rank cannot hold the same logical expert twice. Consequently an
    # expert has at most one copy on every EP rank.
    if active > args.num_experts * (args.ep_size - 1):
        raise ValueError("Active replica budget exceeds the duplicate-free EP placement capacity.")
    return _CapacityPlan(
        primary_slots_per_rank=primary_slots,
        reserved_replicas=reserved,
        active_replicas=active,
        empty_slots=reserved - active,
    )


def _preloaded_replay_payload(
    *,
    layouts: list[np.ndarray],
    owners: list[np.ndarray],
    luts: list[np.ndarray],
    args: argparse.Namespace,
    algorithm: str,
) -> dict[str, object]:
    """Serialize layouts with inactive slots for direct startup preload.

    The legacy replay action language normalizes a full layout through owner
    swaps and replica fills. It cannot represent an arbitrary inactive-slot
    distribution without temporary actions, while the runtime's static
    preload path already validates and installs the exact layout atomically.
    """

    layers: dict[str, object] = {}
    for offset, (layout, owner, lut) in enumerate(zip(layouts, owners, luts, strict=True)):
        layer = args.layer_start + offset
        name = args.layer_name_template.format(layer=layer)
        layers[name] = {
            "slot_to_logical": [int(value) for value in layout.tolist()],
            "owner_slots": [int(value) for value in owner.tolist()],
            "source_logical_to_physical": [[int(value) for value in row] for row in lut.tolist()],
        }
    return {
        "schema_version": 2,
        "source": {
            "initial_layout": "preloaded",
            "algorithm": algorithm,
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
            "layer_name_template": args.layer_name_template,
            "requires_static_preload": True,
        },
        "topology": {
            "ep_size": args.ep_size,
            "num_experts": args.num_experts,
            "num_physical_slots": args.ep_size * args.slots_per_rank,
            "slots_per_rank": args.slots_per_rank,
        },
        "replay": {"actions_by_step": {"1": []}},
        "layers": layers,
    }


def _plan_layer(
    layer: int,
    *,
    args: argparse.Namespace,
    capacity: _CapacityPlan,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    torch.set_num_threads(int(args.worker_threads))
    evaluator = _HybridEvaluator(args)
    layer_started = time.perf_counter()
    optimize_samples = _load_routes(
        args.route_root,
        steps=args.optimize_steps,
        layer=layer,
        ep_size=args.ep_size,
        call_indices=args.call_indices,
        forward_repeats=args.forward_repeats,
        layer_stride=args.expected_total_layers,
    )
    validation_samples = _load_routes(
        args.route_root,
        steps=args.validation_steps,
        layer=layer,
        ep_size=args.ep_size,
        call_indices=args.call_indices,
        forward_repeats=args.forward_repeats,
        layer_stride=args.expected_total_layers,
    )
    source_demand, source_affinity = _source_statistics(
        optimize_samples,
        num_experts=args.num_experts,
    )
    if args.communication_blind_proposals:
        source_affinity.fill(0.0)
    # These are exact identities, not approximations: source statistics retain
    # one row per EP rank, and their affinity sum is the same global pair
    # count previously rebuilt by a second complete route scan.
    demand_by_rank = source_demand
    affinity = source_affinity.sum(axis=0)
    node_omega, _, gamma = _partition_coefficients(args)
    if args.communication_blind_proposals:
        node_omega = 0.0
    partitions = _logical_base_partitions(
        affinity,
        demand_by_rank.sum(axis=0),
        num_nodes=args.ep_size // args.ranks_per_node,
        restarts=args.partition_restarts,
        iterations=args.partition_iterations,
        seed=args.seed + 100_003 * layer,
        omega=node_omega,
        gamma=gamma,
    )
    replica_allocations = _replica_allocations(
        affinity,
        demand_by_rank.sum(axis=0),
        replicas=capacity.active_replicas,
        restarts=args.partition_restarts,
        iterations=args.partition_iterations,
        seed=args.seed + 100_003 * layer + 53_123,
        candidate_limit=args.replica_candidate_limit,
        omega=node_omega,
        gamma=gamma,
    )
    candidates: list[_Candidate] = []
    candidate_jobs: list[dict[str, object]] = []
    seen_replica_sets: set[tuple[bytes, bytes]] = set()
    uniform_statistics: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    cost_cache: dict[bytes, HybridCost] = {}
    for partition_index, partition in enumerate(partitions):
        for combination_index, replica_experts in enumerate(replica_allocations):
            key = (partition.tobytes(), replica_experts.tobytes())
            if key in seen_replica_sets:
                continue
            seen_replica_sets.add(key)
            logical_instances = _logical_instances(
                args.num_experts,
                replica_experts,
                total_slots=args.ep_size * args.slots_per_rank,
            )
            logical_key = logical_instances.tobytes()
            structured = (
                []
                if args.communication_blind_proposals
                else _structured_instance_node_candidates(
                    partition,
                    logical_instances,
                    source_demand,
                    ranks_per_node=args.ranks_per_node,
                )
            )
            structured_route_statistics = (
                _group_route_statistics(
                    optimize_samples,
                    partition,
                    ranks_per_node=args.ranks_per_node,
                )
                if structured
                else None
            )
            if args.generic_instance_seed or not structured:
                statistics = uniform_statistics.get(logical_key)
                if statistics is None:
                    statistics = _uniform_instance_statistics(
                        logical_instances,
                        source_demand,
                        source_affinity,
                    )
                    uniform_statistics[logical_key] = statistics
                candidate_jobs.append(
                    {
                        "logical_instances": logical_instances,
                        "demand_by_source": source_demand,
                        "affinity_by_source": source_affinity,
                        "seed": args.seed + 100_003 * layer + 997 * partition_index + 31 * combination_index,
                        "strategy": f"recursive_classifier_p{partition_index}_c{combination_index}",
                        "initial_instance_statistics": statistics,
                    }
                )
            proxy_rows: list[tuple[float, str, np.ndarray, np.ndarray]] = []
            for structured_strategy, structured_nodes in structured:
                coherent_lut, proxy_score = _group_coherent_node_lut(
                    optimize_samples,
                    logical_instances,
                    structured_nodes,
                    partition,
                    ranks_per_node=args.ranks_per_node,
                    communication_ms_per_token=(
                        args.communication_phase_multiplier
                        * args.hidden_size
                        * args.bytes_per_element
                        * args.inter_ms_per_byte
                    ),
                    assignment_ms_per_assignment=(
                        args.compute_phase_multiplier * args.compute_ms_per_assignment
                        + args.communication_phase_multiplier * args.route_ms_per_assignment
                    ),
                    route_statistics=structured_route_statistics,
                )
                proxy_rows.append(
                    (
                        proxy_score,
                        structured_strategy,
                        structured_nodes,
                        coherent_lut,
                    )
                )
            for _, structured_strategy, structured_nodes, coherent_lut in sorted(
                proxy_rows,
                key=lambda row: row[0],
            )[: args.structured_shortlist]:
                candidate_jobs.append(
                    {
                        "logical_instances": logical_instances,
                        "demand_by_source": source_demand,
                        "affinity_by_source": source_affinity,
                        "seed": args.seed + 100_003 * layer + 997 * partition_index + 31 * combination_index,
                        "strategy": (f"structured_{structured_strategy}_p{partition_index}_c{combination_index}"),
                        "fixed_instance_nodes": structured_nodes,
                        "fixed_initial_lut": coherent_lut,
                    }
                )
            if args.hyperedge_seed and not args.communication_blind_proposals:
                hyperedge_nodes = _hyperedge_instance_nodes(
                    optimize_samples,
                    logical_instances,
                    num_nodes=args.ep_size // args.ranks_per_node,
                    capacity=args.ranks_per_node * args.slots_per_rank,
                    num_experts=args.num_experts,
                    seed=args.seed + 100_003 * layer + 997 * partition_index + 31 * combination_index,
                    sample_limit=args.hyperedge_token_sample,
                )
                candidate_jobs.append(
                    {
                        "logical_instances": logical_instances,
                        "demand_by_source": source_demand,
                        "affinity_by_source": source_affinity,
                        "seed": args.seed + 100_003 * layer + 997 * partition_index + 31 * combination_index,
                        "strategy": f"hyperedge_recursive_p{partition_index}_c{combination_index}",
                        "fixed_instance_nodes": hyperedge_nodes,
                    }
                )
    build_candidate = partial(
        _build_candidate,
        optimize_samples,
        evaluator=evaluator,
        args=args,
        cost_cache=cost_cache,
    )
    if args.candidate_workers == 1:
        built_candidates = [build_candidate(**job) for job in candidate_jobs]
    else:
        with ThreadPoolExecutor(
            max_workers=min(args.candidate_workers, len(candidate_jobs)),
        ) as executor:
            built_candidates = list(executor.map(lambda job: build_candidate(**job), candidate_jobs))
    candidates.extend(candidate for candidate in built_candidates if candidate is not None)
    if not candidates:
        raise RuntimeError(f"No feasible recursive classifier candidate for layer {layer}.")
    best = min(candidates, key=lambda item: item.optimize_cost.total_ms)
    validation_cost = evaluator.evaluate(validation_samples, best.lut)
    comparison_validation_cost: HybridCost | None = None
    if args.comparison_layout == "mirrored-r2":
        if args.ep_size % 2 or args.ep_size * args.slots_per_rank != 2 * args.num_experts:
            raise ValueError("mirrored-r2 comparison requires an even EP size and exactly two full expert copies.")
        _r2_layout, _r2_owners, r2_lut = _mirrored_r2(
            ep_size=args.ep_size,
            num_experts=args.num_experts,
            slots_per_rank=args.slots_per_rank,
        )
        comparison_validation_cost = evaluator.evaluate(validation_samples, r2_lut)
    active_layout = best.layout[best.layout >= 0]
    copy_counts = np.bincount(active_layout, minlength=args.num_experts)
    layer_ms = (time.perf_counter() - layer_started) * 1000.0
    row: dict[str, object] = {
        "layer": layer,
        "strategy": best.strategy,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "strategy": candidate.strategy,
                "planner_ms": candidate.planner_ms,
                "optimize": asdict(candidate.optimize_cost),
            }
            for candidate in sorted(
                candidates,
                key=lambda item: item.optimize_cost.total_ms,
            )
        ],
        "planner_ms": layer_ms,
        "winner_planner_ms": best.planner_ms,
        "exact_route_evaluations": len(cost_cache),
        "uniform_statistics_builds": len(uniform_statistics),
        "alternations": best.alternations,
        "copy_counts": [int(value) for value in copy_counts.tolist()],
        "optimize": asdict(best.optimize_cost),
        "validation": asdict(validation_cost),
        "comparison_validation": (
            asdict(comparison_validation_cost) if comparison_validation_cost is not None else None
        ),
    }
    print(
        f"layer={layer:02d} strategy={best.strategy} candidates={len(candidates)} "
        f"optimize_ms={best.optimize_cost.total_ms:.3f} "
        f"validation_ms={validation_cost.total_ms:.3f} "
        f"planner_ms={layer_ms:.1f}",
        flush=True,
    )
    return layer, best.layout, best.owners, best.lut, row


def main() -> None:
    args = _parse_args()
    if "{layer}" not in args.layer_name_template:
        raise ValueError("--layer-name-template must contain the '{layer}' placeholder.")
    capacity = _validate_configuration(args)
    if args.workers <= 0 or args.candidate_workers <= 0 or args.worker_threads <= 0:
        raise ValueError("workers, candidate-workers, and worker-threads must be positive.")
    layer_ids = tuple(range(args.layer_start, args.layer_start + args.layers))
    worker = partial(_plan_layer, args=args, capacity=capacity)
    wall_started = time.perf_counter()
    if args.workers == 1:
        results = [worker(layer) for layer in layer_ids]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(layer_ids))) as executor:
            results = list(executor.map(worker, layer_ids))
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    results.sort(key=lambda item: item[0])
    layouts = [item[1] for item in results]
    owners = [item[2] for item in results]
    luts = [item[3] for item in results]
    rows = [item[4] for item in results]

    validation_total = sum(float(row["validation"]["total_ms"]) for row in rows)
    if args.comparison_layout == "mirrored-r2":
        comparison_ms = sum(float(row["comparison_validation"]["total_ms"]) for row in rows)
    else:
        comparison_ms = float(args.comparison_validation_ms)
    report = {
        "schema_version": 1,
        "algorithm": "capacity-general-recursive-classifier-v2",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output_layout", "output_report"}
        },
        "layers": rows,
        "aggregate": {
            "replica_slots": capacity.active_replicas,
            "reserved_replica_slots": capacity.reserved_replicas,
            "active_replica_slots": capacity.active_replicas,
            "empty_slots": capacity.empty_slots,
            "optimize_total_ms": sum(float(row["optimize"]["total_ms"]) for row in rows),
            "validation_total_ms": validation_total,
            "planner_total_ms": sum(float(row["planner_ms"]) for row in rows),
            "planner_mean_ms_per_layer": sum(float(row["planner_ms"]) for row in rows) / len(rows),
            "planner_wall_ms": wall_ms,
            "workers": min(args.workers, len(layer_ids)),
            "candidate_workers": args.candidate_workers,
            "exact_route_evaluations": sum(int(row["exact_route_evaluations"]) for row in rows),
            "uniform_statistics_builds": sum(int(row["uniform_statistics_builds"]) for row in rows),
            "comparison_validation_ms": comparison_ms,
            "validation_gain_ms": comparison_ms - validation_total,
            "validation_speedup": comparison_ms / validation_total,
            "e2e_eligible": _is_e2e_eligible(
                layer_start=args.layer_start,
                layers=args.layers,
                expected_total_layers=args.expected_total_layers,
                validation_total_ms=validation_total,
                comparison_validation_ms=comparison_ms,
            ),
        },
    }
    serialization_mode = "preloaded"
    serialization_reason = "inactive physical slots require exact static preload"
    if not capacity.empty_slots:
        try:
            payload = _replay_payload(
                layouts=layouts,
                owners=owners,
                luts=luts,
                args=args,
                algorithm="capacity-general-recursive-classifier-v2",
            )
            serialization_mode = "actions"
            serialization_reason = ""
        except RuntimeError as error:
            serialization_reason = f"legacy action normalization is not feasible: {error}"
            payload = _preloaded_replay_payload(
                layouts=layouts,
                owners=owners,
                luts=luts,
                args=args,
                algorithm="capacity-general-recursive-classifier-v2",
            )
    else:
        payload = _preloaded_replay_payload(
            layouts=layouts,
            owners=owners,
            luts=luts,
            args=args,
            algorithm="capacity-general-recursive-classifier-v2",
        )
    report["aggregate"]["serialization_mode"] = serialization_mode
    report["aggregate"]["serialization_reason"] = serialization_reason
    for path, value in ((args.output_layout, payload), (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"validation_total_ms={validation_total:.6f} "
        f"comparison_ms={comparison_ms:.6f} "
        f"speedup={comparison_ms / validation_total:.6f} "
        f"e2e_eligible={report['aggregate']['e2e_eligible']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
