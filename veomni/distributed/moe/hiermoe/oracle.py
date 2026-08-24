# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Idealized communication-volume heuristics for captured HierMoE routes."""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.distributed as dist

from .perf_model import HierMoEPerfModel
from .topology import Hierarchy


_SNAPSHOT_FORMAT = "veomni.hiermoe.route_snapshot"
_SNAPSHOT_VERSION = 1
_LOCAL_SNAPSHOT_FORMAT = "veomni.hiermoe.local_route"
_LOCAL_SNAPSHOT_VERSION = 1
_CAPTURE_PATH_TEMPLATE = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", "").strip()
_CAPTURE_CALLS: dict[tuple[int, str], int] = {}
_CAPTURED: set[tuple[int, str, int]] = set()
_CAPTURE_LAYER_ORDINALS: dict[int, int] = {}


@dataclass(frozen=True)
class RouteSnapshot:
    """Logical top-k routes for one layer invocation on every EP rank."""

    routes_by_rank: tuple[torch.Tensor, ...]
    num_experts: int
    hidden_size: int
    bytes_per_element: int
    hierarchy: Hierarchy
    logical_to_physical: torch.Tensor
    layer_key: str
    step: int
    call_index: int
    smooth_max_gamma: float = 10.0
    selected_dim: int | None = None

    @property
    def ep_size(self) -> int:
        return len(self.routes_by_rank)

    @property
    def base_experts_per_rank(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def full_replica_slots_per_rank(self) -> int:
        return self.num_experts - self.base_experts_per_rank

    @property
    def owner_ranks(self) -> torch.Tensor:
        return torch.div(self.logical_to_physical, self.base_experts_per_rank, rounding_mode="floor")

    @property
    def communication_dimension(self) -> int:
        if self.selected_dim is not None:
            return int(self.selected_dim)
        return min(2, self.hierarchy.selected_dim)

    def validate(self) -> None:
        if self.ep_size < 1:
            raise ValueError("A route snapshot must contain at least one EP rank.")
        if self.num_experts < 1 or self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be positive and divisible by ep_size={self.ep_size}."
            )
        if self.hierarchy.ep_size != self.ep_size:
            raise ValueError(
                f"Hierarchy EP size {self.hierarchy.ep_size} does not match snapshot EP size {self.ep_size}."
            )
        if not 1 <= self.communication_dimension <= self.hierarchy.selected_dim:
            raise ValueError(
                f"selected_dim={self.communication_dimension} must be in [1, {self.hierarchy.selected_dim}]."
            )
        if tuple(self.logical_to_physical.shape) != (self.num_experts,):
            raise ValueError("logical_to_physical must contain one physical slot per logical expert.")
        expected = torch.arange(self.num_experts, dtype=torch.long)
        actual = torch.sort(self.logical_to_physical.to(torch.long).cpu()).values
        if not torch.equal(actual, expected):
            raise ValueError("logical_to_physical must be a permutation of physical expert slots.")

        top_k: int | None = None
        for rank, routes in enumerate(self.routes_by_rank):
            if routes.ndim != 2:
                raise ValueError(f"Rank {rank} routes must be two-dimensional, got shape={tuple(routes.shape)}.")
            if top_k is None:
                top_k = int(routes.shape[1])
            elif int(routes.shape[1]) != top_k:
                raise ValueError("All EP ranks must use the same top-k width.")
            if routes.numel() == 0:
                continue
            minimum = int(routes.min().item())
            maximum = int(routes.max().item())
            if minimum < 0 or maximum >= self.num_experts:
                raise ValueError(
                    f"Rank {rank} route IDs must be in [0, {self.num_experts}), got [{minimum}, {maximum}]."
                )


@dataclass(frozen=True)
class CommunicationCost:
    total: float
    inter: float
    intra: float
    inter_peak_tokens: int
    intra_peak_tokens: int


@dataclass(frozen=True)
class CurvePoint:
    budget: int
    cost: float
    speedup: float
    remaining_fraction: float
    actions: tuple[str, ...]


@dataclass(frozen=True)
class SwapCurve:
    points: tuple[CurvePoint, ...]
    mappings: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ReplicaCurve:
    points: tuple[CurvePoint, ...]
    copies: tuple[torch.Tensor, ...]


def _normalize_routes(routes: torch.Tensor) -> torch.Tensor:
    routes = routes.detach().to(device="cpu", dtype=torch.long)
    if routes.ndim == 1:
        routes = routes.unsqueeze(-1)
    elif routes.ndim > 2:
        routes = routes.reshape(-1, routes.shape[-1])
    return routes.contiguous()


def save_route_snapshot(snapshot: RouteSnapshot, path: str | Path) -> Path:
    snapshot.validate()
    routes = tuple(_normalize_routes(item) for item in snapshot.routes_by_rank)
    lengths = torch.tensor([item.shape[0] for item in routes], dtype=torch.long)
    max_tokens = int(lengths.max().item())
    top_k = int(routes[0].shape[1])
    padded = torch.full((snapshot.ep_size, max_tokens, top_k), -1, dtype=torch.int32)
    for rank, item in enumerate(routes):
        padded[rank, : item.shape[0]] = item.to(torch.int32)

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _SNAPSHOT_FORMAT,
            "version": _SNAPSHOT_VERSION,
            "routes": padded,
            "route_lengths": lengths,
            "num_experts": snapshot.num_experts,
            "hidden_size": snapshot.hidden_size,
            "bytes_per_element": snapshot.bytes_per_element,
            "ep_size": snapshot.ep_size,
            "hierarchy_group_sizes": list(snapshot.hierarchy.group_sizes),
            "hierarchy_source": snapshot.hierarchy.source,
            "logical_to_physical": snapshot.logical_to_physical.to(device="cpu", dtype=torch.long),
            "layer_key": snapshot.layer_key,
            "step": snapshot.step,
            "call_index": snapshot.call_index,
            "smooth_max_gamma": snapshot.smooth_max_gamma,
            "selected_dim": snapshot.communication_dimension,
        },
        output,
    )
    return output


def load_route_snapshot(path: str | Path) -> RouteSnapshot:
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(source, map_location="cpu")
    if payload.get("format") != _SNAPSHOT_FORMAT or int(payload.get("version", -1)) != _SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported HierMoE route snapshot: {source}")

    padded = payload["routes"].to(torch.long)
    lengths = payload["route_lengths"].to(torch.long)
    routes = tuple(padded[rank, : int(lengths[rank].item())].contiguous() for rank in range(padded.shape[0]))
    snapshot = RouteSnapshot(
        routes_by_rank=routes,
        num_experts=int(payload["num_experts"]),
        hidden_size=int(payload["hidden_size"]),
        bytes_per_element=int(payload["bytes_per_element"]),
        hierarchy=Hierarchy(
            ep_size=int(payload["ep_size"]),
            group_sizes=tuple(int(value) for value in payload["hierarchy_group_sizes"]),
            source=str(payload.get("hierarchy_source", "snapshot")),
        ),
        logical_to_physical=payload["logical_to_physical"].to(torch.long),
        layer_key=str(payload["layer_key"]),
        step=int(payload["step"]),
        call_index=int(payload["call_index"]),
        smooth_max_gamma=float(payload.get("smooth_max_gamma", 10.0)),
        selected_dim=int(payload.get("selected_dim", min(2, len(payload["hierarchy_group_sizes"])) or 1)),
    )
    snapshot.validate()
    return snapshot


def _layer_matches(layer_key: str, requested: str) -> bool:
    requested = requested.strip()
    if not requested:
        return True
    if requested.isdigit():
        match = re.search(r"(?:layers?|layer)\.(\d+)(?:\.|$)", layer_key)
        return match is not None and int(match.group(1)) == int(requested)
    return requested == layer_key or requested in layer_key


def route_capture_enabled() -> bool:
    """Return whether route capture was enabled before process startup."""

    return bool(_CAPTURE_PATH_TEMPLATE)


def route_capture_mode() -> str:
    """Return the configured route capture mode."""

    mode = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE", "global").strip().lower()
    if mode not in {"global", "local"}:
        raise ValueError(f"VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE must be either 'global' or 'local', got {mode!r}.")
    return mode


def _layer_index(layer_key: str) -> int:
    matches = re.findall(r"(?:layers?|layer)\.(\d+)(?:\.|$)", layer_key)
    return int(matches[-1]) if matches else -1


def _capture_layer_key(step: int, layer_key: str | None) -> str:
    if layer_key is not None:
        return layer_key

    layer_ordinal = _CAPTURE_LAYER_ORDINALS.get(int(step), 0)
    _CAPTURE_LAYER_ORDINALS[int(step)] = layer_ordinal + 1
    raw_num_layers = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS", "").strip()
    if raw_num_layers:
        try:
            num_layers = int(raw_num_layers)
        except ValueError as exc:
            raise ValueError("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS must be a positive integer.") from exc
        if num_layers < 1:
            raise ValueError("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS must be a positive integer.")
        layer_ordinal %= num_layers
    return f"model.layers.{layer_ordinal}.mlp.experts"


def _capture_output_path(
    raw_path: str,
    *,
    step: int,
    layer_key: str,
    call_index: int,
    global_rank: int,
    ep_rank: int,
) -> Path:
    return Path(
        raw_path.format(
            step=int(step),
            layer=re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_key),
            layer_index=_layer_index(layer_key),
            call=int(call_index),
            rank=int(global_rank),
            ep_rank=int(ep_rank),
        )
    )


def _save_local_route_snapshot(
    *,
    routes: torch.Tensor,
    path: Path,
    global_rank: int,
    ep_rank: int,
    ep_size: int,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    logical_to_physical: torch.Tensor,
    slot_to_logical: torch.Tensor | None,
    layer_key: str,
    step: int,
    call_index: int,
    smooth_max_gamma: float,
    selected_dim: int,
) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _LOCAL_SNAPSHOT_FORMAT,
            "version": _LOCAL_SNAPSHOT_VERSION,
            "routes": _normalize_routes(routes).to(torch.int32),
            "global_rank": int(global_rank),
            "ep_rank": int(ep_rank),
            "ep_size": int(ep_size),
            "num_experts": int(num_experts),
            "hidden_size": int(hidden_size),
            "bytes_per_element": int(bytes_per_element),
            "hierarchy_group_sizes": list(hierarchy.group_sizes),
            "hierarchy_source": hierarchy.source,
            "logical_to_physical": logical_to_physical.detach().to(device="cpu", dtype=torch.long),
            "slot_to_logical": (
                None if slot_to_logical is None else slot_to_logical.detach().to(device="cpu", dtype=torch.long)
            ),
            "layer": _layer_index(layer_key),
            "layer_key": layer_key,
            "step": int(step),
            "call_index": int(call_index),
            "smooth_max_gamma": float(smooth_max_gamma),
            "selected_dim": int(selected_dim),
        },
        output,
    )
    return output


def maybe_capture_route_snapshot(
    *,
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    ep_group: dist.ProcessGroup | None,
    hierarchy: Hierarchy,
    layer_key: str | None,
    step: int,
    logical_to_physical: torch.Tensor | None = None,
    slot_to_logical: torch.Tensor | None = None,
    smooth_max_gamma: float = 10.0,
    selected_dim: int = 1,
) -> Path | None:
    """Capture one route invocation selected by debug-only environment variables."""

    raw_path = _CAPTURE_PATH_TEMPLATE
    if not raw_path:
        return None
    target_step = int(os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP", "-1"))
    target_layer = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_LAYER", "")
    target_call = int(os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL", "0"))
    layer_key = _capture_layer_key(step, layer_key)
    if (target_step >= 0 and int(step) != target_step) or not _layer_matches(layer_key, target_layer):
        return None

    call_key = (int(step), layer_key)
    call_index = _CAPTURE_CALLS.get(call_key, 0)
    _CAPTURE_CALLS[call_key] = call_index + 1
    capture_key = (int(step), layer_key, call_index)
    if call_index != target_call or capture_key in _CAPTURED:
        return None
    _CAPTURED.add(capture_key)

    local_routes = selected_experts.detach().to(dtype=torch.int32)
    if local_routes.ndim == 1:
        local_routes = local_routes.unsqueeze(-1)
    elif local_routes.ndim > 2:
        local_routes = local_routes.reshape(-1, local_routes.shape[-1])
    local_routes = local_routes.contiguous()

    initialized = dist.is_initialized()
    ep_size = dist.get_world_size(ep_group) if ep_group is not None and initialized else 1
    global_rank = dist.get_rank() if initialized else 0
    ep_rank = dist.get_rank(ep_group) if ep_group is not None and initialized else 0
    mapping = (
        logical_to_physical.detach().to(device="cpu", dtype=torch.long)
        if logical_to_physical is not None
        else torch.arange(num_experts, dtype=torch.long)
    )
    mode = route_capture_mode()
    if mode == "local":
        if "{rank" not in raw_path and "{ep_rank" not in raw_path:
            raise ValueError(
                "Local HierMoE route capture path must contain a {rank} or {ep_rank} field "
                "so ranks on the same host do not overwrite each other."
            )
        output = _capture_output_path(
            raw_path,
            step=step,
            layer_key=layer_key,
            call_index=call_index,
            global_rank=global_rank,
            ep_rank=ep_rank,
        )
        return _save_local_route_snapshot(
            routes=local_routes,
            path=output,
            global_rank=global_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_experts=num_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            hierarchy=hierarchy,
            logical_to_physical=mapping,
            slot_to_logical=slot_to_logical,
            layer_key=layer_key,
            step=step,
            call_index=call_index,
            smooth_max_gamma=smooth_max_gamma,
            selected_dim=selected_dim,
        )

    if slot_to_logical is not None:
        raise RuntimeError(
            "Global HierMoE route capture does not support redundant expert slots; "
            "use VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE=local."
        )
    if ep_size == 1:
        routes_by_rank = (local_routes.cpu(),)
    else:
        shape = torch.tensor(local_routes.shape, dtype=torch.long, device=local_routes.device)
        gathered_shapes = [torch.empty_like(shape) for _ in range(ep_size)]
        dist.all_gather(gathered_shapes, shape, group=ep_group)
        shapes = torch.stack(gathered_shapes).cpu()
        if not bool((shapes[:, 1] == shapes[0, 1]).all().item()):
            raise RuntimeError("All EP ranks must use the same top-k width for route capture.")
        max_tokens = int(shapes[:, 0].max().item())
        padded = torch.full(
            (max_tokens, int(shapes[0, 1].item())),
            -1,
            dtype=torch.int32,
            device=local_routes.device,
        )
        padded[: local_routes.shape[0]] = local_routes
        gathered = torch.empty(
            (ep_size * max_tokens, padded.shape[1]),
            dtype=padded.dtype,
            device=padded.device,
        )
        dist.all_gather_into_tensor(gathered, padded, group=ep_group)
        gathered = gathered.view(ep_size, max_tokens, padded.shape[1]).cpu()
        routes_by_rank = tuple(gathered[rank, : int(shapes[rank, 0].item())].contiguous() for rank in range(ep_size))

    if global_rank != 0:
        return None
    output = _capture_output_path(
        raw_path,
        step=step,
        layer_key=layer_key,
        call_index=call_index,
        global_rank=global_rank,
        ep_rank=ep_rank,
    )
    return save_route_snapshot(
        RouteSnapshot(
            routes_by_rank=tuple(routes.to(torch.long) for routes in routes_by_rank),
            num_experts=int(num_experts),
            hidden_size=int(hidden_size),
            bytes_per_element=int(bytes_per_element),
            hierarchy=hierarchy,
            logical_to_physical=mapping,
            layer_key=layer_key,
            step=int(step),
            call_index=call_index,
            smooth_max_gamma=float(smooth_max_gamma),
            selected_dim=int(selected_dim),
        ),
        output,
    )


def _speedup(baseline: float, current: float) -> float:
    if baseline <= 0.0:
        return 1.0 if current <= 0.0 else 0.0
    return float("inf") if current <= 0.0 else baseline / current


def _remaining_fraction(baseline: float, current: float) -> float:
    if baseline <= 0.0:
        return 1.0 if current <= 0.0 else float("inf")
    return current / baseline


def _curve_point(budget: int, cost: float, baseline: float, actions: Iterable[str]) -> CurvePoint:
    return CurvePoint(
        budget=int(budget),
        cost=float(cost),
        speedup=_speedup(baseline, cost),
        remaining_fraction=_remaining_fraction(baseline, cost),
        actions=tuple(actions),
    )


def _copy_matrix(owner_ranks: torch.Tensor, ep_size: int) -> torch.Tensor:
    copies = torch.zeros((owner_ranks.numel(), ep_size), dtype=torch.bool)
    copies[torch.arange(owner_ranks.numel()), owner_ranks.to(torch.long)] = True
    return copies


def _rank_distance(lhs: int, rhs: int, intra_size: int) -> int:
    if lhs == rhs:
        return 0
    return 1 if lhs // intra_size == rhs // intra_size else 2


def _assign_targets(snapshot: RouteSnapshot, copies: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Assign each routed expert to an independently nearest copy.

    This is deterministic and inexpensive, but it does not jointly optimize a
    token's complete top-k destination set for rank-level deduplication.
    """
    intra_size = (
        int(snapshot.hierarchy.group_sizes[0]) if len(snapshot.hierarchy.group_sizes) >= 2 else snapshot.ep_size
    )
    nearest = torch.empty((snapshot.ep_size, snapshot.num_experts), dtype=torch.long)
    for source_rank in range(snapshot.ep_size):
        for logical in range(snapshot.num_experts):
            candidates = torch.nonzero(copies[logical], as_tuple=False).flatten().tolist()
            if not candidates:
                raise ValueError(f"Logical expert {logical} has no physical copy.")
            nearest[source_rank, logical] = min(
                (int(candidate) for candidate in candidates),
                key=lambda candidate: (_rank_distance(source_rank, candidate, intra_size), candidate),
            )

    targets_by_rank: list[torch.Tensor] = []
    for source_rank, routes in enumerate(snapshot.routes_by_rank):
        targets_by_rank.append(nearest[source_rank].index_select(0, routes.reshape(-1)).view_as(routes))
    return tuple(targets_by_rank)


def communication_cost_from_targets(
    snapshot: RouteSnapshot,
    targets_by_rank: Sequence[torch.Tensor],
    perf_model: HierMoEPerfModel | None = None,
) -> CommunicationCost:
    """Estimate topology-only communication at the runtime dimension.

    This heuristic independently selects a nearest copy for every routed
    expert, assumes zero cost when all traffic is local, and uses a
    topology-level peak approximation. It is neither a mathematical lower
    bound nor an exact prediction of a runtime that still launches collectives.
    """

    model = perf_model or HierMoEPerfModel.default()
    ep_size = snapshot.ep_size
    payload_bytes = snapshot.hidden_size * snapshot.bytes_per_element
    selected_dim = snapshot.communication_dimension
    if selected_dim > 2:
        raise NotImplementedError(
            "The route-analysis heuristic currently supports flat and 2D hierarchies only; "
            f"got group_sizes={snapshot.hierarchy.group_sizes}."
        )
    if selected_dim == 1:
        rank_matrix = torch.zeros((ep_size, ep_size), dtype=torch.long)
        for source_rank, targets in enumerate(targets_by_rank):
            hits = torch.zeros((targets.shape[0], ep_size), dtype=torch.bool)
            hits.scatter_(1, targets.to(torch.long), True)
            rank_matrix[source_rank] = hits.sum(dim=0)
            rank_matrix[source_rank, source_rank] = 0
        peak = int(rank_matrix.max().item()) if rank_matrix.numel() else 0
        stage = 0.0 if peak == 0 else model.a2a.alpha + ep_size * peak * payload_bytes * model.a2a.beta
        return CommunicationCost(2.0 * stage, 2.0 * stage, 0.0, peak, 0)

    intra_size = int(snapshot.hierarchy.group_sizes[0])
    if intra_size <= 0 or ep_size % intra_size != 0:
        raise ValueError(f"Invalid 2D hierarchy intra_size={intra_size} for ep_size={ep_size}.")
    num_nodes = ep_size // intra_size
    inter_matrix = torch.zeros((ep_size, num_nodes), dtype=torch.long)
    intra_matrix = torch.zeros((ep_size, ep_size), dtype=torch.long)
    for source_rank, targets in enumerate(targets_by_rank):
        target_nodes = torch.div(targets, intra_size, rounding_mode="floor")
        node_hits = torch.zeros((targets.shape[0], num_nodes), dtype=torch.bool)
        node_hits.scatter_(1, target_nodes.to(torch.long), True)
        inter_matrix[source_rank] = node_hits.sum(dim=0)
        inter_matrix[source_rank, source_rank // intra_size] = 0

        rank_hits = torch.zeros((targets.shape[0], ep_size), dtype=torch.bool)
        rank_hits.scatter_(1, targets.to(torch.long), True)
        rank_counts = rank_hits.sum(dim=0)
        source_local_rank = source_rank % intra_size
        for target_rank in range(ep_size):
            ingress_rank = (target_rank // intra_size) * intra_size + source_local_rank
            if ingress_rank != target_rank:
                intra_matrix[ingress_rank, target_rank] += rank_counts[target_rank]

    inter_peak = int(inter_matrix.max().item()) if inter_matrix.numel() else 0
    intra_peak = int(intra_matrix.max().item()) if intra_matrix.numel() else 0
    inter_link = model.inter[0] if model.inter else model.a2a
    inter_stage = (
        0.0 if inter_peak == 0 else inter_link.alpha + num_nodes * inter_peak * payload_bytes * inter_link.beta
    )
    intra_stage = (
        0.0 if intra_peak == 0 else model.intra.alpha + intra_size * intra_peak * payload_bytes * model.intra.beta
    )
    return CommunicationCost(
        total=2.0 * (inter_stage + intra_stage),
        inter=2.0 * inter_stage,
        intra=2.0 * intra_stage,
        inter_peak_tokens=inter_peak,
        intra_peak_tokens=intra_peak,
    )


def idealized_communication_cost(
    snapshot: RouteSnapshot,
    copies: torch.Tensor,
    perf_model: HierMoEPerfModel | None = None,
) -> CommunicationCost:
    return communication_cost_from_targets(snapshot, _assign_targets(snapshot, copies), perf_model)


def _selector_cost(
    snapshot: RouteSnapshot,
    logical_to_physical: torch.Tensor,
    perf_model: HierMoEPerfModel,
) -> float:
    routes = torch.cat(snapshot.routes_by_rank, dim=0)
    physical = logical_to_physical.index_select(0, routes.reshape(-1)).view_as(routes)
    per_dim = [
        perf_model.estimate_hierarchical_time(
            physical,
            snapshot.num_experts,
            snapshot.hidden_size,
            snapshot.bytes_per_element,
            snapshot.hierarchy,
            dim,
        )
        for dim in range(1, snapshot.hierarchy.selected_dim + 1)
    ]
    values = torch.tensor(per_dim, dtype=torch.float64)
    gamma = snapshot.smooth_max_gamma
    return float(torch.logsumexp(values * gamma, dim=0).item() / gamma)


def _online_swap_candidates(snapshot: RouteSnapshot) -> list[tuple[int, int]]:
    routes = torch.cat(snapshot.routes_by_rank, dim=0)
    counts = torch.bincount(routes.reshape(-1), minlength=snapshot.num_experts)
    if snapshot.num_experts <= 64:
        indices = torch.triu_indices(snapshot.num_experts, snapshot.num_experts, offset=1).t()
        return [(int(lhs), int(rhs)) for lhs, rhs in indices.tolist()]
    hot = torch.topk(counts, k=min(12, snapshot.num_experts), largest=True).indices
    cold = torch.topk(counts, k=min(12, snapshot.num_experts), largest=False).indices
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for lhs in hot.tolist():
        for rhs in cold.tolist():
            if lhs == rhs:
                continue
            pair = (min(int(lhs), int(rhs)), max(int(lhs), int(rhs)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= 96:
                return pairs
    return pairs


def _swap_mapping(mapping: torch.Tensor, pair: tuple[int, int]) -> torch.Tensor:
    swapped = mapping.clone()
    lhs, rhs = pair
    swapped[lhs], swapped[rhs] = swapped[rhs].clone(), swapped[lhs].clone()
    return swapped


def _mapping_cost(snapshot: RouteSnapshot, mapping: torch.Tensor, perf_model: HierMoEPerfModel) -> float:
    owner_ranks = torch.div(mapping, snapshot.base_experts_per_rank, rounding_mode="floor")
    copies = _copy_matrix(owner_ranks, snapshot.ep_size)
    return idealized_communication_cost(snapshot, copies, perf_model).total


def route_count_greedy_swap_curve(
    snapshot: RouteSnapshot,
    max_pairs: int,
    perf_model: HierMoEPerfModel | None = None,
) -> SwapCurve:
    """Apply a source-agnostic route-count heuristic with disjoint swap pairs.

    This intentionally concatenates routes from all source ranks when scoring
    proposals. It is useful as a cheap online-style heuristic, but it is not an
    exact replay of the runtime planner or its dedup-aware communication model.
    """

    snapshot.validate()
    model = perf_model or HierMoEPerfModel.default()
    initial = snapshot.logical_to_physical.clone()
    baseline = _mapping_cost(snapshot, initial, model)
    selector_baseline = _selector_cost(snapshot, initial, model)
    scored: list[tuple[float, tuple[int, int]]] = []
    for pair in _online_swap_candidates(snapshot):
        lhs_rank = int(initial[pair[0]].item()) // snapshot.base_experts_per_rank
        rhs_rank = int(initial[pair[1]].item()) // snapshot.base_experts_per_rank
        if lhs_rank == rhs_rank:
            continue
        score = _selector_cost(snapshot, _swap_mapping(initial, pair), model)
        if score < selector_baseline:
            scored.append((score, pair))
    scored.sort(key=lambda item: (item[0], item[1]))

    points = [_curve_point(0, baseline, baseline, ())]
    mappings = [initial.clone()]
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    score_idx = 0
    for budget in range(1, max(0, int(max_pairs)) + 1):
        while score_idx < len(scored):
            _, pair = scored[score_idx]
            score_idx += 1
            if pair[0] in used or pair[1] in used:
                continue
            chosen.append(pair)
            used.update(pair)
            break
        mapping = initial.clone()
        for pair in chosen:
            mapping = _swap_mapping(mapping, pair)
        cost = _mapping_cost(snapshot, mapping, model)
        points.append(_curve_point(budget, cost, baseline, (f"{lhs}<->{rhs}" for lhs, rhs in chosen)))
        mappings.append(mapping)
    return SwapCurve(tuple(points), tuple(mappings))


def _best_found_swap_candidates(
    snapshot: RouteSnapshot,
    mapping: torch.Tensor,
    used: set[int],
    candidate_limit: int | None,
    perf_model: HierMoEPerfModel,
) -> list[tuple[int, int]]:
    owner = torch.div(mapping, snapshot.base_experts_per_rank, rounding_mode="floor")
    demand = _route_demand(snapshot)
    intra_size = (
        int(snapshot.hierarchy.group_sizes[0]) if len(snapshot.hierarchy.group_sizes) >= 2 else snapshot.ep_size
    )
    expert_rank_cost = torch.zeros((snapshot.num_experts, snapshot.ep_size), dtype=torch.float64)
    for expert in range(snapshot.num_experts):
        for target_rank in range(snapshot.ep_size):
            expert_rank_cost[expert, target_rank] = sum(
                demand[source_rank, expert] * _weighted_rank_distance(source_rank, target_rank, intra_size, perf_model)
                for source_rank in range(snapshot.ep_size)
            )

    scored: list[tuple[float, tuple[int, int]]] = []
    for lhs in range(snapshot.num_experts):
        if lhs in used:
            continue
        for rhs in range(lhs + 1, snapshot.num_experts):
            if rhs in used or int(owner[lhs].item()) == int(owner[rhs].item()):
                continue
            lhs_owner = int(owner[lhs].item())
            rhs_owner = int(owner[rhs].item())
            delta = (
                expert_rank_cost[lhs, rhs_owner]
                + expert_rank_cost[rhs, lhs_owner]
                - expert_rank_cost[lhs, lhs_owner]
                - expert_rank_cost[rhs, rhs_owner]
            )
            scored.append((float(delta.item()), (lhs, rhs)))
    scored.sort(key=lambda item: (item[0], item[1]))
    if candidate_limit is not None and candidate_limit > 0:
        scored = scored[:candidate_limit]
    return [pair for _, pair in scored]


def best_found_swap_curve(
    snapshot: RouteSnapshot,
    max_pairs: int,
    perf_model: HierMoEPerfModel | None = None,
    candidate_limit: int | None = None,
) -> SwapCurve:
    """Greedily search a wide pair set under the idealized heuristic."""

    snapshot.validate()
    model = perf_model or HierMoEPerfModel.default()
    mapping = snapshot.logical_to_physical.clone()
    baseline = _mapping_cost(snapshot, mapping, model)
    current_cost = baseline
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    points = [_curve_point(0, baseline, baseline, ())]
    mappings = [mapping.clone()]
    for budget in range(1, max(0, int(max_pairs)) + 1):
        best_cost = current_cost
        best_pair: tuple[int, int] | None = None
        best_mapping: torch.Tensor | None = None
        for pair in _best_found_swap_candidates(snapshot, mapping, used, candidate_limit, model):
            candidate = _swap_mapping(mapping, pair)
            cost = _mapping_cost(snapshot, candidate, model)
            if cost < best_cost - 1.0e-12:
                best_cost = cost
                best_pair = pair
                best_mapping = candidate
        if best_pair is not None and best_mapping is not None:
            mapping = best_mapping
            current_cost = best_cost
            chosen.append(best_pair)
            used.update(best_pair)
        points.append(_curve_point(budget, current_cost, baseline, (f"{lhs}<->{rhs}" for lhs, rhs in chosen)))
        mappings.append(mapping.clone())
    return SwapCurve(tuple(points), tuple(mappings))


def _route_demand(snapshot: RouteSnapshot) -> torch.Tensor:
    demand = torch.zeros((snapshot.ep_size, snapshot.num_experts), dtype=torch.float64)
    for rank, routes in enumerate(snapshot.routes_by_rank):
        demand[rank] = torch.bincount(routes.reshape(-1), minlength=snapshot.num_experts).to(torch.float64)
    return demand


def _weighted_rank_distance(
    source_rank: int,
    target_rank: int,
    intra_size: int,
    perf_model: HierMoEPerfModel,
) -> float:
    if source_rank == target_rank:
        return 0.0
    cost = 0.0
    if source_rank // intra_size != target_rank // intra_size:
        link = perf_model.inter[0] if perf_model.inter else perf_model.a2a
        cost += float(link.beta)
    if source_rank % intra_size != target_rank % intra_size:
        cost += float(perf_model.intra.beta)
    return cost


def _nearest_copy_distance(copies: torch.Tensor, rank: int, expert: int, intra_size: int) -> int:
    copy_ranks = torch.nonzero(copies[expert], as_tuple=False).flatten().tolist()
    return min(_rank_distance(rank, int(copy_rank), intra_size) for copy_rank in copy_ranks)


def _online_replica_layouts(
    snapshot: RouteSnapshot,
    owner_ranks: torch.Tensor,
    max_slots_per_rank: int,
) -> list[torch.Tensor]:
    copies = _copy_matrix(owner_ranks, snapshot.ep_size)
    demand = _route_demand(snapshot)
    intra_size = (
        int(snapshot.hierarchy.group_sizes[0]) if len(snapshot.hierarchy.group_sizes) >= 2 else snapshot.ep_size
    )
    used = torch.zeros((snapshot.ep_size,), dtype=torch.long)
    layouts = [copies.clone()]
    for budget in range(1, max_slots_per_rank + 1):
        while bool((used < budget).any().item()):
            best_score = 0.0
            best: tuple[int, int] | None = None
            for rank in range(snapshot.ep_size):
                if int(used[rank].item()) >= budget:
                    continue
                for expert in range(snapshot.num_experts):
                    if bool(copies[expert, rank].item()) or demand[rank, expert] <= 0:
                        continue
                    distance = _nearest_copy_distance(copies, rank, expert, intra_size)
                    score = float(demand[rank, expert].item()) * float(distance)
                    candidate = (rank, expert)
                    if score > best_score or (score == best_score and best is not None and candidate < best):
                        best_score = score
                        best = candidate
            if best is None:
                break
            rank, expert = best
            copies[expert, rank] = True
            used[rank] += 1
        layouts.append(copies.clone())
    return layouts


def _best_found_replica_layouts(
    snapshot: RouteSnapshot,
    owner_ranks: torch.Tensor,
    max_slots_per_rank: int,
    perf_model: HierMoEPerfModel,
) -> list[torch.Tensor]:
    demand = _route_demand(snapshot)
    intra_size = (
        int(snapshot.hierarchy.group_sizes[0]) if len(snapshot.hierarchy.group_sizes) >= 2 else snapshot.ep_size
    )
    rankings: list[list[int]] = []
    for rank in range(snapshot.ep_size):
        scored: list[tuple[float, int]] = []
        for expert in range(snapshot.num_experts):
            owner_rank = int(owner_ranks[expert].item())
            if owner_rank == rank:
                continue
            gain = float(demand[rank, expert].item()) * _weighted_rank_distance(
                rank,
                owner_rank,
                intra_size,
                perf_model,
            )
            scored.append((gain, expert))
        scored.sort(key=lambda item: (-item[0], item[1]))
        rankings.append([expert for _, expert in scored])

    layouts: list[torch.Tensor] = []
    for budget in range(max_slots_per_rank + 1):
        copies = _copy_matrix(owner_ranks, snapshot.ep_size)
        for rank in range(snapshot.ep_size):
            for expert in rankings[rank][:budget]:
                copies[expert, rank] = True
        layouts.append(copies)
    return layouts


def _replica_curve_from_layouts(
    snapshot: RouteSnapshot,
    layouts: Sequence[torch.Tensor],
    perf_model: HierMoEPerfModel,
    budgets: Sequence[int] | None = None,
) -> ReplicaCurve:
    selected_budgets = tuple(range(len(layouts))) if budgets is None else tuple(int(value) for value in budgets)
    selected_layouts = [layouts[budget] for budget in selected_budgets]
    costs = [idealized_communication_cost(snapshot, copies, perf_model).total for copies in selected_layouts]
    baseline = costs[0]
    owner = selected_layouts[0]
    points: list[CurvePoint] = []
    for budget, cost, copies in zip(selected_budgets, costs, selected_layouts, strict=True):
        added = torch.nonzero(copies & ~owner, as_tuple=False)
        actions = (f"expert{int(expert)}@rank{int(rank)}" for expert, rank in added.tolist())
        points.append(_curve_point(budget, cost, baseline, actions))
    return ReplicaCurve(tuple(points), tuple(layout.clone() for layout in selected_layouts))


def route_count_greedy_replica_curve(
    snapshot: RouteSnapshot,
    max_slots_per_rank: int,
    perf_model: HierMoEPerfModel | None = None,
    logical_to_physical: torch.Tensor | None = None,
    budgets: Sequence[int] | None = None,
) -> ReplicaCurve:
    """Generalize the source-local route-count times nearest-distance heuristic."""

    snapshot.validate()
    model = perf_model or HierMoEPerfModel.default()
    mapping = snapshot.logical_to_physical if logical_to_physical is None else logical_to_physical
    owner = torch.div(mapping, snapshot.base_experts_per_rank, rounding_mode="floor")
    layouts = _online_replica_layouts(snapshot, owner, max(0, int(max_slots_per_rank)))
    return _replica_curve_from_layouts(snapshot, layouts, model, budgets)


def best_found_replica_curve(
    snapshot: RouteSnapshot,
    max_slots_per_rank: int,
    perf_model: HierMoEPerfModel | None = None,
    logical_to_physical: torch.Tensor | None = None,
    budgets: Sequence[int] | None = None,
) -> ReplicaCurve:
    """Rank local replicas by source demand and calibrated hierarchy distance.

    Copy assignment remains the independent nearest-copy heuristic used by
    :func:`communication_cost_from_targets`.
    """

    snapshot.validate()
    model = perf_model or HierMoEPerfModel.default()
    mapping = snapshot.logical_to_physical if logical_to_physical is None else logical_to_physical
    owner = torch.div(mapping, snapshot.base_experts_per_rank, rounding_mode="floor")
    layouts = _best_found_replica_layouts(snapshot, owner, max(0, int(max_slots_per_rank)), model)
    return _replica_curve_from_layouts(snapshot, layouts, model, budgets)


def online_greedy_swap_curve(
    snapshot: RouteSnapshot,
    max_pairs: int,
    perf_model: HierMoEPerfModel | None = None,
) -> SwapCurve:
    warnings.warn(
        "online_greedy_swap_curve is a route-count heuristic, not an exact runtime replay; "
        "use route_count_greedy_swap_curve instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return route_count_greedy_swap_curve(snapshot, max_pairs, perf_model)


def online_greedy_replica_curve(
    snapshot: RouteSnapshot,
    max_slots_per_rank: int,
    perf_model: HierMoEPerfModel | None = None,
    logical_to_physical: torch.Tensor | None = None,
    budgets: Sequence[int] | None = None,
) -> ReplicaCurve:
    warnings.warn(
        "online_greedy_replica_curve is a route-count heuristic; use route_count_greedy_replica_curve instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return route_count_greedy_replica_curve(
        snapshot,
        max_slots_per_rank,
        perf_model,
        logical_to_physical,
        budgets,
    )


def oracle_swap_curve(
    snapshot: RouteSnapshot,
    max_pairs: int,
    perf_model: HierMoEPerfModel | None = None,
    candidate_limit: int | None = None,
) -> SwapCurve:
    warnings.warn(
        "oracle_swap_curve is a best-found heuristic, not an oracle or lower bound; "
        "use best_found_swap_curve instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return best_found_swap_curve(snapshot, max_pairs, perf_model, candidate_limit)


def oracle_replica_curve(
    snapshot: RouteSnapshot,
    max_slots_per_rank: int,
    perf_model: HierMoEPerfModel | None = None,
    logical_to_physical: torch.Tensor | None = None,
    budgets: Sequence[int] | None = None,
) -> ReplicaCurve:
    warnings.warn(
        "oracle_replica_curve is a best-found heuristic, not an oracle or lower bound; "
        "use best_found_replica_curve instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return best_found_replica_curve(
        snapshot,
        max_slots_per_rank,
        perf_model,
        logical_to_physical,
        budgets,
    )


def sample_replica_budgets(full_budget: int, practical_max: int = 16) -> tuple[int, ...]:
    if full_budget <= 0:
        return (0,)
    values = set(range(0, min(full_budget, practical_max) + 1))
    value = max(practical_max + 1, 2)
    while value < full_budget:
        values.add(value)
        value *= 2
    values.add(full_budget)
    return tuple(sorted(values))


def subset_replica_curve(curve: ReplicaCurve, budgets: Sequence[int]) -> ReplicaCurve:
    by_budget = {point.budget: (point, copies) for point, copies in zip(curve.points, curve.copies, strict=True)}
    return ReplicaCurve(
        points=tuple(by_budget[budget][0] for budget in budgets),
        copies=tuple(by_budget[budget][1] for budget in budgets),
    )


def finite_plot_speedup(speedup: float, finite_cap: float) -> float:
    return finite_cap if math.isinf(speedup) else speedup
