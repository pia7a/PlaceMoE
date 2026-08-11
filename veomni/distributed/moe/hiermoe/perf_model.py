# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from ....utils.device import get_device_type, get_torch_device, synchronize
from .routing import duplicate_free_counts_by_expert_group
from .topology import Hierarchy


@dataclass(frozen=True)
class LinkCost:
    alpha: float
    beta: float

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "LinkCost":
        return cls(alpha=float(payload["alpha"]), beta=float(payload["beta"]))

    def to_payload(self) -> dict[str, float]:
        return {"alpha": float(self.alpha), "beta": float(self.beta)}


@dataclass(frozen=True)
class PeerTransferCost:
    intra: LinkCost
    inter: LinkCost

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "PeerTransferCost":
        return cls(
            intra=LinkCost.from_payload(dict(payload["intra"])),
            inter=LinkCost.from_payload(dict(payload["inter"])),
        )

    def to_payload(self) -> dict[str, dict[str, float]]:
        return {"intra": self.intra.to_payload(), "inter": self.inter.to_payload()}


@dataclass(frozen=True)
class GradientSyncCost:
    gather: PeerTransferCost
    scatter: PeerTransferCost

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "GradientSyncCost":
        return cls(
            gather=PeerTransferCost.from_payload(dict(payload["gather"])),
            scatter=PeerTransferCost.from_payload(dict(payload["scatter"])),
        )

    def to_payload(self) -> dict[str, dict[str, dict[str, float]]]:
        return {"gather": self.gather.to_payload(), "scatter": self.scatter.to_payload()}


def fit_link_cost(samples: Sequence[tuple[int, float]]) -> LinkCost:
    """Fit a non-negative alpha + beta * bytes model."""

    rows = [(float(size), float(elapsed)) for size, elapsed in samples if size >= 0 and elapsed >= 0.0]
    if len(rows) < 2:
        raise ValueError("At least two valid timing samples are required to fit a link cost.")
    mean_x = sum(row[0] for row in rows) / len(rows)
    mean_y = sum(row[1] for row in rows) / len(rows)
    denominator = sum((row[0] - mean_x) ** 2 for row in rows)
    if denominator <= 0.0:
        raise ValueError("Link-cost timing samples must contain at least two distinct payload sizes.")
    beta = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in rows) / denominator)
    alpha = max(0.0, mean_y - beta * mean_x)
    return LinkCost(alpha=alpha, beta=beta)


def _timed_accelerator_call(function: Callable[[], None]) -> float:
    namespace = get_torch_device()
    event_ctor = getattr(namespace, "Event", None)
    if event_ctor is None:
        started = time.perf_counter()
        function()
        return (time.perf_counter() - started) * 1000.0
    start = event_ctor(enable_timing=True)
    end = event_ctor(enable_timing=True)
    start.record()
    function()
    end.record()
    synchronize()
    return float(start.elapsed_time(end))


def _cross_rank_max(value: float, group: dist.ProcessGroup | None, device: torch.device) -> float:
    # HCCL does not support float64 reductions consistently across Ascend
    # software releases.  Millisecond probe values do not need double
    # precision, so keep the collective on the accelerator-safe float32 path.
    result = torch.tensor([value], dtype=torch.float32, device=device)
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=group)
    return float(result.item())


def _median_probe(
    operation: Callable[[], None],
    *,
    group: dist.ProcessGroup | None,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(max(0, warmup)):
        operation()
    synchronize()
    values = [_cross_rank_max(_timed_accelerator_call(operation), group, device) for _ in range(max(1, repeats))]
    return float(statistics.median(values))


def _fit_all_to_all_link(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    payload_sizes: Sequence[int],
    warmup: int,
    repeats: int,
) -> LinkCost:
    world_size = dist.get_world_size(group)
    samples: list[tuple[int, float]] = []
    for bytes_per_peer in payload_sizes:
        elements_per_peer = max(1, (int(bytes_per_peer) + 3) // 4)
        send = torch.empty((world_size * elements_per_peer,), dtype=torch.float32, device=device)
        receive = torch.empty_like(send)

        def operation(send_buffer: torch.Tensor = send, receive_buffer: torch.Tensor = receive) -> None:
            dist.all_to_all_single(receive_buffer, send_buffer, group=group)

        elapsed = _median_probe(
            operation,
            group=group,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        samples.append((world_size * elements_per_peer * send.element_size(), elapsed))
    return fit_link_cost(samples)


def _peer_for_probe(rank: int, world_size: int, local_world_size: int, *, intra: bool) -> int | None:
    if intra:
        if local_world_size < 2:
            return None
        node_start = rank // local_world_size * local_world_size
        local_rank = rank - node_start
        partner_local = local_rank ^ 1
        partner = node_start + partner_local
        return partner if partner < world_size else None
    if world_size <= local_world_size:
        return None
    node = rank // local_world_size
    nodes = (world_size + local_world_size - 1) // local_world_size
    partner_node = node ^ 1
    if partner_node >= nodes:
        return None
    partner = partner_node * local_world_size + rank % local_world_size
    return partner if 0 <= partner < world_size else None


def _fit_peer_link(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    local_world_size: int,
    intra: bool,
    payload_sizes: Sequence[int],
    warmup: int,
    repeats: int,
) -> LinkCost | None:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    partner = _peer_for_probe(rank, world_size, max(1, int(local_world_size)), intra=intra)
    active = torch.tensor([int(partner is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(active, op=dist.ReduceOp.MAX, group=group)
    if not bool(active.item()):
        return None
    global_ranks = dist.get_process_group_ranks(group)
    peer = None if partner is None else int(global_ranks[partner])
    samples: list[tuple[int, float]] = []
    for payload_bytes in payload_sizes:
        elements = max(1, (int(payload_bytes) + 3) // 4)
        send = torch.empty((elements,), dtype=torch.float32, device=device)
        receive = torch.empty_like(send)

        def operation(send_buffer: torch.Tensor = send, receive_buffer: torch.Tensor = receive) -> None:
            if peer is None:
                return
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send_buffer, peer, group=group),
                    dist.P2POp(dist.irecv, receive_buffer, peer, group=group),
                ]
            )
            for request in requests:
                request.wait()

        elapsed = _median_probe(
            operation,
            group=group,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        samples.append((elements * send.element_size(), elapsed))
    return fit_link_cost(samples)


def fit_perf_model_on_startup(
    base: "HierMoEPerfModel",
    *,
    group: dist.ProcessGroup,
    local_world_size: int,
    payload_sizes: Sequence[int] = (4096, 65536, 1048576),
    warmup: int = 2,
    repeats: int = 5,
) -> "HierMoEPerfModel":
    """Measure the runtime links used by CoRe-MoE placement decisions."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Startup HierMoE performance fitting requires an initialized process group.")
    if dist.get_world_size(group) <= 1:
        raise RuntimeError("Startup HierMoE performance fitting requires at least two EP ranks.")
    device_type = get_device_type()
    device = (
        torch.device(device_type, get_torch_device().current_device()) if device_type != "cpu" else torch.device("cpu")
    )
    a2a = _fit_all_to_all_link(
        group,
        device=device,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    state_intra = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=True,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    state_inter = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=False,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    gather_intra = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=True,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    gather_inter = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=False,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    scatter_intra = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=True,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    scatter_inter = _fit_peer_link(
        group,
        device=device,
        local_world_size=local_world_size,
        intra=False,
        payload_sizes=payload_sizes,
        warmup=warmup,
        repeats=repeats,
    )
    resolved_state_intra = state_intra or base.intra
    resolved_state_inter = state_inter or base.inter[-1]
    state_move = PeerTransferCost(intra=resolved_state_intra, inter=resolved_state_inter)
    gather = PeerTransferCost(intra=gather_intra or resolved_state_intra, inter=gather_inter or resolved_state_inter)
    scatter = PeerTransferCost(
        intra=scatter_intra or resolved_state_intra,
        inter=scatter_inter or resolved_state_inter,
    )
    return HierMoEPerfModel(
        a2a=a2a,
        inter=tuple(resolved_state_inter for _ in base.inter),
        intra=resolved_state_intra,
        source="startup-fit",
        profile_source="startup-fit",
        state_move=state_move,
        gradient_sync=GradientSyncCost(gather=gather, scatter=scatter),
        schema_version=2,
    )


@dataclass(frozen=True)
class HierMoEPerfModel:
    a2a: LinkCost
    inter: tuple[LinkCost, ...]
    intra: LinkCost
    source: str
    state_move: PeerTransferCost | None = None
    gradient_sync: GradientSyncCost | None = None
    schema_version: int = 1
    profile_source: str | None = None

    @classmethod
    def default(cls) -> "HierMoEPerfModel":
        return cls(
            a2a=LinkCost(alpha=1.0, beta=1.0),
            inter=(LinkCost(alpha=0.8, beta=0.75),),
            intra=LinkCost(alpha=0.2, beta=0.25),
            source="default",
        )

    @classmethod
    def from_path(cls, path: str | None) -> "HierMoEPerfModel":
        if not path:
            return cls.default()

        payload = json.loads(Path(path).read_text())
        profile_source = str(payload.get("source", "")).strip() or None
        a2a = payload.get("a2a", {})
        intra = payload.get("intra", {})
        inter_payload = payload.get("inter", [])
        inter = tuple(LinkCost.from_payload(dict(item)) for item in inter_payload)
        state_move_payload = payload.get("state_move")
        gradient_sync_payload = payload.get("gradient_sync")
        return cls(
            a2a=LinkCost.from_payload(a2a),
            inter=inter or (LinkCost(alpha=0.8, beta=0.75),),
            intra=LinkCost.from_payload(intra),
            source=f"file:{path}",
            state_move=(
                PeerTransferCost.from_payload(dict(state_move_payload)) if state_move_payload is not None else None
            ),
            gradient_sync=(
                GradientSyncCost.from_payload(dict(gradient_sync_payload))
                if gradient_sync_payload is not None
                else None
            ),
            schema_version=int(payload.get("schema_version", 1)),
            profile_source=profile_source,
        )

    @property
    def is_profiled(self) -> bool:
        return self.source == "startup-fit" or self.profile_source in {
            "bench_hiermoe_perf_model",
            "startup-fit",
        }

    @property
    def has_runtime_placement_costs(self) -> bool:
        return self.state_move is not None and self.gradient_sync is not None

    @property
    def runtime_cost_status(self) -> str:
        if self.schema_version >= 2 and self.has_runtime_placement_costs:
            return "complete"
        return "fallback"

    def resolved_state_move(self) -> PeerTransferCost:
        if self.state_move is not None:
            return self.state_move
        return PeerTransferCost(intra=self.intra, inter=self.inter[-1])

    def resolved_gradient_sync(self) -> GradientSyncCost:
        if self.gradient_sync is not None:
            return self.gradient_sync
        fallback = self.resolved_state_move()
        return GradientSyncCost(gather=fallback, scatter=fallback)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": int(self.schema_version),
            "a2a": self.a2a.to_payload(),
            "inter": [link.to_payload() for link in self.inter],
            "intra": self.intra.to_payload(),
            "source": self.profile_source or self.source,
        }
        if self.state_move is not None:
            payload["state_move"] = self.state_move.to_payload()
        if self.gradient_sync is not None:
            payload["gradient_sync"] = self.gradient_sync.to_payload()
        return payload

    def estimate_baseline_time(
        self,
        selected_experts: torch.Tensor,
        num_experts: int,
        hidden_size: int,
        bytes_per_element: int,
        ep_size: int | None = None,
    ) -> float:
        ep_size = int(ep_size or num_experts)
        num_local_experts = max(1, num_experts // ep_size)
        counts = duplicate_free_counts_by_expert_group(selected_experts, num_experts, num_local_experts).float()
        n_a2a = float(ep_size * counts.max().item() * hidden_size * bytes_per_element)
        return self.a2a.alpha + n_a2a * self.a2a.beta

    def estimate_hierarchical_time(
        self,
        selected_experts: torch.Tensor,
        num_experts: int,
        hidden_size: int,
        bytes_per_element: int,
        hierarchy: Hierarchy,
        dim: int,
    ) -> float:
        if dim <= 1:
            return self.estimate_baseline_time(
                selected_experts,
                num_experts,
                hidden_size,
                bytes_per_element,
                ep_size=hierarchy.ep_size,
            )

        group_sizes = hierarchy.group_sizes[: dim - 1]
        total = 0.0
        previous_u = 1
        for idx, u_i in enumerate(group_sizes):
            expert_group_size = max(1, num_experts // max(1, hierarchy.ep_size // u_i))
            counts = duplicate_free_counts_by_expert_group(selected_experts, num_experts, expert_group_size).float()
            n_inter = float((u_i / previous_u) * counts.max().item() * hidden_size * bytes_per_element)
            link = self.inter[min(idx, len(self.inter) - 1)]
            total += link.alpha + n_inter * link.beta
            previous_u = u_i

        intra_group_size = max(1, num_experts // hierarchy.ep_size)
        intra_counts = duplicate_free_counts_by_expert_group(selected_experts, num_experts, intra_group_size).float()
        n_intra = float((hierarchy.ep_size / previous_u) * intra_counts.max().item() * hidden_size * bytes_per_element)
        total += self.intra.alpha + n_intra * self.intra.beta
        return total

    def select_dimension(
        self,
        selected_experts: torch.Tensor,
        num_experts: int,
        hidden_size: int,
        bytes_per_element: int,
        hierarchy: Hierarchy,
    ) -> int:
        max_dim = max(1, hierarchy.selected_dim)
        costs = [
            (
                dim,
                self.estimate_hierarchical_time(
                    selected_experts,
                    num_experts,
                    hidden_size,
                    bytes_per_element,
                    hierarchy,
                    dim,
                ),
            )
            for dim in range(1, max_dim + 1)
        ]
        return min(costs, key=lambda item: item[1])[0]
