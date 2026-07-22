"""Optional MoE routing traces for LAER-MoE cost-model validation.

This module is intentionally off by default.  When enabled, it gathers the
top-k token-to-expert histogram across the EP group and emits enough
information for an offline LAER-MoE Sec. 3.2 cost-model validator.  It does
not change routing or computation decisions.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from .timing import current_full_profile_phase


_MOE_VALIDATION_RECORDS: list[dict[str, Any]] = []


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def moe_validation_enabled() -> bool:
    return _env_flag("VEOMNI_MOE_VALIDATOR_ENABLE", False)


def _rank_tuple(ep_group: dist.ProcessGroup | None) -> tuple[int, int, int]:
    global_rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
    if ep_group is not None and dist.is_available() and dist.is_initialized():
        ep_rank = int(dist.get_rank(group=ep_group))
        ep_size = int(dist.get_world_size(group=ep_group))
    else:
        ep_rank = 0
        ep_size = 1
    return global_rank, ep_rank, ep_size


def _expert_owner(num_experts: int, ep_size: int) -> list[int]:
    if ep_size <= 1:
        return [0 for _ in range(num_experts)]
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    experts_per_rank = num_experts // ep_size
    return [expert // experts_per_rank for expert in range(num_experts)]


def record_moe_validation_routing(
    record: dict[str, Any] | None,
    *,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    """Collect R[src_rank, expert_id] for one local MoE forward call.

    Every EP rank must call this when enabled because it performs an all-gather.
    Only EP rank 0 stores the gathered matrix to avoid duplicate payloads.
    """

    if record is None or not moe_validation_enabled():
        return

    global_rank, ep_rank, ep_size = _rank_tuple(ep_group)
    flat = selected_experts.detach().reshape(-1).to(torch.int64)
    local_hist = torch.bincount(flat, minlength=int(num_experts)).to(torch.int64)

    if ep_group is not None and ep_size > 1:
        gathered = torch.empty((ep_size, int(num_experts)), device=local_hist.device, dtype=torch.int64)
        dist.all_gather_into_tensor(gathered, local_hist, group=ep_group)
    else:
        gathered = local_hist.view(1, -1)

    if ep_rank != 0:
        return

    num_layers = int(record.get("num_layers") or 0)
    call_index = int(record.get("call_index") or 0)
    layer = record.get("layer")
    micro_batch = call_index // num_layers if num_layers > 0 else call_index
    routing_histogram = gathered.cpu().tolist()
    expert_owner = _expert_owner(int(num_experts), ep_size)
    tokens_on_rank = [0 for _ in range(ep_size)]
    for src_row in routing_histogram:
        for expert_id, count in enumerate(src_row):
            tokens_on_rank[expert_owner[expert_id]] += int(count)

    _MOE_VALIDATION_RECORDS.append(
        {
            "record_type": "routing_histogram",
            "call_index": call_index,
            "micro_batch": int(micro_batch),
            "phase": record.get("phase") or current_full_profile_phase(),
            "layer": layer,
            "num_layers": num_layers or None,
            "num_experts": int(num_experts),
            "ep_size": int(ep_size),
            "top_k": int(record.get("top_k") or 1),
            "tokens": int(record.get("tokens") or 0),
            "token_expert_assignments": int(record.get("token_expert_assignments") or 0),
            "global_rank": int(global_rank),
            "ep_rank": int(ep_rank),
            "routing_histogram": routing_histogram,
            "expert_owner": expert_owner,
            "tokens_on_rank": tokens_on_rank,
        }
    )


def flush_moe_validation_records(current_step: int) -> dict[str, Any]:
    global _MOE_VALIDATION_RECORDS
    records = _MOE_VALIDATION_RECORDS
    _MOE_VALIDATION_RECORDS = []
    if not records:
        return {}
    return {
        "step": int(current_step),
        "record_type": "moe_validation_routing",
        "records": records,
        "note": (
            "R[src_rank, expert_id] after top-k routing. Under ordinary EP, "
            "S[src_rank, expert_id, dst_rank] is nonzero only for dst_rank=expert_owner[expert_id]."
        ),
    }
