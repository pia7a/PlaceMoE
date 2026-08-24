"""Lightweight per-layer physical MoE load collection for convergence runs."""

from __future__ import annotations

import os
import re
from typing import Any

import torch
import torch.distributed as dist


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PHYSICAL_TOKENS: dict[int, int] = {}
_LAYER_KEYS: dict[int, str] = {}
_DEVICE: torch.device | None = None


def physical_moe_load_enabled() -> bool:
    """Return whether the convergence run requested physical-load metrics."""

    return bool(os.environ.get("VEOMNI_CONVERGENCE_METRICS_DIR"))


def _layer_index(layer_key: str | None) -> int:
    if layer_key:
        match = _LAYER_PATTERN.search(layer_key)
        if match is not None:
            return int(match.group(1))
    raise RuntimeError(f"Physical MoE load metrics require a canonical layer key, got {layer_key!r}.")


def bind_physical_moe_layer_keys(model: Any) -> int:
    """Attach canonical names to expert modules for non-HierMoE baselines."""

    attached = 0
    for name, module in model.named_modules():
        if _LAYER_PATTERN.search(name) is None or not name.endswith(".mlp.experts"):
            continue
        module._veomni_physical_load_layer_key = name
        attached += 1
    return attached


def record_physical_moe_load(
    layer_key: str | None,
    physical_tokens: int,
    *,
    device: torch.device,
) -> None:
    """Accumulate expert-compute tokens on the current physical accelerator.

    ``physical_tokens`` is measured after dispatch and replica remapping, so it
    represents the rows actually consumed by local expert GEMMs. Recomputed
    activation-checkpoint forwards are intentionally counted because they are
    real physical expert work in the training step.
    """

    if not physical_moe_load_enabled():
        return
    global _DEVICE
    layer_index = _layer_index(layer_key)
    _PHYSICAL_TOKENS[layer_index] = _PHYSICAL_TOKENS.get(layer_index, 0) + int(physical_tokens)
    _LAYER_KEYS.setdefault(layer_index, layer_key or f"layer_{layer_index}")
    _DEVICE = device


def _max_over_mean(values: list[int]) -> float:
    mean = sum(values) / len(values) if values else 0.0
    return max(values) / mean if mean > 0 else 0.0


def flush_physical_moe_load(
    *,
    process_group: dist.ProcessGroup | None,
    expected_num_layers: int,
) -> dict[str, Any]:
    """Gather local physical loads and reset the current step's counters."""

    global _DEVICE
    if not physical_moe_load_enabled():
        return {}
    if _DEVICE is None:
        raise RuntimeError("Physical MoE load metrics were enabled but no expert compute was recorded in this step.")

    unexpected = sorted(index for index in _PHYSICAL_TOKENS if not 0 <= index < expected_num_layers)
    if unexpected:
        raise RuntimeError(f"Physical MoE load collector saw out-of-range layer indices: {unexpected}")
    missing = [index for index in range(expected_num_layers) if index not in _PHYSICAL_TOKENS]
    if missing:
        raise RuntimeError(f"Physical MoE load collector missed layers: {missing}")

    local_counts = torch.tensor(
        [_PHYSICAL_TOKENS[index] for index in range(expected_num_layers)],
        dtype=torch.long,
        device=_DEVICE,
    )
    local_rank = torch.tensor([dist.get_rank() if dist.is_initialized() else 0], dtype=torch.long, device=_DEVICE)
    group_size = dist.get_world_size(process_group) if process_group is not None and dist.is_initialized() else 1

    if group_size > 1:
        gathered_counts = torch.empty(group_size * expected_num_layers, dtype=torch.long, device=_DEVICE)
        dist.all_gather_into_tensor(gathered_counts, local_counts, group=process_group)
        gathered_counts = gathered_counts.view(group_size, expected_num_layers)
        gathered_ranks = torch.empty(group_size, dtype=torch.long, device=_DEVICE)
        dist.all_gather_into_tensor(gathered_ranks, local_rank, group=process_group)
    else:
        gathered_counts = local_counts.view(1, expected_num_layers)
        gathered_ranks = local_rank

    rank_ids = [int(value) for value in gathered_ranks.cpu().tolist()]
    by_layer = [[int(value) for value in row] for row in gathered_counts.transpose(0, 1).contiguous().cpu().tolist()]
    layer_ratios = [_max_over_mean(row) for row in by_layer]
    rank_totals = [sum(by_layer[layer][rank] for layer in range(expected_num_layers)) for rank in range(group_size)]
    layer_keys = [_LAYER_KEYS.get(index, f"layer_{index}") for index in range(expected_num_layers)]

    _PHYSICAL_TOKENS.clear()
    _LAYER_KEYS.clear()
    _DEVICE = None
    return {
        "physical_rank_ids": rank_ids,
        "physical_rank_tokens": rank_totals,
        "max_rank_tokens_over_mean": _max_over_mean(rank_totals),
        "layer_keys": layer_keys,
        "physical_rank_tokens_by_layer": by_layer,
        "layer_max_rank_tokens_over_mean": layer_ratios,
    }
