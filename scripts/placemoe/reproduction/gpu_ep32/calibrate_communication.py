#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Calibrate EP32 hierarchical communication on production CUDA collectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from veomni.arguments import HierMoEConfig
from veomni.distributed.moe.hiermoe import rank_dedup_combine, rank_dedup_dispatch
from veomni.distributed.moe.hiermoe.state import configure_hiermoe


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=(256, 1024, 4096, 16384))
    parser.add_argument("--validation-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument(
        "--raw-payload-bytes",
        type=int,
        nargs="+",
        default=(65536, 262144, 1048576, 4194304, 8388608, 33554432),
    )
    # Validate inside the production payload regime.  The 1 MiB point is still
    # dominated by NCCL startup/protocol selection on this cluster, while even
    # the smallest measured hierarchical stage carries multiple MiB.
    parser.add_argument("--raw-validation-payload-bytes", type=int, default=8388608)
    return parser.parse_args()


def _cluster_scope(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.preflight_report.read_bytes()
    preflight = json.loads(raw)
    if not isinstance(preflight, dict):
        raise RuntimeError("EP32 preflight report must be an object.")
    expected = {
        "schema_version": 1,
        "status": "accepted",
        "world_size": 32,
        "ep_size": 32,
        "ranks_per_node": 8,
    }
    mismatches = {key: (preflight.get(key), value) for key, value in expected.items() if preflight.get(key) != value}
    if mismatches:
        raise RuntimeError(f"EP32 preflight report mismatch: {mismatches}")
    nodes = preflight.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise RuntimeError("EP32 preflight report must contain four nodes.")
    if {node.get("node_rank") for node in nodes if isinstance(node, dict)} != {0, 1, 2, 3}:
        raise RuntimeError("EP32 preflight report has invalid node ranks.")
    reference_node = nodes[0]
    if not isinstance(reference_node, dict):
        raise RuntimeError("EP32 preflight node entry must be an object.")
    software_keys = ("accelerator", "torch", "cuda", "nccl", "triton", "nccl_socket_ifname")
    software_scope = {key: reference_node.get(key) for key in software_keys}
    for key in ("accelerator", "torch", "cuda", "triton", "nccl_socket_ifname"):
        if not isinstance(software_scope[key], str) or not software_scope[key]:
            raise RuntimeError(f"EP32 preflight node has invalid {key}.")
    nccl_version = software_scope["nccl"]
    if (
        not isinstance(nccl_version, list)
        or not nccl_version
        or not all(isinstance(value, int) and value >= 0 for value in nccl_version)
    ):
        raise RuntimeError("EP32 preflight node has an invalid NCCL version.")
    node_expected = {"status": "accepted", "devices": 8, **software_scope}
    for node in nodes:
        if not isinstance(node, dict):
            raise RuntimeError("EP32 preflight node entry must be an object.")
        node_mismatches = {
            key: (node.get(key), value) for key, value in node_expected.items() if node.get(key) != value
        }
        if node_mismatches:
            raise RuntimeError(f"EP32 preflight nodes do not share one software and hardware scope: {node_mismatches}")
        for identity_key in ("hostname", "network_interface_address"):
            if not isinstance(node.get(identity_key), str) or not node[identity_key]:
                raise RuntimeError(f"EP32 preflight node has no {identity_key}.")
        pci_bus_ids = node.get("gpu_pci_bus_ids")
        if (
            not isinstance(pci_bus_ids, list)
            or len(pci_bus_ids) != 8
            or not all(isinstance(value, str) and value for value in pci_bus_ids)
        ):
            raise RuntimeError("EP32 preflight node must identify eight GPU PCI buses.")
    source_sha256 = args.source_sha256.lower()
    if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
        raise RuntimeError("communication source SHA-256 is invalid.")
    return {
        "kind": "gpu32-cluster-software-v1",
        "preflight_sha256": hashlib.sha256(raw).hexdigest(),
        "communication_source_sha256": source_sha256,
        "nodes": nodes,
    }


def _routes(
    *,
    rank: int,
    tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    pattern: str,
    device: torch.device,
) -> torch.Tensor:
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by EP size")
    if top_k > ep_size:
        raise ValueError("top_k must not exceed EP size")
    rows = torch.arange(tokens, dtype=torch.long, device=device).view(-1, 1)
    offsets = torch.arange(top_k, dtype=torch.long, device=device).view(1, -1)
    if pattern == "uniform":
        destination_ranks = torch.remainder(rows * top_k + offsets + rank, ep_size)
    elif pattern == "skew":
        # Every source concentrates on one hot node while retaining distinct
        # experts inside each top-k route.
        destination_ranks = torch.remainder(offsets, min(ep_size, 8))
        destination_ranks = destination_ranks.expand(tokens, -1)
    else:
        raise ValueError(f"unknown route pattern: {pattern}")
    experts_per_rank = num_experts // ep_size
    local_experts = torch.remainder(rows + 3 * offsets, experts_per_rank)
    selected = destination_ranks * experts_per_rank + local_experts
    if top_k > 1:
        ordered = selected.sort(dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise RuntimeError(f"{pattern} generator produced duplicate experts")
    return selected.contiguous()


def _stage_endpoint_bytes(
    context: Any,
    *,
    stage: int,
    hidden_size: int,
) -> float:
    unique_send = getattr(context, f"stage{stage}_unique_send_splits")
    unique_recv = getattr(context, f"stage{stage}_unique_recv_splits")
    if stage == 3:
        assignment_send = context.assignment_send_splits
        assignment_recv = context.assignment_recv_splits
    else:
        assignment_send = getattr(context, f"stage{stage}_assignment_send_splits")
        assignment_recv = getattr(context, f"stage{stage}_assignment_recv_splits")
    if any(value is None for value in (unique_send, unique_recv, assignment_send, assignment_recv)):
        raise RuntimeError(f"hierarchical stage {stage} did not expose split sizes")
    hidden_bytes = hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
    metadata_bytes = 3 * 4 if stage == 1 else 2 * 4
    unique_send_total = sum(int(value) for value in unique_send)
    unique_recv_total = sum(int(value) for value in unique_recv)
    assignment_send_total = sum(int(value) for value in assignment_send)
    assignment_recv_total = sum(int(value) for value in assignment_recv)
    dispatch = max(
        hidden_bytes * unique_send_total + metadata_bytes * assignment_send_total,
        hidden_bytes * unique_recv_total + metadata_bytes * assignment_recv_total,
    )
    combine = hidden_bytes * max(unique_send_total, unique_recv_total)
    return float(dispatch + combine)


def _elapsed(events: dict[str, Any], first: str, second: str) -> float:
    return float(events[first][0].elapsed_time(events[first][1])) + float(
        events[second][0].elapsed_time(events[second][1])
    )


def _hierarchical_sample(
    *,
    selected: torch.Tensor,
    hidden_size: int,
    num_experts: int,
    ep_size: int,
) -> tuple[dict[str, float], Any]:
    device = selected.device
    tokens, top_k = selected.shape
    hidden = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    dist.barrier()
    wall_start = torch.cuda.Event(enable_timing=True)
    wall_end = torch.cuda.Event(enable_timing=True)
    wall_start.record()
    permuted, context, _ = rank_dedup_dispatch(hidden, selected, weights, num_experts, dist.group.WORLD)
    rank_dedup_combine(permuted, context)
    stage_bytes = [_stage_endpoint_bytes(context, stage=stage, hidden_size=hidden_size) for stage in (1, 2, 3)]
    wall_end.record()
    torch.cuda.synchronize()
    if context.mode != "hierarchical3d" or context.internal_timing_events is None:
        raise RuntimeError("production dispatch did not expose three-stage hierarchical CUDA timing")
    events = context.internal_timing_events
    required = {
        "stage1_a2a",
        "stage2_a2a",
        "stage3_a2a",
        "combine_stage3_a2a",
        "combine_stage2_a2a",
        "combine_stage1_a2a",
    }
    if not required.issubset(events):
        raise RuntimeError(f"missing internal communication events: {sorted(required - events.keys())}")
    measured = torch.tensor(
        [
            *stage_bytes,
            _elapsed(events, "stage1_a2a", "combine_stage1_a2a"),
            _elapsed(events, "stage2_a2a", "combine_stage2_a2a"),
            _elapsed(events, "stage3_a2a", "combine_stage3_a2a"),
            float(wall_start.elapsed_time(wall_end)),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
    return (
        {
            "stage1_payload_endpoint_bytes": float(measured[0].item()),
            "stage2_payload_endpoint_bytes": float(measured[1].item()),
            "stage3_payload_endpoint_bytes": float(measured[2].item()),
            "actual_stage1_a2a_ms": float(measured[3].item()),
            "actual_stage2_a2a_ms": float(measured[4].item()),
            "actual_stage3_a2a_ms": float(measured[5].item()),
            "actual_hierarchical_wall_ms": float(measured[6].item()),
        },
        context,
    )


def _raw_a2a(
    group: dist.ProcessGroup,
    *,
    payload_bytes: int,
    pattern: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    group_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    total_elements = max(group_size, payload_bytes // element_size)
    total_elements -= total_elements % group_size
    if pattern == "uniform":
        input_splits = [total_elements // group_size] * group_size
    elif pattern == "skew":
        hot = max(1, int(total_elements * 0.75))
        remainder = total_elements - hot
        base, extra = divmod(remainder, max(1, group_size - 1))
        input_splits = [hot] + [base + int(index < extra) for index in range(group_size - 1)]
    else:
        raise ValueError(pattern)
    output_splits = [input_splits[group_rank]] * group_size
    source = torch.zeros((sum(input_splits),), dtype=torch.bfloat16, device=device)
    target = torch.empty((sum(output_splits),), dtype=torch.bfloat16, device=device)
    samples = []
    for iteration in range(warmup + iterations):
        dist.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_to_all_single(
            target,
            source,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        end.record()
        torch.cuda.synchronize()
        value = torch.tensor([float(start.elapsed_time(end))], dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        if iteration >= warmup:
            samples.append(float(value.item()))
    return {
        "group_size": group_size,
        "payload_bytes_per_source": int(sum(input_splits) * element_size),
        "pattern": pattern,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _fit_origin(rows: list[dict[str, float]], feature: str, target: str) -> float:
    numerator = sum(float(row[feature]) * float(row[target]) for row in rows)
    denominator = sum(float(row[feature]) ** 2 for row in rows)
    if denominator <= 0.0:
        raise RuntimeError(f"no feature energy for {feature}")
    return max(0.0, numerator / denominator)


def _diagnostics(rows: list[dict[str, float]], feature: str, target: str, coefficient: float) -> dict[str, Any]:
    actual = [float(row[target]) for row in rows]
    predicted = [coefficient * float(row[feature]) for row in rows]
    residuals = [truth - estimate for truth, estimate in zip(actual, predicted, strict=True)]
    mean = statistics.mean(actual)
    variance = sum((value - mean) ** 2 for value in actual)
    squared = sum(value**2 for value in residuals)
    return {
        "count": len(rows),
        "actual_mean_ms": mean,
        "prediction_mean_ms": statistics.mean(predicted),
        "rmse_ms": math.sqrt(squared / len(rows)),
        "mape_percent": statistics.mean(abs(delta) / truth for delta, truth in zip(residuals, actual, strict=True))
        * 100.0,
        "r_squared": None if variance <= 0.0 else 1.0 - squared / variance,
    }


def _fit_raw_balanced_link(
    rows: list[dict[str, Any]],
    *,
    validation_payload_bytes: int,
    label: str,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    balanced = {int(row["payload_bytes_per_source"]): row for row in rows if row.get("pattern") == "uniform"}
    if validation_payload_bytes not in balanced:
        raise RuntimeError(f"{label} has no held-out payload {validation_payload_bytes}")
    lower_sizes = [size for size in balanced if size < validation_payload_bytes]
    upper_sizes = [size for size in balanced if size > validation_payload_bytes]
    if not lower_sizes or not upper_sizes:
        raise RuntimeError(f"{label} held-out payload must be bracketed by calibration sizes")
    lower_size = max(lower_sizes)
    upper_size = min(upper_sizes)
    lower_ms = float(balanced[lower_size]["median_ms"])
    upper_ms = float(balanced[upper_size]["median_ms"])
    beta = (upper_ms - lower_ms) / float(upper_size - lower_size)
    alpha = lower_ms - beta * float(lower_size)
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha < 0.0 or beta <= 0.0:
        raise RuntimeError(f"{label} produced invalid local alpha-beta coefficients")
    predicted = alpha + beta * float(validation_payload_bytes)
    actual = [float(value) for value in balanced[validation_payload_bytes]["samples_ms"]]
    residuals = [value - predicted for value in actual]
    mean = statistics.mean(actual)
    variance = sum((value - mean) ** 2 for value in actual)
    squared = sum(value**2 for value in residuals)
    diagnostics = {
        "kind": "raw_balanced_a2a_local_alpha_beta_holdout",
        "count": len(actual),
        "held_out_payload_bytes_per_source": validation_payload_bytes,
        "actual_mean_ms": mean,
        "prediction_mean_ms": predicted,
        "rmse_ms": math.sqrt(squared / len(actual)),
        "mape_percent": statistics.mean(abs(delta) / truth for delta, truth in zip(residuals, actual, strict=True))
        * 100.0,
        "r_squared": None if variance <= 0.0 else 1.0 - squared / variance,
    }
    model = {
        "kind": "local_alpha_beta_bracketing_workload_payload",
        "pattern": "uniform",
        "feature": "payload_bytes_per_source",
        "alpha_ms": alpha,
        "beta_ms_per_byte": beta,
        "lower_calibration_payload_bytes": lower_size,
        "upper_calibration_payload_bytes": upper_size,
        "lower_calibration_median_ms": lower_ms,
        "upper_calibration_median_ms": upper_ms,
    }
    return beta, model, diagnostics


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        ep_size = dist.get_world_size()
        if ep_size != 32 or args.ranks_per_node != 8:
            raise RuntimeError(f"expected EP32 as 4x8, got ep_size={ep_size}")
        if args.validation_tokens not in args.tokens:
            raise ValueError("validation-tokens must be one of --tokens")
        configure_hiermoe(
            HierMoEConfig(
                enable=True,
                token_dedup=True,
                communication_mode="hierarchical",
                expert_swap=False,
                hierarchy_group_sizes=[2, args.ranks_per_node, ep_size],
            ),
            dist.group.WORLD,
        )
        rows: list[dict[str, Any]] = []
        latest_context = None
        for tokens in args.tokens:
            for pattern in ("uniform", "skew"):
                selected = _routes(
                    rank=rank,
                    tokens=tokens,
                    top_k=args.top_k,
                    num_experts=args.num_experts,
                    ep_size=ep_size,
                    pattern=pattern,
                    device=torch.device("cuda", local_rank),
                )
                for iteration in range(args.warmup + args.iterations):
                    measured, latest_context = _hierarchical_sample(
                        selected=selected,
                        hidden_size=args.hidden_size,
                        num_experts=args.num_experts,
                        ep_size=ep_size,
                    )
                    if iteration >= args.warmup:
                        rows.append(
                            {
                                "tokens_per_rank": tokens,
                                "pattern": pattern,
                                "iteration": iteration - args.warmup,
                                **measured,
                            }
                        )
        if (
            latest_context is None
            or latest_context.stage1_group is None
            or latest_context.stage2_group is None
            or latest_context.stage3_group is None
        ):
            raise RuntimeError("hierarchical process groups were not created")
        raw = {"stage1_inter_group": [], "stage2_mid_group": [], "stage3_intra_group": []}
        for payload_bytes in args.raw_payload_bytes:
            for pattern in ("uniform", "skew"):
                raw["stage1_inter_group"].append(
                    _raw_a2a(
                        latest_context.stage1_group,
                        payload_bytes=payload_bytes,
                        pattern=pattern,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=torch.device("cuda", local_rank),
                    )
                )
                raw["stage2_mid_group"].append(
                    _raw_a2a(
                        latest_context.stage2_group,
                        payload_bytes=payload_bytes,
                        pattern=pattern,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=torch.device("cuda", local_rank),
                    )
                )
                raw["stage3_intra_group"].append(
                    _raw_a2a(
                        latest_context.stage3_group,
                        payload_bytes=payload_bytes,
                        pattern=pattern,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=torch.device("cuda", local_rank),
                    )
                )
        if rank == 0:
            fit_rows = [row for row in rows if int(row["tokens_per_rank"]) != args.validation_tokens]
            validation_rows = [row for row in rows if int(row["tokens_per_rank"]) == args.validation_tokens]
            production_inter = _fit_origin(fit_rows, "stage1_payload_endpoint_bytes", "actual_stage1_a2a_ms")
            production_mid = _fit_origin(fit_rows, "stage2_payload_endpoint_bytes", "actual_stage2_a2a_ms")
            production_intra = _fit_origin(fit_rows, "stage3_payload_endpoint_bytes", "actual_stage3_a2a_ms")
            inter, inter_model, inter_validation = _fit_raw_balanced_link(
                raw["stage1_inter_group"],
                validation_payload_bytes=args.raw_validation_payload_bytes,
                label="inter-stage link",
            )
            mid, mid_model, mid_validation = _fit_raw_balanced_link(
                raw["stage2_mid_group"],
                validation_payload_bytes=args.raw_validation_payload_bytes,
                label="mid-stage PCIe link",
            )
            intra, intra_model, intra_validation = _fit_raw_balanced_link(
                raw["stage3_intra_group"],
                validation_payload_bytes=args.raw_validation_payload_bytes,
                label="intra-stage NVLink link",
            )
            payload = {
                "schema_version": 4,
                "source": "gpu32-a6000-ep32-communication-calibration",
                "run_name": args.run_name,
                "scope": _cluster_scope(args),
                "topology": {
                    "accelerator": "NVIDIA RTX A6000",
                    "nodes": 4,
                    "gpus_per_node": args.ranks_per_node,
                    "ep_size": ep_size,
                    "ranks_per_node": args.ranks_per_node,
                    "hierarchy_group_sizes": [2, args.ranks_per_node, ep_size],
                    "hidden_size": args.hidden_size,
                    "bytes_per_element": 2,
                },
                "coefficients": {
                    "level_ms_per_byte": [inter, mid, intra],
                    "inter_ms_per_byte": inter,
                    "mid_ms_per_byte": mid,
                    "intra_ms_per_byte": intra,
                },
                "coefficient_features": {
                    "levels": [
                        "raw_balanced_a2a_payload_bytes_per_source",
                        "raw_balanced_a2a_payload_bytes_per_source",
                        "raw_balanced_a2a_payload_bytes_per_source",
                    ]
                },
                "link_models": {
                    "stage1_inter": inter_model,
                    "stage2_mid": mid_model,
                    "stage3_intra": intra_model,
                },
                "fit": {
                    "kind": "physical_link_local_alpha_beta",
                    "raw_validation_payload_bytes": args.raw_validation_payload_bytes,
                },
                "validation": {
                    "held_out_payload_bytes_per_source": args.raw_validation_payload_bytes,
                    "stage1_inter": inter_validation,
                    "stage2_mid": mid_validation,
                    "stage3_intra": intra_validation,
                },
                "production_route_diagnostics": {
                    "held_out_tokens_per_rank": args.validation_tokens,
                    "fit": {
                        "stage1_inter": _diagnostics(
                            fit_rows,
                            "stage1_payload_endpoint_bytes",
                            "actual_stage1_a2a_ms",
                            production_inter,
                        ),
                        "stage2_mid": _diagnostics(
                            fit_rows,
                            "stage2_payload_endpoint_bytes",
                            "actual_stage2_a2a_ms",
                            production_mid,
                        ),
                        "stage3_intra": _diagnostics(
                            fit_rows,
                            "stage3_payload_endpoint_bytes",
                            "actual_stage3_a2a_ms",
                            production_intra,
                        ),
                    },
                    "validation": {
                        "stage1_inter": _diagnostics(
                            validation_rows,
                            "stage1_payload_endpoint_bytes",
                            "actual_stage1_a2a_ms",
                            production_inter,
                        ),
                        "stage2_mid": _diagnostics(
                            validation_rows,
                            "stage2_payload_endpoint_bytes",
                            "actual_stage2_a2a_ms",
                            production_mid,
                        ),
                        "stage3_intra": _diagnostics(
                            validation_rows,
                            "stage3_payload_endpoint_bytes",
                            "actual_stage3_a2a_ms",
                            production_intra,
                        ),
                    },
                },
                "coverage": {
                    "route_patterns": ["uniform", "skew"],
                    "tokens_per_rank": list(args.tokens),
                    "production_hierarchical_samples": rows,
                    "raw_all_to_all": raw,
                },
            }
            _atomic_json(args.output, payload)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "coefficients": payload["coefficients"],
                        "validation": payload["validation"],
                    },
                    indent=2,
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
