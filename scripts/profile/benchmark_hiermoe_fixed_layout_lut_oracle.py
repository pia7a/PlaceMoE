#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Optimize only the source LUT of a fixed redundant-expert layout.

Each EP rank owns one source-LUT row and evaluates all one-expert moves to
another existing physical copy.  Candidate deltas are exact for the cached
Forward routes and the source-aware hybrid communication/assignment model.
Only one local winner per rank is gathered; no candidate-sized collective is
performed.  Accepted moves update the winning rank's cached statistics, and
coordinate descent stops when no globally positive move remains.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.profile.benchmark_hiermoe_forward_lut_cover_oracle import (
    _initialize,
    _layer_name,
    _layer_state,
    _load_route,
    _max_rank_time,
    _parse_int_list,
    _percentile,
    _synchronize,
)
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.statistical_scorer import (
    ForwardLUTCoverCompactStatistics,
    prepare_forward_lut_cover_compact_statistics,
    score_forward_lut_move_compact_statistics,
)
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(2, 3, 4, 5))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(6, 7))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument("--minimum-gain-ms", type=float, default=1e-5)
    parser.add_argument("--max-rounds", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-layout", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def _planner(args: argparse.Namespace) -> GreedyCommunicationPlanner:
    def reducer(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=(args.ranks_per_node, args.ep_size),
            source="fixed-layout-lut-oracle",
            local_world_size=args.ranks_per_node,
        ),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=args.hidden_size,
        bytes_per_element=args.bytes_per_element,
        slots_per_rank=args.slots_per_rank,
        reducer=reducer,
        process_group=dist.group.WORLD,
        traffic_inter_ms_per_byte=args.inter_ms_per_byte,
        traffic_intra_ms_per_byte=args.intra_ms_per_byte,
        traffic_route_ms_per_assignment=args.route_ms_per_assignment,
        traffic_communication_phase_multiplier=args.communication_phase_multiplier,
        traffic_compute_phase_multiplier=args.compute_phase_multiplier,
        forward_compute_per_assignment=args.compute_ms_per_assignment,
        assume_unique_routes=True,
    )


def _local_candidates(
    layout: torch.Tensor,
    source_lut: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts: list[int] = []
    destinations: list[int] = []
    for expert in range(int(num_experts)):
        current = int(source_lut[expert].item())
        copies = torch.nonzero(layout == expert, as_tuple=False).reshape(-1)
        for destination in copies.detach().to(device="cpu").tolist():
            if int(destination) != current:
                experts.append(expert)
                destinations.append(int(destination))
    return (
        torch.tensor(experts, dtype=torch.long, device=layout.device),
        torch.tensor(destinations, dtype=torch.long, device=layout.device),
    )


def _local_state(
    *,
    planner: GreedyCommunicationPlanner,
    selected: torch.Tensor,
    source_lut: torch.Tensor,
    source_rank: int,
    num_experts: int,
) -> tuple[ForwardLUTCoverCompactStatistics, torch.Tensor]:
    physical = source_lut.index_select(0, selected.reshape(-1)).view_as(selected)
    unique = planner._local_packed_counts(physical)
    assignment = planner._local_packed_assignment_counts(physical)
    endpoint = planner._local_traffic_endpoint_statistics(
        unique,
        assignment[:, : planner.ep_size],
        source_rank=source_rank,
    ).squeeze(0)
    compact = prepare_forward_lut_cover_compact_statistics(
        planner,
        selected,
        source_logical_to_physical=source_lut,
        num_experts=num_experts,
    )
    return compact, endpoint


def _endpoint_delta(
    *,
    planner: GreedyCommunicationPlanner,
    compact: ForwardLUTCoverCompactStatistics,
    experts: torch.Tensor,
    destinations: torch.Tensor,
    source_lut: torch.Tensor,
    source_rank: int,
    num_experts: int,
) -> torch.Tensor:
    communication_delta, assignment_delta = score_forward_lut_move_compact_statistics(
        planner,
        compact,
        experts,
        destinations,
        source_logical_to_physical=source_lut,
        num_experts=num_experts,
    )
    return planner._local_traffic_endpoint_statistics(
        communication_delta,
        assignment_delta,
        source_rank=source_rank,
    )


def _cost(planner: GreedyCommunicationPlanner, endpoint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    details = planner._traffic_endpoint_cost_details(endpoint)
    return details[0], details[1]


def _global_endpoints(local_endpoints: torch.Tensor) -> torch.Tensor:
    result = local_endpoints.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _gather_lut(source_lut: torch.Tensor, ep_size: int) -> torch.Tensor:
    gathered = source_lut.new_empty((int(ep_size), source_lut.numel()))
    dist.all_gather_into_tensor(gathered, source_lut.contiguous())
    return gathered


def _evaluate_lut(
    *,
    planner: GreedyCommunicationPlanner,
    source_lut: torch.Tensor,
    steps: tuple[int, ...],
    layer: int,
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    local_endpoints: list[torch.Tensor] = []
    for step in steps:
        selected = _load_route(
            args.route_root,
            step=step,
            layer=layer,
            rank=rank,
            device=device,
        )
        _compact, endpoint = _local_state(
            planner=planner,
            selected=selected,
            source_lut=source_lut,
            source_rank=rank,
            num_experts=args.num_experts,
        )
        local_endpoints.append(endpoint)
    global_endpoints = _global_endpoints(torch.stack(local_endpoints))
    communication, compute = _cost(planner, global_endpoints)
    return (
        float((communication + compute).mean().item()),
        float(communication.mean().item()),
        float(compute.mean().item()),
    )


def _optimize_layer(
    *,
    planner: GreedyCommunicationPlanner,
    layout: torch.Tensor,
    source_lut: torch.Tensor,
    steps: tuple[int, ...],
    layer: int,
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, object]]:
    selected_by_step = [
        _load_route(
            args.route_root,
            step=step,
            layer=layer,
            rank=rank,
            device=device,
        )
        for step in steps
    ]
    local_states = [
        _local_state(
            planner=planner,
            selected=selected,
            source_lut=source_lut,
            source_rank=rank,
            num_experts=args.num_experts,
        )
        for selected in selected_by_step
    ]
    global_endpoints = _global_endpoints(torch.stack([state[1] for state in local_states]))
    initial_communication, initial_compute = _cost(planner, global_endpoints)
    initial_total = initial_communication + initial_compute
    baseline_total = initial_total
    rounds: list[dict[str, object]] = []

    _synchronize(device)
    started = time.perf_counter()
    for round_index in range(int(args.max_rounds)):
        experts, destinations = _local_candidates(
            layout,
            source_lut,
            num_experts=args.num_experts,
        )
        local_candidate_total = global_endpoints.new_empty((experts.numel(),))
        endpoint_delta_by_step: list[torch.Tensor] = []
        if experts.numel():
            candidate_sum = global_endpoints.new_zeros((experts.numel(),))
            for step_index, (compact, _endpoint) in enumerate(local_states):
                endpoint_delta = _endpoint_delta(
                    planner=planner,
                    compact=compact,
                    experts=experts,
                    destinations=destinations,
                    source_lut=source_lut,
                    source_rank=rank,
                    num_experts=args.num_experts,
                )
                endpoint_delta_by_step.append(endpoint_delta)
                communication, compute = _cost(
                    planner,
                    global_endpoints[step_index].unsqueeze(0) + endpoint_delta,
                )
                candidate_sum += communication + compute
            local_candidate_total.copy_(candidate_sum / float(len(steps)))
            local_index = int(local_candidate_total.argmin().item())
            local_cost = float(local_candidate_total[local_index].item())
            local_expert = int(experts[local_index].item())
            local_destination = int(destinations[local_index].item())
        else:
            local_index = -1
            local_cost = float("inf")
            local_expert = -1
            local_destination = -1

        local_winner = torch.tensor(
            [local_cost, float(local_expert), float(local_destination), float(rank)],
            dtype=torch.float32,
            device=device,
        )
        gathered = local_winner.new_empty((args.ep_size, local_winner.numel()))
        dist.all_gather_into_tensor(gathered, local_winner)
        winner_rank_index = int(gathered[:, 0].argmin().item())
        winner_cost = float(gathered[winner_rank_index, 0].item())
        winner_expert = int(gathered[winner_rank_index, 1].item())
        winner_destination = int(gathered[winner_rank_index, 2].item())
        winner_rank = int(gathered[winner_rank_index, 3].item())
        current_cost = float(baseline_total.mean().item())
        gain = current_cost - winner_cost
        if not gain > float(args.minimum_gain_ms):
            break

        winner_delta = global_endpoints.new_zeros(global_endpoints.shape)
        if rank == winner_rank:
            winner_delta.copy_(torch.stack([delta[local_index] for delta in endpoint_delta_by_step]))
            source_lut[winner_expert] = winner_destination
            local_states = [
                _local_state(
                    planner=planner,
                    selected=selected,
                    source_lut=source_lut,
                    source_rank=rank,
                    num_experts=args.num_experts,
                )
                for selected in selected_by_step
            ]
        dist.all_reduce(winner_delta, op=dist.ReduceOp.SUM)
        global_endpoints += winner_delta
        communication, compute = _cost(planner, global_endpoints)
        baseline_total = communication + compute
        rounds.append(
            {
                "round": round_index,
                "source_rank": winner_rank,
                "expert": winner_expert,
                "destination_slot": winner_destination,
                "gain_ms": gain,
                "cost_ms": float(baseline_total.mean().item()),
            }
        )
    _synchronize(device)
    elapsed_ms = _max_rank_time((time.perf_counter() - started) * 1000.0, device)

    final_communication, final_compute = _cost(planner, global_endpoints)
    return source_lut, {
        "round_count": len(rounds),
        "initial_cost_ms": float(initial_total.mean().item()),
        "final_cost_ms": float((final_communication + final_compute).mean().item()),
        "gain_ms": float((initial_total.mean() - (final_communication + final_compute).mean()).item()),
        "initial_communication_ms": float(initial_communication.mean().item()),
        "final_communication_ms": float(final_communication.mean().item()),
        "initial_compute_ms": float(initial_compute.mean().item()),
        "final_compute_ms": float(final_compute.mean().item()),
        "elapsed_ms": elapsed_ms,
        "rounds": rounds,
    }


def main() -> None:
    args = _parse_args()
    if args.layers <= 0:
        raise ValueError("layers must be positive.")
    if args.max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")
    rank, _world_size, device = _initialize(args.ep_size)
    planner = _planner(args)
    input_payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    output_payload = copy.deepcopy(input_payload)
    results: list[dict[str, object]] = []

    if args.warmup:
        layout, _owners, lut = _layer_state(
            input_payload,
            layer=args.layer_start,
            device=device,
            args=args,
        )
        _optimize_layer(
            planner=planner,
            layout=layout,
            source_lut=lut[rank].clone(),
            steps=(args.optimize_steps[0],),
            layer=args.layer_start,
            rank=rank,
            device=device,
            args=argparse.Namespace(**{**vars(args), "max_rounds": 1}),
        )

    wall_started = time.perf_counter()
    for layer in range(args.layer_start, args.layer_start + args.layers):
        layout, _owners, lut = _layer_state(
            input_payload,
            layer=layer,
            device=device,
            args=args,
        )
        original_source_lut = lut[rank].clone()
        optimized_source_lut, optimize = _optimize_layer(
            planner=planner,
            layout=layout,
            source_lut=original_source_lut.clone(),
            steps=args.optimize_steps,
            layer=layer,
            rank=rank,
            device=device,
            args=args,
        )
        validation_baseline = _evaluate_lut(
            planner=planner,
            source_lut=original_source_lut,
            steps=args.validation_steps,
            layer=layer,
            rank=rank,
            device=device,
            args=args,
        )
        validation_final = _evaluate_lut(
            planner=planner,
            source_lut=optimized_source_lut,
            steps=args.validation_steps,
            layer=layer,
            rank=rank,
            device=device,
            args=args,
        )
        gathered_lut = _gather_lut(optimized_source_lut, args.ep_size)
        result = {
            "layer": layer,
            "optimize": optimize,
            "validation": {
                "baseline_ms": validation_baseline[0],
                "final_ms": validation_final[0],
                "gain_ms": validation_baseline[0] - validation_final[0],
                "communication_gain_ms": validation_baseline[1] - validation_final[1],
                "compute_gain_ms": validation_baseline[2] - validation_final[2],
            },
        }
        results.append(result)
        if rank == 0:
            layers = output_payload["layers"]
            layers[_layer_name(layer)]["source_logical_to_physical"] = (
                gathered_lut.detach().to(device="cpu", dtype=torch.long).tolist()
            )
            print(json.dumps(result, sort_keys=True), flush=True)

    if rank == 0:
        if args.output_layout is None or args.output_report is None:
            raise ValueError("Rank 0 requires --output-layout and --output-report.")
        optimize_baseline = sum(float(row["optimize"]["initial_cost_ms"]) for row in results)
        optimize_gain = sum(float(row["optimize"]["gain_ms"]) for row in results)
        validation_baseline = sum(float(row["validation"]["baseline_ms"]) for row in results)
        validation_gain = sum(float(row["validation"]["gain_ms"]) for row in results)
        elapsed = [float(row["optimize"]["elapsed_ms"]) for row in results]
        report = {
            "schema_version": 1,
            "algorithm": "fixed-layout-exact-lut-coordinate-descent-v1",
            "input_layout": str(args.input_layout.resolve()),
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
            "layers": args.layers,
            "accepted_moves": sum(int(row["optimize"]["round_count"]) for row in results),
            "optimize_baseline_ms": optimize_baseline,
            "optimize_gain_ms": optimize_gain,
            "optimize_gain_fraction": optimize_gain / max(optimize_baseline, 1e-12),
            "validation_baseline_ms": validation_baseline,
            "validation_gain_ms": validation_gain,
            "validation_gain_fraction": validation_gain / max(validation_baseline, 1e-12),
            "timing_per_layer_ms": {
                "median": statistics.median(elapsed),
                "p90": _percentile(elapsed, 0.9),
                "maximum": max(elapsed),
            },
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "results": results,
        }
        output_payload["source_lut_refinement"] = {key: value for key, value in report.items() if key != "results"}
        args.output_layout.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_layout.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
