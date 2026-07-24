# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Measure how often the previous route selects the current exact action."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
import zlib
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.planner import PlacementAction, PlacementPlan
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--perf-model-path", type=Path)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _initialize(backend: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "hccl":
        importlib.import_module("torch_npu")
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend=backend)
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, device


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _load_routes(
    route_dir: Path,
    *,
    steps: int,
    layers: int,
    rank: int,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], dict]:
    routes_by_step: list[list[torch.Tensor]] = []
    first_metadata = None
    for step in range(steps):
        step_routes = []
        for layer in range(layers):
            path = route_dir / f"step{step:04d}" / f"layer{layer:02d}_call0_rank{rank:02d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("format") != "veomni.hiermoe.local_route":
                raise ValueError(f"Unsupported local route snapshot: {path}")
            expected = (step, layer, rank)
            actual = (
                int(payload["step"]),
                int(payload["layer"]),
                int(payload["global_rank"]),
            )
            if actual != expected:
                raise ValueError(f"Route metadata mismatch in {path}: expected={expected}, actual={actual}")
            route = payload["routes"].to(dtype=torch.long)
            if route.ndim != 2:
                raise ValueError(f"Expected rank-2 routes in {path}, got {tuple(route.shape)}")
            sorted_route = route.sort(dim=-1).values
            if route.shape[1] > 1 and bool((sorted_route[:, 1:] == sorted_route[:, :-1]).any().item()):
                raise ValueError(f"Captured gate top-k contains duplicate logical experts: {path}")
            step_routes.append(route.to(device=device, non_blocking=True).contiguous())
            if first_metadata is None:
                first_metadata = payload
            elif (
                int(payload["num_experts"]) != int(first_metadata["num_experts"])
                or int(payload["hidden_size"]) != int(first_metadata["hidden_size"])
                or int(payload["bytes_per_element"]) != int(first_metadata["bytes_per_element"])
            ):
                raise ValueError(f"Inconsistent route metadata in {path}")
        routes_by_step.append(step_routes)
    assert first_metadata is not None
    return routes_by_step, first_metadata


def _fixed_r2_layout(
    *,
    num_experts: int,
    ep_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if ep_size % 2 or num_experts % (ep_size // 2):
        raise ValueError(f"Fixed R2 requires even EP and compatible experts, got EP={ep_size}, E={num_experts}")
    slots_per_rank = num_experts // (ep_size // 2)
    logical = torch.arange(num_experts, dtype=torch.long, device=device)
    rank_in_half = torch.div(logical, slots_per_rank, rounding_mode="floor")
    local_slot = torch.remainder(logical, slots_per_rank)
    owners = rank_in_half * slots_per_rank + local_slot
    replicas = (ep_size // 2 + rank_in_half) * slots_per_rank + local_slot
    layout = torch.full((ep_size * slots_per_rank,), -1, dtype=torch.long, device=device)
    layout.scatter_(0, owners, logical)
    layout.scatter_(0, replicas, logical)
    return layout, owners, slots_per_rank


def _action_key(actions: tuple[PlacementAction, ...]) -> tuple[tuple[str, int, int, int, int], ...]:
    return tuple(
        (action.kind, action.src_slot, action.dst_slot, action.src_logical, action.dst_logical) for action in actions
    )


def _plans_digest(plans: list[PlacementPlan]) -> int:
    payload = [
        {
            "actions": _action_key(plan.actions),
            "layout": plan.final_layout,
            "owners": plan.final_owner_slots,
        }
        for plan in plans
    ]
    digest = hashlib.sha256(repr(payload).encode()).hexdigest()
    return int(digest[:15], 16)


def _assert_same_plans(plans: list[PlacementPlan], *, device: torch.device, world_size: int) -> None:
    if not dist.is_initialized():
        return
    local_digest = torch.tensor([_plans_digest(plans)], dtype=torch.int64, device=device)
    gathered = torch.empty((world_size,), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local_digest)
    if bool((gathered != gathered[0]).any().item()):
        raise RuntimeError("EP ranks selected different placement plans.")


def _timed_plan_layers(
    planner: GreedyCommunicationPlanner,
    routes: list[torch.Tensor],
    layouts: list[torch.Tensor],
    owners: list[torch.Tensor],
    *,
    rank: int,
    step: int,
    layer_keys: list[str],
    device: torch.device,
    world_size: int,
    max_swaps: int = 1,
    max_replicas: int = 1,
) -> tuple[list[PlacementPlan], float]:
    if dist.is_initialized():
        dist.barrier()
    _synchronize(device)
    started = time.perf_counter()
    plans = [
        planner.plan(
            route,
            layout,
            owner,
            source_ranks=rank,
            max_swaps=max_swaps,
            max_replicas=max_replicas,
            step=step,
            layer_seed=zlib.crc32(layer_key.encode("utf-8")),
        )
        for route, layout, owner, layer_key in zip(routes, layouts, owners, layer_keys, strict=True)
    ]
    _synchronize(device)
    elapsed = torch.tensor([(time.perf_counter() - started) * 1000.0], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    _assert_same_plans(plans, device=device, world_size=world_size)
    return plans, float(elapsed.item())


def _evaluate_predicted_cost(
    planner: GreedyCommunicationPlanner,
    route: torch.Tensor,
    predicted: PlacementPlan,
    *,
    rank: int,
    step: int,
    layer_key: str,
    device: torch.device,
) -> tuple[float, float]:
    layout = torch.tensor(predicted.final_layout, dtype=torch.long, device=device)
    owners = torch.tensor(predicted.final_owner_slots, dtype=torch.long, device=device)
    _synchronize(device)
    started = time.perf_counter()
    evaluated = planner.plan(
        route,
        layout,
        owners,
        source_ranks=rank,
        max_swaps=0,
        max_replicas=0,
        step=step,
        layer_seed=zlib.crc32(layer_key.encode("utf-8")),
    )
    _synchronize(device)
    return evaluated.baseline_cost.communication, (time.perf_counter() - started) * 1000.0


def _communication_summary(
    baseline: float,
    historical_route: float,
    current_route: float,
) -> dict[str, float]:
    available_gain = baseline - current_route
    return {
        "baseline_cost_ms": baseline,
        "historical_route_cost_ms": historical_route,
        "current_route_cost_ms": current_route,
        "historical_route_speedup": baseline / historical_route,
        "current_route_speedup": baseline / current_route,
        "current_over_historical_speedup": historical_route / current_route,
        "historical_excess_cost_over_current": (historical_route - current_route) / current_route,
        "historical_optimal_gain_capture": (
            (baseline - historical_route) / available_gain if available_gain > 0.0 else 1.0
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
    if args.layers <= 0:
        raise ValueError("--layers must be positive.")
    rank, world_size, device = _initialize(args.backend)
    if world_size != args.ep_size:
        raise ValueError(f"Distributed world size {world_size} must equal ep_size {args.ep_size}.")

    routes_by_step, metadata = _load_routes(
        args.route_dir,
        steps=args.steps,
        layers=args.layers,
        rank=rank,
        device=device,
    )
    num_experts = int(metadata["num_experts"])
    initial_layout, initial_owners, slots_per_rank = _fixed_r2_layout(
        num_experts=num_experts,
        ep_size=args.ep_size,
        device=device,
    )
    layouts = [initial_layout.clone() for _ in range(args.layers)]
    owners = [initial_owners.clone() for _ in range(args.layers)]
    layer_keys = [
        str(
            torch.load(
                args.route_dir / "step0000" / f"layer{layer:02d}_call0_rank{rank:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )["layer_key"]
        )
        for layer in range(args.layers)
    ]

    def reducer(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    perf_model = HierMoEPerfModel.from_path(str(args.perf_model_path) if args.perf_model_path is not None else None)
    planner = GreedyCommunicationPlanner(
        hierarchy=Hierarchy(
            ep_size=args.ep_size,
            group_sizes=tuple(args.group_sizes),
            source="history-reuse",
            local_world_size=args.local_world_size,
        ),
        perf_model=perf_model,
        hidden_size=int(metadata["hidden_size"]),
        bytes_per_element=int(metadata["bytes_per_element"]),
        slots_per_rank=slots_per_rank,
        reducer=reducer,
        process_group=dist.group.WORLD,
        max_copies=args.max_copies,
        candidate_scorer="statistics",
        compact_candidate_collective=False,
        assume_unique_routes=True,
    )

    exact_step0, step0_ms = _timed_plan_layers(
        planner,
        routes_by_step[0],
        layouts,
        owners,
        rank=rank,
        step=0,
        layer_keys=layer_keys,
        device=device,
        world_size=world_size,
    )
    layouts = [torch.tensor(plan.final_layout, dtype=torch.long, device=device) for plan in exact_step0]
    owners = [torch.tensor(plan.final_owner_slots, dtype=torch.long, device=device) for plan in exact_step0]
    history_layouts = [layout.clone() for layout in layouts]
    history_owners = [owner.clone() for owner in owners]
    fixed_layouts = [initial_layout.clone() for _ in range(args.layers)]
    fixed_owners = [initial_owners.clone() for _ in range(args.layers)]

    pair_results = []
    total_action_matches = 0
    total_cost_optimal = 0
    total_predicted_improvements = 0
    total_current_actions = 0
    total_comparisons = 0
    prediction_ms_total = 0.0
    current_exact_ms_total = step0_ms
    mismatch_evaluation_ms_total = 0.0
    maximum_relative_regret = 0.0
    relative_regret_sum = 0.0
    total_baseline_communication = 0.0
    total_historical_communication = 0.0
    total_current_communication = 0.0
    rollout_fixed_r2_communication = 0.0
    rollout_historical_communication = 0.0
    rollout_current_communication = 0.0

    for step in range(1, args.steps):
        predicted, prediction_ms = _timed_plan_layers(
            planner,
            routes_by_step[step - 1],
            layouts,
            owners,
            rank=rank,
            step=step,
            layer_keys=layer_keys,
            device=device,
            world_size=world_size,
        )
        current, current_ms = _timed_plan_layers(
            planner,
            routes_by_step[step],
            layouts,
            owners,
            rank=rank,
            step=step,
            layer_keys=layer_keys,
            device=device,
            world_size=world_size,
        )
        if step == 1:
            history = predicted
        else:
            history, _ = _timed_plan_layers(
                planner,
                routes_by_step[step - 1],
                history_layouts,
                history_owners,
                rank=rank,
                step=step,
                layer_keys=layer_keys,
                device=device,
                world_size=world_size,
            )
        history_final_layouts = [torch.tensor(plan.final_layout, dtype=torch.long, device=device) for plan in history]
        history_final_owners = [
            torch.tensor(plan.final_owner_slots, dtype=torch.long, device=device) for plan in history
        ]
        history_evaluated, _ = _timed_plan_layers(
            planner,
            routes_by_step[step],
            history_final_layouts,
            history_final_owners,
            rank=rank,
            step=step,
            layer_keys=layer_keys,
            device=device,
            world_size=world_size,
            max_swaps=0,
            max_replicas=0,
        )
        fixed_r2_evaluated, _ = _timed_plan_layers(
            planner,
            routes_by_step[step],
            fixed_layouts,
            fixed_owners,
            rank=rank,
            step=step,
            layer_keys=layer_keys,
            device=device,
            world_size=world_size,
            max_swaps=0,
            max_replicas=0,
        )
        prediction_ms_total += prediction_ms
        current_exact_ms_total += current_ms

        action_matches = 0
        cost_optimal = 0
        predicted_improvements = 0
        current_actions = 0
        step_regret_sum = 0.0
        step_regret_max = 0.0
        step_baseline_communication = 0.0
        step_historical_communication = 0.0
        step_current_communication = 0.0
        step_rollout_fixed_r2_communication = 0.0
        step_rollout_historical_communication = 0.0
        step_rollout_current_communication = 0.0
        for layer, (predicted_plan, current_plan) in enumerate(zip(predicted, current, strict=True)):
            predicted_key = _action_key(predicted_plan.actions)
            current_key = _action_key(current_plan.actions)
            action_match = predicted_key == current_key
            action_matches += int(action_match)
            current_actions += int(bool(current_plan.actions))

            if action_match:
                predicted_current_cost = current_plan.final_cost.communication
            elif not predicted_plan.actions:
                predicted_current_cost = current_plan.baseline_cost.communication
            else:
                predicted_current_cost, evaluation_ms = _evaluate_predicted_cost(
                    planner,
                    routes_by_step[step][layer],
                    predicted_plan,
                    rank=rank,
                    step=step,
                    layer_key=layer_keys[layer],
                    device=device,
                )
                mismatch_evaluation_ms_total += evaluation_ms

            baseline = current_plan.baseline_cost.communication
            optimum = current_plan.final_cost.communication
            step_baseline_communication += baseline
            step_historical_communication += predicted_current_cost
            step_current_communication += optimum
            step_rollout_fixed_r2_communication += fixed_r2_evaluated[layer].baseline_cost.communication
            step_rollout_historical_communication += history_evaluated[layer].baseline_cost.communication
            step_rollout_current_communication += current_plan.final_cost.communication
            tolerance = max(1.0e-9, abs(optimum) * 1.0e-7)
            is_cost_optimal = predicted_current_cost <= optimum + tolerance
            is_improvement = predicted_current_cost < baseline
            cost_optimal += int(is_cost_optimal)
            predicted_improvements += int(is_improvement)
            relative_regret = max(0.0, predicted_current_cost - optimum) / max(abs(baseline), 1.0e-12)
            step_regret_sum += relative_regret
            step_regret_max = max(step_regret_max, relative_regret)

        total_action_matches += action_matches
        total_cost_optimal += cost_optimal
        total_predicted_improvements += predicted_improvements
        total_current_actions += current_actions
        total_comparisons += args.layers
        relative_regret_sum += step_regret_sum
        maximum_relative_regret = max(maximum_relative_regret, step_regret_max)
        total_baseline_communication += step_baseline_communication
        total_historical_communication += step_historical_communication
        total_current_communication += step_current_communication
        rollout_fixed_r2_communication += step_rollout_fixed_r2_communication
        rollout_historical_communication += step_rollout_historical_communication
        rollout_current_communication += step_rollout_current_communication
        pair_results.append(
            {
                "previous_step": step - 1,
                "current_step": step,
                "layers": args.layers,
                "action_matches": action_matches,
                "action_match_rate": action_matches / args.layers,
                "cost_optimal": cost_optimal,
                "cost_optimal_rate": cost_optimal / args.layers,
                "predicted_strict_improvements": predicted_improvements,
                "current_exact_actions": current_actions,
                "mean_relative_regret": step_regret_sum / args.layers,
                "max_relative_regret": step_regret_max,
                "prediction_ms_48_layers": prediction_ms,
                "current_exact_ms_48_layers": current_ms,
                "modeled_a2a": _communication_summary(
                    step_baseline_communication,
                    step_historical_communication,
                    step_current_communication,
                ),
                "rollout_modeled_a2a": _communication_summary(
                    step_rollout_fixed_r2_communication,
                    step_rollout_historical_communication,
                    step_rollout_current_communication,
                ),
            }
        )

        history_layouts = history_final_layouts
        history_owners = history_final_owners
        layouts = [torch.tensor(plan.final_layout, dtype=torch.long, device=device) for plan in current]
        owners = [torch.tensor(plan.final_owner_slots, dtype=torch.long, device=device) for plan in current]

    result = {
        "route_dir": str(args.route_dir),
        "steps": args.steps,
        "layers": args.layers,
        "ep_size": args.ep_size,
        "group_sizes": list(args.group_sizes),
        "perf_model_source": perf_model.source,
        "initial_layout": "fixed_r2",
        "comparisons": total_comparisons,
        "action_matches": total_action_matches,
        "action_match_rate": total_action_matches / total_comparisons,
        "cost_optimal": total_cost_optimal,
        "cost_optimal_rate": total_cost_optimal / total_comparisons,
        "predicted_strict_improvements": total_predicted_improvements,
        "current_exact_actions": total_current_actions,
        "mean_relative_regret": relative_regret_sum / total_comparisons,
        "max_relative_regret": maximum_relative_regret,
        "modeled_a2a": _communication_summary(
            total_baseline_communication,
            total_historical_communication,
            total_current_communication,
        ),
        "rollout_modeled_a2a": _communication_summary(
            rollout_fixed_r2_communication,
            rollout_historical_communication,
            rollout_current_communication,
        ),
        "timing_ms": {
            "exact_step0_48_layers": step0_ms,
            "prediction_total": prediction_ms_total,
            "prediction_mean_48_layers": prediction_ms_total / (args.steps - 1),
            "current_exact_total_including_step0": current_exact_ms_total,
            "mismatch_action_evaluation_total": mismatch_evaluation_ms_total,
        },
        "pairs": pair_results,
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
