#!/usr/bin/env python3
"""Reconstruct a committed fixed-pipeline layout from rank-0 action logs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


_METRICS_STEP = re.compile(r"HierMoE metrics step=(?P<step>\d+) ")
_ACTIONS = re.compile(r"hiermoe/expert_swap_pair=(?P<actions>.*?) hiermoe/expert_swap_selector=")
_MIGRATION_JOBS = re.compile(r"hiermoe/pipeline_migration_jobs=(?P<jobs>\d+)")
_MIGRATION_STALE = re.compile(r"hiermoe/pipeline_migration_stale=(?P<stale>\d+)")
_ACTION = re.compile(r"(?P<layer>[^,]+):(?P<kind>swap|replica|empty)\((?P<body>[^)]*)\)")


@dataclass(frozen=True)
class StepRecord:
    step: int
    actions: tuple[tuple[str, str, str], ...]
    migration_jobs: int
    migration_stale: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--layer-count", type=int, default=48)
    parser.add_argument(
        "--layer-template",
        default="model.language_model.layers.{layer}.mlp.experts",
    )
    parser.add_argument(
        "--through-action-step",
        type=int,
        default=None,
        help="Last selected-action step to replay. By default infer it from next-step migration completion.",
    )
    parser.add_argument(
        "--clear-fixed-r2-redundant-slots",
        action="store_true",
        help=(
            "Replay an online-freeze seed: start from fixed R2, clear every non-owner "
            "R2 slot, then apply the logged replica actions."
        ),
    )
    parser.add_argument(
        "--canonical-empty-seed",
        action="store_true",
        help=(
            "Start from the canonical one-owner-per-expert layout with every "
            "reserved redundant slot empty."
        ),
    )
    return parser.parse_args()


def _read_records(path: Path) -> dict[int, StepRecord]:
    records: dict[int, StepRecord] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step_match = _METRICS_STEP.search(line)
            if step_match is None:
                continue
            step = int(step_match.group("step"))
            actions_match = _ACTIONS.search(line)
            action_text = "none" if actions_match is None else actions_match.group("actions")
            actions = tuple(
                (match.group("layer"), match.group("kind"), match.group("body"))
                for match in _ACTION.finditer(action_text)
            )
            jobs_match = _MIGRATION_JOBS.search(line)
            stale_match = _MIGRATION_STALE.search(line)
            records[step] = StepRecord(
                step=step,
                actions=actions,
                migration_jobs=0 if jobs_match is None else int(jobs_match.group("jobs")),
                migration_stale=0 if stale_match is None else int(stale_match.group("stale")),
            )
    if not records:
        raise ValueError(f"No HierMoE metrics records found in {path}.")
    return records


def _infer_committed_action_step(records: dict[int, StepRecord]) -> int:
    committed: list[int] = []
    for action_step, record in records.items():
        if not record.actions:
            continue
        next_record = records.get(action_step + 1)
        layer_count = len({layer for layer, _kind, _body in record.actions})
        if next_record is not None and next_record.migration_jobs >= layer_count and next_record.migration_stale == 0:
            committed.append(action_step)
    if not committed:
        raise ValueError("No action step has a complete, non-stale next-step migration record.")
    return max(committed)


def _fixed_r2_layout(ep_size: int, num_experts: int, slots_per_rank: int) -> tuple[list[int], list[int]]:
    if ep_size <= 1 or ep_size % 2:
        raise ValueError(f"Fixed R2 requires a positive even EP size, got {ep_size}.")
    expected_slots_per_rank = num_experts // (ep_size // 2)
    if expected_slots_per_rank != slots_per_rank:
        raise ValueError(
            f"Fixed R2 expects {expected_slots_per_rank} slots/rank for EP={ep_size}, "
            f"experts={num_experts}; got {slots_per_rank}."
        )
    physical_slots = ep_size * slots_per_rank
    layout = [-1] * physical_slots
    owners = [-1] * num_experts
    for logical in range(num_experts):
        rank_in_half, local_slot = divmod(logical, slots_per_rank)
        first_slot = rank_in_half * slots_per_rank + local_slot
        second_slot = (ep_size // 2 + rank_in_half) * slots_per_rank + local_slot
        layout[first_slot] = logical
        layout[second_slot] = logical
        owners[logical] = first_slot
    return layout, owners


def _canonical_empty_layout(
    ep_size: int,
    num_experts: int,
    slots_per_rank: int,
) -> tuple[list[int], list[int]]:
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError(f"Canonical layout requires num_experts divisible by ep_size, got {num_experts}/{ep_size}.")
    base_slots_per_rank = num_experts // ep_size
    if slots_per_rank < base_slots_per_rank:
        raise ValueError(
            f"Canonical layout needs at least {base_slots_per_rank} slots/rank, got {slots_per_rank}."
        )
    layout = [-1] * (ep_size * slots_per_rank)
    owners = [-1] * num_experts
    for logical in range(num_experts):
        rank, local_slot = divmod(logical, base_slots_per_rank)
        physical_slot = rank * slots_per_rank + local_slot
        layout[physical_slot] = logical
        owners[logical] = physical_slot
    return layout, owners


def _replay_action(layout: list[int], owners: list[int], kind: str, body: str) -> None:
    if kind == "swap":
        lhs_text, rhs_text = body.split("<->", maxsplit=1)
        lhs, rhs = int(lhs_text), int(rhs_text)
        lhs_slot, rhs_slot = owners[lhs], owners[rhs]
        if layout[lhs_slot] != lhs or layout[rhs_slot] != rhs:
            raise ValueError(
                f"Swap {body} does not match owner slots {lhs_slot}:{layout[lhs_slot]}, {rhs_slot}:{layout[rhs_slot]}."
            )
        layout[lhs_slot], layout[rhs_slot] = layout[rhs_slot], layout[lhs_slot]
        owners[lhs], owners[rhs] = rhs_slot, lhs_slot
        return
    if kind == "replica":
        logical_text, slot_text = body.split("->", maxsplit=1)
        logical, dst_slot = int(logical_text), int(slot_text)
        if not 0 <= dst_slot < len(layout):
            raise ValueError(f"Replica destination slot is out of range: {body}.")
        if layout[owners[logical]] != logical:
            raise ValueError(f"Replica source owner is invalid for expert {logical}.")
        victim = layout[dst_slot]
        layout[dst_slot] = logical
        if victim >= 0 and owners[victim] == dst_slot:
            remaining = [slot for slot, value in enumerate(layout) if value == victim]
            if not remaining:
                raise ValueError(f"Replica {body} removes the final copy of victim expert {victim}.")
            owners[victim] = min(remaining)
        return
    logical_text, slot_text = body.split("@", maxsplit=1)
    logical, slot = int(logical_text), int(slot_text)
    if layout[slot] != logical:
        raise ValueError(f"Empty action does not match layout: {body}.")
    layout[slot] = -1


def main() -> None:
    args = _parse_args()
    if args.clear_fixed_r2_redundant_slots and args.canonical_empty_seed:
        raise ValueError("--clear-fixed-r2-redundant-slots and --canonical-empty-seed are mutually exclusive.")
    records = _read_records(args.log)
    through_step = (
        _infer_committed_action_step(records) if args.through_action_step is None else int(args.through_action_step)
    )
    action_steps = sorted(step for step, record in records.items() if record.actions and step <= through_step)
    if not action_steps or action_steps[-1] != through_step:
        raise ValueError(f"No actions found for requested through-action-step={through_step}.")

    logged_layers = {layer for step in action_steps for layer, _kind, _body in records[step].actions}
    configured_layers = {
        args.layer_template.format(layer=layer)
        for layer in range(args.layer_count)
    }
    layer_names = sorted(logged_layers | configured_layers)
    layouts: dict[str, list[int]] = {}
    owners_by_layer: dict[str, list[int]] = {}
    action_counts: dict[str, int] = {}
    for layer in layer_names:
        if args.canonical_empty_seed:
            layout, owners = _canonical_empty_layout(args.ep_size, args.num_experts, args.slots_per_rank)
        else:
            layout, owners = _fixed_r2_layout(args.ep_size, args.num_experts, args.slots_per_rank)
        layouts[layer] = layout
        owners_by_layer[layer] = owners
        action_counts[layer] = 0

    prefix_actions: list[dict[str, str]] = []
    if args.clear_fixed_r2_redundant_slots:
        for layer in layer_names:
            layout = layouts[layer]
            owners = set(owners_by_layer[layer])
            for slot, logical in enumerate(layout):
                if slot in owners:
                    continue
                if logical < 0:
                    raise ValueError(f"Fixed R2 redundant slot {slot} is unexpectedly empty for {layer}.")
                body = f"{logical}@{slot}"
                _replay_action(layout, owners_by_layer[layer], "empty", body)
                prefix_actions.append({"layer": layer, "kind": "empty", "body": body})
                action_counts[layer] += 1

    for step in action_steps:
        for layer, kind, body in records[step].actions:
            _replay_action(layouts[layer], owners_by_layer[layer], kind, body)
            action_counts[layer] += 1

    for layer in layer_names:
        layout = layouts[layer]
        owners = owners_by_layer[layer]
        if any(owner < 0 or layout[owner] != logical for logical, owner in enumerate(owners)):
            raise ValueError(f"Reconstructed owner mapping is invalid for {layer}.")
        if set(range(args.num_experts)) - set(layout):
            raise ValueError(f"Reconstructed layout loses at least one logical expert in {layer}.")

    pending_steps = sorted(step for step, record in records.items() if record.actions and step > through_step)
    actions_by_step = {
        str(step): [
            {
                "layer": layer,
                "kind": kind,
                "body": body,
            }
            for layer, kind, body in records[step].actions
        ]
        for step in action_steps
    }
    if prefix_actions:
        prefix_step = min(action_steps) - 1
        actions_by_step[str(prefix_step)] = prefix_actions
    output = {
        "schema_version": 1,
        "source": {
            "rank0_log": str(args.log.resolve()),
            "initial_layout": (
                "fixed_r2_then_clear_redundant"
                if args.clear_fixed_r2_redundant_slots
                else ("canonical_empty" if args.canonical_empty_seed else "fixed_r2")
            ),
            "committed_action_steps": action_steps,
            "excluded_pending_action_steps": pending_steps,
            "commit_inference": "action step S is committed only when step S+1 reports all layer migrations and no stale migration",
        },
        "topology": {
            "ep_size": args.ep_size,
            "num_experts": args.num_experts,
            "slots_per_rank": args.slots_per_rank,
            "num_physical_slots": args.ep_size * args.slots_per_rank,
        },
        "replay": {
            "actions_by_step": actions_by_step,
        },
        "layers": {
            layer: {
                "slot_to_logical": layouts[layer],
                "logical_owner_slots": owners_by_layer[layer],
                "replayed_action_count": action_counts[layer],
            }
            for layer in layer_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} with {len(layer_names)} layers; "
        f"committed action steps={action_steps}; excluded pending steps={pending_steps}"
    )


if __name__ == "__main__":
    main()
