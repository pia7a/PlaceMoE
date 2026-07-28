# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Apply exhaustive Forward-LUT Cover winners to a replayable static layout."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-group-size", type=int, default=8)
    parser.add_argument("--minimum-gain-ms", type=float, default=0.0)
    return parser.parse_args()


def _layer_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _validate_state(
    *,
    layout: list[int],
    owners: list[int],
    source_lut: list[list[int]],
    num_experts: int,
) -> None:
    if len(owners) != num_experts:
        raise ValueError("Owner table has the wrong expert count.")
    for expert, slot in enumerate(owners):
        if not 0 <= slot < len(layout) or layout[slot] != expert:
            raise ValueError(f"Owner slot {slot} does not contain expert {expert}.")
    for source_rank, row in enumerate(source_lut):
        if len(row) != num_experts:
            raise ValueError(f"Source LUT row {source_rank} has the wrong expert count.")
        for expert, slot in enumerate(row):
            if not 0 <= slot < len(layout) or layout[slot] != expert:
                raise ValueError(
                    f"Source LUT row {source_rank}, expert {expert} references invalid slot {slot}."
                )


def _apply_cover(
    row: dict[str, object],
    action: dict[str, object],
    *,
    service_group_size: int,
) -> None:
    layout = [int(value) for value in row["slot_to_logical"]]
    owners = [int(value) for value in row["owner_slots"]]
    source_lut = [
        [int(value) for value in source_row]
        for source_row in row["source_logical_to_physical"]
    ]
    num_experts = len(owners)
    ep_size = len(source_lut)
    if ep_size <= 0 or len(layout) % ep_size:
        raise ValueError("Physical layout cannot be divided across source ranks.")
    slots_per_rank = len(layout) // ep_size
    source = int(action["source_logical"])
    source_slot = int(action["source_slot"])
    destination = int(action["destination_slot"])
    victim = int(action["victim_logical"])
    target_rank = destination // slots_per_rank
    if int(action["target_rank"]) != target_rank:
        raise ValueError("Cover action target rank disagrees with its destination slot.")
    if not 0 <= source < num_experts or not 0 <= victim < num_experts:
        raise ValueError("Cover action contains an invalid logical expert.")
    if layout[source_slot] != source or owners[source] != source_slot:
        raise ValueError("Cover source does not match the current canonical owner.")
    if layout[destination] != victim:
        raise ValueError("Cover victim does not match the current destination slot.")
    if sum(value == victim for value in layout) <= 1:
        raise ValueError("Cover would remove the victim's final physical copy.")

    layout[destination] = source
    if owners[victim] == destination:
        owners[victim] = min(slot for slot, expert in enumerate(layout) if expert == victim)
    fallback = owners[victim]
    for source_rank in range(ep_size):
        if source_lut[source_rank][victim] == destination:
            source_lut[source_rank][victim] = fallback
    service_start = (target_rank // service_group_size) * service_group_size
    for source_rank in range(service_start, service_start + service_group_size):
        source_lut[source_rank][source] = destination

    _validate_state(
        layout=layout,
        owners=owners,
        source_lut=source_lut,
        num_experts=num_experts,
    )
    row["slot_to_logical"] = layout
    row["owner_slots"] = owners
    row["source_logical_to_physical"] = source_lut


def main() -> None:
    args = _parse_args()
    if args.service_group_size <= 0:
        raise ValueError("service-group-size must be positive.")
    payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_result.read_text(encoding="utf-8"))
    result = copy.deepcopy(payload)
    layers = result.get("layers")
    replay = result.get("replay")
    if not isinstance(layers, dict) or not isinstance(replay, dict):
        raise ValueError("Input layout must contain layer and replay tables.")
    actions_by_step = replay.get("actions_by_step")
    if not isinstance(actions_by_step, dict):
        raise ValueError("Input layout replay has no actions_by_step table.")
    step_actions = actions_by_step.setdefault("1", [])
    if not isinstance(step_actions, list):
        raise ValueError("Input layout step 1 actions must be a list.")

    accepted: list[dict[str, object]] = []
    for result_row in oracle.get("results", []):
        if not isinstance(result_row, dict):
            continue
        optimize = result_row.get("optimize")
        action = result_row.get("action")
        if (
            not bool(result_row.get("accepted"))
            or not isinstance(optimize, dict)
            or float(optimize["gain_ms"]) <= float(args.minimum_gain_ms)
            or not isinstance(action, dict)
        ):
            continue
        layer = int(result_row["layer"])
        name = _layer_name(layer)
        layer_row = layers.get(name)
        if not isinstance(layer_row, dict):
            raise ValueError(f"Input layout has no state for layer {layer}.")
        _apply_cover(
            layer_row,
            action,
            service_group_size=int(args.service_group_size),
        )
        replay_action = {
            "body": f"{int(action['source_logical'])}->{int(action['destination_slot'])}",
            "kind": "replica",
            "layer": name,
        }
        step_actions.append(replay_action)
        accepted.append(
            {
                "layer": layer,
                "gain_ms": float(optimize["gain_ms"]),
                "validation_gain_ms": float(result_row["validation"]["gain_ms"]),
                **{key: int(value) for key, value in action.items()},
            }
        )

    source = result.setdefault("source", {})
    if not isinstance(source, dict):
        raise ValueError("Input layout source metadata must be a table.")
    source.update(
        {
            "algorithm": "forward-lut-exhaustive-cover-static-v1",
            "input_layout": str(args.input_layout.resolve()),
            "oracle_result": str(args.oracle_result.resolve()),
            "service_group_size": int(args.service_group_size),
            "minimum_gain_ms": float(args.minimum_gain_ms),
            "accepted_covers": len(accepted),
            "predicted_gain_ms": sum(float(row["gain_ms"]) for row in accepted),
            "heldout_gain_ms": sum(float(row["validation_gain_ms"]) for row in accepted),
            "covers": accepted,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted_covers": len(accepted),
                "predicted_gain_ms": source["predicted_gain_ms"],
                "heldout_gain_ms": source["heldout_gain_ms"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
