#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Merge independently generated HierMoE hierarchical-layout chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-root", type=Path, required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--output-primary", type=Path, required=True)
    parser.add_argument("--output-full", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _merge_replays(paths: list[Path]) -> dict[str, object]:
    payloads = [_load(path) for path in paths]
    merged = dict(payloads[0])
    merged_layers: dict[str, object] = {}
    merged_actions: list[object] = []
    for payload in payloads:
        layers = payload["layers"]
        if not isinstance(layers, dict):
            raise ValueError("Replay layers must be a mapping.")
        overlap = set(merged_layers) & set(layers)
        if overlap:
            raise ValueError(f"Replay chunks overlap on layers: {sorted(overlap)}.")
        merged_layers.update(layers)
        replay = payload["replay"]
        if not isinstance(replay, dict):
            raise ValueError("Replay metadata must be a mapping.")
        actions_by_step = replay["actions_by_step"]
        if not isinstance(actions_by_step, dict):
            raise ValueError("Replay actions_by_step must be a mapping.")
        rows = actions_by_step["1"]
        if not isinstance(rows, list):
            raise ValueError("Replay step 1 actions must be a list.")
        merged_actions.extend(rows)
    merged["layers"] = dict(sorted(merged_layers.items()))
    merged["replay"] = {"actions_by_step": {"1": merged_actions}}
    return merged


def _merge_reports(paths: list[Path]) -> dict[str, object]:
    payloads = [_load(path) for path in paths]
    merged = dict(payloads[0])
    layers: list[object] = []
    validation: list[object] = []
    aggregate: dict[str, float] = {}
    for payload in payloads:
        layer_rows = payload["layers"]
        validation_rows = payload["validation"]
        chunk_aggregate = payload["aggregate"]
        if not isinstance(layer_rows, list) or not isinstance(validation_rows, list):
            raise ValueError("Report rows must be lists.")
        if not isinstance(chunk_aggregate, dict):
            raise ValueError("Report aggregate must be a mapping.")
        layers.extend(layer_rows)
        validation.extend(validation_rows)
        for key, value in chunk_aggregate.items():
            aggregate[str(key)] = aggregate.get(str(key), 0.0) + float(value)
    layers.sort(key=lambda row: int(row["layer"]))
    validation.sort(key=lambda row: int(row["layer"]))
    merged["layers"] = layers
    merged["validation"] = validation
    merged["aggregate"] = aggregate
    configuration = dict(merged["configuration"])
    configuration["layer_start"] = 0
    configuration["layers"] = len(layers)
    configuration["merged_chunks"] = len(paths)
    merged["configuration"] = configuration
    return merged


def main() -> None:
    args = _parse_args()
    indices = list(range(args.parts))
    primary = _merge_replays([args.part_root / f"primary_{index}.json" for index in indices])
    full = _merge_replays([args.part_root / f"full_{index}.json" for index in indices])
    report = _merge_reports([args.part_root / f"report_{index}.json" for index in indices])
    for path, payload in (
        (args.output_primary, primary),
        (args.output_full, full),
        (args.output_report, report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
