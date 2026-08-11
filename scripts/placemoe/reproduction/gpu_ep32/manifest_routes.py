#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Validate and fingerprint a complete HierMoE route capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item.strip())
    if not steps or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must be a non-empty list of unique integers")
    return steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_route_manifest(
    root: Path,
    *,
    steps: tuple[int, ...],
    layer_start: int,
    layers: int,
    ep_size: int,
) -> dict[str, Any]:
    root = root.resolve()
    expected = {
        f"step{step:04d}/layer{layer:02d}_call0_rank{rank:02d}.pt"
        for step in steps
        for layer in range(layer_start, layer_start + layers)
        for rank in range(ep_size)
    }
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.pt")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Incomplete or contaminated route capture: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )
    digest = hashlib.sha256()
    for relative in sorted(expected):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(root / relative).encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "route_root": str(root),
        "steps": list(steps),
        "layer_start": layer_start,
        "layers": layers,
        "ep_size": ep_size,
        "call_index": 0,
        "files": len(expected),
        "sha256": digest.hexdigest(),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--steps", type=_parse_steps, required=True)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    manifest = build_route_manifest(
        args.route_root,
        steps=args.steps,
        layer_start=args.layer_start,
        layers=args.layers,
        ep_size=args.ep_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(manifest["sha256"])


if __name__ == "__main__":
    main()
