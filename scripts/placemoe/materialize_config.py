# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Materialize a static PlaceMoE runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from veomni.distributed.moe.hiermoe.placemoe.runtime import PlaceMoERuntimeConfig


def write_static_config(initial_artifact: Path, output: Path) -> Path:
    """Write a validated config that preserves a fixed PlaceMoE layout."""
    initial_artifact = initial_artifact.expanduser().resolve()
    if not initial_artifact.is_file():
        raise FileNotFoundError(f"PlaceMoE initial artifact does not exist: {initial_artifact}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "placemoe": {
            "initial_artifact": os.path.relpath(initial_artifact, output.parent),
            "hot_update": {"enabled": False},
        }
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        PlaceMoERuntimeConfig.from_file(temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a canonical PlaceMoE config for an existing static layout.",
    )
    parser.add_argument(
        "--initial-artifact",
        type=Path,
        required=True,
        help="Existing PlaceMoE layout artifact.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(write_static_config(args.initial_artifact, args.output))


if __name__ == "__main__":
    main()
