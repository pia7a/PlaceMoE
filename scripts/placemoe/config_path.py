#!/usr/bin/env python3
"""Resolve one path field from a PlaceMoE config without importing VeOmni."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _load(path: Path) -> Mapping[str, Any]:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json"
        else yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, Mapping):
        raise ValueError("PlaceMoE config must be a mapping.")
    root = payload.get("placemoe", payload)
    if not isinstance(root, Mapping):
        raise ValueError("placemoe must be a mapping.")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("field", choices=("initial_artifact", "calibration_artifact"))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    root = _load(config_path)
    if args.field == "initial_artifact":
        raw = root.get("initial_artifact", "")
    else:
        calibration = root.get("calibration", {})
        if not isinstance(calibration, Mapping):
            raise ValueError("calibration must be a mapping.")
        raw = calibration.get("artifact", "")
    value = os.path.expandvars(os.path.expanduser(str(raw).strip()))
    if not value:
        raise ValueError(f"{args.field} is not configured.")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    print(path.resolve())


if __name__ == "__main__":
    main()
