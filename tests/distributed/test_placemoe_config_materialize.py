# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import json
from pathlib import Path

from scripts.placemoe.materialize_config import write_static_config
from veomni.distributed.moe.hiermoe.placemoe.runtime import PlaceMoERuntimeConfig


def test_static_placemoe_config_is_canonical_and_portable(tmp_path: Path) -> None:
    layout = tmp_path / "ours_layout.json"
    layout.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "configs" / "ours_placemoe.json"

    assert write_static_config(layout, output) == output.resolve()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "placemoe": {
            "hot_update": {"enabled": False},
            "initial_artifact": "../ours_layout.json",
        }
    }
    config = PlaceMoERuntimeConfig.from_file(output)
    assert config.initial_artifact == str(layout.resolve())
    assert not config.hot_update.enabled
