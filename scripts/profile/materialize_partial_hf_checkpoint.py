#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Materialize an independent Hugging Face checkpoint with fewer text layers.

The tool is intentionally conservative:

* the source and destination must be different paths;
* an existing destination is never overwritten;
* weights are written to a staging directory and published atomically;
* non-text-layer tensors (vision tower, embeddings, final norm, LM head, MTP)
  are preserved;
* ``config.json`` and the safetensors index are regenerated and validated;
* a manifest records the exact source, layer selection and tensor inventory.

Qwen3.5 multimodal checkpoints store decoder layers under
``model.language_model.layers`` and keep the decoder configuration under
``text_config``.  Older text checkpoints using ``model.layers`` are also
supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


LAYER_PATTERNS = (
    re.compile(r"^model\.language_model\.layers\.(\d+)\."),
    re.compile(r"^model\.layers\.(\d+)\."),
)
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "partial_checkpoint_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--num-hidden-layers", type=int, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned tensor selection without writing files.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_index(key: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = pattern.match(key)
        if match is not None:
            return int(match.group(1))
    return None


def _update_config(config: dict[str, Any], target_layers: int) -> tuple[dict[str, Any], int]:
    text_config = config.get("text_config")
    target = text_config if isinstance(text_config, dict) else config
    source_layers = target.get("num_hidden_layers")
    if not isinstance(source_layers, int):
        raise ValueError("config.json does not contain an integer num_hidden_layers.")
    if not 0 < target_layers <= source_layers:
        raise ValueError(
            f"num-hidden-layers must be in [1, {source_layers}], got {target_layers}."
        )

    target["num_hidden_layers"] = target_layers
    layer_types = target.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != source_layers:
            raise ValueError(
                "text layer_types must be a list with one entry per source hidden layer."
            )
        target["layer_types"] = layer_types[:target_layers]

    mlp_only_layers = target.get("mlp_only_layers")
    if isinstance(mlp_only_layers, list):
        target["mlp_only_layers"] = [
            layer for layer in mlp_only_layers if int(layer) < target_layers
        ]
    return config, source_layers


def _load_plan(
    source: Path,
    target_layers: int,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, list[str]],
    list[str],
    list[str],
    int,
]:
    config_path = source / CONFIG_NAME
    index_path = source / INDEX_NAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config, source_layers = _update_config(config, target_layers)

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    source_weight_map = index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not source_weight_map:
        raise ValueError(f"{index_path} has no non-empty weight_map.")

    layer_ids = {
        layer_id
        for key in source_weight_map
        if (layer_id := _layer_index(key)) is not None
    }
    expected_ids = set(range(source_layers))
    if layer_ids != expected_ids:
        missing = sorted(expected_ids - layer_ids)
        extra = sorted(layer_ids - expected_ids)
        raise ValueError(
            "checkpoint text-layer IDs do not match config: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    kept_keys = sorted(
        key
        for key in source_weight_map
        if (layer_id := _layer_index(key)) is None or layer_id < target_layers
    )
    dropped_keys = sorted(set(source_weight_map) - set(kept_keys))
    if not dropped_keys and target_layers < source_layers:
        raise ValueError("No text-layer tensors would be dropped; layer prefix detection failed.")

    keys_by_source_shard: dict[str, list[str]] = defaultdict(list)
    for key in kept_keys:
        shard = source_weight_map[key]
        if not isinstance(shard, str):
            raise ValueError(f"Invalid shard name for tensor {key!r}: {shard!r}")
        keys_by_source_shard[shard].append(key)
    return (
        config,
        source_weight_map,
        dict(keys_by_source_shard),
        kept_keys,
        dropped_keys,
        source_layers,
    )


def _copy_assets(source: Path, staging: Path) -> list[str]:
    copied: list[str] = []
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        if entry.name == CONFIG_NAME or entry.name == INDEX_NAME:
            continue
        if entry.suffix == ".safetensors":
            continue
        shutil.copy2(entry, staging / entry.name)
        copied.append(entry.name)
    return sorted(copied)


def _validate_output(
    output: Path,
    *,
    target_layers: int,
    expected_keys: set[str],
) -> None:
    with (output / CONFIG_NAME).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config")
    target = text_config if isinstance(text_config, dict) else config
    if target.get("num_hidden_layers") != target_layers:
        raise ValueError("Output config num_hidden_layers does not match requested target.")
    if isinstance(target.get("layer_types"), list) and len(target["layer_types"]) != target_layers:
        raise ValueError("Output config layer_types was not truncated consistently.")

    with (output / INDEX_NAME).open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    output_weight_map = index.get("weight_map", {})
    if set(output_weight_map) != expected_keys:
        raise ValueError("Output safetensors index key set does not match the selection plan.")

    observed_keys: set[str] = set()
    for shard_name in sorted(set(output_weight_map.values())):
        shard_path = output / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            observed_keys.update(handle.keys())
    if observed_keys != expected_keys:
        raise ValueError("Output shard tensor keys do not match the regenerated index.")
    for key in observed_keys:
        layer_id = _layer_index(key)
        if layer_id is not None and layer_id >= target_layers:
            raise ValueError(f"Output unexpectedly retains dropped tensor {key!r}.")


def _materialize(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must be different paths.")
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists and will not be overwritten: {destination}"
        )
    if args.num_hidden_layers <= 0:
        raise ValueError("--num-hidden-layers must be positive.")

    (
        config,
        source_weight_map,
        keys_by_source_shard,
        kept_keys,
        dropped_keys,
        source_layers,
    ) = _load_plan(source, args.num_hidden_layers)
    source_shards = sorted(keys_by_source_shard)
    summary = {
        "source": str(source),
        "destination": str(destination),
        "source_hidden_layers": source_layers,
        "target_hidden_layers": args.num_hidden_layers,
        "source_tensor_count": len(source_weight_map),
        "kept_tensor_count": len(kept_keys),
        "dropped_tensor_count": len(dropped_keys),
        "source_shard_count": len(set(source_weight_map.values())),
        "output_shard_count": len(source_shards),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        copied_assets = _copy_assets(source, staging)
        output_weight_map: dict[str, str] = {}
        output_shards: list[dict[str, Any]] = []
        total_size = 0
        shard_count = len(source_shards)
        for output_index, source_shard in enumerate(source_shards, start=1):
            source_path = source / source_shard
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            requested_keys = keys_by_source_shard[source_shard]
            with safe_open(source_path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                missing = sorted(set(requested_keys) - available)
                if missing:
                    raise ValueError(
                        f"{source_shard} is missing indexed tensors: {missing[:8]}"
                    )
                tensors = {key: handle.get_tensor(key) for key in requested_keys}
                shard_bytes = sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors.values()
                )
                output_name = (
                    f"model.safetensors-{output_index:05d}-of-{shard_count:05d}.safetensors"
                )
                save_file(tensors, staging / output_name, metadata={"format": "pt"})
            total_size += shard_bytes
            for key in requested_keys:
                output_weight_map[key] = output_name
            output_shards.append(
                {
                    "source_shard": source_shard,
                    "output_shard": output_name,
                    "tensor_count": len(requested_keys),
                    "tensor_bytes": shard_bytes,
                }
            )
            print(
                f"[{output_index}/{shard_count}] {source_shard} -> {output_name}: "
                f"{len(requested_keys)} tensors, {shard_bytes / 2**30:.3f} GiB",
                flush=True,
            )

        with (staging / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": output_weight_map,
        }
        with (staging / INDEX_NAME).open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
            "source_config_sha256": _sha256(source / CONFIG_NAME),
            "source_index_sha256": _sha256(source / INDEX_NAME),
            "output_total_tensor_bytes": total_size,
            "copied_assets": copied_assets,
            "kept_text_layer_ids": list(range(args.num_hidden_layers)),
            "dropped_text_layer_ids": list(range(args.num_hidden_layers, source_layers)),
            "shards": output_shards,
        }
        with (staging / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

        _validate_output(
            staging,
            target_layers=args.num_hidden_layers,
            expected_keys=set(kept_keys),
        )
        os.replace(staging, destination)
        print(f"Published independent checkpoint: {destination}", flush=True)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    _materialize(_parse_args())


if __name__ == "__main__":
    main()
