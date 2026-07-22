#!/usr/bin/env python3
"""Split a JSON-array dataset into balanced JSON-array shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source JSON file containing one top-level array.")
    parser.add_argument("output_dir", type=Path, help="New directory that will contain only JSON shard files.")
    parser.add_argument("--num-shards", type=int, default=16, help="Number of balanced shards to write.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output path: {output_dir}")

    with input_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("The source JSON must contain one top-level array.")
    if args.num_shards > len(records):
        raise ValueError(f"Cannot create {args.num_shards} non-empty shards from {len(records)} records.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    shard_counts: list[int] = []
    try:
        quotient, remainder = divmod(len(records), args.num_shards)
        start = 0
        for shard_index in range(args.num_shards):
            count = quotient + (1 if shard_index < remainder else 0)
            end = start + count
            shard_path = temporary_dir / f"part-{shard_index:05d}-of-{args.num_shards:05d}.json"
            with shard_path.open("w", encoding="utf-8") as handle:
                json.dump(records[start:end], handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            shard_counts.append(count)
            start = end
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    manifest = {
        "source": str(input_path),
        "source_sha256": file_sha256(input_path),
        "num_records": len(records),
        "num_shards": args.num_shards,
        "shard_counts": shard_counts,
    }
    manifest_path = output_dir.parent / f"{output_dir.name}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(records)} records to {args.num_shards} shards in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
