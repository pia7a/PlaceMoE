#!/usr/bin/env python3
"""Materialize the reduced DeepSeek-V3 checkpoint used for HierMoE experiments.

The HierMoE paper evaluates a six-layer DeepSeek-V3 whose model and hidden
dimensions are half of the original model. This generator keeps the routing
problem intact (256 routed experts, top-8, eight routing groups), uses six pure
MoE decoder blocks, and scales all width-dependent dimensions by one half.

The generated checkpoint is deterministic random initialization intended for
systems experiments. It is not a pretrained language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file


InitKind = Literal["normal", "ones", "zeros"]
_DTYPE_NAME_TO_TORCH = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_DTYPE_NAME_TO_SAFETENSORS = {
    "bfloat16": "BF16",
    "float32": "F32",
}
_ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype_name: Literal["bfloat16", "float32"]
    init: InitKind

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= int(dim)
        return result

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=_DTYPE_NAME_TO_TORCH[self.dtype_name]).element_size()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    required = {
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "vocab_size",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing required DeepSeek-V3 config fields: {missing}")
    if int(config.get("first_k_dense_replace", -1)) != 0:
        raise ValueError("The HierMoE reduced checkpoint requires six pure MoE layers (first_k_dense_replace=0).")
    if int(config["num_hidden_layers"]) != 6:
        raise ValueError("The HierMoE reduced checkpoint requires exactly six decoder layers.")
    if int(config["n_routed_experts"]) % int(config["n_group"]) != 0:
        raise ValueError("n_routed_experts must be divisible by n_group.")
    if int(config["num_experts_per_tok"]) > int(config["n_routed_experts"]):
        raise ValueError("num_experts_per_tok cannot exceed n_routed_experts.")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]) != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")


def build_tensor_specs(config: dict) -> list[TensorSpec]:
    hidden = int(config["hidden_size"])
    moe_intermediate = int(config["moe_intermediate_size"])
    shared_intermediate = moe_intermediate * int(config["n_shared_experts"])
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    num_heads = int(config["num_attention_heads"])
    q_lora_rank = int(config["q_lora_rank"])
    kv_lora_rank = int(config["kv_lora_rank"])
    qk_head_dim = int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    kv_out_dim = num_heads * (int(config["qk_nope_head_dim"]) + int(config["v_head_dim"]))
    vocab_size = int(config["vocab_size"])

    specs = [TensorSpec("model.embed_tokens.weight", (vocab_size, hidden), "bfloat16", "normal")]
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        specs.extend(
            (
                TensorSpec(
                    f"{prefix}.self_attn.q_a_proj.weight",
                    (q_lora_rank, hidden),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.q_a_layernorm.weight",
                    (q_lora_rank,),
                    "bfloat16",
                    "ones",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.q_b_proj.weight",
                    (num_heads * qk_head_dim, q_lora_rank),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                    (kv_lora_rank + int(config["qk_rope_head_dim"]), hidden),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.kv_a_layernorm.weight",
                    (kv_lora_rank,),
                    "bfloat16",
                    "ones",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.kv_b_proj.weight",
                    (kv_out_dim, kv_lora_rank),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.self_attn.o_proj.weight",
                    (hidden, num_heads * int(config["v_head_dim"])),
                    "bfloat16",
                    "normal",
                ),
            )
        )
        for expert_idx in range(num_experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"
            specs.extend(
                (
                    TensorSpec(
                        f"{expert_prefix}.gate_proj.weight",
                        (moe_intermediate, hidden),
                        "bfloat16",
                        "normal",
                    ),
                    TensorSpec(
                        f"{expert_prefix}.up_proj.weight",
                        (moe_intermediate, hidden),
                        "bfloat16",
                        "normal",
                    ),
                    TensorSpec(
                        f"{expert_prefix}.down_proj.weight",
                        (hidden, moe_intermediate),
                        "bfloat16",
                        "normal",
                    ),
                )
            )
        specs.extend(
            (
                TensorSpec(f"{prefix}.mlp.gate.weight", (num_experts, hidden), "bfloat16", "normal"),
                TensorSpec(
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                    (num_experts,),
                    "float32",
                    "zeros",
                ),
                TensorSpec(
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    (shared_intermediate, hidden),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    (shared_intermediate, hidden),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                    (hidden, shared_intermediate),
                    "bfloat16",
                    "normal",
                ),
                TensorSpec(f"{prefix}.input_layernorm.weight", (hidden,), "bfloat16", "ones"),
                TensorSpec(f"{prefix}.post_attention_layernorm.weight", (hidden,), "bfloat16", "ones"),
            )
        )
    specs.extend(
        (
            TensorSpec("model.norm.weight", (hidden,), "bfloat16", "ones"),
            TensorSpec("lm_head.weight", (vocab_size, hidden), "bfloat16", "normal"),
        )
    )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise RuntimeError("Tensor specification contains duplicate checkpoint keys.")
    return specs


def pack_shards(specs: Sequence[TensorSpec], max_shard_bytes: int) -> list[list[TensorSpec]]:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive.")
    shards: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    current_bytes = 0
    for spec in specs:
        if current and current_bytes + spec.nbytes > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(spec)
        current_bytes += spec.nbytes
    if current:
        shards.append(current)
    return shards


def stable_tensor_seed(global_seed: int, tensor_name: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{tensor_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def materialize_tensor(spec: TensorSpec, *, global_seed: int, initializer_range: float) -> torch.Tensor:
    dtype = _DTYPE_NAME_TO_TORCH[spec.dtype_name]
    if spec.init == "zeros":
        return torch.zeros(spec.shape, dtype=dtype)
    if spec.init == "ones":
        return torch.ones(spec.shape, dtype=dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_tensor_seed(global_seed, spec.name))
    tensor = torch.empty(spec.shape, dtype=dtype)
    return tensor.normal_(mean=0.0, std=initializer_range, generator=generator)


def shard_file_name(index: int, count: int) -> str:
    return f"model-{index:05d}-of-{count:05d}.safetensors"


def validate_shard(path: Path, specs: Sequence[TensorSpec]) -> None:
    expected = {spec.name: spec for spec in specs}
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        if actual_names != set(expected):
            missing = sorted(set(expected).difference(actual_names))
            extra = sorted(actual_names.difference(expected))
            raise RuntimeError(f"Shard key mismatch for {path}: missing={missing[:8]}, extra={extra[:8]}")
        for name, spec in expected.items():
            tensor_slice = handle.get_slice(name)
            if tuple(tensor_slice.get_shape()) != spec.shape:
                raise RuntimeError(
                    f"Shard shape mismatch for {name}: expected={spec.shape}, actual={tensor_slice.get_shape()}"
                )
            if tensor_slice.get_dtype() != _DTYPE_NAME_TO_SAFETENSORS[spec.dtype_name]:
                raise RuntimeError(
                    f"Shard dtype mismatch for {name}: expected={spec.dtype_name}, actual={tensor_slice.get_dtype()}"
                )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_assets(source: Path | None, destination: Path) -> list[str]:
    if source is None:
        return []
    copied: list[str] = []
    for name in _ASSET_NAMES:
        source_path = source / name
        if not source_path.is_file():
            continue
        shutil.copy2(source_path, destination / name)
        copied.append(name)
    return copied


def normalize_tokenizer_config(destination: Path) -> list[str]:
    tokenizer_config_path = destination / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return []
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    adjustments: list[str] = []
    if not tokenizer_config.get("fix_mistral_regex", False):
        tokenizer_config["fix_mistral_regex"] = True
        adjustments.append("fix_mistral_regex=true")
    if adjustments:
        write_json(tokenizer_config_path, tokenizer_config)
    return adjustments


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_readme(config: dict, *, parameter_count: int, weight_bytes: int, seed: int) -> str:
    return f"""# DeepSeek-V3 HierMoE 6-MoE Half-Width

This is a deterministic randomly initialized checkpoint for distributed
systems experiments. It is **not pretrained** and must not be used for model
quality evaluation.

The architecture follows the scale reduction stated in the HierMoE evaluation:

- 6 pure MoE decoder blocks (`first_k_dense_replace=0`)
- hidden size 3584 (half of 7168)
- routed/shared expert intermediate size 1024 (half of 2048)
- 64 attention heads (half of 128), preserving per-head dimensions
- query/KV LoRA ranks 768/256 (half of 1536/512)
- 256 routed experts, top-8, 8 routing groups
- BF16 weights, no MTP layers

The checkpoint contains {parameter_count:,} parameters and {weight_bytes:,}
bytes of tensor payload. The per-tensor random seed is derived from global seed
{seed} and the fully qualified tensor name, so shard/resume boundaries do not
change the generated values.

Paper: https://arxiv.org/abs/2508.09591
"""


def verify_checkpoint(directory: Path, config: dict, specs: Sequence[TensorSpec]) -> dict:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    expected_names = {spec.name for spec in specs}
    actual_names = set(weight_map)
    if actual_names != expected_names:
        missing = sorted(expected_names.difference(actual_names))
        extra = sorted(actual_names.difference(expected_names))
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing[:8]}, extra={extra[:8]}")

    specs_by_shard: dict[str, list[TensorSpec]] = {}
    for spec in specs:
        specs_by_shard.setdefault(weight_map[spec.name], []).append(spec)
    for file_name, shard_specs in sorted(specs_by_shard.items()):
        validate_shard(directory / file_name, shard_specs)

    expected_bytes = sum(spec.nbytes for spec in specs)
    index_bytes = int(index.get("metadata", {}).get("total_size", -1))
    if index_bytes != expected_bytes:
        raise RuntimeError(f"Tensor byte count mismatch: expected={expected_bytes}, index={index_bytes}")
    config_on_disk = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for key, value in config.items():
        if config_on_disk.get(key) != value:
            raise RuntimeError(f"Config mismatch for {key}: expected={value!r}, actual={config_on_disk.get(key)!r}")
    return {
        "tensor_count": len(specs),
        "parameter_count": sum(spec.numel for spec in specs),
        "tensor_bytes": expected_bytes,
        "shard_count": len(specs_by_shard),
    }


def generate_checkpoint(
    *,
    config_path: Path,
    output_dir: Path,
    tokenizer_source: Path | None,
    seed: int,
    max_shard_bytes: int,
    num_threads: int,
) -> dict:
    config = load_config(config_path)
    specs = build_tensor_specs(config)
    shards = pack_shards(specs, max_shard_bytes)
    staging_dir = output_dir.with_name(f".{output_dir.name}.incomplete")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, int(num_threads)))
    weight_map: dict[str, str] = {}
    initializer_range = float(config.get("initializer_range", 0.02))
    for shard_idx, shard_specs in enumerate(shards, start=1):
        file_name = shard_file_name(shard_idx, len(shards))
        shard_path = staging_dir / file_name
        if shard_path.is_file():
            validate_shard(shard_path, shard_specs)
        else:
            tensors = {
                spec.name: materialize_tensor(
                    spec,
                    global_seed=seed,
                    initializer_range=initializer_range,
                )
                for spec in shard_specs
            }
            save_file(tensors, shard_path, metadata={"format": "pt"})
            del tensors
            validate_shard(shard_path, shard_specs)
        for spec in shard_specs:
            weight_map[spec.name] = file_name
        print(
            f"[{shard_idx:02d}/{len(shards):02d}] {file_name} "
            f"{sum(spec.nbytes for spec in shard_specs) / 2**30:.3f} GiB",
            flush=True,
        )

    tensor_bytes = sum(spec.nbytes for spec in specs)
    write_json(
        staging_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": tensor_bytes},
            "weight_map": weight_map,
        },
    )
    write_json(staging_dir / "config.json", config)
    copied_assets = copy_assets(tokenizer_source, staging_dir)
    tokenizer_adjustments = normalize_tokenizer_config(staging_dir)
    if "generation_config.json" not in copied_assets:
        write_json(
            staging_dir / "generation_config.json",
            {
                "_from_model_config": True,
                "bos_token_id": config.get("bos_token_id"),
                "eos_token_id": config.get("eos_token_id"),
                "transformers_version": "5.2.0",
            },
        )
        copied_assets.append("generation_config.json")
    parameter_count = sum(spec.numel for spec in specs)
    (staging_dir / "README.md").write_text(
        render_readme(config, parameter_count=parameter_count, weight_bytes=tensor_bytes, seed=seed),
        encoding="utf-8",
    )

    verification = verify_checkpoint(staging_dir, config, specs)
    shard_checksums = {
        shard_file_name(index, len(shards)): sha256_file(staging_dir / shard_file_name(index, len(shards)))
        for index in range(1, len(shards) + 1)
    }
    manifest = {
        "architecture": "DeepSeek-V3 HierMoE 6-MoE half-width",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": "bfloat16",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "global_seed": seed,
        "hiermoe_paper": "https://arxiv.org/abs/2508.09591",
        "intended_use": "distributed training systems performance experiments only",
        "max_shard_bytes": max_shard_bytes,
        "randomly_initialized": True,
        "source_config": str(config_path.resolve()),
        "tokenizer_assets": copied_assets,
        "tokenizer_adjustments": tokenizer_adjustments,
        "tokenizer_source": str(tokenizer_source.resolve()) if tokenizer_source is not None else None,
        **verification,
        "shards": shard_checksums,
    }
    write_json(staging_dir / "hiermoe_reduction_manifest.json", manifest)
    checksum_lines = [f"{digest}  {name}" for name, digest in sorted(shard_checksums.items())]
    (staging_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.rename(staging_dir, output_dir)
    return manifest


def parse_size(value: str) -> int:
    normalized = value.strip().lower()
    suffixes = {
        "gib": 2**30,
        "gb": 10**9,
        "mib": 2**20,
        "mb": 10**6,
    }
    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            return int(float(normalized[: -len(suffix)]) * multiplier)
    return int(normalized)


def print_plan(config: dict, specs: Sequence[TensorSpec], shards: Sequence[Sequence[TensorSpec]]) -> None:
    tensor_bytes = sum(spec.nbytes for spec in specs)
    print(
        json.dumps(
            {
                "num_layers": config["num_hidden_layers"],
                "first_k_dense_replace": config["first_k_dense_replace"],
                "hidden_size": config["hidden_size"],
                "moe_intermediate_size": config["moe_intermediate_size"],
                "num_attention_heads": config["num_attention_heads"],
                "n_routed_experts": config["n_routed_experts"],
                "num_experts_per_tok": config["num_experts_per_tok"],
                "tensor_count": len(specs),
                "parameter_count": sum(spec.numel for spec in specs),
                "tensor_bytes": tensor_bytes,
                "tensor_gib": tensor_bytes / 2**30,
                "shard_count": len(shards),
                "largest_shard_gib": max(sum(spec.nbytes for spec in shard) for shard in shards) / 2**30,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    default_config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "model_configs"
        / "deepseek"
        / "DeepseekV3_HierMoE_6MoE_Half.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--seed", type=int, default=20250813)
    parser.add_argument("--max-shard-size", type=parse_size, default=parse_size("4GiB"))
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    specs = build_tensor_specs(config)
    shards = pack_shards(specs, args.max_shard_size)
    if args.dry_run:
        print_plan(config, specs, shards)
        return
    if args.verify_only:
        print(json.dumps(verify_checkpoint(args.output_dir, config, specs), indent=2))
        return
    manifest = generate_checkpoint(
        config_path=args.config,
        output_dir=args.output_dir,
        tokenizer_source=args.tokenizer_source,
        seed=args.seed,
        max_shard_bytes=args.max_shard_size,
        num_threads=args.num_threads,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
