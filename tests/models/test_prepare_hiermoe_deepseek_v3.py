import json
from pathlib import Path

from scripts.deepseek_v3.prepare_hiermoe_reduced_checkpoint import (
    build_tensor_specs,
    load_config,
    normalize_tokenizer_config,
    pack_shards,
    stable_tensor_seed,
)


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "model_configs"
    / "deepseek"
    / "DeepseekV3_HierMoE_6MoE_Half.json"
)


def test_hiermoe_deepseek_config_preserves_routing_scale() -> None:
    config = load_config(_CONFIG_PATH)

    assert config["num_hidden_layers"] == 6
    assert config["first_k_dense_replace"] == 0
    assert config["hidden_size"] == 3584
    assert config["moe_intermediate_size"] == 1024
    assert config["num_attention_heads"] == 64
    assert config["q_lora_rank"] == 768
    assert config["kv_lora_rank"] == 256
    assert config["n_routed_experts"] == 256
    assert config["num_experts_per_tok"] == 8
    assert config["n_group"] == 8
    assert config["topk_group"] == 4


def test_tensor_specs_use_official_split_expert_keys() -> None:
    config = load_config(_CONFIG_PATH)
    specs = build_tensor_specs(config)
    by_name = {spec.name: spec for spec in specs}

    assert by_name["model.layers.0.mlp.experts.255.gate_proj.weight"].shape == (1024, 3584)
    assert by_name["model.layers.5.mlp.experts.255.down_proj.weight"].shape == (3584, 1024)
    assert "model.layers.0.mlp.experts.gate_up_proj" not in by_name
    assert by_name["model.layers.0.mlp.gate.e_score_correction_bias"].dtype_name == "float32"
    assert sum(spec.numel for spec in specs) == 18_191_078_400


def test_sharding_and_tensor_seeds_are_deterministic() -> None:
    config = load_config(_CONFIG_PATH)
    specs = build_tensor_specs(config)
    shards = pack_shards(specs, 4 * 2**30)

    assert len(shards) == 9
    assert all(sum(spec.nbytes for spec in shard) <= 4 * 2**30 for shard in shards)
    assert stable_tensor_seed(20250813, specs[0].name) == stable_tensor_seed(20250813, specs[0].name)
    assert stable_tensor_seed(20250813, specs[0].name) != stable_tensor_seed(20250813, specs[1].name)


def test_config_is_json_round_trip_stable() -> None:
    config = load_config(_CONFIG_PATH)

    assert json.loads(json.dumps(config)) == config


def test_tokenizer_config_enables_transformers_v5_regex_fix(tmp_path: Path) -> None:
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"tokenizer_class": "LlamaTokenizerFast"}))

    adjustments = normalize_tokenizer_config(tmp_path)

    assert adjustments == ["fix_mistral_regex=true"]
    assert json.loads(tokenizer_config.read_text())["fix_mistral_regex"] is True


def test_generated_deepseek_experts_use_opslot_layer_registration() -> None:
    generated_root = (
        Path(__file__).resolve().parents[2] / "veomni" / "models" / "transformers" / "deepseek_v3" / "generated"
    )
    for backend in ("gpu", "npu"):
        source = (generated_root / f"patched_modeling_deepseek_v3_{backend}.py").read_text()
        assert "return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)" in source
