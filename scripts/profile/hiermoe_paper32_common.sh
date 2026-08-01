#!/usr/bin/env bash

# Shared, source-only configuration for the 32-NPU paper matrix.

paper32_host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
paper32_source_root=${paper32_host_root}/src
paper32_container_source_root=/workspace/output/hiermoe_greedy_swap_cover_20260722/src
paper32_container_name=${PAPER32_CONTAINER_NAME:-tzq_hiermoe_paper32_warmcache_20260729}
paper32_image=${PAPER32_IMAGE:-ascendai/veomni:qwen35-main-triton321-torch290-warmcache}
paper32_ssh_key=${PAPER32_SSH_KEY:-/root/.ssh/KeyPair-3bce.pem}
paper32_world_size=${PAPER32_WORLD_SIZE:-32}
paper32_nproc_per_node=${PAPER32_NPROC_PER_NODE:-8}
if ((paper32_world_size % paper32_nproc_per_node != 0)); then
  echo "PAPER32_WORLD_SIZE must be divisible by PAPER32_NPROC_PER_NODE" >&2
  return 2 2>/dev/null || exit 2
fi
paper32_nnodes=$((paper32_world_size / paper32_nproc_per_node))
paper32_artifact_prefix=${PAPER32_ARTIFACT_PREFIX:-paper${paper32_world_size}}
if [[ "${paper32_world_size}" == "64" ]]; then
  paper32_cluster_slug=${PAPER32_CLUSTER_SLUG:-huawei12}
else
  paper32_cluster_slug=${PAPER32_CLUSTER_SLUG:-huawei1}
fi
paper32_master_addr=${PAPER32_MASTER_ADDR:-192.168.0.84}
case "${paper32_cluster_slug}" in
  huawei2)
    paper32_perf_model_rel=hiermoe_perf_model_c009_ep32_20260720/v2/hiermoe_perf_model.json
    ;;
  huawei12)
    paper32_perf_model_rel=hiermoe_perf_model_huawei12_ep64_2d_20260730_v1/hiermoe_perf_model.json
    ;;
  *)
    paper32_perf_model_rel=hiermoe_perf_model_huawei1_ep32_20260730_v1/hiermoe_perf_model.json
    ;;
esac
paper32_perf_model_host=${PAPER32_PERF_MODEL_HOST:-/home/tzq/npu_profile_outputs/${paper32_perf_model_rel}}
paper32_perf_model_container=${PAPER32_PERF_MODEL_CONTAINER:-/workspace/output/${paper32_perf_model_rel}}
# Topology calibration is shared by all model/dataset cases at a fixed world
# size. Model compute calibration remains model-specific.
if [[ "${paper32_world_size}" == "64" ]]; then
  paper32_inter_ms_per_byte=${PAPER32_INTER_MS_PER_BYTE:-4.712480381906582e-08}
  paper32_intra_ms_per_byte=${PAPER32_INTRA_MS_PER_BYTE:-1.166976566082813e-08}
else
  paper32_inter_ms_per_byte=${PAPER32_INTER_MS_PER_BYTE:-3.69235444043744e-08}
  paper32_intra_ms_per_byte=${PAPER32_INTRA_MS_PER_BYTE:-8.5092387758084e-09}
fi
paper32_communication_phase_multiplier=${PAPER32_COMMUNICATION_PHASE_MULTIPLIER:-3.1}
paper32_compute_phase_multiplier=${PAPER32_COMPUTE_PHASE_MULTIPLIER:-4.19}
paper32_hosts=(
  "${PAPER32_RANK0_HOST:-huawei1_node1}"
  "${PAPER32_RANK1_HOST:-huawei1_node2}"
  "${PAPER32_RANK2_HOST:-huawei1_node3}"
  "${PAPER32_RANK3_HOST:-huawei1_node4}"
)
if ((paper32_nnodes > 4)); then
  paper32_hosts+=(
    "${PAPER32_RANK4_HOST:-huawei2_node1}"
    "${PAPER32_RANK5_HOST:-huawei2_node2}"
    "${PAPER32_RANK6_HOST:-huawei2_node3}"
    "${PAPER32_RANK7_HOST:-huawei2_node4}"
  )
fi
if ((${#paper32_hosts[@]} != paper32_nnodes)); then
  echo "unsupported node count ${paper32_nnodes}; this matrix supports 4 or 8 nodes" >&2
  return 2 2>/dev/null || exit 2
fi

paper32_configure_model() {
  local model=$1
  case "${model}" in
    qwen3vl)
      paper32_model_path=/workspace/model/Qwen3-VL-30B-A3B-Instruct
      paper32_model_slug=qwen3vl30b
      paper32_num_layers=48
      paper32_num_experts=128
      paper32_primary_slots=$((paper32_num_experts / paper32_world_size))
      paper32_redundant_slots=${PAPER32_DEFAULT_REDUNDANT_SLOTS:-${paper32_primary_slots}}
      paper32_slots_per_rank=$((paper32_primary_slots + paper32_redundant_slots))
      paper32_hidden_size=2048
      paper32_micro_batch_size=4
      paper32_global_batch_size=$((paper32_world_size * paper32_micro_batch_size))
      paper32_freeze_vit=false
      paper32_train_entrypoint=tasks/train_vlm.py
      paper32_train_config=configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml
      paper32_layer_name_template='model.language_model.layers.{layer}.mlp.experts'
      paper32_is_text_model=0
      paper32_lightweight_timing_default=0
      paper32_compute_ms_per_assignment=${PAPER32_COMPUTE_MS_PER_ASSIGNMENT:-2.8151889676680392e-05}
      paper32_compute_calibration_artifact=results/paper32_qwen3vl30b_compute_calibration_ep32_20260727_audit.json
      ;;
    qwen35|qwen35_20l)
      # Canonical paper workload: an independently materialized checkpoint
      # containing the first 20 of the original 40 text layers.
      paper32_model_path=/workspace/model/Qwen3.5-35B-A3B-20L
      paper32_model_slug=qwen35b20l
      paper32_num_layers=20
      paper32_num_experts=256
      paper32_primary_slots=$((paper32_num_experts / paper32_world_size))
      paper32_redundant_slots=${PAPER32_DEFAULT_REDUNDANT_SLOTS:-${paper32_primary_slots}}
      paper32_slots_per_rank=$((paper32_primary_slots + paper32_redundant_slots))
      paper32_hidden_size=2048
      paper32_micro_batch_size=4
      paper32_global_batch_size=$((paper32_world_size * paper32_micro_batch_size))
      paper32_freeze_vit=true
      paper32_train_entrypoint=tasks/train_vlm.py
      paper32_train_config=configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml
      paper32_layer_name_template='model.language_model.layers.{layer}.mlp.experts'
      paper32_is_text_model=0
      paper32_lightweight_timing_default=0
      paper32_compute_ms_per_assignment=${PAPER32_COMPUTE_MS_PER_ASSIGNMENT:-1.60665178692262e-05}
      paper32_compute_calibration_artifact=results/paper32_qwen35b20l_compute_calibration_ep32_20260730_audit.json
      ;;
    deepseek_v3_6moe_half|deepseek6moe)
      paper32_model_path=/workspace/model/DeepSeek-V3-6MoE-Half
      paper32_model_slug=deepseekv3_6moe_half
      paper32_num_layers=6
      paper32_num_experts=256
      paper32_primary_slots=$((paper32_num_experts / paper32_world_size))
      paper32_redundant_slots=${PAPER32_DEFAULT_REDUNDANT_SLOTS:-${paper32_primary_slots}}
      paper32_slots_per_rank=$((paper32_primary_slots + paper32_redundant_slots))
      paper32_hidden_size=3584
      paper32_micro_batch_size=4
      paper32_global_batch_size=$((paper32_world_size * paper32_micro_batch_size))
      paper32_freeze_vit=false
      paper32_train_entrypoint=tasks/train_text.py
      paper32_train_config=configs/text/deepseek_v3_6moe_half.yaml
      paper32_layer_name_template='model.layers.{layer}.mlp.experts'
      paper32_is_text_model=1
      paper32_lightweight_timing_default=1
      if [[ "${paper32_world_size}" == "64" ]]; then
        paper32_compute_calibration_artifact=${PAPER32_COMPUTE_CALIBRATION_ARTIFACT:-results/paper64_deepseekv3_6moe_half_compute_calibration_ep64_20260730_v1.json}
      else
        paper32_compute_calibration_artifact=${PAPER32_COMPUTE_CALIBRATION_ARTIFACT:-results/paper32_deepseekv3_6moe_half_compute_calibration_ep32_20260730_v3.json}
      fi
      # Route capture and compute calibration do not consume this coefficient.
      # Layout construction must replace this sentinel from an accepted,
      # model-specific calibration artifact.
      paper32_compute_ms_per_assignment=${PAPER32_COMPUTE_MS_PER_ASSIGNMENT:-0}
      ;;
    *)
      echo "unsupported model '${model}'; expected qwen3vl, qwen35_20l, or deepseek_v3_6moe_half" >&2
      return 2
      ;;
  esac

  if [[ -n "${PAPER32_REDUNDANT_SLOTS_OVERRIDE:-}" ]]; then
    if [[ ! "${PAPER32_REDUNDANT_SLOTS_OVERRIDE}" =~ ^[0-9]+$ ]]; then
      echo "PAPER32_REDUNDANT_SLOTS_OVERRIDE must be a non-negative integer" >&2
      return 2
    fi
    paper32_redundant_slots=${PAPER32_REDUNDANT_SLOTS_OVERRIDE}
    paper32_slots_per_rank=$((paper32_primary_slots + paper32_redundant_slots))
  fi

  if [[ -n "${PAPER32_MICRO_BATCH_SIZE_OVERRIDE:-}" ]]; then
    if [[ ! "${PAPER32_MICRO_BATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
      echo "PAPER32_MICRO_BATCH_SIZE_OVERRIDE must be a positive integer" >&2
      return 2
    fi
    paper32_micro_batch_size=${PAPER32_MICRO_BATCH_SIZE_OVERRIDE}
  fi
  if [[ -n "${PAPER32_GLOBAL_BATCH_SIZE_OVERRIDE:-}" ]]; then
    if [[ ! "${PAPER32_GLOBAL_BATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
      echo "PAPER32_GLOBAL_BATCH_SIZE_OVERRIDE must be a positive integer" >&2
      return 2
    fi
    paper32_global_batch_size=${PAPER32_GLOBAL_BATCH_SIZE_OVERRIDE}
  fi
}

paper32_configure_dataset() {
  local dataset=$1
  case "${dataset}" in
    sharegpt4v)
      paper32_dataset_slug=sharegpt4v
      paper32_data_path=/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards
      paper32_data_source_name=sharegpt4v_sft
      ;;
    tulu3)
      paper32_dataset_slug=tulu3
      paper32_data_path=/workspace/dataset/Tulu3/train-00002-of-00006.parquet
      if [[ "${paper32_is_text_model:-0}" == "1" ]]; then
        paper32_data_source_name=tulu3_sft
      else
        paper32_data_source_name=tulu-3-sft-mixture
      fi
      if [[ "${paper32_is_text_model:-0}" == "1" ]]; then
        paper32_freeze_vit=false
      else
        paper32_freeze_vit=true
      fi
      ;;
    *)
      echo "unsupported dataset '${dataset}'; expected sharegpt4v or tulu3" >&2
      return 2
      ;;
  esac
}

paper32_load_compute_calibration() {
  local artifact=${paper32_source_root}/${paper32_compute_calibration_artifact}
  if [[ ! -s "${artifact}" ]]; then
    echo "missing model-specific compute calibration: ${artifact}" >&2
    return 1
  fi
  local calibrated
  calibrated=$(python - "${artifact}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "accepted":
    raise SystemExit(f"calibration status is {payload.get('status')!r}, expected 'accepted'")
value = payload.get("coefficients", {}).get("compute_ms_per_assignment")
if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
    raise SystemExit(f"invalid compute_ms_per_assignment: {value!r}")
print(value)
PY
  )
  paper32_compute_ms_per_assignment=${PAPER32_COMPUTE_MS_PER_ASSIGNMENT:-${calibrated}}
}

paper32_profile_name() {
  printf '%s_profile_%s_%s_%s_p4_%s' \
    "${paper32_artifact_prefix}" \
    "${paper32_cluster_slug}" \
    "${paper32_model_slug}" \
    "${paper32_dataset_slug}" \
    "${PAPER32_PROFILE_TAG:-20260730}"
}

paper32_layout_stem() {
  local method=$1
  local layout_tag_suffix=
  if [[ -n "${PAPER32_LAYOUT_TAG:-}" ]]; then
    layout_tag_suffix="_${PAPER32_LAYOUT_TAG}"
  fi
  printf '%s_%s_%s_%s_%s_b%s_ep%s%s' \
    "${paper32_artifact_prefix}" \
    "${paper32_cluster_slug}" \
    "${paper32_model_slug}" \
    "${paper32_dataset_slug}" \
    "${method}" \
    "${paper32_redundant_slots}" \
    "${paper32_world_size}" \
    "${layout_tag_suffix}"
}

paper32_ablation_layout_stem() {
  local ablation=$1
  local redundant_slots=$2
  local layout_tag_suffix=
  if [[ -n "${PAPER32_LAYOUT_TAG:-}" ]]; then
    layout_tag_suffix="_${PAPER32_LAYOUT_TAG}"
  fi
  printf '%s_%s_%s_%s_ablation_%s_b%s_ep%s%s' \
    "${paper32_artifact_prefix}" \
    "${paper32_cluster_slug}" \
    "${paper32_model_slug}" \
    "${paper32_dataset_slug}" \
    "${ablation}" \
    "${redundant_slots}" \
    "${paper32_world_size}" \
    "${layout_tag_suffix}"
}

paper32_method_slug() {
  case "$1" in
    baseline)
      printf 'veomni_baseline'
      ;;
    r2)
      printf 'fixed_r2_hierarchical_dedup'
      ;;
    eplb)
      printf 'eplb_static_hierarchical_dedup'
      ;;
    hiermoe)
      printf 'hiermoe_exact_p1'
      ;;
    ours)
      printf 'ours_static_hierarchical_dedup'
      ;;
    ours_full_replan)
      printf 'ours_periodic_full_replan_hierarchical_dedup'
      ;;
    *)
      echo "unsupported paper32 method '$1'" >&2
      return 2
      ;;
  esac
}

paper32_method_grad_mode() {
  case "$1" in
    ours|ours_online_lut|ours_full_replan)
      # Hiding redundant-gradient synchronization is part of the complete
      # method.  R2/EPLB intentionally retain blocking synchronization.
      printf 'hidden'
      ;;
    baseline|r2|eplb|hiermoe|dedup)
      printf 'blocking'
      ;;
    static)
      printf 'hidden'
      ;;
    *)
      echo "unsupported paper32 method '$1'" >&2
      return 2
      ;;
  esac
}

paper32_ssh_args() {
  if [[ -n "${paper32_ssh_key}" ]]; then
    printf '%s\n' -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no
  else
    printf '%s\n' -o StrictHostKeyChecking=no
  fi
}
