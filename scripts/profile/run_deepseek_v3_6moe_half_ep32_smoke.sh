#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "${script_dir}/../.." && pwd)
run_tag=${DEEPSEEK_EP32_RUN_TAG:-20260730_r1}
run_name=${DEEPSEEK_EP32_RUN_NAME:-deepseek_v3_6moe_half_tulu3_ep32_aligned_smoke_${run_tag}}

# Full-module event profiling on every rank materially perturbs this short
# smoke workload. MoE A2A device-event timing remains enabled separately.
env \
  E2E_VARIANT=baseline \
  RUN_NAME_OVERRIDE="${run_name}" \
  TRAIN_ENTRYPOINT_OVERRIDE=tasks/train_text.py \
  TRAIN_CONFIG_OVERRIDE=configs/text/deepseek_v3_6moe_half.yaml \
  MODEL_PATH_OVERRIDE=/workspace/model/DeepSeek-V3-6MoE-Half \
  MODEL_CONFIG_PATH_OVERRIDE=/workspace/model/DeepSeek-V3-6MoE-Half \
  DATA_PATH_OVERRIDE=/workspace/dataset/Tulu3/train-00002-of-00006.parquet \
  DATA_SOURCE_NAME_OVERRIDE=tulu3_sft \
  TRAIN_FREEZE_VIT_OVERRIDE=false \
  NNODES_OVERRIDE=4 \
  NPROC_PER_NODE_OVERRIDE=8 \
  DP_REPLICATE_SIZE_OVERRIDE=1 \
  DP_SHARD_SIZE_OVERRIDE=32 \
  EP_SIZE_OVERRIDE=32 \
  NUM_MOE_LAYERS_OVERRIDE=6 \
  MICRO_BATCH_SIZE_OVERRIDE=4 \
  GLOBAL_BATCH_SIZE_OVERRIDE=128 \
  MAX_SEQ_LEN_OVERRIDE=4096 \
  MAX_STEPS_OVERRIDE="${DEEPSEEK_EP32_MAX_STEPS:-4}" \
  FULL_PROFILE_ENABLE_OVERRIDE=0 \
  FULL_PROFILE_START_STEP_OVERRIDE=2 \
  FULL_PROFILE_EVERY_N_OVERRIDE=1 \
  FULL_PROFILE_RANKS_OVERRIDE=all \
  MASTER_ADDR_OVERRIDE="${DEEPSEEK_EP32_MASTER_ADDR:-192.168.0.55}" \
  MASTER_PORT="${DEEPSEEK_EP32_MASTER_PORT:-30730}" \
  HCCL_IF_BASE_PORT="${DEEPSEEK_EP32_HCCL_PORT:-64700}" \
  RANK0_HOST_OVERRIDE=huawei2_node1 \
  RANK1_HOST_OVERRIDE=huawei2_node2 \
  RANK2_HOST_OVERRIDE=huawei2_node3 \
  RANK3_HOST_OVERRIDE=huawei2_node4 \
  RANK0_CONTAINER_OVERRIDE=tzq_hiermoe_paper32_warmcache_20260729 \
  RANK1_CONTAINER_OVERRIDE=tzq_hiermoe_paper32_warmcache_20260729 \
  RANK2_CONTAINER_OVERRIDE=tzq_hiermoe_paper32_warmcache_20260729 \
  RANK3_CONTAINER_OVERRIDE=tzq_hiermoe_paper32_warmcache_20260729 \
  bash "${source_root}/scripts/profile/launch_hiermoe_greedy_e2e_4node.sh"

echo "completed ${run_name}"
