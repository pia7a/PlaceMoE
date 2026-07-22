#!/usr/bin/env bash
# Reproducible two-node EP16 benchmark for joint HierMoE placement.
set -euo pipefail

if [[ -n "${RUN_LOG:-}" && "${VEOMNI_HIERMOE_RUN_WRAPPED:-0}" != "1" ]]; then
    export VEOMNI_HIERMOE_RUN_WRAPPED=1
    set +e
    bash "$0" "$@" >"${RUN_LOG}" 2>&1
    status=$?
    printf '%s\n' "${status}" >"${RUN_EXIT_FILE:-${RUN_LOG}.exit}"
    exit "${status}"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export RUN_NAME=${RUN_NAME:?RUN_NAME must be set}
export RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/pretrain_runs/${RUN_NAME}"}
export MODEL_PATH=${MODEL_PATH:-/workspace/model/Qwen3-VL-30B-A3B-Instruct}
export DATA_PATH=${DATA_PATH:-"/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards"}
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=${MASTER_ADDR:-192.168.0.63}
export MASTER_PORT=${MASTER_PORT:?MASTER_PORT must be set}
export MAX_STEPS=${MAX_STEPS:-22}
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=64
export MAX_SEQ_LEN=4096
export DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
export DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-2}
export DP_REPLICATE_SIZE=1
export DP_SHARD_SIZE=16
export EP_SIZE=16
export NUM_MOE_LAYERS=48
export MOE_IMPL=fused_npu
export ATTN_IMPL=flash_attention_2
export PROFILE_KIND=pretrain
export VERL_MOE_PROFILE_DIR="${REPO_ROOT}/profile/runs/pretrain/${RUN_NAME}"
export VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
export VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-1}
export VEOMNI_FULL_PROFILE_EVERY_N=1
export VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-0}
export VEOMNI_TORCH_PROFILE_ENABLE=0
export VEOMNI_MOE_TIMING_SYNC_EVENTS=0
PAIRS=${PAIRS:-1}
SLOTS=${SLOTS:-1}
ROUTE_MODE=${ROUTE_MODE:-step}
SLOT_ROUNDS=${SLOT_ROUNDS:-}

unset VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH
unset VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP
unset VEOMNI_HIERMOE_ORACLE_CAPTURE_LAYER
unset VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL

cd "${REPO_ROOT}"
HIERMOE_ARGS=(
    --train.hiermoe.enable true
    --train.hiermoe.token_dedup true
    --train.hiermoe.expert_swap true
    --train.hiermoe.expert_swap_mode "${ROUTE_MODE}"
    --train.hiermoe.expert_swap_interval 1
    --train.hiermoe.expert_swap_max_pairs_per_layer "${PAIRS}"
    --train.hiermoe.redundant_slot_increment_per_device "${SLOTS}"
    --train.hiermoe.hierarchy_group_sizes 8 16
    --train.hiermoe.log_interval 1
)
if [[ -n "${SLOT_ROUNDS}" ]]; then
    HIERMOE_ARGS+=(--train.hiermoe.max_slot_op_search_rounds "${SLOT_ROUNDS}")
fi
exec bash scripts/profile/run_qwen3_vl_moe_pretrain_profile_npu_4node.sh "${HIERMOE_ARGS[@]}"
