#!/usr/bin/env bash
# Run a sequential HierMoE expert-swap pair-count sweep on each 4-node NPU rank.
#
# Launch this script in the four rank containers with the same environment
# variables and NODE_RANK=0..3. It reuses the 100-step profile launcher and only
# changes train.hiermoe.expert_swap_max_pairs_per_layer between runs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

PAIR_COUNTS=${HIERMOE_SWAP_PAIR_COUNTS:-"1 2 4 16 32"}
PROFILE_KIND=${PROFILE_KIND:-pretrain}
RUN_PREFIX=${RUN_PREFIX:-qwen3vl_hiermoe_swap_pairs_ep32_mb4_gbs128}
SWEEP_ID=${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29700}
MAX_STEPS=${MAX_STEPS:-6}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
EP_SIZE=${EP_SIZE:-32}
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MOE_IMPL=${MOE_IMPL:-fused_npu}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}

export PROFILE_KIND
export MAX_STEPS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE EP_SIZE NNODES NPROC_PER_NODE MOE_IMPL ATTN_IMPL
export HIERMOE_ENABLE=1
export HIERMOE_TOKEN_DEDUP=true
export HIERMOE_EXPERT_SWAP=true
export VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
export VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-3}
export VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N:-1}
export VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-0}
export VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE:-0}
export VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS:-0}
export MOE_MONITOR_INTERVAL=${MOE_MONITOR_INTERVAL:-1}

index=0
for pairs in ${PAIR_COUNTS}; do
    export HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER="${pairs}"
    export MASTER_PORT=$((MASTER_PORT_BASE + index))
    export RUN_NAME="${RUN_PREFIX}_pairs${pairs}_${MAX_STEPS}step_${SWEEP_ID}"
    mkdir -p "pretrain_runs/${RUN_NAME}"
    echo "=== HierMoE swap pair sweep: pairs=${pairs} RUN_NAME=${RUN_NAME} MASTER_PORT=${MASTER_PORT} ==="
    bash scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh \
        2>&1 | tee "pretrain_runs/${RUN_NAME}/rank${NODE_RANK}.log"
    index=$((index + 1))
done
