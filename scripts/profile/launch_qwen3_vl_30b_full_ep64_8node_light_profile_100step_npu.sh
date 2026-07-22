#!/usr/bin/env bash
# Launch the 8-node / 64-NPU Qwen3-VL-30B-A3B VeOmni light-profile pretrain run.
#
# Run this script on the NPU jump/head node:
#   cd /home/tzq/VeOmni-0.1.11
#   bash scripts/profile/launch_qwen3_vl_30b_full_ep64_8node_light_profile_100step_npu.sh
#
# Useful overrides:
#   START_CONTAINERS=1 bash scripts/profile/launch_qwen3_vl_30b_full_ep64_8node_light_profile_100step_npu.sh
#   RUN_NAME=my_run MASTER_PORT=29501 MAX_STEPS=100 MICRO_BATCH_SIZE=4 bash ...
#
# Logs are written on each node inside the mounted repo:
#   /home/tzq/VeOmni-0.1.11/pretrain_runs/${RUN_NAME}/rank${rank}.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

KEY=${KEY:-/home/tzq/KeyPair-3bce.pem}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-tzq_npu_profile_rank}
START_CONTAINERS=${START_CONTAINERS:-0}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}

TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh}

declare -A HOST_BY_RANK=(
  [0]=192.168.0.63
  [1]=192.168.0.164
  [2]=192.168.0.45
  [3]=192.168.0.93
  [4]=192.168.0.174
  [5]=192.168.0.213
  [6]=192.168.0.136
  [7]=192.168.0.210
)

NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TOTAL_PROCS=$((NNODES * NPROC_PER_NODE))

RUN_NAME=${RUN_NAME:-qwen3vl_full48_sharegpt4v_ep64_mb4_gbs256_100step_8node_$(date +%Y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-192.168.0.63}
MASTER_PORT=${MASTER_PORT:-29500}

MAX_STEPS=${MAX_STEPS:-100}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((TOTAL_PROCS * MICRO_BATCH_SIZE))}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-2}

MODEL_PATH=${MODEL_PATH:-/workspace/model/Qwen3-VL-30B-A3B-Instruct}
DATA_PATH=${DATA_PATH:-/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards}

DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE:-1}
DP_SHARD_SIZE=${DP_SHARD_SIZE:-${TOTAL_PROCS}}
EP_SIZE=${EP_SIZE:-${TOTAL_PROCS}}
NUM_MOE_LAYERS=${NUM_MOE_LAYERS:-48}
MOE_IMPL=${MOE_IMPL:-fused_npu}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
MOE_MONITOR_INTERVAL=${MOE_MONITOR_INTERVAL:-1}

VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-3}
VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N:-10}
VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-0}
VEOMNI_FULL_PROFILE_WITH_BACKWARD=${VEOMNI_FULL_PROFILE_WITH_BACKWARD:-1}
VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE:-0}
VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS:-0}

if [[ "${NNODES}" -ne 8 ]]; then
  echo "This launcher is for 8 nodes; NNODES=${NNODES}" >&2
  exit 2
fi

if [[ ! -f "${KEY}" ]]; then
  echo "SSH key not found: ${KEY}" >&2
  exit 2
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

if [[ "${START_CONTAINERS}" == "1" ]]; then
  echo "Starting/verifying 8-node containers..."
  bash scripts/profile/start_npu_profile_containers_8node.sh
fi

mkdir -p "pretrain_runs/${RUN_NAME}"

ssh_base=(ssh -i "${KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}")

echo "RUN_NAME=${RUN_NAME}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} TOTAL_PROCS=${TOTAL_PROCS}"
echo "MAX_STEPS=${MAX_STEPS} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} MAX_SEQ_LEN=${MAX_SEQ_LEN}"
echo "DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE} DP_SHARD_SIZE=${DP_SHARD_SIZE} EP_SIZE=${EP_SIZE}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo

for rank in 0 1 2 3 4 5 6 7; do
  host=${HOST_BY_RANK[${rank}]}
  container=${CONTAINER_PREFIX}${rank}
  echo "===== launch rank${rank} on ${host} (${container}) ====="

  "${ssh_base[@]}" root@"${host}" \
    "RUN_NAME=${RUN_NAME}" \
    "NODE_RANK=${rank}" \
    "CONTAINER=${container}" \
    "TRAIN_SCRIPT=${TRAIN_SCRIPT}" \
    "NNODES=${NNODES}" \
    "NPROC_PER_NODE=${NPROC_PER_NODE}" \
    "MASTER_ADDR=${MASTER_ADDR}" \
    "MASTER_PORT=${MASTER_PORT}" \
    "MAX_STEPS=${MAX_STEPS}" \
    "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}" \
    "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
    "MAX_SEQ_LEN=${MAX_SEQ_LEN}" \
    "DATA_NUM_WORKERS=${DATA_NUM_WORKERS}" \
    "DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR}" \
    "MODEL_PATH=${MODEL_PATH}" \
    "DATA_PATH=${DATA_PATH}" \
    "DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE}" \
    "DP_SHARD_SIZE=${DP_SHARD_SIZE}" \
    "EP_SIZE=${EP_SIZE}" \
    "NUM_MOE_LAYERS=${NUM_MOE_LAYERS}" \
    "MOE_IMPL=${MOE_IMPL}" \
    "ATTN_IMPL=${ATTN_IMPL}" \
    "MOE_MONITOR_INTERVAL=${MOE_MONITOR_INTERVAL}" \
    "VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE}" \
    "VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP}" \
    "VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N}" \
    "VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS}" \
    "VEOMNI_FULL_PROFILE_WITH_BACKWARD=${VEOMNI_FULL_PROFILE_WITH_BACKWARD}" \
    "VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE}" \
    "VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS}" \
    'bash -s' <<'REMOTE'
set -euo pipefail

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "Container is not running on $(hostname): ${CONTAINER}" >&2
  echo "Run with START_CONTAINERS=1 or start containers first." >&2
  exit 3
fi

docker exec "${CONTAINER}" bash -c "mkdir -p /workspace/task3/VeOmni-0.1.11/pretrain_runs/${RUN_NAME}"

docker exec -d \
  -e RUN_NAME="${RUN_NAME}" \
  -e NODE_RANK="${NODE_RANK}" \
  -e NNODES="${NNODES}" \
  -e NPROC_PER_NODE="${NPROC_PER_NODE}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e MAX_STEPS="${MAX_STEPS}" \
  -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
  -e GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  -e MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  -e DATA_NUM_WORKERS="${DATA_NUM_WORKERS}" \
  -e DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e DATA_PATH="${DATA_PATH}" \
  -e DP_REPLICATE_SIZE="${DP_REPLICATE_SIZE}" \
  -e DP_SHARD_SIZE="${DP_SHARD_SIZE}" \
  -e EP_SIZE="${EP_SIZE}" \
  -e NUM_MOE_LAYERS="${NUM_MOE_LAYERS}" \
  -e MOE_IMPL="${MOE_IMPL}" \
  -e ATTN_IMPL="${ATTN_IMPL}" \
  -e MOE_MONITOR_INTERVAL="${MOE_MONITOR_INTERVAL}" \
  -e VEOMNI_FULL_PROFILE_ENABLE="${VEOMNI_FULL_PROFILE_ENABLE}" \
  -e VEOMNI_FULL_PROFILE_START_STEP="${VEOMNI_FULL_PROFILE_START_STEP}" \
  -e VEOMNI_FULL_PROFILE_EVERY_N="${VEOMNI_FULL_PROFILE_EVERY_N}" \
  -e VEOMNI_FULL_PROFILE_RANKS="${VEOMNI_FULL_PROFILE_RANKS}" \
  -e VEOMNI_FULL_PROFILE_WITH_BACKWARD="${VEOMNI_FULL_PROFILE_WITH_BACKWARD}" \
  -e VEOMNI_TORCH_PROFILE_ENABLE="${VEOMNI_TORCH_PROFILE_ENABLE}" \
  -e VEOMNI_MOE_TIMING_SYNC_EVENTS="${VEOMNI_MOE_TIMING_SYNC_EVENTS}" \
  "${CONTAINER}" \
  bash -c "cd /workspace/task3/VeOmni-0.1.11 && bash ${TRAIN_SCRIPT} > pretrain_runs/${RUN_NAME}/rank${NODE_RANK}.log 2>&1"

echo "started rank${NODE_RANK}; log: /home/tzq/VeOmni-0.1.11/pretrain_runs/${RUN_NAME}/rank${NODE_RANK}.log"
REMOTE
done

cat <<EOF

Launched 8-node run:
  RUN_NAME=${RUN_NAME}

Check rank0 log:
  ssh -i ${KEY} root@192.168.0.63 'tail -f /home/tzq/VeOmni-0.1.11/pretrain_runs/${RUN_NAME}/rank0.log'

Check all containers:
  for h in 192.168.0.63 192.168.0.164 192.168.0.45 192.168.0.93 192.168.0.174 192.168.0.213 192.168.0.136 192.168.0.210; do
    ssh -i ${KEY} root@\$h 'docker ps --format "{{.Names}} {{.Status}}" | grep tzq_npu_profile_rank || true'
  done

After it finishes, summarize from rank0/head container:
  docker exec tzq_npu_profile_rank0 bash -c '
    cd /workspace/task3/VeOmni-0.1.11
    python profile/scripts/summarize_full_timing.py \\
      --run-dir profile/runs/pretrain/${RUN_NAME} \\
      --output-dir profile/processed/pretrain/${RUN_NAME} \\
      --figure-dir profile/figures/pretrain/${RUN_NAME}
  '
EOF
