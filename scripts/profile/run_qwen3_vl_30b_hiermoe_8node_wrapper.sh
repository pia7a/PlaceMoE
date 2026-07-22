#!/usr/bin/env bash
# Launch the Qwen3-VL-30B HierMoE NPU profile run across the eight profile
# containers from the jump/head node.
#
# Run this script on 101.245.99.128. It starts one docker exec per rank and
# forwards the same environment to every node so RUN_NAME and MASTER_PORT stay
# consistent.
#
# Examples:
#   bash scripts/profile/run_qwen3_vl_30b_hiermoe_8node_wrapper.sh
#   MAX_STEPS=16 bash scripts/profile/run_qwen3_vl_30b_hiermoe_8node_wrapper.sh
#   MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=128 MASTER_PORT=29600 bash scripts/profile/run_qwen3_vl_30b_hiermoe_8node_wrapper.sh
#   HIERMOE_FIT_PERF_MODEL_ON_STARTUP=1 bash scripts/profile/run_qwen3_vl_30b_hiermoe_8node_wrapper.sh --train.hiermoe.log_interval 1
#
# Set CHECK_IDLE=0 to skip NPU occupancy checks. Set AUTO_START_CONTAINERS=1 to
# docker start existing stopped profile containers before launching.
# Set CHECK_ONLY=1 to run only preflight checks. Set DRY_RUN=1 to print the
# docker/ssh launch commands after preflight checks without executing them.

set -euo pipefail

KEY=${KEY:-/home/tzq/KeyPair-3bce.pem}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-tzq_npu_profile_rank}
REPO_IN_CONTAINER=${REPO_IN_CONTAINER:-/workspace/task3/VeOmni-0.1.11}
LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh}
CHECK_IDLE=${CHECK_IDLE:-1}
AUTO_START_CONTAINERS=${AUTO_START_CONTAINERS:-0}
CHECK_ONLY=${CHECK_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

MASTER_ADDR=${MASTER_ADDR:-192.168.0.63}
MASTER_PORT=${MASTER_PORT:-29500}

NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAX_STEPS=${MAX_STEPS:-6}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-2}

MODEL_PATH=${MODEL_PATH:-/workspace/model/Qwen3-VL-30B-A3B-Instruct}
DATA_PATH=${DATA_PATH:-/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards}

DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE:-1}
DP_SHARD_SIZE=${DP_SHARD_SIZE:-64}
EP_SIZE=${EP_SIZE:-64}
MOE_IMPL=${MOE_IMPL:-fused_npu}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
MOE_MONITOR_INTERVAL=${MOE_MONITOR_INTERVAL:-1}

VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-3}
VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N:-1}
VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-0}
VEOMNI_HIERMOE_INTERNAL_TIMING=${VEOMNI_HIERMOE_INTERNAL_TIMING:-0}
VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE:-0}
VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS:-0}
VEOMNI_MOE_VALIDATOR_ENABLE=${VEOMNI_MOE_VALIDATOR_ENABLE:-0}

HIERMOE_DEDUP_ONLY=${HIERMOE_DEDUP_ONLY:-0}
HIERMOE_ENABLE=${HIERMOE_ENABLE:-true}
HIERMOE_TOKEN_DEDUP=${HIERMOE_TOKEN_DEDUP:-true}
HIERMOE_EXPERT_SWAP=${HIERMOE_EXPERT_SWAP:-true}
HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP=${HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP:-1}
HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=${HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER:-4}
HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE=${HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE:-0}
HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS=${HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS:-}
HIERMOE_EXPERT_SWAP_INTERVAL=${HIERMOE_EXPERT_SWAP_INTERVAL:-1}
HIERMOE_EXPERT_SWAP_MODE=${HIERMOE_EXPERT_SWAP_MODE:-step}
HIERMOE_HIERARCHY_GROUP_SIZES=${HIERMOE_HIERARCHY_GROUP_SIZES:-}
HIERMOE_FIT_PERF_MODEL_ON_STARTUP=${HIERMOE_FIT_PERF_MODEL_ON_STARTUP:-0}
HIERMOE_PERF_MODEL_MASTER_PORT=${HIERMOE_PERF_MODEL_MASTER_PORT:-$((MASTER_PORT + 37))}
HIERMOE_PERF_MODEL_MESSAGE_BYTES_CSV=${HIERMOE_PERF_MODEL_MESSAGE_BYTES_CSV:-67108864,134217728,268435456,536870912}
HIERMOE_PERF_MODEL_WARMUP=${HIERMOE_PERF_MODEL_WARMUP:-2}
HIERMOE_PERF_MODEL_ITERS=${HIERMOE_PERF_MODEL_ITERS:-5}
HIERMOE_PERF_MODEL_MEASURE_LAST_N=${HIERMOE_PERF_MODEL_MEASURE_LAST_N:-3}
HIERMOE_PERF_MODEL_PATH=${HIERMOE_PERF_MODEL_PATH:-}
HIERMOE_PERF_MODEL_DIR=${HIERMOE_PERF_MODEL_DIR:-}

RUN_NAME=${RUN_NAME:-qwen3vl_hiermoe_ep${EP_SIZE}_mb${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_${MAX_STEPS}step_all8_$(date +%Y%m%d_%H%M%S)}

declare -A HOST_BY_RANK=(
  [0]=local
  [1]=192.168.0.164
  [2]=192.168.0.45
  [3]=192.168.0.93
  [4]=192.168.0.174
  [5]=192.168.0.213
  [6]=192.168.0.136
  [7]=192.168.0.210
)

if [[ "${NNODES}" != "8" ]]; then
  echo "This wrapper is for the current 8-node profile cluster; got NNODES=${NNODES}." >&2
  exit 2
fi

TOTAL_PROCS=$((NNODES * NPROC_PER_NODE))
if ((EP_SIZE <= 0 || TOTAL_PROCS % EP_SIZE != 0)); then
  echo "EP_SIZE=${EP_SIZE} must be a positive divisor of NNODES*NPROC_PER_NODE=${TOTAL_PROCS}." >&2
  exit 2
fi
EP_FSDP_SIZE=$((TOTAL_PROCS / EP_SIZE))

if [[ -z "${HIERMOE_HIERARCHY_GROUP_SIZES}" && "${HIERMOE_ENABLE}" != "0" && "${HIERMOE_ENABLE}" != "false" ]]; then
  if ((EP_SIZE > NPROC_PER_NODE && EP_SIZE % NPROC_PER_NODE == 0)); then
    HIERMOE_HIERARCHY_GROUP_SIZES="${NPROC_PER_NODE},${EP_SIZE}"
  elif ((EP_SIZE > 1)); then
    HIERMOE_HIERARCHY_GROUP_SIZES="${EP_SIZE}"
  fi
fi

if ((EP_FSDP_SIZE > 1)) && [[ "${HIERMOE_EXPERT_SWAP}" == "1" || "${HIERMOE_EXPERT_SWAP}" == "true" ]]; then
  if [[ "${HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP}" == "1" || "${HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP}" == "true" ]]; then
    echo "Expert swap requires ep_fsdp_size=1; got ep_fsdp_size=${EP_FSDP_SIZE}. Disabling HIERMOE_EXPERT_SWAP for this run." >&2
    HIERMOE_EXPERT_SWAP=false
  else
    echo "Expert swap requires ep_fsdp_size=1; got ep_fsdp_size=${EP_FSDP_SIZE}. Set HIERMOE_EXPERT_SWAP=false or HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP=1." >&2
    exit 2
  fi
fi

if [[ ! -f "${KEY}" ]]; then
  echo "SSH key not found: ${KEY}" >&2
  exit 2
fi

ssh_base=(ssh -i "${KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}")

quote() {
  printf "%q" "$1"
}

check_container() {
  local rank=$1
  local host=${HOST_BY_RANK[${rank}]}
  local container=${CONTAINER_PREFIX}${rank}
  local command
  command=$(cat <<REMOTE
set -euo pipefail
if ! docker ps -a --format '{{.Names}}' | grep -qx '${container}'; then
  echo 'Missing container: ${container}' >&2
  exit 4
fi
if ! docker ps --format '{{.Names}}' | grep -qx '${container}'; then
  if [[ '${AUTO_START_CONTAINERS}' == '1' ]]; then
    docker start '${container}' >/dev/null
  else
    echo 'Container is not running: ${container}. Set AUTO_START_CONTAINERS=1 or start it first.' >&2
    exit 4
  fi
fi
docker exec '${container}' test -d '${REPO_IN_CONTAINER}'
REMOTE
)
  if [[ "${host}" == "local" ]]; then
    bash -c "${command}"
  else
    "${ssh_base[@]}" root@"${host}" "${command}"
  fi
}

check_idle() {
  local rank=$1
  local host=${HOST_BY_RANK[${rank}]}
  local command
  command=$(cat <<'REMOTE'
set -euo pipefail
process_lines=$(npu-smi info | awk '
  /Process id/ {in_process_table=1; next}
  in_process_table && /\|[[:space:]]*[0-7][[:space:]]+0[[:space:]]+\|[[:space:]]*[0-9]+/ {print}
')
if [[ -n "${process_lines}" ]]; then
  echo "NPU is busy on $(hostname):" >&2
  echo "${process_lines}" >&2
  exit 5
fi
REMOTE
)
  if [[ "${host}" == "local" ]]; then
    bash -c "${command}"
  else
    "${ssh_base[@]}" root@"${host}" "${command}"
  fi
}

make_env_args() {
  local rank=$1
  local keys=(
    RUN_NAME MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE MAX_STEPS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_SEQ_LEN
    DATA_NUM_WORKERS DATA_PREFETCH_FACTOR
    MODEL_PATH DATA_PATH DP_REPLICATE_SIZE DP_SHARD_SIZE EP_SIZE MOE_IMPL ATTN_IMPL MOE_MONITOR_INTERVAL
    VEOMNI_FULL_PROFILE_ENABLE VEOMNI_FULL_PROFILE_START_STEP VEOMNI_FULL_PROFILE_EVERY_N VEOMNI_FULL_PROFILE_RANKS
    VEOMNI_HIERMOE_INTERNAL_TIMING
    VEOMNI_TORCH_PROFILE_ENABLE VEOMNI_MOE_TIMING_SYNC_EVENTS VEOMNI_MOE_VALIDATOR_ENABLE
    HIERMOE_DEDUP_ONLY HIERMOE_ENABLE HIERMOE_TOKEN_DEDUP HIERMOE_EXPERT_SWAP HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER
    HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS
    HIERMOE_AUTO_DISABLE_EXPERT_SWAP_FOR_EP_FSDP HIERMOE_EXPERT_SWAP_INTERVAL HIERMOE_EXPERT_SWAP_MODE HIERMOE_HIERARCHY_GROUP_SIZES HIERMOE_FIT_PERF_MODEL_ON_STARTUP
    HIERMOE_PERF_MODEL_MASTER_PORT HIERMOE_PERF_MODEL_MESSAGE_BYTES_CSV HIERMOE_PERF_MODEL_WARMUP
    HIERMOE_PERF_MODEL_ITERS HIERMOE_PERF_MODEL_MEASURE_LAST_N HIERMOE_PERF_MODEL_PATH HIERMOE_PERF_MODEL_DIR
  )
  local key
  for key in "${keys[@]}"; do
    printf -- "--env %s=%s " "${key}" "$(quote "${!key}")"
  done
  printf -- "--env NODE_RANK=%s " "${rank}"
}

extra_args=()
for arg in "$@"; do
  extra_args+=("$(quote "${arg}")")
done
extra_args_q=""
if ((${#extra_args[@]} > 0)); then
  extra_args_q="${extra_args[*]}"
fi

repo_q=$(quote "${REPO_IN_CONTAINER}")
launch_script_q=$(quote "${LAUNCH_SCRIPT}")
inner_command="set -euo pipefail; cd ${repo_q}; mkdir -p \"pretrain_runs/\${RUN_NAME}\"; bash ${launch_script_q} ${extra_args_q} 2>&1 | tee \"pretrain_runs/\${RUN_NAME}/rank\${NODE_RANK}.log\""

echo "RUN_NAME=${RUN_NAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "MAX_STEPS=${MAX_STEPS} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "TOTAL_PROCS=${TOTAL_PROCS} EP_SIZE=${EP_SIZE} EP_FSDP_SIZE=${EP_FSDP_SIZE} DP_SHARD_SIZE=${DP_SHARD_SIZE}"
echo "HIERMOE_ENABLE=${HIERMOE_ENABLE} HIERMOE_TOKEN_DEDUP=${HIERMOE_TOKEN_DEDUP} HIERMOE_EXPERT_SWAP=${HIERMOE_EXPERT_SWAP}"
echo "HIERMOE_EXPERT_SWAP_MODE=${HIERMOE_EXPERT_SWAP_MODE} HIERMOE_EXPERT_SWAP_INTERVAL=${HIERMOE_EXPERT_SWAP_INTERVAL} HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=${HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER}"
echo "HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE=${HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE} HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS=${HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS:-auto}"
echo "HIERMOE_HIERARCHY_GROUP_SIZES=${HIERMOE_HIERARCHY_GROUP_SIZES} HIERMOE_FIT_PERF_MODEL_ON_STARTUP=${HIERMOE_FIT_PERF_MODEL_ON_STARTUP}"

for rank in 0 1 2 3 4 5 6 7; do
  echo "Checking rank${rank} host=${HOST_BY_RANK[${rank}]} container=${CONTAINER_PREFIX}${rank}"
  if [[ "${CHECK_IDLE}" == "1" ]]; then
    check_idle "${rank}"
  fi
  check_container "${rank}"
done

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Preflight checks passed. CHECK_ONLY=1, not launching training."
  exit 0
fi

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  host=${HOST_BY_RANK[${rank}]}
  container=${CONTAINER_PREFIX}${rank}
  env_args=$(make_env_args "${rank}")
  remote_command="docker exec ${env_args} ${container} bash -c $(quote "${inner_command}")"
  echo "Launching rank${rank} on ${host}:${container}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${host}" == "local" ]]; then
      echo "${remote_command}"
    else
      echo "${ssh_base[*]} root@${host} ${remote_command}"
    fi
    continue
  fi
  if [[ "${host}" == "local" ]]; then
    bash -c "${remote_command}" &
  else
    "${ssh_base[@]}" root@"${host}" "${remote_command}" &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "RUN_NAME=${RUN_NAME}"
exit "${status}"
