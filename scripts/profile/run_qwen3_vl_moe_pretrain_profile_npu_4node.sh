#!/usr/bin/env bash
# Qwen3-VL MoE four-node Ascend NPU pretraining with lightweight MoE timing JSONL.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

RUN_NAME=${RUN_NAME:-qwen3vl_a3b_npu_ep32_profile_4step_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/pretrain_runs/${RUN_NAME}"}
MODEL_PATH=${MODEL_PATH:-/workspace/model/Qwen3-VL-30B-A3B-Instruct}
DATA_PATH=${DATA_PATH:-/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards}

NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:?NODE_RANK must be set to 0, 1, 2, or 3}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-192.168.0.63}
MASTER_PORT=${MASTER_PORT:-29572}
TOTAL_PROCS=$((NNODES * NPROC_PER_NODE))

MAX_STEPS=${MAX_STEPS:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-2}
DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE:-1}
DP_SHARD_SIZE=${DP_SHARD_SIZE:-${TOTAL_PROCS}}
EP_SIZE=${EP_SIZE:-32}
NUM_MOE_LAYERS=${NUM_MOE_LAYERS:-48}
MOE_IMPL=${MOE_IMPL:-fused_npu}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}

PROFILE_KIND=${PROFILE_KIND:-pretrain}
VEOMNI_PROFILE_ROOT=${VEOMNI_PROFILE_ROOT:-"${REPO_ROOT}/profile"}
VERL_MOE_PROFILE_DIR=${VERL_MOE_PROFILE_DIR:-"${VEOMNI_PROFILE_ROOT}/runs/${PROFILE_KIND}/${RUN_NAME}"}
VERL_MOE_MONITOR_DIR=${VERL_MOE_MONITOR_DIR:-"${VERL_MOE_PROFILE_DIR}/moe_monitor"}
VERL_MOE_TIMING_DIR=${VERL_MOE_TIMING_DIR:-"${VERL_MOE_PROFILE_DIR}/moe_timing"}
VERL_MOE_TIMING_NUM_LAYERS=${VERL_MOE_TIMING_NUM_LAYERS:-${NUM_MOE_LAYERS}}
VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
VEOMNI_FULL_PROFILE_DIR=${VEOMNI_FULL_PROFILE_DIR:-"${VERL_MOE_PROFILE_DIR}/full_timing"}
VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS:-0}
VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-2}
VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N:-1}
VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-all}
VEOMNI_FULL_PROFILE_WITH_BACKWARD=${VEOMNI_FULL_PROFILE_WITH_BACKWARD:-1}
VEOMNI_FULL_PROFILE_RUN_KIND=${VEOMNI_FULL_PROFILE_RUN_KIND:-pretrain}

# Keep torch_npu.profiler disabled by default; the JSONL event profiler below is
# the low-overhead signal used for all-to-all and expert compute breakdowns.
VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE:-0}
VEOMNI_TORCH_PROFILE_DIR=${VEOMNI_TORCH_PROFILE_DIR:-"${VERL_MOE_PROFILE_DIR}/torch_profiler"}
VEOMNI_TORCH_PROFILE_START_STEP=${VEOMNI_TORCH_PROFILE_START_STEP:-2}
VEOMNI_TORCH_PROFILE_END_STEP=${VEOMNI_TORCH_PROFILE_END_STEP:-3}
VEOMNI_TORCH_PROFILE_RANK0_ONLY=${VEOMNI_TORCH_PROFILE_RANK0_ONLY:-true}
VEOMNI_TORCH_PROFILE_EXPORT_RANKS=${VEOMNI_TORCH_PROFILE_EXPORT_RANKS:-0}
VEOMNI_TORCH_PROFILE_RECORD_SHAPES=${VEOMNI_TORCH_PROFILE_RECORD_SHAPES:-false}
VEOMNI_TORCH_PROFILE_PROFILE_MEMORY=${VEOMNI_TORCH_PROFILE_PROFILE_MEMORY:-false}
VEOMNI_TORCH_PROFILE_WITH_STACK=${VEOMNI_TORCH_PROFILE_WITH_STACK:-false}
VEOMNI_TORCH_PROFILE_WITH_MODULES=${VEOMNI_TORCH_PROFILE_WITH_MODULES:-false}
VEOMNI_TORCH_PROFILE_ACTIVITIES=${VEOMNI_TORCH_PROFILE_ACTIVITIES:-cpu,npu}
VEOMNI_MOE_VALIDATOR_ENABLE=${VEOMNI_MOE_VALIDATOR_ENABLE:-0}
VEOMNI_MOE_VALIDATOR_DIR=${VEOMNI_MOE_VALIDATOR_DIR:-"${VERL_MOE_PROFILE_DIR}/moe_validator"}

mkdir -p "${RUN_ROOT}" "${VERL_MOE_MONITOR_DIR}" "${VERL_MOE_TIMING_DIR}" "${VEOMNI_FULL_PROFILE_DIR}"
if [[ "${VEOMNI_TORCH_PROFILE_ENABLE}" == "1" || "${VEOMNI_TORCH_PROFILE_ENABLE}" == "true" ]]; then
    mkdir -p "${VEOMNI_TORCH_PROFILE_DIR}"
fi
if [[ "${VEOMNI_MOE_VALIDATOR_ENABLE}" == "1" || "${VEOMNI_MOE_VALIDATOR_ENABLE}" == "true" ]]; then
    mkdir -p "${VEOMNI_MOE_VALIDATOR_DIR}"
fi

cat > "${RUN_ROOT}/launch_env_rank${NODE_RANK}.txt" <<EOF
RUN_NAME=${RUN_NAME}
RUN_ROOT=${RUN_ROOT}
MODEL_PATH=${MODEL_PATH}
DATA_PATH=${DATA_PATH}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
NPROC_PER_NODE=${NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
MAX_STEPS=${MAX_STEPS}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
MAX_SEQ_LEN=${MAX_SEQ_LEN}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS}
DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR}
DP_REPLICATE_SIZE=${DP_REPLICATE_SIZE}
DP_SHARD_SIZE=${DP_SHARD_SIZE}
EP_SIZE=${EP_SIZE}
NUM_MOE_LAYERS=${NUM_MOE_LAYERS}
MOE_IMPL=${MOE_IMPL}
ATTN_IMPL=${ATTN_IMPL}
VERL_MOE_PROFILE_DIR=${VERL_MOE_PROFILE_DIR}
VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE}
VEOMNI_TORCH_PROFILE_ENABLE=${VEOMNI_TORCH_PROFILE_ENABLE}
VEOMNI_MOE_VALIDATOR_ENABLE=${VEOMNI_MOE_VALIDATOR_ENABLE}
VEOMNI_MOE_VALIDATOR_DIR=${VEOMNI_MOE_VALIDATOR_DIR}
VEOMNI_MOE_TIMING_SYNC_EVENTS=${VEOMNI_MOE_TIMING_SYNC_EVENTS}
EOF

export NPROC_PER_NODE
export VERL_MOE_PROFILE_DIR VERL_MOE_MONITOR_DIR VERL_MOE_TIMING_DIR VERL_MOE_TIMING_NUM_LAYERS
export VEOMNI_FULL_PROFILE_ENABLE VEOMNI_FULL_PROFILE_DIR VEOMNI_FULL_PROFILE_START_STEP
export VEOMNI_FULL_PROFILE_EVERY_N VEOMNI_FULL_PROFILE_RANKS VEOMNI_FULL_PROFILE_WITH_BACKWARD VEOMNI_FULL_PROFILE_RUN_KIND
export VEOMNI_TORCH_PROFILE_EXPORT_RANKS VEOMNI_TORCH_PROFILE_ACTIVITIES
export VEOMNI_MOE_VALIDATOR_ENABLE VEOMNI_MOE_VALIDATOR_DIR
export VEOMNI_MOE_TIMING_SYNC_EVENTS
export TOKENIZERS_PARALLELISM=false
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-16}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}

echo "Run root: ${RUN_ROOT}"
echo "Profile dir: ${VERL_MOE_PROFILE_DIR}"
echo "Full timing dir: ${VEOMNI_FULL_PROFILE_DIR} (enabled=${VEOMNI_FULL_PROFILE_ENABLE})"
echo "Torch profiler dir: ${VEOMNI_TORCH_PROFILE_DIR} (enabled=${VEOMNI_TORCH_PROFILE_ENABLE})"
echo "MoE timing sync events: ${VEOMNI_MOE_TIMING_SYNC_EVENTS}"
echo "torchrun: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
    --model.model_path "${MODEL_PATH}" \
    --model.ops_implementation.moe_implementation "${MOE_IMPL}" \
    --model.ops_implementation.attn_implementation "${ATTN_IMPL}" \
    --model.ops_implementation.cross_entropy_loss_implementation npu \
    --model.ops_implementation.rms_norm_implementation npu \
    --model.ops_implementation.rotary_pos_emb_implementation npu \
    --model.ops_implementation.rotary_pos_emb_vision_implementation npu \
    --model.ops_implementation.swiglu_mlp_implementation eager \
    --model.ops_implementation.load_balancing_loss_implementation eager \
    --data.train_path "${DATA_PATH}" \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.dataloader.num_workers "${DATA_NUM_WORKERS}" \
    --data.dataloader.prefetch_factor "${DATA_PREFETCH_FACTOR}" \
    --data.max_seq_len "${MAX_SEQ_LEN}" \
    --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
    --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --train.max_steps "${MAX_STEPS}" \
    --train.num_train_epochs 1 \
    --train.accelerator.dp_replicate_size "${DP_REPLICATE_SIZE}" \
    --train.accelerator.dp_shard_size "${DP_SHARD_SIZE}" \
    --train.accelerator.ep_size "${EP_SIZE}" \
    --train.moe_load_balance_monitor_interval 1 \
    --train.profile.enable "${VEOMNI_TORCH_PROFILE_ENABLE}" \
    --train.profile.start_step "${VEOMNI_TORCH_PROFILE_START_STEP}" \
    --train.profile.end_step "${VEOMNI_TORCH_PROFILE_END_STEP}" \
    --train.profile.trace_dir "${VEOMNI_TORCH_PROFILE_DIR}" \
    --train.profile.rank0_only "${VEOMNI_TORCH_PROFILE_RANK0_ONLY}" \
    --train.profile.record_shapes "${VEOMNI_TORCH_PROFILE_RECORD_SHAPES}" \
    --train.profile.profile_memory "${VEOMNI_TORCH_PROFILE_PROFILE_MEMORY}" \
    --train.profile.with_stack "${VEOMNI_TORCH_PROFILE_WITH_STACK}" \
    --train.profile.with_modules "${VEOMNI_TORCH_PROFILE_WITH_MODULES}" \
    --train.wandb.enable false \
    --train.checkpoint.output_dir "${RUN_ROOT}/ckpts" \
    --train.checkpoint.save_steps 0 \
    --train.checkpoint.save_epochs 0 \
    --train.checkpoint.hf_save_steps 0 \
    --train.checkpoint.hf_save_epochs 0 \
    --train.checkpoint.save_hf_weights false \
    "$@"
