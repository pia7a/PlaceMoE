#!/usr/bin/env bash
set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
launcher=${source_root}/scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh
variant=${E2E_VARIANT:-greedy}
master_port=${MASTER_PORT:-29950}
run_suffix=${RUN_SUFFIX:-20260722}
swap_interval=${HIERMOE_SWAP_INTERVAL_OVERRIDE:-1}
max_steps=${MAX_STEPS_OVERRIDE:-6}
data_path=${DATA_PATH_OVERRIDE:-/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards}
data_num_workers=${DATA_NUM_WORKERS_OVERRIDE:-4}
data_prefetch_factor=${DATA_PREFETCH_FACTOR_OVERRIDE:-2}
perf_model_path=${HIERMOE_PERF_MODEL_PATH_OVERRIDE:-/workspace/output/hiermoe_perf_model_c009_ep32_20260720/v2/hiermoe_perf_model.json}
greedy_copy_cap=${HIERMOE_GREEDY_MAX_COPIES_OVERRIDE:-4}
greedy_adaptive_topk=${HIERMOE_GREEDY_ADAPTIVE_TOPK_OVERRIDE:-0}
greedy_adaptive_topk_initial=${HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL_OVERRIDE:-32}
greedy_adaptive_topk_strict=${HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT_OVERRIDE:-0}
capture_routes=${HIERMOE_CAPTURE_ROUTES:-0}
capture_mode=${HIERMOE_CAPTURE_MODE_OVERRIDE:-local}
capture_step=${HIERMOE_CAPTURE_STEP_OVERRIDE:--1}
capture_layer=${HIERMOE_CAPTURE_LAYER_OVERRIDE:-}
capture_call=${HIERMOE_CAPTURE_CALL_OVERRIDE:-0}
debug_copy_stats=${HIERMOE_DEBUG_COPY_STATS_OVERRIDE:-0}
debug_copy_layers=${HIERMOE_DEBUG_COPY_LAYERS_OVERRIDE:-1}
debug_copy_groups=${HIERMOE_DEBUG_COPY_GROUPS_OVERRIDE:-2}
key=/home/tzq/KeyPair-3bce.pem

hiermoe_enable=false
dedup_only=0
token_dedup=false
expert_swap=false
max_pairs=0
redundant_slots=0
replica_rounds=0
fixed_r2=0
fixed_pipeline=false
swap_mode=step
swap_selector=current_joint

case "${variant}" in
  baseline)
    ;;
  dedup)
    hiermoe_enable=
    dedup_only=1
    token_dedup=true
    ;;
  fixed_r2)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    ;;
  fixed_r2_greedy_sync)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  fixed_r2_pipeline_grad)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  r2_pipeline)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  r2_planner)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    swap_mode=layer
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  greedy)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=1
    replica_rounds=32
    swap_mode=layer
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  *)
    echo "unsupported E2E_VARIANT=${variant}" >&2
    exit 2
    ;;
esac

run_name=qwen3vl_greedy_ep32_mb4_gbs128_${variant}_${max_steps}step_${run_suffix}
common_env=(
  -e "PYTHONPATH=${source_root}"
  -e "RUN_NAME=${run_name}"
  -e "RUN_ROOT=${source_root}/pretrain_runs/${run_name}"
  -e "MODEL_PATH=/workspace/model/Qwen3-VL-30B-A3B-Instruct"
  -e "DATA_PATH=${data_path}"
  -e "DATA_NUM_WORKERS=${data_num_workers}"
  -e "DATA_PREFETCH_FACTOR=${data_prefetch_factor}"
  -e "NNODES=4"
  -e "NPROC_PER_NODE=8"
  -e "MASTER_ADDR=192.168.0.55"
  -e "MASTER_PORT=${master_port}"
  -e "MAX_STEPS=${max_steps}"
  -e "MICRO_BATCH_SIZE=4"
  -e "GLOBAL_BATCH_SIZE=128"
  -e "MAX_SEQ_LEN=4096"
  -e "DP_REPLICATE_SIZE=1"
  -e "DP_SHARD_SIZE=32"
  -e "EP_SIZE=32"
  -e "HIERMOE_ENABLE=${hiermoe_enable}"
  -e "HIERMOE_DEDUP_ONLY=${dedup_only}"
  -e "HIERMOE_TOKEN_DEDUP=${token_dedup}"
  -e "HIERMOE_COMMUNICATION_MODE=hierarchical"
  -e "HIERMOE_EXPERT_SWAP=${expert_swap}"
  -e "HIERMOE_EXPERT_SWAP_INTERVAL=${swap_interval}"
  -e "HIERMOE_LOG_INTERVAL=1"
  -e "HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=${max_pairs}"
  -e "HIERMOE_EXPERT_SWAP_SELECTOR=${swap_selector}"
  -e "HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE=${redundant_slots}"
  -e "HIERMOE_GREEDY_MAX_COPIES_PER_EXPERT=${greedy_copy_cap}"
  -e "HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS=${replica_rounds}"
  -e "HIERMOE_EXPERT_SWAP_MODE=${swap_mode}"
  -e "HIERMOE_FIXED_PIPELINE_OVERLAP=${fixed_pipeline}"
  -e "HIERMOE_FIT_PERF_MODEL_ON_STARTUP=0"
  -e "HIERMOE_PERF_MODEL_PATH=${perf_model_path}"
  -e "VEOMNI_HIERMOE_FIXED_R2_LAYOUT=${fixed_r2}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK=${greedy_adaptive_topk}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL=${greedy_adaptive_topk_initial}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT=${greedy_adaptive_topk_strict}"
  -e "VEOMNI_HIERMOE_FULL_ROUTE_GATHER_MAX_TOKENS=16384"
  -e "VEOMNI_FULL_PROFILE_ENABLE=1"
  -e "VEOMNI_FULL_PROFILE_START_STEP=3"
  -e "VEOMNI_FULL_PROFILE_EVERY_N=1"
  -e "VEOMNI_FULL_PROFILE_RANKS=0"
  -e "VEOMNI_HIERMOE_INTERNAL_TIMING=0"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS=${debug_copy_stats}"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS=${debug_copy_layers}"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS=${debug_copy_groups}"
  -e "VEOMNI_TORCH_PROFILE_ENABLE=0"
  -e "VEOMNI_MOE_TIMING_SYNC_EVENTS=0"
)

if [[ "${capture_routes}" == "1" ]]; then
  capture_root=${source_root}/route_captures/${run_name}
  common_env+=(
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE=${capture_mode}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH=${capture_root}/step{step:04d}/layer{layer_index:02d}_call{call}_rank{rank:02d}.pt"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP=${capture_step}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_LAYER=${capture_layer}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL=${capture_call}"
  )
fi

launch_remote() {
  local host=$1
  local node_rank=$2
  local container=$3
  ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    docker exec \
    "${common_env[@]}" \
    -e "NODE_RANK=${node_rank}" \
    -w "${source_root}" \
    "${container}" \
    bash "${launcher}" \
    >"${host_root}/${run_name}_rank${node_rank}.host.log" 2>&1
}

launch_remote 192.168.0.190 1 tzq_npu_static_r2_rank1_20260720 &
rank1_pid=$!
launch_remote 192.168.0.109 2 tzq_npu_static_r2_rank2_20260719 &
rank2_pid=$!
launch_remote 192.168.0.9 3 tzq_npu_static_r2_rank3_20260719 &
rank3_pid=$!

docker exec \
  "${common_env[@]}" \
  -e NODE_RANK=0 \
  -w "${source_root}" \
  tzq_npu_coremoe_verify_20260717 \
  bash "${launcher}" \
  >"${host_root}/${run_name}_rank0.host.log" 2>&1
rank0_rc=$?

wait "${rank1_pid}"
rank1_rc=$?
wait "${rank2_pid}"
rank2_rc=$?
wait "${rank3_pid}"
rank3_rc=$?
printf 'run=%s rank0_rc=%s rank1_rc=%s rank2_rc=%s rank3_rc=%s\n' \
  "${run_name}" "${rank0_rc}" "${rank1_rc}" "${rank2_rc}" "${rank3_rc}"
exit $((rank0_rc || rank1_rc || rank2_rc || rank3_rc))
