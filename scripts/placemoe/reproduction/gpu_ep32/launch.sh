#!/usr/bin/env bash
# Launch one HierMoE E2E case on a configured four-node EP32 testbed.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
node_launcher=${script_dir}/run_training_node.sh
# shellcheck source=common.sh
source "${script_dir}/common.sh"
remote_root=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repo_root}}
remote_node_launcher=${remote_root}/scripts/placemoe/reproduction/gpu_ep32/run_training_node.sh
# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"

variant=${E2E_VARIANT:-baseline}
run_name=${RUN_NAME_OVERRIDE:-gpu32_${variant}_$(date +%Y%m%d_%H%M%S)}
master_addr=${MASTER_ADDR_OVERRIDE:-${repro_master_addr}}
repro_require_value PLACEMOE_REPRO_MASTER_ADDR "${master_addr}"
master_port=${MASTER_PORT:-29500}
max_steps=${MAX_STEPS_OVERRIDE:-2}
model_path=${MODEL_PATH_OVERRIDE:-${PLACEMOE_REPRO_QWEN3VL_MODEL_PATH:-}}
data_path=${DATA_PATH_OVERRIDE:-${PLACEMOE_REPRO_SHAREGPT4V_DATA_PATH:-}}
repro_require_value PLACEMOE_REPRO_QWEN3VL_MODEL_PATH "${model_path}"
repro_require_value PLACEMOE_REPRO_SHAREGPT4V_DATA_PATH "${data_path}"
data_source_name=${DATA_SOURCE_NAME_OVERRIDE:-sharegpt4v_sft}
config_path=${CONFIG_PATH_OVERRIDE:-configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml}
num_moe_layers=${NUM_MOE_LAYERS_OVERRIDE:-48}
micro_batch_size=${MICRO_BATCH_SIZE_OVERRIDE:-1}
global_batch_size=${GLOBAL_BATCH_SIZE_OVERRIDE:-32}
freeze_vit=${TRAIN_FREEZE_VIT_OVERRIDE:-false}
perf_model_path=${HIERMOE_PERF_MODEL_PATH_OVERRIDE:-}
fit_perf_model=${HIERMOE_FIT_PERF_MODEL_ON_STARTUP_OVERRIDE:-false}
redundant_slots=${HIERMOE_REDUNDANT_SLOTS_OVERRIDE:-4}
replay_path=${HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE:-}
initial_layout_path=${HIERMOE_INITIAL_LAYOUT_OVERRIDE:-${replay_path}}
placemoe_config=${PLACEMOE_CONFIG_OVERRIDE:-${VEOMNI_PLACEMOE_CONFIG:-}}
grad_mode=${HIERMOE_ABLATION_GRAD_MODE_OVERRIDE:-blocking}

hiermoe_enable=false
token_dedup=true
expert_swap=false
max_pairs=0
selector=current_joint
active_redundant_slots=0
search_rounds=0
fixed_pipeline=false
fixed_r2=0
force_fixed_r2_mirrored_remap=0
replay_mode=off
forward_reuse_cover=0
forward_reuse_cover_patch_remap=0
forward_reuse_cover_empty_seeding=0
cost_model_verify=0
hiermoe_internal_timing=${HIERMOE_INTERNAL_TIMING_OVERRIDE:-0}
online_freeze_calibration_step=${HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP_OVERRIDE:-1}

case "${variant}" in
  baseline)
    ;;
  dedup)
    hiermoe_enable=true
    ;;
  hiermoe_exact_p1)
    hiermoe_enable=true
    expert_swap=true
    max_pairs=1
    selector=hiermoe_exact_p1
    ;;
  replica)
    hiermoe_enable=true
    expert_swap=true
    selector=hiermoe_greedy_cover_p1
    active_redundant_slots=${redundant_slots}
    fixed_r2=1
    force_fixed_r2_mirrored_remap=1
    ;;
  fixed_r2_mirrored_pipeline_grad)
    hiermoe_enable=true
    expert_swap=true
    selector=hiermoe_greedy_cover_p1
    active_redundant_slots=${redundant_slots}
    fixed_pipeline=true
    fixed_r2=1
    force_fixed_r2_mirrored_remap=1
    ;;
  cost_model_verify)
    hiermoe_enable=true
    expert_swap=true
    selector=hiermoe_greedy_cover_p1
    active_redundant_slots=${redundant_slots}
    search_rounds=0
    fixed_r2=1
    force_fixed_r2_mirrored_remap=1
    grad_mode=${HIERMOE_ABLATION_GRAD_MODE_OVERRIDE:-blocking}
    cost_model_verify=1
    hiermoe_internal_timing=1
    ;;
  static_layout)
    if [[ -z "${initial_layout_path}" && -z "${placemoe_config}" ]]; then
      echo "static_layout requires HIERMOE_INITIAL_LAYOUT_OVERRIDE or PLACEMOE_CONFIG_OVERRIDE" >&2
      exit 2
    fi
    hiermoe_enable=true
    expert_swap=true
    selector=hiermoe_greedy_cover_p1
    active_redundant_slots=${redundant_slots}
    grad_mode=${HIERMOE_ABLATION_GRAD_MODE_OVERRIDE:-blocking}
    ;;
  hierarchical_full_static)
    if [[ -z "${replay_path}" || -z "${initial_layout_path}" ]]; then
      echo "hierarchical_full_static requires replay and initial layout paths" >&2
      exit 2
    fi
    hiermoe_enable=true
    expert_swap=true
    selector=hiermoe_greedy_cover_p1
    active_redundant_slots=${redundant_slots}
    fixed_pipeline=true
    replay_mode=static
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    forward_reuse_cover_empty_seeding=1
    grad_mode=${HIERMOE_ABLATION_GRAD_MODE_OVERRIDE:-hidden}
    ;;
  *)
    echo "unsupported E2E_VARIANT=${variant}" >&2
    exit 2
    ;;
esac

if [[ "${expert_swap}" == "true" && -z "${perf_model_path}" && "${fit_perf_model}" != "true" \
  && "${replay_mode}" != "static" && "${fixed_r2}" != "1" \
  && -z "${initial_layout_path}" && -z "${placemoe_config}" ]]; then
  echo "${variant} requires HIERMOE_PERF_MODEL_PATH_OVERRIDE" >&2
  exit 2
fi

remote_specs=("${repro_remote_specs[@]}")

common_env=(
  "PYTHON=${repro_python}"
  "PYTHONPATH=${repo_root}"
  "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF_OVERRIDE:-backend:cudaMallocAsync}"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING_OVERRIDE:-1}"
  "NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT_OVERRIDE:-1}"
  "RUN_NAME=${run_name}"
  "RUN_ROOT=${repo_root}/pretrain_runs/${run_name}"
  "MODEL_PATH=${model_path}"
  "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH_OVERRIDE:-${model_path}}"
  "DATA_PATH=${data_path}"
  "DATA_SOURCE_NAME=${data_source_name}"
  "CONFIG_PATH=${config_path}"
  "NNODES=4"
  "NPROC_PER_NODE=8"
  "MASTER_ADDR=${master_addr}"
  "MASTER_PORT=${master_port}"
  "MAX_STEPS=${max_steps}"
  "EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS_OVERRIDE:-500}"
  "MICRO_BATCH_SIZE=${micro_batch_size}"
  "GLOBAL_BATCH_SIZE=${global_batch_size}"
  "MAX_SEQ_LEN=${MAX_SEQ_LEN_OVERRIDE:-4096}"
  "TRAIN_FREEZE_VIT=${freeze_vit}"
  "NUM_MOE_LAYERS=${num_moe_layers}"
  "DP_SHARD_SIZE=32"
  "EP_SIZE=32"
  "MOE_IMPL=${MOE_IMPL_OVERRIDE:-fused_triton}"
  "MOE_MONITOR_INTERVAL=${MOE_MONITOR_INTERVAL_OVERRIDE:-1}"
  "ATTN_IMPL=${ATTN_IMPL_OVERRIDE:-flash_attention_2}"
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME_OVERRIDE:-${repro_nccl_socket_ifname}}"
  "CUDA_LIB_PATH=${repro_cuda_lib_path}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE_OVERRIDE:-0}"
  "HIERMOE_ENABLE=${hiermoe_enable}"
  "HIERMOE_TOKEN_DEDUP=${token_dedup}"
  "HIERMOE_COMMUNICATION_MODE=hierarchical"
  "HIERMOE_HIERARCHY_GROUP_SIZES=${HIERMOE_HIERARCHY_GROUP_SIZES_OVERRIDE:-${repro_hierarchy_group_sizes}}"
  "HIERMOE_EXPERT_SWAP=${expert_swap}"
  "HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=${max_pairs}"
  "HIERMOE_EXPERT_SWAP_SELECTOR=${selector}"
  "HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE=${active_redundant_slots}"
  "HIERMOE_GREEDY_MAX_COPIES_PER_EXPERT=8"
  "HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS=${search_rounds}"
  "HIERMOE_EXPERT_SWAP_MODE=step"
  "HIERMOE_FIXED_PIPELINE_OVERLAP=${fixed_pipeline}"
  "HIERMOE_FIT_PERF_MODEL_ON_STARTUP=${fit_perf_model}"
  "HIERMOE_PERF_MODEL_PATH=${perf_model_path}"
  "VEOMNI_HIERMOE_COST_MODEL_VERIFY=${cost_model_verify}"
  "VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES=${VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES_OVERRIDE:-${VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES:-0}}"
  "VEOMNI_HIERMOE_INTERNAL_TIMING=${hiermoe_internal_timing}"
  "VEOMNI_HIERMOE_FINAL_ACCUM_DEBUG=${HIERMOE_FINAL_ACCUM_DEBUG_OVERRIDE:-0}"
  "VEOMNI_HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP=${online_freeze_calibration_step}"
  "VEOMNI_HIERMOE_FIXED_R2_LAYOUT=${fixed_r2}"
  "VEOMNI_HIERMOE_FORCE_FIXED_R2_MIRRORED_REMAP=${force_fixed_r2_mirrored_remap}"
  "VEOMNI_HIERMOE_ABLATION_REPLAY_MODE=${replay_mode}"
  "VEOMNI_HIERMOE_ABLATION_REPLAY_PATH=${replay_path}"
  "VEOMNI_HIERMOE_INITIAL_LAYOUT=${initial_layout_path}"
  "VEOMNI_PLACEMOE_CONFIG=${placemoe_config}"
  "VEOMNI_HIERMOE_FORWARD_REUSE_COVER=${forward_reuse_cover}"
  "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP=${forward_reuse_cover_patch_remap}"
  "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_EMPTY_SEEDING=${forward_reuse_cover_empty_seeding}"
  "VEOMNI_HIERMOE_ABLATION_GRAD_MODE=${grad_mode}"
  "VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE=blocking"
  "VEOMNI_FULL_PROFILE_ENABLE=${FULL_PROFILE_ENABLE_OVERRIDE:-0}"
  "VEOMNI_TRUE_STEP_TIME=1"
  "VEOMNI_HIERMOE_CUDA_SEGMENT_SUM=${VEOMNI_HIERMOE_CUDA_SEGMENT_SUM_OVERRIDE:-1}"
  "VEOMNI_MOE_TIMING_ENABLE=${VEOMNI_MOE_TIMING_ENABLE_OVERRIDE:-1}"
  "VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS=${VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS_OVERRIDE:-0}"
  "PLACEMOE_REPRO_EXPECTED_ACCELERATOR=${repro_expected_accelerator}"
  "PLACEMOE_REPRO_EXPECTED_TORCH=${repro_expected_torch}"
  "PLACEMOE_REPRO_EXPECTED_CUDA=${repro_expected_cuda}"
  "PLACEMOE_REPRO_EXPECTED_NCCL=${repro_expected_nccl}"
  "PLACEMOE_REPRO_EXPECTED_TRITON=${repro_expected_triton}"
  "VEOMNI_FULL_PROFILE_START_STEP=${FULL_PROFILE_START_STEP_OVERRIDE:-11}"
  "VEOMNI_FULL_PROFILE_EVERY_N=1"
  "VEOMNI_FULL_PROFILE_RANKS=0"
  "VEOMNI_TORCH_PROFILE_ENABLE=${TORCH_PROFILE_ENABLE_OVERRIDE:-0}"
  "VEOMNI_TORCH_PROFILE_START_STEP=${TORCH_PROFILE_START_STEP_OVERRIDE:-11}"
  "VEOMNI_TORCH_PROFILE_END_STEP=${TORCH_PROFILE_END_STEP_OVERRIDE:-12}"
  "VEOMNI_TORCH_PROFILE_RANK0_ONLY=${TORCH_PROFILE_RANK0_ONLY_OVERRIDE:-true}"
  "GPU_PREFLIGHT_ONLY=${GPU_PREFLIGHT_ONLY:-0}"
  "HIERMOE_CAPTURE_ROUTES=${HIERMOE_CAPTURE_ROUTES:-0}"
  "HIERMOE_CAPTURE_STEP=${HIERMOE_CAPTURE_STEP_OVERRIDE:--1}"
  "HIERMOE_CAPTURE_ROOT=${HIERMOE_CAPTURE_ROOT:-}"
)

quote_env() {
  local target_root=${1:?target root is required}
  local value
  for value in "${common_env[@]}"; do
    value=${value//"${repo_root}"/"${target_root}"}
    printf '%q ' "${value}"
  done
}

launch_remote() {
  local port=$1
  local rank=$2
  local remote_command
  remote_command="cd $(printf '%q' "${remote_root}") && env $(quote_env "${remote_root}") NODE_RANK=${rank} bash $(printf '%q' "${remote_node_launcher}")"
  repro_ssh "${port}" "${remote_command}" \
    >"${repo_root}/pretrain_runs/${run_name}_rank${rank}.host.log" 2>&1
}

mkdir -p "${repo_root}/pretrain_runs"
pids=()
for spec in "${remote_specs[@]}"; do
  port=${spec%%:*}
  rank=${spec##*:}
  launch_remote "${port}" "${rank}" &
  pids+=("$!")
done

set +e
env "${common_env[@]}" NODE_RANK=0 bash "${node_launcher}" \
  >"${repo_root}/pretrain_runs/${run_name}_rank0.host.log" 2>&1
rank0_rc=$?
remote_rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || remote_rc=1
done
set -e

printf 'run=%s rank0_rc=%s remote_rc=%s\n' "${run_name}" "${rank0_rc}" "${remote_rc}"
if ((rank0_rc != 0 || remote_rc != 0)); then
  exit 1
fi
