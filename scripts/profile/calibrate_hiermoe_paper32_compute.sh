#!/usr/bin/env bash

set -euo pipefail

model=${1:-qwen35_20l}
dataset=${2:-sharegpt4v}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"
paper32_configure_model "${model}"
paper32_configure_dataset "${dataset}"

run_tag=${PAPER32_COMPUTE_CALIBRATION_TAG:-20260730}
run_name=${PAPER32_COMPUTE_CALIBRATION_RUN_NAME:-${paper32_artifact_prefix}_${paper32_model_slug}_${paper32_dataset_slug}_compute_calibration_ep${paper32_world_size}_${run_tag}}
output_artifact=${PAPER32_COMPUTE_CALIBRATION_OUTPUT:-${paper32_compute_calibration_artifact}}
master_port=${PAPER32_COMPUTE_CALIBRATION_MASTER_PORT:-31031}
hccl_port=${PAPER32_COMPUTE_CALIBRATION_HCCL_PORT:-64100}
calibration_step=${PAPER32_COMPUTE_CALIBRATION_STEP:-2}
validation_steps=${PAPER32_COMPUTE_VALIDATION_STEPS:-1}
minimum_steps=$((calibration_step + validation_steps + 1))
max_steps=${PAPER32_COMPUTE_CALIBRATION_STEPS:-${minimum_steps}}
if ((max_steps < minimum_steps)); then
  echo "PAPER32_COMPUTE_CALIBRATION_STEPS=${max_steps} is too small; need at least ${minimum_steps}" >&2
  exit 2
fi

calibration_env=(
  "E2E_VARIANT=cost_model_verify"
  "RUN_NAME_OVERRIDE=${run_name}"
  "MASTER_ADDR_OVERRIDE=${paper32_master_addr}"
  "MASTER_PORT=${master_port}"
  "HCCL_IF_BASE_PORT=${hccl_port}"
  "NNODES_OVERRIDE=${paper32_nnodes}"
  "NPROC_PER_NODE_OVERRIDE=${paper32_nproc_per_node}"
  "DP_SHARD_SIZE_OVERRIDE=${paper32_world_size}"
  "EP_SIZE_OVERRIDE=${paper32_world_size}"
  "SSH_KEY_OVERRIDE=${paper32_ssh_key}"
  "MODEL_PATH_OVERRIDE=${paper32_model_path}"
  "MODEL_CONFIG_PATH_OVERRIDE=${paper32_model_path}"
  "TRAIN_ENTRYPOINT_OVERRIDE=${paper32_train_entrypoint}"
  "TRAIN_CONFIG_OVERRIDE=${paper32_train_config}"
  "NUM_MOE_LAYERS_OVERRIDE=${paper32_num_layers}"
  "DATA_PATH_OVERRIDE=${paper32_data_path}"
  "DATA_SOURCE_NAME_OVERRIDE=${paper32_data_source_name}"
  "TRAIN_FREEZE_VIT_OVERRIDE=${paper32_freeze_vit}"
  "MICRO_BATCH_SIZE_OVERRIDE=${paper32_micro_batch_size}"
  "GLOBAL_BATCH_SIZE_OVERRIDE=${paper32_global_batch_size}"
  "MAX_SEQ_LEN_OVERRIDE=${PAPER32_MAX_SEQ_LEN:-4096}"
  "MAX_STEPS_OVERRIDE=${max_steps}"
  "TRAIN_LR_OVERRIDE=${PAPER32_LR:-}"
  "HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP_OVERRIDE=${calibration_step}"
  "VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES=${PAPER32_EXPORT_COST_MODEL_SAMPLES:-1}"
  "VEOMNI_HIERMOE_COST_MODEL_VALIDATION_STEPS=${validation_steps}"
  "FULL_PROFILE_ENABLE_OVERRIDE=0"
  "FULL_PROFILE_START_STEP_OVERRIDE=99"
  "HIERMOE_REDUNDANT_SLOTS_OVERRIDE=${paper32_redundant_slots}"
  "HIERMOE_GREEDY_MAX_COPIES_OVERRIDE=8"
  "HIERMOE_PERF_MODEL_PATH_OVERRIDE=${paper32_perf_model_container}"
)
for ((node_rank = 0; node_rank < paper32_nnodes; ++node_rank)); do
  calibration_env+=(
    "RANK${node_rank}_HOST_OVERRIDE=${paper32_hosts[${node_rank}]}"
    "RANK${node_rank}_CONTAINER_OVERRIDE=${paper32_container_name}"
  )
done
env "${calibration_env[@]}" bash "${script_dir}/launch_hiermoe_greedy_e2e_4node.sh"

python "${script_dir}/extract_hiermoe_compute_calibration.py" \
  --log "${paper32_host_root}/${run_name}_rank0.host.log" \
  --output "${paper32_source_root}/${output_artifact}" \
  --model "${paper32_model_slug}" \
  --ep-size "${paper32_world_size}" \
  --micro-batch-size "${paper32_micro_batch_size}" \
  --global-batch-size "${paper32_global_batch_size}" \
  --maximum-sequence-length "${PAPER32_MAX_SEQ_LEN:-4096}" \
  --joint-max-mape "${PAPER32_JOINT_MAX_MAPE:-5.0}" \
  --joint-target "${PAPER32_JOINT_TARGET:-network}" \
  --source-run "${run_name}"

echo "compute calibration completed: ${run_name}"
