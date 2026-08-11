#!/usr/bin/env bash
# Calibrate the offline Ours scorer for one EP32 experiment configuration.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL [CALIBRATION_DATASET]" >&2
  exit 2
fi

model=$1
dataset=${2:-sharegpt4v}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
repro_configure_model "${model}"
repro_configure_dataset "${dataset}"

calibration_model_path=${MODEL_PATH_OVERRIDE:-${repro_model_path}}
calibration_data_path=${DATA_PATH_OVERRIDE:-${repro_data_path}}
calibration_data_source=${DATA_SOURCE_NAME_OVERRIDE:-${repro_data_source_name}}
calibration_micro_batch=${MICRO_BATCH_SIZE_OVERRIDE:-${repro_micro_batch_size}}
calibration_global_batch=${GLOBAL_BATCH_SIZE_OVERRIDE:-${repro_global_batch_size}}
calibration_max_seq_len=${MAX_SEQ_LEN_OVERRIDE:-${PLACEMOE_REPRO_MAX_SEQ_LEN:-4096}}
calibration_moe_impl=${MOE_IMPL_OVERRIDE:-fused_triton}
calibration_freeze_vit=${TRAIN_FREEZE_VIT_OVERRIDE:-${repro_freeze_vit}}
calibration_dataset_sha256=$(repro_dataset_sha256 "${calibration_data_path}")

: "${PLACEMOE_REPRO_COMM_CALIBRATION:?PLACEMOE_REPRO_COMM_CALIBRATION must point to the shared EP32 communication artifact}"
: "${PLACEMOE_REPRO_CHECKPOINT_SHA256:?PLACEMOE_REPRO_CHECKPOINT_SHA256 must identify the exact model checkpoint}"
communication_calibration=${PLACEMOE_REPRO_COMM_CALIBRATION}
checkpoint_sha256=${PLACEMOE_REPRO_CHECKPOINT_SHA256}
preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:?PLACEMOE_REPRO_PREFLIGHT_REPORT must point to the current four-node preflight report}
communication_source_sha256=$(repro_communication_source_sha256)
if [[ ! -s "${communication_calibration}" || ! -s "${preflight_report}" || ! "${checkpoint_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid communication calibration or checkpoint SHA-256" >&2
  exit 1
fi
calibration_tag=${PLACEMOE_REPRO_CALIBRATION_TAG:-$(date +%Y%m%d_%H%M%S)}
cost_scope_sha256=$(repro_cost_scope_sha256 \
  "${checkpoint_sha256}" "${communication_calibration}" "${calibration_moe_impl}" \
  "${calibration_micro_batch}" "${calibration_global_batch}" "${calibration_max_seq_len}" \
  "${dataset}" "${calibration_dataset_sha256}" "${calibration_data_source}" "${calibration_freeze_vit}")
cost_scope_tag=${cost_scope_sha256:0:12}
run_name=${PLACEMOE_REPRO_CALIBRATION_RUN_NAME:-gpu32_${repro_model_slug}_ours_compute_calibration_${repro_dataset_slug}_${calibration_tag}}
run_log="${repro_source_root}/pretrain_runs/${run_name}_rank0.host.log"
timing_root="${repro_source_root}/profile/runs/pretrain/${run_name}"
cost_model=${PLACEMOE_REPRO_COST_MODEL:-${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${cost_scope_tag}_cost_model.json}
r2_summary=${PLACEMOE_REPRO_R2_SUMMARY:-}

if [[ "${PLACEMOE_REPRO_REUSE_CALIBRATION_RUN:-0}" != "1" ]]; then
  if [[ -e "${run_log}" || -e "${timing_root}" ]]; then
    echo "refusing to append to existing calibration run: ${run_name}" >&2
    exit 1
  fi
  env \
    E2E_VARIANT=cost_model_verify \
    RUN_NAME_OVERRIDE="${run_name}" \
    MODEL_PATH_OVERRIDE="${calibration_model_path}" \
    MODEL_CONFIG_PATH_OVERRIDE="${calibration_model_path}" \
    CONFIG_PATH_OVERRIDE="${repro_config_path}" \
    DATA_PATH_OVERRIDE="${calibration_data_path}" \
    DATA_SOURCE_NAME_OVERRIDE="${calibration_data_source}" \
    NUM_MOE_LAYERS_OVERRIDE="${repro_num_layers}" \
    MICRO_BATCH_SIZE_OVERRIDE="${calibration_micro_batch}" \
    GLOBAL_BATCH_SIZE_OVERRIDE="${calibration_global_batch}" \
    TRAIN_FREEZE_VIT_OVERRIDE="${calibration_freeze_vit}" \
    MAX_SEQ_LEN_OVERRIDE="${calibration_max_seq_len}" \
    MOE_IMPL_OVERRIDE="${calibration_moe_impl}" \
    HIERMOE_REDUNDANT_SLOTS_OVERRIDE="${repro_redundant_slots}" \
    HIERMOE_ABLATION_GRAD_MODE_OVERRIDE="${PLACEMOE_REPRO_CALIBRATION_GRAD_MODE:-blocking}" \
    MAX_STEPS_OVERRIDE=3 \
    FULL_PROFILE_ENABLE_OVERRIDE=0 \
    VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS_OVERRIDE=1 \
    VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES_OVERRIDE=1 \
    MASTER_PORT="${PLACEMOE_REPRO_CALIBRATION_MASTER_PORT:-29940}" \
    HIERMOE_FIT_PERF_MODEL_ON_STARTUP_OVERRIDE=true \
    HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP_OVERRIDE=1 \
    bash "${script_dir}/launch.sh"
  bash "${script_dir}/collect_run.sh" "${run_name}"
fi

if [[ ! -s "${run_log}" ]]; then
  echo "missing Ours calibration log: ${run_log}" >&2
  exit 1
fi
actual_timing_files=$(find "${timing_root}/moe_timing" -type f -name 'moe_timing_rank*.jsonl' | wc -l)
if [[ "${actual_timing_files}" -ne 32 ]]; then
  echo "incomplete Ours calibration timing: expected 32 rank files, found ${actual_timing_files}" >&2
  exit 1
fi

calibrator_args=(
  --run-name "${run_name}"
  --ours-log "${run_log}"
  --phase-timing-root "${timing_root}"
  --phase-step 2
  --layers "${repro_num_layers}"
  --ep-size 32
  --communication-calibration "${communication_calibration}"
  --checkpoint-sha256 "${checkpoint_sha256}"
  --preflight-report "${preflight_report}"
  --communication-source-sha256 "${communication_source_sha256}"
  --cost-scope-sha256 "${cost_scope_sha256}"
  --ranks-per-node 8
  --num-experts "${repro_num_experts}"
  --slots-per-rank "${repro_slots_per_rank}"
  --hidden-size "${repro_hidden_size}"
  --bytes-per-element 2
  --model-id "${model}"
  --dataset-id "${dataset}"
  --dataset-sha256 "${calibration_dataset_sha256}"
  --data-source-name "${calibration_data_source}"
  --micro-batch-size "${calibration_micro_batch}"
  --global-batch-size "${calibration_global_batch}"
  --max-seq-len "${calibration_max_seq_len}"
  --moe-impl "${calibration_moe_impl}"
  --freeze-vit "${calibration_freeze_vit}"
  --output "${cost_model}"
)
if [[ -s "${r2_summary}" ]]; then
  calibrator_args+=(--r2-summary "${r2_summary}")
fi

python_bin=${PYTHON:-${repro_python}}
PYTHONPATH="${repro_source_root}" "${python_bin}" \
  "${script_dir}/calibrate_cost_model.py" "${calibrator_args[@]}"

bash "${script_dir}/publish_artifacts.sh" "${cost_model}"

echo "EP32 Ours cost model ready: ${cost_model}"
