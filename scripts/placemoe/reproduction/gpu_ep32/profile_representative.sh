#!/usr/bin/env bash

# Collect a representative profile after formal timing has completed.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 METHOD DATASET [LAYOUT]" >&2
  exit 2
fi
method=$1
dataset=$2
layout=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
repro_configure_model qwen3vl
repro_configure_dataset "${dataset}"

variant=baseline
grad_mode=blocking
case "${method}" in
  baseline)
    variant=baseline
    ;;
  r2)
    variant=replica
    ;;
  eplb|ours)
    variant=static_layout
    if [[ ! -s "${layout}" ]]; then
      echo "${method} profiling requires an existing layout artifact" >&2
      exit 2
    fi
    if [[ "${method}" == ours ]]; then
      grad_mode=hidden
    fi
    ;;
  *)
    echo "unsupported method: ${method}" >&2
    exit 2
    ;;
esac

run_tag=${PLACEMOE_REPRO_PROFILE_TAG:-$(date +%Y%m%d_%H%M%S)}
run_name="gpu32_profile_${repro_model_slug}_${repro_dataset_slug}_${method}_${run_tag}"
placemoe_config=
if [[ "${method}" == ours ]]; then
  placemoe_config="${repro_source_root}/results/${run_name}_placemoe.json"
  PYTHONPATH="${repro_source_root}" "${PYTHON:-${repro_python}}" "${repro_source_root}/scripts/placemoe/materialize_config.py" \
    --initial-artifact "${layout}" \
    --output "${placemoe_config}"
fi
command=(
  env
  "E2E_VARIANT=${variant}"
  "RUN_NAME_OVERRIDE=${run_name}"
  "MODEL_PATH_OVERRIDE=${repro_model_path}"
  "MODEL_CONFIG_PATH_OVERRIDE=${repro_model_path}"
  "CONFIG_PATH_OVERRIDE=${repro_config_path}"
  "DATA_PATH_OVERRIDE=${repro_data_path}"
  "DATA_SOURCE_NAME_OVERRIDE=${repro_data_source_name}"
  "TRAIN_FREEZE_VIT_OVERRIDE=${repro_freeze_vit}"
  "NUM_MOE_LAYERS_OVERRIDE=${repro_num_layers}"
  "MICRO_BATCH_SIZE_OVERRIDE=${repro_micro_batch_size}"
  "GLOBAL_BATCH_SIZE_OVERRIDE=${repro_global_batch_size}"
  "MAX_SEQ_LEN_OVERRIDE=4096"
  "MAX_STEPS_OVERRIDE=12"
  "HIERMOE_REDUNDANT_SLOTS_OVERRIDE=${repro_redundant_slots}"
  "HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=${grad_mode}"
  "FULL_PROFILE_ENABLE_OVERRIDE=1"
  "FULL_PROFILE_START_STEP_OVERRIDE=11"
  "TORCH_PROFILE_ENABLE_OVERRIDE=1"
  "VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS_OVERRIDE=1"
  "MASTER_PORT=${PLACEMOE_REPRO_PROFILE_MASTER_PORT:-30450}"
)
if [[ -n "${layout}" ]]; then
  if [[ "${method}" == ours ]]; then
    command+=("PLACEMOE_CONFIG_OVERRIDE=${placemoe_config}")
  else
    command+=("HIERMOE_INITIAL_LAYOUT_OVERRIDE=${layout}")
  fi
fi
command+=(bash "${script_dir}/launch.sh")
"${command[@]}"
bash "${script_dir}/collect_run.sh" "${run_name}"
echo "representative profile collected: ${run_name}"
