#!/usr/bin/env bash
# Prepare/reuse the scoped communication and model-compute calibrations for EP32.

set -euo pipefail

mode=${1:-prepare}
if [[ "${mode}" != prepare && "${mode}" != check ]]; then
  echo "usage: $0 [prepare|check]" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
python_bin=${PYTHON:-${repro_python}}
communication_calibration=${PLACEMOE_REPRO_COMM_CALIBRATION:-${repro_source_root}/results/gpu32_ep32_a6000_communication_cost.json}
read -r -a models <<< "${PLACEMOE_REPRO_MODELS:-qwen3vl qwen35_20l}"
read -r -a datasets <<< "${PLACEMOE_REPRO_DATASETS:-sharegpt4v tulu3}"
tag=${PLACEMOE_REPRO_CALIBRATION_TAG:-$(date +%Y%m%d_%H%M%S)}
preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:?PLACEMOE_REPRO_PREFLIGHT_REPORT must point to the current four-node preflight report}
if [[ ! -s "${preflight_report}" ]]; then
  echo "missing current EP32 preflight report: ${preflight_report}" >&2
  exit 1
fi
communication_source_sha256=$(repro_communication_source_sha256)

if [[ ! -s "${communication_calibration}" ]]; then
  if [[ "${mode}" == check ]]; then
    echo "missing communication calibration: ${communication_calibration}" >&2
    exit 1
  fi
  PLACEMOE_REPRO_COMM_CALIBRATION_OUTPUT="${communication_calibration}" \
    PLACEMOE_REPRO_COMM_CALIBRATION_TAG="${tag}" \
    PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
    bash "${script_dir}/calibrate_communication.sh"
fi
PYTHONPATH="${repro_source_root}" "${python_bin}" - "${communication_calibration}" "${preflight_report}" "${communication_source_sha256}" <<'PY' >/dev/null
import sys
from pathlib import Path
from scripts.placemoe.reproduction.gpu_ep32.cost_components import load_communication_calibration
load_communication_calibration(Path(sys.argv[1]), ep_size=32, ranks_per_node=8, hidden_size=2048, bytes_per_element=2, preflight_report=Path(sys.argv[2]), communication_source_sha256=sys.argv[3])
PY

for model in "${models[@]}"; do
  repro_configure_model "${model}"
  checkpoint_sha256=$(repro_checkpoint_sha256)
  for dataset in "${datasets[@]}"; do
    repro_configure_model "${model}"
    repro_configure_dataset "${dataset}"
    dataset_sha256=$(repro_dataset_sha256 "${repro_data_path}")
    cost_scope_sha256=$(repro_cost_scope_sha256 \
      "${checkpoint_sha256}" "${communication_calibration}" fused_triton \
      "${repro_micro_batch_size}" "${repro_global_batch_size}" 4096 \
      "${dataset}" "${dataset_sha256}" "${repro_data_source_name}" "${repro_freeze_vit}")
    cost_model="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${cost_scope_sha256:0:12}_cost_model.json"
    if [[ ! -s "${cost_model}" ]]; then
      if [[ "${mode}" == check ]]; then
        echo "missing model/dataset cost calibration: ${cost_model}" >&2
        exit 1
      fi
      PLACEMOE_REPRO_COMM_CALIBRATION="${communication_calibration}" \
        PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
        PLACEMOE_REPRO_CHECKPOINT_SHA256="${checkpoint_sha256}" \
        PLACEMOE_REPRO_COST_MODEL="${cost_model}" \
        PLACEMOE_REPRO_CALIBRATION_TAG="${tag}" \
        bash "${script_dir}/calibrate_cost_model.sh" "${model}" "${dataset}"
    fi
    PYTHONPATH="${repro_source_root}" "${python_bin}" \
      "${script_dir}/validate_cost_model.py" \
      --cost-model "${cost_model}" \
      --communication-calibration "${communication_calibration}" \
      --preflight-report "${preflight_report}" \
      --communication-source-sha256 "${communication_source_sha256}" \
      --model-id "${model}" \
      --dataset-id "${dataset}" \
      --dataset-sha256 "${dataset_sha256}" \
      --data-source-name "${repro_data_source_name}" \
      --freeze-vit "${repro_freeze_vit}" \
      --checkpoint-sha256 "${checkpoint_sha256}" \
      --cost-scope-sha256 "${cost_scope_sha256}" \
      --layers "${repro_num_layers}" \
      --num-experts "${repro_num_experts}" \
      --slots-per-rank "${repro_slots_per_rank}" \
      --hidden-size "${repro_hidden_size}" \
      --micro-batch-size "${repro_micro_batch_size}" \
      --global-batch-size "${repro_global_batch_size}" \
      --max-seq-len 4096 \
      --moe-impl fused_triton >/dev/null
    printf "model=%s dataset=%s checkpoint_sha256=%s dataset_sha256=%s cost_model=%s\n" \
      "${model}" "${dataset}" "${checkpoint_sha256}" "${dataset_sha256}" "${cost_model}"
  done
done
printf 'communication_calibration=%s\n' "${communication_calibration}"
