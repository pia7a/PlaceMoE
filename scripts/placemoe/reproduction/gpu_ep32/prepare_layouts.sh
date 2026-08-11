#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL DATASET" >&2
  exit 2
fi

model=$1
dataset=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
repro_configure_model "${model}"
repro_configure_dataset "${dataset}"

experiment_model_path=${MODEL_PATH_OVERRIDE:-${repro_model_path}}
experiment_data_path=${DATA_PATH_OVERRIDE:-${repro_data_path}}
experiment_data_source=${DATA_SOURCE_NAME_OVERRIDE:-${repro_data_source_name}}
experiment_micro_batch=${MICRO_BATCH_SIZE_OVERRIDE:-${repro_micro_batch_size}}
experiment_global_batch=${GLOBAL_BATCH_SIZE_OVERRIDE:-${repro_global_batch_size}}
experiment_max_seq_len=${MAX_SEQ_LEN_OVERRIDE:-${PLACEMOE_REPRO_MAX_SEQ_LEN:-4096}}
experiment_moe_impl=${MOE_IMPL_OVERRIDE:-fused_triton}
experiment_freeze_vit=${TRAIN_FREEZE_VIT_OVERRIDE:-${repro_freeze_vit}}
experiment_dataset_sha256=$(repro_dataset_sha256 "${experiment_data_path}")

# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"
: "${PLACEMOE_REPRO_COMM_CALIBRATION:?PLACEMOE_REPRO_COMM_CALIBRATION must point to the shared EP32 communication artifact}"
: "${PLACEMOE_REPRO_CHECKPOINT_SHA256:?PLACEMOE_REPRO_CHECKPOINT_SHA256 must identify the exact model checkpoint}"
checkpoint_sha256=${PLACEMOE_REPRO_CHECKPOINT_SHA256}
preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:?PLACEMOE_REPRO_PREFLIGHT_REPORT must point to the current four-node preflight report}
communication_source_sha256=$(repro_communication_source_sha256)
if [[ ! -s "${PLACEMOE_REPRO_COMM_CALIBRATION}" || ! -s "${preflight_report}" || ! "${checkpoint_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid communication calibration, preflight report, or checkpoint SHA-256" >&2
  exit 1
fi
cost_scope_sha256=$(repro_cost_scope_sha256 \
  "${checkpoint_sha256}" "${PLACEMOE_REPRO_COMM_CALIBRATION}" "${experiment_moe_impl}" \
  "${experiment_micro_batch}" "${experiment_global_batch}" "${experiment_max_seq_len}" \
  "${dataset}" "${experiment_dataset_sha256}" "${experiment_data_source}" "${experiment_freeze_vit}")
cost_scope_tag=${cost_scope_sha256:0:12}
python_bin=${PYTHON:-${repro_python}}
results_root="${repro_source_root}/results"
layout_tag=${PLACEMOE_REPRO_LAYOUT_TAG:-${cost_scope_tag}}
layout_stem="gpu32_${repro_model_slug}_${repro_dataset_slug}_${layout_tag}"
eplb_layout="${results_root}/${layout_stem}_eplb_layout.json"
eplb_report="${results_root}/${layout_stem}_eplb_report.json"
ours_layout="${results_root}/${layout_stem}_ours_layout.json"
ours_report="${results_root}/${layout_stem}_ours_report.json"
layout_bundle="${results_root}/${layout_stem}_layout_bundle.json"
cost_model=${PLACEMOE_REPRO_COST_MODEL:-${results_root}/gpu32_${repro_model_slug}_${repro_dataset_slug}_${cost_scope_tag}_cost_model.json}
eplb_root=${PLACEMOE_REPRO_EPLB_ROOT:-}
repro_require_value PLACEMOE_REPRO_EPLB_ROOT "${eplb_root}"
if [[ ! -s "${eplb_root}/eplb.py" ]]; then
  echo "missing official EPLB implementation: ${eplb_root}/eplb.py" >&2
  exit 1
fi

if [[ "${PLACEMOE_REPRO_REUSE_COST_MODEL:-0}" != "1" || ! -s "${cost_model}" ]]; then
  PLACEMOE_REPRO_COST_MODEL="${cost_model}" \
    PLACEMOE_REPRO_COMM_CALIBRATION="${PLACEMOE_REPRO_COMM_CALIBRATION}" \
    PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
    PLACEMOE_REPRO_CHECKPOINT_SHA256="${checkpoint_sha256}" \
    PLACEMOE_REPRO_CALIBRATION_MASTER_PORT="${PLACEMOE_REPRO_CALIBRATION_MASTER_PORT:-29940}" \
    DATA_PATH_OVERRIDE="${experiment_data_path}" \
    DATA_SOURCE_NAME_OVERRIDE="${experiment_data_source}" \
    MICRO_BATCH_SIZE_OVERRIDE="${experiment_micro_batch}" \
    GLOBAL_BATCH_SIZE_OVERRIDE="${experiment_global_batch}" \
    MAX_SEQ_LEN_OVERRIDE="${experiment_max_seq_len}" \
    MOE_IMPL_OVERRIDE="${experiment_moe_impl}" \
    TRAIN_FREEZE_VIT_OVERRIDE="${experiment_freeze_vit}" \
    bash "${script_dir}/calibrate_cost_model.sh" "${model}" "${dataset}"
fi
if [[ ! -s "${cost_model}" ]]; then
  echo "missing calibrated EP32 cost model after calibration: ${cost_model}" >&2
  exit 1
fi
validated_cost_model=$(PYTHONPATH="${repro_source_root}" "${python_bin}" \
  "${script_dir}/validate_cost_model.py" \
  --cost-model "${cost_model}" \
  --communication-calibration "${PLACEMOE_REPRO_COMM_CALIBRATION}" \
  --preflight-report "${preflight_report}" \
  --communication-source-sha256 "${communication_source_sha256}" \
  --model-id "${model}" \
  --dataset-id "${dataset}" \
  --dataset-sha256 "${experiment_dataset_sha256}" \
  --data-source-name "${experiment_data_source}" \
  --freeze-vit "${experiment_freeze_vit}" \
  --checkpoint-sha256 "${checkpoint_sha256}" \
  --cost-scope-sha256 "${cost_scope_sha256}" \
  --layers "${repro_num_layers}" \
  --num-experts "${repro_num_experts}" \
  --slots-per-rank "${repro_slots_per_rank}" \
  --hidden-size "${repro_hidden_size}" \
  --micro-batch-size "${experiment_micro_batch}" \
  --global-batch-size "${experiment_global_batch}" \
  --max-seq-len "${experiment_max_seq_len}" \
  --moe-impl "${experiment_moe_impl}")
read -r cost_inter cost_mid cost_intra cost_route cost_communication_multiplier cost_compute cost_compute_multiplier < <(
  "${python_bin}" -c \
    'import json, sys; c=json.loads(sys.argv[1])["coefficients"]; print(*(c[k] for k in ("inter_ms_per_byte", "mid_ms_per_byte", "intra_ms_per_byte", "route_ms_per_assignment", "communication_phase_multiplier", "compute_ms_per_assignment", "compute_phase_multiplier")))' \
    "${validated_cost_model}"
)
cost_model_sha256=$(sha256sum "${cost_model}")
cost_model_sha256=${cost_model_sha256%% *}

profile_base="gpu32_profile_${repro_model_slug}_${repro_dataset_slug}_4step"
if [[ "${PLACEMOE_REPRO_REUSE_PROFILE:-0}" == "1" ]]; then
  : "${PLACEMOE_REPRO_ROUTE_ROOT:?PLACEMOE_REPRO_ROUTE_ROOT is required when PLACEMOE_REPRO_REUSE_PROFILE=1}"
  route_root=${PLACEMOE_REPRO_ROUTE_ROOT}
  profile_name=$(basename "${route_root}")
else
  profile_tag=${PLACEMOE_REPRO_PROFILE_TAG:-$(date +%Y%m%d_%H%M%S)}
  profile_name="${profile_base}_${profile_tag}"
  route_root=${PLACEMOE_REPRO_ROUTE_ROOT:-${repro_source_root}/route_captures/${profile_name}}
  remote_route_root=${route_root}
  if [[ "${route_root}" == "${repro_source_root}/"* ]]; then
    remote_route_root=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repro_source_root}}${route_root#${repro_source_root}}
  fi
  if [[ -e "${route_root}" ]]; then
    echo "refusing to merge a fresh route capture into existing root: ${route_root}" >&2
    exit 1
  fi
  for spec in "${repro_remote_specs[@]}"; do
    port=${spec%%:*}
    if repro_ssh "${port}" "test -e '${remote_route_root}'"
    then
      echo "refusing to merge a fresh route capture into existing remote root on port ${port}" >&2
      exit 1
    fi
  done
  env \
    E2E_VARIANT=dedup \
    HIERMOE_CAPTURE_ROUTES=1 \
    HIERMOE_CAPTURE_ROOT="${route_root}" \
    RUN_NAME_OVERRIDE="${profile_name}" \
    MODEL_PATH_OVERRIDE="${experiment_model_path}" \
    MODEL_CONFIG_PATH_OVERRIDE="${experiment_model_path}" \
    CONFIG_PATH_OVERRIDE="${repro_config_path}" \
    DATA_PATH_OVERRIDE="${experiment_data_path}" \
    DATA_SOURCE_NAME_OVERRIDE="${experiment_data_source}" \
    NUM_MOE_LAYERS_OVERRIDE="${repro_num_layers}" \
    MICRO_BATCH_SIZE_OVERRIDE="${experiment_micro_batch}" \
    GLOBAL_BATCH_SIZE_OVERRIDE="${experiment_global_batch}" \
    TRAIN_FREEZE_VIT_OVERRIDE="${experiment_freeze_vit}" \
    MAX_SEQ_LEN_OVERRIDE="${experiment_max_seq_len}" \
    MOE_IMPL_OVERRIDE="${experiment_moe_impl}" \
    MAX_STEPS_OVERRIDE=4 \
    FULL_PROFILE_ENABLE_OVERRIDE=0 \
    MASTER_PORT="${PLACEMOE_REPRO_PROFILE_MASTER_PORT:-29800}" \
    bash "${script_dir}/launch.sh"

  if [[ ! -d "${route_root}" ]]; then
    echo "rank-0 route capture root was not created: ${route_root}" >&2
    exit 1
  fi
  for spec in "${repro_remote_specs[@]}"; do
    port=${spec%%:*}
    temporary_root=$(mktemp -d "/tmp/placemoe_routes_${port}_XXXXXX")
    repro_scp_from "${port}" "${remote_route_root}" "${temporary_root}/"
    cp -a "${temporary_root}/$(basename "${route_root}")/." "${route_root}/"
    rm -rf -- "${temporary_root}"
  done
fi

route_manifest="${route_root}/route_manifest.json"
route_manifest_sha256=$(PYTHONPATH="${repro_source_root}" "${python_bin}" \
  "${script_dir}/manifest_routes.py" \
  --route-root "${route_root}" \
  --steps 0,1,2,3 \
  --layers "${repro_num_layers}" \
  --ep-size 32 \
  --output "${route_manifest}")

layout_bundle_args=(
  --bundle "${layout_bundle}"
  --eplb-layout "${eplb_layout}"
  --eplb-report "${eplb_report}"
  --ours-layout "${ours_layout}"
  --ours-report "${ours_report}"
  --cost-model "${cost_model}"
  --cost-model-sha256 "${cost_model_sha256}"
  --route-manifest-sha256 "${route_manifest_sha256}"
  --route-root "${route_root}"
  --planner-source "${repro_source_root}/scripts/profile/plan_placemoe.py"
  --planner-source "${repro_source_root}/scripts/profile/placemoe_planner.py"
  --planner-source "${repro_source_root}/veomni/distributed/moe/hiermoe/placemoe"
  --eplb-source "${repro_source_root}/scripts/profile/build_hiermoe_eplb_layout.py"
  --eplb-source "${eplb_root}/eplb.py"
  --layers "${repro_num_layers}"
  --ep-size 32
  --ranks-per-node 8
  --hierarchy-group-sizes "${repro_hierarchy_group_sizes}"
  --num-experts "${repro_num_experts}"
  --primary-slots-per-rank "${repro_primary_slots}"
  --redundant-slots-per-rank "${repro_redundant_slots}"
  --slots-per-rank "${repro_slots_per_rank}"
  --hidden-size "${repro_hidden_size}"
  --accelerator "NVIDIA RTX A6000"
  --model-id "${model}"
  --dataset-id "${dataset}"
  --micro-batch-size "${experiment_micro_batch}"
  --global-batch-size "${experiment_global_batch}"
  --max-seq-len "${experiment_max_seq_len}"
  --moe-impl "${experiment_moe_impl}"
  --freeze-vit "${experiment_freeze_vit}"
)

reuse_layouts=0
if [[ "${PLACEMOE_REPRO_REUSE_LAYOUTS:-1}" == "1" \
  && -s "${eplb_layout}" && -s "${eplb_report}" \
  && -s "${ours_layout}" && -s "${ours_report}" \
  && -s "${layout_bundle}" ]] \
  && PYTHONPATH="${repro_source_root}" "${python_bin}" \
    "${script_dir}/validate_layout.py" \
    "${layout_bundle_args[@]}"
then
  reuse_layouts=1
  echo "reusing fingerprint-matched layouts for ${model}/${dataset}"
fi

if [[ "${reuse_layouts}" != "1" ]]; then
  mkdir -p "${results_root}"
  staging_root=$(mktemp -d "${results_root}/.placemoe_layouts_XXXXXX")
  staged_eplb_layout="${staging_root}/$(basename "${eplb_layout}")"
  staged_eplb_report="${staging_root}/$(basename "${eplb_report}")"
  staged_ours_layout="${staging_root}/$(basename "${ours_layout}")"
  staged_ours_report="${staging_root}/$(basename "${ours_report}")"

  PYTHONPATH="${repro_source_root}" "${python_bin}" \
    "${repro_source_root}/scripts/profile/build_hiermoe_eplb_layout.py" \
    --eplb-root "${eplb_root}" \
    --route-root "${route_root}" \
    --profile-steps 0,1,2,3 \
    --layers "${repro_num_layers}" \
    --ep-size 32 \
    --ranks-per-node 8 \
    --num-experts "${repro_num_experts}" \
    --primary-slots-per-rank "${repro_primary_slots}" \
    --redundant-slots-per-rank "${repro_redundant_slots}" \
    --output-layout "${staged_eplb_layout}" \
    --output-report "${staged_eplb_report}"

  PYTHONPATH="${repro_source_root}" "${python_bin}" \
    "${repro_source_root}/scripts/profile/plan_placemoe.py" \
    --route-root "${route_root}" \
    --optimize-steps 0,1,2 \
    --validation-steps 3 \
    --layers "${repro_num_layers}" \
    --expected-total-layers "${repro_num_layers}" \
    --workers "${PLACEMOE_REPRO_LAYOUT_WORKERS:-12}" \
    --candidate-workers "${PLACEMOE_REPRO_LAYOUT_CANDIDATE_WORKERS:-1}" \
    --worker-threads 1 \
    --ep-size 32 \
    --ranks-per-node 8 \
    --hierarchy-group-sizes "${repro_hierarchy_group_sizes}" \
    --num-experts "${repro_num_experts}" \
    --primary-slots-per-rank "${repro_primary_slots}" \
    --redundant-slots-per-rank "${repro_redundant_slots}" \
    --slots-per-rank "${repro_slots_per_rank}" \
    --hidden-size "${repro_hidden_size}" \
    --inter-ms-per-byte "${cost_inter}" \
    --mid-ms-per-byte "${cost_mid}" \
    --intra-ms-per-byte "${cost_intra}" \
    --route-ms-per-assignment "${cost_route}" \
    --communication-phase-multiplier "${cost_communication_multiplier}" \
    --compute-ms-per-assignment "${cost_compute}" \
    --compute-phase-multiplier "${cost_compute_multiplier}" \
    --comparison-layout mirrored-r2 \
    --output-layout "${staged_ours_layout}" \
    --output-report "${staged_ours_report}"

  mv "${staged_eplb_layout}" "${eplb_layout}"
  mv "${staged_eplb_report}" "${eplb_report}"
  mv "${staged_ours_report}" "${ours_report}"
  mv "${staged_ours_layout}" "${ours_layout}"
  PYTHONPATH="${repro_source_root}" "${python_bin}" \
    "${script_dir}/validate_layout.py" \
    --write \
    "${layout_bundle_args[@]}"
  rmdir "${staging_root}"
fi

bash "${script_dir}/publish_artifacts.sh" \
  "${cost_model}" "${eplb_layout}" "${eplb_report}" "${ours_report}" "${ours_layout}" \
  "${layout_bundle}"

echo "GPU layouts ready for ${model}/${dataset}; route_manifest_sha256=${route_manifest_sha256}"
