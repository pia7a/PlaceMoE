#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 MODEL DATASET METHOD [full|smoke]" >&2
  echo "  MODEL: qwen3vl | qwen35_20l | deepseek_v3_6moe_half" >&2
  echo "  DATASET: sharegpt4v | tulu3" >&2
  echo "  METHOD: baseline | dedup | r2 | eplb | hiermoe | ours | ours_online_lut | ours_full_replan | static" >&2
  exit 2
fi

model=$1
dataset=$2
method=$3
mode=${4:-full}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"
paper32_configure_model "${model}"
paper32_configure_dataset "${dataset}"
if [[ "${model}" == "deepseek_v3_6moe_half" || "${model}" == "deepseek6moe" ]]; then
  paper32_load_compute_calibration
fi

pause_file=${PAPER32_PAUSE_FILE:-${paper32_source_root}/.paper32_pause}
if [[ "${method}" != "baseline" ]]; then
  while [[ -e "${pause_file}" ]]; do
    echo "paper32 paused before ${model}/${dataset}/${method}: ${pause_file}"
    sleep 5
  done
fi

case "${mode}" in
  full)
    max_steps=${PAPER32_MAX_STEPS_OVERRIDE:-20}
    stats_start_step=${PAPER32_STATS_START_STEP_OVERRIDE:-11}
    stats_end_step=${PAPER32_STATS_END_STEP_OVERRIDE:-${max_steps}}
    if [[ ! "${max_steps}" =~ ^[1-9][0-9]*$ \
      || ! "${stats_start_step}" =~ ^[1-9][0-9]*$ \
      || ! "${stats_end_step}" =~ ^[1-9][0-9]*$ \
      || "${stats_start_step}" -gt "${stats_end_step}" \
      || "${stats_end_step}" -gt "${max_steps}" ]]
    then
      echo "invalid full-run step window: max=${max_steps} stats=${stats_start_step}-${stats_end_step}" >&2
      exit 2
    fi
    if [[ "${PAPER32_LIGHTWEIGHT_TIMING:-${paper32_lightweight_timing_default:-0}}" == "1" ]]; then
      full_profile_enable=0
      full_profile_start=99
    else
      full_profile_enable=1
      full_profile_start=${stats_start_step}
    fi
    ;;
  smoke)
    max_steps=${PAPER32_SMOKE_STEPS:-2}
    full_profile_enable=0
    full_profile_start=99
    ;;
  *)
    echo "unsupported mode '${mode}'; expected full or smoke" >&2
    exit 2
    ;;
esac

layout_report=
replay=
case "${method}" in
  baseline)
    variant=baseline
    method_slug=$(paper32_method_slug baseline)
    ;;
  dedup)
    variant=dedup
    method_slug=hierarchical_dedup
    ;;
  r2)
    variant=fixed_r2_mirrored_pipeline_grad
    method_slug=$(paper32_method_slug r2)
    ;;
  hiermoe)
    variant=hiermoe_exact_p1
    method_slug=$(paper32_method_slug hiermoe)
    ;;
  eplb|ours|ours_online_lut|ours_full_replan)
    if [[ "${method}" == "ours_online_lut" ]]; then
      variant=hierarchical_full_static_online_lut
      method_slug=ours_online_lut_hierarchical_dedup
      layout_stem=$(paper32_layout_stem ours)
    elif [[ "${method}" == "ours_full_replan" ]]; then
      variant=hierarchical_full_static_periodic_replan
      method_slug=$(paper32_method_slug ours_full_replan)
      layout_stem=$(paper32_layout_stem ours)
    else
      variant=hierarchical_full_static
      method_slug=$(paper32_method_slug "${method}")
      layout_stem=$(paper32_layout_stem "${method}")
    fi
    layout_stem=${PAPER32_LAYOUT_STEM_OVERRIDE:-${layout_stem}}
    replay=${paper32_source_root}/results/${layout_stem}_layout.json
    layout_report=${paper32_source_root}/results/${layout_stem}_report.json
    if [[ "${method}" == "ours_full_replan" && -n "${PAPER32_PLACEMOE_CONFIG:-}" ]]; then
      : # The canonical config owns and validates the initial artifact.
    elif [[ (! -s "${replay}" || ! -s "${layout_report}") && "${PAPER32_DRY_RUN:-0}" != "1" ]]; then
      echo "missing ${method} layout for ${model}/${dataset}; run:" >&2
      echo "  bash ${script_dir}/prepare_hiermoe_paper32_layouts.sh ${model} ${dataset}" >&2
      exit 1
    fi
    ;;
  static)
    variant=hierarchical_full_static
    layout_stem=${PAPER32_STATIC_LAYOUT_STEM:-}
    method_slug=${PAPER32_STATIC_METHOD_SLUG:-}
    if [[ -z "${layout_stem}" || -z "${method_slug}" ]]; then
      echo "static method requires PAPER32_STATIC_LAYOUT_STEM and PAPER32_STATIC_METHOD_SLUG" >&2
      exit 2
    fi
    replay=${paper32_source_root}/results/${layout_stem}_layout.json
    layout_report=${paper32_source_root}/results/${layout_stem}_report.json
    if [[ (! -s "${replay}" || ! -s "${layout_report}") && "${PAPER32_DRY_RUN:-0}" != "1" ]]; then
      echo "missing static layout '${layout_stem}' for ${model}/${dataset}" >&2
      exit 1
    fi
    ;;
  *)
    echo "unsupported method '${method}'; expected baseline, dedup, r2, eplb, hiermoe, ours, ours_online_lut, ours_full_replan, or static" >&2
    exit 2
    ;;
esac

default_grad_mode=$(paper32_method_grad_mode "${method}")
run_tag=${PAPER32_RUN_TAG:-20260730}
run_name=${PAPER32_RUN_NAME:-${paper32_artifact_prefix}_${paper32_model_slug}_${paper32_dataset_slug}_${method_slug}_${mode}_${run_tag}}
master_port=${PAPER32_MASTER_PORT:-30500}
hccl_port=${PAPER32_HCCL_PORT:-62000}
launcher=${script_dir}/launch_hiermoe_greedy_e2e_4node.sh
if [[ "${mode}" == "full" && "${PAPER32_SKIP_COMPLETED:-1}" == "1" \
  && -s "${paper32_source_root}/results/${run_name}_summary.json" ]]
then
  echo "skipping completed ${run_name}"
  exit 0
fi

case_env=(
  "E2E_VARIANT=${variant}"
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
  "NUM_TRAIN_EPOCHS_OVERRIDE=${PAPER32_NUM_TRAIN_EPOCHS:-1}"
  "TOTAL_MAX_STEPS_OVERRIDE=${PAPER32_TOTAL_MAX_STEPS:-}"
  "TRAIN_LR_OVERRIDE=${PAPER32_LR:-}"
  "FULL_PROFILE_ENABLE_OVERRIDE=${full_profile_enable}"
  "FULL_PROFILE_START_STEP_OVERRIDE=${full_profile_start}"
  "FULL_PROFILE_EVERY_N_OVERRIDE=1"
  "FULL_PROFILE_RANKS_OVERRIDE=0"
  "CONVERGENCE_METRICS_ENABLE_OVERRIDE=${PAPER32_CONVERGENCE_METRICS:-0}"
  "MOE_MONITOR_INTERVAL_OVERRIDE=${PAPER32_MOE_MONITOR_INTERVAL:-1}"
  "MOE_MONITOR_JSONL_ENABLE_OVERRIDE=${PAPER32_MOE_MONITOR_JSONL_ENABLE:-1}"
  "MOE_TIMING_ENABLE_OVERRIDE=${PAPER32_MOE_TIMING_ENABLE:-1}"
  "ENV_METRICS_JSONL_ENABLE_OVERRIDE=${PAPER32_ENV_METRICS_JSONL_ENABLE:-1}"
  "HIERMOE_LOG_INTERVAL_OVERRIDE=${PAPER32_HIERMOE_LOG_INTERVAL:-1}"
  "HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=${PAPER32_GRAD_MODE:-${default_grad_mode}}"
  "HIERMOE_GREEDY_MAX_COPIES_OVERRIDE=8"
  "HIERMOE_PERF_MODEL_PATH_OVERRIDE=${paper32_perf_model_container}"
)
for ((node_rank = 0; node_rank < paper32_nnodes; ++node_rank)); do
  if ((node_rank == 0)) && [[ "${PAPER32_RANK0_LOCAL:-0}" == "1" ]]; then
    case_env+=("RANK0_CONTAINER_OVERRIDE=${paper32_container_name}")
  else
    case_env+=(
      "RANK${node_rank}_HOST_OVERRIDE=${paper32_hosts[${node_rank}]}"
      "RANK${node_rank}_CONTAINER_OVERRIDE=${paper32_container_name}"
    )
  fi
done
if [[ "${method}" == "r2" || "${method}" == "eplb" || "${method}" == "ours" \
  || "${method}" == "ours_online_lut" || "${method}" == "ours_full_replan" || "${method}" == "static" ]]
then
  case_env+=("HIERMOE_REDUNDANT_SLOTS_OVERRIDE=${paper32_redundant_slots}")
fi
if [[ "${method}" == "eplb" || "${method}" == "ours" || "${method}" == "ours_online_lut" \
  || "${method}" == "ours_full_replan" \
  || "${method}" == "static" ]]
then
  placemoe_initial_artifact=${PAPER32_PLACEMOE_INITIAL_ARTIFACT:-${paper32_container_source_root}/results/${layout_stem}_layout.json}
  case_env+=(
    "HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE=${placemoe_initial_artifact}"
    "HIERMOE_STATIC_PRELOAD_LAYOUT_PATH_OVERRIDE=${placemoe_initial_artifact}"
  )
fi
if [[ "${method}" == "ours_online_lut" ]]; then
  case_env+=(
    "HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE_OVERRIDE=${paper32_inter_ms_per_byte}"
    "HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE_OVERRIDE=${paper32_intra_ms_per_byte}"
    "HIERMOE_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT_OVERRIDE=${PAPER32_ROUTE_MS_PER_ASSIGNMENT:-8.746548178958447e-05}"
    "HIERMOE_ONLINE_FREEZE_COMMUNICATION_RATIO_OVERRIDE=${paper32_communication_phase_multiplier}"
    "HIERMOE_ONLINE_FREEZE_COMPUTE_RATIO_OVERRIDE=${paper32_compute_phase_multiplier}"
    "HIERMOE_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT_OVERRIDE=${paper32_compute_ms_per_assignment}"
  )
fi
if [[ "${method}" == "ours_full_replan" ]]; then
  case_env+=(
    "HIERMOE_PLACEMOE_CONFIG_PATH_OVERRIDE=${PAPER32_PLACEMOE_CONFIG:-}"
    "HIERMOE_LAYOUT_REFRESH_INTERVAL_OVERRIDE=${PAPER32_LAYOUT_REFRESH_INTERVAL:-${HIERMOE_SWAP_INTERVAL_OVERRIDE:-100}}"
    "HIERMOE_MAPPING_REFRESH_INTERVAL_OVERRIDE=${PAPER32_MAPPING_REFRESH_INTERVAL:-0}"
    "HIERMOE_PLACEMOE_INTER_MS_PER_BYTE_OVERRIDE=${paper32_inter_ms_per_byte}"
    "HIERMOE_PLACEMOE_INTRA_MS_PER_BYTE_OVERRIDE=${paper32_intra_ms_per_byte}"
    "HIERMOE_PLACEMOE_ROUTE_MS_PER_ASSIGNMENT_OVERRIDE=${PAPER32_ROUTE_MS_PER_ASSIGNMENT:-8.746548178958447e-05}"
    "HIERMOE_PLACEMOE_COMMUNICATION_MULTIPLIER_OVERRIDE=${paper32_communication_phase_multiplier}"
    "HIERMOE_PLACEMOE_COMPUTE_MS_PER_ASSIGNMENT_OVERRIDE=${paper32_compute_ms_per_assignment}"
    "HIERMOE_PLACEMOE_COMPUTE_MULTIPLIER_OVERRIDE=${paper32_compute_phase_multiplier}"
    "HIERMOE_PERIODIC_FULL_REPLAN_LAST_STEP_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_LAST_STEP:-2147483647}"
    "HIERMOE_PERIODIC_FULL_REPLAN_WORKERS_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_WORKERS:-48}"
    "HIERMOE_PERIODIC_FULL_REPLAN_CANDIDATE_WORKERS_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_CANDIDATE_WORKERS:-4}"
    "HIERMOE_PERIODIC_FULL_REPLAN_WORKER_THREADS_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_WORKER_THREADS:-1}"
    "HIERMOE_PERIODIC_FULL_REPLAN_CPU_IDS_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_CPU_IDS:-144-191}"
    "HIERMOE_PERIODIC_FULL_REPLAN_TRAIN_CPU_IDS_OVERRIDE=${PAPER32_PERIODIC_FULL_REPLAN_TRAIN_CPU_IDS:-0-143}"
  )
fi

echo "starting ${run_name}"
if [[ "${PAPER32_DRY_RUN:-0}" == "1" ]]; then
  printf 'env'
  printf ' %q' "${case_env[@]}"
  printf ' bash %q\n' "${launcher}"
  exit 0
fi
env "${case_env[@]}" bash "${launcher}"

if [[ "${mode}" == "full" ]]; then
  bash "${script_dir}/collect_hiermoe_paper32_run.sh" "${run_name}"
  if [[ "${PAPER32_SKIP_PAPER_SUMMARY:-0}" == "1" ]]; then
    echo "skipping heavyweight paper summary for ${run_name}"
    echo "completed ${run_name}"
    exit 0
  fi
  summary_grad_mode=not_applicable
  if [[ "${method}" == "r2" || "${method}" == "eplb" || "${method}" == "ours" \
    || "${method}" == "ours_online_lut" || "${method}" == "ours_full_replan" || "${method}" == "static" ]]
  then
    summary_grad_mode=${PAPER32_GRAD_MODE:-${default_grad_mode}}
  fi
  summary_args=(
    --run-name "${run_name}"
    --start-step "${stats_start_step}"
    --end-step "${stats_end_step}"
    --expected-ranks "${paper32_world_size}"
    --grad-mode "${summary_grad_mode}"
    --output "${paper32_source_root}/results/${run_name}_summary.json"
  )
  if [[ -n "${layout_report}" ]]; then
    summary_args+=(--layout-report "${layout_report}")
  fi
  if [[ -n "${replay}" ]]; then
    summary_args+=(--layout-path "${replay}")
  fi
  (
    cd "${paper32_source_root}"
    python scripts/profile/summarize_hiermoe_paper_case.py "${summary_args[@]}"
  )
fi

echo "completed ${run_name}"
