#!/usr/bin/env bash

set -uo pipefail

source_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722/src
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722/src
launcher=${source_root}/scripts/profile/launch_hiermoe_greedy_e2e_4node.sh
collector=${source_root}/scripts/profile/collect_hiermoe_paper_run.sh
summarizer=${source_root}/scripts/profile/summarize_hiermoe_paper_case.py

common=(
  MAX_STEPS_OVERRIDE=20
  FULL_PROFILE_START_STEP_OVERRIDE=11
  FULL_PROFILE_EVERY_N_OVERRIDE=1
  FULL_PROFILE_RANKS_OVERRIDE=0
  HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=blocking
  RANK0_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank0_20260729
  RANK1_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank1_20260729
  RANK2_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank2_20260729
  RANK3_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank3_20260729
)

run_case() {
  local name=$1
  local variant=$2
  local master_port=$3
  local hccl_port=$4
  local replay=${5:-}
  local report=${6:-}
  local -a case_env=(
    "${common[@]}"
    "RUN_NAME_OVERRIDE=${name}"
    "E2E_VARIANT=${variant}"
    "MASTER_PORT=${master_port}"
    "HCCL_IF_BASE_PORT=${hccl_port}"
  )
  if [[ -n "${replay}" ]]; then
    case_env+=(
      "HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE=${container_root}/results/${replay}"
      "HIERMOE_GREEDY_MAX_COPIES_OVERRIDE=8"
    )
  fi

  echo "starting ${name}"
  if ! env "${case_env[@]}" bash "${launcher}"; then
    echo "FAILED ${name}" >&2
    return 1
  fi
  bash "${collector}" "${name}"
  local -a summary_args=(
    --run-name "${name}"
    --start-step 11
    --end-step 20
    --output "${source_root}/results/${name}_summary.json"
  )
  if [[ -n "${report}" ]]; then
    summary_args+=(--layout-report "${source_root}/results/${report}")
  fi
  python "${summarizer}" "${summary_args[@]}"
}

failures=0
run_case paper8h_p0_sharegpt4v_b4_r2_20260729 fixed_r2_mirrored_pipeline_grad 30200 59000 || failures=$((failures + 1))
run_case \
  paper8h_p0_sharegpt4v_b4_hiremoe_20260729 \
  hierarchical_full_static \
  30201 \
  59100 \
  hierarchical_full_static_ep32_48layers_20260728.json || failures=$((failures + 1))
run_case \
  paper8h_p0_sharegpt4v_b4_eplb_20260729 \
  hierarchical_full_static \
  30202 \
  59200 \
  eplb_sharegpt4v_profile4_b4_ep32_48layers_layout_20260729.json \
  eplb_sharegpt4v_profile4_b4_ep32_48layers_report_20260729.json || failures=$((failures + 1))
run_case \
  paper8h_p0_sharegpt4v_b4_ours_20260729 \
  hierarchical_full_static \
  30203 \
  59300 \
  recursive_classifier_refined_v2_ep32_48layers_layout_20260728.json \
  recursive_classifier_refined_v2_ep32_48layers_report_20260728.json || failures=$((failures + 1))

echo "P0 completed failures=${failures}"
exit "${failures}"
