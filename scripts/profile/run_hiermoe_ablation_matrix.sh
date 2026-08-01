#!/usr/bin/env bash

set -euo pipefail

mode=${1:-dry-run}
model=${PAPER32_ABLATION_MODEL:-qwen3vl}
dataset=${PAPER32_ABLATION_DATASET:-sharegpt4v}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"
paper32_configure_model "${model}"
paper32_configure_dataset "${dataset}"

case "${mode}" in
  prepare)
    bash "${script_dir}/sync_hiermoe_paper32_source.sh"
    PAPER32_PREPARE_EPLB=0 \
      PAPER32_REUSE_PROFILE=${PAPER32_REUSE_PROFILE:-0} \
      bash "${script_dir}/prepare_hiermoe_paper32_layouts.sh" "${model}" "${dataset}"
    bash "${script_dir}/prepare_hiermoe_ablation_layouts.sh" "${model}" "${dataset}"
    exit 0
    ;;
  dry-run)
    run_mode=full
    dry_run=1
    ;;
  smoke)
    run_mode=smoke
    dry_run=0
    ;;
  full)
    if [[ "${PAPER32_CONFIRM_ABLATION:-0}" != "1" ]]; then
      echo "full mode runs twelve independent 32-NPU cases" >&2
      echo "set PAPER32_CONFIRM_ABLATION=1 to start it" >&2
      exit 2
    fi
    run_mode=full
    dry_run=0
    ;;
  *)
    echo "usage: $0 [prepare|dry-run|smoke|full]" >&2
    exit 2
    ;;
esac

if [[ "${dry_run}" == "0" && "${PAPER32_ABLATION_SKIP_PREPARE:-0}" != "1" ]]; then
  bash "${script_dir}/sync_hiermoe_paper32_source.sh"
  PAPER32_PREPARE_EPLB=0 \
    PAPER32_REUSE_PROFILE=${PAPER32_REUSE_PROFILE:-0} \
    bash "${script_dir}/prepare_hiermoe_paper32_layouts.sh" "${model}" "${dataset}"
  bash "${script_dir}/prepare_hiermoe_ablation_layouts.sh" "${model}" "${dataset}"
fi

if [[ "${mode}" == "smoke" ]]; then
  default_cases="hyper_rho100 ablation_comm"
else
  default_cases=(
    baseline
    hyper_rho000
    hyper_rho025
    hyper_rho050
    hyper_rho075
    hyper_rho100
    ablation_dedup
    ablation_static_r2
    ablation_comm
    ablation_compute
    ablation_joint
    ablation_online_lut
    ablation_grad_blocking
  )
fi
if [[ -n "${PAPER32_ABLATION_CASES:-}" ]]; then
  read -r -a cases <<< "${PAPER32_ABLATION_CASES}"
elif [[ "${mode}" == "smoke" ]]; then
  read -r -a cases <<< "${default_cases}"
else
  cases=("${default_cases[@]}")
fi

quarter=$((paper32_primary_slots / 4))
run_tag=${PAPER32_ABLATION_RUN_TAG:-20260729}
port_index=0
failures=0

run_case() {
  local case_name=$1
  local method=
  local method_slug=
  local layout_stem=
  local redundant_slots=
  local grad_mode=hidden
  local master_port=$((PAPER32_ABLATION_MASTER_PORT_BASE + port_index))
  local hccl_port=$((PAPER32_ABLATION_HCCL_PORT_BASE + port_index * 100))
  local run_name=paper32_${paper32_model_slug}_${paper32_dataset_slug}_${case_name}_${run_mode}_${run_tag}
  local -a case_env=(
    "PAPER32_DRY_RUN=${dry_run}"
    "PAPER32_RUN_NAME=${run_name}"
    "PAPER32_MASTER_PORT=${master_port}"
    "PAPER32_HCCL_PORT=${hccl_port}"
  )

  case "${case_name}" in
    baseline)
      method=baseline
      ;;
    hyper_rho000)
      method=static
      method_slug=ours_hyper_rho000
      redundant_slots=0
      layout_stem=$(paper32_ablation_layout_stem hyper_rho000 0)
      ;;
    hyper_rho025)
      method=static
      method_slug=ours_hyper_rho025
      redundant_slots=${quarter}
      layout_stem=$(paper32_ablation_layout_stem hyper_rho025 "${redundant_slots}")
      ;;
    hyper_rho050)
      method=static
      method_slug=ours_hyper_rho050
      redundant_slots=$((2 * quarter))
      layout_stem=$(paper32_ablation_layout_stem hyper_rho050 "${redundant_slots}")
      ;;
    hyper_rho075)
      method=static
      method_slug=ours_hyper_rho075
      redundant_slots=$((3 * quarter))
      layout_stem=$(paper32_ablation_layout_stem hyper_rho075 "${redundant_slots}")
      ;;
    hyper_rho100)
      method=ours
      ;;
    ablation_dedup)
      method=dedup
      ;;
    ablation_static_r2)
      method=r2
      ;;
    ablation_comm)
      method=static
      method_slug=ours_comm_initial_lut
      redundant_slots=${paper32_primary_slots}
      layout_stem=$(paper32_ablation_layout_stem comm_initial_lut "${redundant_slots}")
      ;;
    ablation_compute)
      method=static
      method_slug=ours_compute_initial_lut
      redundant_slots=${paper32_primary_slots}
      layout_stem=$(paper32_ablation_layout_stem compute_initial_lut "${redundant_slots}")
      ;;
    ablation_joint)
      method=static
      method_slug=ours_joint_initial_lut
      redundant_slots=${paper32_primary_slots}
      layout_stem=$(paper32_ablation_layout_stem joint_initial_lut "${redundant_slots}")
      ;;
    ablation_online_lut)
      method=ours_online_lut
      ;;
    ablation_grad_blocking)
      method=ours
      grad_mode=blocking
      ;;
    *)
      echo "unknown ablation case '${case_name}'" >&2
      return 2
      ;;
  esac

  case_env+=("PAPER32_GRAD_MODE=${grad_mode}")
  if [[ -n "${redundant_slots}" ]]; then
    case_env+=("PAPER32_REDUNDANT_SLOTS_OVERRIDE=${redundant_slots}")
  fi
  if [[ -n "${layout_stem}" ]]; then
    case_env+=(
      "PAPER32_STATIC_LAYOUT_STEM=${layout_stem}"
      "PAPER32_STATIC_METHOD_SLUG=${method_slug}"
    )
  fi

  echo "starting ${case_name}"
  env "${case_env[@]}" \
    bash "${script_dir}/run_hiermoe_paper32_case.sh" \
      "${model}" "${dataset}" "${method}" "${run_mode}"
}

: "${PAPER32_ABLATION_MASTER_PORT_BASE:=30800}"
: "${PAPER32_ABLATION_HCCL_PORT_BASE:=58000}"
for case_name in "${cases[@]}"; do
  if ! run_case "${case_name}"; then
    failures=$((failures + 1))
    echo "ablation matrix stopping after failed case=${case_name}" >&2
    break
  fi
  port_index=$((port_index + 1))
done

summary_path() {
  local case_name=$1
  printf '%s/results/paper32_%s_%s_%s_%s_%s_summary.json' \
    "${paper32_source_root}" \
    "${paper32_model_slug}" \
    "${paper32_dataset_slug}" \
    "${case_name}" \
    "${run_mode}" \
    "${run_tag}"
}

plot_group() {
  local group=$1
  shift
  local output_stem=paper32_${paper32_model_slug}_${paper32_dataset_slug}_${group}_speedup_vs_veomni_${run_tag}
  local -a plot_args=(
    --model "${paper32_model_slug}"
    --dataset "${paper32_dataset_slug}"
    --output-svg "${paper32_source_root}/results/${output_stem}.svg"
    --output-json "${paper32_source_root}/results/${output_stem}.json"
    --output-csv "${paper32_source_root}/results/${output_stem}.csv"
  )
  local spec
  for spec in "$@"; do
    plot_args+=(--summary "${spec}")
  done
  python "${script_dir}/plot_hiermoe_paper_speedup.py" "${plot_args[@]}" \
    >"${paper32_source_root}/results/${output_stem}.log"
}

all_summaries_exist() {
  local spec
  local path
  for spec in "$@"; do
    path=${spec#*=}
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
  done
}

refresh_full_ours_hidden_summary() {
  local run_name=paper32_${paper32_model_slug}_${paper32_dataset_slug}_hyper_rho100_${run_mode}_${run_tag}
  local layout_stem
  local layout_path
  local layout_report
  local output
  layout_stem=$(paper32_layout_stem ours)
  layout_path=${paper32_source_root}/results/${layout_stem}_layout.json
  layout_report=${paper32_source_root}/results/${layout_stem}_report.json
  output=$(summary_path hyper_rho100)
  (
    cd "${paper32_source_root}"
    python scripts/profile/summarize_hiermoe_paper_case.py \
      --run-name "${run_name}" \
      --start-step 11 \
      --end-step 20 \
      --expected-ranks "${paper32_world_size}" \
      --grad-mode hidden \
      --layout-report "${layout_report}" \
      --layout-path "${layout_path}" \
      --output "${output}"
  ) >"${output%.json}.refresh.log"
}

if [[ "${failures}" == "0" && "${dry_run}" == "0" && "${run_mode}" == "full" ]]; then
  hyperparameter_summaries=(
    "baseline=$(summary_path baseline)" \
    "hyper_rho000=$(summary_path hyper_rho000)" \
    "hyper_rho025=$(summary_path hyper_rho025)" \
    "hyper_rho050=$(summary_path hyper_rho050)" \
    "hyper_rho075=$(summary_path hyper_rho075)" \
    "hyper_rho100=$(summary_path hyper_rho100)"
  )
  static_r2_blocking_summary=${PAPER32_ABLATION_STATIC_R2_BLOCKING_SUMMARY:-}
  if [[ -z "${static_r2_blocking_summary}" \
    && "${paper32_model_slug}" == "qwen3vl30b" \
    && "${paper32_dataset_slug}" == "sharegpt4v" ]]; then
    static_r2_blocking_summary=${paper32_source_root}/results/paper32_qwen3vl30b_sharegpt4v_fixed_r2_hierarchical_dedup_full_huawei2_main20_v1_summary.json
  fi
  ablation_summaries=(
    "baseline=$(summary_path baseline)" \
    "dedup=$(summary_path ablation_dedup)"
  )
  if [[ -n "${static_r2_blocking_summary}" ]]; then
    ablation_summaries+=("static_r2=${static_r2_blocking_summary}")
  fi
  ablation_summaries+=(
    "static_r2_grad_overlap=$(summary_path ablation_static_r2)" \
    "comm_only=$(summary_path ablation_comm)" \
    "compute_only=$(summary_path ablation_compute)" \
    "comm_assignment=$(summary_path ablation_joint)" \
    "full_ours=$(summary_path hyper_rho100)" \
    "online_lut=$(summary_path ablation_online_lut)" \
    "full_ours_grad_blocking=$(summary_path ablation_grad_blocking)"
  )
  if all_summaries_exist "${hyperparameter_summaries[@]}"; then
    plot_group hyperparameters "${hyperparameter_summaries[@]}"
  else
    echo "skipping hyperparameter plot because one or more summaries are absent"
  fi
  if all_summaries_exist "${ablation_summaries[@]}"; then
    plot_group ablation "${ablation_summaries[@]}"
  else
    echo "skipping ablation plot because one or more summaries are absent"
  fi

  hidden_summary=$(summary_path hyper_rho100)
  blocking_summary=$(summary_path ablation_grad_blocking)
  if [[ -s "${blocking_summary}" ]]; then
    refresh_full_ours_hidden_summary
    python "${script_dir}/compare_hiermoe_grad_hiding.py" \
      --hidden-summary "${hidden_summary}" \
      --blocking-summary "${blocking_summary}" \
      --output-json \
        "${paper32_source_root}/results/paper32_${paper32_model_slug}_${paper32_dataset_slug}_grad_hiding_comparison_${run_tag}.json" \
      --output-csv \
        "${paper32_source_root}/results/paper32_${paper32_model_slug}_${paper32_dataset_slug}_grad_hiding_comparison_${run_tag}.csv"
  fi
fi

echo "ablation matrix completed failures=${failures}"
exit "${failures}"
