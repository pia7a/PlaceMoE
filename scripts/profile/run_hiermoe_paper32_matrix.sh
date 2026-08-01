#!/usr/bin/env bash

set -euo pipefail

mode=${1:-full}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"

paper32_preflight() {
  if [[ "${PAPER32_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    return
  fi
  bash "${script_dir}/sync_hiermoe_paper32_source.sh"
  bash "${script_dir}/prepare_hiermoe_paper32_containers.sh"
}

case "${mode}" in
  full)
    if [[ "${PAPER32_CONFIRM_FULL:-0}" != "1" ]]; then
      echo "full mode runs 20 training cases plus four route profiles." >&2
      echo "set PAPER32_CONFIRM_FULL=1 to start it." >&2
      exit 2
    fi
    paper32_preflight
    # Qwen3-VL EP32 is already complete.  The default continuation runs only
    # the remaining Qwen3.5-20L groups; set PAPER32_MODELS explicitly to
    # reconstruct or rerun the complete matrix.
    read -r -a models <<< "${PAPER32_MODELS:-qwen35_20l}"
    read -r -a datasets <<< "${PAPER32_DATASETS:-sharegpt4v tulu3}"
    read -r -a methods <<< "${PAPER32_METHODS:-baseline r2 eplb hiermoe ours}"
    port_index=0
    failures=0
    for model in "${models[@]}"; do
      for dataset in "${datasets[@]}"; do
        group_failures=0
        if ! PAPER32_PROFILE_MASTER_PORT=$((30600 + port_index)) \
          PAPER32_PROFILE_HCCL_PORT=$((52000 + port_index * 100)) \
          bash "${script_dir}/prepare_hiermoe_paper32_layouts.sh" "${model}" "${dataset}"
        then
          failures=$((failures + 1))
          group_failures=$((group_failures + 1))
          echo "skipping ${model}/${dataset} cases: layout preparation failed" >&2
          continue
        fi
        for method in "${methods[@]}"; do
          PAPER32_MASTER_PORT=$((30700 + port_index)) \
          PAPER32_HCCL_PORT=$((54000 + port_index * 100)) \
            bash "${script_dir}/run_hiermoe_paper32_case.sh" \
              "${model}" "${dataset}" "${method}" full \
            || {
              failures=$((failures + 1))
              group_failures=$((group_failures + 1))
            }
          port_index=$((port_index + 1))
        done
        if [[ "${group_failures}" == "0" ]]; then
          paper32_configure_model "${model}"
          paper32_configure_dataset "${dataset}"
          run_tag=${PAPER32_RUN_TAG:-20260730}
          chart_stem=${paper32_artifact_prefix}_${paper32_model_slug}_${paper32_dataset_slug}_speedup_vs_veomni_${run_tag}
          chart_args=(
            --model "${paper32_model_slug}"
            --dataset "${paper32_dataset_slug}"
            --output-svg "${paper32_source_root}/results/${chart_stem}.svg"
            --output-json "${paper32_source_root}/results/${chart_stem}.json"
            --output-csv "${paper32_source_root}/results/${chart_stem}.csv"
          )
          for method in "${methods[@]}"; do
            method_slug=$(paper32_method_slug "${method}")
            run_name=${paper32_artifact_prefix}_${paper32_model_slug}_${paper32_dataset_slug}_${method_slug}_full_${run_tag}
            chart_args+=(
              --summary
              "${method}=${paper32_source_root}/results/${run_name}_summary.json"
            )
          done
          if [[ " ${methods[*]} " == *" baseline "* ]]; then
            python "${script_dir}/plot_hiermoe_paper_speedup.py" "${chart_args[@]}" \
              || {
                failures=$((failures + 1))
                group_failures=$((group_failures + 1))
              }
          else
            echo "skipping ${model}/${dataset} speedup chart: baseline was not requested"
          fi
        else
          echo "skipping ${model}/${dataset} speedup chart: ${group_failures} group failures" >&2
        fi
      done
    done
    echo "paper32 matrix completed failures=${failures}"
    exit "${failures}"
    ;;
  smoke)
    paper32_preflight
    # These two cases cover the two new failure-prone paths without paying for
    # 20 repeated model loads: online exact P1 and 256-expert R2 through Adam.
    PAPER32_MASTER_PORT=${PAPER32_MASTER_PORT:-30790} \
    PAPER32_HCCL_PORT=${PAPER32_HCCL_PORT:-57000} \
      bash "${script_dir}/run_hiermoe_paper32_case.sh" \
        qwen3vl sharegpt4v hiermoe smoke
    PAPER32_MASTER_PORT=$((${PAPER32_MASTER_PORT:-30790} + 1)) \
    PAPER32_HCCL_PORT=$((${PAPER32_HCCL_PORT:-57000} + 100)) \
      bash "${script_dir}/run_hiermoe_paper32_case.sh" \
        qwen35_20l sharegpt4v r2 smoke
    ;;
  dry-run)
    read -r -a models <<< "${PAPER32_MODELS:-qwen35_20l}"
    read -r -a datasets <<< "${PAPER32_DATASETS:-sharegpt4v tulu3}"
    read -r -a methods <<< "${PAPER32_METHODS:-baseline r2 eplb hiermoe ours}"
    for model in "${models[@]}"; do
      for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
          PAPER32_DRY_RUN=1 bash "${script_dir}/run_hiermoe_paper32_case.sh" \
            "${model}" "${dataset}" "${method}" full
        done
      done
    done
    ;;
  *)
    echo "usage: $0 [full|smoke|dry-run]" >&2
    exit 2
    ;;
esac
