#!/usr/bin/env bash

set -euo pipefail

mode=dry-run
case "${1:-}" in
  "") ;;
  --config)
    if [[ $# -lt 2 || $# -gt 3 ]]; then
      echo "usage: $0 [--config PATH] [dry-run|smoke|full]" >&2
      exit 2
    fi
    export PLACEMOE_REPRO_CONFIG=$2
    mode=${3:-dry-run}
    ;;
  dry-run|smoke|full)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 [--config PATH] [dry-run|smoke|full]" >&2
      exit 2
    fi
    mode=$1
    ;;
  -h|--help)
    echo "usage: $0 [--config PATH] [dry-run|smoke|full]"
    exit 0
    ;;
  *)
    echo "usage: $0 [--config PATH] [dry-run|smoke|full]" >&2
    exit 2
    ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"

read -r -a models <<< "${PLACEMOE_REPRO_MODELS:-qwen3vl}"
read -r -a datasets <<< "${PLACEMOE_REPRO_DATASETS:-sharegpt4v tulu3}"
read -r -a methods <<< "${PLACEMOE_REPRO_METHODS:-baseline r2 eplb ours}"
run_tag=${PLACEMOE_REPRO_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
grad_protocol=${PLACEMOE_REPRO_GRAD_PROTOCOL:-paper}
e2e_source=${PLACEMOE_REPRO_E2E_SOURCE:-env_step_time_s}
repeats=${PLACEMOE_REPRO_REPEATS:-1}

case "${mode}" in
  dry-run|smoke|full) ;;
  *)
    echo "usage: $0 [--config PATH] [dry-run|smoke|full]" >&2
    exit 2
    ;;
esac
if [[ ! "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLACEMOE_REPRO_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ "${grad_protocol}" != paper && "${grad_protocol}" != blocking ]]; then
  echo "PLACEMOE_REPRO_GRAD_PROTOCOL must be paper or blocking" >&2
  exit 2
fi
if [[ "${e2e_source}" != env_step_time_s ]]; then
  echo "formal EP32 timing requires PLACEMOE_REPRO_E2E_SOURCE=env_step_time_s" >&2
  exit 2
fi
if [[ "${PLACEMOE_REPRO_FULL_PROFILE_ENABLE:-0}" != "0" || -n "${PLACEMOE_REPRO_FULL_PROFILE_METHODS:-}" ]]; then
  echo "formal EP32 timing does not allow profiler collection; run the representative profiler separately" >&2
  exit 2
fi
for method in "${methods[@]}"; do
  case "${method}" in
    baseline|r2|eplb|ours) ;;
    *)
      echo "unsupported formal method: ${method}" >&2
      exit 2
      ;;
  esac
done
if [[ "${mode}" == full && "${PLACEMOE_REPRO_CONFIRM_FULL:-0}" != "1" ]]; then
  cases=$((${#models[@]} * ${#datasets[@]} * ${#methods[@]} * repeats))
  echo "full mode runs ${cases} independent 5-step training cases plus calibration and route capture." >&2
  echo "set PLACEMOE_REPRO_CONFIRM_FULL=1 to start it." >&2
  exit 2
fi

python_bin=${PYTHON:-${repro_python}}
if [[ "${mode}" != dry-run && ! -x "${python_bin}" ]]; then
  echo "PlaceMoE reproduction Python is not executable: ${python_bin}" >&2
  exit 1
fi

preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:-${repro_source_root}/results/gpu_adaptation/gpu32_preflight_${run_tag}.json}
if [[ "${mode}" != dry-run && "${PLACEMOE_REPRO_RUN_PREFLIGHT:-1}" == "1" ]]; then
  PLACEMOE_REPRO_SYNC_SOURCE=${PLACEMOE_REPRO_SYNC_SOURCE:-1} \
    PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
    bash "${script_dir}/preflight.sh"
fi
if [[ "${mode}" != dry-run && ! -s "${preflight_report}" ]]; then
  echo "missing current EP32 preflight report: ${preflight_report}" >&2
  exit 1
fi

communication_calibration=${PLACEMOE_REPRO_COMM_CALIBRATION:-${repro_source_root}/results/gpu32_ep32_a6000_communication_${run_tag}.json}
if [[ "${mode}" != dry-run && ! -s "${communication_calibration}" ]]; then
  PLACEMOE_REPRO_COMM_CALIBRATION_OUTPUT="${communication_calibration}" \
    PLACEMOE_REPRO_COMM_CALIBRATION_TAG="matrix_${run_tag}" \
    PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
    bash "${script_dir}/calibrate_communication.sh"
fi
if [[ "${mode}" != dry-run && ! -s "${communication_calibration}" ]]; then
  echo "missing EP32 communication calibration: ${communication_calibration}" >&2
  exit 1
fi

if [[ "${mode}" != dry-run ]]; then
  communication_source_sha256=$(repro_communication_source_sha256)
  PYTHONPATH="${repro_source_root}" "${python_bin}" -c 'import sys; from pathlib import Path; from scripts.placemoe.reproduction.gpu_ep32.cost_components import load_communication_calibration; load_communication_calibration(Path(sys.argv[1]), ep_size=32, ranks_per_node=8, hidden_size=2048, bytes_per_element=2, preflight_report=Path(sys.argv[2]), communication_source_sha256=sys.argv[3])' \
    "${communication_calibration}" "${preflight_report}" "${communication_source_sha256}"
fi

port_index=0
for model in "${models[@]}"; do
  repro_configure_model "${model}"
  if [[ "${mode}" == dry-run ]]; then
    checkpoint_sha256=${PLACEMOE_REPRO_CHECKPOINT_SHA256_OVERRIDE:-0000000000000000000000000000000000000000000000000000000000000000}
  else
    checkpoint_sha256=${PLACEMOE_REPRO_CHECKPOINT_SHA256_OVERRIDE:-$(repro_checkpoint_sha256)}
  fi
  if [[ ! "${checkpoint_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid checkpoint fingerprint for ${model}" >&2
    exit 1
  fi
  for dataset in "${datasets[@]}"; do
    repro_configure_model "${model}"
    repro_configure_dataset "${dataset}"
    if [[ "${mode}" == dry-run ]]; then
      dataset_sha256=${PLACEMOE_REPRO_DRY_RUN_DATASET_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}
    else
      dataset_sha256=$(repro_dataset_sha256 "${repro_data_path}")
    fi
    if [[ "${mode}" == dry-run ]]; then
      model_cost_scope=${PLACEMOE_REPRO_DRY_RUN_COST_SCOPE:-0000000000000000000000000000000000000000000000000000000000000000}
    else
      model_cost_scope=$(repro_cost_scope_sha256 \
        "${checkpoint_sha256}" "${communication_calibration}" fused_triton \
        "${repro_micro_batch_size}" "${repro_global_batch_size}" 4096 \
        "${dataset}" "${dataset_sha256}" "${repro_data_source_name}" "${repro_freeze_vit}")
    fi
    model_layout_tag=${PLACEMOE_REPRO_LAYOUT_TAG_OVERRIDE:-${model_cost_scope:0:12}}
    model_cost_model="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_cost_scope:0:12}_cost_model.json"
    model_layout_bundle="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_layout_bundle.json"
    layout_reuse_profile=${PLACEMOE_REPRO_REUSE_PROFILE:-0}
    layout_route_root=${PLACEMOE_REPRO_ROUTE_ROOT:-}
    if [[ -n "${PLACEMOE_REPRO_LAYOUT_PROFILE_TAG:-}" ]]; then
      layout_reuse_profile=1
      layout_route_root="${repro_source_root}/route_captures/gpu32_profile_${repro_model_slug}_${repro_dataset_slug}_4step_${PLACEMOE_REPRO_LAYOUT_PROFILE_TAG}"
    fi

    if [[ "${mode}" != dry-run && ! -s "${model_cost_model}" ]]; then
      PLACEMOE_REPRO_COMM_CALIBRATION="${communication_calibration}" \
        PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
        PLACEMOE_REPRO_CHECKPOINT_SHA256="${checkpoint_sha256}" \
        PLACEMOE_REPRO_COST_MODEL="${model_cost_model}" \
        PLACEMOE_REPRO_CALIBRATION_TAG="matrix_${run_tag}" \
        PLACEMOE_REPRO_CALIBRATION_MASTER_PORT=$((29540 + port_index)) \
        bash "${script_dir}/calibrate_cost_model.sh" "${model}" "${dataset}"
    fi
    if [[ "${mode}" != dry-run ]]; then
      PLACEMOE_REPRO_COMM_CALIBRATION="${communication_calibration}" \
        PLACEMOE_REPRO_PREFLIGHT_REPORT="${preflight_report}" \
        PLACEMOE_REPRO_CHECKPOINT_SHA256="${checkpoint_sha256}" \
        PLACEMOE_REPRO_COST_MODEL="${model_cost_model}" \
        PLACEMOE_REPRO_REUSE_COST_MODEL=1 \
        PLACEMOE_REPRO_REUSE_PROFILE="${layout_reuse_profile}" \
        PLACEMOE_REPRO_ROUTE_ROOT="${layout_route_root}" \
        PLACEMOE_REPRO_LAYOUT_TAG="${model_layout_tag}" \
        PLACEMOE_REPRO_PROFILE_MASTER_PORT=$((29600 + port_index)) \
        bash "${script_dir}/prepare_layouts.sh" "${model}" "${dataset}"
    fi

    declare -A method_summary_lists=()
    declare -A method_grad_modes=()
    declare -A method_variants=()
    declare -A method_replays=()
    declare -A method_layout_reports=()
    declare -A method_layout_bundles=()
    declare -A method_configs=()
    for method in "${methods[@]}"; do
      variant=baseline
      replay=
      layout_report=
      layout_bundle=
      placemoe_config=
      case "${method}" in
        baseline)
          variant=baseline
          ;;
        r2)
          variant=replica
          ;;
        eplb)
          variant=static_layout
          replay="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_eplb_layout.json"
          layout_report="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_eplb_report.json"
          layout_bundle="${model_layout_bundle}"
          ;;
        ours)
          variant=static_layout
          replay="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_ours_layout.json"
          layout_report="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_ours_report.json"
          layout_bundle="${model_layout_bundle}"
          placemoe_config="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${model_layout_tag}_ours_placemoe.json"
          if [[ "${mode}" != dry-run ]]; then
            PYTHONPATH="${repro_source_root}" "${python_bin}" "${repro_source_root}/scripts/placemoe/materialize_config.py" --initial-artifact "${replay}" --output "${placemoe_config}"
          fi
          ;;
      esac
      method_grad_mode=blocking
      if [[ "${grad_protocol}" == paper && "${method}" == ours ]]; then
        method_grad_mode=hidden
      fi
      method_variants["${method}"]="${variant}"
      method_replays["${method}"]="${replay}"
      method_layout_reports["${method}"]="${layout_report}"
      method_layout_bundles["${method}"]="${layout_bundle}"
      method_grad_modes["${method}"]="${method_grad_mode}"
      method_configs["${method}"]="${placemoe_config}"
    done

    group_summaries=()
    dataset_execution_index=0
    method_count=${#methods[@]}
    execution_policy=repeat-major-fixed-order
    for repeat in $(seq 1 "${repeats}"); do
      for ((method_position = 0; method_position < method_count; method_position++)); do
        method=${methods[${method_position}]}
        variant=${method_variants["${method}"]}
        replay=${method_replays["${method}"]}
        layout_report=${method_layout_reports["${method}"]}
        layout_bundle=${method_layout_bundles["${method}"]}
        method_grad_mode=${method_grad_modes["${method}"]}
        placemoe_config=${method_configs["${method}"]}
        dataset_execution_index=$((dataset_execution_index + 1))
        run_name="gpu32_${repro_model_slug}_${repro_dataset_slug}_${method}_${grad_protocol}_${mode}_${run_tag}_r${repeat}"
        if [[ "${mode}" != dry-run && ( -e "${repro_source_root}/profile/runs/pretrain/${run_name}" || -e "${repro_source_root}/pretrain_runs/${run_name}" ) ]]; then
          echo "refusing to append to existing run: ${run_name}; choose a new PLACEMOE_REPRO_RUN_TAG" >&2
          exit 1
        fi
        max_steps=5
        if [[ "${mode}" == smoke ]]; then
          max_steps=2
        fi
        command=(
          env
          "E2E_VARIANT=${variant}"
          "RUN_NAME_OVERRIDE=${run_name}"
          "MODEL_PATH_OVERRIDE=${repro_model_path}"
          "CONFIG_PATH_OVERRIDE=${repro_config_path}"
          "MODEL_CONFIG_PATH_OVERRIDE=${repro_model_path}"
          "DATA_PATH_OVERRIDE=${repro_data_path}"
          "DATA_SOURCE_NAME_OVERRIDE=${repro_data_source_name}"
          "NUM_MOE_LAYERS_OVERRIDE=${repro_num_layers}"
          "MICRO_BATCH_SIZE_OVERRIDE=${repro_micro_batch_size}"
          "GLOBAL_BATCH_SIZE_OVERRIDE=${repro_global_batch_size}"
          "MAX_SEQ_LEN_OVERRIDE=4096"
          "TRAIN_FREEZE_VIT_OVERRIDE=${repro_freeze_vit}"
          "HIERMOE_REDUNDANT_SLOTS_OVERRIDE=${repro_redundant_slots}"
          "MAX_STEPS_OVERRIDE=${max_steps}"
          "EMPTY_CACHE_STEPS_OVERRIDE=${PLACEMOE_REPRO_EMPTY_CACHE_STEPS:-500}"
          "FULL_PROFILE_ENABLE_OVERRIDE=0"
          "TORCH_PROFILE_ENABLE_OVERRIDE=0"
          "MOE_MONITOR_INTERVAL_OVERRIDE=0"
          "VEOMNI_MOE_TIMING_ENABLE_OVERRIDE=0"
          "VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS_OVERRIDE=0"
          "MASTER_PORT=$((29700 + port_index))"
          "HIERMOE_FIT_PERF_MODEL_ON_STARTUP_OVERRIDE=false"
          "HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=${method_grad_mode}"
        )
        if [[ -n "${replay}" ]]; then
          if [[ ! -s "${replay}" && "${mode}" != dry-run ]]; then
            echo "missing layout: ${replay}" >&2
            exit 1
          fi
          if [[ "${method}" == ours ]]; then
            command+=("PLACEMOE_CONFIG_OVERRIDE=${placemoe_config}")
          else
            command+=("HIERMOE_INITIAL_LAYOUT_OVERRIDE=${replay}")
          fi
        fi
        command+=(bash "${script_dir}/launch.sh")
        echo "EP32 execution repeat=${repeat} position=${method_position} index=${dataset_execution_index} method=${method} policy=${execution_policy}"
        printf "%q " "${command[@]}"
        printf "\n"
        if [[ "${mode}" != dry-run ]]; then
          "${command[@]}"
          if [[ "${mode}" == full ]]; then
            PLACEMOE_REPRO_COLLECT_REQUIRE_MOE_TIMING=0 bash "${script_dir}/collect_run.sh" "${run_name}"
            summary_path="${repro_source_root}/results/${run_name}_summary.json"
            summary_args=(
              --run-name "${run_name}"
              --profile-root "${repro_source_root}/profile/runs/pretrain"
              --start-step 3
              --end-step 5
              --grad-mode "${method_grad_mode}"
              --skip-moe-timing
              --cost-model "${model_cost_model}"
              --communication-calibration "${communication_calibration}"
              --preflight-report "${preflight_report}"
              --communication-source-sha256 "${communication_source_sha256}"
              --repeat-index "${repeat}"
              --execution-index "${dataset_execution_index}"
              --execution-policy "${execution_policy}"
              --output "${summary_path}"
            )
            if [[ -n "${layout_report}" ]]; then
              summary_args+=(
                --layout-path "${replay}"
                --layout-report "${layout_report}"
                --layout-bundle "${layout_bundle}"
              )
            fi
            PYTHONPATH="${repro_source_root}" "${python_bin}" \
              "${repro_source_root}/scripts/profile/summarize_hiermoe_paper_case.py" "${summary_args[@]}"
            method_summary_lists["${method}"]+="${summary_path} "
          fi
        fi
        port_index=$((port_index + 1))
      done
    done

    if [[ "${mode}" == full ]]; then
      for method in "${methods[@]}"; do
        method_grad_mode=${method_grad_modes["${method}"]}
        read -r -a method_summaries <<< "${method_summary_lists["${method}"]}"
        aggregate_path="${repro_source_root}/results/gpu32_${repro_model_slug}_${repro_dataset_slug}_${method}_${grad_protocol}_full_${run_tag}_aggregate.json"
        aggregate_args=(
          --method "${method}"
          --model "${repro_model_slug}"
          --dataset "${repro_dataset_slug}"
          --grad-protocol "${grad_protocol}"
          --grad-mode "${method_grad_mode}"
          --expected-repeats "${repeats}"
          --output "${aggregate_path}"
        )
        for summary_path in "${method_summaries[@]}"; do
          aggregate_args+=(--summary "${summary_path}")
        done
        "${python_bin}" "${script_dir}/summarize_repeats.py" "${aggregate_args[@]}"
        group_summaries+=("${method}=${aggregate_path}")
      done
    fi

    if [[ "${mode}" == full ]]; then
      chart_stem="gpu32_ep32_${repro_model_slug}_${repro_dataset_slug}_speedup_vs_veomni_${grad_protocol}_${run_tag}"
      chart_args=(
        --model "${repro_model_label}"
        --dataset "${repro_dataset_slug}"
        --output-svg "${repro_source_root}/results/${chart_stem}.svg"
        --output-json "${repro_source_root}/results/${chart_stem}.json"
        --output-csv "${repro_source_root}/results/${chart_stem}.csv"
      )
      for summary in "${group_summaries[@]}"; do
        chart_args+=(--summary "${summary}")
      done
      PYTHONPATH="${repro_source_root}" "${python_bin}" \
        "${repro_source_root}/scripts/profile/plot_hiermoe_paper_speedup.py" "${chart_args[@]}"
    fi
  done
done
