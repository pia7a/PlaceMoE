#!/usr/bin/env bash

set -euo pipefail

mode=${1:-dry-run}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repro_tag=${PLACEMOE_REPRO_TAG:-canonical_repro_20260802_v1}
master_port_base=${PLACEMOE_REPRO_MASTER_PORT_BASE:-30600}
hccl_port_base=${PLACEMOE_REPRO_HCCL_PORT_BASE:-52000}
default_cases=(
  ep32_qwen3vl_sharegpt4v_16k
  ep64_qwen3vl_tulu3_16k
  ep64_deepseekv3_tulu3_30k
  ep64_qwen3vl_sharegpt4v_16k
  ep64_qwen3vl_sharegpt4v_32k
)
read -r -a cases <<< "${PLACEMOE_REPRO_CASES:-${default_cases[*]}}"
read -r -a methods <<< "${PLACEMOE_REPRO_METHODS:-baseline r2 ours}"

case "${mode}" in
  dry-run|execute)
    ;;
  *)
    echo "usage: $0 [dry-run|execute]" >&2
    exit 2
    ;;
esac

for method in "${methods[@]}"; do
  case "${method}" in
    baseline|r2|ours)
      ;;
    *)
      echo "unsupported reproduction method '${method}'; expected baseline, r2, or ours" >&2
      exit 2
      ;;
  esac
done
if ((${#cases[@]} == 0 || ${#methods[@]} == 0)); then
  echo "the reproduction matrix must contain at least one case and method" >&2
  exit 2
fi
if [[ ! "${master_port_base}" =~ ^[0-9]+$ || ! "${hccl_port_base}" =~ ^[0-9]+$ ]]; then
  echo "reproduction port bases must be integers" >&2
  exit 2
fi
max_master_port=$((master_port_base + (${#cases[@]} - 1) * 20 + ${#methods[@]}))
max_hccl_port=$((hccl_port_base + (${#cases[@]} - 1) * 1000 + ${#methods[@]} * 100))
if ((master_port_base < 1024 || max_master_port > 65535 \
  || hccl_port_base < 1024 || max_hccl_port > 65535)); then
  echo "derived reproduction ports must remain in [1024, 65535]" >&2
  exit 2
fi

topology_env() {
  local world_size=$1
  if [[ "${world_size}" == "32" ]]; then
    printf '%s\n' \
      PAPER32_WORLD_SIZE=32 \
      PAPER32_ARTIFACT_PREFIX=paper32 \
      PAPER32_CLUSTER_SLUG=huawei2 \
      PAPER32_MASTER_ADDR=192.168.0.55 \
      PAPER32_RANK0_HOST=huawei2_node1 \
      PAPER32_RANK1_HOST=huawei2_node2 \
      PAPER32_RANK2_HOST=huawei2_node3 \
      PAPER32_RANK3_HOST=huawei2_node4
  else
    printf '%s\n' \
      PAPER32_WORLD_SIZE=64 \
      PAPER32_ARTIFACT_PREFIX=paper64 \
      PAPER32_CLUSTER_SLUG=huawei12 \
      PAPER32_MASTER_ADDR=192.168.0.84 \
      PAPER32_RANK0_HOST=huawei1_node1 \
      PAPER32_RANK1_HOST=huawei1_node2 \
      PAPER32_RANK2_HOST=huawei1_node3 \
      PAPER32_RANK3_HOST=huawei1_node4 \
      PAPER32_RANK4_HOST=huawei2_node1 \
      PAPER32_RANK5_HOST=huawei2_node2 \
      PAPER32_RANK6_HOST=huawei2_node3 \
      PAPER32_RANK7_HOST=huawei2_node4
  fi
}

case_env() {
  local case_name=$1
  local world_size model dataset max_seq micro_batch global_batch learning_rate
  local compute_calibration= lightweight_timing=0

  case "${case_name}" in
    ep32_qwen3vl_sharegpt4v_16k)
      world_size=32
      model=qwen3vl
      dataset=sharegpt4v
      max_seq=4096
      micro_batch=4
      global_batch=128
      learning_rate=
      ;;
    ep64_qwen3vl_tulu3_16k)
      world_size=64
      model=qwen3vl
      dataset=tulu3
      max_seq=4096
      micro_batch=4
      global_batch=256
      learning_rate=
      ;;
    ep64_deepseekv3_tulu3_30k)
      world_size=64
      model=deepseek_v3_6moe_half
      dataset=tulu3
      max_seq=15360
      micro_batch=2
      global_batch=256
      learning_rate=0
      lightweight_timing=1
      compute_calibration=results/paper64_deepseekv3_6moe_half_compute_calibration_ep64_mb2_ga2_seq15k_lr0_v2_region.json
      ;;
    ep64_qwen3vl_sharegpt4v_16k)
      world_size=64
      model=qwen3vl
      dataset=sharegpt4v
      max_seq=4096
      micro_batch=4
      global_batch=256
      learning_rate=
      ;;
    ep64_qwen3vl_sharegpt4v_32k)
      world_size=64
      model=qwen3vl
      dataset=sharegpt4v
      max_seq=8192
      micro_batch=4
      global_batch=256
      learning_rate=
      ;;
    *)
      echo "unknown reproduction case '${case_name}'" >&2
      return 2
      ;;
  esac

  topology_env "${world_size}"
  printf '%s\n' \
    "PLACEMOE_CASE_WORLD_SIZE=${world_size}" \
    "PLACEMOE_CASE_MODEL=${model}" \
    "PLACEMOE_CASE_DATASET=${dataset}" \
    "PAPER32_PROFILE_TAG=${repro_tag}_${case_name}" \
    "PAPER32_LAYOUT_TAG=${repro_tag}_${case_name}" \
    "PAPER32_RUN_TAG=${repro_tag}_${case_name}" \
    "PAPER32_REUSE_PROFILE=0" \
    "PAPER32_REUSE_LAYOUTS=1" \
    "PAPER32_PREPARE_EPLB=0" \
    "PAPER32_MAX_SEQ_LEN=${max_seq}" \
    "PAPER32_MICRO_BATCH_SIZE_OVERRIDE=${micro_batch}" \
    "PAPER32_GLOBAL_BATCH_SIZE_OVERRIDE=${global_batch}" \
    "PAPER32_LR=${learning_rate}" \
    "PAPER32_MAX_STEPS_OVERRIDE=20" \
    "PAPER32_STATS_START_STEP_OVERRIDE=11" \
    "PAPER32_STATS_END_STEP_OVERRIDE=20" \
    "PAPER32_LIGHTWEIGHT_TIMING=${lightweight_timing}" \
    "PAPER32_SKIP_COMPLETED=1" \
    "PAPER32_SKIP_PAPER_SUMMARY=0" \
    "PAPER32_CONVERGENCE_METRICS=0"
  if [[ -n "${compute_calibration}" ]]; then
    printf '%s\n' "PAPER32_COMPUTE_CALIBRATION_ARTIFACT=${compute_calibration}"
  fi
}

run_preflight() {
  local -a env64 env32
  mapfile -t env64 < <(topology_env 64)
  mapfile -t env32 < <(topology_env 32)
  env "${env64[@]}" bash "${script_dir}/sync_hiermoe_paper32_source.sh"
  env "${env64[@]}" bash "${script_dir}/prepare_hiermoe_paper32_containers.sh"
  env "${env32[@]}" bash "${script_dir}/prepare_hiermoe_paper32_containers.sh"
}

for case_name in "${cases[@]}"; do
  case_env "${case_name}" >/dev/null
done

dry_run_pause_file=
if [[ "${mode}" == "dry-run" ]]; then
  dry_run_pause_dir=$(mktemp -d)
  dry_run_pause_file=${dry_run_pause_dir}/pause
  trap 'rmdir "${dry_run_pause_dir}" 2>/dev/null || true' EXIT
fi

case_index=0
if [[ "${mode}" == "execute" ]]; then
  run_preflight
fi

for case_name in "${cases[@]}"; do
  mapfile -t configured < <(case_env "${case_name}")
  world_size=
  model=
  dataset=
  run_env=()
  for item in "${configured[@]}"; do
    case "${item}" in
      PLACEMOE_CASE_WORLD_SIZE=*) world_size=${item#*=} ;;
      PLACEMOE_CASE_MODEL=*) model=${item#*=} ;;
      PLACEMOE_CASE_DATASET=*) dataset=${item#*=} ;;
      *) run_env+=("${item}") ;;
    esac
  done
  if [[ -z "${world_size}" || -z "${model}" || -z "${dataset}" ]]; then
    echo "incomplete case configuration for ${case_name}" >&2
    exit 2
  fi

  profile_port=$((master_port_base + case_index * 20))
  profile_hccl_port=$((hccl_port_base + case_index * 1000))
  echo "===== ${case_name}: EP${world_size} ${model}/${dataset} ====="
  if [[ "${mode}" == "execute" ]]; then
    env "${run_env[@]}" \
      PAPER32_PROFILE_MASTER_PORT="${profile_port}" \
      PAPER32_PROFILE_HCCL_PORT="${profile_hccl_port}" \
      bash "${script_dir}/prepare_hiermoe_paper32_layouts.sh" "${model}" "${dataset}"
  fi

  method_index=0
  for method in "${methods[@]}"; do
    method_port=$((profile_port + method_index + 1))
    method_hccl_port=$((profile_hccl_port + (method_index + 1) * 100))
    if [[ "${mode}" == "dry-run" ]]; then
      env "${run_env[@]}" \
        PAPER32_DRY_RUN=1 \
        PAPER32_SKIP_COMPLETED=0 \
        PAPER32_PAUSE_FILE="${dry_run_pause_file}" \
        PAPER32_MASTER_PORT="${method_port}" \
        PAPER32_HCCL_PORT="${method_hccl_port}" \
        bash "${script_dir}/run_hiermoe_paper32_case.sh" \
          "${model}" "${dataset}" "${method}" full
    else
      env "${run_env[@]}" \
        PAPER32_MASTER_PORT="${method_port}" \
        PAPER32_HCCL_PORT="${method_hccl_port}" \
        bash "${script_dir}/run_hiermoe_paper32_case.sh" \
          "${model}" "${dataset}" "${method}" full
    fi
    method_index=$((method_index + 1))
  done
  case_index=$((case_index + 1))
done

echo "PlaceMoE paper reproduction ${mode} completed for ${#cases[@]} cases"
