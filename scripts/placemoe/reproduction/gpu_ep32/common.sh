#!/usr/bin/env bash

# Shared, source-only configuration for the GPU EP32 reproduction matrix.

repro_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repro_default_source_root=$(cd "${repro_script_dir}/../../../.." && pwd)
repro_cluster_config_path=${PLACEMOE_REPRO_CONFIG:-${repro_default_source_root}/configs/placemoe/gpu_ep32.env}
if [[ -n "${PLACEMOE_REPRO_CONFIG:-}" && ! -r "${repro_cluster_config_path}" ]]; then
  echo "PLACEMOE_REPRO_CONFIG is not readable: ${repro_cluster_config_path}" >&2
  return 1
fi
if [[ -r "${repro_cluster_config_path}" ]]; then
  # shellcheck disable=SC1090
  source "${repro_cluster_config_path}"
fi

repro_source_root=${repro_default_source_root}
repro_python=${PLACEMOE_REPRO_PYTHON:-${repro_source_root}/.venv/bin/python}
repro_master_addr=${PLACEMOE_REPRO_MASTER_ADDR:-}
repro_nccl_socket_ifname=${PLACEMOE_REPRO_NCCL_SOCKET_IFNAME:-ibs0}
repro_cuda_lib_path=${PLACEMOE_REPRO_CUDA_LIB_PATH:-}
repro_hierarchy_group_sizes=${PLACEMOE_REPRO_HIERARCHY_GROUP_SIZES:-2,8,32}
repro_expected_accelerator=${PLACEMOE_REPRO_EXPECTED_ACCELERATOR:-NVIDIA RTX A6000}
repro_expected_torch=${PLACEMOE_REPRO_EXPECTED_TORCH:-2.9.1+cu129}
repro_expected_cuda=${PLACEMOE_REPRO_EXPECTED_CUDA:-12.9}
repro_expected_nccl=${PLACEMOE_REPRO_EXPECTED_NCCL:-2.27.5}
repro_expected_triton=${PLACEMOE_REPRO_EXPECTED_TRITON:-3.5.1}
repro_perf_model_rel=results/hiermoe_perf_model_gpu_ep32.json
repro_perf_model_host=${repro_source_root}/results/hiermoe_perf_model_gpu_ep32.json
repro_perf_model_container=${repro_source_root}/results/hiermoe_perf_model_gpu_ep32.json

repro_require_value() {
  local name=$1
  local value=${2:-}
  if [[ -z "${value}" ]]; then
    echo "${name} must be set in ${repro_cluster_config_path}" >&2
    return 2
  fi
}

repro_configure_model() {
  local model=$1
  case "${model}" in
    qwen3vl)
      repro_model_path=${PLACEMOE_REPRO_QWEN3VL_MODEL_PATH:-}
      repro_require_value PLACEMOE_REPRO_QWEN3VL_MODEL_PATH "${repro_model_path}"
      repro_model_slug=qwen3vl30b
      repro_model_label=Qwen3-VL-30B-A3B-Instruct
      repro_config_path=configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml
      repro_num_layers=48
      repro_num_experts=128
      repro_primary_slots=4
      repro_redundant_slots=4
      repro_slots_per_rank=8
      repro_hidden_size=2048
      repro_micro_batch_size=4
      repro_global_batch_size=128
      repro_freeze_vit=false
      ;;
    qwen35_20l|qwen35)
      repro_model_path=${PLACEMOE_REPRO_QWEN35_MODEL_PATH:-}
      repro_require_value PLACEMOE_REPRO_QWEN35_MODEL_PATH "${repro_model_path}"
      repro_model_slug=qwen35b20l
      repro_model_label=Qwen3.5-35B-A3B-20L
      repro_config_path=configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml
      repro_num_layers=20
      repro_num_experts=256
      repro_primary_slots=8
      repro_redundant_slots=8
      repro_slots_per_rank=16
      repro_hidden_size=2048
      repro_micro_batch_size=4
      repro_global_batch_size=128
      repro_freeze_vit=false
      ;;
    *)
      echo "unsupported model '${model}'; expected qwen3vl or qwen35_20l" >&2
      return 2
      ;;
  esac

  if [[ -n "${PLACEMOE_REPRO_REDUNDANT_SLOTS_OVERRIDE:-}" ]]; then
    if [[ ! "${PLACEMOE_REPRO_REDUNDANT_SLOTS_OVERRIDE}" =~ ^[0-9]+$ ]]; then
      echo "PLACEMOE_REPRO_REDUNDANT_SLOTS_OVERRIDE must be a non-negative integer" >&2
      return 2
    fi
    repro_redundant_slots=${PLACEMOE_REPRO_REDUNDANT_SLOTS_OVERRIDE}
    repro_slots_per_rank=$((repro_primary_slots + repro_redundant_slots))
  fi
  if [[ -n "${PLACEMOE_REPRO_MICRO_BATCH_SIZE_OVERRIDE:-}" ]]; then
    if [[ ! "${PLACEMOE_REPRO_MICRO_BATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
      echo "PLACEMOE_REPRO_MICRO_BATCH_SIZE_OVERRIDE must be a positive integer" >&2
      return 2
    fi
    repro_micro_batch_size=${PLACEMOE_REPRO_MICRO_BATCH_SIZE_OVERRIDE}
  fi
}

repro_checkpoint_sha256() {
  local digest
  if [[ -s "${repro_model_path}/qwen35_20l_manifest.json" ]]; then
    digest=$(sha256sum "${repro_model_path}/qwen35_20l_manifest.json")
    printf '%s\n' "${digest%% *}"
    return
  fi
  if [[ ! -s "${repro_model_path}/config.json" || ! -s "${repro_model_path}/model.safetensors.index.json" ]]; then
    echo "checkpoint metadata is incomplete: ${repro_model_path}" >&2
    return 1
  fi
  local -a checkpoint_files=(
    "${repro_model_path}/config.json"
    "${repro_model_path}/model.safetensors.index.json"
  )
  local shard
  while IFS= read -r shard; do
    checkpoint_files+=("${shard}")
  done < <(find "${repro_model_path}" -maxdepth 1 -type f -name '*.safetensors' | sort)
  if [[ "${#checkpoint_files[@]}" -le 2 ]]; then
    echo "checkpoint has no safetensor shards: ${repro_model_path}" >&2
    return 1
  fi
  digest=$(cd "${repro_model_path}" && sha256sum "${checkpoint_files[@]##*/}" | sha256sum)
  printf '%s\n' "${digest%% *}"
}
repro_communication_source_sha256() {
  local digest
  digest=$(sha256sum \
    "${repro_script_dir}/common.sh" \
    "${repro_script_dir}/calibrate_communication.sh" \
    "${repro_script_dir}/run_communication_node.sh" \
    "${repro_script_dir}/calibrate_communication.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/all_to_all.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/triton_segment_sum.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/greedy_planner.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/perf_model.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/state.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/__init__.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/placemoe/__init__.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/placemoe/runtime/__init__.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/placemoe/runtime/config.py" \
    "${repro_source_root}/veomni/distributed/moe/hiermoe/topology.py" \
    "${repro_source_root}/veomni/arguments/arguments_types.py" \
    "${repro_source_root}/veomni/arguments/__init__.py" \
    "${repro_source_root}/veomni/distributed/moe/comm.py" \
    "${repro_source_root}/veomni/distributed/moe/timing.py" \
    "${repro_source_root}/veomni/utils/accelerator_timing.py" \
    "${repro_source_root}/veomni/utils/device.py" \
    | sha256sum)
  printf "%s\n" "${digest%% *}"
}


repro_dataset_sha256() {
  local data_path=$1
  local digest
  if [[ -f "${data_path}" ]]; then
    digest=$(sha256sum "${data_path}")
  elif [[ -d "${data_path}" ]]; then
    digest=$(find "${data_path}" -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum)
  else
    echo "dataset path does not exist: ${data_path}" >&2
    return 1
  fi
  printf "%s\n" "${digest%% *}"
}

repro_cost_scope_sha256() {
  local checkpoint_sha256=$1
  local communication_calibration=$2
  local moe_impl=$3
  local micro_batch_size=$4
  local global_batch_size=$5
  local max_seq_len=$6
  local dataset_id=${7:?dataset_id is required}
  local dataset_sha256=${8:?dataset_sha256 is required}
  local data_source_name=${9:?data_source_name is required}
  local freeze_vit=${10:?freeze_vit is required}
  local communication_sha256 digest
  communication_sha256=$(sha256sum "${communication_calibration}")
  communication_sha256=${communication_sha256%% *}
  digest=$(
    {
      printf 'checkpoint=%s\ncommunication=%s\nmodel=%s\n' \
        "${checkpoint_sha256}" "${communication_sha256}" "${repro_model_slug}"
      printf 'moe_impl=%s\nmb=%s\ngbs=%s\nseq=%s\n' \
        "${moe_impl}" "${micro_batch_size}" "${global_batch_size}" "${max_seq_len}"
      printf 'dataset=%s\ndataset_sha256=%s\ndata_source=%s\nfreeze_vit=%s\n' \
        "${dataset_id}" "${dataset_sha256}" "${data_source_name}" "${freeze_vit}"
      printf 'layers=%s\nexperts=%s\nslots=%s\nhidden=%s\ndtype=bf16\n' \
        "${repro_num_layers}" "${repro_num_experts}" "${repro_slots_per_rank}" "${repro_hidden_size}"
      sha256sum \
        "${repro_source_root}/veomni/ops/kernels/moe/group_gemm.py" \
        "${repro_source_root}/veomni/ops/kernels/moe/_kernels/kernel/group_gemm.py" \
        "${repro_source_root}/veomni/distributed/moe/hiermoe/all_to_all.py" \
        "${repro_source_root}/veomni/distributed/moe/hiermoe/triton_segment_sum.py" \
        "${repro_source_root}/veomni/distributed/moe/hiermoe/expert_swap.py" \
        "${repro_source_root}/scripts/placemoe/reproduction/gpu_ep32/calibrate_cost_model.py" \
        "${repro_source_root}/scripts/placemoe/reproduction/gpu_ep32/cost_components.py"
    } | sha256sum
  )
  printf '%s\n' "${digest%% *}"
}

repro_configure_dataset() {
  local dataset=$1
  case "${dataset}" in
    sharegpt4v)
      repro_dataset_slug=sharegpt4v
      repro_data_path=${PLACEMOE_REPRO_SHAREGPT4V_DATA_PATH:-}
      repro_require_value PLACEMOE_REPRO_SHAREGPT4V_DATA_PATH "${repro_data_path}"
      repro_data_source_name=sharegpt4v_sft
      ;;
    tulu3)
      repro_dataset_slug=tulu3
      repro_data_path=${PLACEMOE_REPRO_TULU3_DATA_PATH:-}
      repro_require_value PLACEMOE_REPRO_TULU3_DATA_PATH "${repro_data_path}"
      repro_data_source_name=tulu-3-sft-mixture
      repro_freeze_vit=true
      ;;
    *)
      echo "unsupported dataset '${dataset}'; expected sharegpt4v or tulu3" >&2
      return 2
      ;;
  esac
}

repro_profile_name() {
  printf 'repro_profile_%s_%s_p4_20260729' "${repro_model_slug}" "${repro_dataset_slug}"
}

repro_layout_stem() {
  local method=$1
  printf 'repro_%s_%s_%s_b%s_ep32' \
    "${repro_model_slug}" \
    "${repro_dataset_slug}" \
    "${method}" \
    "${repro_redundant_slots}"
}

repro_ablation_layout_stem() {
  local ablation=$1
  local redundant_slots=$2
  printf 'repro_%s_%s_ablation_%s_b%s_ep32' \
    "${repro_model_slug}" \
    "${repro_dataset_slug}" \
    "${ablation}" \
    "${redundant_slots}"
}

repro_method_slug() {
  case "$1" in
    baseline)
      printf 'veomni_baseline'
      ;;
    r2)
      printf 'fixed_r2_hierarchical_dedup'
      ;;
    eplb)
      printf 'eplb_static_hierarchical_dedup'
      ;;
    hiermoe)
      printf 'hiermoe_exact_p1'
      ;;
    ours)
      printf 'ours_static_hierarchical_dedup'
      ;;
    *)
      echo "unsupported paper32 method '$1'" >&2
      return 2
      ;;
  esac
}

repro_method_grad_mode() {
  case "$1" in
    ours)
      # Hiding redundant-gradient synchronization is part of the complete
      # method.  R2/EPLB intentionally retain blocking synchronization.
      printf 'hidden'
      ;;
    baseline|r2|eplb|hiermoe|dedup)
      printf 'blocking'
      ;;
    static)
      printf 'hidden'
      ;;
    *)
      echo "unsupported paper32 method '$1'" >&2
      return 2
      ;;
  esac
}
