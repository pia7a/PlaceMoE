#!/usr/bin/env bash
# Qwen3-VL/Qwen3.5 four-node, 32-GPU HierMoE E2E launcher.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
cd "${repo_root}"

run_name=${RUN_NAME:?RUN_NAME must be set}
run_root=${RUN_ROOT:-"${repo_root}/pretrain_runs/${run_name}"}
model_path=${MODEL_PATH:?MODEL_PATH must be set by the EP32 launcher}
model_config_path=${MODEL_CONFIG_PATH:-${model_path}}
data_path=${DATA_PATH:?DATA_PATH must be set by the EP32 launcher}
data_source_name=${DATA_SOURCE_NAME:-sharegpt4v_sft}
moe_monitor_interval=${MOE_MONITOR_INTERVAL:-1}

nnodes=${NNODES:-4}
node_rank=${NODE_RANK:?NODE_RANK must be set}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${node_rank}" == "0" ]]; then
    export CUDA_VISIBLE_DEVICES=${PLACEMOE_REPRO_RANK0_DEVICE_ORDER:-0,1,2,7,3,4,5,6}
  else
    export CUDA_VISIBLE_DEVICES=${PLACEMOE_REPRO_REMOTE_DEVICE_ORDER:-0,1,2,3,4,5,6,7}
  fi
fi
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:?MASTER_ADDR must be set}
master_port=${MASTER_PORT:-29500}
world_size=$((nnodes * nproc_per_node))

max_steps=${MAX_STEPS:-2}
empty_cache_steps=${EMPTY_CACHE_STEPS:-500}
micro_batch_size=${MICRO_BATCH_SIZE:-1}
global_batch_size=${GLOBAL_BATCH_SIZE:-32}
max_seq_len=${MAX_SEQ_LEN:-4096}
data_num_workers=${DATA_NUM_WORKERS:-4}
data_prefetch_factor=${DATA_PREFETCH_FACTOR:-2}
freeze_vit=${TRAIN_FREEZE_VIT:-false}
num_moe_layers=${NUM_MOE_LAYERS:-48}

dp_replicate_size=${DP_REPLICATE_SIZE:-1}
dp_shard_size=${DP_SHARD_SIZE:-${world_size}}
ep_size=${EP_SIZE:-${world_size}}
moe_impl=${MOE_IMPL:-fused_triton}
ulysses_size=${ULYSSES_SIZE:-1}
attn_impl=${ATTN_IMPL:-flash_attention_2}

hiermoe_enable=${HIERMOE_ENABLE:-false}
hiermoe_token_dedup=${HIERMOE_TOKEN_DEDUP:-true}
hiermoe_communication_mode=${HIERMOE_COMMUNICATION_MODE:-hierarchical}
hiermoe_expert_swap=${HIERMOE_EXPERT_SWAP:-false}
hiermoe_swap_interval=${HIERMOE_EXPERT_SWAP_INTERVAL:-1}
hiermoe_max_pairs=${HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER:-0}
hiermoe_selector=${HIERMOE_EXPERT_SWAP_SELECTOR:-current_joint}
hiermoe_redundant_slots=${HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE:-0}
hiermoe_max_copies=${HIERMOE_GREEDY_MAX_COPIES_PER_EXPERT:-8}
hiermoe_search_rounds=${HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS:-0}
hiermoe_swap_mode=${HIERMOE_EXPERT_SWAP_MODE:-step}
hiermoe_fixed_pipeline=${HIERMOE_FIXED_PIPELINE_OVERLAP:-false}
hiermoe_perf_model_path=${HIERMOE_PERF_MODEL_PATH:-}
hiermoe_fit_perf_model=${HIERMOE_FIT_PERF_MODEL_ON_STARTUP:-false}
hiermoe_hierarchy_group_sizes=${HIERMOE_HIERARCHY_GROUP_SIZES:-2,8,32}

profile_enable=${VEOMNI_TORCH_PROFILE_ENABLE:-false}
profile_start=${VEOMNI_TORCH_PROFILE_START_STEP:-11}
profile_end=${VEOMNI_TORCH_PROFILE_END_STEP:-20}
profile_rank0_only=${VEOMNI_TORCH_PROFILE_RANK0_ONLY:-true}
profile_dir=${VEOMNI_TORCH_PROFILE_DIR:-"${repo_root}/profile/runs/pretrain/${run_name}/torch_profiler"}

export VERL_MOE_PROFILE_DIR=${VERL_MOE_PROFILE_DIR:-"${repo_root}/profile/runs/pretrain/${run_name}"}
export VERL_MOE_MONITOR_DIR=${VERL_MOE_MONITOR_DIR:-"${VERL_MOE_PROFILE_DIR}/moe_monitor"}
if [[ "${VEOMNI_MOE_TIMING_ENABLE:-1}" == "1" ]]; then
  export VERL_MOE_TIMING_DIR=${VERL_MOE_TIMING_DIR:-"${VERL_MOE_PROFILE_DIR}/moe_timing"}
else
  unset VERL_MOE_TIMING_DIR
fi
export VERL_MOE_TIMING_NUM_LAYERS=${VERL_MOE_TIMING_NUM_LAYERS:-${num_moe_layers}}
export VEOMNI_FULL_PROFILE_ENABLE=${VEOMNI_FULL_PROFILE_ENABLE:-1}
export VEOMNI_FULL_PROFILE_DIR=${VEOMNI_FULL_PROFILE_DIR:-"${VERL_MOE_PROFILE_DIR}/full_timing"}
export VEOMNI_FULL_PROFILE_START_STEP=${VEOMNI_FULL_PROFILE_START_STEP:-11}
export VEOMNI_FULL_PROFILE_EVERY_N=${VEOMNI_FULL_PROFILE_EVERY_N:-1}
export VEOMNI_FULL_PROFILE_RANKS=${VEOMNI_FULL_PROFILE_RANKS:-0}
export VEOMNI_FULL_PROFILE_WITH_BACKWARD=1
export VEOMNI_FULL_PROFILE_RUN_KIND=pretrain
export VEOMNI_MOE_TIMING_SYNC_EVENTS=0
export VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS=${VEOMNI_MOE_TIMING_INDIVIDUAL_SPANS:-0}
export VEOMNI_TRUE_STEP_TIME=${VEOMNI_TRUE_STEP_TIME:-1}
export VEOMNI_HIERMOE_CUDA_SEGMENT_SUM=${VEOMNI_HIERMOE_CUDA_SEGMENT_SUM:-1}
export USE_LIBUV=${USE_LIBUV:-0}
if [[ "${HIERMOE_CAPTURE_ROUTES:-0}" == "1" ]]; then
  capture_root=${HIERMOE_CAPTURE_ROOT:-"${repo_root}/route_captures/${run_name}"}
  export VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE=local
  export VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH="${capture_root}/step{step:04d}/layer{layer_index:02d}_call{call}_rank{rank:02d}.pt"
  export VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP=${HIERMOE_CAPTURE_STEP:--1}
  export VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL=0
  export VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS=${VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS:-${num_moe_layers}}
fi

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibs0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
unset PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=false
cuda_lib_path=${CUDA_LIB_PATH:-}
if [[ -n "${cuda_lib_path}" ]]; then
  export LD_LIBRARY_PATH="${cuda_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -d "${model_path}" ]]; then
  echo "missing model directory: ${model_path}" >&2
  exit 1
fi
if [[ ! -e "${data_path}" ]]; then
  echo "missing dataset: ${data_path}" >&2
  exit 1
fi
if [[ "${world_size}" -ne 32 || "${ep_size}" -ne 32 ]]; then
  echo "paper32 GPU launcher requires 32 processes and EP_SIZE=32" >&2
  exit 1
fi

python_bin=${PYTHON:-${PLACEMOE_REPRO_PYTHON:-${repo_root}/.venv/bin/python}}
if [[ ! -x "${python_bin}" ]]; then
  echo "EP32 paper Python is not executable: ${python_bin}" >&2
  exit 1
fi
torchrun_cmd=()
if [[ -n "${TORCHRUN:-}" ]]; then
  torchrun_cmd=("${TORCHRUN}")
elif [[ -x "$(dirname "${python_bin}")/torchrun" ]]; then
  torchrun_cmd=("$(dirname "${python_bin}")/torchrun")
else
  # Preserve the selected venv for every distributed worker.
  torchrun_cmd=("${python_bin}" -m torch.distributed.run)
fi

if [[ "${GPU_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  "${python_bin}" - \
    "${model_path}" \
    "${data_path}" \
    "${nproc_per_node}" \
    "${NCCL_SOCKET_IFNAME}" \
    "${PLACEMOE_REPRO_EXPECTED_ACCELERATOR:-NVIDIA RTX A6000}" \
    "${PLACEMOE_REPRO_EXPECTED_TORCH:-2.9.1+cu129}" \
    "${PLACEMOE_REPRO_EXPECTED_CUDA:-12.9}" \
    "${PLACEMOE_REPRO_EXPECTED_NCCL:-2.27.5}" \
    "${PLACEMOE_REPRO_EXPECTED_TRITON:-3.5.1}" <<'PY'
import json
import os
import socket
import sys

import torch
import triton

with open(os.path.join(sys.argv[1], "config.json"), encoding="utf-8") as handle:
    model_type = json.load(handle).get("model_type")
if model_type != "qwen3_vl_moe":
    raise SystemExit(f"unsupported model_type={model_type!r}")
if not os.path.exists(sys.argv[2]):
    raise SystemExit(f"dataset is missing: {sys.argv[2]}")
expected_devices = int(sys.argv[3])
if not torch.cuda.is_available() or torch.cuda.device_count() < expected_devices:
    raise SystemExit("CUDA preflight failed")
if not torch.distributed.is_nccl_available():
    raise SystemExit("NCCL is unavailable")
interface = sys.argv[4]
if not os.path.exists(f"/sys/class/net/{interface}"):
    raise SystemExit(f"NCCL socket interface is missing: {interface}")
with open(f"/sys/class/net/{interface}/address", encoding="utf-8") as handle:
    interface_address = handle.read().strip()
expected_accelerator = sys.argv[5]
accelerators = [torch.cuda.get_device_name(index) for index in range(expected_devices)]
if any(name != expected_accelerator for name in accelerators):
    raise SystemExit(f"accelerator mismatch: expected={expected_accelerator!r}, actual={accelerators!r}")
actual_nccl = ".".join(str(value) for value in torch.cuda.nccl.version())
versions = {
    "torch": (torch.__version__, sys.argv[6]),
    "cuda": (torch.version.cuda, sys.argv[7]),
    "nccl": (actual_nccl, sys.argv[8]),
    "triton": (triton.__version__, sys.argv[9]),
}
version_mismatches = {
    name: {"actual": actual, "expected": expected}
    for name, (actual, expected) in versions.items()
    if actual != expected
}
if version_mismatches:
    raise SystemExit(f"software version mismatch: {version_mismatches}")
print(json.dumps({
    "status": "accepted",
    "hostname": socket.gethostname(),
    "network_interface_address": interface_address,
    "gpu_pci_bus_ids": [str(getattr(torch.cuda.get_device_properties(index), "pci_bus_id", "")) for index in range(expected_devices)],
    "model_type": model_type,
    "devices": expected_devices,
    "accelerator": expected_accelerator,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": torch.cuda.nccl.version(),
    "triton": triton.__version__,
    "nccl_socket_ifname": interface,
}, sort_keys=True))
PY
  exit 0
fi

hiermoe_args=(
  --train.hiermoe.enable "${hiermoe_enable}"
  --train.hiermoe.token_dedup "${hiermoe_token_dedup}"
  --train.hiermoe.communication_mode "${hiermoe_communication_mode}"
  --train.hiermoe.expert_swap "${hiermoe_expert_swap}"
  --train.hiermoe.expert_swap_interval "${hiermoe_swap_interval}"
  --train.hiermoe.expert_swap_max_pairs_per_layer "${hiermoe_max_pairs}"
  --train.hiermoe.expert_swap_selector "${hiermoe_selector}"
  --train.hiermoe.redundant_slot_increment_per_device "${hiermoe_redundant_slots}"
  --train.hiermoe.greedy_max_copies_per_expert "${hiermoe_max_copies}"
  --train.hiermoe.max_slot_op_search_rounds "${hiermoe_search_rounds}"
  --train.hiermoe.expert_swap_mode "${hiermoe_swap_mode}"
  --train.hiermoe.fixed_pipeline_overlap "${hiermoe_fixed_pipeline}"
  --train.hiermoe.fit_perf_model_on_startup "${hiermoe_fit_perf_model}"
  --train.hiermoe.use_from_step 0
)
if [[ -n "${hiermoe_perf_model_path}" ]]; then
  hiermoe_args+=(--train.hiermoe.perf_model_path "${hiermoe_perf_model_path}")
fi
if [[ -n "${hiermoe_hierarchy_group_sizes}" ]]; then
  IFS=',' read -r -a hierarchy_sizes <<< "${hiermoe_hierarchy_group_sizes}"
  hiermoe_args+=(--train.hiermoe.hierarchy_group_sizes "${hierarchy_sizes[@]}")
fi

run_dirs=("${run_root}" "${VERL_MOE_MONITOR_DIR}" "${VEOMNI_FULL_PROFILE_DIR}" "${profile_dir}")
if [[ -n "${VERL_MOE_TIMING_DIR:-}" ]]; then
  run_dirs+=("${VERL_MOE_TIMING_DIR}")
fi
mkdir -p "${run_dirs[@]}"

config_path=${CONFIG_PATH:-configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml}
echo "node=${node_rank} run=${run_name} master=${master_addr}:${master_port} model=${model_path}"

"${torchrun_cmd[@]}" \
  --nnodes="${nnodes}" \
  --nproc-per-node="${nproc_per_node}" \
  --node-rank="${node_rank}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  tasks/train_vlm.py "${config_path}" \
  --model.config_path "${model_config_path}" \
  --model.model_path "${model_path}" \
  --model.tokenizer_path "${model_path}" \
  --model.ops_implementation.moe_implementation "${moe_impl}" \
  --model.ops_implementation.attn_implementation "${attn_impl}" \
  --data.train_path "${data_path}" \
  --data.dataloader.type native \
  --data.datasets_type iterable \
  --data.source_name "${data_source_name}" \
  --data.dataloader.num_workers "${data_num_workers}" \
  --data.dataloader.prefetch_factor "${data_prefetch_factor}" \
  --data.max_seq_len "${max_seq_len}" \
  --train.freeze_vit "${freeze_vit}" \
  --train.micro_batch_size "${micro_batch_size}" \
  --train.global_batch_size "${global_batch_size}" \
  --train.max_steps "${max_steps}" \
  --train.num_train_epochs 1 \
  --train.empty_cache_steps "${empty_cache_steps}" \
  --train.accelerator.dp_replicate_size "${dp_replicate_size}" \
  --train.accelerator.dp_shard_size "${dp_shard_size}" \
  --train.accelerator.ulysses_size "${ulysses_size}" \
  --train.accelerator.ep_size "${ep_size}" \
  --train.moe_load_balance_monitor_interval "${moe_monitor_interval}" \
  --train.profile.enable "${profile_enable}" \
  --train.profile.start_step "${profile_start}" \
  --train.profile.end_step "${profile_end}" \
  --train.profile.trace_dir "${profile_dir}" \
  --train.profile.rank0_only "${profile_rank0_only}" \
  --train.wandb.enable false \
  --train.checkpoint.output_dir "${run_root}/ckpts" \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.hf_save_steps 0 \
  --train.checkpoint.hf_save_epochs 0 \
  --train.checkpoint.save_hf_weights false \
  "${hiermoe_args[@]}" \
  "$@" \
  2>&1 | tee "${run_root}/node${node_rank}.log"
