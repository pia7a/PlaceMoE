#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
cd "${repo_root}"

: "${RUN_NAME:?RUN_NAME must be set}"
: "${NODE_RANK:?NODE_RANK must be set}"
: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:?MASTER_PORT must be set}"
: "${OUTPUT_PATH:?OUTPUT_PATH must be set}"
: "${PREFLIGHT_REPORT:?PREFLIGHT_REPORT must be set}"
: "${SOURCE_SHA256:?SOURCE_SHA256 must be set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must define the topology-aware local rank order}"

python_bin=${PYTHON:-${PLACEMOE_REPRO_PYTHON:-${repo_root}/.venv/bin/python}}
export PYTHONPATH=${repo_root}
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibs0}
export NCCL_IB_DISABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF \
  VEOMNI_PLACEMOE_CONFIG \
  VEOMNI_HIERMOE_INITIAL_LAYOUT \
  VEOMNI_HIERMOE_INTERNAL_TIMING
cuda_lib_path=${CUDA_LIB_PATH:-}
if [[ -n "${cuda_lib_path}" ]]; then
  export LD_LIBRARY_PATH="${cuda_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"${python_bin}" -m torch.distributed.run \
  --nnodes=4 \
  --nproc-per-node=8 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  scripts/placemoe/reproduction/gpu_ep32/benchmark_hierarchy.py \
  --run-name "${RUN_NAME}" \
  --preflight-report "${PREFLIGHT_REPORT}" \
  --source-sha256 "${SOURCE_SHA256}" \
  --output "${OUTPUT_PATH}" \
  --hidden-size 2048 \
  --num-experts 128 \
  --top-k 8
