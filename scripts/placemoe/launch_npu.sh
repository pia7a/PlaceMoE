#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <training-entrypoint> <training-config> [VeOmni overrides ...]" >&2
  exit 2
fi

entrypoint=$1
config=$2
shift 2

: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${PLACEMOE_PYTHON:=.venv/bin/python}"

if [[ ! -x "${PLACEMOE_PYTHON}" ]]; then
  echo "PlaceMoE Python is not executable: ${PLACEMOE_PYTHON}" >&2
  echo "Create the documented system-site-packages venv and install PlaceMoE with uv pip, or set PLACEMOE_PYTHON." >&2
  exit 2
fi
if [[ ! -f "${entrypoint}" ]]; then
  echo "Training entrypoint does not exist: ${entrypoint}" >&2
  exit 2
fi
if [[ ! -f "${config}" ]]; then
  echo "Training config does not exist: ${config}" >&2
  exit 2
fi

if [[ "${PLACEMOE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "${PLACEMOE_PYTHON}" -m veomni.distributed.moe.hiermoe.placemoe.cli \
    doctor --config "${config}"
fi

exec "${PLACEMOE_PYTHON}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${entrypoint}" "${config}" "$@"
