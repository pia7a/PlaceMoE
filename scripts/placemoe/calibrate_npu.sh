#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output-json> [calibrator arguments ...]" >&2
  exit 2
fi

output_json=$1
shift

: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${PLACEMOE_PYTHON:=.venv/bin/python}"
: "${EP_SIZE:=$((NNODES * NPROC_PER_NODE))}"

if [[ ! -x "${PLACEMOE_PYTHON}" ]]; then
  echo "PlaceMoE Python is not executable: ${PLACEMOE_PYTHON}" >&2
  exit 2
fi

exec "${PLACEMOE_PYTHON}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m placemoe.calibrate_network \
  --output-json "${output_json}" \
  --ep-size "${EP_SIZE}" \
  --ranks-per-node "${NPROC_PER_NODE}" \
  "$@"
