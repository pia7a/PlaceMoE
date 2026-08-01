#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
calibration_script=${script_dir}/calibrate_hiermoe_paper32_network.sh
repeat_count=${HIERMOE_CALIBRATION_REPEAT_COUNT:-6}
tag=${HIERMOE_CALIBRATION_REPEAT_TAG:-20260731_robust_v1}
message_bytes=${HIERMOE_CALIBRATION_MESSAGE_BYTES:-67108864,83886080,100663296,134217728,201326592,268435456,402653184,536870912}

if ((repeat_count < 4)); then
  echo "HIERMOE_CALIBRATION_REPEAT_COUNT must be at least 4" >&2
  exit 2
fi

run_topology() {
  local world_size=$1
  local cluster_slug=$2
  local hierarchy=$3
  local name_prefix=$4
  local master_port_base=$5
  local hccl_port_base=$6
  local repetition

  for ((repetition = 1; repetition <= repeat_count; ++repetition)); do
    local name=${name_prefix}_r${repetition}_${tag}
    local output=/home/tzq/npu_profile_outputs/${name}/hiermoe_perf_model.json
    if [[ -s "${output}" ]]; then
      echo "reuse completed calibration: ${output}"
      continue
    fi
    echo "run ${cluster_slug}/EP${world_size} repetition ${repetition}/${repeat_count}"
    PAPER32_WORLD_SIZE=${world_size} \
    PAPER32_CLUSTER_SLUG=${cluster_slug} \
    PAPER32_NETWORK_CALIBRATION_NAME=${name} \
    PAPER32_NETWORK_CALIBRATION_MASTER_PORT=$((master_port_base + repetition)) \
    PAPER32_NETWORK_CALIBRATION_HCCL_PORT=$((hccl_port_base + repetition * 32)) \
    PAPER32_NETWORK_CALIBRATION_HIERARCHY_GROUP_SIZES=${hierarchy} \
    PAPER32_NETWORK_CALIBRATION_MESSAGE_BYTES=${message_bytes} \
    PAPER32_NETWORK_CALIBRATION_WARMUP=3 \
    PAPER32_NETWORK_CALIBRATION_ITERS=7 \
    PAPER32_NETWORK_CALIBRATION_MEASURE_LAST_N=5 \
      bash "${calibration_script}"
  done
}

run_topology \
  32 huawei1 8,32 \
  hiermoe_perf_model_huawei1_ep32_robust \
  31300 58000
run_topology \
  64 huawei12 8,64 \
  hiermoe_perf_model_huawei12_ep64_robust \
  31400 60000

echo "completed ${repeat_count} independent calibrations per topology"
