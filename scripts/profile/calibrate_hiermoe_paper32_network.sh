#!/usr/bin/env bash

set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"

calibration_tag=${PAPER32_NETWORK_CALIBRATION_TAG:-20260730}
calibration_name=${PAPER32_NETWORK_CALIBRATION_NAME:-hiermoe_perf_model_${paper32_cluster_slug}_ep${paper32_world_size}_${calibration_tag}}
host_output_root=/home/tzq/npu_profile_outputs/${calibration_name}
container_output_root=/workspace/output/${calibration_name}
bench_source=${PAPER32_PERF_BENCH_SOURCE:-/home/tzq/npu_profile_outputs/hiermoe_perf_model_c009_ep32_20260720/bench_hiermoe_perf_model.py}
bench_name=bench_hiermoe_perf_model.py
master_port=${PAPER32_NETWORK_CALIBRATION_MASTER_PORT:-30931}
hccl_port=${PAPER32_NETWORK_CALIBRATION_HCCL_PORT:-64000}
message_bytes=${PAPER32_NETWORK_CALIBRATION_MESSAGE_BYTES:-1048576,4194304,16777216,67108864,134217728,268435456,536870912}
warmup=${PAPER32_NETWORK_CALIBRATION_WARMUP:-2}
iters=${PAPER32_NETWORK_CALIBRATION_ITERS:-5}
measure_last_n=${PAPER32_NETWORK_CALIBRATION_MEASURE_LAST_N:-3}
hierarchy_group_sizes=${PAPER32_NETWORK_CALIBRATION_HIERARCHY_GROUP_SIZES:-}

if [[ ! -f "${bench_source}" ]]; then
  echo "missing benchmark source: ${bench_source}" >&2
  exit 2
fi

mkdir -p "${host_output_root}"

ssh_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh -o BatchMode=yes -o ConnectTimeout=20 "$@"; then
      return 0
    fi
    sleep $((attempt * 2))
  done
  return 1
}

rsync_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if rsync "$@"; then
      return 0
    fi
    sleep $((attempt * 2))
  done
  return 1
}

for host in "${paper32_hosts[@]}"; do
  ssh_retry "${host}" "mkdir -p '${host_output_root}'" || exit 1
  rsync_retry -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    "${bench_source}" "${host}:${host_output_root}/${bench_name}" || exit 1
done

launch_node() {
  local host=$1
  local node_rank=$2
  local log_path=${host_output_root}/network_calibration_node${node_rank}.log

  ssh_retry "${host}" \
    "docker exec \
      -e HCCL_IF_BASE_PORT='${hccl_port}' \
      -e HCCL_NPU_SOCKET_PORT_RANGE=auto \
      -e HCCL_CONNECT_TIMEOUT=7200 \
      -e HCCL_EXEC_TIMEOUT=7200 \
      -e HCCL_OP_EXPANSION_MODE=AIV \
      -e HCCL_BUFFSIZE=16 \
      -w '${container_output_root}' \
      '${paper32_container_name}' \
      torchrun \
        --nnodes="${paper32_nnodes}" \
        --nproc-per-node="${paper32_nproc_per_node}" \
        --node-rank='${node_rank}' \
        --master-addr='${paper32_master_addr}' \
        --master-port='${master_port}' \
        '${container_output_root}/${bench_name}' \
        --output-json='${container_output_root}/hiermoe_perf_model.json' \
        --details-json='${container_output_root}/hiermoe_perf_model_details_node${node_rank}.json' \
        --message-bytes-csv='${message_bytes}' \
        --ranks-per-node=8 \
        ${hierarchy_group_sizes:+--hierarchy-group-sizes-csv='${hierarchy_group_sizes}'} \
        --warmup='${warmup}' \
        --iters='${iters}' \
        --measure-last-n='${measure_last_n}' \
      >'${log_path}' 2>&1"
}

pids=()
for ((node_rank = 0; node_rank < paper32_nnodes; ++node_rank)); do
  launch_node "${paper32_hosts[${node_rank}]}" "${node_rank}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

for ((node_rank = 0; node_rank < paper32_nnodes; ++node_rank)); do
  host=${paper32_hosts[${node_rank}]}
  rsync_retry -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    "${host}:${host_output_root}/network_calibration_node${node_rank}.log" \
    "${host_output_root}/network_calibration_node${node_rank}.log" || rc=1
done

if ((rc != 0)); then
  echo "network calibration failed; inspect ${host_output_root}/network_calibration_node*.log" >&2
  exit "${rc}"
fi

rsync_retry -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "${paper32_hosts[0]}:${host_output_root}/hiermoe_perf_model.json" \
  "${host_output_root}/hiermoe_perf_model.json" || exit 1
rsync_retry -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "${paper32_hosts[0]}:${host_output_root}/hiermoe_perf_model_details_node0.json" \
  "${host_output_root}/hiermoe_perf_model_details.json" || exit 1

echo "network calibration completed: ${host_output_root}/hiermoe_perf_model.json"
