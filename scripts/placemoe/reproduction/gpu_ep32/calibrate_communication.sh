#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
remote_root=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repo_root}}
# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"

calibration_tag=${PLACEMOE_REPRO_COMM_CALIBRATION_TAG:-$(date +%Y%m%d_%H%M%S)}
run_name=${PLACEMOE_REPRO_COMM_CALIBRATION_RUN_NAME:-gpu32_ep32_a6000_communication_${calibration_tag}}
output=${PLACEMOE_REPRO_COMM_CALIBRATION_OUTPUT:-${repo_root}/results/${run_name}.json}
master_addr=${MASTER_ADDR_OVERRIDE:-${repro_master_addr}}
repro_require_value PLACEMOE_REPRO_MASTER_ADDR "${master_addr}"
master_port=${PLACEMOE_REPRO_COMM_CALIBRATION_MASTER_PORT:-29930}
node_script=${script_dir}/run_communication_node.sh
remote_node_script=${remote_root}/scripts/placemoe/reproduction/gpu_ep32/run_communication_node.sh
preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:?PLACEMOE_REPRO_PREFLIGHT_REPORT must point to the current four-node preflight report}
if [[ ! -s "${preflight_report}" ]]; then
  echo "missing EP32 preflight report: ${preflight_report}" >&2
  exit 1
fi
comm_source_sha256=$(repro_communication_source_sha256)

if [[ -e "${output}" ]]; then
  echo "refusing to overwrite communication calibration: ${output}" >&2
  exit 1
fi
mkdir -p "${repo_root}/pretrain_runs" "$(dirname "${output}")"
common_env=(
  "RUN_NAME=${run_name}"
  "MASTER_ADDR=${master_addr}"
  "MASTER_PORT=${master_port}"
  "OUTPUT_PATH=${output}"
  "PREFLIGHT_REPORT=${preflight_report}"
  "COMM_SOURCE_SHA256=${comm_source_sha256}"
  "NCCL_SOCKET_IFNAME=${repro_nccl_socket_ifname}"
  "CUDA_LIB_PATH=${repro_cuda_lib_path}"
  "PYTHON=${PYTHON:-${repro_python}}"
)
quote_env() {
  local target_root=${1:?target root is required}
  local value
  for value in "${common_env[@]}"; do
    value=${value//"${repo_root}"/"${target_root}"}
    printf "%q " "${value}"
  done
}
launch_remote() {
  local port=$1
  local node_rank=$2
  local command
  command="cd $(printf %q "${remote_root}") && env $(quote_env "${remote_root}") NODE_RANK=${node_rank} bash $(printf %q "${remote_node_script}")"
  repro_ssh "${port}" "${command}" \
    >"${repo_root}/pretrain_runs/${run_name}_rank${node_rank}.host.log" 2>&1
}

pids=()
for spec in "${repro_remote_specs[@]}"; do
  launch_remote "${spec%%:*}" "${spec##*:}" &
  pids+=("$!")
done
set +e
env "${common_env[@]}" NODE_RANK=0 bash "${node_script}" \
  >"${repo_root}/pretrain_runs/${run_name}_rank0.host.log" 2>&1
rank0_rc=$?
remote_rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || remote_rc=1
done
set -e
if ((rank0_rc != 0 || remote_rc != 0)); then
  echo "communication calibration failed: run=${run_name} rank0_rc=${rank0_rc} remote_rc=${remote_rc}" >&2
  exit 1
fi
if [[ ! -s "${output}" ]]; then
  echo "communication calibration produced no artifact: ${output}" >&2
  exit 1
fi
PYTHONPATH="${repo_root}" "${PYTHON:-${repro_python}}" -c \
  'import sys; from pathlib import Path; from scripts.placemoe.reproduction.gpu_ep32.cost_components import load_communication_calibration; load_communication_calibration(Path(sys.argv[1]), ep_size=32, ranks_per_node=8, hidden_size=2048, bytes_per_element=2, preflight_report=Path(sys.argv[2]), communication_source_sha256=sys.argv[3]); print(sys.argv[1])' \
  "${output}" "${preflight_report}" "${comm_source_sha256}"
bash "${script_dir}/publish_artifacts.sh" "${output}"
echo "EP32 communication calibration ready: ${output}"
