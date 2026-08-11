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

tag=${PLACEMOE_REPRO_HIERARCHY_BENCHMARK_TAG:-$(date +%Y%m%d_%H%M%S)}
run_name=${PLACEMOE_REPRO_HIERARCHY_BENCHMARK_RUN_NAME:-gpu32_ep32_hierarchy_${tag}}
output=${PLACEMOE_REPRO_HIERARCHY_BENCHMARK_OUTPUT:-${repo_root}/results/${run_name}.json}
preflight_report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:?PLACEMOE_REPRO_PREFLIGHT_REPORT must point to the current preflight report}
master_addr=${MASTER_ADDR_OVERRIDE:-${repro_master_addr}}
repro_require_value PLACEMOE_REPRO_MASTER_ADDR "${master_addr}"
master_port=${PLACEMOE_REPRO_HIERARCHY_BENCHMARK_MASTER_PORT:-29940}
node_script=${script_dir}/run_hierarchy_node.sh
remote_node_script=${remote_root}/scripts/placemoe/reproduction/gpu_ep32/run_hierarchy_node.sh
rank0_devices=${PLACEMOE_REPRO_RANK0_DEVICE_ORDER:-0,1,2,7,3,4,5,6}
rank1_devices=${PLACEMOE_REPRO_RANK1_DEVICE_ORDER:-0,1,2,3,4,5,6,7}
rank2_devices=${PLACEMOE_REPRO_RANK2_DEVICE_ORDER:-0,1,2,3,4,5,6,7}
rank3_devices=${PLACEMOE_REPRO_RANK3_DEVICE_ORDER:-0,1,2,3,4,5,6,7}

if [[ ! -s "${preflight_report}" ]]; then
  echo "missing EP32 preflight report: ${preflight_report}" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "refusing to overwrite hierarchy benchmark: ${output}" >&2
  exit 1
fi
for order in "${rank0_devices}" "${rank1_devices}" "${rank2_devices}" "${rank3_devices}"; do
  if [[ ! "${order}" =~ ^[0-7](,[0-7]){7}$ ]]; then
    echo "invalid EP32 device order: ${order}" >&2
    exit 2
  fi
done

source_sha256=$(sha256sum \
  "${script_dir}/benchmark_hierarchy.sh" \
  "${script_dir}/run_hierarchy_node.sh" \
  "${script_dir}/benchmark_hierarchy.py" \
  "${repo_root}/veomni/distributed/moe/hiermoe/all_to_all.py" \
  "${repo_root}/veomni/distributed/moe/hiermoe/state.py" \
  "${repo_root}/veomni/distributed/moe/hiermoe/topology.py" \
  "${repo_root}/veomni/arguments/arguments_types.py" \
  | sha256sum)
source_sha256=${source_sha256%% *}
mkdir -p "${repo_root}/pretrain_runs" "$(dirname "${output}")"

common_env=(
  "RUN_NAME=${run_name}"
  "MASTER_ADDR=${master_addr}"
  "MASTER_PORT=${master_port}"
  "OUTPUT_PATH=${output}"
  "PREFLIGHT_REPORT=${preflight_report}"
  "SOURCE_SHA256=${source_sha256}"
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
device_order_for_rank() {
  case "${1:?node rank is required}" in
    0) printf "%s\n" "${rank0_devices}" ;;
    1) printf "%s\n" "${rank1_devices}" ;;
    2) printf "%s\n" "${rank2_devices}" ;;
    3) printf "%s\n" "${rank3_devices}" ;;
    *) return 2 ;;
  esac
}
launch_remote() {
  local port=$1
  local node_rank=$2
  local devices command
  devices=$(device_order_for_rank "${node_rank}")
  command="cd $(printf %q "${remote_root}") && env $(quote_env "${remote_root}") NODE_RANK=${node_rank} CUDA_VISIBLE_DEVICES=$(printf %q "${devices}") bash $(printf %q "${remote_node_script}")"
  repro_ssh "${port}" "${command}" \
    >"${repo_root}/pretrain_runs/${run_name}_rank${node_rank}.host.log" 2>&1
}

pids=()
for spec in "${repro_remote_specs[@]}"; do
  launch_remote "${spec%%:*}" "${spec##*:}" &
  pids+=("$!")
done
set +e
env "${common_env[@]}" NODE_RANK=0 CUDA_VISIBLE_DEVICES="${rank0_devices}" bash "${node_script}" \
  >"${repo_root}/pretrain_runs/${run_name}_rank0.host.log" 2>&1
rank0_rc=$?
remote_rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || remote_rc=1
done
set -e
if ((rank0_rc != 0 || remote_rc != 0)); then
  echo "hierarchy benchmark failed: run=${run_name} rank0_rc=${rank0_rc} remote_rc=${remote_rc}" >&2
  exit 1
fi
if [[ ! -s "${output}" ]]; then
  echo "hierarchy benchmark produced no artifact: ${output}" >&2
  exit 1
fi
PYTHONPATH="${repo_root}" "${PYTHON:-${repro_python}}" -c \
  'import json, sys; from pathlib import Path; payload=json.loads(Path(sys.argv[1]).read_text()); assert payload["schema_version"] == 1; assert payload["source"] == "gpu32-a6000-ep32-hierarchy-benchmark"; assert payload["scope"]["communication_source_sha256"] == sys.argv[2]; print(json.dumps(payload["aggregate"], sort_keys=True)); print(json.dumps(payload["performance_gate"], sort_keys=True))' \
  "${output}" "${source_sha256}"
bash "${script_dir}/publish_artifacts.sh" "${output}"
echo "EP32 hierarchy benchmark ready: ${output}"
