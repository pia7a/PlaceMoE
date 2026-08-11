#!/usr/bin/env bash

# Copy only training source/config files to the remote EP32 nodes and verify them.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
remote_root=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repo_root}}
# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"

verify_files=(
  veomni/distributed/moe/hiermoe/all_to_all.py
  veomni/distributed/moe/hiermoe/triton_segment_sum.py
  veomni/distributed/moe/hiermoe/greedy_planner.py
  veomni/distributed/moe/hiermoe/perf_model.py
  veomni/distributed/moe/hiermoe/state.py
  veomni/distributed/moe/hiermoe/__init__.py
  veomni/distributed/moe/hiermoe/placemoe/__init__.py
  veomni/distributed/moe/hiermoe/placemoe/runtime/__init__.py
  veomni/distributed/moe/hiermoe/placemoe/runtime/config.py
  veomni/distributed/moe/hiermoe/topology.py
  veomni/arguments/arguments_types.py
  veomni/arguments/__init__.py
  veomni/distributed/moe/comm.py
  veomni/distributed/moe/timing.py
  veomni/utils/accelerator_timing.py
  veomni/utils/device.py
  scripts/placemoe/materialize_config.py
  scripts/placemoe/reproduction/gpu_ep32/common.sh
  scripts/placemoe/reproduction/gpu_ep32/calibrate_communication.sh
  scripts/placemoe/reproduction/gpu_ep32/run_communication_node.sh
  scripts/placemoe/reproduction/gpu_ep32/calibrate_communication.py
  scripts/placemoe/reproduction/gpu_ep32/benchmark_hierarchy.sh
  scripts/placemoe/reproduction/gpu_ep32/run_hierarchy_node.sh
  scripts/placemoe/reproduction/gpu_ep32/benchmark_hierarchy.py
  scripts/placemoe/reproduction/gpu_ep32/cost_components.py
  scripts/placemoe/reproduction/gpu_ep32/run_training_node.sh
  scripts/placemoe/reproduction/gpu_ep32/matrix.sh
)

for spec in "${repro_remote_specs[@]}"; do
  port=${spec%%:*}
  rank=${spec##*:}
  echo "syncing EP32 rank ${rank} through ${repro_ssh_host}:${port}"
  git -C "${repo_root}" ls-files -co --exclude-standard -z -- \
      veomni tasks configs scripts pyproject.toml uv.lock \
    | tar -C "${repo_root}" --null -T - -cf - \
    | setsid -w ssh "${repro_ssh_args[@]}" -p "${port}" "${repro_ssh_user}@${repro_ssh_host}" \
        "mkdir -p '${remote_root}' && tar -xf - -C '${remote_root}'"

  for relative_path in "${verify_files[@]}"; do
    local_digest=$(sha256sum "${repo_root}/${relative_path}")
    local_digest=${local_digest%% *}
    remote_digest=$(repro_ssh "${port}" "sha256sum '${remote_root}/${relative_path}'")
    remote_digest=${remote_digest%% *}
    if [[ "${local_digest}" != "${remote_digest}" ]]; then
      echo "source checksum mismatch on rank ${rank}: ${relative_path}" >&2
      exit 1
    fi
  done
done

echo "EP32 source sync completed: ${remote_root}"
