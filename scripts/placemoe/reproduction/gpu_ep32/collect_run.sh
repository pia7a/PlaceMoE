#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

run_name=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
remote_root=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repo_root}}
require_moe_timing=${PLACEMOE_REPRO_COLLECT_REQUIRE_MOE_TIMING:-1}
profile_root="${repo_root}/profile/runs/pretrain/${run_name}"
remote_profile_root="${remote_root}/profile/runs/pretrain/${run_name}"

# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"

mkdir -p "${profile_root}"
for spec in "${repro_remote_specs[@]}"; do
  port=${spec%%:*}
  temporary_root=$(mktemp -d "/tmp/placemoe_profile_${port}_XXXXXX")
  repro_scp_from "${port}" "${remote_profile_root}" "${temporary_root}/"
  cp -a "${temporary_root}/${run_name}/." "${profile_root}/"
  rm -rf -- "${temporary_root}"
done

if [[ "${require_moe_timing}" == "1" ]]; then
  for rank in $(seq 0 31); do
    if ! find "${profile_root}/moe_timing" -type f -name "moe_timing_rank${rank}.jsonl" -print -quit | grep -q .; then
      echo "missing MoE timing for rank ${rank}: ${run_name}" >&2
      exit 1
    fi
  done
fi
echo "collected ${run_name} into ${profile_root}"
