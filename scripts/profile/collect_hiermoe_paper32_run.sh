#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

run_name=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"

rsync_with_retry() {
  local attempt
  for attempt in 1 2 3; do
    if rsync "$@"; then
      return 0
    fi
    if ((attempt < 3)); then
      echo "rsync attempt ${attempt} failed; retrying in 2 seconds" >&2
      sleep 2
    fi
  done
  return 1
}

profile_rel=profile/runs/pretrain/${run_name}
mkdir -p "${paper32_source_root}/${profile_rel}"
for ((rank = 0; rank < paper32_nnodes; ++rank)); do
  if ((rank == 0)) && [[ "${PAPER32_RANK0_LOCAL:-0}" == "1" ]]; then
    # Rank 0 has already written directly into the local shared source tree.
    # Do not try to fetch it through the default remote host alias.
    continue
  fi
  host=${paper32_hosts[rank]}
  rsync_with_retry -az \
    -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
    "root@${host}:${paper32_source_root}/${profile_rel}/" \
    "${paper32_source_root}/${profile_rel}/"
  if ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
    test -f "${paper32_host_root}/${run_name}_rank${rank}.host.log"
  then
    rsync_with_retry -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "root@${host}:${paper32_host_root}/${run_name}_rank${rank}.host.log" \
      "${paper32_host_root}/"
  fi
done

echo "collected ${run_name} into ${paper32_source_root}/${profile_rel}"
