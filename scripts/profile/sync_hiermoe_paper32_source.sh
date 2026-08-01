#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"

excludes=(
  --exclude=/.git/
  --exclude=/.venv/
  --exclude='**/__pycache__/'
  --exclude=/pretrain_runs/
  --exclude=/profile/
  --exclude=/results/
  --exclude=/route_captures/
)
for host in "${paper32_hosts[@]}"; do
  echo "syncing source to ${host}"
  ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
    "mkdir -p '${paper32_source_root}'"
  rsync -az \
    "${excludes[@]}" \
    -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
    "${paper32_source_root}/" \
    "root@${host}:${paper32_source_root}/"
done

echo "paper32 source is synchronized"
