#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

run_name=$1
host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
source_root=${host_root}/src
profile_rel=profile/runs/pretrain/${run_name}
key=/home/tzq/KeyPair-3bce.pem

mkdir -p "${source_root}/${profile_rel}"
for node in \
  "192.168.0.190:1" \
  "192.168.0.109:2" \
  "192.168.0.9:3"
do
  host=${node%%:*}
  rank=${node##*:}
  rsync -az \
    -e "ssh -i ${key} -o StrictHostKeyChecking=no" \
    "root@${host}:${source_root}/${profile_rel}/" \
    "${source_root}/${profile_rel}/"
  if ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    test -f "${host_root}/${run_name}_rank${rank}.host.log"
  then
    rsync -az \
      -e "ssh -i ${key} -o StrictHostKeyChecking=no" \
      "root@${host}:${host_root}/${run_name}_rank${rank}.host.log" \
      "${host_root}/"
  fi
done

echo "collected ${run_name} into ${source_root}/${profile_rel}"
