#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CONFIG MODEL DATASET [full|smoke]" >&2
  exit 2
fi

config=$1
model=$2
dataset=$3
mode=${4:-full}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "${script_dir}/../../.." && pwd)
config=$(realpath "${config}")
case "${config}" in
  "${source_root}"/*) ;;
  *)
    echo "PlaceMoE config must be stored inside ${source_root} so every training container can read it." >&2
    exit 2
    ;;
esac

relative_config=${config#"${source_root}"/}
initial_artifact=$(python "${script_dir}/../config_path.py" "${config}" initial_artifact)
case "${initial_artifact}" in
  "${source_root}"/*) ;;
  *)
    echo "Initial PlaceMoE artifact must be stored inside ${source_root}." >&2
    exit 2
    ;;
esac
relative_artifact=${initial_artifact#"${source_root}"/}
container_source_root=${PLACEMOE_CONTAINER_SOURCE_ROOT:-/workspace/output/hiermoe_greedy_swap_cover_20260722/src}
export VEOMNI_PLACEMOE_CONFIG=${container_source_root}/${relative_config}
container_name=${PAPER32_CONTAINER_NAME:-tzq_hiermoe_paper32_warmcache_20260729}

export PAPER32_WORLD_SIZE=32
export PAPER32_RANK0_HOST=${PAPER32_RANK0_HOST:-huawei1_node1}
export PAPER32_RANK1_HOST=${PAPER32_RANK1_HOST:-huawei1_node2}
export PAPER32_RANK2_HOST=${PAPER32_RANK2_HOST:-huawei2_node1}
export PAPER32_RANK3_HOST=${PAPER32_RANK3_HOST:-huawei2_node2}
export PAPER32_SKIP_PAPER_SUMMARY=${PAPER32_SKIP_PAPER_SUMMARY:-1}

if [[ "${PLACEMOE_SYNC_SOURCE:-1}" == "1" ]]; then
  bash "${source_root}/scripts/profile/sync_hiermoe_paper32_source.sh"
fi

hosts=(
  "${PAPER32_RANK0_HOST}"
  "${PAPER32_RANK1_HOST}"
  "${PAPER32_RANK2_HOST}"
  "${PAPER32_RANK3_HOST}"
)
if [[ "${PLACEMOE_SYNC_ARTIFACT:-1}" == "1" ]]; then
  for host in "${hosts[@]}"; do
    echo "deploying initial PlaceMoE artifact to ${host}"
    ssh -i "${PAPER32_SSH_KEY:-/root/.ssh/KeyPair-3bce.pem}" \
      -o BatchMode=yes -o StrictHostKeyChecking=no "root@${host}" \
      "mkdir -p '${source_root}/$(dirname "${relative_artifact}")'"
    rsync -az -e "ssh -i ${PAPER32_SSH_KEY:-/root/.ssh/KeyPair-3bce.pem} -o StrictHostKeyChecking=no" \
      "${initial_artifact}" "root@${host}:${source_root}/${relative_artifact}"
  done
fi
validation_pids=()
for host in "${hosts[@]}"; do
  (
    echo "validating PlaceMoE config on ${host}"
    ssh -i "${PAPER32_SSH_KEY:-/root/.ssh/KeyPair-3bce.pem}" \
      -o BatchMode=yes -o StrictHostKeyChecking=no "root@${host}" \
      docker exec -w "${container_source_root}" "${container_name}" \
      python scripts/placemoe/validate_config.py "${VEOMNI_PLACEMOE_CONFIG}"
  ) &
  validation_pids+=("$!")
done
for pid in "${validation_pids[@]}"; do
  wait "${pid}"
done

exec bash "${source_root}/scripts/profile/run_hiermoe_paper32_case.sh" \
  "${model}" "${dataset}" ours_full_replan "${mode}"
