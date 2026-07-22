#!/usr/bin/env bash
# Start or recreate the eight Ascend NPU profile containers used by the
# Qwen3-VL VeOmni pretraining profile runs.
#
# Run this script on the jump/head login node where /home/tzq/KeyPair-3bce.pem
# can SSH to every internal node as root.
#
# Examples:
#   bash scripts/profile/start_npu_profile_containers_8node.sh
#   ACTION=restart bash scripts/profile/start_npu_profile_containers_8node.sh
#   RECREATE=1 bash scripts/profile/start_npu_profile_containers_8node.sh

set -euo pipefail

KEY=${KEY:-/home/tzq/KeyPair-3bce.pem}
IMAGE=${IMAGE:-4923f32e1e9b}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-tzq_npu_profile_rank}
ACTION=${ACTION:-start}      # start or restart existing containers
RECREATE=${RECREATE:-0}      # set to 1 to remove and recreate containers
VERIFY=${VERIFY:-1}          # set to 0 to skip docker exec checks
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}

declare -A HOST_BY_RANK=(
  [0]=192.168.0.63
  [1]=192.168.0.164
  [2]=192.168.0.45
  [3]=192.168.0.93
  [4]=192.168.0.174
  [5]=192.168.0.213
  [6]=192.168.0.136
  [7]=192.168.0.210
)

if [[ ! -f "${KEY}" ]]; then
  echo "SSH key not found: ${KEY}" >&2
  exit 2
fi

if [[ "${ACTION}" != "start" && "${ACTION}" != "restart" ]]; then
  echo "ACTION must be start or restart, got: ${ACTION}" >&2
  exit 2
fi

ssh_base=(ssh -i "${KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}")

for rank in 0 1 2 3 4 5 6 7; do
  host=${HOST_BY_RANK[${rank}]}
  name=${CONTAINER_PREFIX}${rank}
  echo "===== rank${rank} ${host} ${name} ====="

  "${ssh_base[@]}" root@"${host}" \
    IMAGE="${IMAGE}" \
    NAME="${name}" \
    ACTION="${ACTION}" \
    RECREATE="${RECREATE}" \
    VERIFY="${VERIFY}" \
    'bash -s' <<'REMOTE'
set -euo pipefail

required_paths=(
  /home/tzq/VeOmni-0.1.11
  /home/share/Qwen3-VL-30B-A3B-Instruct
  /home/share/dataset/ShareGPT4V
  /usr/local/Ascend/driver
  /usr/local/Ascend/firmware
  /usr/local/bin/npu-smi
  /usr/bin/msnpureport
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path missing on $(hostname): ${path}" >&2
    exit 3
  fi
done

mkdir -p /home/tzq/verl-release-v0.8.0 /home/tzq/npu_profile_outputs

if [[ "${RECREATE}" == "1" ]] && docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm -f "${NAME}" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  if [[ "${ACTION}" == "restart" ]]; then
    docker restart "${NAME}" >/dev/null
  else
    docker start "${NAME}" >/dev/null
  fi
else
  docker run -dit \
    --name "${NAME}" \
    --network host \
    --ipc host \
    --security-opt label=disable \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /home/tzq/verl-release-v0.8.0:/workspace/task3/verl-release-v0.8.0 \
    -v /home/tzq/VeOmni-0.1.11:/workspace/task3/VeOmni-0.1.11 \
    -v /home/share/Qwen3-VL-30B-A3B-Instruct:/workspace/model/Qwen3-VL-30B-A3B-Instruct:ro \
    -v /home/share/dataset/ShareGPT4V:/workspace/dataset/ShareGPT4V:ro \
    -v /home/tzq/npu_profile_outputs:/workspace/output \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/bin/msnpureport:/usr/bin/msnpureport \
    "${IMAGE}" /bin/bash >/dev/null
fi

docker ps --filter "name=${NAME}" --format '{{.Names}} {{.Image}} {{.Status}}'

if [[ "${VERIFY}" == "1" ]]; then
  docker exec "${NAME}" bash -lc '
    set -euo pipefail
    cd /workspace/task3/VeOmni-0.1.11
    python - <<PY
import os
checks = {
    "veomni": "/workspace/task3/VeOmni-0.1.11",
    "model": "/workspace/model/Qwen3-VL-30B-A3B-Instruct",
    "dataset": "/workspace/dataset/ShareGPT4V",
    "output": "/workspace/output",
}
for key, path in checks.items():
    print(f"{key}={os.path.exists(path)} path={path}")
PY
    npu-smi info >/dev/null
  '
fi
REMOTE
done
