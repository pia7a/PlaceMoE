#!/usr/bin/env bash
# Reproducible launcher for the Ascend NPU profile containers.
#
# Run this script on the NPU jump/login node, where /home/tzq/KeyPair-3bce.pem
# can SSH to the internal NPU hosts as root.
#
# Common usage:
#   cd /home/tzq/VeOmni-0.1.11
#   bash scripts/profile/start_npu_profile_containers.sh
#
# Options:
#   NODESET=new4     Start the current four-node group: .174/.213/.136/.210.
#   NODESET=original4 Start the original four-node group: .63/.164/.45/.93.
#   NODESET=all8     Start all eight known nodes.
#   ACTION=restart   Restart existing containers instead of only starting them.
#   RECREATE=1       Remove and recreate containers.
#   VERIFY=0         Skip post-start mount/NPU checks.
#
# Examples:
#   NODESET=new4 ACTION=start bash scripts/profile/start_npu_profile_containers.sh
#   NODESET=original4 ACTION=restart bash scripts/profile/start_npu_profile_containers.sh
#   NODESET=all8 RECREATE=1 bash scripts/profile/start_npu_profile_containers.sh

set -euo pipefail

KEY=${KEY:-/home/tzq/KeyPair-3bce.pem}
IMAGE=${IMAGE:-4923f32e1e9b}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-tzq_npu_profile_rank}
NODESET=${NODESET:-new4}
ACTION=${ACTION:-start}
RECREATE=${RECREATE:-0}
VERIFY=${VERIFY:-1}
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

case "${NODESET}" in
  new4)
    ranks=(4 5 6 7)
    ;;
  original4)
    ranks=(0 1 2 3)
    ;;
  all8)
    ranks=(0 1 2 3 4 5 6 7)
    ;;
  *)
    echo "NODESET must be new4, original4, or all8; got: ${NODESET}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${KEY}" ]]; then
  echo "SSH key not found: ${KEY}" >&2
  exit 2
fi

if [[ "${ACTION}" != "start" && "${ACTION}" != "restart" ]]; then
  echo "ACTION must be start or restart; got: ${ACTION}" >&2
  exit 2
fi

ssh_base=(ssh -i "${KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}")

for rank in "${ranks[@]}"; do
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
expected_image_id=$(docker image inspect --format '{{.Id}}' "${IMAGE}")

validate_existing_container() {
  local actual_image_id mounts devices

  actual_image_id=$(docker inspect --format '{{.Image}}' "${NAME}")
  if [[ "${actual_image_id}" != "${expected_image_id}" ]]; then
    echo "Container ${NAME} uses image ${actual_image_id}, expected ${expected_image_id}; rerun with RECREATE=1." >&2
    exit 4
  fi

  mounts=$(docker inspect --format '{{range .Mounts}}{{.Destination}}={{.Source}};{{end}}' "${NAME}")
  required_mounts=(
    /workspace/task3/verl-release-v0.8.0
    /workspace/task3/VeOmni-0.1.11
    /workspace/model/Qwen3-VL-30B-A3B-Instruct
    /workspace/dataset/ShareGPT4V
    /workspace/output
    /usr/local/Ascend/driver
    /usr/local/Ascend/firmware
    /usr/local/sbin
    /usr/local/bin/npu-smi
    /usr/bin/msnpureport
  )
  for mount in "${required_mounts[@]}"; do
    if [[ "${mounts}" != *"${mount}="* ]]; then
      echo "Container ${NAME} is missing mount ${mount}; rerun with RECREATE=1." >&2
      exit 4
    fi
  done

  devices=$(docker inspect --format '{{range .HostConfig.Devices}}{{.PathOnHost}};{{end}}' "${NAME}")
  required_devices=(
    /dev/davinci0
    /dev/davinci1
    /dev/davinci2
    /dev/davinci3
    /dev/davinci4
    /dev/davinci5
    /dev/davinci6
    /dev/davinci7
    /dev/davinci_manager
    /dev/devmm_svm
    /dev/hisi_hdc
  )
  for device in "${required_devices[@]}"; do
    if [[ "${devices}" != *"${device};"* ]]; then
      echo "Container ${NAME} is missing device ${device}; rerun with RECREATE=1." >&2
      exit 4
    fi
  done
}

if [[ "${RECREATE}" == "1" ]] && docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm -f "${NAME}" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  validate_existing_container
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
failures = []
for key, path in checks.items():
    exists = os.path.exists(path)
    print(f"{key}={exists} path={path}")
    if not exists:
        failures.append(f"{key} missing: {path}")
output_probe = os.path.join(checks["output"], ".veomni_write_probe")
try:
    with open(output_probe, "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    os.remove(output_probe)
except OSError as exc:
    failures.append(f"output not writable: {exc}")
if failures:
    raise SystemExit("Container verification failed: " + "; ".join(failures))
PY
    npu-smi info >/dev/null
  '
fi
REMOTE
done
