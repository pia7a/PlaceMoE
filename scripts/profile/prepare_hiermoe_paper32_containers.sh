#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"

create_or_validate() {
  local host=$1
  local node_index=$2
  local existing_image
  ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
    "mkdir -p '$(dirname "${paper32_perf_model_host}")'"
  rsync -az \
    -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
    "${paper32_perf_model_host}" \
    "root@${host}:${paper32_perf_model_host}"
  existing_image=$(ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
    "docker inspect -f '{{.Config.Image}}' '${paper32_container_name}' 2>/dev/null || true")
  if [[ -n "${existing_image}" ]]; then
    if [[ "${existing_image}" != "${paper32_image}" ]]; then
      echo "${host}: existing ${paper32_container_name} uses ${existing_image}, expected ${paper32_image}" >&2
      return 1
    fi
    ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
      "docker start '${paper32_container_name}' >/dev/null || true"
  else
    ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
      "docker run -d \
        --name '${paper32_container_name}' \
        --network=host \
        --privileged \
        --shm-size=128g \
        --add-host=\"\$(hostname):127.0.0.1\" \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/bin/msnpureport:/usr/bin/msnpureport \
        -v /home/share:/workspace/model:ro \
        -v /home/share/dataset:/workspace/dataset:ro \
        -v /home/tzq/EPLB:/workspace/EPLB:ro \
        -v /home/tzq/npu_profile_outputs:/workspace/output \
        -w '${paper32_container_source_root}' \
        '${paper32_image}' sleep infinity >/dev/null"
  fi

  validation_command='python -c "import torch, transformers, veomni; print(torch.__version__, transformers.__version__, veomni.__file__)";'
  validation_command+=' test -d /workspace/model/Qwen3-VL-30B-A3B-Instruct;'
  validation_command+=' test -d /workspace/model/Qwen3.5-35B-A3B-20L;'
  validation_command+=' test -d /workspace/model/DeepSeek-V3-6MoE-Half;'
  validation_command+=' test -e /workspace/dataset/Tulu3/train-00002-of-00006.parquet;'
  validation_command+=' test -d /workspace/dataset/ShareGPT4V;'
  validation_command+=" test -f ${paper32_perf_model_container};"
  if [[ "${node_index}" == "0" ]]; then
    validation_command+=' test -f /workspace/EPLB/eplb.py;'
  fi
  ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${host}" \
    "docker exec '${paper32_container_name}' bash -lc '${validation_command}'"
}

for node_index in "${!paper32_hosts[@]}"; do
  host=${paper32_hosts[node_index]}
  echo "preparing ${host}:${paper32_container_name}"
  create_or_validate "${host}" "${node_index}"
done

echo "paper32 containers are ready"
