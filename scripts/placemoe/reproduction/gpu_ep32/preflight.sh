#!/usr/bin/env bash

# Validate software, model, data, networking, and eight A6000s on every node.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
run_tag=${PLACEMOE_REPRO_PREFLIGHT_TAG:-$(date +%Y%m%d_%H%M%S)}
run_name=gpu32_preflight_${run_tag}
report=${PLACEMOE_REPRO_PREFLIGHT_REPORT:-${repo_root}/results/gpu_adaptation/${run_name}.json}

if [[ "${PLACEMOE_REPRO_SYNC_SOURCE:-1}" == "1" ]]; then
  bash "${script_dir}/sync_source.sh"
fi

env \
  E2E_VARIANT=baseline \
  RUN_NAME_OVERRIDE="${run_name}" \
  MODEL_PATH_OVERRIDE="${PLACEMOE_REPRO_MODEL_PATH:-${PLACEMOE_REPRO_QWEN3VL_MODEL_PATH:-}}" \
  DATA_PATH_OVERRIDE="${PLACEMOE_REPRO_PREFLIGHT_DATA_PATH:-${PLACEMOE_REPRO_SHAREGPT4V_DATA_PATH:-}}" \
  GPU_PREFLIGHT_ONLY=1 \
  FULL_PROFILE_ENABLE_OVERRIDE=0 \
  TORCH_PROFILE_ENABLE_OVERRIDE=0 \
  bash "${script_dir}/launch.sh"

mkdir -p "$(dirname "${report}")"
"${repro_python}" - "${repo_root}" "${run_name}" "${report}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
report = Path(sys.argv[3])
nodes = []
for rank in range(4):
    log_path = root / "pretrain_runs" / f"{run_name}_rank{rank}.host.log"
    accepted = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "accepted":
            accepted = payload
    if accepted is None:
        raise SystemExit(f"no accepted preflight payload in {log_path}")
    accepted["node_rank"] = rank
    nodes.append(accepted)

software_scopes = {
    (node["accelerator"], node["torch"], node["cuda"], tuple(node["nccl"]), node["triton"])
    for node in nodes
}
if len(software_scopes) != 1:
    raise SystemExit(f"EP32 software scope differs across nodes: {software_scopes!r}")
payload = {
    "schema_version": 1,
    "status": "accepted",
    "world_size": 32,
    "ep_size": 32,
    "ranks_per_node": 8,
    "nodes": nodes,
}
report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(report)
PY
