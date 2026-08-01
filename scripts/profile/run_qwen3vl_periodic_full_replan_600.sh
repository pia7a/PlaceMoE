#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "${script_dir}/../.." && pwd)
tag=${QWEN3VL_PERIODIC_REPLAN_TAG:-20260801_v1}
interval=${QWEN3VL_PERIODIC_REPLAN_INTERVAL:-100}
if [[ ! "${interval}" =~ ^(50|100)$ ]]; then
  echo "QWEN3VL_PERIODIC_REPLAN_INTERVAL must be 50 or 100" >&2
  exit 2
fi
last_trigger=$((600 - interval))
run_name=paper32_qwen3vl30b_sharegpt4v_ours_periodic_full_replan_convergence600_${tag}_i${interval}
baseline_run=paper32_qwen3vl30b_sharegpt4v_veomni_baseline_convergence600_20260801_v1
static_run=paper32_qwen3vl30b_sharegpt4v_ours_static_hierarchical_dedup_convergence600_20260801_v1
output_dir=${source_root}/results/qwen3vl_periodic_full_replan600_${tag}_i${interval}

export PAPER32_SSH_KEY=${PAPER32_SSH_KEY:-/home/tzq/KeyPair-3bce.pem}
export PAPER32_RANK0_HOST=${PAPER32_RANK0_HOST:-192.168.0.55}
export PAPER32_RANK1_HOST=${PAPER32_RANK1_HOST:-192.168.0.190}
export PAPER32_RANK2_HOST=${PAPER32_RANK2_HOST:-192.168.0.109}
export PAPER32_RANK3_HOST=${PAPER32_RANK3_HOST:-192.168.0.9}
export PAPER32_MASTER_ADDR=${PAPER32_MASTER_ADDR:-192.168.0.55}
export PAPER32_CLUSTER_SLUG=huawei2
export PAPER32_MAX_STEPS_OVERRIDE=600
export PAPER32_TOTAL_MAX_STEPS=600
export PAPER32_NUM_TRAIN_EPOCHS=4
export PAPER32_STATS_START_STEP_OVERRIDE=501
export PAPER32_STATS_END_STEP_OVERRIDE=600
export PAPER32_LIGHTWEIGHT_TIMING=1
export PAPER32_CONVERGENCE_METRICS=1
export PAPER32_MOE_MONITOR_INTERVAL=0
export PAPER32_MOE_MONITOR_JSONL_ENABLE=0
export PAPER32_MOE_TIMING_ENABLE=0
export PAPER32_ENV_METRICS_JSONL_ENABLE=1
export PAPER32_HIERMOE_LOG_INTERVAL=1
export PAPER32_SKIP_PAPER_SUMMARY=1
export PAPER32_SKIP_COMPLETED=0
export PAPER32_PERIODIC_FULL_REPLAN_LAST_STEP=${last_trigger}
export HIERMOE_SWAP_INTERVAL_OVERRIDE=${interval}

bash "${script_dir}/sync_hiermoe_paper32_source.sh"

PAPER32_RUN_NAME=${run_name} \
PAPER32_MASTER_PORT=${PAPER32_MASTER_PORT:-31630} \
PAPER32_HCCL_PORT=${PAPER32_HCCL_PORT:-59300} \
  bash "${script_dir}/run_hiermoe_paper32_case.sh" qwen3vl sharegpt4v ours_full_replan full

python "${script_dir}/plot_qwen3vl_convergence.py" \
  --baseline "${source_root}/profile/runs/pretrain/${baseline_run}/convergence_metrics/convergence_metrics_rank0.jsonl" \
  --ours "${source_root}/profile/runs/pretrain/${static_run}/convergence_metrics/convergence_metrics_rank0.jsonl" \
  --dynamic "${source_root}/profile/runs/pretrain/${run_name}/convergence_metrics/convergence_metrics_rank0.jsonl" \
  --dynamic-events "${source_root}/profile/runs/pretrain/${run_name}/periodic_full_replan/events.jsonl" \
  --output-dir "${output_dir}" \
  --expected-steps 600 \
  --layer 23 \
  --window 20 \
  --last-n 100

echo "Qwen3-VL periodic full-replan experiment completed: ${output_dir}"
