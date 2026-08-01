#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "${script_dir}/../.." && pwd)
tag=${QWEN3VL_CONVERGENCE_TAG:-20260801_v1}
baseline_run=paper32_qwen3vl30b_sharegpt4v_veomni_baseline_convergence600_${tag}
ours_run=paper32_qwen3vl30b_sharegpt4v_ours_static_hierarchical_dedup_convergence600_${tag}
output_dir=${source_root}/results/qwen3vl_convergence600_${tag}

# The current accepted Qwen3-VL main-method layout and performance model carry
# the huawei2 artifact label. The four execution hosts remain huawei1_node1-4.
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
export PAPER32_ENV_METRICS_JSONL_ENABLE=0
export PAPER32_HIERMOE_LOG_INTERVAL=600
export PAPER32_SKIP_PAPER_SUMMARY=1
export PAPER32_SKIP_COMPLETED=0

bash "${script_dir}/sync_hiermoe_paper32_source.sh"

PAPER32_RUN_NAME=${baseline_run} \
PAPER32_MASTER_PORT=31500 \
PAPER32_HCCL_PORT=59000 \
  bash "${script_dir}/run_hiermoe_paper32_case.sh" qwen3vl sharegpt4v baseline full

PAPER32_RUN_NAME=${ours_run} \
PAPER32_MASTER_PORT=31501 \
PAPER32_HCCL_PORT=59100 \
  bash "${script_dir}/run_hiermoe_paper32_case.sh" qwen3vl sharegpt4v ours full

python "${script_dir}/plot_qwen3vl_convergence.py" \
  --baseline "${source_root}/profile/runs/pretrain/${baseline_run}/convergence_metrics/convergence_metrics_rank0.jsonl" \
  --ours "${source_root}/profile/runs/pretrain/${ours_run}/convergence_metrics/convergence_metrics_rank0.jsonl" \
  --output-dir "${output_dir}" \
  --expected-steps 600 \
  --layer 23 \
  --window 20 \
  --last-n 100

echo "Qwen3-VL convergence experiment completed: ${output_dir}"
