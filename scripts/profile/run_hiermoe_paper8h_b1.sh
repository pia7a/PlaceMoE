#!/usr/bin/env bash

set -euo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
source_root=${host_root}/src
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722/src
launcher=${source_root}/scripts/profile/launch_hiermoe_greedy_e2e_4node.sh
collector=${source_root}/scripts/profile/collect_hiermoe_paper_run.sh
summarizer=${source_root}/scripts/profile/summarize_hiermoe_paper_case.py
route_name=qwen3vl_greedy_ep32_mb4_gbs128_dedup_8step_route_history_8step_20260724
sample_dir=${source_root}/results/sharegpt4v_b1_recursive_classifier_samples_20260729
eplb_layout=eplb_sharegpt4v_profile4_b1_ep32_48layers_layout_20260729.json
eplb_report=eplb_sharegpt4v_profile4_b1_ep32_48layers_report_20260729.json
ours_layout=recursive_classifier_sharegpt4v_profile4_b1_ep32_48layers_layout_20260729.json
ours_report=recursive_classifier_sharegpt4v_profile4_b1_ep32_48layers_report_20260729.json
key=/home/tzq/KeyPair-3bce.pem
rank0_container=tzq_hiermoe_paper8h_rank0_20260729

containers=(
  RANK0_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank0_20260729
  RANK1_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank1_20260729
  RANK2_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank2_20260729
  RANK3_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank3_20260729
)

if [[ ! -f "${source_root}/results/${ours_layout}" ]]; then
  mkdir -p "${sample_dir}"
  build_one_layer() {
    local layer=$1
    docker exec \
      -e "PYTHONPATH=${container_root}" \
      -w "${container_root}" \
      "${rank0_container}" \
      python scripts/profile/plan_placemoe.py \
      --route-root "${container_root}/route_captures/${route_name}" \
      --optimize-steps 0,1,2 \
      --validation-steps 3 \
      --layer-start "${layer}" \
      --layers 1 \
      --slots-per-rank 5 \
      --primary-slots-per-rank 4 \
      --output-layout "${container_root}/results/sharegpt4v_b1_recursive_classifier_samples_20260729/layer${layer}_layout.json" \
      --output-report "${container_root}/results/sharegpt4v_b1_recursive_classifier_samples_20260729/layer${layer}_report.json" \
      >"${sample_dir}/layer${layer}.log" 2>&1
  }
  export -f build_one_layer
  export container_root route_name rank0_container sample_dir
  seq 0 47 | xargs -P 24 -n 1 bash -c 'build_one_layer "$1"' _

  docker exec \
    -e "PYTHONPATH=${container_root}" \
    -w "${container_root}" \
    "${rank0_container}" \
    python scripts/profile/merge_hiermoe_recursive_classifier_layouts.py \
    --input-dir "${container_root}/results/sharegpt4v_b1_recursive_classifier_samples_20260729" \
    --layers 48 \
    --output-layout "${container_root}/results/${ours_layout}" \
    --output-report "${container_root}/results/${ours_report}"
fi

for host in 192.168.0.190 192.168.0.109 192.168.0.9; do
  rsync -az \
    -e "ssh -i ${key} -o StrictHostKeyChecking=no" \
    "${source_root}/results/${eplb_layout}" \
    "${source_root}/results/${ours_layout}" \
    "root@${host}:${source_root}/results/"
done

run_static() {
  local name=$1
  local replay=$2
  local report=$3
  local master_port=$4
  local hccl_port=$5
  env \
    "${containers[@]}" \
    E2E_VARIANT=hierarchical_full_static \
    RUN_NAME_OVERRIDE="${name}" \
    MASTER_PORT="${master_port}" \
    HCCL_IF_BASE_PORT="${hccl_port}" \
    MAX_STEPS_OVERRIDE=20 \
    FULL_PROFILE_START_STEP_OVERRIDE=11 \
    FULL_PROFILE_EVERY_N_OVERRIDE=1 \
    FULL_PROFILE_RANKS_OVERRIDE=0 \
    HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=blocking \
    HIERMOE_REDUNDANT_SLOTS_OVERRIDE=1 \
    HIERMOE_GREEDY_MAX_COPIES_OVERRIDE=8 \
    HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE="${container_root}/results/${replay}" \
    bash "${launcher}"
  bash "${collector}" "${name}"
  python "${summarizer}" \
    --run-name "${name}" \
    --start-step 11 \
    --end-step 20 \
    --layout-report "${source_root}/results/${report}" \
    --output "${source_root}/results/${name}_summary.json"
}

run_static paper8h_p2_sharegpt4v_b1_eplb_20260729 "${eplb_layout}" "${eplb_report}" 30500 62000
run_static paper8h_p2_sharegpt4v_b1_ours_20260729 "${ours_layout}" "${ours_report}" 30501 62100
