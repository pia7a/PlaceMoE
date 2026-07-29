#!/usr/bin/env bash

set -euo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
source_root=${host_root}/src
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722/src
launcher=${source_root}/scripts/profile/launch_hiermoe_greedy_e2e_4node.sh
collector=${source_root}/scripts/profile/collect_hiermoe_paper_run.sh
summarizer=${source_root}/scripts/profile/summarize_hiermoe_paper_case.py
profile_name=paper8h_profile_tulu3_qwen3vl_20260729
route_root=${source_root}/route_captures/${profile_name}
sample_dir=${source_root}/results/tulu3_recursive_classifier_samples_20260729
eplb_layout=eplb_tulu3_profile4_b4_ep32_48layers_layout_20260729.json
eplb_report=eplb_tulu3_profile4_b4_ep32_48layers_report_20260729.json
ours_layout=recursive_classifier_tulu3_profile4_b4_ep32_48layers_layout_20260729.json
ours_report=recursive_classifier_tulu3_profile4_b4_ep32_48layers_report_20260729.json
key=/home/tzq/KeyPair-3bce.pem
rank0_container=tzq_hiermoe_paper8h_rank0_20260729

containers=(
  RANK0_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank0_20260729
  RANK1_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank1_20260729
  RANK2_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank2_20260729
  RANK3_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank3_20260729
)
data=(
  DATA_PATH_OVERRIDE=/workspace/dataset/Tulu3/train-00002-of-00006.parquet
  DATA_SOURCE_NAME_OVERRIDE=tulu-3-sft-mixture
  TRAIN_FREEZE_VIT_OVERRIDE=true
)

env \
  "${containers[@]}" \
  "${data[@]}" \
  E2E_VARIANT=fixed_r2_mirrored_pipeline_grad \
  RUN_NAME_OVERRIDE=${profile_name} \
  MASTER_PORT=30300 \
  HCCL_IF_BASE_PORT=60000 \
  MAX_STEPS_OVERRIDE=4 \
  FULL_PROFILE_START_STEP_OVERRIDE=99 \
  HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=blocking \
  HIERMOE_CAPTURE_ROUTES=1 \
  HIERMOE_CAPTURE_MODE_OVERRIDE=local \
  HIERMOE_CAPTURE_STEP_OVERRIDE=-1 \
  bash "${launcher}"

for node in \
  "192.168.0.190:1" \
  "192.168.0.109:2" \
  "192.168.0.9:3"
do
  host=${node%%:*}
  rank=${node##*:}
  rsync -az \
    -e "ssh -i ${key} -o StrictHostKeyChecking=no" \
    "root@${host}:${route_root}/" \
    "${route_root}/"
done

docker exec "${rank0_container}" mkdir -p /tmp/EPLB
docker cp /home/tzq/EPLB/. "${rank0_container}:/tmp/EPLB/"
docker exec \
  -e "PYTHONPATH=${container_root}" \
  -w "${container_root}" \
  "${rank0_container}" \
  python scripts/profile/build_hiermoe_eplb_layout.py \
  --eplb-root /tmp/EPLB \
  --route-root "${container_root}/route_captures/${profile_name}" \
  --profile-steps 0,1,2,3 \
  --redundant-slots-per-rank 4 \
  --output-layout "${container_root}/results/${eplb_layout}" \
  --output-report "${container_root}/results/${eplb_report}"

mkdir -p "${sample_dir}"
build_one_layer() {
  local layer=$1
  docker exec \
    -e "PYTHONPATH=${container_root}" \
    -w "${container_root}" \
    "${rank0_container}" \
    python scripts/profile/build_hiermoe_recursive_classifier_layout.py \
    --route-root "${container_root}/route_captures/${profile_name}" \
    --optimize-steps 0,1,2 \
    --validation-steps 3 \
    --layer-start "${layer}" \
    --layers 1 \
    --output-layout "${container_root}/results/tulu3_recursive_classifier_samples_20260729/layer${layer}_layout.json" \
    --output-report "${container_root}/results/tulu3_recursive_classifier_samples_20260729/layer${layer}_report.json" \
    >"${sample_dir}/layer${layer}.log" 2>&1
}
export -f build_one_layer
export container_root profile_name rank0_container sample_dir
seq 0 47 | xargs -P 12 -n 1 bash -c 'build_one_layer "$1"' _

docker exec \
  -e "PYTHONPATH=${container_root}" \
  -w "${container_root}" \
  "${rank0_container}" \
  python scripts/profile/merge_hiermoe_recursive_classifier_layouts.py \
  --input-dir "${container_root}/results/tulu3_recursive_classifier_samples_20260729" \
  --layers 48 \
  --output-layout "${container_root}/results/${ours_layout}" \
  --output-report "${container_root}/results/${ours_report}"

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
    "${data[@]}" \
    E2E_VARIANT=hierarchical_full_static \
    RUN_NAME_OVERRIDE="${name}" \
    MASTER_PORT="${master_port}" \
    HCCL_IF_BASE_PORT="${hccl_port}" \
    MAX_STEPS_OVERRIDE=20 \
    FULL_PROFILE_START_STEP_OVERRIDE=11 \
    FULL_PROFILE_EVERY_N_OVERRIDE=1 \
    FULL_PROFILE_RANKS_OVERRIDE=0 \
    HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=blocking \
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

run_static paper8h_p1_tulu3_b4_eplb_20260729 "${eplb_layout}" "${eplb_report}" 30301 60100
run_static paper8h_p1_tulu3_b4_ours_20260729 "${ours_layout}" "${ours_report}" 30302 60200
