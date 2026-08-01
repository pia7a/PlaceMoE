#!/usr/bin/env bash

set -euo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
source_root=${host_root}/src
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722/src
launcher=${source_root}/scripts/profile/launch_hiermoe_greedy_e2e_4node.sh
collector=${source_root}/scripts/profile/collect_hiermoe_paper_run.sh
summarizer=${source_root}/scripts/profile/summarize_hiermoe_paper_case.py
profile_name=paper8h_profile_qwen35_122b_a10b_12l_20260729
route_root=${source_root}/route_captures/${profile_name}
sample_dir=${source_root}/results/qwen35_12l_recursive_classifier_samples_20260729
eplb_layout=eplb_qwen35_122b_a10b_12l_profile4_b4_ep32_layout_20260729.json
eplb_report=eplb_qwen35_122b_a10b_12l_profile4_b4_ep32_report_20260729.json
ours_layout=recursive_classifier_qwen35_122b_a10b_12l_profile4_b4_ep32_layout_20260729.json
ours_report=recursive_classifier_qwen35_122b_a10b_12l_profile4_b4_ep32_report_20260729.json
key=/home/tzq/KeyPair-3bce.pem
rank0_container=tzq_hiermoe_paper8h_rank0_20260729
reuse_profile=${QWEN35_REUSE_PROFILE:-0}
ours_optimize_steps=${QWEN35_OURS_OPTIMIZE_STEPS:-0,1,2}
ours_validation_steps=${QWEN35_OURS_VALIDATION_STEPS:-3}
ours_partition_restarts=${QWEN35_OURS_PARTITION_RESTARTS:-3}
ours_partition_iterations=${QWEN35_OURS_PARTITION_ITERATIONS:-24}
ours_alternations=${QWEN35_OURS_ALTERNATIONS:-3}
ours_lut_iterations=${QWEN35_OURS_LUT_ITERATIONS:-6}
ours_structured_shortlist=${QWEN35_OURS_STRUCTURED_SHORTLIST:-2}

containers=(
  RANK0_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank0_20260729
  RANK1_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank1_20260729
  RANK2_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank2_20260729
  RANK3_CONTAINER_OVERRIDE=tzq_hiermoe_paper8h_rank3_20260729
)
model=(
  MODEL_PATH_OVERRIDE=/workspace/model/Qwen3.5-122B-A10B
  MODEL_CONFIG_PATH_OVERRIDE=${container_root}/configs/model_configs/qwen35_122b_a10b_12l.json
  NUM_MOE_LAYERS_OVERRIDE=12
  TRAIN_FREEZE_VIT_OVERRIDE=true
  RMS_NORM_GATED_IMPLEMENTATION_OVERRIDE=npu
  CAUSAL_CONV1D_IMPLEMENTATION_OVERRIDE=eager
  CHUNK_GATED_DELTA_RULE_IMPLEMENTATION_OVERRIDE=eager
)

if [[ "${reuse_profile}" != "1" ]]; then
  # This four-step route profile is also the bounded compatibility smoke:
  # forward, backward, and Adam.step all execute before layout construction.
  env \
    "${containers[@]}" \
    "${model[@]}" \
    E2E_VARIANT=dedup \
    RUN_NAME_OVERRIDE=${profile_name} \
    MASTER_PORT=30400 \
    HCCL_IF_BASE_PORT=61000 \
    MAX_STEPS_OVERRIDE=4 \
    FULL_PROFILE_START_STEP_OVERRIDE=99 \
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
    rsync -az \
      -e "ssh -i ${key} -o StrictHostKeyChecking=no" \
      "root@${host}:${route_root}/" \
      "${route_root}/"
  done
fi

if [[ ! -s "${source_root}/results/${eplb_layout}" ]]; then
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
    --layers 12 \
    --num-experts 256 \
    --primary-slots-per-rank 8 \
    --redundant-slots-per-rank 4 \
    --output-layout "${container_root}/results/${eplb_layout}" \
    --output-report "${container_root}/results/${eplb_report}"
fi

mkdir -p "${sample_dir}"
build_one_layer() {
  local layer=$1
  docker exec \
    -e "PYTHONPATH=${container_root}" \
    -w "${container_root}" \
    "${rank0_container}" \
    python scripts/profile/plan_placemoe.py \
    --route-root "${container_root}/route_captures/${profile_name}" \
    --optimize-steps "${ours_optimize_steps}" \
    --validation-steps "${ours_validation_steps}" \
    --layer-start "${layer}" \
    --layers 1 \
    --num-experts 256 \
    --slots-per-rank 12 \
    --primary-slots-per-rank 8 \
    --hidden-size 3072 \
    --partition-restarts "${ours_partition_restarts}" \
    --partition-iterations "${ours_partition_iterations}" \
    --alternations "${ours_alternations}" \
    --lut-iterations "${ours_lut_iterations}" \
    --structured-shortlist "${ours_structured_shortlist}" \
    --output-layout "${container_root}/results/qwen35_12l_recursive_classifier_samples_20260729/layer${layer}_layout.json" \
    --output-report "${container_root}/results/qwen35_12l_recursive_classifier_samples_20260729/layer${layer}_report.json" \
    >"${sample_dir}/layer${layer}.log" 2>&1
}
export -f build_one_layer
export container_root profile_name rank0_container sample_dir
export ours_optimize_steps ours_validation_steps ours_partition_restarts ours_partition_iterations
export ours_alternations ours_lut_iterations ours_structured_shortlist
seq 0 11 | xargs -P 12 -n 1 bash -c 'build_one_layer "$1"' _

docker exec \
  -e "PYTHONPATH=${container_root}" \
  -w "${container_root}" \
  "${rank0_container}" \
  python scripts/profile/merge_hiermoe_recursive_classifier_layouts.py \
  --input-dir "${container_root}/results/qwen35_12l_recursive_classifier_samples_20260729" \
  --layers 12 \
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
    "${model[@]}" \
    E2E_VARIANT=hierarchical_full_static \
    RUN_NAME_OVERRIDE="${name}" \
    MASTER_PORT="${master_port}" \
    HCCL_IF_BASE_PORT="${hccl_port}" \
    MAX_STEPS_OVERRIDE=20 \
    FULL_PROFILE_START_STEP_OVERRIDE=11 \
    FULL_PROFILE_EVERY_N_OVERRIDE=1 \
    FULL_PROFILE_RANKS_OVERRIDE=0 \
    HIERMOE_ABLATION_GRAD_MODE_OVERRIDE=blocking \
    HIERMOE_REDUNDANT_SLOTS_OVERRIDE=4 \
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

run_static paper8h_p1_qwen35_122b_a10b_12l_b4_eplb_20260729 "${eplb_layout}" "${eplb_report}" 30401 61100
run_static paper8h_p1_qwen35_122b_a10b_12l_b4_ours_20260729 "${ours_layout}" "${ours_report}" 30402 61200
