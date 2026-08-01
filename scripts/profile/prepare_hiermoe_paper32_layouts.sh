#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL DATASET" >&2
  exit 2
fi

model=$1
dataset=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hiermoe_paper32_common.sh
source "${script_dir}/hiermoe_paper32_common.sh"
paper32_configure_model "${model}"
paper32_configure_dataset "${dataset}"

paper32_rsync_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if rsync "$@"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

profile_name=$(paper32_profile_name)
route_root=${paper32_source_root}/route_captures/${profile_name}
eplb_stem=$(paper32_layout_stem eplb)
ours_stem=$(paper32_layout_stem ours)
reuse_profile=${PAPER32_REUSE_PROFILE:-0}
reuse_layouts=${PAPER32_REUSE_LAYOUTS:-1}
prepare_eplb=${PAPER32_PREPARE_EPLB:-1}
launcher=${script_dir}/launch_hiermoe_greedy_e2e_4node.sh

reuse_ready=0
if [[ "${reuse_layouts}" == "1" \
  && -s "${paper32_source_root}/results/${ours_stem}_layout.json" \
  && -s "${paper32_source_root}/results/${ours_stem}_report.json" ]]; then
  if [[ "${prepare_eplb}" != "1" \
    || ( -s "${paper32_source_root}/results/${eplb_stem}_layout.json" \
      && -s "${paper32_source_root}/results/${eplb_stem}_report.json" ) ]]; then
    reuse_ready=1
  fi
fi

if [[ "${reuse_ready}" == "1" ]]; then
  echo "reusing complete EPLB/Ours layouts for ${model}/${dataset}"
  reuse_files=(
    "${paper32_source_root}/results/${ours_stem}_layout.json"
    "${paper32_source_root}/results/${ours_stem}_report.json"
  )
  if [[ "${prepare_eplb}" == "1" ]]; then
    reuse_files+=(
      "${paper32_source_root}/results/${eplb_stem}_layout.json"
      "${paper32_source_root}/results/${eplb_stem}_report.json"
    )
  fi
  for host in "${paper32_hosts[@]}"; do
    paper32_rsync_retry -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "${reuse_files[@]}" \
      "root@${host}:${paper32_source_root}/results/"
  done
  exit 0
fi

if [[ "${reuse_profile}" != "1" ]]; then
  profile_env=(
    "E2E_VARIANT=${PAPER32_PROFILE_VARIANT:-dedup}"
    "RUN_NAME_OVERRIDE=${profile_name}"
    "MASTER_ADDR_OVERRIDE=${paper32_master_addr}"
    "MASTER_PORT=${PAPER32_PROFILE_MASTER_PORT:-30600}"
    "HCCL_IF_BASE_PORT=${PAPER32_PROFILE_HCCL_PORT:-63000}"
    "NNODES_OVERRIDE=${paper32_nnodes}"
    "NPROC_PER_NODE_OVERRIDE=${paper32_nproc_per_node}"
    "DP_SHARD_SIZE_OVERRIDE=${paper32_world_size}"
    "EP_SIZE_OVERRIDE=${paper32_world_size}"
    "SSH_KEY_OVERRIDE=${paper32_ssh_key}"
    "MODEL_PATH_OVERRIDE=${paper32_model_path}"
    "MODEL_CONFIG_PATH_OVERRIDE=${paper32_model_path}"
    "TRAIN_ENTRYPOINT_OVERRIDE=${paper32_train_entrypoint}"
    "TRAIN_CONFIG_OVERRIDE=${paper32_train_config}"
    "NUM_MOE_LAYERS_OVERRIDE=${paper32_num_layers}"
    "DATA_PATH_OVERRIDE=${paper32_data_path}"
    "DATA_SOURCE_NAME_OVERRIDE=${paper32_data_source_name}"
    "TRAIN_FREEZE_VIT_OVERRIDE=${paper32_freeze_vit}"
    "MICRO_BATCH_SIZE_OVERRIDE=${paper32_micro_batch_size}"
    "GLOBAL_BATCH_SIZE_OVERRIDE=${paper32_global_batch_size}"
    "MAX_SEQ_LEN_OVERRIDE=${PAPER32_MAX_SEQ_LEN:-4096}"
    "MAX_STEPS_OVERRIDE=4"
    "TRAIN_LR_OVERRIDE=${PAPER32_LR:-}"
    "FULL_PROFILE_ENABLE_OVERRIDE=0"
    "FULL_PROFILE_START_STEP_OVERRIDE=99"
    "HIERMOE_CAPTURE_ROUTES=1"
    "HIERMOE_CAPTURE_MODE_OVERRIDE=local"
    "HIERMOE_CAPTURE_STEP_OVERRIDE=-1"
    "HIERMOE_CAPTURE_CALL_OVERRIDE=${PAPER32_CAPTURE_CALL:-0}"
    "HIERMOE_PERF_MODEL_PATH_OVERRIDE=${paper32_perf_model_container}"
  )
  for ((node_rank = 0; node_rank < paper32_nnodes; ++node_rank)); do
    if ((node_rank == 0)) && [[ "${PAPER32_RANK0_LOCAL:-0}" == "1" ]]; then
      profile_env+=("RANK0_CONTAINER_OVERRIDE=${paper32_container_name}")
    else
      profile_env+=(
        "RANK${node_rank}_HOST_OVERRIDE=${paper32_hosts[${node_rank}]}"
        "RANK${node_rank}_CONTAINER_OVERRIDE=${paper32_container_name}"
      )
    fi
  done
  echo "capturing ${profile_name}"
  env "${profile_env[@]}" bash "${launcher}"
fi

mkdir -p "${route_root}"
for ((rank = 0; rank < paper32_nnodes; ++rank)); do
  host=${paper32_hosts[rank]}
  paper32_rsync_retry -az \
    --exclude='.*.pt.*' \
    -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
    "root@${host}:${route_root}/" \
    "${route_root}/"
done

# The builders execute in the rank-0 warm-cache container.  Make its route
# directory complete before starting CPU layout construction.
paper32_rsync_retry -az \
  --exclude='.*.pt.*' \
  -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
  "${route_root}/" \
  "root@${paper32_hosts[0]}:${route_root}/"

if [[ "${PAPER32_CAPTURE_ONLY:-0}" == "1" ]]; then
  echo "route capture ready for ${model}/${dataset}: ${route_root}"
  exit 0
fi

paper32_load_compute_calibration

rank0_exec=(
  ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no
  -o ConnectionAttempts=5 -o ServerAliveInterval=30 -o ServerAliveCountMax=10
  "root@${paper32_hosts[0]}"
  docker exec
  -e "PYTHONPATH=${paper32_container_source_root}"
  -w "${paper32_container_source_root}"
  "${paper32_container_name}"
)

prepared_stems=()
if [[ "${prepare_eplb}" == "1" ]]; then
  echo "building EPLB layout ${eplb_stem}"
  "${rank0_exec[@]}" python scripts/profile/build_hiermoe_eplb_layout.py \
    --eplb-root /workspace/EPLB \
    --route-root "${paper32_container_source_root}/route_captures/${profile_name}" \
    --profile-steps 0,1,2,3 \
    --layer-name-template "${paper32_layer_name_template}" \
    --layers "${paper32_num_layers}" \
    --call-indices "${PAPER32_FORWARD_CALL_INDICES:-0}" \
    --forward-repeats "${PAPER32_FORWARD_REPEATS:-1}" \
    --capture-layer-stride "${paper32_num_layers}" \
    --ep-size "${paper32_world_size}" \
    --ranks-per-node 8 \
    --num-experts "${paper32_num_experts}" \
    --primary-slots-per-rank "${paper32_primary_slots}" \
    --redundant-slots-per-rank "${paper32_redundant_slots}" \
    --output-layout "${paper32_container_source_root}/results/${eplb_stem}_layout.json" \
    --output-report "${paper32_container_source_root}/results/${eplb_stem}_report.json"
  prepared_stems+=("${eplb_stem}")
fi

echo "building Ours layout ${ours_stem}"
"${rank0_exec[@]}" python scripts/profile/build_hiermoe_recursive_classifier_layout.py \
  --route-root "${paper32_container_source_root}/route_captures/${profile_name}" \
  --optimize-steps 0,1,2 \
  --validation-steps 3 \
  --layer-name-template "${paper32_layer_name_template}" \
  --layers "${paper32_num_layers}" \
  --expected-total-layers "${paper32_num_layers}" \
  --call-indices "${PAPER32_FORWARD_CALL_INDICES:-0}" \
  --forward-repeats "${PAPER32_FORWARD_REPEATS:-1}" \
  --workers "${PAPER32_LAYOUT_WORKERS:-12}" \
  --candidate-workers "${PAPER32_LAYOUT_CANDIDATE_WORKERS:-1}" \
  --worker-threads 1 \
  --ep-size "${paper32_world_size}" \
  --ranks-per-node 8 \
  --num-experts "${paper32_num_experts}" \
  --primary-slots-per-rank "${paper32_primary_slots}" \
  --redundant-slots-per-rank "${paper32_redundant_slots}" \
  --slots-per-rank "${paper32_slots_per_rank}" \
  --hidden-size "${paper32_hidden_size}" \
  --inter-ms-per-byte "${paper32_inter_ms_per_byte}" \
  --intra-ms-per-byte "${paper32_intra_ms_per_byte}" \
  --communication-phase-multiplier "${paper32_communication_phase_multiplier}" \
  --compute-ms-per-assignment "${paper32_compute_ms_per_assignment}" \
  --compute-phase-multiplier "${paper32_compute_phase_multiplier}" \
  --comparison-layout mirrored-r2 \
  --output-layout "${paper32_container_source_root}/results/${ours_stem}_layout.json" \
  --output-report "${paper32_container_source_root}/results/${ours_stem}_report.json"
prepared_stems+=("${ours_stem}")

for stem in "${prepared_stems[@]}"; do
  for suffix in layout report; do
    paper32_rsync_retry -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "root@${paper32_hosts[0]}:${paper32_source_root}/results/${stem}_${suffix}.json" \
      "${paper32_source_root}/results/"
  done
done

for host in "${paper32_hosts[@]}"; do
  for stem in "${prepared_stems[@]}"; do
    paper32_rsync_retry -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "${paper32_source_root}/results/${stem}_layout.json" \
      "${paper32_source_root}/results/${stem}_report.json" \
      "root@${host}:${paper32_source_root}/results/"
  done
done

echo "layouts ready for ${model}/${dataset}"
