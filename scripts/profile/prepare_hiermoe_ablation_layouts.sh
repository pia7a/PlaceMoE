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
paper32_load_compute_calibration

profile_name=$(paper32_profile_name)
route_root=${paper32_source_root}/route_captures/${profile_name}
if [[ ! -d "${route_root}" ]] || ! find "${route_root}" -type f -print -quit | grep -q .; then
  echo "missing route profile '${route_root}'" >&2
  echo "run prepare_hiermoe_paper32_layouts.sh ${model} ${dataset} first" >&2
  exit 1
fi
if ((paper32_primary_slots % 4 != 0)); then
  echo "primary slots per rank must divide by four for the requested rho sweep" >&2
  exit 2
fi

rank0_exec=(
  ssh -i "${paper32_ssh_key}" -o StrictHostKeyChecking=no "root@${paper32_hosts[0]}"
  docker exec
  -e "PYTHONPATH=${paper32_container_source_root}"
  -w "${paper32_container_source_root}"
  "${paper32_container_name}"
)

build_layout() {
  local ablation=$1
  local redundant_slots=$2
  local lut_iterations=$3
  local objective=$4
  local stem
  local slots_per_rank=$((paper32_primary_slots + redundant_slots))
  local -a objective_args=()

  stem=$(paper32_ablation_layout_stem "${ablation}" "${redundant_slots}")
  if [[ -s "${paper32_source_root}/results/${stem}_layout.json" \
    && -s "${paper32_source_root}/results/${stem}_report.json" \
    && "${PAPER32_REBUILD_ABLATION_LAYOUTS:-0}" != "1" ]]
  then
    echo "skipping existing ${stem}"
    return 0
  fi
  if [[ "${objective}" == "communication" ]]; then
    objective_args=(
      --route-ms-per-assignment 0
      --compute-ms-per-assignment 0
    )
  elif [[ "${objective}" == "compute" ]]; then
    # Isolate the max non-deduplicated assignment objective.  Disable both
    # hierarchy links and the route/local-processing proxy so no communication
    # feature can affect candidate generation or exact candidate selection.
    objective_args=(
      --inter-ms-per-byte 0
      --intra-ms-per-byte 0
      --route-ms-per-assignment 0
      --communication-blind-proposals
    )
  elif [[ "${objective}" != "joint" ]]; then
    echo "unsupported objective '${objective}'" >&2
    return 2
  fi

  echo "building ${stem}"
  "${rank0_exec[@]}" python scripts/profile/plan_placemoe.py \
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
    --redundant-slots-per-rank "${redundant_slots}" \
    --slots-per-rank "${slots_per_rank}" \
    --hidden-size "${paper32_hidden_size}" \
    --inter-ms-per-byte "${paper32_inter_ms_per_byte}" \
    --intra-ms-per-byte "${paper32_intra_ms_per_byte}" \
    --communication-phase-multiplier "${paper32_communication_phase_multiplier}" \
    --compute-ms-per-assignment "${paper32_compute_ms_per_assignment}" \
    --compute-phase-multiplier "${paper32_compute_phase_multiplier}" \
    --lut-iterations "${lut_iterations}" \
    "${objective_args[@]}" \
    --output-layout "${paper32_container_source_root}/results/${stem}_layout.json" \
    --output-report "${paper32_container_source_root}/results/${stem}_report.json"

  for suffix in layout report; do
    rsync -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "root@${paper32_hosts[0]}:${paper32_source_root}/results/${stem}_${suffix}.json" \
      "${paper32_source_root}/results/"
  done
  for host in "${paper32_hosts[@]}"; do
    rsync -az \
      -e "ssh -i ${paper32_ssh_key} -o StrictHostKeyChecking=no" \
      "${paper32_source_root}/results/${stem}_layout.json" \
      "${paper32_source_root}/results/${stem}_report.json" \
      "root@${host}:${paper32_source_root}/results/"
  done
}

quarter=$((paper32_primary_slots / 4))
requested_layouts=${PAPER32_ABLATION_LAYOUT_CASES:-hyper_rho000 hyper_rho025 hyper_rho050 hyper_rho075 comm_initial_lut compute_initial_lut joint_initial_lut}
read -r -a layout_cases <<< "${requested_layouts}"
for layout_case in "${layout_cases[@]}"; do
  case "${layout_case}" in
    hyper_rho000)
      build_layout hyper_rho000 0 "${PAPER32_LAYOUT_LUT_ITERATIONS:-6}" joint
      ;;
    hyper_rho025)
      build_layout hyper_rho025 "${quarter}" "${PAPER32_LAYOUT_LUT_ITERATIONS:-6}" joint
      ;;
    hyper_rho050)
      build_layout hyper_rho050 "$((2 * quarter))" "${PAPER32_LAYOUT_LUT_ITERATIONS:-6}" joint
      ;;
    hyper_rho075)
      build_layout hyper_rho075 "$((3 * quarter))" "${PAPER32_LAYOUT_LUT_ITERATIONS:-6}" joint
      ;;
    # rho=1 with the full joint objective and optimized LUT is the canonical
    # Ours layout already produced by prepare_hiermoe_paper32_layouts.sh.
    comm_initial_lut)
      build_layout comm_initial_lut "${paper32_primary_slots}" 0 communication
      ;;
    compute_initial_lut)
      build_layout compute_initial_lut "${paper32_primary_slots}" 0 compute
      ;;
    joint_initial_lut)
      build_layout joint_initial_lut "${paper32_primary_slots}" 0 joint
      ;;
    *)
      echo "unknown ablation layout case '${layout_case}'" >&2
      exit 2
      ;;
  esac
done

echo "ablation layouts are ready for ${model}/${dataset}"
