#!/usr/bin/env bash

set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
benchmark=${source_root}/scripts/profile/benchmark_hiermoe_forward_lut_cover_owner.py
route_root=${source_root}/route_captures/qwen3vl_greedy_ep32_mb4_gbs128_dedup_8step_route_history_8step_20260724
input_layout=${source_root}/results/recursive_classifier_refined_v2_ep32_48layers_layout_20260728.json
anchor=${source_root}/results/cover_oracle_layer26_round1_ep32_20260728.json
key=/home/tzq/KeyPair-3bce.pem
master_addr=192.168.0.55
master_port=${MASTER_PORT:-29865}
run_name=${RUN_NAME:-forward_lut_cover_owner_ep32_48layers}
layer_start=${LAYER_START:-0}
layers=${LAYERS:-48}
optimize_steps=${OPTIMIZE_STEPS:-2,3,4,5}
validation_steps=${VALIDATION_STEPS:-6,7}

benchmark_args=(
  --input-layout="${input_layout}"
  --route-root="${route_root}"
  --optimize-steps="${optimize_steps}"
  --validation-steps="${validation_steps}"
  --layer-start="${layer_start}"
  --layers="${layers}"
  --ep-size=32
  --ranks-per-node=8
  --service-group-size=8
  --num-experts=128
  --slots-per-rank=8
  --max-copies=4
  --anchor="${anchor}"
)

launch_remote() {
  local host=$1
  local node_rank=$2
  local container=$3
  ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    docker exec \
    -e "PYTHONPATH=${source_root}" \
    -w "${source_root}" \
    "${container}" \
    torchrun --nnodes=4 --nproc-per-node=8 --node-rank="${node_rank}" \
    --master-addr="${master_addr}" --master-port="${master_port}" \
    "${benchmark}" "${benchmark_args[@]}" \
    >"${host_root}/results/${run_name}_node${node_rank}.log" 2>&1
}

mkdir -p "${host_root}/results"
launch_remote 192.168.0.190 1 tzq_npu_static_r2_rank1_20260720 &
rank1_pid=$!
launch_remote 192.168.0.109 2 tzq_npu_static_r2_rank2_20260719 &
rank2_pid=$!
launch_remote 192.168.0.9 3 tzq_npu_static_r2_rank3_20260719 &
rank3_pid=$!

docker exec \
  -e "PYTHONPATH=${source_root}" \
  -w "${source_root}" \
  tzq_npu_coremoe_verify_20260717 \
  torchrun --nnodes=4 --nproc-per-node=8 --node-rank=0 \
  --master-addr="${master_addr}" --master-port="${master_port}" \
  "${benchmark}" "${benchmark_args[@]}" \
  --output="${source_root}/results/${run_name}.json" \
  >"${host_root}/results/${run_name}_node0.log" 2>&1
rank0_rc=$?

wait "${rank1_pid}"
rank1_rc=$?
wait "${rank2_pid}"
rank2_rc=$?
wait "${rank3_pid}"
rank3_rc=$?
printf 'run=%s rank0_rc=%s rank1_rc=%s rank2_rc=%s rank3_rc=%s\n' \
  "${run_name}" "${rank0_rc}" "${rank1_rc}" "${rank2_rc}" "${rank3_rc}"
exit $((rank0_rc || rank1_rc || rank2_rc || rank3_rc))
