#!/usr/bin/env bash

set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
route_dir=/workspace/output/hiermoe_p4_route_replay_20260720/routes
benchmark=${source_root}/scripts/profile/benchmark_hiermoe_traffic_cost_model.py
key=/home/tzq/KeyPair-3bce.pem
master_addr=192.168.0.55
master_port=${MASTER_PORT:-30083}
run_name=${RUN_NAME:-traffic_cost_model_ep32_48layers}
warmup=${WARMUP:-1}
iterations=${ITERATIONS:-3}

benchmark_args=(
  --route-dir="${route_dir}"
  --layers=48
  --warmup="${warmup}"
  --iterations="${iterations}"
  --ranks-per-node=8
  --num-experts=128
  --hidden-size=2048
  --slot-increment=4
  --max-copies=8
)

launch_remote() {
  local host=$1
  local node_rank=$2
  local container=$3
  ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    docker exec \
    -e "PYTHONPATH=${source_root}" \
    -e "VEOMNI_HIERMOE_INTERNAL_TIMING=1" \
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
  -e "VEOMNI_HIERMOE_INTERNAL_TIMING=1" \
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

echo "run=${run_name} rank0_rc=${rank0_rc} rank1_rc=${rank1_rc} rank2_rc=${rank2_rc} rank3_rc=${rank3_rc}"
exit $((rank0_rc || rank1_rc || rank2_rc || rank3_rc))
