#!/usr/bin/env bash

set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
route_dir=/workspace/output/hiermoe_p4_route_replay_20260720/routes
benchmark=${source_root}/scripts/profile/benchmark_hiermoe_pipeline_overlap.py
key=/home/tzq/KeyPair-3bce.pem
master_addr=192.168.0.55
master_port=${MASTER_PORT:-29977}
run_name=${RUN_NAME:-pipeline_overlap_ep32_l0}
mode=${MODE:-all}
layer=${LAYER:-0}
warmup=${WARMUP:-2}
iterations=${ITERATIONS:-5}
compute_window_ms=${COMPUTE_WINDOW_MS:-50}
foreground_a2a_mib=${FOREGROUND_A2A_MIB:-64}
foreground_transport=${FOREGROUND_TRANSPORT:-rank-dedup}
slot_increment=${SLOT_INCREMENT:-1}
fail_on_threshold=${FAIL_ON_THRESHOLD:-0}
pipeline_stage_timing=${PIPELINE_STAGE_TIMING:-1}
hiermoe_internal_timing=${HIERMOE_INTERNAL_TIMING:-1}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-55000}

benchmark_args=(
  --mode="${mode}"
  --route-dir="${route_dir}"
  --layer="${layer}"
  --num-experts=128
  --hidden-size=2048
  --moe-intermediate-size=768
  --slot-increment="${slot_increment}"
  --group-sizes 8 32
  --ranks-per-node=8
  --max-copies=4
  --compute-window-ms="${compute_window_ms}"
  --foreground-a2a-mib="${foreground_a2a_mib}"
  --foreground-transport="${foreground_transport}"
  --warmup="${warmup}"
  --iterations="${iterations}"
)
if [[ "${fail_on_threshold}" == "1" ]]; then
  benchmark_args+=(--fail-on-threshold)
fi

launch_remote() {
  local host=$1
  local node_rank=$2
  local container=$3
  ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    docker exec \
    -e "PYTHONPATH=${source_root}" \
    -e "VEOMNI_HIERMOE_PIPELINE_STAGE_TIMING=${pipeline_stage_timing}" \
    -e "VEOMNI_HIERMOE_INTERNAL_TIMING=${hiermoe_internal_timing}" \
    -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}" \
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
  -e "VEOMNI_HIERMOE_PIPELINE_STAGE_TIMING=${pipeline_stage_timing}" \
  -e "VEOMNI_HIERMOE_INTERNAL_TIMING=${hiermoe_internal_timing}" \
  -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}" \
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
