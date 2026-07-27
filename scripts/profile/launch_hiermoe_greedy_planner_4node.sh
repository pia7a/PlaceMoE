#!/usr/bin/env bash

set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
route_dir=/workspace/output/hiermoe_p4_route_replay_20260720/routes
benchmark=${source_root}/scripts/profile/benchmark_hiermoe_greedy_planner.py
key=/home/tzq/KeyPair-3bce.pem
master_addr=192.168.0.55
master_port=${MASTER_PORT:-29976}
run_name=${RUN_NAME:-planner_sufficient_stats_no_kernel_ep32_l0}
warmup=${WARMUP:-2}
iterations=${ITERATIONS:-7}
candidate_collective=${CANDIDATE_COLLECTIVE:-full}
layer=${LAYER:-0}
layer_count=${LAYER_COUNT:-1}
layer_execution=${LAYER_EXECUTION:-sequential}
layer_parallel_streams=${LAYER_PARALLEL_STREAMS:-8}
layer_owner=${LAYER_OWNER:-0}
layer_owner_collective=${LAYER_OWNER_COLLECTIVE:-reduce_scatter}
communication_scale=${COMMUNICATION_SCALE:-1.0}
forward_compute_per_assignment=${FORWARD_COMPUTE_PER_ASSIGNMENT:-0.0}
forward_compute_constant=${FORWARD_COMPUTE_CONSTANT:-0.0}
adaptive_topk=${ADAPTIVE_TOPK:-0}
adaptive_topk_initial=${ADAPTIVE_TOPK_INITIAL:-16}
adaptive_topk_strict=${ADAPTIVE_TOPK_STRICT:-0}
early_proxy_topk=${EARLY_PROXY_TOPK:-0}
exact_primitive_topk=${EXACT_PRIMITIVE_TOPK:-0}
post_shortlist_compact_pair=${POST_SHORTLIST_COMPACT_PAIR:-0}
exact_primitive_max_only=${EXACT_PRIMITIVE_MAX_ONLY:-0}
compare_full_exact=${COMPARE_FULL_EXACT:-0}

benchmark_args=(
  --route-dir=${route_dir}
  --layer=${layer}
  --layer-count=${layer_count}
  --layer-execution=${layer_execution}
  --layer-parallel-streams=${layer_parallel_streams}
  --ep-size=32
  --group-sizes 8 32
  --local-world-size=8
  --slot-increment=1
  --phase=steady
  --max-swaps=1
  --max-covers=1
  --max-copies=4
  --communication-scale=${communication_scale}
  --forward-compute-per-assignment=${forward_compute_per_assignment}
  --forward-compute-constant=${forward_compute_constant}
  --candidate-scorer=statistics
  --candidate-collective=${candidate_collective}
  --backend=hccl
  --warmup=${warmup}
  --iterations=${iterations}
)
if [[ "${adaptive_topk}" == "1" ]]; then
  benchmark_args+=(--adaptive-topk --adaptive-topk-initial="${adaptive_topk_initial}")
fi
if [[ "${layer_owner}" == "1" ]]; then
  benchmark_args+=(--layer-owner --layer-owner-collective="${layer_owner_collective}")
fi
if [[ "${adaptive_topk_strict}" == "1" ]]; then
  benchmark_args+=(--adaptive-topk-strict-certificate)
fi
if (( early_proxy_topk > 0 )); then
  benchmark_args+=(--early-proxy-topk="${early_proxy_topk}")
fi
if (( exact_primitive_topk > 0 )); then
  benchmark_args+=(--exact-primitive-topk="${exact_primitive_topk}")
fi
if [[ "${post_shortlist_compact_pair}" == "1" ]]; then
  benchmark_args+=(--post-shortlist-compact-pair)
fi
if [[ "${exact_primitive_max_only}" == "1" ]]; then
  benchmark_args+=(--exact-primitive-max-only)
fi
if [[ "${compare_full_exact}" == "1" ]]; then
  benchmark_args+=(--compare-full-exact)
fi

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
