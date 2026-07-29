#!/usr/bin/env bash
set -uo pipefail

host_root=/home/tzq/npu_profile_outputs/hiermoe_greedy_swap_cover_20260722
container_root=/workspace/output/hiermoe_greedy_swap_cover_20260722
source_root=${container_root}/src
launcher=${source_root}/scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh
variant=${E2E_VARIANT:-greedy}
master_port=${MASTER_PORT:-29950}
run_suffix=${RUN_SUFFIX:-20260722}
swap_interval=${HIERMOE_SWAP_INTERVAL_OVERRIDE:-1}
max_steps=${MAX_STEPS_OVERRIDE:-6}
model_path=${MODEL_PATH_OVERRIDE:-/workspace/model/Qwen3-VL-30B-A3B-Instruct}
model_config_path=${MODEL_CONFIG_PATH_OVERRIDE:-${model_path}}
data_path=${DATA_PATH_OVERRIDE:-/workspace/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs_share_full_shards}
data_source_name=${DATA_SOURCE_NAME_OVERRIDE:-sharegpt4v_sft}
micro_batch_size=${MICRO_BATCH_SIZE_OVERRIDE:-4}
global_batch_size=${GLOBAL_BATCH_SIZE_OVERRIDE:-128}
max_seq_len=${MAX_SEQ_LEN_OVERRIDE:-4096}
data_num_workers=${DATA_NUM_WORKERS_OVERRIDE:-4}
data_prefetch_factor=${DATA_PREFETCH_FACTOR_OVERRIDE:-2}
freeze_vit=${TRAIN_FREEZE_VIT_OVERRIDE:-false}
rms_norm_gated_impl=${RMS_NORM_GATED_IMPLEMENTATION_OVERRIDE:-npu}
causal_conv1d_impl=${CAUSAL_CONV1D_IMPLEMENTATION_OVERRIDE:-eager}
chunk_gated_delta_rule_impl=${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION_OVERRIDE:-eager}
perf_model_path=${HIERMOE_PERF_MODEL_PATH_OVERRIDE:-/workspace/output/hiermoe_perf_model_c009_ep32_20260720/v2/hiermoe_perf_model.json}
greedy_copy_cap=${HIERMOE_GREEDY_MAX_COPIES_OVERRIDE:-4}
greedy_adaptive_topk=${HIERMOE_GREEDY_ADAPTIVE_TOPK_OVERRIDE:-0}
greedy_adaptive_topk_initial=${HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL_OVERRIDE:-32}
greedy_adaptive_topk_strict=${HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT_OVERRIDE:-0}
capture_routes=${HIERMOE_CAPTURE_ROUTES:-0}
capture_mode=${HIERMOE_CAPTURE_MODE_OVERRIDE:-local}
capture_step=${HIERMOE_CAPTURE_STEP_OVERRIDE:--1}
capture_layer=${HIERMOE_CAPTURE_LAYER_OVERRIDE:-}
capture_call=${HIERMOE_CAPTURE_CALL_OVERRIDE:-0}
debug_copy_stats=${HIERMOE_DEBUG_COPY_STATS_OVERRIDE:-0}
debug_copy_layers=${HIERMOE_DEBUG_COPY_LAYERS_OVERRIDE:-1}
debug_copy_groups=${HIERMOE_DEBUG_COPY_GROUPS_OVERRIDE:-2}
hccl_if_base_port=${HCCL_IF_BASE_PORT:-55000}
ablation_replay_path=${HIERMOE_ABLATION_REPLAY_PATH_OVERRIDE:-${source_root}/results/qwen3vl_greedy_ep32_mb4_gbs128_r2_pipeline_6step_hccl_serial_nonblocking_score_20260726_r3_committed_layout.json}
static_preload_layout_path=${HIERMOE_STATIC_PRELOAD_LAYOUT_PATH_OVERRIDE:-}
key=/home/tzq/KeyPair-3bce.pem

hiermoe_enable=false
dedup_only=0
token_dedup=false
expert_swap=false
max_pairs=0
redundant_slots=0
replica_rounds=0
fixed_r2=0
fixed_pipeline=false
swap_mode=step
swap_selector=current_joint
ablation_replay_mode=off
ablation_migration_mode=hidden
ablation_grad_mode=hidden
cpu_planner_mode=off
cpu_train_cores_per_rank=${HIERMOE_CPU_TRAIN_CORES_PER_RANK_OVERRIDE:-8}
npu_layer_owner_blocking=0
npu_layer_owner_collective=${HIERMOE_NPU_LAYER_OWNER_COLLECTIVE_OVERRIDE:-reduce_scatter}
online_freeze_cost_mode=off
online_freeze_calibration_step=${HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP_OVERRIDE:-1}
online_freeze_communication_ratio=${HIERMOE_ONLINE_FREEZE_COMMUNICATION_RATIO_OVERRIDE:-3.1}
online_freeze_compute_ratio=${HIERMOE_ONLINE_FREEZE_COMPUTE_RATIO_OVERRIDE:-4.19}
online_freeze_inter_ms_per_byte=${HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE_OVERRIDE:-6.765449326279194e-08}
online_freeze_intra_ms_per_byte=${HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE_OVERRIDE:-5.02482606728045e-09}
online_freeze_route_ms_per_assignment=${HIERMOE_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT_OVERRIDE:-8.746548178958447e-05}
online_freeze_traffic_intercept_ms=${HIERMOE_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS_OVERRIDE:-16.771503695343263}
cost_model_verify=0
forward_reuse_cover=0
forward_reuse_cover_patch_remap=0
forward_reuse_cover_fast=0
forward_reuse_cover_compute_weight=${HIERMOE_FORWARD_REUSE_COVER_COMPUTE_WEIGHT_OVERRIDE:-1.0}
forward_reuse_cover_compute_ms_per_assignment=${HIERMOE_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT_OVERRIDE:-2.82807e-05}
forward_reuse_cover_min_gain=${HIERMOE_FORWARD_REUSE_COVER_MIN_GAIN_OVERRIDE:-0.0}
forward_reuse_cover_rounds=${HIERMOE_FORWARD_REUSE_COVER_ROUNDS_OVERRIDE:-1}
forward_reuse_cover_only_step=${HIERMOE_FORWARD_REUSE_COVER_ONLY_STEP_OVERRIDE:--1}
forward_reuse_cover_victim_mode=${HIERMOE_FORWARD_REUSE_COVER_VICTIM_MODE_OVERRIDE:-minimum}
forward_reuse_cover_service_scope=${HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE_OVERRIDE:-rank}
forward_reuse_cover_confirm_samples=${HIERMOE_FORWARD_REUSE_COVER_CONFIRM_SAMPLES_OVERRIDE:-1}
forward_reuse_cover_aggregate_service_group=${HIERMOE_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP_OVERRIDE:-0}
forward_reuse_cover_proposal_topk=${HIERMOE_FORWARD_REUSE_COVER_PROPOSAL_TOPK_OVERRIDE:-1}
forward_reuse_cover_empty_seeding=0
hiermoe_internal_timing=${VEOMNI_HIERMOE_INTERNAL_TIMING_OVERRIDE:-0}
force_fixed_r2_mirrored_remap=0
full_profile_start_step=${FULL_PROFILE_START_STEP_OVERRIDE:-3}
full_profile_every_n=${FULL_PROFILE_EVERY_N_OVERRIDE:-1}
full_profile_ranks=${FULL_PROFILE_RANKS_OVERRIDE:-0}
num_moe_layers=${NUM_MOE_LAYERS_OVERRIDE:-48}
redundant_slots_override=${HIERMOE_REDUNDANT_SLOTS_OVERRIDE:-}

case "${variant}" in
  baseline)
    ;;
  dedup)
    hiermoe_enable=
    dedup_only=1
    token_dedup=true
    ;;
  fixed_r2)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    ;;
  fixed_r2_greedy_sync)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  fixed_r2_pipeline_grad)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  fixed_r2_mirrored_pipeline_grad)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    force_fixed_r2_mirrored_remap=1
    ;;
  r2_pipeline)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  cpu_planner_blocking|cpu_planner_background)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    if [[ "${variant}" == "cpu_planner_blocking" ]]; then
      cpu_planner_mode=blocking
    else
      cpu_planner_mode=background
    fi
    ;;
  cpu_process_blocking|cpu_process_background)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    if [[ "${variant}" == "cpu_process_blocking" ]]; then
      cpu_planner_mode=process_blocking
    else
      cpu_planner_mode=process_background
    fi
    ;;
  npu_layer_owner_blocking)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    npu_layer_owner_blocking=1
    ;;
  planner_init_freeze)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=128
    fixed_r2=0
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    # Empty-slot initialization bypasses the steady-state interval check.
    # After step 1 fills all slots, this interval freezes the resulting layout.
    swap_interval=1000
    ;;
  online_freeze_comm|online_freeze_joint)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=128
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    full_profile_start_step=4
    if [[ "${variant}" == "online_freeze_comm" ]]; then
      online_freeze_cost_mode=communication
      # The communication-only control does not have A_max to absorb local
      # route/pack work, so it uses the two-feature coefficients fitted
      # directly to the complete communication region.
      online_freeze_inter_ms_per_byte=${HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE_OVERRIDE:-6.5085072685786e-08}
      online_freeze_intra_ms_per_byte=${HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE_OVERRIDE:-2.0419282740722182e-08}
      online_freeze_route_ms_per_assignment=0.0
      online_freeze_traffic_intercept_ms=${HIERMOE_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS_OVERRIDE:-14.45356840775864}
    else
      online_freeze_cost_mode=joint
    fi
    ;;
  cost_model_verify)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=0
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_grad_mode=hidden
    cost_model_verify=1
    hiermoe_internal_timing=1
    force_fixed_r2_mirrored_remap=1
    full_profile_start_step=1
    ;;
  forward_reuse_cover)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    ;;
  forward_reuse_cover_patch)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    ;;
  forward_reuse_cover_empty_seed)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=1
    fixed_r2=0
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    forward_reuse_cover_empty_seeding=1
    ;;
  forward_reuse_cover_fast)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    forward_reuse_cover_fast=1
    ;;
  forward_reuse_cover_patch_static)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=0
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_replay_mode=static
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    ;;
  forward_reuse_cover_empty_static|hierarchical_primary_static|hierarchical_full_static)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=0
    fixed_r2=0
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_replay_mode=static
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    forward_reuse_cover_empty_seeding=1
    if [[ "${variant}" == "hierarchical_full_static" && -z "${static_preload_layout_path}" ]]; then
      static_preload_layout_path=${ablation_replay_path}
    fi
    ;;
  forward_reuse_cover_patch_multiround)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=0
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    ablation_migration_mode=blocking
    ablation_grad_mode=hidden
    forward_reuse_cover=1
    forward_reuse_cover_patch_remap=1
    forward_reuse_cover_rounds=${HIERMOE_FORWARD_REUSE_COVER_ROUNDS_OVERRIDE:-32}
    forward_reuse_cover_only_step=${HIERMOE_FORWARD_REUSE_COVER_ONLY_STEP_OVERRIDE:-2}
    forward_reuse_cover_victim_mode=${HIERMOE_FORWARD_REUSE_COVER_VICTIM_MODE_OVERRIDE:-minimum}
    ;;
  r2_planner)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=4
    replica_rounds=1
    fixed_r2=1
    swap_mode=layer
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  ablation_g_blocking|ablation_g_hidden|ablation_m_blocking|ablation_m_hidden)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    redundant_slots=4
    fixed_r2=1
    fixed_pipeline=true
    swap_mode=step
    swap_selector=hiermoe_greedy_cover_p1
    if [[ "${variant}" == ablation_g_* ]]; then
      ablation_replay_mode=static
      ablation_migration_mode=blocking
      if [[ "${variant}" == "ablation_g_blocking" ]]; then
        ablation_grad_mode=blocking
      else
        ablation_grad_mode=hidden
      fi
    else
      ablation_replay_mode=step
      ablation_grad_mode=blocking
      if [[ "${variant}" == "ablation_m_blocking" ]]; then
        ablation_migration_mode=blocking
      else
        ablation_migration_mode=hidden
      fi
    fi
    ;;
  greedy)
    hiermoe_enable=true
    token_dedup=true
    expert_swap=true
    max_pairs=1
    redundant_slots=1
    replica_rounds=32
    swap_mode=layer
    swap_selector=hiermoe_greedy_cover_p1
    ;;
  *)
    echo "unsupported E2E_VARIANT=${variant}" >&2
    exit 2
    ;;
esac

if [[ -n "${redundant_slots_override}" ]]; then
  redundant_slots=${redundant_slots_override}
fi
ablation_grad_mode=${HIERMOE_ABLATION_GRAD_MODE_OVERRIDE:-${ablation_grad_mode}}

run_name=${RUN_NAME_OVERRIDE:-qwen3vl_greedy_ep32_mb${micro_batch_size}_gbs${global_batch_size}_${variant}_${max_steps}step_${run_suffix}}
common_env=(
  -e "PYTHONPATH=${source_root}"
  -e "RUN_NAME=${run_name}"
  -e "RUN_ROOT=${source_root}/pretrain_runs/${run_name}"
  -e "MODEL_PATH=${model_path}"
  -e "MODEL_CONFIG_PATH=${model_config_path}"
  -e "DATA_PATH=${data_path}"
  -e "DATA_SOURCE_NAME=${data_source_name}"
  -e "DATA_NUM_WORKERS=${data_num_workers}"
  -e "DATA_PREFETCH_FACTOR=${data_prefetch_factor}"
  -e "TRAIN_FREEZE_VIT=${freeze_vit}"
  -e "RMS_NORM_GATED_IMPL=${rms_norm_gated_impl}"
  -e "CAUSAL_CONV1D_IMPL=${causal_conv1d_impl}"
  -e "CHUNK_GATED_DELTA_RULE_IMPL=${chunk_gated_delta_rule_impl}"
  -e "NNODES=4"
  -e "NPROC_PER_NODE=8"
  -e "MASTER_ADDR=192.168.0.55"
  -e "MASTER_PORT=${master_port}"
  -e "MAX_STEPS=${max_steps}"
  -e "MICRO_BATCH_SIZE=${micro_batch_size}"
  -e "GLOBAL_BATCH_SIZE=${global_batch_size}"
  -e "MAX_SEQ_LEN=${max_seq_len}"
  -e "DP_REPLICATE_SIZE=1"
  -e "DP_SHARD_SIZE=32"
  -e "EP_SIZE=32"
  -e "NUM_MOE_LAYERS=${num_moe_layers}"
  -e "HIERMOE_ENABLE=${hiermoe_enable}"
  -e "HIERMOE_DEDUP_ONLY=${dedup_only}"
  -e "HIERMOE_TOKEN_DEDUP=${token_dedup}"
  -e "HIERMOE_COMMUNICATION_MODE=hierarchical"
  -e "HIERMOE_EXPERT_SWAP=${expert_swap}"
  -e "HIERMOE_EXPERT_SWAP_INTERVAL=${swap_interval}"
  -e "HIERMOE_LOG_INTERVAL=1"
  -e "HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=${max_pairs}"
  -e "HIERMOE_EXPERT_SWAP_SELECTOR=${swap_selector}"
  -e "HIERMOE_REDUNDANT_SLOT_INCREMENT_PER_DEVICE=${redundant_slots}"
  -e "HIERMOE_GREEDY_MAX_COPIES_PER_EXPERT=${greedy_copy_cap}"
  -e "HIERMOE_MAX_SLOT_OP_SEARCH_ROUNDS=${replica_rounds}"
  -e "HIERMOE_EXPERT_SWAP_MODE=${swap_mode}"
  -e "HIERMOE_FIXED_PIPELINE_OVERLAP=${fixed_pipeline}"
  -e "VEOMNI_HIERMOE_PIPELINE_STAGE_TIMING=1"
  -e "VEOMNI_HIERMOE_PIPELINE_PLAN_WORKERS=64"
  -e "VEOMNI_HIERMOE_ABLATION_REPLAY_PATH=${ablation_replay_path}"
  -e "VEOMNI_HIERMOE_ABLATION_REPLAY_MODE=${ablation_replay_mode}"
  -e "VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE=${ablation_migration_mode}"
  -e "VEOMNI_HIERMOE_ABLATION_GRAD_MODE=${ablation_grad_mode}"
  -e "VEOMNI_HIERMOE_STATIC_PRELOAD_LAYOUT_PATH=${static_preload_layout_path}"
  -e "VEOMNI_HIERMOE_CPU_PLANNER_MODE=${cpu_planner_mode}"
  -e "VEOMNI_HIERMOE_CPU_TRAIN_CORES_PER_RANK=${cpu_train_cores_per_rank}"
  -e "VEOMNI_HIERMOE_NPU_LAYER_OWNER_BLOCKING=${npu_layer_owner_blocking}"
  -e "VEOMNI_HIERMOE_NPU_LAYER_OWNER_COLLECTIVE=${npu_layer_owner_collective}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE=${online_freeze_cost_mode}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP=${online_freeze_calibration_step}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_COMMUNICATION_RATIO=${online_freeze_communication_ratio}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_COMPUTE_RATIO=${online_freeze_compute_ratio}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE=${online_freeze_inter_ms_per_byte}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE=${online_freeze_intra_ms_per_byte}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT=${online_freeze_route_ms_per_assignment}"
  -e "VEOMNI_HIERMOE_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS=${online_freeze_traffic_intercept_ms}"
  -e "VEOMNI_HIERMOE_COST_MODEL_VERIFY=${cost_model_verify}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER=${forward_reuse_cover}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP=${forward_reuse_cover_patch_remap}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_FAST=${forward_reuse_cover_fast}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_WEIGHT=${forward_reuse_cover_compute_weight}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT=${forward_reuse_cover_compute_ms_per_assignment}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_MIN_GAIN=${forward_reuse_cover_min_gain}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ROUNDS=${forward_reuse_cover_rounds}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ONLY_STEP=${forward_reuse_cover_only_step}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_VICTIM_MODE=${forward_reuse_cover_victim_mode}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE=${forward_reuse_cover_service_scope}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_CONFIRM_SAMPLES=${forward_reuse_cover_confirm_samples}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP=${forward_reuse_cover_aggregate_service_group}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PROPOSAL_TOPK=${forward_reuse_cover_proposal_topk}"
  -e "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_EMPTY_SEEDING=${forward_reuse_cover_empty_seeding}"
  -e "VEOMNI_HIERMOE_FORCE_FIXED_R2_MIRRORED_REMAP=${force_fixed_r2_mirrored_remap}"
  -e "HIERMOE_FIT_PERF_MODEL_ON_STARTUP=0"
  -e "HIERMOE_PERF_MODEL_PATH=${perf_model_path}"
  -e "VEOMNI_HIERMOE_FIXED_R2_LAYOUT=${fixed_r2}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK=${greedy_adaptive_topk}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL=${greedy_adaptive_topk_initial}"
  -e "VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT=${greedy_adaptive_topk_strict}"
  -e "VEOMNI_HIERMOE_FULL_ROUTE_GATHER_MAX_TOKENS=16384"
  -e "VEOMNI_FULL_PROFILE_ENABLE=1"
  -e "VEOMNI_FULL_PROFILE_START_STEP=${full_profile_start_step}"
  -e "VEOMNI_FULL_PROFILE_EVERY_N=${full_profile_every_n}"
  -e "VEOMNI_FULL_PROFILE_RANKS=${full_profile_ranks}"
  -e "VEOMNI_HIERMOE_INTERNAL_TIMING=${hiermoe_internal_timing}"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS=${debug_copy_stats}"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS=${debug_copy_layers}"
  -e "VEOMNI_HIERMOE_DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS=${debug_copy_groups}"
  -e "VEOMNI_TORCH_PROFILE_ENABLE=0"
  -e "VEOMNI_MOE_TIMING_SYNC_EVENTS=0"
  -e "HCCL_IF_BASE_PORT=${hccl_if_base_port}"
)

if [[ "${capture_routes}" == "1" ]]; then
  capture_root=${source_root}/route_captures/${run_name}
  common_env+=(
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE=${capture_mode}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH=${capture_root}/step{step:04d}/layer{layer_index:02d}_call{call}_rank{rank:02d}.pt"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP=${capture_step}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_LAYER=${capture_layer}"
    -e "VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL=${capture_call}"
  )
fi

launch_remote() {
  local host=$1
  local node_rank=$2
  local container=$3
  ssh -i "${key}" -o StrictHostKeyChecking=no "root@${host}" \
    docker exec \
    "${common_env[@]}" \
    -e "NODE_RANK=${node_rank}" \
    -w "${source_root}" \
    "${container}" \
    bash "${launcher}" \
    >"${host_root}/${run_name}_rank${node_rank}.host.log" 2>&1
}

rank0_container=${RANK0_CONTAINER_OVERRIDE:-tzq_npu_coremoe_verify_20260717}
rank1_container=${RANK1_CONTAINER_OVERRIDE:-tzq_npu_static_r2_rank1_20260720}
rank2_container=${RANK2_CONTAINER_OVERRIDE:-tzq_npu_static_r2_rank2_20260719}
rank3_container=${RANK3_CONTAINER_OVERRIDE:-tzq_npu_static_r2_rank3_20260719}

launch_remote 192.168.0.190 1 "${rank1_container}" &
rank1_pid=$!
launch_remote 192.168.0.109 2 "${rank2_container}" &
rank2_pid=$!
launch_remote 192.168.0.9 3 "${rank3_container}" &
rank3_pid=$!

docker exec \
  "${common_env[@]}" \
  -e NODE_RANK=0 \
  -w "${source_root}" \
  "${rank0_container}" \
  bash "${launcher}" \
  >"${host_root}/${run_name}_rank0.host.log" 2>&1
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
