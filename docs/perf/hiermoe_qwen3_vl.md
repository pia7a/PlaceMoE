# HierMoE Qwen3-VL MoE Benchmark

This benchmark compares the original VeOmni MoE EP communication path against
HierMoE token deduplication and Expert Swap on `Qwen3-VL-30B-A3B-Instruct`.

The NPU profile launcher already overrides the MoE backend with `MOE_IMPL`
and defaults to `fused_npu`. Keep the backend identical between the two runs.

## Baseline

```bash
cd /workspace/task3/VeOmni-0.1.11
MOE_IMPL=fused_npu \
EP_SIZE=<N> \
NODE_RANK=<rank> \
bash scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh \
  --train.accelerator.ep_size <N> \
  --train.hiermoe.enable false
```

## HierMoE Token Dedup

```bash
cd /workspace/task3/VeOmni-0.1.11
MOE_IMPL=fused_npu \
EP_SIZE=<N> \
NODE_RANK=<rank> \
bash scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh \
  --train.accelerator.ep_size <N> \
  --train.hiermoe.enable true \
  --train.hiermoe.token_dedup true \
  --train.hiermoe.expert_swap false
```

## Full HierMoE

```bash
cd /workspace/task3/VeOmni-0.1.11
MOE_IMPL=fused_npu \
EP_SIZE=<N> \
NODE_RANK=<rank> \
bash scripts/profile/run_qwen3_vl_30b_full_ep32_4node_light_profile_100step_npu.sh \
  --train.hiermoe.enable true \
  --train.hiermoe.token_dedup true \
  --train.hiermoe.expert_swap true
```

`train.hiermoe.communication_mode` controls the deduplicated dispatch/combine
path. `hierarchical` is the default and always uses the deepest path supported
by the inferred/configured topology; `direct` always uses rank-level
All-to-All; `auto` evaluates all available dimensions with the current route
and profiled alpha/beta coefficients. The profile launcher exposes the same
setting as `HIERMOE_COMMUNICATION_MODE`.

### Expert-swap selector

`train.hiermoe.expert_swap_selector` selects the expert-swap implementation:

- `current_joint` (default) keeps the current-route communication-plus-compute planner.
- `hiermoe_exact_p1` evaluates every expert pair with the paper-exact,
  duplicate-free hierarchical communication objective, and applies the globally
  best pair only when its cost is strictly lower.
- `legacy_batched` keeps the historical batched P1/P4 estimator while scoring
  the complete `C(num_experts, 2)` pair set. P4 then greedily chooses at most
  four disjoint improving pairs.

Any active swap or replica planner requires measured alpha/beta coefficients.
Set `train.hiermoe.perf_model_path` to JSON emitted by
`profile/scripts/bench_hiermoe_perf_model.py`, or enable
`train.hiermoe.fit_perf_model_on_startup`. Missing, default, or unverified
coefficients fail before the placement manager is constructed.

The exact selector batches additive route statistics for all registered MoE
layers into one EP all-reduce per swap boundary. It is intentionally limited to
`expert_swap_mode=step`, `expert_swap_max_pairs_per_layer=1`, no redundant
slots, at most 256 experts per layer, and at most 64 MiB of packed statistics.
These constraints fail fast instead of silently changing the P1 algorithm.

The profile launcher exposes the selector as
`HIERMOE_EXPERT_SWAP_SELECTOR`. For a direct four-node P1 comparison with the
existing runner, use:

```bash
E2E_VARIANT=p1_current RUN_SUFFIX=<tag> \
  bash /home/tzq/npu_profile_outputs/hiermoe_exact_p1_optional_20260721/run_p1_e2e_4node.sh
E2E_VARIANT=p1_exact RUN_SUFFIX=<tag> \
  bash /home/tzq/npu_profile_outputs/hiermoe_exact_p1_optional_20260721/run_p1_e2e_4node.sh
```

Both variants use one swap pair, no redundant slots, and the full ShareGPT4V
dataset; only the selector differs. The runner defaults to six steps; set
`MAX_STEPS_OVERRIDE=100` for a longer throughput comparison. Exact-selector metrics are logged under
`hiermoe/exact_p1_*`, including statistics, collective and scoring time,
candidate count, and accepted count.

Expert Swap is implemented for `ep_fsdp_size=1`, where EP partitions complete
experts and each EP rank owns whole local expert slots. This matches the
4-node Qwen3-VL profile setting `DP_SHARD_SIZE=32` and `EP_SIZE=32`.
If `ep_fsdp_size > 1`, Expert Swap fails fast rather than moving partial
FSDP shards as if they were complete experts.

FSDP2 CPU offload is also rejected when swap or replica placement is active.
The planner and executor use the accelerator EP process group and migrate live
parameter and optimizer payloads, so CPU-resident expert state is not a
supported placement configuration.

### Current-route swap-then-replica planning

The current-route planner runs a strict greedy swap stage followed by a strict
greedy replica stage. The public budgets select the available actions:

| swap pairs | redundant slots | available actions |
| ---: | ---: | --- |
| `0` | `0` | no-op |
| `>0` | `0` | expert swap |
| `0` | `>0` | expert replica |
| `>0` | `>0` | swap, then replica |

`P=0,R=0` does not construct a placement manager and therefore does not add
the current-route gradient-accumulation or `ep_fsdp_size=1` restrictions.

`train.hiermoe.max_slot_op_search_rounds` is an optional experimental cap for
replica planning. When omitted, the effective budget is
`redundant_slot_increment_per_device * ep_size`. Explicit `0` reserves the
configured redundant slots but disables replica planning; a positive value is
clamped to the same slot capacity. The runtime copy table is sized from the
copies present in the committed layout and is independent of this search
budget.

The cost model is `Tcomm + Tcompute`. `Tcomm` is the sum of forward dispatch,
forward combine, backward dispatch, and backward combine, using deduplicated
token traffic and the configured hierarchical `alpha + beta * bytes` model.
`Tcompute` uses the original, non-deduplicated token-expert assignments and is
`3 * forward_compute_cost * max_rank_assignments`, accounting for one forward
and approximately two forward-equivalent backward expert computations. Both
terms use their independently heaviest EP rank.

Each swap round evaluates every eligible pair formed from experts on the
communication/compute bottleneck ranks and experts on other ranks. Every round
uses global expert-group and sole-expert co-occurrence statistics to evaluate
the exact four-case token deltas without scoring complete layouts. Swap pairs
are disjoint.
The replica stage then evaluates bottleneck experts against available redundant
slots on the post-swap layout. A token selects one physical copy by preferring
a rank already needed by another top-k expert, then the nearest hierarchy
level, then a deterministic route hash.

Both stages retain their global route state across greedy rounds. Swap updates
only the two owner ranks and affected hierarchy-group columns. Replica scoring
uses a fixed bottleneck-expert table and one row per unique token-logical route;
top-k duplicates contribute once to communication and use their multiplicity
for expert-compute assignments. Accepted replicas update only the selected
copy set, routed assignments, token-group occupancy, and global cost. Candidate
statistics use one fused reduction per attempted round; no candidate performs
its own collective or complete-layout rescore.

An action is accepted only when the predicted communication-plus-compute time
strictly decreases. There is no EMA, minimum-gain threshold, or transfer-cost
term in candidate selection. Planning time, expert state transfer, and
redundant-gradient synchronization are measured separately. A negative best
gain immediately stops the corresponding stage.

`expert_swap_mode=step` plans from the previous completed step's raw routes.
`expert_swap_mode=layer` plans from the current layer's exact routes before
dispatch and therefore requires `gradient_accumulation_steps == 1`. Step 0
collects the raw communication and expert-compute calibration sample without
changing placement; later decisions use the most recent completed sample and
do not smooth it.

Redundant-gradient synchronization writes the same summed logical gradient to
every physical copy so replicas stay identical after the optimizer update.
FSDP2 gradient clipping computes its norm from one owner slot per logical
expert, then applies the resulting clip coefficient to every physical copy.

Frozen reference models, such as the DPO reference policy, execute with
placement mapping disabled. They still use token deduplication, but do not
reserve redundant slots or inherit the trainable policy's expert layout.

DCP resume requires the same redundant-slot budget used when saving. Compact
and slot-expanded expert parameters and optimizer states have different tensor
shapes, so checkpoint extra state is validated before DCP loads either tensor
set; cross-budget resume fails fast instead of attempting a metadata-only
layout conversion.

Parameter and optimizer-state payload signatures, byte counts, and tensor
references are validated and frozen across EP ranks before an accepted plan is
executed. The executor migrates swaps first, then fills or retargets redundant
slots, and commits the final layout once so dispatch cannot observe an
intermediate mapping.

Placement logs include the accepted actions, predicted communication/compute
cost, planning time, migration time, redundant-gradient synchronization time,
and route/swap/replica planning breakdowns. Planner metrics also report the
configured replica-round value (`auto` when omitted), total slot capacity,
effective round cap, accepted rounds, score/update/collective time, decision
synchronization, and finalization.

The EP16 layer-24 replay snapshot used for planner regression has SHA256
`bb588b7c8aeba46bbb2f88f92f4cd73eeff2b822311022b0fbdeeaa2b025de1c`.
Run the distributed planner replay once per node with matching rendezvous
arguments; for example, on a two-node, eight-NPU-per-node cluster:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=<0-or-1> \
  --master-addr=<rank0-internal-ip> --master-port=29667 \
  scripts/profile/benchmark_hiermoe_current_route_planner.py \
  --snapshot <step0_layer24_call0.pt> \
  --configs P1S0 P4S0 P0S1-auto P1S1-auto P4S1-auto \
  --warmup 8 --iterations 20 --backend hccl \
  --output <planner-benchmark.json>
```

With 16 ranks, 128 experts, top-k 8, and roughly 16K tokens per rank, the
two-node NPU results below use eight warmups followed by 20 measured plans:

| config | accepted | comm speedup | compute speedup | modeled MoE speedup | median | P90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1S0` | 1 swap | 1.0056x | 1.2597x | 1.0216x | 16.75 ms | 16.94 ms | 25.85 ms |
| `P4S0` | 4 swaps | 1.0089x | 1.3420x | 1.0287x | 72.13 ms | 74.07 ms | 79.85 ms |
| `P0S1-auto` | 15 replicas | 1.0146x | 1.4961x | 1.0406x | 3190.73 ms | 3195.49 ms | 3196.71 ms |
| `P1S1-auto` | 1 swap + 15 replicas | 1.0146x | 1.4358x | 1.0383x | 3208.26 ms | 3211.80 ms | 3212.19 ms |
| `P4S1-auto` | 4 swaps + 15 replicas | 1.0157x | 1.5142x | 1.0424x | 3255.48 ms | 3258.96 ms | 3261.28 ms |

`P1S0` and `P4S0` meet the hard per-layer `max < 100 ms` gate. The automatic
replica configurations do not: 15 strict sequential rounds cost roughly 3.2
seconds per layer, or 153-156 seconds when their median is multiplied by 48
layers. Candidate collectives account for only a few milliseconds; most device
work becomes visible at the required per-round decision synchronization. The
replica path is therefore functionally correct but is not eligible for
end-to-end integration under the current performance gate. These speedups cover
modeled MoE communication and expert compute only; they are not end-to-end
training speedups.

The route-capture analysis script reports a route-count-greedy heuristic and a
best-found heuristic. Both estimate idealized exclusive communication only;
neither is an exact replay of the runtime planner, a mathematical lower bound,
or an end-to-end performance prediction.

## Six-step smoke comparison

Use the same environment as the successful 100-step run, but set:

```bash
MAX_STEPS=6
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128
MAX_SEQ_LEN=4096
DP_REPLICATE_SIZE=1
DP_SHARD_SIZE=32
EP_SIZE=32
MOE_IMPL=fused_npu
ATTN_IMPL=flash_attention_2
VEOMNI_FULL_PROFILE_ENABLE=1
VEOMNI_FULL_PROFILE_START_STEP=3
VEOMNI_FULL_PROFILE_EVERY_N=10
VEOMNI_FULL_PROFILE_RANKS=0
VEOMNI_TORCH_PROFILE_ENABLE=0
VEOMNI_MOE_TIMING_SYNC_EVENTS=0
```

Baseline appends:

```bash
--train.hiermoe.enable false
```

HierMoE appends:

```bash
--train.hiermoe.enable true \
--train.hiermoe.token_dedup true \
--train.hiermoe.expert_swap true
```

Expert Swap candidate generation covers the complete
`C(num_experts, 2)` pair set. For the legacy P1 fallback, the full set is
sharded across EP ranks by default when a layer has more than 64 experts. Set
`VEOMNI_HIERMOE_SWAP_CANDIDATE_SHARDS=1` to force unsharded scoring, or a
positive value such as `16`/`32` to pin the number of shards. P4 fast paths
score the full pair set from globally reduced route statistics. The removed
hot/cold/max-candidate knobs no longer truncate the search space.

Selector experiments:

- `VEOMNI_HIERMOE_SWAP_FAST_2D=1` enables the exact 2D selector estimator.
  It reduces rank-local selector compute on the Qwen3-VL EP32 profile, but is
  still an opt-in tuning knob until it shows consistent end-to-end wins across
  ranks.
- `train.hiermoe.expert_swap_max_pairs_per_layer` controls how many disjoint
  expert pairs a layer may migrate at one swap boundary. The default is `1`,
  which preserves the original single-pair behavior. Larger values reuse the
  same grouped P2P exchange path and batch selected parameter and optimizer
  state slots by peer.
- `train.hiermoe.expert_swap_interval` controls the step interval between
  Expert Swap boundaries. The default is `1`, preserving the paper-style
  every-step behavior. The NPU launcher also exposes this as
  `HIERMOE_EXPERT_SWAP_INTERVAL`.

Pair-count sweep:

```bash
export HIERMOE_SWAP_PAIR_COUNTS="1 2 4 16 32"
export MAX_STEPS=6
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
export EP_SIZE=32
export MOE_IMPL=fused_npu
export HIERMOE_EXPERT_SWAP_INTERVAL=1
bash scripts/profile/sweep_hiermoe_expert_swap_pairs_4node_npu.sh
```

After generating the standard profile tables for every run, plot overhead and
communication load balance with:

```bash
python profile/scripts/plot_hiermoe_expert_swap_pair_sweep.py \
  --baseline-run <baseline_run_name> \
  --runs <pairs1_run> <pairs2_run> <pairs4_run> <pairs16_run> <pairs32_run>
```

## Current 4-node NPU smoke results

These runs use the same hyperparameters as the successful 100-step profile run,
except `MAX_STEPS=6`. Approximate throughput uses
`GLOBAL_BATCH_SIZE * MAX_SEQ_LEN / train_step_total` with
`GLOBAL_BATCH_SIZE=128` and `MAX_SEQ_LEN=4096`.

| run | HierMoE options | rank0 profiled step | train step | approx tokens/s | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3vl_hiermoe_baseline_6step_20260625_1150` | `enable=false` | 3 | 31.76 s | 16.5k | 1.00x |
| `qwen3vl_hiermoe_currentlog_6step_20260626_033702` | `enable=true`, `token_dedup=true`, `expert_swap=true` | 3 | 19.66 s | 26.7k | 1.62x |

The current HierMoE run completed successfully on 4 nodes with status 0 on all
node wrappers. Rank0 logged `selected_dim=2`, `perf_model_source=default`, and
an average dedup ratio of `0.541` for steps 3-6.

Rank0 MoE timing, summed over steps 3-6:

| metric | baseline | HierMoE | ratio |
| --- | ---: | ---: | ---: |
| forward all-to-all cuda time | 63.57 s | 18.50 s | 3.44x lower |
| backward all-to-all cuda time | 33.57 s | 10.32 s | 3.25x lower |
| forward MoE comm region cuda time | 44.96 s | 30.45 s | 1.48x lower |
| forward expert compute cuda time | 1.72 s | 1.68 s | 1.02x lower |
| backward expert compute cuda time | 2.06 s | 2.04 s | 1.01x lower |

The remaining gap between raw all-to-all improvement and end-to-end step speedup
is mostly outside expert compute, in the forward MoE communication region and
surrounding distributed training work. Continue using the same profile breakdown
when evaluating further topology/performance-model or payload-construction
changes.

When `communication_mode=auto`, the conservative default performance model
caps automatic dimension selection at 2D even when `topology=auto` infers a
deeper hierarchy such as `(8, 16, 64)`. Use
`train.hiermoe.perf_model_path` after fitting alpha/beta if a 3D hierarchy
should be selected automatically for 64-card runs.

## Startup alpha/beta fitting

`profile/scripts/bench_hiermoe_perf_model.py` runs before the training
`torchrun` and fits `time_ms = alpha + bytes * beta` for the current rank
layout. It measures global A2A plus the staged HierD groups used by the active
hierarchy, writes a JSON readable by `train.hiermoe.perf_model_path`, and
broadcasts that JSON so every node-local container can read it before step 0.
The benchmark is a separate launch and is not counted in training speedup.

The NPU profile launcher exposes the common controls:

```bash
HIERMOE_FIT_PERF_MODEL_ON_STARTUP=1
HIERMOE_PERF_MODEL_MESSAGE_BYTES_CSV=67108864,134217728,268435456,536870912
HIERMOE_PERF_MODEL_WARMUP=2
HIERMOE_PERF_MODEL_ITERS=5
HIERMOE_PERF_MODEL_MEASURE_LAST_N=3
HIERMOE_PERF_MODEL_MASTER_PORT=$((MASTER_PORT + 37))
```

Use `HIERMOE_HIERARCHY_GROUP_SIZES=8,64` when the fitted benchmark and the
training run should both stay on the current 2D hierarchy. This is useful on
64-card runs while the 3D microbenchmark path is being validated. The launcher
passes the same value to `--hierarchy-group-sizes-csv` for fitting and to
`--train.hiermoe.hierarchy_group_sizes` for training.

For the 64-card all8 short test, keep the existing successful Qwen3-VL
hyperparameters and add:

```bash
export NNODES=8
export NPROC_PER_NODE=8
export MAX_STEPS=6
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=256
export DP_REPLICATE_SIZE=1
export DP_SHARD_SIZE=64
export EP_SIZE=64
export MOE_IMPL=fused_npu
export HIERMOE_ENABLE=true
export HIERMOE_TOKEN_DEDUP=true
export HIERMOE_EXPERT_SWAP=true
export HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=4
export HIERMOE_HIERARCHY_GROUP_SIZES=8,64
export HIERMOE_FIT_PERF_MODEL_ON_STARTUP=1
```

The launcher then appends `--train.hiermoe.perf_model_path` automatically. To
reuse a previous calibration without rerunning the benchmark, set
`HIERMOE_PERF_MODEL_PATH=/path/to/hiermoe_perf_model.json` and leave
`HIERMOE_FIT_PERF_MODEL_ON_STARTUP=0`.

Recorded validation run:

- `qwen3vl_hiermoe_fitperf2d_ep64_mb4_gbs256_6step_all8_20260629_204044`
- Startup fit message sizes: 64/128/256/512 MB, hierarchy `[8,64]`
- Sampled step 3-6 train step total: 121.882 s, 1.438x vs current 64-card baseline
- Sampled MoE all-to-all: 75.186 s, 1.931x vs current 64-card baseline
- Expert Swap: 1.257 s total across 4 profiled steps, 314 ms/profiled step
- Loss over 6 steps: 1.43 -> 0.86

## Metrics

Collect at least these fields from the training logs or wandb:

- `tokens_per_second(M)`
- step time from the existing environment meter
- Event-based comparison fields from `moe_logical_section_summary.csv` and
  `moe_logical_section_by_step_summary.csv`: `backend_path`,
  `logical_section`, `cuda_ms_sum`, `cuda_ms_avg_per_call`, `calls`,
  `tokens`, `token_expert_assignments`.
- Debug wall-clock fields in training logs: `hiermoe/dispatch_wall_ms`,
  `hiermoe/combine_wall_ms`, `hiermoe/local_expert_compute_wall_ms`.
- `hiermoe/dedup_ratio_dispatch`
- `hiermoe/dedup_ratio_combine`
- `hiermoe/selected_dim`
- `hiermoe/expert_swap_pair`
- `hiermoe/perf_model_source`
