# HierMoE Profile Results and Optimization Notes

This note summarizes the current HierMoE engineering state for
`Qwen3-VL-30B-A3B-Instruct` on the NPU `fused_npu` MoE path. The code snapshot
used for the latest 32-card checks is `64ec3e1`.

## Scope

The measured path is FSDP2 + EP with complete experts assigned by the EP
dimension (`ep_fsdp_size=1` in the target runs). `hiermoe.enable=false` keeps the
original fused MoE route unchanged; the HierMoE dispatch/combine and Expert Swap
logic is entered only when `hiermoe.enable=true`.

The 32-card results below were generated on the back-four nodes with:

- `NNODES=4`, `NPROC_PER_NODE=8`, `EP_SIZE=32`
- `MICRO_BATCH_SIZE=4`, `GLOBAL_BATCH_SIZE=128`, `MAX_SEQ_LEN=4096`
- `MAX_STEPS=6`
- `MOE_IMPL=fused_npu`, `ATTN_IMPL=flash_attention_2`
- event timing enabled through the existing MoE/full-timing profile flow

## Current 32-card Results

`train_step_total_ms`, `all_to_all_ms`, and `expert_compute_ms` come from the
generated `summary.json` files. Speedup is computed against the 32-card baseline
run `qwen3vl_baseline_hotpath_ep32_mb4_gbs128_6step_new4_20260628_113349`.

| run | mode | train step total | E2E speedup | A2A total | A2A speedup | loss |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qwen3vl_baseline_hotpath_ep32_mb4_gbs128_6step_new4_20260628_113349` | original fused MoE | 126.088 s | 1.000x | 98.066 s | 1.00x | not plotted |
| `qwen3vl_hiermoe_deduponly_ep32_mb4_gbs128_new4_6step_20260628_225006` | token dedup only, no Expert Swap | 74.848 s | 1.685x | 32.949 s | 2.98x | 1.44 -> 0.85 |
| `qwen3vl_hiermoe_pairs4_restorepatch32_ep32_mb4_gbs128_new4_6step_20260628_220836` | token dedup + Expert Swap, 4 pairs/layer | 70.728 s | 1.783x | 27.234 s | 3.60x | 1.44 -> 0.85 |
| `qwen3vl_hiermoe_swap_pairs_ep32_mb4_gbs128_new4_pairs4_6step_20260628_154421` | best pair sweep result | 70.403 s | 1.791x | see sweep CSV | see sweep CSV | 1.44 -> 0.85 |

The restored current implementation is within about 0.5% of the best recorded
32-card run. The accepted target for the 32-card path is therefore reached:
about `1.79x` end-to-end speedup.

Event-based MoE logical-section averages show that expert compute itself is not
materially changed:

| run | dispatch avg/call | combine avg/call | expert compute avg/call | Expert Swap avg/profile record |
| --- | ---: | ---: | ---: | ---: |
| baseline | 78.61 ms | 91.65 ms | 4.52 ms | disabled path, ~0 ms |
| dedup-only | 13.47 ms | 19.06 ms | 4.54 ms | disabled path, ~0 ms |
| pair=4 restore | 12.10 ms | 16.72 ms | 4.54 ms | 366.68 ms |

Notes:

- The logical-section call counts differ between original all-to-all and
  HierMoE because HierMoE records staged dispatch/combine spans. Use
  `summary.json` for end-to-end and A2A totals, and use avg/call values for
  per-path timing diagnostics.
- The dedup-only run demonstrates that hierarchical token dedup accounts for
  most of the speedup: `1.685x` E2E by itself.
- Expert Swap improves the 32-card result from `1.685x` to about `1.78-1.79x`
  by improving placement/load balance enough to reduce dispatch/combine time,
  while its own step-boundary overhead remains small relative to the step.

## Expert Swap Pair-count Sweep

The pair-count sweep used
`train.hiermoe.expert_swap_max_pairs_per_layer={1,2,4,16,32}` with the same
32-card 6-step setup. Results are in:

- `profile/tables/pretrain/hiermoe_swap_pair_sweep_20260628_154421/hiermoe_expert_swap_pair_sweep.csv`
- `profile/figures/pretrain/hiermoe_swap_pair_sweep_20260628_154421/hiermoe_expert_swap_pair_sweep.png`

| max pairs/layer | train step total | E2E speedup | Expert Swap avg/record | dispatch avg/call | combine avg/call | communication max/mean max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 70.888 s | 1.779x | 339.26 ms | 12.10 ms | 16.84 ms | 1.0485 |
| 2 | 70.897 s | 1.778x | 366.21 ms | 12.16 ms | 16.82 ms | 1.0333 |
| 4 | 70.403 s | 1.791x | 365.38 ms | 12.08 ms | 16.80 ms | 1.0352 |
| 16 | 70.829 s | 1.780x | 367.37 ms | 12.06 ms | 16.82 ms | 1.0298 |
| 32 | 70.580 s | 1.786x | 371.83 ms | 12.11 ms | 16.83 ms | 1.0333 |

Conclusion: `4` pairs/layer is the best observed point. More pairs reduce the
worst communication imbalance slightly, but the E2E gain saturates and can
become noise-level. The default remains `1` for conservative behavior, with the
script/environment knob available for sweeps.

## 64-card Historical Result

The available 64-card profile data is older than the current P2P/batched Expert
Swap implementation and should be treated as bottleneck evidence, not as the
final current 64-card result. It was generated with `EP_SIZE=64`,
`GLOBAL_BATCH_SIZE=256`, `MAX_STEPS=16`.

| run | train step total | E2E speedup | A2A total | A2A speedup | Expert Swap total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3vl_baseline_eventprofile_ep64_mb4_gbs256_16step_all8_20260626_161536` | 627.923 s | 1.000x | 517.048 s | 1.00x | disabled path, ~0 s |
| `qwen3vl_hiermoe_eventprofile_ep64_mb4_gbs256_16step_all8_20260626_154116` | 908.864 s | 0.691x | 298.585 s | 1.73x | 429.254 s |

This explains the earlier 8-node slowdown: A2A was improved by `1.73x`, but the
old Expert Swap path added about `30.66 s` per profiled record. With that
overhead, the A2A improvement cannot translate into E2E speedup.

The current code fixes the main Expert Swap exchange problem by using grouped
P2P sends/receives and packing parameter plus optimizer-state slots by peer,
device, and dtype. That fix has been validated on 32 cards, but the current
64-card path has not been rerun because the front-four nodes were occupied.

## Implemented Optimizations

### Hierarchical token dedup / HierD-AlltoAll

- Added `train.hiermoe.*` config and CLI override support.
- Added `HIERMOE_DEDUP_ONLY=1` in the NPU profile launcher to run default
  expert placement with token dedup and no Expert Swap.
- Builds HierMoE process groups inside the EP communicator only.
- Uses 2D hierarchy for the current 32-card production path.
- Deduplicates tokens at hierarchy group boundaries so a token that targets
  multiple experts in the same group is sent once for that stage.
- Packs metadata and routing weights into a paired payload and uses
  `all_to_all_pair` to avoid separate metadata/weight all-to-all launches.
- Starts split-size exchange early and waits only when the split sizes are
  needed.
- Keeps expert compute on the existing fused NPU backend.
- Keeps combine accumulation in FP32 and uses the NPU-aware dim-0 index-add path
  where available.

### Expert Swap

- Maintains `logical_expert_id -> physical placement` per MoE layer.
- Runs swap only at the step boundary after optimizer update, not in the middle
  of forward/backward.
- Swaps real expert parameter slots and matching optimizer state slots.
- Batches remote swaps by peer, device, and dtype, using P2P instead of EP-wide
  all-to-all for expert migration.
- Handles local same-rank swaps without communication.
- Supports `train.hiermoe.expert_swap_interval`.
- Supports `train.hiermoe.expert_swap_max_pairs_per_layer`; when greater than
  `1`, selects multiple non-overlapping expert pairs per layer.
- Saves and reloads placement/permutation state through the HierMoE state path.
- Fails fast rather than pretending to support unsafe partial-expert migration
  when the expert parameter layout is not the complete-local-expert case.

### Profiling

- Uses event timing for dispatch, combine, expert compute, and Expert Swap.
- Normalizes baseline and HierMoE MoE timing into logical sections:
  `dispatch`, `combine`, `expert_compute`, `expert_swap`.
- Keeps event elapsed reads at flush/summarization points, avoiding hot-path
  synchronization.
- Adds sweep plotting for Expert Swap pair count vs overhead and communication
  imbalance.

## Optimizations Tested and Not Retained

- A replacement for `_candidate_pair_token_counts` based on token-pair keys and
  `bincount` was tested. NPU microbenchmarks showed only noise-level gains
  around `1.000-1.007x`, so it was reverted.
- A split-size overlap variant was tested earlier and produced a slower formal
  6-step run, so it was reverted.
- Direct replacement of combine with `torch_npu.npu_moe_finalize_routing` was
  not retained: it has no usable autograd path in the current environment, and
  BF16 accumulation would change the current FP32 accumulation semantics.

## Current Recommendation

For the verified 32-card setup, use:

```bash
export HIERMOE_ENABLE=true
export HIERMOE_TOKEN_DEDUP=true
export HIERMOE_EXPERT_SWAP=true
export HIERMOE_EXPERT_SWAP_INTERVAL=1
export HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=4
```

If a conservative run is needed, keep `HIERMOE_EXPERT_SWAP_MAX_PAIRS_PER_LAYER=1`.
The 32-card difference between `1` and `4` pairs/layer is small, while `4` is
the best observed point.

No additional optimization should be pursued unless a new profile shows more
than about 5% recoverable overhead in the current code. The next missing
verification, if resources become available, is a fresh 64-card short run with
the current P2P/batched Expert Swap implementation.
