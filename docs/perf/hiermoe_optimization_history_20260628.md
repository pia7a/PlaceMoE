# HierMoE Optimization History

This document summarizes the recorded HierMoE optimization history for the
Qwen3-VL-30B-A3B NPU profile work. The machine-readable ledger is:

`docs/perf/hiermoe_optimization_history_20260628.csv`

The CSV contains one row per run with generated profile summary data, plus the
manual pair-count sweep rows and the early doc-only 4-node smoke comparison.

## Data Sources

- Remote profile tables from rank5 container:
  `/workspace/task3/VeOmni-0.1.11/profile/tables/pretrain/*/summary.json`
- Step timing:
  `step_timing_summary.csv`, especially `hiermoe_expert_swap`
- MoE logical timing:
  `moe_logical_section_summary.csv`, especially dispatch/combine
- Pair-count sweep:
  `profile/tables/pretrain/hiermoe_swap_pair_sweep_20260628_154421/hiermoe_expert_swap_pair_sweep.csv`
- Early smoke result:
  `docs/perf/hiermoe_qwen3_vl.md`

The ledger intentionally includes diagnostics and failed/incomplete runs. Use the
`status` column to distinguish `kept-best`, `kept-analysis`, `candidate`,
`diagnostic`, `not-retained`, and `incomplete`.

## CSV Columns

| column | meaning |
| --- | --- |
| `run_name` | profile run directory or sweep run name |
| `scope` | comparison scope, e.g. `ep32_new4`, `ep64_all8`, or legacy doc-only |
| `optimization_point` | inferred engineering change or experiment represented by the run |
| `status` | whether the run is a baseline, retained result, diagnostic, candidate, or incomplete |
| `baseline_run` | baseline used for speedup calculation |
| `train_step_total_ms` | sampled training step total from `summary.json` or doc-only note |
| `e2e_speedup_vs_baseline` | baseline train step total divided by this run's train step total |
| `all_to_all_ms` | sampled dispatch+combine all-to-all total when available |
| `all_to_all_speedup_vs_baseline` | baseline A2A total divided by this run's A2A total |
| `expert_swap_cuda_ms_sum` | sampled Expert Swap event total |
| `expert_swap_cuda_ms_avg_per_record` | Expert Swap event total divided by profile records |
| `dispatch/combine_*` | event-timing dispatch/combine breakdown from normalized MoE summaries |
| `source` | `summary_json`, `pair_sweep_csv`, or `docs_perf_hiermoe_qwen3_vl_md` |

## Milestones

| stage | representative run | E2E speedup | A2A speedup | Expert Swap overhead | note |
| --- | --- | ---: | ---: | ---: | --- |
| Early 4-node smoke | `qwen3vl_hiermoe_currentlog_6step_20260626_033702` | 1.615x | 3.371x | not recorded | Doc-only result before the later event-summary ledger. |
| 64-card old full path | `qwen3vl_hiermoe_eventprofile_ep64_mb4_gbs256_16step_all8_20260626_154116` | 0.691x | 1.732x | 429.254 s total | A2A improved, but old Expert Swap dominated E2E time. |
| 32-card baseline | `qwen3vl_baseline_hotpath_ep32_mb4_gbs128_6step_new4_20260628_113349` | 1.000x | 1.000x | ~0 | Current 32-card comparison baseline. |
| Dedup only | `qwen3vl_hiermoe_deduponly_ep32_mb4_gbs128_new4_6step_20260628_225006` | 1.685x | 2.976x | ~0 | Isolates HierD-AlltoAll token dedup with default expert placement. |
| Pair sweep best | `qwen3vl_hiermoe_swap_pairs_ep32_mb4_gbs128_new4_pairs4_6step_20260628_154421` | 1.791x | see sweep rows | 1.462 s total | Best recorded 32-card setting: 4 non-overlapping expert pairs/layer. |
| Restored current code | `qwen3vl_hiermoe_pairs4_restorepatch32_ep32_mb4_gbs128_new4_6step_20260628_220836` | 1.783x | 3.602x | 1.467 s total | Current restored implementation, within about 0.5% of best. |

## What Mattered Most

### 1. Hierarchical token dedup was the largest verified gain

The cleanest A/B is baseline vs dedup-only:

| run | train step | E2E speedup | A2A total | A2A speedup |
| --- | ---: | ---: | ---: | ---: |
| baseline | 126.088 s | 1.000x | 98.066 s | 1.000x |
| dedup-only | 74.848 s | 1.685x | 32.949 s | 2.976x |

This shows that the main win is the system-level HierD-AlltoAll token dedup:
duplicate-free token movement reduces the A2A payload enough to move E2E step
time by about 1.68x before Expert Swap.

### 2. Expert Swap was initially the main 64-card blocker

The old 64-card run had better A2A but worse E2E:

| run | train step | E2E speedup | A2A total | A2A speedup | Expert Swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64-card baseline | 627.923 s | 1.000x | 517.048 s | 1.000x | ~0 |
| old 64-card HierMoE | 908.864 s | 0.691x | 298.585 s | 1.732x | 429.254 s |

This is why Expert Swap was reworked to real parameter/optimizer-state
migration with peer-grouped P2P, rather than broad EP all-to-all style exchange.
The current P2P/batched swap path has been validated in the 32-card run, but a
fresh 64-card validation is still not in the ledger.

### 3. Pair count helped, but saturated quickly

The 32-card pair sweep shows that increasing the number of swapped pairs is not
monotonic:

| max pairs/layer | train step | E2E speedup | Expert Swap avg/record | dispatch avg/call | combine avg/call |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 70.888 s | 1.779x | 339.26 ms | 12.10 ms | 16.84 ms |
| 2 | 70.897 s | 1.778x | 366.21 ms | 12.16 ms | 16.82 ms |
| 4 | 70.403 s | 1.791x | 365.38 ms | 12.08 ms | 16.80 ms |
| 16 | 70.829 s | 1.780x | 367.37 ms | 12.06 ms | 16.82 ms |
| 32 | 70.580 s | 1.786x | 371.83 ms | 12.11 ms | 16.83 ms |

The best observed point is 4 pairs/layer. More pairs slightly improve some load
balance metrics, but the E2E benefit is within noise and Expert Swap overhead
does not decrease.

### 4. Hot-path compression turned the idea into a stable speedup

The intermediate `hotpath` and selector runs show that many engineering details
were needed after the algorithmic dedup path worked:

- metadata and routing weights were packed into paired payloads;
- split-size exchange was started early and waited on only when needed;
- dispatch/combine paths reduced small collective launches;
- combine accumulation stayed FP32 while using the NPU-aware index-add path;
- Expert Swap exchange was batched by peer/device/dtype with P2P;
- candidate selector and instrumentation paths were profiled separately.

The retained 32-card result should be read as the combination of HierD-AlltoAll
token dedup plus these hot-path reductions. The CSV includes several candidate
runs around 1.65x-1.77x; those are useful for forensics, but not all were kept.

## Notable Non-retained or Diagnostic Runs

- `qwen3vl_hiermoe_eventprofile_ep64_mb4_gbs256_16step_all8_20260626_154116`:
  useful for root cause analysis, not retained as a final implementation
  result because Expert Swap cost dominated.
- `qwen3vl_hiermoe_g4_ep32_mb4_gbs128_6step_new4_20260628_015341`:
  hierarchy/group-size experiment, slower than the retained path.
- `qwen3vl_hiermoe_global2d_ep32_mb4_gbs128_6step_new4_20260627_175926`:
  selector-side experiment, slower due to high Expert Swap selection/exchange
  overhead.
- `qwen3vl_hiermoe_hotpath_ep32_mb4_gbs128_6step_new4_20260627_203814`:
  a fast candidate outlier in the ledger. It is kept for traceability but not
  treated as a final benchmark because the result was not the retained/repeated
  implementation.
- Internal 4-step runs have high mechanical speedups because they are shorter
  diagnostics and are not directly comparable to 6-step baseline rows.

## Reading the Ledger

For final claims, prefer rows with:

- `status=baseline`
- `status=kept-analysis`
- `status=kept-best`

Use `candidate`, `diagnostic`, and `not-retained` rows to understand engineering
direction, not as final benchmark numbers.

The most important retained engineering measures are:

1. Hierarchical token dedup: baseline to dedup-only, `1.685x` E2E.
2. P2P/batched Expert Swap and placement: dedup-only to pair=4, roughly
   `1.06x` additional relative improvement.
3. Hot-path launch/count reductions: necessary to keep HierMoE overhead small
   enough for the communication savings to appear in E2E step time.
