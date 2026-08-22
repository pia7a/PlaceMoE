# PlaceMoE Refactor Validation

This document records the focused evidence used to validate the PlaceMoE
refactor. It is intentionally not a reproduction of the paper's full
experimental matrix. The goal is to establish that the canonical optimizer
implements the paper flow, produces valid runtime artifacts, improves the
same-runtime behavior over uniform replication, and supports a live update on
all 64 NPUs.

## Scope

- Branch: `refactor/placemoe-cleanup`
- Historical validation container: `tzq_npu_coremoe_verify_20260717`
- Historical distributed testbed: 8 nodes, 8 NPUs per node, EP64
- Common runtime: hierarchical token-deduplicated A2A with hidden
  replica-gradient synchronization
- Canonical artifact: schema version 2 with
  `source.algorithm=placemoe-v1`

The static A/B runs change only the preloaded physical layout `L` and
source-aware mapping `M`. They use the same model, data, runtime, batch and
sequence configuration, and gradient-synchronization mode.

## Production interface regression on 4 nodes

The production cleanup was revalidated on the 4 available nodes
`huawei1_node1`, `huawei1_node2`, `huawei2_node1`, and `huawei2_node2`, with 8
NPUs per node and EP32. The 20-step Qwen3-VL/ShareGPT4V run used
`qwen3vl_ep32_hot_smoke.yaml`, including the canonical initial artifact,
runtime performance model, accepted calibration coefficients, and independent
layout and mapping intervals. All 4 node launchers returned exit code 0.

| Event | Source / apply step | Planner wall time | Installation | Result |
| --- | ---: | ---: | ---: | --- |
| Mapping-only refresh | 1 / 3 | 8.95 s | 0.16 s | `M` updated without moving slots |
| Full layout and mapping refresh | 4 / 18 | 178.43 s | 7.97 s | 11,989 slots moved across 48 layers |

The full planner ran 48 layer jobs concurrently. Its mean per-layer time was
161.78 s, close to the 178.43-s wall time rather than their sum. Exact
held-out evaluation predicted a 1.113x joint-cost improvement over the current
pair. Training continued while both planners ran; after the full update, steps
19 and 20 took 14.49 s and 13.65 s with finite losses of 0.839 and 0.838. This
run verifies canonical configuration loading before model sharding, explicit
calibration propagation, independent refresh scheduling, asynchronous
planning, mapping-only installation, full state migration, and continued
training after the atomic update.

Run name: `placemoe_production_ep32_qwen3vl_hot_smoke_20260803_v4`.

## CPU regression and artifact checks

The focused Python 3.11 regression suite covers routing statistics, replica
allocation, capacity-constrained placement, mapping refinement, alternation,
materialization, schema validation, CLI integration, and runtime hot-update
validation. The final focused run has 78 passing tests and the same
single pre-existing failure recorded before the refactor:

```text
78 passed, 1 failed
```

The failure is
`test_fused_path_skips_eager_scoring_reduces_once_and_reuses_physical_routes`
in `tests/distributed/test_core_moe_planner.py`; its reducer receives shape
`(2, 2)` instead of the test's expected `(2, 7)`. PlaceMoE did not introduce
additional failures. The final hot-replan layer-key fix separately passes 33
targeted tests, and all modified Python files pass Ruff.

Both retained EP64 artifacts pass schema, capacity, owner, `L`, and `M`
validation:

| Workload | Layout SHA-256 | Held-out exact-cost result |
| --- | --- | --- |
| Qwen3.5-20L / Tulu3 | `9b5f0d43db3a4502e03737f9fa745f7450d372f9ea8f7b5d3573219df733bdac` | 4021.023 ms versus 4162.132 ms for mirrored R2, or 1.0351x |
| DeepSeek-V3 6-MoE-layer / Tulu3 | `b5b417529dd164b4dba0ab7c4143002effccc0b27e6a47cbbf9730acd4f3e762` | 3827.923 ms versus 3856.713 ms for mirrored R2, or 1.0075x |

The exact route evaluator, rather than the partitioning surrogate, selects
the final pair.

## EP64 static A/B

The short static comparison uses Qwen3.5-20L on Tulu3, EP64, learning rate 0,
6 training steps, and steps 4--6 as the measurement window. Both variants use
the same hierarchical runtime and hidden replica-gradient synchronization.

| Metric | Mirrored R2 | PlaceMoE | R2 / PlaceMoE |
| --- | ---: | ---: | ---: |
| Forward + backward A2A | 4668.582 ms | 4265.196 ms | 1.0946x |
| Expert compute | 311.454 ms | 293.774 ms | 1.0602x |
| End-to-end step | 70822.455 ms | 70282.807 ms | 1.0077x |
| Dispatch deduplication ratio | 0.5485 | 0.5697 | -- |

All 64 ranks completed successfully. The three-sample window is a functional
and directional gate, not a paper-grade performance claim. It nevertheless
shows that the refactored canonical `L,M` improves communication, expert
compute, and end-to-end time on the common runtime.

Run names:

- `placemoe_refactor_ep64_qwen35_r2_static_ab`
- `placemoe_refactor_ep64_qwen35_placemoe_static_ab`

## EP64 hot-update smoke test

The hot-update test uses Qwen3.5-20L on Tulu3 and starts from the validated
static PlaceMoE artifact. Step 1 captures recent routes and launches the
canonical CLI in a separate CPU process. The generated schema-v2 artifact
preserves the exact 20 runtime layer keys and is applied at step 4.

| Event | Result |
| --- | ---: |
| Route snapshot | 1923.95 ms |
| Asynchronous CPU planner | 86242.43 ms |
| Apply step / source step | 4 / 1 |
| Staleness | 3 steps |
| Migrated physical slots across 20 layers | 6616 |
| Blocking state migration and installation | 4731.51 ms |
| Completed post-update steps | 5--10 |

Every node returned exit code 0. The run proves that recent token routes can
drive the canonical optimizer, that the resulting `L,M` artifact can be
broadcast and validated, and that expert parameters and optimizer states can
be migrated and used without restarting training. It also exposed and fixed
a model-generic correctness issue: periodic planning now serializes the exact
registered runtime layer keys instead of assuming a Qwen-specific template.

Run name: `placemoe_refactor_ep64_qwen35_hot_smoke`.

## Interpretation

The focused evidence satisfies the refactor acceptance gate:

1. the reusable APIs implement the paper's statistics, allocation, placement,
   mapping, alternation, exact selection, and artifact validation flow;
2. fixed replication and historical swap/cover planners remain isolated
   baselines rather than hidden dependencies of the canonical optimizer;
3. the same-runtime EP64 A/B is directionally positive; and
4. a real 20-layer EP64 hot update completes and training continues.

The results do not replace the paper's longer steady-state experiments and
must not be quoted as statistically robust speedups.

## Production bridge validation on 2 nodes

The production-portability refactor was additionally checked on 2 Ascend
nodes with 8 NPUs per node. This gate tests integration contracts rather than
reproducing the paper's performance matrix: both models retain a single MoE
layer to keep the distributed run short, while still exercising real EP16
dispatch, expert execution, replica-gradient synchronization, and optimizer
state handling.

Before the distributed runs, the current canonical planner was replayed on
the historical 48-layer Qwen3-VL/ShareGPT4V EP32 route snapshots. The generated
`layers`, `replay`, `schema_version`, and `topology` payload has the same
semantic SHA-256 (`41a8059d...653db6ea`) as the historical artifact. The exact
held-out cost is also unchanged at 5556.006 ms, versus 7271.245 ms for uniform
replication. This establishes planner-output and cost equivalence independently
of the runtime smoke tests.

| Run | Runtime behavior | Steady-state evidence |
| --- | --- | --- |
| Qwen3-VL / ShareGPT4V | One mapping-only refresh was submitted at step 2 and atomically applied at step 12 without moving expert states. | Steps 13--30 average 2.792 s; final loss and gradient norm are finite. Every step launches a replica-gradient overlap job, hiding 99.56% of its raw synchronization time on average. |
| DeepSeek-V3 / Tulu-3 | A preloaded replicated layout runs with learning rate 0 through the generic fused-expert adapter. | Steps 2--6 average 1.051 s; final loss and gradient norm are finite. Every step launches an overlap job, hiding 99.49% on average. |

The Qwen refresh explicitly passes the current calibration coefficients to the
asynchronous canonical planner. The DeepSeek run uses the same model-independent
adapter resolution as Qwen: modules are selected by supported expert parameter
forms, not model names. Both 16-rank jobs completed on every rank without a
blocking gradient-synchronization fallback.

The validation also exposed a VeOmni configuration round-trip issue: serialized
configurations already contain the automatically injected `ep` dimension.
Reloading them used to append a duplicate dimension and create duplicate
optimizer parameter groups. `AcceleratorConfig` now accepts one matching
serialized entry, canonicalizes identical duplicates from older configurations,
and rejects conflicting entries.
