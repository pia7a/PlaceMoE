# PlaceMoE Code Map

This document is the source of truth for the PlaceMoE refactor. It separates
the paper algorithm, the training runtime, retained baselines, and historical
profiling artifacts. The refactor must keep the existing runtime artifact
schema compatible until all callers have migrated.

## Canonical PlaceMoE flow

The paper describes the following data flow for each MoE layer:

1. collect token-level routing snapshots;
2. derive source-conditioned expert demand and co-selection affinity;
3. retain a bounded shortlist of exact-budget replica allocations;
4. build a capacity-feasible node-to-rank physical layout `L`;
5. optimize the source-aware token-to-copy mapping `M`;
6. alternate `L` and `M` and select the pair with the lowest exact route cost;
7. serialize the selected pair for static preload or a hot update.

The reusable implementation now lives in
`veomni/distributed/moe/hiermoe/placemoe/`:

| Module | Paper responsibility |
| --- | --- |
| `statistics.py` | Source-conditioned demand and co-selection affinity. |
| `allocation.py` | Exact-budget bounded replica-allocation shortlist. |
| `partition.py` | Calibrated capacity-constrained affinity partitioning. |
| `placement.py` | Node-to-rank placement, locality matching, and rank repair. |
| `mapping.py` | Demand-ordered initialization and calibrated mapping update. |
| `optimizer.py` | Bounded layout--mapping alternation and exact-cost callback. |
| `materialize.py` | Physical-slot assignment and mapping relocation. |
| `artifacts.py` | Validated schema-v2 runtime artifacts. |

The canonical CLI is `scripts/profile/plan_placemoe.py`. The older
`build_hiermoe_recursive_classifier_layout.py` name remains only as its
compatibility implementation while downstream imports migrate.

By default, the CLI evaluates only PlaceMoE candidates. The historical
four-node structured-degree2 and token-KMeans hyperedge candidates remain
available through `--include-legacy-structured-candidates` and
`--include-legacy-hyperedge-candidates`; they are not part of the paper path.
All successful runs emit the same preloaded schema-v2 artifact for both
static startup and hot updates.

## Runtime modules retained by PlaceMoE

| Module | Responsibility |
| --- | --- |
| `all_to_all.py` | Hierarchical token-deduplicated dispatch and combine. |
| `routing.py` | Duplicate-free and assignment-load accounting. |
| `perf_model.py` | Communication and expert-compute calibration. |
| `state.py` | Trainer integration, checkpoint state, and lifecycle. |
| `expert_swap.py` | Shared layout installation, state migration, gradient synchronization, and hot-update runtime. Periodic full replanning invokes the canonical CLI and accepts only validated schema-v2 artifacts. Historical online swap selectors remain isolated behind explicit runtime configuration. |
| `online_lut_planner.py` | Mapping-only updates over copies already present in `L`. |
| `metrics.py` | Runtime metrics exposed to the trainer. |

## Retained baselines and experimental planners

The following paths are not the canonical PlaceMoE optimizer, but remain
useful for comparisons or diagnostics:

- fixed/uniform expert replication;
- EPLB-generated static layouts;
- online expert swapping and greedy swap/cover planners;
- the HierMoE placement baseline;
- mapping-only online LUT refinement.

They must not be imported by the canonical offline optimizer except through a
documented compatibility or evaluation interface.

The paper launchers expose these paths as explicit methods. They share the
runtime substrate but do not share an optimizer:

| Method | Role | Layout or mapping source |
| --- | --- | --- |
| `ours` | Canonical static PlaceMoE | Validated schema-v2 `L,M` artifact from `plan_placemoe.py`. |
| `ours_full_replan` | Canonical PlaceMoE hot update | Recent routes are passed to `plan_placemoe.py`; the validated artifact is migrated and atomically installed. |
| `ours_online_lut` | Mapping-only diagnostic | Keeps `L` fixed and updates only the runtime LUT. |
| `r2` | Fixed-replication baseline | Uniform mirrored copies in default expert order. |
| `eplb` | Placement baseline | Externally generated static placement on the common runtime. |
| `hiermoe` | Communication-oriented placement baseline | Legacy exact-P1 selector. |

Online swap/cover selectors in `expert_swap.py` are retained for comparison
and recovery only. They are neither imported by `placemoe/` nor enabled by
the canonical CLI.

## Historical and generated files

Generated `.paper32_*_launcher.sh` snapshots were removed from version
control and are now ignored. They remain reproducible outputs of
`launch_hiermoe_greedy_e2e_4node.sh`, not source code.

Historical benchmark, plotting, and diagnostic scripts will only be removed
after a reference search shows that no canonical launcher, test, or document
depends on them. Git history remains the recovery path for deleted artifacts.

## Baseline before refactoring

The repository started from commit `ff5f980` on branch `master`, with a clean
worktree and four local commits beyond `origin/master`. The targeted planning
test baseline, run in the Python 3.11 NPU validation container, is:

```text
58 passed, 1 failed in 47.48s
```

The pre-existing failure is
`test_fused_path_skips_eager_scoring_reduces_once_and_reuses_physical_routes`
in `tests/distributed/test_core_moe_planner.py`: the reducer receives shape
`(2, 2)` while the test expects `(2, 7)`. Refactoring must not introduce
additional failures; this baseline failure is tracked separately from the
PlaceMoE extraction.

## Minimal completion evidence

The refactor is complete only when all of the following hold:

- the canonical API validates replica budget, slot capacity, `L`, and `M`;
- exact route replay selects the final candidate;
- focused CPU tests match or improve the recorded baseline;
- a same-runtime EP64 comparison shows the optimized pair improving the
  measured joint behavior over uniform replication;
- one EP64 hot update installs a new `L` and `M` without restarting training
  or producing invalid loss values.
