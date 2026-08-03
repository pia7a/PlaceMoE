# PlaceMoE pre-training

PlaceMoE uses one validated configuration file for static placement and
training-time updates. The production entry point is:

```bash
bash scripts/placemoe/pretrain.sh \
  configs/placemoe/qwen3vl_ep32_hot.yaml \
  qwen3vl sharegpt4v full
```

The launcher validates the configuration and initial `L,M` artifact before
starting distributed training. By default, it synchronizes the source tree,
deploys only the referenced initial artifact, and validates both inside every
training container. Set `PLACEMOE_SYNC_SOURCE=0` or
`PLACEMOE_SYNC_ARTIFACT=0` only when an external deployment system has already
distributed the exact files. The current EP32 launcher uses only the four
online nodes `huawei1_node1`, `huawei1_node2`, `huawei2_node1`, and
`huawei2_node2`. Override `PAPER32_RANK{0,1,2,3}_HOST` only when the cluster
allocation changes.

## Configuration

```yaml
placemoe:
  initial_artifact: ../../results/placemoe_layout.json
  runtime_perf_model: ../../../../hiermoe_perf_model_c009_ep32_20260720/v2/hiermoe_perf_model.json
  calibration:
    artifact: calibration/qwen3vl_ep32_huawei2.json
  hot_update:
    enabled: true
    layout_interval_steps: 100
    mapping_interval_steps: 20
    last_update_step: 500
    work_root: ../../profile/runs/pretrain/placemoe_hot_update
    failure_policy: continue
  resources:
    workers: 48
    candidate_workers: 4
    worker_threads: 1
    planner_cpu_ids: 144-191
    training_cpu_ids: 0-143
```

Paths are resolved relative to the configuration file. Calibration artifacts
must have status `accepted`; their coefficients are passed unchanged to every
planner invocation. `runtime_perf_model` supplies the profiled A2A and state
transfer costs used by the training runtime, so production jobs do not depend
on a hidden paper-runner default. `failure_policy: continue` keeps the current `L,M` if an
asynchronous planner job fails, while `raise` stops training.

The two update intervals are independent:

| Layout interval | Mapping interval | Behavior |
| ---: | ---: | --- |
| `0` | `0` | Static `L,M`; no hot updates. |
| `100` | `0` | Recompute and install both `L` and `M` every 100 steps. |
| `0` | `20` | Keep `L` fixed and refresh only the lookup table `M`. |
| `100` | `20` | Refresh `M` every 20 steps and `L,M` every 100 steps. |

When both events are due at the same step, the full `L,M` update subsumes the
mapping-only update. At most one CPU planner runs at a time; later events are
coalesced and processed after the active job completes. Training continues
with the current pair until a validated schema-v2 artifact is ready. A full
update migrates expert parameters and optimizer states at a step boundary and
then atomically installs `L` and `M`; a mapping-only update replaces only the
dispatch lookup table.

## Planner and compatibility interfaces

`scripts/profile/plan_placemoe.py` is the canonical planner CLI. Its
implementation is `scripts/profile/placemoe_planner.py`. The historical
`build_hiermoe_recursive_classifier_layout.py` name is a deprecated wrapper,
and the `VEOMNI_HIERMOE_*` variables remain a compatibility adapter for paper
reproduction scripts. New pre-training jobs should set only
`VEOMNI_PLACEMOE_CONFIG` or use `scripts/placemoe/pretrain.sh`.

The legacy expert swap/cover methods remain available as explicit baselines,
but the canonical planner does not import them. Runtime metrics use the
`placemoe/` prefix; historical `hiermoe/periodic_*` aliases are retained so
existing dashboards continue to work.
