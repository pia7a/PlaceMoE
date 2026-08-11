# PlaceMoE pre-training

PlaceMoE uses one validated configuration file for static placement and
training-time updates. The hardware-neutral runtime interface is
`VEOMNI_PLACEMOE_CONFIG`; use it with the normal VeOmni training launcher on
either GPU or NPU:

```bash
export VEOMNI_PLACEMOE_CONFIG=$(realpath configs/placemoe/qwen3vl_ep32_hot.yaml)
# Start the regular tasks/train_vlm.py command for the selected accelerator.
```

The runtime validates the configuration and initial `L,M` artifact before
installing the placement. Paths are resolved relative to the YAML file, so the
same training command does not need hardware-specific PlaceMoE options.
`scripts/placemoe/reproduction/npu_ep32.sh` launches the original NPU paper
testbed, while `scripts/placemoe/reproduction/gpu_ep32/matrix.sh` launches the
GPU EP32 testbed. Both pass the same `VEOMNI_PLACEMOE_CONFIG` runtime interface.
Their platform-specific variables configure deployment, devices, networking,
and the training job; they do not duplicate PlaceMoE placement, calibration,
or update settings.

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
```

Paths are resolved relative to the configuration file. Calibration artifacts
must have status `accepted`; their coefficients are passed unchanged to every
planner invocation. `runtime_perf_model` supplies the profiled A2A and state
transfer costs used by the training runtime, so production jobs do not depend
on a hidden paper-runner default. `failure_policy: continue` keeps the current
`L,M` if an asynchronous planner job fails, while `raise` stops training.

By default, PlaceMoE automatically reserves at most 25% of the physical CPU
cores visible to the training process for the asynchronous planner. The
selection is balanced across NUMA nodes, keeps SMT siblings together, and
bounds planner parallelism to the reserved physical cores. Set both
`resources.planner_cpu_ids` and `resources.training_cpu_ids` only when an
explicit affinity split is required.

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

## Planner and runtime interfaces

`scripts/profile/plan_placemoe.py` is the canonical planner CLI. Its
implementation is `scripts/profile/placemoe_planner.py`. The historical
`build_hiermoe_recursive_classifier_layout.py` name is a deprecated wrapper.
PlaceMoE runtime configuration is accepted only through
`VEOMNI_PLACEMOE_CONFIG`. The remaining `VEOMNI_HIERMOE_*` settings control
the underlying hierarchical communication runtime and explicit baselines.

The legacy expert swap/cover methods remain available as explicit baselines,
but the canonical planner does not import them. Runtime metrics use only the
`placemoe/` prefix.

## GPU EP32 reproduction contract

This document describes the GPU contract and the formal Qwen3-VL-30B-A3B
experiment matrix. Calibration, layout construction, formal timing, and
profiling are separate phases. Calibration and profiler runs must never be
included in reported end-to-end timing.

## Fixed platform contract

- Four nodes and eight NVIDIA RTX A6000 GPUs per node.
- Torch 2.9.1+cu129, CUDA 12.9, NCCL 2.27.5, and Triton 3.5.1.
- NCCL over the configured interface (`ibs0` for the validated testbed), with
  EP size 32 and hierarchy group sizes `2,8,32`.
- Qwen3-VL-30B-A3B-Instruct: 48 MoE layers, 128 experts, top-8 routing,
  hidden size 2048, and `fused_triton` MoE kernels.
- Micro batch size 4, global batch size 128, maximum sequence length 4096,
  BF16, and approximately 16K input tokens per rank.
- ShareGPT4V uses the vision tower. Tulu-3 freezes the unused vision tower.

`PLACEMOE_REPRO_PYTHON` defaults to the repository `.venv/bin/python`. The
committed configuration template declares the expected software contract,
while a local ignored configuration supplies cluster paths and addresses. Environment
variables may override individual values. This GPU reproduction keeps the
paper model, batch shape, datasets, and comparison methods; the configured CUDA
software stack is the validated GPU platform, not a claim about paper hardware.

The node preflight rejects a different accelerator, software version, model
type, missing dataset, missing NCCL support, or missing configured network
interface. It checks that all four nodes report one software/hardware scope and
records host, NIC, and GPU PCI identities. Communication calibration is bound
to that preflight report, the exact communication source, and a maximum 10%
held-out MAPE.

## Compared methods

| Matrix name | Runtime | Redundancy | Gradient synchronization |
| --- | --- | --- | --- |
| `baseline` | Native VeOmni all-to-all | none | not applicable |
| `r2` | hierarchical deduplicated runtime | two fixed copies / two EP16 groups | blocking |
| `eplb` | hierarchical deduplicated runtime | `B=E=128`, official EPLB placement | blocking |
| `ours` | hierarchical deduplicated runtime | `B=E=128`, PlaceMoE placement and mapping | hidden |

The default `PLACEMOE_REPRO_GRAD_PROTOCOL=paper` matches the paper contract
above. Set `PLACEMOE_REPRO_GRAD_PROTOCOL=blocking` to run an additional
algorithm-only comparison where Replica, EPLB, and PlaceMoE all use blocking
gradient synchronization.

For EPLB and PlaceMoE, a fresh four-step route profile is collected for each
dataset. PlaceMoE optimizes steps 0-2 and selects the candidate using held-out
step 3. EPLB uses steps 0-3. Cost artifacts are accepted only when accelerator,
topology, checkpoint, dataset hash, batch shape, MoE implementation, and
communication calibration provenance match exactly. Reused communication and
cost artifacts are revalidated against the current four-node preflight hash
and the current communication-source fingerprint before route capture starts.
For the static GPU PlaceMoE case, the matrix materializes a minimal canonical
configuration next to the layout artifact with hot updates disabled. EPLB uses
the generic HierMoE initial-layout input; the two methods do not share a
PlaceMoE-specific compatibility variable.

## Running the matrix

Copy the committed template to the ignored local configuration and fill in all
paths, hosts, proxy ports, network interface, device order, and expected
software versions:

```bash
cp configs/placemoe/gpu_ep32.env.example configs/placemoe/gpu_ep32.env
$EDITOR configs/placemoe/gpu_ep32.env

# Keep credentials outside the configuration file.
export PLACEMOE_REPRO_SSH_KEY=/absolute/path/to/key
# Password-based testbeds may instead set PLACEMOE_REPRO_SSH_PASSWORD.

# Validate configuration and print the matrix without launching training.
bash scripts/placemoe/reproduction/gpu_ep32/matrix.sh \
  --config configs/placemoe/gpu_ep32.env dry-run

# Run the eight fixed-order formal cases.
PLACEMOE_REPRO_CONFIRM_FULL=1 PLACEMOE_REPRO_RUN_TAG=qwen3vl_ep32_formal \
  bash scripts/placemoe/reproduction/gpu_ep32/matrix.sh \
  --config configs/placemoe/gpu_ep32.env full

# Secondary placement-only comparison with blocking gradient synchronization.
PLACEMOE_REPRO_CONFIRM_FULL=1 PLACEMOE_REPRO_GRAD_PROTOCOL=blocking \
  PLACEMOE_REPRO_RUN_TAG=qwen3vl_ep32_blocking \
  bash scripts/placemoe/reproduction/gpu_ep32/matrix.sh \
  --config configs/placemoe/gpu_ep32.env full
```

The matrix also accepts `PLACEMOE_REPRO_CONFIG=/path/to/config.env`;
`--config` is the documented entry point and the historical positional mode
remains compatible.
It runs source synchronization and four-node preflight itself. Run preflight
separately only for an early connectivity check. All helper scripts load the
same configuration, including `PLACEMOE_REPRO_REMOTE_REPO_ROOT`.

Source synchronization sends only `veomni/`, `tasks/`, `configs/`,
`scripts/`, `pyproject.toml`, and `uv.lock`. It excludes the ignored local
configuration, credentials, datasets, results, tests, Git metadata, virtual
environments, and unrelated workspace files. Set
`PLACEMOE_REPRO_SYNC_SOURCE=0` only if the exact source is already present
remotely.

By default, each protocol executes two datasets and four methods once: 8
training runs (16 for both protocols). Each run has 5 optimizer steps. Steps
1-2 are warmup and steps 3-5 form the run mean. Methods always execute in the
declared fixed order (baseline, R2, EPLB, PlaceMoE); there is no cyclic
rotation. `PLACEMOE_REPRO_REPEATS` may explicitly request independent repeats
without changing that order. The aggregate JSON retains every run mean and its
cross-run standard deviation. Every single-run summary records its repeat
index, execution index, and fixed-order policy. The E2E metric directly reads
synchronized critical-rank `step_time_s`.
Full/Torch profilers, MoE CUDA-event timing, individual spans, and router
monitor hooks are disabled for formal runs.

After formal timing, collect a separate representative profile. The layout
argument is required for EPLB or PlaceMoE.

```bash
bash scripts/placemoe/reproduction/gpu_ep32/profile_representative.sh \
  ours sharegpt4v /absolute/path/to/ours_layout.json
```

## Artifacts

The `gpu32_*` prefixes below are immutable identifiers from the completed GPU
experiment and remain unchanged so archived result references and readers stay
stable. They are not public script or configuration names.

- `results/gpu32_ep32_a6000_communication_<run-tag>.json`: preflight/source-
  bound EP32 NCCL calibration with held-out validation.
- `results/gpu32_*_cost_model.json`: model/dataset-scoped compute and cost fit.
- `route_captures/gpu32_profile_*`: fresh four-step routing observations.
- `results/gpu32_*_{eplb,ours}_{layout,report}.json`: layouts and validation.
- `results/gpu32_*_layout_bundle.json`: schema-v2 binding of layouts, reports,
  route manifest, cost model, experiment identity, and planner/EPLB sources.
- `results/gpu32_*_aggregate.json`: one or more explicitly configured run
  summaries (one by default), with input hashes, execution order, and complete
  artifact provenance.
- `results/gpu32_ep32_*_speedup_vs_veomni_*`: JSON, CSV, and SVG speedups;
  JSON/CSV retain preflight, communication, cost, source, bundle, layout, report,
  and aggregate-summary hashes.
- `profile/runs/pretrain/gpu32_profile_*`: separate detailed profiler output.

Memory fields use `peak_accelerator_allocated_gib` and
`peak_accelerator_reserved_gib` on both GPU and NPU.

## Validation and reporting gate

```bash
source .venv/bin/activate
make quality
pytest -q tests/distributed/test_placemoe_ep32.py
bash -n scripts/placemoe/reproduction/gpu_ep32/*.sh
```

Run CUDA/NCCL tests on the target allocation; local unit tests do not replace
the four-node preflight. Formal speedups must not be reported until the current
preflight, fresh EP32 communication calibration, scoped cost model, fresh route
capture, all eight default paper-protocol cases, summary validation, and chart
generation succeed. If the blocking placement-only control is requested,
report its additional eight cases separately. Profiler runs remain separate
from both timing matrices.
