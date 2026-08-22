# PlaceMoE pre-training

PlaceMoE integrates through VeOmni's versioned MoE runtime bridge. It uses the
normal model, data, trainer, FSDP2, and distributed-launch interfaces; the only
additional configuration lives under `train.hiermoe.placemoe`.

## 1. Prepare the environment

The validated Ascend stack uses Python 3.11, CANN 9, PyTorch 2.9.0, and
torch-npu 2.9.0.post2. On that stack, install PlaceMoE as the VeOmni extension:

```bash
python -m venv --system-site-packages .venv
uv pip install --python .venv/bin/python --no-deps --no-build-isolation .
source .venv/bin/activate
```

The current aarch64 path requires this validated preinstalled accelerator
stack; it does not provision CANN, PyTorch, or torch-npu into a clean host.
The Dockerfile under `docker/ascend/` follows the same plugin path: the base
image owns the accelerator and general VeOmni dependencies, while uv builds
and installs the current PlaceMoE tree.

```bash
docker build -t placemoe:ascend -f docker/ascend/Dockerfile .
```

To move the validated image to an offline or multi-node cluster, follow
[Packaging and distributing the PlaceMoE Ascend image](placemoe_image_distribution.md).
To start a loaded image with NPU, model, dataset, configuration, and output
mounts, follow [Running PlaceMoE from the validated Ascend
image](placemoe_container_quickstart.md).

## 2. Calibrate a new cluster topology

Run the communication calibrator once with the same node and rank topology as
training. For example, launch the following command on both nodes, changing
only `NODE_RANK`:

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  scripts/placemoe/calibrate_npu.sh calibration/runtime_perf_model.json \
  --hierarchy-group-sizes-csv 8,16
```

Rank 0 writes the fitted artifact. Make it available at the same path on every
node before training. The benchmark measures A2A and hierarchy-level
collectives over multiple message sizes; it is independent of the training
model and dataset.

## 3. Calibrate a new model

Use the intended training YAML to fit the model-dependent planner costs. The
artifact does not need to exist yet: this command derives an isolated
default-layout run from the YAML and disables optimization, checkpoints, and
parameter updates. Launch exactly one complete EP group on the required number
of nodes, changing only `NODE_RANK`:

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29502 \
  placemoe calibrate-model \
  --config configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py \
  --runtime-perf-model calibration/runtime_perf_model.json \
  --output calibration/model_and_topology.json
```

The default run uses 5 steps: 2 warm-up steps, 1 fitting step, and 2 held-out
validation steps. Rank 0 writes a scoped artifact only after validating the
communication, expert-compute, and joint predictions; rejected artifacts
cannot be used by the runtime. Use `tasks/train_text.py` for a text model and
make the accepted artifact available at the same path on every node. The
portable calibrator currently supports the validated 2-level node-to-rank
hierarchy. Multi-node calibration also uses `MASTER_PORT+1` to exchange
node-local timing summaries.

## 4. Configure one VeOmni training YAML

Keep the model, dataset, distributed topology, and PlaceMoE settings in one
file. The following block is the complete PlaceMoE surface:

```yaml
train:
  accelerator:
    ep_size: 16
    dp_shard_size: 16
  hiermoe:
    # Physical capacity reserved for additional copies on each EP rank.
    redundant_slot_increment_per_device: 4
    # Same-node group followed by the complete EP group.
    hierarchy_group_sizes: [8, 16]
    placemoe:
      enabled: true
      base_directory: /shared/placemoe
      initial_artifact: ""
      runtime_perf_model: calibration/runtime_perf_model.json
      calibration:
        artifact: calibration/model_and_topology.json
      hot_update:
        enabled: true
        layout_interval_steps: 100
        mapping_interval_steps: 20
        last_update_step: 1000
        work_root: runs/planner
        failure_policy: continue
      resources:
        workers: 48
        candidate_workers: 4
        worker_threads: 1
```

The preset automatically enables hierarchical token deduplication,
source-aware dispatch, step-boundary installation, and replica-gradient
overlap. It disables the historical swap/cover planners. The remaining values
are deployment inputs and must not be copied blindly:

- `ep_size`, `hierarchy_group_sizes`, and slot capacity describe the cluster
  and replica budget;
- `runtime_perf_model` describes A2A and expert-state transfer on that
  topology;
- `calibration.artifact` contains the communication and expert-compute
  coefficients used by every planner process; and
- model and dataset paths use the existing VeOmni schema.

An initial `L,M` artifact is optional. Without one, the initial mapping routes
to the canonical owners and the first full layout update creates replicas from
profiled routes. Therefore a positive `layout_interval_steps` is required when
`initial_artifact` is empty.

## 5. Validate before launch

Run the deployment doctor on every node:

```bash
placemoe doctor --config configs/my_train.yaml
```

It checks the validated software stack, visible NPUs and CANN, model and data
paths, runtime and planner calibration files, replica capacity, update
schedule, and the canonical PlaceMoE preset. Warnings do not block launch;
every `FAIL` should be resolved first.

## 6. Launch on each node

The launcher does not perform SSH orchestration. It runs the deployment doctor
locally before torchrun, so use the same repository revision, training YAML,
model, dataset, and calibration artifacts on every node:

```bash
# Node 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh tasks/train_vlm.py configs/my_train.yaml

# Node 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh tasks/train_vlm.py configs/my_train.yaml
```

Use `tasks/train_text.py` for a text model. Additional VeOmni command-line
overrides may follow the YAML path. Set `PLACEMOE_SKIP_PREFLIGHT=1` only when an
equivalent check is enforced by the cluster scheduler.

## 7. Static and adaptive modes

`L` and `M` have independent schedules:

| Layout interval | Mapping interval | Behavior |
| ---: | ---: | --- |
| `0` | `0` | Keep a preloaded `L,M` static. |
| `100` | `0` | Recompute and install both `L` and `M` every 100 steps. |
| `0` | `20` | Keep `L` fixed and refresh only the lookup table `M`. |
| `100` | `20` | Refresh `M` every 20 steps and both decisions every 100 steps. |

When both events are due, the full update subsumes the mapping-only update.
At most one CPU planner runs at a time; later events are coalesced while
training continues with the current pair. A completed artifact is schema-
validated before installation. Mapping-only updates do not move expert state;
full updates prevalidate all layer artifacts, migrate parameters and optimizer
states, and install `L` and `M` at a training-step boundary.

`failure_policy: continue` records planner failure and retains the current
pair. Use `raise` when a failed adaptive update must terminate the job.

## 8. Adapting another MoE model

PlaceMoE does not branch on model names. The default adapter supports expert
modules whose leading dimension indexes local expert slots and that expose
either:

- fused `gate_up_proj` and `down_proj`; or
- split `gate_proj`, `up_proj`, and `down_proj`.

These cover the validated Qwen3-VL and DeepSeek-V3 configurations. A model with
another parameter layout registers one `MoEModelAdapter` that returns its
expert-stacked parameters and normalized fused-kernel weights:

```python
from placemoe import register_moe_model_adapter

register_moe_model_adapter(MyModelAdapter())
```

Therefore, a model already integrated through VeOmni's standard model and EP
interfaces needs no PlaceMoE-specific model changes when it uses either default
expert representation. PlaceMoE fails during model binding if no adapter
matches, and fails during the first backward pass if the required
replica-gradient hooks do not execute; it never silently selects blocking
replica synchronization.

The planner remains unchanged because it consumes logical routes, topology,
capacities, and calibration rather than model-specific modules.

## Compatibility and known boundaries

- Static artifacts must match the model's layer keys, expert count, EP size,
  ranks per node, and slots per rank.
- Replicated placement currently requires `ep_fsdp_size=1` and does not support
  FSDP2 CPU offload.
- A mapping-only schedule needs an initial layout containing useful replicas;
  it cannot create copies by itself.
- The legacy `VEOMNI_PLACEMOE_CONFIG` and `VEOMNI_HIERMOE_*` controls remain for
  archived launchers and paper reproduction only. File-based legacy input also
  requires `VEOMNI_PLACEMOE_USE_LEGACY_CONFIG=1`; otherwise the inline PlaceMoE
  block is used. The legacy `config_path` input cannot be mixed with inline
  fields.
- Paper workflows are preserved under `scripts/placemoe/reproduction/` and
  `docs/perf/`; they are not production APIs.
