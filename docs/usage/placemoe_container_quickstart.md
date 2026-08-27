# PlaceMoE container quick start

English | [中文](placemoe_container_quickstart_zh.md)

This guide takes a new Ascend user from a Git checkout to a calibrated,
multi-node PlaceMoE training job. It uses bind mounts for the source tree,
training YAMLs, models, datasets, and outputs. The image therefore provides a
reproducible accelerator and Python environment, while users can update Python
source and YAML files without rebuilding it.

The validated production path uses aarch64 Ascend 910B hosts, Python 3.11,
CANN 9, PyTorch 2.9.0, and torch-npu 2.9.0.post2. For image export and offline
distribution, see [Packaging and distributing the PlaceMoE Ascend
image](placemoe_image_distribution.md). For the complete configuration and
calibration reference, see [PlaceMoE pre-training](placemoe_pretraining.md).

## 1. Build the image and prepare mounted inputs

### 1.1 Check out one revision on every node

Clone PlaceMoE on every training node, and check out the same release tag or
commit. Do not run one distributed job from different revisions.

```bash
git clone https://github.com/pia7a/PlaceMoE.git /home/user/PlaceMoE
cd /home/user/PlaceMoE
git checkout <validated-tag-or-commit>
git rev-parse HEAD
```

The final command must print the same commit on every node. A shared source
directory is also acceptable when it provides the same path and revision to
all nodes.

### 1.2 Build and distribute the image

Build the image once from a clean checkout:

```bash
cd /home/user/PlaceMoE

docker build \
  -t placemoe:ascend-910b-cann9-torch2.9 \
  -f docker/ascend/Dockerfile .

docker run --rm \
  placemoe:ascend-910b-cann9-torch2.9 \
  placemoe --help
```

The Dockerfile uses a public VeOmni Ascend base pinned by digest. Build once
and distribute the resulting image through an OCI registry or `docker save`;
do not independently rebuild nominally identical production images on every
node. Verify that every node reports the same image ID:

```bash
docker image inspect --format '{{.Id}}' \
  placemoe:ascend-910b-cann9-torch2.9
```

### 1.3 Prepare and synchronize mounted directories

Keep mutable inputs and outputs outside the image:

```bash
SOURCE_ROOT=/home/user/PlaceMoE
MODEL_ROOT=/data/models
DATA_ROOT=/data/datasets
CONFIG_DIR=/home/user/placemoe-configs
OUTPUT_DIR=/data/placemoe-output

mkdir -p "$CONFIG_DIR" "$OUTPUT_DIR"
```

The container path, rather than the host path, is the distributed contract:

| Host variable | Container path | Access | Contents |
| --- | --- | --- | --- |
| `SOURCE_ROOT` | `/workspace/PlaceMoE` | Read-only | PlaceMoE and VeOmni source at one pinned commit |
| `MODEL_ROOT` | `/workspace/model` | Read-only | Model directories, configs, and tokenizers |
| `DATA_ROOT` | `/workspace/dataset` | Read-only | Dataset descriptors and shards |
| `CONFIG_DIR` | `/workspace/configs` | Read-only | Complete training YAMLs |
| `OUTPUT_DIR` | `/workspace/output` | Read-write | Calibration artifacts, plans, checkpoints, and logs |

Host paths may differ between nodes, but the container paths must be
identical. When the cluster has no shared filesystem, synchronize source and
configuration before starting the containers. For example, from node 0:

```bash
rsync -av --checksum \
  /home/user/placemoe-configs/ \
  user@node1:/home/user/placemoe-configs/
```

Verify both the source revision and training configuration:

```bash
git -C /home/user/PlaceMoE rev-parse HEAD
sha256sum /home/user/placemoe-configs/my_train.yaml
```

A YAML edit made on the host is immediately visible through the bind mount,
but a running training process does not reread its configuration. Restart the
job after editing the YAML.

### 1.4 Create a complete training YAML

Start from a VeOmni YAML that already loads the intended model and dataset.
Use container paths for all files, for example:

```yaml
model:
  model_path: /workspace/model/my-model

data:
  train_path: /workspace/dataset/my-training-data.yaml
```

Then configure the EP topology and append the PlaceMoE block. The following
example describes 2 nodes with 8 ranks per node:

```yaml
train:
  accelerator:
    ep_size: 16
    dp_shard_size: 16

  hiermoe:
    # Extra physical expert slots reserved on each EP rank.
    redundant_slot_increment_per_device: 4
    # Same-node group followed by the complete EP group.
    hierarchy_group_sizes: [8, 16]

    placemoe:
      enabled: true
      base_directory: /workspace/output/my-run
      # Leave empty for adaptive startup from the default layout.
      initial_artifact: ""
      runtime_perf_model: calibration/runtime_perf_model.json
      calibration:
        artifact: calibration/model_and_topology.json
      hot_update:
        enabled: true
        layout_interval_steps: 100
        mapping_interval_steps: 20
        last_update_step: 1000
        work_root: planner
        failure_policy: continue
      resources:
        workers: 48
        candidate_workers: 4
        worker_threads: 1

  checkpoint:
    output_dir: /workspace/output/my-run/checkpoints
```

These values are examples, not portable tuning defaults:

- `ep_size` must equal the complete EP group used by the job;
- `hierarchy_group_sizes` must describe the real rank hierarchy;
- the redundant-slot budget must fit device memory;
- planner worker counts must fit the host CPU allocation; and
- model and data fields continue to use the normal VeOmni schema.

For one node, set `ep_size` to the local rank count and use a single hierarchy
entry, for example `ep_size: 8` with `hierarchy_group_sizes: [8]`. A redundant
slot budget of `0` is valid and keeps layout optimization enabled while
disabling expert replication and replica-gradient synchronization.

When `initial_artifact` is empty, PlaceMoE initially uses the default layout
and applies the first full layout update after collecting routes. With a
positive replica budget, that update also creates the replicas. A mapping-only
update cannot change the physical layout, so
`layout_interval_steps` must be positive in this mode. To keep a precomputed
layout and mapping static, provide `initial_artifact` and set both update
intervals to `0`.

### 1.5 Start the container on every node

The following command mounts 8 local NPUs. Adapt the device list only when a
host exposes a different number of devices.

```bash
IMAGE=placemoe:ascend-910b-cann9-torch2.9
CONTAINER=placemoe_train
NPU_SMI=$(command -v npu-smi)

test -x "$NPU_SMI"

docker run -dit \
  --name "$CONTAINER" \
  --privileged \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTHONPATH=/workspace/PlaceMoE:/opt/placemoe \
  -e PLACEMOE_PYTHON=/opt/placemoe/.venv/bin/python \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v "$NPU_SMI":/usr/local/sbin/npu-smi:ro \
  -v "$SOURCE_ROOT":/workspace/PlaceMoE:ro \
  -v "$MODEL_ROOT":/workspace/model:ro \
  -v "$DATA_ROOT":/workspace/dataset:ro \
  -v "$CONFIG_DIR":/workspace/configs:ro \
  -v "$OUTPUT_DIR":/workspace/output \
  -w /workspace/PlaceMoE \
  "$IMAGE"
```

Do not mount a host checkout over `/opt/placemoe`: doing so hides the virtual
environment installed in the image. Mounting it at `/workspace/PlaceMoE` and
putting that path first in `PYTHONPATH` keeps the validated dependencies while
loading the mounted Python source.

The validated 910B host requires `--privileged` for the Ascend driver ioctls.
On a hardened cluster, replace it only with the site-provided Ascend container
runtime policy that grants equivalent access.

## 2. Calibrate and start training

### 2.1 Enter and verify the container

On every node:

```bash
docker exec -it placemoe_train bash
cd /workspace/PlaceMoE

npu-smi info
python -c \
  "import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())"
python -c \
  "import veomni.distributed.moe.hiermoe.placemoe.cli as m; print(m.__file__)"
```

The final command must resolve under `/workspace/PlaceMoE`; otherwise the
container is using its image-baked source rather than the mounted checkout.

### 2.2 Prepare both calibration artifacts

PlaceMoE uses 2 calibration stages:

| Stage | Measures | Typical reuse boundary |
| --- | --- | --- |
| Runtime calibration | Hierarchical A2A, expert-state movement, and replica-gradient synchronization | Same accelerator stack and EP topology |
| Model calibration | Communication and expert-compute coefficients used by the planner | Same model, workload, execution configuration, topology, and entrypoint |

The recommended `prepare` command reads both artifact paths from the training
YAML. It reuses valid artifacts, generates missing artifacts, and rejects an
invalid existing artifact rather than silently overwriting it. Run it in the
container on every node, changing only `NODE_RANK`:

```bash
# Node 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe prepare \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py

# Node 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe prepare \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py
```

Use `tasks/train_text.py` for a text-only model. The model stage runs 5
default-layout steps: 2 warm-up steps, 1 fitting step, and 2 held-out
validation steps. Preparation does not generate the final layout `L` and
mapping `M`; the runtime planner derives them from routes collected during the
training job.

Keep `MASTER_PORT` and the next 4 ports free during preparation. Use
`--force-runtime` or `--force-model` only when an existing artifact is known
to be stale and replacement is intended. Rebuilding runtime calibration also
invalidates and rebuilds model calibration because the latter records the
runtime artifact hash.

### 2.3 Run the 2 calibration stages separately when needed

The one-command path is recommended for normal use. Administrators may
calibrate only the cluster topology first:

```bash
# Run on every node; change only NODE_RANK.
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe calibrate-runtime \
  --output /workspace/output/my-run/calibration/runtime_perf_model.json \
  --hierarchy-group-sizes-csv 8,16
```

Then calibrate the model and execution configuration:

```bash
# Run on every node; change only NODE_RANK.
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29502 \
  placemoe calibrate-model \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py \
  --runtime-perf-model \
    /workspace/output/my-run/calibration/runtime_perf_model.json \
  --output \
    /workspace/output/my-run/calibration/model_and_topology.json
```

The standalone commands intentionally write their requested outputs. Use
`placemoe prepare` when cache validation and automatic reuse are desired.

### 2.4 Validate the complete deployment

After calibration, run on every node:

```bash
placemoe doctor --config /workspace/configs/my_train.yaml
```

Resolve every `FAIL` before training. The doctor validates the software stack,
visible NPUs, source bridge, model and data paths, topology, replica capacity,
calibration scope, and hot-update schedule.

### 2.5 Start multi-node training

Use a different free master port from calibration. Start the launcher on every
node; it does not perform SSH orchestration.

```bash
# Node 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml

# Node 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml
```

Use the scheduler to start each node when available. Normal VeOmni command-line
overrides may follow the YAML path. The launcher runs `placemoe doctor`
locally before `torchrun`; set `PLACEMOE_SKIP_PREFLIGHT=1` only when the
cluster scheduler enforces an equivalent check.

## 3. Adapt another model or VeOmni fork

### 3.1 Decide which integration is required

PlaceMoE separates 2 boundaries:

1. VeOmni integrates the model, data pipeline, checkpoint, and EP execution.
2. PlaceMoE discovers the routed-expert tensors through `MoEModelAdapter` and
   controls the MoE runtime through the versioned bridge.

No PlaceMoE-specific model code is needed when a model already uses VeOmni's
standard routed-expert representation:

- a leading local-expert dimension;
- `num_experts` on the expert module; and
- either fused `gate_up_proj`/`down_proj` or split
  `gate_proj`/`up_proj`/`down_proj` parameters.

A different stacked representation implements and registers one
`MoEModelAdapter`. An unstacked `ModuleList` of independent expert modules
must first be converted to an expert-stacked VeOmni representation; merely
registering the list does not make its parameters compatible with slot
movement or grouped expert kernels.

### 3.2 Example adaptation plan: openPangu-Ultra-MoE-718B

This section is an adaptation example, not a claim of validated PlaceMoE
support. The public
[openPangu-Ultra-MoE-718B-V1.1](https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-V1.1)
configuration uses model type `pangu_ultra_moe`, 61 layers, 256 routed
experts, top-8 routing, and one shared expert. Its published remote code uses
an independent-expert `ModuleList` and targets Transformers 4.48.2, whereas
this repository targets Transformers 5.2.0, FSDP2, and VeOmni's stacked EP
kernels. Do not expect the remote-code checkpoint to train under PlaceMoE
without a VeOmni model port.

A production port should proceed in this order:

1. Add the model to VeOmni using the Transformers v5 model workflow. Register
   `pangu_ultra_moe` at import time and make the generated model code derive
   from the pinned Transformers 5.2.0 implementation. Do not edit files under
   a model's `generated/` directory manually.
2. Implement the normal VeOmni EP parallel plan and token-routing path. Verify
   default VeOmni training before enabling PlaceMoE.
3. Convert the 256 routed experts from independent modules to a stacked fused
   or split projection representation, including checkpoint conversion. Keep
   the shared expert outside the routed-expert placement and replication set.
4. If the stacked field names differ from the default contract, implement the
   `matches`, `num_experts`, `expert_parameters`,
   `replace_expert_parameter`, and `kernel_weights` methods from
   `placemoe.MoEModelAdapter`, then register the adapter when the model package
   is imported.
5. First validate model loading and a default-layout forward/backward pass,
   then run `placemoe prepare`, `placemoe doctor`, and a short 2-node adaptive
   job. Confirm loss consistency, layout installation, replica-gradient hook
   execution, and checkpoint save/resume before performance measurement.

The PlaceMoE planner itself remains unchanged because it consumes logical
routes, topology, slot capacities, and calibration rather than model names.
The detailed VeOmni model workflow is documented in the [new-model guide and
checklist](support_new_models/guide_and_checklist.md), and the public adapter
protocol is implemented in `placemoe/model_adapter.py`.

### 3.3 Integrate a user-modified VeOmni checkout

If the user's VeOmni changes are limited to model registration, generated
model code, checkpoint conversion, data processing, or the parallel plan, use
a PlaceMoE release as the branch base and carry those model commits onto it.
The existing PlaceMoE trainer and distributed hooks then remain intact.

For a fork with a custom trainer or training loop, preserve the versioned
`MoERuntimeBridge` lifecycle. In particular, the host must:

- configure the bridge after the EP process group is available;
- expand and bind expert slots during model construction;
- bind the optimizer after it is created;
- execute every trainable forward inside the bridge's training-forward scope;
- report step and microstep boundaries;
- run replica-gradient synchronization after backward;
- invoke the step-end layout/mapping update and metric hooks; and
- shut down planner workers and process groups cleanly.

Custom trainers must also preserve PlaceMoE checkpoint and gradient-norm
semantics. Replica-gradient overlap never silently falls back to blocking
synchronization: missing hooks are treated as integration errors.

An arbitrary upstream VeOmni checkout cannot be made PlaceMoE-compatible by
only importing a package because upstream VeOmni does not expose every
required lifecycle hook. Avoid `sitecustomize` and runtime monkey patches.
Use the [versioned bridge integration
guide](placemoe_veomni_bridge.md) to carry the small, reviewable host boundary
into a fork.

A training framework that is not based on VeOmni additionally needs an
equivalent EP dispatch path, expert-stacked parameter contract, checkpoint
semantics, and implementation of the complete runtime-bridge lifecycle. That
is a framework port rather than a model adapter; the current plugin does not
claim zero-touch support for an arbitrary training framework.

## 4. Common problems and recommendations

### The YAML is not visible on another node

Mount a host configuration directory at `/workspace/configs` on every node.
Synchronize the host file with a shared filesystem, Git, or `rsync`, and
compare `sha256sum` before launch. Docker cannot add a new bind mount to an
already running container; recreate the container if the mount was omitted.

### Source edits are not used inside the container

Do not mount over `/opt/placemoe`. Mount the checkout at
`/workspace/PlaceMoE`, set
`PYTHONPATH=/workspace/PlaceMoE:/opt/placemoe`, and verify the imported module
path as shown in Section 2.1. Restart Python or the training job after editing
source.

Pure Python and YAML edits do not require an image rebuild with this bind-mount
workflow. Rebuild the image after changing dependencies, `pyproject.toml`, the
installed console entry point, native extensions, or the validated
accelerator stack. Production releases should still record one source commit
and one image ID.

### NPU devices exist, but `torch.npu.is_available()` is false

Visible `/dev/davinci*` files are insufficient when the driver ioctls and
libraries are unavailable. Use the validated `--privileged` launch or the
site's equivalent Ascend runtime policy, and mount the host driver and
firmware. Loading the image does not install or upgrade the host driver.

### `npu-smi` is missing

`npu-smi` belongs to the host driver, not PlaceMoE. Resolve it on the host with
`command -v npu-smi` and mount that executable into the container as shown in
Section 1.5. The decisive runtime test is still whether torch-npu can enumerate
and use the devices.

### Calibration appears to hang

Every declared node must start the same command. Verify that:

- `NNODES`, `NPROC_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` match;
- each node has a unique `NODE_RANK` in `[0, NNODES)`;
- the master address is reachable from every container;
- `MASTER_PORT` and the next 4 ports are free; and
- firewalls permit HCCL and torch distributed traffic.

Starting node 0 twice with `NODE_RANK=0` does not replace the missing node 1.
If a wrong command was started, inspect the exact torchrun/calibration process
IDs and terminate those processes before retrying with a new free port.

### `placemoe doctor` reports missing calibration scope keys

The JSON is an old or manually constructed artifact rather than a current
PlaceMoE calibration artifact. Run `placemoe prepare`; do not add scope fields
by hand. If a mismatched artifact already exists, review the path and use the
appropriate force flag only when replacement is intentional.

### A valid artifact is unexpectedly recalibrated or rejected

Runtime calibration is scoped to the accelerator environment and EP topology.
Model calibration also fingerprints the model, representative data,
execution settings, entrypoint, and runtime artifact. A changed model kernel,
dependency, topology, or workload may require `--force-model` or
`--force-runtime`. All nodes must expose byte-identical artifacts.

### HCCL times out

Verify `--network host`, the selected network interface, `MASTER_ADDR`, rank
counts, firewall rules, and HCCL connectivity. Confirm that no previous job is
still using the selected ports.

### Training runs, but no useful replica layout appears

A mapping-only update changes `M` but cannot create expert copies. With an
empty `initial_artifact`, configure a positive `layout_interval_steps` and
reserve nonzero redundant slots. Confirm planner and installation metrics in
the training log.

### The model or configuration runs out of memory

First lower micro-batch size, sequence length, media resolution, or redundant
slot capacity. For an integration-only smoke test, a representative subset of
MoE layers is acceptable, but do not compare its performance with a full-model
result. Re-run model calibration whenever the execution configuration changes.

### Recommended production checks

Before a long run:

1. record the image ID, PlaceMoE commit, YAML checksum, model revision, and
   calibration artifact hashes;
2. run `placemoe doctor` on every node;
3. complete a short multi-node forward/backward and checkpoint-resume test;
4. verify that all nodes install the same `L` and `M`; and
5. measure performance only after correctness and gradient-overlap checks
   pass.
