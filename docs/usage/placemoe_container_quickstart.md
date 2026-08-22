# Running PlaceMoE from the validated Ascend image

This guide starts a single- or multi-node PlaceMoE training job after the
validated image has been loaded on every node. For image build, transfer, and
integrity checks, see [Packaging and distributing the PlaceMoE Ascend
image](placemoe_image_distribution.md). For calibration and the complete YAML
reference, see [PlaceMoE pre-training](placemoe_pretraining.md).

## 1. Prepare host paths

Keep models, datasets, training configurations, calibration artifacts, and
outputs outside the image. Store generated calibration artifacts under the
writable `OUTPUT_DIR`, and reference their container paths from the YAML.

```bash
IMAGE=placemoe:ascend-910b-cann9-torch2.9
CONTAINER=placemoe_train

# Directories containing one or more named models and datasets.
MODEL_ROOT=/path/to/models
DATA_ROOT=/path/to/datasets
# User-owned complete training YAMLs and writable run outputs.
CONFIG_DIR=/path/to/configs
OUTPUT_DIR=/path/to/output
NPU_SMI=$(command -v npu-smi)

mkdir -p "$OUTPUT_DIR"
test -x "$NPU_SMI"
```

The mounts establish the following contract:

| Host variable | Container path | Contents |
| --- | --- | --- |
| `MODEL_ROOT` | `/workspace/model` | Named model directories and tokenizers |
| `DATA_ROOT` | `/workspace/dataset` | Named dataset directories or shards |
| `CONFIG_DIR` | `/workspace/configs` | Complete training YAMLs |
| `OUTPUT_DIR` | `/workspace/output` | Calibration artifacts, checkpoints, logs, planner artifacts, and route snapshots |

Use the same container paths on every node. Training YAML files must contain
container paths such as `/workspace/model/Qwen3-VL-30B-A3B-Instruct`, rather
than the corresponding host paths.

## 2. Start the container

The validated 8-NPU Ascend 910B launch mounts every NPU device and the host
driver and firmware. Adapt the device list only when the host exposes a
different number of NPUs.

```bash
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
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v "$NPU_SMI":/usr/local/sbin/npu-smi:ro \
  -v "$MODEL_ROOT":/workspace/model:ro \
  -v "$DATA_ROOT":/workspace/dataset:ro \
  -v "$CONFIG_DIR":/workspace/configs:ro \
  -v "$OUTPUT_DIR":/workspace/output \
  "$IMAGE"
```

The image already contains the validated checkout and virtual environment at
`/opt/placemoe`. Do not mount another repository over that directory unless
you intentionally want to replace the code contained in the image.

The validated 910B host requires `--privileged` for the Ascend driver ioctls;
device nodes can otherwise be visible while `torch.npu.device_count()` still
returns zero. On a hardened production cluster, replace it only with the
site-provided Ascend container-runtime policy that grants equivalent device
access.

`npu-smi` is a host-driver diagnostic tool rather than part of PlaceMoE. The
command above resolves its host path and exposes it at
`/usr/local/sbin/npu-smi` inside the container.

## 3. Enter and verify the container

```bash
docker exec -it "$CONTAINER" bash
cd /opt/placemoe

npu-smi info
placemoe --help
python -c "import torch, torch_npu, placemoe; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())"
```

Then validate the complete deployment configuration on every node:

```bash
placemoe doctor --config /workspace/configs/my_train.yaml
```

Resolve every `FAIL` before launching training. In particular, verify the
model and data paths, EP topology, replica capacity, calibration artifacts,
and layout and mapping update intervals.

## 4. Start single-node training

For a multimodal model on all 8 local NPUs:

```bash
cd /opt/placemoe

NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml
```

Use `tasks/train_text.py` instead of `tasks/train_vlm.py` for a text-only
model. Normal VeOmni command-line overrides may follow the YAML path.

## 5. Start multi-node training

Start the same image and mounts on every node. Confirm that every node reports
the same image ID:

```bash
docker image inspect --format '{{.Id}}' \
  placemoe:ascend-910b-cann9-torch2.9
```

Choose the reachable IP address of node 0 as `MASTER_ADDR`. In the container
on node 0, run:

```bash
cd /opt/placemoe

NNODES=2 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 \
MASTER_PORT=29500 \
scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml
```

Run the same command in the container on node 1 with only `NODE_RANK` changed:

```bash
cd /opt/placemoe

NNODES=2 \
NODE_RANK=1 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 \
MASTER_PORT=29500 \
scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml
```

The launcher runs `placemoe doctor --config` locally before `torchrun`. It does not
perform SSH orchestration, so both commands must be started through the
cluster scheduler or on their respective nodes.

## 6. Prepare calibration artifacts

Configure `runtime_perf_model` and `calibration.artifact` in the training YAML
to point into `/workspace/output`. The recommended command reuses valid JSON
artifacts and creates only missing ones. For a 2-node, 16-rank deployment, run
it on both nodes and change only `NODE_RANK`:

```bash
NNODES=2 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 \
MASTER_PORT=29501 \
placemoe prepare \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py
```

The first stage measures communication for the topology. The second runs 5
default-layout steps to fit and validate the model-dependent planner costs.
Valid files are not recomputed; an invalid existing file fails explicitly.
Use `--force-runtime` or `--force-model` only when replacement is intended.
After preparation, run `placemoe doctor` and then launch training.

For separate `calibrate-runtime` and `calibrate-model` commands, artifact reuse
rules, and model-versus-topology scope, see the pre-training guide.

## 7. Common failures

- `libascend_hal.so` is missing: the host driver or NPU devices were not
  mounted correctly.
- `placemoe doctor` cannot find a path: the YAML probably contains a host path
  instead of its container path.
- `.venv/bin/python` is missing: a host directory may have been mounted over
  `/opt/placemoe`.
- HCCL times out: verify `--network host`, `MASTER_ADDR`, the port, firewall,
  rank counts, and HCCL connectivity between all nodes.
- An adaptive run has no initial artifact: configure a positive layout update
  interval so PlaceMoE can create replicas after collecting routes.
