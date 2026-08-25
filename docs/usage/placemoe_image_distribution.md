# Packaging and distributing the PlaceMoE Ascend image

English | [中文](placemoe_image_distribution_zh.md)

This guide describes how to preserve a validated PlaceMoE image, transfer it
to an offline Ascend cluster, and load the identical image on every node. A
Docker image archive contains the image and all parent layers, so the target
nodes do not need access to the original base-image registry.

The archive does not contain host drivers, firmware, Docker volumes, models,
datasets, checkpoints, or files mounted when a container runs.

## 1. Build and verify the image

Build from a clean, committed VeOmni checkout containing PlaceMoE. The
validated PlaceMoE Dockerfile uses a public VeOmni Ascend base pinned by
digest.

```bash
docker build \
  -t placemoe:ascend-910b-cann9-torch2.9 \
  -f docker/ascend/Dockerfile.placemoe_9.0.0_torch_npu2.9.0_910b.arm .
```

Record the source revision and image ID, and run a basic smoke check before
packaging:

```bash
git rev-parse HEAD
docker image inspect \
  --format '{{.Id}} {{json .RepoDigests}}' \
  placemoe:ascend-910b-cann9-torch2.9
docker run --rm placemoe:ascend-910b-cann9-torch2.9 placemoe --help
```

Run at least one NPU smoke test and `placemoe doctor` with the intended
training configuration before treating an image as validated. Rebuild the
image from source instead of using `docker commit`; a committed container may
silently include runtime state, credentials, or untracked fixes.

## 2. Export an offline archive

Use `docker save`, not `docker export`. `docker save` preserves image layers,
configuration, entrypoint, environment, and tags. `docker export` only exports
a container filesystem and cannot reproduce the image metadata.

```bash
IMAGE=placemoe:ascend-910b-cann9-torch2.9
ARCHIVE=placemoe-ascend-910b-cann9-torch2.9.tar.gz

docker save "$IMAGE" | gzip -1 > "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
docker image inspect --format '{{.Id}}' "$IMAGE" > "$ARCHIVE.image-id"
```

The image is large because the archive includes CANN, PyTorch, torch-npu, and
all inherited layers. Ensure that both the source and target nodes have enough
temporary disk space. `gzip -1` favors transfer preparation speed; use a
higher compression level only when network capacity is more constrained than
CPU time.

## 3. Transfer and load on every cluster node

Copy the archive, checksum, and recorded image ID to every node. `rsync` is
recommended for large files because interrupted transfers can resume.

```bash
rsync --partial --progress \
  placemoe-ascend-910b-cann9-torch2.9.tar.gz* \
  user@target-node:/opt/placemoe-images/
```

On each target node:

```bash
cd /opt/placemoe-images
sha256sum --check placemoe-ascend-910b-cann9-torch2.9.tar.gz.sha256
gzip -dc placemoe-ascend-910b-cann9-torch2.9.tar.gz | docker load

docker image inspect \
  --format '{{.Id}}' \
  placemoe:ascend-910b-cann9-torch2.9
```

The loaded image ID must match the recorded ID on every node. Use the same
image tag in all distributed launch commands.

## 4. Publish through a registry

For a connected production cluster, an OCI registry is easier to operate than
copying archives. Use an immutable version tag that records the PlaceMoE
revision and validated accelerator stack:

```bash
REGISTRY=registry.example.com/moe
GIT_REV=$(git rev-parse --short=12 HEAD)
VERSION="${GIT_REV}-ascend910b-cann9-torch2.9"

docker tag \
  placemoe:ascend-910b-cann9-torch2.9 \
  "$REGISTRY/placemoe:$VERSION"
docker push "$REGISTRY/placemoe:$VERSION"
```

Each node then loads the identical image with:

```bash
VERSION='replace-with-recorded-immutable-version'
docker pull "registry.example.com/moe/placemoe:$VERSION"
```

Do not reuse or overwrite a published version tag. Record the registry digest
reported by `docker push` in the release notes or deployment manifest.

## 5. Target-cluster requirements

Every target node must provide:

- an `aarch64` host with compatible Ascend 910B firmware and driver;
- Docker and the site's supported Ascend container runtime or equivalent NPU
  device and driver mounts;
- HCCL connectivity and consistent rank-to-node topology;
- shared or identically mounted model, dataset, checkpoint, calibration, and
  PlaceMoE artifact paths; and
- sufficient local disk and shared memory for the image and training job.

Loading the image does not install or update the host Ascend driver. The host
driver and firmware must remain compatible with the CANN 9 runtime inside the
image.

After loading, mount the real model, dataset, and configuration paths and run:

```bash
placemoe doctor --config /path/to/training.yaml
```

Only start distributed training after `doctor` succeeds on every node.
The complete device-mount, container-entry, and single- and multi-node launch
commands are provided in [Running PlaceMoE from the validated Ascend
image](placemoe_container_quickstart.md).

## 6. Release checklist

Before distributing an image:

1. Build from a clean Git revision and record that revision.
2. Verify that the Docker build context contains no credentials or private
   artifacts.
3. Run `placemoe doctor` and an NPU smoke test.
4. Export with `docker save` or push an immutable registry tag.
5. Record the archive checksum or registry digest and image ID.
6. Confirm the same image ID and configuration on every training node.
7. Keep models, datasets, checkpoints, and credentials outside the image.
