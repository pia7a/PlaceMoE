# PlaceMoE Ascend 镜像打包与分发

[English](placemoe_image_distribution.md) | 中文

本文介绍如何保存经过验证的 PlaceMoE 镜像，将其传输到离线 Ascend 集群，并在
所有节点加载完全一致的镜像。Docker 镜像归档包含目标镜像及其所有父层，因此
目标节点无需访问原始基础镜像仓库。

镜像归档不包含宿主机驱动、固件、Docker volume、模型、数据集、checkpoint，
也不包含容器运行时通过 bind mount 挂载的文件。

## 1. 构建并验证镜像

从干净且已提交的 PlaceMoE 源码目录构建镜像。生产 Dockerfile 使用按 digest
固定的公开 VeOmni Ascend 基础镜像。

```bash
docker build \
  -t placemoe:ascend-910b-cann9-torch2.9 \
  -f docker/ascend/Dockerfile .
```

打包前记录源码版本和镜像 ID，并执行基本 smoke test：

```bash
git rev-parse HEAD
docker image inspect \
  --format '{{.Id}} {{json .RepoDigests}}' \
  placemoe:ascend-910b-cann9-torch2.9
docker run --rm placemoe:ascend-910b-cann9-torch2.9 placemoe --help
```

只有在使用目标训练配置通过 `placemoe doctor` 并至少完成一次 NPU smoke test
之后，才能将镜像视为已验证版本。应从源码重新构建镜像，不要使用
`docker commit`；后者可能无意中包含运行状态、凭据或未跟踪的临时修复。

## 2. 导出离线归档

使用 `docker save`，不要使用 `docker export`。`docker save` 会保留镜像层、
配置、entrypoint、环境变量和标签；`docker export` 仅导出容器文件系统，无法
复现镜像元数据。

```bash
IMAGE=placemoe:ascend-910b-cann9-torch2.9
ARCHIVE=placemoe-ascend-910b-cann9-torch2.9.tar.gz

docker save "$IMAGE" | gzip -1 > "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
docker image inspect --format '{{.Id}}' "$IMAGE" > "$ARCHIVE.image-id"
```

归档包含 CANN、PyTorch、torch-npu 及所有继承层，因此文件会很大。请确保源节点
和目标节点都有足够的临时磁盘空间。`gzip -1` 优先保证打包速度；只有当网络容量
比 CPU 时间更紧张时，才建议提高压缩等级。

## 3. 传输并在每个集群节点加载

将归档、checksum 和记录的镜像 ID 复制到每个节点。大文件建议使用 `rsync`，
因为它可以续传中断的传输：

```bash
rsync --partial --progress \
  placemoe-ascend-910b-cann9-torch2.9.tar.gz* \
  user@target-node:/opt/placemoe-images/
```

在每个目标节点执行：

```bash
cd /opt/placemoe-images
sha256sum --check placemoe-ascend-910b-cann9-torch2.9.tar.gz.sha256
gzip -dc placemoe-ascend-910b-cann9-torch2.9.tar.gz | docker load

docker image inspect \
  --format '{{.Id}}' \
  placemoe:ascend-910b-cann9-torch2.9
```

所有节点加载后的镜像 ID 必须与源节点记录的 ID 一致。所有分布式启动命令也必须
使用相同的镜像标签。

## 4. 通过镜像仓库发布

对于能够访问网络的生产集群，使用 OCI registry 通常比复制归档更容易维护。
使用包含 PlaceMoE revision 和已验证加速器软件栈的不可变版本标签：

```bash
REGISTRY=registry.example.com/moe
GIT_REV=$(git rev-parse --short=12 HEAD)
VERSION="${GIT_REV}-ascend910b-cann9-torch2.9"

docker tag \
  placemoe:ascend-910b-cann9-torch2.9 \
  "$REGISTRY/placemoe:$VERSION"
docker push "$REGISTRY/placemoe:$VERSION"
```

各节点随后加载同一个镜像：

```bash
VERSION='replace-with-recorded-immutable-version'
docker pull "registry.example.com/moe/placemoe:$VERSION"
```

不要复用或覆盖已经发布的版本标签。将 `docker push` 输出的 registry digest
记录到 release notes 或部署清单中。

## 5. 目标集群要求

每个目标节点必须提供：

- 带有兼容 Ascend 910B 固件和驱动的 `aarch64` 宿主机；
- Docker，以及站点支持的 Ascend container runtime，或等价的 NPU 设备与驱动
  挂载方式；
- 可正常工作的 HCCL 网络和一致的 rank-to-node 拓扑；
- 共享或以相同路径挂载的模型、数据集、checkpoint、校准文件和 PlaceMoE
  artifact；
- 足够容纳镜像和训练任务的本地磁盘与共享内存。

加载镜像不会安装或升级宿主机 Ascend 驱动。宿主机驱动和固件必须与镜像内的
CANN 9 runtime 兼容。

加载镜像后，挂载真实模型、数据集和训练配置并运行：

```bash
placemoe doctor --config /path/to/training.yaml
```

只有当每个节点上的 `doctor` 都通过后，才能启动分布式训练。完整的设备挂载、
容器进入方式以及单节点和多节点启动命令见
[从已验证 Ascend 镜像运行 PlaceMoE](placemoe_container_quickstart_zh.md)。

## 6. 发布检查清单

分发镜像前：

1. 从干净的 Git revision 构建，并记录该 revision；
2. 确认 Docker build context 不包含凭据或私有 artifact；
3. 运行 `placemoe doctor` 和 NPU smoke test；
4. 使用 `docker save` 导出，或推送不可变 registry 标签；
5. 记录归档 checksum 或 registry digest，以及镜像 ID；
6. 确认每个训练节点使用相同的镜像 ID 和配置；
7. 将模型、数据集、checkpoint 和凭据保存在镜像之外。
