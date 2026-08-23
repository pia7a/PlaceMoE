# PlaceMoE 容器快速入门

[English](placemoe_container_quickstart.md) | 中文

本文面向首次使用 PlaceMoE 的 Ascend 用户，介绍如何从 Git 源码开始，
完成镜像构建、目录挂载、集群与模型校准，并启动多节点训练。本文将源码、
训练 YAML、模型、数据集和输出目录都挂载到容器中：镜像提供可复现的加速器
与 Python 环境，用户修改 Python 源码或 YAML 时则无需重新构建镜像。

当前验证过的生产环境为 aarch64 Ascend 910B、Python 3.11、CANN 9、
PyTorch 2.9.0 和 torch-npu 2.9.0.post2。离线导出和分发镜像请参阅
[PlaceMoE Ascend 镜像打包与分发](placemoe_image_distribution_zh.md)；完整配置和
校准规则请参阅 [PlaceMoE 预训练指南](placemoe_pretraining_zh.md)。

## 1. 构建镜像并准备挂载内容

### 1.1 在所有节点检出同一个源码版本

在每个训练节点上克隆 PlaceMoE，并检出相同的发布标签或提交。一次分布式
任务不能混用不同的源码版本。

```bash
git clone https://github.com/pia7a/PlaceMoE.git /home/user/PlaceMoE
cd /home/user/PlaceMoE
git checkout <validated-tag-or-commit>
git rev-parse HEAD
```

最后一条命令在所有节点上必须输出相同的 commit。也可以使用共享源码目录，
前提是所有节点看到的路径和版本完全一致。

### 1.2 构建并分发镜像

从干净的源码目录构建一次镜像：

```bash
cd /home/user/PlaceMoE

docker build \
  -t placemoe:ascend-910b-cann9-torch2.9 \
  -f docker/ascend/Dockerfile .

docker run --rm \
  placemoe:ascend-910b-cann9-torch2.9 \
  placemoe --help
```

Dockerfile 使用按 digest 固定的公开 VeOmni Ascend 基础镜像。生产环境建议只
构建一次，然后通过 OCI registry 或 `docker save` 将该镜像分发到所有节点；
不要在每个节点独立构建名义上相同的生产镜像。确认各节点的镜像 ID 一致：

```bash
docker image inspect --format '{{.Id}}' \
  placemoe:ascend-910b-cann9-torch2.9
```

### 1.3 准备并同步挂载目录

将可变输入和输出保存在镜像之外：

```bash
SOURCE_ROOT=/home/user/PlaceMoE
MODEL_ROOT=/data/models
DATA_ROOT=/data/datasets
CONFIG_DIR=/home/user/placemoe-configs
OUTPUT_DIR=/data/placemoe-output

mkdir -p "$CONFIG_DIR" "$OUTPUT_DIR"
```

分布式任务真正依赖的是一致的容器路径，而不是一致的宿主机路径：

| 宿主机变量 | 容器路径 | 权限 | 内容 |
| --- | --- | --- | --- |
| `SOURCE_ROOT` | `/workspace/PlaceMoE` | 只读 | 固定到同一个 commit 的 PlaceMoE 和 VeOmni 源码 |
| `MODEL_ROOT` | `/workspace/model` | 只读 | 模型目录、配置和 tokenizer |
| `DATA_ROOT` | `/workspace/dataset` | 只读 | 数据集描述文件和数据分片 |
| `CONFIG_DIR` | `/workspace/configs` | 只读 | 完整的训练 YAML |
| `OUTPUT_DIR` | `/workspace/output` | 读写 | 校准文件、布局方案、checkpoint 和日志 |

不同节点的宿主机路径可以不同，但容器路径必须一致。若集群没有共享文件系统，
请在启动容器前同步源码和配置。例如，在节点 0 上执行：

```bash
rsync -av --checksum \
  /home/user/placemoe-configs/ \
  user@node1:/home/user/placemoe-configs/
```

检查源码版本和训练配置是否一致：

```bash
git -C /home/user/PlaceMoE rev-parse HEAD
sha256sum /home/user/placemoe-configs/my_train.yaml
```

在宿主机上修改 YAML 后，容器内会立即看到新内容；但正在运行的训练进程不会
重新读取配置，因此修改 YAML 后需要重新启动训练任务。

### 1.4 创建完整的训练 YAML

首先准备一份能够正常加载目标模型和数据集的 VeOmni YAML。所有文件路径必须
使用容器内路径，例如：

```yaml
model:
  model_path: /workspace/model/my-model

data:
  train_path: /workspace/dataset/my-training-data.yaml
```

然后配置 EP 拓扑并加入 PlaceMoE 配置。下面的例子表示 2 个节点、每节点 8 个
rank：

```yaml
train:
  accelerator:
    ep_size: 16
    dp_shard_size: 16

  hiermoe:
    # 每个 EP rank 为额外专家副本预留的物理槽位数。
    redundant_slot_increment_per_device: 4
    # 节点内通信组，以及完整的 EP 通信组。
    hierarchy_group_sizes: [8, 16]

    placemoe:
      enabled: true
      base_directory: /workspace/output/my-run
      # 留空表示从默认布局开始自适应更新。
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

以上数值只是配置示例，不能直接作为所有集群的默认值：

- `ep_size` 必须等于训练任务使用的完整 EP group 大小；
- `hierarchy_group_sizes` 必须描述真实的 rank 层级；
- 冗余槽位预算必须符合设备显存限制；
- planner worker 数量必须符合宿主机 CPU 资源；
- 模型和数据配置继续沿用普通 VeOmni YAML 的定义。

当 `initial_artifact` 为空时，PlaceMoE 先使用默认布局，并在第一次完整布局更新前
收集路由数据。mapping-only 更新不能创建专家副本，因此此时
`layout_interval_steps` 必须为正数。若要静态使用预先生成的布局和映射，需要
设置 `initial_artifact`，并将两个更新间隔都设为 `0`。

### 1.5 在所有节点启动容器

下面的命令挂载 8 个本地 NPU。只有当宿主机暴露的 NPU 数量不同时，才需要
调整设备列表。

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

不要将宿主机源码覆盖挂载到 `/opt/placemoe`，否则会遮蔽镜像中的虚拟环境。
将源码挂载到 `/workspace/PlaceMoE`，并把该路径放在 `PYTHONPATH` 首位，可以
同时使用镜像中验证过的依赖和宿主机上的最新 Python 源码。

验证过的 910B 宿主机需要 `--privileged` 才能正常执行 Ascend driver ioctl。
在安全策略更严格的集群中，只能用站点提供的、具有等价设备权限的 Ascend
container runtime 策略替代它。

## 2. 校准并开始训练

### 2.1 进入容器并检查环境

在每个节点执行：

```bash
docker exec -it placemoe_train bash
cd /workspace/PlaceMoE

npu-smi info
python -c \
  "import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())"
python -c \
  "import veomni.distributed.moe.hiermoe.placemoe.cli as m; print(m.__file__)"
```

最后一条命令的输出必须位于 `/workspace/PlaceMoE` 下；否则容器使用的是镜像
构建时写入的源码，而不是当前挂载的源码。

### 2.2 一键准备两类校准文件

PlaceMoE 使用两类校准：

| 阶段 | 测量内容 | 通常可以复用的范围 |
| --- | --- | --- |
| 集群运行时校准 | 分层 A2A、专家状态迁移和副本梯度同步 | 相同加速器软件栈和 EP 拓扑 |
| 模型校准 | planner 使用的通信与专家计算系数 | 相同模型、负载、执行配置、拓扑和训练入口 |

推荐使用 `prepare` 命令。它会从训练 YAML 中读取两个 artifact 路径，复用有效
文件、创建缺失文件，并在已有文件无效时明确报错，而不是静默覆盖。在每个节点
的容器中运行以下命令，只修改 `NODE_RANK`：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe prepare \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py

# 节点 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe prepare \
  --config /workspace/configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py
```

纯文本模型使用 `tasks/train_text.py`。模型校准默认运行 5 个默认布局 step：
2 个 warm-up step、1 个拟合 step 和 2 个 held-out 验证 step。
`prepare` 不会生成最终布局 `L` 和映射 `M`；runtime planner 会根据实际训练中
收集的路由生成它们。

校准期间需要保证 `MASTER_PORT` 以及后续 4 个端口空闲。只有在确定现有文件
已经过期并希望替换时，才使用 `--force-runtime` 或 `--force-model`。重新进行
runtime 校准也会使模型校准失效并重新生成，因为模型校准文件记录了 runtime
artifact 的哈希值。

### 2.3 按需分别执行两类校准

普通用户推荐使用一键命令。集群管理员也可以先只校准集群拓扑：

```bash
# 所有节点都要执行，只修改 NODE_RANK。
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe calibrate-runtime \
  --output /workspace/output/my-run/calibration/runtime_perf_model.json \
  --hierarchy-group-sizes-csv 8,16
```

然后校准模型和执行配置：

```bash
# 所有节点都要执行，只修改 NODE_RANK。
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

两个独立命令会写入指定的输出文件。需要自动校验和复用缓存时，应使用
`placemoe prepare`。

### 2.4 校验完整部署配置

校准完成后，在每个节点运行：

```bash
placemoe doctor --config /workspace/configs/my_train.yaml
```

开始训练前必须解决所有 `FAIL`。doctor 会检查软件环境、可见 NPU、源码 bridge、
模型和数据路径、拓扑、冗余容量、校准文件作用域以及热更新配置。

### 2.5 启动多节点训练

训练应使用与校准不同的空闲 master port。每个节点都需要启动 launcher；该脚本
不会通过 SSH 自动启动其他节点。

```bash
# 节点 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml

# 节点 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh \
  tasks/train_vlm.py \
  /workspace/configs/my_train.yaml
```

如有集群调度器，应由调度器同时启动各节点。YAML 路径后可以继续追加普通
VeOmni 命令行覆盖项。launcher 会在 `torchrun` 前本地执行 `placemoe doctor`；
只有当集群调度器执行了等价检查时，才设置 `PLACEMOE_SKIP_PREFLIGHT=1`。

## 3. 适配其他模型或 VeOmni 分支

### 3.1 判断需要哪种适配

PlaceMoE 将模型与运行时拆分成两个边界：

1. VeOmni 负责模型、数据流水线、checkpoint 和 EP 执行。
2. PlaceMoE 通过 `MoEModelAdapter` 识别 routed expert 张量，并通过版本化 bridge
   控制 MoE runtime。

若模型已经采用 VeOmni 的标准 routed expert 表示，则不需要添加模型专用的
PlaceMoE 代码。标准表示要求：

- 张量首维为本地 expert 维度；
- expert 模块提供 `num_experts`；
- 参数采用融合的 `gate_up_proj`/`down_proj`，或拆分的
  `gate_proj`/`up_proj`/`down_proj`。

若模型使用另一种堆叠参数表示，需要实现并注册一个 `MoEModelAdapter`。如果模型
使用由独立 expert 模块组成、尚未堆叠的 `ModuleList`，则应先将它转换为 VeOmni
的 expert-stacked 表示；仅注册这个列表无法使其参数兼容槽位迁移和 grouped
expert kernel。

### 3.2 适配示例：openPangu-Ultra-MoE-718B

本节只是适配方案示例，不代表 PlaceMoE 已经验证该模型。公开的
[openPangu-Ultra-MoE-718B-V1.1](https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-V1.1)
配置使用 `pangu_ultra_moe` 模型类型、61 层、256 个 routed experts、top-8
routing 和 1 个 shared expert。其公开 remote code 使用独立 expert
`ModuleList`，并以 Transformers 4.48.2 为目标；本仓库则采用 Transformers
5.2.0、FSDP2 和 VeOmni 的堆叠 EP kernel。因此，不能在未进行 VeOmni 模型移植
的情况下，直接用 PlaceMoE 训练该 remote-code checkpoint。

建议按以下顺序进行生产级适配：

1. 按照 VeOmni Transformers v5 模型流程添加该模型。在 import 阶段注册
   `pangu_ultra_moe`，并让生成的模型代码基于固定的 Transformers 5.2.0 实现。
   不要手工修改模型 `generated/` 目录下的文件。
2. 实现普通 VeOmni EP parallel plan 和 token routing 路径。在启用 PlaceMoE
   前，先验证默认 VeOmni 训练。
3. 将 256 个独立 routed experts 转换为融合或拆分的堆叠 projection 表示，并
   实现 checkpoint 转换。shared expert 不应参与 routed expert 的放置和复制。
4. 若堆叠参数字段名与默认接口不同，实现 `placemoe.MoEModelAdapter` 的
   `matches`、`num_experts`、`expert_parameters`、
   `replace_expert_parameter` 和 `kernel_weights`，并在导入模型 package 时
   注册该 adapter。
5. 先验证模型加载和默认布局下的 forward/backward，再运行
   `placemoe prepare`、`placemoe doctor` 和短时间 2 节点自适应训练。测量性能前，
   检查 loss 一致性、布局安装、副本梯度 hook 执行以及 checkpoint 保存与恢复。

PlaceMoE planner 不需要随模型变化，因为它只接收逻辑路由、拓扑、槽位容量和
校准结果，而不依赖模型名称。完整 VeOmni 模型流程见
[新模型指南和检查清单](support_new_models/guide_and_checklist_zh.md)，公共 adapter
接口实现在 `placemoe/model_adapter.py`。

### 3.3 适配用户修改过的 VeOmni

若用户的 VeOmni 改动仅涉及模型注册、生成的模型代码、checkpoint 转换、数据
处理或 parallel plan，建议以 PlaceMoE release 为分支基线，再将这些模型提交
迁移到该分支。这样能够保留现有 PlaceMoE trainer 和 distributed hooks。

如果用户 fork 中包含自定义 trainer 或训练循环，则必须保留版本化
`MoERuntimeBridge` 生命周期。宿主训练框架至少需要：

- 在 EP process group 可用后配置 bridge；
- 在构建模型时扩展并绑定 expert slots；
- 创建 optimizer 后绑定 optimizer；
- 在 bridge 的 training-forward scope 内执行所有可训练 forward；
- 上报 step 和 microstep 边界；
- backward 后执行副本梯度同步；
- 在 step 结束时调用布局/映射更新和 metric hooks；
- 正确关闭 planner workers 和 process groups。

自定义 trainer 还必须保持 PlaceMoE checkpoint 和梯度范数语义。副本梯度 overlap
不会静默退化为阻塞同步；缺失必要 hook 会被视为适配错误。

任意上游 VeOmni checkout 不能只通过 import 一个 package 就自动支持 PlaceMoE，
因为上游 VeOmni 尚未暴露全部必需的生命周期 hook。不要使用 `sitecustomize` 或
runtime monkey patch。请按照[版本化 bridge 适配指南](placemoe_veomni_bridge_zh.md)，
将边界清晰且便于审查的少量宿主接口迁移到用户 fork。

如果用户的训练框架不是基于 VeOmni，还需要提供等价的 EP dispatch、专家堆叠
参数接口、checkpoint 语义，以及完整的 runtime bridge 生命周期。这属于训练框架
移植，而不是模型 adapter；当前插件不声明能够零修改接入任意训练框架。

## 4. 常见问题与建议

### 其他节点在容器中看不到 YAML

在每个节点上把宿主机配置目录挂载到 `/workspace/configs`。通过共享文件系统、
Git 或 `rsync` 同步宿主机文件，并在启动前比较 `sha256sum`。Docker 不能给已经
运行的容器新增 bind mount；若启动时漏掉挂载，需要重新创建容器。

### 容器没有使用修改后的源码

不要覆盖挂载 `/opt/placemoe`。将源码挂载到 `/workspace/PlaceMoE`，设置
`PYTHONPATH=/workspace/PlaceMoE:/opt/placemoe`，并按第 2.1 节检查实际导入路径。
修改源码后需要重启 Python 或训练任务。

采用上述挂载方式时，修改纯 Python 或 YAML 不需要重建镜像。修改依赖、
`pyproject.toml`、安装后的命令入口、native extension 或验证过的加速器软件栈后，
需要重建镜像。正式发布仍应记录唯一的源码 commit 和镜像 ID。

### NPU 设备文件存在，但 `torch.npu.is_available()` 为 false

仅看到 `/dev/davinci*` 并不表示容器拥有必要的 driver ioctl 和动态库。使用验证
过的 `--privileged` 启动方式，或使用站点提供的等价 Ascend runtime 策略，并
挂载宿主机 driver 和 firmware。加载镜像不会安装或升级宿主机 driver。

### 容器中没有 `npu-smi`

`npu-smi` 属于宿主机 driver，而不是 PlaceMoE。先在宿主机执行
`command -v npu-smi`，再按照第 1.5 节将该可执行文件挂载到容器。最终仍应以
torch-npu 能否识别并使用设备作为运行时判断依据。

### 校准命令看起来卡住了

声明的所有节点都必须启动相同命令。请检查：

- `NNODES`、`NPROC_PER_NODE`、`MASTER_ADDR` 和 `MASTER_PORT` 是否一致；
- 每个节点是否使用 `[0, NNODES)` 范围内唯一的 `NODE_RANK`；
- 所有容器是否能够访问 master address；
- `MASTER_PORT` 及后续 4 个端口是否空闲；
- 防火墙是否允许 HCCL 和 torch distributed 通信。

在两个节点都使用 `NODE_RANK=0` 不能替代缺失的节点 1。若启动了错误命令，先
检查确切的 torchrun/calibration 进程 PID，终止这些进程后再使用新的空闲端口重试。

### `placemoe doctor` 报告 calibration scope 字段缺失

该 JSON 是旧格式或手工创建的文件，不是当前 PlaceMoE calibration artifact。
运行 `placemoe prepare`，不要手工添加 scope 字段。若路径中已有不匹配文件，
请先确认路径；只有确定需要替换时，才使用相应的 force 参数。

### 看似有效的 artifact 被重新校准或拒绝

runtime calibration 的作用域包含加速器环境和 EP 拓扑。模型校准还会对模型、
代表性数据、执行设置、entrypoint 和 runtime artifact 生成 fingerprint。修改模型
kernel、依赖、拓扑或工作负载后，可能需要 `--force-model` 或
`--force-runtime`。所有节点必须看到字节完全一致的 artifact。

### HCCL 超时

检查 `--network host`、选用的网络接口、`MASTER_ADDR`、rank 数量、防火墙规则和
HCCL 连通性，并确认没有旧任务继续占用所选端口。

### 训练可以运行，但没有生成有效的副本布局

mapping-only 更新只改变 `M`，不能创建 expert copies。当 `initial_artifact` 为空
时，需要设置正数的 `layout_interval_steps` 并预留非零冗余槽位。检查训练日志中
的 planner 和 installation metrics。

### 模型或配置发生 OOM

优先降低 micro-batch size、序列长度、多媒体分辨率或冗余槽位容量。仅为了验证
系统接入时，可以使用具有代表性的部分 MoE 层，但不能将其性能与完整模型结果
直接比较。执行配置变化后需要重新进行模型校准。

### 建议的生产检查

开始长时间训练前：

1. 记录镜像 ID、PlaceMoE commit、YAML checksum、模型版本和校准文件哈希；
2. 在每个节点运行 `placemoe doctor`；
3. 完成一次短时间多节点 forward/backward 和 checkpoint 恢复测试；
4. 确认所有节点安装了相同的 `L` 和 `M`；
5. 通过正确性和梯度 overlap 检查后再测量性能。
