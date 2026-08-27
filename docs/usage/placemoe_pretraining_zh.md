# PlaceMoE 预训练指南

[English](placemoe_pretraining.md) | 中文

PlaceMoE 通过 VeOmni 的版本化 MoE runtime bridge 集成。它沿用普通的模型、
数据、trainer、FSDP2 和分布式启动接口；新增配置只位于
`train.hiermoe.placemoe` 下。

## 1. 准备环境

验证过的 Ascend 软件栈使用 Python 3.11、CANN 9、PyTorch 2.9.0 和
torch-npu 2.9.0.post2。在该环境中，将 PlaceMoE 安装为 VeOmni extension：

```bash
python -m venv --system-site-packages .venv
uv pip install --python .venv/bin/python --no-deps --no-build-isolation .
source .venv/bin/activate
```

当前 aarch64 路径要求预先安装这套经过验证的加速器软件栈；PlaceMoE 不会在
干净宿主机中自动安装 CANN、PyTorch 或 torch-npu。`docker/ascend/` 下经过
验证的 PlaceMoE Dockerfile 遵循相同的插件边界：基础镜像提供加速器环境和
通用 VeOmni 依赖，uv 则构建并安装当前源码。

```bash
docker build -t placemoe:ascend -f docker/ascend/Dockerfile.placemoe_9.0.0_torch_npu2.9.0_910b.arm .
```

将验证过的镜像迁移至离线或多节点集群，请参阅
[PlaceMoE Ascend 镜像打包与分发](placemoe_image_distribution_zh.md)。加载镜像后
挂载 NPU、模型、数据集、配置和输出目录，请参阅
[从已验证 Ascend 镜像运行 PlaceMoE](placemoe_container_quickstart_zh.md)。

## 2. 配置一份 VeOmni 训练 YAML

将模型、数据集、分布式拓扑和 PlaceMoE 设置保存在同一个文件中。下面是完整的
PlaceMoE 配置接口：

```yaml
train:
  accelerator:
    ep_size: 16
    dp_shard_size: 16
  hiermoe:
    # 每个 EP rank 为额外副本预留的物理槽位容量。
    redundant_slot_increment_per_device: 4
    # 节点内通信组，以及完整 EP 通信组。
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
        failure_policy: raise
      resources:
        workers: 48
        candidate_workers: 4
        worker_threads: 1
```

该 preset 会自动启用分层 token deduplication、source-aware dispatch 和 step
boundary 安装；副本预算为正时还会启用 replica-gradient overlap。它会禁用历史
swap/cover planners。
其余参数都是部署输入，不能盲目复制：

- `ep_size`、`hierarchy_group_sizes` 和 slot capacity 描述集群与副本预算；
- `runtime_perf_model` 描述该拓扑上的 A2A 和 expert-state transfer；
- `calibration.artifact` 包含每个 planner 进程使用的通信与 expert-compute
  系数；
- 模型和数据集路径继续使用已有 VeOmni schema。

单节点配置应使用仅含 rank 层级的 `[8]`，并设置 `ep_size: 8`；多个同构节点使用
`[ranks_per_node, ep_size]`，例如 `[8, 16]`。系统支持将
`redundant_slot_increment_per_device` 设为 `0`：此时 PlaceMoE 仍会优化每个专家
唯一 base copy 的布局，但不执行 replica-gradient synchronization。

初始 `L,M` artifact 是可选的。若未提供，初始 mapping 会将请求路由到 canonical
owner，第一个完整更新再根据已采集 routes 构建布局；若副本预算为正，该更新还会
创建专家副本。因此，当
`initial_artifact` 为空时，`layout_interval_steps` 必须为正数。

## 3. 准备校准文件

PlaceMoE 有两个可复用的校准阶段。它们不会生成 `L` 和 `M`；runtime planner
会根据正式训练任务中采集的 routes 生成这两个决策。

| Artifact | 测量内容 | 可复用范围 |
| --- | --- | --- |
| `runtime_perf_model` | 分层 A2A、状态迁移和梯度同步 | 相同硬件、EP size、每节点 rank 数和通信层级 |
| `calibration.artifact` | 与模型相关的通信和 expert-compute 系数 | 相同模型、执行配置和拓扑 |

推荐命令会检查 YAML 中配置的两个路径，复用有效文件并只创建缺失文件。使用与
训练相同的分布式拓扑在每个节点启动；各节点只修改 `NODE_RANK`：

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe prepare \
  --config configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py
```

在单台 8-NPU 节点上，将 `NNODES=1`、`NODE_RANK=0`、
`NPROC_PER_NODE=8`，并配置 `hierarchy_group_sizes: [8]`。同一个
`prepare` 命令会完成 rank-only 拓扑和模型校准。

模型校准阶段默认运行 5 个默认布局 step：2 个 warm-up step、1 个拟合 step 和
2 个 held-out 验证 step。通过验证的 JSON 会写入每个节点上的配置路径。文本模型
使用 `tasks/train_text.py`。

缓存处理规则如下：

- 所有节点上有效且逐字节相同的 artifact 会被复用；
- 任一节点缺少 artifact 时，会在所有节点重新生成该阶段；
- 已存在但无效或作用域不匹配的 artifact 会中止准备过程，而不会被静默覆盖；
- `--force-runtime` 或 `--force-model` 会显式替换对应阶段。重新执行 runtime
  校准也会重新执行模型校准，因为后者的 provenance 记录了 runtime artifact
  的哈希值。

除模型和拓扑作用域外，缓存校验器还会为模型、代表性数据、执行设置和训练
entrypoint 生成 fingerprint。如果依赖或 kernel 升级没有体现在 YAML 或
entrypoint 中，请使用 `--force-model`。

### 分开运行两个阶段

若只校准集群拓扑，在每个节点运行：

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29501 \
  placemoe calibrate-runtime \
  --output calibration/runtime_perf_model.json \
  --hierarchy-group-sizes-csv 8,16
```

通信 benchmark 与模型和数据集无关。然后拟合模型相关的 planner costs：

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29502 \
  placemoe calibrate-model \
  --config configs/my_train.yaml \
  --entrypoint tasks/train_vlm.py \
  --runtime-perf-model calibration/runtime_perf_model.json \
  --output calibration/model_and_topology.json
```

两个独立命令会有意覆盖指定的输出文件；需要自动复用缓存时，应使用
`placemoe prepare`。当前可迁移的准备流程支持经过验证的两级 node-to-rank
层级，并要求任务恰好包含一个完整 EP group。为 torchrun、timing exchange 和
准备协调保留 `MASTER_PORT` 及其后续 4 个端口。

## 4. 启动前验证

在每个节点运行部署检查：

```bash
placemoe doctor --config configs/my_train.yaml
```

它会检查已验证的软件栈、可见 NPU 和 CANN、模型与数据路径、runtime 和 planner
校准文件、副本容量、更新计划以及 canonical PlaceMoE preset。warning 不会阻止
启动，但必须先解决所有 `FAIL`。

## 5. 在每个节点启动

launcher 不负责 SSH 编排。它会在 torchrun 前执行本地部署检查，因此每个节点
必须使用相同的仓库 revision、训练 YAML、模型、数据集和校准文件：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh tasks/train_vlm.py configs/my_train.yaml

# 节点 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.0.10 MASTER_PORT=29500 \
  scripts/placemoe/launch_npu.sh tasks/train_vlm.py configs/my_train.yaml
```

文本模型使用 `tasks/train_text.py`。YAML 路径后可以继续添加 VeOmni 命令行覆盖
参数。只有当集群调度器已执行等价检查时，才能设置
`PLACEMOE_SKIP_PREFLIGHT=1`。

## 6. 静态与自适应模式

`L` 和 `M` 使用相互独立的更新计划：

| Layout interval | Mapping interval | 行为 |
| ---: | ---: | --- |
| `0` | `0` | 保持预加载的 `L,M` 不变。 |
| `100` | `0` | 每 100 步重新计算并安装 `L` 和 `M`。 |
| `0` | `20` | 保持 `L` 不变，只更新 lookup table `M`。 |
| `100` | `20` | 每 20 步更新 `M`，每 100 步更新两个决策。 |

当两个事件同时到期时，完整更新会包含 mapping-only 更新。任一时刻最多运行一个
CPU planner；训练继续使用当前决策，后续事件会在 planner 运行期间合并。完成的
artifact 必须通过 schema 验证后才能安装。Mapping-only 更新不移动 expert state；
完整更新会预先验证所有层的 artifact，迁移参数和 optimizer states，并在训练 step
边界安装 `L` 和 `M`。

验证和生产发布建议使用 `failure_policy: raise`。只有在自适应更新失败时明确
希望保留当前有效决策并继续训练，才使用 `continue`。

## 7. 适配其他 MoE 模型

PlaceMoE 不根据模型名称分支。默认 adapter 支持首维索引本地 expert slots，且
提供以下任一参数布局的 expert 模块：

- fused `gate_up_proj` 和 `down_proj`；
- split `gate_proj`、`up_proj` 和 `down_proj`。

这些接口覆盖已验证的 Qwen3-VL 和 DeepSeek-V3 配置。采用其他参数布局的模型
需要注册一个 `MoEModelAdapter`，返回按 expert 堆叠的参数和规范化 fused-kernel
weights：

```python
from placemoe import register_moe_model_adapter

register_moe_model_adapter(MyModelAdapter())
```

因此，已经通过 VeOmni 标准模型和 EP 接口集成、且使用上述任一默认 expert 表示
的模型，无需 PlaceMoE 专用修改。如果模型绑定时没有 adapter 能够匹配，PlaceMoE
会立即失败；如果必要的 replica-gradient hooks 没有在第一次 backward 中执行，
PlaceMoE 也会失败，而不会静默选择阻塞式副本同步。

planner 不随模型改变，因为它只接收逻辑 routes、拓扑、容量和校准结果，而不依赖
模型专用模块。

## 兼容性与已知边界

- 静态 artifact 必须匹配模型 layer keys、expert 数量、EP size、每节点 rank 数和
  每 rank slot 数；
- 副本放置当前要求 `ep_fsdp_size=1`，且不支持 FSDP2 CPU offload；
- mapping-only 更新计划需要初始布局中已经存在有用的副本，不能自行创建副本；
- 历史 `VEOMNI_PLACEMOE_CONFIG` 和 `VEOMNI_HIERMOE_*` 控制项仅为归档 launcher
  和论文复现实验保留。文件形式的 legacy input 还要求
  `VEOMNI_PLACEMOE_USE_LEGACY_CONFIG=1`，否则使用 inline PlaceMoE block。
  legacy `config_path` 不能与 inline fields 混用；
- 可选的 golden parity test
  `tests/distributed/test_placemoe_planner_parity.py` 可使用外部提供的 EP32 和
  EP64 reference artifacts 对比生成的布局与映射。
