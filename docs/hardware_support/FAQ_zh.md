# Ascend NPU 常见问题与解决方法

[English](FAQ.md) | 中文

本文汇总在 Ascend NPU 上使用 VeOmni 和 PlaceMoE 时的常见问题与解决方法。

## 问：如何解决 NPU 内存碎片问题？

### 答：设置多 stream 内存复用环境变量

```bash
# 启用 NPU 多 stream 内存复用
export MULTI_STREAM_MEMORY_REUSE=2
```

这会启用 NPU 多 stream 内存复用，减少内存碎片并提高利用率。推荐值为 `2`。

> **说明：**`train.sh` 默认已经设置该环境变量。

## 问：如何配置多节点训练？

### 答：修改 `train.sh` 中的环境变量

下面是一个**双节点示例**，请根据实际集群规模调整：

```bash
# 节点总数，本例为 2
NNODES=${NNODES:=2}
# 当前节点序号；双节点取 0 或 1，每台机器必须不同
NODE_RANK=${NODE_RANK:=0}
# 主节点 IP；所有机器必须相同
MASTER_ADDR=${MASTER_ADDR:=192.168.1.100}
# 主节点通信端口
MASTER_PORT=${MASTER_PORT:=12345}
# 每节点 NPU 数；A2 最多 8 个，A3 最多 16 个
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
```

> **配置位置：**这些参数位于 `train.sh` 第 9–37 行。

参数含义：

- `NNODES`：集群节点总数；
- `NODE_RANK`：每个节点唯一的序号，范围为 0 到 `NNODES-1`；
- `MASTER_ADDR`：主节点 IP，所有机器保持一致；
- `MASTER_PORT`：通信端口，默认值适用于大多数环境；
- `NPROC_PER_NODE`：每节点 NPU 数量，A2 最多 8 个，A3 最多 16 个。

注意：

- 所有节点必须能够通过 `MASTER_ADDR:MASTER_PORT` 通信；
- 所有节点需要相同的配置文件和数据路径；
- 启动前确认节点间网络连通。

## 问：如何解决 `"'liger_kernel' is not supported on Ascend NPU"`？

### 答：在 YAML 中设置 `model.ops_implementation`

```yaml
model:
  ops_implementation:
    # Attention implementation
    attn_implementation: "flash_attention_2"
    # 可选："eager"、"sdpa"、"flash_attention_2"、
    # "flash_attention_3"、"flash_attention_4"、"native-sparse"

    # MoE implementation
    moe_implementation: "fused_npu"
    # 可选："eager"、"fused_npu"
```

> **配置位置：**训练 YAML；相关字段见 `arguments.md` 第 127–135 行。

NPU 优化算子包括：

- `npu_group_gemm`：MoE GroupGEMM 算子（`npu_group_gemm.py:1-114`）；
- `npu_rms_norm`：RMS normalization 算子（`npu_fused_operator.py:20-26`）；
- `npu_rotary_mul`：RoPE positional encoding 算子
  （`npu_fused_operator.py:28-52`）。

VeOmni 会检测 NPU 环境并使用相应算子。Attention 使用 SDPA 或 CANN 内置算子，
MoE 使用 `npu_group_gemm`。

## 问：如何解决 `global batch size should be a multiple of 8/16/32`？

### 答：正确设置 batch size

确保 global batch size 满足：

```text
global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps
```

注意：

- 如果未设置 `global_batch_size`，系统会按
  `micro_batch_size × dp_size` 自动计算；
- `global_batch_size` 必须能够被所有并行维度整除。

## 问：如何设置可见的 NPU 设备？

### 答：使用 `ASCEND_RT_VISIBLE_DEVICES`

```bash
# 只让 NPU 0、1、2、3 对进程可见
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

如果没有设置，系统会自动检测所有可用 NPU：

```bash
# 自动检测可用 NPU 数量
NPROC_PER_NODE=$(ls -l /dev/davinci* | grep -v "davinci_manager" | wc -l)
```

该变量的作用类似 CUDA 的 `CUDA_VISIBLE_DEVICES`，用于控制进程可见的 NPU。

## 问：如何解决 Transformers 版本不兼容？

### 答：使用兼容的 Transformers 版本

```bash
# 查看当前 Transformers 版本
python -c "import transformers; print(transformers.__version__)"

# 使用 uv 安装
uv sync --locked --extra npu --extra audio --group dev

# 或使用 pip 安装
pip install transformers==5.2.0
```

VeOmni 在 `pyproject.toml` 中固定使用 Transformers `5.2.0`。其他 v5 minor
版本可能可以运行，但未经过 CI 验证。
