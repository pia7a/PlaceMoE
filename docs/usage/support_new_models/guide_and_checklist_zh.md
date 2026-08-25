# 新模型支持指南与检查清单

[English](guide_and_checklist.md) | 中文

**概述：**VeOmni 在 HuggingFace 模型之上叠加 FSDP、序列并行（SP）、专家并行
（EP）和 fused kernels。本文按模型类型给出集成步骤与检查清单。完整示例见：

- [qwen3_vl_example.md](./qwen3_vl_example.md)：VLM + MoE（图像/视频、deepstack、EP）；
- [qwen3_omni_moe_example.md](./qwen3_omni_moe_example.md)：全模态 MoE（图像/视频/音频、talker）。

> **适用范围：**VeOmni 当前固定使用 `transformers==5.9.0`，并在
> `veomni/models/transformers/<model>/generated/` 中保存 patchgen 生成的模型
> 文件。本文最初基于的 runtime monkey-patch 流程已经废弃。注册、parallel
> plan、多模态 data transform、trainer 接线和测试等高层检查清单仍然适用；
> 但下文的 modeling patch 步骤应理解为描述生成文件最终完成的工作，实际修改
> 应写入 `<model>_gpu_patch_gen_config.py`。patchgen 的完整操作见
> [patchgen 设计文档](../../design/patchgen.md) 和
> `veomni-migrate-transformers-v5` agent skill。

---

## 不同模型类型的集成复杂度

| 模型类型 | 所需文件 | 主要新增内容 |
| --- | --- | --- |
| Dense 纯文本 LLM | `__init__.py` | SP position embedding slicing |
| VLM（图像/视频） | `__init__.py` + `modeling_*.py` | FSDP dummy forward、ViT 与 LM 中的 SP、position ID function |
| 全模态 MoE | `__init__.py` + 另外 4 个文件 | 上述全部内容，加 audio encoder、fused MoE、EP plan、processor patch |

---

## 分步集成

### 第 0 步：理解目标模型

编写 VeOmni 代码前，先回答：

1. `config.json` 中的 `model_type` 是什么？它将作为 registry key；
2. `config.json` 中的 `architectures[0]` 是什么？它决定模型类；
3. `processor_config.json` 中的 processor class 是什么？它将作为
   `MODEL_PROCESSOR_REGISTRY` key；
4. 是否为 MoE？如果是，需要 `parallel_plan.py`；
5. 是否为多模态（图像/视频/音频）？如果是，需要 processor patch 和 data
   transform；
6. 是否使用多模态 RoPE？如果是，需要 `get_position_id_func`。

### 第 1 步：创建模型目录

```bash
mkdir veomni/models/transformers/your_model_name/
touch veomni/models/transformers/your_model_name/__init__.py
# 复杂模型还需要：
touch veomni/models/transformers/your_model_name/modeling_your_model_name.py
touch veomni/models/transformers/your_model_name/configuration_your_model_name.py  # 需要修正 config 时
touch veomni/models/transformers/your_model_name/processing_your_model_name.py    # 多模态模型
touch veomni/models/transformers/your_model_name/parallel_plan.py                 # MoE 模型
```

### 第 2 步：注册模型（`__init__.py`）

**最小纯文本模型：**

```python
from ...loader import MODELING_REGISTRY

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from transformers.models.your_model import YourModelForCausalLM
    return YourModelForCausalLM
```

**完整多模态 MoE：**

```python
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY

@MODEL_CONFIG_REGISTRY.register("your_model_type")
def register_config():
    from .configuration_your_model import YourModelConfig, apply_veomni_patch
    apply_veomni_patch()
    return YourModelConfig

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from .modeling_your_model import YourModelForCausalLM, apply_veomni_patch
    apply_veomni_patch()
    return YourModelForCausalLM

@MODEL_PROCESSOR_REGISTRY.register("YourModelProcessor")  # processor_config.json 中的准确类名
def register_processor():
    from .processing_your_model import YourModelProcessor, apply_veomni_patch
    apply_veomni_patch()
    return YourModelProcessor
```

> **Registry key 规则：**
>
> - `MODELING_REGISTRY` 和 `MODEL_CONFIG_REGISTRY` 使用 `config.json` 中的
>   `model_type`；
> - `MODEL_PROCESSOR_REGISTRY` 使用 `processor_config.json` 中的 Python 类名
>   字符串。

### 第 3 步：加入 package 的 `__init__.py`

在 [veomni/models/transformers/__init__.py](../../../veomni/models/transformers/__init__.py)
中加入模块：

```python
from . import (
    # ... 现有模型 ...
    your_model_name,  # 新增
)
```

### 第 4 步：Patch 模型（`modeling_*.py`）

标准模式是将 HF module 作为别名导入，定义 patches，并在最后应用：

```python
import transformers.models.your_model.modeling_your_model as hf_your_model

# ... 定义 patches ...

def apply_veomni_patch():
    hf_your_model.YourClass.method = patched_method
```

具体 patch 取决于模型类型，参见后面的检查清单。各 patch 的实现细节见示例文档。

### 第 5 步：定义专家并行方案（仅 MoE，`parallel_plan.py`）

```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj":   Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})
```

> **查找正确路径：**在未 patch 的 HF 模型上运行
> `for name, _ in model.named_parameters(): print(name)`。

### 第 6 步：Patch processor（仅多模态，`processing_*.py`）

两个常见问题：

1. HF 检查 `if audio is not None:`，而 VeOmni 在没有输入时传入 `[]`，因此应
   覆盖为 `if audio:`；
2. keyword argument 不匹配（`audios=` 与 `audio=`），需要与
   `data_transform.py` 实际传入的名称一致。

### 第 7 步：编写 data transform function

在 [veomni/data/data_transform.py](../../../veomni/data/data_transform.py)
中添加 `process_sample_your_model()`。完整函数签名和处理步骤见示例文档。

### 第 8 步：接入 trainer

修改 [veomni/trainer/vlm_trainer.py](../../../veomni/trainer/vlm_trainer.py)。在
`build_model_assets`、`build_data_collate_info`、`build_data_transform` 中加入
新模型类型；根据需要修改 `freeze_module` 或 `build_param_groups`。

### 第 9 步：添加配置文件

创建 `configs/multimodal/your_model/your_model.yaml`，设置
`model.config_path`、`model.attn_implementation`、
`model.moe_implementation`、`train.sp_size` 和 `train.ep_size`。

### 第 10 步：测试

需要添加的测试见下方检查清单。

---

## Patch 速查表

| Patch | 文本 LLM | VLM | 全模态 MoE |
| --- | :---: | :---: | :---: |
| `tie_word_embeddings` config 修正 | 有时 | 有时 | ✓ |
| FSDP dummy forward | — | ✓ | ✓（ViT + Audio） |
| SP：LM position embedding slicing | ✓ | ✓ | ✓ |
| SP：ViT pad+slice | — | ✓ | ✓ |
| SP：`cu_seqlens` padding entry | — | ✓ | ✓ |
| SP：ViT-to-LM fill-back | — | ✓ | ✓ |
| SP：deepstack all-gather | — | 使用 deepstack 时 | ✓ |
| Fused MoE + stacked weights | — | MoE 模型 | ✓ |
| Flash-attn kwargs pop/restore | — | ✓ | ✓ |
| 预计算 `max_seqlen` | — | ✓ | ✓ |
| Position ID transposition | — | ✓ | ✓ |
| `ForCausalLMLoss` | ✓ | ✓ | ✓ |
| `get_position_id_func` | — | ✓ | ✓ |

具体 patch 实现见示例文档。

---

## 检查清单

### 所有新模型

- [ ] `veomni/models/transformers/your_model/__init__.py` 使用
  `@MODELING_REGISTRY.register`
- [ ] 更新 `veomni/models/transformers/__init__.py`

### VLM（图像/视频）

- [ ] ViT encoder 中提供 FSDP `dummy_forward`
- [ ] ViT 中执行 SP `sp_pad_and_slice`，并使用正确 `pad_scale`
- [ ] 为 SP 添加 `cu_seqlens` padding entry
- [ ] SP ViT-to-LM fill-back（`gather_seq_scatter_heads` /
  `gather_heads_scatter_seq`）
- [ ] 使用 VeOmni token ID constants 的 `get_position_id_func`
- [ ] 在 `data_transform.py` 中添加 `process_sample_*`，并在 `VLMTrainer`
  中添加 `build_data_transform`

### MoE 模型

- [ ] `parallel_plan.py` 使用正确 expert weight paths
- [ ] 在 pretrained model base class 上接入 `get_parallel_plan`
- [ ] Stacked-weight `YourModelExperts` 模块与 `fused_moe_forward`
- [ ] 将 `_moe_implementation` 从顶层 config 传入 text sub-config
- [ ] 针对堆叠 expert 参数 patch `_init_weights`

### 全模态模型（音频）

- [ ] Audio encoder 中提供 FSDP `dummy_forward`
- [ ] Audio encoder 中执行 SP gather/slice（`gather_outputs` +
  `slice_input_tensor`）
- [ ] data transform 中提供 `audio_mask`；`build_data_collate_info` 中提供
  `audio_feature_lengths`
- [ ] Patch processor，使用 `if audios:` truthy check

### 所有模型的测试

- [ ] 在 `tests/toy_config/your_model_toy/` 添加 toy config
- [ ] 在 `veomni/data/dummy_dataset.py` 添加 `DummyYourModelDataset`（多模态）
- [ ] 在 `tests/models/utils.py` 添加 `MODEL_TO_DATASET` entry
- [ ] 在 `tests/models/test_models_patch.py` 的 `test_cases` 中添加
  `pytest.param`（Level 1）
- [ ] 在 `tests/e2e/test_e2e_parallel.py` 添加 case、fixture 和 test function
  （Level 2）
- [ ] 对 VLM，将 toy config 加入 `tests/models/test_vlm_trainer.py` 的
  `freeze_vit` smoke test list

---

## 常见问题

| 现象 | 可能原因 | 解决方法 |
| --- | --- | --- |
| backward 时 NCCL hang | ViT/AudioEncoder 缺少 `dummy_forward` | 当输入为 `None` 时，在启用 `fsdp_enabled` 的 rank 上添加并调用它 |
| ViT attention shape mismatch | SP 缺少 `cu_seqlens` padding entry | SP 启用时追加 `cu_seqlens[-1] + pad_seq_len` |
| `masked_scatter` size error | 在 SP-sliced layout 中执行 fill-back | fill-back 前调用 `gather_seq_scatter_heads` |
| `tie_word_embeddings` crash | config 默认值为 `True`，但没有 `get_output_embeddings` | Patch config，设置 `tie_word_embeddings=False` |
| 多样本 batch 中 position IDs 错误 | `(bs, 3, L)` 未转置为 `(3, bs, L)` | 在 model forward 中添加转置检查 |
| 音频输入被静默忽略 | 空列表 `[]` 仍满足 `if audio is not None:` | 在 processor 中改成 `if audio:` |
| EP 未生效 | `parallel_plan` 中的 expert weight paths 不匹配 | 在模型上运行 `named_parameters()` 核对准确路径 |
| Fused MoE 输出错误 | 权重 shape/transpose 不匹配 | 验证 `(num_experts, out, in)` 约定，并检查 `.contiguous()` |

---

## 关键 imports

```python
from veomni.distributed.parallel_state import get_parallel_state

from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,   # (bs, seq, h//sp) → (bs, seq//sp, h)
    gather_outputs,             # 沿指定维度 all-gather，不带 autograd
    gather_seq_scatter_heads,   # (bs, seq//sp, h) → (bs, seq, h//sp)
    slice_input_tensor,         # 沿指定维度为当前 SP rank 切片
    sp_pad_and_slice,           # pad 至 pad_scale 的倍数后切片
    unpad_tensor,               # 移除 tensor padding
)
from veomni.distributed.sequence_parallel.ulysses import _Gather  # 带 autograd 的 all-gather

from veomni.ops import fused_moe_forward
from veomni.ops.kernels.cross_entropy import ForCausalLMLoss

from veomni.utils.constants import (
    AUDIO_INPUT_INDEX,   # input_ids 中音频的 placeholder token ID
    IGNORE_INDEX,        # -100，label mask value
    IMAGE_INPUT_INDEX,   # input_ids 中图像的 placeholder token ID
    VIDEO_INPUT_INDEX,   # input_ids 中视频的 placeholder token ID
)
```
