# 将 PlaceMoE 集成到修改过模型的 VeOmni 分支

[English](placemoe_veomni_bridge.md) | 中文

PlaceMoE 将模型集成与 MoE runtime 集成分离。VeOmni 宿主通过版本化的
`veomni.moe_runtime_bridges` entry point 调用 MoE provider；PlaceMoE 则通过
公共 model-adapter API 发现 expert 模块。

## 推荐流程

如果用户分支的修改仅涉及模型注册、生成的模型代码、checkpoint 转换或
parallel plan：

1. 以一个已发布的 PlaceMoE commit 作为分支基线；
2. 将仅涉及模型的 VeOmni commit cherry-pick 或 rebase 到该基线上；
3. 保留模型原有的 VeOmni 注册和 EP parallel plan；
4. 在训练 YAML 中启用 PlaceMoE；
5. 在分布式训练前运行 `placemoe doctor`。

这样无需修改用户模型的 forward 或训练循环，同时能把 PlaceMoE 宿主 hooks
保留为一个版本化、易于审查的集成边界。

任意上游 VeOmni checkout 无法仅通过零改动 import 自动支持 PlaceMoE，因为
VeOmni 当前尚未暴露 PlaceMoE 所需的全部 dispatch、step boundary、checkpoint
和 backward overlap hooks。不要使用 `sitecustomize`、runtime monkey patch 或
隐式 fallback 来模拟这些 hooks。

## 自动模型支持

PlaceMoE 自动支持 VeOmni 标准 fused MoE 接口使用的两种 expert 表示：

- `gate_up_proj` 和 `down_proj`；
- `gate_proj`、`up_proj` 和 `down_proj`。

每个张量都必须在首维按本地 expert slot 堆叠，expert 模块还必须提供
`num_experts`。该约定覆盖已验证的 Qwen3-VL 和 DeepSeek-V3 路径，也适用于采用
相同 VeOmni 接口的新模型。

采用其他表示的模型需要实现 `MoEModelAdapter`，并在模型 package 中注册一次：

```python
from placemoe import register_moe_model_adapter

register_moe_model_adapter(MyModelAdapter())
```

adapter 只暴露 expert 参数和规范化的 fused-kernel weights；其中不包含放置、
映射、通信或 planner 逻辑。

## 严格启动约定

遇到以下情况时，PlaceMoE 会给出可操作的错误并停止：

- 宿主 bridge API 版本不兼容；
- 没有 model adapter 能匹配 expert 模块；
- expert slot 维度不一致；
- 无法注册 replica-gradient hooks；
- backward 路径没有执行所有必需 hooks。

Replica-gradient overlap 绝不会静默退化为阻塞同步。因此，不完整的模型或宿主
集成会在影响性能或训练语义之前暴露出来。

## 验证

在每个节点运行：

```bash
placemoe doctor --config configs/my_train.yaml
```

`doctor` 会同时检查 bridge provider 与 API 版本、软件栈、拓扑相关校准文件、
路径、副本容量和热更新计划。随后运行一个短时间双节点任务，并在测量端到端性能
之前，将生成的布局和映射 artifact 与已知 route fixture 对比。
