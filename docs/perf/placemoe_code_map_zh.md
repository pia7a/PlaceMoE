# PlaceMoE 代码导览

[English](placemoe_code_map.md) | 中文

本文是 PlaceMoE 重构的代码结构基准，区分论文算法、训练 runtime、保留的基线和
历史 profiling artifact。在所有调用方完成迁移之前，重构必须保持已有 runtime
artifact schema 兼容。

## Canonical PlaceMoE 流程

论文针对每个 MoE 层描述了以下数据流：

1. 采集 token-level routing snapshots；
2. 计算 source-conditioned expert demand 和 co-selection affinity；
3. 保留一个满足精确预算的有界副本分配候选集合；
4. 构造满足容量约束的 node-to-rank 物理布局 `L`；
5. 优化 source-aware token-to-copy mapping `M`；
6. 交替优化 `L` 和 `M`，并选择 exact route cost 最低的组合；
7. 序列化所选组合，用于静态预加载或热更新。

可复用实现位于 `veomni/distributed/moe/hiermoe/placemoe/`：

| 模块 | 论文职责 |
| --- | --- |
| `statistics.py` | Source-conditioned demand 和 co-selection affinity。 |
| `allocation.py` | 满足精确预算的有界副本分配候选集合。 |
| `partition.py` | 经校准的容量约束 affinity partitioning。 |
| `placement.py` | 通用 node-to-rank placement、community-coherent node proposals、locality matching 和 rank repair。 |
| `mapping.py` | Demand-ordered 初始化、经校准的 entry update 和 source-community block mapping。 |
| `optimizer.py` | 有界 layout--mapping 交替优化和 exact-cost callback。 |
| `materialize.py` | 物理 slot assignment 和 mapping relocation。 |
| `artifacts.py` | 经验证的 schema-v2 runtime artifacts。 |
| `seeds.py` | 确定性可行 seeds，避免相对默认布局的不必要回退。 |

Canonical CLI 是 `scripts/profile/plan_placemoe.py`，底层由
`scripts/profile/placemoe_planner.py` 实现。旧名称
`build_hiermoe_recursive_classifier_layout.py` 是为尚未迁移的外部脚本保留的
deprecated wrapper。

CLI 保留论文的 calibrated partition candidate，同时还评估两类 proposal，以防
exact-route search 受到 pairwise surrogate error 影响。Scale-normalized partition
candidate 提供稳健的通用 fallback。默认 community proposal 依据 affinity 对
experts 进行粗粒度分组，通过平衡且拓扑通用的 copy permutation 放置完整
communities，并使用精确 token destination unions 联合映射每个
source-node/community block。当某个 allocation 保留完整 communities 时，它也
支持部分副本预算；否则退回通用 placement。

旧的四节点 `structured_degree2` library 只有在指定
`--include-legacy-structured-candidates` 时才启用；历史 token-KMeans hyperedge
candidate 则通过 `--include-legacy-hyperedge-candidates` 启用。所有成功运行都
输出相同的预加载 schema-v2 artifact，可同时用于静态启动和热更新。

Partition restarts 使用与规模无关的 load weights 来生成多样化 spectral seeds。
由于 pairwise affinity 无法恢复 top-k destination-group union，calibrated paper
path 与 scale-normalized compatibility path 被保留为不同 proposal branches。
paper branch 同时评估跟随 movable copies 迁移的 mapping 和重新执行 demand-ordered
初始化得到的 mapping，随后应用 calibrated coordinate update。compatibility
branch 从 normalized placement 开始，使用论文时期的确定性交换顺序和新 mapping，
评估固定的 normalized mapping tradeoffs，并在各轮间传播其 exact-cost incumbent。
最终 `L,M` 始终只由完整 token-route replay 选择。

当预算恰好是每个 expert 增加一个副本时，default-order uniform plan 会作为可行的
safety seed 保留；只有当其 profiled joint cost 更低时才会胜出。

## PlaceMoE 保留的 runtime 模块

| 模块 | 职责 |
| --- | --- |
| `all_to_all.py` | 分层 token-deduplicated dispatch 和 combine。 |
| `routing.py` | Duplicate-free 与 assignment-load accounting。 |
| `perf_model.py` | 通信和 expert-compute 校准。 |
| `state.py` | Trainer 集成、checkpoint state 和 lifecycle。 |
| `placemoe/runtime/` | 类型化配置、独立的 layout/mapping 调度、canonical planner-process 构造。 |
| `expert_swap.py` | 布局安装、状态迁移、梯度同步和历史 swap/cover 方法的兼容 runtime。Canonical 热更新从 `placemoe/runtime/` 进入，调用 canonical CLI，且只接受通过验证的 schema-v2 artifact。 |
| `online_lut_planner.py` | 在 `L` 中已有副本之上的 mapping-only 更新。 |
| `metrics.py` | 暴露给 trainer 的 runtime metrics。 |

## 保留的基线和实验 planners

以下路径不是 canonical PlaceMoE optimizer，但仍用于比较或诊断：

- fixed/uniform expert replication；
- EPLB 生成的静态布局；
- online expert swapping 和 greedy swap/cover planners；
- HierMoE placement baseline；
- mapping-only online LUT refinement。

除非通过明确记录的兼容或评估接口，否则 canonical offline optimizer 不得导入
这些实现。

论文 launcher 将这些路径作为不同方法显式暴露。它们共享 runtime substrate，
但不共享 optimizer：

| 方法 | 角色 | Layout 或 mapping 来源 |
| --- | --- | --- |
| `ours` | Canonical 静态 PlaceMoE | `plan_placemoe.py` 生成的、通过验证的 schema-v2 `L,M` artifact。 |
| `ours_full_replan` | PlaceMoE 热更新的论文 launcher 兼容名称 | 将近期 routes 传给 `plan_placemoe.py`，迁移并原子安装通过验证的 artifact。 |
| `ours_online_lut` | Mapping-only 诊断方法 | 保持 `L` 不变，只更新 runtime LUT。 |
| `r2` | Fixed-replication baseline | 按默认 expert 顺序均匀镜像副本。 |
| `eplb` | Placement baseline | 公共 runtime 上由外部生成的静态 placement。 |
| `hiermoe` | Communication-oriented placement baseline | Legacy exact-P1 selector。 |

`expert_swap.py` 中的 online swap/cover selectors 只为比较和恢复保留。它们不会被
`placemoe/` 导入，也不会由 canonical CLI 启用。

## 历史文件和生成文件

生成的 `.paper32_*_launcher.sh` snapshots 已从版本控制中删除并加入 ignore。它们
是 `launch_hiermoe_greedy_e2e_4node.sh` 的可复现输出，而不是源码。

只有在 reference search 表明没有 canonical launcher、测试或文档依赖时，才会
删除历史 benchmark、绘图和诊断脚本。Git 历史仍是已删除 artifact 的恢复路径。

## 重构前基线

仓库从 `master` 分支的 commit `ff5f980` 开始重构，当时工作区干净，并比
`origin/master` 多 4 个本地 commit。在 Python 3.11 NPU 验证容器中，目标
planning 测试基线为：

```text
58 passed, 1 failed in 47.48s
```

已有失败位于 `tests/distributed/test_core_moe_planner.py` 的
`test_fused_path_skips_eager_scoring_reduces_once_and_reuses_physical_routes`：
reducer 收到 `(2, 2)` shape，而测试期望 `(2, 7)`。重构不能引入更多失败；该
基线失败与 PlaceMoE 提取分开跟踪。

## 最小完成证据

完成后的 CPU、offline、EP64 静态 A/B 和 EP64 热更新结果记录在
`docs/perf/placemoe_refactor_validation.md`。

只有同时满足以下条件，重构才算完成：

- canonical API 验证副本预算、slot capacity、`L` 和 `M`；
- 最终候选由 exact route replay 选择；
- focused CPU 测试达到或超过记录的基线；
- same-runtime EP64 对比表明优化组合相对 uniform replication 改善 measured
  joint behavior；
- 一次 EP64 热更新在不重启训练且不产生无效 loss 的情况下安装新的 `L` 和 `M`。
