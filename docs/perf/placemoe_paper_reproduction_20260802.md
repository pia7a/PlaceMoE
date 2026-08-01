# PlaceMoE 论文结果复现记录（2026-08-02）

## 目标与口径

本轮在 8 个 Ascend 910B 节点上验证重构后的 canonical PlaceMoE 是否能复现论文结果。实验不运行 Qwen3.5-VL，也不重跑 GPU case。每个可执行 case 仅比较：

- VeOmni；
- Replication：固定 2 份副本、hierarchical token-deduplicated A2A、blocking replica-gradient synchronization；
- canonical PlaceMoE：当前 planner 生成的静态 `L` 和 `M`，hidden replica-gradient synchronization。

完整训练均运行 20 steps，并统计 steps 11--20。DeepSeek-V3 使用 6 个 MoE layers、`max_seq_len=15360`、micro-batch 2、global batch 256 和 `lr=0`。本轮 runner 为 `scripts/profile/run_placemoe_paper_reproduction.sh`。

## 执行矩阵

| Case | 论文展示 | 状态 |
|---|---:|---|
| EP32 Qwen3-VL / ShareGPT4V / 16K tokens/rank | 是 | 3 个方法全部完成 |
| EP64 Qwen3-VL / Tulu-3 / 16K tokens/rank | 是 | 3 个方法全部完成 |
| EP64 DeepSeek-V3-6MoE / Tulu-3 / about 30K tokens/rank | 是 | 2 次 profile 均被 NPU 507035 阻塞 |
| EP64 Qwen3-VL / ShareGPT4V / 16K tokens/rank | 否 | 3 个方法全部完成 |
| EP64 Qwen3-VL / ShareGPT4V / 32K tokens/rank | 否 | 2 次 profile 未完成；最终尝试被 NPU 507035 阻塞 |

## 新运行的绝对结果

以下均为 10 个 steady-state samples 的均值，单位为 ms。A2A 为 forward 与 backward A2A 之和。

| Case | Method | E2E | A2A | Expert compute | Gradient mode |
|---|---|---:|---:|---:|---|
| EP32 ShareGPT4V 16K | VeOmni | 31017.165 | 24630.138 | 1162.795 | N/A |
|  | Replication | 16281.943 | 5704.716 | 1020.293 | blocking |
|  | PlaceMoE | 15808.165 | 5791.090 | 1026.881 | hidden |
| EP64 Tulu-3 16K | VeOmni | 36282.243 | 31198.553 | 1293.512 | N/A |
|  | Replication | 20044.644 | 11076.071 | 1125.259 | blocking |
|  | PlaceMoE | 19110.537 | 10242.552 | 1117.431 | hidden |
| EP64 ShareGPT4V 16K | VeOmni | 45699.716 | 39183.417 | 1284.242 | N/A |
|  | Replication | 22622.125 | 12071.070 | 1171.610 | blocking |
|  | PlaceMoE | 22331.459 | 11957.808 | 1156.096 | hidden |

## 与论文结果的比较

“偏差”按 `新加速比 / 论文加速比 - 1` 计算。

| Case | Method | E2E speedup 新/论文 | 偏差 | A2A speedup 新/论文 | 偏差 | Expert speedup 新/论文 |
|---|---|---:|---:|---:|---:|---:|
| EP32 ShareGPT4V 16K | Replication | 1.905 / 1.862 | +2.33% | 4.318 / 4.193 | +2.96% | 1.140 / 1.126 |
|  | PlaceMoE | 1.962 / 2.329 | -15.74% | 4.253 / 6.939 | -38.71% | 1.132 / 1.147 |
| EP64 Tulu-3 16K | Replication | 1.810 / 1.807 | +0.19% | 2.817 / 2.810 | +0.25% | 1.150 / 1.157 |
|  | PlaceMoE | 1.899 / 2.121 | -10.47% | 3.046 / 3.845 | -20.78% | 1.158 / 1.257 |
| EP64 ShareGPT4V 16K | Replication | 2.020 / 2.028 | -0.38% | 3.246 / 3.264 | -0.56% | 1.096 / 1.092 |
|  | PlaceMoE | 2.046 / 2.338 | -12.48% | 3.277 / 4.228 | -22.50% | 1.111 / 1.139 |

PlaceMoE 相对相同 runtime 的 Replication 的 E2E 加速分别为：

| Case | 新结果 | 论文结果 |
|---|---:|---:|
| EP32 ShareGPT4V 16K | 1.030x | 1.251x |
| EP64 Tulu-3 16K | 1.049x | 1.174x |
| EP64 ShareGPT4V 16K | 1.013x | 1.153x |

## Planner 诊断

canonical planner 在 held-out complete routes 上预测的 PlaceMoE / mirrored-Replication joint-cost speedup 只有：

| Case | Held-out predicted speedup | Planner wall time | Exact route evaluations | Uniform-seed winners |
|---|---:|---:|---:|---:|
| EP32 ShareGPT4V 16K | 1.009 | 125.2 s | 438 | 40 / 48 layers |
| EP64 Tulu-3 16K | 1.027 | 471.7 s | 818 | 30 / 48 layers |
| EP64 ShareGPT4V 16K | 1.007 | 471.4 s | 852 | 42 / 48 layers |

这些预测与实际仅 1.013--1.049x 的 PlaceMoE / Replication E2E 收益方向一致。因此，差距不是运行时没有执行当前布局，也不是计时器失真；当前候选生成与搜索本身没有找到论文历史布局所达到的通信收益。3 个 case 中有 112 / 144 个 layer 最终选择 `placemoe_uniform_seed`，进一步表明当前 affinity-aware candidate 的有效性不足。

## 被阻塞的 case

### DeepSeek-V3

配置日志确认 `lr=0.0`。2 次独立 profile 均出现 NPU vector-core exception，runtime result 为 `507035`；第二次在 `huawei2_node4` 的 ranks 56--61 等 rank 上复现。错误发生在完整 route capture 完成之前，因此不能安全生成新布局，也不能给出 20-step 三方法结果。

证据：仓库上一级目录中的 `paper64_profile_huawei12_deepseekv3_6moe_half_tulu3_p4_canonical_repro_20260802_v1_ep64_deepseekv3_tulu3_30k_rank*.host.log`。

### Qwen3-VL ShareGPT4V 32K/rank

第一次 profile 的 rank-0 launcher 非零退出，其余 7 个 rendezvous 进程已精确清理。随后增加节点 SSH 启动错峰并更换端口；第二次确认 8 个节点各有且仅有 1 个 torchrun parent，但 `huawei1_node1` local rank 4 在 `HcclAlltoAllV` 期间出现 SDMA/vector-core exception，runtime result 为 `507035`。其余进程已清理并复核 8 节点残留均为 0。

证据：仓库上一级目录中的 `paper64_profile_huawei12_qwen3vl30b_sharegpt4v_p4_canonical_repro_20260802_v1_ep64_qwen3vl_sharegpt4v_32k_rank*.host.log`。

## 结论

1. **训练 runtime、计时口径和 Replication 基线可以复现。** 3 个成功 case 的 Replication E2E/A2A speedup 与论文差异均不超过 3%；两个 EP64 case 均小于 1%。
2. **当前 canonical PlaceMoE 仍然有效，但没有达到论文性能。** 它相对 VeOmni 提供 1.899--2.046x E2E 和 3.046--4.253x A2A speedup，也在 3 个 case 中均略快于 blocking Replication；但相对论文，E2E speedup 低 10.47%--15.74%，A2A speedup 低 20.78%--38.71%。
3. **主要差距位于 planner/candidate generation，而不是 runtime。** Expert-compute 结果总体接近，通信收益明显不足；held-out exact evaluator 也只预测 0.7%--2.7% 的 joint-cost 改善。
4. **DeepSeek 与 32K case 本轮没有形成反证或支持证据。** 它们在 route-profile 阶段被重复 NPU 507035 阻塞，应在设备稳定后单独补跑，不能将历史结果与本轮失败混为算法失败。

下一步最有价值的工作不是扩大实验矩阵，而是将论文历史 PlaceMoE layout 与当前 canonical layout 逐层比较，定位为何 legacy/structured candidates 能获得更低的 A2A，而当前候选大多退化为 uniform seed。
