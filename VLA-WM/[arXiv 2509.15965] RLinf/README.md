# RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation

**Authors**: Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, et al.
**Affiliations**: Tsinghua University, Infinigence AI, Peking University, UC Berkeley, Beihang University, SJTU
**Status**: arXiv 2509.15965
**Links**: [arXiv](https://arxiv.org/abs/2509.15965) | [GitHub](https://github.com/RLinf/RLinf)

## 一句话总结

提出 M2Flow（宏到微流变换）范式，将用户编写的高层 RL 工作流自动变换为针对硬件优化的细粒度执行计划，通过弹性流水线 + 上下文切换 + profiling 引导调度，在 reasoning RL 和 embodied RL 上实现 1.07x~2.43x 加速。

## 核心贡献

1. **M2Flow 范式**: 解耦 RL 工作流的逻辑编程与物理执行，开发者写宏观流程，系统自动优化微观执行
2. **弹性流水线 (Elastic Pipelining)**: 空间调度——动态调整 worker 数据处理粒度实现灵活流水线
3. **自动上下文切换 (Context Switching)**: 时间调度——通过设备锁实现 GPU 时分复用
4. **Profiling-guided Scheduler**: 自动搜索最优执行模式（temporal/spatial/hybrid）
5. **自适应通信层**: 支持任意 worker 间通信，自动选择最优通信后端（NCCL/cudaIPC/Gloo）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机：RL 工作流异构性 + 现有系统不灵活 |
| [02 - Background](sections/02-background.md) | RL 工作流特征 + 现有系统效率问题分析 |
| [03 - Design](sections/03-design.md) | M2Flow + 弹性流水线 + 上下文切换 + 调度策略 + 通信 |
| [04 - Evaluation](sections/04-evaluation.md) | Reasoning RL + Embodied RL 实验 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Reasoning RL 加速 | 1.07x ~ 1.70x vs veRL/Slime |
| Embodied RL 加速 | 1.05x ~ 2.43x vs SimpleVLA-RL |
| 支持模型 | Qwen2.5 1.5/7/32B, Qwen3 MoE, OpenVLA, OpenVLA-OFT, Pi0 |
| 支持算法 | GRPO, PPO, DAPO, REINFORCE++ |
| 调度搜索时间 | 0.0007s ~ 5.98s (8~1024 GPUs) |
| 代码量 | 20K 行 Python |
| LIBERO 成功率提升 | 34.33% → 97.83% (OpenVLA-OFT) |

## 与 VLA-WM 系列关系

| 论文 | 关系 |
|------|------|
| [Robo-Dopamine](../[CVPR%202026]%20Robo-Dopamine/) | Dopamine-RL 使用 PPO/Cal-QL，RLinf 可作为其训练基础设施 |
| [pi0](../[CoRL%202025]%20pi0/) | RLinf 已支持 Pi0 模型的 RL 训练 |
| SimpleVLA-RL | RLinf 在 LIBERO 上 2.43x 加速 vs SimpleVLA-RL |
