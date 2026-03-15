# Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation

**Authors**: Huajie Tan, Sixiang Chen, Yijie Xu, Zixiao Wang, Yuheng Ji, Cheng Chi, Yaoxu Lyu, Zhongxia Zhao, Xiansheng Chen, Peterson Co, Shaoxuan Xie, Guocai Yao, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang
**Affiliations**: Peking University, BAAI, University of Sydney, CAS
**Conference**: CVPR 2026
**Links**: [arXiv](https://arxiv.org/abs/2512.23703) | [Project Page](https://robo-dopamine.github.io/) | [GitHub](https://github.com/FlagOpen/Robo-Dopamine) | [HuggingFace](https://huggingface.co/collections/tanhuajie2001/robo-dopamine)

## 一句话总结

提出 Dopamine-Reward（基于 hop-based 相对进度建模 + 多视角融合的通用奖励模型 GRM）和 Dopamine-RL（理论保证 policy-invariant 的奖励塑形框架），实现 one-shot 适配新任务后仅 150 次交互即可从近零提升到 95% 成功率。

## 核心贡献

1. **Dopamine-Reward**: 提出 hop-based step-wise 相对进度建模方法，训练通用奖励模型 GRM（35M 样本，3400+ 小时数据），支持多视角输入
2. **Multi-Perspective Progress Fusion**: 融合 incremental/forward-anchored/backward-anchored 三种视角的进度预测，消除误差累积
3. **Dopamine-RL**: 理论推导 policy-invariant reward shaping（$r = r_{gold} + \gamma\Phi(s_{t+1}) - \Phi(s_t)$），避免 semantic trap
4. **One-Shot GRM Adaptation**: 仅需一条示教轨迹即可适配新任务
5. **大规模数据集**: 3,400 小时、100K 轨迹、350+ 日常操作任务（真实 + 仿真 + 人类视频）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：两大问题 + 两大方案 |
| [01 - Introduction](sections/01-introduction.md) | 动机：IL 局限 → RL 奖励难题 → PRM 两大缺陷 |
| [02 - Related Work](sections/02-related-work.md) | RL for Robotics + Learned PRMs |
| [03 - Method](sections/03-method.md) | GRM 构建 + 多视角融合 + Policy-Invariant Shaping |
| [04 - Experiments](sections/04-experiments.md) | RQ1-4: 奖励精度、成功率、泛化、消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练数据 | 35M 样本 / 3,400+ 小时 / 100K 轨迹 |
| 任务覆盖 | 350+ 日常操作任务 |
| VOC (rank correlation) | 0.953 (8B Multi-View, Sparse) |
| 任务完成判断准确率 | 92.8% (8B Multi-View) |
| 真实世界成功率 | 95.2% (150 rollouts, ~1 小时) |
| 泛化性能下降 | 仅 8-19% vs BC 的 50-60% |
| 模型规模 | 3B / 8B（基于 RoboBrain 2.0） |

## 与 VLA-WM 系列关系

| 论文 | 关系 |
|------|------|
| [RoboBrain](../[CVPR%202025]%20RoboBrain/) | GRM 的 base model 来源 (RoboBrain 2.0) |
| [RoboBrain 2.0](../[Arxiv%202507.02029]%20RoboBrain-2.0/) | 直接前作，提供预训练 VLM |
| [pi0](../[CoRL%202025]%20pi0/) | Dopamine-RL 兼容的 policy 架构之一 |
