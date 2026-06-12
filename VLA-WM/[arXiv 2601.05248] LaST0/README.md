# LaST₀: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

**Authors**: Zhuoyang Liu*, Jiaming Liu*†, Hao Chen*, Jiale Yu, Ziyu Guo, Chengkai Hou, Chenyang Gu, Xiangju Mi, Renrui Zhang, Kun Wu, Zhengping Che†, Jian Tang, Pheng-Ann Heng, Shanghang Zhang✉
**Affiliations**: Peking University / 北航人形机器人创新中心 / CUHK / Simplexity Robotics
**Status**: arXiv 2601.05248 | ICML 2026
**Links**: [arXiv](https://arxiv.org/abs/2601.05248) | [Project](https://vla-last0.github.io/) | [GitHub](https://github.com/ZhuoyangLiu2005/last0)

## 一句话总结

用**隐式时空 CoT**（Latent Spatio-Temporal Chain-of-Thought）替代显式文本/图像推理，通过 **Mixture-of-Transformers 双系统架构**实现低频 latent 推理 + 高频 action 生成的解耦，在真机 10 任务上比 SOTA VLA 提升 13-14%，推理速度 15.4 Hz（比显式 CoT 快 14×）。

## 核心贡献

1. **LaST CoT**: 在 latent 空间构建时空推理链，同时编码未来视觉语义、3D 几何和机器人本体感知信息
2. **Mixture-of-Transformers (MoT) 双系统**: 慢推理 expert（低频，生成 latent CoT）+ 快执行 expert（高频，Flow Matching 生成 action），共享 self-attention 实现信息交互
3. **异步频率训练策略**: 混合不同快慢频率比例训练，推理时自适应选择最优频率

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机：显式 CoT 的延迟与表征瓶颈 |
| [02 - Related Work](sections/02-related-work.md) | VLA、Latent CoT 相关工作 |
| [03 - Method](sections/03-method.md) | LaST CoT 构造 + MoT 架构 + 训练策略 |
| [04 - Experiment](sections/04-experiment.md) | 仿真 + 消融 + 真机实验 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与展望 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 仿真成功率（RLBench 10 任务） | **82%**（vs HybridVLA-7B 74%，π₀.₅ 65%） |
| 真机 Franka 成功率 | **72%**（vs π₀.₅ 59%，CoT-VLA 50%） |
| 推理速度 | **15.4 Hz**（vs CoT-VLA 1.1 Hz，快 **14×**） |
| 模型参数量 | 3.3B（Janus-Pro / DeepSeek-LLM 1.5B 基座） |
| 预训练数据 | 400K+ 轨迹（Open-X, DROID, ROBOMIND 等） |
| latent token 数/模态 | **1 token**（average pooling，够用） |
| 最优时序覆盖 | **4 个未来关键帧** |
| 最优快慢频率比 | **1:4**（混合训练，推理选 1:4） |

## 与 VLA-WM 系列关系

| 论文 | 关系 |
|------|------|
| DreamZero（WAM）| LaST₀ 的 latent 世界状态预测是 WAM 的 latent 空间版本，Last-WAM 项目的前置工作 |
| CoT-VLA（显式 CoT 基线） | LaST₀ 解决了 CoT-VLA 延迟高（1.1 Hz）的问题 |
| [pi0.5](../[CoRL%202025]%20pi0.5/) | 同为 Flow Matching VLA，LaST₀ 在 RLBench 上超越 π₀.₅ 17% |
