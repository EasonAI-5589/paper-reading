# Ctrl-World: A Controllable Generative World Model for Robot Manipulation

**Authors**: Yanjiang Guo\*, Lucy Xiaoyang Shi\*, Jianyu Chen, Chelsea Finn (\*Equal Contribution)
**Affiliations**: Stanford University, Tsinghua University
**Status**: ICLR 2026 | arXiv:2510.10125
**Links**: [arXiv](https://arxiv.org/abs/2510.10125) | [Project](https://ctrl-world.github.io) | [GitHub](https://github.com/Robert-gyj/Ctrl-World)

---

## 一句话总结

用可控多视角世界模型（SVD backbone + 帧级动作条件 + 姿态记忆检索）实现"在想象空间中闭环评估和改进 VLA 策略"，在 DROID 数据集上训练后能零样本泛化到新场景，成功将 π0.5 的指令跟随成功率从 38.7% 提升至 83.4%（+44.7%）。

---

## 核心贡献

1. **多视角联合预测**：同时预测第三人称视角 + 腕部相机，显著减少部分可观测性引起的幻觉
2. **帧级动作条件（Frame-level Action Conditioning）**：在 SVD 的空间 transformer 内通过 cross-attention 将每一帧的动作/位姿嵌入进去，实现厘米级精确控制
3. **姿态条件记忆检索（Pose-conditioned Memory Retrieval）**：稀疏历史帧 + 姿态 embedding 注入，解决长时一致性问题（>20秒）
4. **策略闭环 pipeline**：世界模型 + VLA policy 交互式生成合成轨迹 → 用成功轨迹 SFT 改进 policy

---

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练数据 | DROID 95k 轨迹，564 场景 |
| 视频 backbone | SVD（Stable-Video-Diffusion，1.5B）|
| 预测分辨率 | 192×320 |
| 动作条件步数 | 15步（≈1秒动作块）|
| 历史帧数 | 7帧（stride 1-2s）|
| 长时生成一致性 | >20 秒 |
| FVD (Ctrl-World vs IRASim) | **97.4 vs 138.1**（↓29%）|
| PSNR (Ctrl-World vs 单视角) | **23.56 vs 21.27** |
| 指令跟随相关系数（R²≈） | slope=0.87（世界模型 vs 真实环境）|
| Policy Improvement（avg）| **38.7% → 83.4%（+44.7%）**|
| 训练时间 | ~2-3 天（2×8 H100）|

---

## 与项目的关系

| 关系 | 说明 |
|------|------|
| **我们用的 baseline** | Latent-Act WAM 项目用 Ctrl-World 作为 SVD backbone 版 baseline |
| **backbone** | SVD（不是 Wan2.1！Wan2.1 是候选替换方向）|
| **action 输入** | `observation.state`（绝对 EEF pose），**不是** delta action |
| **训练数据格式** | LeRobot 格式 → 转换为 ctrl-world 格式（libero_ctrlworld）|
| **代码路径** | `/mnt/gyc/Ctrl-World/`，config: `config_libero.py` |

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机：现有 world model 缺啥 |
| [02 - Related Work](sections/02-related-work.md) | 视频生成 + Action-conditioned WM 相关工作 |
| [03 - Problem Formulation](sections/03-problem-formulation.md) | 问题定义 |
| [04 - Method](sections/04-method.md) | 三大核心组件 + 评估/改进 pipeline |
| [05 - Experiments](sections/05-experiments.md) | 世界模型质量 + 策略评估 + 策略改进 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与局限 |

---

## 主要图示

| 图 | 内容 |
|----|------|
| [Figure 1 / Overview](images/overview-page1.jpg) | 论文总览：policy 评估 + policy 改进两个应用场景 |
| [Figure 2 / Architecture](images/figure2-architecture.jpg) | Ctrl-World 架构图：多视角输入、空间/时间 transformer、帧级 cross-attention |
| [Figure 3/4 / Results](images/figure3-4-results.jpg) | 长时序生成质量对比 + 可控性消融（厘米级精度） |
| [Figure 5/6 / Eval](images/figure5-6-eval.jpg) | 一致性展示 + 真实 vs 想象空间对比 |
| [Figure 7/9 / Improvement](images/figure7-9-improvement.jpg) | 指令跟随相关性 + Policy Improvement 结果 |
