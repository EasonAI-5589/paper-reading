# Real-Time Execution of Action Chunking Flow Policies (RTC)

**作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine
**机构**: Physical Intelligence, UC Berkeley
**会议**: **NeurIPS 2025**
**链接**: [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | [项目主页](https://pi.website/research/real_time_chunking)

## 一句话总结

提出 Real-Time Chunking (RTC)，一种纯推理时算法，通过将异步 action chunking 建模为 inpainting 问题（冻住已执行 action + soft masking guidance），让 diffusion/flow-based VLA 在任意推理延迟下都能平滑实时执行，无需重新训练。

## 核心贡献

1. **RTC 算法**: 将异步 action chunking 转化为 inpainting 问题，用 ΠGDM guidance + soft masking 保证跨 chunk 连续性
2. **Soft masking**: 指数衰减的权重矩阵，不只冻住前 $d$ 个 action，而是对所有重叠区域渐进引导
3. **Kinetix 动态仿真 benchmark**: 12 个高动态任务，解决现有 benchmark 太简单的问题
4. **大规模真实实验**: π₀.₅ + 6 个双臂任务，480 episodes，28h 机器人时间，证明 RTC 对延迟完全鲁棒
5. **即插即用**: 纯推理时方法，适用于任何 diffusion/flow-based VLA，不需要改训练

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1（点火柴 demo + 轨迹对比） |
| [01 - Introduction](sections/01-introduction.md) | 动机：物理世界不等你 + action chunking 的不足 |
| [02 - Preliminaries](sections/02-preliminaries.md) | 符号定义 + flow matching 基础 + 延迟数据 + Figure 2/3（mode bifurcation + RTC 设计图） |
| [03 - Method](sections/03-method.md) | **核心方法**: ΠGDM inpainting + soft masking + Algorithm 1 双线程系统 |
| [04 - Experiments](sections/04-experiments.md) | 仿真 12 任务 + 真实 6 任务，RTC 全面最优 |
| [05 - Related Work](sections/05-related-work.md) | 定位：第一个将 inpainting/guidance 用于实时控制 |
| [06 - Discussion](sections/06-discussion.md) | 局限性 + 未来方向 |
| [07 - Appendix](sections/07-appendix.md) | β 消融 + 延迟分解 + soft masking 对比 + 超参数 |

## 关键数字

| 指标 | 数值 |
|------|------|
| RTC 模型延迟 | 97ms (vs baseline 76ms, +28%) |
| π₀ KV cache prefill | 46ms (RTX 4090) |
| 控制频率 | 50Hz (Δt = 20ms) |
| 最大测试延迟 | ~310ms (d≈16, 占 H=50 的 32%) |
| 仿真评估量 | 2048 trials/data point × 12 envs |
| 真实评估量 | 480 episodes, 28h 机器人时间 |
| β (guidance clipping) | 5 |
| BID 延迟 vs RTC | 2.3x (223ms vs 97ms) |

## 方法核心图

```
当前执行的 chunk:  [a₀  a₁  a₂  a₃ | a₄  a₅  a₆  a₇  a₈  a₉  a₁₀ | a₁₁ a₁₂ a₁₃ a₁₄]
                    |-- frozen ---|--- soft decay (exponential) ---|--- free generation --|
Guidance weight:    [  1   1   1   1 | 0.9  0.7  0.5  0.3  0.1  0.05  0  |  0    0    0    0 ]
                    
推理完成前已执行 (d=4)    前一个 chunk 有预测值         超出前一个 chunk 范围
→ 必须匹配                → 渐进引导                    → 完全自由
```

## 与其他 VLA-RTC 论文的关系

| 论文 | 与 RTC 的关系 |
|------|-------------|
| Leave No Observation Behind (2509.23224) | RTC 的前身概念——也关注 VLA action chunk 的实时纠偏 |
| Training-time RTC (2512.05964) | RTC 的训练时改进——将 RTC 的思想融入训练过程 |
| VLASH (2512.01031) | 不同路线——通过异步推理 + 未来状态预测来解决延迟问题 |
