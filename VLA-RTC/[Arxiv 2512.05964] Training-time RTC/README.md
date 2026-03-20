# Training-Time Action Conditioning for Efficient Real-Time Chunking

> **Training-time RTC** — RTC 的训练时替代方案

| 属性 | 值 |
|------|-----|
| arXiv | [2512.05964](https://arxiv.org/abs/2512.05964) |
| 日期 | 2025-12 |
| 作者 | Kevin Black et al. (Physical Intelligence) |
| 课题 | VLA 实时推理加速 |

## 核心思想

RTC（Real-Time Chunking）在推理时通过 inpainting 实现异步 action chunk 生成，但有额外计算开销。本文提出**训练时模拟推理延迟**，直接在训练中 condition on action prefixes，消除推理时开销。

## 方法

- 训练时模拟 inference delay，让模型学习基于 action prefix 生成后续动作
- **无需修改模型架构或运行时**，仅需几行代码改动
- 在 π₀.₆ VLA 上验证

## 关键结果

- 在高推理延迟下**优于 inference-time RTC**
- 真实世界箱子组装 + 咖啡制作任务中，保持 RTC 的性能和速度
- 计算开销更低，**实用的 drop-in 替代方案**

## 与 RTC 的关系

| | RTC (Inference-time) | Training-time RTC |
|---|---|---|
| 修改位置 | 推理时 inpainting | 训练时 conditioning |
| 额外开销 | 有（inpainting 计算） | 无 |
| 需要重训练 | ❌ | ✅ |
| 高延迟表现 | 好 | **更好** |

---

*待深度阅读* 📖
