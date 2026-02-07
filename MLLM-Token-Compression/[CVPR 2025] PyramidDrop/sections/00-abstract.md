# Abstract

> 来源: PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction (CVPR 2025)

---

## 📄 原文

> 💡 **一句话总结**: 提出 PyramidDrop，一个**渐进式** visual token 减少策略——浅层保留全部 token，深层逐步丢弃，形成"金字塔"结构。同时加速训练和推理。

> 💡 **核心卖点**:
> 1. **渐进式减少** — 不是一刀切，而是分阶段逐步减少 image tokens
> 2. **训练 + 推理双加速** — 不仅推理快，训练也能省 40% 时间
> 3. **Plug-and-play** — 也可作为 inference-only 策略
> 4. **基于实证观察** — 浅层 token 全部重要，深层 token 大量冗余

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 训练加速 | 40%+ |
| 推理 FLOPs 降低 | 55% |
| 性能 | 几乎无损 |
| 模型 | LLaVA-NeXT, LLaVA-1.5 |
