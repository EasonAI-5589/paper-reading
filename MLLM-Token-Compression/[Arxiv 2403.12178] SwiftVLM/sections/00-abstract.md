# Abstract

> 来源: SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Bypass (Arxiv 2403.12178)

---

## 📄 原文

> 💡 **一句话总结**: 提出 **bypass** 剪枝范式 — 被剪掉的 visual tokens 不丢弃，而是"绕道"到后续层重新评估重要性，避免过早剪枝导致的不可逆信息损失。

> 💡 **核心卖点**:
> 1. **Bypass 范式** — 被剪 token 不丢弃，forwarded 到后续剪枝层重新评估
> 2. **Layer-wise 分析** — 发现不同层的 token selection 能力不是单调递增的
> 3. **动态规划选层** — 用 DP 找最优剪枝层
> 4. **Fine-grained 任务表现突出** — 在 RefCOCO 等定位任务上远超现有方法
> 5. **Training-free** — 无需训练

### 关键数字速查
| 指标 | 数值 |
|------|------|
| vs FastV (localization) | +46.6% (192 tokens) |
| vs PDrop (localization) | +59.3% (192 tokens) |
| vs SparseVLM (localization) | +76.0% (192 tokens) |
| Non-localization | 99.0% (几乎无损) |
| FLOPs | 与其他方法相当 |
