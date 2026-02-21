# HiDivDrop: Experiments & Analysis

> ⚠️ 基于 web search 和 OpenReview abstract 信息。详细表格待 PDF 解析后补充。

## 实验设置

- **主要模型**: LLaVA-1.5-7B
- **Benchmarks**: 11 个 (具体列表待 PDF 确认)
- **对比方法**: PDrop, ToMe, FastV, PyramidDrop 等

## 主要结果

### Token Compression Performance

| Setting | HiDivDrop | PDrop | Gap |
|---|---|---|---|
| 88.9% pruning | **98.3%** retention | 94.2% | **+4.1%** |
| 91.7% pruning | **96.5%** retention | fails | — |

### Efficiency

| Metric | Before | After |
|---|---|---|
| Prefill latency | 63.6 ms | **28.8 ms** |
| Training acceleration | 1.00× | **1.72×** |

> 💡 98.3% retention at 88.9% pruning 非常接近 lossless。对比 DART 在同样比例下的 93.7%（LLaVA-1.5-7B），HiDivDrop 高出约 4.6%。但要注意 HiDivDrop 需要训练，DART 是 training-free。

> 💡 Training acceleration 1.72× 是 HiDivDrop 的独特优势。大多数 token pruning 方法只加速 inference，HiDivDrop 因为 Late Injection 跳过浅层计算，在 training 时也能减少计算量。

> 💡 91.7% pruning 下 PDrop fails 但 HiDivDrop 仍能保持 96.5%，说明 concave schedule + early exit 在极端压缩下的优势。PDrop 的线性 schedule 可能在深层仍然 prune 过多，导致崩溃。

## 与 DART 的对比分析

| Dimension | HiDivDrop | DART |
|---|---|---|
| Training | Required | Training-free |
| Pruning Criteria | Learned (differentiable Top-K) | Duplication-based (cosine sim) |
| Schedule | Progressive (concave pyramid) | One-shot (after layer 2) |
| FlashAttention | Compatible | Compatible |
| Training Acceleration | 1.72× | N/A |
| 88.9% pruning performance | ~98.3% | ~93.7% |
| Key Insight | Shallow layers are passive → Late Injection | Importance is unreliable → Use duplication |

> 💡 **互补性**：HiDivDrop 解决 "when/how much to prune"，DART 解决 "what criteria to prune by"。理论上可以将 DART 的 duplication-based criteria 插入 HiDivDrop 的 progressive framework 中——Late Injection + Concave Pyramid + Duplication-based selection = 可能更强的组合。

> 💡 **Trade-off**: HiDivDrop 性能更好但需要训练，DART plug-and-play 但性能略低。对于已有训好的模型要快速部署，DART 更实用；对于从头训练或 fine-tune 场景，HiDivDrop 更优。
