# HiDivDrop: Vision Token Reduction in MLLMs via Late Injection and Differentiable Top-K

**Venue**: ICLR 2026 (OpenReview #25145)
**arXiv**: 2503.14075
**日期**: 2025-02-21 深读

---

## 核心思想

HiDivDrop 的核心洞察：**之前的progressive pruning方法误解了shallow layers的角色**（以为浅层对fusion至关重要），且使用了过于rigid的pruning schedule。HiDivDrop通过分析MLLM层级的真实功能，将token pruning策略与层级角色对齐。

## 两大创新

### 1. Late Injection Strategy
- **观察**: 浅层LLM层对视觉-文本融合是"passive"的——视觉token在这些层基本没有被有效处理
- **策略**: 跳过passive shallow layers，将视觉token直接注入到active fusion真正开始的中间层
- **效果**: 减少了浅层的冗余计算，同时不损失fusino质量

### 2. Concave Pyramid Pruning + Early Exit
- **Concave Pyramid**: 非线性的pruning schedule——在fusion阶段初期快速减少token，中间层保留较多，深层再减少（凹形曲线）
- **Early Exit**: 当层间token表示的相似度足够高时，提前退出（不再需要更多层的处理）
- **Differentiable Top-K**: 使用可微分的top-k算子来优化token选择，而非硬阈值
- **Inter-layer Similarity**: 用层间相似度来动态调整pruning rate

## 关键性能数据

| 设置 | 性能保留 | 备注 |
|------|---------|------|
| LLaVA-1.5-7B, 中等压缩 | **98.3%** retention | 11个benchmark平均 |
| LLaVA-1.5-7B, 88.9% pruning | 保持competitive | 优于PDrop 4.1% |
| LLaVA-1.5-7B, 91.7% pruning | **96.5%** retention | PDrop在此ratio下失效 |
| 训练加速 | **1.72×** | 兼容FlashAttention |
| Prefill latency | 63.6ms → 28.8ms | 实际加速显著 |

## 工程细节
- 解耦visual KV projection，与FlashAttention兼容
- 修复了动态pruning带来的position ID mismatch问题
- 适用于训练和推理两个阶段

## 与STAR-Pro的对比分析

### 维度差异
| 维度 | HiDivDrop | STAR-Pro |
|------|-----------|----------|
| **核心问题** | WHERE — 在哪些层注入/退出 | WHAT — 用什么indicator来决定pruning |
| **方法论** | 层级分析 + 动态schedule | Indicator设计 + 一致性分析 |
| **关注点** | 层的角色（passive vs active） | Token importance的可靠性 |
| **训练需求** | 需要训练（differentiable top-k） | Training-free（推测） |
| **FlashAttention** | ✅ 兼容 | 取决于indicator选择 |

### 互补性
- HiDivDrop回答"在哪里pruning效果最好"
- STAR-Pro回答"用什么标准来pruning"
- **两者可以组合**：用HiDivDrop的Late Injection + Concave Schedule确定WHERE，用STAR-Pro的indicator确定WHAT

### 竞争性
- HiDivDrop的98.3% retention在moderate pruning下非常强
- 但它需要训练（differentiable top-k），而STAR-Pro如果是training-free则有部署优势
- HiDivDrop的Early Exit与STAR-Pro的early termination思路可能有overlap

## 值得关注的点

1. **"Shallow layers are passive"** — 这个发现很重要，说明不是所有层都值得花token computation
2. **Differentiable Top-K** — 可微分的token选择，使得pruning schedule可以被end-to-end优化
3. **Inter-layer similarity** — 用层间相似度作为Early Exit的criterion，与STAR-Pro的"indicator consistency across layers"概念相关
4. **训练加速1.72×** — 不仅是推理优化，还能加速训练，这在实际应用中价值很大

## 局限性
- 需要训练/微调来优化pruning schedule（不是plug-and-play）
- 主要在LLaVA系列验证，泛化性待考察
- Concave Pyramid的shape是否对不同任务/模型需要重新调整？
