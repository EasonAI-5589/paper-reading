# ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance

> **Paper**: arXiv 2505.18757v2 (May 2025, updated Nov 2025)
> **Authors**: Duo Li, Zuhao Yang, Xiaoqin Zhang, Ling Shao, Shijian Lu (NTU Singapore)
> **Status**: arXiv preprint only，未明确投稿ECCV（无会议标注）
> **竞品分析目的**: 为 STAR-Pro (ECCV 2026) 提供竞品定位

---

## 1. 核心动机

现有 visual token pruning 方法依赖**单一指标**（attention score / token similarity / output divergence），存在以下缺陷：

| 指标类型 | 代表方法 | 问题 |
|---------|---------|------|
| Cross-modal attention | FastV, SparseVLM | Positional bias (causal decoding偏向后方token), attention分布不均 |
| [CLS] attention | FasterVLM | 过度集中，text-to-visual又过于分散 |
| Token similarity (merge) | ToMe | 性能低于直接pruning |
| Output divergence | VTW, FitPrune | 需要calibration set，难以跨模型迁移 |

**核心洞察**: Token diversity 和 task relevance 是两个**正交因素**，应该分开处理：
- **Diversity** → 解决 intra-modal redundancy（视觉token间相似性）
- **Relevance** → 解决 cross-modal redundancy（视觉token与文本任务的相关性）

**Information Migration 现象**: Cross-modal attention在LLM前半段decoder层强，后半段显著减弱 → 视觉信息已被吸收到文本表征中。

---

## 2. 方法详解

### Stage 1: Diversity-Driven Token Selection（在LLM embedding space，prefilling前）

**位置**: Vision encoder之后、LLM输入之前（embedding space）

**算法**: Greedy Max-Sum Diversification
1. **Pivot选择**: 用vision encoder最后一层的[CLS] attention选最重要的token作为初始pivot
   - Image + AnyRes: 从global thumbnail中选
   - Video: 每帧选最高[CLS] attention token，再从中选最高的
2. **贪心扩展**: 迭代选择与已选集合**累积余弦相似度最小**的token
   ```
   c(t) = argmin_{v ∉ C} Σ_{c∈C} cos(x_v, x_c)
   ```
3. 重复直到保留k个token（如k=288，约10%）

**关键特点**:
- 纯diversity驱动，不考虑task relevance
- 在embedding space操作，避免attention的positional bias
- 增量更新累积相似度，计算高效

### Stage 2: Relevance-Driven Token Compression（在LLM decoder内部）

**位置**: LLM prefilling阶段的后半段decoder层

**机制**: 基于 information migration
1. 在选定的检查层（默认 7L/8 处）计算双向cross-modal attention ratio：
   - α_{t→v}: 文本token对视觉token的attention占比
   - α_{v→t}: 视觉token对文本token的attention占比
2. 当两者都低于阈值τ时，**移除该层之后的所有视觉token**
3. 一次性全部移除，不做渐进式pruning

**关键特点**:
- 不做partial/progressive pruning，因为Stage 1已经移除了大部分token
- 只在一个layer做判断和移除
- 移除后decoding阶段不再有视觉token的计算开销

---

## 3. 关键实验结果

### 3.1 Image Understanding (LLaVA-NeXT-7B, 8 benchmarks)

| 方法 | 保留率 | 平均性能比 |
|------|-------|-----------|
| **ToDRE** | **25%** | **98.2%** |
| DivPrune | 25% | 96.6% |
| FasterVLM | 25% | 96.0% |
| **ToDRE** | **10%** | **95.0%** |
| DivPrune | 10% | 93.5% |
| FasterVLM | 10% | 91.4% |
| FastV | 10% | 88.8% |

### 3.2 效率 (10% retention, LLaVA-NeXT-7B)

| 方法 | FLOPs↓ | Memory↓ | Throughput↑ | 性能 |
|------|--------|---------|-------------|------|
| ToDRE | 6.0T (↓80.9%) | 13.6GB (↓14.5%) | 2.9 s/s (1.9×) | 95.0% |
| DivPrune | 6.0T | 13.6GB | 2.8 s/s | 93.5% |

### 3.3 Ablation

- Stage 2 only → 100.0% performance, 仅8.8%时间节省（因为只加速后半段）
- Stage 1 only (10%) → 95.8%, 59.4%时间节省
- Stage 1+2 (10%) → 96.0%, 61.4%时间节省
- **Stage 2 的贡献有限**: 主要收益来自Stage 1的diversity selection

### 3.4 Cross-Model

- Qwen2.5-VL-7B: 25% → 97.1%, 10% → 92.0%
- InternVL2-8B: 25% → 96.8%, 10% → 91.5%

---

## 4. 与 STAR-Pro 的深度对比

### 4.1 架构对比

| 维度 | ToDRE | STAR-Pro |
|------|-------|----------|
| **Stage 1 位置** | LLM embedding space (prefilling前) | 同样在embedding space |
| **Stage 1 策略** | 纯 diversity (greedy max-sum) | **R + λD 融合** (relevance + diversity) |
| **Stage 2 位置** | LLM decoder内单一layer | LLM decoder内 progressive multi-layer |
| **Stage 2 策略** | 一次性移除所有visual tokens | 渐进式多层pruning |
| **Diversity定义** | 最小化累积余弦相似度 | （待确认STAR-Pro的D定义） |
| **Relevance信号** | Stage 2才用cross-modal attention | **Stage 1就融合relevance** |
| **训练** | Training-free | Training-free |

### 4.2 核心设计差异分析

#### Diversity vs. Relevance 的处理方式

**ToDRE**: 将diversity和relevance视为正交因素，**严格分阶段处理**
- Stage 1 只看diversity → 可能保留diverse但task-irrelevant的tokens
- Stage 2 只看relevance → 用information migration一次性移除

**STAR-Pro**: 在Stage 1就**融合R+λD**
- 优势：早期就能兼顾diversity和task relevance
- 更精细的token selection，不会保留diverse但无用的tokens

#### Stage 2 的粒度

**ToDRE**: 单一layer全部移除
- 简单粗暴，效果有限（ablation显示仅+0.2%性能提升）
- 对长文本生成可能有更大效率收益

**STAR-Pro**: Progressive multi-layer pruning
- 更精细的控制，渐进式减少
- 可能在不同层保留不同程度的视觉信息

### 4.3 STAR-Pro 的潜在优势

1. **Stage 1 融合R+D**: 比纯diversity更有针对性，避免保留"diverse but irrelevant"的tokens
2. **Progressive Stage 2**: 比一次性全删更灵活，可以适应不同任务对视觉信息的需求
3. **ToDRE Stage 2 效果有限**: ablation证明其Stage 2贡献很小（+0.2%性能，2%时间），说明单纯靠information migration的一次性删除效果不佳

---

## 5. ToDRE 弱点分析

### 5.1 Stage 1 纯 Diversity 的局限

- **无task-awareness**: 纯diversity选择可能保留背景中互不相似但任务无关的tokens
- **Pivot依赖[CLS]**: 虽然作者说random也接近，但[CLS] attention本身也有局限
- **线性贪心**: Greedy algorithm只保证近似最优，无法保证全局最优diversity

### 5.2 Stage 2 贡献微弱

- Ablation显示Stage 2对性能提升仅+0.2%，时间节省也有限
- Information migration threshold τ 需要手动设定
- **只在一个layer判断**: 不同输入可能需要不同的pruning timing
- 对短答案任务（大多数benchmark），decoding阶段节省可忽略

### 5.3 实验局限

- 主要在LLaVA-NeXT上验证，Qwen和InternVL的实验没有与其他方法对比
- 效率测试只用了POPE（单token回答），无法体现长文本生成的优势
- 缺少 high-resolution / dense prediction 任务的评估
- 2.6× speedup 的claim主要来自Stage 1的90% pruning

### 5.4 理论声称 vs 实际效果

- 声称diversity和relevance是"正交因素"，但未提供严格的正交性证明
- 声称分开处理更好，但未与"融合处理"（如R+λD）做对比 → **STAR-Pro的机会点**

---

## 6. 对 STAR-Pro 论文写作的启示

### 可以引用的论点
1. ToDRE 验证了 diversity 在 token selection 中的重要性 → STAR-Pro 也使用了 diversity
2. ToDRE 证明纯 attention-based 方法存在 positional bias → STAR-Pro 的 R+λD 融合是更好的解决方案
3. ToDRE Stage 2 效果有限 → 证明 progressive multi-layer pruning（STAR-Pro）可能更有效

### 差异化叙事
- "ToDRE treats diversity and relevance as orthogonal and handles them separately; however, we argue that early fusion of R+λD in Stage 1 yields more task-aware token selection."
- "ToDRE's Stage 2 removes all visual tokens at a single layer, achieving only marginal gains; our progressive multi-layer approach enables finer-grained control."

### 实验对比
- 需要在相同benchmark上与ToDRE直接比较
- 重点展示在10% retention ratio下的性能差距
- 展示progressive pruning vs one-shot removal的ablation

---

*分析日期: 2026-02-21*
*分析目的: STAR-Pro ECCV 2026 竞品分析*
