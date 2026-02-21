# Stop Looking for "Important Tokens" in Multimodal Language Models: Duplication Matters More

**arXiv**: 2502.11494 (Feb 2025)
**作者**: Zichen Wen et al. (上海交大 + 上海AI Lab)
**方法名**: DART (Duplication-Aware Reduction of Tokens)
**日期**: 2025-02-21 深读

---

## 核心论点：Importance-Based Pruning 有根本性缺陷

这篇论文的标题就是它的thesis：**别再找"重要token"了**。四大问题：

### 问题 I: 忽略token间交互的动态性
- Importance score是**静态**计算的：s_i = F(x_i | X)
- 但pruning一个token后，其他token的重要性应该变化
- 两个相似token，删一个后另一个应变得更重要，但static scoring无法捕捉
- 这违反了**独立性假设**

### 问题 II: 与FlashAttention不兼容
- 基于attention score的方法需要显式获取attention map
- FlashAttention不输出attention map → 必须禁用FA → 速度反而变慢
- **实际加速 vs 理论FLOPs减少的矛盾**

### 问题 III: Position Bias
- Attention score对位置有偏差：靠近最后一个token的位置attention更高
- 导致保留的token集中在图像右下角（序列尾部）
- **不反映真实的语义重要性**

### 问题 IV: 不如随机pruning！
- **最震撼的发现**: FastV、SparseVLM在88.9% reduction下表现比随机pruning还差
- 这直接质疑了整个importance-based paradigm的合理性

## DART方法

### 核心思路：去重 > 找重要
- 选少量**pivot tokens** (≤2% of total, e.g., 8个)
- 计算所有token与pivot的**cosine similarity**
- 保留与pivot**最不相似**的token（低duplication = 高信息量）
- 丢弃与pivot高度相似的冗余token

### Pivot Token选择
- K-norm最大的token (默认)
- V-norm / 随机选择 / attention score → **都work**
- 性能差异 <1.2%，说明**去重比选pivot重要得多**

### 理论保证
- Lipschitz连续性 + Hausdorff距离 bound
- ||f(X) - f(R)|| ≤ K√(2(1-ε))B
- ε越大（去重阈值越高），保留的token越"不同"，误差bound越小

## 关键性能数据

### LLaVA-1.5-7B
| Reduction | DART Avg | 次优方法 | 优势 |
|-----------|----------|---------|------|
| ↓66.7% (192 tokens) | **98.8%** | MustDrop 97.2% | +1.6% |
| ↓77.8% (128 tokens) | **98.0%** | MustDrop 95.6% | +2.4% |
| ↓88.9% (64 tokens) | **93.7%** | FiCoCo-V 91.5% | +2.2% |

### LLaVA-Next-7B (88.9% reduction)
- DART: **93.9%** vs 次优 HiRED 91.8%

### 速度
- 1.99× total speedup, 2.99× prefill speedup
- Token reduction overhead: **<0.08s**
- 完全兼容FlashAttention

### 跨模型验证
- Qwen2-VL-7B: 97.0% at 66.7% reduction
- MiniCPM-V2.6: 92.9% at 66.7% reduction
- Video-LLaVA: 接近原始性能 at 50% reduction

## 与STAR-Pro的关系分析

### 这篇是Supporting Evidence还是竞品？

**主要是Supporting Evidence + 部分竞品**

#### Supporting Evidence方面：
1. **直接验证了indicator inconsistency**: "importance score不如random"这个发现 = STAR-Pro的核心motivation的实证基础
2. **Position bias问题**: 与STAR-Pro关注的"indicator不可靠"同源
3. **Static vs dynamic scoring矛盾**: 这正是STAR-Pro要解决的问题的一个具体表现

#### 竞品方面：
1. DART提出了一个**替代范式**（duplication替代importance），而非改进importance indicator
2. 如果STAR-Pro的approach是"设计更好的importance indicator"，DART则说"importance这条路本身就错了"
3. DART的93.7%@88.9% reduction是一个strong baseline

### STAR-Pro比这篇更进一步在哪里？

| 维度 | StopLooking/DART | STAR-Pro (推测) |
|------|-----------------|----------------|
| **诊断** | "Importance不行" → 换paradigm | "Importance可以，但需要更好的indicator" |
| **解法** | 放弃importance，用duplication | 在importance框架内找到一致性更好的indicator |
| **深度** | 观察现象 + 提出替代 | 分析WHY inconsistency + 提出principled solution |
| **理论** | Hausdorff distance bound | Indicator consistency theory |

### STAR-Pro可以引用这篇的角度：
1. 作为motivation的实证支持："现有importance indicator确实有问题"
2. 作为baseline来比较："我们的indicator比duplication-based方法还好"
3. 但需要回应DART的核心挑战："importance paradigm本身是否有问题？"

## 重要发现总结

1. **"不同pivot选择策略保留的token重叠度<50%，但性能相当"** → 说明不存在唯一的"critical token set"，这挑战了importance-based方法的基本假设
2. **Duplication-based方法对pivot选择不敏感** → 鲁棒性强
3. **FLOPs减少 ≠ 实际加速** → SparseVLM FLOPs只多2.8%但speedup少21.6%（因为multi-stage sequential processing）
4. **FastV产生更多hallucination** → importance pruning反而引入偏差

## 局限性
- 只在第2层之后做一次pruning（单点pruning）
- Pivot token数量固定为8个
- 没有考虑text-guided的token selection
- 对OCR等需要细粒度理解的任务，extreme compression下仍有明显下降
