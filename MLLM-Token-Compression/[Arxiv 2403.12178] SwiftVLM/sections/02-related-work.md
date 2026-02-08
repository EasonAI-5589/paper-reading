[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 将现有视觉 token 压缩方法分为 **Text-agnostic** 和 **Text-aware** 两大类，分析各自的优缺点。

---

To reduce the number of visual tokens and improve inference efficiency, existing studies (Zhong et al., 2025; Wang et al., 2025b; Li et al., 2024b) can be broadly classified into two categories.

> 💡 **分类框架**: 按是否利用文本信息来指导 token 压缩，分为两类。

---

### Text-agnostic

Qwen2.5-VL (Bai et al., 2025) merges each group of four neighboring visual tokens into a single token. ToMe (Bolya et al., 2022) performs similarity-based token merging between the attention and MLP blocks. VisionZip (Yang et al., 2025b) retains tokens with high [CLS]-attention scores and merges the remaining ones based on feature similarity, following a strategy similar to VisPruner (Zhang et al., 2025) and Prumerge (Shang et al., 2025). VoCo-LLAMA (Ye et al., 2025b) compresses visual information into a single learnable VoCo token, which is then used for subsequent cross-modal interaction.

> 💡 **Text-agnostic 方法一览**:
> | 方法 | 策略 | 特点 |
> |------|------|------|
> | Qwen2.5-VL | 4 邻近 token → 1 token | 固定规则，简单粗暴 |
> | ToMe | 相似度 merging | 在 attention/MLP 之间操作 |
> | VisionZip | CLS-attention 筛选 + 相似度合并 | 类似 VisPruner、Prumerge |
> | VoCo-LLAMA | 压缩到 1 个可学习 token | 极端压缩 |

---

Despite their efficiency, these methods rely solely on visual cues for token reduction, which limits their ability to preserve query-relevant visual details, particularly when the queried regions are not visually salient.

> 💡 **核心局限**: 不考虑文本查询，只看视觉特征。如果查询涉及的区域视觉上不显著（如小文字、背景物体），就容易被误删。

---

### Text-aware

Q-Former (Li et al., 2023) reduces visual token redundancy by training cross-modal modules that compress hundreds of visual tokens into a small set of learnable tokens. ATP-LLaVA (Ye et al., 2025a) instead introduces trainable modules within the VLM and prunes visual tokens based on importance scores derived from text–vision and vision–vision attention. Although these approaches leverage the text query to guide visual token compression or selection, they require additional trainable components, incurring extra optimization overhead.

> 💡 **需要训练的方法**: Q-Former 和 ATP-LLaVA 都需要额外可学习模块，虽然利用了文本信息但增加了训练成本。

---

Several training-free methods exploit the native cross-modal attention of VLMs. FastV (Chen et al., 2024a) uses T-V attention to assess visual token importance and performs aggressive pruning at a shallow layer. PDrop (Xing et al., 2024) progressively reduces visual tokens across layers, based on the observation that pruning becomes less harmful at deeper layers. FEATHER (Endo et al., 2025) further refines this strategy by mitigating the influence of Rotary Position Embedding (RoPE) (Su et al., 2024) on T-V attention, while SparseVLM (Zhang et al., 2024) performs adaptive layer-wise pruning by estimating redundancy from the rank of the T-V attention matrix. Despite being training-free, these methods assume that tokens pruned early remain unimportant in deeper layers, which often fails in fine-grained visual reasoning, leading to performance degradation.

> 💡 **Training-free 方法（本文直接竞争对手）**:
> | 方法 | 策略 | 局限 |
> |------|------|------|
> | FastV | 浅层一次性激进剪枝 | 信息丢失严重 |
> | PDrop | 逐层渐进减少 | 仍假设早期丢弃的不重要 |
> | FEATHER | 消除 RoPE 影响 + 渐进剪枝 | 仍是 drop 范式 |
> | SparseVLM | 按 attention 矩阵秩自适应剪枝 | 同上 |
>
> **共同假设（也是共同缺陷）**: 被早期丢弃的 token 在深层也不重要 → 在细粒度推理中失败

---

## 🔖 Section 总结

### 核心洞察
1. Text-agnostic 方法高效但忽视查询相关性
2. Text-aware + 需训练的方法有额外开销
3. Training-free drop 方法是 SwiftVLM 的主要竞争对手，共同缺陷是**不可逆丢弃**
4. SwiftVLM 的定位：training-free + text-aware + **不丢弃**（bypass）
