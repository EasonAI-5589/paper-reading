[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 阐述了现有视觉 token 剪枝方法的三大类策略（merge、drop、progressive drop），揭示了它们共同的缺陷——**层间 token 重要性排序不一致**，并提出 bypass 作为第三种范式。

---

Vision–Language Models (VLMs) (Team et al., 2024; Chen et al., 2024b; Alayrac et al., 2022) have rapidly advanced in recent years and emerged as a central paradigm in multimodal learning. These models integrate a visual encoder with a large language model (LLM) (Grattafiori et al., 2024; Achiam et al., 2023) through a cross-modal fusion module, enabling strong performance across a wide range of vision–language tasks (Gao et al., 2025; Lin et al., 2025; Yang et al., 2025a; Wang et al., 2025a). In practice, visual inputs are processed by generating a large number of visual tokens. However, only a small subset of these tokens is critical for text-conditioned reasoning, with the remainder largely increasing latency and computational overhead.

> 💡 **背景**: VLM 的标准范式是 visual encoder + LLM + 跨模态融合。核心问题：视觉 token 数量多但大部分与文本查询无关，造成冗余计算。

---

To reduce the number of visual tokens, prior studies adopt token merging strategies, such as ToMe (Bolya et al., 2022), Qwen-VL (Bai et al., 2025), and VisionZip (Yang et al., 2025b). These methods aggregate visual features based on feature similarity or spatial proximity. While these approaches improve inference efficiency, such compression degrades fine-grained visual details, especially for precise localization tasks.

> 💡 **策略一：Token Merging**: ToMe、Qwen-VL、VisionZip 等方法通过相似度或空间邻近性合并 token。问题是合并会损失细粒度视觉细节（如定位任务需要的精确位置信息）。

---

Another line of work leverages text-to-vision (T–V) attention in VLMs to rank visual tokens and dynamically drop low-ranked ones, as illustrated in Fig.1(b). FastV (Chen et al., 2024a) observes that T–V attention becomes highly concentrated on a small subset of visual tokens from the third layer onward, and thus aggressively drops low-ranked ones in a shallow layer. PDrop (Xing et al., 2024) further shows that aggressive pruning in early layers leads to significant performance degradation, whereas the impact becomes less severe in deeper layers, motivating a progressive dropping strategy. This principle is subsequently adopted by works such as SparseVLM (Zhang et al., 2024) and FEATHER (Endo et al., 2025). However, we find that the importance ranking of visual tokens varies across layers.

> 💡 **策略二：Text-aware Drop**: 利用 T-V attention 排序并丢弃低排名 token。FastV 在浅层激进剪枝；PDrop 发现浅层激进剪枝有害，改为渐进式。但核心问题：**token 重要性排序在不同层之间是变化的**。

---

![Figure 1](../images/a8d3a0e913ab8120540dd22ac32506ac7134131ddd136ddfe30584cd92156193.jpg)
*Figure 1. Comparison of visual token pruning strategies in VLMs. (a)–(b) Existing approaches suffer from irreversible loss of critical visual information once tokens are merged or dropped in shallow layers. (c) We propose Bypass, a pruning strategy that restores previously merged tokens via token alignment. Bypass provides critical visual tokens with an opportunity to be reconsidered at deeper layers with stronger token selection capability.*

> 💡 **Figure 1 批读**:
> - **(a) Merge**: 合并后信息不可恢复
> - **(b) Drop**: 丢弃后 token 永久丢失
> - **(c) Bypass（本文）**: 未选中的 token 不丢弃，而是保留原始状态 → 通过 merge 代理参与中间计算 → 在后续剪枝层通过 alignment 恢复并重新评估
> - 这是本文最核心的 idea 可视化

---

As illustrated in Fig.2, we report the overlap ratio on a TextVQA (Singh et al., 2019) sample between the bottom 50% visual tokens selected by early layers (layers 1–9) and the top 10% visual tokens selected by later layers (layers 10–20) of LLaVA-1.5-7B (Liu et al., 2024a). We observe that visual tokens deemed unimportant and dropped in early layers can become highly important in deeper layers.

![Figure 2](../images/e1258e0fcf7da8781b141fcc133245e5816047a11b7d71dce007ca4f111a8e8b.jpg)
*Figure 2. Layer-wise variation in visual token ranking. For a representative TextVQA example, we report the overlap ratio between the bottom-ranked 50% of visual tokens selected at layers 1–9 and the top-ranked 10% selected at layers 10–20 of LLaVA.*

> 💡 **Figure 2 批读**:
> - 这是 motivation 的关键证据：浅层被排在后 50% 的 token，有相当一部分在深层进入了前 10%
> - 说明浅层的 T-V attention 不能准确反映 token 的最终重要性
> - 这直接解释了为什么 FastV 这类早期剪枝方法在细粒度任务上失败

---

While existing methods perform early-layer pruning to improve efficiency, prematurely dropping task-relevant visual tokens can hinder subsequent reasoning. As shown in Fig.3, methods such as FastV and PDrop force deeper layers to reason over incomplete visual evidence, often resulting in incorrect answers.

![Figure 3](../images/802a623af5d31f9927b335a5b58277a98d97ff291c613fdcf55f7be2b05192a7.jpg)
*Figure 3. Comparison of results from different pruning methods. FastV applies aggressive early-layer pruning, whereas PDrop adopts progressive pruning. Both drop the visual token containing "NASRI", leading to incorrect answers. SwiftVLM preserves the query-relevant token at the final stage and answers correctly.*

> 💡 **Figure 3 批读**:
> - 具体案例：问题需要识别球衣上的 "NASRI" 文字
> - FastV 和 PDrop 都在早期丢弃了包含 "NASRI" 的 token → 答错
> - SwiftVLM 通过 bypass 保留了该 token，在最终阶段正确识别 → 答对
> - 这个例子很好地说明了 fine-grained visual reasoning 的需求

---

Based on these observations, we propose a third pruning paradigm, termed bypass. As illustrated in Fig.1(c), at the first pruning layer, bottom-ranked visual tokens are not immediately discarded. Instead, they are fully preserved and forwarded directly to the next pruning layer for re-ranking of their importance. Meanwhile, these bottom visual tokens are merged according to feature similarity. The merged visual tokens then participate in subsequent inference.

> 💡 **Bypass 核心机制**:
> 1. 第一次剪枝：低排名 token **完整保留**（bypass pathway）
> 2. 同时，这些 token 按相似度合并 → 合并后的 token 参与后续推理（作为代理）
> 3. 在下一个剪枝层：用合并 token 的变化来校正 bypass token → 重新评估重要性

---

At the following pruning layer, we derive a hidden-state offset from the merged visual tokens and use it to adjust the bypassed bottom-ranked tokens, aligning them with text tokens in the current representation space. These corrected tokens are then reintroduced for joint re-evaluation.

> 💡 **Token Alignment**: bypass 的关键技术难点是如何让跳过了中间层计算的 token 与当前层的表示空间对齐。解决方案：用 merge token 在中间层的变化量（offset）来近似校正。

---

This design preserves the complete visual information while allowing each pruning layer to independently assess token importance, thereby avoiding irreversible critical information loss caused by premature pruning in early layers.

---

Furthermore, to determine the pruning layers used for token selection, we conduct a comprehensive layer-wise analysis across two task categories and six benchmark datasets. We first run the vanilla model and record, at each layer, the indices of the top 20% visual tokens selected based on T–V attention. Using the same set of evaluation samples, we then re-run the model while retaining all visual tokens in the first two layers and keeping only the layer-specific top 20% visual tokens from the third layer onward. The layer-wise results are reported in Fig.4.

![Figure 4](../images/465c1bc284a732b9836f68dcf5d5930c3417b102749e2cc99a80c800a56214c6.jpg)
*Figure 4. Non-monotonic layer-wise capability for visual token selection. Across tasks and datasets, we record the layer-wise top 20% visual tokens of the vanilla model and re-evaluate it by retaining all tokens in layers 1–2 and only the layer-specific top 20% from layer 3 onward. Performance is reported relative to the vanilla baseline.*

> 💡 **Figure 4 批读**:
> - 关键发现：**各层选择重要视觉 token 的能力是非单调的**
> - 不是越深的层越好，而是中间层（如 layer 15）表现最强
> - 早期层波动明显，说明浅层的 attention 不稳定
> - 这个发现支持了"选择性剪枝层"的设计，而非简单的逐层渐进剪枝

---

The results indicate that the ability to identify important visual tokens varies across layers and is not monotonically increasing with depth. Moreover, intermediate layers generally exhibiting stronger selection capability. Accordingly, we formulate the pruning-layer selection problem as a dynamic programming task, enforcing a monotonic increase in selection capability across the chosen pruning layers.

> 💡 **剪枝层选择**: 将选择最优剪枝层建模为**动态规划问题**，约束条件是所选层的选择能力单调递增。这避免了手动选层的 heuristic。

---

Based on these two observations, we propose SwiftVLM, a training-free method that performs pruning at layers with strong selection capability while ensuring independent pruning decisions at each stage.

We first identify model-specific optimal pruning layers (e.g., $i$ and $k$ in Fig.1(c)) and fix them for evaluation at test time. After visual token pruning at layer $i$, the unselected visual tokens are preserved and re-evaluated at layer $k$ with high selection capability.

---

The key contributions are summarized as follows:

• We reveal pronounced layer-wise disparities in visual token importance and propose bypass, a novel pruning strategy that forwards unselected visual tokens to subsequent pruning layers, enabling independent selection decisions.

• We reveal that the discriminative capability of layers for identifying critical visual tokens varies significantly across depth, exhibiting non-monotonic behavior.

• We present SwiftVLM, a simple yet effective training-free method that identifies high-discriminability pruning layers via dynamic programming and employs bypass to preserve fine-grained visual details while accelerating inference.

• Extensive experiments across two VLMs on nine benchmarks show SwiftVLM substantially outperforms existing training-free methods.

> 💡 **贡献总结**:
> 1. **观察 1**: 层间 token 重要性差异 → 提出 bypass
> 2. **观察 2**: 层间判别能力非单调 → DP 选层
> 3. **方法**: SwiftVLM = bypass + DP 选层，training-free
> 4. **实验**: 2 个 VLM，9 个 benchmark，全面超越现有 training-free 方法

---

## 🔖 Section 总结

### 核心洞察
1. 现有 drop/merge 剪枝范式的根本问题：**不可逆信息丢失**
2. Token 重要性排序**跨层不一致**——浅层不重要 ≠ 深层不重要
3. 各层选择能力**非单调**——中间层往往最强
4. SwiftVLM 的两大支柱：bypass（保留信息）+ DP 选层（选对层）
