[← 返回 README](../README.md)

# 3. Preliminary Analysis

## 📌 预览
分析 LVLM 推理流程中的计算开销分布（encoding:prefilling:decoding ≈ 1:63.6:0.4），指出 prefilling 是瓶颈；然后从 intra-modal（token similarity）和 cross-modal（attention）两个维度解剖冗余，为两阶段设计提供理论动机。

---

Recently, numerous visual token compression techniques have emerged. Most approaches [2, 10, 35, 55, 62] reduce computational redundancy only within partial stages of the

LVLM inference process, lacking a systematic analysis and overall consideration. To bridge this gap, we provide a deeper analysis organized as follows. In Section 3.1, we review the fundamental architecture and processing flow of existing LVLMs, identifying where redundant computation arises. In the following Section 3.2, we further provide empirical observations and examine the limitations of existing redundancy-reduction strategies, which motivate us to propose a two-stage token pruning method. In the Appendix, a theoretical proof is presented to validate the underlying rationale and structural integrity of the proposed two-stage paradigm.

> 💡 **分析思路**: 现有工作只在局部阶段减少冗余，缺乏全局视角 → ToDRE 做系统分析后提出覆盖 embedding space + decoder 的两阶段方案。

---

# 3.1. Computational Overhead in LVLM Processing Pipeline

Architecture and Processing Flow. Typically, existing LVLMs consist of three main components: a vision encoder, a vision-language projector, and a LLM decoder. Both the encoder and decoder are built upon the Transformer blocks [54]. Given a visual input $V$ , the vision encoder extracts visual features, which are then mapped into a sequence of visual token embeddings $E _ { v }$ by the vision-language projector, aligned with the LLM textual embedding space. Then, $E _ { v }$ is concatenated with text embeddings $E _ { t }$ and system prompt embeddings $E _ { s }$ to form the input sequence for LLM. During the LLM's prefilling stage, all input tokens interact via self-attention to generate a contextualized representation, denoted as $X = \{ z _ { s _ { 1 } } , \dotsc , z _ { s _ { L } } , z _ { v _ { 1 } } , \dotsc , z _ { v _ { M } } , z _ { t _ { 1 } } , \dotsc , z _ { t _ { N } } \}$ , where $L$ , $M$ and $N$ denote the sequence lengths of system prompt token $Z _ { s }$ , visual token $Z _ { v }$ , and text token $\scriptstyle { Z _ { t } }$ , respectively. At each Transformer layer, $X$ is projected into keys and values, which are then stored as KV cache. In the subsequent decoding stage, keys and values are computed and added only for newly generated tokens, while previously computed key-value pairs are retrieved from the cache directly.

> 💡 **LVLM 三阶段推理流程**:
> 1. **Vision Encoding**: ViT 提取特征 → projector 映射到 LLM embedding space
> 2. **LLM Prefilling**: 全量 token（system + visual + text）做 self-attention → 生成 KV cache
> 3. **LLM Decoding**: 逐 token 生成，复用 KV cache
>
> Visual token 在 prefilling 阶段参与 O(n²) 的 self-attention，是计算瓶颈的主要来源。

---

Computational Cost Analysis. Prior studies [19, 38] have shown that the dominant contributors to inference cost in LVLMs are the vision-encoding stage, the LLM prefilling stage, and the LLM decoding stage, each of which incurs substantial self-attention and feed-forward network (FFN) computations. Following previous studies [10, 55], we formulate the calculation of floating-point operations (FLOPs) as follows:

$$
{ \mathrm { F L O P s } } _ { \mathrm { e n c o d i n g } } = { \mathrm { F L O P s } } _ { \mathrm { p r e f i l l i n g } } = T \times \left( 4 n d ^ { 2 } + 2 n ^ { 2 } d + 2 n d m \right) ,
$$

$$
\begin{array} { c } { { \mathrm { F L O P s } _ { \mathrm { d e c o d i n g } } = \displaystyle T \sum _ { t = 1 } ^ { L } \left( 4 d ^ { 2 } + 2 d ( n + t - 1 ) + 2 d m \right) } } \\ { { = T \left( 4 L d ^ { 2 } + 2 L d m + d L ( 2 n + L - 1 ) \right) , } } \end{array}
$$

where $T$ is the number of transformer layers; $n$ and $L$ respectively denote the lengths of the input and output sequences; $d$ is size of the hidden state; and $m$ is the intermediate dimension of the FFN. We take LLaVA-NeXT-7B [37], which employs CLIP-ViT-Large-Patch14 [45] vision encoder and Vicuna-7B-v1.5 [12] LLM decoder, as an example. The relative ratio of FLOPs (with $\scriptstyle n = 3 0 0 0$ and $L { = } 2 0$ ) is approximately encoding:prefilling:decoding $\approx 1 { : } 6 3 . 6 { : } 0 . 4$ . When scaled to LLaVA-NeXT-13B, the relative ratio shifts to 1:121.1:0.8, indicating that the LLM's prefilling and decoding stages roughly double their share of the total computational cost. This underscores the importance of pruning visual tokens as early as possible—ideally prior to or during the LLM prefilling stage—to mitigate the exploding computational burden.

> 💡 **FLOPs 分布（关键数字）**:
>
> | 阶段 | 7B (n=3000, L=20) | 13B |
> |------|-------------------|-----|
> | Encoding | 1 | 1 |
> | Prefilling | **63.6** | **121.1** |
> | Decoding | 0.4 | 0.8 |
>
> - Prefilling 占比碾压性主导 → **在 prefilling 前/期间剪枝最有效**
> - 模型越大，prefilling 占比越高 → 大模型更需要 token 压缩
> - 这解释了为什么 Stage 1（embedding space 压缩）收益远大于 Stage 2

---

# 3.2. Intra- and Inter-Modal Redundancy

The core objective of visual token pruning is to drop redundant tokens while preserving the holistic representational capacity of visual features. Given the critical role of early token pruning in reducing computational cost, we next examine how to effectively identify which visual tokens to prune.

A common practice is to identify the most "important" tokens based on predefined criteria, and then apply tokenlevel pruning or merging strategies. Attention-based methods—such as averaging attention scores [10] or leveraging attention from the [CLS] token to visual tokens [62]—are widely adopted. However, such methods suffer from attention shift, where causal decoding biases attention toward later-positioned visual tokens [55]. Moreover, attention distributions are often imbalanced: [CLS]-based attention is overly concentrated, while text-to-visual attention tends to be dispersed and noisy [62]. These limitations motivate a natural rethinking: what is the essence of visual token redundancy? While earlier studies have not delved deeply into this issue, we argue that token redundancy manifests in two orthogonal components: intra-modal redundancy within the visual signal, and cross-modal redundancy between visual and textual modalities.

> 💡 **Attention-based 方法的两大缺陷**:
> 1. **Attention shift (positional bias)**: Causal decoding 偏向后部 token → 前部有信息的 token 被误删
> 2. **分布不平衡**: [CLS] attention 过于集中；text-to-visual attention 分散且嘈杂
>
> **冗余的两个正交维度**:
> - **Intra-modal**: visual token 之间的相似性（空间冗余）
> - **Cross-modal**: visual token 与 text 的相关性（任务冗余）

---

Intra-modal redundancy occurs when visual tokens exhibit significant similarity, since highly similar tokens contribute little unique information and are thus redundant. Such redundancy can be identified using visual-only signals, typically by measuring cosine similarity. Then, the problem reduces to selecting a minimally redundant subset of tokens. Here, instead of relying on complex designs for redundancy detection, we find that retaining a maximally diverse set of tokens more effectively preserves the visual representation. This observation motivates us to introduce the Diversitydriven Visual Token Selection, acting as the first stage of ToDRE prior to LLM prefilling.

> 💡 **Intra-modal 冗余 → Stage 1 动机**:
> - 相似的 token 携带重复信息 → 保留最 diverse 的子集即可
> - 不用检测"哪些冗余"，而是**直接选最不相似的**
> - 纯视觉信号，不需要 text query → 可以在 LLM 前执行

---

On the other hand, LVLM's multimodal comprehension heavily depends on textual cues [61], giving rise to crossmodal redundancy where visual tokens that are less relevant to the textual information can be safely pruned. In this view, the attention scores between visual and text modalities during the LLM prefilling stage offer a simple yet reliable signal for token reduction. By treating cross-modal attention as a unified whole, we avoid the previously mentioned limitations of attention-based selection strategies. Building on the concept of decoding-stage information migration proposed in VTW [35], we further analyze its behavior during the LLM prefilling stage. As shown in Figure 2, cross-modal attention is prominent in early layers and gradually diminishes in deeper layers, revealing the information migration phenomenon during prefilling: early layers prioritize crossmodal interaction, while deeper layers focus primarily on uni-modality processing. This finding drives us to propose the Relevance-driven Visual Token Reduction, serving as the second stage of ToDRE during LLM prefilling.

> 💡 **Cross-modal 冗余 → Stage 2 动机**:
> - 与 text 无关的 visual token 可安全删除
> - VTW [35] 在 decoding 阶段发现 information migration，ToDRE 将分析扩展到 **prefilling 阶段**
> - 浅层：active cross-modal interaction → 不能删
> - 深层：uni-modal text processing → 可以全删
> - 将 cross-modal attention 作为整体（而非逐 token）评估，避免 positional bias

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| FLOPs 比例 (7B) | encoding:prefilling:decoding ≈ 1:63.6:0.4 |
| FLOPs 比例 (13B) | 1:121.1:0.8 |
| 示例输入长度 n | 3000 |
| 示例输出长度 L | 20 |

### 核心洞察
1. Prefilling 是绝对计算瓶颈 → 越早剪枝越好
2. Attention-based pruning 有 positional bias 和分布不平衡问题
3. 冗余 = intra-modal (diversity) + cross-modal (relevance)，正交分解
4. Information migration 在 prefilling 阶段同样成立 → Stage 2 的理论基础
