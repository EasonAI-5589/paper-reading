[← 返回 README](../README.md)

# 3. Preliminary Analysis

## 📌 预览
本节系统分析 LVLM 推理中的计算开销来源（encoding / prefilling / decoding FLOPs 比例），并将 visual token 冗余分解为 **intra-modal redundancy**（视觉内部相似性）和 **cross-modal redundancy**（视觉-文本无关性），为 ToDRE 两阶段设计提供理论动机。

---

Recently, numerous visual token compression techniques
have emerged. Most approaches [2, 10, 35, 55, 62] reduce
computational redundancy only within _partial stages_ of the
LVLM inference process, lacking a systematic analysis and
_overall_ _consideration_ . To bridge this gap, we provide a
deeper analysis organized as follows. In Section 3.1, we
review the fundamental architecture and processing flow of
existing LVLMs, identifying where redundant computation
arises. In the following Section 3.2, we further provide empirical observations and examine the limitations of existing
redundancy-reduction strategies, which motivate us to propose a two-stage token pruning method. In the Appendix,
a theoretical proof is presented to validate the underlying
rationale and structural integrity of the proposed two-stage
paradigm.

> 💡 **批注**: 开篇指出现有方法只关注部分阶段的冗余，缺乏系统性分析。这为 ToDRE 的全局两阶段设计提供了切入点。

---

## 3.1. Computational Overhead in LVLM Processing Pipeline

**Architecture** **and** **Processing** **Flow.** Typically, existing
LVLMs consist of three main components: a vision encoder,
a vision-language projector, and a LLM decoder. Both the
encoder and decoder are built upon the Transformer blocks
[54]. Given a visual input _V_, the vision encoder extracts
visual features, which are then mapped into a sequence of visual token embeddings _Ev_ by the vision-language projector,
aligned with the LLM textual embedding space. Then, _Ev_ is
concatenated with text embeddings _Et_ and system prompt
embeddings _Es_ to form the input sequence for LLM. During the LLM's prefilling stage, all input tokens interact via
self-attention to generate a contextualized representation, denoted as _X_ = _{_ _**z**_ _s_ 1 _, . . .,_ _**z**_ _sL,_ _**z**_ _v_ 1 _, . . .,_ _**z**_ _vM,_ _**z**_ _t_ 1 _, . . .,_ _**z**_ _tN }_,
where _L_, _M_ and _N_ denote the sequence lengths of system
prompt token _**Z**_ _s_, visual token _**Z**_ _v_, and text token _**Z**_ _t_, respectively. At each Transformer layer, _X_ is projected into
keys and values, which are then stored as KV cache. In the
subsequent decoding stage, keys and values are computed
and added only for newly generated tokens, while previously computed key-value pairs are retrieved from the cache
directly.

> 💡 **批注**: 标准 LVLM pipeline 回顾。关键点：prefilling 阶段所有 token 全参与 self-attention（O(n²)），decoding 阶段只算新 token 但要读整个 KV cache。

---

**Computational Cost Analysis.** Prior studies [19, 38] have
shown that the dominant contributors to inference cost in
LVLMs are the vision-encoding stage, the LLM prefilling
stage, and the LLM decoding stage, each of which incurs
substantial self-attention and feed-forward network (FFN)
computations. Following previous studies [10, 55], we formulate the calculation of floating-point operations (FLOPs)
as follows:


FLOPsencoding = FLOPsprefilling = _T_ _×_ ​4 _nd_ [2] + 2 _n_ [2] _d_ + 2 _ndm_ ​,
(1)



= _T_ ​4 _Ld_ [2] + 2 _Ldm_ + _dL_ (2 _n_ + _L −_ 1)​ _,_
(2)



FLOPsdecoding = _T_



_L_



_t_ =1



​4 _d_ [2] + 2 _d_ ( _n_ + _t −_ 1) + 2 _dm_ 


where _T_ is the number of transformer layers; _n_ and _L_
respectively denote the lengths of the input and output sequences; _d_ is size of the hidden state; and _m_ is the intermediate dimension of the FFN. We take LLaVA-NeXT-7B
[37], which employs CLIP-ViT-Large-Patch14 [45] vision
encoder and Vicuna-7B-v1.5 [12] LLM decoder, as an example. The relative ratio of FLOPs (with _n_ =3000 and _L_ =20)
is approximately encoding:prefilling:decoding _≈_ 1: **63.6** :0.4.
When scaled to LLaVA-NeXT-13B, the relative ratio shifts
to 1: **121.1** :0.8, indicating that the LLM's prefilling and decoding stages roughly double their share of the total computational cost. This underscores the importance of pruning
visual tokens as early as possible—ideally _prior to_ or _during_
the LLM prefilling stage—to mitigate the exploding computational burden.

> 💡 **批注 — 关键数字**:
> - LLaVA-NeXT-7B: encoding : prefilling : decoding ≈ 1 : **63.6** : 0.4
> - LLaVA-NeXT-13B: 1 : **121.1** : 0.8
> - **Prefilling 占绝对主导**（>98%），所以在 LLM 之前或 prefilling 阶段做 pruning 收益最大
> - 这解释了为什么 Stage 1（embedding space pruning）能带来最大的加速

---

## 3.2. Intra- and Inter-Modal Redundancy

The core objective of visual token pruning is to drop redundant tokens while preserving the holistic representational
capacity of visual features. Given the critical role of early
token pruning in reducing computational cost, we next examine how to effectively identify _which_ visual tokens to
prune.
A common practice is to identify the most "important"
tokens based on predefined criteria, and then apply tokenlevel pruning or merging strategies. Attention-based methods—such as averaging attention scores [10] or leveraging
attention from the [CLS] token to visual tokens [62]—are
widely adopted. However, such methods suffer from _atten-_
_tion_ _shift_, where causal decoding biases attention toward
later-positioned visual tokens [55]. Moreover, attention distributions are often imbalanced: [CLS]-based attention is
overly concentrated, while text-to-visual attention tends to
be dispersed and noisy [62]. These limitations motivate a
natural rethinking: _what is the essence of visual token redun-_
_dancy?_ While earlier studies have not delved deeply into
this issue, we argue that token redundancy manifests in two
orthogonal components: _intra-modal redundancy_ within the
visual signal, and _cross-modal redundancy_ between visual
and textual modalities.

> 💡 **批注**: 核心洞察——redundancy 不是一个单一概念，而是两个正交维度：
> 1. **Intra-modal**: visual token 之间的相似性（与任务无关）
> 2. **Cross-modal**: visual token 与 text query 的无关性（与任务相关）
> 
> 这个分解直接对应 ToDRE 的两阶段设计。

---

_Intra-modal redundancy_ occurs when visual tokens exhibit significant similarity, since highly similar tokens contribute little unique information and are thus redundant. Such
redundancy can be identified using visual-only signals, typically by measuring cosine similarity. Then, the problem
reduces to selecting a minimally redundant subset of tokens.
Here, instead of relying on complex designs for redundancy
detection, we find that retaining a maximally diverse set of
tokens more effectively preserves the visual representation.
This observation motivates us to introduce the _Diversity-_
_driven_ _Visual_ _Token_ _Selection_, acting as the first stage of
ToDRE prior to LLM prefilling.

> 💡 **批注**: **逆向思维**——与其检测冗余 token 然后删除，不如直接选择最多样的子集保留。这比 "找冗余" 更直接高效。Max-sum diversification 是组合优化中的经典问题。

---

On the other hand, LVLM's multimodal comprehension
heavily depends on textual cues [61], giving rise to _cross-_
_modal redundancy_ where visual tokens that are less relevant
to the textual information can be safely pruned. In this view,
the attention scores between visual and text modalities during the LLM prefilling stage offer a simple yet reliable signal
for token reduction. By treating cross-modal attention as
a unified whole, we avoid the previously mentioned limitations of attention-based selection strategies. Building on the
concept of decoding-stage _information migration_ proposed
in VTW [35], we further analyze its behavior during the
LLM prefilling stage. As shown in Figure 2, cross-modal
attention is prominent in early layers and gradually diminishes in deeper layers, revealing the _information migration_
phenomenon during prefilling: early layers prioritize crossmodal interaction, while deeper layers focus primarily on
uni-modality processing. This finding drives us to propose
the _Relevance-driven Visual Token Reduction_, serving as the
second stage of ToDRE during LLM prefilling.

> 💡 **批注**: 
> - Cross-modal redundancy 用 attention 衡量是合理的，但不是逐 token 选择，而是**整层移除**
> - 与 VTW 的区别：VTW 用 KL-divergence（需要 calibration），ToDRE 直接看 cross-modal attention ratio（前向传播中即可获得）
> - Information migration 从 VTW 的 decoding 阶段扩展到了 prefilling 阶段

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| LLaVA-NeXT-7B FLOPs 比例 (enc:pre:dec) | 1 : 63.6 : 0.4 |
| LLaVA-NeXT-13B FLOPs 比例 | 1 : 121.1 : 0.8 |
| Prefilling 占比（7B） | >98% |

### 核心洞察
1. **Prefilling 主导计算开销** → 应尽早 pruning
2. **冗余的正交分解**: intra-modal (diversity) + cross-modal (relevance)
3. Attention-based 方法有 positional bias 和分布不均问题
4. Information migration 在 prefilling 阶段同样成立
