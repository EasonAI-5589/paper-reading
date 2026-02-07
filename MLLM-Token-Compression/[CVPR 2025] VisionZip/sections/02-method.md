# 2. VisionZip

> 来源: VisionZip (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 这是方法核心。先讲为什么减少视觉 token 能提效（2.1），再讲冗余观察（2.2），然后是两步压缩方法（2.3），最后是高效微调（2.4）和使用场景（2.5）。

---

### 2.1 Preliminary

> 💡 **2.1 要点预览**: VLM 的三件套架构和计算复杂度分析——为什么减少 $n_{\text{img}}$ 最有效？

**Architecture of VLM.** VLM 由三部分组成：visual encoder → modality projector → LLM。Visual encoder（如 CLIP）将图像转成 visual tokens，projector 对齐到 LLM 的 embedding space，LLM 融合视觉+文本信息生成回答。

**Computation Complexity.** Total FLOPs = $T \times (4nd^2 + 2n^2d + 2ndm)$

其中 $T$ 是 transformer 层数，$n$ 是序列长度，$d$ 是 hidden dim，$m$ 是 FFN intermediate size。

> 💡 **批注**: 关键是 $n^2$ 项！序列长度 $n = n_{\text{sys}} + n_{\text{img}} + n_{\text{question}}$，而 $n_{\text{img}}$ 通常是其他两项之和的 **20 倍**。
> ```
> 大白话: FLOPs 和序列长度的关系：
> ├── 线性项: 4nd² + 2ndm  → 减 token 线性省算力
> └── 二次项: 2n²d         → 减 token 二次方省算力 ⭐
>
> 例: n_img 从 2880 减到 160（减 18 倍）
> → 二次项省 ~18² = 324 倍！
> ```

> 💡 **2.1 小结**: 减少 $n_{\text{img}}$ 是提升 VLM 效率的最有效途径，因为 self-attention 的计算量与序列长度平方成正比。

---

### 2.2 Redundancy Observation

> 💡 **2.2 要点预览**: 实验证明 CLIP/SigLIP 输出的视觉 token 绝大部分冗余——attention 集中在少数 token 上。

In popular VLMs like LLaVA and MiniGemini, the number of vision tokens far exceeds that of text tokens. We randomly sampled one image and visualized the attention of each token from the Vision Encoder's **-2 layer** (the selected layer for obtaining input visual tokens in most VLMs).

> 💡 **批注**: 为什么是 -2 层（倒数第二层）？
> - 最后一层的 token 要和 CLIP text branch 做 contrastive loss 对齐，特征可能 "失真"
> - -2 层更好地保留了图像本身的信息
> - 这是 LLaVA 系列的标准选择

As shown in Fig. 2, both CLIP and SigLIP exhibit an attention pattern concentrated on a limited number of tokens, while the majority receive minimal attention. We analyze the distribution on TextVQA validation set — most visual tokens receive very low attention with weights close to zero.

> 💡 **批注**: 这个观察是 VisionZip 方法的基石。关键发现：
> 1. **少数 token 聚集了大部分 attention**（dominant tokens）
> 2. **多数 token attention ≈ 0**（冗余 tokens）
> 3. 这个现象在 CLIP 和 SigLIP 中都存在（普遍性）
> 4. 在不同图片上也一致（鲁棒性）

> 💡 **2.2 小结**: Vision encoder 自身的 self-attention 机制导致信息高度集中于少数 token，这为 text-agnostic 的 token 压缩提供了理论基础。

---

### 2.3 Informative Visual Token Zip

> 💡 **2.3 要点预览**: VisionZip 的核心两步法——先选 dominant tokens（[CLS] attention），再合并 contextual tokens（key similarity）。

![Figure 3](../images/6813a4b5e049b90da83f37c32c42fa59124ecac14490cf4851e38665e375b398.jpg)
*Figure 3: VisionZip 框架。用 attention score 选 dominant tokens，用 similarity 合并 contextual tokens。*

> 💡 **Figure 3 批读**:
> ```
> 输入: Vision Encoder 输出的全量 visual tokens (e.g., 576)
>       ↓
> Step 1: Dominant Token Selection
> ├── 有 CLS token (CLIP): 用 CLS 对各 token 的 attention score
> ├── 无 CLS token (SigLIP): 用各 token 被其他 token attend 的平均值
> └── 选 Top-K 个 attention 最高的 → dominant tokens
>       ↓
> Step 2: Contextual Token Merging
> ├── 剩余 token 均匀分成 target 和 merge 两组
> ├── 用 Key 向量的点积计算 similarity
> ├── 每个 merge token 分配给最相似的 target
> └── 平均合并 → contextual tokens
>       ↓
> 输出: dominant tokens + contextual tokens → Projector → LLM
> ```

#### Dominant Token Selection

We evaluate the importance of each visual token by examining its attention scores within the vision encoder:

$$S_h = \text{Softmax}\left(\frac{Q_h K_h^\top}{\sqrt{D_h}}\right)$$

Averaging across heads yields $S_{\text{avg}} \in \mathbb{R}^{B \times \text{SeqLen} \times \text{SeqLen}}$.

**For CLIP (has CLS token):** Use CLS token's attention scores to select top-K tokens.

**For SigLIP (no CLS token):** Calculate average attention each token receives from all others.

> 💡 **批注**: 为什么用 [CLS] 的 attention？
> ```
> CLS token 的设计目的就是聚合全图信息
> → CLS 关注哪些 token = 这些 token 包含最多信息
> → 直接用 CLS attention 排序就行，简单暴力有效
> ```
> 对于没有 CLS 的 SigLIP，退而求其次用 "被其他 token 平均关注度" 来代替，思路一致。

![Algorithm 1](../images/e24e3002f75ca359334523473c2b57c8bf923ed18d67363ef2236f23726de908.jpg)
*Algorithm 1: Dominant Token Selection 伪代码*

> 💡 **Algorithm 1 批读**: 核心就 3 步——拿 attention → 取 CLS 那行 → topk 选 token。非常简洁。

#### Contextual Token Merging

Although dominant tokens contain most visual information, we merge remaining tokens to avoid losing small but potentially important details.

> 💡 **批注**: Token merging 的思路借鉴了 ToMe (Token Merging)。关键设计：
> ```
> 1. 为什么用 Key 向量计算 similarity？
>    → Key 在 self-attention 中本来就是 "我包含什么信息" 的摘要
>    → 两个 Key 相似 = 两个 token 信息相似 → 可以合并
>
> 2. 为什么 uniform split？
>    → 均匀采样保证空间覆盖
>    → 避免合并后信息分布不均
>
> 3. 合并方式: 简单平均
>    → 简单有效，没搞复杂的加权
> ```

**Algorithm 2 伪代码**:
```python
remaining = vanilla_tokens.mask(dominant_tokens)  # 去掉 dominant
targets, merge = uniform_split(remaining, M)       # 均匀分成两组
similarity = bmm(merge.K, targets.K.T)             # Key 相似度
assign_idx = similarity.argmax(dim=2)              # 每个 merge → 最相似的 target
context_tokens = avg_merge(assign_idx, targets, merge)  # 平均合并
```

> 💡 **2.3 小结**:
> - Dominant selection: [CLS] attention → Top-K → 信息密度最高的 token
> - Contextual merging: Key similarity → 合并冗余 token → 保留细节
> - 整个过程在 vision encoder 端完成，不需要 LLM 参与
> - Token 数量配比（以 LLaVA-1.5 保留 64 token 为例）：54 dominant + 10 contextual

---

### 2.4 Efficient Tuning

> 💡 **2.4 要点预览**: token 数骤降导致轻微 misalignment，用极少数据微调 projector 即可修复。

The reduction in visual tokens can lead to misalignment, as the VLM model originally trained on all full visual tokens may struggle to adapt to the sudden decrease.

We use minimal instruction tuning data to efficiently fine-tune the **multimodal projector** while keeping other components frozen.

> 💡 **批注**: 微调细节：
> | 项目 | 值 |
> |------|-----|
> | 微调组件 | 仅 projector |
> | 数据量 | 1/10 LLaVA-1.5 数据 |
> | 时间 | 30 min on 8×A800 |
> | 也可用 | 3090 GPU |
> | 效果 | 64 token 时性能从 94.0% → 95.2% |
>
> 这个成本几乎可以忽略不计，性价比极高。

> 💡 **2.4 小结**: VisionZip‡ 通过极低成本的 projector 微调，弥补了 token 减少带来的 misalignment，是实际部署的推荐方式。

---

### 2.5 Usage of VisionZip

> 💡 **2.5 要点预览**: VisionZip 的适用范围和独特优势。

VisionZip can adapt to multiple tasks:
- Image and video understanding
- Multi-turn conversations (previous efficient VLMs could not handle)
- Compatible with all existing LLM acceleration algorithms
- Plug-and-play for vision encoders
- 13B VLM can be faster than 7B VLM

> 💡 **批注**: VisionZip 的核心优势总结：
> ```
> 1. Text-agnostic → 多轮对话不掉分
> 2. 在 encoder 端压缩 → LLM 端所有加速技术都兼容
> 3. Plug-and-play → 不改模型结构
> 4. 13B+VisionZip > 7B → 实际部署中可以用更大模型
> ```

---

## 💡 Section 总结

### 方法流程图
```
Image → Vision Encoder (CLIP/SigLIP)
         ↓ (-2 layer)
    All visual tokens (576/2880)
         ↓
    ┌── Dominant Token Selection (CLS attention → Top-K)
    │   └── e.g., 54 tokens
    └── Contextual Token Merging (Key similarity → avg merge)
        └── e.g., 10 tokens
         ↓
    Selected tokens (64) → Projector → LLM
                            ↑
                    (可选: 微调 projector 30min)
```

### 关键设计选择
| 设计 | 选择 | 理由 |
|------|------|------|
| 重要性度量 | CLS attention | CLS 天然聚合全图信息 |
| 合并度量 | Key 向量相似度 | Key 是信息内容的摘要 |
| 压缩位置 | Vision encoder 端 | Text-agnostic, 效率更高 |
| 微调目标 | 仅 projector | 成本极低, 效果显著 |
