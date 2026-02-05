# 3. Where to Compress Tokens in MLLMs

> ==核心分类维度：按 MLLM 模块位置分类 (Vision Encoder / Projector / LLM / Hybrid)==

---

## 3.1 Token Compression in Vision Encoder

> In MLLMs, visual data are inherently more redundant than text, leading to a substantially larger number of tokens on the vision side than on the language side. For instance, a single high-resolution image can be divided into thousands of patch tokens.
>
> ==视觉数据比文本冗余得多，单张高分辨率图像可产生数千个 patch tokens==

> Since the vision encoder (VE) is the first module to encode visual inputs, reducing visual tokens at this initial stage yields disproportionately large efficiency gains throughout the entire MLLM system.
>
> ==在 VE 阶段压缩收益最大，影响整个 pipeline==

### 3.1.1 Inside-Encoder Compression

> Inside-VE compression directly alters token flow within the encoder, reducing self-attention complexity at an early stage.
>
> ==在 ViT 内部压缩，早期降低 attention 复杂度==

#### Visual Token Dropping

**Similarity-based scoring:**
> Tokens exhibiting high similarity to global representation (CLS token) are deemed redundant and removed. Representative works include TRIM and SAINT.
>
> ==与 CLS token 相似度高的 → 冗余 → 删除==

**Attention-based scoring:**
> VisPruner and HiPrune leverage the CLS token attention to assess visual importance. VFlowOpt constructs importance map by integrating attention-derived relevance with patch-level information entropy.
>
> ==用 attention score 评估重要性（但有 positional bias 问题！）==

**Heuristic-based scoring:**
> EgoPrune leverages domain-specific heuristics from egocentric videos. METEOR adopts layer-adaptive strategy: similarity in shallow layers, class attention in deep layers.
>
> ==启发式方法：EgoPrune 用自我中心视频先验，METEOR 分层处理==

#### Visual Token Merging

> Unlike pruning which deletes tokens, merging aggregates similar tokens into compact representations.
>
> ==Merging 不删除，而是聚合相似 tokens==

**Proximity-based Merging:**
> Spatial and temporal adjacency provide natural bases for identifying redundant tokens. Neighboring patches or consecutive frames typically share similar features.
>
> ==基于空间/时间邻近性：相邻 patch 或连续帧往往相似==

**Similarity-based Merging:**
> Global similarity methods compute token importance via patch-to-class correlation or cluster semantically similar patches. Cross-modal merging methods leverage textual context.
>
> ==基于相似度：语义聚类 / 跨模态融合==

**Hybrid Strategies:**
> Sequential approaches first apply attention-based pruning, then use weighted merging to recover information from discarded tokens.
>
> ==混合策略：先 pruning 去粗粒度冗余，再 merging 恢复信息==

#### Multi-Scale Visual Compression

| 类型 | 代表方法 | 思路 |
|------|----------|------|
| Multi-Layer | LLaVA-STF, METEOR | 聚合多层 ViT 特征 |
| Multi-Encoder | Cambrian-1, METEOR | 结合自监督 + 语言监督编码器 |
| Multi-Resolution | FastVLM, ADMIRE | 高分辨率细节 + 低分辨率全局 |

---

### 3.1.2 Outside-Encoder Compression

> Compression occurs after the vision encoder produces output but before the projector. This design is **plug-and-play** and minimally invasive.
>
> ==VE 输出后、Projector 前压缩，即插即用==

#### Purely-Vision Compression

> VisionZip identifies reusable tokens through importance estimation and representativeness constraints. LLaVA-STF generates compact visual summaries through cross-layer concatenation.
>
> ==不依赖文本，纯视觉压缩：VisionZip、LLaVA-STF==

**⚠️ Visual Attention Bias Problem:**
> Recent works reveal that attention-based selection exhibits bias toward salient regions (foreground objects), neglecting global context. HoloV addresses this by incorporating global visual context.
>
> ==Attention-based 选择偏向前景显著区域，忽略背景！HoloV 解决这个问题==

#### Text-Guided Compression

> PAR parses queries into entities and actions and re-weights visual tokens. QG-VTC computes question-to-vision similarity to guide token retention.
>
> ==利用文本 query 指导压缩：PAR 解析实体+动作，QG-VTC 计算问题-视觉相似度==

> Text-guided compression methods demonstrate particular robustness at Outside-VE position: visual token semantics are fully encoded while cross-modal interaction has not yet begun.
>
> ==Text-guided 在 Outside-VE 位置效果好：视觉已编码完，跨模态还没开始==

---

## 3.2 Token Compression in Projector

> The projector module plays a pivotal role in bridging the vision encoder and the language model.
>
> ==Projector = 视觉编码器和 LLM 之间的桥梁==

### 3.2.1 Transformation-Based Compression

| 方法 | 原理 | 代表作 |
|------|------|--------|
| **Pooling** | 平均池化下采样 | MobileVLM V2 (LDP), DeCo, PLLaVA |
| **Pixel Shuffle** | 空间分辨率 → 通道维度 | InternVL 1.5, NVLM, Qwen2VL |
| **Convolution** | 可学习卷积聚合 | Honeybee (C-Abstractor), MobileVLM V2 |

> Pooling: parameter-free and computationally efficient.
> Pixel Shuffle: trades token count for channel dimensionality.
> Convolution: selectively integrates local information through learnable weights.
>
> ==Pooling 无参高效，Pixel Shuffle 换通道深度，Conv 可学习聚合==

### 3.2.2 Query-Based Compression

**Q-Former (BLIP-2):**
> Q-Former employs a small set of learnable query vectors that interact with frozen visual features via stacked self-attention and cross-attention layers. It efficiently compresses hundreds of visual tokens into only a few.
>
> ==Q-Former：可学习 queries + cross-attention，压缩数百 tokens 到几个==

**Q-Former 变体:**
| 变体 | 改进 |
|------|------|
| Qwen-VL | 单层 cross-attention，简化架构 |
| Honeybee | C-Abstractor / D-Abstractor 增强局部性 |
| MQT | 可变 query 数量，平均减少一半 tokens |
| TG-LLaVA | 文本指导的 query |
| LLaVA-Mini | Modality-Pre Fusion 缓解信息丢失 |

**Cross-Attention-Based:**
> TokenPacker employs coarse-to-fine strategy: downsampled features as queries, iteratively interact with high-resolution features.
>
> ==TokenPacker：粗到细，下采样特征作为 query==

### 3.2.3 Importance-Driven Compression

> Identify relative importance of tokens and selectively prune or merge less informative ones.
>
> ==评估重要性，选择性剪枝/合并==

| 方法类型 | 代表作 | 思路 |
|----------|--------|------|
| Similarity | DynTok, LLaVA-Scissor | cosine similarity / SCC 图分割 |
| Saliency | SeqCompression | K-means++ 聚类 + 合并 |
| Diversity | DivPrune | Max-Min Diversity Problem |

---

## 3.3 Token Compression in LLM

> Given that the LLM component generally contains significantly more parameters than the vision encoder and projector, the resulting sequence incurs substantial computational overhead.
>
> ==LLM 参数最多，压缩收益直接==

### 3.3.1 Compression in Prefilling Stage

> Once a vision token is removed in shallow layers, deeper layers cannot access information from the corresponding image region.
>
> ==浅层删了就没了，深层无法恢复！==

#### Importance-based

**FastV:**
> First to observe that vision tokens receive substantially lower attention scores compared to text tokens, revealing extreme sparsity. FastV prunes half of vision tokens at the second layer.
>
> ==FastV 首次发现：vision tokens attention 极度稀疏 → 第 2 层删 50%==

**PyramidDrop:**
> Identified that redundancy of vision tokens increases with LLM depth → multi-stage progressive pruning.
>
> ==PyramidDrop：越深层冗余越高 → 渐进剪枝==

**⚠️ Attention Bias Problem:**
> Feather noted that vision tokens located near output tokens tend to receive disproportionately high attention scores due to RoPE's long-term decay property.
>
> ==Attention Bias：靠近输出的 tokens 得分偏高（RoPE 长期衰减导致）==

**⚠️ Flash Attention Compatibility:**
> Flash Attention does not directly expose attention scores → need workarounds (selective recomputation).
>
> ==Flash Attention 不暴露 attention scores，需要变通方案==

#### Learnable Module-based

> Trainable components learn to assess token importance or determine compression ratio.
>
> ==可学习模块评估重要性 / 预测压缩率==

| 方法 | 思路 |
|------|------|
| p-MoD | 权重预测器，每层保留 Top R% tokens |
| GlimpsePrune | 预测器估计每层 token 重要性 |
| DyRate | 轻量分类器预测最优剪枝率 |
| ATP-LLaVA | MLP 双头预测实例级阈值 |

#### Token Merging-based

> Softer compression strategy: compute similarity and fuse multiple tokens into fewer representative ones.
>
> ==软压缩：相似度计算 + 聚类/合并==

| 方法 | 思路 |
|------|------|
| LLaVolta | 简单平均池化，多阶段训练补偿性能损失 |
| FiCoCo | 重要 tokens 先选出，再计算相关矩阵指导合并 |
| FrameFusion | 跨帧 cosine similarity 合并时空相似 tokens |
| HoliTom | 直接合并低 attention 的 tokens |

#### Fusion-based

> Cross-attention or self-attention to integrate visual information, avoiding excessively long sequences.
>
> ==用 cross-attention 注入视觉信息，避免长序列==

| 方法 | 思路 |
|------|------|
| Flamingo | GATED XATTN-DENSE layers |
| mPLUG-Owl3 | 文本自注意力 + 跨模态 attention |
| CrossLMM | 压缩 visual + text 作为 queries |
| VoCo-LLaMA | 单个 Vision Compression token |

### 3.3.2 Compression in Decoding Stage (KV Cache)

> Reduce memory and computational overhead of cached K/V tensors during autoregressive decoding.
>
> ==KV Cache 压缩：减少解码阶段的 K/V 缓存开销==

| 方法 | 思路 |
|------|------|
| LOOK-M | 累积 attention scores 估计重要性 |
| MustDrop | Prefilling 保留的 tokens 的 KV 存储 |
| SparseMM | 识别 visual heads，分配更多 KV budget |
| DyCoke | 动态压缩，attention 分布变化时更新 KV |
| Video-XL-2 | Bi-level KVs，动态选择 dense/sparse |

---

## 3.4 Token Compression in Multi-Module (Hybrid)

> Integrate multiple stages of reduction — from early spatial downsampling to late-stage pruning — to maximize efficiency and quality.
>
> ==多模块协同/渐进压缩，端到端优化==

### 3.4.1 Collaborative Compression

| 方法 | 协同方式 |
|------|----------|
| CrossGET | 视觉+语言分支都插入压缩模块 |
| LLaMA-VID | 每帧压缩为 2 tokens (context + content) |
| PAR | 外部冗余 (query 无关) + 内部冗余 (相似) |

### 3.4.2 Progressive Compression

| 方法 | 多阶段策略 |
|------|-----------|
| MustDrop | VE + Prefilling + Decoding 三阶段 |
| DyCoke | 1) 相邻帧 cosine similarity 合并 2) KV 动态剪枝 |
| FiCoCo | Filter → Correlate → Compress |

---

## 💡 Key Takeaways

| 压缩位置 | 优势 | 挑战 |
|----------|------|------|
| Vision Encoder (Inside) | 早期压缩收益大 | 需要修改 VE 架构 |
| Vision Encoder (Outside) | 即插即用 | Attention bias 问题 |
| Projector | 自然融合点 | 信息瓶颈 |
| LLM Prefilling | 参数最多，收益直接 | 删了无法恢复 |
| LLM Decoding | 长生成必需 | Flash Attention 兼容性 |
| Hybrid | 端到端优化 | 设计复杂 |

---

*[返回论文目录](../README.md)*
