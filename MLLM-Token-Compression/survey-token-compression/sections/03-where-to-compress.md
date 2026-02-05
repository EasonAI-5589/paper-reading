# 3. Where to Compress Tokens in MLLMs

> Based on the taxonomy illustrated in Figure 2, we systematically categorize existing token compression methods according to **where** compression is applied within the MLLM architecture.
>
> ==核心分类维度：压缩发生在 MLLM 的哪个位置==

压缩可以在三个模块逐步部署：
1. **Vision Encoder (§3.1)**: 在视觉感知阶段减少计算开销
2. **Projector (§3.2)**: 在视觉→语言空间转换时整合 token reduction
3. **LLM (§3.3)**: 实现整体跨模态效率优化
4. **Hybrid (§3.4)**: 多模块协同压缩

---

## 3.1 Token Compression in Vision Encoder

> Visual data are inherently more redundant than text, leading to a substantially larger number of tokens on the vision side. A single high-resolution image can be divided into thousands of patch tokens.
>
> ==视觉数据冗余度远高于文本，高分辨率图像可产生数千个 patch tokens==

> Since the vision encoder is the first module to encode visual inputs, reducing visual tokens at this initial stage yields **disproportionately large efficiency gains** throughout the entire MLLM system.
>
> ==在 VE 阶段压缩的收益最大：早期减少的 token 会在后续所有模块中累积节省==

### 3.1.1 Inside-Encoder Compression

在 ViT 内部压缩，直接改变 token 流动。核心问题：
1. 如何处理"不重要"的 tokens（pruning or merging）
2. 如何跨层协调压缩（multi-scale）

#### Visual Token Dropping

> Token dropping methods compute importance scores for visual tokens and retain only the most salient ones, directly discarding the remainder. Implementation typically follows a "**ranking + Top-K**" paradigm.
>
> ==Token Dropping = 重要性打分 + 排序 + 保留 Top-K==

**三种打分策略：**

| 策略 | 方法 | 原理 |
|------|------|------|
| **Similarity-based** | TRIM, SAINT | 与全局表示（CLS/平均向量）相似度高的视为冗余 |
| **Attention-based** | VisPruner, HiPrune, MADTP | 利用 ViT attention 权重判断 saliency |
| **Heuristic-based** | EgoPrune, METEOR | 利用任务先验（如第一人称视频的运动区域） |

> METEOR adopts a layer-adaptive strategy: similarity to average token in shallow layers (low-level redundancy), class attention scores in deep layers (semantic information).
>
> ==METEOR 的层自适应策略：浅层用相似度，深层用 class attention==

#### Visual Token Merging

> Unlike pruning which deletes tokens outright, merging aggregates similar tokens into compact representations to preserve information while shortening sequences.
>
> ==Merging vs Dropping：合并保留信息，删除丢弃信息==

**三种合并策略：**

| 策略 | 方法 | 原理 |
|------|------|------|
| **Proximity-based** | Chat-UniVi, TESTA | 空间/时间相邻的 tokens 往往相似 |
| **Similarity-based** | CrossGET, LLaVA-PruMerge | 基于特征空间相似度，可跨越空间距离 |
| **Hybrid** | FiCoCo, LookupViT | 先 pruning 去粗粒度冗余，再 merging 恢复信息 |

> **ToMe (开山之作)**: Bipartite soft matching 合并相似 tokens，简单有效

#### Multi-Scale Visual Compression

> Single-scale compression methods operate at fixed granularity, struggling to obtain comprehensive visual details. Multi-scale approaches coordinate compression across layers, encoders, or resolutions.
>
> ==单尺度压缩的局限：固定粒度无法兼顾细节和全局==

| 类型 | 方法 | 思路 |
|------|------|------|
| **Multi-Layer** | LLaVA-STF, METEOR | 从多个 ViT 层提取特征融合 |
| **Multi-Encoder** | Cambrian-1, METEOR | 结合 DINO (自监督) + CLIP (语言监督) |
| **Multi-Resolution** | FastVLM, ADMIRE | 高分辨率保细节 + 低分辨率保全局 |

---

### 3.1.2 Outside-Encoder Compression

> Compression occurs **after** the vision encoder produces its output but **before** the projector. This position offers stronger plug-and-play capability, requiring no modification to encoder layers.
>
> ==Outside-VE：VE 输出后、Projector 前，即插即用==

#### Purely-Vision Compression

> Purely-vision methods downsample or aggregate encoder outputs based solely on **vision-vision semantic relevance**, independent of user queries.
>
> ==仅依赖视觉信息，不需要文本引导==

代表方法：
- **VisionZip**: 重要性估计 + 代表性约束，可达 16x 压缩
- **Fourier-VLM**: 频域低通滤波去高频冗余
- **LLaVA-PruMerge**: 近邻聚类 + learnable merging，10x 压缩性能接近无压缩

> **Visual Attention Bias Problem**: Early works leverage CLS token attention for selection. However, attention-based selection exhibits bias toward salient regions (foreground), neglecting global context.
>
> ==Attention Bias 问题：CLS attention 偏向前景显著区域，忽略背景上下文==

> **HoloV** addresses this by incorporating global visual context to balance foreground and background tokens from a holistic perspective.
>
> ==HoloV 解决方案：从整体视角平衡前景和背景 tokens==

#### Text-Guided Compression

> When textual prompts provide semantic priors, compression can focus on **question-relevant regions or frames**, realizing context-oriented efficiency.
>
> ==文本引导：利用问题语义聚焦相关区域==

代表方法：
- **PAR**: 解析 query 为 entities/actions，重新加权视觉 tokens
- **QG-VTC**: 计算 question-to-vision 相似度引导保留
- **LongVU**: 跨模态 query + 帧/区域候选，先过滤再精选

> Text-guided methods demonstrate particular robustness at the Outside-VE position: visual token semantics are fully encoded while cross-modal interaction has not yet begun.
>
> ==Outside-VE 位置是 text-guided 的最佳位置：视觉已编码但跨模态还未开始==

---

## 3.2 Token Compression in Projector

> The projector module bridges the vision encoder and the language model. It transforms raw visual embeddings into language-compatible representations.
>
> ==Projector 是视觉→语言的桥梁==

三大类方法：

### 3.2.1 Transformation-Based Compression

> Transformation-based methods reduce tokens by directly transforming the spatial structure of visual feature maps through **lightweight, deterministic transformations**.
>
> ==变换型：轻量级确定性变换==

| 方法 | 原理 | 代表工作 |
|------|------|---------|
| **Pooling** | 参数无关的下采样，保留主语义 | MobileVLM V2 (2×2 avg pool), PLLaVA |
| **Pixel Shuffle** | 空间分辨率换通道维度 | InternVL 1.5, Qwen2VL |
| **Convolution** | 可学习的局部聚合 | Honeybee, VideoLLaMA2 |

> Pixel Shuffle: $Y = \text{reshape}(X, H/r, W/r, C \cdot r^2)$
> 
> ==Pixel Shuffle 公式：减少 $r^2$ 倍 token，增加 $r^2$ 倍通道==

### 3.2.2 Query-Based Compression

> Query-based compression leverages a limited number of **learnable query embeddings** to attend to dense visual features and distill them into a compact representation.
>
> ==Query-based：用少量可学习 query 蒸馏视觉特征==

**Q-Former (BLIP-2):**
> Q-Former employs a small set of learnable query vectors that interact with frozen visual features via stacked self-attention and cross-attention layers.
>
> ==Q-Former = Learnable Queries + Cross-Attention + Frozen Vision Features==

**Q-Former 变体:**
- **Qwen-VL**: 单层 cross-attention，简化架构
- **Honeybee**: C-Abstractor (ResNet+pooling) 和 D-Abstractor (Deformable Attention) 增强局部性
- **LLaVA-Mini**: 额外 Modality Pre-Fusion 模块缓解信息丢失

**Cross-Attention-Based:**
- **TokenPacker**: 下采样特征作 query，与高分辨率 region 配对交互
- **mPLUG-DocOwl2**: 全局视觉特征作 query，crop 特征作 K/V

### 3.2.3 Importance-Driven Compression

> Importance-driven methods estimate the importance of each token and selectively retain the most valuable ones.
>
> ==重要性驱动：估计每个 token 重要性，保留最有价值的==

| 策略 | 方法 | 原理 |
|------|------|------|
| **Similarity-based** | DynTok, LLaVA-Scissor | 局部 token 相似度，分组合并 |
| **Saliency-based** | SeqCompression | K-means++ 聚类后平均合并 |
| **Innovative** | DivPrune | 最大最小多样性问题 (MMDP) |

---

## 3.3 Token Compression in LLM

> The LLM component generally contains significantly more parameters than the vision encoder and projector. The resulting sequence incurs substantial computational overhead.
>
> ==LLM 参数量远大于 VE 和 Projector，所以压缩收益也大==

按生成阶段分类：
- **Prefilling Stage (§3.3.1)**: 首次前向时压缩
- **Decoding Stage (§3.3.2)**: KV Cache 压缩

### 3.3.1 Compression in Prefilling Stage

#### Importance-based

> FastV was among the first to observe that vision tokens receive **substantially lower attention scores** compared to text tokens within the LLM, revealing the **extreme sparsity** in the information carried by vision tokens.
>
> ==FastV 发现：视觉 tokens 的 attention score 远低于文本 tokens（信息极度稀疏）==

> Based on this observation, FastV prunes half of the vision tokens at the **second layer** of the LLM using attention from the last textual token.
>
> ==FastV 策略：在 LLM 第2层，用最后一个文本 token 的 attention 剪掉 50% 视觉 tokens==

> PyramidDrop identified that redundancy of vision tokens tends to **increase with LLM depth**. It introduced a **multi-stage progressive pruning** strategy.
>
> ==PyramidDrop 发现：冗余随深度增加；策略：多阶段渐进剪枝==

**⚠️ Attention Bias Problem:**

> Feather noted that vision tokens located near output tokens tend to receive **disproportionately high attention scores** in shallow layers. This is attributed to the **long-term decay property of RoPE**.
>
> ==Attention Bias：靠近输出的视觉 tokens 在浅层获得不成比例的高 attention（RoPE 衰减特性）==

解决方案：
- **Feather**: 计算重要性时不应用 RoPE
- **AdaTP**: 用独立 text encoder 计算余弦相似度
- **VScan**: 从中间层而非浅层开始剪枝

**⚠️ Flash Attention Compatibility:**

> A technical challenge arises when integrating attention-based pruning with Flash Attention, which **does not directly expose attention scores**.
>
> ==Flash Attention 不暴露 attention scores，与基于 attention 的剪枝不兼容==

解决方案：
- **TopV**: 用特征相似度 + 空间距离替代 attention scores
- **PACT**: 用 hidden state norms + global query vector

#### Learnable Module-based

> Learnable module-based approaches introduce **trainable components** that learn to assess token importance or determine the appropriate compression ratio.
>
> ==可学习模块：训练一个模块来评估 token 重要性或压缩率==

代表方法：
- **p-MoD**: weight predictor 预测每个 token 重要性
- **DyRate**: 轻量级分类器预测最优剪枝比例
- **ATP-LLaVA**: MLP 双头预测自适应阈值

#### Token Merging-based

> Token merging offers a **softer compression** strategy. It computes similarity measures and applies grouping/clustering to fuse multiple tokens into fewer representatives.
>
> ==Merging 是更软的压缩策略：保留信息而非直接丢弃==

代表方法：
- **LLaVolta**: 简单平均池化 + 多阶段训练降低压缩率
- **FiCoCo**: 先选重要 tokens，再计算与剩余 tokens 的相关矩阵引导合并
- **FrameFusion**: 计算与前帧对应位置的相似度，合并时序冗余

#### Fusion-based

> Fusion-based approaches implement compression **indirectly** by leveraging cross-attention or self-attention to integrate visual information into other tokens.
>
> ==Fusion：用 cross/self-attention 间接压缩，避免过长序列==

代表方法：
- **Flamingo**: GATED XATTN-DENSE 层，text 作 Q，visual 作 K/V
- **mPLUG-Owl3**: intra-text self-attention + cross-modal attention
- **VoCo-LLaMA**: 单个 Vision Compression token 吸收所有视觉信息

### 3.3.2 Compression in Decoding Stage (KV Cache)

> KV-cache compression aims to reduce the memory and computational overhead of cached key and value tensors during autoregressive decoding.
>
> ==KV Cache 压缩：减少解码阶段缓存的 K/V 内存开销==

代表方法：
- **DyCoke**: 基于 text-vision attention 动态压缩，只保留高 attention 的 KV pairs
- **Video-XL-2**: Bi-level KVs decoding，动态选择 dense 或 sparse KV
- **LiveVLM**: 先按 attention 丢弃不重要 KV，再合并每帧 KV
- **StreamMem**: 固定大小 KV memory + attention-based 压缩

---

## 3.4 Token Compression in Multi-Module (Hybrid)

> An increasing number of approaches explore compression strategies **across multiple modules** to achieve higher compression efficiency.
>
> ==Hybrid：跨多个模块协同压缩，效率更高==

### 3.4.1 Collaborative Compression

> Collaborative compression involves jointly optimizing compression decisions across multiple modules.
>
> ==协作压缩：多模块联合优化压缩决策==

代表方法：
- **CrossGET**: Vision Encoder + Projector 联合
- **LLaMA-VID**: Vision Encoder + Projector
- **PAR**: Vision Encoder + Projector + LLM

### 3.4.2 Progressive Compression

> Progressive compression applies token reduction incrementally through multiple stages, each contributing to the overall compression.
>
> ==渐进压缩：分多阶段逐步压缩==

代表方法：
- **MustDrop**: 多阶段 pipeline + uncertainty gating 恢复
- **DyCoke**: 动态渐进压缩
- **FiCoCo**: 先选后合的两阶段
- **METEOR**: Vision Encoder + Projector + LLM 全链路渐进压缩

---

## 总结表

| 位置 | 优点 | 缺点 | 代表方法 |
|------|------|------|---------|
| **Vision Encoder** | 收益最大（早期压缩） | 可能影响视觉语义编码 | ToMe, VisionZip, DART |
| **Projector** | 自然压缩点，训练时常用 | 设计空间受限 | Q-Former, TokenPacker, Pooling |
| **LLM** | 可利用跨模态信息 | 前向传播后才能压缩 | FastV, PyramidDrop, DyCoke |
| **Hybrid** | 效率最高 | 设计复杂 | METEOR, MustDrop |
