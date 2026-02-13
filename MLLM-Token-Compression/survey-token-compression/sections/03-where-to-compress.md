# 3. Where to Compress Tokens in MLLMs

> 本章是survey的核心章节，按压缩在MLLM架构中的**位置**系统分类。这一分类维度直接对应了工程实现中的第一个设计决策——在pipeline的哪个环节"动刀"，决定了可获取的信息类型、压缩的副作用传播范围、以及与现有模型的兼容性。

## 总览分类图 (Figure 2)

```
Where to Compress
├── 3.1 Vision Encoder
│   ├── 3.1.1 Inside-Encoder
│   │   ├── A. Visual Token Dropping
│   │   ├── B. Visual Token Merging
│   │   └── C. Multi-Scale Compression
│   └── 3.1.2 Outside-Encoder
│       ├── A. Purely-Vision Compression
│       ├── B. Text-guided Compression
│       └── C. Token Recovery Mechanisms
├── 3.2 Projector
│   ├── 3.2.1 Transformation-Based (Pooling / Pixel Shuffle / Convolution)
│   ├── 3.2.2 Query-Based (Q-Former / Variants / Cross-Attention)
│   └── 3.2.3 Importance-Driven (Similarity / Saliency / Innovative Metrics)
├── 3.3 LLM
│   ├── 3.3.1 Prefilling Stage (Importance / Learnable / Merging / Fusion)
│   └── 3.3.2 Decoding Stage (KV-cache Compression)
└── 3.4 Hybrid (Multi-Module)
    ├── 3.4.1 Collaborative Compression
    └── 3.4.2 Progressive Compression
```

---

## 3.1 Token Compression in Vision Encoder

> 视觉编码器是处理视觉输入的第一个模块，是MLLM pipeline中视觉token的"源头"。在此阶段进行压缩意味着**全链路的效率收益**——被压缩掉的token不会进入后续的Projector和LLM，因此在encoder处减少N个token，其计算节省会在整条pipeline上累积放大。如Figure 3所示，vision encoder阶段的压缩可以进一步分为Inside-Encoder（在ViT内部修改token流）和Outside-Encoder（在encoder输出之后、projector之前进行压缩）两类。

### 3.1.1 Inside-Encoder Compression

在ViT编码器**内部**直接修改token流，在自注意力计算过程中减少token数量，从而降低encoder本身的计算开销。设计围绕两个核心问题展开：(1) 如何识别和处理"不重要"的token——通过pruning丢弃还是通过merging合并？(2) 如何跨多层/多编码器/多分辨率协调压缩，以利用层级化的视觉语义？

#### A. Visual Token Dropping

**核心思路**: 计算每个token的重要性分数 --> 排名 --> 保留Top-K --> 直接丢弃其余token。实现上遵循"ranking + Top-K"范式，配合预定义阈值。

**三种评分策略**:

| 策略 | 核心思想 | 代表工作 | 关键细节 |
|------|---------|---------|---------|
| **Similarity-based** | 衡量token与全局表示(CLS token/聚合特征向量)的相似度，高相似度=冗余，应被移除 | TRIM, SAINT | TRIM利用CLIP嵌入衡量文本-视觉相关性，配合自适应IQR(四分位距)阈值实现层自适应筛选；SAINT采用基于图的方法，在图结构中联合优化剪枝率和冗余阈值，比固定策略更灵活 |
| **Attention-based** | 利用ViT内部的注意力权重判断token的视觉显著性 | VisPruner, HiPrune, VFlowOpt, MADTP, SmartTrim | VisPruner和HiPrune利用CLS token attention评估image partition的重要性；VFlowOpt构建结合visual attention-derived context relevance和patch-level information entropy的importance map；MADTP引入Token Importance Score (TIS)整合class attention、self-attention和cross-modal alignment attention三种机制，用sparsemax + learnable thresholds动态生成pruning masks；SmartTrim将CLS token送入轻量policy network学习跨模态信息指导的重要性分数 |
| **Heuristic-based** | 利用任务/领域先验知识构造评分规则 | EgoPrune, METEOR | EgoPrune针对自中心视频，利用geometric stability和field-of-view dynamics优先保留运动相关区域、剪除静态背景；METEOR采用层自适应策略——浅层用token-to-average similarity（低层冗余以纹理重复为主），深层用CLS-to-token attention（高层语义信息更集中） |

> **[感受]** Token Dropping是最直观的压缩范式，但其核心局限在于**信息不可逆丢失**——一旦token被丢弃，其携带的视觉信息在后续所有层中都不可恢复。这使得dropping方法对重要性评分的准确性极度敏感：评分偏差会直接导致关键视觉区域被误删。三种评分策略中，similarity-based方法假设"与全局表示相似=冗余"，这在背景均匀的场景中成立，但对于复杂场景（如多个不同前景物体），全局表示本身就不准确，导致评分失效。Attention-based方法虽然最为流行，但后续在LLM阶段的研究（如Feather）已经揭示了attention score存在系统性偏差（position bias、foreground bias等），这一问题在encoder内部同样存在但尚未被充分讨论。Heuristic-based方法（如METEOR的层自适应策略）提供了一个重要启示：不同深度的ViT层捕获的信息类型不同，压缩策略也应相应调整——这一思想值得在MLLM效率优化中进一步泛化。对于MLLM效率优化研究者而言，一个值得探索的方向是：能否将dropping与后续的recovery机制结合，在encoder内部dropping后在projector或LLM处进行条件恢复？

#### B. Visual Token Merging

**核心思路**: 与直接丢弃不同，merging将语义相似的token聚合为代表性token，在缩短序列长度的同时**保留被压缩token的信息**。这是一种"软压缩"策略，信息损失通常小于dropping。

| 合并策略 | 核心思想 | 代表工作 | 关键细节 |
|---------|---------|---------|---------|
| **Proximity-based** | 利用空间/时序邻近性作为归纳偏置——相邻token往往高度冗余 | downsampling, pixel-shuffle+channel merging, 3D convolution, adaptive convolution kernels, density-based clustering | 空间合并通过downsampling或pixel-shuffle实现确定性聚合；时序合并通过joint temporal-spatial aggregation同时合并相似帧和patch，或通过frame-level fusion自适应地对连续帧进行加权融合（learnable importance weighting）。利用了相邻token在空间和时序维度上高相关性的归纳偏置，实现高效压缩同时保留局部连贯性 |
| **Similarity-based** | 超越空间邻近性，基于显式的语义相似度度量来聚合token | patch-to-class correlation, density-based clustering into abstracted representations | 全局相似度方法通过patch-to-class correlation计算token重要性，或将语义相似但空间上远离的patch聚类为抽象表示，实现跨空间位置的语义合并 |
| **Cross-modal** | 利用文本上下文指导合并决策，使压缩对齐任务需求 | bidirectional language-aware signals, pipelines combining semantic and spatial similarity | 通过跨模态双向信号交换语言感知信息，或通过结合语义关系和空间相似度的pipeline，使压缩能够适应内容含义而非仅仅依赖token位置 |
| **Hybrid** | 组合多种压缩技术以获得更好的效率-质量权衡 | sequential pruning→weighted merging, learnable abstraction methods | 先用attention-based pruning移除粗粒度冗余，再用weighted merging从被丢弃token中恢复信息并整合到保留token中；可学习的抽象方法使用少量可训练压缩token配合cross-attention lookup高分辨率token，实现灵活压缩比而无需修改架构 |

> **[感受]** Merging相比Dropping的核心优势在于信息保留——通过加权平均或聚类中心来聚合被压缩token的信息，理论上减少了信息的不可逆丢失。但这也引入了新的问题：**合并操作本身的信息损失如何量化？**平均合并会模糊边界细节，加权合并的权重如何确定？Proximity-based方法依赖"空间邻近=语义相似"的归纳偏置，这在自然图像中通常成立，但在文档、图表等结构化视觉输入中可能失效（相邻patch可能属于完全不同的语义单元）。Cross-modal merging是一个特别值得关注的方向——它将任务语义引入encoder内部的压缩决策，但也带来了encoder与LLM之间的耦合，削弱了模块化设计的灵活性。从MLLM效率优化的角度看，Hybrid策略（先prune再merge恢复）是一个务实的设计思路，它承认了单一策略的局限性。未来的研究可以探索更精细的merge操作（如attention-weighted merge而非简单均值），或者在merge时显式保留被合并token的差异信息（类似残差连接的思想），以在高压缩率下维持更好的信息保真度。

#### C. Multi-Scale Compression

> 单尺度方法在固定粒度下运作，难以同时兼顾细节保留和全局理解。多尺度方法通过跨层/跨编码器/跨分辨率的协调来利用层级化的视觉语义，实现更全面的视觉细节捕获。

| 类型 | 核心思想 | 代表工作 | 关键细节 |
|------|---------|---------|---------|
| **Multi-Layer** | 从ViT的不同层提取并融合特征，利用浅层的低级细节和深层的高级语义 | LLaVA-STF, METEOR, Chat-UniVi, LaCo | LLaVA-STF从多个ViT block提取token，通过channel concatenation和convolution融合空间与语义信息；METEOR采用层级pruning——浅层用token-to-average similarity，深层用CLS-to-token attention；Chat-UniVi采用三级cascade aggregation，逐步提取粗/中/细粒度token集合形成统一的多尺度表示；LaCo在早期层进行激进压缩后通过pixel shuffle和MLP实现细节恢复 |
| **Multi-Encoder** | 组合不同架构或训练范式的编码器，获取互补的视觉表示 | Cambrian-1 (DINOv2 + CLIP), METEOR | Cambrian-1证明了将自监督模型(DINOv2)与语言监督模型(CLIP)结合能持续提升vision-centric和OCR任务的性能，凸显了多样化视觉表示的价值；METEOR提出系统化的多编码器框架，消除跨编码器冗余以最大化互补性同时最小化计算开销 |
| **Multi-Resolution** | 高分辨率输入捕获细粒度细节，低分辨率输入提供全局上下文，双路径协调实现效率与质量的平衡 | FastVLM, ADMIRE, LinVT, M3, VideoChat-Flash (HiCo) | FastVLM通过新型混合视觉编码器FastViTHD实现最优的token-resolution平衡；ADMIRE采用dual-path Multi-Resolution Adaptation——低分辨率主干负责全局处理，高分辨率旁路负责细节注入，在文档理解和小目标检测上表现优异；LinVT和M3对视频理解应用多尺度时序pooling，捕获短期动态和长程上下文；VideoChat-Flash引入Hierarchical Condensation (HiCo)，从clip级到segment级渐进地精炼视频语义 |

> **[感受]** 多尺度压缩是一个设计理念层面的重要进步——它承认了"没有单一粒度能适配所有视觉内容"这一现实。Multi-Layer方法本质上是在利用ViT不同深度的"信息层级"（浅层偏纹理、深层偏语义），这与神经科学中视觉皮层的层级处理模型相呼应。Multi-Encoder方法（如Cambrian-1）虽然提升了表示质量，但也带来了计算开销的增加，其效率收益需要与额外encoder的成本做权衡——如何在保持互补性的同时最小化冗余，仍是一个未充分解决的问题。Multi-Resolution方法与实际应用需求最为吻合（用户可能既需要看清文档中的小字，也需要理解整体布局），但dual-path设计引入了如何动态分配计算预算的新问题。从MLLM效率优化的视角看，多尺度方法的一个被忽略的机会是**内容自适应的尺度选择**——对于简单场景可能只需要低分辨率粗粒度表示，而复杂场景才需要多尺度融合，这种动态的尺度路由机制可以进一步提升效率。另外，多尺度方法与token压缩的结合仍处于初期——如何在多尺度表示上做高效的压缩而非简单拼接，是一个有潜力的方向。

---

### 3.1.2 Outside-Encoder Compression

在Vision Encoder输出之后、Projector之前进行压缩。此时视觉token已完整编码但尚未与语言模态对齐。

**核心优势**: **Plug-and-play**——不需要修改编码器内部结构，可以作为独立模块插入任何MLLM架构，最大化了与现有模型的兼容性和部署灵活性。压缩方法可通过衡量vision-vision或vision-text语义相关性来减少token数。

#### A. Purely-Vision Compression

仅基于视觉信号自身的语义相关性（vision-vision semantic relevance）进行压缩，不依赖文本query。这使得压缩结果可跨不同query复用，适合multi-turn对话和batch推理场景。

**Selection-then-Merge范式**: 先选择重要token，再对剩余token进行合并或聚合。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **VisionZip** | 重要性估计 + 代表性约束 | 识别可复用的代表性token，在重要性和覆盖度之间做权衡 |
| **Fourier-VLM** | 频域低通滤波 | 在频域中抑制高频冗余成分，映射回token空间后实现压缩 |
| **LLaVA-STF** | 跨层拼接 + Multi-Block Token Fusion (MBTF) | 通过cross-layer concatenation生成紧凑的视觉摘要 |

**Visual Attention Bias问题**: 早期方法（LLaVA-PruMerge, VTC-CLS, FasterVLM, FoPru, freePruner）利用CLS token和self-attention进行稀疏化，通过self-attention scores选择高贡献token。但研究发现，attention-based selection存在系统性的**前景偏向(foreground bias)**——倾向于选择前景物体token而忽略全局上下文（如背景、空间关系）。**HoloV**通过整合全局视觉上下文来平衡前景和背景token，从holistic perspective解决这一偏差。

**极端压缩 (Extreme Compression)**: 将每帧/每段视频压缩到极少量token:
- **LLaMA-VID**: 每帧压缩为1个Content Token，实现固定预算压缩，适合长视频
- **VideoLLaMA 2**: 通过Spatial-Temporal Convolution (STC)整合帧级patch，使用separable convolution和local aggregation
- **Flash-VStream**: K-means聚类低分辨率特征为Context Synopsis Memory，保留全局时序信息
- **LLaVA-PruMerge**: 最近邻聚类的可学习token合并，通过nearest-neighbor clustering实现灵活压缩

这些方法共享一个核心原则：增强每个保留token的信息密度，在不依赖文本的情况下实现高效压缩，在多图像和多轮对话场景中尤其有优势。

> **[感受]** Purely-Vision压缩的最大优势在于**文本无关性**——压缩结果可被不同query复用，这在multi-turn对话和batch推理中具有明显的工程优势。但这也恰恰是其局限：不利用文本信息意味着无法做query-adaptive的压缩，可能在高压缩率下丢失与特定query高度相关但视觉上不显著的信息。Visual Attention Bias问题是一个深刻的发现——它揭示了ViT的attention分布并不等同于"对下游任务的重要性分布"，这一偏差在MLLM场景下被放大（因为任务多样性远高于纯视觉任务）。HoloV的holistic approach是一个正确的方向，但如何定义"全局视觉上下文"仍缺乏理论指导。极端压缩方法（如LLaMA-VID的每帧1 token）虽然在效率上极为激进，但信息瓶颈也最为严重——对于需要细粒度视觉推理的任务（如counting、spatial reasoning），这种压缩率几乎必然导致显著性能下降。从MLLM效率优化的角度看，一个有价值的研究方向是：设计一种自适应机制，根据输入内容的视觉复杂度动态决定压缩率——简单场景用极端压缩，复杂场景用温和压缩。

#### B. Text-guided Compression

利用文本语义先验（用户query、指令）指导压缩决策，使压缩过程能够聚焦于与当前任务相关的视觉区域，实现context-oriented的效率优化。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **PAR** | 解析query为实体和动作 --> 重新加权视觉token | 将用户query parsing为结构化的entity-action对，以此指导视觉token的重要性重分配 |
| **QG-VTC** | 计算question-to-vision相似度 --> 引导token保留 | 在vision-text相似度空间中评估每个视觉token与query的相关性，实现4x-8x压缩率且性能损失极小 |
| **LongVU** | 跨模态查询与帧/区域候选结合 --> 先帧级过滤再token级选择 | 整合cross-modal queries和frame/region candidates，实现两级过滤：先在帧级移除无关帧，再在token级精选相关区域 |
| **AdaFV** | 自适应cross-modality attention mixture | 基于visual saliency和text-image similarity动态选择visual token |
| **VCM** | Vision Concept Modeling | 动态确定所需视觉概念的数量和空间位置，通过multi-head cross-attention层进行语义对齐，基于selected keywords的数量和相关性估计最优保留token数 |

**Text-guided在Outside-VE位置的独特优势**: 此时视觉token已完整编码（保留了全部视觉信息），但跨模态交互尚未在LLM中开始。这意味着text-guided压缩可以利用文本信号做精准筛选，同时**避免文本偏差干扰低级视觉编码过程**。

**实践范式**: 先purely-vision做文本无关的粗粒度压缩（去除明显冗余） --> 再text-guided做query相关的精细化筛选。这种级联设计兼顾了泛化性和任务特异性。

> **[感受]** Text-guided压缩从理论上最为合理——既然MLLM的最终目标是回答用户query，那么压缩就应该保留与query最相关的视觉信息。但这也引入了一个根本性的"鸡生蛋"问题：在encoder输出阶段，模型尚未真正"理解"用户query的深层意图（特别是需要多步推理的复杂问题），此时基于浅层文本-视觉相似度的筛选可能并不准确。例如，对于"这张图片中有什么不寻常的地方？"这类open-ended query，几乎所有视觉区域都可能是相关的，text-guided压缩的优势将大打折扣。QG-VTC的4x-8x压缩率令人印象深刻，但需要关注其在不同query类型上的性能方差——对于描述性query（"描述这张图片"）和定位性query（"左上角是什么"），最优压缩策略可能截然不同。LongVU的两级过滤（帧级+token级）是一个优雅的设计——它将压缩分解为粗粒度和细粒度两个阶段，每个阶段的决策空间都更小，降低了错误累积的风险。从MLLM效率优化的角度看，text-guided压缩的一个关键挑战是如何在不显著增加计算开销的前提下引入文本信号——如果text-guided module本身的计算成本接近于直接处理所有视觉token，那么压缩的净收益就会大打折扣。

#### C. Token Recovery Mechanisms

在高压缩率下，过度压缩不可避免地导致关键信息丢失。Token Recovery Mechanisms通过动态恢复机制增强压缩的鲁棒性，实现closed-loop的压缩-恢复流程。

| 方法 | 恢复策略 | 关键细节 |
|------|---------|---------|
| **RecoverableCompression** | 基于置信度和冲突阈值触发重采样 | 当MLLM检测到语义不确定性或entropy时，触发对视觉信息的重新采样，重新注入被压缩的token以补偿丢失的视觉证据 |
| **MustDrop** | 通过不确定性门控进行多阶段恢复 | 在multi-stage pipeline中整合recovery，通过uncertainty gating在每个阶段平衡激进压缩和稳定性 |
| **ToCom** | 处理训练-测试压缩率不匹配问题 | 作为plug-and-play layer弥合不同压缩率之间的性能gap，无需重新训练即可跨压缩率工作 |
| **VTC / Video-XL-Pro** | 通过视觉重建监督优化压缩 | VTC利用Stable Diffusion decoder从压缩token重建图像，以重建误差为监督信号优化压缩策略，确保保留的token能忠实还原原始视觉信息 |

> **[感受]** Token Recovery是压缩研究中一个被严重低估的方向。大多数压缩方法将信息丢失视为不可避免的代价，而recovery机制提供了一种"后悔药"——在发现压缩过度时可以动态补救。这一思想与人类视觉处理中的"注意力回溯"机制有异曲同工之妙（当我们意识到遗漏了重要细节时，会重新注视相关区域）。RecoverableCompression的confidence-based触发机制特别有启发性——它将压缩从一个open-loop过程变为closed-loop过程，压缩质量可以通过模型自身的不确定性来监控。ToCom解决的train-test compression rate mismatch是一个极为实际的工程问题——在部署时往往需要根据硬件资源动态调整压缩率，而大多数方法需要为每个压缩率单独训练/微调。VTC使用重建损失作为监督信号是一个巧妙的设计，但它假设"能重建=信息保留充分"，这对于MLLM的下游任务来说可能过于保守（重建需要保留所有像素信息，而VQA只需要保留任务相关信息）。从MLLM效率优化的角度看，recovery机制的计算开销是关键——如果recovery过于频繁或成本过高，可能会抵消压缩的效率增益。理想的recovery应该是轻量级的、有条件触发的，且恢复的信息应该是有针对性的而非全量重采样。

---

## 3.2 Token Compression in Projector

> Projector是视觉编码器与语言模型之间的**桥梁模块**，负责将raw visual embeddings转换为language-compatible representations。它天然位于模态转换的交界处，既有机会利用视觉特征的结构（如空间布局），又可以引入语言先验进行任务导向的压缩。如Figure 4所示，projector阶段的压缩方法可分为transformation-based（确定性变换）、query-based（可学习查询）和importance-driven（重要性驱动）三类。

### 3.2.1 Transformation-Based Compression

通过**确定性变换**直接改变feature map的空间结构来减少token数量。这类方法不依赖可学习的query或复杂的注意力机制，计算开销极低，设计简洁。

#### Pooling

Pooling是计算机视觉中最广泛使用的下采样操作。给定输入特征图 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$，pooling窗口大小为 $k \times k$，输出特征图 $\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C}$ 中每个位置的值为：

$$Y_{i,j,c} = \frac{1}{|\Omega_{i,j}|} \sum_{(u,v) \in \Omega_{i,j}} X_{u,v,c}$$

其中 $\Omega_{i,j}$ 是以 $(i,j)$ 为中心的 $k \times k$ 空间邻域。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **MobileVLM V2** | Lightweight Downsample Projector (LDP) | 采用2x2 average pooling配合pointwise和depthwise convolution，针对移动端部署优化 |
| **DeCo** | 自适应average pooling的有效性验证 | 通过extensive实验分析验证了adaptive average pooling的有效性，展示了其在收敛稳定性和特征提取方面的优势 |
| **AVG-LLaVA** | Visual Granularity Scaler + Visual Granularity Router | 通过stacking average pooling layers构建multi-granularity visual features，再由Visual Granularity Router为每个输入选择最合适的粒度级别 |
| **TC-LLaVA** | 简单global average pooling (视频) | 对视频帧直接应用全局average pooling减少每帧token数，以极简方式实现大幅压缩 |
| **PLLaVA** | 空间+时序自适应average pooling | 同时在空间和时序维度应用adaptive average pooling，实现dual-dimension压缩 |

> **[感受]** Pooling的最大优势在于**零额外参数、零训练成本、实现极简**——一个2x2 average pooling就能将token数减少4倍，几乎没有工程部署障碍。但其局限也同样明显：pooling对所有空间区域施加相同的压缩率，无法区分信息密集区域（如文本区域、小目标）和信息稀疏区域（如天空、墙壁）。AVG-LLaVA的granularity router是对这一局限的一次有意义的尝试——它通过学习为不同输入选择不同粒度，实现了instance-level的自适应，但仍是粗粒度的（整张图片用同一粒度，而非不同区域用不同粒度）。PLLaVA将pooling扩展到时序维度是视频场景下的自然选择，但temporal pooling的一个潜在问题是它可能模糊时序边界（如动作切换点）。从MLLM效率优化的角度看，pooling更适合作为压缩pipeline中的第一级粗粒度降维（fast and cheap），后续再用更精细的方法做selective refinement。单独依赖pooling作为唯一压缩手段，在高分辨率或细粒度任务上可能不够。

#### Pixel Shuffle

Pixel Shuffle通过channel dimensionality来交换空间分辨率——将高分辨率空间token重新排列为更少但通道更深的token。给定输入 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ 和下采样因子 $r$：

$$\mathbf{Y} = \text{reshape}(\mathbf{X}, H/r, W/r, C \cdot r^2)$$

其中 $\mathbf{Y} \in \mathbb{R}^{H/r \times W/r \times (C \cdot r^2)}$。

- 空间token减少 $r^2$ 倍，通道维度相应增加 $r^2$ 倍
- 后接MLP将expanded channel dimension对齐到LLM embedding dimension
- 本质上是"用通道深度换取空间分辨率"——每个输出token包含了原始 $r^2$ 个token的信息
- 代表工作: **InternVL 1.5**, **NVLM**, **InternVL 2.5**, **Qwen2-VL**

> **[感受]** Pixel Shuffle在当前主流MLLM中的采用率极高（InternVL系列、Qwen2-VL等），这本身就说明了其工程实用性。与pooling的平均操作不同，pixel shuffle是**无损的空间-通道变换**——信息并未被丢弃，只是被重新组织到了通道维度。但后续的MLP投影（将expanded channel压缩到LLM embedding维度）实际上才是信息瓶颈所在——这一步的信息损失往往被忽视。一个有趣的观察是：pixel shuffle的下采样因子 $r$ 通常是固定的（如 $r=2$，4倍压缩），缺乏内容自适应性。此外，pixel shuffle隐含假设了空间上相邻的 $r \times r$ patch包含互补信息，这在大多数自然图像中成立，但对于具有强空间规律性的输入（如棋盘格图案、重复纹理）可能导致通道维度中的高冗余。从MLLM效率优化的研究视角看，pixel shuffle已经成为一个strong baseline——新提出的projector压缩方法需要与之做公平对比，证明额外的设计复杂度带来了实质性的性能提升。

#### Convolution

相比pooling和pixel shuffle的确定性操作，卷积通过**可学习权重**选择性整合局部信息，能够捕获任务相关的局部模式。

标准2D卷积将输入特征图 $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$ 映射为 $\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}$：

$$Y_{i,j}^{(o)} = \sum_{c=1}^{C_{in}} \sum_{m=1}^{k_h} \sum_{n=1}^{k_w} \mathbf{W}_{m,n,c}^{(o)} \cdot \mathbf{X}_{i+m-1,j+n-1}^{(c)} + b^{(o)}$$

其中 $\mathbf{W} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}$ 是可学习的卷积核。

- 通过stride > 1或后接pooling实现下采样
- 可学习权重使其能适应任务需求，选择性保留重要的局部信息
- 常与pooling组合使用:
  - **Honeybee的C-Abstractor**: ResNet blocks + average pooling，保留局部结构
  - **MobileVLM V2的LDP**: pointwise convolution + depthwise convolution + average pooling，轻量高效
- 通过stacking convolutional layers或使用variable kernel sizes，可以捕获多尺度抽象特征

> **[感受]** 卷积在projector压缩中的角色本质上是一个"可学习的局部聚合器"——相比pooling的uniform averaging，卷积可以学习到"哪些局部信息更重要"。但这种优势的代价是需要额外训练，且引入了新的参数。Honeybee的C-Abstractor设计特别有洞察力——它意识到标准projector（如线性层或Q-Former）丢失了视觉特征的局部空间结构，而ResNet blocks可以保留这种结构。然而，卷积的固有局限在于其**感受野有限**——即使通过stacking增大感受野，卷积本质上仍是局部操作，对于需要全局上下文的压缩决策（如"这个区域在整张图中是否重要"）可能不如attention-based方法。从MLLM效率优化的角度看，卷积最适合作为transformation-based压缩中的一环，与pooling或pixel shuffle搭配使用，而非独立承担压缩任务。另外，depthwise separable convolution（如MobileVLM V2使用的）是一个值得借鉴的轻量化设计——它在保持可学习性的同时大幅减少了参数量和计算量，特别适合端侧部署场景。

---

### 3.2.2 Query-Based Compression

利用少量**可学习query向量**通过cross-attention机制聚合视觉信息，实现从大量视觉token到固定数量（或可变数量）query token的压缩。这类方法提供了一种灵活的、parameter-efficient的替代方案——query数量直接控制压缩率，且query可以学习选择性地关注任务相关信息。

#### Q-Former (BLIP-2)

Q-Former是query-based压缩的奠基性架构，由BLIP-2首次提出。

**核心机制**:
- **Q (Query)**: 一组可训练query嵌入（数量固定且少量，如32个），初始化为可学习参数
- **K/V (Key/Value)**: 来自frozen vision encoder的patch嵌入（数百个）
- 通过stacked self-attention（query间交互）+ cross-attention（query与视觉token交互）层，queries学习选择性聚合最相关的视觉信息
- 输出: 紧凑的query嵌入集合 --> 线性投影到LLM嵌入空间，作为visual tokens输入LLM
- 本质上将"数百个视觉token --> 固定数量query token"的压缩编码为了一个跨模态注意力学习问题

**后续工作**: MiniGPT-4, InstructBLIP 等在Q-Former基础上进一步引入instruction tuning和task-specific fine-tuning。

> **[感受]** Q-Former是MLLM token压缩领域最具开创性的架构之一——它将"压缩"重新定义为"信息蒸馏"，通过cross-attention让少量query主动地从大量视觉token中"提问并提取"关键信息。这种主动查询的范式比被动的pooling/dropping在理论上更为优雅。然而Q-Former也有显著局限：(1) **query数量固定**——32个query不论输入是简单还是复杂都使用相同容量，存在信息瓶颈或冗余浪费的问题；(2) **训练成本高**——Q-Former需要额外的预训练阶段（如BLIP-2的三阶段训练），这与training-free压缩方法相比是明显劣势；(3) **空间信息丢失**——cross-attention聚合后，query token之间不再保持原始视觉token的空间排列关系，这对需要空间推理的任务不利。从MLLM效率优化的视角看，Q-Former代表了一种"重参数化压缩"的思路——用额外参数和训练换取更好的压缩质量。在当前趋势下（追求training-free、plug-and-play），Q-Former的heavy设计可能不再是最优选择，但其core idea（用可学习query做跨模态信息蒸馏）仍然深刻且被广泛借鉴。

#### Q-Former变体

后续工作对Q-Former进行了简化或增强，以解决其局限性:

| 变体 | 改进方向 | 关键细节 |
|------|---------|---------|
| **Qwen-VL** | 简化：单层cross-attention | 将Q-Former的多层结构简化为单层cross-attention module，大幅降低架构复杂度和计算开销，同时保留了聚合视觉信息和token压缩的能力 |
| **Honeybee** | 增强局部性：C-Abstractor + D-Abstractor | C-Abstractor通过ResNet blocks + average pooling保留局部空间结构；D-Abstractor利用Deformable Attention（reference points + sampling offsets）增强局部性的同时保持输出token数的灵活性 |
| **MQT** | 可变query数量 | 训练时随机采样 $m$ 个query（$m < M$），使模型学会在不同粒度下工作。推理时可根据需求调整query数量——平均可将query数减半且性能损失极小 |
| **TG-LLaVA** | 文本引导的query | 引入可学习latent嵌入编码全局文本语义，通过single-layer Q-Former整合文本和视觉信息，生成text-driven mask应用于视觉特征，在text guidance下精炼压缩 |
| **LLaVA-Mini** | 预融合模态信息 | 增加Modality-Pre Fusion模块，在压缩前先将视觉表示与instruction tokens融合，缓解standalone Q-Former可能导致的视觉信息丢失 |

> **[感受]** Q-Former变体的演化路径清晰地反映了该领域的几个关键洞察。Qwen-VL的单层简化表明，Q-Former的大部分信息提取能力可以在单次cross-attention中完成，多层堆叠的边际收益递减——这对追求效率的MLLM设计具有重要指导意义。Honeybee的D-Abstractor引入Deformable Attention是一个精妙的设计：它在保持query-based框架的同时解决了空间局部性丢失的问题，但也增加了实现复杂度。MQT的可变query数量机制非常值得关注——它实现了**推理时的动态压缩率调整**，无需重新训练，这对于需要在不同硬件条件下部署的MLLM极具实用价值。TG-LLaVA和LLaVA-Mini都试图在压缩阶段引入更多上下文信息（文本语义或多模态融合），其核心动机是减少"盲目压缩"的信息损失。从效率优化的角度看，这些变体揭示了一个设计tension：更精细的压缩需要更多的上下文信息，但获取和处理这些上下文本身也需要计算资源——如何在"压缩质量"和"压缩成本"之间找到sweet spot，是这个方向的核心挑战。

#### Cross-Attention-Based

不完全依赖Q-Former框架，而是直接利用cross-attention机制来识别或提取任务相关的视觉token。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **CATP** | 基于cross-attention概率投票 --> 按聚合重要性剪枝 | 在query tokens和image tokens之间计算cross-attention概率，跨多层和多head累积投票分数，按聚合重要性进行剪枝。考虑了层间差异，提出composite ranking方法 |
| **TokenPacker** | 低分辨率特征作为point-based queries --> Point-to-Region cross-attention | 采用coarse-to-fine视觉信息提取策略：先下采样获得低分辨率表示作为queries，再通过Point-to-Region cross-attention迭代注入高分辨率信息，逐步丰富每个query的语义 |
| **HiRes-LLaVA** | 下采样特征作为queries --> 与原始特征cross-attention | 放弃可学习query，直接将downsampled visual features作为queries，通过与原始高分辨率features的cross-attention生成紧凑但信息丰富的压缩序列 |
| **mPLUG-DocOwl2** | 全局特征作为queries --> cross-attention聚合文本语义 | 针对高分辨率文档理解，使用全局视觉特征作为queries，cropped image features作为keys/values，通过cross-attention聚合文本语义，大幅减少文档图像的visual token数 |
| **QueCC** | 将user query注入视觉表示 --> 携带任务语义的cross-attention | 将用户query的textual features注入visual representations中，使后续cross-attention（downsampled tokens与visual token regions之间）在压缩时能维持与textual task的强关联性 |

> **[感受]** 这组方法相比Q-Former有一个共同的进步：它们不再依赖"从零初始化的可学习query"，而是利用已有的视觉或文本特征作为query的初始化来源——这提供了更好的初始对齐，减少了训练难度。TokenPacker的coarse-to-fine策略特别值得学习：它巧妙地将压缩分解为两步（先粗粒度降维，再精细补充），避免了one-shot压缩的信息损失。HiRes-LLaVA直接用downsampled features作为query的做法非常简洁——它本质上是在问："低分辨率版本能从高分辨率版本中补充哪些细节？"mPLUG-DocOwl2针对文档理解的设计提醒我们，不同视觉域（自然图像vs文档）的最优压缩策略可能大不相同——文档图像的信息分布远比自然图像更均匀，简单的saliency-based压缩可能丢失关键文字信息。从MLLM效率优化的角度看，cross-attention-based方法的一个潜在问题是：cross-attention本身的计算复杂度为 $O(n_{query} \times n_{kv})$，当原始token数很大时（如高分辨率图像），这一步的开销可能不可忽略。如何在保持压缩质量的同时降低压缩操作本身的计算成本，是一个值得关注的方向。

---

### 3.2.3 Importance-Driven Compression

通过显式估计每个token的重要性来选择性保留/合并，不依赖固定长度的query或确定性的空间变换。这类方法的核心假设是：token的重要性可以通过某种可计算的度量来评估，且重要性分布是非均匀的。

#### Various Similarity Metrics

| 方法 | 度量方式 | 关键细节 |
|------|---------|---------|
| **DynTok** | 局部token相似度 --> 自适应分组合并 | 利用视频帧中image patches的varying information density，自适应地分组并合并视觉token——在信息密集区域保留更多token，稀疏区域保留更少。实验发现在CLIP-generated visual representations空间中计算cosine similarity优于在LLM embedding空间中计算 |
| **LLaVA-Scissor** | Semantic Connected Components (SCC)，图连通分量分割 | 将token压缩重新构建为graph connected components partitioning任务：在视觉token之间构建语义相似度图，通过连通分量分析显式覆盖所有语义区域，避免了attention-based方法常见的对salient objects的过度偏向 |
| **SeqCompression** | 基于显著性的 "Cluster and Aggregate" | 在vision encoder和projector之后，用K-means++聚类visual tokens（按embedding similarity），每个cluster内的tokens均值合并为单个代表性token。对比实验表明saliency-based方法显著优于importance-agnostic方法 |
| **DivPrune** | Max-Min Diversity Problem (MMDP) | 将token剪枝形式化为最大化多样性问题——目标是构造一个保留token子集，使其最小pairwise distance最大化。这一形式化确保保留的token在特征空间中尽可能分散，覆盖更广泛的视觉语义 |

> **[感受]** Importance-Driven方法的核心贡献在于提供了多种"重要性"的操作化定义。DynTok发现"在CLIP空间而非LLM空间中计算相似度效果更好"这一实验结论非常有意思——它暗示了视觉token的冗余结构在不同表示空间中的表现不同，最优的压缩空间选择本身就是一个值得研究的问题。LLaVA-Scissor的SCC方法直接回应了attention bias问题——通过图论方法确保所有语义区域都被覆盖，而非仅仅保留最"显著"的区域，这在需要全面视觉理解的任务中应该表现更优。DivPrune将剪枝形式化为MMDP是一个优雅的数学表述，但MMDP本身是NP-hard问题，实际求解需要近似算法，其近似质量和计算开销的trade-off值得关注。SeqCompression的K-means++方法简洁有效，但K-means假设了球形cluster，对于复杂语义分布可能不够灵活。从MLLM效率优化的角度看，这些方法的一个共同局限是：它们定义的"重要性"都是基于视觉特征自身的性质（相似度、多样性），而非基于下游任务的需求。将task-aware信号融入importance estimation（如结合loss gradient或attention feedback），可能是提升这类方法上限的关键。

---

## 3.3 Token Compression in LLM

> LLM是MLLM中参数量最大的模块（通常占总参数的90%以上），视觉token在LLM中的处理开销远超encoder和projector。因此在LLM阶段进行压缩可以获得最直接的推理加速和显存节省。如Figure 5所示，LLM阶段的压缩根据生成过程的两个阶段分为：Prefilling Stage压缩（在第一次前向传播中减少视觉token）和Decoding Stage压缩（在自回归生成过程中压缩KV-cache）。

> 早期关注点集中在Prefilling阶段（因为短问答任务中prefilling占主导），但随着chain-of-thought推理和长文本生成需求的增长，Decoding阶段的KV-cache压缩变得同样甚至更加关键。

### 3.3.1 Compression in Prefilling Stage

在LLM的第一次前向传播中压缩视觉token。**关键特性**: 一旦token在某一层被移除，后续所有深层都无法再访问该token携带的信息——这使得prefilling压缩的决策具有**不可逆性**，对重要性评估的准确性要求极高。

现有方法基于对LLM处理视觉token时的行为模式观察，提出了四类压缩策略：

#### Importance-based

**核心方法**: 利用LLM内部的**text-to-image attention**作为重要性度量，保留attention score高的视觉token，丢弃其余。

| 方法 | 关键创新 | 细节 |
|------|---------|------|
| **FastV** | 首次观察到视觉token在LLM中的"attention sparsity"现象 | 发现视觉token接收到的attention score远低于文本token，揭示了视觉token携带信息的extreme sparsity。基于此观察，在LLM第2层用最后一个文本token的attention分布剪掉50%的视觉token |
| **PyramidDrop** | 发现冗余随LLM深度递增 --> 多阶段渐进剪枝 | 识别出视觉token的冗余度随LLM层深度增加而增大，据此设计multi-stage progressive pruning策略，每个阶段在更深层移除更多token |
| **SparseVLM** | 更细粒度的文本token选择 | 提出更精细的方法来选择用于评估视觉token重要性的文本token，提升了text-to-image attention的评估质量 |
| **VTW** | 激进策略：在某些层完全移除所有视觉token | 基于"视觉信息在足够深的层已被充分吸收"的假设，在特定层后完全删除所有视觉token |
| **TransPrune / VFlowOpt** | 结合attention score + 信息熵图 | 将attention scores与information entropy maps结合，提供更鲁棒的token重要性估计 |
| **CrossMisalign** | vision-to-vision attention | 利用视觉token之间的attention（而非text-to-vision）评估重要性，完全绕过对文本信号的依赖 |

**Attention Bias问题**: 随着attention-based pruning的发展，研究者发现了若干inherent biases：

| 问题 | 发现者 | 问题描述 | 解决方案 |
|------|--------|---------|---------|
| **Positional Bias** | Feather | 靠近输出端（序列末尾）的视觉token获得不成比例的高attention，源于RoPE的长程衰减特性 | 计算重要性时不应用RoPE，消除位置编码带来的偏差 |
| **Text-Vision Misalignment** | AdaTP | 直接使用LLM内部attention可能不准确反映真实的text-vision相关性 | 引入专用text encoder，在独立空间计算文本-视觉cosine相似度，避免LLM attention的偏差 |
| **Shallow Layer Instability** | VScan | 浅层的attention分布最不稳定，在此处做pruning决策误差最大 | 从中间层而非浅层开始剪枝，避开attention bias最严重的区域 |

**Flash Attention兼容性问题**: Flash Attention不直接暴露attention scores（为了优化内存效率），但大多数importance-based方法依赖attention map。

- 常见解决方案：在特定层选择性地重新计算标准attention map，仅在pruning决策层引入开销
- 但如果pruning在多层进行，重复计算的overhead可能显著增加inference latency
- **TopV**: 完全绕过attention分数，使用feature similarity + relative spatial distance + absolute central distance的组合度量
- **PACT**: 结合hidden state norms和global query vector评估重要性
- **GreedyPrune**: 直接用text-vision cosine similarity作为ranking criterion

> **[感受]** FastV的观察（视觉token在LLM中attention极度稀疏）是该领域最具影响力的empirical finding之一——它直接激发了大量后续工作。但这一观察也引发了一个更深层的问题：**为什么LLM对视觉token的attention如此稀疏？**是因为视觉信息本身冗余，还是因为当前MLLM的训练方式导致LLM未能充分利用视觉信息？如果是后者，那么token压缩可能在掩盖一个更根本的问题。Attention Bias的发现具有方法论层面的重要性——它揭示了"用attention score作为importance proxy"这一看似自然的做法存在系统性缺陷。Feather发现的RoPE-induced positional bias尤其值得注意：这意味着在使用RoPE的LLM中，sequence中位置靠后的视觉token天然获得更高attention，但这并不反映其真实重要性。Flash Attention兼容性问题是一个很现实的工程障碍——当前几乎所有production-level LLM推理都使用Flash Attention，如果压缩方法与之不兼容，就很难实际部署。TopV等不依赖attention score的方法虽然绕过了这一问题，但其评估质量是否能与attention-based方法匹配仍需更多验证。对于MLLM效率优化研究者而言，一个重要的研究方向是：设计与Flash Attention原生兼容的pruning机制——不是"绕过"Flash Attention，而是"集成到"Flash Attention的计算流程中。

#### Learnable Module-based

与基于预定义度量的importance-based方法不同，learnable methods引入**额外可训练组件**来学习评估token重要性或预测最优压缩率，实现数据驱动的动态压缩。

| 方法 | 机制 | 关键细节 |
|------|------|---------|
| **p-MoD** | 每层的weight predictor --> 按预测权重排名 --> 保留top R% | 为LLM每一层附加轻量weight predictor（受DynamicViT和AdaViT启发），预测每个token的重要性权重，排序后保留top R%。通过Gumbel-Softmax使框架全程可微分 |
| **GlimpsePrune** | visual token importance predictor --> 基于attention scores估计各层重要性 | 训练visual token importance predictor学习估计每个token在每一层的重要性，基于reused attention scores实现low-overhead prediction |
| **DyRate** | 轻量分类器预测最优剪枝率 | 不预测单个token的重要性，而是预测整个sequence的最优compression ratio，为每个输入instance选择最合适的压缩率 |
| **ATP-LLaVA** | 双prediction head --> 学习instance-specific阈值 | 使用MLP with dual prediction heads学习instance-specific thresholds for token pruning，实现自适应的token reduction during generation |

> **[感受]** Learnable methods的核心优势在于**自适应性**——它们可以根据输入内容和任务需求动态调整压缩策略，而非使用固定规则。p-MoD的per-layer predictor设计体现了一个重要洞察：不同层的最优pruning策略不同（浅层可能应该保留更多token，深层可以更激进地压缩），这与PyramidDrop的层级pruning观察一致。DyRate预测"压缩率而非单token重要性"是一个有趣的设计选择——它将问题从N维（N个token的重要性）简化为1维（一个压缩率标量），大幅降低了预测难度，但也丧失了对单个token的精细控制。这些方法的共同局限在于**训练依赖**：需要额外的训练数据和计算来训练predictor/classifier，且训练好的predictor对新的MLLM架构或新的数据分布可能需要重新训练。从MLLM效率优化的角度看，一个值得思考的问题是：这些额外的learnable module在推理时的overhead是否能被压缩带来的节省所覆盖？对于轻量级的predictor（如单层MLP），答案通常是肯定的，但对于更复杂的prediction module，需要仔细的cost-benefit分析。

#### Token Merging-based

与直接丢弃不同，token merging在LLM内部通过计算相似度并合并语义相似的视觉token来实现"软压缩"，在缩短序列的同时保留被压缩token的信息。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **ToMe** | 二分图软匹配 --> 基于pairwise similarity合并 | 源自ViT加速的bipartite soft matching算法，将tokens分为两组并通过pairwise similarity进行最优匹配和合并 |
| **LLaVolta** | 简单average pooling + 多训练阶段渐进降低压缩率 | 是最早将token merging应用到MLLM setting的工作之一，使用简单直接的average pooling aggressively compress vision tokens，通过multiple training stages progressively lower compression ratios来缓解性能损失 |
| **FiCoCo** | Filter --> Correlate --> Compress 三阶段 | 先筛选重要token子集，再计算保留token与剩余token之间的correlation matrix，最后以最小化信息损失为目标guided合并 |
| **FrameFusion** | 帧间cosine相似度 --> 合并跨帧相似空间区域 | 针对视频场景，计算每个visual token与前一帧对应空间位置token的cosine similarity，合并跨帧的相似spatial regions以消除时序冗余 |
| **CrossMisalign** | 作为visual information recovery mechanism | 引入specialized recovery scheme，将语义最冗余的token与其最相似的对应物合并（基于reused attention key embeddings的dot-product similarity），而非简单丢弃 |
| **HoliTom** | 直接合并低attention score的token | 采用简洁直接的策略：将attention score较低的token直接合并，而非丢弃，在简单性和信息保留之间取得平衡 |

> **[感受]** Token Merging在LLM内部的应用面临一个独特挑战：LLM的hidden states已经是高度contextualized的表示（经过多层self-attention后，每个token都编码了全局上下文），在这种高度entangled的表示空间中，"相似度"的含义不同于在encoder的浅层特征中。简单的cosine similarity是否仍是最佳度量？ToMe的bipartite matching算法虽然高效，但它假设了tokens可以被cleanly分为两组，这在语义复杂的场景中可能过于简化。FiCoCo的三阶段pipeline设计（filter-correlate-compress）最为完整——它不仅考虑了"哪些token应该被压缩"，还考虑了"如何从被压缩token中恢复信息到保留token中"，这种information-aware的合并策略在理论上应该比simple average merge损失更少信息。FrameFusion针对视频的帧间合并是一个自然而有效的设计，但它假设了相邻帧的空间对齐（相同位置的patch跨帧对应），对于存在camera motion或object movement的视频可能需要更复杂的alignment。对于MLLM效率优化研究者而言，LLM内部token merging的一个开放问题是：在LLM的哪一层做merge效果最好？过浅层信息不够contextualized，过深层可能已经来不及（大部分计算已经完成）。

#### Fusion-based

与pruning和merging直接缩短序列长度不同，fusion-based方法通过**cross-attention或self-attention**将视觉信息间接注入到其他token（如文本token或特殊压缩token）中，从而**避免处理过长的视觉token序列**。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **Flamingo** | GATED XATTN-DENSE层 | 在预训练LLM的各层之间插入gated cross-attention layers：文本token作为queries，视觉features作为keys/values。通过gating mechanism控制视觉信息的注入量，实现vision-language的deep interaction |
| **mPLUG-Owl3** | intra-text self-attention + cross-modal attention | 结合text tokens之间的self-attention和text-to-vision的cross-modal attention，使text tokens可以选择性地提取相关视觉信息，避免处理长视觉序列 |
| **CrossLMM** | 压缩视觉token+文本token作为queries --> 原始长序列作为K/V | 将compressed visual tokens和text tokens组合为queries，original long-sequence visual representations作为keys/values，通过visual-to-visual和text-to-visual cross-attention让LLM可以access高分辨率视觉内容而不增加主序列长度 |
| **VoCo-LLaMA** | 单个Vision Compression token | 引入单个VoCo (Vision Compression) token，修改attention mask使所有文本token仅attend to VoCo token（而非原始视觉token）。VoCo token通过attention机制吸收所有视觉内容，成为唯一的visual information接口 |
| **Victor** | 可学习visual register tokens | 在visual tokens后附加learned visual register tokens，这些register通过attention机制从visual tokens中吸收内容。在更深层中，原始visual tokens被全部丢弃，仅保留register tokens，实现极致压缩 |

> **[感受]** Fusion-based方法代表了一种根本不同的压缩哲学——不是"减少token"，而是"改变信息流动方式"。Flamingo的gated cross-attention设计在历史上极为重要——它证明了视觉信息不必以concatenated tokens的形式输入LLM，而可以通过cross-attention在每一层按需注入。这种架构设计从根源上避免了长视觉序列的 $O(n^2)$ 问题。VoCo-LLaMA的"single compression token"思路是极端压缩的代表——将所有视觉信息压缩到一个token中。理论上这是可能的（毕竟一个高维向量可以编码大量信息），但实际上单个token的信息瓶颈限制了模型在complex visual reasoning中的表现。Victor的register tokens机制与VoCo类似但更灵活（多个register），其"先吸收再丢弃"的两阶段策略是一个巧妙的设计。然而，fusion-based方法的一个被忽视的问题是：cross-attention层本身的参数和计算开销——如果需要在LLM的每一层都插入cross-attention，额外的参数量和FLOPs可能不小。从MLLM效率优化的角度看，fusion-based方法最适合在模型设计阶段就融入（如Flamingo），对于已有的MLLM做后处理式压缩则不太适用。这也意味着这类方法更适合"建新模型"的研究者，而非"加速现有模型"的研究者。

---

### 3.3.2 Compression in Decoding Stage

> 主要指**KV-cache压缩**——在自回归解码过程中减少缓存的key-value对，降低每步生成的内存占用和计算开销。随着multimodal chain-of-thought推理的兴起，输出长度从几句话扩展到数百甚至数千token，KV-cache的内存消耗和计算负载已成为generation过程中的critical bottleneck。

KV-cache压缩的核心思想是：不需要保留所有历史token的KV对，可以通过pruning、quantization或merging策略选择性地保留最重要的KV对。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **LOOK-M** | 累积attention scores --> 保留最近窗口 + top-K重要KV对 | 最早的multimodal KV-cache压缩工作之一，结合recency（最近窗口保留）和importance（累积attention排名），同时考虑时效性和重要性 |
| **MustDrop** | prefilling和decoding双阶段压缩 --> 仅保留最终层retained的视觉KV | 跨prefilling和decoding两个阶段的全链路压缩——在prefilling阶段筛选出的视觉token的KV对被选择性地保留到decoding阶段 |
| **SparseMM** | 识别visual heads --> 为其分配更多KV-cache预算 | 通过OCR-based task识别出哪些attention heads主要负责visual understanding（"visual heads"），为这些heads分配更多KV-cache budget，对non-visual heads采用更激进的压缩策略 |
| **DyCoke** | text-vision attention引导 --> 动态保留高attention的KV对 | 在每个decoding step中，仅保留text-vision attention score最高的KV pairs。若attention分布在后续步骤中显著变化，KV cache会相应更新 |
| **Video-XL-2** | Bi-level KV解码 --> 当前query动态选择dense或sparse KV表示 | 针对视频帧的大量时序冗余，模型根据当前query动态选择从dense或sparse KV representations中检索，丢弃大量query-irrelevant的KV pairs |
| **LiveVLM** | 先丢弃不重要视觉token的KV --> 再合并每帧KV为单个tuple | 针对视频流场景的两级策略：先基于attention scores丢弃不重要视觉token的KV对，再将每帧剩余KV merge为单个KV tuple |
| **InfiniPot-V** | Temporal-axis Redundancy (TaR) + Value Norm (VaN) 双指标 | 整合两种互补的重要性评估标准：TaR识别跨时间步的冗余KV对，VaN利用value vector的norm来评估信息量，共同指导压缩 |
| **StreamMem** | 固定大小KV memory --> 基于attention scores压缩 | 在固定大小的KV memory内运作，基于visual tokens和generic queries之间的attention scores进行压缩，适合实时问答的streaming场景 |

> **[感受]** KV-cache压缩是一个与纯LLM领域高度交叉的方向（StreamLLM、FastGen、H2O等），但MLLM场景引入了独特的挑战：视觉KV对与文本KV对的重要性分布可能截然不同，且视觉KV对通常占据KV-cache的大部分容量。SparseMM的"visual heads"概念非常有洞察力——它揭示了LLM中不同attention heads对visual information的处理是不均匀的，这种head-level的分析可以指导更精细的cache allocation策略。DyCoke的动态更新机制回应了一个关键问题：在自回归生成过程中，模型对不同视觉区域的需求是变化的——第一个生成token可能需要关注图像左侧，而后续token可能需要右侧的信息，因此static pruning可能不够。Video-XL-2的bi-level KV design对视频场景特别有意义——大量帧的KV都是高度冗余的，只有query-relevant的帧才需要dense representation。从MLLM效率优化的角度看，KV-cache压缩的实用价值可能比prefilling压缩更大，因为它直接影响了long-form generation（如详细描述、chain-of-thought推理）的latency和memory，且对模型质量的影响相对更可控（因为仍保留了完整的prefilling过程）。然而，一个被忽视的问题是：当prefilling和decoding阶段的压缩同时应用时，两者之间的交互效应尚不清楚——prefilling压缩可能已经改变了KV-cache的分布特性，使得为uncompressed setting设计的KV-cache压缩策略不再最优。

---

## 3.4 Token Compression in Multi-Module

> 前面3.1-3.3讨论了在单个模块内部进行的压缩。然而，单模块压缩面临一个固有局限：每个模块能获取的信息是有限的（encoder处缺乏文本信号，projector处缺乏LLM的deep reasoning信号，LLM处视觉信息已经过projector的转换）。多模块压缩通过跨模块协调，试图突破单模块的信息瓶颈，实现系统级的全局最优。

### 3.4.1 Collaborative Compression

多模块间**联合**和**协调**地减少token，不同模块的压缩决策相互配合而非独立运作。

| 方法 | 策略 | 关键细节 |
|------|------|---------|
| **CrossGET** | 在visual和language分支的self-attention和FFN之间插入CrossGET模块 | 最早采用multi-module token compression的工作之一，在vision和language分支的self-attention和FFN层之间插入CrossGET模块，跨层减少token数。解决了早期方法"先提取视觉信息才能利用文本引导"的局限，使视觉处理层能更早获得文本监督信号 |
| **LLaMA-VID** | 视觉token与文本queries cross-modal交互 --> 上下文token + 内容token | 利用视觉token与textual queries的cross-modal interaction生成两种互补token——context token（全局上下文）和content token（局部细节），使每帧仅需2个token即可表示，实现极端压缩 |
| **PAR** | 区分external + internal redundancy --> query rewriting + semantic clustering + token router | 提供了对visual token redundancy的细粒度分析：external redundancy指与任务无关的token（通过query rewriting和semantic retrieval移除），internal redundancy指token间的语义重复（通过semantic clustering和token router消除）。这种双管齐下的策略实现了更精确的压缩 |

> **[感受]** Collaborative Compression的核心insight是：不同模块拥有互补的信息来源，通过协调可以做出比任何单模块更好的压缩决策。CrossGET的设计尤其有启发性——它在视觉处理的早期阶段就引入了语言信号，这挑战了"视觉编码应该独立于文本"的传统假设。但这种跨模块耦合也带来了训练和部署的复杂性——修改了visual和language分支的内部结构，使得模型难以使用pre-trained components的权重。LLaMA-VID的context+content dual-token设计是一个elegant的solution——用一个token捕获"全局是什么"，另一个token捕获"局部细节"，每帧仅2个token就能工作。PAR对冗余的external/internal分类是一个有价值的概念框架——它明确区分了"不需要的信息"和"重复的信息"，这两种冗余需要不同的策略来处理。从MLLM效率优化的角度看，collaborative compression代表了该领域的成熟方向，但也面临更高的工程复杂度。它最适合"从头设计新MLLM"的场景，对于"加速现有MLLM"的需求，单模块的plug-in方法可能更为实际。一个未被充分探索的问题是：不同模块的压缩之间是否存在最优的"压缩比分配"——即总压缩率固定时，应该在encoder/projector/LLM分别压缩多少才能最小化性能损失？

### 3.4.2 Progressive Compression

跨多个阶段的**渐进式**压缩pipeline——在MLLM的不同处理阶段依次进行压缩，每个阶段利用当前可获得的信息做进一步精炼，从early visual processing一直延伸到late-stage LLM inference。

| 方法 | 多阶段策略 | 关键细节 |
|------|----------|---------|
| **MustDrop** | Vision encoding --> Prefilling --> Decoding 全链路压缩 | 在vision encoding阶段通过carefully designed mechanisms跨视觉编码、prefilling和decoding三个阶段进行压缩，结合merging highly similar spatial tokens、dual-attention filtering和output-aware KV cache策略，实现端到端全inference pipeline加速 |
| **FiCoCo** | Filter --> Correlate --> Compress 三阶段 | 将token压缩分解为三个递进问题：(1) Filter确定哪些token应被丢弃，(2) Correlate计算保留token与被丢弃token的关联以确定保留位置，(3) Compress将丢弃token的信息融合到保留token中以保留关键信息 |
| **DyCoke** | 阶段1: 帧间cosine相似度合并 --> 阶段2: KV-cache动态剪枝 | 两阶段策略：第一阶段通过计算相邻帧对应token的cosine similarity进行帧间合并（消除时序冗余），第二阶段在decoding时基于text-vision attention动态剪枝KV cache（保持query relevance） |

**核心趋势**: 从单模块孤立优化 --> 全系统端到端协同压缩。这一趋势表明，单一位置的压缩收益可能已接近饱和，未来的增益需要来自pipeline级别的全局优化。

> **[感受]** Progressive Compression是token压缩领域最有前景的方向之一——它承认了"不同阶段可获得的信息不同，因此应该在每个阶段做当时可以做到的最好的压缩"。MustDrop的全链路设计最为彻底——从encoder到decoding的每一步都有压缩，形成了一个完整的compression cascade。但这种全链路设计也引入了一个关键问题：**误差累积**——前一阶段的压缩误差会传播并放大到后续阶段。如何控制multi-stage compression中的error propagation（例如通过每阶段的recovery mechanism或error-aware decision making），是progressive compression需要解决的核心技术挑战。FiCoCo的三阶段分解（filter-correlate-compress）在概念上最为清晰，且"correlate"阶段（计算保留与丢弃token之间的关联）是一个巧妙的设计——它不是简单地丢弃不重要token，而是先理解被丢弃信息与保留信息的关系，再做信息融合。DyCoke的两阶段策略（spatial merging + temporal KV pruning）针对视频场景很有针对性，但两个阶段之间的最优划分点（在哪一层从阶段1切换到阶段2）可能需要task-specific tuning。从MLLM效率优化的角度看，progressive compression的终极目标应该是：**建立一个统一的、可微分的压缩框架**，让系统自动学习在pipeline的每个位置应该压缩什么、压缩多少，而非依赖人工设计的multi-stage heuristics。这也许需要将压缩决策纳入MLLM的端到端训练目标中，使压缩成为模型能力的一部分而非后处理步骤。

---

## 个人笔记与总结

### Section 3 的核心洞察

1. **压缩位置决定了信息可获取性**：在encoder处只有视觉信号，在projector处可以引入跨模态信号，在LLM处有完整的text-vision交互信息。越靠后的位置信息越丰富，但可节省的计算越少（因为前面的计算已经完成了）。

2. **不可逆性 vs 恢复能力**：大多数压缩方法是one-shot且不可逆的。Token Recovery Mechanisms（如RecoverableCompression、MustDrop）是一个被低估但重要的方向。

3. **Attention Bias是一个cross-cutting concern**：无论在encoder内部（attention-based dropping）还是LLM内部（importance-based pruning），基于attention score的重要性评估都存在系统性偏差（positional bias、foreground bias），这需要被每一种基于attention的方法认真对待。

4. **从单模块到多模块**：单模块压缩的收益正在趋近饱和，Progressive和Collaborative Compression代表了下一阶段的研究范式。

5. **Training-free vs Re-train的张力**：Plug-in（training-free）方法部署方便但压缩质量受限，Re-train方法质量更高但需要额外训练成本。两者之间的sweet spot（如lightweight fine-tuning配合plug-in compression）值得探索。

### 与MLLM效率优化研究最相关的方向

- **自适应压缩率**: 根据输入复杂度和query类型动态调整压缩率（AVG-LLaVA、MQT、DyRate等已有初步尝试）
- **Flash Attention兼容的压缩**: 设计不依赖显式attention map的importance estimation（TopV、PACT等）
- **端到端的压缩-性能联合优化**: 将压缩决策纳入训练目标，而非作为独立的post-hoc步骤
- **跨模块压缩比分配**: 在总压缩预算固定时，如何最优地分配各模块的压缩率

### 关键方法标记

- **高影响力奠基工作**: FastV (LLM prefilling), BLIP-2/Q-Former (Projector), ToMe (Encoder merging)
- **工程实用性最强**: Pixel Shuffle (InternVL/Qwen2-VL), Average Pooling, FastV
- **学术创新性最高**: FiCoCo (三阶段progressive), LLaVA-Scissor (SCC图论方法), DivPrune (MMDP形式化)
- **最有启发性的观察**: FastV的attention sparsity, Feather的RoPE bias, SparseMM的visual heads
