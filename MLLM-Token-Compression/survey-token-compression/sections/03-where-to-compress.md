# 3. Where to Compress Tokens in MLLMs

> 本章是survey的核心章节，按压缩在MLLM架构中的**位置**系统分类。

## 总览分类图 (Figure 2)

```
Where to Compress
├── 3.1 Vision Encoder
│   ├── 3.1.1 Inside-Encoder
│   │   ├── Visual Token Dropping
│   │   ├── Visual Token Merging
│   │   └── Multi-Scale Compression
│   └── 3.1.2 Outside-Encoder
│       ├── Purely-Vision Compression
│       └── Text-guided Compression
├── 3.2 Projector
│   ├── 3.2.1 Transformation-Based
│   ├── 3.2.2 Query-Based
│   └── 3.2.3 Importance-Driven
├── 3.3 LLM
│   ├── 3.3.1 Prefilling Stage
│   └── 3.3.2 Decoding Stage
└── 3.4 Hybrid (Multi-Module)
    ├── 3.4.1 Collaborative Compression
    └── 3.4.2 Progressive Compression
```

---

## 3.1 Token Compression in Vision Encoder

> 视觉编码器是处理视觉输入的第一个模块，在此阶段压缩可获得全链路的效率收益。

### 3.1.1 Inside-Encoder Compression

在ViT编码器**内部**直接修改token流，减少自注意力复杂度。

#### A. Visual Token Dropping

**核心思路**: 计算每个token的重要性分数 → 排名 → 保留Top-K → 丢弃其余。

**三种评分策略**:

| 策略 | 方法 | 代表工作 |
|------|------|---------|
| **Similarity-based** | 衡量token与全局表示(CLS token等)的相似度，高相似=冗余 | TRIM, SAINT |
| **Attention-based** | 利用ViT注意力权重判断显著性 | VisPruner, HiPrune, VFlowOpt, MADTP, SmartTrim |
| **Heuristic-based** | 利用任务先验(如自中心视频的几何稳定性) | EgoPrune, METEOR |

**TRIM**: 用CLIP嵌入衡量文本-视觉相关性 + 自适应IQR阈值
**SAINT**: 基于图的方法，联合优化剪枝率和冗余阈值
**METEOR**: 浅层用平均相似度，深层用class attention（层自适应策略）

#### B. Visual Token Merging

**核心思路**: 将相似token聚合为代表性token，保留信息但缩短序列。

| 合并策略 | 描述 | 代表工作 |
|---------|------|---------|
| **Proximity-based** | 空间/时序邻近的token合并 | downsampling, pixel-shuffle+channel merging, 3D convolution |
| **Similarity-based** | 跨越空间距离的语义相似token合并 | patch-to-class correlation, density-based clustering |
| **Cross-modal** | 利用文本上下文指导合并决策 | bidirectional language-aware signals |
| **Hybrid** | 先attention-based剪枝 → 再weighted merging恢复信息 | learnable abstraction methods |

**时序维度合并**:
- Joint temporal-spatial aggregation: 同时合并相似帧和patch
- Frame-level fusion: 自适应地对连续帧进行加权融合

#### C. Multi-Scale Compression

> 单尺度方法固定粒度，难以兼顾细节和全局。多尺度方法通过跨层/跨编码器/跨分辨率协调来利用层级语义。

| 类型 | 描述 | 代表工作 |
|------|------|---------|
| **Multi-Layer** | 从ViT多层提取并融合特征 | LLaVA-STF, METEOR, Chat-UniVi, LaCo |
| **Multi-Encoder** | 组合不同架构/训练范式的编码器 | Cambrian-1 (DINOv2 + CLIP), METEOR |
| **Multi-Resolution** | 高分辨率=细节，低分辨率=全局 | FastVLM, ADMIRE, LinVT, M3, VideoChat-Flash |

---

### 3.1.2 Outside-Encoder Compression

在Vision Encoder输出之后、Projector之前进行压缩。
**优势**: Plug-and-play，不需要修改编码器。

#### A. Purely-Vision Compression

仅基于视觉信号（vision-vision semantic relevance）进行压缩，不依赖文本。

**Selection-then-merge范式**:
- **VisionZip**: 重要性估计 + 代表性约束 → 选择可复用token
- **Fourier-VLM**: 频域低通滤波抑制高频冗余
- **LLaVA-STF**: 跨层拼接 + Multi-Block Token Fusion (MBTF)

**Visual Attention Bias问题**:
- 早期方法（LLaVA-PruMerge, VTC-CLS, FasterVLM）利用CLS token和self-attention做稀疏化
- 但attention-based选择存在**偏向前景对象、忽略全局上下文**的偏差
- **HoloV**: 通过整合全局视觉上下文来平衡前景和背景token

**极端压缩 (Extreme Compression)**:
- **LLaMA-VID**: 每帧压缩为1个Content Token
- **Flash-VStream**: K-means聚类低分辨率特征为Context Synopsis Memory
- **LLaVA-PruMerge**: 最近邻聚类的可学习token合并

#### B. Text-guided Compression

利用文本语义先验指导压缩，关注query相关区域。

| 方法 | 策略 |
|------|------|
| **PAR** | 解析query为实体和动作 → 重新加权视觉token |
| **QG-VTC** | 计算question-to-vision相似度 → 引导token保留 (4×-8× 压缩) |
| **LongVU** | 跨模态查询与帧/区域候选结合 → 先帧级过滤再token级选择 |

**Text-guided在Outside-VE位置的优势**: 视觉token已完整编码，cross-modal交互尚未开始 → 减少文本对低级视觉编码的干扰。

**实践范式**: 先purely-vision做文本无关压缩 → 再text-guided做query相关精炼

#### C. Token Recovery Mechanisms

在高压缩率下通过恢复机制增强鲁棒性:
- **RecoverableCompression**: 基于置信度和冲突阈值触发重采样
- **MustDrop**: 通过不确定性门控进行多阶段恢复
- **ToCom**: 处理训练-测试压缩率不匹配问题
- **VTC / Video-XL-Pro**: 通过视觉重建监督优化压缩

---

## 3.2 Token Compression in Projector

> Projector是视觉与语言模态之间的桥梁，天然适合进行token压缩。

### 3.2.1 Transformation-Based Compression

通过**确定性变换**直接改变feature map的空间结构来减少token数。

#### Pooling

$$Y_{i,j,c} = \frac{1}{|\Omega_{i,j}|} \sum_{(u,v) \in \Omega_{i,j}} X_{u,v,c}$$

| 方法 | 策略 |
|------|------|
| **MobileVLM V2** | Lightweight Downsample Projector (LDP), 2×2 average pooling |
| **DeCo** | 自适应average pooling的有效性验证 |
| **AVG-LLaVA** | Visual Granularity Scaler + Visual Granularity Router，选择最合适粒度 |
| **TC-LLaVA** | 简单global average pooling (视频) |
| **PLLaVA** | 空间+时序自适应average pooling |

#### Pixel Shuffle

$$\mathbf{Y} = \text{reshape}(\mathbf{X}, H/r, W/r, C \cdot r^2)$$

- 空间token减少 $r^2$ 倍，通道维度相应增加
- 后接MLP对齐到LLM嵌入维度
- 代表: **InternVL 1.5**, **NVLM**, **InternVL 2.5**, **Qwen2-VL**

#### Convolution

$$Y_{i,j}^{(o)} = \sum_{c=1}^{C_{in}} \sum_{m=1}^{k_h} \sum_{n=1}^{k_w} \mathbf{W}_{m,n,c}^{(o)} \cdot \mathbf{X}_{i+m-1,j+n-1}^{(c)} + b^{(o)}$$

- 通过可学习权重选择性整合局部信息
- 常与pooling组合: Honeybee的C-Abstractor (ResNet blocks + avg pooling), MobileVLM V2的LDP (pointwise + depthwise conv + avg pooling)

---

### 3.2.2 Query-Based Compression

利用可学习query向量通过cross-attention聚合视觉信息。

#### Q-Former (BLIP-2)

**核心机制**:
- $Q$: 可训练query嵌入（少量，如32个）
- $K/V$: 来自frozen vision encoder的patch嵌入
- 通过stacked self-attention + cross-attention，queries选择性聚合任务相关视觉信息
- 输出: 紧凑的query嵌入 → 线性投影到LLM嵌入空间

**后续工作**: MiniGPT-4, InstructBLIP

#### Q-Former变体

| 变体 | 改进 |
|------|------|
| **Qwen-VL** | 单层cross-attention，降低复杂度 |
| **Honeybee** | C-Abstractor (保留局部结构) + D-Abstractor (Deformable Attention增强局部性) |
| **MQT** | 可变query数量（训练时随机采样 $m < M$），支持不同粒度 |
| **TG-LLaVA** | 可学习latent嵌入编码全局文本语义 → text-driven masking |
| **LLaVA-Mini** | 增加Modality-Pre Fusion模块，融合视觉+指令后再压缩 |

#### Cross-Attention-Based

| 方法 | 策略 |
|------|------|
| **CATP** | 基于cross-attention概率投票 → 按聚合重要性剪枝 |
| **TokenPacker** | 低分辨率特征作为point-based queries → Point-to-Region cross-attention逐步注入高分辨率信息 |
| **HiRes-LLaVA** | 下采样特征作为queries → 与原始特征cross-attention生成紧凑序列 |
| **mPLUG-DocOwl2** | 全局特征作为queries → cross-attention聚合文本语义 (高分辨率文档) |
| **QueCC** | 将用户query注入视觉表示 → 携带任务语义的cross-attention |

---

### 3.2.3 Importance-Driven Compression

通过估计每个token的重要性来选择性保留/合并。

#### Various Similarity Metrics

| 方法 | 度量方式 |
|------|---------|
| **DynTok** | 局部token相似度 → 自适应分组合并 (CLIP空间优于LLM空间) |
| **LLaVA-Scissor** | Semantic Connected Components (SCC)，图连通分量分割 |
| **SeqCompression** | 基于显著性的 "Cluster and Aggregate" (K-means++ 聚类 → 均值合并) |
| **DivPrune** | Max-Min Diversity Problem (MMDP)，最大化保留token集合的最小距离 |

#### Saliency-Based

- **SeqCompression**: 实验表明显著性方法 > 重要性无关方法

#### Innovative Metrics-Based

- **DivPrune**: 将token剪枝形式化为最大化多样性问题

---

## 3.3 Token Compression in LLM

> LLM参数量最大，压缩在此阶段可显著降低计算开销。

### 3.3.1 Compression in Prefilling Stage

在LLM的第一次前向传播中压缩，移除的token在后续深层无法再被访问。

#### Importance-based

**核心方法**: 利用text-to-image attention作为重要性度量。

| 方法 | 关键创新 |
|------|---------|
| **FastV** | 首次观察到视觉token在LLM中注意力远低于文本token → 在第2层用最后文本token的attention剪掉50% |
| **PyramidDrop** | 发现冗余随LLM深度增加 → 多阶段渐进剪枝 |
| **SparseVLM** | 更细粒度的文本token选择方法 |
| **VTW** | 激进方法：在某些层完全移除所有视觉token |
| **TransPrune / VFlowOpt** | 结合attention score + 信息熵图 |

**Attention Bias问题**:
- **Feather**: 发现靠近输出端的视觉token获得不成比例的高attention（RoPE的长程衰减特性）→ 建议不应用RoPE计算重要性
- **AdaTP**: 引入专用text encoder计算文本-视觉cosine相似度，避免attention偏差
- **VScan**: 从中间层而非浅层开始剪枝（避免浅层bias最严重的区域）

**Flash Attention兼容性问题**:
- Flash Attention不直接暴露attention scores → 需要选择性地在特定层重新计算attention map
- **TopV**: 绕过attention分数，使用feature similarity + relative spatial distance + absolute central distance

#### Learnable Module-based

| 方法 | 机制 |
|------|------|
| **p-MoD** | 每层的weight predictor → 按预测权重排名 → 保留top R% |
| **GlimpsePrune** | visual token importance predictor → 基于attention scores估计各层重要性 |
| **DyRate** | 轻量分类器预测最优剪枝率 |
| **ATP-LLaVA** | 双prediction head → 学习instance-specific阈值 |

#### Token Merging-based

| 方法 | 策略 |
|------|------|
| **ToMe** | 二分图软匹配 → 基于pairwise similarity合并 (源自ViT加速) |
| **LLaVolta** | 简单average pooling + 多训练阶段渐进降低压缩率 |
| **FiCoCo** | Filter → Correlate → Compress 三阶段 |
| **FrameFusion** | 帧间cosine相似度 → 合并跨帧相似空间区域 |
| **HoliTom** | 直接合并低attention score的token |

#### Fusion-based

通过cross-attention或self-attention将视觉信息注入其他token，间接实现压缩。

| 方法 | 策略 |
|------|------|
| **Flamingo** | GATED XATTN-DENSE层，文本作为queries，视觉作为K/V |
| **mPLUG-Owl3** | intra-text self-attention + cross-modal attention |
| **CrossLMM** | 压缩视觉token+文本token作为queries，原始长序列作为K/V → visual-to-visual + text-to-visual cross-attention |
| **VoCo-LLaMA** | 单个Vision Compression token → 修改attention使文本token仅attend to VoCo token |
| **Victor** | 可学习visual register tokens → 通过attention吸收视觉内容 → 后续丢弃原始视觉token |

---

### 3.3.2 Compression in Decoding Stage

> 主要指**KV-cache压缩**，减少自回归解码的内存和计算开销。

| 方法 | 策略 |
|------|------|
| **LOOK-M** | 累积attention scores → 保留最近窗口 + top-K重要KV对 |
| **MustDrop** | prefilling和decoding双阶段压缩 → 仅保留最终层retained的视觉KV |
| **SparseMM** | 识别visual heads → 为其分配更多KV-cache预算 |
| **DyCoke** | text-vision attention引导 → 动态保留高attention的KV对 |
| **Video-XL-2** | Bi-level KV解码 → 当前query动态选择dense或sparse KV表示 |
| **LiveVLM** | 先丢弃不重要视觉token的KV → 再合并每帧KV为单个tuple |
| **InfiniPot-V** | Temporal-axis Redundancy (TaR) + Value Norm (VaN) 双指标 |
| **StreamMem** | 固定大小KV memory → 基于attention scores压缩 |

---

## 3.4 Token Compression in Multi-Module

### 3.4.1 Collaborative Compression

多模块间**联合**和**协调**地减少token。

| 方法 | 策略 |
|------|------|
| **CrossGET** | 在visual和language分支的self-attention和FFN之间插入CrossGET模块 |
| **LLaMA-VID** | 视觉token与文本queries cross-modal交互 → 上下文token + 内容token |
| **PAR** | 区分external redundancy (任务无关) 和 internal redundancy (token间冗余) → query rewriting + semantic clustering + token router |

### 3.4.2 Progressive Compression

跨多个阶段的**渐进式**压缩pipeline。

| 方法 | 多阶段策略 |
|------|----------|
| **MustDrop** | Vision encoding → Prefilling → Decoding 全链路压缩 |
| **FiCoCo** | Filter → Correlate → Compress 三阶段 |
| **DyCoke** | 阶段1: 帧间cosine相似度合并 → 阶段2: KV-cache动态剪枝 |

**趋势**: 从单模块孤立优化 → 全系统端到端协同压缩

---

## 个人笔记

<!-- 在此添加你对Section 3的思考和关键洞察 -->

### 最重要的方法 (个人标记)
- TODO

### 与我的工作最相关的方向
- TODO

