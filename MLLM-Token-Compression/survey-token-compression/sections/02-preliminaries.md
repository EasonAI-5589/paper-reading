# 2. Preliminaries

## 2.1 MLLM架构

### 2.1.1 三组件设计

现代MLLM采用统一的**三组件架构**，将视觉感知与语言生成串联在一条pipeline中：

```
输入图像/视频 → [Vision Encoder Ev] → [Projector P] → [LLM G] → 文本输出
                                            ↑
                                 文本指令 → [Tokenizer Et] ──┘
```

#### (1) Vision Encoder $\mathcal{E}_v$

- 通常基于预训练的 **SigLIP** 或 **CLIP** 视觉编码器
- 将视觉输入（图像或视频帧）转换为dense visual token序列：
  $$\mathbf{Z}^v = \mathcal{E}_v(\mathcal{X}^v) \in \mathbb{R}^{n_v \times d_v}$$
- $n_v$：视觉token数量（由输入分辨率和patch大小决定，如224×224图像、14×14 patch → $n_v = 256$）
- $d_v$：视觉特征维度

#### (2) Projector $\mathcal{P}$

- 充当**视觉与语言模态之间的桥梁**，负责特征空间对齐
- 将视觉特征从视觉编码器的 $d_v$ 维空间映射到LLM的嵌入空间 $d_l$ 维：
  $$\mathbf{H}^v = \mathcal{P}(\mathbf{Z}^v) \in \mathbb{R}^{n_v \times d_l}$$
- 常见实现形式：简单MLP（如LLaVA）、带下采样的卷积projector、Q-Former（如BLIP-2）、Resampler（如Flamingo）
- **注意**：部分projector设计本身就具有token数量压缩的功能（如Q-Former将$n_v$个视觉token压缩为固定数量的query token）

#### (3) LLM $\mathcal{G}$

- 作为多模态推理的核心引擎，处理拼接后的视觉+文本token序列：
  $$\mathbf{Y} = \mathcal{G}([\mathbf{H}^v; \mathcal{E}_t(\mathcal{X}^t)])$$
- 其中 $\mathcal{E}_t$ 为文本tokenizer，$\mathcal{X}^t$ 为文本输入
- 视觉token和文本token共享同一个Transformer的自注意力机制，参与统一的序列建模

> **[感受]** 三组件设计的简洁性既是MLLM成功的原因，也是效率问题的根源。Vision Encoder产出的token数量由分辨率和patch size机械地决定，与下游任务的实际信息需求完全解耦——这意味着无论你是问"图片里有没有猫"还是"请逐字识别文档中的所有文字"，进入LLM的视觉token数量是一样的。从效率优化的角度看，这三个组件各自提供了不同的压缩切入点：(1) 在Encoder内部压缩，可以利用视觉注意力模式但缺乏文本信息的指导；(2) 在Projector处压缩，可以同时接触视觉和语言两侧的信息；(3) 在LLM内部压缩，拥有最丰富的跨模态交互信号但此时压缩的收益受限于前面已经完成的prefill计算。理想的方案可能是在每个组件上都做适度的压缩（而不是在单一位置做激进压缩），形成一个"渐进式漏斗"结构。

### 2.1.2 计算复杂度分析

对于序列长度 $n$、隐藏维度 $d$、FFN中间维度 $m$ 的单层Transformer，FLOPs分解为：

$$\text{Layer FLOPs} = \underbrace{4nd^2}_{\text{QKV投影 + 输出投影}} + \underbrace{2n^2d}_{\text{自注意力计算}} + \underbrace{2ndm}_{\text{FFN前馈}}$$

$L$ 层Transformer的总FLOPs：

$$\text{Total FLOPs} = L \times (4nd^2 + 2n^2d + 2ndm)$$

其中总序列长度 $n = n_t + n_v$（文本token数 + 视觉token数）。

**关键瓶颈分析**：

- $4nd^2$ 和 $2ndm$ 项：与 $n$ 成**线性**关系，受模型维度 $d$ 和 $m$ 主导
- $2n^2d$ 项：与 $n$ 成**二次**关系，即自注意力机制的计算复杂度
- **当 $n_v$ 很大时**（如高分辨率图像 $n_v > 2000$，或长视频 $n_v > 10000$），$2n^2d$ 项主导整体计算开销
- 显存占用同样受二次复杂度影响：KV-cache的大小为 $O(n \times d \times L)$，在自回归生成过程中持续增长

> **[感受]** 复杂度公式清晰地揭示了为什么token压缩是MLLM效率优化中ROI最高的方向之一：减少 $n_v$ 可以同时降低注意力计算的二次项和KV-cache的线性存储开销，而且这个收益随着序列长度的增加而加速放大。举一个具体的例子：假设 $n_t = 100$，$n_v = 2000$，将 $n_v$ 压缩到500，则总序列长度从2100降到600，注意力FLOPs降至原来的约 $(600/2100)^2 \approx 8.2\%$——这是一个数量级的减少。相比之下，量化（减小 $d$ 的有效精度）和剪枝（减少 $L$）虽然也能提升效率，但它们改变的是模型本身的能力，而token压缩改变的是输入的信息密度，两者是正交的。这也解释了为什么token压缩方法可以与量化、FlashAttention等技术叠加使用。需要注意的是，上述分析假设了标准的dense attention；如果未来MLLM广泛采用稀疏注意力或线性注意力，$2n^2d$ 项的主导地位会被削弱，那时token压缩的边际收益也会相应下降——但从目前的技术发展看，dense attention仍将在较长时间内是主流。

## 2.2 Token压缩

### 2.2.1 形式化定义

给定原始的混合token序列 $\mathbf{H} \in \mathbb{R}^{N \times d_l}$（其中 $N = n_t + n_v$），token压缩定义为一个映射函数：

$$\mathbf{H}_{\text{comp}} = \mathcal{C}(\mathbf{H}) \in \mathbb{R}^{M \times d_l}, \quad M < N$$

**压缩率**定义为：

$$R_{\text{comp}} = \frac{N}{M}$$

- $R_{\text{comp}} = 4\times$ 表示token数量减少为原来的 $1/4$
- 压缩率越高意味着效率增益越大，但信息损失的风险也越高
- 实际应用中，$4\times$ 到 $16\times$ 是常见的压缩范围；部分极端视频场景探索了 $64\times$ 以上的压缩率

> **[感受]** 形式化定义看似简单，但隐含了一个重要的设计选择：压缩函数 $\mathcal{C}$ 的输出维度 $M$ 是**固定的还是自适应的**？大多数现有方法使用固定的压缩率（如统一丢弃75%的token），但这忽略了一个事实——不同输入的信息冗余程度差异巨大。一张纯白背景上的单个物体可能只需要极少数token就能充分表示，而一张信息密集的文档图像可能每个token都携带关键信息。从MLLM效率优化的角度，真正理想的压缩函数应该是 $M = f(\mathbf{H}, \mathcal{X}^t)$，即压缩后的token数量同时取决于视觉内容本身和文本query的信息需求。这需要在压缩策略中引入"信息量估计"模块——这可能是一个高价值的研究方向。

### 2.2.2 两种核心冗余类型

论文将可压缩的冗余归纳为两大类：

#### (i) Intra-Visual 冗余（视觉内部冗余）

视觉token序列内部存在大量可压缩的冗余，具体表现为：

- **背景区域重复**：图像中大面积的天空、墙壁、地面等均匀区域产生的token高度相似
- **相邻patch相似性**：空间上相邻的patch在特征空间中往往高度相关，尤其在非边缘区域
- **视频帧间冗余**：相邻帧之间的变化通常很小（尤其在低运动场景），大量token携带重复信息

**对应的压缩手段**：空间聚合（spatially adjacent token merging/pooling）、时序聚合（temporal token selection/aggregation）

#### (ii) Cross-Modal 冗余（跨模态冗余）

并非所有视觉token对于给定的文本query都具有同等价值：

- 用户问"图片中猫的颜色是什么"时，背景中的建筑物、树木等视觉token对回答该问题完全无关
- 这些与文本query语义不相关的视觉token构成了**跨模态冗余**
- 跨模态冗余的判定具有**动态性**——同一张图片在不同query下，"冗余"的token集合是不同的

**对应的压缩手段**：文本引导的token筛选（text-guided filtering/scoring），利用视觉-文本的attention分数识别并移除低相关性token

### 2.2.3 关键观察：视觉token的主导地位

- 在典型的MLLM输入中，视觉token数量通常是文本token的 **20倍以上**（如 $n_v = 2000$，$n_t = 100$）
- 这意味着总序列长度 $n$ 几乎完全由 $n_v$ 决定
- 因此，绝大多数token压缩方法**主要针对减少 $n_v$**，即压缩视觉token而非文本token
- 这也解释了为什么多数方法的压缩函数实际上是 $\mathcal{C}: \mathbb{R}^{n_v \times d} \rightarrow \mathbb{R}^{m_v \times d}$（$m_v \ll n_v$），文本token保持不变

> **[感受]** 两种冗余类型的区分对于设计压缩方法具有直接的指导意义，但也揭示了一个根本性的张力：**Intra-Visual冗余的消除是text-agnostic的**（不需要知道用户问什么就能做），而**Cross-Modal冗余的消除是text-dependent的**（必须知道query才能判断哪些token冗余）。这意味着两类方法最适合部署在pipeline的不同位置：Intra-Visual压缩适合在Vision Encoder内部或Projector之前进行（此时尚未接触到text input），Cross-Modal压缩适合在LLM内部进行（此时视觉和文本token已经在同一attention空间中交互）。从研究的角度看，一个有趣的问题是：这两种冗余之间是否存在交互效应？即先消除intra-visual冗余是否会让cross-modal冗余更容易被识别（因为留下的token信息密度更高、更容易区分相关/不相关）？如果答案是肯定的，那么"先空间压缩、再跨模态筛选"的级联策略应该优于在单一阶段试图同时处理两种冗余。此外，20倍的token数量不对称也暗示了当前MLLM架构设计的一个潜在缺陷——或许我们需要从根本上重新设计视觉编码方式，使其产出的token数量与下游语义需求更匹配，而不是先盲目编码再事后压缩。

