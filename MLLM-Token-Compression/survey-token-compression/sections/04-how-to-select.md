# 4. How to Select the Desirable Token Compression Strategy

> 本章从5个关键设计维度进行对比分析，为实践者提供策略选择指南。随着token压缩方法的爆发式增长（50+方法），研究者面对的核心问题不再是"能不能压缩"，而是"在我的具体场景下，应该选择什么压缩策略"。本章正是为了回答这一问题而设计的决策路线图。

## 决策分类图 (Figure 6)

```
How to Select Strategy
├── §4.1 Temporal-Enhanced Compression for Videos
├── §4.2 Purely-Visual vs. Text-guided Compression
├── §4.3 Token Merging vs. Token Dropping
├── §4.4 Plug-in Methods vs. Re-training Methods
└── §4.5 Efficient Training vs. Efficient Inference
```

论文对每个维度分析了其技术优缺点，并基于部署约束提供了实践建议。五个维度并非正交——例如，一个视频场景下的plug-in方法可能同时涉及§4.1（时序压缩）、§4.3（merging vs. dropping的选择）和§4.4（是否需要训练）。理解这些维度间的交叉关系，是做出合理设计决策的关键。

---

## 4.1 Temporal-Enhanced Compression for Video Input

视频比静态图像多出**时序维度**，带来根本性的效率挑战。随着视频时长增加或帧采样率提高，视觉token数量爆炸式增长——一段数分钟的视频经过逐帧patch化后可轻松产生数万甚至数十万个token。虽然现有的空间压缩策略（参见§3.1）可以直接应用于视频的每一帧，但它们忽略了跨帧冗余（cross-frame redundancy）这一视频特有的巨大压缩空间。论文围绕视频场景提出三大核心挑战：

1. **Spatial-temporal interaction**: 如何联合压缩空间 $(h,w)$ 和时间 $t$ 维度，形成紧凑但富有表达力的表示
2. **Temporal structure preservation**: 压缩后如何保留时空结构信息（运动估计、时序定位、事件边界检测）
3. **Scalability to extreme lengths**: 如何扩展到包含数千至数万帧的小时级视频，而非仅处理短视频片段

> **[感受]** 这三大挑战的划分非常精准，恰好对应了视频理解任务的三个递进层次：基础的内容理解（需要空间-时序联合压缩）、精细的时序推理（需要保留时间结构）、以及极端场景的可扩展性（需要全新的架构设计）。从MLLM效率优化的角度看，视频场景的压缩潜力远大于静态图像——因为视频的时序冗余（相邻帧的重叠度通常高达90%以上）提供了巨大的压缩空间，但同时也更加危险：过度压缩可能导致关键的动态信息（如快速动作、场景切换、细微表情变化）被不可逆地丢失。一个值得深思的问题是：这三大挑战目前大多被独立解决，但理想的方法应该是一个统一框架，能够在空间-时序联合压缩的同时保留时间结构，并且自然地扩展到极长视频。目前距离这种统一方案还有相当距离。

### 4.1.1 Spatial-Temporal Compression

联合时空压缩策略可以大致分为**固定（fixed）** 和**动态（dynamic）** 两类，同时出现了结合两者优势的**混合（hybrid）** 策略。

| 类别 | 方法 | 核心思路 | 代表工作 |
|------|------|---------|---------|
| **Fixed - Pooling** | 时序维度上average pooling相邻token | 最简单直接，对相邻帧token取平均，抑制冗余但牺牲运动细节 | PLLaVA, Video-ChatGPT |
| **Fixed - Convolution** | 2D/3D卷积联合时空下采样 | 更显式地建模时空信息：VideoLLaMA2用3D STC Connector + RegStage保留局部动态；Qwen2-VL用2D卷积融合相邻帧特征；Qwen2.5-VL用3D卷积同时在空间(4x)和时间(2x)下采样 | VideoLLaMA2 (STC Connector), Qwen2-VL, Qwen2.5-VL |
| **Fixed - Query-based** | 可学习query通过attention聚合所有token | 不做pooling而是学习一组紧凑的query token（如Q-Former、Token Learner、Resampler），通过attention机制从所有视觉token中提取信息。Clapper用cross-attention捕获帧间动态，LinVT和CrossLMM利用用户query引导压缩 | Clapper, LinVT, CrossLMM |
| **Fixed - Sequential** | 按时序处理+时间戳嵌入+循环记忆 | 利用线性复杂度 $O(n)$ 的序列模型高效编码长视频。BLIP-3-Video提出Grouped Sequential Model，按时间顺序处理并按空间位置分组，每组维护独立的时序记忆+时间戳位置编码，最终聚合为16-32个video-level token。STORM用Mamba State Space Model的双向扫描捕获时空依赖 | BLIP-3-Video, STORM |
| **Dynamic - Merging** | 自适应合并跨帧相似/冗余token | 根据token相似度动态决定合并策略。TESTA、AuroraCap、DyCoke通过余弦相似度识别跨帧冗余token并合并。InTI引入轻量权重预测网络为相邻帧空间共位token生成动态合并权重。Learnable VTM为每个token分配可学习的显著性分数来决定合并比例 | TESTA, AuroraCap, DyCoke, InTI, Learnable VTM |
| **Dynamic - Pruning** | 丢弃时序低显著性/冗余token | 直接裁剪不重要的token而非合并。LongVU提出三阶段压缩pipeline：先跨帧对齐，再计算余弦相似度丢弃高度相似的token，最终实现极端空间压缩。TimeChat-Online通过feature-level冗余度量保留时序动态信息，丢弃冗余token，且证明feature-level度量优于pixel-level | LongVU, TimeChat-Online |
| **Hybrid - Global-Local** | 全局事件聚类 + 局部帧级聚合 | 平衡全局覆盖与局部细节。PruneVid、Chat-UniVi、FiLA-Video做全局事件级聚类后进行事件内聚合。LongVLM在clip内做局部token merging并结合全局语义表示。TempMe和Video-XL使用层次化merging或Visual Summarization Tokens。HiCom分组采样+指令条件压缩。Quicksviewer用Gumbel-Softmax确定信息密度并按块重采样 | LongVLM, Video-XL, TempMe, PruneVid, Chat-UniVi, HiCom |
| **Hybrid - Slow-Fast** | 高分辨率慢通路(空间细节) + 低分辨率快通路(运动) | 受动作识别启发，借鉴SlowFast网络的双流设计。SlowFast-LLaVA、LLaVA-Video、Clapper使用两条通路：慢通路低帧率但高空间分辨率保留细节，快通路高帧率但紧凑token捕获运动。Keye-VL 1.5进一步改进，动态将显著帧分配到慢通路、静态帧分配到快通路，显著提升效率 | SlowFast-LLaVA, LLaVA-Video, Clapper, Keye-VL 1.5 |
| **Hybrid - Memory-bank** | 长期记忆 + 短期记忆互补 | 用记忆机制处理超长序列。Flash-VStream引入STAR memory：Context Synopsis Memory将低分辨率特征聚类为质心以保留全局时序趋势，Detail Augmentation Memory为关键帧选择性保留高分辨率token。MovieChat用滑动窗口+双记忆（短期捕获当前窗口细节，长期聚合历史语义），容量超限时周期性merge。VidCompress用memory-augmented cross-clip attention增强记忆 | MovieChat, VidCompress, Flash-VStream |

> **[感受]** 这个分类从fixed到dynamic再到hybrid，清晰地展示了视频token压缩方法的演化脉络。Fixed方法（pooling/conv/query/sequential）胜在简单可控、训练稳定，但"一刀切"的压缩率无法适应视频内容的动态变化——一段包含快速动作和静止场景的视频，显然需要对不同片段施加不同的压缩强度。Dynamic方法（merging/pruning）解决了这一问题，但引入了额外的相似度计算开销和不稳定的压缩率。Hybrid方法是目前最有前景的方向，但代价是设计复杂度显著增加。从研究选题的角度看，我认为最值得关注的是两个方向：(1) **Sequential Model**（BLIP-3-Video、STORM）——它们用 $O(n)$ 复杂度处理长序列，特别是Mamba等State Space Model的引入为视频token压缩提供了新的计算范式，不再受限于attention的二次复杂度；(2) **Slow-Fast Pathway**——Keye-VL 1.5的动态帧分配思想值得深入挖掘，因为它本质上是在做帧级别的自适应资源分配，这与人类观看视频时"重要时刻仔细看、无聊片段扫一眼"的注意力分配模式一致。一个尚未被充分探索的方向是将这些方法与视频编码器（如视频codec中的I/P/B帧结构）的先验知识结合——视频编码器已经在压缩层面做了大量关于帧间冗余的分析，这些信息可以被直接复用以指导token级别的压缩决策。

### 4.1.2 Temporal Structure Preservation

token合并/剪枝在减少序列长度的同时，可能**模糊甚至丢失时空位置信息**。当多个帧的token被合并为一个代表性embedding，或者某些帧的token被直接丢弃时，"这个事件发生在视频的第几秒"这一关键信息可能消失。这直接影响需要精确时间感知的任务，如时序定位（temporal grounding）、动作估计（motion estimation）、以及时序问答（temporally grounded QA）。

论文归纳了三种保留时间信息的策略：

| 方法 | 策略 | 技术细节 | 代表工作 |
|------|------|---------|---------|
| **Temporal Positional Embeddings** | 为视觉token增加时间位置信息 | 最直接的方法。BLIP-3-Video在按时序分组处理时维护时间戳位置编码。TimeChat-Online在剪枝后保留原始的Video-RoPE位置编码，使保留下来的token仍携带相对于原始视频的时空位置。PVC使用绝对位置嵌入 $t = [0, 1, ..., T]$ 或相对位置嵌入 $t = [0, \frac{1}{T-1}, ..., \frac{T-2}{T-1}, 1]$ | BLIP-3-Video, TimeChat-Online, PVC |
| **Temporal Encoding Modules** | 专用时序编码组件 | 引入独立的时序建模模块。STORM用Mamba State Space Model的MambaMixer层，通过双向扫描同时捕获空间和时序依赖。PVC采用渐进式编码策略：每帧token先独立编码，再逐步与前帧信息融合，确保未被前帧覆盖的新信息被补充进来，维护累积时序上下文 | STORM (MambaMixer), PVC (渐进式编码) |
| **Special Timestamp Tokens** | 插入显式时间戳token到视觉序列 | 在token序列中显式嵌入时间标记。Video-XL-2在视觉token序列中交错插入时间戳表示作为独立token，增强模型的时序感知能力。Qwen3-VL更进一步，采用**文本格式的时间编码策略**，每个视频patch被前缀一个格式化的时间戳文本字符串（如`<3.0 seconds>`），完全脱离传统的Video-RoPE，实现基于时间戳的事件定位 | Video-XL-2, Qwen3-VL |

> **[感受]** 时间结构保留是视频token压缩中一个容易被忽视但极其重要的问题。很多压缩方法在VQA等粗粒度任务上表现良好，但一旦面对需要精确时序推理的任务（如"视频第15秒发生了什么"、"A事件和B事件哪个先发生"），性能就急剧下降——因为压缩过程中时间信息被隐式地"平滑"掉了。三种策略的演进体现了对这一问题认识的深化：从简单地给token附加位置嵌入（可能在merging后失去意义），到用专用模块显式建模时序结构，再到Qwen3-VL将时间戳转化为文本token这一颇具创意的做法。Qwen3-VL的文本时间戳方案尤其值得关注——它将时间信息从连续的位置嵌入空间转移到了LLM最擅长处理的离散文本空间，巧妙地利用了LLM对文本的强大理解能力来弥补位置嵌入在压缩后的信息损失。从效率优化的研究角度看，这里暴露了一个根本性的tension：**更强的压缩必然导致更多的位置信息损失**。目前的三种方法都是在"事后补救"——先压缩再想办法保留时间信息。更根本的解决思路可能是在压缩机制的设计中就将时序保留作为约束条件内化进去，例如在merging的相似度计算中加入时序位置的惩罚项，或者在pruning的token重要性评分中给时序边界附近的token以更高权重。

### 4.1.3 Extreme-Long Video Compression

小时级视频（数千至数万帧）的理解对MLLM提出了极端的效率挑战——不仅token数量可能达到数十万级别，还面临GPU显存的物理限制。这需要在输入采样、编码压缩、显存管理和推理加速等多个维度进行专用设计。

| 方向 | 代表工作 | 关键特点 |
|------|---------|---------|
| **Memory-bank** | MovieChat (sliding window + dual memory), Flash-VStream (STAR memory) | MovieChat开创性地将滑动窗口与双重记忆机制结合：短期记忆捕获当前窗口细节，长期记忆聚合历史语义，实现在24GB GPU上处理10,000帧。Flash-VStream进一步引入更精密的flash memory架构，实现实时响应用户查询 |
| **Video-XL系列** | Video-XL -> Video-XL-Pro -> Video-XL-2 | 展示了清晰的迭代演进：Video-XL引入动态区间分割+Visual Summarization Tokens (VSTs)，2048帧/16x-32x压缩/95%准确率。Video-XL-Pro通过ReCoT框架引入可重构能力：动态token捕获运动模式、语义引导masking聚焦密集区域、query-aware选择剪枝低相关token，8000帧/接近99%准确率。Video-XL-2将优化重心从训练时转向推理时，通过KV cache稀疏化在单GPU上处理10,000帧 |
| **Query-aware** | LinVT, Long-VMNet (固定5880 token memory bank) | 在视频QA场景中，只有与用户问题相关的帧子集才有信息价值。LinVT通过时空显著性分析识别候选区域，再根据文本query进行过滤和聚合。Long-VMNet使用固定大小memory bank（5880 tokens），一次视频扫描后可跨查询复用，<1GB内存支持10小时视频 |
| **Keyframe-based** | ReTaKe (帧间距离检测关键帧 -> 非关键帧KV剪枝) | 通过帧间距离检测关键帧峰值并标记为pivot，对非pivot帧仅保留KV cache中的高attention token。利用LLM先验知识实现即插即用，可处理8x更长序列 |
| **Hybrid architecture** | TimeViper (Mamba-Transformer混合) | 采用Mamba-Transformer混合架构，结合Mamba的线性复杂度处理长序列全局建模与Transformer的精确attention处理局部细节，实现10,000帧处理能力 |

**Summary**: 极长视频理解需要多维度协同: 1) 自适应关键帧采样减少输入冗余 2) 多模块协作编码实现渐进式压缩 3) query-aware策略根据用户意图动态调整 4) KV-cache稀疏化提升推理效率。这一演进方向体现了从孤立的单点优化向系统级、任务感知、端到端设计的转变。

> **[感受]** 极长视频理解是token压缩领域最具挑战性也最具实用价值的前沿方向。Video-XL系列的三代演进（Video-XL -> Video-XL-Pro -> Video-XL-2）非常具有代表性，展示了从"固定压缩"到"语义引导压缩"再到"推理时KV稀疏化"的技术路线升级。值得注意的是，几乎所有成功处理10,000+帧的方法都采用了某种形式的**层次化设计**——先做粗粒度的帧级筛选（关键帧采样），再做细粒度的token级压缩——这与人类理解长视频的认知过程高度一致：先快速浏览把握整体结构，再对关键片段仔细分析。从效率优化研究的角度看，我认为这个方向有两个关键的未解决问题：(1) **流式处理能力**——目前大多数方法仍然假设视频是完整可用的（offline processing），但实际应用（如视频监控、直播分析）需要在线处理能力，这意味着memory bank的更新策略需要更加精巧；(2) **压缩率与信息损失的理论保证**——当我们将10小时视频压缩到不到6000个token时，信息损失到底有多大？目前缺乏理论工具来回答这个问题，都是靠benchmark上的经验评估。此外，Long-VMNet的"一次扫描、跨查询复用"设计非常实用——对于视频搜索等需要对同一视频回答多个问题的场景，避免了重复编码的巨大开销，这种amortized设计思路值得更多研究者关注。

---

## 4.2 Purely-Visual vs. Text-guided Compression

token压缩方法根据是否利用文本信息可分为两大分支：**purely-visual compression**（基于视觉内在冗余，不依赖文本query）和**text-guided compression**（利用文本语义指导视觉token的选择和保留）。这一区分对应了§1.2中冗余分类的两个维度——intra-visual冗余（空间/时序冗余）和cross-modal冗余（跨模态对齐冗余）。

### 对比表 (Table 3)

| 维度 | Purely-Visual | Text-guided |
|------|--------------|-------------|
| **方法** | 基于视觉内在冗余（重复对象、均匀背景、语义等价区域）保留信息token，不依赖任何文本输入 | 基于文本语义（用户指令或query）选择与之对齐的视觉token，利用text-to-vision attention或相似度作为选择信号 |
| **适用场景** | 多轮对话（压缩结果可跨轮次复用）、流式视频（无需等待query）、视觉字幕生成、易部署场景 | 单轮对话、长视频QA（需要极高压缩率）、高压缩率场景、视觉定位（visual grounding） |
| **优点** | 文本无关，一次压缩可跨query复用；低延迟；对视觉丰富场景通用性强；直接在LLM之前减少token，避免LLM浅层的计算和内存开销 | 高压缩率下仍保持高准确率；任务相关保留，丢弃与query无关的token；对VQA、grounding等对齐敏感的任务效果好 |
| **缺点** | 可能保留与query无关的token（如用户问"猫的颜色"但保留了大量背景建筑token）；在极高压缩率下精度下降更明显 | 需要针对每个新query重新编码/选择历史视觉token；多轮对话效率低（每轮都要重新压缩）；流式场景不适用 |
| **代表工作** | DeCo, VisionZip, DART, HoloV, TimeChat-Online | FastV, SparseVLM, Q-Former, QueCC, PyramidDrop, LLaVA-Mini |

> **[感受]** Purely-visual和text-guided的对比，本质上是**效率与精度的经典trade-off在token压缩领域的具体体现**。Purely-visual方法的最大优势在于"一次压缩，多次复用"——这在生产环境中极其重要：想象一个视频理解API，同一段视频可能被不同用户用不同的query查询，如果每次都要重新做text-guided压缩，成本将非常高昂。Text-guided方法的优势则在于"精准投递"——只保留query需要的视觉信息，能在极高压缩率下仍保持不错的性能。但论文中未充分讨论的一个问题是：text-guided压缩是否会引入**confirmation bias**——模型可能过度关注与query表面匹配的token，而忽略了回答问题所需的上下文信息（例如，问"这个人在做什么"时，不仅需要看人本身，还需要看周围环境才能准确判断动作的语义）。VisionZip的发现——"更紧凑的视觉token反而能产生更好的视觉表示"——值得深思，它暗示当前MLLM中存在大量的信息冗余，purely-visual压缩在某些情况下甚至可能是一种隐式的正则化。

### Takeaway

> **两者是互补而非竞争的**。论文建议的实践设计范式：**先purely-visual获得紧凑视觉表示**（在Vision Encoder或Projector阶段完成，去除intra-visual冗余）→ **再text-guided在LLM内做query相关精炼**（在LLM内部利用text-visual交互信号进一步剪枝cross-modal冗余token）。这种两阶段设计同时获得了purely-visual的低延迟和text-guided的高精度。

> **[感受]** 这个"先visual后text-guided"的两阶段设计建议是Section 4中最有工程指导价值的结论之一。它巧妙地利用了MLLM pipeline的自然分阶段特性——在token进入LLM之前先做一轮无差别的"粗筛"（去掉明显的空间/时序冗余），进入LLM之后再根据query做"精选"（去掉与当前问题无关的token）。这本质上是一种**coarse-to-fine**的压缩范式。但我认为，更有前景的方向是在purely-visual阶段就引入**轻量级的text先验**——不是完整的query理解，而是诸如"这是一个OCR任务"还是"这是一个粗粒度场景理解任务"这样的task-type信号，从而在早期就能做出更intelligent的压缩决策，同时不引入完整text-guided压缩的重编码开销。FCoT-VL的工作已经初步探索了这个方向，值得关注。此外，在多轮对话场景中，如何在第一轮的text-guided压缩结果基础上增量更新（而非每轮从头压缩），是一个有价值但尚未被充分研究的问题。

---

## 4.3 Token Merging vs. Token Dropping

Token merging和token dropping（也称pruning）是token压缩范式中的两个基本操作。它们的核心区别在于压缩的"方式"：**merging是soft策略**，将不太重要的token聚合到代表性embedding中；**dropping是hard策略**，直接丢弃被判定为不重要或与任务无关的token。一个自然的问题是：这两种操作应该被同等对待吗？

### 对比表 (Table 4)

| 维度 | Token Merging | Token Dropping |
|------|--------------|----------------|
| **本质** | **Soft**策略，将视觉上冗余的token聚合为紧凑的代表性嵌入，通过加权平均等方式融合信息 | **Hard**策略，直接丢弃被判定为不重要、信息量低或与任务无关的token |
| **优点** | (i) 保留整体语义和细粒度信息（被合并token的信息没有完全丢失）；(ii) 适合压缩低层视觉特征中的空间冗余；(iii) 在处理密集/时序冗余的视觉输入时效果好 | (i) 保留稀疏但显著的语义（被保留的token未经任何修改，信息纯净）；(ii) 适合压缩高层视觉特征；(iii) 实现简单、计算开销低 |
| **缺点** | 可能模糊空间/时序局部性（合并操作"平滑"了位置信息）；当合并跨越语义边界时可能引入噪声 | 可能丢失上下文中的细微线索；在剪枝过程中被移除的微妙上下文信息不可恢复 |
| **代表** | ToMe, TESTA, HoliTom, MustDrop | VisPruner, MADTP, DivPrune, DART, FlexSelect, CDPruner, DTD |

> **[感受]** Merging vs. dropping的对比揭示了token压缩中一个深层的信息论问题：**信息到底应该被"浓缩"还是"筛选"？** Merging对应"浓缩"——将N个token的信息压缩到M个token中，每个保留的token承载更多信息但可能引入模糊性。Dropping对应"筛选"——保留最重要的M个token不做任何修改，信息纯净但总信息量减少。从信息论角度看，merging的信息保留上界更高（理论上可以无损），但实际中加权平均等简单融合方式远未达到这个上界；dropping的信息损失是确定的（丢掉的token的信息完全消失），但保留下来的信息是精确的。我认为这两种策略的本质差异更类似于有损压缩中的"量化"（merging，降低精度但保留全部token）和"采样"（dropping，保持精度但减少token数）。在实际应用中，选择哪种策略应该取决于下游任务对信息的需求模式：需要全局语义理解的任务（如image captioning）更适合merging，需要局部精确信息的任务（如OCR、grounding）更适合dropping。

### 关键发现

论文汇总了近期研究中关于merging vs. dropping的若干重要发现：

- **LLMC+ 分析**: 对空间冗余场景进行了定量分析，发现**drop-based策略在Vision Encoder和LLM中均优于merge-based**策略。这一结果有些反直觉——直觉上merging保留了更多信息，为什么反而不如dropping？可能的解释是：merging引入的"模糊"比dropping引入的"遗漏"对LLM的影响更大
- **DART / FEATHER 的位置偏差发现**: attention scores作为token重要性的代理指标存在**系统性的位置偏差**——偏向图像右下角或序列后部的token。这意味着基于attention score的pruning/merging决策可能是有偏的，与token的实际语义重要性不完全一致
- **HoloV 的过度聚焦发现**: MLLMs存在过度关注"highlighted tokens"而忽略整体上下文（holistic context）的倾向。这暗示aggressive dropping可能加剧这一偏差——在压缩后的稀疏token集上，模型更容易"只见树木不见森林"

此外，基于attention score的token选择可能与**Flash Attention不兼容**（因为Flash Attention不显式存储attention矩阵），这在工程实现中是一个不可忽视的问题。近期越来越多的工作转向**基于feature-level相似度的选择策略**，用特征级冗余度来替代attention magnitude作为选择信号，实现更稳定、更context-aware的压缩。

> **[感受]** 这三个关键发现对于实际设计压缩方法具有直接指导意义。LLMC+的发现（dropping优于merging处理空间冗余）表面上否定了merging，但我认为需要更细致地解读：这个结论可能高度依赖于具体的merging实现方式（简单平均 vs. 加权融合 vs. 学习式聚合），且可能在不同类型的冗余（空间冗余 vs. 时序冗余 vs. 语义冗余）上有不同结论。DART/FEATHER发现的**attention位置偏差**是一个很有价值的negative result——它提醒我们，在MLLM中直接使用attention score作为token重要性指标是不可靠的，这与NLP领域中attention ≠ explanation的经典发现一脉相承。这一发现也解释了为什么一些基于random selection的baseline有时能与attention-based方法持平甚至更优。HoloV的"过度聚焦"发现则暗示了一个更深层的问题：**token压缩可能会放大MLLM本身的注意力偏差**，形成一种恶性循环——压缩依赖attention来选择token，而attention本身已经有偏差，压缩后的稀疏输入进一步强化了这种偏差。如何打破这个循环是一个重要的开放问题。

### 趋势

> 趋向**自适应混合策略**：不再固定使用merging或dropping，而是根据模态特征和冗余类型动态切换soft聚合和hard剪枝。Merging为密集/时序冗余的视觉输入提供平滑聚合，dropping在仅需稀疏高层语义时更为有效。未来的框架可能受益于自适应混合设计，根据模态特性和冗余类型动态切换soft聚合与hard剪枝。

> **[感受]** 自适应混合策略是一个显而易见的发展方向，但实现起来面临一个关键问题：**如何高效地判断当前输入应该用merging还是dropping？** 这个判断本身就需要额外的计算开销。一种可能的思路是利用输入的全局统计量（如token embedding的方差、attention entropy等）来做粗粒度的策略选择——方差低（信息均匀分布）时用merging更合适，方差高（信息高度集中在少数token）时dropping更有效。另一种更激进的思路是完全放弃这种二元选择，设计一种**连续的压缩操作**——用一个可学习的参数控制每个token被"保留多少信息"，从完全保留（不压缩）到部分保留（merging的效果）到完全丢弃（dropping的效果），形成一个统一的压缩框架。这也与MustDrop的多阶段设计思想一致：在不同的pipeline阶段对同一批token施加不同的压缩操作。

---

## 4.4 Plug-in Methods vs. Re-training Methods

从模型适配的角度，token压缩方法可以分为两大阵营：**plug-in方法**（无需额外训练，直接集成到冻结的预训练模型中）和**re-training方法**（引入可学习模块，需要额外的微调或端到端训练）。这一区分直接关系到方法的**部署成本、跨模型迁移性和性能上限**。

### 对比表 (Table 5)

| 维度 | Plug-in | Re-training |
|------|---------|-------------|
| **方法** | 无参数或极少参数的策略，可直接集成到现有frozen模型中，无需任何额外训练 | 引入可学习的压缩模块，需要额外训练（fine-tuning或end-to-end优化）来获得压缩能力 |
| **特点** | (i) Training-free，无参数或极少参数 (ii) 轻量级，易于部署 (iii) 跨模型迁移性好——同一策略可应用于不同MLLM (iv) 在细粒度任务上性能上限受限 | (i) 性能上限更高——能学到task-adaptive的压缩策略 (ii) 需要额外训练成本和数据 (iii) 跨模型迁移性差——训练的压缩模块可能不适用于其他MLLM (iv) 工程复杂度高 |
| **代表** | FastV, SparseVLM, PyramidDrop, MustDrop | Honeybee, DeCo, TokenPacker, HiCo |

> **[感受]** Plug-in vs. re-training的选择在实践中往往是由**部署约束**而非**学术追求**决定的。如果你是在一个已经部署的MLLM服务上做效率优化（比如给公司已有的InternVL API降低推理成本），plug-in方法几乎是唯一选择——你不可能为了加一个压缩模块就重新训练整个模型。如果你是在从头设计一个MLLM（比如训练自己的视觉语言模型），那么re-training方法的性能优势不容忽视。FCoT-VL的实验发现也很有意思——当前的training-free token压缩方法在高分辨率视觉理解和复杂文本推理任务上仍然存在显著的性能下降，这说明plug-in方法的"免费午餐"是有边界的。从研究价值的角度看，我认为plug-in方法更有学术意义——因为它迫使你在不能依赖"让模型自己学"的情况下，真正理解token冗余的本质并设计出精巧的启发式或理论驱动的压缩策略。

### Plug-in的四种策略

论文将plug-in方法细分为四种具体的技术路线：

1. **参数无关空间变换 (Parameter-free Spatial Transformations)**: 最简单直接的plug-in策略。包括global/adaptive pooling（TC-LLaVA, PLLaVA, DeCo, AVG-LLaVA），通过空间池化操作将token数量降低到预定义的大小。完全无参数，适用于任何backbone，但压缩率固定且对内容不敏感

2. **像素重排 (Pixel Rearrangement)**: 通过pixel shuffle和space-to-depth变换重新组织token（NVLM, InternVL 1.5）。将空间维度"折叠"到通道维度，减少token数量但增加每个token的特征维度，从而在不丢失信息的前提下减少序列长度。这是一种准无损的压缩方式

3. **相似度压缩 (Similarity-based Token Compression)**: 基于token间相似度进行分组合并或多样性选择。DynTok动态分组视频token并进行组内merging；LLaVA-Scissor利用Semantic Connected Components保留语义区域同时减少冗余；DivPrune通过最大化多样性选择信息量最大的token子集

4. **推理时KV-cache压缩 (Inference-time KV Cache Compression)**: 在LLM解码阶段对KV-cache进行动态剪枝。DyCoke用attention引导的方式裁剪KV cache；MustDrop采用output-aware KV policy，基于输出token的需求决定哪些KV entry可以被安全移除。这类方法不减少prefilling阶段的token数，但显著加速解码阶段

> **[感受]** 这四种plug-in策略从"粗暴"到"精巧"形成了一个清晰的谱系。策略1（空间变换）和策略2（像素重排）本质上是固定的几何操作，不考虑内容语义，胜在简单可靠但压缩率-性能trade-off不够好。策略3（相似度压缩）引入了内容感知能力，但需要额外计算相似度矩阵的开销。策略4（KV-cache压缩）最精巧，因为它在LLM内部利用了最丰富的语义信号（attention pattern），但只加速解码而不减少prefilling成本。从系统设计的角度看，这四种策略其实可以且应该被**组合使用**——在不同的pipeline阶段各用一种：先用策略1或2在projector处做粗粒度的空间降采样（减少进入LLM的token数），再用策略3在LLM浅层做内容感知的token选择，最后用策略4在解码阶段做KV-cache稀疏化。这种多阶段组合是MustDrop的核心设计理念，也代表了plug-in方法的发展方向。一个值得探索的问题是：这四种策略各自的最优操作点在哪里？即在pipeline的哪个位置、用多大的压缩率，能获得整体最优的效率-性能trade-off？

### 趋势

> **混合策略日益受到关注**：lightweight plug-in做早期空间降采样 -> re-trained cross-attention/query-based模块做语义精炼 -> KV-cache剪枝加速解码。这种渐进式集成（exemplified by MustDrop的多阶段设计），反映了将plug-in方法的部署灵活性与re-training方法的任务适应性相结合的趋势。

> **[感受]** 这个"plug-in + re-training"的混合趋势非常合理，但也引出了一个工程层面的挑战：**如何在不破坏plug-in方法的即插即用特性的前提下引入re-training组件？** 理想的设计应该是模块化的——plug-in部分可以独立工作（提供一个decent的baseline），re-training组件作为可选的"增强包"在有训练资源时进一步提升性能。这种设计理念类似于LoRA对于LLM fine-tuning的意义——基础模型不变，附加的可学习模块是可插拔的。从更宏观的角度看，plug-in vs. re-training的界限正在模糊化：一些方法虽然需要"训练"，但训练成本极低（如仅训练几个learnable query token），实际上介于plug-in和re-training之间。未来可能出现的一种范式是**meta-learning for compression**——在大规模数据上预训练一个通用的压缩策略生成器，对于新的MLLM只需少量样本就能适配出task-specific的压缩参数。

---

## 4.5 Efficient Training vs. Efficient Inference

token压缩的优化目标可以分为两个不同的阶段：**Efficient Training**（减少预训练/SFT阶段的token数以降低训练成本）和**Efficient Inference**（减少推理阶段的token数以降低延迟和显存消耗）。虽然两者都在"减少token"，但它们面临的约束、验证成本和采用现状截然不同。

### 对比表 (Table 6)

| 维度 | Efficient Training | Efficient Inference |
|------|-------------------|-------------------|
| **目标** | 减少预训练/SFT的token数 -> 降低训练成本（训练MLLM需要处理数十亿到数万亿token，成本巨大） | 在prefilling/decoding阶段减少token -> 降低推理延迟和显存消耗 |
| **方法论特点** | 方法相对简单（主流MLLM倾向于使用成熟的、经过验证的压缩策略）；验证成本极高（每次修改压缩策略都需要重新训练模型） | 方法多样化（是该领域研究最活跃的方向）；验证成本低（只需在现有模型上测试） |
| **研究现状** | 研究较少，方法有限 | 研究极多，是token压缩领域的主战场 |
| **代表** | Flamingo, Q-Former, LLaVA-OneVision, Qwen2.5-VL, InternVL3.5 | FastV, SparseVLM, PyramidDrop, VisionZip, SparseMM |

> **[感受]** Training vs. inference的不对称研究现状本身就是一个有趣的现象：推理侧方法百花齐放，训练侧方法却寥寥无几。这种不对称的根本原因不是技术上的，而是**经济上的**——验证一个推理侧压缩方法只需要在现有模型上跑几个benchmark，几个GPU小时就够了；验证一个训练侧压缩方法则需要从头训练一个完整的MLLM，可能需要数千GPU小时，这使得绝大多数研究组无法承担训练侧方法的探索成本。这种不对称也导致了一个令人遗憾的研究gap：训练侧可能存在巨大的效率提升空间，但因为验证门槛太高而很少被探索。对于有充足计算资源的大型实验室（如Google、Meta、字节等），训练侧的token压缩创新可能是一个"low-hanging fruit"——因为竞争者少，而潜在收益大。

### 主流MLLM的训练压缩策略 (Table 7)

纵观主流MLLM的发展，它们在训练阶段采用的token压缩策略呈现出清晰的演化趋势：

| 年份 | 模型 | 训练压缩策略 | 特点 |
|------|------|------------|------|
| 2022 | Flamingo | GATED XATTN-DENSE | 早期探索，通过门控交叉注意力控制视觉信息流入LLM |
| 2023 | BLIP-2, mPLUG-Owl, Qwen-VL, Video-LLaMA, MiniGPT-4 | Q-Former及变体 | 用可学习query token通过cross-attention压缩视觉表示，成为2023年的主流范式 |
| 2024 | Video-ChatGPT, PLLaVA, LongVLM, VideoLLaMA 2 | Temporal + Spatial Pooling | 视频场景开始普及，简单的时空池化成为标配 |
| 2024 | LLaVA-OneVision | Bilinear Interpolation | 用双线性插值减少每帧token数 |
| 2024 | LLaVA-Video | Average Spatial Pooling | 更简单的平均空间池化 |
| 2025 | InternVL系列, Qwen2系列 | Pixel Shuffle | 通过像素重排将空间维度折叠到通道维度，当前最主流的训练压缩方案 |
| 2025 | Seed1.5-VL | Average Pooling | 回归最简单的平均池化 |

> **[感受]** 这张表揭示了一个非常有趣的趋势：主流MLLM的训练压缩策略从复杂走向简单。2023年Q-Former是一个相当复杂的设计（引入额外的cross-attention模块和可学习query），但到了2025年，最强的模型反而采用了pixel shuffle或average pooling这样极其简单的方案。这说明在训练侧，**简单可靠比复杂精巧更重要**——因为训练是一次性成本，且训练过程中模型可以学会适应压缩带来的信息损失。另一个值得注意的现象是，Q-Former范式在2023年之后逐渐被抛弃，取而代之的是更"暴力"的压缩方式（pooling/interpolation/pixel shuffle）。这可能是因为Q-Former虽然理论上能做task-adaptive压缩，但在实际训练中引入了额外的优化困难（query token的学习可能不稳定），且与简单pooling方法相比，性能提升并不显著。此外，LLaVolta提出的阶段式训练（先在多token上训练再逐步减少）是一个有前景但尚未被广泛采用的思路，它本质上是一种课程学习（curriculum learning），值得更深入的研究。

### 为什么训练压缩方法未被主流MLLM广泛采用？

论文提出一个关键观察：大量推理侧的token压缩方法原则上也可以用于训练（因为LLM的prefilling和训练都涉及对序列的单次前向传播），但实际上**几乎没有主流MLLM采用这些更先进的压缩方法来加速训练**。原因有三：

1. **兼容性问题 (Compatibility Issues)**: 很多prefilling加速方法（尤其是基于attention score的token选择）与**Flash Attention不兼容**——Flash Attention不显式计算和存储完整的attention矩阵，而这些方法依赖attention score来判断token重要性。由于Flash Attention是现代MLLM训练的标配（对训练效率至关重要），这种不兼容性直接阻碍了这些方法在训练中的应用

2. **验证成本 (Validation Cost)**: 采用新的压缩策略需要在训练中验证其效果，而**训练验证比推理验证昂贵得多**——可能需要完整地训练一个模型才能判断新策略是否有效。这种高昂的试错成本使得研究者倾向于保守选择，只有当新方法被证明是breakthrough时才会被采用

3. **归纳偏置 (Inductive Bias)**: 现有的推理侧压缩方法往往是基于特定任务/benchmark的观察来设计的（如"attention score低的token不重要"），引入了较强的归纳偏置。这些偏置在特定的推理场景下可能成立，但**在训练中可能导致分布外(OOD)场景的性能退化**——因为训练数据覆盖的场景远比推理benchmark广泛。对于追求通用能力的MLLM来说，任何可能导致特定能力退化的做法都是不可接受的

> **[感受]** 这三个原因的分析切中要害，但我认为还有一个未被提及的重要原因：**生态惯性和工程复杂度**。主流MLLM的训练pipeline已经非常复杂（分布式训练、混合精度、数据并行、梯度检查点等），引入一个新的token压缩模块意味着需要与所有这些工程组件进行兼容性测试和性能调优，这个工程成本远超学术论文中呈现的难度。Flash Attention的兼容性问题尤其具有杀伤力——它不仅是一个技术问题，更是一个生态问题：整个训练基础设施（包括DeepSpeed、Megatron、FSDP等）都围绕Flash Attention构建，任何与之不兼容的方法都面临巨大的集成障碍。从研究策略的角度看，这暗示了一个高价值的研究方向：**设计与Flash Attention原生兼容的token压缩方法**——不依赖显式的attention score，而是通过feature-level的相似度或可学习的选择机制来做token压缩。此外，归纳偏置的问题也值得深思：推理侧方法在benchmark上的"成功"可能只是一种假象——因为benchmark本身的分布与训练数据的分布不同。一个在TextVQA上表现很好的attention-based pruning策略，可能会在训练中破坏模型在general image captioning上的能力。这也是为什么主流MLLM宁可用"笨"但安全的pooling/pixel shuffle，也不敢用"聪明"但有风险的attention-based pruning。

---

## 章节总结

Section 4提供的五个维度构成了一个**多维度决策空间**：面对具体的部署场景，研究者需要同时在这五个维度上做出选择。例如，一个"长视频流式问答"的场景可能需要：§4.1中的memory-bank式时序压缩 + §4.2中的先purely-visual后text-guided的两阶段策略 + §4.3中的以merging为主的soft压缩 + §4.4中的plug-in方法（因为需要跨不同LLM部署） + §4.5中聚焦efficient inference。这些维度之间的交叉组合效应——某些组合是否存在协同增益或相互冲突——是一个值得系统研究但目前缺乏实验支持的重要问题。
