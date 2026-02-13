# Towards Efficient Multimodal Large Language Models: A Survey on Token Compression

> **论文信息**
> - **标题**: Towards Efficient Multimodal Large Language Models: A Survey on Token Compression
> - **作者**: Linli Yao*, Long Xing*, Yang Shi* 等 (北京大学, 中科大, 南洋理工, 中科院, 港大, 微软, 阿里云, 国防科大, 快手)
> - **发表**: Journal of LaTeX Class Files, November 2025
> - **资源**: [GitHub](https://github.com/yaolinli/MLLM-Token-Compression)

---

## 目录导航

| 编号 | 章节 | 笔记文件 | 状态 |
|------|------|--------|------|
| 1 | Introduction | [01-introduction.md](sections/01-introduction.md) | ✅ |
| 2 | Preliminaries | [02-preliminaries.md](sections/02-preliminaries.md) | ✅ |
| 3 | Where to Compress Tokens in MLLMs | [03-where-to-compress.md](sections/03-where-to-compress.md) | ✅ |
| 4 | How to Select the Desirable Strategy | [04-how-to-select.md](sections/04-how-to-select.md) | ✅ |
| 5 | Benchmarks and Metrics | [05-benchmarks-metrics.md](sections/05-benchmarks-metrics.md) | ✅ |
| 6 | Application Scenarios | [06-applications.md](sections/06-applications.md) | ✅ |
| 7 | Open Challenges and Future Work | [07-challenges-future.md](sections/07-challenges-future.md) | ✅ |
| 8 | Conclusion | [08-conclusion.md](sections/08-conclusion.md) | ✅ |

---

## 一句话总结

本survey按**压缩位置**（Vision Encoder / Projector / LLM / Hybrid）和**压缩机制**（pruning / merging / fusion / query-based）两个维度，系统梳理了MLLM视觉token压缩领域的50+代表性工作，并提供了实用的策略选择路线图。

## 核心分类体系 (Quick Reference)

```
Token Compression in MLLMs
├── Vision Encoder (§3.1)
│   ├── Inside-Encoder: Token Dropping / Merging / Multi-Scale
│   └── Outside-Encoder: Purely-Vision / Text-guided
├── Projector (§3.2)
│   ├── Transformation-Based: Pooling / Pixel Shuffle / Convolution
│   ├── Query-Based: Q-Former / Cross-Attention
│   └── Importance-Driven: Similarity / Saliency / Novel Metrics
├── LLM (§3.3)
│   ├── Prefilling: Importance / Learnable / Merging / Fusion
│   └── Decoding: KV-cache Compression
└── Hybrid (§3.4)
    ├── Collaborative Compression
    └── Progressive Compression
```

## 策略选择决策树 (Quick Reference)

```
How to Select Strategy (§4)
├── §4.1 视频时序增强压缩 (Fixed / Dynamic / Hybrid)
├── §4.2 Purely-Visual vs Text-guided (互补，先视觉后文本)
├── §4.3 Token Merging vs Dropping (soft聚合 vs hard丢弃)
├── §4.4 Plug-in vs Re-training (易部署 vs 高性能)
└── §4.5 Efficient Training vs Efficient Inference
```

---

## 个人思考与笔记

### 与我的研究的关联

1. **压缩位置选择与MLLM架构设计的耦合关系**：本survey的核心分类维度——按架构位置（Encoder/Projector/LLM）组织压缩方法——直接对应了MLLM效率优化中的第一个设计决策。对于正在设计高效MLLM推理系统的研究者来说，这意味着压缩策略不应独立于架构设计，而应作为架构的"原生组件"来考虑。例如，如果采用了轻量级Projector（如简单MLP），则Projector处的压缩空间有限，压力需要转移到Encoder或LLM；而如果采用了Q-Former类Projector，则Projector本身就已完成了一次压缩。

2. **Training-free方法作为推理加速的"即插即用"工具**：FastV、PyramidDrop等training-free方法可以直接应用于任何现有MLLM而无需重新训练。这对于需要在多个基座模型上快速验证效率优化策略的研究流程非常有价值——可以先用plug-in方法建立baseline，确认压缩可行后再投入re-training以获取更好的performance。

3. **KV-cache压缩与视觉token压缩的交叉地带**：Survey中讨论的LLM decoding阶段的KV-cache压缩（§3.3.2）与NLP领域的KV-cache优化（如GQA、MQA、PagedAttention）存在天然的交叉。MLLM场景的特殊性在于：视觉token通常占据KV-cache的大部分空间（因为视觉token数量远多于文本token），因此选择性地压缩视觉token的KV-cache可能比通用的KV-cache压缩策略更高效。

4. **视频理解场景的极端效率需求**：Survey中关于极长视频压缩（§4.1.3）的讨论直接关联到流式视频理解和具身AI等高影响力应用。这些场景下，token数量可达数十万级别，仅靠token压缩可能不够——可能需要与Mamba等线性复杂度架构、Ring Attention等分布式注意力技术协同使用。

5. **自适应压缩与MLLM的"元认知"能力**：Survey指出的content-aware和task-aware自适应压缩方向（§7.2）与MLLM的"元认知"能力建设密切相关——模型需要具备"判断当前输入和任务需要多少视觉信息"的能力。这种能力本身可能需要在预训练阶段就开始培养，而非仅作为推理时的后处理策略。

### 值得深入阅读的论文

1. **FastV** [Chen et al.] — LLM内部基于attention score的training-free token pruning。选读理由：作为最具影响力的plug-in方法之一，FastV的核心思想极为简洁（利用LLM浅层的attention分布识别不重要的视觉token并在后续层中跳过），实现成本低、泛化性好，是理解"LLM内部压缩"范式的最佳入口。同时，它揭示了一个重要的实验发现：LLM浅层的attention已经足够区分重要和不重要的视觉token。

2. **VisionZip** [提出基于信息论的视觉token压缩] — 选读理由：作为少数尝试从信息论角度分析token冗余的工作，VisionZip为"理论驱动的压缩"提供了一个有价值的起点。理解其信息论分析框架有助于思考如何为token压缩建立更严格的理论基础（对应§7.1的挑战）。

3. **ToMe (Token Merging)** [Bolya et al., 2023] — ViT中的二分匹配token合并。选读理由：作为整个token merging范式的奠基性工作，ToMe的核心算法（bipartite soft matching）被后续大量MLLM方法所继承和改进。理解ToMe的设计选择和局限性，是理解后续所有merging类方法的基础。

4. **LLaVA-Mini** — 选读理由：代表了"极致压缩"的方向，探索了将视觉token压缩到极少数量（如1个token）后MLLM还能保持多少能力。这种极端实验对于理解"视觉token的信息下限"非常有价值——它帮助回答"MLLM到底需要多少视觉信息"这个根本问题。

5. **DeCo** [105] 和 **DART** [183] — 分析压缩对MLLM内部表示学习影响的工作。选读理由：作为§7.1中提到的少数具有理论分析深度的工作，它们试图回答"为什么压缩有时有效、有时失效"这个关键问题，对于建立压缩的理论框架有启发意义。

6. **VisionThink** [190] — 基于强化学习的自适应压缩决策。选读理由：代表了§7.2中"自适应压缩"的前沿探索方向，将压缩率的选择从人工超参转变为可学习的策略。RL框架在token压缩中的应用是一个新颖且有潜力的方向。

7. **Video-XL系列** [Video-XL → Video-XL-Pro → Video-XL-2] — 极长视频压缩。选读理由：作为面向小时级视频的代表性工作系列，展示了如何通过渐进式设计将压缩能力从2048帧扩展到8000帧。对于视频MLLM效率优化来说，这个系列提供了完整的技术演进路线图。

8. **M3 (Multi-resolution Multi-modal)** [91] — 多分辨率自适应方法。选读理由：M3的实验发现（自然场景仅需9个token vs 文档理解需要576个token）是整个自适应压缩方向的核心motivation来源，同时其多分辨率设计提供了一种实现content-aware压缩的实际框架。

### 可能的改进方向

1. **建立信息论驱动的token重要性评估框架**：当前方法依赖attention权重或相似度等启发式指标来判断token重要性（§7.1），一个有价值的改进方向是设计基于条件互信息的可计算近似指标——例如，训练一个轻量级的"信息量估计器"（基于token特征预测该token对最终输出的信息贡献），用于替代attention权重作为更准确的重要性评分。这个估计器可以在少量标注数据上训练，然后zero-shot迁移到新任务。

2. **设计"框架友好型"的动态压缩方法**：针对§8中指出的与Flash Attention等主流推理框架不兼容的问题，可以探索"分块固定长度压缩"——将输入token按空间位置分组为固定大小的block，在每个block内部做动态压缩但保证输出的block大小一致。这样既保留了内容自适应性（不同block压缩不同内容），又兼容Flash Attention的block结构要求。具体来说，可以设计一种"block-wise adaptive token pooling"机制。

3. **Token压缩与量化的联合优化**：系统研究token压缩（减少序列长度）与模型量化（减少比特宽度）的协同效应和最优组合策略。直觉上两者的收益应该近似正交（一个减少n，一个减少每个元素的大小），但在极端压缩+极端量化的情况下可能出现互相干扰。一个具体的研究问题是：在总计算预算固定的情况下，"保留更多token + 更激进量化"和"更少token + 更高精度"哪种组合更优？

4. **Speculative Compression（投机压缩）**：借鉴speculative decoding的思路，设计一种"先激进压缩尝试回答 → 检测输出质量/置信度 → 低于阈值时自动回退使用更多token"的两阶段框架。对于大多数"简单"输入（通用VQA、简单场景描述），第一阶段的激进压缩即可给出高质量答案；对于少数"困难"输入（OCR、密集推理），自动触发精细模式。这样可以在保证worst-case质量的前提下，大幅提升average-case效率。

5. **面向预训练阶段的原生Token压缩能力**：而非将token压缩作为推理时的后处理优化，探索在MLLM预训练阶段就引入可学习的token压缩模块——让模型在预训练过程中学会根据输入内容和任务需求自适应地调整视觉token粒度。这可能需要设计新的预训练目标函数，同时优化任务性能和token效率。虽然这个方向的实验成本较高（需要从头预训练），但其潜在收益也最大——因为它从根本上改变了MLLM处理视觉信息的方式，而非在已有架构上做修补。

---

*最后更新: 2025-02*
