# 5. Benchmarks and Metrics

本节提供MLLM token压缩研究中常用的benchmarks（§5.1）和评估指标（§5.2）的详细概览。

## 5.1 Benchmarks (Table 8)

### 图像理解 Benchmarks

Table 8汇总了token压缩研究中广泛使用的图像和视频理解benchmarks。根据所评估的主要能力，这些benchmarks可以归为若干类别。

| Benchmark | 题型 | 评估指标 | 样本数 | 测评能力 |
|-----------|------|---------|--------|---------|
| GQA-testdev-balanced | Open | Accuracy | 12,578 | 通用图像感知 |
| VQA-v2-testdev | Open | Accuracy | 107,394 | 通用图像感知 |
| VizWiz-val | Open | Accuracy | 4,319 | 通用图像感知 |
| POPE | Y/N | F1-Score | 3,000 | 通用图像感知 |
| TextVQA-val | Open | Accuracy | 5,000 | OCR |
| ScienceQA-Image-test | MQA, Y/N | Accuracy | 2,017 | 知识 |
| MathVista-testmini | MQA, Open | Accuracy | 1,000 | 知识/推理 |
| MathVerse-testmini | MQA, Open | Accuracy | 3,940 | 知识/推理 |
| MMMU | MQA, Open | Accuracy | 11,550 | 知识/推理 |
| MME | Y/N | Perception Score | 2,374 | 综合 |
| MMBench-en-dev | MQA | Accuracy | 4,329 | 综合 |
| MM-Vet | Open | GPT-Score | 218 | 综合 |
| SeedBench-Image | MQA | Accuracy | 14,280 | 综合 |
| LLaVA-Bench^W | Open | GPT-Score | 60 | 综合 |

> **[感受]** 从表中可以看到，图像理解benchmarks的样本规模差异极大——VQA-v2有10万+样本，而LLaVA-Bench^W仅有60个样本、MM-Vet仅218个样本。样本量如此悬殊的benchmarks被放在一起做性能对比，统计可靠性差异巨大：在60个样本上0.5%的精度差异可能完全是噪声，而在10万样本上0.1%的差异就可能是显著的。对做token压缩的研究者而言，一个关键的实践问题是：当论文声称"在X benchmark上仅下降0.3%精度"时，必须结合该benchmark的样本量来判断这个数字是否有统计意义。此外，通用图像感知类benchmark（GQA、VQA-v2、VizWiz、POPE）占了4个，但OCR类仅TextVQA一个——而OCR恰恰是对token压缩最敏感的任务类型（需要细粒度像素级信息），这种benchmark分布的不均衡可能导致压缩方法在"平均分"上看起来很好，但在实际部署中最需要的细粒度任务上表现不佳。

### 视频理解 Benchmarks

| Benchmark | 题型 | 评估指标 | 样本数 | 测评能力 |
|-----------|------|---------|--------|---------|
| ActivityNet-QA-test | Open | Accuracy, GPT-Score | 8,000 | 综合 |
| MVBench | MQA | Accuracy | 4,000 | 时序理解 |
| EgoSchema | MQA | Accuracy | 5,063 | 长视频 |
| LongVideoBench-val | MQA | Accuracy | 1,337 | 长视频 |
| MLVU-dev | MQA, Open | Accuracy, GPT-Score | 2,593 | 长视频综合 |
| Next-QA-MC-test | MQA | Accuracy | 8,564 | 综合 |
| Video-ChatGPT | Open | GPT-Score | 3,493 | 综合 |
| Video-MME | MQA | Accuracy | 2,700 | 综合 |

> **[感受]** 视频理解benchmarks的数量和多样性明显不如图像侧——仅8个benchmark，且其中"综合"类占了4个，专门测试时序理解的仅MVBench一个。这对token压缩研究是一个隐患：视频token压缩的核心挑战在于如何在压缩时序冗余的同时保留细粒度的时序动态信息（如动作顺序、因果关系、状态变化），但现有benchmarks对这些能力的细粒度测试严重不足。此外，长视频benchmarks（EgoSchema、LongVideoBench、MLVU）的"长"定义差异很大——有的是几分钟，有的是几十分钟——但token压缩在不同视频长度下的性能衰减曲线可能呈现非线性特征，缺乏按视频长度分层的系统评估会掩盖压缩方法在超长视频上的真实表现。对于做视频token压缩的研究者来说，仅在这些benchmarks上报告平均分是不够的，应该按视频时长、时序复杂度等维度做细粒度分析。

### Benchmark分类

**图像理解**:
- **General Image Perception（通用图像感知）**: 评估自然图像中的基本视觉识别能力，包括物体识别、场景理解、属性判断和空间关系推理
- **OCR（光学字符识别）**: 衡量识别和解释非结构化视觉格式中嵌入的文本内容的能力。这一技能对于MLLM与人类之间的有效交互至关重要
- **Knowledge（知识）**: 评估将视觉感知与领域特定知识或通用世界知识相结合的能力
- **Reasoning（推理）**: 超越单纯感知，要求基于视觉内容并结合先验知识进行逻辑推理和问题解决
- **Integrated（综合）**: 将视觉感知和推理任务合并到单一benchmark中，提供全面的多模态理解能力评估

**视频理解**:
- **Temporal Understanding（时序理解）**: 衡量捕捉和解读时序动态的能力，包括动作序列、运动模式和事件定位
- **Long Video Understanding（长视频理解）**: 评估在长视频（从几分钟到几十分钟）上的处理和推理能力
- **Integrated Video Understanding（综合视频理解）**: 结合多个评估维度，提供对视频场景中感知和推理能力的全面评估

> **[感受]** 这个分类体系的粒度偏粗，对token压缩研究的指导价值有限。以"General Image Perception"为例，它包含了物体识别、场景理解、属性判断、空间关系推理等多种子能力——这些子能力对token压缩的敏感度截然不同：物体识别可能只需要少数关键区域的token，而空间关系推理则需要保留物体间的相对位置信息，压缩策略完全不同。同样，"Integrated"类别过于笼统，无法揭示压缩在哪些具体维度上造成了性能损失。从MLLM效率优化的角度看，我们迫切需要一种**压缩感知的benchmark分类法**——不是按"任务需要什么能力"分类，而是按"任务对视觉信息的哪些维度（空间细节、全局语义、时序动态、文本-视觉对齐）最敏感"来分类，这样才能直接指导压缩方法的选择和压缩率的设定。

---

## 5.2 Metrics

MLLM token压缩方法的评估主要从两个视角出发：下游任务表现（**effectiveness**，效果）和计算效率（**efficiency**，效率），包括理论指标和实际指标。

### 5.2.1 Effectiveness (效果)

效果评估通常遵循原始benchmarks的标准。

| 指标 | 描述 |
|------|------|
| **Accuracy** | 衡量模型预测是否匹配ground-truth答案，是大多数benchmark的主指标。适用于有确定性正确答案的任务（MQA、Y/N、Open QA等） |
| **GPT-Score** | 利用GPT对MLLM的开放式回答进行数值评分。主要用于image captioning等无唯一正确答案的生成式任务，以及开放式视频问答 |

> **[感受]** 当前effectiveness指标的最大问题在于它们衡量的是"最终答案对不对"，而非"压缩丢失了什么信息"。Accuracy作为离散指标（对/错），无法反映压缩造成的渐进性语义退化——一个模型可能在答案仍然正确的情况下，其内部表征已经丢失了大量视觉细节，只是这些细节恰好不影响当前问题的答案。GPT-Score虽然是连续值，但其评分标准不透明且不稳定（不同版本的GPT评分可能不一致），更关键的是它无法区分"因为压缩导致的质量下降"和"模型本身的生成随机性"。对于token压缩研究，理想的effectiveness指标应该能测量**信息保留度**——例如压缩前后模型内部表征的相似度、或者在同一输入上回答一系列由粗到细的问题时的精度衰减曲线。这种"压缩感知"的评估方式才能真正揭示不同压缩方法的信息保留特性，而非仅仅看最终的答案正确率。

### 5.2.2 Efficiency (效率)

效率可以从几个互补的方面进行评估：

| 指标 | 描述 | 注意事项 |
|------|------|---------|
| **Token Retention Count/Ratio** | 压缩后保留的视觉token绝对数量或相对百分比。token压缩方法通常在相同的保留数量/比例下进行比较 | 相同保留率 ≠ 相同推理延迟——压缩位置（encoder内部/projector/LLM内部）对实际延迟影响巨大 |
| **Prefilling/Decoding FLOPs** | 预填充和解码阶段的理论计算量（以浮点运算次数衡量） | 硬件无关的理论指标，便于跨平台对比 |
| **Prefilling/Decoding Latency** | 模型处理输入（prefilling）和生成输出token（decoding）的实际墙钟时间 | 硬件相关——与FLOPs不同，latency受具体基础设施和实现的影响很大 |
| **Memory Usage** | 推理时的峰值内存占用 | 对资源受限设备部署至关重要。token压缩可减少attention KV-cache和中间表示所需的内存，但减少量高度依赖于压缩的实现方式 |

> **[感受]** 论文正确指出了"相同token保留率 ≠ 相同推理延迟"这一关键洞察，但实际上问题更深层：Token Retention Ratio作为最常用的效率指标，本质上是一个**代理指标（proxy metric）**而非直接的效率指标。在encoder内部压缩掉50%的token和在LLM第10层之后压缩掉50%的token，虽然保留率相同，但前者节省了整个projector和LLM所有层的计算，后者只节省了LLM后续层的计算——实际加速比可能差5-10倍。更大的问题是，当前论文普遍缺乏**端到端的效率-效果Pareto分析**：不同方法在不同保留率下的精度-延迟-内存三维trade-off曲线才是最有决策价值的信息，但几乎没有论文做这种系统对比。FLOPs和Latency之间的鸿沟也值得关注——由于GPU的并行特性和内存带宽瓶颈，减少50%的FLOPs往往只能减少20-30%的实际延迟，这使得仅报告FLOPs的论文可能高估了其方法的实际价值。Memory Usage指标虽然被列出，但在实践中报告它的论文不到一半，这对于端侧部署场景是一个严重的信息缺失。

---

## 评估体系的问题 (§7.4)

从评估角度看，现有token压缩方法的效率和效果主要通过下游多模态任务来评估。论文指出当前评估实践存在三个关键限制：

### 限制1：缺乏系统性任务分类 (Lack of Systematic Task Categorization)

如Table 8所示，benchmarks按粗粒度类别分组，对token压缩如何影响特定视觉理解能力（如空间关系推理 vs 物体追踪）以及特定内容域（如表格解读 vs 图表解释）的洞察力有限。这种粗粒度分类掩盖了压缩对不同子能力的差异化影响。

> **[感受]** 这个限制揭示了一个根本性问题：当前的benchmark设计是为了评估MLLM的能力，而非为了评估token压缩方法的特性。一个好的token压缩评估体系应该能回答"这个方法保留了什么、丢失了什么"，而不只是"这个方法在某些任务上掉了多少分"。例如，一个pruning方法可能在VQA-v2上精度不变但在需要精确空间关系理解的任务上严重退化，而粗粒度的"通用图像感知"分类完全无法捕捉这种差异。从研究选题的角度看，**设计一套专门面向token压缩的诊断性benchmark**（类似于NLP中的CheckList或BIG-Bench的设计思路）是一个有价值且目前空白的研究方向——它应该按信息类型（空间细节、纹理、颜色、位置关系、文字等）组织测试用例，使研究者能精确定位压缩方法的信息丢失模式。

### 限制2：低效评估流程 (Inefficient Evaluation Processes)

当前评估通常需要在10+个benchmarks上运行，每个包含数千个样本。许多benchmarks在评估焦点上存在大量重叠，导致评估冗余严重、资源利用低效。完整评估一个压缩方法可能需要数天的GPU时间，这本身与"追求效率"的研究目标形成了讽刺性矛盾。

> **[感受]** 评估效率低下的问题在实践中比论文描述得更严重。以一个典型的token压缩论文为例：如果要在14个图像benchmark和8个视频benchmark上、以3-5个不同压缩率分别评估，总共需要数十次完整推理流程。对于计算资源有限的实验室，光是"跑完所有benchmark"就可能需要一周以上。这导致了两个不良后果：(1) 研究者倾向于cherry-pick对自己方法有利的benchmark子集来报告，降低了可比性；(2) 快速迭代实验变得困难，限制了方法探索的效率。一个可行的解决方案是构建一个**轻量级代理benchmark集**——通过分析现有benchmark之间的相关性矩阵，选出一组互信息最大化、冗余最小化的核心评估子集，可能5-6个benchmark就能覆盖90%以上的评估信息量。这不仅能加速评估，还能提高跨论文的可比性。

### 限制3：缺乏统一评估标准 (Absence of Consistent Evaluation Standards)

不同工作使用不同的benchmark和指标组合，每项工作强调不同的优势，导致跨方法的公平比较极为困难。尽管近期有工作[296]引入了专门针对token压缩的更具挑战性的评估设定，但系统化、标准化的评估框架仍然缺失。

> **[感受]** 缺乏统一评估标准是整个token压缩领域最大的"基础设施债务"。不同论文在以下维度都可能不同：(1) 基座模型（LLaVA-1.5 vs InternVL vs Qwen-VL）；(2) 图像分辨率和预处理方式；(3) benchmark的具体split和评估脚本版本；(4) 推理设置（beam search vs greedy, temperature等）。这些差异使得即使两篇论文都报告了"MMBench上的accuracy"，数字也可能不直接可比。从MLLM效率优化社区的角度看，建立一个类似于LM-Evaluation-Harness的**统一token压缩评估平台**——固定基座模型、标准化评估流程、提供效率-效果联合报告——将是推动领域进步的关键基础设施。这也是为什么论文[296]的工作方向（专门为token压缩设计评估设定）值得关注和跟进，它可能成为未来标准化的起点。此外，一个被完全忽视的评估维度是**压缩方法的鲁棒性**——在输入分布偏移、对抗扰动、或极端压缩率下方法的性能稳定性，这对实际部署至关重要但几乎没有论文系统评估过。
