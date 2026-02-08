# 5. Infrastructures

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

> 💡 **Section 概览**: 基础设施部分介绍三个方面：大规模训练基础设施（FlagScale 框架 + 5 个优化策略）、强化微调基础设施（VeRL）、推理基础设施（混合精度量化）。

---

### 5.1 Large-Scale Training Infrastructure

> 💡 **5.1 要点预览**: 基于 FlagScale 框架的大规模训练优化，包含混合并行、内存预分配、数据预处理、分布式加载、容错。

To improve the efficiency and stability of multimodal model training, we have developed and integrated a series of key optimization techniques, including hybrid parallelism strategies, memory pre-allocation, distributed data loading, kernel fusion, and fine-grained compute-communication overlapping. These optimizations significantly enhance both resource utilization and training throughput. For data preprocessing, we build upon the Megatron–Energon framework [30] and incorporate custom optimization strategies. Our system supports dynamic mixing of multiple datasets containing diverse modalities, including plain text, single image, multiple images, and video, while also allowing for strict sample order preservation within each dataset. A custom WebDataset-based format [1] enables compatibility with various data modalities and greatly reduces preprocessing time while improving flexibility and scalability in data handling.

#### 5.1.1 Multi-Dimensional Hybrid Parallelism

Multimodal models differ significantly from conventional LLMs in both architecture and data characteristics [33]. On the architectural side, multimodal models are inherently heterogeneous: the vision module (e.g., ViT with Adaptor) is typically a small-scale encoder-only component, while the language module is a much larger decoder-only transformer. On the data side, training samples include plain text, single images, multi-image sequences, and videos. The number of image tokens, text tokens, and the length of the fused token sequence can vary dramatically between samples.

These heterogeneities pose substantial challenges to distributed training frameworks. To address this, we implemented several targeted strategies in our custom framework, FlagScale [12]:

• Non-uniform Pipeline Parallelism [43]: Since the ViT module appears early in the model and has relatively low computational cost, we reduce the number of LLM layers in the first pipeline stage, thereby improving training throughput without increasing memory overhead.

• Separate Recompute Strategy: During the annealing stage, the vision input may contain up to 20,000–30,000 tokens, frequently causing an Out-of-Memory (OOM) error in the ViT module. To mitigate this, we enable recompute [8, 26] only in the ViT module to reduce memory usage of intermediate activations, while disabling recompute in the LLM module to preserve computational efficiency.

> 💡 **5.1.1 批注**: 两个关键策略:
> - 非均匀 Pipeline Parallelism: ViT 小、LLM 大，所以第一个 PP stage 放更少 LLM 层
> - 分离重计算: 只在 ViT 做 recompute（因为它可能有 20K-30K tokens），LLM 不做（保持效率）

#### 5.1.2 Pre-Allocate Memory

In the supervised fine-tuning training process of RoboBrain 2.0, input lengths vary significantly across samples. PyTorch's default caching memory allocator [49] can lead to memory fragmentation under such dynamic input conditions, frequently resulting in OOM errors. A common but inefficient workaround is to call torch.cuda.empty_cache() before every forward pass, which severely degrades performance. Instead, we take a more efficient approach by analyzing PyTorch's memory allocation mechanism. Fragmentation often results from the lack of a sufficiently large and contiguous cached memory block for new tensors, prompting new allocations and worsening fragmentation. To address this, we introduce a memory pre-allocation strategy: we compute the maximum sequence length across the entire dataset before training, and pad all samples to this maximum length in the first step. This ensures that tensors can reuse pre-allocated memory blocks, reducing fragmentation and maintaining throughput.

> 💡 **5.1.2 批注**: 巧妙的内存优化——第一步用最大序列长度 pad 所有样本，让 PyTorch 预分配足够大的连续内存块。之后就能复用这些块，避免碎片化。比每次 empty_cache() 高效得多。

#### 5.1.3 Data Pre-Processing

We adopt native Megatron-Energon [30] for unified data loading, eliminating the need for external training frameworks. Additionally, we optimized the preprocessing pipeline to reduce time consumption by up to 90%. We evaluated and compared two preprocessing strategies:

• Preprocessing Both JSON and Images. Using the default Megatron-Energon data pipeline, both JSON metadata and images are compressed into binary files for WebDataset. However, this approach suffers from two major issues: (1) Low efficiency: Preprocessing 320,000 samples can take over 2 hours. (2) Inconsistent image readers: Megatron-Energon uses cv2, while models such as RoboBrain 2.0 use PIL, introducing subtle differences that may affect training performance.

• Preprocessing JSON Only (Recommended). In our optimized pipeline, only JSON files are preprocessed, and images are kept in their original form. Image preprocessing is deferred to the TaskEncoder module using the same preprocessor as Qwen2.5-VL. (1) High efficiency: Preprocessing 320,000 samples takes less than 10 minutes. (2) Alignment with model input: Ensures image handling is fully aligned between preprocessing and training, eliminating inconsistency and improving model performance.

> 💡 **5.1.3 批注**: 只预处理 JSON，图像延迟到 TaskEncoder 处理。320K 样本从 2h+ 降到 <10min，90% 提速。还避免了 cv2/PIL 不一致问题。

#### 5.1.4 Distributed Data Loading

To minimize the I/O burden on compute nodes, we reduce redundant data loading in large-scale distributed training. Unlike single-node setups, GPUs in distributed training systems play different roles depending on the chosen parallel strategy. Data loading typically occurs along the data parallel (DP) dimension, where each DP rank handles a unique data shard. However, in multi-dimensional hybrid parallelism (e.g., DP-PP-TP), only a subset of GPU processes actually need to load data: (1) In each Pipeline Parallel (PP) [42] group, only the first and last stages need to perform data loading. (2) Within Tensor Parallel (TP) [58] groups, only one GPU per group is required to load data, with others receiving data via broadcast. This design significantly reduces redundant I/O operations and improves overall data throughput.

> 💡 **5.1.4 批注**: 在 DP-PP-TP 混合并行中，只有需要数据的 GPU 才加载：PP 只有首尾 stage、TP 只有一个 GPU 加载后广播。减少冗余 I/O。

#### 5.1.5 Fault Tolerance

To handle both hardware and software failures during training, we co-designed fault-tolerant mechanisms between our FlagScale [12] training framework and the system platform. Common errors, such as LostCard, KubeNodeNotReady, are automatically detected and trigger automatic job recovery and restart, ensuring minimal disruption. Furthermore, our custom DataLoader module based on Megatron-Energon supports full data state recovery, allowing seamless resumption from the most recent checkpoint with complete consistency in data loading and sample shuffling states.

> 💡 **5.1.5 批注**: 自动检测硬件故障（如 GPU 掉卡、K8s 节点失联）并自动重启。DataLoader 支持完整的数据状态恢复，保证 checkpoint 后数据顺序一致。

---

### 5.2 Reinforcement Fine-Tuning Infrastructure

> 💡 **5.2 要点预览**: 使用 VeRL 框架做 RLVR（基于可验证奖励的强化学习）。

We employ Reinforcement Learning with Verifiable Rewards (RLVR) to enhance RoboBrain 2.0 using VeRL [68], an open-source RL framework specifically designed for post-training LLMs and VLMs. Based on the HybridFlow architecture [56], VeRL features a hybrid-controller model that integrates both a global controller for inter-RL-role dataflow coordination and distributed controllers for intra-RL-role parallel processing. This architecture enables efficient execution of complex post-training workflows while ensuring scalability. VeRL's support for multiple RL algorithms (e.g., GRPO) and seamless LLM integration makes it particularly suitable for RoboBrain 2.0's reinforcement fine-tuning (RFT) requirements. The framework enables high-performance model tuning with minimal overhead through its optimized dataflow management and parallel processing capabilities. Its efficient handling of large-scale training tasks and rigorous reward verification establishes VeRL as an ideal platform for advancing RoboBrain 2.0's capabilities via RLVR.

> 💡 **5.2 小结**: VeRL = 火山引擎开源的 RL 框架。HybridFlow 架构，支持 GRPO 等算法。用于 Stage 3 的 RFT phase。

---

### 5.3 Inference Infrastructure

> 💡 **5.3 要点预览**: FlagScale 推理框架 + 混合精度量化策略。

To improve the efficiency of model inference, we adopt FlagScale [12], also a multi-backend inference framework, which can automatically search for the optimal inference engine and configuration parameters based on the performance characteristics of different models on heterogeneous hardware accelerators, thereby effectively reducing inference latency. Given the high sensitivity of embodied AI models to accuracy, we further introduce a mixed-bit quantization strategy [40, 70]. This strategy enhances inference efficiency and resource utilization while maintaining model performance. Specifically, the vision encoder retains full-precision floating-point computation to ensure the accuracy of key feature extraction. In contrast, during the language module, weights are quantized to 8-bit integers, while activations are preserved in 16-bit floating-point format. This mixed-precision approach significantly reduces computational overhead and memory usage with negligible impact on model accuracy. Moreover, the quantization process is minimally invasive to existing inference pipelines and can be flexibly integrated into current systems. In end-to-end embodied tasks, weight-only quantization alone achieves approximately a 30% reduction in inference latency, demonstrating the effectiveness and practicality of the proposed method in real-world deployment scenarios.

> 💡 **5.3 小结**:
> ```
> 混合精度量化策略:
> ├── Vision Encoder: 全精度（FP16/32）— 保证特征提取精度
> └── Language Model: W8A16（权重 INT8，激活 FP16）
>
> 效果: 推理延迟降低 ~30%，精度损失可忽略
> ```

---

## 💡 Section 总结

### 基础设施全景
```
训练:
├── 框架: FlagScale (开源)
├── 并行: 非均匀 PP + TP + DP
├── 内存: 预分配策略（首步 pad 到最大长度）
├── 数据: 只预处理 JSON，90% 提速
├── I/O: 只在需要的 GPU 加载
└── 容错: 自动检测故障 + 状态恢复

强化学习: VeRL (火山引擎开源)

推理:
├── 框架: FlagScale
└── 量化: ViT FP + LLM W8A16, 延迟 -30%
```

### 核心洞察
1. **FlagScale 是 BAAI 的核心基础设施**: 同时支持训练和推理
2. **多模态训练的特殊挑战**: ViT 和 LLM 的异构性需要专门的并行策略
3. **工程优化务实有效**: 内存预分配、JSON-only 预处理等都是实用的工程优化
4. **混合量化策略合理**: ViT 保持全精度（空间感知对精度敏感），LLM 量化（文本生成容忍度高）
