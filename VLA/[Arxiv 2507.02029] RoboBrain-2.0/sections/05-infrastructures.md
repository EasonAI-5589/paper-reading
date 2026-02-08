# 5 Infrastructures

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

### 5.1 Large-Scale Training Infrastructure

To improve the efficiency and stability of multimodal model training, we have developed and integrated a series of key optimization techniques, including hybrid parallelism strategies, memory pre-allocation, distributed data loading, kernel fusion, and fine-grained compute-communication overlapping. These optimizations significantly enhance both resource utilization and training throughput. For data preprocessing, we build upon the Megatron–Energon framework [30] and incorporate custom optimization strategies. Our system supports dynamic mixing of multiple datasets containing diverse modalities, including plain text, single image, multiple images, and video, while also allowing for strict sample order preservation within each dataset. A custom WebDataset-based format [1] enables compatibility with various data modalities and greatly reduces preprocessing time while improving flexibility and scalability in data handling.

> 💡 **训练框架**: FlagScale (BAAI 开源) + Megatron-Energon，支持混合模态数据的动态混合。

#### 5.1.1 Multi-Dimensional Hybrid Parallelism

Multimodal models differ significantly from conventional LLMs in both architecture and data characteristics [33]. On the architectural side, multimodal models are inherently heterogeneous: the vision module (e.g., ViT with Adaptor) is typically a small-scale encoder-only component, while the language module is a much larger decoder-only transformer. On the data side, training samples include plain text, single images, multi-image sequences, and videos. The number of image tokens, text tokens, and the length of the fused token sequence can vary dramatically between samples.

These heterogeneities pose substantial challenges to distributed training frameworks. To address this, we implemented several targeted strategies in our custom framework, FlagScale [12]:

• Non-uniform Pipeline Parallelism [43]: Since the ViT module appears early in the model and has relatively low computational cost, we reduce the number of LLM layers in the first pipeline stage, thereby improving training throughput without increasing memory overhead.

• Separate Recompute Strategy: During the annealing stage, the vision input may contain up to 20,000–30,000 tokens, frequently causing an Out-of-Memory (OOM) error in the ViT module. To mitigate this, we enable recompute [8, 26] only in the ViT module to reduce memory usage of intermediate activations, while disabling recompute in the LLM module to preserve computational efficiency.

> 💡 **并行策略要点**:
> ```
> 挑战: 多模态模型的异构性
> ├── ViT (小) vs LLM (大) → 计算不均衡
> └── 样本长度差异大 (text vs image vs video)
>
> 解决:
> ├── Non-uniform PP: 第一个 pipeline stage 少放 LLM layers (因为 ViT 在前面)
> └── Separate Recompute: ViT 开 recompute (省内存), LLM 关 recompute (保速度)
> ```
> **实用经验**: ViT 可能有 20K-30K tokens 导致 OOM，只对 ViT 做 gradient checkpointing 是一个好的 trade-off。

#### 5.1.2 Pre-Allocate Memory

In the supervised fine-tuning training process of RoboBrain 2.0, input lengths vary significantly across samples. PyTorch's default caching memory allocator [49] can lead to memory fragmentation under such dynamic input conditions, frequently resulting in OOM errors. A common but inefficient workaround is to call torch.cuda.empty_cache() before every forward pass, which severely degrades performance. Instead, we take a more efficient approach by analyzing PyTorch's memory allocation mechanism. Fragmentation often results from the lack of a sufficiently large and contiguous cached memory block for new tensors, prompting new allocations and worsening fragmentation. To address this, we introduce a memory pre-allocation strategy: we compute the maximum sequence length across the entire dataset before training, and pad all samples to this maximum length in the first step. This ensures that tensors can reuse pre-allocated memory blocks, reducing fragmentation and maintaining throughput.

> 💡 **内存预分配**:
> - 问题: 变长输入 → PyTorch 内存碎片 → OOM
> - 常见 hack: 每次 forward 前 `torch.cuda.empty_cache()` → 性能很差
> - **解决**: 第一步用最大 seq length pad → 预分配内存 → 后续复用
> - 这个技巧对做 VLM 训练很实用！

#### 5.1.3 Data Pre-Processing

We adopt native Megatron-Energon [30] for unified data loading, eliminating the need for external training frameworks. Additionally, we optimized the preprocessing pipeline to reduce time consumption by up to 90%. We evaluated and compared two preprocessing strategies:

• Preprocessing Both JSON and Images. Using the default Megatron-Energon data pipeline, both JSON metadata and images are compressed into binary files for WebDataset. However, this approach suffers from two major issues: (1) Low efficiency: Preprocessing 320,000 samples can take over 2 hours. (2) Inconsistent image readers: Megatron-Energon uses cv2, while models such as RoboBrain 2.0 use PIL, introducing subtle differences that may affect training performance.

• Preprocessing JSON Only (Recommended). In our optimized pipeline, only JSON files are preprocessed, and images are kept in their original form. Image preprocessing is deferred to the TaskEncoder module using the same preprocessor as Qwen2.5-VL. (1) High efficiency: Preprocessing 320,000 samples takes less than 10 minutes. (2) Alignment with model input: Ensures image handling is fully aligned between preprocessing and training, eliminating inconsistency and improving model performance.

> 💡 **数据预处理优化**:
> ```
> 方案 A (默认): JSON + Images → binary WebDataset
>   ❌ 320K samples = 2+ hours
>   ❌ cv2 vs PIL 不一致
>
> 方案 B (推荐): JSON only → WebDataset, images 原样保留
>   ✅ 320K samples = <10 min (20x 加速)
>   ✅ 和 Qwen2.5-VL preprocessor 对齐
> ```
> **经验**: cv2 和 PIL 读图有微妙差异（色彩空间、resize 方式），对训练结果有影响。

#### 5.1.4 Distributed Data Loading

To minimize the I/O burden on compute nodes, we reduce redundant data loading in large-scale distributed training. Unlike single-node setups, GPUs in distributed training systems play different roles depending on the chosen parallel strategy. Data loading typically occurs along the data parallel (DP) dimension, where each DP rank handles a unique data shard. However, in multi-dimensional hybrid parallelism (e.g., DP-PP-TP), only a subset of GPU processes actually need to load data: (1) In each Pipeline Parallel (PP) group, only the first and last stages need to perform data loading. (2) Within Tensor Parallel (TP) groups, only one GPU per group is required to load data, with others receiving data via broadcast. This design significantly reduces redundant I/O operations and improves overall data throughput.

> 💡 **分布式数据加载**: PP 组只有首尾 stage 加载数据, TP 组只有 1 个 GPU 加载然后 broadcast。减少冗余 I/O。

#### 5.1.5 Fault Tolerance

To handle both hardware and software failures during training, we co-designed fault-tolerant mechanisms between our FlagScale [12] training framework and the system platform. Common errors, such as LostCard, KubeNodeNotReady, are automatically detected and trigger automatic job recovery and restart, ensuring minimal disruption. Furthermore, our custom DataLoader module based on Megatron-Energon supports full data state recovery, allowing seamless resumption from the most recent checkpoint with complete consistency in data loading and sample shuffling states.

> 💡 **容错机制**: 自动检测 GPU 掉卡/节点故障 → 自动恢复训练。DataLoader 也能恢复 shuffle 状态。这对大规模训练（512 GPUs）很重要。

---

### 5.2 Reinforcement Fine-Tuning Infrastructure

We employ Reinforcement Learning with Verifiable Rewards (RLVR) to enhance RoboBrain 2.0 using VeRL [68], an open-source RL framework specifically designed for post-training LLMs and VLMs. Based on the HybridFlow architecture [56], VeRL features a hybrid-controller model that integrates both a global controller for inter-RL-role dataflow coordination and distributed controllers for intra-RL-role parallel processing. This architecture enables efficient execution of complex post-training workflows while ensuring scalability. VeRL's support for multiple RL algorithms (e.g., GRPO) and seamless LLM integration makes it particularly suitable for RoboBrain 2.0's reinforcement fine-tuning (RFT) requirements. The framework enables high-performance model tuning with minimal overhead through its optimized dataflow management and parallel processing capabilities. Its efficient handling of large-scale training tasks and rigorous reward verification establishes VeRL as an ideal platform for advancing RoboBrain 2.0's capabilities via RLVR.

> 💡 **RL 基础设施**: 用 VeRL (开源 RL 框架) 做 GRPO。HybridFlow 架构支持 multi-role 并行（policy/reference/reward model 同时运行）。

---

### 5.3 Inference Infrastructure

To improve the efficiency of model inference, we adopt FlagScale [12], also a multi-backend inference framework, which can automatically search for the optimal inference engine and configuration parameters based on the performance characteristics of different models on heterogeneous hardware accelerators, thereby effectively reducing inference latency. Given the high sensitivity of embodied AI models to accuracy, we further introduce a mixed-bit quantization strategy [40, 70]. This strategy enhances inference efficiency and resource utilization while maintaining model performance. Specifically, the vision encoder retains full-precision floating-point computation to ensure the accuracy of key feature extraction. In contrast, during the language module, weights are quantized to 8-bit integers, while activations are preserved in 16-bit floating-point format. This mixed-precision approach significantly reduces computational overhead and memory usage with negligible impact on model accuracy. Moreover, the quantization process is minimally invasive to existing inference pipelines and can be flexibly integrated into current systems. In end-to-end embodied tasks, weight-only quantization alone achieves approximately a 30% reduction in inference latency, demonstrating the effectiveness and practicality of the proposed method in real-world deployment scenarios.

> 💡 **推理优化: 混合精度量化**:
> ```
> Vision Encoder: FP32 (保精度)
> LLM Weights: INT8 (省内存)
> LLM Activations: FP16 (保精度)
>
> 效果: 推理延迟降低 ~30%，精度无明显损失
> ```
> **设计理由**: Vision Encoder 对精度敏感（feature extraction），不能量化；LLM weights 冗余大，可以量化。

---

## 💡 Section 总结

### 基础设施亮点
| 技术 | 作用 | 框架 |
|------|------|------|
| Non-uniform PP | ViT/LLM 负载均衡 | FlagScale |
| Separate Recompute | ViT 省内存, LLM 保速度 | FlagScale |
| Memory Pre-allocation | 避免 PyTorch 内存碎片 | 自研 |
| JSON-only Preprocessing | 20x 预处理加速 | Megatron-Energon |
| GRPO with VeRL | 高效 RL 训练 | VeRL |
| Mixed-bit Quantization | 30% 推理加速 | FlagScale |

### 核心洞察
1. **工程细节很实用**: memory pre-allocation 和 JSON-only preprocessing 是很好的 VLM 训练 tips
2. **FlagScale 生态**: BAAI 的全栈训练+推理框架，从 parallelism 到 fault tolerance 到 inference
3. **这个 section 对做系统的人有参考价值**，对做算法的人可以快速跳过
