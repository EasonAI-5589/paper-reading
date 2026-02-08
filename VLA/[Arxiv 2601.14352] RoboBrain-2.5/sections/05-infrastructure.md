[← 返回 README](../README.md)

# 5 Infrastructure

## 📌 预览
训练基础设施的三个关键优化：混合并行策略、动态预分配内存、跨加速器（NVIDIA↔摩尔线程）训练与推理。

---

During the training of RoboBrain 2.5, we build upon the infrastructure established in RoboBrain 2.0 [33, 72] while further strengthening and systematizing the core training pipeline. The overall system adopts a multi-dimensional hybrid parallelism strategy, combined with distributed data loading optimizations, and a deeply optimized memory pre-allocation mechanism tailored for multi-modal long-sequence training. These improvements significantly enhance hardware utilization efficiency and overall training throughput.

On the data side, our implementation is based on the Megatron–Energon [40] framework with substantial in-house optimizations. This design enables unified format representation and online mixed training of heterogeneous modalities, including text, single-image, multi-image, and video samples. At the same time, we strictly preserve intra-dataset sample ordering to satisfy the requirements of instruction alignment and temporal consistency. By adopting a customized WebDataset [4] sample format, the system achieves compatibility with diverse data types while substantially reducing offline preprocessing overhead and improving the flexibility and extensibility of the data pipeline.

> 💡 **批注**: 基于 Megatron-Energon + FlagScale 框架，支持文本/单图/多图/视频的混合训练。

---

## 5.1 Hybrid Parallelism

Multi-modal large models exhibit pronounced heterogeneity in both model architecture and computational characteristics [48]. The visual component typically consists of a relatively lightweight ViT-based encoder (with adapter modules), whereas the language component is dominated by a large-scale decoder-only architecture. Although the visual encoder has a smaller parameter footprint, its computational cost becomes non-trivial when training with a high proportion of visual or video samples.

To address this architectural heterogeneity, we leverage the heterogeneous training experience accumulated in our in-house distributed framework, FlagScale [20], and adopt an uneven pipeline parallelism strategy [56]. Specifically, the ViT module is placed at the front of the model, and the number of language layers assigned to the first pipeline stage is reduced accordingly. This design balances computational load across pipeline stages, mitigates pipeline bubbles, and improves overall pipeline efficiency.

> 💡 **批注**: 不均匀流水线并行——ViT 放在第一个 pipeline stage，减少该 stage 的 LM 层数，平衡计算负载。TP=2, PP=2。

---

## 5.2 Dynamic pre-Allocated Memory

In RoboBrain 2.5 training, sequence lengths vary significantly across samples. Combined with PyTorch's default CUDA caching memory allocator, this dynamic-shape workload often leads to severe GPU memory fragmentation and, in extreme cases, out-of-memory (OOM) failures. A common workaround is to invoke torch.cuda.empty_cache() [61] before each iteration; however, this approach disrupts memory reuse and substantially degrades training performance.

To resolve this issue, we conduct an in-depth analysis of CUDA memory allocation and reuse behavior and propose a dynamic unified padding strategy based on dual data streams.

• Before training begins, the maximum sequence length observed in the training set is collected;
• In the first training iteration, all samples are padded to this maximum length, enabling one-time memory pre-allocation during initialization;
• In subsequent iterations, tensors reuse the pre-allocated memory, effectively suppressing memory fragmentation;
• Only when the visual token length exceeds the current maximum does the system trigger a full cache cleanup and re-pad samples to the new maximum length.

This strategy strikes a practical balance between memory efficiency and training performance, providing both stability and high throughput in large-scale multi-modal long-sequence training scenarios.

> 💡 **批注**: 动态内存预分配策略——首次迭代按最大长度 pad 并预分配，后续复用。避免了 `empty_cache()` 的性能损失和动态形状导致的 OOM。实用的工程优化。

---

## 5.3 Cross-Accelerator Training and Inference

Leveraging FlagScale's distributed training capabilities on heterogeneous accelerator clusters, together with VLM-specific kernel and communication optimizations, we successfully complete end-to-end training of RoboBrain 2.5 on a thousand-device cluster composed of non-NVIDIA accelerators. The resulting loss convergence behavior closely matches that observed on NVIDIA platforms, with the final convergence gap controlled within $0.62\%$.

Furthermore, the trained checkpoints are seamlessly migrated to NVIDIA-based platforms for downstream evaluation. Across a range of mainstream benchmarks, the resulting performance remains highly consistent with models trained natively on NVIDIA hardware. This RoboBrain 2.5 case study demonstrates that FlagOS/FlagScale's cross-accelerator training and inference capabilities have matured to a level that is reliable, practical, and production-ready for large-scale multi-modal model training.

> 💡 **批注**: 
> - 在摩尔线程（MTT）千卡集群上完成端到端训练，loss 收敛差距仅 0.62%
> - 训练好的 checkpoint 可无缝迁移到 NVIDIA 平台评估
> - 这是 BAAI 展示 FlagScale 的国产化适配能力——具有政治和产业意义

---

## 🔖 Section 总结

### 核心洞察
1. 工程实现基于 FlagScale + Megatron-Energon，是 BAAI 的自研基础设施
2. 不均匀 PP 解决多模态模型的计算异构性
3. 动态内存预分配是针对变长多模态序列的实用优化
4. 跨加速器训练验证了国产 GPU（摩尔线程）可达到与 NVIDIA 几乎一致的效果
