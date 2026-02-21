[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 FSR 的核心思想、方法和实验结果。

---

In this paper, we propose FSR, a training-free visual token pruning framework inspired by human visual perception, which addresses the fundamental challenge of allocating a limited token budget in VLMs. FSR explicitly models the progressive coordination between local evidence and global context through a three-stage process: focusing on task-critical regions, scanning for complementary contextual cues, and refining sparse representations via aggregation. By jointly considering visual saliency, conditional global coverage, and redundancy-aware refinement, FSR preserves both query-relevant evidence and holistic scene information under strict token constraints.

> 💡 **方法总结**: FSR 的核心 = progressive coordination between local and global。三阶段设计不是 engineering trick，而是有认知科学支撑的系统性方案。

---

Extensive experiments across diverse model architectures, input resolutions, and image–video benchmarks demonstrate that FSR consistently achieves a superior accuracy–efficiency trade-off compared to prior methods. These results highlight the effectiveness of human-inspired local–global coordination as a general paradigm for efficient multimodal inference, and position FSR as a practical solution for deploying large-scale VLMs under real-world resource constraints.

> 💡 **最终评价**:
> - **优势**: (1) 认知科学启发的三阶段设计，(2) 动态 Focus/Scan 分配，(3) 理论覆盖保证，(4) 跨模型/分辨率/模态的一致性优势
> - **局限性**（论文未讨论）: 
>   - 依赖 CLIP text encoder 计算 relevance（Qwen2.5-VL 需要适配）
>   - CCS 的贪心采样复杂度 O(K_S × N × d) 在超大 token 数时可能成为瓶颈
>   - 没有与 training-based 方法对比
>   - 没有在更多架构（InternVL、Phi-3V 等）上验证
> - **未来方向**: 
>   - 与 FlashAttention 的深度集成
>   - 多帧视频场景下的时序感知 Focus/Scan
>   - 与 quantization/distillation 等其他效率方法的正交组合
