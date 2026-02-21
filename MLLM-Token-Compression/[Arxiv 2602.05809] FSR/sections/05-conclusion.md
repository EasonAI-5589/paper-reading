[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 FSR 的贡献，强调人类认知启发的局部-全局协调范式。

---

In this paper, we propose FSR, a trainingfree visual token pruning framework inspired by human visual perception, which addresses the fundamental challenge of allocating a limited token budget in VLMs. FSR explicitly models the progressive coordination between local evidence and global context through a three-stage process: focusing on task-critical regions, scanning for complementary contextual cues, and refining sparse representations via aggregation. By jointly considering visual saliency, conditional global coverage, and redundancy-aware refinement, FSR preserves both query-relevant evidence and holistic scene information under strict token constraints.

Extensive experiments across diverse model architectures, input resolutions, and image–video benchmarks demonstrate that FSR consistently achieves a superior accuracy–efficiency trade-off compared to prior methods. These results highlight the effectiveness of human-inspired local– global coordination as a general paradigm for efficient multimodal inference, and position FSR as a practical solution for deploying large-scale VLMs under real-world resource constraints.

> 💡 **结论批注**:
> - 论文没有显式讨论 limitations，但可以识别几个：
>   1. Focus 阶段依赖 CLIP text encoder → 对非 CLIP 架构需适配（已在 Qwen 实验中体现）
>   2. CCS 是 greedy 的，虽有 2-近似保证但非最优
>   3. 超参数（α,β,ρ,κ）虽固定但可能对某些 task 分布不是最优
>   4. 未讨论 batch inference 场景（不同样本 K_F 不同→padding/ragged tensor）
>   5. Refine 阶段的 weighted merge 会改变 token 的语义，对 LLM 的影响未深入分析
