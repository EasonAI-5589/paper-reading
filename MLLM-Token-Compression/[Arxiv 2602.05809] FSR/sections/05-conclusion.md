[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 FSR 的贡献和实验结论。

---

In this paper, we propose FSR, a training-free visual token pruning framework inspired by human visual perception, which addresses the fundamental challenge of allocating a limited token budget in VLMs. FSR explicitly models the progressive coordination between local evidence and global context through a three-stage process: focusing on task-critical regions, scanning for complementary contextual cues, and refining sparse representations via aggregation. By jointly considering visual saliency, conditional global coverage, and redundancy-aware refinement, FSR preserves both query-relevant evidence and holistic scene information under strict token constraints.

Extensive experiments across diverse model architectures, input resolutions, and image–video benchmarks demonstrate that FSR consistently achieves a superior accuracy–efficiency trade-off compared to prior methods.

> 💡 **Conclusion 批读**:
> - 论文没有 Limitations 部分，这是一个遗憾
> - **未讨论的局限**:
>   1. CLIP text encoder 依赖——限制了对 Qwen2.5-VL 等新架构的完全适用性
>   2. ρ=0.9 固定阈值——是否真正 "dynamic" 值得商榷
>   3. 在极端压缩（90%+）下性能下降仍然明显
>   4. 没有与 training-based 方法比较
> - **FSR 的核心价值**: 将 pruning 问题重新框架化为 local/global 动态分配，这个思路本身有启发性

---

## 🔖 Section 总结

### FSR 整体评价
**优势**:
- 三阶段设计清晰直觉，有认知科学 motivation
- 动态 budget 分配是真正的创新点
- 实验覆盖面广，在多数设置下稳定领先
- CCS 有 2-approximation 理论保证
- Training-free, plug-and-play

**不足**:
- CLIP text encoder 依赖是最大硬伤
- 相对 CDPruner 的提升幅度不大（通常 <1%）
- Refine 阶段贡献有限，尤其在大模型上
- 没有讨论 limitations
