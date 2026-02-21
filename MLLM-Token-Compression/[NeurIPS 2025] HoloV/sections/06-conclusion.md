[← 返回 README](../README.md)

# 6 Conclusion

---

We present HoloV, a holistic token pruning framework that addresses two critical limitations of attention-based visual compression: 1) semantic fragmentation from over-pruning non-salient regions, and 2) static importance estimation ignoring token interdependencies. The core innovation lies in variance-modulated dynamic scoring and capacity-constrained allocation, which preserve holistic context. Extensive experiments validate our method's effectiveness in maintaining both perceptual details and abstract spatial reasoning capabilities under aggressive token reduction.

> 💡 **总结**:
> 两个关键创新：
> 1. **Variance-modulated dynamic scoring**: 用语义多样性（variance）+ [CLS] attention 混合评分，替代纯 attention 评分
> 2. **Capacity-constrained allocation**: crop-wise 自适应分配剪枝配额，保证空间覆盖
> 
> 这两个设计共同解决了 attention-first 方法的两个核心问题：语义碎片化和静态重要性估计。

---

## 💡 个人总评

### 优点
1. **问题分析深入**: Section 3 的三个分析（信息冗余、位置偏置、注意力分散）非常透彻，Random vs FastV 实验尤其精彩
2. **方法简洁有效**: Crop-wise 分配 + variance 评分，没有复杂的训练或优化，工程上非常友好
3. **实验全面**: 3 种架构 × 3 种剪枝率 × 9+ baseline × 12 benchmark，覆盖面极广
4. **高剪枝率鲁棒**: 88.9% 剪枝下的 95.8% 性能保留是非常强的数字

### 不足/疑问
1. **与 CDPruner 的对比不够充分**: CDPruner（同为 NeurIPS 2025）也强调 diversity，两者的核心理念相似但实现不同（DPP vs variance）。论文中 CDPruner 没有出现在 Table 1 中，可能因为发表时间接近
2. **Visual Context Refetching 的消融缺失**: 这个机制在主实验中是否启用？对性能贡献多大？论文没有清楚说明
3. **Qwen 实验只和 FastV 比**: 缺少与 DART、HiRED 等更强 baseline 的对比
4. **Variance 评分的计算开销**: 需要计算 O(M²) 的 pair-wise similarity，当 crop 内 token 数较多时可能不可忽略。虽然 Table 4 显示总体开销可控
5. **τ 参数未消融**: crop allocation 的 sharpness 参数 τ 的影响没有讨论

### 与领域内其他工作的关系
- **vs CDPruner**: 都关注 diversity，但 CDPruner 用 DPP（更数学化），HoloV 用 variance（更直观）。CDPruner 是 instruction-centric，HoloV 是 vision-centric
- **vs DART (EMNLP25)**: DART 是 Table 1 中最强 baseline，两者差距在 1-2%。DART 的方法和 HoloV 有什么不同？值得深入了解
- **vs SCOPE (NeurIPS 2025)**: SCOPE 也关注"saliency + coverage"的平衡，和 HoloV 的思路有相似之处
- **整体趋势**: 2024-2025 年的 token pruning 方法从"纯 attention 评分"转向"多样性/覆盖度保证"，HoloV 是这个趋势的代表作之一
