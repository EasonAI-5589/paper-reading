[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览

总结 HoloV 的核心贡献和局限性。

---

We present HoloV, a holistic token pruning framework that addresses two critical limitations of attention-based visual compression: 1) semantic fragmentation from over-pruning non-salient regions, and 2) static importance estimation ignoring token interdependencies. The core innovation lies in variance-modulated dynamic scoring and capacity-constrained allocation, which preserve holistic context. Extensive experiments validate our method's effectiveness in maintaining both perceptual details and abstract spatial reasoning capabilities under aggressive token reduction.

> 💡 **总结批注**: HoloV 解决的两个核心问题：
> 1. **语义碎片化**: 过度剪枝非显著区域导致全局语义断裂
> 2. **静态重要性估计**: 现有方法忽略 token 间的相互依赖关系
>
> 核心技术创新：**方差调制动态评分** (diversity variance + [CLS] attention) + **容量约束分配** (crop-wise adaptive allocation)

---

**Acknowledgments and Disclosure of Funding**

This work was supported by the National Natural Science Foundation of China (Grant No.62506318); Guangdong Provincial Department of Education Project (Grant No.2024KQNCX028); CAAI-Ant Group Research Fund; Scientific Research Projects for the Higher-educational Institutions (Grant No.2024312096), Education Bureau of Guangzhou Municipality; Guangzhou-HKUST(GZ) Joint Funding Program.

---

**Limitations and Future Work.** HoloV demonstrates robust performance in preserving holistic visual context but faces two key limitations: its dependence on fixed spatial crop partitioning may hinder fine-grained semantic capture in complex scenes, and minor accuracy declines persist even at high pruning ratios (e.g., $4.2\%$ drop when pruning $88.9\%$ visual tokens). To address these, future work could prioritize adaptive crop, sparse attention, multi-modality extensions (e.g., 3D data), and integration with hallucination mitigation, while optimizing for edge computing energy efficiency.

> 💡 **局限性和未来方向**:
> - **局限 1**: 固定空间 crop 划分不够灵活，复杂场景下可能影响细粒度语义捕获
> - **局限 2**: 高剪枝率下仍有 4.2% 性能下降
> - **未来方向**: 自适应 crop、稀疏注意力、3D 数据扩展、幻觉缓解集成、边缘计算优化

---

## 🔖 Section 总结

### 核心洞察
1. HoloV 的成功归因于两个核心创新：diversity variance 评分 + crop-wise 分配
2. 方法虽然简单（plug-and-play、training-free），但理论上有保证，实验上一致最优
3. 固定 crop 划分是最大局限，自适应 crop 是最有价值的未来方向
