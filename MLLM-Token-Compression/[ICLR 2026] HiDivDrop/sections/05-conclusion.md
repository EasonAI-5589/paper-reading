[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结HiDivDrop的三大贡献，以及对MLLM层级计算分配的新理解。

---

In summary, our study challenges prevailing assumptions about visual processing in MLLMs and demonstrates that shallow layers only act as passive propagators for visual tokens. By introducing HiDivDrop with Late Injection, Concave Pyramid Pruning, and Early Exit, we align pruning with the true hierarchical dynamics of multimodal integration. Our findings not only achieve state-of-the-art efficiency–accuracy trade-offs, but also provide new insights into how MLLMs allocate computation across layers, paving the way for more principled and scalable multimodal architectures.

> 💡 **结论批读**:
> - 本文的贡献不仅是方法层面的（HiDivDrop框架），更是认知层面的（对MLLM层级功能的新理解）
> - 浅层 = 传播者、中层 = 稀疏融合中心、深层 = 语言推理——这个三层结构是universal的
> - 未来方向：
>   1. 将HiDivDrop扩展到更多MLLM架构（如Qwen2-VL等）
>   2. 结合Pre-LLM压缩（如connector层面的压缩）
>   3. 探索不同任务下的动态层级划分

---

## 🔖 整篇论文的最终总结

### HiDivDrop vs 现有方法

| 方法 | 压缩方式 | 关注点 | 压缩率 | 性能保持 |
|------|---------|--------|--------|---------|
| FastV | 单次early pruning | 简单高效 | 86% | 87.9% |
| PDrop | 均匀progressive | 渐进减少 | 47% | 100.2% |
| TwigVLM | 两阶段 | twig block辅助 | 89% | 95.3% |
| **HiDivDrop** | **三段式层级** | **层级功能对齐** | **89%** | **98.3%** |

### 与STAR-Pro的互补关系
| 维度 | HiDivDrop | STAR-Pro |
|------|-----------|----------|
| 核心问题 | WHERE — 在哪些层做什么 | WHAT — 用什么indicator选token |
| 层级理解 | shallow/middle/deep三段 | 未显式区分层级 |
| Token选择 | DTop-K (attention-based) | Star indicator (multi-criteria) |
| 训练方式 | PT+FT端到端 | indicator训练 |
| 互补可能 | 可在HiDivDrop框架中替换DTop-K为STAR indicator | 可借鉴Late Injection思路 |
