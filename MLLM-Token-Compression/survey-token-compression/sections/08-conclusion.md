# 8. Conclusion

> MLLMs represent a significant advancement in cross-modal understanding, yet **computational efficiency remains a critical bottleneck**.
>
> ==MLLM 是跨模态理解的重大进展，但计算效率仍是关键瓶颈==

> Token compression emerges as a promising solution by reducing redundancy across MLLM components, enhancing both training and inference efficiency while alleviating long-context reasoning complexity.
>
> ==Token Compression 是有前景的解决方案：减少冗余、提升效率、缓解长上下文复杂度==

---

## 领域演进

> The field has evolved from:
> - **single-module to multi-module** compression
> - **fixed-rate to adaptive dynamic** approaches
> - **static images to complex video sequences**
>
> ==三个演进方向：单模块→多模块，固定率→自适应，静态图像→复杂视频==

---

## 持续的挑战

> However, key challenges persist:
>
> 1. The absence of **unified evaluation frameworks** for token compression
> 2. Limited integration with **mainstream training/inference acceleration libraries**
> 3. Insufficient synergy with **other MLLM efficiency techniques**
>
> ==三大持续挑战：统一评估框架缺失、与主流库集成不足、与其他效率技术协同不够==

---

## 本综述贡献

> This survey provides a **systematic foundation** for advancing efficient, scalable, and practically deployable multimodal large language models through strategic token compression methodologies.
>
> ==本综述为高效、可扩展、可实际部署的 MLLM 提供系统性基础==

**核心贡献：**
1. **分类框架 (Taxonomy)**: 按 MLLM 架构位置分类 (VE / Projector / LLM / Hybrid)
2. **方法选择指南 (How to Select)**: 帮助从业者选择最优策略
3. **开放挑战 (Open Challenges)**: 指明未来研究方向

---

## 关键要点回顾

| 维度 | 要点 |
|------|------|
| **Where** | Vision Encoder / Projector / LLM / Hybrid 各有优劣 |
| **How** | Pruning vs Merging, Text-guided vs Purely-Visual, Plug-in vs Re-train |
| **Video** | 时空联合压缩、时间结构保持、超长视频处理 |
| **Challenges** | 理论基础、自适应性、细粒度任务、评估标准 |

---

*综述阅读完成！📚*

*笔记整理 by 3号机*
*2026-02-06*
