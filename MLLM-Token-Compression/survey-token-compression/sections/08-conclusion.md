# 8. Conclusion

## 📄 原文

> MLLMs represent a significant advancement in cross-modal understanding, yet computational efficiency remains a critical bottleneck.
>
> ==MLLMs 在跨模态理解上取得重大进展，但计算效率仍是关键瓶颈==

> Token compression emerges as a promising solution by reducing redundancy across MLLM components, enhancing both training and inference efficiency while alleviating long-context reasoning complexity.
>
> ==Token Compression 是有前景的解决方案：减少冗余、提升效率、缓解长上下文推理复杂度==

---

## 领域演进

> The field has evolved from:
> - single-module to **multi-module compression**
> - fixed-rate to **adaptive dynamic approaches**
> - static images to **complex video sequences**
>
> ==演进方向：单模块 → 多模块 / 固定率 → 自适应 / 静态图像 → 复杂视频==

---

## 仍存在的挑战

> However, key challenges persist:
> 1. **The absence of unified evaluation frameworks** for token compression
> 2. **Limited integration with mainstream training or inference acceleration libraries**
> 3. **Insufficient synergy with other MLLM efficiency techniques**
>
> ==关键挑战：缺乏统一评估框架、与主流加速库集成有限、与其他效率技术协同不足==

---

## 本综述的贡献

> This survey provides a systematic foundation for advancing efficient, scalable, and practically deployable multimodal large language models through strategic token compression methodologies.
>
> ==本综述为高效、可扩展、可部署的 MLLM 提供系统性基础==

---

## 💡 全文总结

### 核心内容回顾

| 章节 | 核心内容 |
|------|----------|
| §1 Introduction | 问题定义：视觉 tokens 爆炸导致 O(n²) 复杂度 |
| §2 Preliminaries | MLLM 架构 (VE → Projector → LLM) + 两种冗余 |
| §3 Where to Compress | 按位置分类：VE (Inside/Outside) / Projector / LLM / Hybrid |
| §4 How to Select | 5 个选择维度的决策指南 |
| §5 Benchmarks | 图像/视频 Benchmarks + 效果/效率指标 |
| §6 Applications | 医学影像、文档、遥感、具身智能、流式视频等 |
| §7 Challenges | 缺乏理论、缺乏自适应、细粒度任务性能下降、评估不统一 |

### 方法论框架

```
MLLM Token Compression
├── Where to Compress
│   ├── Vision Encoder
│   │   ├── Inside-VE (Dropping/Merging/Multi-Scale)
│   │   └── Outside-VE (Purely-Visual/Text-Guided)
│   ├── Projector (Transformation/Query/Importance)
│   ├── LLM (Prefilling/Decoding)
│   └── Hybrid (Collaborative/Progressive)
│
└── How to Select
    ├── Temporal-Enhanced (Fixed/Dynamic/Hybrid)
    ├── Purely-Visual vs. Text-Guided
    ├── Merging vs. Dropping
    ├── Plug-in vs. Re-training
    └── Training vs. Inference
```

### 关键 Takeaways

1. **视觉冗余极高**：自然场景只需 ~9 tokens/图像，OCR 需要 144-576
2. **Attention Bias 问题**：位置偏差 + 显著区域过度关注 → 用 similarity 替代
3. **Merging + Dropping 互补**：密集输入用 Merging，稀疏语义用 Dropping
4. **混合策略最有效**：早期 plug-in 压缩 → 中期 re-training 精炼 → 后期 KV cache 剪枝
5. **未来方向**：理论基础、自适应压缩、细粒度任务优化、统一评估

---

*[返回论文目录](../README.md)*

---

*笔记由 3号机 📚 整理*
*2026-02-06*
