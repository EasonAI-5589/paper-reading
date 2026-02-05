# 📄 Towards Efficient Multimodal Large Language Models: A Survey on Token Compression

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Towards Efficient Multimodal Large Language Models: A Survey on Token Compression |
| **作者** | Linli Yao, Long Xing, Yang Shi, Sida Li, Yuanxin Liu 等 (北大、中科大、南洋理工等) |
| **发布** | 2025年12月 (TechRxiv v1.0) |
| **类型** | Survey |
| **论文链接** | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1) |
| **GitHub** | [yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression) |

## 一句话总结

MLLM 视觉 Token 压缩综述，按压缩位置 (Vision Encoder / Projector / LLM) 分类，覆盖 100+ 篇论文，提供方法选择指南。

## 论文结构

| # | 章节 | 笔记 | 状态 |
|---|------|------|------|
| 0 | Abstract | [📝](./sections/00-abstract.md) | ✅ |
| 1 | Introduction | [📝](./sections/01-introduction.md) | ✅ |
| 2 | Preliminaries | [📝](./sections/02-preliminaries.md) | ✅ |
| 3 | Token Compression Methods | [📝](./sections/03-methods.md) | ✅ ⭐ |
| 4 | How to Select | [📝](./sections/04-how-to-select.md) | ✅ ⭐ |
| 5 | Experiments & Benchmarks | [📝](./sections/05-experiments.md) | ✅ |
| 6 | Application Scenarios | [📝](./sections/06-applications.md) | ✅ |
| 7 | Open Challenges | [📝](./sections/07-challenges.md) | ✅ |
| 8 | Conclusion | [📝](./sections/08-conclusion.md) | ✅ |

## 核心分类框架

```
Where to Compress?
├── Vision Encoder (§3.1)
│   ├── Inside-Encoder (Token Dropping/Merging)
│   └── Outside-Encoder (Purely-Vision/Text-guided)
├── Projector (§3.2)
│   ├── Transformation-based (Pooling/PixelShuffle/Conv)
│   ├── Query-based (Q-Former)
│   └── Importance-driven
├── LLM (§3.3)
│   ├── Prefilling Stage
│   └── Decoding Stage (KV Cache)
└── Hybrid (§3.4)
```

## Key Takeaways

1. **Vision tokens 冗余极高** — 自然场景只需 ~9 tokens/image，OCR 需要 144-576
2. **Attention-based 剪枝有位置偏差** — 用 similarity 替代更稳定
3. **Merging vs Dropping 互补** — 混合策略效果最佳
4. **Text-guided 适合单轮，Purely-visual 适合多轮**
5. **混合策略最有效** — 早期 plug-in → 中期 re-training → 后期 KV cache 剪枝

---

*笔记由 3号机 📚 整理*
*首次阅读：2026-02-06*
