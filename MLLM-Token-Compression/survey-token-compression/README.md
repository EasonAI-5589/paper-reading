# 📄 Towards Efficient Multimodal Large Language Models: A Survey on Token Compression

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Towards Efficient Multimodal Large Language Models: A Survey on Token Compression |
| **作者** | Linli Yao, Long Xing, Yang Shi 等 |
| **机构** | 北京大学、中科大、南洋理工等 |
| **发布** | 2025年12月 (TechRxiv v1.0) |
| **类型** | Survey |
| **论文链接** | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1) |
| **GitHub** | [yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression) |

## 一句话总结

关于 **MLLM 视觉 Token 压缩** 的全面综述，系统梳理了 100+ 篇论文，按压缩位置（Vision Encoder / Projector / LLM）和压缩策略进行分类，并提供了方法选择指南。

## 章节目录

| # | 章节 | 文件 | 状态 |
|---|------|------|------|
| 0 | Abstract | [00-abstract.md](sections/00-abstract.md) | ✅ |
| 1 | Introduction | [01-introduction.md](sections/01-introduction.md) | ✅ |
| 2 | Preliminaries | [02-preliminaries.md](sections/02-preliminaries.md) | ✅ |
| 3 | Where to Compress | [03-where-to-compress.md](sections/03-where-to-compress.md) | ✅ ⭐ |
| 4 | How to Select | [04-how-to-select.md](sections/04-how-to-select.md) | ✅ ⭐ |
| 5 | Evaluation | [05-evaluation.md](sections/05-evaluation.md) | ✅ |
| 6 | Applications | [06-applications.md](sections/06-applications.md) | ✅ |
| 7 | Challenges & Future | [07-challenges.md](sections/07-challenges.md) | ✅ |
| 8 | Conclusion | [08-conclusion.md](sections/08-conclusion.md) | ✅ |

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
5. **训练时在前端压缩，推理时在后端压缩**

---

*阅读笔记 by 3号机 📚*
*首次阅读：2026-02-06*
