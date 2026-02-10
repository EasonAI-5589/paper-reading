# Towards Efficient Multimodal Large Language Models: A Survey on Token Compression

> **论文信息**
> - **标题**: Towards Efficient Multimodal Large Language Models: A Survey on Token Compression
> - **作者**: Linli Yao*, Long Xing*, Yang Shi* 等 (北京大学, 中科大, 南洋理工, 中科院, 港大, 微软, 阿里云, 国防科大, 快手)
> - **发表**: Journal of LaTeX Class Files, November 2025
> - **资源**: [GitHub](https://github.com/yaolinli/MLLM-Token-Compression)

---

## 目录导航

| 编号 | 章节 | 笔记文件 | 状态 |
|------|------|--------|------|
| 1 | Introduction | [01-introduction.md](sections/01-introduction.md) | ✅ |
| 2 | Preliminaries | [02-preliminaries.md](sections/02-preliminaries.md) | ✅ |
| 3 | Where to Compress Tokens in MLLMs | [03-where-to-compress.md](sections/03-where-to-compress.md) | ✅ |
| 4 | How to Select the Desirable Strategy | [04-how-to-select.md](sections/04-how-to-select.md) | ✅ |
| 5 | Benchmarks and Metrics | [05-benchmarks-metrics.md](sections/05-benchmarks-metrics.md) | ✅ |
| 6 | Application Scenarios | [06-applications.md](sections/06-applications.md) | ✅ |
| 7 | Open Challenges and Future Work | [07-challenges-future.md](sections/07-challenges-future.md) | ✅ |
| 8 | Conclusion | [08-conclusion.md](sections/08-conclusion.md) | ✅ |

---

## 一句话总结

本survey按**压缩位置**（Vision Encoder / Projector / LLM / Hybrid）和**压缩机制**（pruning / merging / fusion / query-based）两个维度，系统梳理了MLLM视觉token压缩领域的50+代表性工作，并提供了实用的策略选择路线图。

## 核心分类体系 (Quick Reference)

```
Token Compression in MLLMs
├── Vision Encoder (§3.1)
│   ├── Inside-Encoder: Token Dropping / Merging / Multi-Scale
│   └── Outside-Encoder: Purely-Vision / Text-guided
├── Projector (§3.2)
│   ├── Transformation-Based: Pooling / Pixel Shuffle / Convolution
│   ├── Query-Based: Q-Former / Cross-Attention
│   └── Importance-Driven: Similarity / Saliency / Novel Metrics
├── LLM (§3.3)
│   ├── Prefilling: Importance / Learnable / Merging / Fusion
│   └── Decoding: KV-cache Compression
└── Hybrid (§3.4)
    ├── Collaborative Compression
    └── Progressive Compression
```

## 策略选择决策树 (Quick Reference)

```
How to Select Strategy (§4)
├── §4.1 视频时序增强压缩 (Fixed / Dynamic / Hybrid)
├── §4.2 Purely-Visual vs Text-guided (互补，先视觉后文本)
├── §4.3 Token Merging vs Dropping (soft聚合 vs hard丢弃)
├── §4.4 Plug-in vs Re-training (易部署 vs 高性能)
└── §4.5 Efficient Training vs Efficient Inference
```

---

## 个人思考与笔记

<!-- 在此添加你的个人理解、灵感、与自己工作的关联等 -->

### 与我的研究的关联
- TODO: 填写

### 值得深入阅读的论文
- TODO: 填写

### 可能的改进方向
- TODO: 填写

---

*最后更新: 2025-02*
