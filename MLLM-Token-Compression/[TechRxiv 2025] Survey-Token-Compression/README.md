# Towards Efficient MLLMs: A Survey on Token Compression

**作者**: Linli Yao, Long Xing, Yang Shi, Sida Li, Yuanxin Liu, et al. (PKU, USTC, NTU, etc.)  
**来源**: TechRxiv 2025  
**链接**: [GitHub](https://github.com/yaolinli/MLLM-Token-Compression)

## 一句话总结

系统性综述 MLLM 中的 **Token 压缩技术**，按架构位置分类（Vision Encoder / Projector / LLM / Hybrid），提供方法选择路线图，并讨论四大未解决挑战。

## 核心贡献

1. **架构位置分类体系**: 按 VE / Projector / LLM / Hybrid 系统组织 50+ 方法
2. **方法选择路线图**: 5 个决策维度（时序 / Visual vs. Text-guided / Merging vs. Dropping / Plug-in vs. Retrain / Training vs. Inference）
3. **开放挑战识别**: 理论缺失、自适应性不足、细粒度任务性能下降、评估标准不统一

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义 + Survey 结构概览 |
| [01 - Introduction](sections/01-introduction.md) | 动机（三个核心问题）+ 与已有 survey 的区别 + 贡献 |
| [02 - Preliminaries](sections/02-preliminaries.md) | MLLM 三组件架构 + 计算复杂度 + Token Compression 形式化 |
| [03 - Where to Compress](sections/03-methods.md) | ⭐ 核心分类：VE / Projector / LLM / Hybrid 四大类方法详解 |
| [04 - How to Select](sections/04-how-to-select.md) | ⭐ 选择路线图：5 个决策维度 + 视频时序压缩专题 |
| [05 - Benchmarks & Metrics](sections/05-experiments.md) | 评测 benchmark（图像 14 个 + 视频 8 个）+ 效率指标 |
| [06 - Applications](sections/06-applications.md) | 应用场景：医学影像、文档、遥感、具身 AI、流式视频 |
| [07 - Challenges](sections/07-challenges.md) | 四大挑战：理论 / 自适应 / 性能下降 / 评估 |
| [08 - Conclusion](sections/08-conclusion.md) | 总结：三个演进维度 + 剩余挑战 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 收录方法数 | 50+ |
| 分类维度 | 4 个压缩位置 × 5 个决策维度 |
| 常用 Benchmark | 图像 14 个 + 视频 8 个 |
| Visual tokens 比 Text tokens 多 | ~20× |
| 自然场景最低 tokens | 9 tokens/image (M3) |
| OCR 任务需要 tokens | 144-576 tokens/image |

---

*3号机批读 @ 2026-02-08*
