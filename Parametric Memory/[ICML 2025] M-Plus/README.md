# M+: Extending MemoryLLM with Scalable Long-Term Memory

**作者**: Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gutfreund, Rogério Feris, Zexue He
**会议**: ICML 2025 | **年份**: 2025
**链接**: [arXiv](https://arxiv.org/abs/2502.00592) | [GitHub](https://github.com/wangyu-ustc/MemoryLLM)

## 一句话总结

在 MemoryLLM 基础上增加 **长期记忆 (LTM)**：将被 random drop 的 memory tokens 存入 CPU 端长期记忆池，配合 co-trained retriever 在生成时检索，将知识保留范围从 <20k 扩展到 **160k+ tokens**，GPU 开销不变。

## 核心贡献

1. **Long-Term Memory (LTM)**: 将 MemoryLLM 丢弃的 memory tokens 存入 CPU 端 LTM（最大 150k tokens），而非永久丢弃
2. **Co-trained Retriever**: 两层 MLP 的 query/key projector，在 hidden state 空间做检索，每层只检索一次（而非每个 head 每层检索），大幅降低延迟
3. **Multi-LoRA Design**: 更新过程和生成过程各用一套 LoRA 权重（写/读分离）
4. **三阶段 Data Curriculum**: MemoryLLM 续训 → 长文档训练 → LTM 训练
5. **知识保留从 20k → 160k+**: 在 Long Book QA 和 Event QA 上大幅超越基线

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | Token-level vs Latent-space memory + M+ 动机 |
| [02 - Related Work](sections/02-related-work.md) | Token-level 和 Latent-space memory 综述 |
| [03 - Methodology](sections/03-methodology.md) | Memory 结构 + Retriever + Multi-LoRA + Data Curriculum |
| [04 - Experiments](sections/04-experiments.md) | Long Book QA、知识保留、消融实验 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 基座模型 | Llama-3.1-8B |
| Short-term memory | 10,240 tokens/层 |
| LTM 检索 | 2,560 tokens/层 |
| LTM 最大容量 | 150k tokens |
| 知识保留范围 | 160k+ tokens (vs MemoryLLM <20k) |
| 训练 | 8× A100, deepspeed-stage-2 |
