# Titans: Learning to Memorize at Test Time

**作者**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
**来源**: arXiv 2501.00663 | **年份**: 2024
**链接**: [arXiv](https://arxiv.org/abs/2501.00663)

## 一句话总结

提出 **神经长期记忆模块**（Neural Long-term Memory），作为 meta in-context learner 在测试时学习记忆，并基于此设计 Titans 架构族（MAC/MAG/MAL 三种变体），在语言建模、常识推理、基因组学和时间序列任务上超越 Transformer 和现代线性循环模型，且能扩展到 2M+ 上下文窗口。

## 核心贡献

1. **神经长期记忆 (Neural LTM)**: 用 surprise metric（梯度）+ momentum + weight decay 的在线学习框架，将历史信息压缩到深层 MLP 参数中，测试时仍可持续更新
2. **三种记忆融合架构**: MAC（记忆作为上下文，注意力决定存储什么）、MAG（记忆与滑窗注意力门控融合）、MAL（记忆作为层，与注意力串联）
3. **可并行化训练**: 将 momentum + weight decay 的梯度下降重写为 matmul 操作，支持 chunk-wise 并行扫描
4. **Persistent Memory**: 学习任务级知识的数据无关参数，缓解注意力的 initial token bias
5. **表达能力**: 证明 Titans 超越 TC⁰，理论上比 Transformer 和大多数线性循环模型更强

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + Memory Perspective + 贡献路线图 |
| [02 - Preliminaries](sections/02-preliminaries.md) | Attention、Linear Attention、现代线性 RNN 的记忆视角 |
| [03 - Neural Memory](sections/03-neural-memory.md) | §3.1 长期记忆设计 + §3.2 并行化 + §3.3 Persistent Memory |
| [04 - Titans Architectures](sections/04-titans-architectures.md) | MAC / MAG / MAL 三种变体 + 架构细节 |
| [05 - Experiments](sections/05-experiments.md) | 语言建模、NIAH、BABILong、时间序列、DNA、消融实验 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型规模 | 170M / 340M / 400M / 760M |
| 训练数据 | FineWeb-Edu 15B-30B tokens |
| 上下文扩展 | > 2M tokens |
| S-NIAH 16K (MAC) | 98.4 / 97.4 / 95.2 (PK/N/W) |
| BABILong | 超越 GPT-4（参数量少 ~70x） |
| 时间序列 | 7/7 数据集 SOTA |
