# Nested Learning: The Illusion of Deep Learning Architectures

**作者**: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)
**会议**: NeurIPS 2025 | **年份**: 2025
**链接**: [arXiv](https://arxiv.org/abs/2512.24695)

## 一句话总结

提出 **Nested Learning (NL)** 范式，将深度学习模型统一表示为**嵌套的多层优化问题**，揭示优化器（SGD/Adam等）本质是联想记忆模块，并基于此设计 Self-Modifying Learning Module 和 Continuum Memory System (CMS)，实现 Hope 持续学习架构。

## 核心贡献

1. **Nested Learning 范式**: 将模型+优化器统一表示为嵌套的多层优化问题，每个组件有自己的 "context flow"
2. **优化器 = 联想记忆**: 证明 SGD+momentum、Adam 等优化器本质是压缩梯度信息的联想记忆模块；Adam 是 element-wise L2 回归目标的最优联想记忆
3. **Self-Modifying Learning Module**: 能学习自身更新算法的序列模型（自我引用/self-referential）
4. **Continuum Memory System (CMS)**: 泛化传统"长期/短期记忆"为**连续频谱**的多时间尺度记忆系统
5. **Hope 架构**: 结合 self-modifying module + CMS 的持续学习模型，在语言建模、知识注入、few-shot 泛化上表现优异

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机（LLM 的静态性 = 顺行性遗忘）+ 神经科学启发 + 贡献路线图 |
| [02 - Core Theory](sections/02-core-theory.md) | NL 范式 + 优化器即联想记忆 + CMS 设计 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 论文长度 | ~80 页（含附录）|
| 关键发现 | Adam = 最优 element-wise L2 联想记忆 |
| CMS 频率 | 从 Gamma (fast) 到 Delta (slow)，连续频谱 |
| 评估任务 | 持续学习、长上下文、语言建模、in-context recall |
