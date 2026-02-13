# Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

**作者**: Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu  
**机构**: Alibaba Group / Wuhan University  
**arXiv**: [2601.01885](https://arxiv.org/abs/2601.01885) | **日期**: 2026-01-05  
**简称**: AgeMem

## 一句话总结

将 LTM 和 STM 管理统一为 agent 的 tool-based actions，通过三阶段渐进式 RL + step-wise GRPO 端到端训练，在 5 个 long-horizon benchmark 上超越所有 memory-augmented baseline。

## 核心贡献

1. **统一记忆框架 AgeMem**：将 LTM（ADD/UPDATE/DELETE）和 STM（RETRIEVE/SUMMARY/FILTER）操作暴露为 tool interface，让 agent 自主决定何时、如何管理记忆
2. **三阶段渐进式 RL 训练**：Stage 1 学 LTM 构建 → Stage 2 学 STM 噪声过滤 → Stage 3 联合记忆协调推理；context 在 Stage 1→2 重置防信息泄漏
3. **Step-wise GRPO**：将终端奖励广播到整条轨迹所有 step，解决 memory 操作导致的稀疏/不连续奖励问题
4. **多维度奖励设计**：$R_{task}$（任务完成）+ $R_{context}$（压缩/预防/保留）+ $R_{memory}$（存储质量/维护/语义相关性）+ penalty
5. **全面实验验证**：5 个 benchmark（ALFWorld/SciWorld/PDDL/BabyAI/HotpotQA），2 个 backbone（Qwen2.5-7B/Qwen3-4B），平均提升 49.59%/23.52%

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + Figure 1 框架对比 + 三大挑战 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | LTM / STM / RL for LLMs 三条线 |
| [03 - Method](sections/03-method.md) | 问题定义 + 6 个 memory tool + 三阶段 RL + Step-wise GRPO + 奖励函数 |
| [04 - Experiments](sections/04-experiments.md) | 主实验 + Memory Quality + STM 有效性 + Tool 使用分析 + 消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + Limitations |
| [06 - Appendix](sections/06-appendix.md) | Tool 实现细节 + 奖励公式 + 算法伪代码 + Case Study + 数据集/实现细节 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Avg. performance (Qwen2.5-7B) | 41.96% (+49.59% vs no-memory) |
| Avg. performance (Qwen3-4B) | 54.31% (+23.52% vs no-memory) |
| RL 训练提升 | +8.53 / +8.72 pp over AgeMem-noRL |
| Memory Quality (MQ) | 0.533 / 0.605 (best) |
| STM token 节省 | 3.1% / 5.1% vs RAG |
| Memory tools | 6 个 (ADD/UPDATE/DELETE/RETRIEVE/SUMMARY/FILTER) |
| 训练数据 | 仅 HotpotQA，零样本迁移到其余 4 个 benchmark |
| 硬件 | 8× RTX 4090 (48GB) |

## 与相关论文的关系

| 对比论文 | 关键区别 |
|----------|----------|
| **MemSkill** (2602.02474) | MemSkill 用 skill bank 管 memory 操作（Controller+Executor+Designer），AgeMem 直接把 memory 操作作为 tool action |
| **Mem-T** (2601.23014) | 都用 RL 训练 memory 管理，但 AgeMem 统一了 LTM+STM，Mem-T 只管层次化 LTM |
| **MemGen** (ICLR 2026) | MemGen 是 latent memory（hidden state），AgeMem 是 explicit text memory（tool-based） |
| **Memory-R1** | 都是 RL-based memory management，AgeMem 多了 STM 管理和三阶段训练 |

## BibTeX

```bibtex
@article{yu2026agentic,
  title={Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents},
  author={Yu, Yi and Yao, Liuyi and Xie, Yuexiang and Tan, Qingquan and Feng, Jiaqi and Li, Yaliang and Wu, Libing},
  journal={arXiv preprint arXiv:2601.01885},
  year={2026}
}
```
