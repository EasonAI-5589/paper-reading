# MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents

**作者**: Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, Wenya Wang  
**机构**: Nanyang Technological University, Singapore (王文雅组)  
**arXiv**: [2602.02474](https://arxiv.org/abs/2602.02474) | **日期**: 2026-02-02  
**代码**: [GitHub](https://github.com/ViktorAxelsen/MemSkill)

---

## 一句话总结

将 agent memory 的操作（提取/整合/剪枝）从固定规则变成**可学习、可进化的 skill**，通过 Controller (RL 选 skill) + Executor (LLM 执行) + Designer (LLM 进化 skill bank) 的闭环优化，在对话记忆、多跳问答、具身任务上全面超越 baseline。

---

## 核心贡献

1. **Memory Skills 抽象**: 把 memory 操作表示为结构化、可复用的 skill bank（从 4 个初始原语开始进化）
2. **Skill-conditioned Memory Generation**: Controller 选 Top-K skill + Executor 一次 LLM call 生成 memory，支持 span-level 处理
3. **闭环进化**: RL 训练 skill 选择 + LLM Designer 从 hard cases 进化 skill bank，形成 use → evolve 循环
4. **强泛化**: 跨模型（LLaMA→Qwen）、跨数据集（LoCoMo→LongMemEval/HotpotQA）均有效
5. **可解释性**: 进化出的 skill 可读可检查，自动适配不同领域（对话 vs 具身）

---

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要：skill-based memory 的核心 idea |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 三大设计原则 + Figure 1 对比图 + 贡献列表 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | Agent Memory Systems + Self-Evolving Agents 两条线 |
| 3. Method | [03-method.md](sections/03-method.md) | **核心**：Skill Bank + Controller + Executor + Designer + 闭环优化 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 4 个 benchmark 结果 + Ablation + 分布偏移迁移 + Case Study |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结 + Impact Statement |

---

## 关键数字

| 指标 | MemSkill (LLaMA) | MemSkill (Qwen) | 最强 Baseline |
|------|-------------------|------------------|---------------|
| LoCoMo L-J | 50.96 | 52.07 | A-MEM 48.41 |
| LongMemEval L-J | 59.41 | 59.90 | CoN 56.93 |
| ALF-Seen SR | 47.86 | 60.00 | CoN 57.86 |
| ALF-Unseen SR | 47.01 | 64.18 | CoN 53.73 |

---

## 架构一览

| 组件 | 类型 | 功能 | 是否训练 |
|------|------|------|----------|
| Controller | MLP | 选 Top-K skill (Gumbel-Top-K) | ✅ PPO |
| Executor | LLM (70B) | 执行 skill 生成 memory | ❌ 固定 |
| Designer | LLM | 分析 hard cases → 进化 skill bank | ❌ 固定 |
| Skill Bank | 结构化文本 | 可复用 memory 操作 | 进化（非梯度） |

---

## 跟相关工作的区别

| 方法 | 搜索对象 | 优化方式 |
|------|----------|----------|
| Mem-α / Memory-R1 | 固定操作的选择 | RL |
| MemEvolve | Memory 架构 | 元优化 |
| **MemSkill** | **Memory 操作策略 (skill)** | **RL + LLM 进化** |

---

## BibTeX

```bibtex
@article{zhang2026memskill,
  author    = {Haozhen Zhang and Quanyu Long and Jianzhu Bao and Tao Feng and Weizhi Zhang and Haodong Yue and Wenya Wang},
  title     = {MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents},
  journal   = {CoRR},
  volume    = {abs/2602.02474},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.02474},
  eprinttype = {arXiv},
  eprint    = {2602.02474}
}
```

---

*批读 by 3号机 📚 | 2026-02-13*
