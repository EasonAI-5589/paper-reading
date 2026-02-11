# Mem-T: Densifying Rewards for Long-Horizon Memory Agents

| 项目 | 信息 |
|------|------|
| **作者** | Yanwei Yue, Guibin Zhang, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, Yan Zhang |
| **机构** | PKU, NUS, NTU |
| **日期** | 2025-01 (arXiv: 2601.23014) |
| **链接** | [arXiv](https://arxiv.org/abs/2601.23014) · [GitHub](https://github.com/yanweiyue/Mem-T) · [HuggingFace](https://huggingface.co/EdwinYue/Mem-T-4B) |
| **关键词** | Memory Agent, Hierarchical Memory, Dense Reward, RL, GRPO, Tree Search |
| **批读日期** | 2026-02-08 |

## 一句话总结

Mem-T 提出层次化记忆架构（Working/Factual/Experiential/Raw）+ MoT-GRPO 树引导强化学习训练范式，将稀疏终端奖励转化为密集逐步信号，联合优化记忆构建与检索，在 4B 模型上超越 gpt-4o-mini 驱动的 SOTA 系统。

## 核心贡献

1. **统一层次化记忆框架 Mem-T** — 集成四种记忆（Working/Factual/Experiential/Raw）× 三种操作（Formation/Evolution/Retrieval），是首个全覆盖且全可训练的记忆 Agent
2. **MoT-GRPO 树引导优化** — 通过 Memory Operation Tree 做分支 rollout，实现节点级 reward backpropagation + 双尺度 advantage estimation，解决长序列 credit assignment 问题
3. **Hindsight Credit Assignment** — 从下游检索效果反向归因到上游记忆构建操作，实现 construction 和 retrieval 的联合优化

## Section 导航

| Section | 文件 | 要点 |
|---------|------|------|
| Abstract | [00-abstract.md](notes/00-abstract.md) | 问题定义 + 方法概述 |
| 1. Introduction | [01-introduction.md](notes/01-introduction.md) | Memory Agent 演化、credit assignment 挑战、三大贡献 |
| 2. Related Work | [02-related-work.md](notes/02-related-work.md) | Memory 架构分类 + RL for Memory Agents |
| 3. Method | [03-method.md](notes/03-method.md) | Mem-T 工作流 + MoT-GRPO (Retrieval & Construction) |
| 4. Experiments | [04-experiments.md](notes/04-experiments.md) | LoCoMo SOTA + OOD 泛化 + 消融 + Case Study |
| 5. Conclusion | [05-conclusion.md](notes/05-conclusion.md) | 总结与展望 |
| Appendix | [06-appendix.md](notes/06-appendix.md) | 数据集细节 + 训练配置 + 敏感性分析 + Training Curves |

## 关键数字

| 指标 | 数值 |
|------|------|
| LoCoMo F1 (Mem-T 4B) | **58.65** (vs 次优 Mem0 43.71, +14.92) |
| LoCoMo F1 (Mem-T 8B) | **58.53** |
| HotpotQA F1 (OOD) | **66.35** (vs Mem-α 58.80) |
| LongMemEval Acc (OOD) | **65.80** |
| NarrativeQA F1 (OOD) | **30.29** |
| 推理 token 节省 (vs GAM) | **~24.45%** |
| 训练步数 | 200 步 (retrieval RL) + 10k ops (construction SFT) |
| Base Model | Qwen3-4B / Qwen3-8B |
| 训练树参数 | G=3 树, depth=4, N_ν=3 分支节点 |

## 我的评价

**优势**：
- MoT-GRPO 是解决 long-horizon sparse reward 的优雅方案，比 flat GRPO 信息量更大但比 MCTS 轻量
- 四层记忆 + 双阶段训练的系统设计非常完整，消融实验证明每个组件都有贡献
- 4B 模型超 gpt-4o-mini 说明训练范式 > base model 能力，对小模型部署很有启发
- OOD 泛化结果有说服力，说明学到的是通用记忆管理策略

**局限**：
- Evidence Alignment Gate 依赖 ground-truth evidence annotation，实际部署场景未必有
- 记忆数据库无限增长的问题没有讨论（遗忘/压缩机制？）
- Construction training 用的是 offline SFT 而非 on-policy RL，不是真正端到端
- 8B 模型没有比 4B 更好，暗示可能存在记忆系统设计上的瓶颈

---

## BibTeX

```bibtex
@article{yue2026memt,
  author       = {Yanwei Yue and
                  Guibin Zhang and
                  Boci Peng and
                  Xuanbo Fan and
                  Jiaxin Guo and
                  Qiankun Li and
                  Yan Zhang},
  title        = {Mem-T: Densifying Rewards for Long-Horizon Memory Agents},
  journal      = {CoRR},
  volume       = {abs/2601.23014},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2601.23014},
  eprinttype   = {arXiv},
  eprint       = {2601.23014}
}
```
