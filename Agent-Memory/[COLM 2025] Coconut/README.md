# Training Large Language Models to Reason in a Continuous Latent Space (Coconut)

**作者**: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian  
**机构**: FAIR at Meta, UC San Diego  
**会议**: ICLR 2025  
**链接**: [arXiv 2412.06769](https://arxiv.org/abs/2412.06769) | [Code](https://github.com/facebookresearch/coconut)

## 一句话总结

Coconut 让 LLM 在连续 latent space（而非离散 token 空间）做推理——用 last hidden state 作为 "continuous thought" 直接反馈为下一步输入，训练时用多阶段课程从 CoT 渐进替换为 latent thought，推理时涌现出 BFS-like 的多路径探索能力。

## 核心贡献

1. **Continuous Thought 范式**：将 LLM 的 last hidden state 直接作为下一步输入 embedding（不经过 LM head 解码），让推理脱离语言空间约束
2. **多阶段课程训练**：从完整 CoT 出发，逐阶段用 continuous thought 替换 language reasoning step，利用语言链监督引导 latent reasoning 学习
3. **BFS-like 推理涌现**：continuous thought 能同时编码多个候选推理路径，模型自发展现出类似广度优先搜索的行为（非贪心），在需要规划的任务上超越 CoT
4. **ProsQA 数据集**：新提出的 DAG 结构逻辑推理数据集，比 ProntoQA 更需要搜索和规划能力
5. **效率优势**：在逻辑推理任务上，Coconut 用更少的 token 达到比 CoT 更高的准确率

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：Coconut 核心思想概览 |
| [01 - Introduction](sections/01-introduction.md) | 动机：语言空间推理的根本限制 + Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | CoT 推理 + Latent reasoning 相关工作 |
| [03 - Method](sections/03-method.md) | Coconut 方法详解：架构、训练、推理 + Figure 2 |
| [04 - Latent Tree Search](sections/04-latent-tree-search.md) | ProsQA 实验 + BFS 涌现分析 (核心分析章节) |
| [05 - Experiments](sections/05-experiments.md) | GSM8k/ProntoQA/ProsQA 综合实验 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与展望 |
| [07 - Appendix](sections/07-appendix.md) | 数据集构造、更多实验 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Base Model | GPT-2 (主实验) |
| GSM8k Acc | 34.1% (Coconut) vs 42.9% (CoT) vs 16.5% (No-CoT) |
| ProntoQA Acc | 99.8% (Coconut) vs 98.8% (CoT) |
| ProsQA Acc | 97.0% (Coconut) vs 77.5% (CoT) |
| Token 效率 | ProsQA: 14.2 tokens (Coconut) vs 49.4 (CoT) |
| 训练阶段数 | 逻辑推理 6+1 stages, GSM8k 3+1 stages |
| Llama 3-8B GSM8k | 43.6% (Coconut) vs 42.2% (No-CoT) |

## 与 MemGen/VisMem 的关系

Coconut 是 **latent reasoning** 的理论基础：
- **Coconut**: hidden state → 下一步推理输入（latent reasoning chain）
- **MemGen**: 把 latent thought 扩展为 **latent memory**——推理过程中产生的 hidden state 可以跨 episode 保存和复用
- **VisMem**: 进一步扩展到视觉模态，用 latent token 作为短期/长期视觉记忆

共同思路：**不在离散文本空间做中间表示，而是直接在连续向量空间操作**。

## BibTeX

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```
