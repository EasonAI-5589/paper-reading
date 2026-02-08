# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

| 元信息 | |
|---|---|
| **标题** | MemGen: Weaving Generative Latent Memory for Self-Evolving Agents |
| **作者** | Guibin Zhang†, Muxin Fu†, Shuicheng Yan (NUS) |
| **会议** | ICLR 2026 |
| **arXiv** | [2509.24704](https://arxiv.org/abs/2509.24704) |
| **GitHub** | [KANABOON1/MemGen](https://github.com/KANABOON1/MemGen) |
| **日期** | 2025-10-14 |

## 一句话总结

MemGen 提出动态生成式隐式记忆框架：通过 RL 训练的 Memory Trigger 监测推理状态决定何时激活记忆，Memory Weaver 将经验生成为 latent token 序列无缝织入推理过程，在不修改 LLM 参数的前提下超越参数化记忆和检索式记忆，并自发涌现出 planning/procedural/working memory 三种人类记忆层级。

## 核心贡献

1. **推理-记忆交织框架**：首次实现 token 级粒度的推理-记忆动态交织，Memory Trigger（RL 训练的 LoRA）+ Memory Weaver（生成式 LoRA）两个轻量模块协同工作，reasoner 参数完全冻结
2. **跨范式兼容性**：Weaver 可用 SFT 或 GRPO 训练，兼容任意 LLM backbone，且可与检索式记忆（ExpeL 等）无缝集成
3. **涌现式记忆层级**：无需显式监督，latent memory 自发分化为 planning memory（高层规划）、procedural memory（工具使用/格式）、working memory（上下文一致性），通过 post-hoc intervention 实验验证
4. **全面实验验证**：9 个 benchmark × 3 个 backbone，超 ExpeL/AWM 最多 38.22%，超 GRPO 最多 13.44%，且展现强跨域泛化和持续学习能力

## Section 导航

| Section | 文件 | 内容 |
|---|---|---|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要、论文定位、一句话总结 |
| §1 Introduction | [01-introduction.md](sections/01-introduction.md) | 研究动机、三种记忆范式对比、MemGen 核心思想 |
| §2 Related Work | [02-related-work.md](sections/02-related-work.md) | Agent Memory、Latent Computation、LLM Decoding & RL |
| §3 Preliminary | [03-preliminary.md](sections/03-preliminary.md) | 符号定义、问题形式化、记忆调用粒度 |
| §4 Methodology | [04-methodology.md](sections/04-methodology.md) | MemGen 完整方法：Trigger + Weaver + 检索集成 |
| §5 Experiments | [05-experiments.md](sections/05-experiments.md) | 主实验、泛化、持续学习、记忆层级分析 |
| §6 Conclusion | [06-conclusion.md](sections/06-conclusion.md) | 总结与展望 |
| Appendix | [07-appendix.md](sections/07-appendix.md) | 优化算法细节、超参数、额外实验、Latent Token 示例 |

---

## BibTeX

```bibtex
@inproceedings{zhang2025memgen,
  title={MemGen: Weaving Generative Latent Memory for Self-Evolving Agents},
  author={Guibin Zhang and Muxin Fu and Shuicheng Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
