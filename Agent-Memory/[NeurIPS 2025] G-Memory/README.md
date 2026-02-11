# G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems

**作者**: Guibin Zhang\*, Muxin Fu\*, Guancheng Wan, Miao Yu, Kun Wang†, Shuicheng Yan†  
**机构**: NUS, Tongji University, UCLA, A\*STAR, NTU  
**会议**: NeurIPS 2025 Spotlight | **年份**: 2025  
**链接**: [arXiv 2506.07398](https://arxiv.org/abs/2506.07398) | [GitHub](https://github.com/bingreeky/GMemory)

## 一句话总结

G-Memory 是一个面向多 Agent 系统的层次化图记忆架构，通过 insight graph（高层洞察）、query graph（查询关联）、interaction graph（细粒度对话轨迹）三层图结构组织和检索协作经验，使 MAS 具备跨试次自我进化能力，在 embodied action 上提升高达 20.89%，knowledge QA 提升 10.12%。

## 核心贡献

1. **瓶颈识别**：系统梳理现有 MAS 记忆机制，指出其过于简化（仅保留最终结果/缺乏跨试次记忆）是阻碍 MAS 自我进化的根本原因
2. **三层图记忆架构**：提出 insight graph + query graph + interaction graph 的层次化结构，分别存储高层策略洞察、任务元信息及关联、细粒度 agent 对话轨迹
3. **双向记忆遍历**：向上检索 generalizable insights，向下提取 condensed interaction subgraphs，为每个 agent 提供 role-specific 的记忆
4. **Plug-and-play**：无需修改原始 MAS 框架，在 AutoGen/DyLAN/MacNet 三个框架、五个 benchmark、三种 LLM 上均有效
5. **Token 高效**：相比其他记忆方案，G-Memory 在更少 token 消耗下获得更大性能提升

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：三层图记忆 + 双向遍历 + 主要结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机：MAS 缺乏自进化能力 + Figure 1 架构概览 |
| [02 - Related Work](sections/02-related-work.md) | 单 Agent 记忆、MAS 记忆、MAS 框架 |
| [03 - Preliminary](sections/03-preliminary.md) | MAS 形式化 + 三层图的数学定义 |
| [04 - G-Memory](sections/04-g-memory.md) | 核心方法：粗粒度检索 → 双向遍历 → 层次更新 |
| [05 - Experiment](sections/05-experiment.md) | 5 benchmark × 3 LLM × 3 MAS，主实验 + 成本分析 + 消融 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 局限 |
| [07 - Appendix](sections/07-appendix.md) | 实验细节 + 额外结果 + Prompt 模板 + Discussion |

## 关键数字

| 指标 | 数值 |
|------|------|
| Embodied action 最大提升 | +20.89% (ALFWorld, MacNet+Qwen-14b) |
| Knowledge QA 最大提升 | +10.12% (HotpotQA, AutoGen+Qwen-14b) |
| MAS 框架 | AutoGen, DyLAN, MacNet |
| LLM Backbone | GPT-4o-mini, Qwen-2.5-7b, Qwen-2.5-14b |
| Benchmark | ALFWorld, SciWorld, PDDL, HotpotQA, FEVER |
| Token 开销 | 仅 ~1.4×10⁶ 额外 token (vs MetaGPT-M 的 2.2×10⁶) |

## 💡 与 MemGen 的关系

G-Memory 和 MemGen 是同一作者（Guibin Zhang, NUS Shuicheng Yan 组）的两个互补工作：
- **G-Memory**：token-level 显式记忆，面向**多 Agent** 系统，图结构组织协作轨迹
- **MemGen**：latent-level 隐式记忆，面向**单 Agent**，生成式记忆嵌入推理过程
- 共同点：都关注 cross-trial 记忆、都超越了简单 RAG 式检索

## BibTeX

```bibtex
@inproceedings{zhang2025gmemory,
  author       = {Guibin Zhang and
                  Muxin Fu and
                  Guancheng Wan and
                  Miao Yu and
                  Kun Wang and
                  Shuicheng Yan},
  title        = {G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems},
  booktitle    = {Advances in Neural Information Processing Systems ({NeurIPS})},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.07398},
  eprinttype   = {arXiv},
  eprint       = {2506.07398}
}
```
