# MemEvolve: Meta-Evolution of Agent Memory Systems

**作者**: Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou†, Shuicheng Yan†  
**机构**: OPPO AI Agent Team, LV-NUS lab  
**日期**: 2025.12.23 | **链接**: [arXiv 2512.18746](https://arxiv.org/abs/2512.18746) | [Code](https://github.com/bingreeky/MemEvolve)

## 一句话总结

提出 MemEvolve 元进化框架，不仅让 Agent 积累经验，还能自动搜索最优记忆架构（encode/store/retrieve/manage 四组件），在 4 个 benchmark 上提升 Flash-Searcher 等框架高达 17.06%，并展现跨任务/跨模型/跨框架泛化能力。

## 核心贡献

1. **EvolveLab 统一代码库**：将 12 种主流自进化记忆系统分解为 (Encode, Store, Retrieve, Manage) 四组件模块化设计空间，提供统一实现和评测
2. **MemEvolve 元进化框架**：双层优化 — 内层积累经验，外层通过 Diagnose-and-Design 搜索更优记忆架构，形成良性循环
3. **强实验验证**：Flash-Searcher 提升至 74% (xBench)、74.71% (WebWalkerQA)；在 TaskCraft 上进化的架构可直接迁移到其他 benchmark/LLM/框架

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 & 2 |
| [01 - Introduction](sections/01-introduction.md) | 动机：固定记忆架构的局限 + 自适应学习者类比 |
| [02 - Related Work](sections/02-related-work.md) | LLM Agent 系统 + Agent Memory 架构综述 |
| [03 - EvolveLab](sections/03-evolvelab.md) | 形式化 + 四组件模块化设计空间 + 统一代码库 |
| [04 - MemEvolve](sections/04-memevolve.md) | 双层进化 + Diagnose-and-Design 进化算子 |
| [05 - Experiments](sections/05-experiments.md) | 四大 benchmark 实验 + 跨域泛化 + 进化动态分析 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 贡献者 |
| [07 - Appendix](sections/07-appendix.md) | EvolveLab 实现细节 + 数据集 + 进化记忆系统可视化 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 实现的记忆系统 | 12 种 (Voyager, ExpeL, AWM, G-Memory 等) |
| 模块化组件 | 4 个 (Encode, Store, Retrieve, Manage) |
| 进化轮数 | K_max = 3, 每轮保留 Top-1 扩展 3 个后代 |
| 最大提升 | +17.06% (Kimi K2 on WebWalkerQA) |
| 评测 Benchmark | GAIA, WebWalkerQA, xBench-DS, TaskCraft |
| 跨任务迁移增益 | 2.0–9.09% (TaskCraft → 其他) |

## 与相关工作的关系

| 论文 | 关系 |
|------|------|
| **G-Memory** (同作者 Guibin Zhang) | EvolveLab 中 12 个基线之一；图 + 层次化记忆 |
| **MemGen** (同作者) | 生成式隐式记忆，MemEvolve 是更上层的架构搜索 |
| **AgentKB** | MemEvolve 进化的起点之一 |
| **ExpeL / AWM / Voyager** | EvolveLab 中的经典基线 |

## BibTeX

```bibtex
@misc{zhang2025memevolve,
  title={MemEvolve: Meta-Evolution of Agent Memory Systems},
  author={Zhang, Guibin and Ren, Haotian and Zhan, Chong and Zhou, Zhenhong and Wang, Junhao and Zhu, He and Zhou, Wangchunshu and Yan, Shuicheng},
  year={2025},
  eprint={2512.18746},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
