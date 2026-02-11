# Memory in the Age of AI Agents: A Survey

## 元信息

| 项目 | 内容 |
|------|------|
| **标题** | Memory in the Age of AI Agents: A Survey — Forms, Functions and Dynamics |
| **作者** | Yuyang Hu†, Shichun Liu†, Yanwei Yue†, Guibin Zhang†ò, Boyang Liu, ... (50+ 作者) |
| **机构** | NUS, 人大, 复旦, 北大, NTU, 同济, UCSD, HKUST(GZ), Griffith, Georgia Tech, OPPO, Oxford |
| **类型** | Survey |
| **arXiv** | [2512.13564](https://arxiv.org/abs/2512.13564) |
| **GitHub** | [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) |
| **日期** | 2025 |

## 一句话总结

提出 **Forms–Functions–Dynamics** 三角分类框架，系统梳理 agent memory 研究：按 **form**（token-level / parametric / latent）、**function**（factual / experiential / working）、**dynamics**（formation / evolution / retrieval）三个维度组织现有工作，并明确了 agent memory 与 LLM memory、RAG、context engineering 的边界。

## 核心贡献

1. **多维分类框架**: 提出 Forms–Functions–Dynamics 三角，超越传统 long/short-term 二分法
2. **概念边界厘清**: 系统对比 Agent Memory vs. LLM Memory / RAG / Context Engineering
3. **形式化定义**: 统一的 memory lifecycle 抽象（Formation F / Evolution E / Retrieval R 三算子）
4. **前沿方向**: 八大 frontier — retrieval→generation、automated management、RL+memory、multimodal、multi-agent shared memory、world model memory、trustworthy memory、human-cognitive connections
5. **资源汇编**: 25+ benchmarks + 25+ open-source frameworks 的全面对比

## Section 导航

| Section | 文件 | 主题 | 关键内容 |
|---------|------|------|----------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要与目录 | 论文元信息、作者、摘要、目录结构 |
| §1 | [01-introduction.md](sections/01-introduction.md) | Introduction | 研究动机、新分类法的必要性、五个核心问题、贡献 |
| §2 | [02-preliminaries.md](sections/02-preliminaries.md) | Preliminaries | 形式化定义（agent/memory）、AM vs LLM Memory/RAG/CE |
| §3 | [03-memory-forms.md](sections/03-memory-forms.md) | Form | Token-level (1D/2D/3D) / Parametric / Latent memory |
| §4 | [04-memory-functions.md](sections/04-memory-functions.md) | Functions | Factual / Experiential / Working memory |
| §5 | [05-memory-dynamics.md](sections/05-memory-dynamics.md) | Dynamics | Formation / Evolution / Retrieval 全生命周期 |
| §6 | [06-resources.md](sections/06-resources.md) | Resources | 25+ benchmarks + 25+ frameworks 对比 |
| §7 | [07-frontiers.md](sections/07-frontiers.md) | Frontiers | 8 大前沿方向 |
| §8 | [08-conclusion.md](sections/08-conclusion.md) | Conclusion | 总结与展望 |
| Refs | [09-references.md](sections/09-references.md) | References | 完整参考文献 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 正文页数 | ~76 页 |
| 参考文献 | ~400+ 篇 |
| 图表 | 11 Figures + 9 Tables |
| Memory Forms | 3 (Token-level / Parametric / Latent) |
| Memory Functions | 3 (Factual / Experiential / Working) |
| Memory Dynamics | 3 (Formation / Evolution / Retrieval) |
| Benchmarks 汇总 | 25+ |
| Frameworks 汇总 | 25+ |
| 前沿方向 | 8 个 |

## 核心框架图

![Figure 1: Overview of agent memory taxonomy](images/59c7dcb89b84c5659faf913c40baa21d0d721fb0004a4a3bb8b6dfab62df4dc9.jpg)

*Figure 1: Forms–Functions–Dynamics 三角分类总览。Memory artifacts 按其主要 form 和 function 定位，并映射了代表性系统。*

---

## BibTeX

```bibtex
@article{hu2025memory,
  author       = {Yuyang Hu and Shichun Liu and Yanwei Yue and Guibin Zhang and
                  Boyang Liu and Fangyi Zhu and Jiahang Lin and Honglin Guo and
                  Shihan Dou and Zhiheng Xi and Senjie Jin and Jiejun Tan and
                  Yanbin Yin and Jiongnan Liu and Zeyu Zhang and Zhongxiang Sun and
                  Yutao Zhu and Hao Sun and Boci Peng and Zhenrong Cheng and
                  Xuanbo Fan and Jiaxin Guo and Xinlei Yu and Zhenhong Zhou and
                  Zewen Hu and Jiahao Huo and Junhao Wang and Yuwei Niu and
                  Yu Wang and Zhenfei Yin and Xiaobin Hu and Yue Liao and
                  Qiankun Li and Kun Wang and Wangchunshu Zhou and Yixin Liu and
                  Dawei Cheng and Qi Zhang and Tao Gui and Shirui Pan and
                  Yan Zhang and Philip Torr and Zhicheng Dou and Ji{-}Rong Wen and
                  Xuanjing Huang and Yu{-}Gang Jiang and Shuicheng Yan},
  title        = {Memory in the Age of {AI} Agents},
  journal      = {CoRR},
  volume       = {abs/2512.13564},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2512.13564},
  doi          = {10.48550/ARXIV.2512.13564},
  eprinttype   = {arXiv},
  eprint       = {2512.13564}
}
```
