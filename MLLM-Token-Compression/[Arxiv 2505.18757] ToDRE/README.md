# ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance

## 元信息

| 项目 | 内容 |
|------|------|
| **标题** | ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance |
| **作者** | Duo Li, Zuhao Yang, Xiaoqin Zhang, Ling Shao, Shijian Lu |
| **机构** | NTU Singapore, ZJUT China, Terminus AI Lab / UCAS |
| **ArXiv** | [2505.18757](https://arxiv.org/abs/2505.18757) |
| **提交日期** | 2025-05-24 (v2: 2025-11-19) |
| **关键词** | Visual Token Pruning, LVLM Efficiency, Token Diversity, Task Relevance, Training-free |

## 一句话总结

ToDRE 将 visual token 冗余分解为 **intra-modal diversity** 和 **cross-modal task relevance** 两个正交因素，Stage 1 用 greedy max-sum diversification 在 embedding space 保留多样性子集（90% pruning），Stage 2 利用 information migration 现象在 LLM decoder 深层移除全部剩余 visual tokens，实现 2.6× 加速、95.0% 性能保持。

## 核心贡献

1. **正交分解洞察**：证明 token diversity 与 task relevance 是正交因素，应分阶段分别处理，优于单一指标（attention / similarity）
2. **Stage 1 — Greedy Max-Sum Diversification**：在 LLM embedding space 中用贪心算法选择最大多样性子集，规避 attention 的 positional bias
3. **Stage 2 — Information Migration Pruning**：发现 cross-modal attention 在 LLM decoder 后半段显著衰减，自适应选层后移除全部 visual tokens
4. **Training-free & Plug-and-play**：兼容 FlashAttention，在 LLaVA-NeXT / Qwen2.5-VL / InternVL2 上均有效

## 章节导航

| 章节 | 文件 | 要点 |
|------|------|------|
| Abstract | [sections/00-abstract.md](sections/00-abstract.md) | 问题定义与总结 |
| 1. Introduction | [sections/01-introduction.md](sections/01-introduction.md) | 动机、information migration 发现 |
| 2. Related Work | [sections/02-related-work.md](sections/02-related-work.md) | LVLM & token compression 文献 |
| 3. Preliminary Analysis | [sections/03-preliminary-analysis.md](sections/03-preliminary-analysis.md) | FLOPs 分析、冗余分类 |
| 4. Method | [sections/04-method.md](sections/04-method.md) | 核心算法：diversity selection + relevance pruning |
| 5. Experiments | [sections/05-experiments.md](sections/05-experiments.md) | 12 benchmarks, 4 LVLMs |
| 6. Conclusion | [sections/06-conclusion.md](sections/06-conclusion.md) | 总结 |

## Citation Landscape

### 本文核心对比方法
- **FastV** [Chen et al., 2024]: attention-based pruning in LLM decoder，有 positional bias
- **FasterVLM** [Zhang et al., 2024]: [CLS] attention in embedding space，attention 过于集中
- **SparseVLM** [Zhang et al., 2024]: attention-based，迁移性差（7B→13B 性能崩塌）
- **DivPrune** [Alvar et al., 2025]: diversity-based pruning，最接近的 baseline（ToDRE 在其基础上加了 Stage 2）
- **VTW** [Lin et al., 2025]: KL-divergence 选层，需要 calibration set
- **GlobalCom2** [Liu et al., 2025]: 仅支持图像输入

### 关键引用
- **ToMe** [Bolya et al., 2022]: vision encoder 内 token merging（bipartite soft matching）
- **FlashAttention** [Dao et al., 2022]: ToDRE 与其兼容
- **Max-sum diversification** [Gollapudi & Sharma, VLDB 2009]: Stage 1 算法来源

### 下游关联
- ToDRE 的 diversity + relevance 分离思路可与 STAR-Pro 的 R+λD 融合策略形成对比
