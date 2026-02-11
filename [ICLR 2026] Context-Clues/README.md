# Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs

**作者**: Michael Wornow*, Suhana Bedi*, Miguel Angel Fuentes Hernandez, Ethan Steinberg, Jason Alan Fries, Christopher Ré, Sanmi Koyejo, Nigam H. Shah  
**机构**: Stanford University, Prealize Health  
**会议**: ICLR 2026  
**链接**: [GitHub](https://github.com/som-shahlab/long_context_clues) | [HuggingFace Models](https://huggingface.co/)

## 一句话总结

首次系统评估长上下文模型（Mamba/Hyena/GPT/Llama）在 EHR 临床预测任务上的效果，发现 Mamba-16k 在 EHRSHOT 14 个任务中 9 个达到 SOTA，同时揭示 EHR 数据三个独特属性（copy-forwarding、不规则时间间隔、疾病进展）对模型性能的影响。

## 核心贡献

1. **亚二次架构首次大规模 EHR 训练**：在 250 万患者数据上训练 Mamba/Hyena/Llama/GPT，Mamba-16k 在 EHRSHOT 14 个任务中 9 个超过先前 SOTA（CLMBR-t-base），平均 AUROC +0.03
2. **长上下文系统性评估**：首次系统评估上下文长度（512→16k）对 EHR 建模的影响，发现 Mamba 和 Llama 随上下文增长性能提升，但 Hyena 在 4k 后急剧下降
3. **EHR 数据三大独特属性量化分析**：首次定量分析 copy-forwarding（token 重复）、irregular time intervals（不规则时间间隔）、disease progression（疾病进展）对模型的影响，发现长上下文模型对这些属性更鲁棒
4. **开源模型权重和代码**：发布所有预训练模型权重和代码

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机：EHR FM 受限于 512 token，亚二次架构的机会 |
| [02 - Background](sections/02-background.md) | EHR FM 综述 + 长上下文架构 + Related Work |
| [03 - Methods](sections/03-methods.md) | 数据集、Tokenization、4 种架构、EHRSHOT 评估、3 个 EHR 属性定义 |
| [04 - Results](sections/04-results.md) | 核心实验结果：长上下文提升性能 + 3 个属性的影响分析 |
| [05 - Discussion](sections/05-discussion.md) | 讨论 + Limitations + Future Work |
| [06 - Conclusion](sections/06-conclusion.md) | 结论 |
| [07 - Appendix](sections/07-appendix.md) | 数据集详情、评估细节、模型架构、训练配置、EHR 属性指标、Few-shot/Zero-shot 结果 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练数据 | 2.5M 患者, 3.5B 临床事件 |
| 模型参数 | ~120M（统一规模） |
| 上下文长度范围 | 512 → 16,384 tokens |
| 最佳模型 | Mamba-16k, 平均 AUROC 0.807 |
| 超过 SOTA 任务数 | 9/14 (EHRSHOT) |
| AUROC 提升 | +0.03 over CLMBR-t-base (0.777) |
| 词表大小 | 39,818 tokens |
| 评估任务 | 14 个二分类任务（3 类） |
| 不规则性影响 | Q4 vs Q1 Brier loss 高 14% |

## 与 Agent Memory 的关联

这篇论文虽然聚焦 EHR 领域，但核心问题是**长上下文作为记忆机制**的有效性：
- **长上下文 ≈ 完整记忆**：将患者一生的医疗记录塞入上下文窗口，本质上是用上下文长度替代外部记忆
- **亚二次架构 ≈ 高效记忆压缩**：Mamba 的 SSM 将长序列压缩到固定维度的隐状态中，类似 latent memory
- **EHR 数据的特殊挑战**：copy-forwarding（冗余记忆）、irregular intervals（不均匀时间跨度）、disease progression（信息衰减）是所有 agent memory 系统都可能遇到的问题

## BibTeX

```bibtex
@inproceedings{DBLP:journals/corr/abs-2412-16178,
  author       = {Michael Wornow and
                  Suhana Bedi and
                  Miguel Angel Fuentes Hernandez and
                  Ethan Steinberg and
                  Jason Alan Fries and
                  Christopher R{\'{e}} and
                  Sanmi Koyejo and
                  Nigam H. Shah},
  title        = {Context Clues: Evaluating Long Context Models for Clinical Prediction
                  Tasks on EHRs},
  booktitle    = {International Conference on Learning Representations ({ICLR})},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2412.16178},
  doi          = {10.48550/ARXIV.2412.16178},
  eprinttype    = {arXiv},
  eprint       = {2412.16178}
}
```
