# EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models

**作者**: Michael Wornow*, Rahul Thapa*, Ethan Steinberg, Jason A. Fries†, Nigam H. Shah†  
**机构**: Stanford University (CS + BMIR + Stanford Healthcare)  
**会议**: NeurIPS 2023 (Datasets and Benchmarks Track)  
**链接**: [arXiv](https://arxiv.org/abs/2307.02028) | [Website](https://ehrshot.stanford.edu) | [GitHub](https://github.com/som-shahlab/ehrshot-benchmark)

## 一句话总结

Stanford 发布首个纵向 EHR benchmark（6,739 patients, 41.6M events）+ 临床 FM（CLMBR-T-base, 141M）+ 15 个 few-shot 临床预测任务，为 EHR 领域的 FM 评估提供了可复现的基础设施。

## 核心贡献

1. **EHRSHOT 数据集**：6,739 patients 的完整纵向 EHR（非仅 ICU），平均 2.3x events / 95.2x encounters per patient vs MIMIC-IV
2. **CLMBR-T-base 模型**：141M 参数 autoregressive transformer，在 2.57M patients 上预训练，首批公开权重的结构化 EHR FM
3. **15 个 few-shot 任务**：4 类任务（运营、实验室、新诊断、胸片），自然低标签率，系统评估 FM 的 sample efficiency
4. **完整可复现流程**：数据(DUA) + 模型权重 + 代码 + OMOP-CDM 标准，端到端可复现

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 三大贡献概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机：EHR 数据/FM 缺失 + Figure 1 整体流程 |
| [02 - Related Work](sections/02-related-work.md) | 现有 EHR 数据集/benchmark 对比 + Table 1 |
| [03 - Dataset](sections/03-dataset.md) | **核心** - 数据源、cohort 构造、15 个任务定义 |
| [04 - Baseline Models](sections/04-baseline-models.md) | Count-based GBM vs CLMBR-T-base 架构 |
| [05 - Results](sections/05-results.md) | Few-shot 评估结果 + Figure 3 |
| [06 - Discussion](sections/06-discussion.md) | 局限性与社会影响 |
| [07 - Conclusion](sections/07-conclusion.md) | 总结 |
| [08 - Appendix](sections/08-appendix.md) | 详细任务定义、数据格式、模型细节、完整结果图 |

## 关键数字

| 指标 | 数值 |
|------|------|
| EHRSHOT 患者数 | 6,739 |
| 总临床事件 | 41.6M |
| 总 encounters | 921,499 |
| 预训练患者数 | 2.57M (源库 3.67M) |
| 模型参数 | 141M |
| Hidden dim | 768 |
| 任务数 | 15 (9 binary + 5 multiclass + 1 multilabel) |
| vs MIMIC-IV events/patient | **2.3x** |
| vs MIMIC-IV encounters/patient | **95.2x** |
| 预训练时间 | ~4 days on 1x V100 |
| 词表大小 | 65,536 codes |
| Context window | 5,952 events (496 × 12 layers) |

## BibTeX

```bibtex
@inproceedings{wornow2023ehrshot,
  title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models},
  author={Wornow, Michael and Thapa, Rahul and Steinberg, Ethan and Fries, Jason A. and Shah, Nigam H.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```
