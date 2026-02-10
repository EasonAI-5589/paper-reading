[← 返回 README](../README.md)

# Abstract

## 📌 预览
三大贡献：EHRSHOT 数据集（6,739 patients 纵向 EHR）、CLMBR-T-base（141M FM）、15 个 few-shot 临床预测任务。

---

While the general machine learning (ML) community has benefited from public datasets, tasks, and models, the progress of ML in healthcare has been hampered by a lack of such shared assets. The success of foundation models creates new challenges for healthcare ML by requiring access to shared pretrained models to validate performance benefits. We help address these challenges through three contributions.

> 💡 **开篇定位**: Healthcare ML 的核心痛点 = 缺少公共数据集 + 缺少公开的预训练模型。FM 时代让这个问题更突出——你必须有 pretrained model 才能验证 FM 的价值。

First, we publish a new dataset, EHRSHOT, which contains deidentified structured data from the electronic health records (EHRs) of 6,739 patients from Stanford Medicine. Unlike MIMIC-III/IV and other popular EHR datasets, EHRSHOT is longitudinal and not restricted to ICU/ED patients.

> 💡 **贡献 1 - EHRSHOT 数据集**: 关键词是 **longitudinal**（纵向，完整医疗时间线）而非仅 ICU snapshot。6,739 patients 虽然不多，但每个 patient 的数据深度远超 MIMIC。

Second, we publish the weights of CLMBR-T-base, a 141M parameter clinical foundation model pretrained on the structured EHR data of 2.57M patients. We are one of the first to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR. We provide an end-to-end pipeline for the community to validate and build upon its performance.

> 💡 **贡献 2 - CLMBR-T-base**: 重要区分——这是 **structured EHR FM**（处理诊断码、lab 值等结构化数据），不是 clinical NLP 模型。GatorTron/ClinicalBERT 处理的是临床文本，两者互补。2.57M patients 预训练规模可观。

Third, we define 15 few-shot clinical prediction tasks, enabling evaluation of foundation models on benefits such as sample efficiency and task adaptation. Our model and dataset are available via a research data use agreement from our website. Code to reproduce our results is available here.

> 💡 **贡献 3 - 15 个 few-shot 任务**: 专门为评估 FM 的 few-shot 能力设计，而非传统 supervised 场景。这是 Context Clues 等后续工作的评估基础。

---

## 🔖 Section 总结

### 核心洞察
1. Healthcare ML 缺少 ImageNet 级别的公共基础设施（数据 + 模型 + 任务）
2. 结构化 EHR FM 是一个被忽视的方向——大多数公开模型只处理临床文本
3. Few-shot 评估是衡量 FM 价值的关键——标注数据在医疗领域极其昂贵
