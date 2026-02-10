[← 返回 README](../README.md)

# 4 Baseline Models

## 📌 预览
两个 baseline：(1) Count-based GBM — 简单但强的传统方法，(2) CLMBR-T-base — 141M autoregressive transformer FM。重点关注 CLMBR-T-base 的架构设计。

---

We measure the performance of two baseline models on our dataset: (1) a gradient boosting machine (GBM) that uses count-based featurizations of patients to make predictions, (2) an autoregressive language model ("CLMBR-T-base") that ingests medical codes as tokens and was pretrained on the full longitudinal structured EHRs of 2.57M patients from our source institution [42, 8].

> 💡 **为什么选这两个 baseline**: 一个代表传统 ML（count features + tree model），一个代表 FM（pretrained transformer）。对比这两者能直接回答"预训练到底有没有用"。

We chose these two models as our baselines for several reasons. First, language modeling has achieved state-of-the-art results on clinical prediction tasks [42, 33, 25, 29, 20], while count-based featurization remains a simple but competitive baseline [32, 33, 42]. Second, most prior FMs trained on structured EHR data have not had their model weights published, and were developed and tested exclusively on nonstandard data formats like MIMIC-III [49]. This makes it nearly impossible to conduct a fair comparison of prior models, which often requires re-implementation or significant modification to work across datasets [13]. This is one of the key challenges we are attempting to solve with EHRSHOT. We pre-train our own FM from scratch to have full control over its training, and publish its model weights so the community can reproduce and build upon our results.

> 💡 **不做模型对比的原因**: 不是不想比，而是 **没法比**——其他 structured EHR FM 要么没公开权重，要么用非标准数据格式。这恰恰是 EHRSHOT 要解决的问题。

---

### Count-based Features (GBM)

Count-based featurization is a well-established baseline for EHR tasks, valued for its simplicity and effectiveness [32]. The fundamental idea involves converting each patient's timeline into a count vector, where each element contains the number of occurrences of a specific medical concept prior to the prediction time of a task. These patient vectors are combined into a count matrix, which is high-dimensional and sparse. We use a technique called ontology expansion to increase the density of representation and improve the accuracy of code coverage by acknowledging the parent/child hierarchical relationships between medical concepts [4]. After generating our ontology-expanded count matrix, we train a gradient boosting machine (GBM) model on the EHRSHOT train split, and tune hyperparameters on the validation split. We use the LightGBM implementation [18]. We also evaluate a Logistic Regression and Random Forest model as baselines. Their results can be seen in Appendix in Figures 10 and 11. For clarity, we exclude them from the following analyses, as they perform roughly at par with the count-based GBM model.

> 💡 **Count-based GBM 要点**:
> - 每个患者 → 一个计数向量（维度 = 所有唯一医疗编码数，值 = 该编码在 prediction time 前出现次数）
> - **Ontology expansion** 是关键技巧：利用 OMOP 编码层次结构（如 ICD10 E10.1 → E10 → E08-E13），让稀疏向量变密
> - LightGBM 训练，LR 和 RF 效果差不多所以省略
> - 这个 baseline 虽然简单，但在高标签量场景下非常有竞争力

---

### CLMBR-T-base

Clinical Language-Model-Based Representations using Transformers (CLMBR-T-base). CLMBR-T-base is an autoregressive model designed to predict the next medical code in a patient's timeline given previous codes. This objective enables it to learn robust global patterns for clinical prediction tasks. It is based on the CLMBR model originally developed in [42], but following [8] we substitute a transformer in place of a GRU as its base model. Our model employs causally masked local attention. This ensures forward-only flow of information which is vital for prediction tasks, and is in contrast to BERT-based models which are bidirectional in nature [42]. Note that our model does not process clinical text, only structured information. Our model has 141M trainable parameters, a hidden dimension of 768, and a next code prediction objective. This provides our version of CLMBR-T-base with minute-level resolution rather than the day-level aggregation of the original model formulation [42]. We leave training larger versions of CLMBR to future work.

> 💡 **CLMBR-T-base 架构要点**:
> ```
> 输入: 患者时间线 [code_1, code_2, ..., code_n]（每个 code = 诊断/药物/lab 等）
> 模型: Causal Transformer (12 layers, hidden=768, local attention window=496)
> 目标: Next code prediction（类 GPT）
> 输出: 每个时间步的 d-维 patient representation
> ```
> - **Causal masking** = 单向注意力，只看过去不看未来，适合时序预测
> - **Local attention** (window=496): 每层只看最近 496 个 events，12 层叠加 → 有效窗口 5,952 events
> - **Minute-level resolution**: 原始 CLMBR 用 GRU + day-level aggregation，这个版本用 transformer + 保留分钟级精度
> - 141M 参数 ≈ BERT-base 级别（110M），不算大，但对 EHR 来说已经足够
> - **重要**：不处理文本，只处理 structured codes

More details about our baseline models can be found in the Appendix in Section D.

---

## 🔖 Section 总结

### 模型对比

| 特征 | Count-based GBM | CLMBR-T-base |
|------|----------------|--------------|
| 类型 | 传统 ML | Foundation Model |
| 输入 | 计数向量 (sparse) | Code 序列 (temporal) |
| 预训练 | ❌ | ✅ 2.57M patients |
| 参数量 | 少 (tree model) | 141M |
| 时序信息 | ❌ (bag-of-codes) | ✅ (autoregressive) |
| 下游适配 | 从头训练 GBM | Frozen backbone + LR head |

### 核心洞察
1. **GBM 丢掉了时序信息**：count vector 是 bag-of-codes，不关心事件顺序
2. **CLMBR-T-base 保留时序**：autoregressive 建模捕捉 "先诊断 A → 再用药 B" 这种模式
3. **下游评估方式不同**：GBM 在 few-shot examples 上从头训练；CLMBR-T-base 冻结 backbone，只训练 LR head
