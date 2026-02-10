[← 返回 README](../README.md)

# 6 Discussion

## 📌 预览
讨论 EHRSHOT 的价值定位、5 个局限性、社会影响。

---

We believe that EHRSHOT represents a useful contribution to the ML community by enabling more reproducible healthcare ML research. The release of our pretrained CLMBR-T-base model's weights will allow the community to replicate and build upon our work. Our results identify opportunities for improving pretrained models in few-shot settings.

Acquiring labeled EHR data is expensive and time-consuming. Additionally, certain rare conditions may only be present in a small cohort of patients out of millions within a health system [29]. Thus, model performance in low-label settings is of paramount importance in healthcare. As our results in Section 5 demonstrate, pretrained FMs can yield large performance gains in few-shot settings. While we acknowledge that the tasks themselves may not be the most clinically meaningful, we believe that EHRSHOT offers a valuable contribution by providing a reproducible and rigorous point of comparison for different technical approaches to developing clinical FMs.

> 💡 **坦诚的定位**: 作者承认任务本身不一定是最有临床意义的，EHRSHOT 的价值在于提供 **可复现的技术对比平台**。这很务实——先解决"能不能比"的问题，再解决"比什么"的问题。

---

### Limitations

There are several limitations to this work.

First, we only release structured data – i.e. we do not publish any of the clinical text or images associated with our patients. While many datasets for medical images exist [3], publishing clinical text remains a challenge [41].

Second, we only consider one type of foundation model (CLMBR-T-base) for our experiments [42]. We look forward to seeing the additional foundation models that the community applies to our benchmark.

Third, we release a very small cohort of patients (< 1%) from our source EHR database, and specifically select these patients for the tasks that we define. Releasing our full pretraining dataset would be infeasible from a governance and effort perspective. Thus, while necessary in order to publish our EHR dataset and still broader than existing ICU-specific datasets, our cohort selection process limits the types of questions we can answer and does not reflect the full diversity of medical data.

Fourth, as we only were able to evaluate our pretrained FM on Stanford Medicine data, it is unclear how well our pretrained model will perform at other institutions. We anticipate there will be some drop in performance, but the extent is unclear.

Fifth, several of our tasks are "low label" in the most extreme sense – for example, the Celiac task only has 13 positive patients in its test set. This makes obtaining low variance estimates of model performance difficult. We aim to mitigate this by adding additional patients to our benchmark in future releases.

> 💡 **5 个局限性总结**:
> 1. **只有 structured data**（无文本/影像）→ 无法评估多模态 FM
> 2. **只测了一个 FM** → 需要社区贡献更多模型
> 3. **Cohort bias**: <1% patients 被选入，且为任务定制选择 → 不代表总体分布
> 4. **单站点**: 仅 Stanford，跨院迁移性未知（这是 Context Clues 后续要解决的问题）
> 5. **极端低标签**: Celiac test 只有 13 positive → 评估方差大

---

### Societal Implications

We believe that the release of this dataset can help spur positive innovations for improving clinical care with ML. However, we recognize that there are patient privacy concerns anytime EHR data is released. We believe we sufficiently mitigate this risk through the rigorous deidentification process on which our data is subjected [5]. Additionally, we gate access to the dataset through a research data use agreement. Another concern is that models trained on biased data will reflect those biases [7]. Thus, the pretrained FM that we release may propagate biases in care delivery or outcomes present in our source EHR database [7]. However, we hope that by encouraging the full release of models, we can help the community better identify and mitigate these issues [26].

> 💡 **隐私与偏见**: 两个核心社会影响问题。隐私通过去标识化 + DUA 缓解。偏见问题更深——Stanford 的患者群体不代表美国总体（Table 4 显示 White 55%, Asian 15%, Black 4%），模型可能继承这些偏见。

---

## 🔖 Section 总结

### 核心洞察
1. EHRSHOT 定位是 **技术对比平台**，不是临床决策工具
2. 单站点 + 小 cohort + 无文本 是主要局限，为后续工作留出空间
3. 跨站点泛化是最重要的开放问题——这正是 Context Clues 要解决的
