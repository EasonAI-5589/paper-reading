[← 返回 README](../README.md)

# 7 Conclusion

## 📌 预览
简短总结三大贡献，呼吁社区走向更可复现的开放模型开发。

---

We publish EHRSHOT, a benchmark containing the structured data of 6,739 patients' full longitudinal medical timelines specifically geared towards few-shot evaluation of foundation models for clinical data. Unlike most prior work, EHRSHOT contains longitudinal health data rather than a single department (e.g. ICU). We define a set of 15 tasks ranging from well-studied outcomes like 30-day readmission to lesser explored settings such as anticipating abnormal lab values. Finally, we release the weights of a foundation model pretrained on over 2.57M patient timelines and publish the code needed to replicate our results. We hope that this work represents a first step towards moving the field of ML for healthcare towards more reproducible and open model development.

> 💡 **结语**: EHRSHOT 是一个 "first step"——作者很清楚这不是终点。后续的 Context Clues (2024) 就在 EHRSHOT 基础上进一步研究了 FM 的上下文学习能力，证明了这个 benchmark 的持续价值。

---

## 🔖 Section 总结
- EHRSHOT = 纵向 EHR 数据 + 15 个 few-shot 任务 + 公开 FM 权重 + 完整代码
- 核心愿景：推动 healthcare ML 走向可复现和开放
