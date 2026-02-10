[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
简短结论，呼吁从 NLP 架构的简单迁移转向数据驱动的领域适配。

---

Long context models have unlocked a broad range of natural language applications through their ability to ingest and reason over massive amounts of information. Translating these gains to EHR data could benefit patients by enabling the modeling of an entire lifetime. Thus, we present the first systematic evaluation of how context length impacts EHR modeling. We find that long context subquadratic models such as Mamba are capable of achieving state-of-the-art results on clinical prediction tasks. This represents a sharp break from prior work in EHR FMs, as shown in Table 1, which generally utilized BERT-based models limited to context windows of 512 tokens. We also find that longer context models are more robust to three distinct aspects of EHR data that had been underexplored in prior literature on sequence modeling. We hope our work inspires future efforts to identify interesting sequence modeling challenges from non-NLP domains and encourages further research towards applying non-transformer architectures to structured EHR data.

> 💡 **论文的深层贡献**:
> 1. **技术层面**: Mamba-16k EHR SOTA，打破 BERT-512 的范式
> 2. **方法论层面**: 建立了"领域属性分析"的 framework——迁移 NLP 架构时，先分析目标领域数据的独特属性
> 3. **对 Agent Memory 的启示**: 长上下文作为记忆机制是有效的，但需要考虑领域数据的特殊性（冗余、不规则时间、信息复杂度增长）

---

## 🔖 Section 总结

### 核心 Takeaway
- **从 512 到 16k**: EHR FM 的范式转变
- **Mamba > Transformer**: 在 EHR 长上下文场景下
- **数据驱动的领域适配**: 不能盲目迁移 NLP 架构，需要理解领域数据特性
