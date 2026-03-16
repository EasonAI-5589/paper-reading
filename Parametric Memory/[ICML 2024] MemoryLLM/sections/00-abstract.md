[← 返回 README](../README.md)

# Abstract

## 📌 预览
提出 MemoryLLM：在 Transformer 隐空间中嵌入固定大小 memory pool，实现知识的自我更新。

---

Existing Large Language Models (LLMs) usually remain static after deployment, which might make it hard to inject new knowledge into the model. We aim to build models containing a considerable portion of self-updatable parameters, enabling the model to integrate new knowledge effectively and efficiently. To this end, we introduce MEMORYLLM, a model that comprises a transformer and a fixed-size memory pool within the latent space of the transformer. MEMORYLLM can self-update with text knowledge and memorize the knowledge injected earlier. Our evaluations demonstrate the ability of MEMORYLLM to effectively incorporate new knowledge, as evidenced by its performance on model editing benchmarks. Meanwhile, the model exhibits long-term information retention capacity, which is validated through our custom-designed evaluations and long-context benchmarks. MEMORYLLM also shows operational integrity without any sign of performance degradation even after nearly a million memory updates. Our code and model are open-sourced.

> 💡 **Abstract 批读**:
> - **核心问题**: LLM 部署后是静态的，如何持续注入新知识？
> - **解决方案**: 在 Transformer 隐空间中嵌入**固定大小** memory pool（self-updatable parameters）
> - **三大评估维度**: (1) 新知识整合（模型编辑）, (2) 长期记忆保留, (3) 鲁棒性（百万次更新）
> - **与 Titans 的关系**: MemoryLLM 把知识存在 memory tokens（隐向量）中，Titans 把知识存在 MLP 参数中。两者都是 parametric memory，但机制不同。

---

## 🔖 Section 总结
MemoryLLM 的独特之处：不是用梯度更新参数，而是用 Transformer 自身的前向传播来"写入"新的 memory tokens。这使得更新过程极其高效（不需要反向传播）。
