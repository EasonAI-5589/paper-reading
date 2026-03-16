[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
四大评估维度：模型编辑、长上下文 QA、知识保留、鲁棒性（百万次更新）。

---

## 4.1 Evaluation Protocols

Three aspects:
1. **Integration of New Knowledge**: Model editing (§4.3) + QA tasks (§4.4)
2. **Knowledge Retention**: Long context QA (§4.4) + custom retention experiments (§4.5)
3. **Robustness**: Nearly 1M updates test (§4.6)

---

## 4.3 Model Editing

MemoryLLM demonstrates substantial improvements over existing model editing methods (ROME, MEMIT, MEND, etc.) on standard benchmarks. The key advantage: MemoryLLM can inject multi-sentence knowledge, while traditional model editing is limited to single-fact edits.

> 💡 **批注**: Model editing 基准上的优势来自于 MemoryLLM 能处理更长的知识文本，不局限于单句事实。

## 4.4 Long Context Evaluation

MemoryLLM achieves competitive or superior performance on long-context QA benchmarks compared to RAG methods and long-context models. Key finding: effective up to ~16k-20k tokens of injected knowledge.

> 💡 **批注**: 16k-20k 的限制正是 M+ 要解决的问题——通过增加 LTM 把保留范围扩展到 160k+。

## 4.5 Knowledge Retention

Custom experiments show MemoryLLM can retain knowledge across multiple update steps, with exponential decay matching the theoretical prediction of $1/e$.

## 4.6 Model Integrity

Nearly 1 million updates without performance degradation.

> 💡 **批注**: 这是一个极其重要的工程验证。大多数参数更新方法在多次更新后会退化（catastrophic forgetting），但 MemoryLLM 的 random dropping + 训练时正则化保证了稳定性。

---

## 🔖 Section 总结

### 关键数字速查
| 评估 | 结果 |
|------|------|
| 模型编辑 | 超越 ROME/MEMIT 等传统方法 |
| 长上下文 QA | 有效范围 ~16k-20k tokens |
| 知识保留 | 符合 $1/e$ 理论衰减 |
| 鲁棒性 | ~1M 次更新无退化 |

### 核心洞察
1. MemoryLLM 在模型编辑上的优势来自于能处理多句知识（非单句事实）
2. 16k-20k 的保留限制是 memory pool 固定大小的天然瓶颈 → M+ 通过 LTM 解决
3. 百万次更新的鲁棒性证明了 random dropping 策略的有效性
