[← 返回 README](../README.md)

# Abstract

## 📌 预览
ToDRE 提出两阶段、无需训练的 visual token 剪枝框架：Stage 1 基于 token diversity 做 greedy max-sum diversification 保留多样化子集；Stage 2 基于 task relevance 在 LLM decoder 中移除 cross-modal attention 衰减后的所有 visual token。90% 剪枝率下保持 95.0% 性能，2.6× 加速。

---

Visual token pruning aims to compress and prune redundant visual tokens which play a critical role in efficient inference with large vision-language models (LVLMs). However, most existing work estimates visual redundancy using a single metric, such as cross-modal attention or visual token similarity. We show that visual token diversity and task-specific token relevance are two crucial yet orthogonal factors that complement each other in conveying useful information and should therefore be treated separately for more effective visual token pruning. Building upon this insight, we design

> 💡 **核心洞察**: 现有方法用单一指标衡量冗余（attention 或 similarity），而 ToDRE 指出 diversity 和 relevance 是两个**正交**因素，应分别处理。

---

TODRE, a two-stage and training-free framework that incorporates Token Diversity and task RElevance for effective token compression and efficient LVLM inference. Instead of pruning redundant tokens, we introduce a greedy max-sum diversification algorithm that selects and retains a subset of diverse and representative visual tokens after the vision encoder. On top of that, ToDRE leverages an "information migration" mechanism to eliminate task-irrelevant visual tokens within certain decoder layers of large language model (LLM) to further improve token pruning and LVLM inference. Extensive experiments show that ToDRE prunes $90 \%$ of visual tokens after the vision encoder as well as all visual tokens in certain LLM decoder layers, leading to a $2 . 6 \times$ speed-up in total inference time while maintaining $9 5 . 0 \%$ model performance plus excellent model compatibility.

> 💡 **方法概述**:
> - **Stage 1**: Vision encoder 之后，greedy max-sum diversification 保留 diverse token 子集
> - **Stage 2**: LLM decoder 中，利用 "information migration" 在 cross-modal attention 衰减的层移除全部 visual token
> - **结果**: 90% 剪枝、2.6× 加速、95.0% 性能保持

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| Visual token 剪枝率 | 90% |
| 推理加速 | 2.6× |
| 性能保持 | 95.0% |
| 训练需求 | 无（training-free） |

### 核心洞察
1. Token diversity 和 task relevance 是正交的两个维度
2. 两阶段设计：先保多样性、再删无关性
3. Training-free + plug-and-play，模型兼容性强
