[← 返回 README](../README.md)

# Abstract

## 📌 预览
Coconut 的核心 pitch：LLM 不需要在语言空间推理，可以直接用 hidden state 作为 "continuous thought" 反馈，涌现 BFS-like 搜索。

---

Large language models (LLMs) are restricted to reason in the "language space", where they typically express the reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language space may not always be optimal for reasoning. For example, most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs.

> 💡 **核心动机**: 语言空间推理有两个浪费：(1) 大部分 token 只是为了语法通顺，不贡献推理；(2) 关键决策 token 需要复杂规划，但 LLM 对每个 token 分配的计算量相同。这是 CoT 的根本矛盾。

To explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we introduce a new paradigm Coconut (Chain of Continuous Thought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed "continuous thought"). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space.

> 💡 **Continuous Thought 的实现**: 极其简洁——不经过 LM head 解码成 token，直接把 last hidden state 当作下一步的 input embedding。相当于跳过了 "hidden → softmax → token → embedding" 的信息瓶颈，让信息在连续空间无损传递。

This latent reasoning paradigm leads to the emergence of an advanced reasoning pattern: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT.

> 💡 **BFS 涌现**: 这是最 surprising 的发现。因为 hidden state 是连续向量，它可以同时 "叠加" 编码多个候选路径（类似量子叠加态），而离散 token 必须选一个。这让模型能延迟决策，先广度探索再收敛——而且这不是训练目标强制的，是 naturally emerged。

Coconut outperforms CoT on certain logical reasoning tasks that require substantial search during planning, and shows a better trade-off between accuracy and efficiency.

> 💡 **Abstract 小结**: Coconut 在需要搜索规划的逻辑推理任务上超越 CoT，而且更高效（fewer tokens）。但注意 "certain" 这个词——在 GSM8k 数学推理上还没超过 CoT，说明 latent reasoning 的优势主要在需要 backtracking/planning 的场景。

Last updated: November 4, 2025 Code: https://github.com/facebookresearch/coconut

---

## 🔖 Section 总结

### 核心洞察
1. 语言空间推理 ≠ 最优推理——大部分 token 浪费在语法上
2. Continuous thought = last hidden state 直接反馈，跳过信息瓶颈
3. 连续向量天然能编码多路径 → BFS 涌现
4. 在规划密集型任务上超越 CoT，效率更高
