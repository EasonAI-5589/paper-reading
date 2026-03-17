[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文提出 In-Place TTT，通过将 MLP 的最终投影矩阵作为快速权重、设计与 Next-Token-Prediction 对齐的训练目标、并引入 chunk-wise 更新机制，使现有 LLM 无需从头重训即可获得推理时持续学习能力。

---

The static "train then deploy" paradigm fundamentally limits Large Language Models (LLMs) from dynamically adapting their weights in response to continuous streams of new information inherent in real-world tasks. Test-Time Training (TTT) offers a compelling alternative by updating a subset of model parameters (fast weights) at inference time, yet its potential in the current LLM ecosystem is hindered by critical barriers including architectural incompatibility, computational inefficiency and misaligned fast weight objectives for language modeling. In this work, we introduce In-Place Test-Time Training (In-Place TTT), a framework that seamlessly endows LLMs with Test-Time Training ability. In-Place TTT treats the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights, enabling a "drop-in" enhancement for LLMs without costly retraining from scratch. Furthermore, we replace TTT's generic reconstruction objective with a tailored, theoretically-grounded objective explicitly aligned with the Next-Token-Prediction task governing autoregressive language modeling. This principled objective, combined with an efficient chunk-wise update mechanism, results in a highly scalable algorithm compatible with context parallelism. Extensive experiments validate our framework's effectiveness: as an in-place enhancement, it enables a 4B-parameter model to achieve superior performance on tasks with contexts up to 128k, and when pretrained from scratch, it consistently outperforms competitive TTT-related approaches. Ablation study results further provide deeper insights on our design choices. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.

> 💡 **核心问题**: 传统 TTT（Test-Time Training）虽然提供了推理时更新权重的思路，但在当前 LLM 生态中面临三大障碍：**(i) 架构不兼容** — 已有 TTT 方法（如 TTT-Linear、TTT-MLP）需要用自定义的 TTT 层替换注意力层，无法直接应用于已经训练好的 LLM；**(ii) 计算效率低** — 推理时的在线梯度更新带来显著开销，难以扩展到长上下文场景；**(iii) 目标函数不对齐** — 传统 TTT 使用通用的自重构（reconstruction）目标来更新快速权重，但这与 LLM 的核心任务——下一个 token 预测（NTP）——并不一致，导致快速权重学到的表征无法有效服务于语言建模。

> 💡 **解决方案 — In-Place TTT**: 三大设计选择分别对应解决上述三大障碍。**第一，复用 MLP 下投影矩阵 $W_{down}$ 作为快速权重**：MLP block 是所有 Transformer LLM 的标配组件，将其最终投影矩阵指定为可在推理时更新的快速权重，实现了"即插即用"（drop-in）——无需修改架构，无需从头重训，直接在现有预训练模型上启用 TTT 能力。**第二，chunk-wise 更新机制**：将输入序列划分为 chunk，在每个 chunk 内批量计算梯度并更新快速权重，而非逐 token 更新；这不仅大幅降低计算开销，还天然兼容 context parallelism（上下文并行），使算法可扩展到超长上下文。**第三，NTP 对齐的训练目标**：用一个理论上有据可依的、与 Next-Token-Prediction 直接对齐的目标函数替代传统的自重构目标，确保快速权重的更新方向与语言建模的最终目标一致。

> 💡 **关键结果**: 作为即插即用增强（不重训），In-Place TTT 使一个 4B 参数模型在长达 128k token 的上下文任务上取得了优于基线的表现——这说明该方法能让已有的中等规模模型有效利用超长上下文信息。此外，从头预训练时，In-Place TTT 也持续优于同类 TTT 方法（如 TTT-Linear），验证了其设计选择在根本层面的优越性，而非仅仅是工程技巧的改进。

---

## 🔖 Section 总结

### 核心洞察
1. **"位置"选择决定可行性**：将快速权重锚定在 MLP $W_{down}$ 上是整个框架的关键前提——MLP 是 Transformer 的通用组件，选择它意味着方法对任意 LLM 架构通用；而 $W_{down}$ 作为 MLP 的最终投影，直接影响该层的输出表征，是最具杠杆效应的更新位置。
2. **目标函数对齐是 TTT 在 LLM 中落地的核心瓶颈**：传统 TTT 的自重构目标本质上是一种无监督辅助任务，与 NTP 目标之间存在 gap。本文将快速权重的更新目标直接与 NTP 对齐，从理论上保证了梯度方向的一致性，这是性能提升的主要来源。
3. **Chunk-wise 更新同时解决效率和并行性**：逐 token 更新快速权重在序列长度上是 $O(n)$ 次串行梯度步，而 chunk-wise 方法将其降为 $O(n/c)$ 次，同时每个 chunk 内的计算可以并行化，与现代分布式训练/推理基础设施（context parallelism）自然兼容。
