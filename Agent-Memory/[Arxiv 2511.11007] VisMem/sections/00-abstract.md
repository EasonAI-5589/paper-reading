[← 返回 README](../README.md)

# Abstract

## 📌 预览
VisMem 提出认知对齐的双记忆框架：短期视觉主导记忆 + 长期语义主导记忆，以 latent token 形式无侵入地插入 VLM 自回归生成流，12 个 benchmark 平均提升 11%。

---

Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a "visual processing bottleneck": a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and long-term semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for fine-grained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of 11.0% relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.

> 💡 **问题背景**: VLM 在复杂视觉任务中表现受限于 **visual processing bottleneck**：在长序列自回归生成过程中，模型逐渐失去对视觉证据的 grounding，过度依赖累积的文本上下文，导致 visual forgetting / hallucination。

> 💡 **动机来源**: 受 Dennis Norris 人类认知记忆理论启发——人类记忆天然区分两类：
> - **短期记忆** → 视觉主导（fine-grained perceptual retention）
> - **长期记忆** → 语义主导（abstract semantic consolidation）

> 💡 **核心方法**: 提出 VisMem，一个认知对齐的双记忆框架：短期视觉记忆模块保留细粒度感知信息，长期语义记忆模块提炼抽象语义表示，以 **latent token** 形式无侵入地插入 VLM 自回归生成流，推理时动态调用。两个模块分别对应两个互补目标：
> - **Perceptual fidelity**（感知保真度）：生成过程中始终对齐原始视觉证据，不丢失细节
> - **Semantic consistency**（语义一致性）：长序列推理中保持语义连贯，抵抗 hallucination

> 💡 **实验结果**: 覆盖 12 个 benchmark，涵盖理解 / 推理 / 生成三大能力，相对 vanilla 模型平均提升 **+11.0%**，超过所有对比方法。

> 💡 **定位**: 与同组 MemGen（解决 agent 推理中的经验记忆）形成互补——VisMem 专攻 VLM 推理中的**视觉记忆**问题。

---

## 🔖 Section 总结

### 核心洞察
1. Visual processing bottleneck 是 VLM 的根本性限制，不仅影响理解，也影响推理和生成
2. 短期 vs 长期记忆的认知心理学二分法是一个优雅的架构设计灵感
3. Latent space 范式（而非 pixel/token level）是效率和效果的最佳平衡点
