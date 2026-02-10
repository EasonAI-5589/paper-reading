[← 返回 README](../README.md)

# Abstract

## 📌 预览
VisMem 提出认知对齐的双记忆框架：短期视觉主导记忆 + 长期语义主导记忆，以 latent token 形式无侵入地插入 VLM 自回归生成流，12 个 benchmark 平均提升 11%。

---

Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a "visual processing bottleneck": a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and long-term semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for fine-grained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of 11.0% relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.

> 💡 **核心问题**: VLM 在长序列自回归生成中逐渐"遗忘"视觉证据，过度依赖累积的文本上下文——作者称之为 **visual processing bottleneck**。这是一个被广泛观察到的现象（visual forgetting / hallucination 的根源之一）。

> 💡 **关键创新**: 受 Dennis Norris Theory 启发，将人类的短期/长期记忆二分法映射到 VLM 架构：
> - **短期记忆** → 视觉主导（fine-grained perceptual retention）
> - **长期记忆** → 语义主导（abstract semantic consolidation）
> 
> 这与 MemGen（同组工作）的纯文本 latent memory 形成互补：MemGen 解决 agent 推理中的经验记忆，VisMem 解决 VLM 推理中的视觉记忆。

> 💡 **数字**: 12 个 benchmark，3 大能力（理解/推理/生成），平均 +11%。这个提升幅度相当显著，尤其是在 7B 模型上。

---

## 🔖 Section 总结

### 核心洞察
1. Visual processing bottleneck 是 VLM 的根本性限制，不仅影响理解，也影响推理和生成
2. 短期 vs 长期记忆的认知心理学二分法是一个优雅的架构设计灵感
3. Latent space 范式（而非 pixel/token level）是效率和效果的最佳平衡点
