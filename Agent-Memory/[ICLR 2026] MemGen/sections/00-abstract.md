[← 返回 README](../README.md)

# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

**Guibin Zhang†, Muxin Fu†, Shuicheng Yan**

National University of Singapore · †Equal Contribution

> 💡 **论文定位**：这篇工作直击 Agent Memory 的核心痛点——参数记忆有灾难性遗忘，检索式记忆是死板的"查字典"，都无法实现人类那种推理与记忆交织的流畅认知。MemGen 提出的"生成式隐式记忆"是一个很新颖的 framing。

Agent memory shapes how Large Language Model (LLM)-powered agents, akin to the human brain, progressively refine themselves through environment interactions. Existing paradigms remain constrained: parametric memory forcibly adjusts model parameters, and retrieval-based memory externalizes experience into structured databases, yet neither captures the fluid interweaving of reasoning and memory that underlies human cognition. To address this gap, we propose MemGen, a dynamic generative memory framework that equips agents with a human-esque cognitive faculty. It consists of a memory trigger, which monitors the agent's reasoning state to decide explicit memory invocation, and a memory weaver, which takes the agent's current state as stimulus to construct a latent token sequence as machine-native memory to enrich its reasoning. In this way, MemGen enables agents to recall and augment latent memory throughout reasoning, producing a tightly interwoven cycle of memory and cognition.

> 💡 **关键数字**：超 ExpeL/AWM 38.22%，超 GRPO 13.44%，且自发涌现 planning/procedural/working memory 三层记忆层级——这个 emergent 发现是本文最有启发性的贡献。

Extensive experiments across eight benchmarks show that MemGen surpasses leading external memory systems such as ExpeL and AWM by up to 38.22%, exceeds GRPO by up to 13.44%, and exhibits strong cross-domain generalization ability. More importantly, we find that without explicit supervision, MemGen spontaneously evolves distinct human-like memory faculties, including planning memory, procedural memory, and working memory, suggesting an emergent trajectory toward more naturalistic forms of machine cognition.

> 💡 **一句话总结**：用 RL 训练的 memory trigger 决定"何时想起"，用 LoRA memory weaver 生成 latent tokens 作为"机器原生记忆"注入推理流，实现推理与记忆的动态交织。

📅 Date: October 14, 2025 | 🔗 Github: https://github.com/KANABOON1/MemGen
