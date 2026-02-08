[← 返回 README](../README.md)

# 2 Related Work

> 💡 **Related Work 结构清晰**：三条线索——(1) LLM/Agent Memory 三大范式，(2) Latent Computation 两大方向，(3) LLM Decoding & RL 的技术联系。每条线都精准地把 MemGen 的位置标出来了。

**LLM & Agent Memory.** As outlined in Section 1, existing memory mechanisms designed to evolve the problem-solving capacity of LLM agents can be broadly categorized into three classes: (I) parametric memory, which either integrates past experiences directly into agent parameters through finetuning, as in FireAct (Chen et al., 2023), AgentLumos (Yin et al., 2024), and others (Zhang et al., 2024a; Fu et al., 2025), or maintains them in external parameter modules (Tack et al., 2024; Wang et al., 2024a); (II) retrieval-based memory, which abstracts prior experiences into transferable knowledge (Zhang et al., 2025a; Zhao et al., 2024), or distills them into reusable tools and skills (Zheng et al., 2025; Wang et al., 2025b; Qiu et al., 2025b,a); and (III) latent memory, which leverages implicit representations to encode and retrieve experience (Wang et al., 2024b, 2025a; Hu et al., 2025b; Liu et al., 2024; Sun et al., 2025). Our MemGen falls within the latent memory paradigm, yet distinguishes itself from prior approaches through its more human-esque interweaving of reasoning and memory, as well as its generative, rather than purely retrieval-based, nature.

> 💡 **三类记忆的代表工作速查**：
> | 范式 | 代表 | 特点 |
> |------|------|------|
> | 参数记忆 | FireAct, AgentLumos, AgentTuning | 改 LLM 参数 |
> | 检索记忆 | ExpeL, AWM, SkillWeaver, Alita | 外部数据库 |
> | 隐式记忆 | MemoryLLM, M+, MemGen (ours) | Latent representations |

**Latent Computation.** Our method is also closely related to latent computation, wherein latent states are employed to intervene in or reshape the LLM's reasoning process (Zhu et al., 2025). Prominent paradigms include: (I) architecturally enabling native latent reasoning, exemplified by Coconut (Hao et al., 2024), CODI (Shen et al., 2025), LatentR3 (Zhang et al., 2025b) and CoLaR (Tan et al., 2025), which render the LLM's inference process inherently latent and machine-native; and (II) employing latent computation to steer LLM generation, as in LaRS (Xu et al., 2023), LatentSeek (Li et al., 2025a), SoftCoT (Xu et al., 2025c,b), and Coprocessor (Liu et al., 2024), which leverage latent representations to modulate the quality of generated outputs. These aforementioned works have greatly inspired the latent memory design in this paper: Latent memory can likewise be viewed as an instantiation of the latter, supplementing essential memory context to enhance the problem-solving capacity of agents (Wang et al., 2024b, 2025a).

> 💡 **Latent Computation 两条路线**：
> - **Native latent reasoning** (Coconut, CODI)：让 LLM 直接在 latent space 推理，不输出文字 CoT
> - **Latent steering** (LatentSeek, SoftCoT, Co-processor)：用 latent tokens 引导生成质量
>
> MemGen 属于后者的延伸，但加入了"记忆"的语义——不只是提升当前推理质量，而是编码和利用过往经验。

**LLM Decoding & RL.** Two additional topics that relate to our work are LLM decoding and reinforcement learning (RL). From the decoding perspective, MemGen dynamically generates and inserts latent tokens, which shares similarity with speculative decoding where a drafter model receives the current decoding context and produces subsequent drafted tokens (Cai et al., 2024; Fu et al., 2024; Li et al., 2025b; Goel et al., 2025). However, these methods primarily aim to accelerate LLM inference, whereas MemGen focuses on leveraging latent states as effective carriers of memory. From the RL perspective, MemGen employs rule-based RL to train the memory trigger, which is closely related to reinforcement learning with variable reward (RLVR), including GRPO from DeepSeek-R1 (DeepSeek-AI et al., 2025) and its various derivatives (Qian et al., 2025; Wu et al., 2025a; Wei et al., 2025; Fu et al., 2025). While there exist efforts combining RL with agent memory, to our knowledge, most do not address self-improving memory; for example, MemAgent (Yu et al., 2025) and MEM1 (Zhou et al., 2025) focus on handling long-context inputs rather than evolving memory mechanisms.

> 💡 **与 Speculative Decoding 的类比很巧妙**：都是用一个小模型在主模型推理过程中"插入"额外 tokens，但目的截然不同——speculative decoding 加速推理，MemGen 增强记忆。这个类比帮助理解 MemGen 的实现机制。
