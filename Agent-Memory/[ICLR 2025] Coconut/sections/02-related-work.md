[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两条主线：(1) CoT 推理方法及其理论分析，(2) LLM 中的 latent reasoning 研究。Coconut 的定位是把两者结合——用 CoT 的训练信号引导 latent reasoning。

---

**Chain-of-thought (CoT) reasoning.** We use the term chain-of-thought broadly to refer to methods that generate an intermediate reasoning process in language before outputting the final answer. This includes prompting LLMs (Wei et al., 2022; Khot et al., 2022; Zhou et al., 2022), or training LLMs to generate reasoning chains, either with supervised finetuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024; Havrilla et al., 2024; Shao et al., 2024; Yu et al., 2024a). Madaan and Yazdanbakhsh (2022) classified the tokens in CoT into symbols, patterns, and text, and proposed to guide the LLM to generate concise CoT based on analysis of their roles. Recent theoretical analyses have demonstrated the usefulness of CoT from the perspective of model expressivity (Feng et al., 2023; Merrill and Sabharwal, 2023; Li et al., 2024). By employing CoT, the effective depth of the transformer increases because the generated outputs are looped back to the input (Feng et al., 2023). These analyses, combined with the established effectiveness of CoT, motivated our design of feeding the continuous thoughts back into the LLM as input embeddings.

> 💡 **CoT 增加有效深度**: Feng et al. (2023) 的理论分析很关键——CoT 之所以有效，是因为输出 token 被 looped back to input，相当于增加了 Transformer 的有效层数。Coconut 直接把 hidden state loop back，本质上是同样的机制，但跳过了离散化瓶颈。

While CoT has proven effective for certain tasks, its autoregressive generation nature makes it challenging to mimic human reasoning on more complex problems (LeCun, 2022; Hao et al., 2023), which typically require planning and search. There are works that equip LLMs with explicit tree search algorithms (Xie et al., 2023; Yao et al., 2023; Hao et al., 2024), or train the LLM on search dynamics and trajectories (Lehnert et al., 2024; Gandhi et al., 2024; Su et al., 2024). In our analysis, we find that after removing the constraint of a language space, a new reasoning pattern similar to BFS emerges, even though the model is not explicitly trained in this way.

> 💡 **对比 Tree of Thoughts / RAP**: ToT (Yao et al., 2023) 和 RAP (Hao et al., 2023) 需要外部搜索算法 + 多次 LLM 调用，开销大。Coconut 的 BFS 是 **内化** 在模型内部的——一次 forward pass 的 hidden state 就同时编码了多条路径。效率差距巨大。

**Latent reasoning in LLMs.** Previous works mostly define latent reasoning in LLMs as the hidden computation in transformers (Yang et al., 2024; Biran et al., 2024). Yang et al. (2024) constructed a dataset of two-hop reasoning problems and discovered that it is possible to recover the intermediate variable from the hidden representations. Biran et al. (2024) further proposed to intervene the latent reasoning by "back-patching" the hidden representation. Shalev et al. (2024) discovered parallel latent reasoning paths in LLMs. Another line of work has discovered that, even if the model generates a CoT to reason, the model may actually utilize a different latent reasoning process. This phenomenon is known as the unfaithfulness of CoT reasoning (Wang et al., 2022; Turpin et al., 2024).

> 💡 **CoT unfaithfulness**: 模型表面上在 "step-by-step" 推理，但内部可能用的是完全不同的 latent reasoning path。这暗示 LLM 已经在做 latent reasoning 了，只是被迫要输出 language token。Coconut 就是让模型 "不用装了"，直接在 latent space 做。

To enhance the latent reasoning of LLMs, previous research proposed to augment it with additional tokens. Goyal et al. (2023) pretrained the model by randomly inserting a learnable `<pause>` tokens to the training corpus. This improves LLM's performance on a variety of tasks, especially when followed by supervised finetuning with `<pause>` tokens. On the other hand, Pfau et al. (2024) further explored the usage of filler tokens, e.g., "...", and concluded that they work well for highly parallelizable problems. However, Pfau et al. (2024) mentioned these methods do not extend the expressivity of the LLM like CoT; hence, they may not scale to more general and complex reasoning problems.

> 💡 **Pause token vs Coconut**: Pause token 只是给模型更多 forward pass 的机会（增加计算量），但每次 forward 的输入还是固定的 token embedding。Coconut 的 continuous thought 是 **动态生成的、信息丰富的** hidden state——不仅增加计算量，还传递了前一步推理的完整信息。实验也证实 Coconut 显著优于 pause token (Table 1)。

Wang et al. (2023) proposed to predict a planning token as a discrete latent variable before generating the next reasoning step. Recently, it has also been found that one can "internalize" the CoT reasoning into latent reasoning in the transformer with knowledge distillation (Deng et al., 2023) or a special training curriculum which gradually shortens CoT (Deng et al., 2024). Yu et al. (2024b) also proposed to distill a model that can reason latently from data generated with complex reasoning algorithms. These training methods can be combined to our framework, and specifically, we find that breaking down the learning of continuous thoughts into multiple stages, inspired by iCoT (Deng et al., 2024), is very beneficial for the training.

> 💡 **iCoT 是 Coconut 的训练基础**: iCoT (Deng et al., 2024) 的 "渐进删除 CoT token" 策略直接启发了 Coconut 的多阶段课程训练。区别在于：iCoT 只是删除 language token（模型必须 internally 完成推理），而 Coconut 用 continuous thought 替换删除的 token——给模型一个显式的 latent reasoning 通道。

Other work explores alternative architectures for latent reasoning, including looped transformers (Giannou et al., 2023; Fan et al., 2024), diffusion models in sentence embedding space (Barrault et al., 2024). Different from these works, we focus on general multi-step reasoning tasks and aim to investigate the unique properties of latent reasoning in comparison to language space. In addition to reasoning tasks, Pham et al. (2023) also explored using continuous space for multi-agent communication.

Building on Coconut, Zhu et al. (2025b) developed a theoretical framework demonstrating that continuous CoT can be more efficient than discrete CoT on certain tasks by encoding multiple reasoning paths in superposition states. Subsequently, Zhu et al. (2025a) analyzed the training dynamics to explain how such superposition emerges under the Coconut training objective.

> 💡 **后续理论工作**: Zhu et al. 的两篇 follow-up 提供了理论基础——continuous thought 确实能以 "superposition" 方式编码多条推理路径，这不是 artifact 而是有理论保证的。这也解释了为什么 Coconut 能涌现 BFS。

---

## 🔖 Section 总结

### 关键对比

| 方法 | 推理空间 | 搜索能力 | 可微分 |
|------|---------|---------|--------|
| CoT | 语言（离散） | 贪心/线性 | ❌ |
| ToT/RAP | 语言 + 外部搜索 | BFS/MCTS | ❌ |
| Pause Token | 语言（固定 embedding） | 无 | ✅ |
| iCoT | 内化到 hidden computation | 隐式 | ❌ |
| **Coconut** | **连续 latent space** | **涌现 BFS** | **✅** |

### 核心洞察
1. CoT 增加有效深度的理论 → 为 hidden state loop-back 提供理论支撑
2. Pause token 只增加计算量，不传递推理信息 → Coconut 两者兼得
3. iCoT 的渐进训练策略 → Coconut 多阶段课程的直接灵感来源
4. CoT unfaithfulness → LLM 已经在做 latent reasoning，Coconut 只是让它显式化
