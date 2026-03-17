[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

本节阐述了当前 LLM "训练后部署"静态范式的根本局限，对比了 In-Context Learning 和 Test-Time Training 两条应对路线，指出现有 TTT 方法面临的三大壁垒（专用层需从头预训练、顺序更新难并行、重建目标与 NTP 不对齐），并提出 In-Place TTT 框架逐一攻克，最后概述实验结果。

---

Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of complex tasks (Brown et al., 2020; Chowdhery & et al., 2022; Touvron et al., 2023; OpenAI, 2024). This success is largely built on a static "train then deploy" paradigm, where a model first acquires knowledge from massive corpora and then kept fixed during inference. Yet this design imposes a fundamental limitation: once deployed, the model's weights cannot be updated, preventing dynamic adaptation to the specific context provided by streaming input tokens. Consequently, at test time, the model is constrained in its ability to process and reason over long-horizon, evolving tasks (Chan et al., 2024; Starace et al., 2025), and to continuously learn from unbounded streams of experience like humans (Silver & Sutton, 2025).

> 💡 开篇定位了全文的核心矛盾：静态权重 vs. 动态上下文。"train then deploy" 意味着模型权重在部署后冻结，无法随推理时的流式输入做自适应更新。这直接限制了两类能力：(1) 长程/演化任务的推理能力；(2) 像人类一样从无界经验流中持续学习的能力。注意这里引用了 Silver & Sutton (2025) 的 continual learning 视角——把 TTT 放在 "迈向持续学习" 的更大叙事中。

In-context learning (Brown et al., 2020; Wei et al., 2023) offers a way to mitigate this problem via maintaining all past input tokens in the context. However, its effectiveness is tethered to the model's context window, restricted by the quadratic complexity of the de facto attention mechanism (Vaswani et al., 2017). This bottleneck has spurred a line of research into architectural solutions aimed at efficiently extending the context window (Beltagy et al., 2020; Peng et al., 2023; Child et al., 2019; Dao et al., 2023). Differently, Test-Time Training (TTT) has emerged as a new paradigm (Sun et al., 2020; Wang et al., 2021; Sun et al., 2024; Behrouz et al., 2024; Yau et al., 2025). Instead of merely making a static model more efficient, TTT enables the model to dynamically update the parameters and adapt to any specific context, directly targeting the aforementioned limitation. Specifically, TTT introduces a small subset of model parameters, called fast weights (Schlag et al., 2021), which can be updated on the fly for each new input. By minimizing a self-supervised reconstruction objective, these fast weights compress and internalize contextual information, functioning as an expressive, online evolving state.

> 💡 ICL vs. TTT 的范式对比是本段的核心。ICL 把所有历史 token 保留在上下文中——本质上是把"记忆"外挂在 KV cache 里，代价是 O(n^2) 的注意力复杂度和有限窗口。TTT 换了一条路：不靠加长上下文，而是把信息压缩进可更新的 fast weights（参数记忆），用自监督重建目标在线更新。这里的 fast weights 概念来自 Schlag et al. (2021)，与 Titans 论文中的"持久记忆"思路一脉相承——都是把上下文信息从显式 token 序列转化为隐式参数状态。关键区别在于：ICL 是 retrieval-based（从缓存中检索），TTT 是 learning-based（通过梯度下降学习）。

Despite its conceptual appeal, unleashing TTT's potential within the current LLM ecosystem is hindered by critical barriers: (i) Existing TTT methods often rely on specialized layers beyond standard Transformer blocks, which usually demand costly pretraining from scratch to achieve satisfactory performance. (Sun et al., 2020; Wang et al., 2021; Zhang et al., 2025; Sun et al., 2024); (ii) the canonical TTT mechanism is inherently sequential (Sun et al., 2020; 2024). While existing works explore chunk-wise acceleration (Sun et al., 2023; Behrouz et al., 2024; Irie & Gershman, 2025; Yau et al., 2025), TTT's role as the primary token mixer forces a reliance on small chunks to maintain performance, thereby bottlenecking the massive parallelism required to saturate modern accelerators; and (iii) the prevalent use of a generic reconstruction objective for TTT's fast weights updating is not explicitly tailored for the causal, Next-Token Prediction task that governs autoregressive LMs, potentially hindering their ultimate performance.

> 💡 三大壁垒，逐条拆解：
>
> - **壁垒 (i)：架构侵入性高。** 之前的 TTT 方法（如 TTT-Linear, TTT-MLP）需要引入专门的 TTT 层替换注意力层，这意味着必须从头预训练才能让模型学会使用这些新层。对已有的大模型生态极不友好——你不可能为了加 TTT 就把 Qwen/LLaMA 从头训一遍。
> - **壁垒 (ii)：并行化瓶颈。** TTT 的梯度下降更新天然是顺序的（token-by-token），虽然可以用 chunk-wise 方式加速，但当 TTT 层作为主要的 token mixer 时，chunk 不能太大（否则丢失精度），导致无法充分利用 GPU 的大规模并行能力。这是计算效率问题。
> - **壁垒 (iii)：目标函数错配。** 现有 TTT 用的是通用的自监督重建目标（如 masked autoencoder 式的重建），但 LLM 的核心任务是因果 Next-Token Prediction。两者之间存在 gap——fast weights 学到的信息未必是对 NTP 最有用的信息。

To bridge this gap, we introduce In-Place Test-Time Training (In-Place TTT), a framework designed to seamlessly endow LLMs with Test-Time Training capabilities by directly addressing the aforementioned barriers. Our core insight is to repurpose existing MLP blocks with an in-place design rather than introducing a new, specialized layer (tackling barrier i). Specifically, In-Place TTT treats the final projection matrix of MLP blocks as their fast weights, updating it in-place during inference. This "drop-in" design requires no modifications to the model's architecture, preserving the integrity of pre-trained weights and enabling on-the-fly adaptation without costly retraining from scratch.

> 💡 **核心设计洞察：复用现有 MLP，而非新增层。** 这是本文最关键的贡献。具体做法是把 MLP block 的最后一个投影矩阵（即 down projection）当作 fast weights，在推理时原地（in-place）更新。为什么这个设计如此重要？
>
> 1. **零架构改动**：不需要往 Transformer 里插入任何新模块，直接在已有的 MLP 上操作。这意味着可以拿任何预训练好的 LLM（如 Qwen3-4B）直接用，只需少量 continual training 就能激活 TTT 能力。
> 2. **保留预训练权重的完整性**：fast weights 只是 MLP 的一个子矩阵，其余参数不动。模型原有的知识不会被破坏。
> 3. **"drop-in" 即插即用**：从工程角度看，这大大降低了 TTT 的落地门槛。

To tackle the computational inefficiency and objective misalignment, we further design a bespoke adaptation mechanism for language modeling. Following previous works (Sun et al., 2023; Behrouz et al., 2024; Irie & Gershman, 2025; Zhang et al., 2025; Yau et al., 2025), we replace the inefficient per-token updates with a scalable chunk-wise update rule (tackling barrier ii). Furthermore, our in-place design operates complementarily to the attention mechanism. This synergy obviates the need for small chunks required by standalone TTT layers, thereby ensuring high throughput on modern accelerators. Concurrently, we move beyond the generic reconstruction targets of prior work (Sun et al., 2024; Zhang et al., 2025) and introduce a novel objective explicitly aligned with the Next-Token Prediction (NTP) goal (tackling barrier iii). Grounded in a rigorous theoretical analysis, we show this NTP-aligned objective encourages the fast weights to store predictively useful information for autoregressive language modeling, leading to a highly effective and scalable algorithm.

> 💡 对壁垒 (ii) 和 (iii) 的解法：
>
> - **解决并行化问题**：采用 chunk-wise 更新（和前人一致），但关键区别在于——因为 In-Place TTT 的 MLP 更新与注意力机制是互补的（attention 仍然负责 token mixing），所以 chunk 可以取得很大而不损失性能。这与之前 TTT 作为唯一 token mixer 时必须用小 chunk 的情况形成对比。chunk 越大，并行度越高，GPU 利用率越高。
> - **解决目标错配问题**：提出与 NTP 对齐的新目标函数，不再用通用重建目标。理论分析表明，这个对齐的目标能引导 fast weights 存储"对预测下一个 token 有用的信息"，而非泛泛的上下文重建信息。这是一个从"压缩"到"预测"的目标转移。

Grounded in these principled design choices, our In-Place TTT provides a practical and effective framework for enhancing LLMs with dynamic, continual adaptation. We conduct extensive experiments on language modeling tasks of various compute scales, using them as a practical proxy to probe the model's potential on long-horizon, evolving tasks. Through relatively cheap continual training, our In-Place TTT enables Qwen3-4B-Base to achieve superior performance on tasks with contexts up to 128k, and when pretrained from scratch, it consistently outperforms competitive TTT-related methods by conducting pretraining from scratch on up to 32k-length corpora, validating the architectural merit of our framework. Finally, ablation studies on state size, chunk size, and fast weight objectives provide deeper insights, confirming the critical role of each design choice. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.

> 💡 贡献总结段。两条实验路线验证了方法的有效性：
>
> 1. **Continual training 路线**：在 Qwen3-4B-Base 上做少量继续训练，即可在 128k 长上下文任务上取得优异表现。这验证了 "drop-in" 设计的实用性——不需要从头训练。
> 2. **From-scratch pretraining 路线**：在 32k 长度语料上从头预训练，持续优于竞争性 TTT 方法。这验证了架构本身的优越性。
>
> 注意最后一句话的定位："a promising step towards a paradigm of continual learning in LLMs"——作者把 In-Place TTT 放在了 LLM 持续学习范式的叙事框架下。

---

## 🔖 Section 总结

### 核心洞察

1. **问题本质**：LLM 的 "train then deploy" 范式导致推理时权重冻结，无法动态适应流式输入上下文。ICL 虽能部分缓解，但受限于 O(n^2) 注意力复杂度和有限窗口。

2. **TTT 的定位**：与 ICL 互补的 learning-based 范式——不通过检索 KV cache 中的显式 token，而是通过梯度下降将上下文压缩进 fast weights（参数记忆）。

3. **现有 TTT 的三大壁垒及 In-Place TTT 的对应解法**：

| 壁垒 | 核心问题 | In-Place TTT 解法 |
|------|---------|-----------------|
| (i) 架构侵入 | 需引入专用层，必须从头预训练 | 复用现有 MLP 的 down projection 作为 fast weights，零架构改动 |
| (ii) 并行化瓶颈 | 顺序更新 + 小 chunk 限制 GPU 利用率 | MLP 更新与 attention 互补，允许大 chunk，提升并行度 |
| (iii) 目标错配 | 通用重建目标与 NTP 任务不对齐 | 提出 NTP-aligned 目标函数，引导 fast weights 存储预测性信息 |

4. **核心设计哲学**：不做加法（不加新层），做复用（repurpose existing MLP）。这是本文区别于 TTT-Linear/TTT-MLP/Titans 等前作的最大创新点——把 TTT 从"需要特殊架构"变成"任何 Transformer LLM 的即插即用增强"。
