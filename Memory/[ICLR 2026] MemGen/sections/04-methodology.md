[← 返回 README](../README.md)

# 4 Methodology

## 4.1 MemGen: Interleaving Memory and Reasoning

Just as a person is the sum of their past experiences (McAdams, 2001), memory critically shapes an agent's actions (Xiong et al., 2025). Existing agent memory systems, however, often lack the flexibility of human cognition. In the human brain, reasoning and memory form a seamless continuum: active reasoning in the frontoparietal control network and memory retrieval in the hippocampus and prefrontal cortices operate interweavingly, generating a "continuous flow of thoughts" (Su et al., 2025a). By contrast, many agent memory paradigms retrieve information once at task onset and append it coarsely to the query. MemGen is designed precisely to bridge this gap.

> 💡 **认知科学的灵感**：额顶控制网络（推理）和海马体/前额皮层（记忆）的协同工作产生"连续思维流"——这是 MemGen 的认知科学基础。现有方法是"先查记忆再推理"的串行模式，而 MemGen 让两者交织进行。

As shown in Figure 2, the reasoning process in an agent equipped with MemGen unfolds autoregressively, driven by a frozen core LLM, the reasoner πθ. For a given state sₜ, πθ generates the action aₜ = (z_{t,1}, ..., z_{t,Lₜ}). MemGen continuously monitors the token-by-token generation process and performs on-demand memory insertion. At each token-generation step j, a **memory trigger** T_trigger monitors the reasoner's internal cognitive state to determine if a moment of reflection, i.e., a memory invocation, is necessary. Specifically, as the reasoner generates the token sequence z_{t,<j}, it produces a corresponding sequence of hidden state vectors, H_{t,<j} = (h_{t,1}, ..., h_{t,j-1}), where each h_{t,k} ∈ ℝ^{d_model}. The trigger takes the current hidden states H_{t,<j} as a representation of the reasoner's current metacognitive state and computes an invocation probability:

$$
p_j = \sigma\left(\mathcal{T}_{\text{trigger}}(\mathbf{h}_{t,1}, \dots, \mathbf{h}_{t,j-1})\right),
$$

from which a binary decision, d_j ~ Bernoulli(p_j) ∈ {INVOKE, SKIP}, is sampled.

> 💡 **核心机制分解**：
> 1. Reasoner 正常自回归生成 → 产生 hidden states 序列
> 2. Memory trigger 监控 hidden states → 输出 INVOKE/SKIP 决策
> 3. 若 INVOKE → Memory weaver 基于 hidden states 生成 K 个 latent tokens
> 4. Latent tokens 注入 hidden states → Reasoner 继续生成
>
> 这个循环在整个推理过程中反复发生，实现了 memory 与 reasoning 的 interleaving。

If the decision is to [SKIP], πθ proceeds with its standard autoregressive generation, i.e., z_{t,j} ~ πθ(· | sₜ, z_{t,<j}). However, if the decision is to INVOKE, the reasoning process is momentarily paused. This summons the second core component of our framework: the **memory weaver** W_weaver, which takes the same cognitive state H_{t,<j} as a stimulus to perform a generative act of recollection. It synthesizes a bespoke, machine-native latent memory, formalized as M_t ∈ ℝ^{K × d_model} with fixed length K:

$$
\mathbf{M}_t := [\mathbf{m}_{t,1}, \mathbf{m}_{t,2}, \cdots, \mathbf{m}_{t,K}] = \mathcal{W}_{\text{weaver}}(\mathbf{H}_{t,<j}),
$$

where the memory is generated not merely from the parametric knowledge encoded within W_weaver but may also incorporate cues retrieved from external memory databases (detailed implementation is elaborated in Section 4.3). Crucially, M_t is not a verbatim restatement of prior content but a selective reconstruction, filtered and integrated through W_weaver, akin to the hippocampus consolidating fragments of recollection into human memory (Spens and Burgess, 2024).

> 💡 **"生成式重构"而非"检索式提取"**：这是 MemGen 与所有 retrieval-based memory 的本质区别。人类记忆也不是"播放录像带"，而是每次回忆都是一次重新建构（reconstructive memory）。W_weaver 就是在模拟这个过程。

Once formed, the latent memory is woven seamlessly into the reasoner's ongoing dynamics: its hidden states are prepended to H_{t,<j}, upon which the reasoner resumes generation conditioned on this enriched context,

$$
\mathbf{z}_{t,j} \sim \pi_{\theta}\big(\cdot \mid s_t, \mathbf{z}_{t,<j}, \mathbf{M}_t\big).
$$

This iterative cycle of generation, monitoring, invocation, weaving, and reintegration elevates reasoning from a linear unfolding to a recursive dialogue with memory, all without altering the frozen reasoner πθ, and thereby preserving its general capabilities.

> 💡 **"与记忆的递归对话"**——这个表述精准地描述了 MemGen 的本质。传统 Agent 推理是线性展开的，MemGen 让它变成了推理-记忆的螺旋式上升。

---

## 4.2 Memory Trigger: Learning to Invocate Memory

In this section, we describe the concrete instantiation and training recipe of the memory trigger T_trigger. Recall from Section 4.1 that T_trigger serves as a metacognitive monitor, observing the evolving reasoning state of the frozen reasoner πθ and producing a binary decision d_j ∈ {INVOKE, SKIP} (0→SKIP, 1→INVOKE) that determines whether the memory weaver should be invoked at token j.

**Instantiation.** We instantiate T_trigger as a lightweight LoRA adapter attached to the reasoner πθ. At the decoding step j of the timestep t, it receives the sequence of all hidden states, H_{t,<j} ∈ ℝ^{(j-1) × d_model}; conditioned on this context, T_trigger outputs the action probability P(d_j = INVOKE) ∈ [0, 1].

> 💡 **句子粒度激活策略**：不在每个 token 处都运行 trigger，而只在标点符号（逗号、句号等）处激活。灵感来自 Anthropic 的 interpretability 研究（SepLLM）——句子边界是语义转折的关键点。这大幅降低了计算开销。

To avoid excessive computational overhead, we adopt a sentence-granularity activation strategy, inspired by recent studies on LLM interpretability (Anthropic, 2025; Chen et al., 2024a), which find that interventions between sentences can more effectively guide LLMs' reasoning path. Specifically, we define a delimiter token set D (e.g., commas, periods) and let the trigger act only when the current token falls in D. The invocation decision is computed as:

$$
d_j = \text{Bernoulli}(p_j), \quad p_j = \begin{cases} 0 & \text{if } z_j \notin \mathcal{D}, \\ \mathcal{T}_{\text{trigger}}(\mathbf{H}_{t,<j}) & \text{if } z_j \in \mathcal{D}, \end{cases}
$$

which ensures that T_trigger is invoked only at semantically significant boundaries, preserving decoding efficiency. We validate that MemGen does not incur excessive inference delay in Section D.3.3.

**Training Recipe.** The memory trigger is trained via reinforcement learning, motivated by the need to balance two competing desiderata: ensuring that critical latent memories are invoked to improve task performance, while avoiding unnecessary or spurious invocations that could disrupt reasoning or incur computational overhead.

> 💡 **Reward-adaptive penalty 的设计很巧妙**：不是简单地惩罚所有 INVOKE 决策，而是用高奖励轨迹的平均激活率作为基线——如果你的激活率超过"成功案例"的平均水平，才会被惩罚。这让 trigger 学会"精准开枪"。

Given a batch of seen tasks H = {(xᵢ, τᵢ)}ᵢ₌₁ᴺ, the frozen reasoner πθ generates candidate trajectories while the memory weaver W_weaver remains fixed. At each activated step, the trigger selects an action d̃_j ∈ {INVOKE, SKIP} and receives a reward r(τᵢ) reflecting the quality of the resulting trajectory with respect to the task objective. To encourage sparse yet strategically critical memory invocation, we introduce a reward-adaptive penalty, which discourages unnecessary activations while preserving essential ones, into the objective:

$$
\max_{\phi} \mathbb{E}_{\tau_i \sim \pi_\theta, \tilde{\mathbf{d}} \sim \mathcal{T}_{\text{trigger}}^\phi} \Big[ R(\tau_i) - \lambda \sum_{i,j} \max(0, \tilde{d}_{i,j} - \bar{p}) \Big],
$$

where p̄ is computed as the mean activation probability across high-reward trajectories, i.e., those with reward exceeding the batch median:

$$
\bar{p} = \frac{1}{|\mathcal{H}_{\text{high}}|} \sum_{i \in \mathcal{H}_{\text{high}}} \frac{1}{|\tau_i|} \sum_j \tilde{d}_{i,j}, \quad \mathcal{H}_{\text{high}} = \{i : R(\tau_i) \ge \text{median}_k(R(\tau_k))\},
$$

which ensures that T_trigger learns to invoke memory selectively at key decision points, maximizing task reward while maintaining computational efficiency.

> 💡 **训练流程总结**：先训 weaver（固定 trigger），再训 trigger（固定 weaver）——分阶段训练避免了联合优化的不稳定性。Trigger 的 RL 训练本质上是一个稀疏激活的强化学习问题。

---

## 4.3 Memory Weaver: Synthesizing and Inserting Latent Memory

In this section, we elaborate on the weaver W_weaver, the memory carrier within the MemGen framework. When the agent assimilates new experiences, this information is exclusively internalized into the parameters of W_weaver, leaving the core reasoner πθ entirely unmodified. At junctures where the reasoner requires experiential support, a context-dependent hook activates the weaver to synthesize and externalize pertinent knowledge as a usable memory.

> 💡 **"经验只进 Weaver，不碰 Reasoner"**——这是 MemGen 避免灾难性遗忘的核心保证。类比：Reasoner 是"大脑皮层"（通用能力），Weaver 是"海马体"（经验记忆），两者物理分离。

To be more specific, recall from Equation (5) that after the T_trigger signals the need for memory at step j, W_weaver accepts H_{t,<j} (as the hook) and generates a latent token sequence M_t (as the memory) for πθ.

**Instantiation.** We instantiate W_weaver using another LoRA adapter attached to πθ. Formally, given the incoming hook H_{t,<j} ∈ ℝ^{(j-1) × d_model}, the weaver outputs a latent memory matrix: M_t = W_weaver^{θ'}(H_{t,<j}) ∈ ℝ^{K × d_model}, where K denotes the fixed length of the latent memory sequence and θ' are the trainable LoRA parameters. The synthesized M_t is then prepended to the current hidden states of πθ to guide subsequent token generation, as described in Equation (6).

> 💡 **两个独立的 LoRA adapter**：Trigger 和 Weaver 各用一个 LoRA，挂在同一个 frozen LLM 上。推理时根据需要动态切换——这在工程上很优雅，类似 LoRA adapter 的热切换。

**Training Recipe.** The training of W_weaver proceeds over a batch of past trajectories H = {(xᵢ, τᵢ)}ᵢ₌₁ᴺ. Distinct from conventional agent tuning, which directly integrates experiential data into the parameters of πθ (Chen et al., 2025; Yin et al., 2024), MemGen internalizes experiential knowledge solely into W_weaver, which ensures that πθ's general capabilities remain intact.

Crucially, this separation makes MemGen agnostic to optimization strategies and compatible with diverse LLM backbones. Whether employing supervised fine-tuning (SFT) or RL-based objectives such as GRPO or DAPO, the weaver can be updated under a unified goal: optimizing the generation process of latent memory so as to maximize downstream reward. Formally, let Π_θ^{W_{θ'}, T}(· | x) denote the process of rolling out a trajectory for a task x by πθ in conjunction with weaver W_{θ'} and trigger T. Given a reward functional R, the objective updates only θ' by maximizing the expected reward:

$$
\max_{\theta_{\text{lora}}} \mathbb{E}_{(x_i, \tau_i) \sim \mathcal{H}} \mathbb{E}_{\tau \sim \Pi_\theta^{\mathcal{W}_{\theta'}, \mathcal{T}}(\cdot | x_i)} \big[ R(x_i, \tau) \big],
$$

where the gradients from R are propagated solely to θ', thereby equipping the weaver to supply precisely the memories that improve end-to-end performance without altering πθ.

> 💡 **优化目标的优雅之处**：Equation (10) 表面上是一个标准的 RL/SFT 目标，但梯度只流向 weaver 的 LoRA 参数——这意味着 weaver 学到的是"什么样的 latent memory 能最大化 reasoner 的任务表现"，而不是直接学任务本身。

**Integration with Retrieval-based Memory.** Although the memory generation above primarily draws on the weaver's parametric knowledge, it can be combined with external memory sources. When triggered, any retrieval-based system (e.g., MemoryBank, ExpeL) can provide textual memory, which is merged with the hook H_{t,<j} and fed into W to produce latent memory. This allows W to integrate internal knowledge and external information, supplying the reasoner with richer memory support. Implementation details and results are placed in Appendix E.

> 💡 **与检索式记忆的兼容**：MemGen 不是要取代 ExpeL/MemoryBank，而是可以把它们的检索结果作为额外输入。实验表明（Table 8），MemGen + ExpeL 的组合在 ALFWorld 上达到 75.9%，远超单独使用任一方法。这说明 generative memory 和 retrieval memory 是互补的。
