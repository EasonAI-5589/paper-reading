[← 返回 README](../README.md)

# 3. Methodology

## 📌 预览
VisMem 的完整技术细节。三个核心模块：(1) Memory Invocation（特殊 token 触发机制）；(2) Memory Formation（Query Builder + 双路 Memory Former）；(3) Training Recipe（两阶段 GRPO）。

---

## 3.1. Preliminary

**Problem Formulation.** Based on the interaction process of VLMs, we formulate the problem and introduce the notations used. We first define a policy model $\mathcal{P}$, which is powered by a base VLM. Given a visual task to be solved, feeding a instruction-vision pair $(I, V)$ sampled from a task distribution $\mathcal{D}$, the policy model unfolds a corresponding trajectory $\tau$ at a timestep $t$, including pairs of current state $s_t$ of the environment and the action $a_t$ performed by the model. Here, the state of the environment includes textual contexts and visual observations. Internally, the action is generated sequentially by the token-by-token autoregressive decoding of the model, yielding the output token sequence $\{x_{t,1}, x_{t,2}, \ldots, x_{t,l}\}$. The generation of $i$-th output token $x_{t,i}$ could be presented as:

$$x_{t,i} \sim \mathcal{P}(\cdot \mid s_t, x_{<i}),$$

where the prediction is conditioned on the current environment state and previously generated tokens. To endow the model with vision memory, a vision memory system $\mathcal{M}$ is adhered to the policy model, thus, the objective is to optimize the memory-enhanced model jointly and to maximize its expected performance:

$$\max_{\mathcal{P}, \mathcal{M}} \mathbb{E}_{(I,V) \sim \mathcal{D}, \tau \sim (\mathcal{P}, \mathcal{M})} [S(\tau)],$$

where $S(\cdot)$ denotes the quantifiable performance results, e.g., accuracy or signal from a reward model.

> 💡 **形式化框架**: 把 VLM 推理看作 RL 问题——policy model 生成 trajectory，记忆系统 $\mathcal{M}$ 附加在 policy 上。目标是联合优化 $\mathcal{P}$ 和 $\mathcal{M}$。这与 MemGen 的 RL 框架一脉相承，但 VisMem 的 $\mathcal{M}$ 包含视觉和语义两路。

---

**Motivation.** Building on the Dennis Norris Theory [38], which aligns with contemporary models of human memory, the coordinated operation of short- and long-term visual memories surmounts the "visual processing bottleneck". Short-term latent visual memory maintains fine-grained detail for immediate use and is thus visually dominant; by contrast, long-term latent visual memory abstracts across experiences to enable flexible reuse and is therefore semantically dominant. Taking the task illustrated in Fig. 2 as a case in point, "find the classic Lay's on the shelf" entails the deployment of short-term vision memory, retaining visual details for immediate perceptual demands, while "get in the promotion" triggers generalized semantic knowledge about the "promotion label" acquired from historical scenarios, which is grounded in long-term latent memory, to facilitate the comprehension of the task-based sight. Existing paradigms for enhancing visual capabilities fail to adequately consider vision memory, thus, our VisMem proposes a latent memory method to bridge this gap. More theoretical foundations are in Appendix 6.

> 💡 **直观理解短期 vs 长期记忆**:
> - **短期记忆**: "薯片袋子上写的是什么？包装颜色？" → 需要回到图片看细节（视觉主导）
> - **长期记忆**: "促销标签长什么样？" → 不需要看当前图，是从以往经验中提取的语义知识（语义主导）
> 
> 这个区分很关键：短期记忆从 **vision encoder** 提取，长期记忆从 **language model** 提取。

**Memory System.** Based on previous contents, the task could be further disassembles into two main interactive parts: memory invocation (Sec. 3.2): related to "where and how to invoke the short- or long-term vision memory"; memory formation (Sec. 3.3): related to "what content should the short- or long-term vision memory convey". Additionally, these two decomposed processes interact closely with each other, with distinct priorities and objectives, requiring a meticulously designed training recipe (Sec. 3.4).

---

## 3.2. Memory Invocation

> 💡 **3.2 要点预览**: 如何在自回归生成中触发记忆？答案：扩展词表，添加 4 个特殊 token，模型自己学会在需要的时候输出它们。

As illustrated in Fig. 2, our latent vision memory invocation strategy largely aligns with the standard generation pipeline of VLMs, thereby preserving their robust fundamental visual capabilities. Typically, VLMs generate rationales and answers; however, such pure text sequences lack the granularity to capture fine-grained visual perceptions and semantics, which poses challenges to accurate visual understanding, reasoning, and generation. This limitation arises because during inference, VLMs tend to prioritize accumulated textual context over visual evidence, a phenomenon particularly pronounced in long sequences [17, 25, 72, 78]. To address this, we extend the vocabulary $\mathcal{V}$ of VLMs by incorporating four additional memory-operation tokens, resulting in $\mathcal{V}' = \mathcal{V} \cup \{<m_I^s>, <m_E^s>, <m_I^l>, <m_E^l>\}$. Here, $<m_I>$ and $<m_E>$ form paired invocation and end tokens, where the superscripts $s$ and $l$ denote short- or long-term memory, respectively. Specifically, we register these as indivisible special tokens in the tokenizer and enlarge the embedding matrix from $\mathbb{R}^{|\mathcal{V}| \times d}$ to $\mathbb{R}^{(|\mathcal{V}|+4) \times d}$, where $d$ is the dimension of the model. Furthermore, we initialize the embeddings of the invocation tokens ($<m_I^s>$ and $<m_I^l>$) using the embedding vector of a delimiter token with small perturbations, and update these embeddings during training to facilitate faster convergence. The two end tokens ($<m_E^s>$ and $<m_E^l>$) are treated as structural markers; they are initialized analogously with a lower learning rate. In practice, we also employ constrained decoding to encourage well-formed invocation-end pairs.

> 💡 **4 个特殊 Token**:
> | Token | 含义 | 初始化 |
> |-------|------|--------|
> | `<m_I^s>` | 短期记忆调用开始 | 分隔符 embedding + 小扰动，正常 lr |
> | `<m_E^s>` | 短期记忆调用结束 | 分隔符 embedding + 小扰动，低 lr |
> | `<m_I^l>` | 长期记忆调用开始 | 同上 |
> | `<m_E^l>` | 长期记忆调用结束 | 同上 |
> 
> 生成流中的样子：`... text text <m_I^s> [latent memory tokens] <m_E^s> text text ...`
> 
> 用分隔符初始化是巧妙的——因为记忆调用通常发生在句子/段落边界处，和分隔符的语义位置吻合。

![Figure 2](../images/2b5d4af5306dcefcc9aaae5ce3b620837ed7dcb4334ab871ec910766b30310aa.jpg)
*Figure 2. The overview of our proposed VisMem.*

> 💡 **Figure 2 批读**: VisMem 的完整架构图。关键信息：
> 1. **左侧**：输入的 instruction + image 经过 VLM 标准处理
> 2. **中间**：自回归生成中，遇到 invocation token 时触发 memory formation
> 3. **右侧**：Query Builder 从 hidden states 提取查询 → 分发给短期/长期 Memory Former
> 4. **短期 Memory Former** 挂载在 vision encoder（LoRA）→ 生成视觉细节 token
> 5. **长期 Memory Former** 挂载在 language model（LoRA）→ 生成语义知识 token
> 6. 生成的 latent memory token 插回生成流
> 
> 注意：Query Builder 是共享的，但两个 Memory Former 是独立的 LoRA adapter。

Specifically, the latent vision memory invocation tokens function as triggers for initiating memory insertion, based on the continuous internal cognitive states. During autoregressive generation (see Eq. (4)), upon the output of an invocation token, the memory former immediately initiates the latent vision memory formation procedure:

$$x_{t,i} \to \begin{cases} \text{invocation,} & x_{t,i} \in \{<m_I^s>, <m_I^l>\} \\ \text{continue,} & \text{otherwise} \end{cases}.$$

The resulting latent vision memory, whether short- or long-term as dictated by the specific token type, is subsequently inserted right after the already output invocation token. Following this insertion, the corresponding end token for short ($<m_E^s>$) or long memory ($<m_E^l>$) is automatically appended to resume token-by-token decoding:

$$x_{t,i} \sim \mathcal{P}(\cdot \mid s_t, x_{t,<i}, \{m_I, m_1, ..., m_N, m_E\}).$$

> 💡 **调用流程**:
> ```
> Step 1: 模型正常解码 → 输出 <m_I^s>
> Step 2: 暂停解码 → 触发 Memory Formation
> Step 3: Query Builder 从 hidden states 生成 query
> Step 4: Short-term Memory Former 生成 N_s 个 latent tokens
> Step 5: 插入 [m_1, ..., m_N] + <m_E^s>
> Step 6: 恢复正常解码，后续 token 可以 attend 到这些 memory tokens
> ```
> 这与 MemGen 的 `<MEM>` token 触发机制类似，但 VisMem 有两种类型的触发。

---

## 3.3. Memory Formation

> 💡 **3.3 要点预览**: 记忆"装什么"？由两个组件决定——Query Builder（提问）和 Memory Former（回答）。

To activate the vision memory capability of VLMs, we integrate two memory components: short-term vision memory, which encodes rich visual evidence, and long-term vision memory, which primarily encodes high-level, knowledge-based visual pertinent semantics, without modifying the core VLM and damaging general abilities. This integration leverages short-term memory to enhance advanced visual perception and comprehension, while long-term memory enables the generalization of semantic experiences during reasoning, thus comprehensively enhancing the overall visual performance. As illustrated in Fig. 2, the memory formation process hinges on two core components: a query builder $\mathcal{B}$, which is responsible for generating queries to hook memory; and memory formers $\mathcal{F}_s$ and $\mathcal{F}_l$, which are dedicated to constructing latent visual memories.

### Query Builder

Through this process, we transform hidden states incorporating current cognition into a more efficient and accurate memory query. Initially, we instantiate a lightweight transformer encoder denoted as $\mathcal{B}$ and a learnable memory query $\mathbf{Q}_{init} = \{q_1, ..., q_K\}$, where $K$ represents the length of the query sequence and each $q \in \mathbb{R}^d$. Given the state at a particular time, $\mathcal{B}$ encodes the query sequence based on internal visual and contextual hidden states to retrieve the corresponding latent memory contents. During each invocation, as the policy model generates the current output token sequence, i.e., the token sequence starting from the initial position or from the end of the previous invocation, it accordingly produces a sequence of hidden state vectors $\{h_1, \ldots, h_z\}$. Similarly, visual encoder produces visual hidden state vectors $\{v_1, \ldots, v_y\}$. Thus, the combination of them $\mathbf{H} = \{v_1, \ldots, v_y, h_1, \ldots, h_z\} \in \mathbb{R}^{(y+z) \times d}$, characterizing the multi-modal cognitive state at the time, where $y$ and $z$ denote the lengths. Subsequently, we concatenate the initialized memory query to the rear of these hidden states to update the queried semantic information:

$$\mathbf{Q} = \mathcal{B}([\mathbf{H}, \mathbf{Q}_{init}])[-K:],$$

where we select the output of the last layer of the encoder (see Eq. (10)), and take the last $K$ encoded vectors as the memory query $\mathbf{Q} \in \mathbb{R}^{K \times d}$ to hook latent memory. Furthermore, we employ a masked attention to exclusively enable attention propagation from the query to the hidden states $\mathbf{H}$, while suppressing attention in the reverse direction, i.e., from $\mathbf{H}$ to $\mathbf{Q}$ (see Eq. (11)). Here, both short- and long-term memory share the same query builder $\mathcal{B}$.

> 💡 **Query Builder 架构细节**:
> ```
> 输入: [视觉 hidden states (v_1...v_y) | 文本 hidden states (h_1...h_z) | 可学习 query (q_1...q_K)]
>        ─────────────────── H ──────────────────   ──── Q_init ────
> 
> 处理: Transformer Encoder (masked attention)
>        - Q_init 可以 attend to H ✓
>        - H 不能 attend to Q_init ✗ (单向 mask)
> 
> 输出: 取最后 K 个向量作为 memory query Q
> ```
> 
> 这个设计类似 **Q-Former**（BLIP-2）！Q_init 作为 bottleneck query 从多模态 hidden states 中提取信息。masked attention 保证 H 不被 query 污染——因为 H 来自原始 VLM，不应该被改动。
> 
> **关键**: 短期和长期记忆**共享同一个 Query Builder**。区分在于后续的 Memory Former。

### Latent Memory Former

Distinct from many existing paradigms [26, 44, 70], we internalize the latent vision memory into lightweight formers, preserving the general abilities of base VLMs and ensuring the compatibility of our paradigm. We initialize two lightweight LoRA adapters, which are respectively designated as the short-term memory former $\mathcal{F}_s$ and long-term memory former $\mathcal{F}_l$, attached to the vision encoder and the final language model of the VLM, without directly tampering with the core parameters. More precisely, we first append the generated memory query $\mathbf{Q}$ along with a set of learnable memory tokens after the corresponding target token sequence $\mathbf{X}$. Then we process it by short-term or long-term memory former, which contextualizes and embeds the latent memory information:

$$\mathbf{M}_{s/l} = \mathcal{F}_{s/l}([\mathbf{X}, \mathbf{Q}, \mathbf{M}_{init}])[-N_{s/l}:],$$

where short- and long-term latent vision memory $\mathbf{M}_{s/l} \in \mathbb{R}^{N_{s/l} \times d}$, while $N_s$ and $N_l$ are the predetermined lengths of memory tokens, which can be taken from $\{2, 4, 8, 16, 32\}$. For the short-term pathway, the resultant memory representation is concatenated with the visual token stream, and pass through the original projector to align it with the representation space of the language model. The two memory formers serve as dedicated memory carriers, exclusively storing visual evidences and semantic knowledge within themselves. When the policy model executes a memory invocation, the incoming memory query triggers externalization of useful short- or long-term memory. These memories are seamlessly inserted into the token generation process alongside the invocation and end signals and barely interfere with the original generation, as specified in Eq. (4).

> 💡 **Memory Former 架构详解**:
> 
> | | 短期 Memory Former $\mathcal{F}_s$ | 长期 Memory Former $\mathcal{F}_l$ |
> |--|---|---|
> | **挂载位置** | Vision Encoder | Language Model |
> | **实现** | LoRA adapter | LoRA adapter |
> | **输入** | 视觉 token $\mathbf{X}$ + Query $\mathbf{Q}$ + 可学习 $\mathbf{M}_{init}$ | 文本 token $\mathbf{X}$ + Query $\mathbf{Q}$ + 可学习 $\mathbf{M}_{init}$ |
> | **输出** | $N_s$ 个 latent tokens（默认 8） | $N_l$ 个 latent tokens（默认 16） |
> | **对齐** | 经过 projector 对齐到 LM 空间 | 已在 LM 空间 |
> | **存储内容** | 当前图片的细粒度视觉证据 | 抽象语义知识 |
> 
> **关键设计决策**:
> - 短期记忆挂 vision encoder → 因为需要访问原始视觉特征
> - 长期记忆挂 language model → 因为语义知识存储在 LM 参数中
> - 用 LoRA 而非全参数 → 不破坏原始能力，可即插即用
> - 短期记忆输出需要过 projector → 对齐到语言空间后才能插入生成流
> - $N_l > N_s$（16 > 8）→ 语义知识更抽象、更丰富，需要更多 token 表达

---

## 3.4. Training Recipe

> 💡 **3.4 要点预览**: 两阶段 GRPO 训练。Stage I 训记忆内容（冻结 policy），Stage II 训调用策略（冻结 memory former）。

We design a two-stage training procedure based on GRPO [43], whose optimization objectives are to optimize the effective formation and invocation of latent memory. The first stage enhances the utility of memory, while the second stage maximizes the reward of each invocation, thereby accelerating the convergence of different components steadily. More detailed algorithms and implementations are present in Appendix 7.2 and 8.3.

### Stage I: Memory Formation Optimization

In this stage, we update the query builder $\mathcal{B}$, and memory formers $\mathcal{F}_{s/l}$ while keeping the policy model $\mathcal{P}$ frozen. Initially, during the autoregressive generation process, we randomly invoke either short- or long-term memory upon detecting the delimiter, thereby acquiring initial memory capabilities. Then, the scope of memory invocations is extended to the intervals between delimiters, this not only provides a richer trajectory of memory interactions but also enables memory invocation at arbitrary positions within the generation sequence. The core objective is to maximize the performance improvement relative to trajectory without memory integration $\Delta S(\tau) = S(\tau) - S(\tau_{base})$, thereby enhancing the quality of the memory formation (full function in Eq. (14)):

$$\max_{\mathcal{F}_{s/l}, \mathcal{B}} \mathbb{E}_{\tau \sim \mathcal{P}(\cdot|x, \mathbf{M}_{s/l}), \mathbf{M}_{s/l} \sim \mathcal{F}_{s/l}(\mathbf{Q}), \mathbf{Q} \sim \mathcal{B}(\mathbf{H})} [\Delta S(\tau)].$$

> 💡 **Stage I 详解**:
> - **冻结**: Policy model $\mathcal{P}$
> - **更新**: Query Builder $\mathcal{B}$ + Memory Formers $\mathcal{F}_{s/l}$
> - **触发方式**: 先随机在分隔符处触发（热身），后扩展到任意位置
> - **优化目标**: $\Delta S(\tau)$ = 有记忆的分数 − 无记忆的分数
> 
> 这个设计很聪明：用 $\Delta S$ 而非绝对分数作为奖励，确保只有**真正有帮助的**记忆才会被强化。如果记忆反而帮倒忙，$\Delta S < 0$，会被惩罚。

### Stage II: Memory Invocation Optimization

In this process, we update part parameters $\theta$ of the policy model $\mathcal{P}$, and keeps all the memory formation components frozen. At this stage, the policy model $\mathcal{P}$ is required to invoke memory efficiently and accurately, which entails two core requirements: selecting the correct memory type and avoiding invalid invocations. Thus, we add two penalties to the objective, which could be optimized by (full function in Eq. (15)):

$$\max_{\theta} \mathbb{E}_{\tau \sim \mathcal{P}(\cdot|x, \mathbf{M}_{s/l})} [\Delta S(\tau) - \alpha(p_{type} + p_{neg})],$$

where $\alpha$ denotes the penalty intensity. The type penalty, $p_{type} = \max(0, S(\tau_{rev}) - S(\tau))$, serves to penalize the erroneous selection of memory types, where $\tau_{rev}$ represents the invocation of an alternative memory type. In parallel, the negative penalty $p_{neg} = \max(0, \bar{S} - S(\tau))$ is designed to penalize invocations with negative returns, aiming to enhance efficiency. Here, $\bar{S}$ denotes the mean of quantifiable scores across candidate trajectories.

> 💡 **Stage II 详解**:
> - **冻结**: Query Builder $\mathcal{B}$ + Memory Formers $\mathcal{F}_{s/l}$
> - **更新**: Policy model 的部分参数 $\theta$（学习何时何地输出调用 token）
> - **两个惩罚项**:
>   1. **Type Penalty** $p_{type}$: 如果换另一种记忆类型反而更好 → 说明选错了类型 → 惩罚
>   2. **Negative Penalty** $p_{neg}$: 如果调用记忆后分数低于平均 → 说明不该调用 → 惩罚
> 
> **与 MemGen/Mem-T 的训练范式对比**:
> | | VisMem | MemGen | Mem-T |
> |--|--------|--------|-------|
> | RL 算法 | GRPO (两阶段) | GRPO | 密集化奖励 RL |
> | Stage I | 训记忆内容 | — | 训记忆 agent |
> | Stage II | 训调用策略 | — | — |
> | 分离训练 | ✓ (先内容后调用) | ✗ (端到端) | ✗ |
> | 惩罚机制 | type + negative penalty | — | 密集化奖励 |
> 
> VisMem 的两阶段分离训练是最大特色：先确保"记忆有用"，再学"何时用"。MemGen 是端到端训练，Mem-T 用密集化奖励。

---

## 🔖 Section 总结

### 架构速查
```
VLM 自回归解码
  ↓ 输出 <m_I^s> 或 <m_I^l>
  ↓
Query Builder (共享, transformer encoder)
  ├── 输入: 视觉 H_v + 文本 H_t + learnable Q_init
  ├── Masked attention: Q→H ✓, H→Q ✗
  └── 输出: memory query Q (K=8 tokens)
  ↓
Memory Former (LoRA adapter)
  ├── Short-term: 挂 vision encoder → N_s=8 latent tokens → projector → 插入生成流
  └── Long-term: 挂 language model → N_l=16 latent tokens → 直接插入生成流
  ↓
<m_E^s> 或 <m_E^l> → 恢复正常解码
```

### 关键数字速查
| 参数 | 值 |
|------|-----|
| Memory query 长度 $K$ | 8 |
| 短期记忆 token 数 $N_s$ | 8 |
| 长期记忆 token 数 $N_l$ | 16 |
| 特殊 token 数 | 4 |
| LoRA rank | 16 |
| LoRA α | 32 |
| Stage I lr | 5e-5 |
| Stage II lr | 1e-5 |
| GRPO group size | 16 |

### 核心洞察
1. Query Builder 类似 Q-Former，用 masked attention 从多模态 hidden states 提取查询
2. 短期记忆挂 vision encoder（视觉主导），长期记忆挂 language model（语义主导）——与认知理论完美对应
3. 两阶段训练解耦了"记忆质量"和"调用策略"，避免了同时优化的不稳定性
4. Type penalty 和 negative penalty 保证模型学会选对类型、避免无效调用
