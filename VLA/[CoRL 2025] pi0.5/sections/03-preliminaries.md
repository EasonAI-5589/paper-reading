[← 返回 README](../README.md)

# III. Preliminaries

## 📌 预览
介绍 VLA 的基础框架：模仿学习目标、token 化架构、以及 π0 的 flow matching 动作表示。

---

Vision-language-action models (VLAs) are typically trained via imitation learning on diverse robot demonstration datasets $\mathcal{D}$, by maximizing the log-likelihood of an action $\mathbf{a}_t$ (or, more generally, an action chunk $\mathbf{a}_{t:t+H}$) given an observation $\mathbf{o}_t$ and a natural language task instruction $\ell$: max $\mathbb{E}_{(\mathbf{a}_{t:t+H}, \mathbf{o}_t, \ell) \sim \mathcal{D}} \log(\pi_\theta(\mathbf{a}_{t:t+H} | \mathbf{o}_t, \ell))$. The observation typically contains one or more images $\mathbf{I}_t^1, ..., \mathbf{I}_t^n$ and proprioceptive state $\mathbf{q}_t$, which captures the position of the robot's joints. VLA architectures follow the design of modern language and vision-language models, with modality-specific tokenizers that map inputs and outputs to discrete ("hard") or continuous ("soft") token representations, and a large, autoregressive transformer backbone that is trained to map from input to output tokens. The weights of these models are initialized from pre-trained vision-language models. By encoding policy inputs and outputs into tokenized representations, the imitation learning problem described above can be cast as a simple next-token-prediction problem over a sequence of observation, instruction and action tokens, and we can leverage the scalable tools of modern machine learning to optimize it.

> 💡 **VLA 核心框架**:
> - **输入**: 图像 $\mathbf{I}_t^{1:n}$ + 本体感知 $\mathbf{q}_t$ + 语言指令 $\ell$
> - **输出**: 动作块 $\mathbf{a}_{t:t+H}$ (action chunk)
> - **训练**: 模仿学习 = 最大化条件对数似然
> - **架构**: VLM 初始化 + 模态特定 tokenizer + 自回归 Transformer
> - 核心思想：所有模态 → token → next-token-prediction

---

In practice, the choice of tokenizers for image and text inputs follows those of modern vision-language models. For actions, prior work has developed effective, compression-based tokenization approaches [64], which we use in this work during pretraining. A number of recent VLA models have also proposed to represent the action distribution via diffusion [55, 84, 52] or flow matching [8], providing a more expressive representation over continuous-valued action chunks. During the post-training phase of our model, we will build on the design of the $\pi_0$ model [8], which represents the action distribution via flow matching. In this design, the tokens corresponding to actions receive the partially denoised actions from the previous step of flow matching as input, and output the flow matching vector field. These tokens also use a different set of model weights, which we refer to as an "action expert," analogously to a mixture of experts architecture. This action expert can specialize to flow matching-based action generation, and can be significantly smaller than the rest of the LLM backbone.

> 💡 **动作表示的两种范式**:
> 1. **离散 token** (FAST tokenizer) — 训练快、可扩展，但推理慢（需自回归解码）
> 2. **Flow matching** (π0 的 action expert) — 连续动作、推理快、精度高，但训练慢
>
> **π0.5 的策略**: 预训练用离散 token（快），后训练加 flow matching action expert（精）—— 两全其美！
>
> **Action Expert**: 独立的小 transformer，专门做 flow matching 动作生成，类似 MoE 中的专家

---

## 🔖 Section 总结

### 关键概念速查
| 概念 | 说明 |
|------|------|
| Action Chunk | 一次预测 H 步动作 $\mathbf{a}_{t:t+H}$ |
| FAST Tokenizer | 压缩型动作 token 化，训练高效 |
| Flow Matching | 连续动作生成，迭代去噪 |
| Action Expert | 小型 transformer，专做动作生成 |
| VLM 初始化 | 从预训练 VLM 继承视觉-语言知识 |

### 核心洞察
1. VLA 将机器人控制问题转化为**序列预测问题**
2. 离散 token 和 flow matching 各有优劣，π0.5 在不同阶段分别使用
3. Action Expert 的设计允许动作生成模块独立于语言模型主体
