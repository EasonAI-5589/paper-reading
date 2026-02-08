# 3. MM-RLHF-Reward Model

> 来源: MM-RLHF (ICML 2025)

---

## 📄 原文

In this section, we explore how to train a high-quality reward model using the MM-RLHF dataset to provide a robust supervision signal for subsequent model alignment. The reward model is designed to combine critique generation and scoring (Figure 3), ensuring a comprehensive evaluation process.

> 💡 **Section 概览**: 本节提出 Critique-Based Reward Model——先生成 critique，再打分。这是从传统 scalar RM 到可解释 RM 的范式升级。

![Figure 3](../images/6395d304a30341edcf14938ecbf951c4bf345fe8a19a61e7e4cbdf5d6e839c60.jpg)
*Figure 3: Critique-Based Reward Model 训练流程。用户 query + 模型 response → 人工排序标注 → GPT-4o 扩展标注 → 双任务训练：(1) Learning to Provide Critique (2) Learning Scoring。*

> 💡 **Figure 3 批读**:
> ```
> 训练流程:
> ├── 输入: user query + model responses
> ├── 人工标注: ranking + 简短理由
> ├── GPT-4o 扩展: 简短理由 → 详细 critique
> └── 双任务训练:
>     ├── Task 1 (Critique Head h_l): 学习生成 critique
>     │   └── Loss: 标准 language modeling loss
>     └── Task 2 (Scoring Head h_r): 学习打分
>         └── Loss: 基于 critique 的 pairwise ranking loss
>         └── 训练时用 teacher-forcing (GT critique)
> 
> 推理流程:
> input → Critique Head 生成 critique → Scoring Head 基于 critique 打分
> ```

---

### 3.1 Background and Limitations of Standard Reward Models

Reward models are a key component for aligning model outputs with human preferences. Typically, a reward model starts with a pretrained LLM φ, where the LLM head $h_l$ is replaced with a linear reward head $l_r$, enabling the model to output a scalar reward value. These models are trained using human-provided pairwise comparisons. Given a query **x**, a preferred response $y_w$ and a less preferred response $y_l$, the reward model is optimized to assign higher rewards to preferred responses:

![Equation: Reward Loss](../images/eq_reward_loss.png)

$$\ell_{\mathrm{Reward}}(\theta) = \mathbb{E}_{\mathbf{x}, y_w, y_l} \Big[ -\log \sigma \Big( r(y_w | \mathbf{x}) - r(y_l | \mathbf{x}) \Big) \Big]$$

> 💡 **标准 RM 的训练**: Bradley-Terry model——让 preferred response 的 reward 高于 non-preferred 的。这是经典的 pairwise ranking loss。

Despite their utility, standard reward models face significant limitations. First, they fail to fully utilize the rich and detailed feedback provided by high-quality human annotations, such as textual explanations and nuanced reasoning. Second, scalar rewards lack transparency, making it difficult for humans to understand how the reward is generated. These challenges highlight the need for a more interpretable and robust reward model that leverages critiques as intermediate reasoning steps.

> 💡 **标准 RM 的两个核心问题**:
> 1. **信息浪费**: 人工标注提供了丰富的文字理由，但 scalar RM 只学了排序关系，白白浪费了这些 textual feedback
> 2. **不可解释**: 输出一个数字，无法知道为什么这个 response 好/差

---

### 3.2 Critique-Based Reward Model Training

> 💡 **3.2 要点预览**: 两步走——(1) 扩展人工标注为详细 critique → (2) 双任务联合训练（critique 生成 + scoring）。

**Extending to critique-based training.** To overcome the limitations of traditional reward models, we propose a critique-based training framework: the model first generates a critique $c$ conditioned on the query **x**. This critique serves as an intermediate reasoning step, providing context for scoring responses. The critique-based reward model comprises two components:
1. **Critique Head ($h_l$)**: Generates critiques $c_w$ and $c_l$ for the preferred ($y_w$) and less preferred ($y_l$) responses, respectively, based on the query **x**.
2. **Scoring Head ($h_r$)**: Assigns scalar rewards based on the generated critiques, enabling more fine-grained evaluation.

> 💡 **模型架构**:
> ```
> Critique-Based RM = Critique Head (语言头) + Scoring Head (奖励头)
>                     共享 backbone MLLM
> 
> Critique Head: 保持原始 LLM head，生成文本 critique
> Scoring Head: 新加的 linear head，基于 (query + response + critique) 输出 scalar score
> ```

**Learning to provide critique from enhanced annotation.** The critique head ($h_l$) is trained to align with human-provided annotations. The loss function for critique generation is:

$$\ell_{\mathrm{Critique}}(\theta) = \mathbb{E}_{\mathbf{x}, y, c} \Big[ -\sum_{t=1}^{|c|} \log \pi_\theta(c_t | c_{<t}, \mathbf{x}, y) \Big]$$

where $c_t$ is the $t$-th token in the critique $c$, $c_{<t}$ denotes the tokens preceding $c_t$, and $\pi_\theta(c_t | c_{<t}, \mathbf{x}, y)$ is the probability of token $c_t$ given its context, query **x**, and model response $y$.

> 💡 **Critique Loss**: 就是标准的 next-token prediction loss，目标是让模型学会生成 critique 文本。

However, as shown in Figure 3, while human-provided scoring reasons are highly accurate, they tend to be concise. Directly using these concise annotations as training targets for the reward model's language head does not yield significant performance improvements. To address this issue, we use **GPT-4o** to augment the human annotations by adding more detail and improving the fluency of the critiques. These enhanced scoring reasons are then used as the training targets for the language head. To prevent GPT-4o from introducing hallucinated content or irrelevant analysis, we impose strict constraints in the prompt (Table 7), to ensure the model only expands on the original content without introducing speculative or uncertain information.

> 💡 **Annotation Enhancement（关键步骤）**:
> ```
> 人工标注理由 (concise)
>   ↓ GPT-4o expansion (with strict constraints)
> Enhanced critique (detailed, fluent)
>   ↓ 作为 Critique Head 的训练目标
> ```
> **为什么需要 GPT-4o 扩展？**
> - 人工标注的理由虽然准确，但太简短（如"描述错误"、"有幻觉"）
> - 直接用简短理由训练，效果不好
> - GPT-4o 在**不引入新信息**的约束下，把简短理由扩展成详细的分析
> - Prompt 严格要求"只扩展原有内容，不引入推测或不确定信息"

**Scoring loss with teacher-forcing.** $h_r$ computes scalar rewards based on the query **x**, response $y$, and critique $c$. During training, we adopt a **teacher-forcing strategy**, where the scoring head uses **ground truth critiques** instead of critiques generated by itself. This avoids potential noise from model-generated critiques in the early stages of training. The scoring loss is defined as:

$$\ell_{\mathrm{Score}}(\theta) = \mathbb{E}_{\mathbf{x}, y_w, y_l} \Big[ -\log \sigma \Big( r(\mathbf{x}, y_w, c_w) - r(\mathbf{x}, y_l, c_l) \Big) \Big]$$

where $c_w$ and $c_l$ are the ground truth critiques for the preferred response $y_w$ and less preferred response $y_l$, respectively, $r(\mathbf{x}, y, c)$ is the reward score computed from **x**, $y$, and $c$.

> 💡 **Teacher-forcing 策略**: 训练时 scoring head 用的是 GT critique（不是模型自己生成的）。这避免了训练早期模型生成的 critique 质量差导致的错误传播。**推理时**才用模型自己生成的 critique。

**Joint training objective.** The overall training objective combines the critique generation loss and the scoring loss:

$$\ell_{\mathrm{Total}}(\theta) = \ell_{\mathrm{Critique}}(\theta) + \ell_{\mathrm{Score}}(\theta)$$

**Inference.** During inference, the critique head ($h_l$) generates a critique $c$ conditioned on the query **x** and response $y$. The scoring head ($h_r$) then uses **x**, $y$, and the generated critique $c$ to compute the final reward score $r(\mathbf{x}, y, c)$. This two-step process mirrors the human evaluation process by explicitly reasoning about critiques before scoring.

> 💡 **训练 vs 推理对比**:
> | | Critique 来源 | 目的 |
> |---|---|---|
> | **训练** | GT critique (teacher-forcing) | 避免噪声传播 |
> | **推理** | 模型生成的 critique | 完全自主 |
>
> 这类似于 Chain-of-Thought: 先"想"（critique）再"答"（score）。

**MM-RLHF-RewardBench.** To evaluate the effectiveness of the signals provided by our reward model in guiding subsequent model training, we randomly sample 10 examples from each category of the MM-RLHF dataset to create a test set. Each example includes multiple model responses and their corresponding rankings, enabling the generation of several comparison pairs. This results in a total of **170 pairs** for evaluation. We design two evaluation metrics:
1. **Traditional Accuracy (ACC)**: Measures the proportion of cases where the model correctly identifies the preferred response.
2. **ACC+**: Measures the proportion of cases where the model correctly ranks **all** response pairs for a given sample. This metric emphasizes the model's ability to handle challenging cases, such as those with small ranking differences or hard-to-distinguish pairs.

> 💡 **RewardBench 设计**: 
> - 170 comparison pairs, 5 categories (MCQ, Long, Short, Safety, Video)
> - ACC: 简单准确率（pairwise 正确率）
> - ACC+: 更严格——一个 query 的**所有** pairs 都对才算对
> - ACC+ 是更有意义的指标，因为它测试的是 RM 的排序一致性

---

### 3.3 Discussion

In the MLLM community, there is currently no unified paradigm for the design of reward models. Some approaches rely on traditional reward models [58], which lack interpretability due to their reliance on scalar outputs. Others directly use LLMs to generate rankings [67], which heavily depend on instruction-following capabilities and often exhibit high variance in scoring. In the broader LLM community, works such as [74] explore reward models that first generate critiques. However, their focus is primarily on improving the reliability of model-generated critiques, such as increasing scoring confidence through multiple sampling—a goal distinct from ours. To the best of our knowledge, **this is the first study to explore how MLLMs can effectively leverage human annotations to enhance both interpretability and the final model's scoring ability.**

> 💡 **定位**: 三种 RM 范式的对比:
> | 范式 | 代表 | 优点 | 缺点 |
> |------|------|------|------|
> | Scalar RM | LLaVA-RLHF | 简单高效 | 不可解释，浪费 annotation |
> | LLM-as-Judge | LLaVA-Critic | 可解释 | 依赖 instruction-following，高方差 |
> | **Critique-Based RM (本文)** | MM-RLHF-Reward | 可解释 + 稳定 | 需要高质量 annotation |

---

## 💡 Section 总结

### 核心洞察
1. **从 scalar 到 critique**: 传统 RM 只输出数字，浪费了人工标注的丰富信息。Critique-Based RM 先推理再打分。
2. **GPT-4o 扩展 annotation 是关键**: 人工标注太简短，直接用效果不好。GPT-4o 在约束下扩展后，ACC+ 提升 17%！
3. **Teacher-forcing 防止错误传播**: 训练时用 GT critique，推理时用模型自生成。
4. **Multiple sampling 反而降低性能**: 与 LLM 领域的经验不同——因为模型已经能生成较准确的 critique，额外采样引入了噪声。
5. **7B 模型超过 72B**: MM-RLHF-Reward-7B 超过 Qwen2-VL-72B 和多个闭源模型，说明高质量数据+好的训练范式 > 模型规模。

### 对 Apple Assignment 的价值
- **Critique-Based RM** 是一个重要的技术创新，可以在 assignment 中详细讨论
- 如何将 concise human annotations 转化为 detailed model-friendly critiques 是一个实用的工程技巧
- Teacher-forcing + 联合训练的策略值得借鉴
