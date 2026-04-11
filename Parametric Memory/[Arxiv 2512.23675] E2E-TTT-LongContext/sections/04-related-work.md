[← 返回 README](../README.md)

# 4. Related Work

## 📌 预览

Related work 写得异常详细（几乎占全文 1/5），因为作者要解释 TTT 和 continual learning / meta-learning / fast weights / dynamic evaluation 这一堆**历史很长的"亲戚家族"**的关系。这个 section 的真正价值：**把 TTT 放到一个更大的概念地图里**。结构：
- 4.1 Continual Learning —— 把本文放在 CL 的历史脉络
- 4.2 Test-Time Training —— TTT 的三个 subcategories + "focus on the present" 哲学
- 4.3 Fast Weights & FWPs —— Schmidhuber 1992 起的古老思想，本文与 Clark et al. 2022 的渊源
- 4.4 Learning to Learn —— 与 MAML 的区别

---

## 4.1 Continual Learning

> 💡 **4.1 要点预览**: 对 continual learning 这个老字号研究方向的重新定位 —— 本文不是"每天更新一次模型"的那种 CL，而是"每个 test instance 都开一次迷你训练"的 TTT 视角下的 CL。

Most of today's AI systems remain static after deployment, even though the world keeps changing. The high-level goal of continual learning is to enable AI systems to keep changing with the world, similar to how humans improve throughout their lives [36, 22].

Conventionally, continual learning as a research field has focused on learning from a distribution that gradually changes over time [68, 96, 33]. For example, one could update a chatbot model every hour using new knowledge from the Internet, while typical use cases of the model may require knowledge from both the past and the present [80, 56, 100]. More formally, at each timestep, we sample new training and test data from the current distribution, update our model using the new training data, and then evaluate it on all the test data up to the current timestep. Under this setting, most algorithms focus on not forgetting the past when learning from the present [75, 64, 57, 30].

> 💡 **批注 — "传统 CL" 和"本文 CL"的对比**:
>
> | 维度 | 传统 Continual Learning | 本文 CL 框架 |
> |---|---|---|
> | 时间尺度 | 小时/天 | 每个 test instance |
> | 个性化 | 无 (所有用户共享) | 有 (每个 sequence 一次) |
> | 主要关切 | Catastrophic forgetting | 如何有效压缩 |
> | 评估 | 过去 + 现在的所有 test | 当前 test sequence |
>
> 这个对比能帮你快速理解 —— 本文的 "continual learning as framework" 是借**概念**不是借**具体算法**。经典 EWC / LwF / GEM 等方法在本文里都没用上，因为 TTT 的时间尺度是"每句话重来一次"，根本不存在"跨 instance forgetting"这个问题。

---

## 4.2 Test-Time Training

> 💡 **4.2 要点预览**: TTT 家族的系统分类 —— 三个子类别：**(1) TTT on Nearest Neighbors**（最古老，从 1970s 的 locally weighted regression 起）、**(2) TTT for Novel Instances**（通过 self-supervision 做泛化，Sun 2020 / AlphaProof 2024）、**(3) TTT on Sequences**（本文所在类别，长 context / 视频 / 机器人）。

The algorithmic framework of test-time training has the same high-level goal as continual learning, but it focuses on two aspects where human learning stands out from the forms of continual learning in the conventional literature.

First, each person has a unique brain that learns within the context of their individual life. This personalized form of continual learning is quite different from, for example, the chatbot model that is fine-tuned hourly using the latest information available worldwide. While such a model does change over time, it is still the same at any given moment for every user and every problem instance.

Second, most human learning happens without a boundary between training and testing. Consider your commute to work this morning. It is both "testing" because you did care about getting to work this very morning, and "training" because you were also gaining experience for future commutes. But in machine learning, the train-test split has always been a fundamental concept.

The concept of test-time training is introduced to realize these two special aspects of human learning. Training typically involves formulating a learning problem (such as empirical risk minimization) and then solving it. Following [86], test-time training is defined as any kind of training that formulates a potentially different learning problem based on each individual test instance.

> 💡 **TTT 的两个人类学习特征**:
>
> 1. **个性化**：每个 test instance 有自己的"训练" —— 每个人脑子都不一样，每条 sequence 都得到自己的 $W_T$。
> 2. **无 train/test 界限**：你通勤既是"测试今天能不能到公司"也是"训练以后怎么通勤" —— 本文的 decode 阶段 self-training 就是这个思想。
>
> **TTT 的正式定义** (Sun 2020)：任何基于单个 test instance 重新形成一个学习问题的 training。关键字 "based on individual test instance" —— 这是 TTT 和 domain adaptation 的根本区别。

This concept has a rich history in AI. A well-known example in NLP is dynamic evaluation, pioneered by Mikolov et al. [72] and extended by Krause et al. [60], which our Subsection 2.1 builds upon. In computer vision, early examples have also emerged in applications such as face detection [52], video segmentation [73], super-resolution [83], and 3D reconstruction [70]. Next, we discuss three popular forms of test-time training today, with an emphasis on their connections to each other and to historical examples.

### 4.2.1 TTT on Nearest Neighbors: Larger Effective Capacity

One simple form of test-time training was called locally weighted regression in the 1970s [85, 18], local learning in the 1990s [12], and KNN-SVM in the 2000s [109]: Given a test instance, find its nearest neighbors in the training set, and then train (or fine-tune) the model on these neighbors before making a prediction. This procedure can significantly increase the effective capacity of the model; for example, it allows a linear model to fit a highly nonlinear ground truth [85].

This simple form captures one of the key intuitions of test-time training. In the conventional view of machine learning, a model, once trained, no longer changes at test time. As a consequence, it must prepare to be good at all possible inputs in the future. This task can be very hard, because being good at all possible futures limits the model's capacity to be good at any particular one. But only one future is actually going to happen. So why not train our model once this future happens?

Recently, [35] extended this idea to modern language models and observed a similar benefit of larger effective model capacity after test-time training, and [45] further improved these results through better strategies for neighbor selection. In addition, [46] showed that test-time training on neighbors from the training set is also effective with RL for reasoning tasks, and [5] developed the same idea for visual-motor tasks.

> 💡 **4.2.1 批读 — 最古老的 TTT 形式**:
>
> "给定 test 点，在训练集里找 KNN，只对这些邻居重新训练一个小模型。"—— 这个思路被反复发明：
>
> - 1970s: Locally Weighted Regression (Stone 1977, Cleveland 1979)
> - 1990s: Local Learning (Bottou & Vapnik 1992)
> - 2000s: KNN-SVM (Zhang et al. 2006)
> - 2023: Hardt & Sun 的 "TTT on nearest neighbors for LLMs"
> - 2024: Hubotter et al. 的 "Active fine-tuning of LLMs"
>
> **核心 insight (本节最重要的一段)**：一个 static model 必须"擅长所有可能的未来"—— 这个 framing 在 Section 3.4.1 已经出现过（解释为什么 TTT-E2E 早期 token 就比 full attention 好）。现在这里是 TTT 思想体系的**统一哲学**：
>
> > "model must prepare to be good at all possible inputs in the future" ← too hard
> > "but only one future is actually going to happen" ← focus on this one instead
>
> 这是本文反复引用的 **"focus on the present"** 的哲学根基。

### 4.2.2 TTT for Novel Instances: Better Generalization

As models become larger today, their competence is often limited not by their capacity, but by the amount of available training data, especially when they need to generalize to novel test instances that are "out-of-distribution". In this case, it is even harder to prepare for all possible test instances in the future, especially the novel ones, with a static model. But once a specific test instance is given, we can use it to generate relevant data, which we can then use for training [88]. In other words, the "neighbors" for TTT do not have to come from the training set; they can also be generated on-the-fly.

Since the test instance is unlabeled, one way to make it useful for training is through self-supervision, which generates new pairs of inputs and labels for an auxiliary task such as masked reconstruction (e.g., BERT [23] and MAE [37]). While the auxiliary task is different from the main prediction task, improving performance in one can help the other through their shared representations. This form of TTT can significantly improve generalization under distribution shifts [88, 28].

Recently, TTT has been an important part of AlphaProof [44], which achieved IMO silver-medal standard in 2024. Given each test problem, their system first generates a targeted curriculum of easier problems by prompting a language model, and then performs reinforcement learning on the generated data. Another recent work, Akyurek et al. [2], found TTT effective for few-shot reasoning tasks such as ARC-AGI. Their system generates augmentations of the few-shot demonstrations in the test problem then performs supervised learning.

> 💡 **4.2.2 批读 — TTT 的第二形态：通过自监督做泛化**:
>
> 核心区别：
> - 4.2.1 的 TTT 邻居**来自训练集**。
> - 4.2.2 的 TTT 邻居**在 test time 生成** —— 通过 masked reconstruction (Sun et al. 2020, TTT MAE) / augmentation (Akyurek et al. ARC-AGI) / RL curriculum (AlphaProof)。
>
> AlphaProof 在 IMO 2024 拿了银牌 —— 这是 TTT 思想在 reasoning 领域的重要成果。

### 4.2.3 TTT on Sequences: Longer Memory

In all the forms of TTT discussed so far, the model is reset after each prediction because the test instances are independent. However, humans do not constantly reset their minds. Our memory of how to solve the previous learning problem often helps with the current one, because our experience in the world is much closer to a correlated sequence of data than independent ones.

Sequential applications, such as videos and robotics, offer a playground that bridges this difference. For example, [34] extended TTT with self-supervision to a manipulation policy whose input is a video stream of the robot's workstation, and found that no reset leads to a much larger improvement. Recently, [101] extended the same idea to video segmentation using a model trained with only images. In this case, TTT can be viewed as compressing the context from previous frames into the weights of the model without learning to learn, similar to the naive version of our method in Subsection 2.1.

**TTT-KVB.** Text, like videos, is a form of sequence. In Subsection 2.4, we have discussed TTT-KVB as the most relevant line of prior work [87, 110, 19], which includes variants such as MesaNet [98], Titans [7], and Nested Learning [6]. The popularity of TTT-KVB has two side effects:

- Because the KVB objective is inspired by self-attention, which stores the keys and values, many think that long-context TTT is about memorization instead of generalization.
- Because TTT(-KVB) layers are drop-in replacements for self-attention layers, many also think of long-context TTT as an approach to architecture design.

Our work shows that long-context TTT does not need to memorize the association between the keys and values. In addition, our method is derived purely under the formulation of a continual learning problem, with minimal changes to the architecture.

> 💡 **4.2.3 批读 — 本文所在的子类别 + 对 TTT-KVB 的"祛魅"**:
>
> 这个子类别的 key distinguishing feature 是**"不 reset"** —— 前一个 prediction 的更新**保留**给下一个 prediction 用（而 4.2.1 和 4.2.2 都是 instance-independent 的）。
>
> **Robotics/Video 的先行工作**：
> - Hansen et al. 2020: TTT on manipulation policy with video stream → "不 reset 比 reset 好很多"
> - Wang et al. 2023: TTT on video streams
>
> **然后作者做了一次很重要的"祛魅"**，批评了 TTT-KVB 这个主流 (Titans/MesaNet/Nested Learning) 家族的两个 misconceptions：
>
> 1. **"Long-context TTT = memorization"** ← 错。本文说明你**不需要存 KV pairs**。长 context 靠的是压缩后的**泛化**能力。
> 2. **"Long-context TTT = architecture design"** ← 错。本文说明架构改动是 minimal 的，改动 primarily 在**训练/推理流程**上。
>
> 这两段话基本是在和 Titans/MesaNet/Nested Learning 作者们的**立场辩论**。

---

## 4.3 Fast Weights and Fast Weight Programmers

> 💡 **4.3 要点预览**: Schmidhuber 从 1992 年就开始提"fast weights" —— 这是另一个被本文重新 unify 的老思想线。关键点：TTT 可以看作 fast weights 的一个 special case；而 Clark et al. 2022 是**最相关的一篇**，作者坦率承认是本文的直接 inspiration。

The general idea of fast weights is to update the parameters of a "fast" model on only the most relevant data, as opposed to the conventional practice of updating a "slow" model on all data [94]. This idea has existed since the 1980s [26, 38, 97]. Because the most relevant data can often include the test instance itself, test-time training can be viewed as a special case of fast weights, with a heavier emphasis on the formulation of an explicit learning problem.

The general idea of fast weight programmers (FWPs) is to update the fast weights at test time with a "slow" model (as a programmer) that, in turn, is updated less frequently, if at all [79]. In our method, the inner-loop weights $W$ can be viewed as "fast" and the outer-loop weights $\theta$ as "slow". Therefore, our method can be viewed as a special case of FWPs [58]. Next, we briefly review some of the literature on FWPs in the order of relevance.

> 💡 **批注 — Fast Weights 术语地图**:
>
> | 术语 | 含义 | 本文对应物 |
> |---|---|---|
> | **Slow weights** | 不频繁更新（或从不更新）的权重 | outer loop 参数 $\theta$ (= $W_0$ + 其他) |
> | **Fast weights** | 频繁更新（每个 test instance 都更新）的权重 | inner loop 的 $W_t$ |
> | **Fast Weight Programmer (FWP)** | "slow model" 作为 programmer 去更新 "fast model" | 本文的 outer-inner loop 结构 |
>
> 所以 TTT = FWP 的一个特例，而 TTT-E2E = FWP + Meta-learning + NTP loss。

**Clark et al. [17].** This work is the most relevant to ours in methodology. Given a Transformer baseline with full attention, they add an MLP layer as fast weights, whose initialization is trained as slow weights along with the rest of the model. Similar to ours, their method updates the fast weights by taking a gradient step on the next-token prediction loss computed over each chunk (mini-batch) of tokens. Their method significantly improves perplexity compared to the baseline but does not improve efficiency, since their combined architecture does not have linear complexity. In addition, their design adds the fast weights only to the end of the model instead of interleaving them with attention layers. In our experiments, interleaving proves to be critical for maintaining the performance gain on top of larger baselines. Nevertheless, we find Clark et al. to be a valuable inspiration. An earlier work [102] also contains sketches of a similar idea with limited experiments.

> 💡 **Clark et al. 2022 批读 — "祖父论文"**:
>
> 这可能是本文**方法论上最接近**的前辈：
>
> **相同点**：
> - 都加 MLP 作 fast weights
> - 都用 NTP loss 更新 fast weights
> - 都是 chunk-wise (mini-batch) 更新
> - 都用 outer loop 学 fast weights 的初始化
>
> **本文的两点推进**：
> 1. Clark et al. 只在**模型末尾**加一层 fast weight MLP → 本文**interleave** 到多个 block（最后 1/4）里。作者说"interleaving proves to be critical for larger baselines"。
> 2. Clark et al. 还是 full attention（没有线性复杂度） → 本文用 SWA 做到线性复杂度。
>
> **本文把 Clark et al. 的实验结果从"perplexity improvement on a Transformer"扩展成了"一个和 full attention scaling 等同的 RNN"**。这一段写得非常诚实，直接点名致敬。

**FWPs for long context.** Many methods addressing the problem of long context have roots in the literature of FWPs. In particular, [79] (Schmidhuber, 1992) has been a major source of inspiration for modern RNN layers, such as linear attention [55, 77], DeltaNet [76, 106], and Gated DeltaNet [105], one of our baselines. In addition, some of the work on TTT for long context [87, 110] (discussed in Subsection 4.2) can also be viewed as FWPs, due to the connection between TTT and fast weights. Notably, one instantiation in Irie et al. [49] uses MLPs as layer-wise fast weights for long context, preceding the similar instantiation in [87].

**Other FWPs.** While the FWPs above can be interpreted through TTT, many other varieties cannot. For example, [50] designs the fast weights to be programmed by themselves, [51] builds an image generator using the images as fast weights, [48] applies continuous-time extensions of FWPs to time-series classification, while [47] and [31] demonstrate how the choice of update rules affects the expressiveness of FWPs on formal language recognition tasks. In fact, all networks with some gating mechanism, such as Transformers with SwiGLU blocks [82], can also be viewed as FWPs [32].

> 💡 **批注 — "所有带 gating 的网络都是 FWP"的宽广观点**:
>
> 这是一个非常宽的抽象：**SwiGLU MLP 本身就是一个 FWP** —— 因为 gating 可以被看作一个 slow model 在"programming" downstream activations。Mamba 的作者也有这个论断。
>
> 从这个角度看，现代 LLM 的**几乎每一层**都在做某种形式的 fast weight programming，只是程度和显式度不同。

---

## 4.4 Learning to Learn

> 💡 **4.4 要点预览**: 快速讨论和 MAML 的关系 —— 同样是 outer loop 学 inner loop 初始化，但 problem setting 不同。MAML 的 inner loop 学**一整个数据集**；本文的 inner loop 学**一条序列**。

For decades, researchers have been arguing that learning to learn, also known as meta-learning or bi-level optimization, should be an important component of intelligence [78, 9, 93, 61]. Perhaps the most relevant work in this field is MAML [27]. Similar to our work, MAML also has an outer loop that learns the initialization of the inner loop through gradients of gradients. The main difference between MAML and our work lies in the problem setting. Specifically, their inner loop learns from an entire dataset at a time, so the outer loop requires a large collection of datasets. In contrast, our work addresses the problem of language modeling by casting it as learning to learn. In principle, any supervised learning problem can be cast into our problem formulation.

> 💡 **4.4 批读 — 和 MAML 的根本差别**:
>
> | 维度 | MAML (Finn 2017) | TTT-E2E (本文) |
> |---|---|---|
> | Inner loop 输入 | 一整个 few-shot 数据集 | 一条 sequence |
> | Inner loop 步数 | 5-10 步 | $T/b$ 步 (几百步) |
> | Outer loop 需要 | 很多个**不同 task 的 dataset** | 一个**标准的 text corpus** |
> | 应用场景 | Meta-learning for few-shot | Language modeling as meta-learning |
>
> 本文最有意思的 philosophical claim：**"In principle, any supervised learning problem can be cast into our problem formulation."** —— 这句话说明本文不是只在长 context 上有用，而是一个**通用的训练范式**。任何监督任务都可以变成"把训练集当作 context，把 test 当作 prediction"的形式。不过这只是理论 claim，paper 里没做实验验证。

---

## 🔖 Section 4 总结

### TTT 家族地图（本文所处位置）

```
                      Learning to Learn
                     (Thrun & Pratt 1998)
                             │
                    ┌────────┴────────┐
                 MAML              Fast Weights
               (Finn 2017)    (Schmidhuber 1992)
                    │                 │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
         TTT on Nearest Neighbors    TTT on Sequences
         (Stone 1977 → Hardt 2023)  (Sun 2020 → This paper)
                                         │
                             ┌───────────┴───────────┐
                        TTT-KVB family           TTT-E2E (ours)
                   (Titans, MesaNet, Nested)
                       [层内 KV 重建]              [整网 NTP]
                       [E2E at train, ✗ test]  [E2E at both]
```

### 本文在 TTT 家族中的定位

1. **TTT 的形态**：Sequences (4.2.3)，不 reset。
2. **TTT 的 loss**：Next-token prediction (E2E at test time)。
3. **学习的对象**：Fast weights (MLP weights)。
4. **初始化来源**：Meta-learning outer loop (E2E at training time)。
5. **与 TTT-KVB 的差别**：替换掉 layer-wise KVB loss。
6. **与 Clark et al. 2022 的差别**：Interleaving into multiple blocks + linear complexity via SWA。
7. **与 MAML 的差别**：Problem setting (sequence vs dataset)。

### 本文对 TTT 研究方向的"祛魅"论点

- Long-context TTT 不需要 memorization / KV storage。
- Long-context TTT 不是 architecture design 问题，而是 continual learning 问题。
- "Focus on the present" 而不是"擅长所有可能的未来" —— 这是 TTT 的哲学内核。
