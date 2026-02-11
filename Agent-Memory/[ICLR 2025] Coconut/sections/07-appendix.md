[← 返回 README](../README.md)

# Appendix

## 📌 预览
数据集构造细节（GSM8k/ProntoQA/ProsQA 示例 + ProsQA DAG 构造算法）、推理效率时钟时间、$c=3$ 不稳定分析、大模型实验。

---

## A Datasets

### A.1 Examples

We provide some examples of the questions and CoT solutions for the datasets used in our experiments.

**GSM8k**

Question = "John cuts his grass to 2 inches. It grows .5 inches per month. When it gets to 4 inches he cuts it back down to 2 inches. It cost $100 to get his grass cut. How much does he pay per year?"

Steps = ["«4 - 2 = 2»", "«2 / .5 = 4»", "«12 / 4 = 3»", "«100 * 3 = 300»"]
Answer = "300"

**ProntoQA**

Question = "Brimpuses are not luminous. Shumpuses are amenable. Each yumpus is a lorpus. Gorpuses are shumpuses. Each zumpus is a grimpus. Gorpuses are rompuses. Dumpuses are not floral. Lempuses are cold. Brimpuses are impuses. Every lorpus is floral. Every rompus is transparent. Grimpuses are muffled. Rompuses are yumpuses. Rompuses are wumpuses. Zumpuses are fast. Wumpuses are bitter. Every sterpus is orange. Each lorpus is a vumpus. Yumpuses are feisty. Each yumpus is a lempus. Gorpuses are snowy. Zumpuses are gorpuses. Every lorpus is a sterpus. Stella is a brimpus. Stella is a zumpus. True or false: Stella is not floral."

Steps = ["Stella is a zumpus. Zumpuses are gorpuses.", "Stella is a gorpus. Gorpuses are rompuses.", "Stella is a rompus. Rompuses are yumpuses.", "Stella is a yumpus. Each yumpus is a lorpus.", "Stella is a lorpus. Every lorpus is floral.", "Stella is floral."]
Answer = "False"

**ProsQA**

Question = "Every shumpus is a rempus. Every shumpus is a yimpus. Every terpus is a fompus. Every terpus is a gerpus. Every gerpus is a brimpus. Alex is a rempus. Every rorpus is a scrompus. Every rorpus is a yimpus. Every terpus is a brimpus. Every brimpus is a lempus. Tom is a terpus. Every shumpus is a timpus. Every yimpus is a boompus. Davis is a shumpus. Every gerpus is a lorpus. Davis is a fompus. Every shumpus is a boompus. Every shumpus is a rorpus. Every terpus is a lorpus. Every boompus is a timpus. Every fompus is a yerpus. Tom is a dumpus. Every rempus is a rorpus. Is Tom a lempus or scrompus?"

Steps = ["Tom is a terpus.", "Every terpus is a brimpus.", "Every brimpus is a lempus."]
Answer = "Tom is a lempus."

> 💡 **三个数据集的难度对比**:
> - **GSM8k**: 多步算术推理，需要精确符号操作。Step 用 «» 标记计算步骤。
> - **ProntoQA**: 线性推理链 (A→B→C→...→结论)，只需逐步 follow 规则，无需搜索。
> - **ProsQA**: DAG 结构，有分支和死胡同。二选一问题 "Is X a Y or Z?"，需要在图中找到从 X 到正确答案的路径。这才是真正考验搜索和规划能力的任务。

---

### A.2 Construction of ProsQA

To construct the dataset, we first compile a set of typical entity names, such as "Alex" and "Jack," along with fictional concept names like "lorpus" and "rorpus," following the setting of ProntoQA (Saparov and He, 2022). Each problem is structured as a binary question: "Is [Entity] a [Concept A] or [Concept B]?" Assuming [Concept A] is the correct answer, we build a directed acyclic graph (DAG) where each node represents an entity or a concept. The graph is constructed such that a path exists from [Entity] to [Concept A] but not to [Concept B].

Algorithm 1 describes the graph construction process. The DAG is incrementally built by adding nodes and randomly connecting them with edges. To preserve the validity of the binary choice, with some probability, we enforce that the new node cannot simultaneously serve as a descendant to both node 0 and 1. This separation maintains distinct families of nodes and balances their sizes to prevent model shortcuts.

> 💡 **DAG 构造的关键设计**:
> - 用 bitwise OR 的 label 系统（1=node 0 后代, 2=node 1 后代, 3=两者, 0=都不是）控制图结构
> - 35% 概率新节点不能是 node 1 的后代，35% 不能是 node 0 的后代 → 维持两个 "家族" 分离
> - 采样权重偏好深层节点 → 确保推理路径足够长
> - 平均 23 个节点、36 条边、最短路径 3.8 步 → 足够复杂

### A.3 Statistics

| Dataset | Training | Validation | Test |
|---------|----------|------------|------|
| GSM8k | 385,620 | 500 | 1,319 |
| ProntoQA | 9,000 | 200 | 800 |
| ProsQA | 17,886 | 300 | 500 |

| # Nodes | # Edges | Shortest Path Len | # Shortest Paths |
|---------|---------|-------------------|-----------------|
| 23.0 | 36.0 | 3.8 | 1.6 |

---

## B Clock-Time Reasoning Efficiency Metric

We present a clock-time comparison to evaluate reasoning efficiency. The reported values represent the average inference time per test case (in seconds), with a batch size of 1, measured on an Nvidia A100 GPU.

| Method | GSM8k | ProntoQA | ProsQA |
|--------|-------|----------|--------|
| No-CoT | 0.03 | 0.03 | 0.08 |
| CoT | 0.26 | 0.85 | 0.47 |
| Coconut | 0.09 | 0.11 | 0.15 |

> 💡 **实际推理速度**: Coconut 比 CoT 快 3-8 倍！虽然有 sequential forward passes 的开销，但因为生成的 token 数少很多（不需要解码+重新编码），实际墙钟时间显著更快。

---

## C More Discussion

### C.1 Using More Continuous Thoughts

In Figure 8 (II), we present the performance of Coconut on GSM8k using $c \in \{0, 1, 2\}$. When experimenting with $c = 3$, we observe a slight performance drop accompanied by increased variance. Analysis of the training logs indicates that adding three continuous thoughts at once – particularly during the final stage transition – leads to a sharp spike in training loss, causing instability. Future work will explore finer-grained schedules, such as incrementally adding continuous thoughts one at a time while removing fewer language tokens, as in iCoT (Deng et al., 2024). Additionally, combining language and latent reasoning—e.g., generating the reasoning skeleton in language and completing the reasoning process in latent space—could provide a promising direction for improving performance and stability.

> 💡 **$c=3$ 的不稳定性**: 一次替换太多 language token → 训练 loss spike → 不稳定。解决方向：更细粒度的渐进替换（像 iCoT 那样 token-by-token 而非 step-by-step）。

### C.2 Coconut with Larger Models

We experimented with Coconut on GSM8k using Llama 3.2-3B and Llama 3-8B (Dubey et al., 2024) with $c = 1$.

| Model | no-CoT | Coconut |
|-------|--------|---------|
| Llama 3.2-3B | 26.0 | 31.7 |
| Llama 3-8B | 42.2 | 43.6 |

We observe consistent performance gains across both Llama 3.2-3B and Llama 3-8B models compared to the no-CoT baseline, though these improvements are not as pronounced as those previously demonstrated with GPT-2. One possible reason is that larger models have already undergone extensive language-focused pre-training, making the transition to latent reasoning more challenging.

> 💡 **大模型上提升有限**: Llama 3-8B 上只有 +1.4% (42.2→43.6)，远不如 GPT-2 上的提升。可能原因：
> 1. 大模型的 language reasoning 已经很强，latent reasoning 的边际收益小
> 2. 预训练时形成的 language-centric 表示可能抵抗 latent mode 的学习
> 3. $c=1$ 可能对大模型不够，需要更多 continuous thoughts
> 
> 这也是为什么作者强调 "scale to pretraining" 是关键未来方向——从头就学 latent reasoning 可能比 SFT 阶段引入更有效。

We emphasize that the primary goal of this paper is to highlight the promising attributes of latent-space reasoning and to initiate exploration in this new direction. Universally surpassing language-based CoT likely requires significant research efforts dedicated to latent space pre-training. We are encouraged by recent progress in this area (Geiping et al., 2025; Barrault et al., 2024; Gladstone et al., 2025). While these recent models provide scalable methods for latent representation learning, the latent spaces have not yet been explicitly optimized for reasoning. Integrating these recent advancements with Coconut presents an exciting and promising avenue for future research.

---

## 🔖 Section 总结

### 核心洞察
1. ProsQA 的 DAG 构造巧妙，通过控制节点家族分离确保任务有效性
2. Coconut 实际推理速度比 CoT 快 3-8 倍
3. $c$ 不能无限增大，$c=3$ 就不稳定了
4. 大模型上提升有限 → latent reasoning 需要从 pretraining 开始
