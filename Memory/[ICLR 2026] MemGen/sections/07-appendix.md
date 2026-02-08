[← 返回 README](../README.md)

# Appendix

> 💡 **Appendix 内容丰富**：包含 SFT/GRPO 两种优化算法的完整推导（B）、实验细节（C）、大量补充实验（D）、与检索记忆的集成（E）、latent token 案例展示（F）、记忆功能研究的完整方法论（G）。以下摘录关键部分。

---

## A Additional Related Works

LLM Decoding & RL. Two additional topics that relate to our work are LLM decoding and reinforcement learning (RL). From the decoding perspective, MemGen dynamically generates and inserts latent tokens, which shares similarity with speculative decoding where a drafter model receives the current decoding context and produces subsequent drafted tokens. However, these methods primarily aim to accelerate LLM inference, whereas MemGen focuses on leveraging latent states as effective carriers of memory. From the RL perspective, MemGen employs rule-based RL to train the memory trigger, which is closely related to RLVR, including GRPO from DeepSeek-R1 and its various derivatives. While there exist efforts combining RL with agent memory, most do not address self-improving memory; for example, MemAgent (Yu et al., 2025) and MEM1 (Zhou et al., 2025) focus on handling long-context inputs rather than evolving memory mechanisms.

---

## B Optimization Algorithm on Memory Weaver

### B.1 Combining MemGen with SFT

The objective of Supervised Fine-Tuning is to train the memory weaver to generate latent memories that guide the frozen reasoner πθ to replicate the behavior observed in a dataset of high-quality demonstration trajectories. The SFT loss function:

$$
\mathcal{L}_{\text{SFT}}(\theta') = -\mathbb{E}_{(x_i, \tau_i^*) \sim \mathcal{H}} \left[ \sum_{\ell=0}^{T_i-1} \sum_{j=1}^{L_t} \log \pi_\theta(\mathbf{z}_{i,t,j}^* \mid s_{i,t}, \mathbf{z}_{i,t,<j}^*, \mathbf{M}_{i,t,j}) \right],
$$

where the latent memory M_{i,t,j} = W_{θ'}(H_{i,t,<j}). The gradients are computed exclusively with respect to the weaver's parameters θ'.

> 💡 **SFT 训练的要点**：Reasoner 和 Trigger 都冻结，只更新 Weaver。用 expert trajectory 的 token-level NLL loss 训练——weaver 学习"生成什么样的 latent memory 才能让 reasoner 复现 expert 行为"。

### B.2 Combining MemGen with GRPO

The memory weaver can also be trained using GRPO. For each task xᵢ, generate K distinct trajectories, compute group-relative advantage:

$$
A(\tau_{i,k}) = R(\tau_{i,k}) - \bar{R}(\mathcal{G}_i),
$$

The final GRPO objective:

$$
\mathcal{V}(\theta') = \mathbb{E}_{x_i \sim \mathcal{H}, \mathcal{G}_i \sim \Pi_\theta^{\mathcal{W}_{\theta'}, \mathcal{T}}} \left[ \frac{1}{K} \sum_{k=1}^{K} A(\tau_{i,k}) \log \Pi_\theta^{\mathcal{W}_{\theta'}, \mathcal{T}}(\tau_{i,k} \mid x_i) - \beta \text{KL}(\Pi_\theta^{\mathcal{W}_{\theta'}, \mathcal{T}}(\cdot \mid x_i) \| \Pi_{\text{ref}}(\cdot \mid x_i)) \right],
$$

where gradients are computed only for the weaver's parameters θ'.

> 💡 **SFT vs GRPO 训练 Weaver**：SFT 需要 expert demonstrations（收集成本高），GRPO 只需要 reward signal（更灵活）。实验中 GRPO 变体普遍优于 SFT（Table 1），但 SFT 在 GPQA 等知识密集任务上有时更好。

---

## C Experimental Details

### C.1 Training Dataset Setup

Training Datasets: Official training splits of all evaluated datasets, except PopQA (no training set; use TriviaQA-trained model). Training the Memory Weaver first (without trigger), then training the Memory Trigger (with fixed weaver).

### C.2 Parameter Configurations

| Settings | Hyperparameters |
|----------|----------------|
| SFT | batch=4, lr=1e-5, epochs=2, warmup=0.1, cosine schedule |
| GRPO | batch=8, epochs=2, beta=0.0, lr=1e-5, cosine schedule |
| LoRA | r=16, alpha=32, target=[q_proj, v_proj], dropout=0.1 |

> 💡 **LoRA 配置很轻量**：只训 q_proj 和 v_proj，r=16——这意味着 trigger 和 weaver 的可训练参数量非常小（大约几 MB），却能带来显著的性能提升。

---

## D Extra Results

### D.1 Continual Learning Result

**Table 4** Continual learning results of Qwen2.5-1.5B-Instruct across four datasets (AQuA→GPQA→GSM8K→KodCode).

| Trained On | Method | AQuA | GPQA | GSM8K | KodCode |
|-----------|--------|------|------|-------|---------|
| - | Vanilla | 41.34 | 11.62 | 39.51 | 24.55 |
| AQuA | SFT | 42.52 | 16.67 | 42.10 | 18.20 |
| AQuA | MemGen SFT | **43.31** | **19.70** | 39.80 | 19.55 |
| GPQA | SFT | 38.55 | 17.17 | 45.74 | 18.50 |
| GPQA | MemGen SFT | **39.85** | **20.72** | **47.96** | **28.80** |
| GSM8K | SFT | 33.46 | 13.13 | 52.31 | 19.45 |
| GSM8K | MemGen SFT | **38.43** | **21.72** | **55.67** | 19.75 |
| KodCode | SFT | 28.61 | 2.53 | 24.14 | **54.10** |
| KodCode | MemGen SFT | **40.34** | **20.09** | **53.72** | 52.95 |

> 💡 **最后一行最震撼**：训完 4 个域后，SFT 的 GPQA 从 11.62% 崩到 2.53%，而 MemGen 保持 20.09%。AQuA 也是 40.34% vs 28.61%。这就是"经验存在 weaver 里、不动 reasoner"的威力。

### D.2 Trigger Frequency Visualization

![Figure 7: Qwen2.5-1.5B 触发频率](../images/3169f885e5fec8f783562b988378fb8ac5df4cce4c66d5aa166bf25de85bdaca.jpg)
> **Figure 7** Memory invocation frequency across benchmarks at inference (trained on MemGen SFT+Qwen2.5-1.5B+GSM8K).

![Figure 8: SmolLM3-3B 触发频率](../images/c0ec64ab4f6c5516d7928d4a4b96e9c794a9211cc8779691d5d4db9e195df8ba.jpg)
> **Figure 8** Memory invocation frequency across benchmarks at inference (trained on MemGen SFT+SmolLM3-3B+GSM8K).

> 💡 **跨 backbone 的一致性**：无论是 1.5B、3B 还是 8B，trigger 在训练域（GSM8K）上都比在其他域更频繁地调用记忆——这个行为模式是 robust 的。

![Figure 9: GSM8K 训练泛化](../images/22ef291c4ffd8d3bcf11d4b87285eb007cbd456916a864f424a073949cc6effe.jpg)
> **Figure 9** The generalization study of MemGen. We train MemGen SFT on GSM8K and evaluate it on all four datasets.

![Figure 10: KodCode 训练泛化](../images/fe4723fb9e3d5c660d5fd80d76980f12daaeae99a38205f859a7d1b2fb0ad7a7.jpg)
> **Figure 10** The generalization study of MemGen. We train MemGen SFT on KodCode and evaluate it on all four datasets.

### D.3 Framework Analysis

#### D.3.1 Ablation Study

**Table 5** Ablation study of different memory invocation strategies.

| Memory Invocation Strategy | GPQA | KodCode | TriviaQA |
|---------------------------|------|---------|----------|
| Random (p=0.2) | 15.66 | 54.55 | 63.55 |
| Random (p=0.5) | 16.66 | 52.95 | 57.28 |
| Random (p=0.8) | 12.63 | 53.60 | 62.22 |
| All delimiters activated | 17.34 | 56.20 | 64.15 |
| **MemGen's dedicated Trigger** | **18.28** | **58.16** | **65.02** |

> 💡 **Trigger 的必要性验证**：随机触发（任何 p 值）都不如训练好的 trigger，甚至"在所有标点处触发"也不如——说明"在正确的时机想起正确的记忆"是需要学习的能力，不能靠启发式规则。

#### D.3.2 Analysis of Memory Weaver

**Table 6** Ablation study of the latent weaver instantiation.

| Base LLM: Qwen2.5-1.5B | GPQA | KodCode | TriviaQA |
|------------------------|------|---------|----------|
| LoRA (r=16, α=32) | 18.28 | 58.16 | 65.02 |
| Full SFT | 21.21 | 60.00 | 67.10 |

> 💡 **Full SFT weaver 更强但代价更大**：全参数 weaver 在所有指标上优于 LoRA weaver，但参数量大幅增加。LoRA 版本已经足够 competitive，是实际部署的首选。

#### D.3.3 Efficiency Analysis

**Table 7** Average per-task inference time (seconds) and task performance.

| Model & Method | KodCode Time | KodCode Acc | ALFWorld Time | ALFWorld Acc | TriviaQA Time | TriviaQA Acc |
|---------------|-------------|------------|--------------|-------------|--------------|-------------|
| Qwen2.5-1.5B Vanilla | 11.96 | 24.55 | 21.17 | 22.54 | 2.18 | 32.10 |
| Qwen2.5-1.5B SFT | 2.01 | 55.83 | 10.79 | 36.57 | 1.98 | 63.84 |
| Qwen2.5-1.5B MemGen SFT | 2.94 | 58.16 | 12.94 | 40.30 | 2.05 | 65.02 |
| SmolLM-3B Vanilla | 13.12 | 37.05 | 34.82 | 18.96 | 4.26 | 10.47 |
| SmolLM-3B MemGen SFT | 3.48 | 62.65 | 14.69 | 50.60 | 3.16 | 68.13 |
| Qwen3-8B Vanilla | 16.99 | 49.10 | 55.42 | 58.93 | 8.70 | 52.18 |
| Qwen3-8B MemGen SFT | 7.56 | 66.15 | 20.08 | 85.82 | 6.25 | 77.22 |

> 💡 **推理时间显著减少**：MemGen 不仅不增加推理时间，反而因为更高效的推理路径大幅减少了时间。SmolLM-3B 在 ALFWorld 上从 34.82s→14.69s（-58%），同时准确率 +31.64%。这个结果对实际部署非常友好。

---

## E Integration with Retrieval-based Memory

When triggered, any retrieval-based system can provide textual memory, which is merged with the hook H_{t,<j} and fed into W to produce latent memory. The query is the decoded text generated so far: q_{t,j} = Decode(z_{t,<j}). Retrieved snippets C_t are encoded and concatenated with H_{t,<j}:

$$
\mathbf{M}_t = \mathcal{W}_{\text{weaver}}\big([\mathbf{H}_{t,<j}; \mathbf{E}_t]\big).
$$

**Table 8** Integration with ExpeL (backbone: SmolLM3-3B).

| Method | ALFWorld | TriviaQA | PopQA |
|--------|----------|----------|-------|
| Vanilla LLM | 18.96 | 10.47 | 8.23 |
| ExpeL | 36.18 | 46.20 | 28.16 |
| MemGen + ExpeL (w/o parametric memory) | 45.60 | 53.20 | 39.50 |
| **MemGen + ExpeL (w/ parametric memory)** | **75.90** | **76.40** | **60.23** |

> 💡 **最强组合**：MemGen + ExpeL (w/ parametric memory) 达到 ALFWorld 75.9%、TriviaQA 76.4%。即使关闭 weaver 自身的参数记忆只用 ExpeL 的检索结果，MemGen 仍然显著提升 ExpeL（+9.42% ALFWorld, +11.34% PopQA）——说明 weaver 作为"记忆合成器"本身就有巨大价值。

---

## F Latent Memory Token Demonstration

> 💡 **Case Studies 的观察**：
> - TriviaQA cases 中反复出现的 latent patterns: `[UPPORT...',eniable certif]` 和 `[essengeryyyyMMddELCOME certif]` 交替出现
> - GSM8K cases 中的 patterns: `[.keyword_pick]`, `[-animate.]`, `[-login.]`, `[ecies.]` 等
> - 同一 cluster 内的 pattern 高度一致（如 Cluster 2 大量使用 `[-animate.]`）
>
> 这些乱码实际上是 weaver 在 latent space 中编码的"行为指令"——告诉 reasoner 下一步应该怎么推理。虽然人类读不懂，但 LLM 能"理解"。

---

## G Memory Functional Study

### G.1 Visualization Process

Sequence Representation: compute mean embedding m̄ᵢ = (1/K) Σ m_{i,l} for each memory sequence, then apply t-SNE for 2D projection and K-means (N=4) for clustering.

### G.2 Failure Taxonomy

Eight failure types: Planning Failure, Compositional Reasoning, Tool Parsing Error, Tool Response Error, Answer Formatting Failure, Demand Misunderstanding, Think-Act Inconsistency, False Belief.

### G.3 Annotating Failure Modes and Filtering Latent Memory

Inference-time Filtering: for each new memory sequence, compute its mean embedding m̄_new, find top-k (k=10) nearest neighbors from E_comp = E_vocab ∪ {μ₁, ..., μ_N}. If target cluster centroid μⱼ ∈ S_k(m̄_new), the entire memory sequence is discarded.

> 💡 **消融方法的细节**：不是在训练时去掉某个 cluster，而是在推理时用 nearest neighbor 检查新生成的 memory 是否"属于"目标 cluster，如果是就丢弃。这保证了消融实验的公平性——其他所有组件（trigger、weaver 参数）都不变。

![Figure 11: SmolLM3-3B t-SNE](../images/6c71a22f0fe5ecd63ffc524191699f1a9eda967b4e4ea0ecd5313995395d0036.jpg)
> **Figure 11** (Up) t-SNE visualization of latent memories generated by MemGen+SmolLM3-3B across datasets; (Down) Latent memory visualization within the TriviaQA and GSM8K datasets, clustered using K-means.
