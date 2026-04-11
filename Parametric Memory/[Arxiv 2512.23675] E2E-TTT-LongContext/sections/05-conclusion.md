[← 返回 README](../README.md)

# 5. Conclusion & Author Contributions

## 📌 预览

Conclusion 只有一段。核心点：TTT-E2E 不是一个"架构设计"，而是一个**通用的长上下文建模方法**，可以叠在任何 baseline 架构之上。最后一句的生物学类比（SWA = 短期记忆，TTT 权重 = 长期记忆）是整篇论文的 ending note。

Author Contributions 写得非常详细（包括每个人离开/回归项目的时间点），这在 ML paper 里比较少见，值得读。

---

## 5 Conclusion

We have introduced TTT-E2E, a general method for long-context language modeling. In principle, TTT can be applied to any baseline architecture. For our experiments, this baseline is a Transformer with sliding-window attention. Adding our method to this baseline induces a hierarchy often found in biological memory, where the weights updated at test time can be interpreted as long-term memory and the sliding window as short-term memory. We believe that these two classes of memory will continue to complement each other, and stronger forms of short-term memory will further improve the combined method.

> 💡 **Conclusion 批读 — 三个核心 claims**:
>
> **1. TTT-E2E 是一个 method，不是 architecture**:
> 作者刻意强调 "general method" —— TTT-E2E 可以叠在 **任何** baseline 之上：full attention Transformer、SWA Transformer、Mamba、GDN...... 实验只选了 SWA Transformer 因为它简单干净。这是对"长 context = 架构改动"这一主流认知的再次否认。
>
> **2. 生物学记忆层级类比**:
>
> | 生物记忆 | 本文对应物 | 时间尺度 |
> |---|---|---|
> | Working Memory / STM | SWA 窗口 (k=8K) | 秒级 |
> | Long-term Memory | TTT 更新的 MLP 权重 | 分钟级 (整条 sequence) |
> | Semantic/Skill Memory | Pre-trained frozen 权重 | 整个 pre-train |
>
> 作者没有提到第三层但其实它隐含在**"双 MLP"**的设计里 —— static MLP 保留 pre-trained knowledge (技能记忆)，动态 MLP 做 TTT (工作记忆 → 长期记忆)。
>
> **3. "Stronger short-term memory will help"**:
> 未来改进方向：如果把 SWA 换成更强的 short-term memory（更大窗口？稀疏 attention？RingAttention？），TTT-E2E 的总体性能也会提升。这暗示本文方法**不是终点**，而是一个"骨架 × TTT"的组合的一个实例。

---

## Author Contributions

We state the contributions of each of the six core contributors.

**Arnuv Tandon** and **Karan Dalal** led the investigations into scaling with training compute and scaling with context length, developed the codebase, conducted the final experiments, managed our cluster, and played a central role in every aspect of this research, including its overall direction.

**Xinhao Li** developed and conducted most of the early experiments, including the toy experiments, co-developed our early codebase with Marcel, and contributed features for large-scale training.

**Daniel Koceja** led a set of mid-project experiments that brought clarity to the team's investigation into scaling with training compute.

**Marcel Rød** developed most of the early codebase and provided expertise to improve latency.

**Yu Sun** served as the project lead, making decisions on most day-to-day matters and on the project's overall direction. He developed the idea of TTT-E2E, designed most of the early experiments, and established the experimental protocols for the team. He also wrote the paper.

Yu Sun started the project in October 2024 together with Xinhao Li, Karan Dalal, and Daniel Koceja. Arnuv Tandon and Marcel Rød joined the project in November 2024. Karan Dalal and Daniel Koceja left from November 2024 through March 2025, and rejoined the project in April 2025. Marcel Rød left the project in May 2025. Xinhao Li and Daniel Koceja left in September 2025.

> 💡 **Author Contributions 批读 — 少见的项目时间线写法**:
>
> 这是很罕见的**项目考古学**：
>
> - **2024-10**: Yu Sun + Xinhao Li + Karan Dalal + Daniel Koceja 启动。
> - **2024-11**: Arnuv Tandon 和 Marcel Rød 加入。
> - **2024-11 ~ 2025-03**: Karan 和 Daniel **离开**（可能去别处做 intern / 另有项目）。
> - **2025-04**: Karan 和 Daniel **回归**。
> - **2025-05**: Marcel Rød 离开。
> - **2025-09**: Xinhao Li 和 Daniel Koceja 离开。
> - **2025-12**: 论文提交 arXiv。
>
> 整整 14 个月，**6 个核心贡献者**，只有 Yu Sun 和 Arnuv Tandon 是**全程**。这种写法很像博士论文的 acknowledgement 风格，把"谁在什么时候做了什么"写得非常清楚。对 ML 研究合作的透明度来说是个好 practice。
>
> **这类写法的意义**：让读者理解"为什么这篇 paper 看上去是一个超大合作但又有明确的 lead author"—— Yu Sun 是整个 TTT 方向的持续推动者（他之前写过 Sun et al. 2020, 2023, 2024 / TTT-KVB 等），这篇是他在 Astera Institute 重新组织的新一轮。

---

## 📎 Appendix 快速指引

（正文没有完整展开 appendix，但有两个地方要留意）

### Appendix A: Recipe for the Toy Example

- Toy 架构：Transformer 去掉所有 attention（只剩 MLP）。
- 训练 context：128
- 模型大小：125M 的一半（embed dim 384, heads 6）
- Learning rate sweep：3e-3 (full attention), 5e-3 (TTT 家族)
- 数据：DCLM

### Appendix B: Basic Recipe (Table 3)

| Params | Blocks | Dim | Heads | Pre-train Tokens | Pre-train LR | Fine-tune LR |
|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 | 2.5B | 3e-3 | 4e-4 |
| 350M | 24 | 1024 | 16 | 7B | 1.5e-3 | 4e-4 |
| 760M | 24 | 1536 | 16 | 15B | 1.25e-3 | 4e-4 |
| 1.3B | 24 | 2048 | 32 | 26B | 1e-3 | 4e-4 |
| 2.7B | 32 | 2560 | 32 | 54B | 8e-4 | 4e-4 |

- Fine-tune 用 5% of pre-train 的 token 数
- RoPE $\theta$: 500K (pre-train 8K), 1M (16K), 2M (32K), 5M (64K), 10M (128K)
- 所有 TTT baseline 用 Llama 3 tokenizer（比 Llama 2 好 0.01 loss Δ）

### Appendix C: Baseline Improvements

1. **Latency**: PyTorch baselines 原本用 FlashAttention-2，升级到 FlashAttention-3，效率大幅提升（GPU 为 Hopper H200）。
2. **QK Norm**: 本文发现 QK normalization 让 TTT-E2E 训练更稳。为公平对比，也给 SWA 家族 baseline 加上。Gated DeltaNet 760M 的 loss 从 2.814 → 2.809 (pre-train), 2.691 → 2.683 (32K fine-tune)。

### Appendix D: Decoding Evaluation 细节

- Softmax temperature = 1
- Top-p = 0.95
- Repetition penalty = 1.1 (HuggingFace 标准实践)
- 评测器 Qwen-3-8B-Base

### Figure 9 (Appendix)

把 Figure 1 左图的 "Loss Δ" 换成绝对 loss 值画出来：
- Full attention 和 Hybrid 的 loss **一直随 context 单调下降**（长 context 越长越好）。
- SWA / Mamba 2 / GDN / TTT-KVB 在 32K 之后 loss **反而上升** —— 因为 fine-tune 时 batch size 是固定的 sequence 数 × context length，context 一长 sequence 数就减少，gradient variance 上升；而这些方法又无法有效利用长 context，variance 的害处 > context 的益处。
- **TTT-E2E** 的 loss 平滑下降至 128K ——没有那个悖论的 uptick。

---

## 🔖 整篇 paper 的一页总结

### 一句话

**不要把长 context 当架构问题 —— 就用标准 SWA Transformer，在推理时继续做 NTP 梯度下降，配上 meta-learning 学一个"适合被 TTT"的初始化。结果：scaling 像 full attention，latency 像 RNN。**

### 三个核心创新

| 创新 | 对应 Section | 一句话 |
|---|---|---|
| **E2E at Test Time** | 2.1, 2.4 | Inner loop 的 loss 直接是最终 NTP loss（vs KVB/Titans 的 layer-wise reconstruction） |
| **E2E at Training Time** | 2.2, 2.3 | Outer loop 用 meta-learning + gradients of gradients 学 $W_0$（vs dynamic evaluation 的 static loss） |
| **Mini-batch + SWA** | 2.3 | mini-batch 让 TTT 并行且稳定，SWA 负责 mini-batch 内部的短期记忆 |

### 三个关键实验

| 实验 | 结果 |
|---|---|
| Context scaling (Figure 1, 6) | **唯一**全 context 段 parallel with full attention 的方法 |
| Compute scaling (Figure 5) | 760M/48B 往上和 full attention 趋势一致 |
| NIAH (Table 2) | **诚实的失败** —— 压缩机制必然丢失 needle |

### 三个你需要记住的数字

1. **2.7×** 快于 full attention @ 128K prefill
2. **1/4** block 的 MLP 被 TTT（性价比最好的存储量）
3. **0.013** 仅换 loss (KVB → NTP) 就带来的 loss 改善 —— 比架构改动重要

### 三个未来问题

1. 训练速度 —— 能否写一个支持二阶梯度的 FlashAttention kernel？
2. Recall 任务 —— 能否在 TTT-E2E 上叠一个小型 retrieval module 弥补 NIAH 的劣势？
3. Production scale —— 能否先用 full attention pre-train，再 fine-tune 成 TTT-E2E？

### 和相关论文的对比（本地 Parametric Memory 目录）

| 论文 | 和 TTT-E2E 的关系 |
|---|---|
| **In-Place TTT** (ICLR 2026) | 都复用 MLP 当 fast weights，但 In-Place 不做 meta-learning outer loop，靠 LM-aligned target 对齐；TTT-E2E 靠二阶梯度的 outer loop |
| **Titans** (arXiv 2501.00663) | TTT-KVB 家族代表，本文 Section 2.4 直接以它为对偶推导起点 |
| **Nested Learning** (NeurIPS 2025) | 同样是"多层 associative memory"视角，但每层独立；TTT-E2E 是"一层超大 RNN" |
| **MemoryLLM / M+** (ICML 24/25) | Parametric memory pool 路线（显式 memory 槽位），和 TTT-E2E 的"权重差压缩"是两条正交路线 |
| **ParamMem** (arXiv 2602.23320) | 跨样本的 LoRA 记忆，和 TTT-E2E 的 in-context 权重更新是正交关系 |

---

*End of batched reading. 欢迎在 section 文件里提问，我会把解答补充到对应位置。* 📚
