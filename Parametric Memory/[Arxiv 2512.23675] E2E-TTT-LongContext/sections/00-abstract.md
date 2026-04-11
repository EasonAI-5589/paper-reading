[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要是整篇论文的一页版：把长上下文建模**重新定义**为 continual learning 问题，用**标准 Transformer + SWA** 作骨架，在测试时继续做 NTP 训练把 context 压进权重，训练时用 meta-learning 学一个更好的初始化。最终结果：**和 full attention 一样的 scaling + 常数推理延迟**。

---

We formulate long-context language modeling as a problem in continual learning rather than architecture design. Under this formulation, we only use a standard architecture – a Transformer with sliding-window attention. However, our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights. In addition, we improve the model's initialization for learning at test time via meta-learning at training time. Overall, our method, a form of Test-Time Training (TTT), is End-to-End (E2E) both at test time (via next-token prediction) and training time (via meta-learning), in contrast to previous forms. We conduct extensive experiments with a focus on scaling properties. In particular, for 3B models trained with 164B tokens, our method (TTT-E2E) scales with context length in the same way as Transformer with full attention, while others, such as Mamba 2 and Gated DeltaNet, do not. However, similar to RNNs, TTT-E2E has constant inference latency regardless of context length, making it 2.7× faster than full attention for 128K context. Our code is publicly available.

> 💡 **摘要的三个"反直觉"点**:
>
> 1. **不改架构**：前面几年的长 context 工作（Mamba 2、Gated DeltaNet、TTT-KVB、Titans、MesaNet、Nested Learning）都在"设计新的 sequence modeling layer 来替换 self-attention"。本文说："不用换，就用最普通的 SWA Transformer，改训练/推理流程就行。"
>
> 2. **测试时继续训练**：大多数语言模型部署后是**冻结**的 —— 参数就是推理时的参数。本文在测试时每个 mini-batch 都对模型的 MLP 做一次梯度下降，所以模型权重在推理过程中是**在变的**。context 不再存在 KV cache 里，而是**"压"进了 MLP 权重**。
>
> 3. **双重 End-to-End**：既要 test time E2E（inner loop 的损失直接是最终 next-token prediction loss，不是 layer-wise 的 reconstruction），也要 train time E2E（outer loop 直接优化"TTT 之后"的 loss，通过 gradients of gradients）。TTT-KVB 只满足后者，dynamic evaluation 只满足前者，本文两者都要。

> 💡 **为什么 2.7× 加速？**
> 128K context 下，full attention 要对所有 previous tokens 做 $O(T^2)$ 的 prefill。TTT-E2E 的 SWA 骨架只看 8K 窗口，所以 prefill 是 $O(T)$；又因为 window 固定，**每 token 的 latency 是常数**。2.7× 是在 H100 上实测的 prefill 加速比。

---

## 🔖 摘要一图流

```
长 context 建模
      │
      ├─ 传统路线 (架构问题)
      │    └─ Mamba 2 / Gated DeltaNet / TTT-KVB / Titans ...
      │       设计新 layer → 替换 self-attention
      │       问题：长 context 下 scaling 趋势不如 full attention
      │
      └─ 本文路线 (continual learning 问题)  ⬅️
           └─ 标准 SWA Transformer + 测试时 NTP 梯度下降
              +  训练时 meta-learning (gradients of gradients)
              ─────────────────────────────────────────
              测试时 E2E + 训练时 E2E  =  TTT-E2E
              ─────────────────────────────────────────
              结果：scaling 像 full attention, latency 像 RNN
```

---

*Figure 1 按"Abstract 不放图"规则放到 [01-introduction.md](01-introduction.md)。*
