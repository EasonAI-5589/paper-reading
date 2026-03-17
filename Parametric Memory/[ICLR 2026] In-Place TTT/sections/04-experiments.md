[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
三个核心研究问题：Q1 作为 drop-in 增强预训练 LLM 的效果；Q2 从头训练与已有 TTT 方案的对比；Q3 关键设计选择的消融分析。同时包含 Section 5 结论。

---

## 4.1 In-Place TTT as a Drop-in Enhancement for Pre-trained LLMs

Base model: Qwen3-4B-Base (original context window 32k). Two-stage continual training: ~20B tokens at 32k context + ~15B tokens at 128k context.

**Table 1: RULER benchmark results (average accuracy %)**

| Model | 4k | 8k | 16k | 32k | 64k | 128k | 256k (extrapolation) |
|---|---|---|---|---|---|---|---|
| Mistral-7B | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 | - |
| GLM3-6B | 87.8 | 83.4 | 78.6 | 69.9 | 56.0 | 42.0 | - |
| Phi3-medium-14B | 93.3 | 93.2 | 91.1 | 86.8 | 78.6 | 46.1 | - |
| Llama3-8B | 92.8 | 90.3 | 85.7 | 79.9 | 76.3 | 69.5 | - |
| Qwen3-4B (Instruct) | 95.1 | 93.6 | 91.0 | 87.8 | 77.8 | 66.0 | - |
| Baseline (Qwen3-4B-Base) | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| **In-Place TTT** | **96.1** | **95.6** | **92.7** | **89.3** | **78.7** | **77.0** | **43.9** |

> 💡 **Table 1 批读**:
> - **长上下文提升显著**: 64k 从 74.3→78.7 (+4.4), 128k 从 74.8→77.0 (+2.2), 256k 外推从 41.7→43.9
> - **短上下文无损**: 4k/8k/16k/32k 几乎持平甚至略有提升（8k 从 94.1→95.6），说明 In-Place TTT 不以牺牲短上下文为代价
> - **对比更大模型**: 4B 的 In-Place TTT 在 64k-128k 超越 Phi3-medium-14B (78.7 vs 78.6, 77.0 vs 46.1)，参数量不到 1/3
> - **核心意义**: 这是 drop-in 增强——不改变模型架构，只需少量 continual training 即可获得长上下文能力提升

---

**Table 2: Extension to LLaMA-3.1-8B and Qwen3-14B-Base**

| Base Model | Method | 4k | 8k | 16k | 32k | 64k | 64k+YaRN |
|---|---|---|---|---|---|---|---|
| LLaMA-3.1-8B | Baseline | 93.9 | 92.1 | 92.5 | 91.1 | 81.6 | – |
| LLaMA-3.1-8B | In-Place TTT | 94.4 | 93.0 | 93.3 | 91.7 | 83.7 | – |
| Qwen3-14B | Baseline | 96.8 | 95.0 | 94.6 | 90.7 | 67.9 | 81.3 |
| Qwen3-14B | In-Place TTT | 97.2 | 95.7 | 95.2 | 91.2 | 70.6 | 82.5 |

> 💡 **Table 2 批读**:
> - **跨模型泛化**: 从 4B 到 8B 到 14B 都有一致提升，说明 In-Place TTT 不是针对特定模型的 trick
> - **LLaMA-3.1-8B**: 64k 提升 +2.1 (81.6→83.7)，LLaMA 本身已是强基线（原生 128k context）
> - **Qwen3-14B**: 64k 提升 +2.7 (67.9→70.6)；搭配 YaRN 也有提升 (81.3→82.5)，说明与位置编码外推方法正交互补
> - **规模越大效果是否更好？** 从绝对值看 14B 的 64k 提升 (+2.7) > 8B (+2.1) > 4B 的 continual training 提升，暗示 In-Place TTT 可能在更大模型上更有价值

---

## 4.2 Pre-training from Scratch: A Comparative Analysis

Baselines at 500M and 1.5B scales: SWA (Sliding Window Attention), GLA (Gated Linear Attention), DeltaNet, LaCT (Large Chunk TTT).

**Figure 2**: Sliding Window Perplexity on Pile dataset. In-Place TTT consistently achieves lower perplexity than all baselines at both 500M and 1.5B scales, from 2k to 32k context.

> 💡 **Figure 2 批读**:
> - **击败所有 SSM/线性注意力方案**: GLA、DeltaNet 这些方案在 language modeling 上被 In-Place TTT 超越，说明 TTT 的梯度更新范式比固定规则更新更强
> - **击败 LaCT**: LaCT 是之前的 TTT 方案（需要额外参数块），In-Place TTT 用更少参数实现更好效果
> - **两个规模一致**: 500M 和 1.5B 都保持优势，说明方法的 scalability

---

**Table 3: 4B model results on common sense reasoning + long-context**

| Architecture | HellaSwag | ARC-E | ARC-C | MMLU | PIQA | RULER-4k | RULER-8k | RULER-16k |
|---|---|---|---|---|---|---|---|---|
| Full Attn. (Baseline) | 55.67 | 64.52 | 33.19 | 36.43 | 72.63 | 45.77 | 38.09 | 6.58 |
| SWA (Baseline) | 54.92 | 64.18 | 32.85 | 36.06 | 72.58 | 14.77 | 9.91 | 5.07 |
| Full Attn. + I.P. TTT | **55.85** | **64.98** | 32.34 | **37.42** | **73.29** | **49.98** | **43.82** | **19.99** |
| SWA + I.P. TTT | 55.24 | 64.60 | **33.70** | 36.48 | 72.03 | 28.33 | 26.80 | 7.57 |

> 💡 **Table 3 批读**:
> - **常识推理持平或微升**: HellaSwag/ARC-E/MMLU/PIQA 加 TTT 后基本持平或略有提升，说明 TTT 不影响基础能力
> - **长上下文大幅提升**: Full Attn. 的 RULER-16k 从 6.58→19.99 (3x 提升), RULER-8k 从 38.09→43.82
> - **SWA 同样受益**: SWA 的 RULER-8k 从 9.91→26.80 (2.7x)，说明即使是滑动窗口注意力也能通过 TTT 补偿全局信息
> - **但绝对值仍低**: RULER-16k 的 19.99 仍然不高，说明从头训练 4B 模型的长上下文能力本身就有限；drop-in 增强预训练模型（Section 4.1）效果更好

---

## 4.3 Ablation Studies

**State Size**: Larger fast weights → better performance (scaling with number of TTT-enabled layers).

> 💡 **状态大小**: 增加 TTT-enabled layers 的数量 = 增加 fast weights 的总容量。这与 Titans 的 deep memory 结论一致——更多可训练参数 = 更大的在线记忆容量。

**Chunk Size**: C=512 and C=1024 are optimal; too small (256) or too large (2048) degrades performance.

> 💡 **Chunk Size 的 sweet spot**:
> - C 太小 (256): 每个 chunk 内信息太少，梯度噪声大，更新质量差
> - C 太大 (2048): 更新频率太低，无法及时适应上下文变化；且单步更新需要压缩太多信息
> - C=512~1024 是最优区间，平衡了更新质量和更新频率

**LM-Aligned Objective**: Both Conv1D and W_target projection are crucial. Conv1D is essential for long context, W_target for short context.

> 💡 **LM-Aligned Objective 组件分析**:
> - **Conv1D 对长上下文必要**: 没有 Conv1D 的局部特征提取，TTT 在长序列上退化——因为缺少局部结构信息作为学习信号
> - **W_target 对短上下文必要**: W_target 将自回归重构目标对齐到 LM 目标空间；缺少它则 TTT 的学习方向与 LM 不一致
> - **两者缺一不可**: 这解释了为什么之前的 TTT 方法效果有限——它们的 objective 与 LM 任务不对齐

**Efficiency**: In-Place TTT introduces negligible overhead in both throughput and memory at 8k-128k contexts.

> 💡 **效率**: 几乎零额外开销——这是 in-place 设计的核心优势。因为复用已有 MLP 参数而非新增参数块，所以模型大小不变，推理成本不变，只在 forward pass 中增加了 chunk-wise 的参数更新步骤。

---

## 5. Conclusion

We introduced In-Place Test-Time Training, a practical framework that resolves the critical barriers of TTT for LLMs. Principled design choices are proposed including an in-place mechanism that repurposes existing MLP blocks, an efficient chunk-wise update rule, and a theoretically-grounded objective aligned with language modeling. Extensive experiments validate that our approach not only serves as a powerful "drop-in" enhancement for pre-trained LLMs but also outperforms strong baselines when trained from scratch. By providing a scalable solution for on-the-fly adaptation, our work makes a promising step towards a new paradigm of more dynamic, continual learning for LLMs.

> 💡 **结论批注**: 三个贡献总结——(1) in-place 复用 MLP，(2) chunk-wise 更新规则，(3) LM-aligned objective。最重要的实践意义是 drop-in enhancement：无需改架构、无需从头训练，对已有 LLM 做少量 continual training 即可提升长上下文能力。

---

## 🔖 Section 总结

### 关键数字速查
| 实验 | In-Place TTT | Baseline | 提升 |
|------|-------------|----------|------|
| RULER 64k (Qwen3-4B) | 78.7 | 74.3 | +4.4 |
| RULER 128k (Qwen3-4B) | 77.0 | 74.8 | +2.2 |
| RULER 64k (LLaMA-3.1-8B) | 83.7 | 81.6 | +2.1 |
| RULER 64k (Qwen3-14B) | 70.6 | 67.9 | +2.7 |
| RULER-16k from scratch (Full Attn.) | 19.99 | 6.58 | +13.41 |
| Perplexity (500M/1.5B) | Best | 超越 GLA/DeltaNet/LaCT | - |

### 核心洞察
1. **Drop-in 增强是最大卖点**: 对已有预训练 LLM 做 continual training 即可获得长上下文提升，且短上下文不损失——这是最直接的应用价值
2. **跨模型泛化**: 4B/8B/14B 都有效，与 YaRN 等位置编码方法正交互补
3. **From-scratch 也强**: 击败 GLA/DeltaNet 等 SSM 方案，证明 TTT 范式在 language modeling 上的竞争力
4. **Chunk size = 512~1024 是 sweet spot**: 平衡更新质量与更新频率
5. **LM-aligned objective 是关键创新**: Conv1D + W_target 两个组件缺一不可，解决了 TTT 目标与 LM 目标不对齐的问题
6. **零额外开销**: in-place 复用已有参数，不增加模型大小和推理成本
