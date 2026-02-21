[← 返回 README](../README.md)

# 5 Analysis and Discussion

## 📌 预览
深入分析 DART 的效率、pivot token 选择、pruning layer 选择、pivot 数量、模态影响。核心发现：pivot 选择不敏感（random 也行）；不同 pivot 策略保留的 token 集重叠 <50% 但性能相似；深层剪枝甚至可以减少 hallucination。

---

## 5.1 Efficiency Analysis

As shown in Table 2, we compare the total inference time, prefill time, FLOPs, and KV cache memory of multiple methods. (i) DART achieves a 2.99× speedup in prefill and 1.99× speedup in inference, while its performance on POPE degrades by less than 3% versus the vanilla model. (ii) Analysis reveals although FLOPs reduction is similar across methods, their speeds vary significantly. For instance, SparseVLM increases FLOPs by 2.8% versus DART, but its speedup drops 21.6%, showing FLOPs alone poorly measure acceleration. (iii) We evaluate performance-latency trade-offs using actual latency. Figure 4 shows some methods underperform random token retention. SparseVLM and MustDrop suffer speed degradation from sequential token processing. FastV's biased attention scores yield worse performance. In contrast, DART integrates Flash Attention with under 0.08s overhead, achieving better performance-speed balance.

> 💡 **关键 takeaway**: FLOPs ≠ 实际速度。SparseVLM 的 FLOPs 只比 DART 多 2.8%，但 wall-clock time 慢 21.6%。原因：(1) 不兼容 FA，(2) sequential token processing overhead。Figure 4 的 performance-latency Pareto 图比纯看 benchmark 数字更有说服力。

---

## 5.2 Influence from Selection of Pivot Tokens

In this section, we investigate whether pivot token selection in DART significantly affects its performance. Table 8 in Appendix A.1 evaluates pivot tokens based on criteria such as maximum (♠), minimum (♡) attention scores, K-norm, V-norm, and random selection. Results show that various strategies achieve over 94.9% of the vanilla model's performance across benchmarks. Even DART with randomly selected pivot tokens incurs only a 1.2% performance drop compared to the best strategy and outperforms the previous importance-based methods by 2.1%. This observation shows the robustness in the selection of pivot tokens in DART, and highlights the crucial role of duplication in token reduction, as selecting "important" pivot tokens based on attention scores is only 0.2% better than selecting "unimportant" ones as pivot tokens.

> 💡 **Pivot 选择鲁棒性**:
> | 策略 | Avg. |
> |------|------|
> | V-norm♠ (最佳) | 97.2% |
> | K-norm♠ | 96.8% |
> | Random | 96.0% |
> | V-norm♡ (最差) | 94.9% |
> | **SparseVLM (对比)** | **93.9%** |
> | **FastV (对比)** | **81.5%** |
>
> **最差的 DART 配置仍然优于所有 importance-based 方法**。"Important" vs "unimportant" pivot 差异仅 0.2%，说明 duplication removal 本身才是关键。

---

Furthermore, on the MME benchmark, we analyze the visual tokens retained by selecting pivot tokens based on K-norm♠ and K-norm♡. Interestingly, statistical analysis shows that the overlap between tokens preserved by these two strategies is, on average, less than 50%. Despite this low overlap, both strategies achieve highly effective results, indicating the existence of multiple distinct groups of tokens which should not be pruned. This finding challenges the conventional notion of a single critical token set defined by importance scores, demonstrating that diverse token subsets with minimal overlap can yield comparable performance.

> 💡 **重叠率 <50% 但性能相似**: 这是非常深刻的发现。它说明不存在一个唯一的 "关键 token 集"——有多组不同的 token 子集都能支撑模型性能。这与 importance-based 方法 "找到唯一最重要 token 集" 的 narrative 根本矛盾。从信息论角度：token space 中有多个近似最优的覆盖集（covering set），它们互不相同但都能近似表示完整信息。

---

## 5.3 Influence from Choice of the Pruned Layer and the Number of Pivot Tokens

We explore the impact of layer on model performance. As expected, pruning deeper layers yields performance closer to the vanilla model but increases latency, as shown in Figure 6. However, we observe two intriguing findings: (i) Pruning at layers 10, 15, and 20 surprisingly outperforms the vanilla model (Fig. 6(a)), consistent with Fig. 1, suggesting that removing duplicate tokens may reduce hallucinations in MLLMs on the POPE. (ii) At deeper layers (e.g., 15, 20), the latency-minimizing points correspond to pruning all vision tokens, yet performance drops only by 0.1%∼1.6%. This highlights a modality imbalance in MLLMs, indicating underutilization of the visual modality.

> 💡 **两个 surprising findings**:
> (1) **深层剪枝 > vanilla**: 在 POPE 上，layer 10/15/20 处剪枝后性能比不剪还好！这暗示冗余 vision tokens 在深层可能反而干扰模型，产生 hallucination。
> (2) **深层删所有 vision tokens 也只降 0.1-1.6%**: 这表明 LLM 在浅层就已经把关键视觉信息编码到了 text/system tokens 中（类似 FastV 发现的 "anchor token" 现象）。Vision tokens 在深层几乎是 "dead weight"。

---

Furthermore, we delved into the impact of the number of pivot tokens on performance. As depicted in Figure 5, choosing either an insufficient or an excessive number of pivot tokens leads to suboptimal outcomes. When a limited number of pivot tokens (e.g., one or two), the lack of diversity among these tokens may impede their ability to comprehensively represent the entire feature space. In contrast, when an overly large number of pivot tokens, for example, 20 or more, are chosen, the majority of retained visual tokens tend to be pivot tokens. In extreme cases, our approach starts to resemble the importance-based method, where pivot tokens essentially transform into important tokens, overlooking the impact of duplication factors.

> 💡 **Pivot 数量的 sweet spot**: 太少（1-2）→ diversity 不足，不能覆盖整个特征空间；太多（20+）→ 保留的 token 大多是 pivot 本身，退化为 importance-based 方法。最佳值约 8 个（4 visual + 4 text）。这是一个 bias-variance tradeoff 的具象化。

---

## 5.4 Influence from Modalities of Pivot Tokens

We further analyze the impact of the source of pivot tokens on the overall performance of DART, with a particular focus on understanding whether guidance from the language modality is essential for effective token reduction. We evaluate the performance implications of selecting pivot tokens exclusively from either the visual or text modality, aiming to quantify the influence of each modality. As illustrated in Figure 7, the absence of pivot tokens from either modality leads to a noticeable decline in performance. This demonstrates that information from both modalities contributes to the token reduction process to varying degrees. Moreover, it highlights that we provide an effective method for incorporating textual guidance without the need to explicitly compute cross-modal attention scores while remaining compatible with Flash Attention.

> 💡 **双模态 pivot 更好**: 全部 tokens > 仅 visual > 仅 text。这说明 cross-modal 信息对于判断 "哪些 vision tokens 冗余" 是有帮助的。巧妙的是，DART 通过在 pivot 中同时包含 visual 和 text tokens，隐式地引入了 cross-modal guidance，而无需显式计算 cross-attention。这是 "不需要 attention scores 也能获得 cross-modal 信号" 的优雅解法。

---
