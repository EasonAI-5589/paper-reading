[← 返回 README](../README.md)

# 4. Results

## 📌 预览
核心实验结果：(1) Mamba-16k AUROC 0.807 达到 SOTA；(2) copy-forwarding、irregular intervals、disease progression 都负相关模型性能；(3) 长上下文模型更鲁棒。

---

First, we evaluate each of our models on the 14 EHRSHOT clinical prediction tasks. Overall results are shown in Figure 1b, and per-task results in Appendix Figure 9. Our best performing model is Mamba with a context length of 16k tokens. It achieves the highest average AUROC across all tasks, beating the prior state-of-the-art by 0.03 points. Second, we analyze how three EHR-specific properties – event repetition from copy-forwarding, irregularly spaced inter-event times, and disease progression – impact model performance. After stratifying EHRSHOT patients into quartiles by each property, we find that each property negatively correlates with model performance. However, longer context models exhibit more robustness as they perform better across all quartiles.

---

## 4.1 Longer Contexts Improve Prediction Making for Certain Architectures

Our best performing model is Mamba at its maximum context length of 16k tokens, with a mean AUROC of 0.807 (+0.03 points over prior SOTA). This can be seen in Figure 1b. Each line represents a separate model architecture. The y-axis is mean AUROC across the 14 EHRSHOT tasks, and the x-axis is the context length. The dotted purple line is the AUROC (0.777) achieved by the best overall prior model, CLMBR-t-base, which had a context length of 512 tokens (Wornow et al., 2023).

> 💡 **核心结果**:
> - **Mamba-16k**: AUROC 0.807（+0.03 over CLMBR-t-base 的 0.777）
> - **Mamba 趋势**: 随上下文增长持续提升（1k→4k→8k→16k）
> - **Llama 趋势**: 温和提升（512→4k）
> - **GPT 趋势**: 不稳定（可能因 absolute PE 导致）
> - **Hyena 趋势**: 4k 后急剧下降！

Several trends appear in Figure 1b. Both Mamba (green) and Llama (orange) show increased performance at longer context lengths, demonstrating the value of additional EHR data when making clinical predictions. In contrast, Hyena (red) exhibits a sharp decrease in performance after exceeding a context length of 4k. This shows that including more tokens into the context does not always improve performance across architectures. The impact of context length on GPT (blue) appears less clear, which could be due to its usage of absolute positional embeddings (see Section 4.4 for additional analysis). Results on individual tasks are in Appendix Figure 9.

> 💡 **Hyena 为什么崩溃？** 论文没有深入解释，但 Hyena 用的是隐式长卷积。可能的原因：
> 1. EHR 数据的高重复性可能使长卷积过拟合局部模式
> 2. EHR 中的离散事件不像自然语言那样有连续的语义流
> 3. 120M 参数对 16k 长卷积可能不够
> 
> 这是一个重要的 negative result：不是所有亚二次架构都适合长上下文 EHR。

To more explicitly model the passage of time, we also train a version of our models using the Artificial Time Tokens (ATT) technique proposed in CEHR-BERT (Pang et al., 2021). However, as shown in Appendix Figure 12, we see slightly worse performance with this tokenization strategy.

> 💡 **ATT 无效**: 在 token 间插入时间标记（D1, W2, M3 等）反而降低性能。可能原因：ATT 占据了宝贵的上下文窗口位置，减少了实际临床事件的数量。位置编码已经隐式捕获了顺序信息。

---

## 4.2 Copy-Forwarding Creates Noisy Repetition Harming Model Performance

**EHR-OMOP Analysis.** We measure the n-gram repetition rate (RR) across all 0.5M EHR-OMOP validation patients and plot the frequency of each observed RR in Figure 3 in blue. We perform the same calculations on the WikiText-103 dataset and overlay them in orange as "WikiText" as a point of comparison (Merity et al., 2016).

![Figure 3](../images/6640ab1bbc1ca942eef90c1a418b42f2212f35b46abbfb470b5d1b688b5fd1f7.jpg)
*Figure 3: EHR data exhibits a higher degree of repetition than natural language, as measured by n-gram repetition rates. "EHR-OMOP" (blue) vs "WikiText" (orange) for n=1,2,3,4.*

> 💡 **Figure 3 批读**:
> - EHR 的 3-gram 和 4-gram 重复率远高于 WikiText
> - 很多患者的 1-gram RR > 80%，意味着超过 80% 的 token 都重复出现过
> - WikiText 的 4-gram 重复率几乎为 0，但 EHR 中相当普遍
> - 这说明 copy-forwarding 确实是 EHR 数据的突出特征

**EHRSHOT Stratification.** Next, we evaluated how the repetitiveness of a patient's timeline affects model performance on the EHRSHOT benchmark using Brier score. Using 1-gram repetition rate as the metric, patients were grouped into quartiles from Q1 (lowest) to Q4 (highest).

We repeated this analysis with the EHR FMs trained in this work (Table 2, top). Model performance consistently degrades as repetition increases, indicating that highly repetitive sequences are more challenging to model. Notably, longer context versions of Mamba and Llama achieve significantly lower Brier scores across all quartiles compared to their shorter counterparts.

> 💡 **Table 2 (Repetitiveness) 关键发现**:
> - Mamba 1k: Q1=0.0644, Q4=0.0790 (差距 23%)
> - Mamba 16k: Q1=0.0605, Q4=0.0746 (差距 23%, 但绝对值全面更低)
> - 长上下文不能消除重复性的负面影响，但能在所有分位都降低 Brier score

---

## 4.3 Irregular Inter-Token Time Intervals Are Harder to Model

**EHR-OMOP Analysis.** We first quantify the degree to which EHR data exhibits irregularity in the intervals of time between consecutive events.

![Figure 2](../images/1c9fbde8e863f8da62da718817ddcd32519353b0a0e9d2596f0294efd82b2eab.jpg)
*Figure 2: EHR data exhibits a high degree of variation in time intervals between events. Mean, standard deviation, and IQR of inter-event times (log scale).*

> 💡 **Figure 2 批读**:
> - X 轴是对数尺度，范围从 10¹ 秒（~10秒）到 10⁹ 秒（~30年）
> - 大多数患者的 inter-event 标准差在 10⁷-10⁸ 秒（115 天到 3.2 年）
> - 这意味着同一患者的就诊间隔从几分钟到几年不等——极端不规则

**EHRSHOT Stratification.** Table 2 extends this analysis to the EHR FMs trained in this work. While model performance still degrades with increased irregularity, longer context versions of Mamba and Llama consistently outperform their shorter counterparts across all quartiles.

> 💡 **Table 2 (Irregularity) 关键发现**:
> - Mamba 16k 在所有分位都显著优于 Mamba 1k
> - CLMBR-t-base (512) 的 Q4 Brier score (0.0777) 远高于 Mamba 16k 的 Q4 (0.0723)
> - **最不规则 vs 最规则患者**：Brier 差异约 10-14%

---

## 4.4 Disease Progression Effects Are Better Modeled with Longer Contexts

**EHR-OMOP Analysis.** Figure 4 shows that tokens later in a patient's timeline are more difficult to predict (higher perplexity), even when conditioning on all prior tokens. This contrasts with natural language, where later tokens tend to have lower perplexity (Kaplan et al., 2020; Peng et al., 2023b). We hypothesize this is because diseases naturally become more complex and varied with aging.

![Figure 4](../images/03d04c55f3323585d79134808dce1dda78dca1f65a251ef61a1065c4fe1bd3b8.jpg)
*Figure 4: Median perplexity (PPL) by token position for GPT, Hyena, Llama, Mamba across varying context lengths. The upward trend in PPL is almost immediate, even within the first hundred tokens.*

> 💡 **Figure 4 批读 — 这是论文最有洞察力的结果之一**:
> - **所有模型、所有上下文长度的 perplexity 都随 token 位置上升**
> - 这与 NLP 的规律完全相反！在 NLP 中，给模型更多上下文，后面的 token 更容易预测
> - 在 EHR 中，即使给了完整的病史，未来的诊断仍然越来越难预测——因为疾病在进展
> - **Mamba 和 Llama 的长上下文版本在所有位置都有更低的 perplexity，且差距在后期更大**
> - **GPT 出现 10+ 点的 perplexity 尖刺**，这是 absolute PE 导致的——换成 RoPE 后消失（见 Appendix Figure 11）
> 
> 对 agent memory 的启示：agent 的"世界"也在变化。早期记忆对预测后期行为的帮助可能递减——这挑战了"越多记忆越好"的简单假设。

Longer context versions of Mamba and Llama consistently achieve lower perplexities across all token positions compared to shorter contexts, with the gap widening at later tokens. This suggests that a more complete view of the patient's timeline helps handle increasing token complexity due to aging. In contrast, Hyena's longer context models perform worse, replicating our original EHRSHOT results. For GPT, results are mixed: longer contexts (2k and 4k) achieve lower perplexities at later tokens but exhibit significant spikes. This appears to be caused by GPT's usage of absolute positional embeddings – replacing them with rotary positional embeddings (ROPE) (Su et al., 2024) mitigated these spikes as seen in Appendix Figure 11. Thus, despite its popularity in the EHR FM community (see Table 1), we recommend discontinuing the GPT architecture in favor of Llama or other more modern decoder-only architectures.

> 💡 **GPT vs Llama 的启示**: GPT 在 EHR 社区最流行（Table 1 中最多），但本文建议弃用 GPT 改用 Llama。核心原因：absolute PE 在长上下文时引起 perplexity 尖刺，RoPE 能解决。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| Mamba-16k 平均 AUROC | 0.807 |
| CLMBR-t-base AUROC | 0.777 |
| 提升 | +0.03 |
| SOTA 任务数 | 9/14 |
| 最不规则 vs 最规则 Brier 差异 | ~14% |
| Hyena 4k→16k | 急剧下降 |

### 核心洞察
1. **Mamba 是 EHR 长上下文的最佳选择**：持续受益于更长上下文
2. **Hyena 不适合 EHR**：4k 后崩溃，原因不明但结果清晰
3. **GPT 应被 Llama 取代**：absolute PE 在长上下文时有严重问题
4. **EHR 的 perplexity 随位置上升**：与 NLP 相反，更多历史不能让预测更容易
5. **ATT（显式时间编码）无效**：反而略微降低性能
