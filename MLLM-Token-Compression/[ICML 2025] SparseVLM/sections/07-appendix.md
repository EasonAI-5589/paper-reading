[← 返回 README](../README.md)

# Appendix

## 📌 预览
附录包含：视觉 token 冗余分析、FlashAttention 兼容方案、计算预算详细估算、数据集描述、实现细节、效率分析、可视化案例。

---

## A. The Redundancy of Visual Tokens in VLMs

In non-textual tasks, such as classification or detection, downsampling is commonly used to reduce visual redundancy and enhance model training efficiency (Zhang et al., 2024b). Figure 7 illustrates this process, showing the reduction of tokens from 1166 to 576 in a downsampled image, resulting in a 50% efficiency boost but a 15% information loss (entropy decreased from 7.44 to 6.13). This trade-off is acceptable for such tasks. Conversely, for text-related tasks like visual question answering (VQA), which involve both text and vision modalities, a distinct approach is required. Highlighting the most information-dense text (88% of total text) alongside the region pertinent to the query in the image (38% of total image), we observe that image information is typically sparser than textual data. Hence, our SparseVLM method incrementally prunes visual token redundancy, maintaining crucial information for task accuracy. This strategy enhances model efficiency.

![Figure 7](../images/566b598be134004b841ef16b8e925383895e5ec415121cc44489fb634275baf7.jpg)
*Figure 7. Analysis of visual redundancy in different vision tasks.*

> 💡 **Figure 7 批读**: 分类/检测任务直接下采样可以接受（15% 信息损失）。但 VQA 任务中，88% 的文本信息密集，只有 38% 的图像区域与问题相关 → 需要智能裁剪而非盲目下采样。

---

## B. Compatibility with FlashAttention

To ensure compatibility between SparseVLM and FlashAttention (Dao et al., 2022) when extracting the matrix $A$ or $P$, we devise the dual-flash attention operation to directly obtain the average attention scores relative to the text raters. This operation is lightweight and enjoys the efficiency of FlashAttention.

Specifically, the first forward pass operates identically to the original FlashAttention, generating the necessary hidden states. In the second forward pass, we introduce a specially designed $V$ matrix. In this matrix, for the rows corresponding to the text raters we wish to analyze, we set the values to the reciprocal of the number of text raters. This configuration allows the inner product between the attention map and the $V$ matrix to return the mean value of the attention scores for the selected text raters directly in FlashAttention. With the mean value, we perform a top-$k$ selection to identify the visual tokens to retain. Tokens that are excluded during this process are converted into masks, which are then applied to the hidden states produced by the first FlashAttention pass to complete the pruning operation.

> 💡 **Dual-Flash Attention 要点**:
> 1. **第一次 forward**: 正常 FlashAttention，生成 hidden states
> 2. **第二次 forward**: 用特殊 $V$ 矩阵（rater 行设为 $1/n$，其余为 0）
>    - 内积结果直接就是 rater 平均注意力分数
> 3. Top-$k$ 选择保留的视觉 token → 生成 mask → 应用到第一次的 hidden states
>
> 这样不需要显式存储完整注意力矩阵，保持 FlashAttention 的内存效率。

---

## C. Computing Budget Detailed Estimation

**Estimation of Visual Token Significance.** Each visual token undergoes $L_t - 1$ additions and one division. With $L_v$ visual tokens in total, the number of FLOPs for this stage is $(L_t - 1 + 1) \times L_v = L_t \times L_v$.

**Relevant Text Selection.** The FLOPs for equation 6 can be approximately simplified to the matrix multiplication between $H_v$ and $H_q$. The result has a shape of $L_v \times L_t$, where each element undergoes $D$ multiplications and additions. Therefore, the FLOP count can be expressed as $L_t \times L_v \times 2D$.

**Sparsification Level Adaptation.** The rank of a matrix is typically computed using singular value decomposition (SVD). The FLOPs involved can be approximated as $L_t \times L_v \times \min(L_t, L_v)$.

**Token Aggregation.** Total FLOPs: $L_r \times (3L_r - 1) \times 2D + L_r$.

**Token Reconstruction.** FLOPs: $D \times (L_r - C)$.

> 💡 **计算量汇总**: 所有额外开销（rater 选择、rank 计算、聚类重构）都是轻量级的，远小于裁剪带来的计算节省。

---

## D. Dataset

> 💡 详细的数据集描述见原文。覆盖 GQA、MMBench、MME、POPE、ScienceQA、VQA-v2、TextVQA、MMVet、TGIF-QA、MSVD-QA、MSRVTT-QA、ActivityNet-QA。

---

## E. Implementation Details

All experiments are conducted on a single NVIDIA A100-80G GPU. The implementation is carried out in Python 3.10, utilizing PyTorch 2.1.2, CUDA 11.8, and transformers 4.31.0. The inference follows the evaluation settings established by LLaVA (Liu et al., 2024b).

---

## G. More Detailed Efficiency Analysis

![Figure 8](../images/8cf306ac487857096badb3c7d2f1f64e854689e71a85c24c7fbb3d9f257b9eb1.jpg)
*Figure 8. Trade-offs for SparseVLM on LLaVA: (a) Latency vs. Accuracy, and (b) FLOPs vs. Accuracy.*

> 💡 **Figure 8 批读**: LLaVA 上的 latency-accuracy 和 FLOPs-accuracy 权衡曲线。SparseVLM 始终在 Pareto 前沿上方，优于 random sparse。

---

![Figure 9](../images/9c227260cf2740e46a01b5a9bab17d1a0181ab7b61aa113e29e36738b6a6648c.jpg)
*Figure 9. Trade-offs for SparseVLM on MGM: (a) Latency vs. Accuracy, and (b) FLOPs vs. Accuracy.*

> 💡 **Figure 9 批读**: MGM 上的趋势与 LLaVA 一致，SparseVLM 在各种效率水平下都保持更高精度。

---

![Figure 10](../images/eaa611875347dc062be2dd11cb4c05858d6a3bdd9f4ad4e28fbf3af1e1ea04e8.jpg)
*Figure 10. Trade-offs for SparseVLM on Video-LLaVA: (a) Latency vs. Accuracy, and (b) Token budget vs. Accuracy.*

> 💡 **Figure 10 批读**: Video-LLaVA 上同样表现优异，token budget 减少时精度下降平缓。

---

## H. More Sparsification Visualization

Figure 11 showcases a diverse array of visualization examples that demonstrate the application of SparseVLM across a spectrum of visual question-answering (VQA) prompts. These visualizations offer a deeper insight into how our SparseVLM processes and responds to different types of queries posed in a visual context.

![Figure 11](../images/3f25fbb551e071367090f79df8a324dc02687ae3a632897e39dc5215b0f43058.jpg)
*Figure 11. More visualization examples of SparseVLM on different prompts. Best viewed in color.*

> 💡 **Figure 11 批读**: 更多 VQA 可视化案例，进一步验证 SparseVLM 根据不同问题聚焦不同区域的能力。
