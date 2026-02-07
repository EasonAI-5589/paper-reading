# Appendix

> 来源: SparseVLM (ICML 2025)

---

> 💡 **Appendix 概览**: 包含 A-H 共 8 个附录，涵盖视觉冗余分析、FlashAttention 兼容、计算详细估算、数据集细节、实现细节、效率分析、更多可视化。这里摘录关键内容。

---

## A. The Redundancy of Visual Tokens in VLMs

![Figure 7](../images/566b598be134004b841ef16b8e925383895e5ec415121cc44489fb634275baf7.jpg)
*Figure 7: 不同视觉任务中的视觉冗余分析。下采样从 1166 到 576 tokens 提升 50% 效率但损失 15% 信息。*

> 💡 **Figure 7 批读**: 对比分类/检测任务（下采样可接受）和 VQA 任务（需要精准保留相关区域）。图片信息通常比文本更稀疏——图片中 38% 区域与问题相关，而文本中 88% 是有用的。这就是 SparseVLM 的理论基础。

---

## B. Compatibility with FlashAttention

> 💡 **批注**: 这是工程上最巧妙的部分。由于 FlashAttention 不显式存储注意力矩阵，SparseVLM 设计了 **dual-flash attention**:
> ```
> 第一次 Forward: 正常 FlashAttention → 得到 hidden states
> 第二次 Forward: 用特殊 V 矩阵（rater 行设为 1/n，其余为 0）
>   → 内积直接返回 rater 的平均注意力分数
>   → 然后 top-k 选择保留的视觉 token
>   → 生成 mask 应用到第一次的 hidden states 上
> ```
> 代价是多一次 FlashAttention forward，但矩阵是特殊稀疏的，实际开销小。

---

## G. More Detailed Efficiency Analysis

![Figure 8](../images/8cf306ac487857096badb3c7d2f1f64e854689e71a85c24c7fbb3d9f257b9eb1.jpg)
*Figure 8: LLaVA 上的 Latency vs. Accuracy 和 FLOPs vs. Accuracy 权衡曲线。*

![Figure 9](../images/9c227260cf2740e46a01b5a9bab17d1a0181ab7b61aa113e29e36738b6a6648c.jpg)
*Figure 9: MiniGemini 上的权衡曲线。*

![Figure 10](../images/eaa611875347dc062be2dd11cb4c05858d6a3bdd9f4ad4e28fbf3af1e1ea04e8.jpg)
*Figure 10: VideoLLaVA 上的权衡曲线。*

> 💡 **批读**: 三个模型的权衡曲线都显示 SparseVLM（蓝线）始终在 Pareto 前沿，即在相同 latency/FLOPs 下精度最高，或相同精度下效率最高。Random sparse（灰线）则明显最差，说明"聪明地剪"比"随机剪"重要得多。

---

## H. More Sparsification Visualization

![Figure 11](../images/3f25fbb551e071367090f79df8a324dc02687ae3a632897e39dc5215b0f43058.jpg)
*Figure 11: 更多 SparseVLM 在不同 prompt 上的可视化示例。*

> 💡 **Figure 11 批读**: 展示了多种场景（室内/室外、物体/文字、单目标/多目标）下 SparseVLM 的剪枝效果。每种情况都能准确保留与问题相关的视觉区域，验证了 text-aware 策略的通用性。
