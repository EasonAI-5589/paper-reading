[← 返回 README](../README.md)

# 5. Conclusion

In this paper, we identify limitations in existing token pruning methods for visual grounding tasks and perform a systematic analysis of VLMs' multi-stage visual processing pipelines. Results reveal task-specific demands, where grounding relies on global spatial reference frames disrupted by pruning. To mitigate this, we propose Nüwa, a two-stage framework with Boids-inspired aggregation and text-guided refinement to preserve spatial integrity. Extensive experiments across 13 datasets and multiple VLMs show state-of-the-art performance on VQA (95% retention) and VG (47.2% retention), with 89% TFLOPs and 62% prefill reductions via 88.9% token pruning.

> 💡 **总体评价**:
>
> **优点**:
> 1. 分析部分（Sec 2）非常扎实，RPME 实验直接证明了 spatial integrity 是 VG 退化根因
> 2. VG 上的大幅提升（7% → 47%）确实解决了一个被忽视的关键问题
> 3. 方法设计原则清晰：空间均匀性 + 局部聚合 + text-guided 精炼
> 4. 效率开销极小（+1ms prefill），工程友好
>
> **不足**:
> 1. VG 绝对性能仍然只有 vanilla 的一半——47% retention 在实际应用中可能不够
> 2. Stage 2 创新有限，本质是 FastV 思路的变体
> 3. 只在 LLaVA-1.5/NeXT 上验证，缺少 Qwen2-VL / InternVL 等更新模型的实验
> 4. Boids 类比虽有趣但有些牵强——实际操作是静态的 grid partition + weighted merging
> 5. 没有与 STAR-Pro 直接对比（同期工作，但分析角度互补）
>
> **对 STAR-Pro 的启示**:
> - Nüwa 证明了**位置嵌入策略**是 VG 退化的关键因素，STAR-Pro 需要确保自己的方法也保留了空间完整性
> - Region partition 是一个简单有效的 baseline——如果 STAR-Pro 不做 region-aware 处理，在 VG 上可能也会退化
> - RPME 分析可以作为 STAR-Pro 论文中引用的 related insight
