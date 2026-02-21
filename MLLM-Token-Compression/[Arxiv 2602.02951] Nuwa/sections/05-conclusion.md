[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 Nüwa 的核心发现和贡献。

---

In this paper, we identify limitations in existing token pruning methods for visual grounding tasks and perform a systematic analysis of VLMs' multi-stage visual processing pipelines. Results reveal task-specific demands, where grounding relies on global spatial reference frames disrupted by pruning. To mitigate this, we propose Nüwa, a two-stage framework with Boids-inspired aggregation and text-guided refinement to preserve spatial integrity. Extensive experiments across 13 datasets and multiple VLMs show state-of-the-art performance on VQA (95% retention) and VG (47.2% retention), with 89% TFLOPs and 62% prefill reductions via 88.9% token pruning.

> 💡 **批注**: 结论简洁有力。Nüwa 的最大价值不仅是方法本身，更是揭示了一个被忽视的问题：**token pruning 会系统性破坏空间参考系，导致 VG 任务崩溃**。三条核心发现（Finding 1-3）+ PE 策略分类（PERC/PESP/RPME）是对该领域的重要理论贡献。
> 
> **局限性与未来方向**：
> - 论文没有讨论 dynamic resolution（如 Qwen2.5-VL 的 any-resolution）下的适用性
> - Stage 2 的 text-guided pruning 贡献有限（ablation 显示），是否有更好的设计？
> - 47% VG 保留率虽然远超其他方法，但距离实用（>80%）仍有差距
> - 是否可以将 RPME 策略直接集成到现有方法中作为通用改进？
