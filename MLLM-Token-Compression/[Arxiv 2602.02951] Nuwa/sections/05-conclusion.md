[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结全文核心发现和成果。

---

In this paper, we identify limitations in existing token pruning methods for visual grounding tasks and perform a systematic analysis of VLMs' multi-stage visual processing pipelines. Results reveal task-specific demands, where grounding relies on global spatial reference frames disrupted by pruning. To mitigate this, we propose Nuwa, a two-stage framework with Boids-inspired aggregation ¨ and text-guided refinement to preserve spatial integrity. Extensive experiments across 13 datasets and multiple VLMs show state-of-the-art performance on VQA $9 5 \%$ retention) and VG $4 7 . 2 \%$ retention), with $89 \%$ TFLOPs and $62 \%$ prefill reductions via $8 8 . 9 \%$ token pruning.

> 💡 **批注**: Conclusion 简洁有力。值得注意的是 VG 47.2% 保持率虽然是 SOTA，但绝对值仍然不高——这说明 token pruning 对空间任务的影响是根本性的，仅靠推理时的修补难以完全恢复。未来方向可能包括：(1) training-based 的空间感知 pruning；(2) 更强的位置编码方案（如 RoPE-2D）；(3) 针对不同任务动态调整 pruning 策略。
>
> 另一个值得思考的问题：Nüwa 的分析（Finding 1-3）是否意味着所有 training-free pruning 方法在 VG 上都有天花板？如果是，那么 VG 任务可能需要完全不同的压缩范式（如 learned downsampling）。
