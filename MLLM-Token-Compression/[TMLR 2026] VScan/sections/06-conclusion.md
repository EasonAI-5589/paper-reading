[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览
总结 VScan 的贡献和局限。

---

In this work, we present a comprehensive empirical study to understand how visual information is processed across both the visual encoding and LLM decoding stages. Building on these insights, we propose VScan—a two-stage, training-free visual token reduction framework—to accelerate LVLM inference while maintaining robust performance. Specifically, we design complementary global and local scanning strategies to select informative visual tokens that preserve rich visual details during visual encoding, and further refine this token set via middle layer pruning in the LLM decoding stage based on textual relevance. Extensive experiments across 4 LVLM architectures and 16 image and video benchmarks demonstrate that our approach consistently outperforms existing state-of-the-art methods, achieving a superior trade-off between efficiency and accuracy.

> 💡 **总结要点**：
> 1. 实证分析 → 方法设计，逻辑完整
> 2. 两阶段 training-free：visual encoder (global+local scan + merge) + LLM (middle layer pruning)
> 3. 4 模型 × 16 benchmark 全面 SOTA

---

**Limitations.** One potential limitation of this work is the inherent trade-off between efficiency and accuracy: while the proposed VScan significantly reduces inference cost of LVLMs, aggressive token pruning may still distort visual information and lead to degraded performance, particularly on challenging tasks that demand fine-grained understanding or compositional reasoning.

> 💡 **局限性批读**：
> - 高压缩率下仍有信息损失，尤其是需要细粒度理解的任务
> - 作者提到了但没解决的问题：
>   - 如何自适应选择压缩率？（当前是固定的 R₁, R₂）
>   - 不同任务/图片的最优压缩率可能不同
>   - Multi-turn conversation 支持（Appendix A.5 简要讨论了，可以 re-select）
