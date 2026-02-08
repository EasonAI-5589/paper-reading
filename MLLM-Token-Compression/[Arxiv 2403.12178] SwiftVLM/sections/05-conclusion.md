[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结全文贡献和启示。

---

In this work, we revisit visual token pruning in VLMs and reveal that visual token importance varies substantially across layers. This observation explains why existing drop-based pruning methods, which rely on early selection decisions, often struggle on tasks requiring fine-grained visual reasoning. To better preserve visual information, we introduce a novel pruning strategy, termed bypass, and integrate it into our proposed pruning framework, SwiftVLM. This design allows each pruning layer to perform token selection in a relatively independent manner. Experimental results demonstrate that bypass consistently outperforms drop, suggesting its potential as a promising pruning paradigm.

> 💡 **Conclusion 批读**:
> - 核心贡献回顾：发现层间 token 重要性差异 → 提出 bypass 范式 → SwiftVLM 框架
> - **关键定位**: bypass 作为一种新的剪枝范式（与 merge 和 drop 并列的第三种）
> - **未提及的局限性**:
>   - 只在 LLaVA 系列上验证，未测试 InternVL、Qwen-VL 等
>   - 只用了 2 个剪枝层（3 层中的 2 次剪枝），更多层的 bypass 效果未知
>   - DP 选层需要在训练集上评估，有一定前期成本
>   - 未讨论与 training-based 方法的组合

---

## 🔖 Section 总结

### 核心洞察
1. Bypass 是与 merge/drop 并列的第三种剪枝范式
2. 核心优势：保留完整视觉信息 + 各层独立决策
3. 论文整体定位：simple, training-free, effective
