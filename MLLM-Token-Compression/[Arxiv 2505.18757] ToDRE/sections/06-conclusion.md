[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
总结 ToDRE 的两大核心贡献（diversity-driven selection + relevance-driven reduction），重申关键实验数字。

---

In this work, we systematically analyze redundancy in
LVLM inference and identify two key inefficiencies: (1)
redundant visual tokens that inflate intra-modal computation, and (2) tokens that contribute little cross-modal information during decoding. To address these inefficiencies,
we propose TODRE, a training-free, architecture-agnostic
framework that first selects a maximally diverse subset of visual tokens via a greedy max-sum diversification algorithm,
then removes all remaining visual tokens once cross-modal
attention fades. Experiments on twelve image- and videolanguage benchmarks show that ToDRE prunes up to 90%
of visual tokens while preserving 95.0% of the original performance, achieving 2.6 _×_ faster inference and 14.5% lower
memory usage than uncompressed baselines.

> 💡 **批注**: 结论简洁有力，核心数字重申：90% pruning、95.0% 性能、2.6× 加速、14.5% 内存节省。
> 
> **个人总评**:
> - **优势**: 方法简单优雅（两阶段各只有一个核心操作），training-free，跨模型迁移性强
> - **局限**: Stage 2 在短回答 benchmark 上效率提升有限；diversity selection 的 O(nk) 在超长视频（token 数极大）时可能成为瓶颈
> - **与 STAR-Pro 对比**: STAR-Pro 用 R+λD 融合 relevance 和 diversity，ToDRE 证明分开处理更好——这是一个有趣的 design choice 分歧

---

## 🔖 Section 总结

### ToDRE 全文核心要点回顾
| 维度 | 内容 |
|------|------|
| 问题 | LVLM 推理中 visual token 冗余导致计算/内存开销大 |
| 洞察 | 冗余分为 intra-modal (diversity) 和 cross-modal (relevance)，正交 |
| Stage 1 | Greedy max-sum diversification，embedding space，保留 k=10% tokens |
| Stage 2 | Cross-modal attention ratio < τ 时整层移除 visual tokens |
| 结果 | 90% pruning, 95.0% performance, 2.6× speedup, 14.5% memory ↓ |
| 适用性 | Training-free, architecture-agnostic, 兼容 FlashAttention |
