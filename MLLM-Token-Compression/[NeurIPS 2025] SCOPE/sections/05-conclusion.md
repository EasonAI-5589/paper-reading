[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 SCOPE 的核心贡献和意义。

---

While existing approaches predominantly rely on attention-based saliency to prune redundant tokens, they often neglect semantic coverage, leading to incomplete visual representations. To overcome this limitation, we propose SCOPE, a novel visual token pruning framework that jointly models both token saliency and coverage. Our method introduces a set-coverage score based on pairwise token similarities and calculates a token-coverage gain for each candidate token. By incorporating saliency scores into this gain, we derive the SCOPE score, which guides an iterative token selection process. Empirical evaluations on LLaVA 1.5 and LLaVA-Next across multiple vision-language benchmarks show that SCOPE consistently outperforms state-of-the-art pruning approaches, achieving strong performance even under aggressive token reduction. We believe that our approach offers a principled and effective framework for evaluating the value of visual tokens in MLLMs.

> 💡 **Conclusion 批读**: 简洁有力的总结。强调了 SCOPE 的核心创新（saliency + coverage 联合建模）和主要成果（一致超越 SOTA）。最后一句话提升了方法的定位——不仅是 pruning 方法，更是一个评估 visual token 价值的框架。

---

## 🔖 Section 总结

### SCOPE 全文核心要点
1. **问题**: Saliency-only pruning 导致语义不完整 + 注意力偏斜问题
2. **方法**: SCOPE score = Coverage Gain × Saliency^α，贪心迭代选择
3. **结果**: 9× 压缩保留 96% 性能，一致超越所有 baselines
4. **特点**: Training-free，即插即用，额外开销极小
