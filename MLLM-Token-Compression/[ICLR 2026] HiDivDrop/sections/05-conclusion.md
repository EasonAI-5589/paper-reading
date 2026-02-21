[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 HiDivDrop 的核心贡献和对 MLLM 视觉处理层级本质的新认识。

---

In summary, our study challenges prevailing assumptions about visual processing in MLLMs and demonstrates that shallow layers only act as passive propagators for visual tokens. By introducing HiDivDrop with Late Injection, Concave Pyramid Pruning, and Early Exit, we align pruning with the true hierarchical dynamics of multimodal integration. Our findings not only achieve state-of-the-art efficiency–accuracy trade-offs, but also provide new insights into how MLLMs allocate computation across layers, paving the way for more principled and scalable multimodal architectures.

> 💡 **Conclusion 批读**:
> - 核心贡献：**挑战了浅层重要性的流行假设**
> - 方法论意义：剪枝策略应与模型的**实际层级功能**对齐
> - 学术意义：揭示了 MLLM 的三阶段信息处理动态（传声筒→融合→推理）
> - 实践意义：SOTA 效率-精度 trade-off
> - 展望：为更有原则性、可扩展的多模态架构铺路

---

# Ethics Statement

ETHICS STATEMENT

This work does not present any ethical concerns. Our research focuses on methodological contributions and efficiency analysis without involving sensitive data, human subjects, or applications that could raise ethical risks.

# Reproducibility Statement

REPRODUCIBILITY STATEMENT

We have taken several steps to ensure the reproducibility of our work. All experimental settings, including dataset descriptions, training details, and hyperparameter selections, are clearly documented in the main text and appendix. We further provide extensive ablation studies to justify our design choices. Upon acceptance, we will release the full codebase and scripts to facilitate replication of our results.

> 💡 **可复现性**:
> - 实验设置详尽（主文 + 附录）
> - 消融实验充分
> - 承诺开源代码（acceptance 后）
> - 目前为 double-blind review 阶段

---

## 🔖 Section 总结

### 本文的三层贡献
1. **分析层面**：三阶段层级结构（Propagator → Sparse Fusion Hub → Language Reasoning）
2. **方法层面**：HiDivDrop = Late Injection + Concave Pyramid Pruning + Early Exit + ILVAS + DTop-K
3. **工程层面**：Persistent PE + FlashAttention 兼容 + Parallel Decoupling → 理论加速变实际加速

### 局限性（原文未明确讨论，个人观察）
1. 仅在 LLaVA-1.5 框架验证，未测试 Qwen-VL、InternVL 等其他架构
2. 注入/退出层需要 per-model 分析（虽然 Appendix F 展示了跨模型的一致趋势）
3. training-based 方法需要额外训练成本（虽然训练时间本身也减少了）
4. 单图场景为主，多图/视频场景未充分探索
