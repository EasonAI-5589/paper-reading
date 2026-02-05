# 6. Conclusion

## 📄 原文

> In this paper, we propose FastV, a **plug-and-play** inference cost optimization method for Large Vision-Language Models.
>
> ==FastV：即插即用的 LVLM 推理成本优化方法==

> Our insight for FastV arises from our observation that the **attention computation over visual tokens is of extreme inefficiency** in the deep layers of popular LVLMs though they take up a large portion of input tokens.
>
> ==核心洞察：Visual tokens 占大部分输入，但深层 attention 计算极度低效==

> FastV prunes out the unnecessary visual tokens according to the **attention score ranking**, which results in **significant inference cost reduction without sacrificing performance**.
>
> ==方法：按 attention score 排序剪枝 → 大幅减少推理成本，性能不降==

---

## 💡 全文总结

### 核心发现
- Visual tokens 在深层 attention 极度稀疏（只有 system prompt 的 1/472）
- 原因：浅层信息已聚合到 "anchor tokens"，深层不再需要 visual tokens

### 方法
- **FastV**：在第 K 层后按 attention score 剪枝 R% 的 visual tokens
- 最佳配置：K=2, R=50%

### 效果
- 45% FLOPs 减少
- 性能几乎无损
- 视频任务甚至性能提升

### 特点
- Plug-and-play：无需重新训练
- 灵活可调：K 和 R 可根据需求调整
- 适用性广：LLaVA、Qwen-VL、Video-LLaVA 等

---

## 与 STAR-Pro 的关系

| 维度 | FastV | STAR-Pro |
|------|-------|----------|
| **发现** | Attention 在深层稀疏 | Attention 在不同阶段不一致 |
| **方法** | 单阶段剪枝 (LLM Decoder) | 两阶段剪枝 (VE + Decoder) |
| **重要性评估** | Attention score | Stage 1: similarity+diversity; Stage 2: text raters |
| **局限** | 依赖 attention，有 positional bias | - |

**STAR-Pro 可以引用 FastV：**
1. ✅ 支持：Vision tokens 在深层确实冗余
2. ⚠️ 指出局限：单阶段剪枝无法适应 guidance 演化
3. 💡 改进：两阶段 adaptive-to-progressive 范式

---

*[返回论文目录](../README.md)*
