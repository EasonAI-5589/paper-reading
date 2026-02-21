[← 返回 README](../README.md)

# Supplementary Material

## 📌 预览
附录包含：benchmark 详细说明、两阶段正交性理论证明、pivot 选择消融、阈值 τ 消融、检测层消融、decoding 阶段 attention 分析、case study（长文本生成 + token 可视化对比）。

---

## 理论分析：Intra- 和 Cross-Modal 冗余的正交性

> 💡 **核心证明**:
> - 将 visual 和 text embedding 映射到正交子空间: $W_V^\top W_T = 0$
> - Intra-modal 冗余 $D_\kappa(V)$: visual token 间的 kernel similarity
> - Cross-modal 冗余 $R_\rho(V, T)$: visual token 对 text 的冗余
> - **结论**: $\text{Cov}(D_\kappa, R_\rho) = 0$ → 两者统计独立
> - **意义**: 理论证明两阶段各自优化不会互相干扰，可以分开处理
>
> **批判性思考**: 正交子空间假设 ($W_V^\top W_T = 0$) 在实际模型中不严格成立（LLM 的 embedding space 是共享的，visual 和 text token 经过对齐后会有一定重叠）。但该证明提供了理想化条件下的理论支撑，实验结果（Stage 1+2 性能提升）也佐证了近似正交性。

---

## Pivot 选择消融 (Appendix Table 1)

> 💡 **消融结果**:
> - [CLS] attention: 98.9% (25%) / 96.0% (10%) — 最优
> - Random: 98.6% / 95.8% — 仅差 0.2-0.3%
> - Center (最近均值): 98.8% / 95.6%
> - Farthest (最远均值): 98.7% / 95.8%
>
> **结论**: Pivot 选择对最终结果影响很小 → diversification 过程本身主导了质量。[CLS] 只是锦上添花。无 [CLS] 的编码器可放心用 random。

---

## 阈值 τ 消融 (Appendix Table 2)

> 💡 **消融结果**:
> - τ = 0.03: 79:22, 99.9% — 过于保守，时间反而增加（因 attention ratio 计算开销）
> - τ = 0.05: 73:24, 100.0%
> - **τ = 0.10**: 72:35, **100.1%** — 最佳平衡点
> - τ = 0.15: 72:25, 100.0% — 更激进但无额外收益
>
> **结论**: τ 在 0.05-0.15 范围内都表现良好，不敏感。默认 0.10。

---

## 检测层消融 (Appendix Table 3)

> 💡 **消融结果**:
> - L/2: 64:19, **87.9%** — 灾难性下降！浅层 cross-modal interaction 仍活跃
> - 5L/8: 58:14, 96.2% — 稍早，有性能损失
> - 6L/8: 68:54, 98.3% — 接近最优
> - **7L/8**: 70:15, **100.0%** — 最佳平衡
> - L (最后一层): 78:51, 99.9% — 几乎无加速
>
> **结论**: 7L/8 是 sweet spot。太早会中断 cross-modal alignment，太晚无加速效果。

---

## Decoding 阶段 Attention 分析 (Appendix Figure 1)

> 💡 **关键发现**:
> - Output token 对 visual token 的 attention 在所有层都 < 5%
> - 浅层主要 attend to system prompt，深层 attend to system prompt + text
> - **结论**: Decoding 阶段 visual token 几乎无用 → 进一步证实 Stage 2 在 prefilling 后期全删 visual token 的合理性
> - 这也解释了为什么即使 Stage 2 only 也能 100% 无损

---

## Case Study: 长文本生成 (Appendix Figure 2)

> 💡 **Video Detail Caption 对比**:
> - **ToDRE**: 准确识别事件和活动
> - **FastV**: 动作描述模糊，遗漏关键物体
> - **FasterVLM**: 生成泛化描述，错误识别主体
> - **意义**: 长文本生成场景中 diversity-based selection 的优势更明显

---

## Token 可视化对比 (Appendix Figure 3)

> 💡 **Attention-driven vs Diversity-driven**:
> - Attention-based: token 分布集中在少数高 attention 区域 → 信息覆盖窄
> - ToDRE: token 分布分散，覆盖更广的空间和语义区域 → 更适合开放式问题
> - 在 7 个 benchmark 上一致观察到此模式

---

## 🔖 Section 总结

### 核心洞察
1. 正交性证明虽基于理想化假设，但实验佐证了近似正交性
2. 所有超参数（pivot 策略、τ、检测层）都不敏感 → 实用性强
3. Decoding 阶段 visual attention < 5% → 全删 visual token 完全合理
4. 长文本和可视化 case study 直观展示了 diversity 的优势
