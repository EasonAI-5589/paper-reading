# 1. Introduction

> 来源: SwiftVLM (Arxiv 2403.12178)

---

## 📄 原文

> 💡 **Section 概览**: 两个关键发现 → bypass 范式 → SwiftVLM

---

### 🔑 发现 1: Token 重要性跨层变化

![Figure 2](../images/e1258e0fcf7da8781b141fcc133245e5816047a11b7d71dce007ca4f111a8e8b.jpg)
*Figure 2: 浅层被判定不重要的 visual tokens 在深层可能变得非常重要*

> 💡 **Figure 2 批读**:
> ```
> 实验: 取浅层 (1-9) 排名后 50% 的 token
>        vs 深层 (10-20) 排名前 10% 的 token
> 
> 结果: 存在显著重叠!
>   → 浅层认为"不重要"的 token，深层认为"很重要"
>   → 过早剪枝 = 不可逆信息损失
> ```
> 这直接挑战了 FastV 和 PDrop 的假设。

### 🔑 发现 2: 层的 selection 能力非单调

![Figure 4](../images/465c1bc284a732b9836f68dcf5d5930c3417b102749e2cc99a80c800a56214c6.jpg)
*Figure 4: 不同层选择 visual token 的能力不是单调递增的，中间层反而最强*

> 💡 **Figure 4 批读**:
> ```
> 逐层只保留 top-20% token 的性能:
>   Layer 3-5:  波动大
>   Layer 10-15: 性能最高 (选择能力最强)
>   Layer 20+:   性能下降
>
> 结论: 不是越深越好！应该选 selection 能力最强的层来做剪枝
> ```

### Bypass 范式

![Figure 1](../images/a8d3a0e913ab8120540dd22ac32506ac7134131ddd136ddfe30584cd92156193.jpg)
*Figure 1: (a) Merge (b) Drop (c) Bypass — 三种剪枝策略对比*

> 💡 **Figure 1 批读**:
> ```
> (a) Merge (ToMe 等):
>     被剪 token 合并 → 丢失细粒度信息
>
> (b) Drop (FastV, PDrop 等):
>     被剪 token 直接丢 → 不可逆信息损失
>
> (c) Bypass (SwiftVLM):
>     被剪 token 保留 → 绕道到下一个剪枝层
>     → 在下一层重新评估重要性
>     → 之前被误剪的 token 有机会被"复活"
> ```
> **核心理念**: 给被剪 token 一次"复活"的机会！

![Figure 3](../images/802a623af5d31f9927b335a5b58277a98d97ff291c613fdcf55f7be2b05192a7.jpg)
*Figure 3: FastV 和 PDrop 都丢掉了包含 "NASRI" 的 token，导致回答错误。SwiftVLM 在最终阶段保留了相关 token。*

> 💡 **批注**: 这个例子很直观 — 问 "What does the back of his shirt say?" 需要看球衣号码/名字。FastV/PDrop 在浅层就把那个区域的 token 剪了，SwiftVLM 的 bypass 让它在深层被重新选中。

---

## 💡 Section 总结

### 核心洞察
1. **Token 重要性跨层不一致** — 浅层判断不准，不应该过早做最终决定
2. **层的 selection 能力非单调** — 中间层最强，应选这些层做剪枝
3. **Bypass > Drop** — 给 token 第二次机会，避免不可逆损失
