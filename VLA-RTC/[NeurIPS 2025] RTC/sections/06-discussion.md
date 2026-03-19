[← 返回 README](../README.md)

# 6 Discussion and Future Work

## 📌 预览
简短的 discussion 总结了 RTC 的贡献和局限。

---

Real-time chunking is an inference-time algorithm for asynchronous execution of action chunking policies that demonstrates speed and performance across simulation and real-world experiments, including under significant inference delays. However, this work is not without limitations: it adds significant computational overhead compared to methods that sample directly from the base policy, and it is applicable only to diffusion- and flow-based policies. Additionally, while our real-world experiments cover a variety of challenging manipulation tasks, there are more dynamic settings that could benefit even more from real-time execution. One example is legged locomotion, which is represented in our simulated benchmark but not our real-world results.

> 💡 **局限性分析**:
> 1. **计算开销**: 每个 denoising step 多一次反向传播（76ms → 97ms，+28%）。对于已经很慢的模型来说不是大问题，但如果模型本身很快（如 consistency policy），这个开销就不可忽略了
> 2. **只适用于 diffusion/flow-based policy**: 不能用在 autoregressive VLA（如 RT-2 [8]）或 VQ-based policy（如 VQ-BeT [34]）上。这是因为 inpainting 依赖于 iterative denoising 的结构
> 3. **真实实验缺少 locomotion**: 仿真有腿式运动任务，但真实世界只测了操作任务。高动态 locomotion（如四足跑步）可能从 RTC 获益更大
> 
> **我的补充局限**:
> - RTC 依赖于推理延迟的 **可预测性**（用历史延迟估计未来延迟）。如果延迟剧烈波动（如不稳定网络），可能需要更保守的策略
> - 软件复杂度增加：双线程 + 反向传播 + 动态 execution horizon，工程实现比同步推理复杂很多

---

## Appendix 关键内容

### A.2 Guidance Weight Clipping (β)

![Figure 7](../images/9eef3d12e72737add7d01e11a778e95d9ba34e293b8e92a5ffb88c52073cffb6.jpg)
*Figure 7: β 消融。Top left: guidance 权重在 τ=0 时趋向无穷。Top right: β 的仿真消融，β≥5 后无边际收益。Bottom: β 过高导致 action chunk 发散和高加速度。*

> 💡 **Figure 7 批读**:
> - 5 步 denoising 时，如果 $\beta \geq 4.25$，clipping 只影响第一步 ($\tau=0$)
> - $\beta = 5$ 是 sweet spot：足够大以提供有效 guidance，又不会导致发散
> - Bottom right: β 越高 → 最大加速度越高 → action 越 jerky。这说明 β 截断不只是数值稳定性问题，也是输出质量问题

### A.3 Latency Measurements

| Method | Latency |
|--------|---------|
| **RTC** | **97ms** |
| BID N=16 (no forward) | 115ms |
| BID N=16 (shared backbone) | 169ms |
| BID N=16 (full) | 223ms |
| Vanilla π₀.₅ | 76ms |

> 💡 **延迟对比**: RTC 只比 vanilla 慢 21ms (+28%)，但 BID 最多慢 147ms (+194%)。而且 BID 的 batch=16 已经是减半了（仿真用 32）。

### A.4 Decay Schedule Comparison

![Figure 8](../images/dcd90fa430188c15f9a534a5940a3e719db1eeab647f6ba32a46f6924a3000b2.jpg)
*Figure 8: Left: 不同衰减函数对比（指数 > 线性 > 其他）。Right: Diffuser 的 replacement inpainting vs RTC 的 guidance inpainting。*

> 💡 **Figure 8 批读**:
> - 指数衰减最好，线性衰减接近。其他方案（constant、step）都差一些
> - Diffuser 的 replacement 方法（每步直接覆盖已知 action）不如 guidance 方法。这是因为 replacement 不通过梯度信号影响整个 chunk，只是机械地覆盖部分维度

---

## 🔖 Section 总结

### 核心洞察
1. **RTC 的局限是诚实的**: 计算开销、只适用 diffusion/flow、缺少 locomotion 验证
2. **β=5 是经验值但有理论支持**: 5 步 denoising 只影响第一步，足够保守
3. **RTC vs BID 延迟**: 97ms vs 115-223ms，RTC 明显更高效
4. **Guidance > Replacement**: 梯度引导比简单覆盖更能保证全局一致性
