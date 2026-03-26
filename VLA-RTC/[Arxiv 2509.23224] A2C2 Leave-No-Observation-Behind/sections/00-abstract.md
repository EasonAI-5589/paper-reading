[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文提出 Asynchronous Action Chunk Correction (A2C2)，核心不是重写整段 action chunk，而是在每个控制步加一个轻量 correction head，用最新 observation 对当前要执行的 base action 做 residual 校正。它不需要重训 base policy，并且和 RTC 这类异步执行方案正交。

---

To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA’s action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model’s competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic KINETIX task suite (12 tasks) and LIBERO SPATIAL, our method yields consistent success rate improvements across increasing delays and execution horizons $( + 2 3 \%$ point and $+ 7 \%$ point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.

> 💡 **Abstract 批读**:
> - **问题**: VLA 为了减少推理次数会输出 action chunks，但这会把执行阶段变得越来越 open-loop。只要存在推理延迟 `d` 或较长 execution horizon，机器人执行的动作就会越来越依赖旧 observation，闭环反应性下降。
> - **方法**: A2C2 = 在 chunk 执行期间，每个控制步都运行一个小型 correction head。它读取 **latest observation + base action + chunk 内位置特征 + base policy 特征**，输出一个 per-step residual action，用来修正当前要执行的动作。
> - **关键卖点**: 这是一个 **plug-in** 校正模块，不替换 base policy，也 **不需要重新训练 base policy**。同时它和 RTC 这种处理 inter-chunk continuity 的方法是 **正交** 的，理论上可以叠加。
> - **实验**: 作者在两个层次上验证它：
>   1. **KINETIX**: 12 个高动态任务，用来看 delay 和 long horizon 下的鲁棒性
>   2. **LIBERO SPATIAL**: 标准多模态 manipulation benchmark，用来看真正 VLA 设置下是否仍然有效
> - **结果**: 相比 RTC，A2C2 在摘要里报告了 `+23%` point 和 `+7%` point 的提升；更关键的是，它在 **zero-delay but long-horizon** 的条件下也还能提分。这说明作者修的不是单一通信延迟，而是更一般的 **step-level reactivity 缺口**。

## 🔖 小结

### 核心洞察
1. **问题不只在慢推理**: 即使没有显式 delay，只要 horizon 够长，chunk 执行本身也会越来越 open-loop。
2. **A2C2 的定位很清楚**: 大模型继续负责 chunk 级规划，小模型负责 step 级闭环纠偏。
3. **它和 RTC 不是替代关系**: RTC 修 chunk 切换，A2C2 修 chunk 内反应性，两者作用层级不同。
