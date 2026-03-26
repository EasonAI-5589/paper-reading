[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 的任务是把问题讲透：大 VLA 很强，但也更慢；action chunking 虽然减少了推理次数，却把执行阶段一步步推向 open-loop；RTC 修的是 chunk 切换连续性，而 A2C2 想修的是 chunk 内每一步的闭环反应性。

---

Recent advances in large vision-language-action (VLA) models have significantly expanded the capability of robots to generalize across tasks and environments (Black et al., 2024; Gemini Robotics Team et al., 2025; NVIDIA et al., 2025; TRI LBM Team et al., 2025). However, large model size requires high computational cost to output the actions for each step, which leads to high inference latency (Kawaharazuka et al., 2025; Black et al., 2025). Especially in dynamic control, such delays become critical. A robot relying on long action sequences predicted from outdated observations can drift, overlook cues, or fail in tasks demanding rapid reactions, such as catching moving objects or stabilizing unstable systems.

The trend of scaling up neural network policies using foundation models brings representational benefits (Sartor & Thompson, 2025), but also incurs a latency problem. For instance, large VLA models such as $\pi_0$ (Black et al., 2024) or OpenVLA (Kim et al., 2024) have billions of parameters and often require hundreds of milliseconds to generate a single action chunk. These action chunks are predicted solely from the previous observation and then executed in an open-loop manner, without incorporating new sensory input during their execution. In addition, latency not only delays execution but also prevents the policy from incorporating the latest observations, thereby weakening its ability to produce reactive behaviors. This is particularly problematic in tasks where the environment changes rapidly during inference. For instance, following a moving object on a cluttered table or grasping a utensil while other objects are being placed, the robot should adjust its action sequence to new sensory inputs. In these scenarios, actions computed from outdated observations accumulate errors over time, which lowers success rates and, in some cases, leads to task failure. This is the central challenge we address in this work.

> 💡 **问题深化**:
> - **显式延迟**: observation 到 action chunk 之间存在推理时间，动作天然会“滞后”
> - **隐式 open-loop**: 一旦 chunk 生成完成，执行期间就不再看新 observation
> - **真正棘手的点**: 第二层问题不会因为把推理做得更快就自动消失，它解释了为什么即使没有显式 delay，只要 horizon 拉长，性能也仍然会掉。

![Figure 1](../images/8d5a7917ccb501d07be482a58b25c9f6de3ff17ee001299f50de04c206737eb8.jpg)

*Figure 1: 异步 action chunk 执行示意图。每个执行动作至少基于落后 `d` 步的 observation，最坏情况会落后 `d + e` 步。*

> 💡 **Figure 1 批读**: 这张图基本就是全文的问题陈述。
> - `H`: chunk 总长度
> - `e`: 实际执行多少步之后再请求新 chunk
> - `d`: 推理延迟
>
> 一旦 `d` 和 `e` 同时变大，机器人执行的就不是“刚刚看到世界之后做出的动作”，而是“基于过去世界快照生成的计划残片”。A2C2 后面所有设计，都是围绕着怎么把这个 stale feedback 问题补回来。

Conventional approaches attempt to mitigate the latency of large models through action chunking (Zhao et al., 2023; Black et al., 2024). By predicting long sequences of actions at once, these methods reduce the frequency of expensive inference calls. However, the chunking strategies can impact performance; robots may experience waiting time during inference, and inconsistencies can arise between successive chunks (Liu et al., 2025). To address this, SmolVLA (Shukor et al., 2025) introduces synchronous execution of the policy, and Real Time Chunking (RTC) (Black et al., 2025) ensures smoother continuity between chunks under asynchrony for diffusion-based action generation. However, these methods still assume that the model predicts fixed-length horizons, which means reactivity to new sensory input remains limited.

Another line of work adopts hierarchical architectures inspired by dual-system reasoning (Kahneman, 2011). Large models serve as a high-level planner (System 2), while smaller policies act as fast executors (System 1). Examples include Hi Robot (Shi et al., 2025), which combines a VLM at the high level with a VLA at the low level, and GR00T-N1 (NVIDIA et al., 2025), which uses a compact policy to refine continuous action chunks. However, since the low-level executor has to wait for predictions from the high-level model, the latency still persists. Consequently, while chunking and hierarchical approaches alleviate some issues, they do not fundamentally solve the challenge of maintaining responsiveness to new observations under the inference delays inherent to VLAs with a large number of parameters.

> 💡 **已有方案为什么不够**:
> - **action chunking**: 省的是推理次数，但没解决 chunk 执行期间越来越看不见新 observation 的问题
> - **RTC**: 修的是 inter-chunk continuity，避免新旧 chunk 切换时打架，**但是在推理时使用的都是推理起点的observation**
> - **层级架构**: 把大模型当 planner、小模型当 executor，但如果低层仍要等高层结果，延迟依然在
>
> 所以 A2C2 的切入点不是“再做一个更平滑的 chunk stitching”，而是把 **step-level feedback** 单独拿出来补。

To mitigate this problem, in this paper, we propose Asynchronous Action Chunk Correction (A2C2), which is a lightweight correction head that can be executed at every timestep to complement the outputs of large VLA models. Unlike conventional approaches such as action chunking and asynchronous inference, our method introduces a lower-level correction layer that directly integrates the most recent observation referring to the action chunks that high-level model outputs. This correction head does not compete with base (high-level) policies like diffusion- or VLA-based chunk generators; instead, it enhances them by injecting real-time feedback to maintain responsiveness under inference delays and long horizons. Through this design, the proposed framework achieves robustness against dynamic environmental changes and external disturbances, thereby mitigating the critical latency bottleneck in deploying large-scale VLA models for real-time robotic control.

In our experiments on the Kinetix tasks, we measure a `35%` point increase in success rate over naive execution and `23%` point increase over RTC in the presence of delay. For long execution horizons, we measure a `12%` point success rate increase over naive execution and `7%` point increase over RTC.

In summary, the contributions of this work are as follows:

- We first formulated delays in policy inference with VLAs that generate action chunks.
- A lightweight add-on action correction policy (A2C2) is introduced to improve reactivity, which can be applied to any VLA model independent of the underlying architecture.
- The method showed substantial improvements in success rates on dynamic tasks and robot manipulation benchmarks with varied inference delays.

> 💡 **A2C2 的定位与贡献**:
> - 它不是替代 base policy，而是给 base chunk 加一个逐步 residual correction 层
> - 它不和 diffusion policy、VLA、RTC 这些路线竞争，而是往它们上面再补一层实时反馈
> - 作者先给出延迟形式化，再给出 plug-in 校正头，最后用 Kinetix 和 LIBERO 证明这件事在动态任务和真实 VLA 设置下都成立

## 🔖 Section 总结

### 核心洞察
1. **问题不只是推理慢**: action chunk 一旦开始执行，就会天然失去对新 observation 的持续利用。
2. **RTC 和 A2C2 解决的不是同一个层级**: RTC 修 chunk 切换，A2C2 修 chunk 内反应性。
3. **A2C2 的出发点很直接**: 把大模型保留在 chunk 级规划，把小模型拉回 step 级反应。
