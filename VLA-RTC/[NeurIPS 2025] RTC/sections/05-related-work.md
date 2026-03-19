[← 返回 README](../README.md)

# 5 Related Work

## 📌 预览
Related Work 覆盖了 5 个方向：action chunking & VLA、减少推理延迟、inpainting & guidance、实时控制、BID。RTC 的独特定位是：第一个将 inpainting/guidance 应用到**实时控制**的工作。

---

**Action chunking, VLAs, and cascade control.** Inspired in part by human motor control [33], action chunking has recently emerged as the de facto standard in imitation learning for visuomotor control [68, 11]. Learning to generate action chunks from human data requires expressive statistical models, such as variational inference [68, 19], diffusion [11, 12, 69, 68, 46, 59], flow matching [5, 6], vector quantization [34, 3, 44], or byte-pair encoding [47]. Recently, some of these methods have been scaled to billions of parameters, giving rise to VLAs [7, 13, 30, 5, 71, 10, 9, 70, 24, 47, 37], a class of large models built on pre-trained vision-language model backbones. With the capacity to fit ever-growing robot datasets [13, 29, 62, 15, 41, 27], as well as Internet knowledge from vision-language pre-training, VLAs have achieved impressive results in generalizable robot manipulation. When applied to real-world robots, action chunking policies are often used in conjunction with a lower-level, higher-frequency control loop—such as a PID controller—which translates the outputs of the policy (e.g., joint positions) to hardware-specific control signals (e.g., joint torques). In these cases, action chunking policies can be viewed as a form of cascade control [14], with the learned policy acting the outermost control loop. However, this is not always the case: for example, our simulated experiments use learned policies that output torques and forces directly. As such, we defer any exploration of the intersection between cascade control theory and learned action chunking policies to future work.

> 💡 **Action Chunking 生态梳理**:
> - **生成模型选择**: VAE [68] → Diffusion [11] → Flow Matching [5] → VQ [34] → BPE [47]
> - **VLA 代表作**: RT-2 [7,8], OpenVLA [30], π₀ [5], π₀.₅ [24], FAST [47], RDT-1B [37]
> - **Cascade control 视角**: VLA 是最外层控制环，PID 是内层。RTC 不改变这个架构，只优化外层的执行策略。

---

**Reducing inference latency.** A natural approach to improve the real-time capabilities of a model is to simply speed it up. For instance, consistency policy [49] distills diffusion policies to elide expensive iterative denoising. Streaming diffusion policy [23] proposes an alternative training recipe that allows for very few denoising steps per controller timestep. Kim et al. [31] augment OpenVLA [30] with parallel decoding to elide expensive autoregressive decoding. More broadly, there is a rich literature on optimizing inference speed, both for diffusion models [52, 38, 56, 17] and large transformers in general [32, 25, 35]. Unfortunately, these directions cannot reduce inference cost below one forward pass. So long as this forward pass takes longer than the controller's sampling period, other methods will be needed for real-time execution.

> 💡 **"加速推理"路线的局限**:
> - Consistency Policy [49]: 蒸馏去掉迭代 denoising
> - Streaming Diffusion Policy [23]: 每个控制步只做很少 denoising steps
> - OpenVLA + parallel decoding [31]: 并行解码替代自回归
> - **共同局限**: 不管怎么优化，延迟不可能低于**单次前向传播**。只要前向传播 > Δt，就需要异步方案。
> - **RTC 与这些方法正交**: 你可以先加速模型，再用 RTC 处理剩余延迟。两者可以叠加。

---

**Inpainting and guidance.** There is a rich literature on image inpainting with pre-trained diffusion and flow models [48, 55, 40, 42]. In our work, we incorporate one such method [48] into our novel real-time execution framework with modifications (namely, soft masking and guidance weight clipping) that we find necessary for our setting. For sequential decision-making, Diffuser [26] pioneered diffusion-based inpainting for following state and action constraints in long-term planning, though their inpainting method is not guidance-based. (See Appendix A.4 for a comparison to the inpainting method from Diffuser applied to our setting.) Diffuser and other work [64, 1] have also guided diffusion models with value functions to solve reinforcement learning (RL) problems. Our work is distinct in that it is the first to apply either inpainting or guidance to real-time control.

> 💡 **Inpainting 在控制中的先驱**:
> - **Diffuser [26]**: 第一个在决策中用 diffusion inpainting，但方法不同（直接替换，非 guidance-based）
> - Appendix A.4 对比了 Diffuser 的 inpainting 和 RTC 的 ΠGDM guidance → RTC 明显更好
> - **RTC 的新颖性**: 第一个把 inpainting/guidance 用于**实时控制**（而非规划）

---

**Real-time execution.** Real-time control has been studied long before the advent of VLAs. Similar to action chunking, model predictive control (MPC; [51]) generates plans over a receding time horizon; like our method, it parallelizes execution and computation, and uses the prior chunk to warm-start planning for the next. Though recent works combining learning methods with MPC have demonstrated real-time control capabilities in narrow domains [53, 21], they rely on explicit, hand-crafted dynamics models and cost functions. These methods are not applicable to our setting, which considers model-free imitation learning policies and tests them on unstructured, open-world manipulation tasks. Separately, in reinforcement learning, a variety of prior works have developed time-delayed decision-making methods [57, 16, 54, 63, 66, 67]. However, these approaches are not always applicable to imitation learning, and none of them leverage action chunking. Most recently, hierarchical VLA designs [58, 4] have emerged where the model is split into a System 2 (high-level planning) and System 1 (low-level action generation) component. The System 2 component contains the bulk of the VLA's capacity and runs at a low frequency, while the System 1 component is lightweight and fast. This approach is orthogonal to ours, and comes with its own tradeoffs (e.g., limiting the size of the System 1 component and requiring its own training recipe).

> 💡 **MPC vs RTC**:
> | 对比项 | MPC | RTC |
> |--------|-----|-----|
> | 模型 | 显式动力学模型 | 隐式（学习的 VLA） |
> | 目标函数 | 手工设计 | 从 demonstration 学习 |
> | 异步执行 | ✅ (warm-start) | ✅ (inpainting) |
> | 适用场景 | 窄域（已知动力学） | 开放世界操作 |
> 
> **System 1/2 分离** (Gemini Robotics [58], GR00T [4]): 大模型低频思考 + 小模型高频执行。跟 RTC 正交——可以在 System 2 上加 RTC。

---

**Bidirectional Decoding.** The most closely related prior work is Bidirectional Decoding (BID; [39]), which enables fully closed-loop control with pre-trained action chunking policies via rejection sampling. While Liu et al. [39] do not consider inference delay, the BID algorithm can be used to accomplish the same effect as our guidance-based inpainting. We compare to BID in our simulated benchmark, finding that it underperforms RTC while using significantly more compute.

> 💡 **BID vs RTC 最终对比**:
> - BID: 采样 N 个 chunk → 挑最好的（rejection sampling）→ 计算量 O(N)
> - RTC: 1 个 chunk + guidance → 计算量 O(1) + backprop
> - 结果: RTC 效果更好、计算更少、实现更简单

---

## 🔖 Section 总结

### RTC 的定位
```
加速推理 (Consistency, Streaming, Parallel Decoding)
    ↓ 不够：单次前向传播 > Δt
异步执行 (MPC warm-start, RL delay methods)  
    ↓ 不够：需要显式模型或不适用 IL
Action chunking + inpainting/guidance
    ↓ RTC: 第一个用于实时控制的 inpainting 方法
System 1/2 分离 (Gemini Robotics, GR00T)
    ↑ 正交：可以叠加使用
```
