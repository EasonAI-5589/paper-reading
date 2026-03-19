[← 返回 README](../README.md)

# 5 Related Work

## 📌 预览

相关工作分为五大类：action chunking/VLA、推理加速、inpainting/guidance、实时控制、BID。这里可以看到 RTC 在整个 landscape 中的位置。

---

**Action chunking, VLAs, and cascade control.** Inspired in part by human motor control [33], action chunking has recently emerged as the de facto standard in imitation learning for visuomotor control [68, 11]. Learning to generate action chunks from human data requires expressive statistical models, such as variational inference [68, 19], diffusion [11, 12, 69, 68, 46, 59], flow matching [5, 6], vector quantization [34, 3, 44], or byte-pair encoding [47]. Recently, some of these methods have been scaled to billions of parameters, giving rise to VLAs [7, 13, 30, 5, 71, 10, 9, 70, 24, 47, 37], a class of large models built on pre-trained vision-language model backbones. With the capacity to fit ever-growing robot datasets [13, 29, 62, 15, 41, 27], as well as Internet knowledge from vision-language pre-training, VLAs have achieved impressive results in generalizable robot manipulation. When applied to real-world robots, action chunking policies are often used in conjunction with a lower-level, higher-frequency control loop—such as a PID controller—which translates the outputs of the policy (e.g., joint positions) to hardware-specific control signals (e.g., joint torques). In these cases, action chunking policies can be viewed as a form of cascade control [14], with the learned policy acting the outermost control loop. However, this is not always the case: for example, our simulated experiments use learned policies that output torques and forces directly. As such, we defer any exploration of the intersection between cascade control theory and learned action chunking policies to future work.

> 💡 **VLA 生态总结**:
> - Action chunk 的生成方式：VAE (ACT) → Diffusion (DP) → Flow matching (π₀) → VQ (VQBET) → BPE (FAST)
> - VLA = VLM backbone + action head，规模到了数十亿参数
> - 有趣的视角：action chunking policy 可以看作 **cascade control** 的外环（但作者说这个方向留给未来）

---

**Reducing inference latency.** A natural approach to improve the real-time capabilities of a model is to simply speed it up. For instance, consistency policy [49] distills diffusion policies to elide expensive iterative denoising. Streaming diffusion policy [23] proposes an alternative training recipe that allows for very few denoising steps per controller timestep. Kim et al. [31] augment OpenVLA [30] with parallel decoding to elide expensive autoregressive decoding. More broadly, there is a rich literature on optimizing inference speed, both for diffusion models [52, 38, 56, 17] and large transformers in general [32, 25, 35]. Unfortunately, these directions cannot reduce inference cost below one forward pass. So long as this forward pass takes longer than the controller's sampling period, other methods will be needed for real-time execution.

> 💡 **推理加速 vs RTC**:
> - 加速方法：consistency distillation, streaming DP, parallel decoding, quantization...
> - **关键论点**：这些方法有下限——**至少需要一次 forward pass**
> - 只要一次 forward > $\Delta t$（控制周期），纯加速就不够用
> - **RTC 是正交的**：不是加速模型，而是让模型在延迟下仍能实时控制。两者可以叠加使用。

---

**Inpainting and guidance.** There is a rich literature on image inpainting with pre-trained diffusion and flow models [48, 55, 40, 42]. In our work, we incorporate one such method [48] into our novel real-time execution framework with modifications (namely, soft masking and guidance weight clipping) that we find necessary for our setting. For sequential decision-making, Diffuser [26] pioneered diffusion-based inpainting for following state and action constraints in long-term planning, though their inpainting method is not guidance-based. (See Appendix A.4 for a comparison to the inpainting method from Diffuser applied to our setting.) Diffuser and other work [64, 1] have also guided diffusion models with value functions to solve reinforcement learning (RL) problems. Our work is distinct in that it is the first to apply either inpainting or guidance to real-time control.

> 💡 **Inpainting 脉络**:
> - 图像 inpainting：ΠGDM [55], Pokle et al. [48] → RTC 的技术基础
> - 决策 inpainting：Diffuser [26]（用简单的 overwrite 而非 guidance）
> - RTC 的**贡献**：第一个把 inpainting/guidance 用于**实时控制**

---

**Real-time execution.** Real-time control has been studied long before the advent of VLAs. Similar to action chunking, model predictive control (MPC; [51]) generates plans over a receding time horizon; like our method, it parallelizes execution and computation, and uses the prior chunk to warm-start planning for the next. Though recent works combining learning methods with MPC have demonstrated real-time control capabilities in narrow domains [53, 21], they rely on explicit, hand-crafted dynamics models and cost functions. These methods are not applicable to our setting, which considers model-free imitation learning policies and tests them on unstructured, open-world manipulation tasks. Separately, in reinforcement learning, a variety of prior works have developed time-delayed decision-making methods [57, 16, 54, 63, 66, 67]. However, these approaches are not always applicable to imitation learning, and none of them leverage action chunking. Most recently, hierarchical VLA designs [58, 4] have emerged where the model is split into a System 2 (high-level planning) and System 1 (low-level action generation) component. The System 2 component contains the bulk of the VLA's capacity and runs at a low frequency, while the System 1 component is lightweight and fast. This approach is orthogonal to ours, and comes with its own tradeoffs (e.g., limiting the size of the System 1 component and requiring its own training recipe).

> 💡 **实时控制相关工作**:
> - **MPC**：也是"执行当前计划同时规划下一步"→ 和 RTC 思路类似，但 MPC 依赖手工动力学模型
> - **延迟 RL**：有一些方法处理 delayed MDP，但不适用于 imitation learning + action chunking
> - **System 1/2 分层 VLA**（如 Gemini Robotics, GR00T N1）：把大模型跑低频、小模型跑高频 → 和 RTC **正交**，可以叠加
>   - RTC 的优势：不需要改模型架构或训练流程

---

**Bidirectional Decoding.** The most closely related prior work is Bidirectional Decoding (BID; [39]), which enables fully closed-loop control with pre-trained action chunking policies via rejection sampling. While Liu et al. [39] do not consider inference delay, the BID algorithm can be used to accomplish the same effect as our guidance-based inpainting. We compare to BID in our simulated benchmark, finding that it underperforms RTC while using significantly more compute.

> 💡 **BID vs RTC**:
> - BID 也是推理时方法，用 rejection sampling 保持连续性
> - 但：采样 64 个 chunk → **计算量巨大** → 延迟是 RTC 的 2.3 倍
> - 而且性能还不如 RTC（尤其是高 delay 时）
> - BID 没考虑 inference delay（原论文假设延迟为 0）

---

## 🔖 Section 总结

### RTC 在 landscape 中的位置

| 方法类别 | 代表 | 与 RTC 的关系 |
|---------|------|-------------|
| 推理加速 | Consistency, FAST | 正交，可叠加 |
| System 1/2 | Gemini Robotics, GR00T | 正交，可叠加 |
| Inpainting | ΠGDM, Diffuser | RTC 的技术基础 |
| 异步控制 | MPC, delayed RL | RTC 解决相同问题但无需动力学模型 |
| 闭环 chunking | BID | 最直接竞争对手，RTC 更优 |
