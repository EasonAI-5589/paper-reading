[← 返回 README](../README.md)

# 5 Related Work

## 📌 预览
Related Work 把 RTC 放在五条相关研究线索中：action chunking & VLA、减少推理延迟、inpainting & guidance、实时控制、以及最直接的竞争者 BID。

---

**Action chunking, VLAs, and cascade control.** Inspired in part by human motor control [33], action chunking has recently emerged as the de facto standard in imitation learning for visuomotor control [68, 11]. Learning to generate action chunks from human data requires expressive statistical models, such as variational inference [68, 19], diffusion [11, 12, 69, 68, 46, 59], flow matching [5, 6], vector quantization [34, 3, 44], or byte-pair encoding [47]. Recently, some of these methods have been scaled to billions of parameters, giving rise to VLAs [7, 13, 30, 5, 71, 10, 9, 70, 24, 47, 37], a class of large models built on pre-trained vision-language model backbones. With the capacity to fit ever-growing robot datasets [13, 29, 62, 15, 41, 27], as well as Internet knowledge from vision-language pre-training, VLAs have achieved impressive results in generalizable robot manipulation. When applied to real-world robots, action chunking policies are often used in conjunction with a lower-level, higher-frequency control loop—such as a PID controller—which translates the outputs of the policy (e.g., joint positions) to hardware-specific control signals (e.g., joint torques). In these cases, action chunking policies can be viewed as a form of cascade control [14], with the learned policy acting the outermost control loop. However, this is not always the case: for example, our simulated experiments use learned policies that output torques and forces directly. As such, we defer any exploration of the intersection between cascade control theory and learned action chunking policies to future work.

> 💡 **Action chunking 生态**:
> - Action chunk 的生成方法百花齐放：VAE [68]、Diffusion [11]、Flow Matching [5]、VQ [34]、BPE [47]
> - VLA 是把 action chunking 跟 VLM backbone 结合，获得了语言理解+泛化能力
> - **Cascade control 视角**: 把 VLA 看成外层控制器（高级决策），PID 是内层控制器（底层执行）。这个框架很自然但作者说留给未来工作

---

**Reducing inference latency.** A natural approach to improve the real-time capabilities of a model is to simply speed it up. For instance, consistency policy [49] distills diffusion policies to elide expensive iterative denoising. Streaming diffusion policy [23] proposes an alternative training recipe that allows for very few denoising steps per controller timestep. Kim et al. [31] augment OpenVLA [30] with parallel decoding to elide expensive autoregressive decoding. More broadly, there is a rich literature on optimizing inference speed, both for diffusion models [52, 38, 56, 17] and large transformers in general [32, 25, 35]. Unfortunately, these directions cannot reduce inference cost below one forward pass. So long as this forward pass takes longer than the controller's sampling period, other methods will be needed for real-time execution.

> 💡 **加速推理 vs RTC — 互补而非替代**:
> - Consistency Policy [49]: 蒸馏去掉 iterative denoising → 1步生成
> - Streaming Diffusion [23]: 改训练方式让每个控制步只需少量 denoise
> - OpenVLA + parallel decoding [31]: 去掉自回归的串行瓶颈
> - **但！这些方法的下限是 1 次 forward pass**。只要 1 次 forward 就超过 Δt，就还需要异步方法
> - **RTC 跟这些方法是正交的**: 你可以先用这些方法加速，然后再用 RTC 处理剩余延迟

---

**Inpainting and guidance.** There is a rich literature on image inpainting with pre-trained diffusion and flow models [48, 55, 40, 42]. In our work, we incorporate one such method [48] into our novel real-time execution framework with modifications (namely, soft masking and guidance weight clipping) that we find necessary for our setting. For sequential decision-making, Diffuser [26] pioneered diffusion-based inpainting for following state and action constraints in long-term planning, though their inpainting method is not guidance-based. (See Appendix A.4 for a comparison to the inpainting method from Diffuser applied to our setting.) Diffuser and other work [64, 1] have also guided diffusion models with value functions to solve reinforcement learning (RL) problems. Our work is distinct in that it is the first to apply either inpainting or guidance to real-time control.

> 💡 **Inpainting 的谱系**:
> - 图像 inpainting: RePaint [40], ΠGDM [55], Pokle et al. [48]
> - 决策 inpainting: Diffuser [26] 先驱，用 replacement 方式（直接覆盖）而非 guidance
> - **RTC 的定位**: 首次把 guidance-based inpainting 用于**实时控制**
> - Appendix 比较了 Diffuser 的 replacement inpainting vs RTC 的 guidance inpainting，后者更好

---

**Real-time execution.** Real-time control has been studied long before the advent of VLAs. Similar to action chunking, model predictive control (MPC; [51]) generates plans over a receding time horizon; like our method, it parallelizes execution and computation, and uses the prior chunk to warm-start planning for the next. Though recent works combining learning methods with MPC have demonstrated real-time control capabilities in narrow domains [53, 21], they rely on explicit, hand-crafted dynamics models and cost functions. These methods are not applicable to our setting, which considers model-free imitation learning policies and tests them on unstructured, open-world manipulation tasks. Separately, in reinforcement learning, a variety of prior works have developed time-delayed decision-making methods [57, 16, 54, 63, 66, 67]. However, these approaches are not always applicable to imitation learning, and none of them leverage action chunking. Most recently, hierarchical VLA designs [58, 4] have emerged where the model is split into a System 2 (high-level planning) and System 1 (low-level action generation) component. The System 2 component contains the bulk of the VLA's capacity and runs at a low frequency, while the System 1 component is lightweight and fast. This approach is orthogonal to ours, and comes with its own tradeoffs (e.g., limiting the size of the System 1 component and requiring its own training recipe).

> 💡 **RTC vs 其他实时方法**:
> | 方法类别 | 代表 | 与 RTC 的关系 |
> |---------|------|-------------|
> | MPC | [51, 53] | 类似的异步+warm-start思路，但需要显式动力学模型 |
> | Delay-aware RL | [57, 16, 54] | 处理延迟但不用 action chunking，不直接适用 IL |
> | Hierarchical VLA | Gemini Robotics [58], GR00T [4] | System 1/2 拆分，正交于 RTC，但需要专门训练 |
> 
> RTC 的独特定位: **model-free + inference-time only + action chunking**

---

**Bidirectional Decoding.** The most closely related prior work is Bidirectional Decoding (BID; [39]), which enables fully closed-loop control with pre-trained action chunking policies via rejection sampling. While Liu et al. [39] do not consider inference delay, the BID algorithm can be used to accomplish the same effect as our guidance-based inpainting. We compare to BID in our simulated benchmark, finding that it underperforms RTC while using significantly more compute.

> 💡 **BID vs RTC — 最直接的竞争者**:
> - **BID 思路**: 采样一批 chunk，用拒绝采样选跟前一个 chunk 最兼容的
> - **优点**: 简单直觉，不需要反向传播
> - **缺点**: 需要大 batch (64个 chunk) → 计算量 huge；依赖采样多样性，不保证能采到好的
> - **RTC 优势**: 通过 gradient guidance **确定性地**把 chunk 拉向兼容方向，计算量可控

---

## 🔖 Section 总结

### 核心洞察
1. **RTC 在方法论谱系中的定位**: 图像 inpainting + flow matching → 实时 action chunking
2. **跟加速方法正交**: Consistency Policy、Streaming Diffusion 减少单次延迟，RTC 处理剩余延迟
3. **跟 Hierarchical VLA 正交**: System 1/2 需要改架构和训练，RTC 不改任何东西
4. **BID 是最直接对比**: 同样是 inference-time method for action chunking continuity，但 RTC 更高效更有效
