[← 返回 README](../README.md)

# 6 Related Work

## 📌 预览
Related Work 覆盖了 3 条主线：chunk-based imitation learning / VLA、异步 chunk 执行、减少推理延迟。A2C2 的独特定位是：它不重做 chunk stitching，也不只追求把模型变快，而是给现有 chunk-based policy 增加一个 **step-level correction layer**。

---

Imitation learning and VLAs: Imitation learning (IL) trains agents from demonstrations provided by humans or expert policies, and has been a representative approach in learning robotic control (Osa et al., 2018). Recent advances have introduced generative sequence models to improve consistency and scalability. Diffusion Policy (Chi et al., 2023) utilizes diffusion models for action generation, enabling it to handle the multimodality of data distribution in imitation learning. In parallel, the Action Chunking Transformer (ACT) (Zhao et al., 2023) proposes a transformer-based policy that outputs action chunks rather than single-step actions, producing coherent behaviors while enabling faster inference. In addition, flow-based approaches, such as Flow Policy (Zhang et al., 2024), generate actions by learning continuous transport maps instead of iterative denoising.

Building on these foundations, a new class of vision-language-action (VLA) foundation models has emerged (Kawaharazuka et al., 2024), including $\pi_0$ (Black et al., 2024), OpenVLA (Kim et al., 2024), GR00T (NVIDIA et al., 2025), and SmolVLA (Shukor et al., 2025). These models adopt chunk-based prediction as the de facto standard for inference, similar to ACT (Zhao et al., 2023). Vision-language-action (VLA) models achieve broad task generalization by aligning multimodal inputs, but their architectures are considerably larger than diffusion- or transformer-based imitation policies. For instance, $\pi_0$ has about 3B parameters and OpenVLA around 7B, which makes inference latency significant even on modern GPU-accelerated hardware. While these models demonstrate the promise of scaling and multimodal grounding, their computational footprint exacerbates the latency problem in real-time control.

> 💡 **Chunk-based VLA 生态梳理**:
> - **生成策略主线**: IL 里的 action generation 已经从单步预测走向 action chunking，代表路线包括 ACT、Diffusion Policy、Flow Policy
> - **VLA 扩展**: 这些 chunk-based 生成范式进一步被扩展到 VLA，形成 $\pi_0$、OpenVLA、GR00T、SmolVLA 这类大模型机器人策略
> - **共同前提**: 这些方法默认接受一个现实交换条件: 用 chunk 换更少的推理次数，但执行阶段也更容易变 open-loop
> - **A2C2 的位置**: 它不替换 chunk-based policy，而是在这个既定前提上补一个逐步 correction 头，让 chunk 执行时重新看见最新 observation

Asynchronous chunk execution: As model sizes increase, inference latency becomes a significant bottleneck, motivating asynchronous policy frameworks. In particular, SmolVLA (Shukor et al., 2025) proposed a server-client architecture for mitigating inference delays. In this setup, the server receives observations and performs inference with a delay of $d$ control steps (including communication latency), then transmits an action chunk of horizon $H$ to the client. Then, the client executes these actions sequentially. However, because the $d$ delayed actions are not yet available at execution time, the client continues executing actions from the previous chunk until the new chunk arrives. This design ensures continuity but introduces the risk of inconsistency between consecutive chunks. For example, the earlier chunk may predict avoiding an obstacle by moving left, while the newly received chunk may instead suggest moving right. Such mismatches across chunks can cause jerky motion and noticeable performance degradation, especially in dynamic environments.

To fix the chunk mismatches, Real Time Chunking (RTC) (Black et al., 2025) is proposed. It is an inference-time algorithm that enables smooth asynchronous execution for action-chunking policies by posing chunk switching as an inpainting problem. Specifically, it generates the next action chunk while executing the current one, freezing actions guaranteed to execute and inpainting the rest.

> 💡 **异步执行路线对比**:
> - **SmolVLA server-client 架构**: 先把“大模型推理有延迟”这件事系统化，明确 client 需要在旧 chunk 上继续执行，直到新 chunk 到达
> - **RTC**: 重点解决 **inter-chunk continuity**，也就是新旧 chunk 切换时不要打架
> - **A2C2**: 重点解决 **intra-chunk reactivity**，也就是当前正在执行的 action 能不能根据最新 observation 及时修正
> - **两者关系**: RTC 更像 chunk stitching，A2C2 更像 per-step calibration；它们不是互斥方案，而是可以叠加的两层补丁

Reducing inference latency: One natural way to enhance a model's real-time performance is to reduce its inference time. Streaming Diffusion Policy (Høeg et al., 2024) or Streaming Flow Policy (Jiang et al., 2025) presents a new training procedure that enables faster inference. More generally, optimizations such as model compression (Lin et al., 2024) or memory optimization (Kwon et al., 2023) of models can also improve inference speed. However, as long as model scale and communication overhead prevent action generation from being faster than the control step, the challenges highlighted in this work remain unresolved.

> 💡 **加速推理路线的局限**:
> - Streaming Diffusion Policy、Streaming Flow Policy、模型压缩、内存优化，目标都是把推理做得更快
> - 这条线当然重要，但它默认的解法仍然是“尽量缩短前向时间”
> - A2C2 的论点更强一些：**只要单次前向传播仍慢于 control step，系统里就需要一个能在 step 级反应的补偿模块**
> - 所以 A2C2 和加速路线是正交的：你可以先把模型加速，再用 A2C2 处理剩余的实时性缺口

## 🔖 Section 总结

### A2C2 的定位
```text
Chunk-based IL / VLA
    ↓ 已经很强，但执行期越来越 open-loop
异步执行 (SmolVLA server-client, RTC)
    ↓ RTC 修的是 chunk 切换连续性
减少推理延迟 (streaming / compression / optimization)
    ↓ 仍未必快到每个 control step 都能重跑大模型
A2C2
    → 在 chunk 内增加 step-level correction
    → 用小头补最新 observation 反馈
    → 与 RTC / 加速路线正交，可叠加
```
