[← 返回 README](../README.md)

# 2 Preliminaries and Motivation

## 📌 预览

定义核心符号和概念：action chunk、prediction horizon、execution horizon、inference delay。然后用具体数字说明为什么现有 VLA 无法满足实时约束，以及 naive 异步方法为什么不行。

---

We begin with an action chunking policy denoted by $\pi(\mathbf{A}_t | \mathbf{o}_t)$, where $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, ..., \mathbf{a}_{t+H-1}]$ is a chunk of future actions, $\mathbf{o}_t$ is an observation, and $t$ indicates a controller timestep. We call $H$ the prediction horizon. When action chunking policies are rolled out, only the first $s \leq H$ actions from each chunk are executed. We call $s$ the execution horizon; often it is shorter than the prediction horizon, but still much greater than 1 (e.g., $s \approx H/2$ [11, 5, 24]). Chunked execution ensures temporal consistency at the expense of reactivity. A long execution horizon reduces a policy's responsiveness to new information, while a short one increases the likelihood of mode-jumping, jerky behavior resulting from discontinuities between chunks.

> 💡 **关键定义**:
> - **Prediction horizon $H$**：模型一次性预测的动作数量（如 π₀.₅ 的 $H=50$）
> - **Execution horizon $s$**：实际执行的动作数量（$s \leq H$，通常 $s \approx H/2$）
> - 这里有个核心 **trade-off**：
>   - $s$ 大 → 时间一致性好，但反应慢（因为要等 $s$ 步才看下一个观测）
>   - $s$ 小 → 反应快，但 chunk 边界跳变概率高

---

In this paper, we consider policies trained with conditional flow matching [36], though our method can also be used with diffusion policies by converting them to flow policies at inference time [48, 18]. To generate an action chunk from a flow policy, random noise $\mathbf{A}_t^0$ is first sampled from a standard Gaussian, and then the flow's velocity field, $\mathbf{v}_\pi$ (a learned neural network) is integrated from $\tau = 0$ to 1 using the update rule

$$\mathbf{A}_t^{\tau+\frac{1}{n}} = \mathbf{A}_t^\tau + \frac{1}{n} \mathbf{v}_\pi(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau),$$

where $\tau \in [0, 1)$ denotes a flow matching timestep, and $n$ determines the number of denoising steps.

> 💡 **Flow matching 基础**:
> - 从高斯噪声出发，沿着学到的速度场 $\mathbf{v}_\pi$ 积分，逐步去噪到 clean action chunk
> - $\tau$：flow matching 的时间步（0→1 是噪声→干净）
> - $n$：去噪步数（通常 5-10 步，比 diffusion 需要的少很多）
> - 注意：diffusion policy 可以在推理时转成 flow policy [48, 18]，所以 RTC **也适用于 diffusion policy**

---

Now, let $\Delta t$ be sampling period of the controller, i.e., the duration of a controller timestep, and let $\delta$ be the time it takes for the policy to generate an action chunk. We say that a system is real-time if it is guaranteed to produce a response (in our case: $\mathbf{a}_t$) to an event (receiving $\mathbf{o}_t$) within a fixed time constraint ($\Delta t$). If $\delta \leq \Delta t$, then meeting the real-time constraint is trivial, since an entire chunk can be generated between two controller timesteps. However, this is near impossible to achieve with modern VLAs. For example, with an RTX 4090 GPU, the 3 billion parameter $\pi_0$ VLA spends 46ms on the KV cache prefill alone, before any denoising steps [5], and targets a 50Hz control frequency ($\Delta t = 20\text{ms}$). Run in remote inference for mobile manipulation, $\pi_0$ lists 13ms of network latency, in perfect conditions with a wired connection. In a more realistic setting, the network overhead alone could easily exceed 20ms. Kim et al. [31], who optimize the 7B OpenVLA model [30] specifically for inference speed, achieve no better than 321ms of latency on a server-grade A100 GPU.

> 💡 **为什么现有 VLA 不满足实时约束**:
> 
> | 模型 | 延迟 | 控制频率 | 差距 |
> |------|------|---------|------|
> | π₀ (3B) | 46ms（仅 KV prefill）| 50Hz (20ms) | ❌ 仅 prefill 就超标 |
> | π₀ 远程推理 | +13ms（理想网络）| 50Hz | ❌ 网络延迟就够呛 |
> | OpenVLA (7B) | 321ms（A100 优化后）| — | ❌ 差得远 |
> 
> **核心结论**：$\delta \gg \Delta t$，所以**同步推理**（做完再执行）是不可能的。必须异步。

---

![Figure 2](../images/cf37db21a222a8da4baad12d439f07783cd2306d7c376f5d5c2744370ad8da25.jpg)
*Figure 2: An illustration of a typical bifurcation between consecutive chunks. Inference is started between timesteps 3 and 4. The original chunk (black) had planned to go above the obstacle while the newly generated chunk (red) goes below the obstacle. However, the new chunk is not available until d=7 steps later. A naive async algorithm might jump from $a_{10}$ to $a'_{11}$, inducing a very high, out-of-distribution acceleration. Temporal ensembling, i.e., interpolating between chunks, reduces the acceleration but produces poor actions.*

> 💡 **Figure 2 批读**:
> - 这是理解整篇论文动机的核心图！
> - **问题场景**：旧 chunk 计划走障碍物上方，新 chunk 决定走下方 → 在交接点产生巨大跳变
> - **Naive async**：直接切换 → 高加速度，out-of-distribution
> - **Temporal ensembling**：取平均 → 两条路径的平均可能穿过障碍物！这就是 multimodality 的问题
> - 这个图完美说明了为什么需要 inpainting：新 chunk 必须**兼容**已经开始执行的旧 chunk

---

Naive synchronous inference, the default in many prior works [5, 30, 8, 24, 31, 59], simply starts inference at the end of the execution horizon and waits while the policy generates the next chunk. When $\delta > \Delta t$, this introduces visible pauses between chunks that not only slow down execution but also change the dynamics of the robot, introducing distribution shift between training and evaluation. To develop a real-time strategy, we must first introduce asynchronous inference, where inference is started early and happens concurrently with execution.

> 💡 **同步推理的问题**:
> - 执行完 $s$ 个动作 → 停下来等推理完成 → 继续执行
> - 停顿不仅浪费时间，还改变了机器人的动力学（训练时没有这种停顿！）→ **distribution shift**

---

We define $d := \lfloor \delta / \Delta t \rfloor$ and call this quantity the inference delay, corresponding to number of controller timesteps between when $\mathbf{o}_t$ is received and when $\mathbf{A}_t$ is available. Let $\mathbf{a}_{t'|t}$ denote the $(t'-t)$-th action of chunk $\mathbf{A}_t$, generated from observing $\mathbf{o}_t$. If $\mathbf{A}_0$ is currently executing, and we desire an execution horizon of $s$, then an asynchronous algorithm must start inference at $s - d$. So long as $d \leq H - s$, then this strategy will satisfy the real-time constraint and guarantee that an action is always available when it is needed. However, since the policy cannot know what will happen between steps $s - d$ and $s$ while generating $\mathbf{A}_{s-d}$, the transition point between $\mathbf{a}_{s-1|0}$ and $\mathbf{a}_{s|s-d}$ may be arbitrarily discontinuous and out-of-distribution. Similar to a too-short execution horizon, this strategy leads to jerky behavior that is worsened dramatically with higher delays; see Figure 2.

> 💡 **Inference delay 的形式化**:
> - $d = \lfloor \delta / \Delta t \rfloor$：推理延迟，以控制时间步为单位
> - 异步约束：$d \leq H - s$（否则新 chunk 来不及生成）
> - 必须在 $s - d$ 时就开始推理（提前量 = $d$ 步）
> - **但是**：推理开始后、新 chunk 可用前这 $d$ 步里发生了什么，模型不知道！→ 不连续
> 
> 例：π₀.₅ 真实世界设置中，$H=50$, $\Delta t=20\text{ms}$, 推理约 97ms → $d \approx 5-6$

---

![Figure 3](../images/fc42f41da60235cac9a996a8be54b8edbbacf8801f2329866e11689034e8c720.jpg)
*Figure 3: A diagram illustrating how action generation attends to the previous action chunk in real-time chunking. If inference starts after the execution of $a_{-1}$ and the inference delay is $d = 4$, then the newly generated chunk will not be available until after $a_3$ is consumed. Therefore, $a_{0:3}$ are "frozen" and are attended to with a full guidance weight of 1. In the intermediate region, $a_{4:10}$, actions from the previous chunk are available but may be updated. This region is attended to with an exponentially decreasing guidance weight. Finally, the last $s = 5$ actions are beyond the end of the previous chunk, and need to be freshly generated.*

> 💡 **Figure 3 批读** — RTC 的核心机制图：
> - **红色区域 (frozen, W=1)**：$a_{0:3}$ 必须冻结，因为推理完成时它们已经被执行了
> - **橙色→黄色区域 (soft mask, W 递减)**：$a_{4:10}$ 来自旧 chunk，可以参考但不强制匹配。越远的越不确定 → 权重指数衰减
> - **绿色区域 (fresh, W=0)**：超出旧 chunk 范围，需要全新生成
> - 这就是 **soft masking inpainting**：不是简单的 0/1 mask，而是有梯度的权重

---

## 🔖 Section 总结

### 关键数字速查

| 量 | 含义 | π₀.₅ 真实值 |
|---|------|------------|
| $H$ | Prediction horizon | 50 |
| $s$ | Execution horizon | 25 |
| $\Delta t$ | 控制周期 | 20ms (50Hz) |
| $\delta$ | 推理延迟 | ~97ms (RTC) |
| $d$ | 推理延迟（步数）| ~5-6 |

### 核心洞察
1. 实时约束 = $\delta \leq \Delta t$，但现有 VLA **远远无法满足**
2. 同步推理 → 停顿 + distribution shift
3. Naive 异步 → chunk 边界不连续 + OOD
4. 需要一种方法让新 chunk "兼容"旧 chunk 的已执行部分
