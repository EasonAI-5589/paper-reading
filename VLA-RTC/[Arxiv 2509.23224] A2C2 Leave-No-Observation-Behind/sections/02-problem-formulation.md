[← 返回 README](../README.md)

# 2 Problem Formulation

## 📌 预览
这一节给出全文最重要的形式化：chunk 长度 `H`、执行 horizon `e`、推理延迟 `d`，以及三者之间的可行区间。如果这组符号关系没看懂，后面的方法、实验和 A2C2 的定位都会显得很飘。

---

We consider an action chunk execution with an imitation learning (IL) policy. As illustrated in Figure 1, an action chunk $A_t = \{a_t, \ldots, a_{t + H - 1}\}$ is from IL policy $\pi$ based on the observation $o_t$ and a language instruction $l$. $H$ is the horizon length, the training sequence length of the IL model $\pi$. We assume it uses $e$ steps of the action chunk, and define it as the execution horizon. Policy predicts the action chunk every $e$ steps as follows:

$$
A_t = \{a_t, \ldots, a_{t + H - 1}\} = \pi(o_t, l).
$$

> 💡 **Chunk 定义 批读**:
> - `H` 决定单次推理吐出多少动作
> - `e` 决定当前 chunk 会被执行多久之后才请求下一个 chunk
> - `H` 偏大时，单次推理更划算，但单个 chunk 更像 open-loop 计划
>
> 这一步其实是在给全文立坐标系：后面所有“实时性”讨论，都是围绕一个 `H` 长度的 chunk 怎么被异步消费展开的。

Also, there is an inference latency. We define the delay $d$ as the number of control steps between receiving an observation $o_t$ and obtaining the corresponding action chunk $A_t$. Formally, it is computed as

$$
d = \left\lfloor \frac{\delta}{\Delta t} \right\rfloor,
$$

where $\delta$ represents the combined inference and communication time, and $\Delta t$ denotes the duration of a single control step.

> 💡 **Delay 定义 批读**:
> - 作者把 **模型推理时间** 和 **通信时间** 统一压进 `d`
> - 所以这里分析的不只是“本地 GPU 太慢”，也包括真实部署里常见的 **client-server VLA** 场景
> - `d` 一旦用控制步数而不是秒来表达，后面讨论等待时间、chunk 用尽和 stale observation 都会非常直观

To control delayed, chunked action execution, the agent executes one action per step till a new chunk arrives asynchronously. Additionally, we assume that the policy server can handle only one inference at a time. If the execution horizon $e$ is shorter than the delay $d$, there will be no action during the model inference, which leads to waiting time. On the other hand, if the execution horizon $e$ is longer than $H - d$, there is no action remaining during the inference time. Therefore, the execution horizon $e$ needs to satisfy $d \leq e \leq H - d$.

In this setting, the agent needs to use the actions that are always based on past observations. Each executed action corresponds to an observation at least $d$ steps old. And in the worst case, the agent may need to execute an action that is generated from the $d + e$ steps past observations.

> 💡 **可行区间与真正问题**:
> - `e < d` 会出现动作断档，机器人只能等
> - `e > H - d` 会导致推理期间 chunk 用完
> - 即使系统工作在合法区间内，执行动作仍然一定是 **stale action**
>
> 所以这节最关键的不是不等式本身，而是它正式说明了：**异步 chunk 执行天然会让 action 落后于 observation**。A2C2 要修的对象不是“让大模型更快”，而是“让最终执行动作别完全受制于旧 observation”。

## 🔖 小结

### 核心洞察
1. **`H / e / d` 是全文坐标系**: `H` 管 chunk 长度，`e` 管消费速度，`d` 管推理与通信滞后。
2. **可行执行本身就有硬约束**: `e` 既不能小于 `d`，也不能大于 `H - d`，否则系统不是等待就是断粮。
3. **合法不代表闭环**: 即使 `d ≤ e ≤ H - d`，执行动作依然至少落后 `d` 步，最坏会落后 `d + e` 步。
