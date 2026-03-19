[← 返回 README](../README.md)

# 4 Experiments

## 📌 预览

实验回答三个问题：(1) RTC 在高动态环境下 vs baselines 表现如何？(2) Soft masking 有多重要？(3) RTC 对真实机器人的性能和速度有什么影响？分为模拟（Kinetix 12 任务）和真实世界（π₀.₅ 6 任务）两部分。

---

In our experiments, we aim to answer the following questions. First, how does RTC compare to existing methods in highly dynamic and stochastic environments, and under increasing inference delays? Second, how important is soft masking (Sec. 3.2) to RTC? Third, how does RTC affect the performance and speed of real-world dexterous robots?

We first evaluate RTC using a benchmark of 12 highly dynamic and stochastic environments in the Kinetix [43] simulator. We use this benchmark to compare the performance of RTC to other methods under simulated inference delays, as well as investigate the effect of soft masking. Then, using the $\pi_{0.5}$ VLA [24] as the base model, we evaluate the performance and speed of RTC on 6 challenging bimanual dexterous manipulation tasks, including 2 mobile manipulation tasks.

> 💡 **实验设计思路**:
> - 模拟实验：控制变量，可以精确调节 inference delay，验证算法的鲁棒性
> - 真实世界：展示实际价值，用 PI 自家的 π₀.₅ 跑 6 个高难度双臂任务

---

## 4.1 Simulated Benchmark

Most simulated imitation learning benchmarks are quasi-static, and standard chunked execution with a long enough execution horizon can achieve near-perfect success rates [11]. We instead create a benchmark of 12 dynamic tasks in Kinetix [43], which uses force-based control, so inference delay necessitates asynchronous execution (there is no concept of "holding position"). We select 10 existing environments and create 2 new ones such that all environments involve dynamic motions like throwing, catching, and balancing. To simulate imperfect actuation, we add Gaussian noise to the actions, making closed-loop corrections crucial for success.

> 💡 **为什么要新 benchmark**:
> - 现有 benchmark（robomimic、ManiSkill 等）太 quasi-static → open-loop 就能搞定
> - Kinetix 用 **force-based control**（直接输出力/力矩），没有"停住不动"的概念 → 推理延迟直接影响结果
> - 任务类型：投掷、接住、平衡——都是高动态的
> - 加了动作噪声 → 闭环修正是成功的关键

---

**Setup.** To generate data for imitation learning, we first train expert policies using RPO [50] and a binary success reward. For each environment, we train 6 expert policies with different seeds and then generate a 1M transition dataset with a different policy selected each episode. We then train action chunking flow policies with a prediction horizon of $H = 8$ and a 4-layer MLP-Mixer [61] architecture for 32 epochs. We report binary success rates with 2048 rollouts per data point, and simulate delays between 0 (fully closed-loop) and 4 (the maximum supported when $H = 8$).

> 💡 **模拟实验设置**:
> - 数据：RL 专家（RPO，6 seeds）→ 1M transition → 模仿学习
> - 模型：MLP-Mixer, $H=8$（很短的 horizon，适合测试 delay 的影响）
> - 评估：2048 rollouts（统计充分！）
> - Delay 范围：0-4（对于 $H=8$ 来说，$d=4$ 已经是极限）

---

**Baselines.** We compare against the following baselines:

- **Naive async.** This strategy does not pay attention to the previous action chunk at all when generating a new one, naively switching chunks as soon as the new one is ready.
- **Bidirectional decoding (BID; [39]).** This strategy uses rejection sampling to keep continuity across chunks. We use a batch size of $N = 32$, mode size of $K = 3$, and a checkpoint trained for 8 epochs as the weak policy.
- **Temporal ensembling (TE; [68]).** This strategy involves keeping a buffer of predicted action chunks and executing an average of all actions predicted for a particular timestep.

> 💡 **Baselines 分析**:
> - **Naive async**：无脑切换，最简单但最差
> - **BID**：用 rejection sampling 保持连续性——采样一批 chunk，选与旧 chunk 最一致的。问题：需要采样 64 个 chunk（32 strong + 32 weak），计算代价巨大
> - **TE**：ACT 论文 [68] 的方法，取多个 chunk 的平均。问题：multimodal 分布的平均不一定有效

---

![Figure 5](../images/b01822a7ef5ca22d7ee4449fc1cdb742470a8d029ad55f3be5b09a2d79ae7e84.jpg)
*Figure 5: Top left: Kinetix environments; each involves getting a green object on the left to touch a blue one on the right. Bottom left: Execution horizon vs. solve rate with a fixed inference delay of 1. Only RTC and BID take full advantage of faster updates. Right: Inference delay vs. solve rate with a fixed execution horizon of $s = \max(d, 1)$. RTC outperforms all baselines. Each data point represents 2048 trials, and 95% Wilson score intervals are shaded in.*

> 💡 **Figure 5 批读** — 核心模拟结果：
> 
> **右图（delay vs. solve rate）** — 最重要的结果：
> - TE 在所有 delay 下都很差（即使 $d=0$！）→ 证明 multimodality 是真问题
> - RTC 随 delay 增加仍然保持最好性能
> - RTC 在 $d=3,4$ 时比 BID 优势更明显（BID 开始崩了）
> - Soft masking (实线) vs Hard masking (虚线)：小 $d$ 时差别更大
> 
> **左图（execution horizon vs. solve rate）**：
> - 只有 RTC 和 BID 能从更短的 execution horizon（更频繁的更新）中获益
> - Naive async 和 TE 减小 $s$ 反而变差（因为 chunk 切换更频繁 → 不连续更多）

---

**Results.** Figure 5 shows the simulated results. In the delay plots (right): TE performs poorly across the board, even with an inference delay of $d = 0$, illustrating the multi-modality of our benchmark—averages of valid actions are not necessarily valid. RTC shows the most robustness to inference delays, outperforming BID, and the gap widens with increasing delay; note that BID uses significantly more compute than RTC by sampling batches of 64 action chunks, 32 from a strong model and 32 from a weak model. Additionally, we find that hard masking somewhat underperforms soft masking, particularly when $d$ is smaller, supporting our claims in Sec. 3.2. Finally, in the execution horizon plot (left), we find that thanks to its continuity across chunks, RTC is better able to take advantage of closed-loop corrections, always performing better with a decreasing execution horizon.

> 💡 **模拟结果总结**:
> 1. **RTC >> BID >> TE**（在高动态任务上）
> 2. RTC 对 delay 最鲁棒——甚至在 $d=4$（$H/2$！）时仍有不错的成功率
> 3. BID 计算量是 RTC 的 2-3 倍，但性能更差
> 4. Soft masking 在小 delay 时帮助最大
> 5. RTC 能真正利用更短的 execution horizon（更好的闭环修正）

---

## 4.2 Real-World Results

Next, we deploy our full real-time chunking system to the real world. We use the $\pi_{0.5}$ VLA [24] as our base policy, and evaluate RTC on a bimanual system with two 6-DoF arms and parallel jaw grippers. Unlike our simulated benchmark, the robots use position control, and so synchronous inference—stopping between chunks—is a reasonable default strategy, used in many prior works [5, 24, 31, 47]. Our goal is to improve upon synchronous inference in a combination of both performance and speed.

> 💡 **真实世界设置的关键区别**：
> - 位置控制（不是力控）→ 同步推理时机器人可以"停住"而不会摔倒
> - 所以同步推理是合理的 baseline（但慢）
> - 目标：**同时提升**性能和速度

---

**Setup.** We use $\pi_{0.5}$ ($H = 50$, $\Delta t = 20\text{ms}$) with $n = 5$ denoising steps, giving a model latency of 76ms for the baselines and 97ms for RTC. We use remote inference over LAN, which adds 10-20ms of latency, giving a starting inference delay around $d \approx 6$ for RTC. However, we would like to understand how the system behaves with higher inference latencies, simulating, e.g., scaling up the model size or running inference on a distant cloud server. Thus, we also evaluate all methods with +100ms and +200ms of injected latency, corresponding to $d \approx 11$ and $d \approx 16$, respectively.

> 💡 **延迟设置**:
> 
> | 设置 | 额外延迟 | 总 $d$ | 模拟场景 |
> |------|---------|--------|---------|
> | 基础 | 0 | ~6 | LAN 远程推理 |
> | +100ms | +100ms | ~11 | 更大模型 |
> | +200ms | +200ms | ~16 | 云推理 |
> 
> 注意 $d=16$ 意味着 $16 \times 20\text{ms} = 320\text{ms}$ 延迟，占 $H=50$ 的 32%！

---

**Tasks and scoring.** Each episode gets an integer score corresponding to how many substeps of the task it completed successfully. We evaluate the following tasks:

- **Light candle** (5 steps, 40s cutoff). Pick up a match and matchbox, strike the match, use it to light a candle, and drop it in a bowl.
- **Plug ethernet** (6 steps, 120s cutoff). Pick up the end of an ethernet cable, reorient it, plug it into a server rack, and repeat the process for the other end.
- **Make bed, mobile** (3 steps, 200s cutoff). Move the corner of a blanket and 2 pillows from the foot to the head of a bed.
- **Shirt folding** (1 step, 300s cutoff). Fold a shirt from a flattened position.
- **Batch folding** (4 steps, 300s cutoff). Take a varied, crumpled clothing item out of a bin, flatten it, fold it, then place it neatly on a pile.
- **Dishes in sink, mobile** (8 steps, 300s cutoff). Move 4 varied items from a counter into a sink.

> 💡 **任务分析**:
> - 6 个任务覆盖了多种挑战：
>   - **精细操作**：划火柴（对实时性要求极高）、插网线（精确对准）
>   - **长 horizon**：叠衣服、整理碗碟（需要持续稳定的动作）
>   - **移动操作**：铺床、碗碟入水池（需要底盘移动 + 手臂协调）
> - 每个任务有 **substep 评分**，不只是二值成功/失败
> - **Light candle 是唯一不能重试的任务**——火柴划着后必须一次性成功

---

**Baselines.** We compare to the following baselines:

- **Synchronous.** This corresponds to the default inference strategy in prior work [5, 24, 31, 47], which executes $s = 25$ actions and then pauses while the new chunk is generated.
- **TE, sparse.** This is similar to naive async; it executes $s = 25$ actions at a time while computing the next chunk in parallel. We found it significantly reduced jerkiness to also apply TE.
- **TE, dense.** This strategy is the closest to the original TE in Zhao et al. [68]. We run inference as often as possible, resulting in $s = d$ for every chunk.

We do not compare to BID [39] in the real world, as we found in simulation that it underperforms RTC while using significantly more compute—when applied to $\pi_{0.5}$ with a batch size of 16, BID has 2.3 times the latency of our method (see A.3 for latency measurements).

> 💡 **Baselines 注意**:
> - Synchronous 是 PI 自己之前用的默认方法（π₀, π₀.₅ 论文中）
> - BID 被排除了——模拟中已经证明它比 RTC 差且贵 2.3 倍
> - TE sparse/dense 是两种不同频率的 temporal ensembling

---

![Figure 6](../images/57e3a1534497d9eef97a7d00e2502b04177edb490b4d65007daee287da3a3e83.jpg)
*Figure 6: Top: Controller steps vs. cumulative progress for each task, aggregated across all delays. Left: Time vs. cumulative progress aggregated across all tasks (X-axis is log scale). Right: Inference delay vs. average throughput, defined as the proportion of task completed divided by duration of episode averaged over episodes.*

> 💡 **Figure 6 批读** — 最重要的真实世界结果：
> 
> **右图（delay vs. throughput）** — headline result：
> - RTC 在所有 delay 下都是最好的
> - **RTC 对注入延迟完全鲁棒**——从 +0 到 +200ms 几乎没有性能下降！
> - Synchronous 线性下降
> - **TE 两个变体在 +100/+200ms 时直接挂了**——震荡太大触发了机器人保护性停机
> 
> **上面六个子图（per-task 进度曲线）**：
> - Light candle：RTC 优势最大（不能重试 → 实时性直接决定成功率）
> - 其他任务中，即使最终分数相似，RTC 也**更早完成更多步骤**（更少犯错、更少重试）
> 
> **左图（时间 vs. 总体进度）**：
> - RTC 一直领先，最终完成度更高
> - 即使去掉推理暂停时间（top 子图用 controller steps），RTC 仍然更快 → **不只是因为没暂停，而是动作质量更好**

---

**Results.** We present the results in Figure 6. In average task throughput, a measurement of both speed and performance, RTC achieves the best score at all inference delays with a statistically significant result at +100 and +200ms. RTC is completely robust to injected delay, showing no degradation, whereas synchronous degrades linearly and both TE variants do not run at all due to causing such high oscillations that the robot's protective stop is triggered (see videos). Inspecting the per-task results (Figure 5, top), we can conclude that RTC helps with more than just execution speed: it completes tasks faster than synchronous inference even when inference pauses are removed. All tasks, except for light candle, allow for retrying until the time limit (and $\pi_{0.5}$ does, in general, exhibit robust retrying behavior). Even though synchronous inference often reaches a similar final score, RTC often completes more of the task earlier in the episode, reflecting fewer mistakes and less retrying. In light candle, the most precision-sensitive task—and also the only one without retrying—RTC shows a large advantage in final score, reflecting a higher overall success rate. Interestingly, the same is true in bed making, even though that task does elicit retrying. The policy particularly struggles to manipulate the pillows, and bed making is the hardest task overall, which may be why RTC has a strong effect.

> 💡 **真实世界结果总结**:
> 1. **RTC 对延迟完全鲁棒** — 从 ~120ms 到 ~320ms 延迟无性能下降
> 2. **TE 在高延迟下不可用** — 触发保护性停机（这是非常有说服力的结果）
> 3. RTC 不仅更快，而且**动作质量更高**（即使去掉暂停时间仍然更快 → 更少重试）
> 4. 在**划火柴**（最精细、不可重试）上优势最大
> 5. 480 个 episode，28 小时机器人执行时间 — 评估非常扎实

---

## 🔖 Section 总结

### 关键数字速查

| 指标 | RTC | Synchronous | TE |
|------|-----|-------------|-----|
| 对 +200ms 延迟的鲁棒性 | ✅ 无下降 | ❌ 线性下降 | ❌ 保护停机 |
| 速度提升（vs sync） | ~20% | baseline | — |
| 计算开销（vs vanilla） | +28% | baseline | ~0% |
| 评估规模 | 480 episodes / 28h |

### 核心洞察
1. RTC 在真实世界中不仅更快，而且更准确
2. 对延迟的鲁棒性是 RTC 最大的卖点——意味着可以用更大的模型或云推理
3. TE 方法在高延迟下完全不可用
4. Light candle 是最有说服力的任务——不可重试 + 高精度需求
