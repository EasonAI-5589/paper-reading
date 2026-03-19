[← 返回 README](../README.md)

# 4 Experiments

## 📌 预览
实验分仿真和真实两部分。仿真用 Kinetix 的 12 个高动态任务测试不同延迟下的鲁棒性；真实世界用 π₀.₅ 在 6 个双臂操作任务上评估，总计 480 episodes、28 小时机器人执行时间。核心结论：RTC 在所有延迟下都最优，且是唯一对延迟完全鲁棒的方法。

---

In our experiments, we aim to answer the following questions. First, how does RTC compare to existing methods in highly dynamic and stochastic environments, and under increasing inference delays? Second, how important is soft masking (Sec. 3.2) to RTC? Third, how does RTC affect the performance and speed of real-world dexterous robots?

We first evaluate RTC using a benchmark of 12 highly dynamic and stochastic environments in the Kinetix [43] simulator. We use this benchmark to compare the performance of RTC to other methods under simulated inference delays, as well as investigate the effect of soft masking. Then, using the π₀.₅ VLA [24] as the base model, we evaluate the performance and speed of RTC on 6 challenging bimanual dexterous manipulation tasks, including 2 mobile manipulation tasks.

> 💡 **实验设计三个问题**:
> 1. RTC vs baselines 在不同延迟下的表现？
> 2. Soft masking 有多重要？
> 3. 真实世界中 RTC 对速度和性能的影响？

---

## 4.1 Simulated Benchmark

Most simulated imitation learning benchmarks are quasi-static, and standard chunked execution with a long enough execution horizon can achieve near-perfect success rates [11]. We instead create a benchmark of 12 dynamic tasks in Kinetix [43], which uses force-based control, so inference delay necessitates asynchronous execution (there is no concept of "holding position"). We select 10 existing environments and create 2 new ones such that all environments involve dynamic motions like throwing, catching, and balancing. To simulate imperfect actuation, we add Gaussian noise to the actions, making closed-loop corrections crucial for success.

> 💡 **为什么要新 benchmark**:
> - 现有 benchmark（如 Diffusion Policy 用的那些）太简单——quasi-static，长 execution horizon 的 open-loop 就能搞定
> - Kinetix 用**力控制**（force-based），没有"保持位置"的概念 → 推理延迟时机器人不会静止，而是会飘走
> - 加了 action 噪声 → 必须 closed-loop 修正，不能靠 open-loop 过关

---

**Setup.** To generate data for imitation learning, we first train expert policies using RPO [50] and a binary success reward. For each environment, we train 6 expert policies with different seeds and then generate a 1M transition dataset with a different policy selected each episode. We then train action chunking flow policies with a prediction horizon of $H = 8$ and a 4-layer MLP-Mixer [61] architecture for 32 epochs. We report binary success rates with 2048 rollouts per data point, and simulate delays between 0 (fully closed-loop) and 4 (the maximum supported when $H = 8$).

> 💡 **仿真实验配置**:
> | 配置项 | 值 |
> |--------|-----|
> | Expert 训练 | RPO, 6 seeds × 12 envs |
> | 数据集 | 1M transitions/env |
> | Policy 架构 | 4-layer MLP-Mixer |
> | $H$ (prediction horizon) | 8 |
> | 训练 epochs | 32 |
> | 评估 rollouts | 2048/data point |
> | 延迟范围 | 0 ~ 4 步 |

---

**Baselines.** We compare against the following baselines:

- **Naive async.** This strategy does not pay attention to the previous action chunk at all when generating a new one, naively switching chunks as soon as the new one is ready.
- **Bidirectional decoding (BID; [39]).** This strategy uses rejection sampling to keep continuity across chunks. We use a batch size of $N = 32$, mode size of $K = 3$, and a checkpoint trained for 8 epochs as the weak policy.
- **Temporal ensembling (TE; [68]).** This strategy involves keeping a buffer of predicted action chunks and executing an average of all actions predicted for a particular timestep.

> 💡 **Baselines 分析**:
> | 方法 | 原理 | 计算量 | 需要训练？ |
> |------|------|--------|-----------|
> | Naive async | 直接切换新 chunk | 1x | 否 |
> | BID | 采样多个 chunk，挑最连续的 | **64x**（32 strong + 32 weak） | 需要 weak model |
> | TE | 对重叠 action 取平均 | 1x | 否 |
> | **RTC** | Inpainting + soft masking | ~1.3x（反向传播） | 否 |
> 
> BID 的计算量是 RTC 的 ~50 倍，但效果还更差。这是 RTC 最有说服力的对比。

---

![Figure 5](../images/b01822a7ef5ca22d7ee4449fc1cdb742470a8d029ad55f3be5b09a2d79ae7e84.jpg)
*Figure 5: Top left: Kinetix 环境（绿色物体碰到蓝色物体即成功）。Bottom left: Execution horizon vs solve rate（固定 delay=1）。Right: Inference delay vs solve rate（固定 $s = \max(d, 1)$）。每个数据点 2048 trials，95% Wilson 置信区间。*

> 💡 **Figure 5 批读——仿真核心结果**:
> 
> **右图（delay vs solve rate）——最重要**:
> - **TE 全面垮掉**: 即使 $d=0$ 也很差，说明 Kinetix 的任务是多模态的（取平均 = 无效 action）
> - **Naive async**: $d=0$ 时还行，delay 增大后迅速下降
> - **BID**: 比 naive 好，但 delay 增大后也明显下降
> - **RTC**: 所有 delay 下都最优，且**下降幅度最小**——对延迟最鲁棒
> - **RTC soft > RTC hard**: Soft masking 在小 $d$ 时优势明显
> 
> **左图（execution horizon vs solve rate）**:
> - RTC 和 BID 是唯二能从更短 execution horizon 中获益的方法（越短 = 越频繁更新 = 更 closed-loop）
> - 其他方法缩短 execution horizon 反而变差（mode-jumping 更严重）

---

## 4.2 Real-World Results

Next, we deploy our full real-time chunking system to the real world. We use the π₀.₅ VLA [24] as our base policy, and evaluate RTC on a bimanual system with two 6-DoF arms and parallel jaw grippers. Unlike our simulated benchmark, the robots use position control, and so synchronous inference—stopping between chunks—is a reasonable default strategy, used in many prior works [5, 24, 31, 47]. Our goal is to improve upon synchronous inference in a combination of both performance and speed.

> 💡 **真实世界 vs 仿真的关键区别**:
> - 仿真用力控（force control）→ 停下来 = 失控
> - 真实用位控（position control）→ 停下来 = 保持位置（所以同步推理在真实世界是可行的 baseline）
> - 这意味着 RTC 在真实世界的竞争对手更强——同步推理至少能"停住等"

---

**Setup.** We use π₀.₅ ($H = 50$, $\Delta t = 20\text{ms}$) with $n = 5$ denoising steps, giving a model latency of 76ms for the baselines and 97ms for RTC. We use remote inference over LAN, which adds 10-20ms of latency, giving a starting inference delay around $d \approx 6$ for RTC. However, we would like to understand how the system behaves with higher inference latencies, simulating, e.g., scaling up the model size or running inference on a distant cloud server. Thus, we also evaluate all methods with +100ms and +200ms of injected latency, corresponding to $d \approx 11$ and $d \approx 16$, respectively.

> 💡 **真实世界延迟配置**:
> | 条件 | 总延迟 | $d$ (控制步) |
> |------|--------|-------------|
> | 基线 (LAN) | ~110ms | ~6 |
> | +100ms 注入 | ~210ms | ~11 |
> | +200ms 注入 | ~310ms | ~16 |
> 
> +200ms 对应 $d \approx 16$，相当于 prediction horizon $H=50$ 的 32%。这是非常极端的延迟。

---

**Tasks and scoring.** Each episode gets an integer score corresponding to how many substeps of the task it completed successfully. We evaluate the following tasks:

- **Light candle** (5 steps, 40s cutoff). Pick up a match and matchbox, strike the match, use it to light a candle, and drop it in a bowl.
- **Plug ethernet** (6 steps, 120s cutoff). Pick up the end of an ethernet cable, reorient it, plug it into a server rack, and repeat the process for the other end.
- **Make bed, mobile** (3 steps, 200s cutoff). Move the corner of a blanket and 2 pillows from the foot to the head of a bed.
- **Shirt folding** (1 step, 300s cutoff). Fold a shirt from a flattened position.
- **Batch folding** (4 steps, 300s cutoff). Take a varied, crumpled clothing item out of a bin, flatten it, fold it, then place it neatly on a pile.
- **Dishes in sink, mobile** (8 steps, 300s cutoff). Move 4 varied items from a counter into a sink.

> 💡 **任务设计分析**:
> - **精度敏感**: Light candle（点火柴需要精确的力和时序）、Plug ethernet（插口对准）
> - **长时间操作**: Batch folding（300s）、Dishes（300s，8 步）
> - **移动操作**: Make bed 和 Dishes 需要移动底座
> - **关键**: Light candle 是**唯一不允许重试**的任务（火柴点了就点了），最能体现 RTC 的优势
> 
> 评估规模: 6 tasks × 4 methods × ~10 trials × 多种延迟 = **480 episodes, 28 小时机器人时间**

---

**Baselines.**

- **Synchronous.** Default in prior work [5, 24, 31, 47]. Execute $s = 25$ actions then pause while generating next chunk.
- **TE, sparse.** Execute $s = 25$ actions while computing next chunk in parallel. Apply TE on overlapping steps.
- **TE, dense.** Run inference as often as possible ($s = d$). Always ≥2 overlapping chunks to ensemble.

We do not compare to BID [39] in the real world, as we found in simulation that it underperforms RTC while using significantly more compute—when applied to π₀.₅ with a batch size of 16, BID has 2.3 times the latency of our method (see A.3 for latency measurements).

> 💡 **BID 被排除的原因**: 仿真中已经证明 BID 效果更差且计算量更大（π₀.₅ 上 BID 延迟是 RTC 的 2.3 倍）。合理的决策。

---

![Figure 6](../images/57e3a1534497d9eef97a7d00e2502b04177edb490b4d65007daee287da3a3e83.jpg)
*Figure 6: Top: 各任务的 controller steps vs cumulative progress。Left: 时间 vs cumulative progress（所有任务汇总，X轴 log scale）。Right: Inference delay vs average throughput（任务完成比例/时间）。TE 在 +100ms 和 +200ms 延迟下振荡太大，触发机器人保护性停机。*

> 💡 **Figure 6 批读——真实世界核心结果**:
> 
> **右下图（delay vs throughput）——最关键**:
> - **RTC 在所有延迟下都最优**，且对延迟完全鲁棒（曲线几乎平坦！）
> - **Synchronous**: 随延迟线性下降（因为停顿时间越来越长）
> - **TE (两种)**: +100ms 和 +200ms 直接**无法运行**——振荡太大触发安全停机！
> 
> **上排（per-task progress curves）**:
> - **Light candle**: RTC 优势最大（唯一不允许重试的任务）。同步推理在延迟大时成功率明显下降
> - **Make bed**: RTC 也有明显优势——枕头操作是最难的部分
> - **其他任务**: RTC 一般更早完成（更少错误和重试），即使最终得分相似
> 
> **关键发现**: 即使去掉推理暂停时间（纯"有效运动时间"），RTC 完成任务仍然更快！说明 RTC 不只是"消除暂停"那么简单——**它的连续控制信号本身就比间断的更好**。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 仿真任务数 | 12 (Kinetix) |
| 真实任务数 | 6 (双臂) |
| 真实评估量 | 480 episodes, 28h |
| RTC 延迟 (真实) | 97ms model + 10-20ms network |
| 基线延迟 | 76ms model + 10-20ms network |
| TE 在高延迟下 | **无法运行**（触发安全停机） |

### 核心洞察
1. **RTC 是唯一对延迟完全鲁棒的方法**: 从 ~110ms 到 ~310ms 几乎无性能下降
2. **TE 在真实世界高延迟下直接失败**: 振荡触发保护性停机，这是此前未被报告的问题
3. **RTC 不只是更快，也更准**: 去掉暂停时间后仍然比 synchronous 快完成任务
4. **Light candle 是最好的 test case**: 高精度 + 不可重试 + 时间敏感，RTC 优势最明显
