[← 返回 README](../README.md)

# Appendix

## 📌 预览
附录包含两部分：A 为定性示例（3D 空间推理 + 时间价值估计的可视化），B 为 Bounded Global Progress 的数学证明。

---

## A Qualitative Examples

This section provides a comprehensive set of qualitative examples that illustrate the capabilities of RoboBrain 2.5 in various embodied AI tasks. As Capabilities like pointing, affordance, planning, etc. are similar to those shown in RoboBrain 2.0 [72], the examples in this section ONLY demonstrate the model's proficiency in 3D spatial reasoning, temporal value estimation, showcasing its potential for real-world applications.

### A.1 Examples on 3D Spatial Reasoning

> 💡 **A.1 概览**: 3D 空间推理的定性示例聚焦三个方面：细粒度空间约束遵循、多步组合推理、跨环境泛化。

**Compliance with Fine-Grained Spatial Constraints.** RoboBrain 2.5 excels at interpreting spatially constrained instructions and generating accurate 3D manipulation traces that adhere to both relative positional requirements and metric constraints.

**Multi-Step Compositional Reasoning.** Complex manipulation tasks often demand multi-step reasoning to decompose high-level goals into executable sub-tasks.

**Generalization Across Environments and Objects.** RoboBrain 2.5's 3D spatial reasoning generalizes to diverse indoor settings and object categories.

---

![](../images/3be0c84268e9f674948b79aca955c9d3ba77f02cd97a07ea6986e2f78690797b.jpg)

![](../images/b070ffe31754103e4bd2c3b1fb1d70562df5cb8dcd345f077fc051b4f84498e5.jpg)

![](../images/06b19e09bfe64bce36fc302c480f188d8c91f1942584290db9b1cc982906534c.jpg)

![](../images/6b7d4bb26e2645fb3b3e3c76f866beb67d6cf76282e8dc12c6b364103fd5d0d2.jpg)

![](../images/cf450369b0d414da356df1d457dae1167d1ec923e562996af39d1884fa966247.jpg)

![](../images/fc98d5da2748b8830218456c8293d473c8de37c337e33cb6b4b6784156f6e11d.jpg)

![](../images/b714f08e0ec3287f5675a1b8a8be2a5b4a53273e3a9e22268a06520a4b6be19a.jpg)

![](../images/f2d396f6d134362580d038a5f3e995b21cf4d7f9b51dbff15b0ff98f16b171a4.jpg)
*Figure 3: Visualization of TraceSpatial-Bench Rollouts and RoboBrain 2.5's Predicted Traces. The red mask marks the ground-truth starting point, the purple 3D bounding box represents the ground-truth endpoint, and the 2D projection of RoboBrain 2.5's predicted 3D spatial trace is displayed.*

> 💡 **Figure 3 批读**: 展示了 Reasoning Step 2-6 的不同复杂度任务。红色 mask = GT 起点，紫色 3D bbox = GT 终点，彩色线 = 模型预测的 3D trace 投影。

---

![](../images/6b46f8f8fd3f931f36e7939183961afd3acb6d8a089cdc82635dc2500b5b9bdf.jpg)

![](../images/05c53ddd0d3b301fa604a43a18753374b4173ad831464a9f4c13013b9a8cae65.jpg)

![](../images/3ca57d39d1e76461ebed194a194a2a6eefee7eaa23665282094c1a1e4eba074b.jpg)

![](../images/308215aaba953c9bd60e0428eaaf60590eeef519d9c53a361fd0e1510aeb6378.jpg)

![](../images/2bf0801e689e718cdc608f3881bb055167b2e065a0f596e9723bb241d5c566e0.jpg)

![](../images/ad60ea47da0eded32551c4ea22b855714ad2445b4092343e49d819a1b02b0b16.jpg)

![](../images/754fbd00ce8735ac25f350cc911096c63485fea6deb98114fe15af06dff5a2f1.jpg)

![](../images/6bcbe528054f7f53dc729c3669fbcd060bffa2a796aadb9842acc0baa9bfbe30.jpg)

![](../images/97af2e5f03ef136262328f3c306292094b5761bf3f229abea408fc5f3b6457a5.jpg)

![](../images/57f8808e3423b0da2c888a9a38fa667ba9b219635343b20b1c77c6d0bdbf1140.jpg)
*Figure 4: Visualization of TraceSpatial-Bench Rollouts and RoboBrain 2.5's Predicted Traces.*

---

![](../images/ac89c783c9297cb2dff50dd52e26337ebbc74f606d3fbe61843c8f5975193c48.jpg)

![](../images/4bc7a73a71518a010b611f0575ad55956b0cb17de5dfb9895ee5dffac5c581a3.jpg)

![](../images/5adc3f23011e7ab4ff7205dcaca1e4070a107d456e0ff878f275337918de7cc5.jpg)

![](../images/6e019afe274b9f26b2a8fe998384600be96248fc7b1401e8b0bd7306c0c511e1.jpg)

![](../images/756c7dbd57be34a8610c6fb3e09fe23abc16a6b1f7716c7fd3a052a84878726a.jpg)

![](../images/2af17d5a5087b6af5d1f2284e88d5e85e2ce3cff9ba806556a028fed851d280e.jpg)

![](../images/5a42e0630382ccef4cf8d38ead688974d30742f2b532092115cac46210bef2fe.jpg)

![](../images/d957e6a0c42511f8f68f85dcf334e2eb865d786e78539ce49f54c4d0ff805b34.jpg)

![](../images/581cee7541ba3a9da24be935c846453d8dd6f99dbe73413da0ee155862ddb4bd.jpg)
*Figure 5: Visualization of TraceSpatial-Bench Rollouts and RoboBrain 2.5's Predicted Traces.*

---

**Application on RoboTwin 2.0**

![](../images/43b1b5e7385eef573c2e27ffff5559556a7c36abe3edd081ee8014f20208faf0.jpg)
*Figure 6: RoboTwin 2.0 Rollouts — AgiLex Dual-Arm tasks: Click Bell; Click Alarm clock; Blocks Ranking.*

> 💡 **Figure 6 批读**: 展示细粒度空间辨别——在多个干扰物中识别特定目标（如"离牛奶盒最近的"、"第二大的"）。

![](../images/7972b79e6a0d3c3453a26e937a83aa7b850a0070d65c93aa97b04ea0efb725a0.jpg)
*Figure 7: RoboTwin 2.0 Rollouts — AgiLex Dual-Arm tasks: Handover Block; Handover Mic; Hanging Mug; Move Can Pot.*

![](../images/317544704e9e57eb7915cb839965f493cfcafa3233d251325218ab5ab646248a.jpg)
*Figure 8: RoboTwin 2.0 Rollouts — AgiLex Dual-Arm tasks: Move Playingcard Away; Move Stapler Pad; Open Laptop; Place A2B Left.*

![](../images/0704b29b58a9069b05a5f6bba155214ac45d2554a02a69af8577e165093b8cac.jpg)
*Figure 9: RoboTwin 2.0 Rollouts — AgiLex Dual-Arm tasks: Place A2B Right; Place Bread Basket; Place Bread Skillet; Place Burger Fries.*

---

### A.2 Examples on Temporal Value Estimation

> 💡 **A.2 概览**: 时间价值估计的定性示例展示三方面：多任务泛化、时间间隔鲁棒性、真实 RL 部署。

**Dense Value Predictions on Diverse Tasks.**

![](../images/1d665a3f5afcf362aef6c5ab6293a4de41e475eef9f7a3f8e627267f9a5773fb.jpg)
*Figure 10: RoboBrain 2.5 Progress Predictions across Diverse Tasks. Hop (instantaneous change) and accumulated Progress on unseen validation tasks.*

> 💡 **Figure 10 批读**: 多种任务（叠碗、折裤子、清桌子等）的 Hop 和 Progress 曲线。成功轨迹中 Hop 持续为正，Progress 单调递增。

---

**Robustness to Temporal Intervals.**

![](../images/c71f75a2586ddd4afb586f3302cce517caff7f9c195a33b847569d66296a7a19.jpg)
*Figure 11: Progress Estimation Consistency across Sampling Intervals (Δt = 10, 25, 50, 100 frames).*

> 💡 **Figure 11 批读**: 不同采样率下 Progress 曲线高度重合——模型学会了将物理进度与时间间隔解耦。Δt 越大，单步 Hop 越大，但累积 Progress 不变。

---

**Visualization of Different Progress Estimation Modes.**

![](../images/fabc7dae245a5feb06afca3838bf5e54957a9780d53afa4997ac258346fcff49.jpg)
*Figure 12: RoboBrain 2.5 Progress Predictions across three modes: incremental, forward-anchored, backward-anchored.*

> 💡 **Figure 12 批读**: 三种估计模式在未见验证任务上都产生一致的单调 Progress 曲线，验证了多视角融合策略的有效性。

---

**Real-World RL Rollout Visualization.**

![](../images/02d175eee6567a3c483903c2618625d166e016cca4490ca61fd9818315e2c11a.jpg)
*Figure 13: Robustness to Artificial Disturbance during Real-World Execution. (a) Human interference shifts target. (b) Robot misses → Progress drops. (c-d) Recovery. (e-f) Successful insertion. Policy trained ~20 min, >95% success rate.*

> 💡 **Figure 13 批读**:
> - 这是整篇论文最令人印象深刻的实验——人为干扰下的闭环恢复
> - 人手移动目标 → 机器人错过 → **Progress 立刻大幅下降**（红点）→ 策略自动调整 → 成功完成
> - 只训练了 ~20 分钟就达到 >95% 成功率，说明 Dense Value 作为 RL reward 的高效性
> - 这证明了 RoboBrain 2.5 作为 reward model 在闭环 RL 中的实际价值

---

## B Proof of Bounded Global Progress

> 💡 **B 概览**: 证明 hop-based 迭代更新保证 $\Phi^*(s) \in [0,1]$。用数学归纳法，分正 hop 和负 hop 两种情况。

The proof shows that iteratively applying predicted relative progress hops guarantees $\Phi^{\star}(s) \in [0, 1]$, provided $\Phi^{\star}(s_0) = 0$ and $H \in [-1, 1]$.

**Update rule:**

$$\Phi^{\star}(s_{t}) = \begin{cases} \Phi^{\star}(s_{t-1}) + H \cdot [1 - \Phi^{\star}(s_{t-1})] & \text{if } H \geq 0 \\ \Phi^{\star}(s_{t-1}) + H \cdot \Phi^{\star}(s_{t-1}) & \text{if } H < 0 \end{cases}$$

**Base Case:** $\Phi^{\star}(s_0) = 0 \in [0, 1]$. ✓

**Inductive Step (assume $G = \Phi^{\star}(s_{t-1}) \in [0,1]$):**

- **Case 1 ($H \geq 0$):** $\Phi^{\star}(s_t) = G + H(1-G) = H + G(1-H)$. Since $G, H \in [0,1]$: lower bound $\geq 0$, upper bound $\leq H + (1-H) = 1$. ✓
- **Case 2 ($H < 0$):** $\Phi^{\star}(s_t) = G(1+H)$. Since $H \in [-1,0)$: $(1+H) \in [0,1)$, so $G(1+H) \in [0, 1)$. ✓

> 💡 **证明批读**: 
> - 正 hop 情况：$\Phi^*(s_t) = H + G(1-H)$ 是 $H$ 和 $G$ 的凸组合，自然有界
> - 负 hop 情况：$\Phi^*(s_t) = G(1+H)$ 是 $G$ 乘以 $[0,1)$ 中的因子，必然缩小
> - 直觉：正向前进永远无法超过 1（因为归一化到剩余距离），后退永远无法低于 0（因为归一化到已走距离）
> - 这个性质是 hop formulation 相比朴素 $\Delta\Phi$ 的核心优势

---

## 🔖 Section 总结

### 核心洞察
1. Figure 13（人为干扰恢复）是最有说服力的实验——证明了 Dense Value 在真实闭环 RL 中的价值
2. 不同采样率下 Progress 曲线高度一致（Figure 11），说明模型学会了物理进度而非帧率
3. Bounded progress 证明虽然简单，但保证了实用中不会出现值爆炸
