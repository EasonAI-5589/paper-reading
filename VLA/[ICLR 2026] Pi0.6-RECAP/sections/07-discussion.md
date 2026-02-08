[← 返回 README](../README.md)

# VII. Discussion and Future Work

## 📌 预览
总结 RECAP 的贡献，指出三个未来方向：(1) 自动化 reward + reset (2) 更好的 exploration (3) fully online RL。

---

Training policies that can achieve the same robustness, speed, and fluency on real-world tasks as people presents a major challenge in robotic learning. In this paper, we discussed how learning from experience, through a combination of DAgger-style coaching and RL, can begin to address this challenge. We describe RECAP, a method for training VLAs with autonomous trials, reward feedback, and human interventions, and present results for a model trained with RECAP, $\pi_{0.6}^{*}$ on a set of realistic tasks: making espresso drinks, folding diverse laundry, and assembling boxes. At the core of RECAP is an RL method that is well-suited for scalable training of VLA policies, using advantage conditioning for policy extraction with value functions. The data for this RL method is collected with a combination of autonomous rollouts and human interventions, correcting mistakes with interventions while finetuning the details of the behavior on autonomous data. Our experiments show that RECAP can improve both the success rate and throughput of the VLA, more than doubling the throughput on some of the harder tasks, and decreasing the number of failures by roughly $2\times$.

> 💡 **总结批读**:
> - RECAP = DAgger-style coaching + offline RL + advantage conditioning
> - 关键成功因素：advantage conditioning 使得 RL 可以 scale 到大 VLA
> - 实际效果：throughput 2×、failure rate 0.5×

---

There are several directions for improvement with RECAP. First, our system is not fully autonomous: it relies on human labeling and effort for reward feedback, interventions, and episode resets. A number of prior works have explored ways to automate these components [84, 85], and VLAs offer new ways to provide for more automated data collection, for example by using high-level policies [86] to reason through resetting the scene. Second, our system is relatively naïve in how it approaches exploration: exploration is largely greedy, relying on stochasticity in the policy and human interventions to explore new solutions. This is reasonable when the initial imitation learning policy already takes reasonable actions, but there is plenty of room for improvement with more sophisticated exploration methods. Lastly, RECAP performs iterated "offline" updates (i.e., it collects a batch of data, retrains the model, and repeats), rather than running a fully online RL loop where the policy and value function are updated in real time as data is collected. We make this decision out of convenience, but extending our approach into a fully concurrent online RL framework is a promising direction for future work.

> 💡 **三个局限/未来方向**:
> 1. **Not fully autonomous**: 需要人类标注 reward、提供 interventions、reset 环境
>    - 解决方向：VLA-based auto-reset、automated reward (VLM judge?)
> 2. **Naive exploration**: 完全依赖 policy stochasticity + interventions
>    - 解决方向：curiosity-driven exploration、goal-conditioned exploration
> 3. **Batch offline updates**: 收集一批 → 重训 → 再收集
>    - 解决方向：fully online RL（边收集边更新）
>    - 这对大模型来说是计算挑战

---

More broadly, training VLAs with RL is perhaps the most direct path to get to performance levels that are adequate for real-world use cases. RL with VLAs presents a number of challenges, from the difficulty of large-scale RL training of high capacity models to sample complexity, autonomy, and delayed feedback. While existing RL frameworks designed for smaller-scale systems or "virtual" domains such as LLMs can provide a good starting point, more research will be needed to make RL a practical tool for VLA training. We hope that our work represents a meaningful step in this direction.

> 💡 **大局观**:
> - VLA + RL 是通向"实际可用"机器人的最直接路径
> - LLM 的 RLHF 经验可以借鉴，但机器人有独特挑战（物理世界、延迟反馈、安全性）
> - RECAP 是这个方向的"有意义的一步"

---

## 🔖 Section 总结

### 核心洞察
1. RECAP 的局限都是可以被解决的工程问题，不是理论瓶颈
2. Fully online RL for VLAs 是最有前景但最有挑战的方向
3. 自动化 reward 和 reset 可以大幅降低人力成本
