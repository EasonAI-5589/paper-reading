# VLAW 批读笔记 · Conclusions & Appendix

---

## 6. Conclusions and Discussions

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks.

Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **局限性总结**（作者自述 + 补充）：
>
> **作者提到的**：
> - 仅在 5 类任务上验证，泛化到更多任务未知
>
> **作者没提但值得注意的**：
> 1. **计算开销未报告**：50K steps world model fine-tune 和生成 2500 条合成轨迹的实际时间是多少？对于研究者复现非常重要。
> 2. **World model 的误差累积**：closed-loop rollout 跑 20 秒已经不错，但更复杂的长任务（比如多步装配）会怎样？
> 3. **Reward model 的准确性上限**：fine-tuned Qwen3-VL 在 Appendix C 里的结果仍有 FN=12（22 个成功被判为失败），说明还有大量有效合成数据被丢弃——这是一个潜在的性能天花板。
> 4. **迭代次数**：只做了 2 次迭代，是否会继续改进？两次迭代的曲线都在上升，没有看到收敛的迹象。
> 5. **任务间的迁移**：5 类任务是联合训练的（multi-task），不同任务的 world model 和 policy 是否相互干扰？没有单任务 baseline 对比。

---

## Appendix A: Relation to Regularized Reinforcement Learning

Under the regularized RL setting, the optimal improved policy:
$$\pi^*(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(o,a)}{\beta}\right)$$

Since flow-matching VLA policies don't provide explicit log-likelihood, we define a surrogate FM divergence:
$$D_{\mathrm{FM}}(\pi^*(\cdot \mid o), \pi_\theta(\cdot \mid o)) \triangleq \mathbb{E}_{a \sim \pi^*(\cdot \mid o)} \left[ \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

The projection step then becomes:
$$\theta^* = \arg\min_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[ w(o,a) \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

Setting $\gamma \to 1$ and binary reward → reduces to Eq. 4 (the actual training objective).

> 💡 **这个推导的价值与局限**：
> - **价值**：把 binary-filtered BC 纳入 AWR（Peng et al. 2019）框架，说明这不只是"拍脑袋的 heuristic"，有理论基础
> - **局限**：
>   1. 用 binary weight（0/1）近似 advantage weight（$\exp(A/\beta)$）是一个粗糙近似——理论上用连续的 advantage 加权会更好
>   2. $\gamma \to 1$ 的假设意味着不 discount 未来奖励，对于长 horizon 任务可能不合适
>   3. 离线数据近似 on-policy 采样是 offline RL 的标准近似，这里的处理与 AWR 原论文一致

---

## Appendix B: Task Details

**Success Criteria（明确定义）：**
- **Stacking**: 块 A 稳定放在块 B 上，保持一段时间不倒
- **Open Book**: 书皮打开超过预设角度并保持
- **Erase Marks**: 结束时白板上无可见记号
- **Scooping**: 至少一定量的目标物进入碗中，且大部分在碗内
- **Drawing**: 产生一条视觉上完整的闭合圆（端点误差在容忍范围内）

> 💡 **Success criteria 设计得很合理**：都是基于最终状态的判断（outcome-based），不需要实时 reward，符合现实中人工判断的方式。但"至少一定量"和"容忍范围内"这类描述不够精确，可能导致不同评估者之间的主观差异。

**评估细节：** 每次迭代后评估 50 次（与 real rollout 数量相同）；DSRL 只评估 10 次（"too time-consuming"）。

> 💡 **DSRL 只评 10 次这件事值得关注**：DSRL 在 50 轮 online rollout 期间同步更新 policy，所以每次都需要重置环境，确实费时。但 10 次评估的统计显著性很低（标准误差大），DSRL 的结果基本无法与其他方法做可靠比较。

---

## Appendix C: Reward Model Details

We use Qwen3-VL-4B-Instruct. Each trajectory is downsampled into a 16-frame video. Fine-tune for 200 steps with batch size 128.

**Threshold Strategy:** Set threshold $\alpha=0.8$ on P(yes). Naive yes/no output is too optimistic.

**Table 3: Confusion Matrix Comparison（manually labeled 40 trajectories）:**

| | Direct Yes/No Output | | Threshold p(yes) > 0.8 | |
|---|---|---|---|---|
| | Pred Success | Pred Failure | Pred Success | Pred Failure |
| **GT Success** | 15 | 7 | 10 | 12 |
| **GT Failure** | **8** | 10 | **2** | 16 |

> 💡 **数字解读**：
> - **Threshold 的主要效果**：FP 从 8 降到 2（↓75%）——大幅减少把失败轨迹误判为成功的情况。这直接减少了污染 policy 训练的"伪成功"合成数据。
> - **代价**：FN 从 7 升到 12——更多真正成功的轨迹被保守地判为失败，导致有用数据损失。在 40 个样本里，GT Success 只有 22 个，threshold 后只保留了 10 个（45%的召回率），丢弃了过半的成功数据。
> - **这里有一个有趣的 tradeoff**：更高的 threshold → 更纯净的训练数据 → 每条数据质量更高，但数量更少。0.8 可能不是最优，可以做 threshold 敏感性分析（论文未做）。
> - **16 帧 downsampling**：原始轨迹可能有几百帧，压缩到 16 帧可能丢失关键交互时刻（接触瞬间往往就几帧）。这可能是 FN 高的原因之一。

---

## 总体评价

### 优点

1. **问题定位准确**：world model 的 over-optimism 和 physical fidelity 问题是真实存在的，解决方案（用 online rollout fine-tune）简单有效
2. **方法简洁**：没有特殊的 architecture 设计，核心贡献在于训练数据和 pipeline 设计
3. **实验扎实**：在真实机器人上做实验，5 类 contact-rich 任务，每类 50 次评估，结果有说服力
4. **理论联系**：将方法与 AWR / regularized RL 框架联系起来，增强了方法的可解释性

### 不足

1. **计算开销未报告**：复现成本不清楚
2. **关键消融缺失**：没有"直接用 pretrained world model 生成数据"的消融，难以量化 world model fine-tune 这一步单独的贡献
3. **DSRL 评估不足**（10次 vs 50次），对比不公平
4. **只做 2 次迭代**：不清楚更多迭代是否继续改进还是会收敛/过拟合
5. **Reward model 准确性有限**：threshold=0.8 下召回率仅约 45%，有大量有效合成数据被丢弃

### 与相关工作的关系

| 工作 | 关系 |
|------|------|
| π₀.₅ (Physical Intelligence) | 本文使用的 base VLA model |
| Ctrl-World (Guo et al. 2025a) | 本文使用的 base world model（第一作者同一团队） |
| π₀.₆* (Physical Intelligence) | 同类 online RL for VLA，但不用 world model；advantage-conditioned SFT |
| DayDreamer (Wu et al. 2023) | 先驱工作，real-world MBRL，但模型容量小、单任务 |
| WMPO (Zhu et al. 2025) | 同期工作，也是 world model + VLA，但方法不同 |
| World-Gymnast (Sharma et al. 2026) | 同期工作，在 world model 里做 RL 训练 robot |
