# Conclusion + Appendix

---

## 6. Conclusions and Discussions

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks. Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **结论诚实地承认了局限**：只在 5 类任务上验证，泛化到更广泛任务还是未知数。
>
> 💡 **未来方向**：随着基础视频模型（如 Genie 3、Sora 类）越来越强，世界模型的物理保真度会进一步提升，VLAW 框架的天花板也会随之提高——这是一个随时间自动升值的方向。
>
> 💡 **局限性小结（作者未提但值得注意的）**：
> 1. **任务覆盖窄**：5 类任务都在 DROID 平台，场景单一（固定桌面，Franka 机械臂）
> 2. **奖励模型的召回问题**：P(yes)>0.8 阈值漏掉了约 55% 的真实成功轨迹（Table 3），大量好数据被浪费
> 3. **世界模型 grounding 的样本效率**：每个任务需要 50 个真实 rollout × 2 迭代 = 100 次真实部署，是否可以更少？
> 4. **跨任务世界模型的干涉问题**：5 个任务同时训练一个世界模型，任务之间的物理规律差异很大，是否存在负迁移？论文没有分析
> 5. **实验规模有限**：每个任务只评估 50 次，置信区间没有报告，0.92 vs 0.88 的差距统计显著吗？

---

## Appendix A: Relation to Regularized Reinforcement Learning (详细推导)

Under the regularized RL setting, the optimal improved policy admits a closed-form solution:

$$
\pi^\star(a \mid o) \propto \pi_\mathrm{ref}(a \mid o) \exp\left(\frac{A^{\pi_\mathrm{ref}}(o, a)}{\beta}\right)
$$

Since the target distribution $\pi^\star$ is generally not representable within a finite parametric policy class, policy improvement is performed via a projection step:

$$
\theta^\star = \arg\min_\theta \mathbb{E}_{o \sim \mathcal{D}} \left[ D(\pi^\star(\cdot \mid o), \pi_\theta(\cdot \mid o)) \right]
$$

**AWR for flow-matching policies.** In standard AWR (Peng et al., 2019), the divergence $D$ is the KL divergence, yielding a weighted log-likelihood objective. However, because our VLA policy uses flow-matching and does not provide explicit action likelihoods, we introduce a surrogate divergence:

$$
D_\mathrm{FM}(\pi^\star(\cdot \mid o), \pi_\theta(\cdot \mid o)) \triangleq \mathbb{E}_{a \sim \pi^\star(\cdot \mid o)} [\mathcal{L}_\mathrm{FM}(\theta; o, a)]
$$

This yields the projection step:

$$
\theta^\star \approx \arg\min_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[ w(o, a) \mathcal{L}_\mathrm{FM}(\theta; o, a) \right]
$$

where $w(o, a) \propto \exp\left(\frac{A^{\pi_\mathrm{ref}}(o, a)}{\beta}\right)$.

Setting $\gamma \to 1$ and assigning a large negative reward to failure trajectories, this reduces to Eq. 4.

> 💡 **推导的本质**：把 flow-matching SFT 套上 AWR 的外衣。关键近似有两个：
> 1. 用 flow-matching loss 替代 KL divergence（合理，因为 flow-matching loss 本质是学习目标分布）
> 2. 二值权重代替连续 advantage weight（粗糙，但 binary reward 场景下近似合理）
>
> 💡 **这个推导的价值**：不是为了严格证明，而是给出"为什么这样做有理论依据"的解释，让审稿人接受这个设计选择。实际上 Table 2 的结果才是真正的证明。

---

## Appendix B: Task Details

**Success Criteria:**

- **Stacking**: Block A stably placed on top of Block B, stack remains upright for a short holding period.
- **Open Book**: Front cover opened beyond a predefined angle and remains open at episode end.
- **Erase Marks**: All visible marker strokes removed from whiteboard at episode end.
- **Scooping**: At least a minimum amount of target object transferred into the bowl, majority inside the bowl (not spilled).
- **Drawing**: Single closed curve forming a visually complete circle, endpoints meet with small gap tolerance.

> 💡 **成功标准的设计**：都是 outcome-based（最终状态判断），不是 process-based（过程是否正确）。这跟奖励模型的评估方式一致——输入最终帧/轨迹视频判断是否成功。
>
> 💡 **Drawing 任务的难点**：要求"endpoints meet with small gap tolerance"——闭合圆，这对机器人来说需要全局规划意识，不只是局部运动控制。这也解释了为什么 base model 只有 0.22 的成功率。

---

## Appendix C: Reward Model Details

We use Qwen3-VL-4B-Instruct as the vision–language reward model. Each trajectory is temporally downsampled into a 16-frame video. We finetune the model for 200 steps with batch size 128.

We observe that directly prompting for binary yes/no is overly optimistic. Using P(yes) > 0.8 threshold substantially reduces false-positive trajectories.

### Table 3: Reward Model Confusion Matrices (n=40)

**Original (Direct Yes/No):**

|  | Pred: Success | Pred: Failure |
|--|--------------|--------------|
| GT: Success | 15 | 7 |
| GT: Failure | **8** | 10 |

**Ours (P(yes) > 0.8):**

|  | Pred: Success | Pred: Failure |
|--|--------------|--------------|
| GT: Success | 10 | 12 |
| GT: Failure | **2** | 16 |

> 💡 **数字背后的含义**：
> - Direct 方法：精度 65%（15/23），假阳率 44%（8/18）——接近一半的"成功"标签是错的，会严重污染训练数据
> - 阈值方法：精度 83%（10/12），假阳率 11%（2/18）——干净多了，但代价是漏掉了 12/22=55% 的真实成功案例
>
> 💡 **一个值得 follow-up 的方向**：能不能设计更好的 VLM 奖励模型，同时保持高精度和高召回？比如 chain-of-thought 推理、多帧对比、或者专门的机器人 reward VLM（RoboReward, Lee et al., 2026 已经在做这件事）。

---

## 总结：VLAW 的价值与局限

### 核心价值
| 维度 | 内容 |
|------|------|
| **问题** | 真实 rollout 太贵，现有世界模型物理保真度不够 |
| **方案** | 用少量 rollout 修正世界模型 → 大量合成数据 → 迭代提升 VLA |
| **亮点** | 在真实机器人接触丰富任务上验证，+39.2% 绝对提升 |
| **简洁性** | 没有复杂的 RL 算法，全程 SFT，适配 flow-matching 策略 |

### 主要局限
| 局限 | 程度 |
|------|------|
| 任务覆盖窄（5类，同一平台） | 中等 |
| 奖励模型召回低（55% 真阳被漏） | 值得改进 |
| 跨任务负迁移未分析 | 未知 |
| 实验统计严谨性（无置信区间） | 轻微 |

### 与 STAR-Pro 的关系
> 💡 STAR-Pro 如果需要在真实机器人上做 post-training，VLAW 的框架非常值得参考——特别是"用少量真实 rollout grounding 世界模型"这个核心思路。Ctrl-World 本身就是同一组人做的，VLAW 相当于给了一个完整的使用范式。
