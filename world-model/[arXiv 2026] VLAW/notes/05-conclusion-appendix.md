# Conclusion + Appendix

---

## 6. Conclusions and Discussions

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks. Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **作者自述的局限**：5 个任务类别远不够证明 generalization。关键问题是：世界模型只在收集了 online rollout 的任务上变好，换个新任务（没有 online data）效果未知。
>
> 💡 **更深层的思考**：
>
> 1. **世界模型的泛化 vs. 特化**：本文的做法本质是 task-specific fine-tuning——需要每个任务都收集真实 rollout 来修正世界模型。如果有一个足够好的基础世界模型（比如 Genie 3），是否还需要 fine-tuning？这是未来方向。
>
> 2. **真实 rollout 的最低需求**：每个任务 50 个 rollout 是否是最优数量？有没有 sample efficiency 的下界？论文没做这个 ablation。
>
> 3. **迭代次数**：只做了 2 次迭代。3 次、5 次会继续提升吗？还是会饱和？Figure 7 暗示提升在放缓（Ours-1→Ours-2 的增量在不同任务上不均匀）。

---

## Appendix A: Relation to Regularized RL (详细推导)

The derivation follows AWR (Peng et al., 2019):

1. Start from regularized RL objective: maximize reward while staying close to reference policy
2. Optimal policy: $\pi^\star(a|o) \propto \pi_\mathrm{ref}(a|o) \exp(A^{\pi_\mathrm{ref}}(o,a)/\beta)$
3. Key innovation: replace KL-based projection with flow-matching surrogate divergence:
   $$D_\mathrm{FM}(\pi^\star, \pi_\theta) \triangleq \mathbb{E}_{a \sim \pi^\star}[\mathcal{L}_\mathrm{FM}(\theta; o, a)]$$
4. With $\gamma \to 1$ and binary rewards → $w(o,a) \in \{0,1\}$ → recovers Eq. 4

> 💡 **理论贡献评估**：
> - 不是新理论，是 AWR 的 flow-matching 适配版
> - 关键近似：连续 advantage weight → 二值权重；这在 binary reward 设定下损失最小
> - 实际意义：给"只在成功轨迹上做 SFT"提供了 RL 理论 justification
> - 但不要过度解读——核心是工程方案有效，理论是包装

---

## Appendix B: Task Details

**Success Criteria:**
- **Stacking**: Block A stably on top of B, stack upright for holding period
- **Open Book**: Cover opened beyond predefined angle, stays open
- **Erase Marks**: No detectable marks remain on whiteboard
- **Scooping**: Minimum amount transferred to bowl, majority inside bowl
- **Drawing**: Single closed curve forming visually complete circle

> 💡 **评估标准合理**：都是 outcome-based（看最终状态），不是 process-based。每个任务评估 50 次。DSRL 只评估 10 次（太慢了）——这使 DSRL 的结果统计功效较低，但整体趋势还是清晰的。

---

## Appendix C: Reward Model Details

- Base model: Qwen3-VL-4B-Instruct
- Input: 16-frame downsampled trajectory video
- Fine-tuning: 200 steps, batch size 128
- Threshold: P('yes') > 0.8

### Table 3: Reward Model Confusion Matrix (40 manually labeled trajectories)

**Direct Yes/No:**
|  | Predicted Success | Predicted Failure |
|--|---|---|
| GT Success | 15 | 7 |
| GT Failure | **8** | 10 |

**Threshold P(yes) > 0.8:**
|  | Predicted Success | Predicted Failure |
|--|---|---|
| GT Success | 10 | 12 |
| GT Failure | **2** | 16 |

> 💡 **奖励模型分析**：
> - Threshold 方法把 FP 从 8 降到 2——在机器人 RL 里，false positive 意味着"给错误行为正激励"，这是致命的
> - 代价是 FN=12（30% 的成功轨迹被漏掉），但在合成数据量足够大（500/任务）的情况下，精确度比召回率重要
> - **40 个样本的评估集太小**——统计显著性存疑，但趋势明确
>
> 💡 **没讨论的问题**：
> - 奖励模型在合成轨迹上的表现如何？合成视频的视觉质量不如真实视频，可能导致 domain gap
> - 如果世界模型产生了 out-of-distribution 的视觉伪影，奖励模型能否识别？
> - 这两个问题是 VLAW pipeline 的潜在 failure mode

---

## 总体评价

### 优点
1. **问题定位精准**：世界模型的过度乐观偏差是真实存在的问题，诊断到位
2. **方案简洁有效**：没有复杂的 RL 算法，就是 SFT on filtered data，但通过世界模型放大了数据量
3. **真实机器人实验**：5 个接触丰富任务，不是仿真里的简单任务
4. **理论与工程兼顾**：AWR for flow-matching 的理论解释合理

### 不足
1. **迭代次数太少**（只有 2 次）：无法判断是否会持续改善或饱和
2. **任务数量有限**（5 类）：泛化性证据不足
3. **计算成本未讨论**：50K steps 微调世界模型需要多少 GPU 时间？生成 500 个轨迹需要多久？
4. **世界模型误差传播分析缺失**：合成数据里的物理错误如何影响策略？有没有负面案例？
5. **奖励模型评估集太小**（40 个样本）
6. **没有与 π₀.₆\* 直接对比**——它们的 baseline 是 Filtered BC 和 DSRL，而不是同期最强的 VLA post-training 方法

### 与 STAR-Pro 的潜在关联
这篇论文与 MLLM Token Compression 方向不直接相关，但：
- 如果 STAR-Pro 涉及 robot learning / embodied AI，世界模型作为数据放大器的思路可以借鉴
- 奖励模型（VLM-as-judge）的 threshold 策略是通用技巧
