# VLAW 批读笔记 · Appendix

---

## Appendix A: Relation to Regularized Reinforcement Learning

Under regularized RL, the optimal improved policy:

$$\pi^*(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(o,a)}{\beta}\right)$$

因为 flow-matching VLA 没有显式 log-likelihood，定义 surrogate divergence：

$$D_{\mathrm{FM}}(\pi^*, \pi_\theta) \triangleq \mathbb{E}_{a \sim \pi^*(\cdot \mid o)} \left[ \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

Projection step：

$$\theta^* = \arg\min_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[ w(o,a) \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

令 $\gamma \to 1$、binary reward → 化简为实际训练目标 Eq.4。

> 💡 **推导价值**：把"在成功轨迹上做 BC"这一 heuristic 与 AWR（Peng et al. 2019）正式联系起来，有理论依据。
>
> **三个近似叠加，保证较弱**：
> 1. binary weight（0/1）≈ advantage weight（exp(A/β)）：粗糙，理论上连续加权更优
> 2. $\gamma \to 1$（不 discount 未来奖励）：长 horizon 任务可能不合适
> 3. offline 数据 ≈ on-policy 采样：标准 offline RL 近似
>
> 本节更像"事后理论解释"而非"从理论推导方法"，与其说是贡献，不如说是增强可读性的工具。

---

## Appendix B: Task Details

**成功判断标准（明确定义）：**

| 任务 | 成功条件 |
|------|---------|
| Stacking | 块 A 稳定放在块 B 上，保持一段时间不倒 |
| Open Book | 书皮打开超过预设角度并保持 |
| Erase Marks | 结束时白板上无可见记号 |
| Scooping | 至少一定量目标物进入碗中，大部分在碗内（非洒出） |
| Drawing | 产生视觉上完整的闭合圆，端点误差在容忍范围内 |

> 💡 **成功标准设计合理**：都是 outcome-based（基于最终状态判断），不需要实时 reward，符合真实场景的人工判断方式。但"至少一定量"、"容忍范围内"等描述偏定性，不同评估者可能有主观差异，理想情况下应给出量化阈值。

**评估细节：** 所有方法评估 50 次/task（DSRL 只评 10 次，"too time-consuming"）。

> 💡 **DSRL 评估不足的问题**：50 次评估标准误差约 ±7%，10 次约 ±15%。DSRL 的结果基本无法与其他方法做可靠的统计比较，这是实验设计的瑕疵。

---

## Appendix C: Reward Model Details

**设置：** Qwen3-VL-4B-Instruct，每条轨迹下采样为 16 帧视频。Fine-tune 200 steps（bs=128）。Threshold $\alpha = 0.8$。

**Table 3: 混淆矩阵对比（人工标注 40 条轨迹）：**

| | 直接 Yes/No 输出 | | Threshold p(yes) > 0.8 | |
|---|---|---|---|---|
| | 预测成功 | 预测失败 | 预测成功 | 预测失败 |
| **GT 成功** | 15 | 7 | 10 | 12 |
| **GT 失败** | **8** | 10 | **2** | 16 |

> 💡 **数字解读：**
> - **Threshold 的主要效果**：FP 从 8 降到 2（↓75%）——大幅减少把失败轨迹误判为成功的情况，直接减少了污染 policy 训练的"伪成功"合成数据
> - **代价**：FN 从 7 升到 12——GT Success 共 22 个，threshold 后只保留 10 个（**召回率仅 45%**），超过一半的有效成功数据被丢弃
>
> **潜在问题**：召回率 45% 意味着大量有效合成数据被白白丢弃。0.8 可能不是最优阈值，precision-recall tradeoff 可以做更细致的分析（论文未做）。
>
> **16 帧下采样的风险**：原始轨迹可能有几百帧，压缩到 16 帧可能错过关键交互时刻（接触往往就几帧），这可能是 FN 偏高的原因之一。
