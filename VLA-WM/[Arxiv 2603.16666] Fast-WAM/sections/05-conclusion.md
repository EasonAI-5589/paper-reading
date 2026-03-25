[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览

Conclusion 简洁有力：核心发现重述 + 一个未来方向。局限性在正文中未详细讨论，但值得补充分析。

---

## 📄 原文

> In this paper, we revisited a basic question in World Action Models: whether their gains come primarily from explicit future imagination at test time or from video modeling during training. To study this question, we introduced Fast-WAM, a WAM architecture that retains video co-training during training while skipping future prediction at inference time, enabling direct action generation from world-grounded latent representations.

> Across simulation benchmarks and real-world robotic tasks, Fast-WAM achieves strong performance without embodied pretraining while running in real time. More importantly, controlled comparisons show that Fast-WAM remains competitive with imagine-then-execute variants, whereas removing video co-training leads to a much larger degradation.

> These results suggest that the main value of video prediction in WAMs may lie more in learning better world representations during training than in generating future observations at test time.

> An important direction for future work is to study the effect of larger-scale pretraining data and model scaling on this design.

---

## 💡 局限性分析（论文未详细讨论，批读补充）

| 局限性 | 说明 |
|--------|------|
| **规模效应不明** | 在更大模型或更长时域上，推理时未来想象可能变得更重要 |
| **真实世界任务单一** | 只有毛巾折叠一个真实任务，更多样化的任务（接触丰富、高精度装配）可能有不同结论 |
| **$\lambda$ 敏感度未分析** | 视频联合训练权重 $\lambda$ 的选择对结果的影响完全未报告 |
| **长时域规划未涉及** | 论文只做 single action chunk generation，没有 auto-regressive rollout；长时域场景可能需要未来想象 |
| **与 LeWM 无直接对比** | 谢赛宁建议一起看，但两者用完全不同的技术路线（扩散模型 vs JEPA），缺少实验对比 |
| **措辞谨慎** | 用 "may lie" 和 "suggest" 而非强 claim——合理但也留有余地 |

---

## 💡 总体评价

**评分: 8.5/10**

### 优点

1. **问题选得好**: 简单、基本、重要，但此前被忽视。WAM 社区需要这种"退一步想清楚"的工作
2. **实验设计精巧**: 控制变量实验是真正的 apple-to-apple comparison，在同一框架内隔离单一变量，说服力极强
3. **实际性能强**: 无预训练达到甚至超过有预训练的 SOTA（Motus +4%, π0.5 +12%）
4. **实用价值高**: 190ms 实时推理，是唯一满足真实机器人闭环控制要求的 WAM 方案
5. **论文写作清晰**: 问题-方法-实验的逻辑链条紧密，每个实验都直接服务于核心问题

### 不足

1. 真实世界任务只有一个（毛巾折叠）
2. 缺少 $\lambda$ 消融和更多超参数分析
3. 长时域/多步 rollout 场景未涉及
4. 没有可视化分析 latent world representation 的质量差异

### 对具身智能领域的影响

这篇论文可能改变 WAM 的研究重心——从"怎么更好地想象未来"转向"怎么更好地通过视频训练学表征"。与 LeWM 的发现互相印证：

| | Fast-WAM | LeWM |
|---|---|---|
| **解决的痛点** | WAM 推理太慢 | JEPA 训练太难 |
| **核心洞见** | 视频预测的价值在训练不在推理 | 轻量 JEPA 也能学到好的世界表征 |
| **共同指向** | **不需要暴力堆大模型，高效的世界建模训练 + 轻量推理就足够** |

→ 具身智能正在从"大力出奇迹"走向"聪明地训练 + 高效地推理"。
