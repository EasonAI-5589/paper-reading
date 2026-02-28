[← 返回 README](../README.md)

# 6. Conclusions and Discussions

> 来源: VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model (arXiv 2602.12063)

---

## 📄 原文

> 💡 **Section 概览**: 很短的结论段，诚实承认了两个局限（5 类任务、世界模型仅对收集了 online 数据的任务高保真），并展望了世界模型作为通用机器人策略训练范式的未来潜力。

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks. Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **结论段批读**：
>
> **作者主动承认的局限**：
> 1. 只评估 5 类任务（评估规模偏小）
> 2. 世界模型的高保真度仅限于「收集了在线数据的任务」（不能开箱即用于新任务）
>
> **作者没说的局限**（更值得关注）：
> ```
> 1. 只迭代 2 次
>    └── 未展示收敛曲线，不知道几次后会饱和或发散
>
> 2. 无 π₀.₆* 直接对比
>    └── Physical Intelligence 最新方法，与 VLAW 定位最相似
>       不对比 = 无法判断引入世界模型是否「值得」额外复杂度
>
> 3. 计算成本未报告
>    └── 50K steps 世界模型微调 + 500 条轨迹生成的 GPU 时间？
>
> 4. 只有 DROID 平台
>    └── 一种机械臂 + 一套相机配置，硬件泛化性未知
>
> 5. 失败案例未分析
>    └── 哪些场景下 VLAW 失败了？为什么？论文没有 failure case 分析
> ```
>
> **展望的可信度**：「base video model 进步 + 大规模机器人交互数据」是合理预测，VLAW 这套框架确实可以随着 Ctrl-World 等基础世界模型的能力提升而受益。

---

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

> 💡 **Impact Statement 只是模板填写**，没有实质内容。机器人操作技术实际上有不少社会影响值得讨论（如对制造业劳动力的影响），但论文选择了最保守的写法。

---

## Acknowledgment

This work was supported by The Robotics and AI Institute and ONR grant N00014-22-1-2621.

---

## 🔖 Section 总结

### 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐ | 问题诊断精准（过度乐观偏差），方案直接有效 |
| 实验扎实度 | ⭐⭐⭐ | 5 类任务、2 次迭代，规模偏小；缺少与 π₀.₆* 的对比 |
| 方法简洁性 | ⭐⭐⭐⭐⭐ | 极简——修正世界模型 + 过滤 SFT，有理论背书 |
| 工程实用性 | ⭐⭐⭐⭐ | 在 DROID 真实机器人上验证，可复现性强 |

### 核心洞察

1. **最重要的贡献**：「过度乐观偏差」的诊断和修复方案——这个问题此前被观察到但没有被系统解决
2. **方法的可扩展性**：VLAW 对基础模型无依赖性，可以插入任何 VLA + 世界模型组合
3. **对研究方向的启示**：如果 STAR-Pro 等下一代 VLA 需要大量多样化训练数据，「用少量真实 rollout 修正世界模型 → 生成大量合成数据」这套思路具有直接参考价值
