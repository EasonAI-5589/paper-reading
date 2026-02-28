# Section 6: Conclusions and Discussions

---

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks. Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **结论段很短，但诚实**：作者主动承认了两个局限：
> 1. **只评估 5 类任务**：这是最主要的局限，所有结论的泛化性存疑
> 2. **世界模型只对收集了 online 数据的任务高保真**：这意味着世界模型的泛化能力有限，不能直接用于未见过任务

> 💡 **更深层的局限（作者没说的）**：
> - **只迭代 2 次**：未展示迭代收敛曲线，不知道几次后会饱和或发散
> - **无 π₀.₆* 直接对比**：π₀.₆* 是 Physical Intelligence 用 offline advantage-conditioned SL 做的方法，本文方法引入了世界模型，但没有跟它直接对比，不知道额外复杂度是否值得
> - **计算成本**：50K steps 世界模型微调 + 500 条轨迹生成的计算成本没有报告
> - **只用 DROID 平台**：一种机械臂、一套相机配置，硬件泛化性未知

> 💡 **展望的准确性**：「base video models 进步 + 大规模机器人交互数据」是当前机器人学习的两大趋势，作者的判断合理。VLAW 这套框架确实可以随着世界模型基础能力的提升而受益。

---

## 总体评价

**贡献度**: ⭐⭐⭐⭐（4/5）  
**实验扎实度**: ⭐⭐⭐（3/5）  
**创新性**: ⭐⭐⭐⭐（4/5）

**一句话评价**：诊断了「世界模型过度乐观」这一真实问题，给出了直接有效的解决方案（加入失败案例 + 迭代），实验结果显著，但实验规模偏小（5任务、2次迭代），泛化性有待验证。

**与 STAR-Pro 研究方向的关系**：VLAW 用世界模型做数据增强来提升 VLA 策略，如果 STAR-Pro 也需要大量训练数据，这套 pipeline 可能是有价值的参考——尤其是「用少量真实 rollout 修正世界模型偏差」的思路。
