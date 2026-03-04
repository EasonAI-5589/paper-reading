[← 返回 README](../README.md)

# 6. Conclusions and Discussions

## 📌 预览

总结 VLAW 的贡献和局限，并对 world model-based robot training 的未来方向做展望。

---

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks. Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **作者自述的局限**：只在 5 类任务上验证。Scaling 到更多任务是 future work——但 world model fine-tune 的计算开销会随任务数线性增长，实际可行性需要进一步验证。
>
> **未提到但值得关注的局限**：
> 1. **计算开销未报告**：50K steps world model fine-tune + 2500 条合成轨迹生成的实际时间是多少？
> 2. **只迭代 2 次**：两轮后成功率曲线还在上升，没有看到收敛——更多迭代是否会继续改进还是 overfit？
> 3. **Reward model 召回率低**：threshold=0.8 下约 45% 的真实成功轨迹被丢弃，是 pipeline 的效率瓶颈
> 4. **任务间共享 world model**：5 类任务共享同一个 world model（co-training），不同任务的物理动力学差异大，是否存在任务间干扰？论文未分析

---

## 🔖 总体评价

### ✅ 优点

1. **问题定位准确**：world model 的 over-optimism 和 physical fidelity 问题真实存在，解决方案（online rollout fine-tune）简单有效
2. **方法极其简洁**：无 architecture 改动，核心贡献在于训练数据分布和 pipeline 设计，易于工程复现（理论上）
3. **实验扎实**：真实机器人、5 类 contact-rich 任务、每类 50 次评估，有说服力
4. **交互 metric 设计好**：confusion matrix 直接测物理交互结果预测，比纯视频质量指标更有任务相关性
5. **数据效率高**：一轮 VLAW ≈ 两轮 Filtered BC，50 条 real rollout 撬动 500 条合成数据

### ❌ 不足

1. **计算开销不透明**：训练时间、GPU 资源消耗均未报告，影响对实用性的判断
2. **关键 ablation 缺失**：没有「用 pretrained Ctrl-World 直接生成（不 fine-tune）」的 ablation，无法独立量化 world model fine-tune 的单独贡献——这是最大的实验遗漏
3. **DSRL 评估不公平**：10 次 vs. 其他方法 50 次，统计可靠性不足
4. **只迭代 2 次**：未见收敛曲线，不知道更多迭代是否继续有效
5. **Reward model 效率低**：threshold=0.8 下召回率约 45%，超过一半成功轨迹被丢弃

### 与相关工作关系一览

| 工作 | 关系 |
|------|------|
| π₀.₅（Physical Intelligence, 2025b） | 本文 base VLA model |
| Ctrl-World（Guo et al., 2025a） | 本文 base world model，第一作者同一团队 |
| π₀.₆*（Physical Intelligence, 2025a） | 同类 offline RL for VLA，但不用 world model |
| DayDreamer（Wu et al., 2023） | 先驱：real-world MBRL，单任务、小模型 |
| WMPO（Zhu et al., 2025） | 同期：world model + VLA policy optimization |
| World-Gymnast（Sharma et al., 2026） | 同期：在 world model 里做 RL 训练 robot |
| AWR（Peng et al., 2019） | 方法的理论基础 |
