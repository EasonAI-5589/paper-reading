# VLAW 批读笔记 · Conclusions

---

## 6. Conclusions and Discussions

In this paper, we propose VLAW, an iterative improvement pipeline that jointly enhances both the vision–language–action (VLA) policy and the action-conditioned world model. We demonstrate that VLAW consistently improves performance across multiple contact-rich manipulation tasks.

Although the learned world model achieves high fidelity on the downstream tasks from which online data are collected, our current evaluation is limited to five task categories. Scaling online rollout data to a broader and more diverse set of tasks is a promising direction for future work. We believe that, as base video models continue to advance and large-scale robot interaction data become increasingly available, world-model-based training will provide a powerful new paradigm for learning generalist robotic policies.

> 💡 **作者自述的局限**：只在 5 类任务上验证。Scaling 到更多任务是 future work——但这也意味着 world model fine-tune 的计算开销会线性增长，是否可行需要进一步验证。

---

## 总体评价

### ✅ 优点

1. **问题定位准确**：world model 的 over-optimism 和 physical fidelity 问题真实存在，解决方案（online rollout fine-tune）简单有效
2. **方法极其简洁**：无特殊 architecture 设计，核心贡献在于训练数据的分布和 pipeline 设计
3. **实验扎实**：真实机器人、5 类 contact-rich 任务、每类 50 次评估，有说服力
4. **理论联系清晰**：将 binary-filtered BC 与 AWR / regularized RL 联系起来，有理论依据
5. **数据效率高**：一轮 VLAW ≈ 两轮 Filtered BC，50 条 real rollout 撬动 500 条合成数据

### ❌ 不足

1. **计算开销未报告**：world model 50K steps fine-tune + 生成 2500 条合成轨迹的实际时间是多少？复现成本不透明
2. **关键消融缺失**：没有"用 pretrained Ctrl-World 直接生成（不 fine-tune）"的消融，无法独立量化 world model fine-tune 这一步的贡献
3. **DSRL 评估不足**：只评 10 次 vs. 其他方法 50 次，对比不公平
4. **只迭代 2 次**：两轮后曲线仍在上升，没有看到收敛——更多迭代是否继续改进？是否会 overfit？未知
5. **Reward model 召回率低**：threshold=0.8 下 recall ≈ 45%（10/22 成功轨迹被保留），大量有效合成数据被丢弃

### 与相关工作的关系

| 工作 | 关系 |
|------|------|
| π₀.₅ (Physical Intelligence, 2025b) | 本文 base VLA model |
| Ctrl-World (Guo et al., 2025a) | 本文 base world model（第一作者同一团队） |
| π₀.₆* (Physical Intelligence, 2025a) | 同类在线 RL for VLA，不用 world model；advantage-conditioned SFT |
| DayDreamer (Wu et al., 2023) | 先驱：real-world MBRL，但模型容量小、单任务 |
| WMPO (Zhu et al., 2025) | 同期：world model + VLA，方法不同 |
| World-Gymnast (Sharma et al., 2026) | 同期：在 world model 里做 RL 训练 robot |
| AWR (Peng et al., 2019) | 方法的理论基础 |
