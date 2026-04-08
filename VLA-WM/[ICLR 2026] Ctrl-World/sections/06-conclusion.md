[← 返回 README](../README.md)

# 6. Conclusion

## 总结

Ctrl-World 展示了一个可控世界模型能做到的两件事：
1. **在想象空间评估策略**：指令跟随行为与真实世界高度相关（slope=0.87）
2. **生成合成数据改进策略**：新物体/新指令任务成功率从 38.7% → 83.4%（+44.7%）

---

## 局限性

| 局限 | 原因 | 可能的解法 |
|------|------|----------|
| 精细物理交互不精确（碰撞、旋转） | 视频 backbone 物理建模有限 | 更强的物理 aware backbone（如 Wan2.1、Cosmos）|
| 对初始观测敏感 | 分布外场景泛化有限 | 更多/更多样的训练数据 |
| 无法改善低级执行成功率 | world model 精度瓶颈 | 迭代 rollout + fine-tuning |
| 依赖人工成功/失败判断 | 无自动 reward model | VLM 作为 reward model（future work）|

---

## 未来方向

- 迭代 policy rollout + fine-tuning（互相改进）
- VLM 自动打分取代人工标注
- 更强物理 backbone 提升建模精度

---

## 💡 批读注解

这篇论文最大的贡献是**提供了一个 workflow**，而不仅仅是一个模型：

```
初始观测 → World Model Rollout → 人/VLM 筛选成功轨迹 → SFT policy → 回到第一步迭代
```

这个 loop 在实验里被证明有效，而且完全不需要真实机器人 rollout。

这对我们项目的启示：
- 我们的 Ctrl-World finetune on LIBERO 就是在搭建这个 loop 的第一步
- 评估 pipeline（eval_ctrlworld.py + compute_metrics.py）是验证 world model 质量的工具
- 最终目标应该是：world model 质量足够好 → 可以拿来生成 LIBERO 的合成成功轨迹 → 改进某个 VLA 在 LIBERO 上的表现
