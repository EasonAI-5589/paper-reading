[← 返回 README](../README.md)

# V. Conclusion

## 📌 预览
总结 Action-Sketcher 的核心贡献和未来方向。

---

In this work, we presented **Action-Sketcher**, a VLA framework designed to address spatial ambiguity and temporal brittleness in long-horizon robotic manipulation.

**核心成就**：
- 引入 Visual Sketch → 将推理过程通过 See-Think-Sketch-Act 循环操作化
- 在推理模式（生成可解释视觉计划）和动作执行（以 Sketch 为条件）之间自适应切换
- 超越 SOTA，尤其在需要复杂空间 grounding 和多步序列规划的任务上

**消融验证**：Visual Sketch 是核心组件，提供了高层推理与底层控制之间的鲁棒桥梁。其显式性质不仅提升了自主性能，还实现了有效的 human-in-loop 纠错。

---

## 局限性与未来方向（Appendix F）

> 💡 **批注（基于论文内容推断）**:
> 1. **主要瓶颈**：当前 61% 的失败源于 Sketch 生成不准确 → 如何提升空间 grounding 精度是最迫切的问题
> 2. **计算开销**：Reasoning Mode 引入额外推理步骤 → 如何在效率与精度间平衡
> 3. **泛化性**：训练于特定场景类型（桌面整理/倒茶等），对完全不同的任务类型泛化待验证
> 4. **3D 理解**：当前 Sketch 是 2D 图像平面上的投影 → 3D 场景中的完整 SE(3) 操作仍需更丰富的表达

---

## 对我们项目的启示（MoT / RLinf）

> 💡 **与我们工作的关联**:
> - **显式中间表达的价值**：Action-Sketcher 证明了把隐式计划"外显化"的重要性。MoT 的 Brain/Cerebellum 双模块也是一种显式功能分离，但意图表达仍是隐式的。
> - **Human-in-Loop 设计**：如果 world model as simulator 的 RL 框架也能暴露类似的可解释中间状态（如子目标 Sketch），可以极大提升调试和干预效率。
> - **长时域任务分解**：See-Think-Sketch-Act 的多步骤子任务分解方式，对 RLinf 中设计更复杂 reward 可能有参考价值——每个子任务的 Sketch 完成情况可作为稠密 reward 信号。
