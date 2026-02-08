[← 返回 README](../README.md)

# Appendix

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 概要总结

Appendix 包含两个部分：

### A. Qualitative Examples（定性示例）

展示 RoboBrain 2.0 在各种具身 AI 任务上的可视化结果，共 7 类任务：

| 子节 | 任务 | Figure 范围 | 说明 |
|------|------|------------|------|
| A.1 | Pointing（指向） | Figure 5-20 | 蓝色点表示模型的空间指代预测，包含不同 Reasoning Step（1-5步推理）|
| A.2 | Affordance（可操作性） | 后续图 | 物体部件功能识别（如杯子的把手、瓶盖、勺子等）|
| A.3 | Trajectory（轨迹） | 后续图 | 未来轨迹预测的可视化 |
| A.4 | EgoPlan2 | 后续图 | 第一人称视角活动规划示例 |
| A.5 | Close-Loop Interaction | 后续图 | 闭环交互的 Observation-Thought-Action 序列 |
| A.6 | Multi-Robot Planning | 后续图 | 多机器人协作规划示例 |
| A.7 | Synthetic Benchmarks | 后续图 | 合成基准测试示例 |

> 💡 **Appendix A 要点**:
> - Pointing 示例最多（Figure 5-20），展示了从简单（"指出橙色盒子"）到复杂（"指出离电视最远的黑色物体"）的空间推理
> - Reasoning Step 从 1 到 5 不等，说明模型可以进行多步推理
> - Affordance 示例展示了功能性理解（"杯子哪个部分用来喝水？" → 指向杯口）
> - Close-Loop 示例展示了完整的 OTA 链

### B. Prompts Details（Prompt 详情）

列出了各任务使用的 prompt 模板：

| 子节 | 任务 | 说明 |
|------|------|------|
| B.1 | Pointing（坐标） | "Point out all instances of {label} in the image" 等 28 个模板 |
| B.2 | Trajectory（坐标） | 轨迹预测的 prompt 格式 |
| B.3 | Affordance（Bbox） | "Which part of X can be used to Y?" 格式 |
| B.4 | General Spatial Analysis | 自由问答形式的空间分析 |
| B.5 | Long-horizon Planning | 长程规划的 prompt + scene graph 输入格式 |
| B.6 | Closed Loop Conversation | 闭环交互的多轮对话格式 |
| B.7 | Multi-Robot Planning | 多机器人协作的结构化 prompt |

> 💡 **Appendix B 要点**:
> - Prompt 设计体现了任务的层次性：坐标输出 vs 文本输出 vs 结构化输出
> - Multi-Robot Planning 的 prompt 包含完整的 scene graph JSON + 机器人规格 + 工具列表
