# 3. ShareRobot Dataset

> 来源: RoboBrain (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 这是本文最核心的贡献。详细介绍 ShareRobot 数据集的设计、筛选、标注和统计。

---

### 3.1 Overview

ShareRobot is a comprehensive dataset, facilitates more efficient task execution by transforming abstract concepts into concrete actions.

> 💡 **六大特点**:
> ```
> ShareRobot 特性:
> ├── Fine-grained: 每帧都有低层指令（vs OXE 只有高层描述）
> ├── Multi-dimensional: planning + affordance + trajectory 三维标注
> ├── High quality: 严格筛选 51,403 实例
> ├── Large scale: 1,027,990 QA pairs（最大开源机器人规划数据集）
> ├── Rich diversity: 102 场景、12 种机器人、107 种原子任务
> └── Easy scalability: Pipeline 可扩展
> ```

---

### 3.2 Data Selection

Based on the Open X-embodiment dataset, we carefully selected 51,403 instances.

> 💡 **筛选标准** — 这部分很重要，说明数据质量把控：
> ```
> 六条筛选规则:
> ├── 分辨率 > 128px（去掉低质量视频）
> ├── 描述准确（去掉无描述/模糊描述的）
> ├── 任务成功（去掉失败案例）
> ├── 视频 > 30 帧（太短的原子任务太少）
> ├── 物体不被遮挡（要能看到 affordance）
> └── 轨迹清晰（要能标注 trajectory）
> ```
> 💡 **批注**: 这些规则很合理。特别是"成功任务"这条 — 机器人数据里失败案例很多，
> 但拿来训 planning 模型会引入错误先验。不过反过来想，失败案例其实也有学习价值
> （学什么不该做），但这篇论文没做这个。

---

### 3.3 Data Labeling

> 💡 **三种标注方式**:

**Planning Labeling**: 
- 每个演示提取 30 帧 + 高层描述 → Gemini 分解为低层指令 → 3 个人工标注员审核
- 设计 10 种问题类型 × 5 个模板 → 随机选 2 个 → 生成 QA pairs
- 51,403 实例 → **1,027,990 QA pairs**

> 💡 **数据放大策略**: 关键是模板多样化。每个实例生成 ~20 个 QA pairs（10 问题类型 × 2 模板），
> 这样从 5 万实例膨胀到 100 万 QA pairs。合理但需要注意模板带来的同质性。

**Affordance Labeling**:
- 筛选 6,522 张图片，标注 bounding box {l_x, l_y, r_x, r_y}
- 人工审核确保 affordance 区域和指令对齐

**Trajectory Labeling**:
- 筛选 6,870 张图片，标注 gripper 轨迹（至少 3 个 {x, y} 坐标）
- 人工审核确保轨迹和指令对齐

> 💡 **数据量对比**:
> ```
> Planning:  51,403 实例 → 1,027,990 QA pairs  ← 主力数据
> Affordance: 6,522 张图片 + bbox                ← 数量较少
> Trajectory: 6,870 张图片 + waypoints            ← 数量较少
> ```
> Affordance 和 Trajectory 的标注量远小于 Planning，这也解释了为什么后两者用 LoRA 微调而不是全模型训练。

---

### 3.4 Data Statistics

![Figure 3](../images/fac14631f6cdcbe8c435a9074e9bf32a0d5670e74af39b4deae5664e59ba8477.jpg)
*Figure 3: ShareRobot 数据集多样性 — (a) 23 个源数据集 (b) 12 种机器人 (c) 107 种原子任务分布*

> 💡 **Figure 3 批读**:
> ```
> 数据来源: 23 个 OXE 子数据集
> 
> 机器人多样性: 12 种 embodiment
> ├── Franka Panda (最多)
> ├── UR5
> ├── Google Robot
> └── 其他 9 种
> 
> Top 5 原子任务:
> ├── pick   ← 频率最高
> ├── move
> ├── reach
> ├── lift
> └── place
> 
> 132 种原子动作 → 符合真实机器人场景的长尾分布
> ```

| 数据划分 | 训练集 | 测试集 |
|----------|--------|--------|
| Planning QA | 1,000,000 | 2,050 |
| Affordance | 6,000 | 522 |
| Trajectory | 6,000 | 870 |

---

## 💡 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 总实例数 | 51,403 |
| QA pairs | 1,027,990 |
| 场景数 | 102 |
| 机器人种类 | 12 |
| 原子任务类型 | 107 (后来统计 132) |
| 源数据集 | 23 (from OXE) |
| 人工标注员 | 3 人 |

### 核心洞察
1. **数据是核心贡献** — ShareRobot 是目前最大的开源机器人规划数据集
2. **从粗到细的标注** — OXE 高层描述 → Gemini 分解 → 人工审核，是个可复制的 pipeline
3. **Affordance/Trajectory 数据量小** — 只有 ~6K 图片，这可能是性能瓶颈
4. **数据质量把控严格** — 6 条筛选规则 + 3 人审核，但 Gemini 生成的低层指令质量如何？论文没有详细分析
