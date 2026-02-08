# 1. Introduction

> 来源: RoboBrain (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: Introduction 讲了三件事：(1) MLLM 在机器人领域的局限性，(2) 三大核心能力的具体含义，(3) ShareRobot + RoboBrain 的解决思路。

Recent advancements in Multimodal Large Language Models (MLLMs) have significantly advanced the pursuit of Artificial General Intelligence (AGI). By leveraging extensive multimodal datasets sourced from the internet and employing self-supervised learning techniques, MLLMs demonstrate exceptional capabilities in visual perception and understanding human language instructions, excelling in tasks such as visual question answering, image captioning, and sentiment analysis.

> 💡 **背景**: MLLM 在通用视觉理解上很强，但...

Despite significant progress in MLLMs, the exploration of their application in robotics remains in its early stages. Recent studies have examined the application of MLLMs in robotics, focusing on planning and subgoal decomposition, action sequencing, and replanning and feedback. However, their effectiveness in robotic scenarios—particularly for long-horizon manipulation tasks—reveals significant limitations.

> 💡 **Gap**: 现有 MLLM 用于机器人时有三个缺陷：
> 1. 不会拆任务（Planning）
> 2. 不知道抓哪里（Affordance）
> 3. 不知道怎么走过去（Trajectory）
>
> **根本原因**: 缺乏大规模、细粒度的机器人操作数据集。这才是核心 bottleneck。

For instance, consider a robotic arm tasked with lifting a teapot and pouring water into a cup. The MLLM should be capable of decomposing this task into sub-tasks, such as "approach the teapot and lift it", "move the teapot until the spout is positioned over the cup", and "tilt the teapot to pour". For each sub-task, the MLLM must utilize affordance perception to accurately identify the graspable regions of the teapot. Additionally, trajectory prediction is essential for determining the complete path from the starting point to the graspable part of the teapot.

> 💡 **例子详解**:
> ```
> 任务: "倒茶"
> 
> Planning 层:
> ├── Step 1: 靠近茶壶并拿起 ← 高层规划
> ├── Step 2: 移到杯子上方
> └── Step 3: 倾倒
> 
> Affordance 层 (Step 1):
> └── 茶壶把手区域 → bounding box [l_x, l_y, r_x, r_y]
> 
> Trajectory 层 (Step 1):
> └── 从当前位置到把手的 2D 路径 → [(x1,y1), (x2,y2), ...]
> ```
> 这个例子很好地展示了三层能力的级联关系。

---

### Contributions

> 💡 **贡献预览**:

The main contributions are:

1. **RoboBrain** — 统一的 MLLM，将抽象指令转化为具体动作（planning + affordance + trajectory 一站式）
2. **训练策略** — 精心设计 robot data vs general data 比例（约 4:6），多阶段训练，支持长视频和高分辨率
3. **ShareRobot 数据集** — 标注三维信息的高质量异构数据集
4. **SOTA 结果** — 在 RoboVQA、OpenEQA 等多个 benchmark 上达到最优

> 💡 **我的评价**:
> - Contribution 1 和 3 是核心，模型架构本身并不新（就是 LLaVA + LoRA）
> - 真正的创新在于 **数据**：从 Open X-Embodiment 精选 51K 实例 → 标注三维信息 → 生成 100 万 QA pairs
> - 训练策略中 robot:general = 4:6 的比例是经过消融实验验证的，这个发现有实用价值

---

## 💡 Section 总结

### 关键信息
| 项目 | 内容 |
|------|------|
| 核心问题 | MLLM 缺乏 planning/affordance/trajectory 三大机器人能力 |
| 根因 | 缺乏大规模细粒度机器人数据集 |
| 方案 | ShareRobot 数据集 + RoboBrain 多阶段训练 |
| 架构 | LLaVA (SigLIP + MLP + Qwen2.5-7B) + A-LoRA + T-LoRA |

### 关键洞察
- 模型不新，**数据为王** — 整篇论文的核心贡献是 ShareRobot 数据集
- "Abstract to Concrete" 的框架很直觉：先 plan → 再看哪能抓 → 再规划轨迹
- 和 RT-2 等直接输出动作的方案不同，RoboBrain 走的是**显式中间表示**路线
