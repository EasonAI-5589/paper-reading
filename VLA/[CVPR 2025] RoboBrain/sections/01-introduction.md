# 01 - Introduction

## 📌 预览
Introduction 部分阐述了 MLLM 在机器人操作中的局限性（缺乏 planning、affordance、trajectory 三大能力），引出 ShareRobot 数据集和 RoboBrain 模型的动机，并列出四点主要贡献。

---

# 1. Introduction

Recent advancements in Multimodal Large Language Models (MLLMs) have significantly advanced the pursuit of Artificial General Intelligence (AGI). By leveraging extensive multimodal datasets sourced from the internet and employing self-supervised learning techniques, MLLMs demonstrate exceptional capabilities in visual perception and understanding human language instructions, excelling in tasks such as visual question answering [3, 15, 16], image captioning [28, 42, 45], and sentiment analysis [18, 21]. Despite significant progress in MLLMs, the exploration of their application in robotics remains in its early stages, highlighting a crucial area for further research and innovation.

> 💡 **研究背景**: 开篇确立了 MLLM 在 AGI 追求中的进展，但指出其在机器人领域的应用仍处于早期阶段。这为后文引出机器人领域的具体挑战做了铺垫。

Recent studies have examined the application of MLLMs in robotics, focusing on planning and subgoal decomposition [6, 31], action sequencing [8, 9], and replanning and feedback [49, 57, 98]. However, their effectiveness in robotic scenarios—particularly for long-horizon manipulation tasks—reveals significant limitations. These limitations stem from the current MLLMs' lack of three critical robotic capabilities: planning, affordance perception, and trajectory prediction, as illustrated in Fig. 1. For instance, consider a robotic arm tasked with lifting a teapot and pouring water into a cup. The MLLM should be capable of decomposing this task into sub-tasks, such as "approach the teapot and lift it", "move the teapot until the spout is positioned over the cup", and "tilt the teapot to pour". For each sub-task, such as "approach and grasp the teapot", the MLLM must utilize affordance perception to accurately identify the graspable regions of the teapot. Additionally, trajectory prediction is essential for determining the complete path from the starting point to the graspable part of the teapot. This challenge for existing MLLMs primarily arises from the scarcity of large-scale, fine-grained datasets specifically designed for robotic operation tasks.

> 💡 **核心问题定义**: 通过 "提茶壶倒水" 这个具体例子，清晰地解释了三层能力的层次关系：
> 1. **Planning（抽象层）**: 将 "倒水" 分解为多个子任务
> 2. **Affordance（感知层）**: 识别茶壶的可抓取区域
> 3. **Trajectory（执行层）**: 规划从起点到抓取点的完整路径
> 
> 这个 abstract → concrete 的渐进式分解是本文的核心 insight，也区别于之前只关注单一能力的工作。

To empower the RoboBrain's core capabilities that transition from abstract instruction comprehension to concrete action expression. we first introduce ShareRobot, a largescale, fine-grained dataset specifically designed for robotic operation tasks. Specifically, we label multi-dimensional information such as task planning, object affordance, and end-effector trajectory. Building upon ShareRobot, we developed RoboBrain, an MLLM model based on the LLaVA [48] architecture, aimed at enhancing the perception and planning capabilities of robots in complex tasks. In the process of training RoboBrain, we meticulously designed the ratio of robotic data to general multi-modal data, implemented a multi-stage training strategy, and incorporated long videos and high-resolution images. This approach endowed RoboBrain with powerful visual information perception capabilities in robotic scenarios, supporting historical frame memory and high-definition image input, thereby further enhancing the ability in robotic manipulation planning. Extensive experimental results demonstrate that RoboBrain outperforms existing models across multiple robotic benchmarks, including RoboVQA [73] and OpenEQA [61], achieving state-of-the-art performance. Additionally, it shows competitive results in trajectory and affordance prediction accuracy. These findings validate the effectiveness of the proposed dataset and framework in enhancing robotic brain capabilities. In summary, the main contributions of this paper are as follows:

> 💡 **解决方案概述**: 数据 + 模型双管齐下。训练中特别强调了机器人数据与通用数据的比例设计（后文实验显示 4:6 是最优比例），以及多阶段训练策略。支持长视频和高分辨率图像输入是实用性的关键。

• We propose RoboBrain, a unified multimodal large language model designed for robotic manipulation, which facilitates more efficient task execution by transforming abstract instruction into concrete actions.   
• We meticulously designed the ratio of robotic data to general multi-modal data, implemented a multi-stage training strategy, and incorporated long videos and highresolution images. This approach provided RoboBrain with historical frame memory and high-resolution image input, thereby further enhancing its capabilities in robotic manipulation planning.   
• We introduce ShareRobot, a high-quality heterogeneous dataset that labels multi-dimensional information, including task planning, object affordance, and end-effector trajectory, effectively enhancing various robotic capabilities.   
Comprehensive experimental results demonstrate that RoboBrain achieves state-of-the-art performance across various robotic benchmarks, highlighting its potential for real-world applications in robotics.

> 💡 **贡献总结**:
> 1. **RoboBrain 模型** — 统一的 MLLM，abstract → concrete
> 2. **训练策略** — 数据配比 + 多阶段 + 长视频/高分辨率
> 3. **ShareRobot 数据集** — 多维度标注（planning + affordance + trajectory）
> 4. **SOTA 实验结果** — 多个机器人 benchmark 上最优
> 
> 注意：贡献表述中模型和训练策略分开列，但实际上是紧密耦合的。数据集作为独立贡献也说明了数据工程在这个领域的重要性。

---

## 🔖 Section 总结
Introduction 清晰地定义了 MLLM 在机器人操作中的三层能力缺口（planning → affordance → trajectory），并以 "abstract to concrete" 为主线，提出 ShareRobot 数据集和 RoboBrain 模型作为解决方案。核心 insight 是将机器人操作理解为从抽象指令到具体动作的层次化分解过程。
