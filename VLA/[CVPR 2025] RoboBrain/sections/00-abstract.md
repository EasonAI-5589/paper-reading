# 00 - Title & Abstract

## 📌 预览
RoboBrain 是一个基于 MLLM 的统一机器人操作模型，核心解决三个能力：任务规划（Planning）、可供性感知（Affordance Perception）、轨迹预测（Trajectory Prediction）。同时提出 ShareRobot 数据集提供多维度标注。

---

# RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete

Yuheng $\mathrm { J i ^ { 2 , 3 , 6 , * } }$ , Huajie Tan1,2,∗, Jiayu $\operatorname { S h i } ^ { 1 , 2 , * }$ , Xiaoshuai $\mathrm { H a o } ^ { 2 , * , \dagger }$ , Yuan Zhang1,2, Hengyuan Zhang1,2   
Pengwei Wang2,†, Mengdi Zhao2, Yao $\mathrm { \ M u ^ { 5 } }$ , Pengju $\mathbf { A n } ^ { 1 , 2 }$ , Xinda Xue1,2, Qinghang $\mathrm { S u ^ { 2 , 4 } }$ , Huaihai Lyu2,3,6 Xiaolong Zheng3,6, Jiaming Liu1,2, Zhongyuan Wang2, Shanghang Zhang1,2,B   
1 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University 2 Beijing Academy of Artificial Intelligence 3 Institute of Automation, Chinese Academy of Sciences 4 Institute of Information Engineering, Chinese Academy of Sciences 5 The University of Hong Kong 6 School of Artificial Intelligence, University of Chinese Academy of Sciences

> 💡 **作者背景**: 来自北大、BAAI、中科院自动化所、港大等机构的联合团队。通讯作者 Shanghang Zhang（张珊珊）是北大计算机系教授，长期从事多模态学习和机器人领域研究。

![](../images/82ad37a1378b1ec730894c6f36e09e1b3bf694a7e07c39ba0b21418e3ba6e99d.jpg)  
Figure 1. Overview of RoboBrain. RoboBrain consists of three key robotic capabilities: planning capability, affordance perception, and trajectory prediction. RoboBrain outperforms previous MLLMs in robotics tasks. The bottom part shows the composition of RoboBrain's training data and provides a specific example of visual question answering from our proposed ShareRobot. Best viewed on screen.

> 💡 **Figure 1 解读**: 这张总览图展示了 RoboBrain 的三大核心能力：(1) Planning — 将高层指令分解为子任务；(2) Affordance — 感知物体可交互区域；(3) Trajectory — 预测末端执行器运动轨迹。下半部分展示了训练数据组成和 ShareRobot 的 VQA 示例。这个 abstract-to-concrete 的渐进式框架是论文的核心设计理念。

# Abstract

Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various multimodal contexts. However, their application in robotic scenarios, particularly for long-horizon manipulation tasks, reveals significant limitations. These limitations arise from the current MLLMs lacking three essential robotic brain capabilities: Planning Capability, which involves decomposing complex manipulation instructions into manageable sub-tasks; Affordance Perception, the ability to recognize and interpret the affordances of interactive objects; and Trajectory Prediction, the foresight to anticipate the complete manipulation trajectory necessary for successful execution. To enhance the robotic brain's core capabilities from abstract to concrete, we introduce ShareRobot, a high-quality heterogeneous dataset that labels multi-dimensional information such as task planning, object affordance, and end-effector trajectory. ShareRobot's diversity and accuracy have been meticulously refined by three human annotators. Building on this dataset, we developed RoboBrain, an MLLMbased model that combines robotic and general multi-modal data, utilizes a multi-stage training strategy, and incorporates long videos and high-resolution images to improve its robotic manipulation capabilities. Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various robotic tasks, highlighting its potential to advance robotic brain capabilities. Project website: RoboBrain.

> 💡 **摘要要点梳理**:
> - **问题**: 当前 MLLM 在机器人长序操作任务中缺乏三个关键能力
> - **数据**: ShareRobot — 标注了 task planning、object affordance、end-effector trajectory 的高质量异构数据集，经 3 名人类标注员精细校验
> - **模型**: RoboBrain — 基于 MLLM，混合机器人与通用多模态数据，多阶段训练，支持长视频和高分辨率图像
> - **结果**: 在多个机器人 benchmark 上达到 SOTA
> - **核心思想**: "from abstract to concrete" — 从抽象指令理解到具体动作执行的渐进式能力增强

---

## 🔖 Section 总结
本文提出 RoboBrain，一个统一的多模态大语言模型用于机器人操作，核心创新点在于：(1) 明确定义了机器人大脑需要的三个能力层次（规划→可供性→轨迹）；(2) 构建了大规模多维度标注数据集 ShareRobot；(3) 设计了多阶段训练策略实现 "abstract to concrete" 的能力增强。
