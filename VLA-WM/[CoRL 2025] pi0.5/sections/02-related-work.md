[← 返回 README](../README.md)

# II. Related Work

## 📌 预览
Related Work 从四个方向定位 π0.5：(1) 通用机器人操作策略、(2) 非机器人数据协同训练、(3) 语言推理与规划、(4) 开放世界泛化系统。

---

**Generalist robot manipulation policies.** Recent works have demonstrated that broadening the training data distribution for robot manipulation policies from narrow, single-task datasets to diverse datasets that span many scenes and tasks [17, 25, 80, 63, 41, 6, 30, 67, 1] allows the resulting policies to not only solve a wider range of tasks out of the box, but also improves their ability to generalize to new scenes and tasks [9, 63, 62, 22]. Training such generalist policies requires new modeling approaches that can handle the scale and diversity of datasets that often span hundreds of different tasks and scenes. Vision-language-action models (VLAs) [23, 92, 42, 8, 83, 90, 55, 45, 3, 75, 64, 76, 84, 7, 37] offer an appealing solution: by fine-tuning pre-trained vision-language models for robot control, VLAs can leverage the semantic knowledge acquired from web-scale pretraining and bring it to bear on the robotics problem. When combined with highly expressive action decoding mechanisms like flow matching [8], diffusion [55, 84, 52], or advanced action tokenization schemes [64], VLAs can perform a wide range of complex manipulation tasks in the real world. However, despite impressive language following abilities, VLAs are still typically evaluated in environments that closely match their training data. While some studies suggest that simple skills like picking up objects or opening drawers can be made to generalize simply by collecting robot data in a broader set of environments [14, 67, 28, 49, 64], it is challenging to apply the same approach to more complex, long-horizon tasks like cleaning up a kitchen, where achieving broad coverage of plausible scenarios via brute-force scaling of robot data collection is infeasible. In our experiments, we evaluate $\pi _ { 0 . 5 }$ in entirely new scenes, such as new kitchens and bedrooms that were not seen in training, showing that our VLA can generalize to entirely new scenes by leveraging not only direct first-hand experience on the target mobile manipulator platform, but also information from other data sources. These sources include data from other (non-mobile) robots, high-level semantic subtask prediction, and data from the web.

> 💡 **通用操作策略的现状**:
> - VLA 通过微调预训练 VLM 实现语义知识迁移
> - 动作解码：flow matching (π0)、diffusion (RDT-1B)、FAST tokenization
> - **现有局限**: VLA 通常在训练分布内评估；简单技能可以靠扩数据泛化，但复杂长时域任务不行
> - **π0.5 的区别**: 在全新场景评估 + 利用非目标域数据源

---

**Non-robot data co-training.** A number of prior works have sought to use diverse non-robot data to improve the generalization of robot policies. Prior methods have explored initializing vision encoders from computer vision datasets [85, 58, 57, 18], or leveraging off-the-shelf task planners [38, 48, 73, 81]. VLA policies are typically initialized from a pre-trained vision-language model, which has been exposed to large amounts of internet vision and language data [23, 92, 42]. Notably, the VLA architecture is flexible and allows to map between input and output sequences of multi-modal vision, language, and action tokens. As such, VLAs broaden the design space of possible transfer approaches beyond simple weight initialization, by supporting the co-training of a single, unified architecture on not just robot action imitation data, but any dataset that interleaves one or multiple of the aforementioned modalities. Prior works have demonstrated that co-training VLAs with data mixtures used for VLM training [23, 92, 86] can improve their generalization ability, e.g., when interacting with new objects or unseen scene backgrounds. In this work, we go beyond VLM data co-training and design a system for co-training VLAs with a broader set of robotics-relevant supervision sources, including data from other robots, high-level semantic subtask predictions, and verbal language instructions. While multi-task training and co-training are not new ideas, we show that the specific combination of data sources in our system enables mobile robots to perform complex and long-horizon behaviors in entirely new environments. We believe that this level of generalization, particularly when accounting for the complexity of the tasks, goes significantly beyond the results demonstrated in prior works.

> 💡 **协同训练的演进**:
> - 早期：视觉编码器初始化 (R3M, MVP)
> - 中期：VLM 权重初始化 (PaLM-E, RT-2, OpenVLA)
> - 当前：VLM 数据混合训练 (MAGMA)
> - **π0.5**: 更广泛的监督信号 — 跨机器人数据 + 语义子任务 + 语言指令 + 网络数据

---

**Robot reasoning and planning with language.** A number of prior works have shown that augmenting end-to-end policies with high-level reasoning can significantly improve performance for long-horizon tasks [2, 36, 44, 74, 71, 4, 16, 11, 53, 88, 51, 59, 13, 70, 91, 65, 72, 47, 76, 89], particularly when high-level subtask inference can benefit from large pretrained LLMs and VLMs. Our method also uses a two-stage inference procedure, where we first infer a high-level semantic subtask (e.g., "pick up the plate"), and then predict the action based on this subtask. Many prior methods have employed two separate models for this purpose, with a VLM predicting semantic steps and a separate low-level policy executing those steps [2, 71, 13, 24, 70, 72, 47]. Our method uses the same exact model for both high-level and low-level inference, in a recipe that more closely resembles chain-of-thought [82] or test-time compute [39] methods, though unlike embodied chain-of-thought methods [88, 46, 61], the high-level inference process still runs at a lower frequency than low-level action inference.

> 💡 **层级推理的定位**:
> - 传统方法：VLM (高层) + 独立低层策略 = 两个模型
> - π0.5：**同一个模型**同时做高层和低层推理，类似 chain-of-thought
> - 区别于 ECoT 等方法：高层推理频率低于低层动作推理

---

**Robotic learning systems with open-world generalization.** While most robotic learning systems are evaluated in environments that closely match the training data, a number of prior works have explored broader open-world generalization. When the robot's tasks are restricted to a more narrow set of basic primitives, such as picking up objects, methods that allow for task-specific assumptions (e.g., grasp prediction, or incorporating model-based planning and control) have been shown to generalize broadly, even to entirely new homes [40, 20, 60, 56, 29]. However, such methods do not readily generalize to the full range of possible tasks that a generalist robot might need to perform. More recently, large-scale datasets collected across many domains [41, 68, 63, 67, 14, 49] have been shown to enable generalization of simple but end-to-end learned tasks to new environments [33, 31, 67, 69, 26, 49, 28, 64]. However, the tasks in these demonstrations are still relatively simple, typically less than a minute in length and often with relatively low success rates. We show that $\pi _ { 0 . 5 }$ can perform long, multistage tasks, such as putting all of the dishes in the sink or picking all of the clothing off the floor of a new bedroom, while generalizing to entirely new homes.

> 💡 **开放世界泛化的现状**:
> - 窄任务（如抓取）+ 任务特定假设 → 可以泛化到新家庭 (Roomba, DexNet)
> - 大规模数据集 + 端到端学习 → 简单任务可泛化，但通常 < 1 分钟，成功率低
> - **π0.5 的突破**: 长时域 (10-15 min)、多阶段、灵巧任务，在全新家庭中执行

---

## 🔖 Section 总结

### 核心洞察
1. **VLA 是整合异构数据的最佳载体** — 序列建模框架天然支持多模态
2. **co-training 的关键进步** — 从"仅 VLM 数据"扩展到"跨机器人 + 语义 + 语言指令"
3. **统一模型 vs 分离模型** — π0.5 用同一模型做高层+低层推理，更接近 chain-of-thought
4. **泛化复杂度的跨越** — 从简单抓取泛化到长时域多阶段操作
