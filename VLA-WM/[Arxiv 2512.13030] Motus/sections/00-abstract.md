[← 返回 README](../README.md)

# Abstract

> 来源: Motus: A Unified Latent Action World Model (arXiv 2512.13030)

---

## 📌 预览

本文提出 Motus，一个**统一的 latent action 世界模型**，核心思路是：

1. **MoT 架构**统一三个预训练专家（VLM + VGM + Action），共享注意力但保留各自 FFN
2. **UniDiffuser 式调度器**灵活切换 5 种建模模式（VLA / WM / IDM / VGM / 联合预测）
3. **光流 → latent action**：从像素级运动中提取 "delta action"，让 action expert 能在海量无标签视频上预训练
4. **三阶段训练 + 六层数据金字塔**：从 web 视频到目标机器人，逐步注入先验

关键结果：仿真 +15% over X-VLA / +45% over $`\pi_{0.5}`$，真实世界 +11~48%。

---

## 📄 原文 + 批注

While a general embodied agent must function as a unified system, current methods are built on isolated models for understanding, world modeling, and control.

> 💡 **开篇定义问题**：当前具身智能的三大能力——理解（understanding）、世界建模（world modeling）、控制（control）——各自独立建模。这是整篇论文要解决的核心矛盾：**碎片化 vs 统一化**。

This fragmentation prevents unifying multimodal generative capabilities and hinders learning from large-scale, heterogeneous data.

> 💡 **碎片化带来两个后果**：
> 1. 无法统一多模态生成能力（你不能在一个模型里同时做视频生成和动作预测）
> 2. 无法从大规模异构数据中学习（互联网视频没有动作标签、不同机器人动作空间不同）
>
> 这两个后果正好对应后文的 **Challenge 1**（统一建模）和 **Challenge 2**（异构数据利用）。

In this paper, we propose Motus, a unified latent action world model that leverages existing general pretrained models and rich, sharable motion information.

> 💡 **Motus 的两个关键词**：
> - **leverages existing pretrained models**：不从头训练，而是复用已有的 VLM 和 VGM 的先验知识
> - **sharable motion information**：运动信息是跨具身体共享的——无论是人还是机器人，光流都能描述运动

Motus introduces a Mixture-of-Transformer (MoT) architecture to integrate three experts (i.e., understanding, video generation, and action) and adopts a UniDiffuser-style scheduler to enable flexible switching between different modeling modes (i.e., world models, vision-language-action models, inverse dynamics models, video generation models, and video-action joint prediction models).

> 💡 **架构设计的两大创新**：
>
> **① MoT (Mixture-of-Transformer)**：三个专家共享多头自注意力（论文称 **Tri-model Joint Attention**），但各自保留独立的 FFN。类似 Bagel 在多模态理解 + 生成上的做法，但这里扩展到了具身智能的三路：
> ```
> Understanding Expert (Qwen3-VL-2B)  ─┐
> Video Generation Expert (Wan 2.2 5B) ─┼─ 共享 Multi-Head Self-Attention
> Action Expert (同深度 Transformer)    ─┘
>                                        各自独立 FFN
> ```
>
> **② UniDiffuser 式调度器**：核心思想是给视频和动作分配**不同的扩散 timestep 和噪声尺度** $`\tau_v`$ 和 $`\tau_a`$。通过控制哪个模态加噪、哪个不加，就能灵活切换 5 种模式：
>
> | 模式 | 视频 $`\tau_v`$ | 动作 $`\tau_a`$ | 对应分布 |
> |------|----------|----------|----------|
> | VLA | 不加噪（条件） | 加噪（生成） | $`p(\mathbf{a} \mid \mathbf{o}, \ell)`$ |
> | World Model | 加噪（生成） | 不加噪（条件） | $`p(\mathbf{o} \mid \mathbf{o}_t, \mathbf{a})`$ |
> | IDM | 不加噪（条件） | 加噪（生成） | $`p(\mathbf{a} \mid \mathbf{o}_{t:t+k})`$ |
> | VGM | 加噪（生成） | 无 | $`p(\mathbf{o} \mid \mathbf{o}_t, \ell)`$ |
> | Joint Prediction | 都加噪 | 都加噪 | $`p(\mathbf{o}, \mathbf{a} \mid \mathbf{o}_t, \ell)`$ |
>
> 这个设计非常优雅——同一个模型，不同的噪声配置，就是不同的具身基础模型。

Motus further leverages the optical flow to learn latent actions and adopts a recipe with three-phase training pipeline and six-layer data pyramid, thereby extracting pixel-level "delta action" and enabling large-scale action pretraining.

> 💡 **Latent Action 的核心思路**：
>
> **光流 (optical flow)** 描述的是相邻帧之间像素级的位移——本质上就是"画面里的东西往哪个方向动了多少"。论文把这个称为 **pixel-level "delta action"**，即像素级别的动作变化量。
>
> 为什么这很重要？因为：
> - 互联网视频有光流但没有机器人动作标签
> - 不同机器人的动作空间不同（维度、范围、语义都不同）
> - 但光流是**通用的运动表达**——人手抓杯子和机械臂抓杯子，光流模式是相似的
>
> 具体做法：用 DC-AE（深度压缩自编码器）把高维光流压缩成 **14 维 latent action**（匹配典型机器人动作空间维度），训练时 90% 无标签数据做自监督重建 + 10% 有标签数据做弱监督对齐。
>
> **三阶段训练**：
> ```
> Stage 1: Video Pretrain     → 只训练 VGM，学习视觉动态
> Stage 2: Latent Action Pretrain → 全模型训练（VLM 冻结），用 latent action
> Stage 3: Embodiment SFT     → 在目标机器人数据上微调，用真实 action
> ```
>
> **六层数据金字塔**（从底到顶，数据量递减、质量递增）：
> ```
> Level 1: Web Data（互联网视频/图文，最大量）
> Level 2: Egocentric Human Videos（第一人称人类视频）
> Level 3: Synthetic Data（仿真数据）
> Level 4: Task-agnostic Data（任务无关的机器人随机采样数据）
> Level 5: Multi-Robot Task Trajectory Data（多机器人任务轨迹）
> Level 6: Target-Robot Task Trajectory Data（目标机器人轨迹，最少但最精准）
> ```

Experiments show that Motus achieves superior performance against state-of-the-art methods in both simulation (a +15% improvement over X-VLA and a +45% improvement over $`\pi_{0.5}`$) and real-world scenarios (improved by +11~48%), demonstrating unified modeling of all functionalities and priors significantly benefits downstream robotic tasks.

> 💡 **实验亮点**：
> - **仿真**（RoboTwin 2.0，50 个操作任务）：Motus 88.66% vs X-VLA 72.80%（+15%）vs $`\pi_{0.5}`$ 42.98%（+45%）
> - **真实世界**（AC-One + Agilex-Aloha-2 双臂平台）：AC-One 平均 63.22% vs $`\pi_{0.5}`$ 14.79%（+48%），Agilex 平均 59.30% vs $`\pi_{0.5}`$ 48.60%（+11%）
> - 所有模型在同等条件下训练（40k SFT 步数），公平对比预训练策略的有效性
>
> **核心结论**：统一建模所有功能和先验，对下游机器人任务有显著收益。不是单纯的"大力出奇迹"，而是统一框架本身带来的增益。

---

![Figure 1](../images/055663917d2dd1ccfa8195052de90e53c125c4438f3f4259402d5c772dc27db6.jpg)

*Figure 1. Motus Architecture. $`a_t \ldots a_{t+k}`$ 是动作，$`z_t \ldots z_{t+k}`$ 是 latent actions，$`\tau_v`$ 和 $`\tau_a`$ 分别是视频生成模型和 action expert 的 rectified flow timestep。*

> 💡 **Figure 1 批读**：
>
> 这张图展示了 Motus 的完整架构，从左到右理解：
>
> **输入侧**（左）：
> - 语言指令 $`\ell`$ 和当前观测 $`\mathbf{o}_t`$ 输入 VLM（Qwen3-VL-2B）
> - 当前帧 $`\mathbf{o}_t`$ 同时输入 VGM（Wan 2.2 5B）作为条件帧
>
> **核心结构**（中）：
> - 三个专家并行排列，各自有独立的 FFN
> - **Tri-model Joint Attention**：三个专家的 token 拼接后做联合自注意力，实现跨模态信息交换
> - 每个专家有 AdaLN 注入各自的 rectified flow timestep
>
> **输出侧**（右）：
> - VGM 输出未来视频帧 $`\mathbf{o}_{t+1:t+k}`$
> - Action Expert 输出动作 $`\mathbf{a}_{t+1:t+k}`$（SFT 阶段）或 latent action $`\mathbf{z}_{t+1:t+k}`$（预训练阶段）
>
> **关键设计**：$`\tau_v`$ 和 $`\tau_a`$ 是独立控制的——这就是 UniDiffuser 式调度器的精髓。通过设置不同的 timestep 组合，同一个模型可以表现为 VLA、WM、IDM、VGM 或联合预测模型。

---

## 💡 Section 总结

### 核心信息速查

| 指标 | 值 |
|------|-----|
| 模型名称 | Motus |
| 团队 | 清华大学 THBI Lab + 北京大学 + 地平线机器人 |
| VGM 骨干 | Wan 2.2 5B |
| VLM 骨干 | Qwen3-VL-2B |
| 统一的建模范式 | VLA、WM、IDM、VGM、Video-Action Joint Prediction |
| 核心创新 | MoT + UniDiffuser 调度器 + 光流 latent action |
| 训练方案 | 三阶段训练 + 六层数据金字塔 |

### 核心洞察

1. **统一 ≠ 简单拼接**：MoT 架构让每个专家保留专业能力（独立 FFN），同时通过共享注意力实现知识融合，比 UWM 的单一 backbone 更合理
2. **光流是天然的跨具身桥梁**：不同机器人动作空间各异，但光流描述的运动模式是通用的，这让 action expert 的大规模预训练成为可能
3. **从 UniDiffuser 借鉴的调度器是最优雅的设计**：一个模型、一套参数，通过噪声配置切换 5 种模式，避免了多模型维护的工程复杂度
4. **数据金字塔解决了"量 vs 质"的矛盾**：底层海量 web 数据提供通用先验，顶层少量目标机器人数据提供精准适配，中间层逐步过渡
