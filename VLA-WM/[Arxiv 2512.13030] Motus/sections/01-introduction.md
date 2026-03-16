[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

Introduction 分五层递进：现有方法碎片化（5 种独立范式） → 两大核心挑战（统一多模态生成 + 异构数据利用） → Motus 方案（MoT + Tri-model Joint Attention + UniDiffuser 调度器） → Latent Action（光流 → DC-AE → 低维向量） → 三阶段训练 + 六层数据金字塔。Figure 1 展示了完整架构。

---

A unified model is essential for embodied agents to integrate a spectrum of cognitive functions—from understanding scenes and instructions, imagining possible futures, to predicting consequences and generating actions—into a unified whole. However, existing methods model these capabilities in isolation: some rely on vision-language-action models (VLAs) [5, 8, 11, 26, 31, 36, 60, 65] to learn static policies from vision and language; others use world models or generative approaches built on predicted futures [4, 7, 19, 21, 25, 28, 39, 41, 53, 56, 62]; and $`\mathcal{F}_1`$ [32] combines VLAs and inverse dynamics models (IDMs) by explicitly imagining future visual observations, but it excludes world models or video generation models (VGMs), resulting in incomplete unification. These approaches fragment what should be a unified system into 5 separate modeling tasks:

> 💡 **现有方法的碎片化问题**：
> - **VLA 派**：用 VLM 的理解能力直接输出动作，但不建模未来（没有 world model）
> - **WM / VGM 派**：先预测未来画面再推理动作，但缺乏直接动作输出能力
> - **$`\mathcal{F}_1`$**：把 VLA 和 IDM 拼起来（先想象未来图像，再从图像推动作），但不包含 world model 和 video generation model，统一不完全
>
> 核心观点：这 5 种能力本该在一个系统里，现在却被拆成了 5 个独立模型。

---

• VLA: $`p(\pmb{a}_{t+1:t+k} \mid \pmb{o}_t, \ell)`$ . • WM: $`p(\pmb{o}_{t+1:t+k} \mid \pmb{o}_t, \pmb{a}_{t+1:t+k})`$ . • IDM: $`p(\pmb{a}_{t+1:t+k} \mid \pmb{o}_{t:t+k})`$ . • VGM: $`p(\pmb{o}_{t+1:t+k} \mid \pmb{o}_t, \ell)`$ . • Video-Action Joint Prediction Model: $`p(\pmb{o}_{t+1:t+k}, \pmb{a}_{t+1:t+k} \mid \pmb{o}_t, \ell)`$ .

> 💡 **5 种分布的直觉理解**：
>
> | 范式 | 公式 | 一句话 | 输入 → 输出 |
> |------|------|--------|-------------|
> | **VLA** | $`p(\pmb{a} \mid \pmb{o}_t, \ell)`$ | 看图说动作 | 当前图像 + 语言指令 → 动作序列 |
> | **WM** | $`p(\pmb{o} \mid \pmb{o}_t, \pmb{a})`$ | 给动作预测未来 | 当前图像 + 动作序列 → 未来图像序列 |
> | **IDM** | $`p(\pmb{a} \mid \pmb{o}_{t:t+k})`$ | 看视频推动作 | 连续图像帧 → 推断中间的动作 |
> | **VGM** | $`p(\pmb{o} \mid \pmb{o}_t, \ell)`$ | 看图生视频 | 当前图像 + 语言指令 → 未来视频（不含动作） |
> | **Joint** | $`p(\pmb{o}, \pmb{a} \mid \pmb{o}_t, \ell)`$ | 同时预测视频 + 动作 | 当前图像 + 语言指令 → 未来视频 + 动作 |
>
> 关键区分：VLA 和 IDM 都输出动作，但 VLA 只看当前帧 + 语言指令，IDM 看的是连续多帧（已知未来观测）。WM 和 VGM 都输出未来图像，但 WM 需要动作作为条件，VGM 用语言指令替代动作。Joint 是最完整的——同时生成视频和动作。

---

Two fundamental challenges (detailed in Sec. 3) hinder the integration of these capabilities. First, unifying such multimodal generative capabilities within one framework is nontrivial. While unified world models (UWMs) [64] offer a theoretical prototype, they are typically trained from scratch or with limited priors, lacking either robust vision-language understanding from vision-language models (VLMs) or rich physical interaction knowledge from VGMs. Second, embodied intelligence demands the ability to learn from large-scale heterogeneous data—including internet videos, egocentric human demonstrations, and multi-robot trajectories—but action spaces vary widely across embodiments, and most video data lack action labels, making it difficult to pretrain action experts with general motion and interaction priors.

> 💡 **两大核心挑战**：
>
> **挑战 1：如何统一 5 种分布？**
> - UWM [64] 做了初步探索，但它从头训练（train from scratch），没有利用 VLM 的理解能力和 VGM 的物理交互知识
> - 换句话说：从零开始训一个什么都能做的模型，效果不如把已有的强力 pretrained expert 整合起来
>
> **挑战 2：如何利用异构数据？**
> - 数据来源极其多样：互联网视频、人类自我中心视频、多机器人轨迹
> - 核心矛盾：**绝大多数视频数据没有 action 标注**（只有像素，没有关节角度/末端速度）
> - 不同机器人的 action space 差异巨大（7-DoF 机械臂 vs 双臂 vs 灵巧手），无法直接共享
> - 这导致 action expert 无法像 VLM / VGM 那样在海量数据上预训练

---

To address these challenges, we propose Motus, a unified latent action world model that integrates pretrained experts within a Mixture-of-Transformers (MoT) architecture. Our approach unifies the 5 key distributions by connecting a video generator (generative expert), an action expert, and a vision-language understanding expert via shared multi-head self-attention layers—a design we term Tri-model Joint Attention—which preserves specialized functionalities while enabling cross-modal knowledge fusion. To further coordinate multimodal generation, Motus incorporates a UniDiffuser-like scheduler, allocating distinct timesteps and noise scales to each modality (e.g., videos and actions). This enables a unified manner for simultaneous modeling marginal, conditional, and joint distributions, as well as adaptive switching among different inference modes (e.g., VLA, WM, IDM, VGM, Video-Action Joint Prediction Model).

> 💡 **Motus 的核心设计——解决挑战 1**：
>
> **Tri-model Joint Attention = 三个 expert 共享 self-attention，各自保留 FFN**
> - 三个 pretrained expert：① VLM（理解专家，如 Qwen2-VL）② VGM（视频生成专家，如 Wan2.1）③ Action Expert（动作专家）
> - 架构灵感来自 Bagel [18] 的 MoT：每层 Transformer 中，self-attention 是共享的（跨模态信息交换），但 FFN 是各 expert 独立的（保持专业性）
> - 好处：**既不破坏各 expert 的预训练权重，又能让三种模态互相看到彼此的 token**
>
> **UniDiffuser-like Scheduler = 通过噪声时间步控制生成模式**
> - 关键思想：给视频和动作分配**独立的 diffusion timestep** $`\tau_v`$ 和 $`\tau_a`$
> - 当 $`\tau_a = 0`$（动作无噪声）、$`\tau_v > 0`$（视频有噪声）→ WM 模式（给定动作，预测未来视频）
> - 当 $`\tau_v = 0`$、$`\tau_a > 0`$ → VLA 模式（给定观测，预测动作）
> - 当 $`\tau_v > 0`$、$`\tau_a > 0`$ → Joint 模式（同时生成视频 + 动作）
> - 一个模型，通过调噪声就能切换 5 种推理模式，非常优雅

---

![Figure 1](../images/055663917d2dd1ccfa8195052de90e53c125c4438f3f4259402d5c772dc27db6.jpg)

> **Figure 1. Motus Architecture.** $`a_t \ldots a_{t+k}`$ 是动作，$`z_t \ldots z_{t+k}`$ 是 latent action，$`\tau_v`$ 和 $`\tau_a`$ 分别是视频生成模型和动作专家的 rectified flow timestep。

> 💡 **Figure 1 解读**：
> - 左侧：三个 expert 的 token 序列被拼接在一起，送入共享的 self-attention（Tri-model Joint Attention）
> - 中间：每个 expert 有自己的 FFN（MoT 结构），保持各自的参数空间
> - 右侧：UniDiffuser scheduler 分别为视频和动作设置噪声时间步，控制哪些模态是"条件"（无噪声）、哪些是"生成目标"（有噪声）
> - 底部：Latent Action 模块——光流经过 DC-AE 编码为低维 latent，与真实 action 对齐

---

Additionally, to leverage heterogeneous data at scale, we introduce latent actions, which encode motion patterns from optical flow as a pixel-level "delta action". This representation bridges visual dynamics with control signals, enabling the action expert to be pretrained on diverse unlabeled videos and robot trajectories. Specifically, a pretrained deep compression autoencoder (DC-AE) with additional lightweight downsampling modules is used to reconstruct optical flow, whereas its encoded low-dimensional latents are supervised with a few action labels, both task-related and task-agnostic, thus steering the focus towards patterns associated with robotic activities.

> 💡 **Latent Action——解决挑战 2（无 action 标注的海量视频怎么用？）**：
>
> **核心思路**：optical flow（光流）是一种"像素级别的 delta action"——每个像素移动了多少，本质上就是"运动信号"
>
> **流程**：
> 1. 从任意视频中提取**光流**（不需要 action 标注，只需要连续两帧）
> 2. 用预训练的 **DC-AE**（Deep Compression AutoEncoder）将光流压缩为低维向量（约 14 维，接近机器人 action 的维度）
> 3. 用**少量有 action 标注的数据**（约 10%）监督这个 latent，让它和真实 action 对齐
>
> **为什么用光流而不是直接用 RGB 帧？**
> - RGB 帧包含大量与运动无关的外观信息（颜色、纹理、背景）
> - 光流天然只编码"运动"，是跨 embodiment 的通用表达——不管是机械臂、人手还是双臂机器人，光流都能表示"什么东西在动、往哪动、动多快"
>
> **监督方式**：task-related（目标任务的 action 标注）+ task-agnostic（通用机器人数据的标注），引导 latent 关注与机器人活动相关的运动模式

---

Subsequently, Motus undergoes a three-phase pretraining–finetuning pipeline (i.e., video pretraining, latent action pretraining, and embodiment-specific action finetuning) on a six-layer data pyramid spanning web-scale, egocentric human, simulation, task-agnostic, multi-robotic, and target-robotic data. This recipe aligns behaviors across different embodiments within the motion space described by optical flows and shares such interaction knowledge with target embodiments to enhance the generalization in downstream tasks, thereby providing the action expert with pretraining like other experts.

> 💡 **三阶段训练 + 六层数据金字塔**：
>
> **三阶段**：
> | 阶段 | 目标 | 数据 |
> |------|------|------|
> | Phase 1: Video Pretraining | 训练视频生成能力 | 互联网视频（最大规模） |
> | Phase 2: Latent Action Pretraining | 训练 action expert 的运动先验 | 光流 latent（无需 action 标注） |
> | Phase 3: Action Finetuning | 对齐到目标机器人 | 目标 embodiment 的有标注数据 |
>
> **六层数据金字塔**（从宽到窄）：
> 1. Web-scale 互联网视频（最多，最泛）
> 2. Egocentric 人类自我中心视频
> 3. Simulation 仿真数据
> 4. Task-agnostic 通用机器人数据
> 5. Multi-robotic 多机器人数据
> 6. Target-robotic 目标机器人数据（最少，最专）
>
> 核心理念：**让 action expert 也能像 VLM 和 VGM 一样享受大规模预训练**——通过光流 latent action 作为桥梁，把海量无标注视频的运动知识传递给 action expert。

---

Overall, our contributions can be summarized as follows:

• A unified embodied foundation model that integrates five mainstream paradigms (i.e., WMs, IDMs, VLAs, VGMs, and Video-Action Joint Prediction Models) without compromising general multimodal priors.

• A scalable robotic recipe with a three-phase training pipeline and six-layer data pyramid that leverages optical flow-based latent action to learn cross-embodiment transferable motion knowledge.

• Extensive experiments show that Motus significantly outperforms state-of-the-art approaches in both simulation (a $`+15\%`$ improvement over X-VLA [60] and a $`+45\%`$ improvement over $`\pi_{0.5}`$ [8]) and real-world scenarios (improved by $`+11{\sim}48\%`$), demonstrating that large-scale general and domain-specific priors can be effectively fused to enhance the generalization of policy learning.

> 💡 **三大贡献总结**：
> 1. **统一建模**：一个模型 = VLA + WM + IDM + VGM + Joint，且不损失各 pretrained expert 的能力
> 2. **可扩展训练方案**：三阶段 pipeline + 六层数据金字塔 + 光流 latent action → 跨 embodiment 可迁移的运动知识
> 3. **SOTA 性能**：仿真 +15%（vs X-VLA）/ +45%（vs $`\pi_{0.5}`$），真实世界 +11~48%

---

> 💡 **Motus 与现有方法的定位对比**：
>
> | 方法 | 统一的范式 | 利用 pretrained expert | 跨 embodiment 预训练 | 大规模无标注视频 |
> |------|-----------|----------------------|---------------------|-----------------|
> | **OpenVLA** [31] | VLA only | VLM ✅ | ❌ | ❌ |
> | **$`\pi_0`$ / $`\pi_{0.5}`$** [8] | VLA only | VLM ✅ | ❌ | ❌ |
> | **$`\mathcal{F}_1`$** [32] | VLA + IDM | VLM ✅ | ❌ | ❌ |
> | **UWM** [64] | VLA+WM+IDM+VGM+Joint | ❌（from scratch）| ❌ | ❌ |
> | **Motus** | VLA+WM+IDM+VGM+Joint | VLM ✅ + VGM ✅ + Action Expert ✅ | ✅（光流 latent） | ✅（六层金字塔） |
>
> Motus 的核心优势：**唯一一个同时做到"完全统一 5 种范式"+"利用 3 个 pretrained expert"+"跨 embodiment 大规模预训练"的方法**。

---

## Section 总结

**核心问题**：现有 embodied AI 方法把理解、预测、控制拆成独立模型，无法统一，也无法利用海量无标注视频。

**Motus 的回答**：
1. **架构层面**：用 MoT（Mixture-of-Transformers）的 Tri-model Joint Attention 把三个 pretrained expert（VLM + VGM + Action Expert）融合到一个模型中，共享 self-attention 做跨模态融合，各自 FFN 保持专业性
2. **调度层面**：用 UniDiffuser-like scheduler 为不同模态分配独立噪声时间步，一个模型通过调节噪声即可切换 VLA / WM / IDM / VGM / Joint 五种推理模式
3. **数据层面**：用光流（optical flow）作为"像素级 delta action"，经 DC-AE 压缩为低维 latent action，让 action expert 也能在海量无标注视频上预训练
4. **训练层面**：三阶段 pipeline（视频预训练 → latent action 预训练 → 目标 embodiment 微调）+ 六层数据金字塔，从互联网规模数据逐步聚焦到目标任务

**关键 takeaway**：Motus 的设计哲学是"不要从头训练，要站在巨人的肩膀上"——把已有的强力 pretrained model 通过巧妙的架构设计整合起来，比从零训一个 unified model（如 UWM）效果好得多。
