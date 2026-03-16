[← 返回 README](../README.md)

# 2. Related Works

## 📌 预览

Related Works 分两个方向：2.1 Unified Multimodal Models（统一多模态模型——从通用多模态到具身基础模型的演进）和 2.2 Latent Action Models（隐动作模型——如何在没有动作标签的情况下从视觉动态中提取可控信号）。Motus 的定位是：既要统一五大建模范式（不像 UWM 从零训练），又要用光流驱动的 latent action 实现跨 embodiment 预训练（不像已有方法只用 RGB/DINOv2 重建）。

---

## 2.1. Unified Multimodal Models

Unified multimodal models jointly model various modalities and tasks within a single generative framework [29, 40, 45, 47, 49, 52], showing broad applications across several domains [35, 54, 63]. In particular, Bagel [18] achieves unification via MoT [30], sharing the multi-head self-attention layers between understanding experts and generation experts. In contrast, existing embodied foundation models are developed independently, spawning multiple disparate paradigms: some leverage the text-image understanding capabilities of VLMs to learn action prediction [6, 8, 27], while others utilize VGMs to generate video sequences and infer actions from consecutive frames [19, 21, 62]. Recently, $`\mathcal{F}_1`$ [32] extends VLAs to explicitly imagine future visual states and output actions by IDMs, thereby merging both models. Furthermore, UWM [64] unifies WMs, VLAs, IDMs, VGMs, and Video-Action Joint Prediction Models within a single diffusion backbone, making an initial exploration of complete robotic models. Unlike UWM, our method goes beyond unified modeling by further incorporating internet-scale general multimodal priors and specialized priors from massive robotic trajectories.

> **通俗解读**：
>
> 这一段梳理了"统一多模态模型"从通用 AI 到具身智能的发展脉络：
>
> **通用多模态统一**：Bagel 等模型用 MoT（Mixture-of-Transformers）把"理解"和"生成"放在同一个框架里，共享注意力层但各自有独立 FFN。这是 Motus 架构灵感的来源。
>
> **具身领域的现状——各自为政**：
> - **VLM → VLA 路线** [6, 8, 27]：OpenVLA、$`\pi_{0.5}`$、GR00T 等，用 VLM 的图文理解能力来预测动作。优势是语言理解强，但不会"想象未来"（没有 video generation）。
> - **VGM → Action 路线** [19, 21, 62]：UniSim、ViDAR 等，先生成未来视频帧，再用 IDM 从帧间推断动作。优势是有物理先验，但缺乏语言指令理解。
> - **VLA + IDM 混合** [$`\mathcal{F}_1`$]：让 VLA 先"想象"未来图像，再用 IDM 推动作。但它不包含 World Model 和纯 VGM 功能，统一得不彻底。
> - **从零统一** [UWM]：把 5 种分布全部统一在一个 diffusion backbone 里。但从零训练，缺乏 VLM/VGM 预训练先验。
>
> **Motus 的差异化**：在 UWM 的"统一五分布"基础上，进一步引入 internet-scale 通用预训练先验（VLM + VGM）和大规模机器人轨迹的专用先验，不用从零开始训。

---

## 2.2. Latent Action Models

Latent actions mitigate the scarcity of action labels by capturing visual dynamics, and are typically derived by coupling IDMs with forward dynamics models (FDMs) to reconstruct the next frame conditioned on the previous one [9, 10, 20, 37]. Initially, RGB images are used for supervision, but this introduces task-irrelevant appearance information [58]. To remove such interference, a common approach is restricting autoencoder's capacity to encode low-dimensional latents [15, 38, 55], thereby reducing the inclusion of redundancy. AdaWorld [22] attempts to decouple the representations, such as $`\beta`$-VAE [23], in order to retain only the useful factors. Other approaches explore alternative reconstruction objectives, e.g., DINOv2 features [11, 15, 50], object keypoints [17, 51, 57], and language instructions [16], which carries rich semantic and spatial features. Moreover, LAOM [34] employs a few action labels to encourage the model to focus on robotic activities. Building on these advances and inspired by optical flow as a universal motion expression [12, 46, 61], we use it to align cross-embodiment behaviors and learn latent actions to facilitate large-scale pretraining.

> **通俗解读**：
>
> Latent Action 要解决的核心问题：**大量视频数据没有动作标签，怎么从中提取"动作信息"？**
>
> 基本思路是 **IDM + FDM**：用逆动力学模型（IDM）从相邻两帧推断"发生了什么动作"，用前向动力学模型（FDM）验证"给定这个动作能否重建下一帧"。两者配合训练出 latent action 编码器。
>
> **演进路线**（重建目标越来越好）：
> 1. **RGB 重建** [Genie, LAPA]：直接重建下一帧像素。问题——包含大量与任务无关的外观信息（背景、光照），latent action 会"记住"无用的东西。
> 2. **限制容量**：用低维 bottleneck 或 $`\beta`$-VAE 强制压缩，逼迫模型只编码关键运动信息。
> 3. **特征重建** [MOTO, UniVLA]：不重建 RGB，改为重建 DINOv2 特征、物体关键点、语言指令等。语义信息更丰富，但缺少像素级的运动细节。
> 4. **少量动作标签辅助** [LAOM]：用少量有标注数据引导模型关注机器人动作相关的模式。
>
> **Motus 的创新**：用**光流（optical flow）**作为重建目标。光流天然编码了"每个像素从哪里移动到哪里"，是一种通用的运动表达：
> - 不含外观信息（只有位移，没有颜色/纹理）
> - 保留像素级运动细节（比 DINOv2 特征更精细）
> - 跨 embodiment 通用（人手移动物体和机械臂移动物体的光流模式可以对齐）

---

## Related Work 总结

| 类别 | 代表方法 | 核心思路 | Motus 的区别 |
|------|---------|---------|-------------|
| VLM → VLA | OpenVLA, $`\pi_{0.5}`$, GR00T | VLM 图文理解 → 预测 action | 缺 video generation 先验，不会"想象未来" |
| VGM → Action | UniSim, ViDAR | 生成未来视频 → IDM 推 action | 缺 language 理解能力 |
| VLA + IDM | $`\mathcal{F}_1`$ | VLA 想象未来 + IDM 推动作 | 不含 WM 和 VGM，统一不彻底 |
| Unified (from scratch) | UWM | 单一 diffusion backbone 统一 5 种分布 | 从零训练，缺 VLM/VGM 预训练先验 |
| Latent action: RGB 重建 | Genie, LAPA | IDM + FDM 重建 RGB 下一帧 | 含大量无关外观信息 |
| Latent action: 特征重建 | MOTO, UniVLA | 重建 DINOv2/keypoint/language | 语义好但缺像素级运动细节 |
| **Motus（本文）** | — | MoT 统一三专家 + 光流 latent action | 统一 5 种分布 + 预训练先验 + 跨 embodiment 光流对齐 |

> **一句话小结**：Motus 同时解决了两个方向的痛点——在"统一建模"上引入预训练先验（不从零训），在"latent action"上用光流替代 RGB/特征重建（更精确、更通用）。
