[← 返回 README](../README.md)

# 3. Method

## 📌 预览
Method 分四个子模块：(1) 问题定义，(2) LaST₀ 整体架构，(3) Latent Spatio-Temporal CoT 的构造，(4) MoT 双系统的异步频率协调，(5) 训练策略。

---

## 3.1 问题定义

> We formulate the robot manipulation task as a probabilistic sequence decision-making problem. At each timestep t, the policy receives a natural language instruction l_t and visual observations I_t... The objective of the VLA model π_θ is to generate an optimal action sequence a_{t:t+H}.

> 💡 **标准 VLA 公式**：`(语言指令, 视觉观测) → 动作序列`，7-DoF（位置 3 + 旋转 3 + 夹爪 1）。

---

## 3.2 LaST₀ 架构

### 视觉编码器
> For each input RGB observation, we employ SigLIP-Large to extract semantic features... these encoded features serve a dual purpose: the current frame acts as real-time contextual input to the MoT experts, while future frames provide ground-truth target embeddings for the visual component of the latent CoT.

> 💡 **SigLIP-Large 双用途**：当前帧 → 输入给 expert；未来帧 → 作为 latent CoT 的监督 target。

---

### 点云编码器（仅训练时）
> We integrate a large-scale, pretrained point cloud encoder, Uni3D, which explicitly captures object geometry and spatial knowledge. Note that, unlike the vision encoder, the point cloud encoder is not used during inference. Instead, Uni3D is solely employed to encode ground-truth point clouds into compact 3D feature representations within the latent CoT space.

> 💡 **推理时不需要点云！** Uni3D 只在训练时提供几何监督信号，推理时慢 expert 直接预测几何 latent（学会了就不需要原始点云了）。这是个很实用的设计——部署时不需要深度传感器。

---

### MoT 骨干网络
> We transform the standard decoder-only transformer into a unified MoT architecture. Unlike conventional transformers that apply a homogeneous set of weights to all tokens, our MoT design introduces task-specific parameter sets for all non-embedding components, including FFN, attention projections (W_Q, W_K, W_V, W_O), and Layer Normalizations, while maintaining a shared global self-attention context.

> 💡 **MoT 的实现方式**：
> ```
> 原始 Transformer (24层):
>   每层: Self-Attn → FFN
>
> MoT:
>   每层: Shared Self-Attn → {FFN_slow 或 FFN_fast}
>   + 独立的 Attn 投影 (WQ/WK/WV/WO) per expert
>   + 独立的 LayerNorm per expert
>
> 共享维度: d=2048（DeepSeek-LLM 1.5B）
> ```
> **关键**：共享 self-attention 让两个 expert 在同一个 KV 空间交互，快 expert 能直接读慢 expert 生成的 latent tokens。

---

## 3.3 Latent Spatio-Temporal CoT

### Latent Embedding 构造
> For each future timestep k ∈ {1,...,H}, we extract features from three complementary modalities:
> - Future RGB frames I_{t+k} → visual latents z^v_k via SigLIP-Large
> - Future point clouds P_{t+k} → geometric latents z^p_k via Uni3D
> - Future robot states s_{t+k} → proprioceptive latents z^s_k via action tokenizer

> 💡 **三种 latent 的互补性**：
> | 模态 | 编码内容 | 对操作的价值 |
> |------|---------|------------|
> | 视觉 z^v | 语义（物体类别、位置） | "什么东西在哪" |
> | 点云 z^p | 3D 几何（形状、深度） | "物体的精确空间结构" |
> | 本体 z^s | 关节角度/速度 | "机器人自身状态" |

---

> To ensure high inference efficiency, we apply average pooling to compress the feature maps of each modality into a single representative token.

> 💡 **1 token per modality**：平均池化到 1 个 token，三种模态每帧共 3 个 token，H=4 帧共 12 个 token。极其紧凑，后面消融实验证明 1 token 已经够用。

---

> We then organize these tokens in an interleaved, chronological order:
> Z_GT = [z^v_1, z^p_1, z^s_1, z^v_2, z^p_2, z^s_2, ..., z^v_H, z^p_H, z^s_H]

> 💡 **时序交错排列**：同一时刻的三模态紧挨着，保留时间因果性，鼓励模型学跨模态耦合动态。

---

### 序列结构与特殊 Token
> We introduce three special tokens: `<latent_start>`, `<latent_end>`, and `<latent_pad>`.
> - **训练时**：用 Z_GT 替换 `<latent_pad>` 占位符 → teacher forcing
> - **推理时**：慢 expert 自回归填充 `<latent_pad>` 位置

> 💡 **训练 vs 推理**：
> ```
> 训练:  [<latent_start>, z^v_1_GT, z^p_1_GT, z^s_1_GT, ..., <latent_end>]
> 推理:  [<latent_start>, z^v_1_pred, z^p_1_pred, z^s_1_pred, ..., <latent_end>]
>                                ↑ 自回归逐步生成
> ```

---

### Latent 监督策略
> We train the slow reasoning expert using continuous latent regression rather than discrete token likelihoods. The loss is defined as cosine similarity:
>
> L_latent = Σ_t (1 - ẑ_t · z^GT_t / (||ẑ_t|| ||z^GT_t||))

> 💡 **为什么用 cosine loss 而不是 MSE**：
> - latent 向量的方向（语义）比大小（magnitude）更重要
> - cosine loss 只优化方向对齐，对尺度不敏感
> - 与 CLIP、DINO 等 contrastive 方法一脉相承

---

## 3.4 双系统频率协调

> We introduce an asynchronous frequency mechanism... using a set of update ratios κ (e.g., κ∈{2,4,8}).
> - **慢 expert**：在 t mod κ = 0 时激活，生成 latent CoT
> - **快 expert**：每步激活，基于最新 latent CoT + 当前观测生成 action
> - 两次关键帧之间：慢 expert 休眠，快 expert 用缓存的 latent CoT

> 💡 **异步频率的直觉**：
> ```
> t=0:  慢 expert 推理 → 生成 latent CoT [z1,z2,z3,z4]
> t=0:  快 expert 生成 action_0（基于 z1-z4）
> t=1:  快 expert 生成 action_1（仍用 z1-z4，慢 expert 休眠）
> t=2:  快 expert 生成 action_2（仍用 z1-z4）
> t=3:  快 expert 生成 action_3（仍用 z1-z4）
> t=4:  慢 expert 推理 → 更新 latent CoT [z5,z6,z7,z8]
> ...
> κ=4 时：慢 expert 4步一次，快 expert 每步一次
> ```

---

## 3.5 训练策略

### 大规模预训练
- **数据**：400K+ 轨迹（Open-X-Embodiment, DROID, ROBOMIND 等）
- **目标**：让两个 expert 在多样化机器人数据上学到通用表征

### SFT 联合优化
> The action expert is trained with randomly mixed fast-slow operating ratios (e.g., 1:1, 1:2, 1:4), which exposes it to latent conditions updated at varying delays.

> 💡 **混合频率训练的好处**：
> - 快 expert 学会在不同"latent 新鲜度"下都能正常工作
> - 推理时可自由选择 κ，不需要重新训练
> - 实验证明混合训练（82%）优于固定比例（75-79%）
