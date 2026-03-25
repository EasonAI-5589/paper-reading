[← 返回 README](../README.md)

# 3. Method

## 📌 预览

方法章节分三部分：① 问题形式化（WAM 的两因素分解）→ ② Fast-WAM 架构（MoT + 结构化注意力掩码 + Flow Matching 训练）→ ③ 控制变体设计（实验的关键）。整体思路是"先提问，再设计，最后构造对照组"。

---

## 3.1 Problem Formulation

### 标准 VLA 策略

$$p(a_{1:H} | o, l) \tag{1}$$

直接从观测 $o$ 和指令 $l$ 映射到动作序列 $a_{1:H}$。

### 现有 WAM 的 imagine-then-execute 形式

$$p(a_{1:H} | o, l) = \int p(v_{1:T} | o, l) \cdot p(a_{1:H} | o, l, v_{1:T}) \, dv_{1:T} \tag{2}$$

先预测未来观测 $v_{1:T}$，再基于想象的未来生成动作。

> 💡 **两因素分析**: WAM 的有效性可能来自两个独立因素：
> - **(i) 训练时的视频预测目标** → 帮助模型学到物理上有意义的 latent 表征
> - **(ii) 推理时的显式未来生成** → 为动作预测提供额外前瞻信息
>
> 现有 WAM 把这两个因素耦合在一起。**Fast-WAM 的核心就是解耦它们。**

### Fast-WAM 的形式

$$p_\theta(a_{1:H} | o, l) = p_\theta(a_{1:H} | z(o, l)) \tag{3-4}$$

其中 $z(o, l)$ 是视频骨干网络的 latent world representation，通过**单次前向编码**得到，而非通过采样/去噪未来观测 $v_{1:T}$。

> 💡 **形式上 Fast-WAM 和 VLA 一样**——都是 obs→action 的直接映射。区别在于 Fast-WAM 的编码器是用视频预测目标训练过的，所以其 latent space 包含了世界动力学知识。

---

## 3.2 Model Architecture

### 整体设计

Fast-WAM 基于 Mixture-of-Transformer (MoT) 架构，共享注意力机制：

```
输入 Token 三组:
├── 🟢 Clean latent tokens（第一帧观测）→ 共享 visual anchor
├── 🔴 Noisy latent tokens（未来视频帧）→ 仅训练时使用
└── 🔵 Action tokens → Action Expert 处理
```

### 架构组件

| 组件 | 来源 | 说明 |
|------|------|------|
| **Video DiT** | Wan2.2-5B (预训练) | 世界建模骨干，5B 参数 |
| **Text Encoder** | Wan2.2 内置 T5 | 编码语言指令，cross-attention 提供给所有 tokens |
| **Video VAE** | Wan2.2 预训练 | 视觉观测 → latent tokens；多相机图像拼接后输入 |
| **Action Expert DiT** | 新训练 | 与 Video DiT 同架构，隐层 $d_a=1024$，约 1B 参数 |

### 结构化注意力掩码（Figure 2b — 全文最关键的设计）

**训练时**:

|  | f0 (clean) | f1..fh (noisy) | a1..ah (action) |
|---|:---:|:---:|:---:|
| **f0 (clean)** | ✅ self | ❌ | ❌ |
| **f1..fh (noisy)** | ✅ attend | ✅ 双向 | ❌ |
| **a1..ah (action)** | ✅ attend | ❌ | ✅ 双向 |

**推理时** — 去掉整个 noisy video 分支:

|  | f0 (clean) | a1..ah (action) |
|---|:---:|:---:|
| **f0 (clean)** | ✅ self | ❌ |
| **a1..ah (action)** | ✅ attend | ✅ 双向 |

> 💡 **注意力掩码是整个方法的精髓**:
>
> 1. Action tokens **不能**看到未来视频 tokens → 防止未来信息泄露到动作分支
> 2. Clean 第一帧 tokens **不看**任何其他 tokens → 保持纯净的视觉锚点
> 3. 推理时直接删掉 noisy video tokens → 无需任何架构修改即可切换到高效模式
>
> 这个设计确保了：训练时视频目标帮助 Video DiT 学到好的世界表征，但这些表征通过第一帧 clean tokens 的 KV cache 传递给 Action Expert，而非通过未来帧。

### 推理流程

```
1. 当前帧 → VAE → clean latent tokens
2. Clean tokens → Video DiT 单次前向传播 → latent world features (KV cache)
3. Action noise → Action Expert → 10 步去噪（attend to clean tokens KV cache）→ 动作 chunk
```

**不实例化任何 noisy video tokens，不做任何视频去噪。** 因此时延从 810ms (IDM) 降到 190ms。

---

## 3.3 Training Objective

标准 Flow Matching 目标。给定目标变量 $y$（动作 chunk 或视频 latents）:

**插值构造**:
$$y_t = (1-t)y + t\epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{5}$$

**速度场预测**:
$$\mathcal{L}_{FM}(y) = \mathbb{E}_{y, \epsilon, t} \left[ \|f_\theta(y_t, t, o, l) - (\epsilon - y)\|_2^2 \right] \tag{6}$$

分别对动作和视频实例化:

$$\mathcal{L}_{act} = \mathcal{L}_{FM}(a_{1:H}) \tag{7}$$
$$\mathcal{L}_{vid} = \mathcal{L}_{FM}(z_{1:T}) \tag{8}$$

**总训练目标**:

$$\mathcal{L} = \mathcal{L}_{act} + \lambda \mathcal{L}_{vid} \tag{9}$$

> 💡 $\lambda$ 控制视频联合训练的权重。论文没明确报告 $\lambda$ 的值——这是一个缺失的消融。噪声调度使用 logit-normal 分布（与 Wan2.2 一致），推理用 10 步去噪 + CFG scale 1.0。

---

## 3.4 Controlled Variants（控制变体设计）

### 实验设计的核心逻辑

| 变体 | 训练时视频目标 | 推理时生成未来 | 对应 WAM 范式 |
|------|:---:|:---:|------|
| **Fast-WAM** | ✅ | ❌ | 本文提出 |
| **Fast-WAM-Joint** | ✅ | ✅（联合去噪） | WAM [4], Motus [5] |
| **Fast-WAM-IDM** | ✅ | ✅（先视频后动作） | LingBot-VA [3], ViDAR [7] |
| **Fast-WAM w.o. video co-train** | ❌ | ❌ | 消融控制组 |

> 💡 **实验设计的精巧之处**:
>
> 所有变体共享同一实现框架（骨干、tokenization、训练配方、数据），只改变一个因素：
> - Fast-WAM vs Joint/IDM → 改变的是**推理方式**（是否生成未来）
> - Fast-WAM vs w.o. video co-train → 改变的是**训练目标**（是否有视频损失）
>
> 预期结果解读：
> - 如果推理时未来想象很重要 → Fast-WAM 应该明显差于 Joint/IDM
> - 如果训练时视频建模很重要 → w.o. video co-train 应该明显差于 Fast-WAM
>
> **实际结果：后者成立，前者不成立。** → 训练时视频建模是核心。

### 具体实现细节

- **Fast-WAM-Joint**: 允许 video tokens 和 action tokens 之间的双向注意力
- **Fast-WAM-IDM**: 按 LingBot-VA [3] 的做法，对 GT 视频 tokens 以 $p = 0.5$ 概率加噪声增强（noise augmentation）
- **Fast-WAM w.o. video co-train**: 去掉 $\mathcal{L}_{vid}$，只保留 $\mathcal{L}_{act}$，其他全部不变
