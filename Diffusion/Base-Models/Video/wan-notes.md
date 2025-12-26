# Wan 系列视频生成模型

## 论文信息

| 版本 | 发布时间 | arXiv | 核心特性 |
|------|----------|-------|----------|
| **Wan 2.1** | 2025-02 | [2503.20314](https://arxiv.org/abs/2503.20314) | 基础 DiT + Flow Matching |
| **Wan 2.1-VACE** | 2025-05 | - | 视频编辑 All-in-one |
| **Wan 2.2** | 2025-07 | - | MoE 架构（首个开源） |
| **Wan 2.2-S2V** | 2025-08 | [2508.18621](https://arxiv.org/abs/2508.18621) | 音频驱动视频生成 |

**GitHub**: https://github.com/Wan-Video/Wan2.1 / https://github.com/Wan-Video/Wan2.2

**Technical Report**: 60 pages, 33 figures

---

## Section 1: Motivation & Problem Definition

### 1.1 研究问题定义

#### 核心任务
> 构建开源的、高质量的视频生成基础模型，在性能上达到甚至超越闭源商业模型（如 Sora）。

#### 问题范畴
- 视频生成（Text-to-Video, Image-to-Video）
- 视频编辑
- 长视频生成

#### 现有方法的问题

##### 问题1：开源模型性能落后
- **现有做法**: CogVideo、Open-Sora 等开源模型
- **局限**: 与 Sora、Runway 等商业模型差距大
- **理想状态**: 开源模型达到商业级质量

##### 问题2：效率与质量的权衡
- **现有做法**: 增大模型获得更好质量
- **局限**: 推理慢，显存需求高
- **理想状态**: 小模型也能高质量

##### 问题3：缺乏统一的评估标准
- **现有做法**: 各家使用不同评估指标
- **局限**: 难以公平对比
- **理想状态**: 标准化自动评估

### 1.2 本文方法与核心创新

#### 总体方案
Wan 是一个完整的视频基础模型套件，基于 Diffusion Transformer (DiT) 范式构建。

#### 关键创新点

##### 1. 3D Causal VAE
- **是什么**: 新型时空压缩 VAE，支持无限长度 1080P 视频
- **解决的问题**: 高效编解码长视频

##### 2. Flow Matching + DiT
- **是什么**: 使用 Flow Matching 替代传统 DDPM
- **解决的问题**: 更直的采样轨迹，更少推理步数

##### 3. MoE 架构 (Wan 2.2)
- **是什么**: 高噪声/低噪声双专家设计
- **解决的问题**: 增大容量但不增加推理成本

##### 4. 大规模数据策展
- **是什么**: 数十亿图像+视频的训练数据
- **解决的问题**: 提升泛化能力

---

## Section 2: Related Work

### 2.1 视频生成模型

| 工作 | 核心方法 | 优势 | 局限 |
|------|----------|------|------|
| Sora | DiT + 时空patch | ✓ 高质量长视频 | ✗ 闭源 |
| CogVideo | 3D VAE + DiT | ✓ 开源 | ✗ 质量不足 |
| Open-Sora | 复现 Sora | ✓ 开源 | ✗ 训练数据少 |
| HunyuanVideo | Dual-stream DiT | ✓ 开源 | ✗ 效率问题 |

### 2.2 效率优化

| 工作 | 核心方法 | 优势 | 局限 |
|------|----------|------|------|
| LCM | 一致性蒸馏 | ✓ 4步生成 | ✗ 质量下降 |
| FlashAttention | IO优化 | ✓ 精确计算 | ✗ 加速有限 |

---

## Section 3: Method

### 3.0 Preliminary

#### 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 视频 latent | $T \times H \times W \times C$ |
| $t$ | 时间步 | $[0, 1]$ |
| $v_\theta$ | 速度场网络 | - |

#### Flow Matching 基础

传统 Diffusion:
$$
dx = f(x, t)dt + g(t)dW
$$

Flow Matching:
$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

**直觉理解**: Flow Matching 学习从噪声到数据的"速度场"，轨迹更直，步数更少。

### 3.1 3D Causal VAE

#### 核心思想
> 设计支持时序因果的 3D VAE，实现高压缩比的视频编解码。

#### 架构示意
```
Video (T×H×W×3) → Encoder3d → Latent (T'×H'×W'×C) → Decoder3d → Video
                    ↓
            CausalConv3d (保持时序因果)
```

#### 关键设计

**CausalConv3d**:
- 只看过去帧，不看未来帧
- 支持流式编解码
- 可处理任意长度视频

**压缩比**:
- 空间: 8x
- 时间: 4x (首帧 1x)

### 3.2 DiT 架构 (WanModel)

#### 核心思想
> 基于 Transformer 的扩散模型，支持多任务（T2V/I2V/VACE）。

#### 架构示意
```
┌─────────────────────────────────────────────────────────┐
│                      WanModel                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Video Latent → Patch Embed (Conv3d) → patches          │
│                                                         │
│  Time t → Sinusoidal → MLP → 6-dim modulation           │
│                                                         │
│  Text → T5 Encoder → Cross-Attention                    │
│                                                         │
│  patches + time + text → N × WanAttentionBlock → Head   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### WanAttentionBlock

```python
class WanAttentionBlock:
    - self_attn: WanSelfAttention (3D RoPE + Flash Attention)
    - cross_attn: WanT2VCrossAttention / WanI2VCrossAttention
    - ffn: Feed-Forward Network
```

**3D RoPE**: 时空位置编码，分别对 T/H/W 维度编码

### 3.3 MoE 架构 (Wan 2.2)

#### 核心思想
> 使用双专家设计，高噪声阶段和低噪声阶段使用不同专家。

#### 架构示意
```
              ┌──────────────────────────┐
              │   High-Noise Expert      │
              │   (Layout Focus)         │ ← t ≥ boundary
              │   14B params             │
              └──────────────────────────┘
                          │
   Timestep ──→ Router ───┤
                          │
              ┌──────────────────────────┐
              │   Low-Noise Expert       │
              │   (Detail Refinement)    │ ← t < boundary
              │   14B params             │
              └──────────────────────────┘
```

#### 关键公式

**路由机制**:
$$
\text{Expert} = \begin{cases}
\text{High-Noise} & \text{if } t \geq \text{boundary} \\
\text{Low-Noise} & \text{if } t < \text{boundary}
\end{cases}
$$

**效率分析**:
- 总参数: 27B
- 每步激活: 14B
- 推理成本: 与单个 14B 模型相同

### 3.4 训练策略

#### 损失函数

Flow Matching Loss:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

#### 训练配置

| 配置项 | 取值 |
|--------|------|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| 数据规模 | 数十亿图像+视频 |
| 训练卡数 | 数千 H100 |

### 3.Y 方法总结

#### Pipeline 流程

| 阶段 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 1 | T5 Encoder | Text | Text Embeddings |
| 2 | 3D VAE Encoder | Video | Latent |
| 3 | DiT Denoising | Noise + Condition | Clean Latent |
| 4 | 3D VAE Decoder | Latent | Video |

---

## Section 4: Experiments

### 4.1 主要结果

#### 实验设置
- **Benchmark**: VBench, EvalCrafter, 内部评估
- **对比方法**: Sora, Runway Gen-3, Kling, CogVideo

#### 关键发现

| 对比项 | 数据 | 说明 |
|--------|------|------|
| Wan 14B vs Open-Sora | 显著领先 | 开源 SOTA |
| Wan 14B vs Sora | 接近/部分超越 | 达到商业级 |
| Wan 1.3B vs 14B | 质量略低但可用 | 8GB 可运行 |

### 4.2 MoE 消融 (Wan 2.2)

| 配置 | Validation Loss | 结论 |
|------|-----------------|------|
| Dense 14B | baseline | - |
| MoE 27B (A14B) | 最低 | MoE 有效 |
| 单专家 27B | 高于 MoE | MoE 设计合理 |

**关键洞察**: MoE 的双专家设计比单纯增大模型更有效。

### 4.3 效率分析

| 模型 | 参数量 | VRAM | 生成时间 (5s视频) |
|------|--------|------|-------------------|
| Wan 1.3B | 1.3B | 8.19GB | 快 |
| Wan 14B | 14B | ~40GB | 中等 |
| Wan 2.2 MoE | 27B (A14B) | ~40GB | 中等 |

---

## Section 5: Takeaways

### 核心贡献
1. 开源达到商业级质量的视频生成模型
2. 首个开源 MoE 视频生成模型
3. 完整的技术报告和代码

### 对 Efficiency 研究的启发
- **MoE 路由**: 基于噪声水平的专家切换是新思路
- **Flow Matching**: 比传统 Diffusion 更易加速
- **1.3B 小模型**: 证明小模型也能高质量

### 与其他工作的联系
- **TurboDiffusion**: 在 Wan 上实现 100-200x 加速
- **π₀**: 使用类似的 Flow Matching 架构

---

## BibTeX

```bibtex
@article{wan2025wan,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={{Team Wan}},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}
```
