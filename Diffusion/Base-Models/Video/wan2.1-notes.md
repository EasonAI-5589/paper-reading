# Wan 2.1: 开源视频生成基础模型

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Wan: Open and Advanced Large-Scale Video Generative Models |
| **作者** | Team Wan (阿里巴巴) |
| **发布时间** | 2025-03 |
| **arXiv** | [2503.20314](https://arxiv.org/abs/2503.20314) |
| **GitHub** | https://github.com/Wan-Video/Wan2.1 |
| **Technical Report** | 60 pages, 33 figures |

### 模型变体

| 变体 | 参数量 | 任务 | VRAM |
|------|--------|------|------|
| Wan2.1-T2V-1.3B | 1.3B | 文生视频 | 8.19GB |
| Wan2.1-T2V-14B | 14B | 文生视频 | ~40GB |
| Wan2.1-I2V-14B | 14B | 图生视频 | ~40GB |
| Wan2.1-VACE | 14B | 视频编辑 | ~40GB |

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

### 1.2 本文方法与核心创新

#### 总体方案
Wan 2.1 基于 Diffusion Transformer (DiT) + Flow Matching 范式构建。

#### 关键创新点

| 创新 | 说明 | 解决的问题 |
|------|------|------------|
| **3D Causal VAE** | 时空压缩 VAE，支持无限长度 1080P | 高效编解码长视频 |
| **Flow Matching** | 替代传统 DDPM | 更直的轨迹，更少步数 |
| **3D RoPE** | 时空位置编码 | 视频帧间关系建模 |

---

## Section 2: Related Work

| 工作 | 核心方法 | 优势 | 局限 |
|------|----------|------|------|
| Sora | DiT + 时空patch | ✓ 高质量长视频 | ✗ 闭源 |
| CogVideo | 3D VAE + DiT | ✓ 开源 | ✗ 质量不足 |
| Open-Sora | 复现 Sora | ✓ 开源 | ✗ 训练数据少 |
| HunyuanVideo | Dual-stream DiT | ✓ 开源 | ✗ 效率问题 |

---

## Section 3: Method

### 3.0 Preliminary: Flow Matching

传统 Diffusion (SDE):
$$
dx = f(x, t)dt + g(t)dW
$$

Flow Matching (ODE):
$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

**直觉理解**: Flow Matching 学习从噪声到数据的"速度场"，轨迹更直，步数更少。

### 3.1 3D Causal VAE

#### 核心思想
> 设计支持时序因果的 3D VAE，实现高压缩比的视频编解码。

#### 架构
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
| 维度 | 压缩比 |
|------|--------|
| 空间 (H, W) | 8x |
| 时间 (T) | 4x (首帧 1x) |

### 3.2 DiT 架构 (WanModel)

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

### 3.3 训练策略

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

### 3.Y Pipeline 总结

| 阶段 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 1 | T5 Encoder | Text | Text Embeddings |
| 2 | 3D VAE Encoder | Video | Latent |
| 3 | DiT Denoising | Noise + Condition | Clean Latent |
| 4 | 3D VAE Decoder | Latent | Video |

---

## Section 4: Experiments

### 4.1 主要结果

| 对比项 | 结果 | 说明 |
|--------|------|------|
| Wan 14B vs Open-Sora | 显著领先 | 开源 SOTA |
| Wan 14B vs Sora | 接近/部分超越 | 达到商业级 |
| Wan 1.3B vs 14B | 质量略低但可用 | 8GB 可运行 |

### 4.2 效率分析

| 模型 | 参数量 | VRAM | 备注 |
|------|--------|------|------|
| Wan 1.3B | 1.3B | 8.19GB | 消费级可用 |
| Wan 14B | 14B | ~40GB | 需专业卡 |

---

## Section 5: Takeaways

### 核心贡献
1. 开源达到商业级质量的视频生成模型
2. 完整的技术报告和代码
3. 1.3B 小模型证明效率与质量可兼得

### 与其他工作的联系
- **TurboDiffusion**: 在 Wan 上实现 100-200x 加速
- **π₀**: 使用类似的 Flow Matching 架构
- **Wan 2.2**: 后续 MoE 架构升级

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
