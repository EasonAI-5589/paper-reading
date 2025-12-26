# Wan 2.2: 首个开源 MoE 视频生成模型

## 模型信息

| 项目 | 内容 |
|------|------|
| **发布时间** | 2025-07 |
| **arXiv** | 无独立论文，技术说明在 GitHub |
| **GitHub** | https://github.com/Wan-Video/Wan2.2 |
| **HuggingFace** | https://huggingface.co/Wan-AI |
| **官网** | https://wan.video/research-and-open-source |

### 模型变体

| 变体 | 参数量 | 激活参数 | 任务 |
|------|--------|----------|------|
| Wan2.2-T2V-A14B | 27B | 14B | 文生视频 |
| Wan2.2-I2V-A14B | 27B | 14B | 图生视频 |
| Wan2.2-S2V-14B | 14B | 14B | 音频驱动视频 (2025-08) |

---

## 核心创新：MoE 架构

### 与 Wan 2.1 的区别

| 特性 | Wan 2.1 | Wan 2.2 |
|------|---------|---------|
| 架构 | Dense DiT | **MoE DiT** |
| 参数量 | 14B | 27B (A14B) |
| 推理成本 | 14B | 14B（相同） |
| 质量 | 高 | **更高** |

### MoE 设计：双专家路由

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

### 路由机制

$$
\text{Expert} = \begin{cases}
\text{High-Noise Expert} & \text{if } t \geq \text{boundary} \\
\text{Low-Noise Expert} & \text{if } t < \text{boundary}
\end{cases}
$$

### 设计直觉

| 阶段 | 专家 | 任务 |
|------|------|------|
| 高噪声 (t 大) | High-Noise Expert | 生成整体布局、构图 |
| 低噪声 (t 小) | Low-Noise Expert | 细化细节、纹理 |

**核心洞察**: 去噪过程的不同阶段需要不同的能力，专门化的专家比通用模型更有效。

### 效率分析

| 指标 | 数值 |
|------|------|
| 总参数 | 27B |
| 每步激活 | 14B |
| 显存需求 | ~40GB（与 Wan 2.1 14B 相同） |
| 推理速度 | 与 Wan 2.1 14B 相同 |

---

## 消融实验

| 配置 | Validation Loss | 结论 |
|------|-----------------|------|
| Dense 14B | baseline | - |
| MoE 27B (A14B) | 最低 | ✓ MoE 有效 |
| 单专家 27B | 高于 MoE | MoE 设计合理 |

**关键洞察**: MoE 的双专家设计比单纯增大模型更有效。

---

## 特性改进

相比 Wan 2.1，Wan 2.2 改进：

1. **更稳定的视频合成** - 减少不真实的相机移动
2. **更好的风格化场景支持** - 多样化艺术风格
3. **更高的生成质量** - 细节更丰富

---

## Wan 2.2-S2V：音频驱动视频

| 项目 | 内容 |
|------|------|
| **发布时间** | 2025-08-26 |
| **arXiv** | [2508.18621](https://arxiv.org/abs/2508.18621) |
| **任务** | 音频 + 图像 → 视频 |

音频驱动的电影级视频生成，是 Wan 2.2 系列的扩展。

---

## 与 TurboDiffusion 的关系

TurboDiffusion 支持 Wan 2.2：

| 模型 | 原始时间 | 加速后 | 加速比 |
|------|----------|--------|--------|
| Wan2.2-I2V-14B (720P) | - | 35.4s | 119x |

---

## Takeaways

### 核心贡献
1. **首个开源 MoE 视频生成模型**
2. **不增加推理成本的情况下提升质量**
3. 基于时间步的专家路由是新思路

### 对 Efficiency 研究的启发
- MoE 是增大容量但不增加推理成本的有效方法
- 基于噪声水平的专家切换可能比传统 token-level MoE 更适合 Diffusion

---

## 相关链接

- Wan 2.1 笔记: [wan2.1-notes.md](wan2.1-notes.md)
- TurboDiffusion 笔记: [../../Efficiency/turbodiffusion.md](../../Efficiency/turbodiffusion.md)
