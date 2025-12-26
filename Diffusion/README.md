# Diffusion Models

## 目录结构

```
Diffusion/
├── Base-Models/
│   └── Video/           # 视频生成基础模型
│       └── wan-notes.md # Wan 2.1/2.2 笔记
├── Efficiency/          # 效率优化方法（综合）
│   └── turbodiffusion.md
└── pdfs/
```

---

## 论文索引

### Base Models - Video

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| Wan 2.1 Technical Report | 2025-03 | [2503.20314](https://arxiv.org/abs/2503.20314) | [notes](Base-Models/Video/wan-notes.md) ✅ |
| Wan 2.2 (MoE) | 2025-07 | - | - |
| Sora | 2024-02 | - | - |
| CogVideo | 2024 | - | - |
| HunyuanVideo | 2024 | - | - |

### Base Models - Image

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| FLUX.1 | 2024 | - | - |
| Stable Diffusion 3 | 2024 | [2403.03206](https://arxiv.org/abs/2403.03206) | - |

### Efficiency

> 效率优化方法往往融合多种技术（蒸馏+量化+注意力），不再细分子目录。

| Paper | Date | arXiv | 涉及技术 | Notes |
|-------|------|-------|----------|-------|
| **TurboDiffusion** | 2025-12 | [2512.16093](https://arxiv.org/abs/2512.16093) | rCM + SLA + W8A8 | [notes](Efficiency/turbodiffusion.md) ✅ |
| SageAttention V1/V2/V3 | 2024-2025 | - | Attention 量化 | TurboDiffusion 依赖 |
| SLA (Sparse-Linear Attention) | 2025 | - | 稀疏注意力 | TurboDiffusion 依赖 |
| rCM | 2025 | - | 步数蒸馏 | TurboDiffusion 依赖 |
| Motion Consistency Model | 2024-06 | [2406.06890](https://arxiv.org/abs/2406.06890) | 步数蒸馏 | - |
| Consistency Models | 2023-03 | [2303.01469](https://arxiv.org/abs/2303.01469) | 步数蒸馏 | - |
| LCM | 2023 | [2310.04378](https://arxiv.org/abs/2310.04378) | 步数蒸馏 | - |
| DMD | 2024 CVPR | - | 步数蒸馏 | - |
| ViDiT-Q | 2025 ICLR | [2406.02540](https://arxiv.org/abs/2406.02540) | 量化 | - |
| FlashAttention-2 | 2023 | [2307.08691](https://arxiv.org/abs/2307.08691) | 注意力 IO | - |

---

## 效率优化技术分类

### 作用于 Pipeline 的不同环节

```
┌─────────────────────────────────────────────────────────────┐
│                 Diffusion Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Text Encoder (T5/CLIP)  ← 通常不优化                        │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DiT / UNet                              │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │ Attention Layers                              │  │   │
│  │  │  • SageAttention (INT8 量化)                  │  │   │
│  │  │  • SLA (稀疏注意力)                           │  │   │
│  │  │  • FlashAttention (IO 优化)                   │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │ Linear Layers                                 │  │   │
│  │  │  • W8A8 / W4A8 量化                           │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  Sampler (100步 → 4步)                                      │
│  • rCM / Consistency Models / LCM / DMD                     │
│       ↓                                                     │
│  VAE Decoder  ← 通常不优化                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 各技术加速贡献

| 技术类型 | 代表方法 | 加速比 | 质量影响 |
|----------|----------|--------|----------|
| **步数蒸馏** | rCM, CM, LCM | **25x** | 需训练 |
| 注意力稀疏 | SLA | 3-5x | 需训练 |
| 注意力量化 | SageAttention | 2-3x | 无损 |
| Linear 量化 | W8A8 | 1.5-2x | 极小 |
| IO 优化 | FlashAttention | 2-4x | 无损 |

### 综合方案

**TurboDiffusion** = rCM + SLA + SageAttention + W8A8 → **100-200x 加速**

---

## 与 VLA/Robotics 的关联

Diffusion 效率优化对机器人策略很重要：

- **Diffusion Policy**: 需要实时推理（<100ms）
- **Flow Matching**: π₀ 使用 Flow Matching 替代 Diffusion
- **边缘部署**: 机器人显存有限，需要量化

---

*Last updated: 2025-12-26*
