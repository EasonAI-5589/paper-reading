# Diffusion Models

## 目录结构

```
Diffusion/
├── Base-Models/
│   ├── Video/           # 视频生成基础模型
│   │   ├── Wan/         # 阿里 Wan 系列
│   │   ├── Sora/        # OpenAI
│   │   ├── CogVideo/    # 智谱
│   │   └── HunyuanVideo/# 腾讯
│   └── Image/           # 图像生成基础模型
│       ├── SDXL/
│       ├── FLUX/
│       └── SD3/
├── Efficiency/          # 效率优化方法
│   ├── Distillation/    # 步数蒸馏
│   ├── Quantization/    # 量化加速
│   └── Attention/       # 注意力优化
└── pdfs/
```

---

## 论文索引

### Base Models - Video

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| Wan 2.1 Technical Report | 2025-03 | [2503.20314](https://arxiv.org/abs/2503.20314) | [notes](Base-Models/Video/wan-notes.md) ✅ |
| Wan 2.2 (MoE) | 2025-07 | - | - |

### Base Models - Image

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| FLUX.1 | 2024 | - | - |
| Stable Diffusion 3 | 2024 | [2403.03206](https://arxiv.org/abs/2403.03206) | - |

### Efficiency - Distillation

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| TurboDiffusion (100-200x) | 2025-12 | [2512.16093](https://arxiv.org/abs/2512.16093) | [notes](Efficiency/Distillation/turbodiffusion.md) ✅ |
| Motion Consistency Model | 2024-06 | [2406.06890](https://arxiv.org/abs/2406.06890) | - |
| Consistency Models | 2023-03 | [2303.01469](https://arxiv.org/abs/2303.01469) | - |
| DMD (1-step) | 2024 CVPR | - | - |
| LCM (2-4 steps) | 2023 | [2310.04378](https://arxiv.org/abs/2310.04378) | - |

### Efficiency - Quantization

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| ViDiT-Q (W8A8/W4A8) | 2025 ICLR | [2406.02540](https://arxiv.org/abs/2406.02540) | - |
| QuantSparse | 2025 | [2509.23681](https://arxiv.org/abs/2509.23681) | - |
| TFMQ-DM | 2024 CVPR | - | - |

### Efficiency - Attention

| Paper | Date | arXiv | Notes |
|-------|------|-------|-------|
| SLA (Sparse-Linear Attention) | 2025 | [2509.24006](https://arxiv.org/abs/2509.24006) | - |
| FlashAttention-2 | 2023 | [2307.08691](https://arxiv.org/abs/2307.08691) | - |
| SageAttention | 2024 | - | - |
| AsymRnR (Token Reduction) | 2024 | [2412.11706](https://arxiv.org/abs/2412.11706) | - |

---

## 效率优化方法总览

### 1. 步数蒸馏 (Step Distillation)

将多步采样（50-100步）压缩到少步（1-4步）：

| 方法 | 步数 | 核心思想 | 代表工作 |
|------|------|----------|----------|
| Consistency Model | 1-2 | 直接预测终点 | CM, LCM, MCM |
| Distribution Matching | 1 | 分布匹配蒸馏 | DMD |
| Trajectory Matching | 4-8 | 轨迹分布匹配 | TDM |
| Rectified Flow | 1-4 | 直线化轨迹 | InstaFlow |

### 2. 量化加速 (Quantization)

降低精度以加速计算：

| 精度 | 方法 | 加速比 | 质量损失 |
|------|------|--------|----------|
| W8A8 | ViDiT-Q | ~2x | 极小 |
| W4A8 | QuantSparse | ~3x | 小 |
| FP8 | Native | ~1.5x | 无 |

### 3. 注意力优化 (Attention)

降低 O(n²) 注意力复杂度：

| 方法 | 类型 | 加速比 | 特点 |
|------|------|--------|------|
| FlashAttention | IO优化 | 2-4x | 精确计算 |
| SageAttention | 量化+IO | 3-5x | INT8 加速 |
| Sparse Attention | 稀疏 | 2-10x | 跳过不重要token |
| Linear Attention | 线性 | O(n) | 近似计算 |

### 4. 综合方案

TurboDiffusion = SLA + rCM + W8A8 → **100-200x 加速**

---

## 与 VLA/Robotics 的关联

Diffusion 效率优化对机器人策略很重要：

- **Diffusion Policy**: 需要实时推理（<100ms）
- **Flow Matching**: π₀ 使用 Flow Matching 替代 Diffusion
- **边缘部署**: 机器人显存有限，需要量化

---

*Last updated: 2025-12-26*
