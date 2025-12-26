# TurboDiffusion: 100-200x 视频扩散加速框架

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | TurboDiffusion: Accelerating Video Diffusion by 100-200x |
| **作者** | Jintao Zhang et al. (清华大学机器学习组) |
| **发布时间** | 2025-12-18 |
| **arXiv** | [2512.16093](https://arxiv.org/abs/2512.16093) |
| **GitHub** | https://github.com/thu-ml/TurboDiffusion |

---

## Section 1: Motivation & Problem Definition

### 1.1 研究问题定义

#### 核心任务
> 在保持视频质量的前提下，将视频扩散模型的端到端生成速度提升 100-200 倍。

#### 问题范畴
- 视频扩散模型加速
- 注意力机制优化
- 步数蒸馏
- 模型量化

#### 现有方法的问题

##### 问题1：推理步数过多
- **现有做法**: DDPM/Flow Matching 需要 50-100+ 步采样
- **局限**: 每步都需要完整前向传播
- **理想状态**: 1-4 步高质量生成

##### 问题2：注意力计算复杂度高
- **现有做法**: Full Attention O(n²)
- **局限**: 长视频（77帧+）计算量爆炸
- **理想状态**: 稀疏或线性注意力

##### 问题3：模型参数量大
- **现有做法**: FP16/BF16 推理（14B+ 参数）
- **局限**: 显存占用高，推理慢
- **理想状态**: 低精度推理不损质量

### 1.2 本文方法与核心创新

#### 总体方案
TurboDiffusion 是一个综合加速框架，结合三大技术：
1. **SLA** (Sparse-Linear Attention) - 注意力加速
2. **rCM** (score-regularized Consistency Model) - 步数蒸馏
3. **W8A8** - 权重和激活量化

#### 加速效果
```
┌──────────────────────────────────────────────────────────────┐
│                    TurboDiffusion 加速效果                    │
├──────────────────────────────────────────────────────────────┤
│  模型                      │ 原始时间  │ 加速后   │ 加速比   │
├──────────────────────────────────────────────────────────────┤
│  Wan2.1-T2V-14B-720P      │ 4648s    │ 22.7s   │ 205x    │
│  Wan2.1-T2V-14B-480P      │ 1237s    │ 6.5s    │ 190x    │
│  Wan2.1-T2V-1.3B-480P     │ 121s     │ 1.8s    │ 67x     │
│  Wan2.2-I2V-14B-720P      │ 2560s    │ 35.4s   │ 72x     │
└──────────────────────────────────────────────────────────────┘
```

---

## Section 2: Related Work

### 2.1 步数蒸馏

| 工作 | 核心方法 | 步数 | 优势 | 局限 |
|------|----------|------|------|------|
| Consistency Model | ODE 轨迹一致性 | 1-2 | ✓ 极少步数 | ✗ 训练不稳定 |
| LCM | 引导蒸馏 | 2-4 | ✓ 稳定 | ✗ 质量略降 |
| DMD | 分布匹配 | 1 | ✓ 一步生成 | ✗ 复杂 |
| **rCM** | Score 正则化 | 1-4 | ✓ 稳定+高质量 | - |

### 2.2 注意力优化

| 工作 | 核心方法 | 加速比 | 优势 | 局限 |
|------|----------|--------|------|------|
| FlashAttention | IO 优化 | 2-4x | ✓ 精确 | ✗ 仍是 O(n²) |
| SageAttention | INT8 量化 | 3-5x | ✓ 快 | ✗ 仍是 O(n²) |
| Linear Attention | 核近似 | O(n) | ✓ 线性 | ✗ 质量损失大 |
| **SLA** | 稀疏+线性 | 高 | ✓ 质量保持 | - |

### 2.3 量化

| 工作 | 精度 | 目标 | 优势 | 局限 |
|------|------|------|------|------|
| ViDiT-Q | W8A8/W4A8 | DiT | ✓ 高压缩 | ✗ 视频支持有限 |
| TFMQ-DM | Mixed | Diffusion | ✓ 自适应 | ✗ 复杂 |
| **TurboDiffusion** | W8A8 | Video DiT | ✓ 简洁高效 | - |

---

## Section 3: Method

### 3.0 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TurboDiffusion 系统架构                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input                                                              │
│    ├── Text Prompt → T5 Encoder → Text Embeddings                  │
│    └── Image (I2V) → VAE Encoder → Latent                          │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   TurboDiffusion Core                        │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  DiT Model (Wan 14B/1.3B)                                   │   │
│  │    ├── SageSLA Attention (--attention_type sagesla)         │   │
│  │    │     └── top-k sparse selection (--sla_topk 0.1)        │   │
│  │    ├── W8A8 Quantized Linear (--quant_linear)               │   │
│  │    └── rCM Sampler (--num_steps 1-4)                        │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          ↓                                          │
│  Output                                                             │
│    └── VAE Decoder → Video (480p/720p, 77 frames)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Sparse-Linear Attention (SLA)

#### 核心思想
> 将 O(n²) 的 Full Attention 转化为稀疏计算，只选择 top-k 比例的 key 进行注意力。

#### 算法流程

**Step 1: Block Pooling**
```
Q, K, V ∈ R^{N×d}
Q_pool, K_pool = pool(Q, K)  # 分块池化
```

**Step 2: Score 计算与 Top-k 选择**
```python
# Smooth-k 技术：减去均值避免极端分布
K_smooth = K_pool - mean(K_pool)
pooled_score = Q_pool @ K_smooth.T

# 选择 top-k keys
lut, sparse_map = topk(pooled_score, k=top_k_ratio * N)
```

**Step 3: Sparse Attention**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{sparse}}^T}{\sqrt{d}}\right) V_{\text{sparse}}
$$

其中 $K_{\text{sparse}}, V_{\text{sparse}}$ 只包含 top-k 选中的位置。

#### 关键设计

**SageSLA = SLA + INT8 量化**
```python
class SageSparseLinearAttention:
    # 使用 SpargeAttn 库的硬件加速
    output = spas_sage_attn._qattn(Q_int8, K_int8, V_fp16, sparse_map)
```

**复杂度分析**:
| 方法 | 复杂度 | 77帧 720P (N≈10⁶) |
|------|--------|-------------------|
| Full Attention | O(n²) | 10¹² FLOPs |
| SLA (k=0.1) | O(k·n²) | 10¹¹ FLOPs |

### 3.2 rCM (Score-Regularized Consistency Model)

#### 核心思想
> 通过 Score 正则化，实现稳定的 1-4 步一致性蒸馏。

#### Consistency Model 回顾

传统 CM 目标：直接预测轨迹终点
$$
f_\theta(x_t, t) \approx x_0 \quad \forall t \in [0, T]
$$

问题：训练不稳定，尤其在高噪声区域。

#### rCM 改进

**Score 正则化**:
$$
\mathcal{L}_{\text{rCM}} = \mathcal{L}_{\text{CM}} + \lambda \cdot \mathbb{E}_{t, x_t} \left[ \| \nabla_{x_t} \log p(x_t) - s_\theta(x_t, t) \|^2 \right]
$$

- $\mathcal{L}_{\text{CM}}$: 一致性损失
- 第二项: Score Matching 正则化（稳定训练）

#### 采样配置

| 参数 | 取值 | 说明 |
|------|------|------|
| `--num_steps` | 1-4 | 采样步数（默认 4） |
| `--sigma_max` | 80.0 | 最大噪声水平 |

**加速贡献**:
- 原始 Wan: 100 步
- TurboDiffusion: 4 步
- **加速: 25x**

### 3.3 W8A8 量化

#### 核心思想
> 将 Linear 层的权重和激活都量化到 INT8，利用 Tensor Core 加速。

#### 量化流程

**Per-block 量化**:
```python
def int8_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入: x (fp16/bf16)
    输出: x_int8, scale (per-block)
    """
    scale = x.abs().max(dim=-1, keepdim=True) / 127
    x_int8 = (x / scale).round().to(torch.int8)
    return x_int8, scale
```

**INT8 GEMM**:
```python
def int8_linear(x: Tensor, weight: Tensor) -> Tensor:
    x_int8, x_scale = int8_quant(x)
    w_int8, w_scale = int8_quant(weight)
    out_int32 = gemm_cuda(x_int8, w_int8)  # Tensor Core INT8 GEMM
    out = out_int32 * (x_scale * w_scale)  # 反量化
    return out
```

#### 加速贡献

| 精度 | 计算吞吐 (A100) | 相对加速 |
|------|-----------------|----------|
| FP16 | 312 TFLOPS | 1x |
| INT8 | 624 TOPS | ~2x |

### 3.4 综合加速分析

```
原始 Wan2.1-T2V-14B-720P: 4648s
                ↓
┌─────────────────────────────────────┐
│ 加速组件          │ 加速比  │ 累计  │
├─────────────────────────────────────┤
│ rCM (100→4 steps) │ 25x    │ 25x  │
│ SLA (top-k=0.1)   │ 3-5x   │ 75x  │
│ W8A8 量化         │ 1.5-2x │ 150x │
│ 工程优化          │ 1.3x   │ 205x │
└─────────────────────────────────────┘
                ↓
TurboDiffusion: 22.7s (205x 加速)
```

### 3.Y 方法总结

| 组件 | 技术 | 目标 | 加速贡献 |
|------|------|------|----------|
| Attention | SLA + SageAttention | O(n²) → O(k·n²) | 3-5x |
| Steps | rCM | 100+ → 4 | 25x |
| Precision | W8A8 | FP16 → INT8 | 1.5-2x |

---

## Section 4: Experiments

### 4.1 主要结果

#### 实验设置
- **硬件**: RTX 5090 (单卡)
- **基线**: 原始 Wan 系列 (FP16, 100 步)
- **评估**: VBench, 人工评估

#### 关键发现

| 模型 | 分辨率 | 原始时间 | 加速后 | 加速比 | 质量保持 |
|------|--------|----------|--------|--------|----------|
| Wan2.1-T2V-1.3B | 480P | 121s | 1.8s | 67x | ✓ |
| Wan2.1-T2V-14B | 480P | 1237s | 6.5s | 190x | ✓ |
| Wan2.1-T2V-14B | 720P | 4648s | 22.7s | **205x** | ✓ |
| Wan2.2-I2V-14B | 720P | 2560s | 35.4s | 72x | ✓ |

### 4.2 消融实验

#### 各组件贡献

| 配置 | 时间 | 加速比 |
|------|------|--------|
| Baseline (Wan 14B-720P) | 4648s | 1x |
| + rCM (4 steps) | 185.9s | 25x |
| + SLA | 61.9s | 75x |
| + W8A8 | 32.0s | 145x |
| + 工程优化 | 22.7s | **205x** |

#### SLA Top-k 敏感性

| top-k | 速度 | 质量 (VBench) |
|-------|------|---------------|
| 1.0 (Full) | 1x | 0.820 |
| 0.3 | 1.8x | 0.818 |
| 0.1 | 3.2x | 0.815 |
| 0.05 | 4.5x | 0.805 |

**结论**: top-k=0.1 是质量-速度的最佳平衡点。

### 4.3 与其他方法对比

| 方法 | 步数 | 注意力 | 量化 | 综合加速 |
|------|------|--------|------|----------|
| LCM | 4 | Full | FP16 | ~25x |
| DMD + FlashAttn | 1 | Flash | FP16 | ~50x |
| ViDiT-Q | 50 | Full | W4A8 | ~4x |
| **TurboDiffusion** | 4 | SLA | W8A8 | **100-200x** |

---

## Section 5: Takeaways

### 核心贡献
1. **首个 100-200x 综合加速框架**：结合多种技术
2. **SLA**: 新型稀疏-线性注意力机制
3. **rCM 适配**: 稳定的视频一致性蒸馏
4. **开源完整实现**: 代码+权重

### 对 Efficiency 研究的启发

| 洞察 | 说明 |
|------|------|
| **组合优于单一** | 多技术叠加加速效果显著 |
| **稀疏注意力可行** | top-k=0.1 即可保质量 |
| **视频一致性蒸馏难** | rCM 的 Score 正则化是关键 |
| **工程优化重要** | 额外 1.3x 来自 kernel 优化 |

### 与其他工作的联系
- **Wan**: TurboDiffusion 专为 Wan 系列优化
- **SageAttention**: SLA 基于 SageAttention 的 INT8 kernel
- **Consistency Models**: rCM 是 CM 的改进版本

---

## 使用示例

```bash
# 克隆仓库
git clone https://github.com/thu-ml/TurboDiffusion

# 下载预训练权重
# (包含 rCM 蒸馏 + SLA 训练的权重)

# T2V 推理
python wan2.1_t2v_infer.py \
    --attention_type sagesla \
    --sla_topk 0.1 \
    --quant_linear \
    --num_steps 4 \
    --prompt "A cat playing piano"
```

---

## BibTeX

```bibtex
@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion by 100-200x},
  author={Zhang, Jintao and others},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
```
