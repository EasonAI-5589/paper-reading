[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
在 LLaVA-1.5/NeXT (7B/13B)、Qwen2.5-VL-7B、LLaVA-Video 上全面评估，覆盖 image + video 任务，66.7%~90% 压缩比。

---

## 4.1 Experimental Setup

**Models**: LLaVA-1.5-7B/13B, LLaVA-NeXT-7B/13B, Qwen2.5-VL-7B, LLaVA-Video-7B-Qwen2

**Benchmarks (Image)**: VQAv2, GQA, ScienceQA, TextVQA, POPE, MME, MMBench (EN/CN), MM-Vet
**Benchmarks (Video)**: MLVU, MVBench, Video-MME, MMVU, MMWorld

**Hardware**: RTX 3090 (24GB) for 7B; 48GB GPU for 13B/video

**Hyperparameters**: α=3, β=1, ρ=0.9, κ=1

> 💡 **批注**: 实验覆盖面很广——6 个模型 × 多个压缩比 × 14+ benchmarks。比 CDPruner 的实验范围更大（CDPruner 没测 Qwen2.5-VL 和 video）。

## 4.2 Main Results

### 4.2.1 LLaVA-1.5-7B (Table 1)

| 压缩比 | FSR Avg. | CDPruner | VisPruner | 差距 |
|--------|----------|----------|-----------|------|
| 66.7% (192 tokens) | **99.1%** | 98.5% | 98.2% | +0.6% |
| 77.8% (128 tokens) | **98.3%** | 97.6% | 96.7% | +0.7% |
| 88.9% (64 tokens) | **96.1%** | 95.7% | 93.5% | +0.4% |

> 💡 **批注**: 
> - 在低压缩比（66.7%）下优势不大，所有方法都接近满分
> - **88.9% 压缩比是关键区分点**：FSR 96.1% vs CDPruner 95.7%，差距不算大但稳定领先
> - MMVet 上差距最明显：64 tokens 下 FSR 32.6 vs CDPruner 29.6（+3.0），说明 FSR 在复杂推理任务上优势更大
> - **与 STAR-Pro 对比参考**: 需要在相同设置下比较，但 FSR 的 baseline 数字可以直接引用

### 4.2.2 LLaVA-NeXT-7B (Table 2, High-Resolution)

2880 tokens → 960/640/320 tokens：

| 压缩比 | FSR Avg. | CDPruner | 差距 |
|--------|----------|----------|------|
| 66.7% (960) | **100.0%** | 99.4% | +0.6% |
| 77.8% (640) | **99.9%** | 99.3% | +0.6% |
| 88.9% (320) | **97.6%** | 97.3% | +0.3% |

> 💡 **批注**: 高分辨率场景下 FSR 优势更明显——66.7% 压缩下保留 100% 性能！这说明高分辨率图像的冗余更大，FSR 的 Focus-Scan 策略能有效利用这种冗余。

### 4.2.3 Qwen2.5-VL-7B (Table 3, Advanced Architecture)

**⚠️ 重要**: Qwen2.5-VL 没有 CLIP text encoder，FSR 做了适配——Focus stage 仅用 self-attention 聚合，省略 instruction relevance。

| 压缩比 | FSR Avg. | HoloV | FastV |
|--------|----------|-------|-------|
| 50% | **97.9%** | 95.6% | 92.0% |
| 60% | **96.4%** | 94.2% | 89.8% |
| 80% | **91.9%** | 88.6% | 84.6% |
| 90% | **84.0%** | 82.1% | 78.3% |

> 💡 **批注**: 
> - 即使没有 instruction relevance（fallback 模式），FSR 仍大幅领先 HoloV（+1.9%~3.3%）
> - 但只比较了 FastV 和 HoloV，**没有比较 CDPruner**（因为 CDPruner 也依赖 CLIP text encoder）
> - 这揭示了一个重要问题：**CLIP text encoder 依赖限制了方法的通用性**

### 4.2.4 Video (Table 4)

LLaVA-Video-7B, 32 frames：FSR 在 60% 压缩下保留 99.6% 性能，80% 下保留 98.2%。

> 💡 **批注**: Video 实验只比较了 HoloV，baseline 不够丰富。但证明了 FSR 可以扩展到 video 模态。

### 4.2.5 13B Models (Tables 5-6)

LLaVA-1.5-13B 和 LLaVA-NeXT-13B 上也保持领先。

> 💡 **有趣发现**: LLaVA-NeXT-13B 在 640 tokens（77.8% 压缩）下，FSR 达到 101.7%——**超过了不剪枝的 baseline**！说明密集 visual tokens 可能引入噪声，适度剪枝反而有益。

## 4.3 Efficiency Analysis (Table 7)

LLaVA-1.5-7B, 64 tokens:
- FLOPs 减少 ~75%
- KV cache 压缩 **9×**
- Prefill 加速 **3.9×**
- Decode latency: 22.317 ms（最低）

> 💡 **批注**: FSR 的计算开销（Focus scoring + CCS + Refine）是 negligible 的，不影响总推理延迟。这是 training-free 方法的共同优势。

## 4.4 Ablation Study (Figure 5)

1. **α, β 消融**: α=0,β=1（仅 saliency）和 α=1,β=0（仅 relevance）都不如 α=3,β=1
2. **Scan 有效性**: 加入 Scan 后性能显著提升，尤其在高压缩比下
3. **κ 消融**: κ=1 最优，κ=5 性能下降（over-smoothing）

> 💡 **批注**: 
> - 消融证明三个阶段都有贡献，但贡献大小为 Focus > Scan > Refine
> - κ=5 的 over-smoothing 问题说明 Refine 需要谨慎——过多 merge 会破坏 token 的语义独特性
> - 13B 模型上 Refine 的增益更小，说明大模型对 peripheral information loss 更鲁棒

---

## 🔖 Section 总结

### 关键数字速查

| 设置 | FSR | CDPruner | 差距 |
|------|-----|----------|------|
| LLaVA-1.5-7B, 64 tokens | 96.1% | 95.7% | +0.4% |
| LLaVA-1.5-7B, 128 tokens | 98.3% | 97.6% | +0.7% |
| LLaVA-NeXT-7B, 640 tokens | 99.9% | 99.3% | +0.6% |
| LLaVA-NeXT-13B, 320 tokens | 100.0% | 99.0% | +1.0% |
| Qwen2.5-VL, 80% reduction | 91.9% | N/A | vs HoloV +3.3% |

### 实验局限
1. Qwen2.5-VL 上缺少 CDPruner 对比（共同依赖 CLIP text encoder）
2. Video 实验 baseline 不够丰富
3. 没有测试更新的模型（如 InternVL2.5）
