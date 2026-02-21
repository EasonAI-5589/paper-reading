[← 返回 README](../README.md)

# 4 Experiment

## 📌 预览
全面实验：6个 VLM backbone × 多个 benchmark × 多个压缩比，外加效率分析和消融实验。

---

# 4.1 Experimental setup

In this section, we describe the experimental configurations used to evaluate the proposed FSR framework, including the model architectures, implementation details, and benchmarks.

Model architectures. We evaluate FSR on a diverse set of VLMs covering both image and video modalities. For static image understanding, we use the LLaVA series (LLaVA-1.5-7B/13B, LLaVA-NeXT-7B/13B) and Qwen2.5-VL-7B. For video understanding, we extend our evaluation to LLaVA-Video-7B-Qwen2. FSR is applied in a fully training-free, plug-and-play manner at inference time, without modifying any model weights.

> 💡 **实验覆盖面**:
> - 6 个模型：LLaVA-1.5-7B/13B, LLaVA-NeXT-7B/13B, Qwen2.5-VL-7B, LLaVA-Video-7B
> - 图像 + 视频两大模态
> - 全部 training-free、plug-and-play

---

Implementation Details. All experiments were implemented using PyTorch 2.1.2 and Python 3.10 with CUDA 12.4. Regarding hardware configurations, experiments on 7B parameter models were conducted on NVIDIA GeForce RTX 3090 (24GB). Experiments involving larger architectures(13B) and video models (LLaVA-Video7B-Qwen2) were performed on NVIDIA GPUs with 48GB memory. The default hyperparameters for FSR are set as follows: $\alpha = 3$ , $\beta = 1$ , $\rho = 0 . 9$ , and $\kappa = 1$ , unless otherwise specified.

Evaluation benchmarks. We conduct experiments on comprehensive benchmarks spanning image and video tasks. For image understanding, we cover open-ended QA (VQAv2 Goyal et al. (2017)), compositional reasoning (GQA Hudson and Manning (2019), ScienceQA Lu et al. (2022)), OCR (TextVQA Singh et al. (2019)), and general capability assessment (POPE Li et al. (2023b), MME Fu et al. (2025a),

MMBench Liu et al. (2024), MM-Vet Yu et al. (2023)). For video understanding, we employ three recent benchmarks: MLVU Zhou et al. (2025) for multi-task long video analysis, MVBench Li et al. (2024b) for fine-grained temporal perception, and Video-MME Fu et al. (2025b) for comprehensive multimodal evaluation. To further assess expert-level and world-model-oriented video understanding, we additionally evaluate on MMVU Zhao et al. (2025) and MMWorld He et al. (2024). To ensure fair comparison, we standardize the evaluation setup by strictly applying the same prompts, post-processing steps, and metrics across all models.

> 💡 **Benchmark 覆盖**:
> - **图像** (8个): VQAv2, GQA, ScienceQA, TextVQA, POPE, MME, MMBench(EN/CN), MM-Vet
> - **视频** (5个): MLVU, MVBench, Video-MME, MMVU, MMWorld
> - 总共 13 个 benchmark，非常全面

---

# 4.2 Main Results

## 4.2.1 FSR for Standard Benchmarks (LLaVA-1.5-7B)

![Table 1](../images/ddbd87bceae58f98b9937ae081482e008f76c2bfbcc0e69035a24cd8cdbf306a.jpg)
*Table 1: Performance comparison of different pruning methods on LLaVA-1.5-7B.*

> 💡 **Table 1 核心数据速查 (LLaVA-1.5-7B)**:
> - 64 token 下 FastV 只有 72.0%，FSR 96.1% → **差距巨大**
> - MMVet（复杂推理）: FSR 32.6 vs CDPruner 29.6（64 tokens）→ 复杂任务优势明显

---

## 4.2.2 FSR for High-Resolution Inputs (LLaVA-NeXT-7B)

![Table 2](../images/dfd800c35c7f76c1ebfcb31d9a038233d7c2356a66996ed75bd5fa9246edf88f.jpg)
*Table 2: Performance comparison of different pruning methods on LLaVA-NeXT-7B.*

> 💡 **Table 2 核心数据速查 (LLaVA-NeXT-7B, 2880 tokens)**:
> - 高分辨率输入冗余更多 → FSR 的动态分配更有效
> - 960 token 时达到 100% → 说明 2880 tokens 中 2/3 是冗余的

---

## 4.2.3 FSR for Advanced Architectures (Qwen2.5-VL-7B)

![Table 3](../images/819f7ee66216d9a85088b3cc5bb366d617980826e2c205d9f4e7014be35e3915.jpg)
*Table 3: Performance comparison on Qwen2.5-VL-7B.*

> 💡 **Table 3 核心数据速查**:
> - Qwen2.5-VL 自带 dynamic resolution + native token merging → 更难的 baseline
> - FSR 适配：Focus 用 self-attention map（无 CLIP text encoder）
> - 90% 压缩: FSR 84.0% vs HoloV 82.1% vs FastV 78.3%
> - **注意**: 只和 FastV/HoloV 比，baseline 较少（可能其他方法难以适配 Qwen2.5-VL）

---

## 4.2.4 FSR for Video Understanding (LLaVA-Video-7B)

![Table 4](../images/e6fdff9621a9770318305fa7c4f5e1b5152a7c8ccf7ac017d8b0cd94cf29f36c.jpg)
*Table 4: Performance comparison on LLaVA-Video-7B.*

> 💡 **Table 4 核心数据速查**:
> - 32 frames/video, 50%-80% 压缩
> - 60% pruning: FSR 99.6% vs HoloV 98.5%
> - 80% pruning: FSR 98.2% vs HoloV 98.0%
> - 50% pruning 时 FSR 甚至超过 full tokens（100.3%）→ 剪枝去噪

---

## 4.2.5 FSR for Large-Scale Models (13B)

![Table 5](../images/4f01199ecfd2bc79794e6369676b2fffe2d4af0f9165a1ab393c0b0e2c7380d8.jpg)
*Table 5: Performance comparison on LLaVA-1.5-13B.*

![Table 6](../images/f44dbc7eec5eae2cb71e47db2cf70bd84f43ebdf22ad9ff4bf47742a74d931df.jpg)
*Table 6: Performance comparison on LLaVA-NeXT-13B.*

> 💡 **Table 5&6 亮点**:
> - LLaVA-1.5-13B, 64 tokens: FSR 96.7% vs CDPruner 96.3%
> - **LLaVA-NeXT-13B, 960 tokens: FSR 102.1%** → 剪枝后超过 full token baseline！
> - LLaVA-NeXT-13B, 640 tokens: FSR 101.7% → 仍然超过 baseline
> - 说明大模型中冗余 token 实际是噪声，pruning = denoising

---

# 4.3 Efficiency Analysis

![Table 7](../images/71e50e2a9e87c0028d6c52a9f1e0a2833bd57e9cc054ba35e334f38dab6f1aa7.jpg)
*Table 7: Comparison of efficiency and performance metrics on LLaVA-1.5-7B.*

> 💡 **Table 7 效率数据 (LLaVA-1.5-7B, 64 tokens)**:
> - FSR vs CDPruner: Prefill 速度相当，decode 更快，Score 更高（61.9 vs 60.8）
> - FSR pipeline 几乎无额外系统开销

---

# 4.4 Ablation Study

![](../images/aec45bd02122f6aa9cc0c707a7af7ff3f061e676e453326f28a0244c1e88aba8.jpg)
*Fig. 5 Ablation study on LLaVA-1.5-7B, LLaVA-NeXT-7B, and LLaVA-NeXT-13B across varying pruning ratios, validating the impact of dual-pathway hyperparameters (α, β), focus-conditioned scanning, and aggregation refinement ratio (κ).*

> 💡 **消融实验要点**:
>
> **1. α 和 β 的影响**:
> - 仅 saliency (α=0,β=1): 性能差 → 不关注 query
> - 仅 relevance (α=1,β=0): 也差 → 缺少视觉显著性
> - α=3,β=1 最优 → relevance 主导但 saliency 不可缺
>
> **2. Scan 阶段的贡献**:
> - Focus + Scan > Focus alone，尤其在高压缩率下
> - 高压缩率时 Scan 更关键（Focus 预算有限）
>
> **3. Refine 阶段 (κ)**:
> - κ=1 最优; κ=5 过度 smooth 反而降性能
> - 大模型(13B) 对 Refine 不太敏感 → 大模型更 robust

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B, 64 tokens | 96.1% avg retained |
| LLaVA-NeXT-7B, 960 tokens | 100.0% (无损) |
| LLaVA-NeXT-13B, 960 tokens | 102.1% (超越 baseline) |
| FLOPs reduction (64 tokens) | ~75% |
| KV cache compression | 9× |
| Prefill speedup | 3.9× |
| 最优超参 | α=3, β=1, ρ=0.9, κ=1 |

### 核心洞察
1. FSR 在所有模型、所有压缩率上都是 SOTA 或 near-SOTA
2. 高分辨率和大模型场景下优势更明显（冗余越多越有效）
3. 有时 pruning 能超越 full tokens → token 冗余 = 噪声
4. 三阶段各有贡献，缺一不可，但在不同条件下贡献比例不同
