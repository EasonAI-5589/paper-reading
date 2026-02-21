# Stop Looking for "Important Tokens" in Multimodal Language Models: Duplication Matters More

> **DART (Duplication-Aware Reduction of Tokens)**
> Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, Linfeng Zhang
> Shanghai Jiao Tong University, Shanghai AI Laboratory, Sun Yat-sen University, Peking University
> arXiv: [2502.11494](https://arxiv.org/abs/2502.11494) | GitHub: [ZichenWen1/DART](https://github.com/ZichenWen1/DART)

## 一句话总结

**不要再找 "重要 token" 了**——importance-based token pruning 常常不如 random pruning。DART 用 token duplication（而非 importance）指导 vision token 裁剪，88.9% 压缩下仍保持 93.7% 性能，1.99× 实际加速，兼容 FlashAttention。

## 核心发现

1. **Importance-based ≤ Random**: FastV/SparseVLM 在 88.9% 压缩下 2/3 benchmark 不如 random pruning（Figure 2）
2. **Duplication > Importance**: 去重比选重要 token 更有效——信息论视角下应最大化 retained information 而非 importance sum
3. **多组最优解存在**: 不同 pivot 策略保留的 token 集重叠 <50%，但性能相似——不存在唯一的 "关键 token 集"
4. **Token pruning 可减少 hallucination**: 深层剪枝后 POPE 超越 vanilla model

## DART 方法

```
输入: Vision tokens X = {x₁, ..., xₙ}, 目标保留比例 r
1. 选 k 个 pivot tokens P (k ≤ 8, 可用 K-norm / random)
2. 计算 cosine similarity: dup(pᵢ, xⱼ) = pᵢ⊤xⱼ / (‖pᵢ‖‖xⱼ‖)
3. 保留与所有 pivot 相似度最低的 r·n 个 token
输出: 精简后的 token 集 R, |R| = r·n
```

**关键优势**: 不需要 attention scores → 兼容 FlashAttention；O(kn) 计算 → ≤0.08s overhead；pivot 选择不敏感 → robust

## 主要结果

| Model | 压缩比 | DART | 第二名 | FastV |
|-------|--------|------|--------|-------|
| LLaVA-1.5-7B | 88.9% | **93.7%** | 91.5% (FiCoCo) | 77.3% |
| LLaVA-Next-7B | 88.9% | **93.9%** | 91.8% (HiRED) | 86.4% |
| LLaVA-1.5-13B | 88.9% | **94.7%** | 81.0% (FastV) | 81.0% |
| Qwen2-VL-72B | 88.9% | **92.2%** | 88.0% (FastV) | 88.0% |

实际加速: **1.99× total, 2.99× prefill** (LLaVA-Next-7B)

## 批读目录

| Section | 文件 |
|---------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) |
| 5. Analysis | [05-analysis.md](sections/05-analysis.md) |
| 6. Conclusion & Limitations | [06-conclusion.md](sections/06-conclusion.md) |
| Appendix | [07-appendix.md](sections/07-appendix.md) |

## Citation Landscape

### 本文挑战的方法
| 方法 | 会议 | 核心思路 | DART 的批判 |
|------|------|---------|------------|
| **FastV** | ECCV 2024 | Attention score → 删低 attention 的 image token | Position bias, 不兼容 FA, 不如 random |
| **SparseVLM** | ICML 2025 | Text-guided attention → token sparsification | 不兼容 FA, 部分 benchmark 不如 random |
| **ToMe** | ICLR 2023 | Bipartite matching → token merging | 破坏 cross-modal interaction |
| **HiRED** | AAAI 2025 | CLS attention → partition-aware token selection | 依赖 attention scores |

### DART 胜出的对手
| 方法 | 来源 | 88.9% 压缩 (LLaVA-1.5-7B) |
|------|------|---------------------------|
| FastV | ECCV 2024 | 77.3% |
| SparseVLM | ICML 2025 | 84.6% |
| PDrop | CVPR 2025 | 78.1% |
| MustDrop | 2024.11 | 90.1% |
| FiCoCo-V | 2024.11 | 91.5% |
| **DART** | **本文** | **93.7%** |

### 扩展应用
- **Audio**: Phi-4-Multimodal ASR，50% 压缩下 DART WER 34.03 vs FastV 134.19
- **VLA**: CogACT 机器人操作，DART 75.2% 超越 vanilla 74.8%

### 关键引用
- Chen et al. 2024 (FastV) — 本文主要对比基线
- Dao et al. 2022 (FlashAttention) — DART 兼容性的关键
- Bolya et al. 2023 (ToMe) — Token merging 的先驱
- Nguyen et al. 2023 — Over-smoothing 现象，DART 的理论动机

## 个人评价

**优点**:
- 🎯 核心 insight 简洁有力：duplication > importance
- 📊 实验极其充分：4 MLLM × 10+ benchmarks × 3 压缩比
- 🔧 工程友好：兼容 FA，overhead ≤0.08s，plug-and-play
- 🔬 分析深入：<50% overlap 发现很有启发性

**不足**:
- 理论分析较弱（Lipschitz bound 太 loose，对 importance-based 方法同样适用）
- 未讨论 dynamic resolution models（如 Qwen2-VL 的 naive resize vs dynamic tiling）
- Pivot 数量 = 8 似乎是固定的，未探讨与模型/图像复杂度的自适应

**与 FastV 的关系**: DART 的 §3.2 对 FastV 的批判非常到位。FastV 的核心 insight "深层不需要 image tokens" 是对的，但 "用 attention score 选 token" 是错的。DART 本质上保留了 FastV 的 "在某层执行 pruning" 框架，但把选择标准从 importance 换成了 duplication。

---
*批读完成于 2026-02-21 | 3号机*
