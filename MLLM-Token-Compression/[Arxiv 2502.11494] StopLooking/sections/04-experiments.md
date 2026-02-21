[← 返回 README](../README.md)

# 4 Experiments

## 📌 预览
在 4 个 MLLM（LLaVA-1.5-7B/13B, LLaVA-Next-7B, Qwen2-VL-7B/72B, MiniCPM-V2.6）上，覆盖 10+ image benchmarks 和 4 video benchmarks。DART 在所有压缩比下均优于现有方法，88.9% 压缩下领先第二名 2.2%。

---

**Experiment Setting.** We conduct experiments on over four MLLMs across ten image-based and four video-based benchmarks. For details on implementation, please refer to Appendix C.

## 4.1 Main Results

**Image understanding task.** The results presented in Tables 1 and 3 highlight DART's exceptional performance across diverse image understanding tasks under varying token configurations. We observe that (i) with only 192 tokens, DART achieves an impressive 98.8% average performance, substantially outperforming second-best MustDrop by 1.6%. (ii) This trend strengthens under aggressive reduction ratios, with DART leading by 2.2% using just 64 tokens. (iii) Moreover, DART scales seamlessly to advanced and larger models like LLaVA-Next-7B and Qwen2-VL-72B (See Tab. 7), achieving 93.9% with only 11.1% tokens, outperforming all competitors significantly. (iv) Inspired by (Wen et al., 2025), we apply DART during training. DART † in Table 1 shows better performance-efficiency trade-offs, maintaining full performance with just 192 visual tokens, highlighting the strong adaptability of our method. These results demonstrate DART's efficiency in leveraging limited tokens while preserving critical information, showcasing robust performance across tasks, model architectures, and model size. For more comparisons, please refer to Tables 4, 5, and Appendix A.3.

> 💡 **Image 结果要点**:
> | 压缩比 | 保留 tokens | DART Avg. | 第二名 | 差距 |
> |---------|------------|-----------|--------|------|
> | 66.7% | 192 | 98.8% | 97.2% (MustDrop) | +1.6% |
> | 77.8% | 128 | 98.0% | 95.6% (MustDrop) | +2.4% |
> | 88.9% | 64 | 93.7% | 91.5% (FiCoCo-V) | +2.2% |
>
> **关键观察**: (1) 压缩越激进，DART 的优势越明显；(2) DART† (training-time) 能在 192 tokens 下达到 100.4%，说明 token pruning 甚至能起到正则化作用；(3) 在 LLaVA-Next-7B 上 88.9% 压缩时 DART 93.9% 远超第二名 HiRED 91.8%。

---

**Video Understanding Task.** To assess DART's capabilities in video understanding, we integrate it with Video-LLaVA (Lin et al., 2023) and benchmark it against state-of-the-art methods, including FastV (Chen et al., 2024). Following established protocols, Video-LLaVA processes videos by sampling 8 frames and extracting 2048 vision tokens, with 50% retained for evaluation. As demonstrated in Table 6, DART surpasses FastV across all benchmarks, achieving a notable 4.0 score on MSVD, 46.3% accuracy on TGIF, and 56.7% accuracy on MSRVT. With an average accuracy of 58.0% and an evaluation score of 3.7, DART demonstrates superior reasoning over complex multimodal data.

> 💡 **Video 结果**: DART 在 50% retention 下全面超越 FastV，且接近 vanilla Video-LLaVA（58.0% vs 58.2%）。视频场景下 frame 间冗余更严重，duplication-based 方法的优势更自然。

---

### Table 1: LLaVA-1.5-7B 主要结果（精选）

| Method | Tokens | GQA | MMB | MME | POPE | SQA | VQAV2 | VQAText | Avg. |
|--------|--------|-----|-----|-----|------|-----|-------|---------|------|
| Vanilla | 576 | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 58.2 | 100% |
| FastV | 64 | 46.1 | 48.0 | 1256 | 48.0 | 51.1 | 55.0 | 47.8 | 77.3% |
| FiCoCo-V | 64 | 52.4 | 60.3 | 1591 | 76.0 | 68.1 | 71.3 | 53.6 | 91.5% |
| MustDrop | 64 | 53.1 | 60.0 | 1612 | 68.0 | 63.4 | 69.3 | 54.2 | 90.1% |
| **DART** | **64** | **55.9** | **60.6** | **1765** | **73.9** | **69.8** | **72.4** | **54.4** | **93.7%** |
| **DART†** | **64** | **57.1** | **64.7** | **1823** | **79.3** | **71.1** | **74.6** | **54.7** | **97.2%** |

> 💡 **Table 1 深入分析**: 在最极端的 88.9% 压缩下，FastV 只剩 77.3%（几乎不可用），而 DART 保持 93.7%。特别注意 POPE（hallucination 指标）：FastV 48.0 vs DART 73.9——FastV 的 position bias 导致严重 hallucination。DART† 更是达到 97.2%，接近无压缩性能。

---

### Table 2: 推理效率对比（LLaVA-Next-7B）

| Method | Tokens↓ | Total Time | Prefill Time | FLOPs↓ | POPE | Total Speedup | Prefill Speedup |
|--------|---------|-----------|-------------|--------|------|---------------|-----------------|
| Vanilla | 2880 | 36:16 | 22:51 | 100% | 86.5 | 1.00× | 1.00× |
| FastV | 320 | 18:17 | 7:41 | 12.8% | 78.3 | 1.98× | 2.97× |
| SparseVLM | 320 | 23:11 | - | 15.6% | 82.3 | 1.56× | - |
| **DART** | **320** | **18:13** | **7:38** | **12.8%** | **84.1** | **1.99×** | **2.99×** |

> 💡 **效率分析**: DART 和 FastV 的 FLOPs 和速度几乎相同（都兼容 FA 后），但 DART POPE 高 5.8 分。SparseVLM 因为不兼容 FA，虽然 FLOPs 只多 2.8%，但 speedup 低 21.6%（1.56× vs 1.99×）。这完美诠释了 "FLOPs 不等于 wall-clock time"。

---

### Table 3-5: 更多模型结果

| Model | 压缩比 | FastV Avg. | DART Avg. | 差距 |
|-------|--------|-----------|-----------|------|
| LLaVA-Next-7B | 88.9% | 86.4% | 93.9% | +7.5% |
| Qwen2-VL-7B | 88.9% | 84.0% | 87.5% | +3.5% |
| MiniCPM-V2.6 | 88.9% | 68.4% | 76.1% | +7.7% |
| Qwen2-VL-72B | 88.9% | 88.0% | 92.2% | +4.2% |
| LLaVA-1.5-13B | 88.9% | 81.0% | 94.7% | +13.7% |

> 💡 **跨模型泛化**: DART 在所有模型上都大幅领先 FastV。最惊人的是 LLaVA-1.5-13B 上 13.7% 的差距。这说明 duplication 现象在不同架构中普遍存在，且 importance-based 方法的问题也是普遍的。

---
