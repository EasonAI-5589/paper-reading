[← 返回 README](../README.md)

# 5 Experiments

## Experimental Setting

- **Models**: LLaVA-NeXT-7B/13B, Qwen2.5-VL-7B-Instruct, InternVL2-8B
- **Benchmarks**: 8 image (MME, ScienceQA, GQA, POPE, MMBench-EN/CN, VizWiz, VQAv2) + 4 video (VideoMME, EgoSchema, MLVU, LongVideoBench)
- **Hardware**: 4× NVIDIA RTX 3090

## 5.1 Benchmarking

### Image Understanding (Table 1)

| Setting | ToDRE | 2nd Best | Gap |
|---------|-------|----------|-----|
| 25% retain (LLaVA-7B) | **98.2%** | 96.6% (DivPrune) | +1.6% |
| 10% retain (LLaVA-7B) | **95.0%** | 93.5% (DivPrune) | +1.5% |
| 25% retain (LLaVA-13B) | **97.3%** | 96.6% (DivPrune) | +0.7% |
| 10% retain (LLaVA-13B) | **93.6%** | 92.5% (DivPrune) | +1.1% |

> 💡 **批注**：ToDRE 在所有设置下均超过 DivPrune（最强 baseline），且差距在高压缩率下更大。FastV 和 SparseVLM 在 13B 上严重退化，说明 attention-based 方法迁移性差。

### Video Understanding (Table 2)

ToDRE at 10% retain: 100.9% average（超过 baseline！）。At 25%: 103.1%。

> 💡 **批注**：Video 场景下 pruning 后性能反超 baseline 是一个有趣的现象。作者归因于"减少了冗余 visual tokens 对 task-relevant information 的干扰"。这在 video 中更明显，因为帧间冗余更大。

### Cross-Model (Table 3)

- Qwen2.5-VL-7B: 25% → 97.1%, 10% → 92.0%
- InternVL2-8B: 25% → 96.8%, 10% → 91.5%

> 💡 **批注**：跨模型泛化性良好。注意 Qwen2.5-VL 和 InternVL2 的 vision encoder 架构与 LLaVA 不同，但 ToDRE 依然有效——说明 embedding space 中的 diversity 选择是 architecture-agnostic 的。

## 5.2 Efficiency (Table 4)

At 10% retention on LLaVA-NeXT-7B:

| Method | FLOPs (T) | Memory (GB) | Throughput (s/s) | Perf |
|--------|-----------|-------------|------------------|------|
| Baseline | 31.4 | 15.9 | 1.5 | 100% |
| ToDRE | **6.0** (↓80.9%) | **13.6** (↓14.5%) | **2.9** (1.9×) | **95.0%** |

> 💡 **批注**：论文标题说 2.6× speed-up，但 throughput 表上是 1.9×。2.6× 可能是算 FLOPs ratio (31.4/6.0≈5.2×) 或考虑了其他因素。实际 wall-clock 加速受限于 overhead（similarity 计算等）。

## 5.3 Ablation Study (Table 5)

| Config | Total Time | Performance |
|--------|-----------|-------------|
| Baseline (2880 tokens) | 77:04 | 100.0% |
| Stage 2 only | 70:15 (↓8.8%) | **100.0%** (lossless!) |
| Stage 1 only (25%) | 48:10 (↓37.5%) | 98.8% |
| Stage 1 + Stage 2 (25%) | 44:18 (↓42.5%) | 98.9% |
| Stage 1 only (10%) | 31:18 (↓59.4%) | 95.8% |
| Stage 1 + Stage 2 (10%) | **29:43** (↓61.4%) | **96.0%** |

> 💡 **批注**：关键发现：
> 1. Stage 2 alone 是 **lossless** 的（100.0%），验证了 information migration 假设——深层的 visual tokens 确实已经没用了
> 2. 两个 stage 组合后性能略微提升（95.8% → 96.0%），说明 Stage 2 不仅提升效率，还能去除干扰
> 3. 效率增益主要来自 Stage 1（embedding space pruning），Stage 2 的增益在短答案任务中有限但在长文本生成中会更显著
