# 5. Experiments

---

## 5.1 Evaluation Tasks

> We conduct a wide range of evaluation including image captioning, VQA, multimodal reasoning, video QA and fine-grained benchmarks like MME.
>
> ==评估范围广：captioning、VQA、多模态推理、视频QA、细粒度benchmark==

---

## 5.2 Model Settings

| 任务类型 | 模型 |
|----------|------|
| Image Understanding | LLaVA-1.5-7B, LLaVA-1.5-13B, Qwen-VL |
| Video Understanding | Video-LLaVA |

---

## 5.3 Main Results ⭐

### Image Understanding

**Table 1: 不同 FastV 配置的性能/计算平衡**

| Model | FastV 配置 | FLOPs | Nocaps | Flickr30k | A-OKVQA | MMMU | Avg |
|-------|-----------|-------|--------|-----------|---------|------|-----|
| LLaVA-1.5-7B | Baseline | 100% | 99.8 | 67.9 | 76.7 | 34.8 | 69.8 |
| | K=2, R=50% | **55%** | 99.7 | 67.5 | 77.0 | 34.4 | **69.7** |
| | K=2, R=75% | 33% | 94.6 | 63.6 | 75.5 | 34.8 | 67.1 |
| | K=2, R=90% | 20% | 72.1 | 43.7 | 70.1 | 35.0 | 55.2 |

> FastV (K=2, R=50%) could achieve about **45% FLOPs reduction** for different LVLMs **without sacrificing the performance**.
>
> ==K=2, R=50% 是最佳配置：45% FLOPs 减少，性能无损==

### Latency 测试

**Table 4: 实际推理延迟对比**

| Model | Total-Time | GPU-Memory | Score | Latency/Example |
|-------|------------|------------|-------|-----------------|
| LLaVA-1.5-7B | 6:34 | 19G | 76.7 | 0.344s |
| LLaVA-1.5-13B | 10:17 | 38G | 82.0 | 0.539s |
| LLaVA-1.5-13B + FastV | **6:30** | 30G | 80.5 | **0.341s** |

> An 13B model with FastV could inference **as fast as a 7B model** with **superior performance**.
>
> ==13B + FastV 速度与 7B 相当，但性能更好！==

### Video Understanding

**Table 6: Video QA 结果**

| Model | FLOPs | TGIF | MSVD | MSRVTT |
|-------|-------|------|------|--------|
| Video-LLaVA | 100% | 0.18 | 0.70 | 0.56 |
| + FastV (K=2, R=50%) | **52.3%** | **0.21** | 0.71 | 0.55 |

> To our surprise, FastV could generally **improves** the Video-QA tasks performance while saving 40%+ computations especially for the TGIF task.
>
> ==惊喜发现：Video-QA 性能反而提升！视频冗余比图像更严重==

> The redundancy information problem is **more severe for video understanding** as multiple images from the video are transformed to tokens.
>
> ==视频冗余更严重：LLaVA 576 tokens/image，Video-LLaVA 2048 tokens/video==

---

## 5.4 Ablation Studies ⭐

### K 和 R 的影响

> When K is small, lowering R would improve the performance with a smaller FLOPs reduction ratio. In contrast, when K is large, adjusting R has limited impact.
>
> ==K 小时 R 影响大，K 大时 R 影响小==

### 对比实验 (Table 7)

| 设置 | Nocaps | Flickr30k | A-OKVQA | MMMU |
|------|--------|-----------|---------|------|
| LLaVA-1.5-7B Baseline | 100.3 | 70.2 | 78.5 | 34.5 |
| (a) Train with 50% tokens | 98.5 | 68.5 | 76.8 | 33.5 |
| (b) **FastV (K=2, R=50%)** | **100.1** | **70** | **78.4** | **34.6** |
| (c) FastV Random | 99.5 | 68.3 | 78.2 | 34.2 |
| (d) FastV (prune system prompt) | 89.2 | 64.3 | 69.2 | 33.8 |
| (e) FastV (prune first half sys) | 17.5 | 27.8 | Failed | Failed |
| (f) FastV (prune instruction) | 77.3 | 50.1 | 56.5 | 29.5 |
| (g) StreamingLLM | 13.2 | 21.4 | Failed | Failed |

**关键发现：**

1. **(b) vs (a)**：推理时动态剪枝 > 训练时减少 tokens
2. **(b) vs (c)**：Attention-based > Random 选择
3. **(d)(e)(f)**：剪枝 system prompt / instruction 会严重损害性能
4. **(g)**：StreamingLLM 不适用于 LVLM

> ==FastV 的核心设计（基于 attention 剪枝 image tokens）是正确的==

---

## 💡 Key Takeaways

1. **最佳配置**：K=2, R=50% → 45% FLOPs 减少，性能无损
2. **13B + FastV 优于 7B**：速度相当，性能更好
3. **视频效果更好**：冗余更严重，FastV 甚至能提升性能
4. **只能剪 image tokens**：剪 system prompt / instruction 会崩

---

*[返回论文目录](../README.md)*
