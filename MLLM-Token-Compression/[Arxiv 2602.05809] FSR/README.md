# FSR: Focus-Scan-Refine — From Human Visual Perception to Efficient Visual Token Pruning

**作者**: Yuanchao Bai et al.  
**机构**: Harbin Institute of Technology / Zhejiang University  
**来源**: Arxiv 2602.05809 (2026.02)  
**代码**: [GitHub](https://github.com/ILOT-code/FSR)

## 一句话总结
受人类视觉认知启发的 **training-free** visual token pruning 框架，通过三阶段 **Focus**（双通道评分锁定局部证据）→ **Scan**（条件采样补充全局上下文）→ **Refine**（加权聚合丰富上下文锚点），动态分配 local/global token 预算。

## 核心贡献
1. **三阶段人类认知模拟**: Focus（选择性注意）→ Scan（外周扫描）→ Refine（集成编码），将 pruning 重新框架化为 local/global 动态分配
2. **双通道评分**: 融合 [CLS] attention saliency 和 CLIP text encoder instruction relevance（ϕ = r̂^α · ŝ^β）
3. **条件上下文采样 (CCS)**: 基于 Focus 集合的 Farthest Point Sampling，有 2-approximation 理论保证
4. **Refine 聚合**: 将丢弃 token 加权合并到 Scan anchors，不增加 budget，Focus tokens 不动保持高保真
5. **广泛实验**: 6 个 VLM backbone × 14+ benchmarks，涵盖 image/video，稳定 SOTA

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 三阶段框架概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 三类方法分析 + 贡献 (Figure 1-2) |
| [02 - Related Work](sections/02-related-work.md) | Attention / Similarity / Joint 三分法 |
| [03 - Method](sections/03-method.md) | Focus 双通道 + Scan CCS + Refine 聚合 (Figure 3-4, Eq.1-9) |
| [04 - Experiments](sections/04-experiments.md) | 主实验 + 效率 + 消融 (Table 1-7, Figure 5) |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B, 64 tokens (88.9%), Avg. | **96.1%** (vs CDPruner 95.7%) |
| LLaVA-1.5-7B, 128 tokens (77.8%), Avg. | **98.3%** (vs CDPruner 97.6%) |
| LLaVA-NeXT-7B, 960 tokens (66.7%), Avg. | **100.0%** |
| LLaVA-NeXT-13B, 320 tokens (88.9%), Avg. | **100.0%** (超过 unpruned baseline) |
| Qwen2.5-VL-7B, 80% reduction, Avg. | **91.9%** (vs HoloV 88.6%) |
| Prefill 加速 (64 tokens) | **3.9×** |
| KV cache 压缩 | **9×** |

## 方法速览

```
Input Image → Vision Encoder → N visual tokens + [CLS] attention
                                    ↓
              ┌─── Stage I: Focus ───┐
              │ s_i ← [CLS] attention (saliency)
              │ r_i ← CLIP text encoder cosine sim (relevance) ⚠️
              │ ϕ_i ← r̂^3 · ŝ^1
              │ K_F ← cumulative ϕ ≥ 0.9·Z
              │ F = top-K_F tokens (local evidence)
              └──────────────────────┘
                                    ↓
              ┌─── Stage II: Scan ───┐
              │ K_S = K - K_F
              │ CCS: Farthest Point Sampling from F
              │ S = K_S tokens (global context)
              └──────────────────────┘
                                    ↓
              ┌─── Stage III: Refine ─┐
              │ D = V \ (F ∪ S) 丢弃集
              │ Top-M = κ·|S| tokens → merge to nearest S anchor
              │ F 不动, S enriched
              └──────────────────────┘
                                    ↓
              Output: Ṽ = F ∪ S (K tokens) → LLM
```

## ⚠️ 关键局限

1. **CLIP Text Encoder 依赖**: Focus 阶段需要额外 CLIP text encoder 计算 instruction relevance → 对 Qwen2.5-VL 等新架构需 fallback（省略 relevance）
2. **提升幅度有限**: 相对 CDPruner 通常 <1% 的 Avg. 提升
3. **ρ 固定阈值**: ρ=0.9 对所有任务一样，"动态"程度有限
4. **Refine 贡献有限**: 大模型上增益小，κ 过大反而 over-smoothing

## Citation Landscape

| 类别 | 代表方法 | FSR 的关系 |
|------|----------|-----------|
| Attention-based | FastV, PruMerge, SparseVLM, PyramidDrop | FSR Focus 阶段部分借鉴 [CLS] attention |
| Similarity-based | DivPrune, DART | FSR Scan 阶段类似 FPS/diversity selection |
| Joint | CDPruner, VisPruner, HoloV, VisionZip | 最直接竞品，FSR 用三阶段替代联合优化 |
| Token Merging | PruMerge | FSR Refine 阶段类似 merge 操作 |
| 认知科学 | Yarbus (1967), Alvarez (2011) | FSR 的 narrative 基础 |

---

*批读日期: 2026-02-21*
