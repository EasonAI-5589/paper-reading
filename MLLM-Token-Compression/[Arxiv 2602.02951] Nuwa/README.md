# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

**作者**: Yihong Huang, Fei Ma, Yihua Shao, Jingcai Guo, Zitong Yu, Laizhong Cui, Qi Tian  
**单位**: 鹏城实验室, 西安电子科技大学, 香港理工大学, 大湾区大学, 深圳大学, 华为  
**链接**: [arXiv](https://arxiv.org/abs/2602.02951) | [GitHub](https://github.com/Man-PaperRejected/Nuwa)  
**日期**: 2026-02

## 一句话总结

Nüwa 发现现有 token pruning 方法在 Visual Grounding 上严重退化的根因是**全局空间参考系被破坏**，提出两阶段 spatial-aware pruning（Boids-inspired 空间锚点 + text-guided 精炼），在 88.9% token 削减下 VQA 保留 95%、VG 保留 47%（此前仅 7%）。

## 核心贡献

1. **Task-specific 分析**：系统剖析 VLM 视觉处理 pipeline，发现 grounding 任务依赖全局空间参考系（Global Spatial Reference Frame），pruning 破坏位置嵌入导致 VG 崩溃
2. **RPME 位置策略**：提出 Relative Position Mapping Extension，通过线性映射将 pruned tokens 的 PE 扩展至原始全范围，恢复空间完整性
3. **Stage 1 — Boids-inspired 空间聚合**：Separation（网格分区）→ Alignment（CLS attention × L2-norm 选代表 token）→ Aggregation（语义相似度 × 空间邻近度加权合并，Pillar/Collector 角色区分）
4. **Stage 2 — Text-guided pruning**：在 LLM 中间层利用文本语义余弦相似度进一步筛选 task-relevant visual tokens
5. **SOTA 表现**：13 个数据集、多个 VLM 上验证，VG 性能从 7% 提升至 47%，VQA 从 94% 提升至 95%，TFLOPs 降 89%

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 动机：VG 退化问题 + 三个关键问题 |
| 2. Dissecting the Pipeline | [02-analysis.md](sections/02-analysis.md) | 三大发现：baseline 对比、多阶段 pipeline、空间完整性 |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | Stage 1 (Separation/Alignment/Aggregation) + Stage 2 (Text-Modulated) |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | VQA/VG 主实验 + 效率分析 + 消融 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Token 削减率 | 88.9% (576→64) |
| VQA 性能保留 | 94.91% (64 tokens) |
| VG 性能保留 | 47.19% (64 tokens, RefCOCO avg) |
| VG 性能保留 (192 tokens) | 79.29% |
| TFLOPs 降低 | 89% |
| Prefill 加速 | 62% (124ms→46ms) |
| 额外开销 | +0.01 TFLOPs, +1ms prefill |

## Citation Landscape

### 关键引用 (被引次数)

| 论文 | 年份 | 引用数 | 关系 |
|------|------|--------|------|
| FastV (An Image is Worth 1/2 Tokens) | 2024 | 374 | 核心 baseline，PESP 策略 |
| SparseVLM | 2024 | 219 | 核心 baseline，LLM multi-layer pruning |
| LLaVA-PruMerge | 2024 | 237 | 核心 baseline，encoder-side |
| PyramidDrop | 2024 | 155 | 核心 baseline，LLM multi-layer |
| VisionZip | 2024 | 125 | 核心 baseline，PERC 策略 |
| DivPrune | 2025 | 59 | 相关 diversity-based 方法 |
| FEATHER | 2024 | 21 | 质疑 pruning 有效性的工作 |
| Token Merging (ToMe) | 2022 | 768 | 经典 token merging |
| FlashAttention | 2022 | 3597 | 兼容性声明 |

### 该论文被引

暂无引用（2026-02 新论文）。

### 与 STAR-Pro 的关系

Nüwa 与 STAR-Pro (Shao et al., 2025a) 在引用中直接关联。两者都关注 VG 退化问题，但切入角度不同：
- **Nüwa**: 空间完整性（spatial integrity）— 保留全局位置参考系
- **STAR-Pro**: indicator inconsistency — token importance 指标在不同任务间不一致

---

## BibTeX

```bibtex
@article{huang2026nuwa,
  title={N{\"u}wa: Mending the Spatial Integrity Torn by VLM Token Pruning},
  author={Huang, Yihong and Ma, Fei and Shao, Yihua and Guo, Jingcai and Yu, Zitong and Cui, Laizhong and Tian, Qi},
  journal={arXiv preprint arXiv:2602.02951},
  year={2026}
}
```
