# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

> **Paper:** arXiv 2602.02951v1  
> **Authors:** Yihong Huang et al. (鹏城实验室/西电/港理工/大湾区大学/深大/华为)  
> **Code:** https://github.com/Man-PaperRejected/Nuwa  
> **Read Date:** 2026-02-21  
> **Purpose:** STAR-Pro ECCV 2026 竞品分析

---

## 1. 核心问题和动机

### 1.1 发现的问题
现有 token pruning 方法在 VQA 任务上表现良好（保留 ~94%），但在 **Visual Grounding (VG)** 任务上严重退化（仅保留 ~7%）。

### 1.2 三个关键发现 (Findings)

**Finding 1:** 高级 pruning 方法在 VQA 上相比简单 baseline（random sampling, average pooling）优势有限；所有方法在 VG 上都系统性退化，且 average pooling 反而最好。

**Finding 2:** VLM 的视觉处理是多阶段的 pipeline：
- 早期层：全局语义整合（task-independent）
- 中间层：细粒度 object-centric 聚焦（task-dependent，VG 对视觉信息需求更高）
- 用 Visual Attention Entropy (VAE) 和 Object-Centric Cohesion (OCC) 两个指标量化

**Finding 3:** VG 退化的根本原因是 **Global Spatial Reference Frame 的丢失**。Token pruning 破坏了 position embedding 构建的全局空间参考系。
- 提出三种 PE 策略分类：PERC（压缩）、PESP（稀疏保留）、RPME（相对位置映射扩展）
- 实验证明 RPME 可以恢复部分 VG 性能（VisionZip +7.5%~+16.9%）

### 1.3 Motivation 总结
**核心论点：** Token pruning 撕裂了 spatial integrity，需要修补。空间完整性来自 token 位置信息之间的交互构建的全局空间参考框架。

---

## 2. 方法细节

### 2.1 Stage 1: Spatial Cohesion Pruning (Vision Encoder 后)
灵感来源：Boids 群体智能算法 (Reynolds, 1998)，三个操作对应 separation/alignment/cohesion。

#### (1) Separation — Grid Partitioning
- 将 N² 个 visual tokens 划分为 M×M 个非重叠局部区域
- 目的：保持空间完整性，后续操作在 region 级别进行

#### (2) Alignment — Salience Identification
- 在每个 region 中选择代表性 benchmark tokens 作为聚合中心
- Salience score = CLS attention × L2-norm of key vector
  - `S(ti) = α_cls,i · ||k_i||_2`
- L2-norm 作为 information capacity 的度量（灵感来自 ViT registers 研究）

#### (3) Aggregation — Spatial Proximity
**角色分配：**
- **Pillar Tokens** (top 25% L2-norm)：特征不修改，类似 ViT registers
- **Collector Tokens** (其余)：从空间邻居聚合特征

**权重矩阵 W：**
- Semantic Similarity: `A_ij = ReLU(cosine_sim(v_i, v_j))` — 只考虑正相关
- Spatial Proximity: `P_ij = 1 - min(1, d(p_i,p_j)/d_thresh)` — 惩罚远距离聚合
- Collector: `W_ij = A_ij · P_ij`; Pillar: `W_ij = δ_ij`
- 最终: `V'_B = Ŵ · V`（行归一化后的加权聚合）

### 2.2 Stage 2: Text-Modulated Pruning (LLM 中间层)
- 在 LLM 中间层（multimodal alignment 之后）进行
- 文本查询向量: `q̄ = mean(q_1, ..., q_K)`（text tokens 的平均嵌入）
- 相关性得分: `R_i = cosine_sim(proj(v'_i), q̄)`
- 保留 top-K_final 个视觉 token
- **本质上就是 FastV 的变体**，作者自己说"Stage-2 implementation is analogous to FASTV"

### 2.3 Position Embedding 策略
- 使用 RPME 策略：保留 pruned tokens 的相对空间距离，通过线性映射扩展到原始 PE 范围
- 这是 VG 性能提升的关键

---

## 3. 关键实验结果

### 3.1 VQA (LLaVA-1.5-7B)
| Token Budget | Nüwa | VisionZip | SparseVLM | FastV |
|---|---|---|---|---|
| 192 (↓66.7%) | **98.80%** | 98.26% | 96.11% | 89.53% |
| 128 (↓77.8%) | **97.87%** | 97.63% | 93.36% | 85.04% |
| 64 (↓88.9%) | **94.91%** | 93.99% | 89.93% | 79.36% |

VQA 上优势不大，比 VisionZip 提升 ~1%。

### 3.2 Visual Grounding (RefCOCO, LLaVA-1.5-7B)
| Token Budget | Nüwa | Pooling | VisionZip | SparseVLM | FastV |
|---|---|---|---|---|---|
| 192 | **79.29%** | — | — | — | — |
| 128 | **75.20%** | ~40% | 8.1% | 12.84% | 18.55% |
| 64 | **47.19%** | ~24% | 7.28% | 1.88% | 3.81% |

**VG 上是碾压级提升：** 从 7% → 47%（64 tokens），从 ~12% → 75%（128 tokens）。

### 3.3 Efficiency
- 64 tokens: 0.6476 TFLOPs (vs VisionZip 0.6461), prefill 46ms (vs VisionZip 45ms)
- 相比 vanilla (576 tokens): ↓89% TFLOPs, ↓62% prefill time
- 额外计算开销可忽略

---

## 4. 与 STAR-Pro 的对比分析

### 4.1 Motivation 差异（完全不同）

| | Nüwa | STAR-Pro |
|---|---|---|
| 核心问题 | Spatial integrity 被破坏导致 VG 退化 | Indicator inconsistency + importance evolution 导致 pruning 不准 |
| 切入角度 | 空间参考系的保持 | 重要性评估指标本身的可靠性 |
| 分析方法 | PE 策略分类 + position reconstruction | R+λD framework 分析 attention vs similarity 差异 |
| 关注任务 | 特别关注 VG 任务 | 主要关注 VQA（通用性能保留）|

### 4.2 方法对比

| | Nüwa (2-stage) | STAR-Pro (2-stage) |
|---|---|---|
| **Stage 1 位置** | Vision encoder 输出后 | Vision encoder 输出后 |
| **Stage 1 方法** | Grid partition → Salience selection → Spatial-semantic aggregation | R+λD adaptive selection（attention + similarity 加权）|
| **Stage 1 特点** | Boids-inspired, pillar/collector 角色区分, 空间邻近性约束 | Rate-distortion 优化理论驱动 |
| **Stage 2 位置** | LLM 中间层（单层） | LLM 多层 progressive |
| **Stage 2 方法** | Text-guided cosine similarity pruning（类似 FastV） | Progressive multi-layer pruning with multi-token text raters |
| **Stage 2 特点** | 简单的 mean-pooled text query | Multi-token raters，逐层渐进 |
| **训练** | Training-free | Training-free |

### 4.3 关键差异

1. **STAR-Pro Stage 2 更精细：** Nüwa Stage 2 本质就是 FastV 变体（单层, mean text query），STAR-Pro 是多层渐进 + multi-token text raters，理论上更能捕捉 importance evolution
2. **Nüwa 更关注空间：** 整个设计围绕 spatial integrity，有 region partition + RPME，STAR-Pro 不特别处理空间
3. **理论框架不同：** Nüwa 用群体智能类比（弱），STAR-Pro 用 R+λD 信息论框架（强）
4. **评估维度不同：** Nüwa 额外评了 VG（其他方法通常不评），STAR-Pro 聚焦 VQA

---

## 5. 弱点和可攻击点

### 5.1 方法层面

1. **Stage 2 过于简单：** 自己承认"analogous to FASTV"，只是 mean-pooled text query + cosine similarity，缺乏对 importance evolution 的考虑。单层 pruning 无法适应不同层的 task-specific 需求。

2. **Grid partition 不灵活：** M×M 固定网格过于刚性，无法适应图像内容的语义分布。高信息密度区域和低信息密度区域得到相同的 token budget。

3. **Pillar/Collector 的 75% 分位数阈值是硬编码的：** 没有理论依据，只是启发式选择。

4. **Boids 类比牵强：** Separation/Alignment/Cohesion 的命名来自 Boids，但实际操作（grid partition / CLS attention / weighted aggregation）与 Boids 算法的动态迭代本质完全不同。这只是 branding。

5. **Spatial proximity threshold 需要调参：** 最优 τ=26%，但不同图像/任务可能需要不同阈值。

6. **只在 LLaVA-1.5 上验证：** 模型泛化性有限。LLaVA-NeXT 只在 appendix 中简单提及。没有在 Qwen2-VL、InternVL 等更新模型上验证。

### 5.2 实验层面

1. **VG 的 baseline 选择不公平：** 其他方法（FastV, SparseVLM, VisionZip）本来就不是为 VG 设计的，它们在 VG 上的灾难性表现（1-8%）更多是 PE 处理的 bug 而非方法本身的缺陷。Nüwa 的巨大提升很大程度来自 RPME（position reconstruction），而非 Stage 1 的 Boids-inspired 聚合。

2. **VQA 提升有限：** 比 VisionZip 只好 ~1%，在 64 tokens 下 94.91% vs 93.99%。说明核心方法在 VQA 上没有显著优势。

3. **没有 ablation 分离 RPME 的贡献：** Table 3 显示 RPME alone 就能显著提升 VG（VisionZip +16.9%），但主实验没有将 RPME 应用于其他 baseline 做公平对比。

4. **Ablation Table 8 暴露问题：** 去掉 region partition 后 VG 从 45% 跌到 6%，说明几乎所有 VG 提升都来自 region partition（即 RPME），而非 salience selection 或 aggregation。

### 5.3 STAR-Pro 可以利用的攻击角度

1. **"spatial integrity 只是 PE 问题"：** Nüwa 的核心贡献可以简化为"正确处理 position embedding"+ 简单的 grid-based merging。这不是一个 pruning 方法的创新，而是修 bug。

2. **Stage 2 缺乏深度：** STAR-Pro 的 progressive multi-layer pruning 在理论和实践上都更 principled。

3. **VQA 上 STAR-Pro 应该有优势：** Nüwa 的 VQA 提升主要来自 Stage 1 的 better token selection，但 STAR-Pro 的 R+λD 框架在重要性评估上更准确。

4. **STAR-Pro 可以吸收 RPME：** Position embedding 的正确处理是一个 orthogonal 的 trick，STAR-Pro 完全可以加入 RPME 来提升 VG 性能。

---

## 6. 总结

Nüwa 是一篇分析驱动的工作，核心贡献是：
1. **发现了 token pruning 破坏 spatial reference frame 的问题**（这是真正的贡献）
2. **提出 RPME 位置编码策略**（最关键的 trick）
3. **两阶段框架**（Stage 1 有新意但 Stage 2 很弱）

对 STAR-Pro 的威胁程度：**中等**。两者 motivation 完全不同，方法也不同。Nüwa 在 VG 上有明显优势，但 STAR-Pro 如果不关注 VG 则不受影响。如果 STAR-Pro 想覆盖 VG，只需加入 RPME 即可。
