# FSR: Focus-Scan-Refine — 竞品深度分析

> **论文**: Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning
> **arXiv**: 2602.05809 (2026.02.05)
> **机构**: Harbin Institute of Technology / Zhejiang University
> **代码**: https://github.com/ILOT-code/FSR
> **分析目的**: STAR-Pro ECCV 2026 竞品分析

---

## 1. 核心思想

FSR 是一个 **training-free、plug-and-play** 的 visual token pruning 框架，灵感来自人类视觉认知过程。核心创新在于将 token 选择分解为三个显式阶段，模拟人类回答视觉问题时的认知流程：

1. **Focus** → 聚焦关键局部证据（类比：人眼首先锁定任务相关区域）
2. **Scan** → 扫描全局上下文（类比：人眼扩展视野获取补充信息）
3. **Refine** → 精炼上下文表示（类比：大脑通过 ensemble coding 整合外围信息）

**核心卖点**: 动态分配 local evidence vs. global context 的 token 预算，而非静态比例。

---

## 2. 三阶段详解

### 2.1 Stage I: Focus — 双通路评分

**目标**: 识别最关键的局部视觉证据。

**方法**: Dual-pathway scoring = visual saliency + instruction relevance

- **Visual saliency** $s_i$: 来自 vision encoder 的 [CLS] attention map（多头平均）
- **Instruction relevance** $r_i$: visual token 与 CLIP text embedding 的 cosine similarity
- **融合分数**: $\phi_i = \hat{r}_i^\alpha \hat{s}_i^\beta$，默认 α=3, β=1（强调 instruction relevance）
- **动态预算**: 累积信息密度达到阈值 ρ=0.9 时停止，即 $K_F = \min\{k | \sum_{j=1}^k \phi_{\pi(j)} \geq \rho Z\}$

**关键设计**: ρ 阈值机制使得 Focus 的 token 数量是自适应的 — 简单任务少分配（如存在性判断 Focus=9），复杂任务多分配（如推理 Focus=15）。

### 2.2 Stage II: Scan — 条件上下文采样 (CCS)

**目标**: 在剩余预算 $K_S = K - K_F$ 中选择与 Focus 集互补的全局上下文 token。

**方法**: Conditional Context Sampling — 本质是以 Focus 集为初始中心的 **Farthest Point Sampling**

- 每轮选择与当前 anchor 集距离最远的 token: $i^* = \arg\max_{i \notin \mathcal{A}} \min_{j \in \mathcal{A}} (1 - \cos(\bar{v}_i, \bar{v}_j))$
- 保证补充 token 与 Focus 证据最大程度不同，避免冗余

**理论保证**: 基于 greedy k-center clustering 经典结果，CCS 的覆盖半径 ≤ 2× 最优解（2-近似）。

### 2.3 Stage III: Refine — 加权聚合

**目标**: 将被丢弃 token 中的有用信息聚合到 Scan anchor 中，不增加 token 预算。

**方法**:
- 每个被丢弃 token 分配到最近的 Scan anchor
- 只聚合 top-M 个最相似的（M = κ|S|, κ=1）
- 加权合并: $v_{j^*} \leftarrow \frac{w_{j^*} v_{j^*} + w_i v_i}{w_{j^*} + w_i}$
- **Focus 集不变** — 保持局部证据的高保真度

**关键**: Refine 只作用于 Scan tokens，避免污染高置信度的 Focus tokens。

---

## 3. 与人类视觉感知的类比

| 人类认知阶段 | FSR 阶段 | 机制 |
|---|---|---|
| 选择性注意 — 聚焦任务相关区域 | Focus | dual-pathway scoring + ρ 阈值 |
| 扩展注意 — 扫描全局上下文 | Scan | Farthest Point Sampling (CCS) |
| Ensemble coding — 外围信息整合 | Refine | similarity-based 加权聚合 |

这个类比有一定道理，但也有 **过度包装** 之嫌 — CCS 本质就是 diversity sampling，Refine 就是 token merging，认知科学的框架主要是叙事工具。

---

## 4. 关键实验结果

### 4.1 LLaVA-1.5-7B（主实验）

| 压缩率 | FSR Avg. | CDPruner | VisPruner | 领先幅度 |
|---|---|---|---|---|
| 66.7% (192 tokens) | **99.1%** | 98.5% | 98.2% | +0.6% |
| 77.8% (128 tokens) | **98.3%** | 97.6% | 96.7% | +0.7% |
| 88.9% (64 tokens) | **96.1%** | 95.7% | 93.5% | +0.4% |

### 4.2 LLaVA-NeXT-7B（高分辨率）

| 压缩率 | FSR | CDPruner | VisPruner |
|---|---|---|---|
| 66.7% (960 tokens) | **100.0%** | 99.4% | 99.2% |
| 77.8% (640 tokens) | **99.9%** | 99.3% | 98.5% |
| 88.9% (320 tokens) | **97.6%** | 97.3% | 95.4% |

### 4.3 Qwen2.5-VL-7B（先进架构，仅对比 FastV 和 HoloV）

FSR 大幅领先 HoloV（80% 压缩: 91.9% vs 88.6%），但 **未与 CDPruner/VisPruner 对比**。

### 4.4 13B 模型
- LLaVA-1.5-13B: FSR 96.7% vs CDPruner 96.3%（64 tokens）
- LLaVA-NeXT-13B: FSR **102.1%** vs CDPruner 101.2%（960 tokens），剪枝后反而比原模型好

### 4.5 视频 (LLaVA-Video-7B)
- 仅与 HoloV 对比，领先但幅度小

### 4.6 效率
- 64 tokens: FLOPs ↓75%, KV cache ↓9×, prefill 加速 3.9×
- 额外开销可忽略（与 CDPruner 相当）

---

## 5. 与 STAR-Pro 的异同分析

### 5.1 关键对比

| 维度 | FSR | STAR-Pro |
|---|---|---|
| **阶段数** | 3（Focus → Scan → Refine） | 2（Adaptive Selection → Progressive Merging） |
| **Training-free** | ✅ | ✅ |
| **Plug-and-play** | ✅ | ✅ |
| **Token 选择信号** | [CLS] attention + CLIP text relevance | LLM cross-attention (更深层的信号) |
| **全局上下文** | Farthest Point Sampling (CCS) | Progressive merging 隐式保留 |
| **Token 合并** | 仅对 Scan tokens 做 weighted merge | Progressive merging（多轮渐进） |
| **动态预算** | ρ 阈值自动分配 Focus/Scan 比例 | Adaptive threshold 自动决定保留数 |
| **理论保证** | 2-近似覆盖保证 | 无显式理论保证（可补） |
| **人类视觉类比** | 三阶段认知过程 | 无（更工程化的叙事） |

### 5.2 FSR 相对 STAR-Pro 的优势

1. **显式的 local/global 分离**: FSR 明确区分了"关键证据"和"全局上下文"，概念清晰
2. **理论覆盖保证**: CCS 有 2-近似保证，审稿人会喜欢
3. **三阶段叙事**: 认知科学包装，story-telling 更强
4. **Refine 阶段保护 Focus tokens**: 不污染高置信 token，设计合理

### 5.3 STAR-Pro 相对 FSR 的优势

1. **更深层的 attention 信号**: STAR-Pro 用 LLM decoder 的 cross-attention（经过多层推理后的注意力），比 FSR 的 vision encoder [CLS] attention 更能反映 LLM 的实际需求
2. **Progressive merging 更灵活**: 多轮渐进式合并比一次性 Refine 更精细
3. **不依赖 CLIP text encoder**: FSR 的 instruction relevance 需要额外的 CLIP text encoder 计算 — 对 Qwen2.5-VL 等没有 CLIP 的模型需要特殊适配（论文中承认 Qwen2.5-VL 实验省略了 relevance 项）
4. **更简洁的设计**: 两阶段 < 三阶段，Occam's razor

### 5.4 弱点 / 可攻击点

1. **CLIP text encoder 依赖**: FSR 的 dual-pathway scoring 需要 CLIP text encoder，但现代 VLM（如 Qwen2.5-VL, InternVL2.5）可能不使用 CLIP 或用不同的 text encoder。论文在 Qwen2.5-VL 实验中不得不去掉 instruction relevance — 这是一个架构通用性问题
2. **Scan 阶段的 greedy 性质**: CCS 虽有理论保证，但 Farthest Point Sampling 倾向于选择 outlier tokens（噪声/边界 token），在极端压缩下可能浪费预算
3. **Refine 的聚合可能 over-smooth**: 论文自己承认 κ 过大会模糊表示（κ=5 性能饱和/下降）
4. **实验对比不充分**: Qwen2.5-VL 和 Video 实验只与 FastV/HoloV 对比，缺少 CDPruner/VisPruner 基线
5. **超参数较多**: α, β, ρ, κ 四个超参数，虽然声称默认值通用，但不同模型可能需要调整
6. **领先幅度小**: 在多数设置下 FSR vs CDPruner 差距仅 0.3~0.7%，在误差范围边缘

---

## 6. "Training-free 三阶段 SOTA" 结论验证

### 6.1 是否真的 SOTA？

**基本成立，但有保留**:
- 在 LLaVA-1.5-7B/13B、LLaVA-NeXT-7B/13B 上，FSR 在大多数设置下确实 beat CDPruner（当前最强 baseline）
- 但领先幅度很小（通常 <1%），且部分 benchmark 上并非全胜（如 POPE 在 64 tokens 设置下 FSR 85.7 vs CDPruner 87.5）
- **FSR 未与 STAR-Pro 直接对比**（因为 STAR-Pro 尚未发表），所以"超过 STAR-Pro"无法确认

### 6.2 是否超过 STAR-Pro？

**无法直接确认** — 论文中没有 STAR-Pro 的数据。需要我们自己跑对比实验。

关键观察：
- FSR 的 Avg. 在 LLaVA-1.5-7B 64 tokens 设置为 96.1%
- 如果 STAR-Pro 在相同设置下能达到 96.5%+，则 FSR 并非 SOTA
- FSR 在高分辨率（LLaVA-NeXT）场景优势更明显，这是需要重点关注的赛道

---

## 7. 对 STAR-Pro 论文的建议

### 7.1 实验层面
- **必须加 FSR 为 baseline**: 它是同期最强 training-free 方法
- **重点在极端压缩率**: FSR 在 88.9% 压缩下仍有 96.1%，STAR-Pro 需要在此设置下展现优势
- **Qwen2.5-VL 是差异化战场**: FSR 在此模型上需去掉 instruction relevance，STAR-Pro 如果不依赖 CLIP 则有天然优势

### 7.2 叙事层面
- FSR 用了认知科学包装，STAR-Pro 可以从不同角度讲 story（如 information-theoretic 或 progressive refinement）
- 强调 STAR-Pro 的 **架构通用性**（不需要 CLIP text encoder）
- 强调 **更少的超参数** 和 **更简洁的设计**

### 7.3 技术层面
- FSR 的 CCS 理论保证是一个亮点，STAR-Pro 可考虑补充类似的理论分析
- STAR-Pro 的 progressive merging 与 FSR 的 one-shot refine 可以做 ablation 对比

---

## 8. 总结

FSR 是一个设计精良、包装到位的 training-free token pruning 方法。三阶段设计在概念上清晰，认知科学类比增强了叙事力。实验覆盖面广，在多数设置下 marginally 领先 CDPruner。

**对 STAR-Pro 的威胁等级: 🟡 中等**
- 性能领先幅度小，STAR-Pro 有机会在实验中追平或超过
- FSR 的 CLIP 依赖是一个结构性弱点
- 但其 story-telling 和实验完整度值得学习

---

*分析日期: 2026-02-21 | 分析者: 3号机*
