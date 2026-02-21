[← 返回 README](../README.md)

# 3. Proposed Method

## 📌 预览
FSR 三阶段技术细节：Focus（双通道评分 + 累积密度阈值）→ Scan（条件上下文采样 CCS + 覆盖理论保证）→ Refine（similarity-based 加权聚合）。

---

## 3.1 Inspiration from Human Visual Perception

Cognitive science research indicates that when answering visual questions, humans:
1. **Prioritize** local regions highly relevant to the query (selective attention)
2. **Scan** the global context when local evidence is insufficient (peripheral scanning)
3. **Aggregate** peripheral information via ensemble coding into summary statistics (holistic representation)

> 💡 **批注**: Figure 3 展示了认知科学的三步模型。这个 framing 在叙事上很有效，但实际算法与认知过程的映射是 loose analogy 而非严格建模。不过作为 motivation 足够有说服力。

**Problem Formulation**: Given visual tokens **V** = {**v**_i}^N, query **q**, budget K ≪ N, find compressed subset **Ṽ** ⊂ **V** with |**Ṽ**| = K.

## 3.2 Stage I: Focus on Local Evidence

### Dual-Pathway Scoring

**Pathway 1 — Visual Saliency**: 使用 vision encoder 的 [CLS] attention map：

$$s_i = \frac{1}{H} \sum_{h=1}^{H} \mathbf{A}_h[\text{CLS}, i]$$

> 💡 **批注**: 这是标准的 [CLS] attention saliency，与 FasterVLM/HiRED 类似。用 vision encoder 内部的 attention，不需要 LLM forward pass。

**Pathway 2 — Instruction Relevance**: 使用 **CLIP text encoder** 编码 query，计算 cosine similarity：

$$r_i = \cos(\bar{\mathbf{v}}_i, \bar{\mathbf{t}})$$

> 💡 **⚠️ 关键依赖问题 — CLIP Text Encoder**:
> - FSR 需要额外的 CLIP text encoder 来计算 instruction relevance
> - 这意味着：(1) 额外的计算开销（虽然 CLIP text encoder 很轻）；(2) **架构依赖**——需要 visual tokens 与 CLIP text space 对齐
> - **对于 Qwen2.5-VL 等没有 CLIP text encoder 的架构，FSR 必须 fallback**：论文中 Qwen2.5-VL 实验省略了 instruction relevance，仅用 self-attention 聚合作为 Focus score
> - CDPruner 同样依赖 CLIP text encoder，这是同一类方法的共同局限
> - **与 STAR-Pro 对比**: STAR-Pro 不依赖外部 text encoder，使用 LLM 内部的 cross-attention，架构兼容性更强

### Fused Priority Score

$$\phi_i = \hat{r}_i^\alpha \cdot \hat{s}_i^\beta$$

其中 α=3, β=1（默认）。两个 score 先归一化到 [0,1]，然后用幂指数融合。

> 💡 **批注**: α=3, β=1 说明 instruction relevance 的权重远大于 visual saliency（3倍指数）。这与直觉一致——task-relevant 比 visually salient 更重要。消融实验（Figure 5）验证了 α=3,β=1 是最优配置。

### Dynamic Budget K_F

$$K_F = \min\{k \mid \sum_{j=1}^{k} \phi_{\pi(j)} \geq \rho Z\}$$

按 ϕ 降序排列，取前 k 个使得累积 score ≥ ρ·Z（默认 ρ=0.9）。

> 💡 **核心创新 — 动态分配**:
> - 如果 query 集中在单一 object → 少数 token 就能覆盖 90% 的 score → K_F 小，留更多 budget 给 Scan
> - 如果 query 涉及多 object/关系推理 → score 分散 → K_F 大
> - 这就是 Figure 1 中动态变化的来源
> - **潜在问题**: ρ=0.9 是固定阈值，对所有任务一视同仁。是否可以根据任务类型自适应调整？

## 3.3 Stage II: Scan for Global Context

### 3.3.1 Conditional Context Sampling (CCS)

剩余 budget K_S = K - K_F 用于选择全局上下文。采用 **Farthest Point Sampling (FPS)** 变体：

初始化锚点集 A = F（Focus 集合）。每次迭代选择与当前 A 最远的 token：

$$\Delta(i, \mathcal{A}) = \min_{j \in \mathcal{A}} (1 - \cos(\bar{\mathbf{v}}_i, \bar{\mathbf{v}}_j))$$
$$i^* = \arg\max_{i \notin \mathcal{A}} \Delta(i, \mathcal{A})$$

重复 K_S 次，最终 S = A \ F。

> 💡 **批注**: 
> - CCS 本质是 **conditioned on Focus set 的 FPS**。以 Focus 集合为起始点做 farthest point sampling，确保选出的 Scan tokens 与 Focus tokens 互补（最不相似）
> - 与 DivPrune 的 max-min diversity selection 思想类似，但 DivPrune 没有先确定 Focus 集合
> - **计算复杂度**: O(K_S · N · |A|) cosine similarity 计算，随着 A 增长线性增加。对于 N=576, K=64 这个量级，开销可忽略
> - **与 CDPruner 对比**: CDPruner 用 DPP (determinantal point process) 做多样性采样，计算复杂度更高 O(N·K²)，但可能更优。FSR 的贪心 CCS 有理论保证（2-approximation）

### 3.3.2 Theoretical Coverage Guarantee

CCS 可以视为 k-center clustering with fixed centers 的贪心算法。根据经典结果：

$$\max_{v \in V} \min_{u \in \mathcal{K}} d(v, u) \leq 2 R_{\text{opt}}(\mathcal{F})$$

> 💡 **批注**: 2-approximation 保证来自 Gonzalez (1985) 和 Hochbaum & Shmoys (1985) 的经典结果。这个理论保证是 FSR 相对于 CDPruner 的一个卖点——虽然 DPP 也有理论性质，但 FSR 的保证更直接且计算更轻。

## 3.4 Stage III: Refine by Aggregation

**目标**: 将丢弃集 D = V \ (F ∪ S) 中的信息聚合到 Scan anchors 中，不改变 token 数量。

**关键设计**: 只聚合到 **Scan tokens**，不动 Focus tokens（保持高保真的 local evidence）。

Step 1: 对每个丢弃 token i ∈ D，找最近的 Scan anchor：
$$j^*(i) = \arg\max_{j \in \mathcal{S}} \cos(\bar{\mathbf{v}}_i, \bar{\mathbf{v}}_j)$$

Step 2: 选 Top-M = κ·|S| 个最高相似度的丢弃 token（默认 κ=1）

Step 3: 加权合并（权重用 priority score ϕ）：
$$\mathbf{v}_{j^*} \leftarrow \frac{w_{j^*} \mathbf{v}_{j^*} + w_i \mathbf{v}_i}{w_{j^*} + w_i}$$

> 💡 **批注**:
> - Refine 阶段类似 PruMerge 的 token merging，但只作用于 Scan anchors
> - κ=1 意味着每个 Scan anchor 平均只吸收 1 个丢弃 token（实际是总共 M = |S| 个 token 分配到各 anchor）
> - **消融**: κ=5 时性能反而下降（over-smoothing），κ=1 最优
> - **与 STAR-Pro 对比**: STAR-Pro 的 merge 发生在 LLM attention 层内部，FSR 的 merge 在进入 LLM 之前的预处理阶段
> - Focus tokens 不参与 merge 是关键设计——保证 local evidence 不被稀释

最终输出 **Ṽ** = F ∪ S（|F| + |S| = K 个 tokens），其中 S 中的 token 已被 Refine enriched。

---

## 🔖 Section 总结

### 算法流程
```
Input: V = {v_i}^N, query q, budget K
│
├─ Stage I: Focus
│  ├─ s_i ← [CLS] attention saliency
│  ├─ r_i ← CLIP text encoder cosine similarity  ⚠️ 需要额外 CLIP text encoder
│  ├─ ϕ_i ← r̂_i^3 · ŝ_i^1
│  └─ K_F ← cumulative ϕ ≥ 0.9·Z → F = top-K_F tokens
│
├─ Stage II: Scan
│  ├─ K_S ← K - K_F
│  ├─ CCS: farthest point sampling conditioned on F
│  └─ S = K_S complementary tokens
│
├─ Stage III: Refine
│  ├─ 对 D = V\(F∪S) 中 token，找最近 S anchor
│  ├─ 选 top-M = κ·|S| 个，加权合并到 anchor
│  └─ F 不动，只 enrich S
│
└─ Output: Ṽ = F ∪ S (K tokens)
```

### 超参数
| 参数 | 默认值 | 含义 |
|------|--------|------|
| α | 3 | instruction relevance 的幂指数 |
| β | 1 | visual saliency 的幂指数 |
| ρ | 0.9 | cumulative density threshold |
| κ | 1 | refine merge ratio |

### 关键设计选择
1. **CLIP text encoder 依赖**: 最大的架构局限，Qwen2.5-VL 需要 fallback
2. **Focus 不参与 merge**: 保证 local evidence 高保真
3. **CCS 的 2-approximation**: 理论保证优于无保证的启发式方法
4. **动态 K_F**: 通过 ρ threshold 实现 task-dependent 分配
