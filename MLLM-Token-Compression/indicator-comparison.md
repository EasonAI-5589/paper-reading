# Visual Token Pruning：Indicator 全景对比

> 整理时间：2026-02-24  
> 覆盖论文：FastV · VisionZip · DivPrune · SCOPE · CDPruner · SparseVLM · VisionTrim · HoloV · VScan · Nuwa · FSR · IDPruner · HiDivDrop（共 13 篇）

---

## 一、分类框架

各方法的 indicator 设计可以按「信号来源」和「优化目标」两个维度分类：

| 信号来源 | 单信号方法 | 融合方法 |
|---------|-----------|---------|
| **视觉自注意力**（CLS attn） | FastV, VisionZip | — |
| **多样性 / 覆盖** | DivPrune | SCOPE, HoloV |
| **跨模态文本** | SparseVLM (LLM attn) | CDPruner, FSR, Nuwa, VisionTrim |
| **重要性 + 多样性** | — | IDPruner (MMR), HiDivDrop (DTop-K) |
| **时序一致性** | HiDivDrop (ILVAS) | VScan (Global+Local) |

---

## 二、逐篇 Indicator 详解

---

### 1. FastV（ECCV 2024）

**信号**：LLM 浅层 self-attention

$$\text{score}_i = \frac{1}{H}\sum_h A_{h,\ \text{last\_text} \to v_i}^{(K)}$$

- 在 LLM 第 $K$ 层（默认 K=2），取最后一个文本 token 对各视觉 token 的 attention
- 多头取平均，直接 top-R% 选取
- **特点**：极简，training-free；"一刀切"剪枝，后续层全生效
- **局限**：第 2 层 attention 尚不稳定，高 attention token 集中一处，覆盖率低

---

### 2. VisionZip（CVPR 2025）

**信号**：Vision encoder 倒数第 2 层的 CLS attention

$$\text{Dominant score}_i = \frac{1}{H}\sum_h A_{h,\text{CLS} \to v_i}^{(L-1)}$$

- 选 top-K 为 **dominant tokens**，剩余按 cosine similarity 合并进最相近的 dominant token → **contextual tokens**
- **特点**：两步（select + merge），避免纯 prune 的信息损失
- **局限**：CLS attention 只捕全局显著，无文本引导，深层 attention 偏稀疏

---

### 3. SCOPE（NeurIPS 2025）

**信号**：Coverage Marginal Gain × CLS Attention Saliency

$$\text{SCOPE}(v, \mathcal{S}) = \underbrace{\Delta(v;\mathcal{S})}_{\text{coverage gain}} \times \underbrace{A_v^\alpha}_{\text{saliency}}$$

Coverage gain 定义：

$$\Delta(v;\mathcal{S}) = \sum_{u \in \mathcal{V}} \max\bigl(\text{sim}(u,v) - C(u,\mathcal{S}),\ 0\bigr)$$

$$C(u,\mathcal{S}) = \max_{s \in \mathcal{S}} \text{sim}(u,s)$$

- 先发现问题：**纯 saliency 的 θ-coverage 甚至低于 random**（attention 集中 → 覆盖差）
- SCOPE score = 乘法融合：信息量 × 覆盖增量
- 贪心迭代选取，每步选最大 SCOPE score 的 token
- **特点**：Coverage function 是 submodular，贪心有 $(1-1/e)$ 近似保证；$\alpha$ 调节 saliency 强度（默认 1.0）

---

### 4. CDPruner（NeurIPS 2025）

**信号**：Conditional DPP kernel = 视觉多样性 × 指令相关性

$$\tilde{L}_{ij} = \tilde{r}_i \cdot \underbrace{L_{ij}}_{\cos(v_i,v_j)} \cdot \tilde{r}_j$$

Log-det 后自然分解：

$$\log\det(\tilde{L}_S) = \underbrace{\sum_{i\in S}\log \tilde{r}_i^2}_{\text{relevance}} + \underbrace{\log\det(L_S)}_{\text{diversity}}$$

- $L_{ij}$：visual token 间 cosine similarity（多样性）
- $\tilde{r}_i$：第 $i$ 个视觉 token 与指令 embedding 的归一化 cosine similarity（相关性）
- MAP inference via 贪心 + Cholesky 分解，复杂度 $O(nm^2)$，额外延迟 <10ms
- **特点**：优雅的 DPP 条件化，training-free；两种 embedding 方式兼容有/无 CLIP text encoder 的模型

---

### 5. DivPrune（CVPR 2025）

**信号**：纯多样性，Max-Min Distance Problem (MMDP)

$$\tilde{E}_v^* = \arg\max_{|\tilde{E}_v|=\tilde{M}} \min_{\gamma,\omega \in \tilde{E}_v} d(\gamma, \omega), \quad d = 1 - \text{cos}$$

- 贪心：每步选「离已选集合最远」的 token（类似 k-center 聚类）
- 一次预计算完整距离矩阵后迭代选取
- **特点**：最纯粹的多样性导向，无需任何注意力信号，training-free
- **局限**：只关注极端 pair，忽略全局多样性；无文本引导；完全忽略 token 重要性

---

### 6. IDPruner（Arxiv 2602.13315）

**信号**：MMR（Maximal Marginal Relevance）= 重要性 - 多样性惩罚

$$v^* = \arg\max_{v_i \notin \mathcal{S}} \Bigl[\lambda \cdot \widehat{\text{Imp}}(v_i) - (1-\lambda) \cdot \underbrace{\max_{v_j \in \mathcal{S}} \cos(v_i,v_j)}_{m_i}\Bigr]$$

- $\text{Imp}$ 来自 **VisionSelector** 的 DiffTopK 输出（端到端训练）+ min-max 归一化
- $m_i$ = 候选 token 与已选集合中任意 token 的最大余弦相似度（用 max 而非 mean）
- 高效 O(KN) 更新：维护 m 向量，每步 `m ← max(m, sim(V, v*))`
- **特点**：显式平衡重要性与多样性，λ 可调；与 SCOPE 的区别：用减法而非乘法组合
- ⚠️ **非 training-free**：依赖 VisionSelector（需训练可学习评分模块）

---

### 7. VisionTrim（ICLR 2026）

**DVTS 信号**：Global CLS attention + Local LTAM（dual-kernel）自适应融合

$$S_i = \alpha \hat{S}_i^g + (1-\alpha) S_i^l, \quad \alpha = \frac{\sigma_l^2}{\sigma_g^2 + \sigma_l^2}$$

- **Global** $S_i^g$：ViT 倒数第 2 层 CLS attention，多头均值后 softmax
- **Local** $S_i^l$（LTAM）：局部 k×k 窗口内的 dual-kernel 亲和度
  - $\kappa_{\text{feat}}$：特征空间高斯距离
  - $\kappa_{\text{pos}}$：空间位置高斯距离
  - $\kappa^* = \kappa_{\text{feat}} + w_3 \kappa_{\text{pos}}$
- **自适应权重**：方差大 = 信号不稳定 → 给小权重

**TGVC 信号**（补充）：CLIP text encoder → text-visual similarity → 文本引导聚类 merge

- **特点**：multi-stage（ViT + LLM 各一次），prune+merge 两用

---

### 8. HoloV（NeurIPS 2025）

**信号**：Diversity Variance + CLS Attention，crop-wise 自适应分配

$$\mathcal{H}_i^c = \gamma_c \mathcal{V}_i^c + \mathcal{A}_i^c, \quad \gamma_c = \frac{\mathbb{E}[\|\mathcal{A}^c\|]}{\mathbb{E}[\|\mathcal{V}^c\|]}$$

- **多样性方差** $\mathcal{V}_i^c$：token $i$ 与 crop 内其他 token 相似度的方差
  - 高方差 = 该 token 与周围差异大 = 语义独特
- $\mathcal{A}_i^c$：CLS attention（全局显著性）
- 自适应缩放因子 $\gamma_c$ 确保两者量级一致
- **Crop-wise 分配**：$w_c \propto$ crop 平均 holistic score，配额 $q_c$ 按权重分配（设上下限防垄断）
- **特点**：在 LLM 之前剪枝（prefill 超线性收益）；另有 Fast VCR 在高不确定性时补回信息

---

### 9. HiDivDrop（ICLR 2026）

**两层 indicator 设计：ILVAS（在哪剪）+ DTop-K（剪什么）**

**ILVAS**（Inter-Layer Visual Attention Similarity）：

$$\text{ILVAS}(l) = \text{sim}\left(\tilde{A}_i^{(l)},\ \tilde{A}_i^{(l+n)}\right)$$

- 衡量 layer $l$ 对 token 重要性的评估在未来层的稳定性
- 选 ILVAS 曲线的局部最大值作为剪枝层：$\mathcal{F} = \{10, 14, 16, 18\}$

**DTop-K**（可微 token 选择）：

1. 重要性分数 $c_i$ → 归一化排名 $c'_i \in [0,1]$  
2. 软掩码：$\text{Mask}_i = \sigma\!\left(\lambda(c'_i - a)\right)$，$a$ 是**可学习的剪枝阈值**  
3. 前向用 hard threshold，反向用 sigmoid 梯度

- **三段式架构**：Late Injection（跳过浅层）+ Concave Pyramid（中间层渐进）+ Early Exit（深层停止）
- **特点**：唯一将"剪枝时机"作为 indicator 研究的方法；DTop-K 使 indicator 端到端可学习
- ⚠️ 需要训练

---

### 10. VScan（TMLR 2026）

**三阶段 indicator**：

| 阶段 | Indicator | 来源 |
|------|-----------|------|
| Global Scan | 深层 CLS attention（output layer） | Vision Encoder |
| Local Scan | 浅层 CLS attention（layer 6）+ 非重叠窗口划分 | Vision Encoder |
| Middle Layer Pruning | last instruction token → visual tokens attention | LLM 第 16 层 |

- Global 和 Local 各取 $R_1/2$，union 后做 Token Merging（cosine similarity → average pooling）
- **中间层而非早期层**：实验验证 k=16 >> k=2（FastV 的位置），避免位置偏差
- **特点**：training-free，兼容 FlashAttention（auxiliary vanilla attention pass）

---

### 11. SparseVLM（ICML 2025）

**信号**：LLM 内部 text→visual 注意力，自适应 rank-based 剪枝量

$$P = A[\mathbb{L}, \mathbb{I}] \in \mathbb{R}^{L_t \times L_v}, \quad \tilde{p}_j = \frac{1}{|\text{raters}|}\sum_{i \in \text{raters}} P_{ij}$$

- **Text Rater 筛选**：先用 $H_v \cdot H_q^T$ 过滤出视觉相关文本 token（超均值才成为 rater），排除代词/介词噪声
- **Rank-based 自适应裁剪量**：$N = \lambda \times (L_v - \text{rank}(P))$
  - 低 rank → 高冗余 → 多剪；每层自动决定裁剪量
- Token Recycling：density peak clustering → 求和重构，减少信息损失
- **特点**：逐层自适应，无需预设固定压缩率；training-free

---

### 12. Nuwa（Arxiv 2602.02951）

**Stage 1 信号**：CLS attention × Key vector L2-norm

$$S(t_i) = \alpha_{\text{cls},i} \times \|\mathbf{k}_i\|_2$$

- **双乘积**：CLS attention 给全局重要性，key L2-norm 给信息容量
- Grid 分区内选 top-k 作为 benchmark token
- 高 L2-norm 的 **Pillar Token** 不参与聚合（保护 ViT register token）
- 聚合权重 = 语义相似度 × 空间邻近度（防止语义相似但位置遥远的 token 被合并）

**Stage 2 信号**（LLM 中间层）：text-visual cosine similarity（多模态对齐完成后再剪）

- **特点**：专为空间定位任务设计，是目前对「空间保持」研究最深入的方法

---

### 13. FSR（Arxiv 2602.05809）

**三段式 indicator，模拟人类视觉感知**

**Focus**（局部证据，动态 budget）：

$$\phi_i = \alpha \hat{r}_i + \beta \hat{s}_i, \quad K_F = \min\left\lbrace k : \sum_{j=1}^k \phi_{\pi(j)} \geq \rho \cdot Z \right\rbrace$$

- $s_i$：CLS attention（视觉显著性）
- $r_i$：CLIP text encoder cosine similarity（指令相关性）
- 动态 budget：保留覆盖 90% 总 $\phi$ 质量所需的最少 token 数

**Scan**（全局背景，CCS 算法）：

- Conditional Context Sampling ≈ Farthest Point Sampling 变体
- 每步选「与 Focus + 已选 Scan token 综合相似度最低」的 token，最大化信息增量
- 有理论覆盖界保证

**Refine**（边缘信息聚合）：

- 剩余 token 按 similarity 分配到最近 Scan center，加权聚合
- 只修改 Scan token，保护 Focus token 高保真度

---

## 三、横向对比总表

| 论文 | Venue | 主信号来源 | 文本引导 | 多样性设计 | 剪枝位置 | Training-free |
|------|-------|-----------|---------|-----------|---------|--------------|
| **FastV** | ECCV 2024 | LLM 浅层 attn | ✗ | ✗ | LLM 内（L2） | ✅ |
| **VisionZip** | CVPR 2025 | ViT CLS attn | ✗ | Merge | ViT 后 | ✅ |
| **DivPrune** | CVPR 2025 | — | ✗ | MMDP（最大最小距离） | LLM 前 | ✅ |
| **SCOPE** | NeurIPS 2025 | ViT CLS attn | ✗ | Coverage marginal gain | LLM 前 | ✅ |
| **CDPruner** | NeurIPS 2025 | — | ✅（CLIP/LLM emb） | DPP（条件化） | LLM 前 | ✅ |
| **SparseVLM** | ICML 2025 | LLM self-attn | ✅（Rater 筛选） | Rank-adaptive | LLM 内（逐层） | ✅ |
| **VisionTrim** | ICLR 2026 | ViT CLS + LTAM | ✅（CLIP text） | TGVC merge | ViT + LLM | ✅（部分 FT） |
| **HoloV** | NeurIPS 2025 | ViT CLS + Diversity Var | ✗ | Crop 分配 + 方差 | LLM 前 | ✅ |
| **VScan** | TMLR 2026 | ViT deep+shallow CLS | ✅（last instr token） | Global+Local 互补 | ViT + LLM 中 | ✅ |
| **Nuwa** | Arxiv | ViT CLS × L2-norm | ✅（cosine，LLM 中） | 空间邻近约束 | ViT + LLM 中 | ✅ |
| **FSR** | Arxiv | ViT CLS + CLIP text | ✅（CLIP text） | CCS（Farthest） | LLM 前 | ✅ |
| **IDPruner** | Arxiv | VisionSelector（训练） | ✗ | MMR（减法平衡） | LLM 前 | ❌ |
| **HiDivDrop** | ICLR 2026 | ILVAS + DTop-K（训练） | ✗ | 层间一致性 | LLM 内（多层） | ❌ |

---

## 四、关键设计趋势

### 1. 纯 CLS attention 已不够用

从 FastV/VisionZip 到后续方法，大家都意识到单一 saliency 覆盖率差（SCOPE 直接测出 saliency-only θ-coverage < random）。后续方法纷纷在 CLS attention 基础上叠加多样性约束。

### 2. 多样性建模的三条技术路线

| 路线 | 代表方法 | 核心思想 |
|------|---------|---------|
| **组合优化** | DivPrune (MMDP), CDPruner (DPP), SCOPE (submodular coverage) | 用有理论保证的组合优化目标 |
| **迭代贪心** | IDPruner (MMR), FSR (CCS) | 每步选「重要且新颖」的 token |
| **空间分区** | HoloV (crop), Nuwa (grid), VScan (window) | 把空间划分为区域，保证覆盖均匀 |

### 3. 文本引导成为标配

CDPruner、SparseVLM、VisionTrim、Nuwa、FSR、VScan 全都引入文本信号。信号来源从 CLIP text encoder → LLM 内部 cross-attention 演进。

### 4. 剪枝位置从单点到多阶段

- FastV：单层（LLM L2）
- VisionZip/DivPrune/SCOPE/CDPruner：Vision Encoder 后
- VisionTrim/VScan/Nuwa：ViT + LLM 双阶段
- HiDivDrop：浅层（Late Injection）+ 中间层（渐进）+ 深层（Early Exit）三段式

### 5. Merge 优于纯 Prune

VisionZip、VisionTrim (TGVC)、VScan (merging)、SparseVLM (recycling)、FSR (Refine)、Nuwa (aggregation)、HoloV (VCR) 全都保留被丢弃 token 的信息，而非直接丢弃。

---

## 五、各方法适用场景速查

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 极简部署，无需调参 | FastV | 一行代码插入，效果尚可 |
| 高压缩率 + 保性能 | SCOPE / CDPruner | 有覆盖保证，不损语义 |
| 空间定位任务（OCR/REC） | Nuwa | 专为空间保持设计 |
| 多轮对话，问题动态变化 | CDPruner / SparseVLM / FSR | 文本引导，每次问题都重新评分 |
| 需要端到端优化 | HiDivDrop / IDPruner | 可学习 indicator |
| 视频 / 超长序列 | HoloV / PyramidDrop | crop 分配 / 多层渐进更稳定 |
