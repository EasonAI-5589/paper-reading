# Visual Token Pruning：Indicator 全景对比

> 整理时间：2026-02-25  
> 覆盖论文：FastV · PyramidDrop · VisionZip · DivPrune · SCOPE · CDPruner · SparseVLM · VisionTrim · HoloV · VScan · Nuwa · FSR · IDPruner · HiDivDrop（共 14 篇）

---

## 一、分类框架

各方法的 indicator 设计可以按「信号来源」和「优化目标」两个维度分类：

| 信号来源 | 单信号方法 | 融合方法 |
|---------|-----------|---------|
| **ViT 自注意力**（CLS attn） | VisionZip | — |
| **LLM 内部 attention**（saliency proxy） | FastV, PyramidDrop | — |
| **多样性 / 覆盖** | DivPrune | SCOPE, HoloV |
| **跨模态文本**（显式 text-guided） | SparseVLM (LLM attn) | CDPruner, FSR, Nuwa, VisionTrim |
| **重要性 + 多样性** | — | IDPruner (MMR), HiDivDrop (DTop-K) |
| **时序一致性** | HiDivDrop (ILVAS) | VScan (Global+Local) |

---

## 二、逐篇 Indicator 详解

---

### 1. FastV（ECCV 2024）

**信号**：LLM 浅层 self-attention（全 token 平均 received attention）

$$\phi_{\text{attn}}(v_i) = \frac{1}{H \cdot |\mathcal{Q}_i|}\sum_h \sum_{j \in \mathcal{Q}_i} A_{h,\, j \to v_i}^{(K)}, \quad \mathcal{Q}_i = \{j : j > p_i\}$$

- 在 LLM 第 $K$ 层（默认 K=2），计算每个 visual token 从**所有后续 token**（因果掩码下含后续 image token + 全部 instruction token）收到的 attention 均值
- 多头 × 多 query position 取 mean，去除 bottom $R\%$（$R$ 为裁剪比例，默认 50%），保留前 $(1{-}R)\%$ token
- **特点**：极简，training-free；"一刀切"剪枝，后续层全生效
- **局限**：第 2 层 attention 尚不稳定，高 attention token 集中一处，覆盖率低

---

### 2. PyramidDrop（CVPR 2025）

**信号**：LLM 各 stage 末尾 last instruction token → image tokens 的 attention score（渐进多阶段）

$$\text{score}_i = q_j^{t_I} \cdot (k_j^{v_i})^T$$

- 将 LLM 均分为 $S$ 个 stage（默认 $S=4$，32 层模型每 8 层一个 stage）
- 每个 stage 末尾，计算 **last instruction token** 的 query 与所有 image token 的 key 的点积作为重要性排名
- 保留 top-$\lambda$ 比例的 token，丢弃剩余：$V_s = \lambda^s \cdot V$
- 默认 $\lambda=0.5$ 时，token 数量呈指数递减：$V \to V/2 \to V/4 \to V/8$
- 直接复用 self-attention 的 Q/K 矩阵，零额外参数

**核心观察**：视觉 token 冗余度随 LLM 层数**单调递增** — 浅层（L2）丢弃任何比例都掉点，深层（L24）保留 10% 性能也不降。均匀压缩忽略了这一层级差异。

**特点**：
- 渐进式剪枝完美匹配冗余度变化规律
- **训练 + 推理双加速**（唯一同时适用于两个阶段的方法）：训练 40%+，推理 55%+，性能几乎无损
- PyramidDrop 训练迫使模型学会**更紧凑的视觉表示**（Figure 3：训练后模型在任意压缩率下都优于 vanilla）
- FlashAttention 兼容，额外开销 $O(N) \times (S-1)$，可忽略
- 更高分辨率 + PyramidDrop 的训练成本甚至**低于原始分辨率**，但性能更好

**关键性能数据**：

| 模型 | 设置 | Avg Token | Avg Perf (%) | FLOPs |
|------|------|-----------|-------------|-------|
| LLaVA-1.5-7B | Vanilla | 576 | 100% | 3.82T |
| LLaVA-1.5-7B | PyramidDrop | 192 (avg) | 96.8% | 1.78T |
| LLaVA-1.5-7B | FastV | 192 (avg) | 90.6% | 2.01T |
| LLaVA-1.5-7B | SparseVLM | 192 (avg) | 95.5% | — |
| LLaVA-1.5-7B | PyramidDrop (极端) | 64 (avg) | 87.6% | — |
| LLaVA-1.5-7B | FastV (极端) | 64 (avg) | 73.7% | — |

> 极端压缩（64 tokens）下 PyramidDrop 仍保持 87.6%，而 FastV 骤降至 73.7% — 渐进式丢弃的鲁棒性远优于一次性丢弃。

**训练加速**：

| 模型 | Vanilla GPU-h | PDrop GPU-h | 加速比 | 性能变化 |
|------|:---:|:---:|:---:|:---:|
| LLaVA-NeXT-7B (p5) | 366 | 218 | **40.4%** | −0.1 |
| LLaVA-NeXT-7B (p9) | 483 | 269 | **44.3%** | +0.6 |
| LLaVA-1.5-7B | 104 | 79 | **24.0%** | +0.7 |
| Video-LLaVA | 183 | 132 | **27.8%** | −0.01 |

**局限**：
- 纯 saliency 驱动，**无多样性设计**（高 attention token 可能聚集一处，空间覆盖差）
- 无 merge 机制，被丢弃 token 的信息完全丧失
- $\lambda$ 和 $S$ 是全局超参数，不同任务（通用 vs 细粒度 OCR）需求不同但无法自适应
- 仅在 LLaVA 系列验证，未验证 Qwen2.5-VL 等新型架构

---

### 3. VisionZip（CVPR 2025）

**信号**：Vision encoder 倒数第 2 层的 CLS attention

$$\text{Dominant score}_i = \frac{1}{H}\sum_h A_{h,\text{CLS} \to v_i}^{(L-1)}$$

- 选 top-K 为 **dominant tokens**；从剩余 token 中均匀采样少量为 target（数量 = 所需 contextual token 数），其余为 merge token，以 Key 向量点积度量相似度，将 merge token 分配到最相似的 target 做均值聚合 → **contextual tokens**
- 最终输出 = dominant tokens + contextual tokens
- **特点**：两步（select + merge），避免纯 prune 的信息损失
- **局限**：CLS attention 只捕全局显著，无文本引导，深层 attention 偏稀疏

---

### 4. SCOPE（NeurIPS 2025）

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

### 5. CDPruner（NeurIPS 2025）

**信号**：Conditional DPP kernel = 视觉多样性 × 指令相关性

$$\tilde{L}_{ij} = \tilde{r}_i \cdot \underbrace{L_{ij}}_{\cos(v_i,v_j)} \cdot \tilde{r}_j$$

Log-det 后自然分解：

$$\log\det(\tilde{L}_S) = \underbrace{\sum_{i\in S}\log \tilde{r}_i^2}_{\text{relevance}} + \underbrace{\log\det(L_S)}_{\text{diversity}}$$

- $L_{ij}$：visual token 间 cosine similarity（多样性）
- $\tilde{r}_{i}$：第 $i$ 个视觉 token 与指令 embedding 的归一化 cosine similarity（相关性）
- MAP inference via 贪心 + Cholesky 分解，复杂度 $O(nm^2)$，额外延迟 <10ms
- **特点**：优雅的 DPP 条件化，training-free；两种 embedding 方式兼容有/无 CLIP text encoder 的模型

---

### 6. DivPrune（CVPR 2025）

**信号**：纯多样性，Max-Min Distance Problem (MMDP)

$$\tilde{E}_v^* = \arg\max_{|\tilde{E}_v|=\tilde{M}} \min_{\gamma \neq \omega \in \tilde{E}_v} d(\gamma, \omega), \quad d(\gamma,\omega) = 1 - \cos(\gamma, \omega)$$

- 贪心：每步选「离已选集合最远」的 token（类似 k-center 聚类）
- 一次预计算完整距离矩阵后迭代选取
- **特点**：最纯粹的多样性导向，无需任何注意力信号，training-free
- **局限**：只关注极端 pair，忽略全局多样性；无文本引导；完全忽略 token 重要性

---

### 7. IDPruner（Arxiv 2602.13315）

**信号**：MMR（Maximal Marginal Relevance）= 重要性 - 多样性惩罚

$$v^* = \arg\max_{v_i \notin \mathcal{S}} \Bigl[\lambda \cdot \widehat{\text{Imp}}(v_i) - (1-\lambda) \cdot \underbrace{\max_{v_j \in \mathcal{S}} \cos(v_i,v_j)}_{m_i}\Bigr]$$

- $\text{Imp}$ 来自 **VisionSelector** 的 DiffTopK 输出（端到端训练）+ min-max 归一化
- $m_{i}$ = 候选 token 与已选集合中任意 token 的最大余弦相似度（用 max 而非 mean）
- 高效 O(KN) 更新：维护 m 向量，每步 `m ← max(m, sim(V, v*))`
- **特点**：显式平衡重要性与多样性，λ 可调；与 SCOPE 的区别：用减法而非乘法组合
- ⚠️ **非 training-free**：依赖 VisionSelector（需训练可学习评分模块）

---

### 8. VisionTrim（ICLR 2026）

**DVTS 信号**：Global CLS attention + Local LTAM（dual-kernel）自适应融合

$$S_i = \alpha \hat{S}_i^g + (1-\alpha) S_i^l, \quad \alpha = \frac{\sigma_l^2}{\sigma_g^2 + \sigma_l^2}$$

- **Global** $S_{i}^{g}$：ViT 倒数第 2 层 CLS attention，多头均值后 softmax
- **Local** $S_{i}^{l}$（LTAM）：局部 k×k 窗口内的 dual-kernel 亲和度
  - $\kappa_{\text{feat}}$：特征空间高斯距离
  - $\kappa_{\text{pos}}$：空间位置高斯距离
  - $\kappa^{*} = \kappa_{\text{feat}} + w_{3} \kappa_{\text{pos}}$
- **自适应权重**：方差大 = 信号不稳定 → 给小权重

**TGVC 信号**（补充）：CLIP text encoder → text-visual similarity → 文本引导聚类 merge

- **特点**：multi-stage（ViT + LLM 各一次），prune+merge 两用

---

### 9. HoloV（NeurIPS 2025）

**信号**：Diversity Variance + CLS Attention，crop-wise 自适应分配

$$\mathcal{H}_i^c = \gamma_c \mathcal{V}_i^c + \mathcal{A}_i^c, \quad \gamma_c = \frac{\mathbb{E}[\|\mathcal{A}^c\|]}{\mathbb{E}[\|\mathcal{V}^c\|]}$$

- **多样性方差** $\mathcal{V}_{i}^{c}$：token $i$ 与 crop 内其他 token 相似度的方差
  - 高方差 = 该 token 与周围差异大 = 语义独特
- $\mathcal{A}_{i}^{c}$：CLS attention（全局显著性）
- 自适应缩放因子 $\gamma_{c}$ 确保两者量级一致
- **Crop-wise 分配**：$w_{c} \propto$ crop 平均 holistic score，配额 $q_{c}$ 按权重分配（设上下限防垄断）
- **特点**：在 LLM 之前剪枝（prefill 超线性收益）；另有 Fast VCR 在高不确定性时补回信息

---

### 10. HiDivDrop（ICLR 2026）

**两层 indicator 设计：ILVAS（在哪剪）+ DTop-K（剪什么）**

**ILVAS**（Inter-Layer Visual Attention Similarity）：

$$\text{ILVAS}(l) = \text{sim}\left(\tilde{A}_i^{(l)},\ \tilde{A}_i^{(l+n)}\right)$$

- 衡量 layer $l$ 对 token 重要性的评估在未来层的稳定性
- 选 ILVAS 曲线的局部最大值作为剪枝层：`F = {10, 14, 16, 18}`

**DTop-K**（可微 token 选择）：

1. 重要性分数 $c_{i}$ → 归一化排名 $c'_{i} \in [0,1]$  
2. 软掩码：$\text{Mask}_{i} = \sigma\!\left(\lambda(c'_{i} - a)\right)$，$a$ 是**可学习的剪枝阈值**  
3. 前向用 hard threshold，反向用 sigmoid 梯度

- **三段式架构**：Late Injection（跳过浅层）+ Concave Pyramid（中间层渐进）+ Early Exit（深层停止）
- **特点**：唯一将"剪枝时机"作为 indicator 研究的方法；DTop-K 使 indicator 端到端可学习
- ⚠️ 需要训练

---

### 11. VScan（TMLR 2026）

**三阶段 indicator**：

| 阶段 | Indicator | 来源 |
|------|-----------|------|
| Global Scan | 深层 CLS attention（output layer） | Vision Encoder |
| Local Scan | 浅层 CLS attention（layer 6）+ 非重叠窗口划分 | Vision Encoder |
| Middle Layer Pruning | last instruction token → visual tokens attention | LLM 第 16 层 |

- Global 和 Local 各取 $R_{1}/2$，union 后做 Token Merging（cosine similarity → average pooling）
- **中间层而非早期层**：实验验证 k=16 >> k=2（FastV 的位置），避免位置偏差
- **特点**：training-free，兼容 FlashAttention（auxiliary vanilla attention pass）

---

### 12. SparseVLM（ICML 2025）

**信号**：LLM 内部 text→visual 注意力，自适应 rank-based 剪枝量

$$P = A[\mathbb{L}, \mathbb{I}] \in \mathbb{R}^{L_t \times L_v}, \quad \tilde{p}_j = \frac{1}{|\text{raters}|}\sum_{i \in \text{raters}} P_{ij}$$

- **Text Rater 筛选**：先用 $H_{v} \cdot H_{q}^{T}$ 过滤出视觉相关文本 token（超均值才成为 rater），排除代词/介词噪声
- **Rank-based 自适应裁剪量**：$N = \lambda \times (L_{v} - \text{rank}(P))$
  - 低 rank → 高冗余 → 多剪；每层自动决定裁剪量
- Token Recycling：density peak clustering → 求和重构，减少信息损失
- **特点**：逐层自适应，无需预设固定压缩率；training-free

---

### 13. Nuwa（Arxiv 2602.02951）

**Stage 1 信号**：CLS attention × Key vector L2-norm

$$S(t_i) = \alpha_{\text{cls},i} \times \|\mathbf{k}_i\|_2$$

- **双乘积**：CLS attention 给全局重要性，key L2-norm 给信息容量
- Grid 分区内选 top-k 作为 benchmark token
- 高 L2-norm 的 **Pillar Token** 不参与聚合（保护 ViT register token）
- 聚合权重 = 语义相似度 × 空间邻近度（防止语义相似但位置遥远的 token 被合并）

**Stage 2 信号**（LLM 中间层）：text-visual cosine similarity（多模态对齐完成后再剪）

- **特点**：专为空间定位任务设计，是目前对「空间保持」研究最深入的方法

---

### 14. FSR（Arxiv 2602.05809）

**三段式 indicator，模拟人类视觉感知**

**Focus**（局部证据，动态 budget）：

$$\phi_i = \alpha \hat{r}_i + \beta \hat{s}_i, \quad K_F = \min\left\lbrace k : \sum_{j=1}^k \phi_{\pi(j)} \geq \rho \cdot Z \right\rbrace$$

- $s_{i}$：CLS attention（视觉显著性）
- $r_{i}$：CLIP text encoder cosine similarity（指令相关性）
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
| **FastV** | ECCV 2024 | LLM 浅层 attn | ✅（全 token received attn，L2 单次） | ✗ | LLM 内（L2） | ✅ |
| **PyramidDrop** | CVPR 2025 | LLM last-instr attn | ✅（渐进多阶段） | ✗ | LLM 内（多层） | ✅ |
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
- PyramidDrop：LLM 内多层（S=4 个 stage，每 stage 末尾丢弃一次）— 首个将「渐进式多阶段」引入 LLM 内部的方法
- VisionZip/DivPrune/SCOPE/CDPruner：Vision Encoder 后
- VisionTrim/VScan/Nuwa：ViT + LLM 双阶段
- HiDivDrop：浅层（Late Injection）+ 中间层（渐进）+ 深层（Early Exit）三段式

### 5. Merge 优于纯 Prune

VisionZip、VisionTrim (TGVC)、VScan (merging)、SparseVLM (recycling)、FSR (Refine)、Nuwa (aggregation)、HoloV (VCR) 全都保留被丢弃 token 的信息，而非直接丢弃。

---

## 五、Indicator 分类体系

> 二分法（Saliency / Diversity）是常见框架，但不够完整。以下提出一个二维分类（A: Saliency × B: Representativeness），更准确地描述 14 篇方法的 indicator 设计空间。

---

### 5.1 两个正交维度

---

#### 维度 A：Saliency（个体重要性）

**核心问题**：这个 token 本身有多重要？  
**操作对象**：单个 token 的评分，独立于其他已选 token。

Saliency 内部存在**本质分裂**，必须再细分：

##### A1. Visual Saliency（视觉显著性，task-agnostic）

> *"Is this token intrinsically important in the image, regardless of the question?"*

信号来自 **视觉编码器内部**，与用户问题无关。对同一张图，不论问什么，评分结果相同。

- **典型信号**：ViT 倒数第 2 层 CLS token 的 attention weight（`A_CLS→vi`）
- **代表方法**：VisionZip、SCOPE（saliency 项）、HoloV（`𝒜ᶜ` 项）、VScan（Global/Local CLS）、VisionTrim（`Sᵍ` 项）、FSR（`sᵢ`）、Nuwa（`α_cls` × `‖kᵢ‖₂`）、IDPruner（VisionSelector）、HiDivDrop（DTop-K）

> ⚠️ **FastV 不属于 A1**：FastV 在 LLM 第 2 层计算每个 visual token 从所有后续 token（含后续 image token + 全部 instruction token）收到的平均 attention。由于 instruction token 参与了 attention 均值，信号依赖文本输入（不同问题 → instruction 对 visual token 的 attention 不同 → 排名不同），属于 A2。但 image-to-image attention（text-agnostic）也参与平均，加之 L2 文本-视觉交互尚浅，text dependency 被显著稀释，实际分布近似 task-agnostic saliency。

##### A2. Task Relevance（任务相关性，task-aware）

> *"Is this token relevant to the user's current query?"*

信号必须结合文本输入计算，**同一张图不同问题评分不同**，实现动态剪枝。

- **典型信号**：
  - visual-text cosine similarity（CLIP text encoder 编码问题）
  - LLM 内部 cross-attention（text token → visual token）
- **代表方法**：
  - *LLM 内部 attention 路线*：FastV（LLM L2 全 token → visual received attn 均值）、PyramidDrop（multi-stage last-instr Q·K → visual attn）、SparseVLM（text rater 筛选后 text → visual attn）、VScan（LLM L16 last-instr attn）、Nuwa（Stage 2 text-visual cos sim）
  - *CLIP text encoder 路线*：CDPruner（`r̃ᵢ` = cos(vᵢ, instr_emb)）、FSR（`rᵢ` = CLIP text cos sim）、VisionTrim（TGVC `S_t2v` = CLIP text → visual sim）

> ⚠️ **A1 与 A2 的根本区别**：A1 只需 vision encoder，推理时无额外文本交互；A2 需要语言侧信息，天然支持 per-query 个性化剪枝，但依赖 CLIP text encoder 或 LLM 中间层激活的可访问性。

---

#### 维度 B：Representativeness（集合代表性）

**核心问题**：选出的 token 子集是否充分代表了原始 token 集合——在语义空间和物理空间两个层面？  
**操作对象**：**集合级别（set-level）**的目标函数或几何约束，单个 token 的价值依赖于已选集合或其空间位置。

根据代表性所保障的空间不同，分为两个子维度：

##### B1. Semantic Diversity（语义多样性）

> *"Are the selected tokens semantically non-redundant in the embedding space?"*

在特征/语义空间中确保所选子集不冗余。四种主要数学实现，性质各异：

| 技术路线 | 优化目标 | 数学工具 | 理论保证 | 代表方法 |
|---------|---------|---------|---------|---------|
| **极端距离**（Pairwise） | $\max \min_{i \neq j \in S} d(v_{i}, v_{j})$ | MMDP | 2-近似 | DivPrune |
| **集合体积**（Volumetric） | $\max \det(\tilde{L}_{S})$ | DPP | NP-hard，贪心 $(1{-}1/e)$ | CDPruner |
| **软覆盖**（Coverage） | $\max \sum_{u} \max_{s \in S} \text{sim}(u, s)$ | Submodular | 贪心 $(1{-}1/e)$ | SCOPE |
| **边际增益**（Marginal） | $\max \lambda \cdot \text{Imp}(v) - (1{-}\lambda) \cdot \max_{j \in S}\text{sim}(v,v_{j})$ | MMR（贪心迭代） | — | IDPruner |

**四种实现的核心区别**：
- **MMDP** 只关注最近邻的那对 token，一旦最小距离确定，其余 pair 不影响结果。对极端情况敏感，不关注全局分布。
- **DPP** 最大化所有向量张成的超体积，天然考虑全局几何结构，但计算更重（行列式）。
- **Submodular Coverage** 把「每个 token 被最近邻代表的程度」累加，等价于 facility-location 问题，直观且可扩展。
- **MMR** 是唯一把 Saliency 和 Diversity **显式加权相减**的方案，λ 提供连续可调的权衡，但无全局近似保证。

##### B2. Spatial Coverage（空间覆盖）

> *"Have we preserved at least one representative token from every spatial region?"*

在物理二维空间中确保所选 token 覆盖图像各区域，与语义无关。B1（语义多样性）和纯 Saliency 方法均可能在空间上失效：高 saliency token 往往集中于图像中心/前景；高语义 diversity 子集可能几何上偏斜（如全选图像一角的多种纹理）。

**实现机制**：将 token 网格划分为不重叠的局部区域（crop/grid/window），在每个区域内独立选取，从几何上保证不遗漏任何角落。

- **代表方法**：HoloV（crop-wise adaptive allocation）、Nuwa（M×M grid partition）、VScan（Local Scan windows）
- ⚠️ **VisionTrim 不在此列**：LTAM 使用局部 k×k 窗口计算 per-token 亲和度分数，是局部 saliency 信号（A1），而非「分区后各区独立选 token」的空间覆盖保证机制。

---

### 5.2 各方法的维度归属

> 各维度定义参见 5.1。每格给出该方法在对应维度的核心公式与机制概述，"—" 表示该方法不涉及此维度。

| 方法 | A1 Visual Saliency | A2 Task Relevance | B1 Semantic Diversity | B2 Spatial Coverage |
|------|-----|-----|-----|-----|
| **PyramidDrop** | — | $\text{score}\_i = q\_j^{t\_I} \cdot (k\_j^{v\_i})^T$：每个 stage 末尾计算 last-instruction token 的 query 与各 image token 的 key 的点积，按分数降序保留比例 $\lambda$（$\lambda \in (0,1)$，默认 0.5）的 token，丢弃其余。复用 self-attention 的 Q/K 矩阵，零额外参数。默认 $S{=}4$ stage，第 $s$ 阶段后剩余 token 数为 $V\_s = \lambda^s V$，呈指数递减 | — | — |
| **FastV** | — | $\phi\_{\text{attn}}(v\_i) = \text{mean}\_{h,\, j \in \mathcal{Q}\_i} A\_{h,\, j \to v\_i}^{(K=2)}$，$\mathcal{Q}\_i = \{j : j > p\_i\}$：在 LLM 第 2 层计算每个 visual token 从所有后续 token（因果掩码下含后续 image token + 全部 instruction token）收到的 attention 均值（多头 × 多 query position 取 mean），one-shot 去除 bottom $R\%$（$R$ 为裁剪比例，默认 50%），保留前 $(1{-}R)\%$ token。因 instruction token 参与均值，信号隐式依赖文本输入（不同问题 → 不同排名），但 image-to-image attention 也参与平均，加之 L2 交互极浅，text dependency 被显著稀释，实际分布近似 task-agnostic saliency | — | — |
| **VisionZip** | 取 ViT 倒数第 2 层 CLS token 对各 visual token 的 attention 多头均值作为 saliency score，选分数最高的 K 个为 dominant tokens。信号完全来自 vision encoder，与文本无关。对无 CLS token 的模型（如 SigLIP），改用每个 token 从所有其他 token 收到的平均 attention | — | Token Merging（Algorithm 2）：从剩余非 dominant token 中**均匀采样**少量 token 作为 target（数量 = 期望的 contextual token 数，如 10 个），其余全部为 merge token；以 Key 向量点积度量相似度，将每个 merge token 分配到最相似的 target 上做**均值聚合**，生成 contextual tokens。最终输出 = dominant tokens + contextual tokens。属信息保留机制，非显式 diversity 目标 | — |
| **DivPrune** | — | — | MMDP（Max-Min Diversity Problem）：从全部视觉 token 中选出目标数量的子集，使子集内任意两 token 间余弦距离（= 1 − cosine similarity）的最小值最大化。预计算全 token 对距离矩阵（一次矩阵乘法）后，贪心迭代每步选离已选集合最远的 token（farthest-first traversal）。纯多样性驱动，不使用任何 saliency 信号，2-近似保证 | — |
| **SCOPE** | $A\_v^\alpha$：取 ViT 倒数第 2 层 CLS attention 作为 per-token saliency，指数 $\alpha$（默认 1.0）控制 saliency 权重。最终选择准则 $v^* = \arg\max\_{v \notin \mathcal{S}} \Delta(v;\mathcal{S}) \cdot A\_v^\alpha$，以乘法融合 saliency 与 coverage marginal gain | — | Submodular Coverage：边际增量 $\Delta(v;\mathcal{S}) = \sum\_{u \in \mathcal{V}} \max(\text{sim}(u,v) - C(u,\mathcal{S}),\, 0)$，其中 $C(u,\mathcal{S}) = \max\_{s \in \mathcal{S}} \text{sim}(u,s)$ 为 $u$ 在已选集合中的最大覆盖度。底层集合覆盖函数 $f(\mathcal{S}) = \sum\_u C(u,\mathcal{S})$ 是单调次模函数，纯 coverage 贪心具 $(1{-}1/e)$ 近似保证。作者实验证明纯 saliency 选择的 $\theta$-coverage 甚至低于 random | — |
| **CDPruner** | — | $\tilde{r}\_i = \frac{\cos(v\_i, e\_{\text{instr}})}{\sum\_k \cos(v\_k, e\_{\text{instr}})}$：将每个 visual token 与指令 embedding 的 cosine similarity 归一化，作为 task-relevance 权重。兼容两种 embedding 来源：CLIP text encoder（有独立文本编码器的架构）或 LLM 首层 hidden state（无独立编码器时） | Conditional DPP：条件核 $\tilde{L}\_{ij} = \tilde{r}\_i \cdot \cos(v\_i,v\_j) \cdot \tilde{r}\_j$，使 $\log\det(\tilde{L}\_S) = \sum\_{i \in S} \log \tilde{r}\_i^2 + \log\det(L\_S)$，将 relevance 与 diversity 天然分解到行列式中联合优化。MAP inference 经贪心 + Cholesky 增量分解实现，$O(nm^2)$，实测额外延迟 <10ms | — |
| **SparseVLM** | — | $\tilde{p}\_j = \frac{1}{\lvert\text{raters}\rvert} \sum\_{i \in \text{raters}} P\_{ij}$，$P = A[\mathbb{L}, \mathbb{I}]$：从 LLM self-attention 中抽取 text→visual 子矩阵 $P$；先以 $H\_v \cdot H\_q^T$ 过滤出视觉相关的 text rater（超均值才入选，排除代词、介词等噪声 token），再对筛选后的 rater 取 attention 均值作为每个 visual token 的重要性分数 | Rank-adaptive budget + Merge：$N = \lambda (L\_v - \text{rank}(P))$，以 attention 矩阵 $P$ 的秩衡量冗余度——低秩意味着 visual token 间高度冗余，因此分配更大裁剪量。该机制逐层独立运行，无需预设固定压缩率。被裁剪 token 经 density peak clustering 聚合后以 Token Recycling 求和重构到保留 token 中 | — |
| **VisionTrim** | DVTS 双信号自适应融合：$S\_i = \alpha \hat{S}\_i^g + (1{-}\alpha) S\_i^l$，$\alpha = \sigma\_l^2 / (\sigma\_g^2 + \sigma\_l^2)$（逆方差加权：等价于 $\alpha \propto 1/\sigma\_g^2$，即全局信号方差越大，其权重 $\alpha$ 越小，自动倾向更稳定的信号源）。全局信号 $\hat{S}\_i^g$：ViT 倒数第 2 层 CLS attention 多头均值经 softmax 归一化。局部信号 $S\_i^l$（LTAM）：在 $k{\times}k$ 局部窗口内以 dual-kernel $\kappa^* = \kappa\_{\text{feat}} + w\_3 \kappa\_{\text{pos}}$（特征空间高斯 + 空间位置高斯）计算每个 token 与其邻域的亲和度均值，转换为概率分布后作为 per-token 局部显著性分数 | TGVC $S\_{t2v}$：以 CLIP text encoder 编码用户指令，计算指令 embedding 与各 visual token 的 cosine similarity，得到 per-token 文本相关性分数，驱动后续聚类 merge 阶段的 token 选择 | TGVC merge：以 $S\_{t2v}$ 对 token 排序，将低相关性 token 按语义相似度聚类到高 $S\_{t2v}$ 的 anchor token 上，加权聚合保留语义信息 | — |
| **HoloV** | $\mathcal{A}\_i^c$：ViT CLS attention 多头均值，衡量每个 token 在 crop $c$ 中的全局语义显著性。作为 holistic score 的基础分量参与后续融合 | — | $\mathcal{V}\_i^c$：token $i$ 与 crop 内所有其他 token 的 cosine similarity 的**方差**（高方差表明该 token 与邻居差异大，语义独特性强）。Holistic score $\mathcal{H}\_i^c = \gamma\_c \mathcal{V}\_i^c + \mathcal{A}\_i^c$，其中 $\gamma\_c = \mathbb{E}[\lVert\mathcal{A}^c\rVert] / \mathbb{E}[\lVert\mathcal{V}^c\rVert]$ 自适应缩放使两项量级一致。该设计为 per-token 局部唯一性度量，非 MMDP/DPP/Submodular 式集合级目标。另有 Fast VCR 机制在模型输出高不确定性时动态补回被剪 token | Crop-wise adaptive allocation：$w\_c \propto \text{avg}(\mathcal{H}^c)$，每个 crop 按平均 holistic score 分配 token 配额 $q\_c$，并设上下限防止单一 crop 垄断所有额度，从几何上保证多区域覆盖 |
| **HiDivDrop** | DTop-K（可微 token 选择）：对每个 token 的重要性分数 $c\_i$ 做归一化排名 $c'\_i \in [0,1]$，再施加软掩码 $\text{Mask}\_i = \sigma(\lambda(c'\_i - a))$，其中 $a$ 为可学习剪枝阈值。前向传播使用 hard threshold，反向传播以 sigmoid 梯度近似实现端到端训练。⚠️ 非 training-free。另：$\text{ILVAS}(l) = \text{sim}(\tilde{A}^{(l)}, \tilde{A}^{(l+n)})$ 评估各层 attention 排名的跨层稳定性，选 ILVAS 曲线**局部最大值**作为剪枝层位（$F = \{10, 14, 16, 18\}$），决定在哪层剪枝而非剪哪些 token | — | —（方法名称含 "Div" 但多样性体现于三段式架构设计——Late Injection 跳过浅层 / Concave Pyramid 中间层渐进 / Early Exit 深层停止——而非 indicator 层面的显式 diversity 目标函数） | — |
| **VScan** | 双尺度 union：Global Scan 取 ViT **output layer** 的 CLS attention 捕捉高层语义显著性，Local Scan 取 ViT **layer 6** 的 CLS attention 保留低层细节信息，两组各选 $R\_1/2$ 个 token 后取 union 实现多粒度互补 | LLM **第 16 层** last-instruction token 对 visual tokens 的 attention。关键设计选择：选中间层而非浅层（实验验证 k=16 远优于 k=2），因浅层 attention 存在位置偏差。以 auxiliary vanilla attention pass 提取 attention map，与 FlashAttention 兼容 | Global/Local union 后对 selected tokens 做 cosine similarity → average pooling token merging，将高相似 token 对聚合为单个代表 token 以保留被丢弃 token 的信息。属信息保留机制，非显式 diversity 目标 | Local Scan 将 token grid 划分为非重叠窗口，每个窗口内独立选取 top-k，从几何上保证图像各空间区域至少保有代表 token，避免全局 CLS attention 导致选择集中于单一区域 |
| **Nuwa** | $S(t\_i) = \alpha\_{\text{cls},i} \times \lVert \mathbf{k}\_i \rVert\_2$：将 ViT CLS attention 与 key vector 的 L2-norm 相乘——前者衡量全局重要性，后者反映信息容量。高 $\lVert k \rVert$ 的 **Pillar Token**（ViT 的 register token）被识别并保护，不参与后续聚合 | Stage 2（LLM 中间层）：在多模态 embedding 经过足够层数的交互充分对齐之后，以 text-visual cosine similarity 作为 indicator 执行第二次剪枝，选择性保留与当前指令高相关的 token | 聚合权重 = 语义 cosine similarity × 空间邻近度高斯核，双约束确保只有语义相近且空间相邻的 token 才被合并，防止语义相似但物理位置遥远的 token 被错误归并。属空间感知的信息保留机制 | M×M grid 分区：将 token grid 划分为 $M \times M$ 个子区域，每区域内独立选取 top-k benchmark token，从几何上保证空间均匀覆盖。该设计专为 OCR、REC 等需要精确空间定位的任务优化 |
| **FSR** | $s\_i$：ViT CLS attention 衡量每个 token 的视觉显著性，与 $r\_i$ 共同加权组合构成 Focus score $\phi\_i$ | $r\_i$：CLIP text encoder 编码指令后与各 visual token 计算 cosine similarity 得到指令相关性分数。Focus score 融合为 $\phi\_i = \alpha \hat{r}\_i + \beta \hat{s}\_i$（$\hat{\cdot}$ 为归一化），并以动态 budget $K\_F = \min\{k : \sum\_{j=1}^k \phi\_{\pi(j)} \geq \rho Z\}$（默认 $\rho{=}0.9$）自适应决定保留 token 数——图像语义越简单，Focus token 越少 | CCS（Conditional Context Sampling）：≈ Farthest Point Sampling 变体，每步选与 Focus + 已选 Scan token 的综合相似度**最低**的 token 以最大化信息增量，具理论覆盖界保证。Refine 阶段：剩余 token 按 similarity 分配到最近 Scan center 并加权聚合（仅修改 Scan token，保护 Focus token 高保真度） | — |
| **IDPruner** | $\widehat{\text{Imp}}(v\_i)$：由 VisionSelector 模块以 DiffTopK 机制生成 + min-max 归一化。VisionSelector 仅以 visual feature 为输入，不使用文本信息，属**学习型 A1**（⚠️ 非 training-free，需端到端训练） | — | MMR：$v^* = \arg\max\_{v\_i \notin \mathcal{S}}[\lambda \cdot \widehat{\text{Imp}}(v\_i) - (1{-}\lambda) \cdot \max\_{v\_j \in \mathcal{S}} \cos(v\_i, v\_j)]$，以**减法**显式平衡 importance 与 diversity（对比 SCOPE 的乘法融合），$\lambda{=}0.5$ 最优。$O(KN)$ 复杂度，维护 max-similarity 向量逐步增量更新。无全局近似保证（vs DPP/Submodular 的 $(1{-}1/e)$） | — |

### 5.3 分类体系的核心观察

**观察 1：纯 saliency → saliency + B1 的演进轨迹**  
从 VisionZip（纯 A1）→ SCOPE（A1 + B1 Submodular Coverage）→ CDPruner（A2 + B1 DPP），主线是在重要性评分基础上叠加语义多样性约束。FastV 和 PyramidDrop 走的是另一条轨迹——同样利用 LLM 内部 attention 信号（A2），但完全放弃 B1，专注于剪枝策略的效率设计（FastV 单次 L2 全 token received attn → PyramidDrop 多阶段 last-instr Q·K），说明 "saliency-only" 的设计空间通过工程优化仍有很大价值。

**观察 2：A2（Task Relevance）信号的两个代际**  
A2 信号从 FastV（2024）就已出现（LLM L2 全 token received attn 均值），并非 2025 年之后的新发明。但两代方法在设计意图上有本质区别：
- **第一代（隐式 A2）**：FastV（全 token received attn 均值，text 信号被 img-to-img attn 稀释）、PyramidDrop（last-instr Q·K 点积，text 信号更集中）——使用 LLM attention 作为**显著性代理**，设计目标是效率而非文本语义对齐，text 信号是副产品
- **第二代（显式 A2）**：CDPruner、SparseVLM、VisionTrim、Nuwa、FSR、VScan——**有意设计**文本引导信号，目标是让剪枝结果对当前 query 敏感，实现 per-query 个性化选择  

信号来源进一步分化为两条技术路线：
- **CLIP text encoder 路线**（CDPruner、VisionTrim、FSR）：轻量，但依赖 CLIP 架构，对 Qwen2.5-VL 等无单独 CLIP encoder 的新型架构需适配
- **LLM 内部 attention 路线**（SparseVLM、VScan L16、Nuwa Stage 2）：更通用，但需要额外 forward pass 或访问中间层激活

**观察 3：B2（空间覆盖）是被长期忽视的隐性瓶颈**  
DivPrune 和 SCOPE 纯粹在语义空间操作（仅有 B1，无 B2），对 OCR、空间关系推理、REC 等需要精确空间定位的任务存在系统性劣势。Nuwa 的实验明确验证了这一点。

**观察 4：MMDP 在 B1 各实现中理论最弱**  
MMDP 仅优化最近邻 pair，是对全局多样性的粗糙近似。DPP 和 Submodular Coverage 均有更严格的全集近似保证，且 Submodular Coverage 在直觉上更贴近"让每个 token 都被代表"这一目标。

**观察 5：A1 + B2 的组合尚未被充分探索**  
目前多数方法是 A1+B1 或 A2+B1 的组合，而 A1+B2（空间感知的视觉显著性，无需文本）在理论上是一个被低估的设计点——HoloV 是最接近的，但其 Diversity Variance 仍是语义信号（B1）而非纯几何信号（B2）。
