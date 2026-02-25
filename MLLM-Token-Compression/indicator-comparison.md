# Visual Token Pruning：Indicator 全景对比

> 整理时间：2026-02-25  
> 覆盖论文：FastV · PyramidDrop · VisionZip · DivPrune · SCOPE · CDPruner · SparseVLM · VisionTrim · HoloV · VScan · Nuwa · FSR · IDPruner · HiDivDrop（共 14 篇）

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

$$\text{score}_i = \frac{1}{H}\sum_h A_{h,\ t_{\text{last}} \to v_i}^{(K)}$$

- 在 LLM 第 $K$ 层（默认 K=2），取最后一个文本 token 对各视觉 token 的 attention
- 多头取平均，直接 top-R% 选取
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

- 选 top-K 为 **dominant tokens**，剩余按 cosine similarity 合并进最相近的 dominant token → **contextual tokens**
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

$$\tilde{E}_v^* = \arg\max_{|\tilde{E}_v|=\tilde{M}} \min_{\gamma,\omega \in \tilde{E}_v} d(\gamma, \omega), \quad d = 1 - \text{cos}$$

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
| **FastV** | ECCV 2024 | LLM 浅层 attn | ✅（last token，L2 单次） | ✗ | LLM 内（L2） | ✅ |
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

> 二分法（Saliency / Diversity）是常见框架，但不够完整。以下提出一个三维分类，更准确地描述 14 篇方法的 indicator 设计空间。

---

### 5.1 三个正交维度

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

> ⚠️ **FastV 不属于 A1**：虽然 FastV 在 LLM 第 2 层取 attention，但其公式 $A_{t_\text{last} \to v_i}^{(K=2)}$ 明确使用**最后一个文本 token**作为 query，信号依赖文本输入（不同问题 → 不同排名），严格来说属于 A2。之所以常被误归为 A1，是因为第 2 层文本-视觉交互尚浅，attention 分布近似于 task-agnostic saliency。

##### A2. Task Relevance（任务相关性，task-aware）

> *"Is this token relevant to the user's current query?"*

信号必须结合文本输入计算，**同一张图不同问题评分不同**，实现动态剪枝。

- **典型信号**：
  - visual-text cosine similarity（CLIP text encoder 编码问题）
  - LLM 内部 cross-attention（text token → visual token）
- **代表方法**：
  - *LLM 内部 attention 路线*：FastV（LLM L2 last-token → visual attn）、PyramidDrop（multi-stage last-instr → visual attn）、SparseVLM（text rater 筛选后 text → visual attn）、VScan（LLM L16 last-instr attn）、Nuwa（Stage 2 text-visual cos sim）
  - *CLIP text encoder 路线*：CDPruner（`r̃ᵢ` = cos(vᵢ, instr_emb)）、FSR（`rᵢ` = CLIP text cos sim）、VisionTrim（TGVC `S_t2v` = CLIP text → visual sim）

> ⚠️ **A1 与 A2 的根本区别**：A1 只需 vision encoder，推理时无额外文本交互；A2 需要语言侧信息，天然支持 per-query 个性化剪枝，但依赖 CLIP text encoder 或 LLM 中间层激活的可访问性。

---

#### 维度 B：Diversity（集合多样性）

**核心问题**：选出的 token 子集语义是否足够不冗余？  
**操作对象**：**集合级别（set-level）**的目标函数，单个 token 的价值依赖于已选集合。

Diversity 内部同样有四种数学实现，性质不同：

| 子类 | 优化目标 | 数学工具 | 理论保证 | 代表方法 |
|------|---------|---------|---------|---------|
| **B1 极端距离**（Pairwise） | $\max \min_{i \neq j \in S} d(v_{i}, v_{j})$ | MMDP | 2-近似 | DivPrune |
| **B2 集合体积**（Volumetric） | $\max \det(\tilde{L}_{S})$ | DPP | NP-hard，贪心 $(1{-}1/e)$ | CDPruner |
| **B3 软覆盖**（Coverage） | $\max \sum_{u} \max_{s \in S} \text{sim}(u, s)$ | Submodular | 贪心 $(1{-}1/e)$ | SCOPE |
| **B4 边际增益**（Marginal） | $\max \lambda \cdot \text{Imp}(v) - (1{-}\lambda) \cdot \max_{j \in S}\text{sim}(v,v_{j})$ | MMR（贪心迭代） | — | IDPruner |

**B1 vs B2 vs B3 的核心区别**：
- **B1（MMDP）** 只关注最近邻的那对 token，一旦最小距离确定，其余 pair 不影响结果。对极端情况敏感，不关注全局分布。
- **B2（DPP）** 最大化所有向量张成的超体积，天然考虑全局几何结构，但计算更重（行列式）。
- **B3（Submodular Coverage）** 把「每个 token 被最近邻代表的程度」累加，等价于 facility-location 问题，直观且可扩展。
- **B4（MMR）** 是唯一把 Saliency 和 Diversity **显式加权相减**的方案，λ 提供连续可调的权衡，但无全局近似保证。

---

#### 维度 C：Spatial Coverage（空间覆盖）

**核心问题**：选出的 token 是否覆盖了图像的每个空间区域？  
**操作对象**：**几何约束**，确保二维空间的均匀分布，与语义无关。

> *"Have we preserved at least one representative token from every spatial region?"*

这一维度在 Saliency-only 和 Diversity-only 方法中均可能失效：高 saliency token 往往集中于图像中心/前景；高 diversity 子集在语义上分散但可能几何上偏斜（如全选图像一角的多种纹理）。

**实现机制**：将 token 网格划分为不重叠的局部区域（crop/grid/window），在每个区域内独立选取，从几何上保证不遗漏任何角落。

- **代表方法**：HoloV（crop-wise adaptive allocation）、Nuwa（M×M grid partition）、VScan（Local Scan windows）
- ⚠️ **VisionTrim 不在此列**：LTAM 使用局部 k×k 窗口计算 per-token 亲和度分数，是局部 saliency 信号（A1），而非「分区后各区独立选 token」的空间覆盖保证机制。

---

#### 维度 D：Stability（层间稳定性）[次要维度]

**核心问题**：该层对 token 重要性的判断，在后续层是否仍然成立？  
**代表方法**：HiDivDrop（ILVAS）

这是唯一一个把「indicator 可靠性」本身作为信号的方法，用于**决定在哪一层剪枝**，而非剪哪些 token。

---

### 5.2 各方法的维度归属

> 每个单元格给出**核心公式 + 机制概述**，"—" 表示该方法不涉及此维度。

---

#### PyramidDrop（CVPR 2025）

| 维度 | 内容 |
|------|------|
| **A1** | — |
| **A2** | $\text{score}_i = q_j^{t_I} \cdot (k_j^{v_i})^T$，各 stage 末尾取 last instruction token 的 query 与 image token 的 key 点积排名。复用 self-attn Q/K，零额外参数。保留 top-$\lambda$，token 数 $V_s = \lambda^s V$ 指数递减（默认 $\lambda{=}0.5$, $S{=}4$：$V \to V/2 \to V/4 \to V/8$）。 |
| **B** | — |
| **C** | — |
| **D** | — |

---

#### FastV（ECCV 2024）

| 维度 | 内容 |
|------|------|
| **A1** | — |
| **A2** | $\text{score}_i = \frac{1}{H}\sum_h A_{h,\, t_{\text{last}} \to v_i}^{(K=2)}$，LLM 第 2 层最后文本 token 对各 visual token 的 attention 多头均值，一次性 top-R% 保留，后续层全生效。信号依赖文本输入（$t_\text{last}$ 是指令末 token），但在 L2 文本-视觉交互极浅，实际表现接近 task-agnostic saliency。 |
| **B** | — |
| **C** | — |
| **D** | — |

---

#### VisionZip（CVPR 2025）

| 维度 | 内容 |
|------|------|
| **A1** | $\text{Dominant}_i = \frac{1}{H}\sum_h A_{h,\text{CLS} \to v_i}^{(L-1)}$，ViT 倒数第 2 层 CLS attention 多头均值，选 top-K 为 dominant tokens。信号来自 vision encoder 内部，与文本无关。 |
| **A2** | — |
| **B** | Merge：剩余 token 按 cosine similarity 合并到最近的 dominant token → contextual tokens，避免纯 prune 的信息丢失。非显式 diversity 目标，而是信息保留机制。 |
| **C** | — |
| **D** | — |

---

#### DivPrune（CVPR 2025）

| 维度 | 内容 |
|------|------|
| **A1** | — |
| **A2** | — |
| **B** | **B1 MMDP**：$\tilde{E}_v^* = \arg\max_{|\tilde{E}_v|=M} \min_{\gamma,\omega \in \tilde{E}_v} (1 - \cos(\gamma,\omega))$，贪心迭代每步选「离已选集合最远」的 token（类 k-center）。预计算完整距离矩阵。纯多样性驱动，无任何 saliency 信号。2-近似保证。 |
| **C** | — |
| **D** | — |

---

#### SCOPE（NeurIPS 2025）

| 维度 | 内容 |
|------|------|
| **A1** | $A_v^\alpha$，ViT CLS attention saliency，$\alpha$ 调节强度（默认 1.0），作为 SCOPE score 的乘法因子。 |
| **A2** | — |
| **B** | **B3 Submodular Coverage**：$\text{SCOPE}(v,\mathcal{S}) = \Delta(v;\mathcal{S}) \times A_v^\alpha$，其中 $\Delta(v;\mathcal{S}) = \sum_u \max(\text{sim}(u,v) - \max_{s \in \mathcal{S}}\text{sim}(u,s),\, 0)$。Coverage gain × saliency 乘法融合。贪心迭代选取，$(1-1/e)$ 近似保证。关键发现：纯 saliency 的 θ-coverage 甚至低于 random。 |
| **C** | — |
| **D** | — |

---

#### CDPruner（NeurIPS 2025）

| 维度 | 内容 |
|------|------|
| **A1** | — |
| **A2** | $\tilde{r}_i = \text{norm\_cos}(v_i, \text{instr\_emb})$，视觉 token 与指令 embedding 的归一化 cosine similarity。支持 CLIP text encoder 或 LLM 内部 embedding 两种来源。 |
| **B** | **B2 Conditional DPP**：$\tilde{L}_{ij} = \tilde{r}_i \cdot L_{ij} \cdot \tilde{r}_j$，其中 $L_{ij} = \cos(v_i, v_j)$。Log-det 分解：$\log\det(\tilde{L}_S) = \sum_{i \in S}\log\tilde{r}_i^2 + \log\det(L_S)$，天然分解为 relevance + diversity。MAP inference 用贪心 + Cholesky，$O(nm^2)$，<10ms。 |
| **C** | — |
| **D** | — |

---

#### SparseVLM（ICML 2025）

| 维度 | 内容 |
|------|------|
| **A1** | — |
| **A2** | $\tilde{p}_j = \frac{1}{|\text{raters}|}\sum_{i \in \text{raters}} P_{ij}$，$P = A[\mathbb{L}, \mathbb{I}]$。先用 $H_v \cdot H_q^T$ 筛选视觉相关 text rater（超均值才入选，排除代词/介词噪声），再取筛选后 text → visual attention 均值。 |
| **B** | Rank-adaptive budget + Merge：$N = \lambda(L_v - \text{rank}(P))$，低 rank → 高冗余 → 多剪，**逐层自动**决定裁剪量，无需预设固定压缩率。Token Recycling：density peak clustering → 求和重构（保留信息，非显式 diversity 目标）。 |
| **C** | — |
| **D** | — |

---

#### VisionTrim（ICLR 2026）

| 维度 | 内容 |
|------|------|
| **A1** | 双信号自适应融合（DVTS）：$S_i = \alpha \hat{S}_i^g + (1-\alpha) S_i^l$，$\alpha = \frac{\sigma_l^2}{\sigma_g^2 + \sigma_l^2}$（信号方差大 → 不稳定 → 权重小）。**全局** $S_i^g$：ViT 倒数第 2 层 CLS attention 多头均值 + softmax。**局部** $S_i^l$（LTAM）：局部 $k{\times}k$ 窗口内 dual-kernel 亲和度 $\kappa^* = \kappa_\text{feat} + w_3 \kappa_\text{pos}$（特征空间高斯距离 + 空间位置高斯距离），计算每个 token 与其局部邻域的平均亲和度作为局部显著性分数。LTAM 是 per-token 的局部 saliency 评估，非空间分区覆盖机制。 |
| **A2** | TGVC $S_{t2v}$：CLIP text encoder 编码指令 → text-visual cosine similarity，用于文本引导的聚类 merge 阶段。 |
| **B** | TGVC merge：按 $S_{t2v}$ 引导聚类，将低分 token 合并到高相关性 token，保留信息同时对齐文本语义。 |
| **C** | — （LTAM 的局部窗口是为了计算局部 saliency 分数，而非「分区 → 各区独立选 token」的空间保证机制。） |
| **D** | — |

---

#### HoloV（NeurIPS 2025）

| 维度 | 内容 |
|------|------|
| **A1** | $\mathcal{A}_i^c$：CLS attention，全局显著性信号。 |
| **A2** | — |
| **B** | $\mathcal{V}_i^c$：token $i$ 与 crop 内其他 token 相似度的**方差**（高方差 = 与邻居差异大 = 语义独特）。Holistic score $\mathcal{H}_i^c = \gamma_c \mathcal{V}_i^c + \mathcal{A}_i^c$，$\gamma_c = \frac{\mathbb{E}[\|\mathcal{A}^c\|]}{\mathbb{E}[\|\mathcal{V}^c\|]}$ 自适应缩放两者量级。注意：这是 per-token 的局部唯一性度量，非 B1-B4 集合级目标函数。另有 Fast VCR 在高不确定性时补回被剪 token 信息。 |
| **C** | Crop-wise adaptive allocation：$w_c \propto \text{avg}(\mathcal{H}^c)$，配额 $q_c$ 按 crop 权重分配，设上下限防止单 crop 垄断，从几何上保证多区域覆盖。 |
| **D** | — |

---

#### HiDivDrop（ICLR 2026）

| 维度 | 内容 |
|------|------|
| **A1** | DTop-K（可微 token 选择）：重要性 $c_i$ → 归一化排名 $c'_i \in [0,1]$ → 软掩码 $\text{Mask}_i = \sigma(\lambda(c'_i - a))$，$a$ 是**可学习阈值**。前向用 hard threshold，反向用 sigmoid 梯度。⚠️需端到端训练。 |
| **A2** | — |
| **B** | — （名称含 "Div" 但 diversity 体现在三段式架构设计：Late Injection 跳过浅层 + Concave Pyramid 中间层渐进 + Early Exit 深层停止，而非 indicator 层面的显式 diversity 目标函数。） |
| **C** | — |
| **D** | $\text{ILVAS}(l) = \text{sim}(\tilde{A}^{(l)},\, \tilde{A}^{(l+n)})$，衡量第 $l$ 层 attention 排名在 $n$ 层后的稳定性。选 ILVAS 曲线的**局部最大值**作为剪枝层（如 $\{10, 14, 16, 18\}$）。唯一将 "indicator 可靠性" 本身作为信号的方法，决定**在哪里剪**而非剪什么。 |

---

#### VScan（TMLR 2026）

| 维度 | 内容 |
|------|------|
| **A1** | **双尺度 union**：Global Scan 用 ViT **output layer** CLS attention（深层，捕捉高级语义显著性）；Local Scan 用 ViT **layer 6** CLS attention（浅层，保留低级细节）。两组各取 $R_1/2$，union 后互补覆盖不同粒度的视觉信息。 |
| **A2** | LLM **第 16 层** last instruction token → visual tokens attention。关键设计：选中间层（k=16）而非浅层（k=2），实验验证 k=16 >> k=2（FastV 的位置），因为浅层存在位置偏差。兼容 FlashAttention（单独开 auxiliary vanilla attention pass 提取 attention map）。 |
| **B** | Merge：union 选出的 token 之间做 cosine similarity → average pooling token merging，保留被丢弃 token 的信息。是信息保留手段，非显式 diversity 目标。 |
| **C** | Local Scan 的**非重叠窗口划分**：将 token grid 分为不重叠的局部窗口，每窗口内独立选 top-k，从几何上保证各空间区域至少有代表 token，避免全局 CLS attention 导致的区域遗漏。 |
| **D** | — |

---

#### Nuwa（Arxiv 2602.02951）

| 维度 | 内容 |
|------|------|
| **A1** | $S(t_i) = \alpha_{\text{cls},i} \times \|\mathbf{k}_i\|_2$，CLS attention × key vector L2-norm 双乘积：CLS attn 给全局重要性，$\|k_i\|_2$ 给信息容量（高 L2-norm 的 **Pillar Token** 被保护，不参与聚合）。 |
| **A2** | Stage 2（LLM 中间层）：text-visual cosine similarity，在多模态对齐完成后再执行第二次剪枝。选择中间层是因为浅层文本-视觉对齐不充分。 |
| **B** | Merge 聚合：权重 = 语义 cosine similarity × 空间邻近度高斯核，双约束防止「语义相似但空间遥远」的 token 被错误合并。是空间感知的信息保留机制。 |
| **C** | M×M **grid 分区**：将 token grid 划分为 M×M 子区域，每区域内独立选 top-k benchmark token，从几何上保证空间均匀覆盖。专为 OCR / 空间关系推理 / REC 等定位任务设计。 |
| **D** | — |

---

#### FSR（Arxiv 2602.05809）

| 维度 | 内容 |
|------|------|
| **A1** | $s_i$：CLS attention（视觉显著性），作为 Focus score 的组成部分。 |
| **A2** | $r_i$：CLIP text encoder 编码指令后与 visual token 的 cosine similarity（指令相关性）。Focus score 融合：$\phi_i = \alpha \hat{r}_i + \beta \hat{s}_i$，**动态 budget**：$K_F = \min\{k : \sum_{j=1}^k \phi_{\pi(j)} \geq \rho \cdot Z\}$（保留覆盖 90% 总 $\phi$ 质量所需的最少 token 数，图像越简单 token 越少）。 |
| **B** | **B3 CCS**（Conditional Context Sampling）：≈ Farthest Point Sampling 变体，每步选「与 Focus + 已选 Scan token 综合相似度**最低**」的 token，最大化信息增量。有理论覆盖界保证。Refine 阶段：剩余 token 按 similarity 分配到最近 Scan center 加权聚合（只修改 Scan token，保护 Focus 高保真度）。 |
| **C** | — |
| **D** | — |

---

#### IDPruner（Arxiv 2602.13315）

| 维度 | 内容 |
|------|------|
| **A1** | $\widehat{\text{Imp}}(v_i)$：VisionSelector 的 DiffTopK 输出 + min-max 归一化。VisionSelector 是端到端训练的可学习评分模块（⚠️非 training-free），仅以 visual feature 为输入，不使用文本信息，属于**学习型 A1**。 |
| **A2** | — |
| **B** | **B4 MMR**：$v^* = \arg\max_{v_i \notin \mathcal{S}} [\lambda \cdot \widehat{\text{Imp}}(v_i) - (1-\lambda) \cdot \max_{v_j \in \mathcal{S}} \cos(v_i, v_j)]$，**减法**显式平衡重要性与多样性（vs SCOPE 的乘法）。$\lambda=0.5$ 最优。高效 $O(KN)$ 增量更新：维护 $m$ 向量，每步 $m \leftarrow \max(m, \text{sim}(V, v^*))$。无全局近似保证（vs B2/B3 有 $(1-1/e)$）。 |
| **C** | — |
| **D** | — |

---

### 5.3 分类体系的核心观察

**观察 1：纯 saliency → saliency+diversity 的演进轨迹**  
从 VisionZip（纯 A1）→ SCOPE（A1 + B3 Coverage）→ CDPruner（A2 + B2 DPP），主线是在重要性评分基础上叠加多样性约束。FastV 和 PyramidDrop 走的是另一条轨迹——同样使用 LLM text→visual attention（A2），但完全放弃多样性，专注于剪枝策略的效率设计（单次 L2 → 渐进多阶段），说明 "saliency-only" 的设计空间通过工程优化仍有很大价值。

**观察 2：A2（Task Relevance）信号的两个代际**  
A2 信号从 FastV（2024）就已出现（LLM L2 last-token attn），并非 2025 年之后的新发明。但两代方法在设计意图上有本质区别：
- **第一代（隐式 A2）**：FastV、PyramidDrop——使用 LLM text→visual attention 作为**显著性代理**，设计目标是效率而非文本语义对齐，text 信号是副产品
- **第二代（显式 A2）**：CDPruner、SparseVLM、VisionTrim、Nuwa、FSR、VScan——**有意设计**文本引导信号，目标是让剪枝结果对当前 query 敏感，实现 per-query 个性化选择  

信号来源进一步分化为两条技术路线：
- **CLIP text encoder 路线**（CDPruner、VisionTrim、FSR）：轻量，但依赖 CLIP 架构，对 Qwen2.5-VL 等无单独 CLIP encoder 的新型架构需适配
- **LLM 内部 attention 路线**（SparseVLM、VScan L16、Nuwa Stage 2）：更通用，但需要额外 forward pass 或访问中间层激活

**观察 3：维度 C 是被长期忽视的隐性瓶颈**  
DivPrune 和 SCOPE 纯粹在语义空间操作，对 OCR、空间关系推理、REC 等需要精确空间定位的任务存在系统性劣势。Nuwa 的实验明确验证了这一点。

**观察 4：B1（MMDP）在理论上最弱**  
B1 仅优化最近邻 pair，是对全局多样性的粗糙近似。B2（DPP）和 B3（Submodular）均有更严格的全集近似保证，且 B3 的 Submodular Coverage 在直觉上更贴近"让每个 token 都被代表"这一目标。

**观察 5：A1 + C 的组合尚未被充分探索**  
目前多数方法是 A1+B 或 A2+B 的组合，而 A1+C（空间感知的视觉显著性，无需文本）在理论上是一个被低估的设计点——HoloV 是最接近的，但其 Diversity Variance 仍是语义信号而非纯几何信号。
