[← 返回 README](../README.md)

# 3. Methodology

Our analysis reveals that existing token pruning methods fail on spatial localization tasks by disrupting the global spatial reference frame. This motivates three core design principles for effective visual token compression: (1) preserving spatial uniformity to ensure consistent coverage; (2) aggregating redundant information in a vision-centric, cohesive manner while retaining local salience (Stage1); and (3) applying text-modulated fine-grained filtering to select task-relevant tokens based on textual semantics (Stage2).

> 💡 三个设计原则分别对应三个 Finding：空间均匀性 ← Finding 3, 视觉聚合 ← Finding 1 (pooling 好), text-guided ← Finding 2 (task-specific)

## 3.1 Stage 1: Spatial Cohesion Pruning in the Vision Encoder

This stage reduces the initial $N^2$ visual tokens to a dense, spatial-preserving sequence via three sequential operations.

### 3.1.1 Separation via Grid Partitioning

Partition the input token grid $\mathcal{T} = \{t_1, t_2, \dots, t_{N^2}\}$ into $M \times M$ non-overlapping local regions $\mathcal{R}_{i,j}$. Subsequent selection and aggregation occur at the region level, enabling a complete global coordinate system.

> 💡 简单但关键——网格分区天然保证了空间均匀覆盖，避免所有选出的 token 集中在图像某一区域。这是 RPME 思想的直接实现。

### 3.1.2 Alignment via Salience Identification

Within each region $\mathcal{R}$, select representative benchmark tokens as aggregation centers. Salience score combines CLS attention and information capacity (L2-norm of key vector):

$$S(t_i) = \alpha_{\text{cls},i} \cdot \|\mathbf{k}_i\|_2$$

where $\alpha_{\text{cls},i}$ is the attention weight from [CLS] token. In each local region $\mathcal{R}_k$, select the $k$ tokens with highest salience scores to form the Benchmark Token set $\mathcal{T}_B$.

> 💡 **为什么用 L2-norm?**
> - 纯 CLS attention 在 ViT 深层分布稀疏（可能受 attention sink 影响）
> - L2-norm 高的 key vector 被认为是 "register tokens"（Darcet et al., 2024），信息容量大
> - 两者相乘 = 全局重要性 × 信息丰富度
> - 这和 VisionZip 只用 CLS attention 的做法形成对比

### 3.1.3 Aggregation via Spatial Proximity

Merge features from other tokens into the benchmark set $\mathcal{T}_B$, guided by role assignment and spatial proximity.

#### Role Assignment: Pillars and Collectors

Differentiate benchmark tokens by information capacity:
- **Pillar Tokens** ($\mathcal{T}_P$): L2-norm in top quartile — features remain **unmodified** (like registers)
- **Collector Tokens** ($\mathcal{T}_C$): the rest — aggregate from spatial neighbors

$$\mathcal{T}_P = \{t_i \in \mathcal{T}_B \mid \|\mathbf{k}_i\|_2 \geq \text{Quantile}(\{\|\mathbf{k}_j\|_2\}_{t_j \in \mathcal{T}_B}, 0.75)\}$$

> 💡 **Pillar/Collector 的设计动机**:
> - 参考 Darcet et al. (2024) 关于 ViT registers 的发现：高 L2-norm token 经常被其他 token attend，修改它们会影响全局
> - 因此保护高 L2-norm token 不被 merge 污染
> - 只有信息密度较低的 Collector 才做聚合——避免破坏关键特征

#### Weighted Aggregation

Weight matrix $\mathbf{W} \in \mathbb{R}^{K \times N^2}$ combines:

**Semantic Similarity Matrix** ($\mathbf{A}$) — only positive correlations:
$$A_{ij} = \text{ReLU}(\text{sim}(\mathbf{v}_i, \mathbf{v}_j))$$

**Spatial Proximity Matrix** ($\mathbf{P}$) — penalize long-range aggregation:
$$P_{ij} = 1 - \max\left(1, \frac{d(p_i, p_j)}{d_{\text{thresh}}}\right)$$

Final aggregation weight:
$$W_{ij} = \begin{cases} \delta_{ij} & \text{if } t_i \in \mathcal{T}_P \text{ (Pillar)} \\ A_{ij} \cdot P_{ij} & \text{if } t_i \in \mathcal{T}_C \text{ (Collector)} \end{cases}$$

Row-normalized: $\mathbf{V}'_B = \hat{\mathbf{W}} \mathbf{V}$

> 💡 **聚合策略分析**:
> - **语义 × 空间** 的乘积权重确保只合并"语义相近且物理相邻"的 token
> - ReLU 过滤负相关——避免语义相反的 token 被合并（这比 ToMe 的 bipartite matching 更保守）
> - 空间距离阈值 $d_{\text{thresh}}$ 控制聚合范围，消融实验显示 26% 最优
> - **与 PruMerge 的区别**: PruMerge 用全局 cosine similarity 做 merging，不考虑空间距离；Nüwa 加入空间约束
> - **与 VisionZip 的区别**: VisionZip 只 select 不 merge（或用简单 mean pooling）；Nüwa 做加权聚合

## 3.2 Stage 2: Text-Modulated Pruning in the LLM

After Stage 1, aggregated vision tokens $\mathbf{V}'_B$ are fed into the LLM. At an intermediate layer (after initial multimodal alignment), apply text-guided pruning:

1. **Textual query**: average-pool text token embeddings: $\bar{\mathbf{q}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{q}_k$

2. **Relevance score**: cosine similarity in shared embedding space:
$$R_i = \text{sim}(\text{proj}(\mathbf{v}'_i), \bar{\mathbf{q}})$$

3. **Select** top-$K_{\text{final}}$ visual tokens by $R_i$.

> 💡 **Stage 2 评价**:
> - 本质上就是 FastV 的思路（在 LLM 中间层做一次 pruning），但用 text-vision cosine similarity 而非 attention score
> - 实际创新有限——消融实验（Table 8）也显示 Stage 2 的贡献不如 Stage 1 大
> - 但两阶段设计符合 Finding 2 的分析：先在 encoder 侧保留空间完整性，再在 LLM 侧利用 text 语义精炼
> - **关键洞察**: proj(·) 是 multimodal projection layer（如 LLaVA 的 linear projector），在 shared space 中衡量相关性比在独立空间中更准确
