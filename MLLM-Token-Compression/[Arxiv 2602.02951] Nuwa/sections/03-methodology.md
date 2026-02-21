[← 返回 README](../README.md)

# 3 Methodology

## 📌 预览
Nüwa 的方法论分为两个阶段：Stage 1 在 Vision Encoder 输出端进行 Boids-inspired 空间感知 pruning（Separation → Alignment → Aggregation）；Stage 2 在 LLM 中间层进行 text-guided pruning。

---

Our analysis reveals that existing token pruning methods fail on spatial localization tasks by disrupting the global spatial reference frame. This motivates three core design principles for effective visual token compression: (1) preserving spatial uniformity to ensure consistent coverage; (2) aggregating redundant information in a vision-centric, cohesive manner while retaining local salience (Stage1); and (3) applying text-modulated fine-grained filtering to select task-relevant tokens based on textual semantics (Stage2). We apply these principles in Nüwa, a two-stage pruning framework.

> 💡 **批注**: 三条设计原则直接对应前面三个 Finding：
> 1. Spatial uniformity → Finding 3（保持空间参考系）
> 2. Vision-centric aggregation → Finding 1（pooling 的优势来自空间保持）
> 3. Text-modulated filtering → Finding 2（中间层的 task-specific 处理）

---

## 3.1 Stage 1: Spatial Cohesion Pruning in the Vision Encoder

This stage reduces the initial $N^2$ visual tokens in the vision encoder to a dense, spatial-preserving sequence via three sequential operations.

> 💡 **批注**: Stage 1 的目标是把 $N^2$（如 576 = 24×24）个 token 压缩到更少数量，同时保持空间完整性。

---

### 3.1.1 Separation via Grid Partitioning

To maintain spatial integrity, we partition the input token grid $\mathcal{T} = \{t_1, t_2, \dots, t_{N^2}\}$ into $M \times M$ non-overlapping local regions $\mathcal{R}_{i,j}$. Subsequent selection and aggregation occur at the region level, enabling a complete global coordinate system.

> 💡 **批注**: Grid Partitioning 是最简单但最关键的操作——它保证了 pruning 后 token 在空间上的均匀分布，直接实现了 RPME 的目标。每个 region 都会保留至少一个代表 token，确保全局覆盖。

---

### 3.1.2 Alignment via Salience Identification

Within each region $\mathcal{R}$, we select representative benchmark tokens as aggregation centers. These tokens should exhibit high global salience; we initially use attention scores from the [CLS] token. However, analysis indicates sparse distributions in deeper vision encoder layers. To mitigate this, we incorporate information capacity, defined as the L2-norm of the token's key vector ($\|\mathbf{k}_i\|_2$), as a secondary criterion. The resulting salience score $S(t_i)$ for token $t_i$ is the product of its global attention score and information capacity:

$$S(t_i) = \alpha_{\text{cls},i} \cdot |\mathbf{k}_i|_2 \tag{3}$$

where $\alpha_{\text{cls},i}$ is the attention weight from the [CLS] token. In each local region $\mathcal{R}_k$, we select the $k$ tokens with the highest salience scores to form the Benchmark Token set $\mathcal{T}_B$.

> 💡 **批注**: Salience score = CLS attention × L2-norm of key vector。这个设计有两个考量：
> - **CLS attention**：反映全局语义重要性
> - **Key L2-norm**：反映信息容量（information capacity）
> 
> 为什么要加 L2-norm？因为 ViT 深层的 CLS attention 分布变得稀疏，单独用不够稳定。L2-norm 高的 token 往往是 "register tokens"（Darcet et al., 2024），在解码时被频繁关注。

---

### 3.1.3 Aggregation Via Spatial Proximity

This operation merges features from other tokens into the benchmark set $\mathcal{T}_B$, guided by role assignment and spatial proximity, yielding a semantically rich and spatially complete token sequence.

---

**Role Assignment: Pillars and Collectors.**

We differentiate benchmark tokens in $\mathcal{T}_B$ by information capacity. Recent works (Darcet et al., 2024; Lappe and Giese, 2025) identify high-norm tokens in ViTs as registers — frequently attended during decoding and often task-agnostic. Modifications to these can shift feature distributions and affect predictions. Thus, we classify tokens with $\|\mathbf{k}_i\|_2$ in the top quartile as Pillar Tokens ($\mathcal{T}_P$), whose features remain unmodified. The rest are Collector Tokens ($\mathcal{T}_C$), which aggregate from spatial neighbors.

$$\mathcal{T}_P = \{t_i \in \mathcal{T}_B \mid |\mathbf{k}_i|_2 \geq \text{Quantile}(\{|\mathbf{k}_j|_2\}_{t_j \in \mathcal{T}_B}, 0.75)\}; \quad \mathcal{T}_C = \mathcal{T}_B \setminus \mathcal{T}_P \tag{4}$$

> 💡 **批注**: **Pillar vs Collector 的区分是一个精妙的设计**：
> - **Pillar Tokens**（top 25% L2-norm）：类似于 ViT 中的 register tokens，是全局特征的锚点，**不做任何修改**
> - **Collector Tokens**（其余 75%）：从空间邻居聚合信息
> 
> 为什么不改 Pillar？因为高 L2-norm 的 token 是 task-agnostic 的全局参考点，修改它们会导致特征分布偏移。这与 Darcet (2024) 的 register token 发现一致。

---

**Weighted Aggregation.**

High semantic similarity does not imply aggregability; relying solely on it for global features is inadequate, as it risks disrupting object-centric representations by merging spatially distant tokens. Thus, we balance it with spatial proximity to form a weight matrix $\mathbf{W} \in \mathbb{R}^{K \times N^2}$, where $K = |\mathcal{T}_B|$, combining semantic and proximity matrices.

**Semantic Similarity Matrix ($\mathbf{A}$):** We consider only positively correlated semantic information. Element $A_{ij}$ is defined as:

$$A_{ij} = \text{ReLU}\left(\text{sim}(\mathbf{v}_i, \mathbf{v}_j)\right) = \text{ReLU}\left(\frac{\mathbf{v}_i \cdot \mathbf{v}_j}{|\mathbf{v}_i||\mathbf{v}_j|}\right) \tag{5}$$

**Spatial Proximity Matrix ($\mathbf{P}$):** To penalize long range aggregation, we define a proximity matrix allowing each benchmark token to aggregate features within an extended local neighborhood, enabling limited cross-region interaction. Element $P_{ij}$ is computed as:

$$P_{ij} = 1 - \max\left(1, \frac{d(p_i, p_j)}{d_{\text{thresh}}}\right) \tag{6}$$

where $d(p_i, p_j)$ is the Euclidean distance between $p_i$ and $p_j$, and $d_{\text{thresh}}$ is a predefined threshold.

Based on role assignment, the final aggregation weight $W_{ij}$ is defined as:

$$W_{ij} = \begin{cases} \delta_{ij} & \text{if } t_i \in \mathcal{T}_P \text{ (Pillar Token)} \\ A_{ij} \cdot P_{ij} & \text{if } t_i \in \mathcal{T}_C \text{ (Collector Token)} \end{cases} \tag{7}$$

where $\delta_{ij}$ is the Kronecker delta, ensuring Pillar Tokens only aggregate from themselves.

The weight $\hat{\mathbf{W}}$ is row-normalized from $\mathbf{W}$, the original feature matrix is $\mathbf{V} \in \mathbb{R}^{N^2 \times D}$. The updated feature matrix for benchmark tokens, $\mathbf{V}'_B \in \mathbb{R}^{K \times D}$, is computed as $\mathbf{V}'_B = \hat{\mathbf{W}} \mathbf{V}$.

> 💡 **批注**: 聚合权重 = **语义相似度 × 空间邻近度**，这是核心创新：
> - 纯语义相似度聚合（如 PruMerge）会把空间上远但语义相似的 token 合并，破坏物体表示
> - 加入空间邻近度约束后，只合并局部邻域内的相似 token
> - **ReLU** 过滤负相关（反义）的语义，只保留正向关联
> - **距离阈值** $d_{\text{thresh}}$ 控制聚合范围（ablation 显示 26% 最大距离最优）
> - Pillar Token 的 Kronecker delta 确保它们不被修改

---

## 3.2 Stage 2: Text-Modulated Pruning in the LLM

Following Stage 1, the aggregated vision tokens $\mathbf{V}'_B$ are fed into the LLM for multimodal feature interaction. We apply a second round of task-oriented pruning at an intermediate layer, after initial multimodal alignment (Shukor and Cord, 2024), where textual and visual features converge in a shared space. To guide this pruning, we first derive a holistic textual query vector $\bar{\mathbf{q}}$ by average-pooling the embeddings $\{\mathbf{q}_1, \dots, \mathbf{q}_K\}$ of text tokens:

$$\bar{\mathbf{q}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{q}_k \tag{8}$$

We calculate a relevance score $R_i$ for each visual token $t'_i$ (with updated feature vector $\mathbf{v}'_i$, the $i$-th token of $\mathbf{V}'_B$) by measuring its cosine similarity to the query vector in the shared embedding space:

$$R_i = \text{sim}(\text{proj}(\mathbf{v}'_i), \bar{\mathbf{q}}) = \frac{\text{proj}(\mathbf{v}'_i) \cdot \bar{\mathbf{q}}}{|\text{proj}(\mathbf{v}'_i)| \cdot |\bar{\mathbf{q}}|} \tag{9}$$

where $\text{proj}(\cdot)$ denotes the multimodal projection layer mapping visual features into the common text-vision embedding space. Finally, we retain only the top-$K_{\text{final}}$ visual tokens with the highest relevance scores $R_i$, passed to subsequent LLM layers for final reasoning and response generation.

> 💡 **批注**: Stage 2 相对简单：在 LLM 中间层用 **text-visual cosine similarity** 做 top-K 选择。
> - 时机选择在 multimodal alignment 之后（中间层），此时文本和视觉特征已经在共享空间中
> - 与 FastV 的 attention-based pruning 不同，这里用 **cosine similarity** 到文本 query 的平均池化
> - 实现上类似 FastV，但 pruning 依据从 attention score 换成了 text-visual relevance
> 
> 注意 Stage 2 的 ablation（Table 8）显示 random pruning 替代 text-guided pruning 后收益有限，说明 Stage 1 的贡献远大于 Stage 2。
