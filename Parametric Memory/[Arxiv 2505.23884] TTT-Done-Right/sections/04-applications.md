[← 返回 README](../README.md)

# 4. LaCT for N-Dimensional Data

## 📌 预览

本节介绍 LaCT 如何适配三种不同模态的数据：4.1 节将 NVS（novel view synthesis）中的图像集合（无序集合）用整个序列作为一个超大 chunk；4.2 节针对文本序列（1D 有序）使用 shifted block-wise causal mask 维持自回归性；4.3 节针对视频帧序列（图像序列）用 3 帧为 chunk 并设计 interleaved noisy/clean 的训练格式。三种任务展示了 LaCT 通过调整 chunk size 和 update/apply 顺序来适配任意数据结构的通用性。

---

In this section, we introduce the three tasks we address using LaCT—novel view synthesis, language modeling, and autoregressive video generation. These tasks have different inherent data structures and we address them with corresponding design choices. The full model architecture details for these data types are provided in Appendix B.

> 💡 **批注：N 维数据适配的核心思路**
>
> LaCT 的"N 维数据"适配不是通过设计专用架构实现的，而是通过两个统一的旋钮：
> 1. **Chunk size**：与数据的内在结构对齐（一张图 = 一个 chunk，N 帧 = 一个 chunk，2K tokens = 一个 chunk）
> 2. **Update/Apply 顺序**：决定 TTT 层等价于哪种注意力掩码，从而决定信息流向
>
> 这种设计哲学使得同一个 LaCT 框架无需改动核心代码就能适配图像集合、文本序列和视频序列。

---

# 4.1 Novel View Synthesis - Image Set

Novel view synthesis (NVS)[27, 28] aims to render images of a static scene from previously unseen viewpoints. Formally, given a set of $N$ input posed images $\{ ( I _ { i } , P _ { i } ) \} _ { i = 1 } ^ { N }$ of a static scene, where $I _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ is an RGB image and $P _ { i }$ is its corresponding camera pose, the model needs to synthesize new images from novel camera poses that typically do not overlap with the input views.

We find that NVS is an effective test bench for evaluating a model's online memory and compression capabilities. Firstly, NVS is challenging as it requires spatial compression, dense retrieval, and basic physical reasoning. Secondly, NVS can be formulated as a non-generative task, significantly reducing training computation and the need for extensive model parameters to store world knowledge, thereby enabling rapid experimentation. Thirdly, the substantial redundant information in dense input views incentivizes the model to learn effective compressions. Given these observations, we use NVS for our initial research iterations. We find that some of the insights gained are transferrable to other tasks.

> 💡 **批注：为什么 NVS 是测试 TTT 记忆能力的理想 benchmark？**
>
> 作者对 NVS 的选择是经过深思熟虑的，有三个关键原因：
>
> 1. **任务挑战性**：NVS 需要跨视角的 3D 空间推理，必须压缩大量重叠冗余的多视角信息，是测试"有损压缩质量"的严苛场景。
>
> 2. **计算可行性**：NVS 是判别性（non-generative）任务，不需要模型存储海量世界知识（对比语言模型需要记住所有语言知识），因此可以用小模型（0.3B）快速实验不同架构设计。
>
> 3. **信息冗余性**：128 张视角高度重叠的图像中，相邻视角有大量重复像素，迫使模型学会有效的信息压缩而非简单记忆。
>
> 这三点使 NVS 成为 LaCT 研究的"快速迭代平台"，论文中大量消融实验（Figure 7、8）都是在 NVS 上做的。

---

Our NVS model follows the basic LaCT diagram in Sec. 3. Both the posed input images and poses of the target novel views are tokenized by patchify and linear layers, following LVSM [29]. The window attention exactly covers the tokens from a single image. The LaCT layer adapts a single-round of strided block-wise causal mask (Fig. 3d), which updates the fast weight using all input image tokens, and applies to both the input and target tokens. The update step resembles a prefill stage, while the apply operation resembles parallel decoding. During rendering of novel views, each test-time training layer functions as a static weight layer, making the entire model a static vision transformer [30]. We illustrate this design in Figure 9.

> 💡 **批注：NVS 的 Chunk 设计——整个序列作为一个超大 Chunk**
>
> NVS 的关键设计决策：**chunk = 全部输入图像 tokens（最多 1M tokens）**
>
> 这对应 Figure 3d 的 strided block-wise causal mask：
> - **Update 阶段**（类似 prefill）：用所有输入图像的 tokens 更新一次 fast weights，将场景的 3D 信息压缩进 $W$
> - **Apply 阶段**（类似 parallel decoding）：用目标视角的 query tokens 查询 fast weights，并行生成所有目标视角
>
> **推理时的优雅性质**：一旦 fast weights 更新完毕（prefill 完成），渲染新视角时 TTT 层退化为静态权重层（$W$ 不再更新），整个模型变成普通的 Vision Transformer，只需前向传播即可，无额外开销。
>
> **Window Attention 的角色**：覆盖单张图片内的所有 tokens，处理图片内部的局部空间结构（patch 之间的邻近关系），而 TTT 层负责跨图片的信息聚合（不同视角的一致性）。

---

# 4.2 Language Modeling - Text Sequence

Autoregressive language models predict the probability distribution of the next token given preceding tokens, $p _ { \theta } ( x _ { n } | x _ { 1 } , \dots , x _ { n - 1 } )$ . Text sequences lack inherent chunk structures, so for LaCT, we define chunk size as a hyperparameter (e.g., 2048 or 4096 tokens). We utilize the shifted block-wise causal mask as in Fig. 3(c) for the TTT apply-update sequence to avoid seeing future tokens in a chunk. Since LaCT lacks per-token causality within each chunk, we employ sliding window attention—with window size equal to the chunk size—to efficiently model per-token causal dependencies. The sliding window is integrated into the same TTT layer with shared QKV similar to GAU [25]. We illustrate the detailed architecture in Fig. 10 and pseudocode 2.

> 💡 **批注：语言模型中的 LaCT——如何处理缺乏内在 Chunk 结构的文本？**
>
> 文本序列是 LaCT 最具挑战性的应用，因为文本没有天然的 chunk 边界（不像图像有边界、视频有帧边界）。作者的解决方案：
>
> **1. Shifted Block-wise Causal Mask（Fig. 3c）**：
> - 目的：维持严格的自回归性（token $x_n$ 只能看到 $x_1, ..., x_{n-1}$）
> - 实现：先 update（用当前 chunk 的信息更新 $W$），再 apply（用更新后的 $W$ 对当前 chunk 的 query 作答）
> - 注意：这里"先 update 后 apply"实际上在 apply 时当前 chunk 的信息已经被编码进 $W$ 了，所以需要 "shift" 来避免泄露未来信息——即当前 chunk 的 fast weight 更新只用于回答下一个 chunk 的 query
>
> **2. Sliding Window Attention（SWA）**：
> - 目的：补充 chunk 内部的 per-token 因果依赖（TTT 层无法区分 chunk 内 token 顺序）
> - window size = chunk size，保证每个 token 能看到其前 chunk_size 个 token 的精确依赖
> - 与 TTT 层共享 QKV（类似 GAU [25]），节省一次 QKV 投影的计算
>
> **这种设计的权衡**：语言模型中 chunk 内部信息的有序性被 TTT 层忽略，但 SWA 层补偿了这一信息损失。实验表明即使如此，LaCT（带 Muon + 大状态）仍能超越 DeltaNet/GLA 等 per-token 方法。

---

# 4.3 Autoregressive Video Diffusion - Image Sequences

Chunkwise autoregressive video diffusion iteratively denoises a number of subsequent video frames, conditioned on the previously generated clean frames, where each chunk can contain thousands of visual tokens. We use teacher-forcing training by interleaving noisy and clean frame chunks. Specifically, a video of $\mathbf { N }$ frame chunks is structured as:

$$
S = [ X _ { 1 } ^ { \mathrm { n o i s e } } , X _ { 1 } , X _ { 2 } ^ { \mathrm { n o i s e } } , X _ { 2 } , \dots , X _ { N } ^ { \mathrm { n o i s e } } ]
$$

where each noisy chunk $X _ { i } ^ { \mathrm { n o i s e } }$ is produced by adding unit Gaussian noise $\epsilon$ to the $i$ -th clean video chunk as $X _ { i } ^ { \mathrm { n o i s e } } = X _ { i } ( 1 - t _ { i } ) + \epsilon t _ { i }$ and $t _ { i } \in [ 0 , 1 ]$ denotes the strength of chunk-independent noise.

> 💡 **批注：Interleaved Noisy-Clean 格式的设计动机**
>
> 视频扩散的 teacher forcing 训练格式是 $[X_1^{noise}, X_1, X_2^{noise}, X_2, ...]$，这个设计解决了两个问题：
>
> **问题 1：低 token 利用率**
> 如果只用 $[X_1, X_2^{noise}]$（历史 clean + 当前 noisy），每个 token 要么是条件输入（无监督信号），要么是去噪目标，利用率只有 50%。
> 解法：将 clean chunk 也加入序列，用去噪目标监督所有 noisy chunks，利用率 > 50%。
>
> **问题 2：chunk 独立的噪声强度**
> 不同 chunk 使用独立的噪声强度 $t_i$，避免了模型对特定噪声时间步的过拟合，类似于扩散模型中的随机 timestep 采样。
>
> **与 fast weight 的协同**：只在 clean chunk 上更新 fast weights（见下文），确保 fast weight 只存储"确定性的干净帧信息"，避免将噪声信息写入记忆。

---

To handle such a data structure, we employ the strided block-wise causal mask in Fig. 3d for LaCT. Specifically, it applies fast weights to each chunk sequentially while only updating fast weights on clean chunks. This simple strategy ensures that each denoising operation only accesses previously cleaned frames. The windowed attention uses a non-overlapping window with 2 consecutive chunks (i.e., $[ X _ { i } , X _ { i + 1 } ^ { \mathrm { n o i s e } } ] )$ to build temporal and spatial locality. Within each window, the attention from $X _ { i }$ to $X _ { i + 1 } ^ { \mathrm { n o i s e } }$ is excluded. We incorporate the first noisy chunk by shifting all attention and TTT masking patterns similar to Fig. 3c. The details of this hybrid architecture and more efficient trainings are in the Appendix B.3.

> 💡 **批注：视频扩散的 LaCT 设计——Strided Mask + Clean-Only Update**
>
> 视频扩散的关键设计选择：
>
> **1. 只在 Clean Chunks 上更新 Fast Weights**：
> - Fast weight $W$ 只从 $X_1, X_2, ...$ 学习（干净帧），不从 $X_i^{noise}$ 学习
> - 这实现了"记忆 = 历史干净帧的压缩"，为当前去噪提供正确的上下文
> - 对应 strided block-wise causal mask（Fig. 3d）：只在子集 chunks 上 update
>
> **2. Window Attention 覆盖 2 个连续 Chunks**：
> - 窗口 $[X_i, X_{i+1}^{noise}]$ 捕获当前 clean 帧和下一个 noisy 帧之间的时空局部性
> - 但禁止 $X_i \to X_{i+1}^{noise}$ 的注意力（防止去噪过程看到同一时刻的 clean 版本，避免 shortcut）
>
> **3. 第一个 Chunk 的特殊处理**：
> - 第一个 noisy chunk $X_1^{noise}$ 没有前置的 clean chunk 可以依赖
> - 用类似 Fig. 3c 的 shift 处理这个边界情况，确保第一个 chunk 的去噪也有合理的条件信息
>
> **整体效果**：Fast weight 充当"视频历史的压缩记忆"，Window Attention 处理相邻帧的时空连续性，二者协同实现长视频的一致性生成。

---

## 🔖 Section 总结

### 关键数字速查
| 任务 | Chunk Size | 数据结构特点 | Update/Apply 顺序 |
|-----|-----------|------------|-----------------|
| Novel View Synthesis | 全序列（最多 1M tokens）| 无序图像集合 | Strided: 先全 Update 后全 Apply |
| Language Modeling | 2K / 4K tokens | 1D 有序序列 | Shifted Causal: 交替 Update-Apply（移位）|
| Video Diffusion | 3 帧（~数千 visual tokens）| 有序图像序列 | Strided: 只在 Clean 帧 Update |

### 核心洞察
1. **NVS 是天然的 TTT 测试场**：无序多视角图像集合 + 大量视角冗余 = 完美测试 fast weight 压缩能力，且非生成性任务使计算可行。
2. **语言模型的 Chunk 边界不重要**：即使文本没有天然 chunk 结构，通过 shifted causal mask + SWA 补偿，LaCT 仍能获得强性能，说明大 chunk 策略的优势（高效率 + 大状态）超过了 chunk 边界对模型的影响。
3. **视频扩散的关键设计**：只对 clean 帧更新 fast weights，避免将噪声信息写入记忆，这是一个简洁但关键的工程决策，仅需修改几行代码即可实现。
4. **统一框架的普适性**：三种任务使用相同的 LaCT 核心代码，只通过调整 chunk size 和 update/apply mask 来适配不同数据结构，展示了框架的高度可复用性。
