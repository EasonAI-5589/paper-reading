# 3. Proposed Method

> 来源: DivPrune (Arxiv 2503.02175)

---

## 📄 原文

> 💡 **Section 概览**: 这是全文最核心的部分。先介绍 LMM 基本框架，然后形式化定义 token pruning 问题，最后将其转化为 MMDP 并给出求解算法。

---

![Figure 2](../images/d4a555a2315535d9b90784a997208386cbe2e52f45b97ef7de143897b1ebf6fd.jpg)
*Figure 2: LMM 架构总览 + DivPrune 的应用位置。右侧展示了方法的具体步骤。*

### 3.1 Large Multimodal Models (LMMs)

> 💡 **3.1 要点预览**: LMM 的数学表示——输入怎么变成 token，token 怎么生成输出。

An LMM typically processes a pair of inputs, denoted as $(T, V)$, where $T$ is the text input and $V$ is the visual input such as image or video. The text input is mapped to $N$ textual tokens $\mathbf{E_t} = \{t_1, \ldots, t_N\}$ using a text encoder. Similarly, the visual input is processed by a corresponding vision encoder. Specifically, it takes visual information $V$ as input and outputs image features, that are further converted to $M$ (generally $M \gg N$) vision tokens $\mathbf{E_v} = \{v_1, \dots, v_M\}$ using a projector layer.

The textual tokens and visual tokens are then combined to be fed to an LLM to generate the prediction in an autoregressive manner:

$$P(y_1, ..., y_{\hat{N}} \mid \mathbf{E_t}, \mathbf{E_v}) = \prod_{i=1}^{\hat{N}} P(y_i \mid y_{<i}, \mathbf{E_t}, \mathbf{E_v})$$

> 💡 **3.1 小结**: 标准 LMM 流程。关键信息：$M \gg N$，即视觉 token 远多于文本 token，所以剪视觉 token 最划算。
> ```
> 典型数量:
> - LLaVA 1.5: M = 576 视觉 token
> - LLaVA 1.6: M = 1728~2880 视觉 token
> - LLaVA-NeXT-Video: M = 144 × 8帧 = 1152 视觉 token
> - 文本 token N: 通常几十到几百
> ```

---

### 3.2 Token Pruning

> 💡 **3.2 要点预览**: 把 token pruning 形式化为一个优化问题。

Reducing the number of input tokens in an integrated LLM within LMMs helps to lower memory usage and inference latency. Since visual tokens tend to have more redundancy, they are generally selected for pruning.

The problem of token pruning can be defined as follows: given a set of visual tokens $\mathbf{E_v}$ with $|\mathbf{E_v}| = M$ and the subset size $\tilde{M}$ ($\tilde{M} < M$), the goal is to select a subset, $\tilde{\mathbf{E}}_{\mathbf{v}}$, while preserving key information necessary for accurate predictions. The objective is to identify a mapping function $f$ that minimizes the difference in the model's output before and after pruning:

$$\text{Find:} \quad f: \mathbf{E}_v \to \tilde{\mathbf{E}}_\mathbf{v}$$
$$\text{Objective:} \quad \min_f \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}})$$
$$\text{Subject to:} \quad |\tilde{\mathbf{E}}_v| = \tilde{M}$$

where $\mathcal{P} = P(y_1, \ldots, y_{\hat{N}} \mid \mathbf{E_t}, \mathbf{E_v})$ and $\tilde{\mathcal{P}} = P(y_1, ..., y_{\hat{N}} \mid \mathbf{E_t}, f(\mathbf{E_v}))$.

> 💡 **批注**: 用大白话说：
> ```
> 目标: 从 M 个视觉 token 中选 M̃ 个
> 约束: 选完之后模型输出要和原来尽量一样
> 挑战: 这个优化问题没有直接的闭式解
> ```
> 关键问题是：怎么衡量"尽量一样"？直接最小化输出差异需要跑模型，开销太大。所以需要一个代理目标（proxy objective）。

> 💡 **3.2 小结**: 形式化了问题，但直接求解不可行。下一步就是用 MMDP 作为代理目标。

---

### 3.3 DivPrune: Method Overview

> 💡 **3.3 要点预览**: 全文最核心——如何把 token pruning 转化为 MMDP，以及怎么高效求解。

We proposed a diversity-based token pruning method by reformulating the problem in (2) to select a subset of $\tilde{M}$ elements that maximizes the diversity, thereby reducing redundancy. Specifically, we define token pruning as Max–Min Diversity Problem (MMDP) [34] where the goal is to find the set $\tilde{\mathbf{E}}_{\mathbf{v}}$ among all possible sets with $\tilde{M}$ samples in $\mathbf{E_v}$ that has the maximum minimum distance between its elements:

$$\text{Find } \tilde{\mathbf{E}}_{\mathbf{v}} = \arg\max\left[\min_{\gamma, \omega \in S}\left(d(\gamma, \omega)\right) : \forall S \subset \mathbf{E}_{\mathbf{v}}\right]$$

where $S$ is an arbitrary set in $\mathbf{E_v}$ with $\tilde{M}$ elements and $(\gamma, \omega)$ are arbitrary elements in $S$. The distance is measured using the cosine distance:

$$d(\gamma, \omega) = 1 - \frac{\gamma \cdot \omega}{\|\gamma\| \|\omega\|}$$

> 💡 **批注**: 这是全文的精华！让我用大白话解释 MMDP：
> ```
> 想象你有 576 个球散布在高维空间中
> 你要从中选出 57 个球（剪枝 90%）
> 
> Max-Min Diversity 的目标:
> 在你选出的 57 个球中，找到距离最近的那一对
> 让这个"最近距离"尽可能大
> 
> 直觉: 如果最近的两个球都很远，说明所有球都分散得很开
>        → 没有冗余，覆盖面广
> 
> 类比: 就像在一块田里种 57 棵树
>       要让任意两棵树之间的最小间距最大化
>       → 树会均匀分布在整块田上
> ```

> 💡 **为什么用余弦距离而不是欧氏距离？**
> 因为在高维空间中（4096维），余弦距离衡量的是向量方向的差异，不受向量长度影响。两个语义相似的 token 方向接近（余弦距离小），语义不同的 token 方向不同（余弦距离大）。

---

#### 算法详解

A solution for the MMDP problem in (3) is a subset of $\mathbf{E_v}$ that maximizes diversity by minimizing redundancy between elements. Since the number of tokens is generally limited (e.g., 576 in LLaVA 1.5) and the solvers are not generally designed for GPU acceleration, we obtain exact solution for the problem. Notably, the overhead of the selection process using GPU is negligible compared to the computations within the LLM.

**Algorithm 1: DivPrune**

```
输入: M̃ (子集大小), E_v (视觉 tokens), Ẽ_v (选中子集)
初始化: Ẽ_v = [], R = E_v (候选列表)

第一阶段: 选第一个 token
  对候选列表 R 中每个 token i:
    计算 i 到 R 中所有其他 token 的最小距离 d_min
  选出 d_min 最大的 token k，从 R 移到 Ẽ_v

第二阶段: 迭代选后续 token
  while |Ẽ_v| < M̃:
    对候选列表 R 中每个 token i:
      计算 i 到已选集合 Ẽ_v 中所有 token 的最小距离 d_min
    选出 d_min 最大的 token k，从 R 移到 Ẽ_v

返回 Ẽ_v
```

> 💡 **算法批读**:
> ```
> 第一阶段（选种子）:
> - 在所有 token 中，找那个"离自己最近邻居最远"的 token
> - 直觉: 这个 token 最"孤立"，代表了一个独特的信息区域
>
> 第二阶段（贪心扩展）:
> - 每次从候选集中选一个离已选集最远的 token
> - "最远"指的是: 该 token 到已选集中最近 token 的距离最大
> - 直觉: 每次选最"不像"已选 token 的那个
>
> 复杂度: O(M × M̃) 次距离计算
> 优化: 预先算好距离矩阵（一次矩阵乘法），避免重复计算
> ```

> 💡 **批注**: 这个贪心算法其实就是经典的 **Farthest Point Sampling (FPS)** 的变体！在 3D 点云处理中（如 PointNet++）广泛使用。DivPrune 的贡献在于把这个思路引入到 LMM token pruning 中，并证明了其有效性。

The proposed method can also be applied to the features (i.e., hidden states) in the intermediate layers of the LLM. In this case, our method is not applied to the visual tokens, but to the features corresponding to the visual tokens obtained from a decoder layer to select a subset before feeding them to the subsequent layers.

> 💡 **批注**: DivPrune 不仅可以在 projector 之后（Layer 0）剪枝，也可以在 LLM 的中间层剪枝。后面的消融实验会分析不同层的效果。

---

## 💡 Section 总结

### 方法流程图
```
Visual Tokens (M 个)
       ↓
计算余弦距离矩阵 (M × M)
       ↓
MMDP 贪心求解:
  1. 选最"孤立"的 token 作为种子
  2. 迭代: 每次选离已选集最远的 token
  3. 直到选够 M̃ 个
       ↓
Selected Tokens (M̃ 个) → 送入 LLM
Pruned Tokens (M - M̃ 个) → 丢弃
```

### 关键数字速查
| 项目 | 值 |
|------|-----|
| Token 维度 | 4096 (LLaVA) |
| 距离度量 | 余弦距离 |
| 算法类型 | 贪心 (Farthest Point Sampling 变体) |
| 求解方式 | 精确解（token 数量有限） |
| 额外开销 | 可忽略（一次矩阵乘法） |

### 核心洞察
1. **MMDP 是一个巧妙的代理目标**：把不可直接优化的"输出差异最小化"转化为可高效求解的"多样性最大化"
2. **贪心算法有效**：因为 token 数量有限（几百到几千），贪心就能得到不错的解
3. **和 FPS 的联系**：这本质上就是高维空间中的最远点采样
