# SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Bypass

Chen Qian * 1 Xinran $\mathbf { Y } \mathbf { u } ^ { \mathrm { ~ \ast ~ 1 ~ } }$ Danyang Li 1 Guoxuan Chi 1 Zheng Yang 1 Qiang Ma 1 Xin Miao 1

# Abstract

Visual token pruning is a promising approach for reducing the computational cost of vision–language models (VLMs), and existing methods often rely on early pruning decisions to improve efficiency. While effective on coarsegrained reasoning tasks, they suffer from significant performance degradation on tasks requiring fine-grained visual details. Through layer-wise analysis, we reveal substantial discrepancies in visual token importance across layers, showing that tokens deemed unimportant at shallow layers can later become highly relevant for text-conditioned reasoning. To avoid irreversible critical information loss caused by premature pruning, we introduce a new pruning paradigm, termed bypass, which preserves unselected visual tokens and forwards them to subsequent pruning stages for re-evaluation. Building on this paradigm, we propose SwiftVLM, a simple and training-free method that performs pruning at model-specific layers with strong visual token selection capability, while enabling independent pruning decisions across layers. Experiments across multiple VLMs and benchmarks demonstrate that SwiftVLM consistently outperforms existing pruning strategies, achieving superior accuracy–efficiency trade-offs and more faithful visual token selection behavior.

# 1. Introduction

Vision–Language Models (VLMs) (Team et al., 2024; Chen et al., 2024b; Alayrac et al., 2022) have rapidly advanced in recent years and emerged as a central paradigm in multimodal learning. These models integrate a visual encoder with a large language model (LLM) (Grattafiori et al., 2024; Achiam et al., 2023) through a cross-modal fusion module, enabling strong performance across a wide range of vision–language tasks (Gao et al., 2025; Lin et al., 2025;

1Tsinghua University, Beijing, China. Correspondence to: Chen Qian <chen.cronus.qian@gmail.com>.

![](images/a8d3a0e913ab8120540dd22ac32506ac7134131ddd136ddfe30584cd92156193.jpg)  
(c) Bypass   
Figure 1. Comparison of visual token pruning strategies in VLMs. (a)–(b) Existing approaches suffer from irreversible loss of critical visual information once tokens are merged or dropped in shallow layers. (c) We propose Bypass, a pruning strategy that restores previously merged tokens via token alignment. Bypass provides critical visual tokens with an opportunity to be reconsidered at deeper layers with stronger token selection capability.

Yang et al., 2025a; Wang et al., 2025a). In practice, visual inputs are processed by generating a large number of visual tokens. However, only a small subset of these tokens is critical for text-conditioned reasoning, with the remainder largely increasing latency and computational overhead.

To reduce the number of visual tokens, prior studies adopt token merging strategies, such as ToMe (Bolya et al., 2022), Qwen-VL (Bai et al., 2025), and VisionZip (Yang et al., 2025b). These methods aggregate visual features based on feature similarity or spatial proximity. While these approaches improve inference efficiency, such compression degrades fine-grained visual details, especially for precise localization tasks.

Another line of work leverages text-to-vision (T–V) attention in VLMs to rank visual tokens and dynamically drop low-ranked ones, as illustrated in Fig.1(b). FastV (Chen et al., 2024a) observes that T–V attention becomes highly concentrated on a small subset of visual tokens from the third layer onward, and thus aggressively drops low-ranked

![](images/e1258e0fcf7da8781b141fcc133245e5816047a11b7d71dce007ca4f111a8e8b.jpg)  
Figure 2. Layer-wise variation in visual token ranking. For a representative TextVQA example, we report the overlap ratio between the bottom-ranked $50 \%$ of visual tokens selected at layers 1–9 and the top-ranked $10 \%$ selected at layers 10–20 of LLaVA.

ones in a shallow layer. PDrop (Xing et al., 2024) further shows that aggressive pruning in early layers leads to significant performance degradation, whereas the impact becomes less severe in deeper layers, motivating a progressive dropping strategy. This principle is subsequently adopted by works such as SparseVLM (Zhang et al., 2024) and FEATHER (Endo et al., 2025). However, we find that the importance ranking of visual tokens varies across layers.

As illustrated in Fig.2, we report the overlap ratio on a TextVQA (Singh et al., 2019) sample between the bottom $50 \%$ visual tokens selected by early layers (layers 1–9) and the top $10 \%$ visual tokens selected by later layers (layers 10–20) of LLaVA-1.5-7B (Liu et al., 2024a). We observe that visual tokens deemed unimportant and dropped in early layers can become highly important in deeper layers.

While existing methods perform early-layer pruning to improve efficiency, prematurely dropping task-relevant visual tokens can hinder subsequent reasoning. As shown in Fig.3, methods such as FastV and PDrop force deeper layers to reason over incomplete visual evidence, often resulting in incorrect answers.

Based on these observations, we propose a third pruning paradigm, termed bypass. As illustrated in Fig.1(c), at the first pruning layer, bottom-ranked visual tokens are not immediately discarded. Instead, they are fully preserved and forwarded directly to the next pruning layer for re-ranking of their importance. Meanwhile, these bottom visual tokens are merged according to feature similarity. The merged visual tokens then participate in subsequent inference.

At the following pruning layer, we derive a hidden-state offset from the merged visual tokens and use it to adjust the bypassed bottom-ranked tokens, aligning them with text tokens in the current representation space. These corrected tokens are then reintroduced for joint re-evaluation.

![](images/802a623af5d31f9927b335a5b58277a98d97ff291c613fdcf55f7be2b05192a7.jpg)  
Figure 3. Comparison of results from different pruning methods. FastV applies aggressive early-layer pruning, whereas PDrop adopts progressive pruning. Both drop the visual token containing “NASRI”, leading to incorrect answers. SwiftVLM preserves the query-relevant token at the final stage and answers correctly.

This design preserves the complete visual information while allowing each pruning layer to independently assess token importance, thereby avoiding irreversible critical information loss caused by premature pruning in early layers.

Furthermore, to determine the pruning layers used for token selection, we conduct a comprehensive layer-wise analysis across two task categories and six benchmark datasets. We first run the vanilla model and record, at each layer, the indices of the top $20 \%$ visual tokens selected based on T–V attention. Using the same set of evaluation samples, we then re-run the model while retaining all visual tokens in the first two layers and keeping only the layer-specific top $20 \%$ visual tokens from the third layer onward. The layer-wise results are reported in Fig.4.

The results indicate that the ability to identify important visual tokens varies across layers and is not monotonically increasing with depth. Moreover, intermediate layers generally exhibiting stronger selection capability. Accordingly, we formulate the pruning-layer selection problem as a dynamic programming task, enforcing a monotonic increase in selection capability across the chosen pruning layers.

Based on these two observations, we propose SwiftVLM, a training-free method that performs pruning at layers with strong selection capability while ensuring independent pruning decisions at each stage.

We first identify model-specific optimal pruning layers (e.g., $i$ and $k$ in Fig.1(c)) and fix them for evaluation at test time. After visual token pruning at layer $i$ , the unselected visual tokens are preserved and re-evaluated at layer $k$ with high selection capability.

![](images/465c1bc284a732b9836f68dcf5d5930c3417b102749e2cc99a80c800a56214c6.jpg)  
Figure 4. Non-monotonic layer-wise capability for visual token selection. Across tasks and datasets, we record the layer-wise top $20 \%$ visual tokens of the vanilla model and re-evaluate it by retaining all tokens in layers 1–2 and only the layer-specific top $20 \%$ from layer 3 onward. Performance is reported relative to the vanilla baseline.

The key contributions are summarized as follows:

• We reveal pronounced layer-wise disparities in visual token importance and propose bypass, a novel pruning strategy that forwards unselected visual tokens to subsequent pruning layers, enabling independent selection decisions.   
• We reveal that the discriminative capability of layers for identifying critical visual tokens varies significantly across depth, exhibiting non-monotonic behavior.   
• We present SwiftVLM, a simple yet effective trainingfree method that identifies high-discriminability pruning layers via dynamic programming and employs bypass to preserve fine-grained visual details while accelerating inference.   
• Extensive experiments across two VLMs on nine benchmarks show SwiftVLM substantially outperforms existing training-free methods.

# 2. Related Work

To reduce the number of visual tokens and improve inference efficiency, existing studies (Zhong et al., 2025; Wang et al., 2025b; Li et al., 2024b) can be broadly classified into two categories.

Text-agnostic. Qwen2.5-VL (Bai et al., 2025) merges each group of four neighboring visual tokens into a single token. ToMe (Bolya et al., 2022) performs similaritybased token merging between the attention and MLP blocks. VisionZip (Yang et al., 2025b) retains tokens with high [CLS]-attention scores and merges the remaining ones based on feature similarity, following a strategy similar to Vis-Pruner (Zhang et al., 2025) and Prumerge (Shang et al., 2025). VoCo-LLAMA (Ye et al., 2025b) compresses visual information into a single learnable VoCo token, which is then used for subsequent cross-modal interaction.

Despite their efficiency, these methods rely solely on visual

cues for token reduction, which limits their ability to preserve query-relevant visual details, particularly when the queried regions are not visually salient.

Text-aware. Q-Former (Li et al., 2023) reduces visual token redundancy by training cross-modal modules that compress hundreds of visual tokens into a small set of learnable tokens. ATP-LLaVA (Ye et al., 2025a) instead introduces trainable modules within the VLM and prunes visual tokens based on importance scores derived from text–vision and vision–vision attention. Although these approaches leverage the text query to guide visual token compression or selection, they require additional trainable components, incurring extra optimization overhead.

Several training-free methods exploit the native cross-modal attention of VLMs. FastV (Chen et al., 2024a) uses T-V attention to assess visual token importance and performs aggressive pruning at a shallow layer. PDrop (Xing et al., 2024) progressively reduces visual tokens across layers, based on the observation that pruning becomes less harmful at deeper layers. FEATHER (Endo et al., 2025) further refines this strategy by mitigating the influence of Rotary Position Embedding (RoPE) (Su et al., 2024) on T-V attention, while SparseVLM (Zhang et al., 2024) performs adaptive layer-wise pruning by estimating redundancy from the rank of the T-V attention matrix. Despite being training-free, these methods assume that tokens pruned early remain unimportant in deeper layers, which often fails in fine-grained visual reasoning, leading to performance degradation.

# 3. Method

# 3.1. Preliminary: Attention in VLMs

Let $L$ denote the total number of tokens participating in computation. Let $h \in \mathbb { R } ^ { L \times d }$ denote the hidden states of all tokens. The query and key matrices are obtained via linear projections,

$$
\mathbf {Q} = \mathbf {h} \mathbf {W} _ {Q}, \quad \mathbf {K} = \mathbf {h} \mathbf {W} _ {K}. \tag {1}
$$

A single-head attention matrix $A \in \mathbb { R } ^ { L \times L }$ in a VLM is then defined as

$$
A = \operatorname {S o f t m a x} \left(\frac {\mathbf {Q K} ^ {\top}}{\sqrt {d}}\right). \tag {2}
$$

VLMs adopt causal attention, under which each token is restricted to attending only to preceding tokens. As a result, the last text token attends to all input tokens. In practice, we extract its attention scores as the cross-modal component to evaluate the importance of visual tokens. Note that positional information is preserved during the pruning process.

# 3.2. Pruning Layer Selection

In this section, we focus on how to accurately select pruning layers with high discriminative capability. Note that we exclude the first two layers from our analysis, as these layers exhibit distinct characteristics compared to other layers (Lad et al., 2024; Kang et al., 2025).

For a model with $L$ layers, we first record the top $V \%$ visual tokens selected by T–V attention at each layer using the vanilla model. Keeping the text and image inputs unchanged, we then re-evaluate the model by retaining all tokens in the first two layers and only the layer-specific top $V \%$ visual tokens from the third layer onward, producing a layer-wise performance profile. This performance sequence reflects the ability of each layer to identify task-relevant visual tokens. We formulate this as:

$$
\left\{x _ {i} \right\} _ {i = 1} ^ {L}, \quad x _ {i} \in \mathbb {R}. \tag {3}
$$

Intuitively, the progressively selected pruning layers should exhibit monotonically increasing performance in this sequence. Let the maximum performance before layer $i$ be denoted as:

$$
M _ {i} = \max  _ {j <   i} x _ {j}. \tag {4}
$$

Based on the condition $x _ { i } > M _ { i }$ , we can identify multiple candidate sets $S$ of pruning layers.

$$
S = \left\{i _ {1}, i _ {2}, \dots , i _ {k} \right\}, \quad 3 \leq i _ {1}, \dots \leq i _ {k} \leq L. \tag {5}
$$

Ideally, model performance can be expressed as a function of the selected pruning layers.

$$
y (t) = \left\{ \begin{array}{l l} x _ {2}, & 1 \leq t <   i _ {1}, \\ x _ {i _ {2}}, & i _ {1} \leq t <   i _ {2}, \\ \vdots & \\ x _ {i _ {K}}, & i _ {K} \leq t \leq L. \end{array} \right. \tag {6}
$$

As the impact of visual token selection propagates through subsequent layers, we reformulate layer selection as an optimization problem that maximizes the overall layer contribution under a fixed budget of $m$ pruning layers.

Let $i _ { K + 1 } = L , i _ { 0 } = 2$ . Then the model performance is formulated as:

$$
P (s) = \frac {\sum_ {k = 0} ^ {K} x _ {i _ {k}} \left(i _ {k + 1} - i _ {k}\right)}{L - 2}. \tag {7}
$$

Let $U ( s )$ denote the integral in the numerator. If the previous update occurs at layer $i _ { k - 1 }$ and the next at layer $j$ , then the marginal area contribution of current update $i$ is:

$$
\Delta U (i | i _ {k - 1}, j) = \left(x _ {i} - x _ {i _ {k - 1}}\right) (j - i). \tag {8}
$$

This constitutes a dynamic programming problem. Consider the last update: it can occur either at the current layer $i$ or at

a later layer $j$ . The necessary and sufficient condition for $j$ to be preferable to $i$ is:

$$
x _ {j} (L - j) \geq x _ {i} (L - i) - x _ {i _ {m - 1}} (j - i). \tag {9}
$$

This establishes the state transition equation. The optimal solution, and therefore the optimal pruning layers, follows directly.

As shown in Fig.4, we conduct layer selection experiments using LLaVA-1.5-7B on three localization datasets (RefCOCO, RefCOCO+, RefCOCOg) and three nonlocalization datasets (TextVQA, GQA, V2-VQA). From the training split of each dataset, 1,000 instances are randomly sampled for evaluation.

Despite dataset-specific variations, consistent patterns can still be observed across datasets. In particular, early layers exhibit noticeable fluctuations, and performance consistently peaks around layer 15, suggesting shared characteristics in layer-wise token discriminability.

Performance metrics are first normalized across all datasets and then averaged to obtain $\left\{ x _ { i } \right\} _ { i = 1 } ^ { L }$ . Following the above layer selection protocol, layers 3, 11, and 15 are selected as pruning layers.

# 3.3. Architecture

For each model, we first select a set of pruning layers, denoted as layers $x$ and $y$ in Fig.5.

The first pruning operation is performed after layer $x$ . Based on the attention map produced by layer $x$ , we extract the T–V attention scores between the last text token and all visual tokens. The top-ranked visual tokens are retained and directly propagated to layer $x + 1$ for further inference. The remaining low-ranked visual tokens are grouped according to the similarity between their hidden states, measured by

$$
s _ {i, j} = \frac {\left(\mathbf {h} _ {i} ^ {x}\right) ^ {\top} \mathbf {h} _ {j} ^ {x}}{\left| \mathbf {h} _ {i} ^ {x} \right| \left| \mathbf {h} _ {j} ^ {x} \right|}, \tag {10}
$$

where $\mathbf { h } _ { i }$ and $\mathbf { h } _ { j }$ denote the hidden states of visual tokens i and $j$ , respectively. Visual tokens within the same group are then merged by averaging their hidden states across feature dimensions, yielding a single merged token

$$
\tilde {\mathbf {h}} _ {g m} ^ {x} = \tilde {\mathbf {h}} _ {g} ^ {x} = \frac {1}{| \mathcal {G} _ {g} |} \sum_ {i \in \mathcal {G} _ {g}} \mathbf {h} _ {i} ^ {x}, \tag {11}
$$

which participates in the computation of layer $x + 1$ .

Here, we propose a new pruning strategy termed bypass. Instead of permanently discarding unselected visual tokens, bypass preserves these tokens and forwards them through a side pathway to the next pruning layer, where they reparticipate in the pruning selection process.

![](images/b3198d4e364c33dea49653d377741b70e4ded0b56553b1344fcdd5d32ba78371.jpg)  
Figure 5. SwiftVLM architecture overview. (a) After layer $x$ , unselected visual tokens are grouped for bypassing, with the resulting merged tokens participating in subsequent computation. (b) Before layer $_ y$ , token alignment is applied to restore grouped tokens, enabling re-evaluation of visual tokens at layers with stronger token selection capability.

Before the pruning layer $y$ , we re-evaluate the importance of all visual tokens. For each group formed by merged tokens, we estimate the average offset of the group as

$$
\Delta \mathbf {h} _ {g m} = \tilde {\mathbf {h}} _ {g m} ^ {y - 1} - \tilde {\mathbf {h}} _ {g m} ^ {x}. \tag {12}
$$

To align the visual tokens transmitted through the bypass pathway with the deeper representations of other tokens, we correct each visual token in group $g$ as follows:

$$
\hat {\mathbf {h}} _ {i} ^ {y - 1} = \mathbf {h} _ {i} ^ {x} + \Delta \mathbf {h} _ {g m}, \quad i \in \mathcal {G} _ {g}. \tag {13}
$$

Using the aligned visual tokens and the key projection matrix $W _ { K } ^ { y }$ of pruning layer $y$ , we construct the key representations. The query is obtained by projecting the last text token from layer $y - 1$ with $W _ { Q } ^ { y }$ . We then compute the T–V attention and perform visual token selection once again. At this stage, only the selected important visual tokens are retained to participate in the subsequent prefill computation.

# 3.4. Representation Alignment Analysis

Transformer (Vaswani et al., 2017) layers adopt a residual formulation, where the hidden states are updated as

$$
\mathbf {h} ^ {\ell} = \mathbf {h} ^ {\ell - 1} + \mathcal {F} ^ {\ell} \left(\mathbf {h} ^ {\ell - 1}\right), \tag {14}
$$

with $\mathcal { F } ^ { \ell } ( \cdot )$ denoting the combined attention and feedforward transformation at layer $\ell$ .

For a visual token $i$ belonging to group $\mathcal { G } _ { g }$ , its hidden state in the vanilla model evolves from layer $x + 1$ to layer $y - 1$ as

$$
\mathbf {h} _ {i} ^ {y - 1} = \mathbf {h} _ {i} ^ {x} + \sum_ {\ell = x + 1} ^ {y - 1} \mathcal {F} ^ {\ell} \left(\mathbf {h} _ {i} ^ {\ell - 1}\right). \tag {15}
$$

Taking the average over all tokens in group $\mathcal { G } _ { g }$ , we obtain

$$
\tilde {\mathbf {h}} _ {g} ^ {y - 1} = \tilde {\mathbf {h}} _ {g} ^ {x} + \sum_ {\ell = x + 1} ^ {y - 1} \frac {1}{| \mathcal {G} _ {g} |} \sum_ {i \in \mathcal {G} _ {g}} \mathcal {F} ^ {\ell} \left(\mathbf {h} _ {i} ^ {\ell - 1}\right), \tag {16}
$$

We denote by $\Delta \mathbf { h } _ { g }$ the accumulated group-level residual update.

In Sec.4.4, we obtain $\Delta \mathbf { h } _ { g }$ from the vanilla model and compare it with $\Delta \mathbf { h } _ { g m }$ . Under fine-grained grouping, their low-dimensional projections show near-complete overlap, providing empirical support for the proposed offset-based approximation.

# 3.5. FLOPs Computation

We consider a setting where visual tokens are pruned after the $K$ -th VLM layer, removing a fraction $D \%$ of visual tokens. Let $n _ { v }$ and $n _ { t }$ denote the numbers of visual tokens and non-visual tokens, respectively, with $T$ layers, hidden dimension $d$ , and FFN intermediate dimension $m$ . The total number of tokens is $n = n _ { v } + n _ { t }$ . and the token count after pruning becomes $\hat { n } = ( 1 - D \% ) * n _ { v } + n _ { t }$ . The resulting FLOPs $F$ are given by:

$$
C _ {n} = \left(4 n d ^ {2} + 2 n ^ {2} d + 3 n d m\right), \tag {17}
$$

$$
F = K \times C _ {n} + (T - K) \times C _ {\hat {n}}. \tag {18}
$$

Furthermore, we analyze the additional computational overhead introduced by the proposed operation. Let $R$ denote the number of low-ranked visual tokens and $Z$ the number of merged tokens.

The merge step incurs an overhead of 2RZd. Representation

Table 1. Performance comparison under different visual token budgets. $( + )$ and (g) denote RefCOCO+ and RefCOCOg, respectively.   

<table><tr><td rowspan="2">Method</td><td colspan="4">Localization</td><td colspan="7">Non-localization</td><td rowspan="2">FLOPs (T)</td></tr><tr><td>RefCOCO</td><td>(+)</td><td>(g)</td><td>Avg.</td><td>VQA\(^{\text{Text}}\)</td><td>GQA</td><td>SQA</td><td>MME</td><td>MMB</td><td>POPE</td><td>Avg.</td></tr><tr><td colspan="13">Upper Bound, 576 Tokens (100%)</td></tr><tr><td rowspan="2">Vanilla</td><td>75.9</td><td>67.0</td><td>70.7</td><td rowspan="2">100%</td><td>46.9</td><td>61.4</td><td>69.6</td><td>1509</td><td>64.6</td><td>86.8</td><td rowspan="2">100%</td><td rowspan="2">4.29</td></tr><tr><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td colspan="13">Retain 192 Tokens (↓ 66.7%)</td></tr><tr><td rowspan="2">FastV(ECCV&#x27;24)</td><td>30.6</td><td>25.8</td><td>29.7</td><td rowspan="2">40.3%</td><td>43.6</td><td>57.2</td><td>69.4</td><td>1471</td><td>63.2</td><td>82.0</td><td rowspan="2">95.9%</td><td rowspan="2">1.71</td></tr><tr><td>40.3%</td><td>38.5%</td><td>42.0%</td><td>93.0%</td><td>93.2%</td><td>99.7%</td><td>97.5%</td><td>97.8%</td><td>94.5%</td></tr><tr><td rowspan="2">VisionZip(CVPR&#x27;25)</td><td>7.0</td><td>5.7</td><td>6.3</td><td rowspan="2">8.9%</td><td>45.2</td><td>58.9</td><td>68.8</td><td>1460</td><td>62.9</td><td>86.6</td><td rowspan="2">97.5%</td><td rowspan="2">1.71</td></tr><tr><td>9.2%</td><td>8.5%</td><td>8.9%</td><td>96.4%</td><td>95.9%</td><td>98.9%</td><td>96.8%</td><td>97.4%</td><td>99.8%</td></tr><tr><td rowspan="2">PDrop(CVPR&#x27;25)</td><td>22.2</td><td>18.2</td><td>18.7</td><td rowspan="2">27.6%</td><td>42.9</td><td>55.5</td><td>69.2</td><td>1365</td><td>63.2</td><td>81.1</td><td rowspan="2">93.8%</td><td rowspan="2">1.72</td></tr><tr><td>29.2%</td><td>27.2%</td><td>26.4%</td><td>91.5%</td><td>90.4%</td><td>99.4%</td><td>90.5%</td><td>97.8%</td><td>93.4%</td></tr><tr><td rowspan="2">SparseVLM(ICML&#x27;25)</td><td>8.7</td><td>7.5</td><td>7.1</td><td rowspan="2">10.9%</td><td>45.8</td><td>58.9</td><td>69.1</td><td>1447</td><td>64.2</td><td>86.7</td><td rowspan="2">98.0%</td><td rowspan="2">1.72</td></tr><tr><td>11.5%</td><td>11.2%</td><td>10.0%</td><td>97.7%</td><td>95.9%</td><td>99.3%</td><td>95.9%</td><td>99.4%</td><td>99.9</td></tr><tr><td rowspan="2">FEATHER(ICCV&#x27;25)</td><td>52.0</td><td>45.5</td><td>45.4</td><td rowspan="2">66.9%</td><td>42.9</td><td>58.6</td><td>70.5</td><td>1431</td><td>63.9</td><td>84.4</td><td rowspan="2">96.5%</td><td rowspan="2">1.82</td></tr><tr><td>68.5%</td><td>67.9%</td><td>64.2%</td><td>91.5%</td><td>95.4%</td><td>101.3%</td><td>94.8%</td><td>98.9%</td><td>97.2%</td></tr><tr><td rowspan="2">SwiftVLM</td><td>66.6</td><td>58.5</td><td>60.6</td><td rowspan="2">86.9%</td><td>45.3</td><td>60.7</td><td>69.0</td><td>1503</td><td>64.5</td><td>87.1</td><td rowspan="2">99.0%</td><td rowspan="2">1.75</td></tr><tr><td>87.7%</td><td>87.3%</td><td>85.7%</td><td>96.6%</td><td>98.9%</td><td>99.1%</td><td>99.6%</td><td>99.8%</td><td>100.3%</td></tr><tr><td colspan="13">Retain 128 Tokens (↓ 77.8%)</td></tr><tr><td rowspan="2">FastV(ECCV&#x27;24)</td><td>12.8</td><td>11.1</td><td>13.8</td><td rowspan="2">17.7%</td><td>39.7</td><td>53.6</td><td>68.5</td><td>1377</td><td>62.3</td><td>77.7</td><td rowspan="2">91.3%</td><td rowspan="2">1.29</td></tr><tr><td>16.9%</td><td>16.6%</td><td>19.5%</td><td>84.6%</td><td>87.3%</td><td>98.4%</td><td>91.3%</td><td>96.4%</td><td>89.5%</td></tr><tr><td rowspan="2">VisionZip(CVPR&#x27;25)</td><td>4.6</td><td>3.6</td><td>4.3</td><td rowspan="2">5.8%</td><td>44.4</td><td>57.5</td><td>68.9</td><td>1441</td><td>62.0</td><td>85.1</td><td rowspan="2">96.1%</td><td rowspan="2">1.29</td></tr><tr><td>6.0%</td><td>5.4%</td><td>6.1%</td><td>94.7%</td><td>93.6%</td><td>99.0%</td><td>95.5%</td><td>96.0%</td><td>98.0%</td></tr><tr><td rowspan="2">PDrop(CVPR&#x27;25)</td><td>3.0</td><td>2.3</td><td>2.3</td><td rowspan="2">3.6%</td><td>39.9</td><td>54.3</td><td>70.2</td><td>1322</td><td>61.9</td><td>80.9</td><td rowspan="2">91.8%</td><td rowspan="2">1.28</td></tr><tr><td>4.0%</td><td>3.4%</td><td>3.3%</td><td>85.1%</td><td>88.4%</td><td>100.9%</td><td>87.6%</td><td>95.8%</td><td>93.2%</td></tr><tr><td rowspan="2">SparseVLM(ICML&#x27;25)</td><td>4.8</td><td>3.9</td><td>4.1</td><td rowspan="2">6.0%</td><td>42.0</td><td>57.4</td><td>69.8</td><td>1418</td><td>63.6</td><td>86.0</td><td rowspan="2">95.8%</td><td rowspan="2">1.30</td></tr><tr><td>6.3%</td><td>5.8%</td><td>5.8%</td><td>89.6%</td><td>93.5%</td><td>100.3%</td><td>94.0%</td><td>98.5%</td><td>99.1%</td></tr><tr><td rowspan="2">FEATHER(ICCV&#x27;25)</td><td>39.0</td><td>34.3</td><td>35.2</td><td rowspan="2">50.8%</td><td>41.2</td><td>56.5</td><td>69.6</td><td>1453</td><td>63.2</td><td>83.3</td><td rowspan="2">95.0%</td><td rowspan="2">1.44</td></tr><tr><td>51.4%</td><td>51.2%</td><td>49.8%</td><td>87.8%</td><td>92.0%</td><td>100%</td><td>96.3%</td><td>97.8%</td><td>96.0%</td></tr><tr><td rowspan="2">SwiftVLM</td><td>55.2</td><td>46.6</td><td>47.4</td><td rowspan="2">69.8%</td><td>41.8</td><td>59.2</td><td>68.5</td><td>1477</td><td>63.9</td><td>86.1</td><td rowspan="2">96.7%</td><td rowspan="2">1.31</td></tr><tr><td>72.7%</td><td>69.6%</td><td>67.0%</td><td>89.1%</td><td>96.4%</td><td>98.4%</td><td>97.9%</td><td>98.9%</td><td>99.2%</td></tr></table>

alignment adds an extra cost of $R d$ . Projecting the last text token to form the query costs $2 d ^ { 2 }$ , while projecting the visual tokens and computing the subsequent dot products introduce costs of $2 n _ { v } d ^ { 2 }$ and $2 n _ { v } d .$ , respectively. Let $r$ denote the ratio of visual tokens retained at layer $y$ . The overall computational overhead $F _ { o }$ is thus given by

$$
\mathrm {F} _ {\mathrm {o}} = 2 R Z d + R d + 2 n _ {v} d + 2 d ^ {2} + 2 (1 - r) n _ {v} d ^ {2}. \tag {19}
$$

# 4. Experiments

# 4.1. Overall Performance

Datasets. We categorize inference tasks into localization and non-localization types, where the former emphasizes fine-grained visual details and the latter focuses on holistic information integration. We evaluate our method on nine widely used benchmarks, including RefCOCO, RefCOCO+,

RefCOCOg (Kazemzadeh et al., 2014; Yu et al., 2016), TextVQA, GQA (Hudson & Manning, 2019), SQA (Lu et al., 2022), MME (Bolya et al., 2022), MMB (Liu et al., 2024c), POPE (Li et al., 2024a). For TextVQA, we follow prior work (Endo et al., 2025) and exclude OCR prompt to better evaluate how pruning affects visual understanding.

Main Results. Since the average RefCOCO bounding box covers about 102 visual tokens, Tab. 1 reports the performance of different methods on LLaVA-1.5-7B under two visual token budgets (192 and 128). Across non-localization tasks, all methods achieve competitive performance, including VisionZip, which employs text-agnostic feature compression.

In contrast, performance differences become pronounced on localization tasks. Notably, PDrop and SparseVLM do not preserve the positional information of visual tokens

![](images/620580f23e83e70b374cf62cb469e93fb7d3e7b3938ad1ab5e9912f93c8c8503.jpg)  
Please provide the bounding box coordinate of the region this sentence describes: “small white car”

![](images/9a44f2658cd9e352c30d9cdd8bb272701389146af400c4f0e7860e2935f89121.jpg)

![](images/9c931e1ee5ab072308c2889ef1e639828b10fffc4802872483d76c335e21bc3c.jpg)

![](images/448d63f0e93cfca84d7529dbdb64f923c7eeb79cddbb700dbb1c4c596544e6bb.jpg)  
(a) Avg. 192 Visual Tokens  
(b) Avg. 128 Visual Tokens  
Figure 6. Visualization of method performance under varying tasks and computation budgets.

after pruning, leading to substantial performance degradation (Chien et al., 2025). FEATHER mitigates the impact of RoPE by recomputing attention, resulting in higher FLOPs compared to other methods. Moreover, despite eliminating RoPE effects, the ability of different layers to discriminate important visual tokens in FEATHER remains nonmonotonic, and low-ranked visual tokens are still dropped after the initial pruning stage. As a result, FEATHER underperforms SwiftVLM by by roughly $20 \%$ .

Visualization. We visualize examples from RefCOCO and TextVQA, showing the retained visual tokens as image patches along with the final answers. As illustrated in Fig.6, FEATHER and PDrop adopt drop-based pruning and discard task-relevant visual tokens (e.g., the car in localization and the signboard in VQA), leading to incomplete or incorrect answers.

# 4.2. Efficiency Study

Following SparseVLM, we implement SwiftVLM in a FlashAttention-compatible (Dao et al., 2022) manner and report the corresponding latency results in Tab.2. Compared to the vanilla model, all pruning-based methods achieve noticeable speedups. FastV attains the largest acceleration since it performs pruning only once.

Unlike FLOPs computation, FlashAttention does not provide direct access to attention maps, requiring attention scores to be recomputed in practice. Consequently, SwiftVLM incurs lower latency than SparseVLM, as it only computes attention between the final text token and visual

Table 2. Efficiency study on LLaVA-1.5-7B. Total Time denotes the wall-clock time required to process the entire POPE dataset. Prefilling Time refers to the average prefill latency per sample. $\Delta$ indicates the speedup factor relative to the vanilla model.   

<table><tr><td>Tokens</td><td>Method</td><td>Total Time (s)</td><td>Δ</td><td>Prefilling Time (ms)</td><td>Δ</td></tr><tr><td>576</td><td>Vanilla</td><td>850.7</td><td>-</td><td>67.3</td><td>-</td></tr><tr><td rowspan="3">192</td><td>FastV</td><td>551.8</td><td>1.54×</td><td>34.7</td><td>1.92×</td></tr><tr><td>SparseVLM</td><td>612.3</td><td>1.39×</td><td>40.7</td><td>1.65×</td></tr><tr><td>SwiftVLM</td><td>573.8</td><td>1.48×</td><td>37.6</td><td>1.79×</td></tr><tr><td rowspan="3">128</td><td>FastV</td><td>539.4</td><td>1.58×</td><td>32.8</td><td>2.05×</td></tr><tr><td>SparseVLM</td><td>583.9</td><td>1.46×</td><td>37.5</td><td>1.79×</td></tr><tr><td>SwiftVLM</td><td>546.2</td><td>1.56×</td><td>33.0</td><td>2.04×</td></tr></table>

Table 3. Ablation study. $\mathrm { X _ { S } }$ denotes layer selection. $\mathrm { X _ { M } }$ denotes token merging, and $\mathrm { X _ { B } }$ denotes the bypass mechanism.   

<table><tr><td>Tokens</td><td>Method</td><td>RefCOCO</td><td>VQA Text</td></tr><tr><td rowspan="4">192</td><td>Baseline</td><td>42.6</td><td>43.2</td></tr><tr><td>+ Xs</td><td>64.5</td><td>45.3</td></tr><tr><td>+ Xs + Xm</td><td>63.7</td><td>44.8</td></tr><tr><td>+ Xs + Xm + XB</td><td>66.6</td><td>45.3</td></tr><tr><td rowspan="4">128</td><td>Baseline</td><td>23.2</td><td>41.2</td></tr><tr><td>+ Xs</td><td>42.8</td><td>40.1</td></tr><tr><td>+ Xs + Xm</td><td>51.9</td><td>40.7</td></tr><tr><td>+ Xs + Xm + XB</td><td>55.2</td><td>41.8</td></tr></table>

tokens, whereas SparseVLM requires attention computation for all text tokens.

# 4.3. Ablation Study

We adopt PDrop as the baseline and augment it with positional encoding updates. Based on this configuration, we progressively introduce layer selection, token merging, and bypass, with results reported in Tab. 3.

Under the 192-token setting, pruning at layers with monotonically increasing selection capability yields the largest gains, while token merging degrades performance due to unnecessary information compression under sufficient computation budget. In contrast, under the more constrained 128-token setting, token merging becomes beneficial, as aggressive dropping would otherwise remove critical visual information. Overall, pruning with bypass consistently provides stable performance improvements across different budget settings.

# 4.4. Why Bypass Works?

To investigate why visual tokens forwarded through bypass can still participate effectively in subsequent computation after representation alignment, we analyze the lowdimensional projections of token offsets as described in Sec.3.4. Under the 128-token setting, we visualize the re-

![](images/1ac16612ab760f5bf360afc2735315eafe1aa5476cf6e21112da1c4c1ce5e12b.jpg)  
Visual Token Hidden-State Changes Visual Token Hidden-State Changes   
(a) Fine-grained Token Merging

![](images/9f10ba18bd8da69a2c22b6fe2e150d5b085f5778ba12da1e6693a0b9784b13e1.jpg)  
(b) Coarse-grained Token Merging   
Figure 7. t-SNE visualization of visual token hidden-state changes. Colors denote similarity-based token groups. In the vanilla model, • shows per-token changes and $\times$ shows the groupwise mean. In our method, each group is merged into a single token, its change from layer 3 to layer 10 is shown as a $\star$ . At $n = 1 8$ , merged tokens account for less than $5 \%$ .

![](images/e2c89e96290627e1b9d99d2183f253ccbfc4f1361620656997b67a964a38f829.jpg)

![](images/503fa0c5759459a601c8584ac247232e191b98a74bff14f6400eec05127ee31d.jpg)  
Figure 8. Token selection overlap with vanilla for drop and bypass. Under an equal computational budget, the overlap distribution and mean are reported over 4,000 cases by comparing the tokens selected at layer 15 under different pruning schemes with those selected by the vanilla model, in order to assess their impact on intrinsic selection behavior.

sults for a sample in TextVQA, as shown in the Fig.7(a). Here, Merged Token corresponds to the offset $\Delta h _ { g m }$ . For each bypassed group, we additionally run the vanilla model. Vanilla Token records the actual hidden-state changes of individual tokens within the group after layer 10, while Vanilla Group Mean represents the average hidden-state change computed from these tokens. We observe that the vanilla group mean closely overlaps with the merged token offset and remains highly consistent with the changes of individual tokens within the group. We then substantially reduce the number of merged tokens and report the results for the same example in Fig.7(b).

Given that VLMs employ causal attention, the hidden-state evolution of a visual token can actually only be influenced by preceding visual tokens. Moreover, since attention fundamentally operates through similarity-based interactions, we hypothesize that visual tokens with similar semantics exhibit similar transformation directions in the representation space, and can thus be well approximated by the changes of

Table 4. Performance comparison on LLaVA-NeXT-7B.   

<table><tr><td>Method</td><td>RefCOCO</td><td>VQA Text</td><td>GQA</td><td>MMB</td><td>Rel. Acc</td></tr><tr><td colspan="6">Upper Bound, Retain 100% Tokens</td></tr><tr><td>Vanilla</td><td>85.3</td><td>65.5</td><td>63.9</td><td>67.9</td><td>100%</td></tr><tr><td colspan="6">Retain 33.3% Tokens</td></tr><tr><td>FastV</td><td>40.5</td><td>58.7</td><td>59.0</td><td>48.3</td><td>75.1%</td></tr><tr><td>FEATHER</td><td>68.8</td><td>62.6</td><td>62.5</td><td>67.5</td><td>92.8%</td></tr><tr><td>SwiftVLM</td><td>80.7</td><td>64.1</td><td>63.6</td><td>68.0</td><td>98.0%</td></tr><tr><td colspan="6">Retain 22.2% Tokens</td></tr><tr><td>FastV</td><td>26.1</td><td>52.6</td><td>56.9</td><td>46.0</td><td>66.9%</td></tr><tr><td>FEATHER</td><td>53.1</td><td>60.9</td><td>61.9</td><td>66.5</td><td>87.5%</td></tr><tr><td>SwiftVLM</td><td>79.6</td><td>62.4</td><td>63.5</td><td>67.7</td><td>97.1%</td></tr></table>

the corresponding merged token.

# 4.5. Why Is Bypass Better Than Drop?

Under the 128-token setting, we compare the visual tokens retained at layer 15 by drop and bypass with the top $5 \%$ and top $10 \%$ tokens selected by the vanilla model, and report their overlap ratios on TextVQA and RefCOCO in Fig.8.

Bypass exhibits a higher overlap with the vanilla model, indicating its ability to preserve visual tokens that are critical for reasoning. This overlap gap is more pronounced on Ref-COCO, consistent with the larger performance differences observed across datasets under the 128-token setting in the ablation study.

# 4.6. Generalization

To evaluate generalization, following prior work, we conduct experiments on LLaVA-NeXT (Liu et al., 2024b) across four datasets. Due to image padding removal in LLaVA-NeXT, performance is compared using visual token retention ratios. SwiftVLM consistently outperforms other methods, with particularly notable gains on localization datasets.

# 5. Conclusion

In this work, we revisit visual token pruning in VLMs and reveal that visual token importance varies substantially across layers. This observation explains why existing drop-based pruning methods, which rely on early selection decisions, often struggle on tasks requiring fine-grained visual reasoning. To better preserve visual information, we introduce a novel pruning strategy, termed bypass, and integrate it into our proposed pruning framework, SwiftVLM. This design allows each pruning layer to perform token selection in a relatively independent manner. Experimental results demonstrate that bypass consistently outperforms drop, suggesting its potential as a promising pruning paradigm.

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35:23716–23736, 2022.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.   
Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024a.   
Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024b.   
Chien, T.-C., Lin, C.-K., Tsai, S.-F., Lai, R.-C., Chen, H.-J., and Sun, M. Grounding-aware token pruning: Recovering from drastic performance drops in visual grounding caused by pruning. arXiv preprint arXiv:2506.21873, 2025.   
Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. Flashat-´ tention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.   
Endo, M., Wang, X., and Yeung-Levy, S. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22826– 22835, 2025.   
Gao, C., Liu, Z., Chi, Z., Huang, J., Fei, X., Hou, Y., Zhang, Y., Lin, Y., Fang, Z., Jiang, Z., et al. Vlaos: Structuring and dissecting planning representations and paradigms in vision-language-action models. arXiv preprint arXiv:2506.17561, 2025.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.   
Kang, S., Kim, J., Kim, J., and Hwang, S. J. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9339– 9350, 2025.   
Kazemzadeh, S., Ordonez, V., Matten, M., and Berg, T. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798, 2014.   
Lad, V., Lee, J. H., Gurnee, W., and Tegmark, M. The remarkable robustness of llms: Stages of inference? arXiv preprint arXiv:2406.19384, 2024.   
Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308, 2024a.   
Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023.   
Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pp. 323–340. Springer, 2024b.   
Lin, T., Zhang, W., Li, S., Yuan, Y., Yu, B., Li, H., He, W., Jiang, H., Li, M., Song, X., et al. Healthgpt: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation. arXiv preprint arXiv:2502.09838, 2025.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024a.   
Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024c.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.   
Shang, Y., Cai, M., Xu, B., Lee, Y. J., and Yan, Y. Llavaprumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22857– 22867, 2025.   
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.   
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Wang, J., Wang, M., Zhou, Z., Yan, J., Wu, L., et al. The sharpness disparity principle in transformers for accelerating language model pre-training. arXiv preprint arXiv:2502.19002, 2025a.   
Wang, Q., Ye, H., Chung, M.-Y., Liu, Y., Lin, Y., Kuo, M., Ma, M., Zhang, J., and Chen, Y. Corematching: A co-adaptive sparse inference framework with token and neuron pruning for comprehensive acceleration of visionlanguage models. arXiv preprint arXiv:2505.19235, 2025b.   
Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.

Yang, Q., Zhang, C., Fan, L., Ding, K., Ye, J., and Xiang, S. Re-ranking reasoning context with tree search makes large vision-language models stronger. arXiv preprint arXiv:2506.07785, 2025a.   
Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., and Jia, J. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19792–19802, 2025b.   
Ye, X., Gan, Y., Ge, Y., Zhang, X.-P., and Tang, Y. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24972–24982, 2025a.   
Ye, X., Gan, Y., Huang, X., Ge, Y., and Tang, Y. Voco-llama: Towards vision compression with large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29836–29846, 2025b.   
Yu, L., Poirson, P., Yang, S., Berg, A. C., and Berg, T. L. Modeling context in referring expressions. In European conference on computer vision, pp. 69–85. Springer, 2016.   
Zhang, Q., Cheng, A., Lu, M., Zhang, R., Zhuo, Z., Cao, J., Guo, S., She, Q., and Zhang, S. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20857–20867, 2025.   
Zhang, Y., Fan, C.-K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D., Okuno, T., Nakata, Y., Keutzer, K., et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024.   
Zhong, Y., Liu, Z., Li, Y., and Wang, L. Aim: Adaptive inference of multi-modal llms via token merging and pruning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20180–20192, 2025.