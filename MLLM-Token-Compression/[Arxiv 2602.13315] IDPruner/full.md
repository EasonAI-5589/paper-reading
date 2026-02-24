# IDPruner: Harmonizing Importance and Diversity in Visual Token Pruning for MLLMs

Yifan $\mathbf { T a n } ^ { 1 , 2 }$ , Yifu $\mathbf { S u n } ^ { 2 }$ , Shirui Huang2, Hong Liu2, Guanghua $\mathbf { Y } \mathbf { u } ^ { 2 }$ , Jianchen $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 2 }$ , Yangdong Deng1

1School of Software, Tsinghua University, 2Tencent

# Abstract

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities, yet they encounter significant computational bottlenecks due to the massive volume of visual tokens. Consequently, visual token pruning, which substantially reduces the token count, has emerged as a critical technique for accelerating MLLM inference. Existing approaches focus on token importance, diversity, or an intuitive combination of both, without a principled framework for their optimal integration. To address this issue, we first conduct a systematic analysis to characterize the trade-off between token importance and semantic diversity. Guided by this analysis, we propose the Importance and Diversity Pruner (IDPruner), which leverages the Maximal Marginal Relevance (MMR) algorithm to achieve a Paretooptimal balance between these two objectives. Crucially, our method operates without requiring attention maps, ensuring full compatibility with FlashAttention and efficient deployment via one-shot pruning. We conduct extensive experiments across various model architectures and multimodal benchmarks, demonstrating that IDPruner achieves state-of-theart performance and superior generalization across diverse architectures and tasks. Notably, on Qwen2.5-VL-7B-Instruct, IDPruner retains $9 5 . 1 8 \%$ of baseline performance when pruning $7 5 \%$ of the tokens, and still maintains $8 6 . 4 0 \%$ even under an extreme $90 \%$ pruning ratio. Our code is available at https: //github.com/Tencent/AngelSlim.

# 1 Introduction

Multimodal Large Language Models (MLLMs) have achieved significant success in artificial intelligence. These models typically encode images or videos into sequences of visual tokens, which are then processed together with textual inputs by the language model to generate text responses (Liu et al., 2023b,a). For instance, $\mathrm { Q w e n } 2 . 5 – \mathrm { V L }$ generates approximately 2,691 visual tokens when processing a single 1080p image $( 1 9 2 0 { \times } 1 0 8 0 )$ , with each token representing a $2 8 \times 2 8$ pixel patch. The high number of visual tokens creates a heavy computational burden, limiting the efficiency and practical deployment of MLLMs (Zhou et al., 2024). Thus, visual token pruning (Wang et al., 2025; Shao et al., 2025), which aims to reduce the number of visual tokens while maintaining model performance, has emerged as a critical technique for achieving efficient MLLM inference.

![](images/6d92df049fe74d544e2eab0acf6c17ee3c96c09e9dbc189a8092891613c2d70e.jpg)  
Figure 1: Performance comparison across four architectures and eight benchmarks. IDPruner (outermost boundary) consistently outperforms baselines in both (a) aggregated performance across four diverse MLLM architectures and (b) fine-grained benchmark breakdown for Qwen2.5-VL. This demonstrates the superior cross-architecture generalization and task-specific robustness of our method.

Existing pruning strategies generally fall into two categories: importance-based and diversitybased methods. Importance-based approaches (Chen et al., 2024a; Yang et al., 2025b,a) select salient tokens, focusing on foreground objects, but often sacrificing the background context essential for global reasoning. In contrast, diversity-based methods (Alvar et al., 2025; Zou et al., 2025) maximize semantic coverage to reduce redundancy but risk retaining task-irrelevant noise while missing fine-grained details. Recent hybrid approaches (Zhang et al., 2024c, 2025a; Li et al., 2025) attempt to combine these complementary criteria but lack rigorous analysis, relying on intuition-based integration that yields suboptimal performance. Therefore, a systematic analytical framework is needed to characterize the interaction between importance and diversity and derive optimal integration strategies.

![](images/1ad54b8acaadd4d1e9fbfe1d8d5ca8ca259784f741f7a45657332237eb6074a6.jpg)  
Figure 2: Overview of the IDPruner framework. Left: Integration of our one-shot visual token pruning into the MLLM inference pipeline. Right: The core mechanism computes Importance Scores (Red) and a Similarity Matrix (Blue), utilizing an MMR selection process to harmonize importance and diversity. This approach operates without attention maps and remains compatible with FlashAttention.

To address this, we first conduct a systematic analysis to investigate the trade-off between token importance and semantic diversity. As shown in Figure 3, our analysis reveals that current approaches fail to effectively balance these two critical dimensions. To overcome this limitation, we introduce the Importance and Diversity Pruner (IDPruner), a novel pruning strategy designed to balance these criteria optimally. Specifically, as illustrated in Figure 2, we cast visual token pruning as a re-ranking problem in information retrieval and adapt the Maximal Marginal Relevance (MMR) (Carbonell and Goldstein-Stewart, 1998) algorithm to model the interplay between token importance and semantic diversity explicitly. This approach selects tokens that jointly maximize both importance and diversity.

IDPruner achieves state-of-the-art performance, as demonstrated by comprehensive evaluations on multimodal benchmarks. Notably, on the Qwen2.5- VL-7B-Instruct model, even under an extreme compression ratio of $90 \%$ , our method retains $\mathbf { 8 6 . 4 0 \% }$ of the baseline performance, significantly outperforming existing competitive approaches. Crucially, unlike progressive pruning strategies that dynamically change sequence lengths, IDPruner performs one-shot pruning at an early stage, which makes it easier to integrate into inference engines like vLLM (Kwon et al., 2023). Furthermore, our method works without requiring attention information, ensuring full compatibility with FlashAttention (Dao et al., 2022) to maximize inference efficiency.

The main contributions of this work are summarized as follows:

• We conduct a systematic analysis to characterize the trade-off between token importance and semantic diversity, providing a theoretical basis for their integration.

• We propose IDPruner, which adapts the Maximal Marginal Relevance (MMR) algorithm to visual token pruning, enabling the optimal harmonization of importance and diversity.

• Extensive experiments demonstrate that our method achieves state-of-the-art performance and exceptional cross-architecture generalization, as visualized in Figure 1, while supporting one-shot pruning and FlashAttention acceleration, offering a practical solution for efficient MLLM deployment.

# 2 Related work

Large Multimodal Models and Visual Token Pruning. Recent Multimodal Large Language Models (MLLMs) (Liu et al., 2023a; Wang et al., 2024; Zhu et al., 2025b) have demonstrated impressive capabilities across various visual tasks, yet they encounter significant computational bottlenecks due to the massive volume of visual tokens. Static-resolution models like LLaVA-1.5 (Liu et al.,

2023a) and LLaVA-NeXT (Liu et al., 2024a) require 576 and 2,880 input tokens per image, respectively, while newer architectures such as the Qwen-VL (Bai et al., 2025), LLaVA-OneVision (Li et al., 2024a), and InternVL (Zhu et al., 2025b) series demand comparable token budgets for highresolution processing. Consequently, visual token pruning, which eliminates unnecessary tokens, has emerged as a crucial technique for accelerating MLLM inference. Current research typically falls into two categories: importance-based methods and diversity-based methods.

Importance-based Token Pruning. Importancebased approaches reduce computational overhead by retaining only the most salient tokens. Early studies rely on attention scores from LLM decoder layers (Chen et al., 2024a; Zhang et al., $2 0 2 4 \mathrm { e }$ ; Xing et al., 2024; Zhang et al., 2025b; Ye et al., 2025; Han et al., 2025), while subsequent research discovers that the attention of the [CLS] token in Vision Transformers (ViT) provides a more effective importance measure (Yang et al., 2025b; Liu et al., 2025; Zhang et al., 2024d; Tong et al., 2025). To mitigate limitations such as FlashAttention incompatibility, recent work has introduced alternative metrics, including optimal transport and L2 norms (Yang et al., 2025a; Dhouib et al., 2025). Beyond training-free methods, approaches like VisionSelector (Zhu et al., 2025a) employ learnable modules to estimate token importance, achieving state-of-theart performance through end-to-end training. Despite their effectiveness in capturing region-specific details, these methods often overlook global context, potentially causing information loss in background areas.

Diversity-based Token Pruning. Diversity-based approaches aim to preserve information coverage by regarding visual tokens as a collective set, minimizing redundancy to retain a representative subset of visual features. DivPrune (Alvar et al., 2025) formulates this task as a Max-Min Diversity Problem, solving it via a greedy algorithm to maximize semantic coverage, while DART (Wen et al., 2025) employs a parallelizable strategy that selects pivot tokens and eliminates their nearest neighbors to maintain diversity. However, maximizing redundancy reduction often comes at the cost of missing fine-grained details in focal regions, as these methods may indiscriminately retain task-irrelevant noise.

Hybrid Strategies. Synergizing importance and diversity typically yields superior performance compared to single-criterion methods. VisPrune (Zhang et al., 2024c) allocates token budgets based on both [CLS] attention and diversity, while CDPruner (Zhang et al., 2025a) employs Determinantal Point Processes (DPP) to balance these objectives. Other approaches explore alternative integration strategies, such as ensuring spatial coverage via regionbased allocation (Zou et al., 2025; Arif et al., 2025) or modeling pruning as a set cover problem to optimize multimodal coverage (Li et al., 2025; Deng et al., 2025). Although effective, these methods typically rely on heuristic integration strategies without a systematic analytical framework. In this work, we address this limitation by introducing a systematic framework that optimally harmonizes importance and diversity.

# 3 An Empirical Analysis of the Importance-Diversity Trade-off

# 3.1 Quantifying Importance and Diversity

Visual token pruning strategies typically focus on either importance-based selection or diversity preservation; however, balancing these two goals remains challenging. To systematically analyze the relationship between these two paths, we first reformulate the visual token pruning problem.

Definition 1 (Visual Token Pruning). Let $\nu =$ $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ denote the set of $N$ visual tokens, where each token $v _ { i } \in \mathbb { R } ^ { d }$ represents a $d$ dimensional feature vector. Visual token pruning aims to select a subset $s \subset \nu$ with $| S | = K < N$ tokens, where $K$ is a pre-defined budget constraint.

To decouple the combining strategy from any specific importance estimator, we pre-define an importance vector w representing the weight of each token, regardless of how w is calculated. Based on this, we define the retention metric:

Definition 2 (Importance Retention Ratio). The importance retention ratio of a subset $s$ is defined as the normalized sum of retained scores:

$$
\mathcal { T } ( S ) = \frac { \sum _ { v _ { k } \in S } w _ { k } } { \sum _ { v _ { i } \in \mathcal { V } } w _ { i } }
$$

This metric quantifies the proportion of total information retained by the subset, ranging from $O$ to 1.

In contrast to importance, which focuses on individual token utility, we characterize the spatial distribution of the selected subset using the Hopkins Statistic (Hopkins and Skellam, 1954), a measure that quantifies the degree of clustering in a dataset. A high Hopkins value indicates strong clustering, meaning that selected tokens concentrate in specific semantic regions and thus exhibit high redundancy.

Definition 3 (Diversity Metric via Hopkins Statistic). Let $\boldsymbol { \mathcal { S } }$ denote the selected token subset with $| { \cal S } | = m$ . We construct a reference set $\mathcal { R }$ by randomly sampling m points from the same feature space as $s$ . Let $d ( x , y )$ denote the cosine distance from point $x$ to its nearest neighbor in set $\mathcal { V }$ . The Hopkins Statistic is defined as:

$$
H ( S ) = \frac { \sum _ { r \in { \mathcal R } } d ( r , S ) } { \sum _ { r \in { \mathcal R } } d ( r , S ) + \sum _ { v \in S } d ( v , S \setminus \{ v \} ) }
$$

In this formulation, ${ \mathcal { S } } \setminus \{ v \}$ denotes the set difference, representing the subset $s$ excluding the specific token v to ensure the distance is calculated against its nearest neighbor.

Intuitively, $H ( S )  1$ indicates high redundancy due to significant clustering, while $H ( S ) $ 0 signifies a regularly spaced distribution with maximal semantic diversity.

# 3.2 Simulation on Real Token Manifolds

To identify the optimal strategy for harmonizing importance and diversity, we conduct a systematic analysis to explore their interaction. Specifically, we employ real visual tokens extracted from the Vision Transformer of the Qwen2.5-VL-7BInstruct model as feature vectors. Real features are essential as they preserve complex manifold structures—such as semantic clustering and sparsity—that synthetic data typically fails to capture.

For token importance, we adopt a randomized approach where the score for each token is sampled independently from a uniform distribution $\mathcal { U } ( 0 , 1 )$ . This setup decouples the evaluation of selection strategies from the bias of any specific pre-trained importance scorer.

We evaluate five representative strategies that make different trade-offs between importance and diversity:

• Greedy Importance: Selects tokens with the highest importance scores, ignoring diversity.

• Greedy Diversity: Iteratively selects the token that maximizes distance to the current subset via Farthest Point Sampling (Resende et al., 2010), prioritizing diversity over importance.

• Naive Hybrid: A two-stage approach that first selects top- $k$ tokens by importance, then applies Farthest Point Sampling within this subset.

• Determinantal Point Processes (DPP): Models diversity probabilistically via the determinant of a kernel matrix (Macchi, 1975).

• Maximal Marginal Relevance (MMR): A joint optimization framework that explicitly balances importance and redundancy. We provide the detailed formulation of this mechanism in Section 3.3.

# 3.3 The Maximal Marginal Relevance (MMR) Mechanism

Maximal Marginal Relevance (MMR) (Carbonell and Goldstein-Stewart, 1998) provides a framework for this joint optimization. Initially proposed for information retrieval, the core idea of MMR is that an ideal result set should balance two criteria: high relevance to the query and low redundancy among selected items.

Adapting this principle to visual token pruning, the algorithm iteratively selects the token $v ^ { * }$ from the candidate set $\mathcal { V } \backslash \mathcal { S }$ that maximizes the following objective:

$$
\begin{array} { c } { v ^ { * } = \arg \underset { v _ { i } \in \mathcal { V } \backslash \mathcal { S } } { \operatorname* { m a x } } \left[ \lambda \cdot \mathrm { I m p } ( v _ { i } ) \right. } \\ { \left. - ( 1 - \lambda ) \cdot \underset { v _ { j } \in \mathcal { S } } { \operatorname* { m a x } } \mathrm { S i m } ( v _ { i } , v _ { j } ) \right] } \end{array}
$$

where $\nu$ represents the set of all visual tokens, $\boldsymbol { \mathcal { S } }$ denotes the currently selected subset, $\operatorname { I m p } ( { \cdot } )$ represents the normalized importance score, $\mathrm { S i m } ( \cdot , \cdot )$ measures the pairwise similarity between tokens, and $\lambda$ is a hyperparameter balancing the two terms.

By subtracting the maximum similarity between the candidate and the current subset $\boldsymbol { \mathcal { S } }$ , the algorithm explicitly penalizes tokens that are semantically close to any already selected token, while prioritizing important tokens.

# 3.4 Comparative Analysis against Heuristic Baselines

We conducted the simulation on 200 randomly sampled images from the MMBench dataset (Liu et al., 2023c) to systematically evaluate the efficacy of the proposed strategies.

Figure 3 illustrates the trade-off between importance retention and diversity for each strategy. The theoretical optimum resides in the top-left corner, corresponding to subsets that maximize $\mathcal { T }$ while minimizing $H$ , thereby maximizing diversity. As illustrated, the single-objective baselines occupy the sub-optimal extremes: Greedy Importance (Red node) achieves maximum $\mathcal { T }$ at the cost of a high Hopkins Statistic $H \approx 1 )$ ), whereas Greedy Diversity (Blue node) minimizes $H$ but suffers from a low Importance Retention Ratio.

![](images/4a55ce01392018b6f9996159dc05d3c5889eff92be5779c22710e5da9687a1a7.jpg)  
Figure 3: Pareto Frontier Analysis. We visualize the trade-off between the Hopkins Statistic $( H )$ and the Importance Retention Ratio $\left( \mathcal { D } \right)$ . The ideal pruning strategy should approach the top-left corner, achieving a high Importance Retention Ratio $( \mathcal { T }  1 )$ ) while minimizing the Hopkins Statistic ${ ( H  0 ) }$ ). The MMR mechanism (Orange) constructs a superior Pareto frontier that strictly dominates the Naive Hybrid strategy (Purple) and envelopes the DPP solution (Green).

Crucially, the trajectory generated by MMR (Orange curve) forms a superior Pareto Frontier. It strictly dominates the Naive Hybrid strategy (Purple curve), maintaining a higher $\mathcal { T }$ for any given level of $H$ , confirming the efficacy of our joint optimization framework. Furthermore, it effectively envelopes the DPP solution (Green node), demonstrating that our joint optimization framework provides the most robust mechanism for harmonizing these conflicting objectives.

# 4 Harmonizing Importance and Diversity via MMR

# 4.1 Token Importance Estimation

The analysis in Section 3 has demonstrated that the MMR mechanism effectively harmonizes diversity and importance. However, applying this framework in practice requires a computable importance metric.

To this end, we adopt the importance estimation mechanism of VisionSelector (Zhu et al., 2025a), which currently represents the state-of-theart among importance-based pruning approaches. Specifically, this method employs a trainable estimation module coupled with a differentiable selection mechanism, DiffTopK, to learn token importance through end-to-end training. To maintain consistency with the training phase, we utilize the output of the DiffTopK mechanism as our raw importance scores, denoted as w.

However, since MMR involves a direct subtraction between importance and similarity, both metrics must have comparable scales to prevent one from dominating the selection process. We therefore apply min-max normalization to the raw importance vector w to define the normalized importance metric:

$$
\mathrm { I m p } ( v _ { i } ) = \frac { w _ { i } - \mathrm { m i n } ( \mathbf { w } ) } { \mathrm { m a x } ( \mathbf { w } ) - \mathrm { m i n } ( \mathbf { w } ) + \epsilon }
$$

where $\epsilon$ is a small constant for numerical stability. This procedure maps importance scores to the interval $[ 0 , 1 ]$ , ensuring they are commensurate with the similarity constraint.

# 4.2 Quantifying Collective Redundancy

In addition to importance, the MMR framework requires a metric to quantify semantic redundancy. In the latent feature space of MLLMs, tokens representing similar visual concepts tend to cluster together. Thus, we define the pairwise similarity between a candidate token $v _ { i }$ and a reference token $v _ { j }$ using cosine similarity:

$$
\mathrm { S i m } ( v _ { i } , v _ { j } ) = \frac { v _ { i } ^ { \top } v _ { j } } { \| v _ { i } \| \| v _ { j } \| }
$$

where $\lVert \cdot \rVert$ denotes the Euclidean norm. This metric enables the algorithm to identify tokens that are semantically similar to those already selected.

# 4.3 IDPruner: An MMR-based Selection Strategy

Building upon the normalized importance and semantic similarity metrics defined above, we formally present the Importance and Diversity Pruner (IDPruner). This method harmonizes the two conflicting objectives within the MMR framework to iteratively construct the optimal subset. At each step $t$ , IDPruner selects the token $v ^ { * }$ from the remaining candidates $\nu \backslash S _ { t - 1 }$ by maximizing the following objective:

$$
v ^ { * } = \arg \operatorname* { m a x } _ { v _ { i } \in \mathcal { V } \backslash S _ { t - 1 } } [ \lambda \cdot \mathrm { I m p } ( v _ { i } ) - ( 1 - \lambda ) \cdot m _ { i } ]
$$

<table><tr><td>Method</td><td>AI2D EM</td><td>ChartQA Relaxed</td><td>DocVQA Anls</td><td>MMBCN Score</td><td>MMB Score</td><td>MME Score</td><td>MMStar Avg</td><td>OCRBench Acc</td><td>POPE Acc</td><td>SQA EM</td><td>VQAText EM</td><td>Avg</td></tr><tr><td>Baseline</td><td>82.48</td><td>83.68</td><td>94.90</td><td>80.41</td><td>83.08</td><td>1702</td><td>61.88</td><td>85.30</td><td>87.80</td><td>88.45</td><td>82.74</td><td>100.0%</td></tr><tr><td colspan="10">Retain 25% Tokens (75% Compression Ratio)</td><td></td><td></td><td></td></tr><tr><td colspan="10">Importance-based methods</td></tr><tr><td></td><td>75.68</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.16%</td></tr><tr><td>FastV VisionZip</td><td>77.40</td><td>68.20 67.20</td><td>81.20 71.48</td><td>73.20 76.12</td><td>76.12 78.78</td><td>1636 1637</td><td>51.08</td><td>43.00 46.50</td><td>85.20 85.76</td><td>83.49 83.99</td><td>80.06 76.21</td><td>87.55%</td></tr><tr><td>HiPrune</td><td>77.49</td><td>68.60</td><td>73.52</td><td>76.03</td><td>78.09</td><td>1619</td><td>54.86 54.43</td><td></td><td>86.02</td><td>84.18</td><td>76.43</td><td>87.80%</td></tr><tr><td>VisionSelector</td><td>79.60</td><td>72.00</td><td>93.24</td><td>75.86</td><td>78.78</td><td>1688</td><td>55.78</td><td>47.10 72.50</td><td>86.74</td><td>85.08</td><td>80.39</td><td>94.22%</td></tr><tr><td colspan="10">Diversity-based methods</td><td></td><td></td><td></td></tr><tr><td>DivPrune</td><td>77.98</td><td>62.00</td><td>85.32</td><td>75.77</td><td>77.84</td><td>1650</td><td>52.97</td><td>58.40</td><td>85.88</td><td>83.94</td><td>75.88</td><td>89.26%</td></tr><tr><td>DART</td><td>74.35</td><td>60.80</td><td>78.90</td><td>73.88</td><td>76.72</td><td>1625</td><td>52.90</td><td>46.00</td><td>84.34</td><td>84.33</td><td>71.68</td><td>85.74%</td></tr><tr><td colspan="10">Hybrid strategies</td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td>77.62</td><td>68.04</td><td>77.39</td><td>75.69</td><td>78.87</td><td>1657</td><td>54.01</td><td>48.70</td><td>85.68</td><td>84.18</td><td>75.17</td><td>88.31%</td></tr><tr><td>SCOPE</td><td>78.92</td><td>71.20</td><td>85.40</td><td>77.75</td><td>79.38</td><td>1684</td><td>56.86</td><td>61.70</td><td>86.78</td><td>85.23</td><td>79.66</td><td>92.51%</td></tr><tr><td>IDPruner</td><td>80.51</td><td>74.32</td><td>93.16</td><td>76.63</td><td>79.73</td><td>1695</td><td>56.49</td><td>74.00</td><td>87.06</td><td>85.52</td><td>80.83</td><td>95.18%</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td colspan="10">Retain 10% Tokens (90% Compression Ratio) Importance-based methods</td></tr><tr><td>FastV</td><td>67.23</td><td>39.48</td><td>51.90</td><td>53.26</td><td>55.58</td><td>1332</td><td>38.02</td><td>24.10</td><td>76.31</td><td>79.28</td><td>72.59</td><td>68.07%</td></tr><tr><td>VisionZip</td><td>70.60</td><td>41.56</td><td>37.94</td><td>66.67</td><td>71.05</td><td>1462</td><td>45.19</td><td>23.40</td><td>81.06</td><td>83.24</td><td>61.06</td><td>71.84%</td></tr><tr><td>HiPrune</td><td>69.82</td><td>43.96</td><td>39.89</td><td>67.44</td><td>70.88</td><td>1438</td><td>45.04</td><td>23.70</td><td>80.70</td><td>82.65</td><td>62.51</td><td>72.22%</td></tr><tr><td>VisionSelector</td><td>74.81</td><td>62.68</td><td>87.00</td><td>68.99</td><td>71.65</td><td>1569</td><td>46.93</td><td>55.50</td><td>82.69</td><td>81.95</td><td>74.52</td><td>85.39%</td></tr><tr><td>Diversity-based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune</td><td>70.11</td><td>41.36</td><td>66.20</td><td>69.42</td><td>72.16</td><td>1529</td><td>44.46</td><td>31.80</td><td>81.91</td><td>80.96</td><td>62.72</td><td>76.09%</td></tr><tr><td>DART</td><td>67.88</td><td>34.84</td><td>49.86</td><td>63.92</td><td>67.35</td><td>1451</td><td>42.93</td><td>24.30</td><td>79.70</td><td>80.96</td><td>54.06</td><td>69.80%</td></tr><tr><td>Hybrid strategies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">VisPruner</td><td>81.11</td><td></td><td>59.66</td><td>72.60%</td></tr><tr><td>SCOPE</td><td>69.88 71.63</td><td>42.68 50.04</td><td>50.85 56.45</td><td>66.84 71.22</td><td>70.96 75.43</td><td>1442 1608</td><td>44.14 48.74</td><td>24.40 34.10</td><td>81.03 84.10</td><td>82.25</td><td>70.61</td><td>79.35%</td></tr><tr><td>IDPruner</td><td>75.16</td><td>62.48</td><td>85.98</td><td>71.65</td><td>74.66</td><td>1618</td><td>47.48</td><td>53.90</td><td>85.43</td><td>82.80</td><td>74.43</td><td>86.47%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Comparison results on comprehensive Image-Language benchmarks on Qwen-2.5-7B-Instruct.

where $m _ { i } = \operatorname* { m a x } _ { v _ { j } \in S _ { t - 1 } } \mathrm { S i m } ( v _ { i } , v _ { j } )$ represents the maximum similarity between the candidate $v _ { i }$ and any token in the currently selected set, and $\lambda \in [ 0 , 1 ]$ is the hyperparameter balancing importance and diversity.

# Algorithm 1 IDPruner

Require: Tokens $\nu$ , Raw Importance Scores w,   
Budget $K$ , Hyperparameter $\lambda$   
Ensure: Pruned subset $s$   
1: $\mathrm { I m p }  ( \mathbf { w } - \operatorname* { m i n } \mathbf { w } ) / ( \operatorname* { m a x } \mathbf { w } - \operatorname* { m i n } \mathbf { w } + \epsilon )$   
2: $\pmb { \mathcal { S } } \gets \emptyset , \mathbf { m } \gets \mathrm { f i l l } ( N , - 1 . 0 )$   
3: for $t = 1$ to $K$ do   
4: if $t = 1$ then   
5: $v ^ { * } \gets \arg \operatorname* { m a x } _ { v _ { i } \in \mathcal { V } } \mathrm { I m p } ( v _ { i } )$   
6: else   
7: $\begin{array} { l } { { v ^ { * } \  \ \mathrm { a r g } \mathrm { m a x } _ { v _ { i } \notin { \cal S } } [ \lambda \mathrm { I m p } ( v _ { i } ) \ - \ ( 1 \ - } } \\ { { \lambda ) m _ { i } ] } } \end{array}$   
8: end if   
9: $\begin{array} { r l } & { \mathcal { S }  \mathcal { S } \cup \{ v ^ { * } \} } \\ & { \mathbf { m }  \operatorname* { m a x } ( \mathbf { m } , \mathrm { S i m } ( \mathcal { V } , v ^ { * } ) ) } \end{array}$   
10:   
11: end for   
12: return $s$

To minimize computational overhead, we adopt an efficient updating strategy. Instead of recomputing the similarity scores for all pairs at every step, we maintain a vector $\mathbf { m } \in \mathbb { R } ^ { N }$ that tracks the maximum similarity for each candidate. After selecting $v ^ { * }$ , we simply update this vector: $m _ { i } \gets \operatorname* { m a x } ( m _ { i } , \mathrm { S i m } ( v _ { i } , v ^ { * } ) )$ . This implementation reduces the computational complexity from $O ( K ^ { 2 } N )$ to $O ( K N )$ , rendering the overhead negligible relative to the model’s forward pass. The complete procedure is summarized in Algorithm 1.

# 5 Experiments

# 5.1 Experimental Setup

Model Architectures. We conduct our main experiments on widely adopted MLLMs, including Qwen2.5-VL-7B-Instruct (Bai et al., 2025) and LLaVA-1.5-7B. (Liu et al., 2023b).

Evaluation benchmarks. We conduct comprehensive evaluations on image and video understanding tasks. For image-language understanding, we employ 10 widely-used datasets: MME (Fu et al., 2023), MMBench (Liu et al., 2023c), MMStar (Chen et al., 2024b), POPE (Li et al., 2023), ScienceQA (Lu et al., 2022), AI2D (Kembhavi et al., 2016), TextVQA (Singh et al., 2019), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2020), and OCRBench (Liu et al., 2024b). For video-language understanding, we include 3 benchmarks: Vinoground (Zhang et al., 2024a),

<table><tr><td>Method</td><td>AI2D EM</td><td>ChartQA Relaxed</td><td>DocVQA Anls</td><td>MMBCN Score</td><td>MMB Score</td><td>MME Score</td><td>MMStar Avg</td><td>OCRBench Acc</td><td>POPE Acc</td><td>SQA EM</td><td>VQAText EM</td><td>Avg</td></tr><tr><td>Baseline</td><td>52.78</td><td>18.12</td><td>24.09</td><td>50.17</td><td>62.20</td><td>1463</td><td>32.74</td><td>19.80</td><td>85.86</td><td>66.19</td><td>47.78</td><td>100.0%</td></tr><tr><td colspan="9">Retain 128 Tokens (77% Compression Ratio)</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Importance-based methods</td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>50.58</td><td>13.84</td><td>12.69</td><td>46.13</td><td>57.90</td><td>1213</td><td>30.89</td><td>12.10</td><td>72.04</td><td>65.44</td><td>31.76</td><td>81.59%</td></tr><tr><td>VisionZip</td><td>50.81</td><td>16.80</td><td>19.93</td><td>48.88</td><td>60.05</td><td>1374</td><td>32.52</td><td>18.50</td><td>82.28</td><td>66.34</td><td>45.40</td><td>94.86%</td></tr><tr><td>HiPrune</td><td>51.98</td><td>17.00</td><td>20.79</td><td>49.74</td><td>60.14</td><td>1386</td><td>32.20</td><td>18.50</td><td>82.14</td><td>66.19</td><td>45.54</td><td>95.63%</td></tr><tr><td>VisionSelector</td><td>50.74</td><td>16.20</td><td>20.83</td><td>49.23</td><td>60.31</td><td>1379</td><td>34.22</td><td>18.40</td><td>83.06</td><td>66.39</td><td>45.29</td><td>95.51%</td></tr><tr><td>Diversity-based methods DivPrune</td><td></td><td></td><td>18.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DART</td><td>51.65 52.82</td><td>16.36 15.32</td><td>15.02</td><td>46.13 44.50</td><td>57.99</td><td>1354</td><td>32.79</td><td>17.70</td><td>85.16</td><td>66.63</td><td>43.75</td><td>93.09%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>56.53</td><td>1309</td><td>30.55</td><td>14.00</td><td>77.08</td><td>66.04</td><td>34.90</td><td>85.69%</td></tr><tr><td>Hybrid strategies</td><td></td><td></td><td>20.05</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td>51.68</td><td>16.56</td><td></td><td>49.74</td><td>60.57</td><td>1382</td><td>32.47</td><td>18.20</td><td>83.57</td><td>66.63</td><td>46.21</td><td>95.39%</td></tr><tr><td>SCOPE</td><td>51.30</td><td>17.20</td><td>21.56</td><td>49.14</td><td>60.22</td><td>1374</td><td>32.68</td><td>19.30</td><td>84.41</td><td>66.58</td><td>46.84</td><td>96.77%</td></tr><tr><td>IDPruner</td><td>51.55</td><td>17.40</td><td>21.55</td><td>49.23</td><td>60.40</td><td>1428</td><td>33.44</td><td>18.80</td><td>84.69</td><td>66.88</td><td>46.39</td><td>97.26%</td></tr><tr><td colspan="9">Retain 64 Tokens (88% Compression Ratio)</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Importance-based methods</td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>49.42</td><td>12.24</td><td>9.74</td><td>40.38</td><td>47.08</td><td>964</td><td>27.78</td><td>4.00</td><td>61.38</td><td>63.71</td><td>17.69</td><td>66.68%</td></tr><tr><td>VisionZip HiPrune</td><td>51.20</td><td>15.76</td><td>15.75</td><td>46.31</td><td>56.70</td><td>1289</td><td>31.40</td><td>16.40</td><td>78.18</td><td>66.44</td><td>42.75</td><td>89.14%</td></tr><tr><td>VisionSelector</td><td>51.20</td><td>15.68</td><td>16.18</td><td>46.13</td><td>57.30</td><td>1257</td><td>31.15</td><td>17.40</td><td>78.17</td><td>66.48</td><td>43.00</td><td>89.56%</td></tr><tr><td>Diversity-based methods</td><td>50.65</td><td>15.12</td><td>17.29</td><td>46.65</td><td>58.33</td><td>1310</td><td>32.58</td><td>17.00</td><td>79.94</td><td>65.84</td><td>42.98</td><td>90.49%</td></tr><tr><td>DivPrune</td><td>50.00</td><td>15.12</td><td>15.69</td><td>44.16</td><td>54.64</td><td>1271</td><td>31.68</td><td>15.70</td><td>84.18</td><td>66.44</td><td>40.83</td><td>87.82%</td></tr><tr><td>DART</td><td>51.55</td><td>15.20</td><td>12.35</td><td>40.29</td><td>53.26</td><td>1195</td><td>28.50</td><td>12.40</td><td>71.03</td><td>66.83</td><td>30.98</td><td>79.88%</td></tr><tr><td>Hybrid strategies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td>51.33</td><td>15.28</td><td>16.29</td><td>44.76</td><td>57.39</td><td>1315</td><td>31.02</td><td>17.00</td><td>81.20</td><td>66.73</td><td>43.45</td><td>89.77%</td></tr><tr><td>SCOPE</td><td>50.71</td><td>15.56</td><td>17.84</td><td>46.82</td><td>58.16</td><td>1320</td><td>32.18</td><td>17.70</td><td>83.07</td><td>66.63</td><td>44.33</td><td>91.90%</td></tr><tr><td>IDPruner</td><td>50.55</td><td>16.32</td><td>18.59</td><td>46.99</td><td>57.65</td><td>1329</td><td>31.49</td><td>17.30</td><td>83.83</td><td>66.68</td><td>44.73</td><td>92.34%</td></tr><tr><td colspan="9">Retain 32 Tokens (94% Compression Ratio)</td><td></td><td></td><td></td><td></td></tr><tr><td>Importance-based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">FastV</td><td></td><td>64.20</td><td></td><td></td></tr><tr><td></td><td>49.03</td><td>11.96</td><td>8.38</td><td>30.24</td><td>38.23</td><td>844</td><td>27.15</td><td>3.00</td><td>57.50</td><td></td><td>10.98</td><td>59.83%</td></tr><tr><td>VisionZip</td><td>51.17</td><td>14.04</td><td>13.10</td><td>39.35</td><td>50.43</td><td>1133</td><td>29.90</td><td>13.90</td><td>72.42</td><td>66.98</td><td>38.15</td><td>81.15%</td></tr><tr><td>HiPrune</td><td>51.42</td><td>14.32</td><td>13.22</td><td>39.95</td><td>51.98</td><td>1149</td><td>29.28</td><td>14.20</td><td>73.26</td><td>66.78</td><td>38.29</td><td>81.87%</td></tr><tr><td>VisionSelector</td><td>50.06</td><td>13.52</td><td>14.00</td><td>43.81</td><td>54.90</td><td>1194</td><td>32.07</td><td>14.80</td><td>75.04</td><td>65.49</td><td>38.94</td><td>84.12%</td></tr><tr><td colspan="9">Diversity-based methods</td><td>80.47</td><td>65.69</td><td>36.21</td><td>80.78%</td></tr><tr><td>DivPrune DART</td><td>50.74 51.10</td><td>14.20 13.32</td><td>12.25 10.79</td><td>37.89 33.51</td><td>51.72 47.08</td><td>1149</td><td>28.68</td><td>13.80 10.70</td></table>

Table 2: Comparison results on comprehensive Image-Language benchmarks on LLaVA-1.5-7B.

VideoMME (Fu et al., 2025), and SEED-Bench (Li et al., 2024b). To ensure fair comparison and reproducibility, we utilize the LMMs-Eval framework (Zhang et al., 2024b), strictly following the default settings and metrics for each task.

Comparison methods. We compare IDPruner with representative state-of-the-art approaches across different paradigms, including importancebased methods like FastV (Chen et al., 2024a), VisionZip (Yang et al., 2025b), HiPrune (Liu et al., 2025), and VisionSelector (Zhu et al., 2025a), diversity-based methods like DivPrune (Alvar et al., 2025) and DART (Wen et al., 2025), as well as hybrid strategies that combine multiple criteria, such as VisPruner (Zhang et al., 2024c), and SCOPE (Deng et al., 2025).

Implementation Details. Unless otherwise specified, the hyperparameter $\lambda$ of IDPruner, which balances importance and diversity, is set to 0.5.

# 5.2 Main Results

Results on Qwen2.5-VL-7B-Instruct. We evaluate our method on Qwen2.5-VL-7B-Instruct under $2 5 \%$ and $10 \%$ token retention settings. As shown in Table 1, IDPruner achieves state-of-the-art average scores of $9 5 . 1 8 \%$ and $8 6 . 4 7 \%$ , respectively. Compared to existing strategies, our method achieves a better balance between keeping fine details and maintaining global context. Specifically, for tasks requiring fine details, such as OCRBench, our method ranks among the top two, while also maintaining global information to surpass VisionSelector on hallucination benchmarks, including POPE. Consequently, on benchmarks such as MME and AI2D, which require both overall understanding and detailed capture, IDPruner demonstrates a clear

lead over other methods.

Results on LLaVA-1.5-7B. We extend our experiments to the LLaVA-1.5-7B model, which operates with a fixed resolution of 576 visual tokens per image. Accordingly, we evaluate performance under three distinct retention settings: 128, 64, and the extreme 32 tokens. As shown in Table 2, IDPruner consistently achieves state-of-the-art results across all pruning ratios. While VisionSelector is surpassed by the hybrid method SCOPE on this architecture, IDPruner maintains its lead, achieving an average score of $8 7 . 4 3 \%$ even with only 32 tokens, demonstrating its robustness across diverse architectures.

In summary, our method exhibits remarkable performance consistency across a diverse range of architectures. Notably, strong baselines exhibit architecture-specific vulnerabilities; for instance, VisionSelector underperforms on LLaVA1.5, whereas SCOPE loses competitiveness on the advanced LLaVA-OneVision-1.5, as detailed in Appendix A. In contrast, IDPruner maintains exceptional robustness. It consistently achieves state-ofthe-art results across all evaluated models, validating the universality of our framework in harmonizing token importance and diversity.

Table 3: Comparison results on Video-Language benchmarks on Qwen2.5-VL-7B-Instruct with $25 \%$ token retention.   

<table><tr><td>Method</td><td>Vinoground Group</td><td>VideoMME Perception</td><td>SEED-Bench All</td><td>Avg</td></tr><tr><td>Baseline</td><td>20.20</td><td>61.33</td><td>74.12</td><td>100.0%</td></tr><tr><td>FastV</td><td>12.80</td><td>59.44</td><td>69.22</td><td>84.56%</td></tr><tr><td>VisionZip</td><td>12.80</td><td>59.67</td><td>72.11</td><td>85.98%</td></tr><tr><td>HiPrune</td><td>11.80</td><td>59.93</td><td>71.97</td><td>84.41%</td></tr><tr><td>VisionSelector</td><td>10.80</td><td>59.19</td><td>70.75</td><td>81.81%</td></tr><tr><td>DivPrune</td><td>14.00</td><td>58.00</td><td>72.11</td><td>87.06%</td></tr><tr><td>DART</td><td>12.60</td><td>59.52</td><td>71.27</td><td>85.19%</td></tr><tr><td>VisPruner</td><td>11.40</td><td>59.44</td><td>71.89</td><td>83.45%</td></tr><tr><td>SCOPE</td><td>12.80</td><td>60.00</td><td>72.63</td><td>86.40%</td></tr><tr><td>IDPruner</td><td>13.40</td><td>59.48</td><td>72.68</td><td>87.13%</td></tr></table>

# 5.3 IDPruner for Video Understanding

Beyond static image benchmarks, we extend IDPruner to video understanding tasks, evaluating its performance on Vinoground, VideoMME, and SEED-Bench at a $7 5 \%$ pruning ratio. As shown in Table 3, purely importance-based methods exhibit significant performance degradation. This is primarily due to their inability to handle the high temporal redundancy in videos. In contrast, diversitybased methods maintain strong performance with an average score of $8 7 . 0 6 \%$ . Notably, IDPruner

achieves the best average performance of $8 7 . 1 3 \%$ by jointly considering both the preservation of important details and the reduction of temporal redundancy.   
Table 4: Efficiency analysis on Vinoground on Qwen2.5-VL-7B-Instruct with $25 \%$ token retention. FA: FlashAttention compatibility.   

<table><tr><td>Method</td><td>FA</td><td>Prefill(ms)</td><td>E2E Latency(ms)</td></tr><tr><td>VisPruner</td><td>×</td><td>1459.95</td><td>1600.81</td></tr><tr><td>SCOPE</td><td>×</td><td>1677.81</td><td>1818.40</td></tr><tr><td>IDPruner</td><td>✓</td><td>1337.76</td><td>1478.32</td></tr></table>

# 5.4 Efficiency and Practicality

We compare the efficiency of hybrid pruning strategies on Qwen2.5-VL-7B using the Vinoground benchmark at a $7 5 \%$ pruning ratio. As shown in Table 4, IDPruner achieves the best efficiency among hybrid strategies, due to its lightweight diversity calculation and being attention-map-free. This design ensures full compatibility with FlashAttention, yielding the lowest prefill time of $1 3 3 7 . 7 6 \mathrm { m s }$ and an end-to-end latency of 1478.32 ms.

# 6 Conclusion

Recent progress in visual token pruning shows that hybrid strategies are surpassing methods that rely only on importance or diversity, becoming the new standard in this field. However, there is a lack of systematic analysis on how to effectively harmonize these two objectives. In this study, we provide a framework to analyze this trade-off and demonstrate that the Maximal Marginal Relevance (MMR) mechanism is an effective strategy to achieve an optimal balance. Based on this insight, we propose IDPruner, a method that explicitly balances token importance and semantic redundancy. Extensive evaluations show that our method achieves state-of-the-art performance and remains robust across different model architectures. We believe this work offers a solid foundation for systematically balancing importance and diversity, enabling more efficient MLLMs.

# References

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. 2025. Divprune: Diversitybased visual token pruning for large multimodal models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9392–9401.

Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. 2025. Hired: Attention-guided token dropping for efficient inference of high-resolution visionlanguage models. In AAAI Conference on Artificial Intelligence.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Jaime G. Carbonell and Jade Goldstein-Stewart. 1998. The use of mmr, diversity-based reranking for reordering documents and producing summaries. ACM SIGIR Forum, 51:209 – 210.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth $1 / 2$ tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and 1 others. 2024b. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS).

Jinhong Deng, Wen Li, Joey Tianyi Zhou, and Yang He. 2025. Scope: Saliency-coverage oriented token pruning for efficient multimodel llms. ArXiv, abs/2510.24214.

Mohamed Achraf Dhouib, Davide Buscaldi, Sonia Vanier, and Aymen Shabou. 2025. Pact: Pruning and clustering-based token reduction for faster visual language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14582–14592.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108– 24118.

Jiayi Han, Liang Du, Yiwen Wu, Guanming Liang, Xiangguo Zhou, Weibo Zheng, Donghong Han, and

Zixun Sun. 2025. Adav: Adaptive text-visual redirection for vision-language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4985–4997.

Brian Hopkins and J. Gordon Skellam. 1954. A new method for determining the type of distribution of plant individuals. Annals of Botany, 18:213–227.

Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. ArXiv, abs/1603.07396.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. ArXiv, abs/2408.03326.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. 2024b. Seedbench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308.

Yangfu Li, Hongjian Zhan, Tianyi Chen, Qi Liu, and Yue Lu. 2025. Why $1 + 1 < 1$ in visual token pruning: Beyond naive integration via multi-objective balanced covering. ArXiv, abs/2505.10118.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024a. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. In NeurIPS.

Jizhihui Liu, Feiyi Du, Guangdao Zhu, Niu Lian, Jun Li, and Bin Chen. 2025. Hiprune: Training-free visual token pruning via hierarchical attention in visionlanguage models. arXiv preprint arXiv:2508.00553.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2023c. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, ChengLin Liu, Lianwen Jin, and Xiang Bai. 2024b. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12).

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS).

Odile Macchi. 1975. The coincidence approach to stochastic point processes. Advances in Applied Probability, 7(1):83–122.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244.

Minesh Mathew, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. 2020. Docvqa: A dataset for vqa on document images. corr abs/2007.00398 (2020). arXiv preprint arXiv:2007.00398.

Mauricio GC Resende, Rafael Martí, Micael Gallego, and Abraham Duarte. 2010. Grasp and path relinking for the max–min diversity problem. Computers & Operations Research, 37(3):498–508.

Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. 2025. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. ArXiv, abs/2507.20198.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025. Stop looking for important tokens in multimodal language models: Duplication matters more. ArXiv, abs/2502.11494.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and 1 others. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, and 1 others. 2025a. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19803–19813.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025b. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19792–19802.

Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. 2025. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24972–24982.

Jianrui Zhang, Cai Mu, and Yong Jae Lee. 2024a. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. 2024b. Lmms-eval: Reality check on the evaluation of large multimodal models. Preprint, arXiv:2407.12772.

Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, and Ruixuan Li. 2025. Flowcut: Rethinking redundancy via information flow for efficient vision-language models. ArXiv, abs/2505.19536.

Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. 2024c. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. 2024d. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv e-prints, pages arXiv– 2412.

Zekun Wang, Minghua Ma, Zexin Wang, Rongchuan Mu, Liping Shan, Ming Liu, and Bing Qin. 2025. Effivlm-bench: A comprehensive benchmark for evaluating training-free acceleration in large visionlanguage models. Preprint, arXiv:2506.00479.

Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. 2025a. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. ArXiv, abs/2506.10967.

Weichen Zhang, Zhui Zhu, Ningbo Li, Kebin Liu, and Yunhao Liu. 2025b. Adaptinfer: Adaptive token pruning for vision-language model inference with dynamical text guidance. arXiv preprint arXiv:2508.06084.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1 others. 2024e. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417.

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. 2024. A survey on efficient inference for large language models. ArXiv, abs/2404.14294.

Jiaying Zhu, Yurui Zhu, Xin Lu, Wenrui Yan, Dong Li, Kunlin Liu, Xueyang Fu, and Zheng-Jun Zha. 2025a. Visionselector: End-to-end learnable visual token compression for efficient multimodal llms. Preprint, arXiv:2510.16598.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, and 1 others. 2025b. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

Xin Zou, Di Lu, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Xu Zheng, Linfeng Zhang, and Xuming Hu. 2025. Don’t just chase "highlighted tokens" in mllms: Revisiting visual holistic context retention. ArXiv, abs/2510.02912.

# A Additional Experimental Results

# A.1 Results on Qwen2.5-VL-3B-Instruct

To evaluate the scalability of our method on smaller language models, we conduct experiments on Qwen2.5-VL-3B-Instruct. As shown in Table 5, IDPruner consistently outperforms competitive baselines at both $2 5 \%$ and $10 \%$ token retention ratios. Notably, when retaining $2 5 \%$ of the tokens, our method achieves an average score of $9 4 . 4 2 \%$ , effectively matching the unpruned baseline. Even under the aggressive $10 \%$ retention setting, IDPruner maintains a high average performance of $8 5 . 7 1 \%$ , outperforming the second-best method (VisionSelector) by $1 . 2 9 \%$ .

# A.2 Results onLLaVA-OneVision-1.5-8B-Instruct

We further assess the cross-architecture generalization on LLaVA-OneVision-1.5-8B-Instruct, which integrates advanced visual encoding strategies. As shown in Table 6, IDPruner achieves the best results among existing state-of-the-art methods. Under the $25 \%$ retention setting, our method achieves an average score of $9 2 . 0 0 \%$ , outperforming the strongest baseline, VisionSelector, by $0 . 3 7 \%$ . In the more challenging $10 \%$ retention scenario, IDPruner exhibits strong robustness, achieving an average score of $8 1 . 5 5 \%$ . It significantly outperforms purely importance-based methods such as VisionZip and HiPrune, which suffer from severe degradation due to the loss of global context. Additionally, it surpasses the competitive VisionSelector by $1 . 4 4 \%$ , confirming that harmonizing importance and diversity is particularly effective for advanced architectures.

# B Ablation Study: Integration Strategies and Hyperparameters

We investigate the efficacy of different integration strategies and the impact of the hyperparameter $\lambda$ , which controls the trade-off between token importance and diversity. Using VisionSelector as the fixed base importance estimator, we compare our IDPruner (MMR) mechanism against two representative baselines: a determinantal point process based method (DPP) and a Naive Hybrid strategy that combines importance filtering with Farthest Point Sampling (FPS). Table 7 summarizes the results on Qwen2.5-VL-7B-Instruct at a $25 \%$ token retention ratio.

Superiority of MMR Mechanism. The integration strategy plays a pivotal role in model performance. As evidenced in Table 7, IDPruner consistently outperforms the Naive Hybrid strategy across comparable $\lambda$ settings and also surpasses the DPP-based baseline. The Naive Hybrid approach typically prioritizes tokens with the highest importance scores before applying Farthest Point Sampling (FPS) to enhance diversity. However, this two-stage paradigm fails to address the inherent redundancy among high-importance tokens, resulting in a selected subset that lacks sufficient diversity. In contrast, IDPruner employs a unified scoring mechanism that simultaneously manages importance and redundancy. By dynamically penalizing semantically repetitive tokens during selection, our method achieves a more effective balance, thereby demonstrating superior robustness over heuristic hybrid strategies.

Hyperparameter Selection. The hyperparameter $\lambda$ controls the balance between token importance and semantic diversity. For IDPruner, the performance follows an inverted U-shape pattern, peaking at $\lambda = 0 . 5$ with an average performance of $9 5 . 5 6 \%$ . This confirms that setting $\lambda = 0 . 5$ successfully strikes an optimal balance between token importance and semantic diversity, enabling IDPruner to leverage both properties for maximum performance.

# C Limitations

Despite the promising results achieved by IDPruner, we acknowledge certain limitations in this study. First, constrained by computational resources, we have not yet evaluated our method on long-context video understanding benchmarks. This restricts the comprehensive verification of our method’s effectiveness in scenarios involving extremely long temporal sequences, thereby limiting the scope of applicable scenarios. Second, due to time constraints, we did not conduct a fine-grained measurement or exhaustive search for the hyperparameter $\lambda$ . While the current settings demonstrate strong robustness, a more thorough optimization could potentially yield further performance improvements.

# D Visualization

To intuitively understand how IDPruner harmonizes importance and diversity compared to existing approaches, we visualize the spatial distribution of retained visual tokens across multiple samples. Figure 4 presents a comparison of token selection masks under a $2 5 \%$ retention ratio. As consistently observed across diverse scenes, DivPrune tends to produce a uniform distribution, often overlooking semantic details. VisionSelector overly concentrates on foreground objects at the expense of background information coverage. In contrast, IDPruner successfully balances both, capturing salient features while maintaining essential background context necessary for global reasoning.

Table 5: Comparison results with different methods on Qwen2.5-VL-3B-Instruct.   

<table><tr><td>Method</td><td>AI2D EM</td><td>ChartQA Relaxed</td><td>DocVQA Anls</td><td>MMBCN Score</td><td>MMB Score</td><td>MME Score</td><td>MMStar Avg</td><td>OCRBench Acc</td><td>POPE Acc</td><td>SQA EM</td><td>VQAText EM</td><td>Avg</td></tr><tr><td>Baseline</td><td>79.11</td><td>83.56</td><td>92.48</td><td>73.28</td><td>77.32</td><td>1517</td><td>56.05</td><td>80.10</td><td>87.41</td><td>80.81</td><td>78.79</td><td>100.0%</td></tr><tr><td colspan="10">Retain 25% Tokens (75% Compression Ratio)</td><td></td><td></td><td></td></tr><tr><td colspan="10">Importance-based methods</td></tr><tr><td></td><td>72.70</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.51</td><td>86.02%</td></tr><tr><td>FastV VisionZip</td><td>74.19</td><td>70.04 71.32</td><td>75.98 70.11</td><td>63.40 67.35</td><td>66.92 71.22</td><td>1437 1452</td><td>47.39</td><td>36.60 42.50</td><td>86.42 85.51</td><td>79.33 81.36</td><td>68.12</td><td>87.34%</td></tr><tr><td>HiPrune</td><td>73.83</td><td>72.76</td><td>72.10</td><td>67.27</td><td>72.34</td><td>1449</td><td>49.37 48.93</td><td>41.30</td><td>85.86</td><td>80.91</td><td>69.27</td><td>87.67%</td></tr><tr><td>VisionSelector</td><td>75.19</td><td>73.72</td><td>90.24</td><td>68.81</td><td>72.59</td><td>1521</td><td>49.97</td><td>61.80</td><td>85.36</td><td>80.37</td><td>76.86</td><td>93.62%</td></tr><tr><td colspan="10">Diversity-based methods</td><td></td><td></td></tr><tr><td>DivPrune</td><td>73.06</td><td>62.96</td><td>78.46</td><td>67.10</td><td>71.82</td><td>1459</td><td>48.38</td><td>51.40</td><td>86.81</td><td>80.22</td><td>68.91</td><td>88.15%</td></tr><tr><td>DART</td><td>71.08</td><td>65.20</td><td>79.72</td><td>65.38</td><td>71.05</td><td>1428</td><td>48.78</td><td>41.80</td><td>80.97</td><td>80.91</td><td>68.25</td><td>86.17%</td></tr><tr><td colspan="10">Hybrid strategies</td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td>74.29</td><td>68.20</td><td>72.52</td><td>67.35</td><td>70.88</td><td>1458</td><td>49.74</td><td>44.80</td><td>86.59</td><td>81.46</td><td>69.62</td><td>87.87%</td></tr><tr><td>SCOPE</td><td>75.84</td><td>74.00</td><td>82.40</td><td>68.81</td><td>72.94</td><td>1471</td><td>50.35</td><td>56.00</td><td>86.62</td><td>80.96</td><td>74.04</td><td>91.98%</td></tr><tr><td>IDPruner</td><td>75.94</td><td>75.84</td><td>90.00</td><td>69.42</td><td>73.80</td><td>1505</td><td>49.49</td><td>64.90</td><td>86.26</td><td>80.42</td><td>76.90</td><td>94.42%</td></tr><tr><td colspan="10">Retain 10% Tokens (90% Compression Ratio)</td></tr><tr><td></td><td>Importance-based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>65.87</td><td>29.72</td><td>36.89</td><td>48.37</td><td>51.98</td><td>1257</td><td>37.28</td><td>13.90</td><td>79.50</td><td>77.05</td><td>57.75</td><td>65.30%</td></tr><tr><td>VisionZip</td><td>67.65</td><td>51.60</td><td>37.88</td><td>59.62</td><td>63.06</td><td>1338</td><td>42.82</td><td>21.40</td><td>81.14</td><td>80.47</td><td>51.56</td><td>72.75%</td></tr><tr><td>HiPrune</td><td>67.75</td><td>53.20</td><td>41.15</td><td>59.45</td><td>63.14</td><td>1326</td><td>41.08</td><td>20.30</td><td>80.90</td><td>80.96</td><td>53.31</td><td>73.00%</td></tr><tr><td>VisionSelector</td><td>70.50</td><td>65.92</td><td>79.94</td><td>59.97</td><td>64.69</td><td>1374</td><td>42.86</td><td>45.20</td><td>82.66</td><td>80.61</td><td>71.57</td><td>84.42%</td></tr><tr><td>Diversity-based methods DivPrune</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DART</td><td>67.71</td><td>43.12</td><td>58.03</td><td>61.25</td><td>65.12</td><td>1389</td><td>40.43</td><td>27.90</td><td>82.24</td><td>79.18</td><td>56.87</td><td>75.50%</td></tr><tr><td></td><td>67.49</td><td>47.56</td><td>60.23</td><td>57.99</td><td>63.83</td><td>1299</td><td>42.18</td><td>23.40</td><td>74.20</td><td>78.63</td><td>58.02</td><td>74.09%</td></tr><tr><td colspan="10">Hybrid strategies</td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td>67.75</td><td>47.92</td><td>48.65</td><td>59.28</td><td>63.32</td><td>1305</td><td>41.51</td><td>22.50</td><td>78.74</td><td>79.77</td><td>54.95</td><td>73.19%</td></tr><tr><td>SCOPE IDPruner</td><td>69.75 71.79</td><td>56.24</td><td>55.01</td><td>64.26</td><td>67.18</td><td>1390 1438</td><td>44.35</td><td>30.80 45.50</td><td>83.34 84.51</td><td>80.47 80.57</td><td>62.58</td><td>79.37%</td></tr><tr><td></td><td></td><td>63.32</td><td>79.38</td><td>63.57</td><td>68.21</td><td></td><td>44.05</td><td></td><td></td><td></td><td>70.02</td><td>85.71%</td></tr></table>

Table 6: Comparison results with different methods on LLaVA-OneVision-1.5-8B-Instruct.   

<table><tr><td>Method</td><td>AI2D EM</td><td>ChartQA Relaxed</td><td>DocVQA Anls</td><td>MMBCN Score</td><td>MMB Score</td><td>MME Score</td><td>MMStar Avg</td><td>OCRBench Acc</td><td>POPE Acc</td><td>SQA EM</td><td>VQAText EM</td><td>Avg</td></tr><tr><td>Baseline</td><td>84.20</td><td>86.40</td><td>97.87</td><td>78.52</td><td>85.31</td><td>1594</td><td>68.25</td><td>80.90</td><td>88.91</td><td>98.76</td><td>79.65</td><td>100.0%</td></tr><tr><td colspan="10">Retain 25% Tokens (75% Compression Ratio)</td><td></td><td></td><td></td></tr><tr><td colspan="10">Importance-based methods</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>50.21</td><td>33.60</td><td></td><td>89.19</td><td>52.54</td><td>75.05%</td></tr><tr><td>FastV VisionZip</td><td>74.48 69.95</td><td>39.16 27.48</td><td>56.53 23.69</td><td>70.36 65.89</td><td>77.66 75.26</td><td>1440 1419</td><td>47.01</td><td>20.80</td><td>81.53 80.46</td><td>84.78</td><td>36.15 65.14%</td></tr><tr><td>HiPrune</td><td>70.27</td><td>19.76</td><td>21.01</td><td>63.57</td><td>72.51</td><td>1339</td><td></td><td></td><td>81.36</td><td>30.21</td><td>61.59%</td></tr><tr><td>VisionSelector</td><td>77.85</td><td>76.32</td><td>94.08</td><td>74.74</td><td>80.07</td><td>1569</td><td>46.58 57.42</td><td>19.60 55.50</td><td>77.29 86.73</td><td>94.70</td><td>77.53 91.63%</td></tr><tr><td colspan="10">Diversity-based methods</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune DART</td><td>78.85 65.52 77.82 69.00</td><td>81.53 84.77</td><td>73.20 72.25</td><td>80.67 79.81</td><td>1533 1564</td><td>56.68 56.80</td><td>51.50 46.90</td><td>88.14 84.77</td><td>91.42 93.11</td><td>75.00 73.89</td><td>88.12% 87.83%</td></tr><tr><td colspan="10">Hybrid strategies</td></tr><tr><td></td><td>75.45</td><td>45.88</td><td>61.00</td><td>69.93</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisPruner SCOPE</td><td>54.84</td><td>75.68</td><td>70.27</td><td>76.55 78.18</td><td>1468 1505</td><td>50.68 54.52</td><td>36.70 46.60</td><td>85.82 87.02</td><td>89.24 90.18</td><td>69.01 72.77</td><td>79.01% 84.30%</td></tr><tr><td colspan="10">78.21 IDPruner 79.18</td></tr><tr><td></td><td>74.48</td><td></td><td>91.82</td><td>73.71</td><td>81.27 1588</td><td>57.97</td><td>57.80</td><td>88.13</td><td>95.14</td><td>77.50</td><td>92.00%</td></tr><tr><td colspan="10">Retain 10% Tokens (90% Compression Ratio)</td></tr><tr><td>Importance-based methods FastV</td><td>70.95</td><td>22.72</td><td>29.77</td><td>62.46</td><td>69.07 1303</td><td>43.66</td><td>16.70</td><td>73.83</td><td>82.35</td><td>38.71</td><td>62.08%</td></tr><tr><td>VisionZip</td><td>69.14 19.64</td><td>13.20</td><td>58.16</td><td>65.38</td><td>1259</td><td>42.12</td><td>12.00</td><td>76.59</td><td>78.53</td><td>21.28</td><td>56.09%</td></tr><tr><td>HiPrune</td><td>68.39 16.24</td><td>12.86</td><td>56.01</td><td>63.83</td><td>1208</td><td>42.13</td><td>9.10</td><td>73.72</td><td>77.29</td><td>18.97</td><td>53.92%</td></tr><tr><td>VisionSelector</td><td>73.38 61.00</td><td>71.31</td><td>68.04</td><td>74.91</td><td>1466</td><td>50.74</td><td>34.60</td><td>82.62</td><td>88.05</td><td>67.67</td><td>80.11%</td></tr><tr><td>Diversity-based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune</td><td>73.74 37.68</td><td>56.57</td><td>67.18</td><td>73.80</td><td>1477</td><td>46.31</td><td>30.10</td><td>85.33</td><td>86.51</td><td>65.98</td><td>75.02%</td></tr><tr><td>DART</td><td>72.80 40.48</td><td>52.77</td><td>63.14</td><td>70.19</td><td>1378</td><td>47.42</td><td>27.30</td><td>75.79</td><td>84.88</td><td>61.58</td><td>71.65%</td></tr><tr><td>Hybrid strategies VisPruner</td><td>69.82</td><td></td><td>35.18</td><td></td><td></td><td>43.24</td><td>18.20</td><td>78.33</td><td>84.23</td><td>54.53</td><td>65.46%</td></tr><tr><td colspan="10">SCOPE</td></tr><tr><td></td><td>25.20</td><td>33.04</td><td>61.77 66.07</td><td>67.70 73.54</td><td>1375 1471</td><td>48.25</td><td>25.70</td><td>84.11</td><td>85.97</td><td>61.02</td><td>72.35%</td></tr><tr><td>IDPruner</td><td>72.05 74.45</td><td>57.92</td><td>47.65 70.43</td><td>69.67 75.69</td><td>1511</td><td>50.02</td><td>38.70</td><td>85.02</td><td>88.70</td><td>72.37</td><td>81.55%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>Method</td><td>AI2D EM</td><td>ChartQA Relaxed</td><td>DocVQA Anls</td><td>MME Score</td><td>OCRBench Acc</td><td>POPE Acc</td><td>SQA EM</td><td>VQAText EM</td><td>Avg</td></tr><tr><td>Baseline</td><td>82.48</td><td>83.68</td><td>94.90</td><td>1701</td><td>85.30</td><td>87.80</td><td>88.45</td><td>82.74</td><td>100.00</td></tr><tr><td colspan="10">Strategy 1: DPP + VisionSelector</td></tr><tr><td>DPP+VisionSelector</td><td>79.70</td><td>73.36</td><td>93.02</td><td>1691</td><td>73.00</td><td>86.96</td><td>84.73</td><td>80.69</td><td>94.95</td></tr><tr><td colspan="10">Strategy 2: IDPruner (MMR Mechanism)</td></tr><tr><td>IDPruner (λ = 0.1)</td><td>78.08</td><td>67.36</td><td>87.94</td><td>1680</td><td>62.00</td><td>86.59</td><td>83.59</td><td>77.65</td><td>90.78</td></tr><tr><td>IDPruner (λ = 0.3)</td><td>80.05</td><td>73.88</td><td>91.26</td><td>1671</td><td>70.50</td><td>86.91</td><td>84.09</td><td>79.69</td><td>94.10</td></tr><tr><td>IDPruner (λ = 0.5)</td><td>80.51</td><td>74.32</td><td>93.16</td><td>1695</td><td>74.00</td><td>87.06</td><td>85.52</td><td>80.83</td><td>95.56</td></tr><tr><td>IDPruner (λ = 0.7)</td><td>80.25</td><td>74.12</td><td>93.35</td><td>1710</td><td>74.00</td><td>87.07</td><td>85.13</td><td>80.75</td><td>95.56</td></tr><tr><td>IDPruner (λ = 0.9)</td><td>79.66</td><td>72.72</td><td>93.29</td><td>1705</td><td>72.80</td><td>86.96</td><td>84.88</td><td>80.61</td><td>94.97</td></tr><tr><td colspan="10">Strategy 3: Naive Hybrid Selector</td></tr><tr><td>Hybrid (λ = 0.1)</td><td>78.79</td><td>66.52</td><td>86.90</td><td>1700</td><td>59.90</td><td>86.09</td><td>83.64</td><td>78.12</td><td>90.47</td></tr><tr><td>Hybrid (λ = 0.3)</td><td>79.50</td><td>72.72</td><td>90.71</td><td>1704</td><td>62.60</td><td>86.56</td><td>83.34</td><td>79.91</td><td>92.73</td></tr><tr><td>Hybrid (λ = 0.5)</td><td>79.95</td><td>74.36</td><td>92.20</td><td>1702</td><td>64.30</td><td>87.17</td><td>84.33</td><td>80.82</td><td>93.84</td></tr><tr><td>Hybrid (λ = 0.7)</td><td>79.18</td><td>75.08</td><td>93.10</td><td>1680</td><td>66.40</td><td>86.63</td><td>84.83</td><td>80.94</td><td>94.10</td></tr><tr><td>Hybrid (λ = 0.9)</td><td>79.31</td><td>73.84</td><td>93.42</td><td>1681</td><td>71.90</td><td>86.66</td><td>84.58</td><td>80.45</td><td>94.69</td></tr></table>

Table 7: Ablation study of integration strategies on Qwen2.5-VL-7B-Instruct with $2 5 \%$ token retention. We use VisionSelector as the base importance scorer. $\lambda$ controls the trade-off between importance and diversity.

# E Empirical Verification of Non-Negative Similarity

A potential concern regarding the MMR mechanism is the behavior of the redundancy penalty term, $( 1 - \lambda ) \cdot \mathrm { S i m } ( v _ { i } , v _ { j } )$ . If the cosine similarity $\mathrm { S i m } ( v _ { i } , v _ { j } )$ were to yield negative values (implying an angle $\theta > 9 0 ^ { \circ }$ between feature vectors), the intended penalty would transform into a reward.

To address this validity concern, we empirically analyzed the geometric properties of the visual token space. We randomly selected 100 images from the MMBench dataset and computed the pairwise angles between all visual tokens extracted from the Qwen2.5-VL-7B-Instruct model.

As illustrated in Figure 5, the distribution of pairwise angles exhibits a distinct pattern. The distribution is overwhelmingly concentrated within the range of $[ 0 ^ { \circ } , 8 5 ^ { \circ } ]$ , with a peak density at approximately $7 4 ^ { \circ }$ . Crucially, there is zero probability mass beyond the $9 0 ^ { \circ }$ threshold (indicated by the red dashed line).

Since $\mathrm { S i m } ( v _ { i } , v _ { j } ) = \cos ( \theta _ { i j } )$ and $\cos ( \theta ) \geq 0$ for all $\theta \in [ 0 ^ { \circ } , 9 0 ^ { \circ } ]$ , this empirical evidence confirms that all similarity scores in our framework are strictly non-negative. Consequently, the term $( 1 - \lambda ) \cdot \mathrm { S i m } ( v _ { i } , v _ { j } )$ consistently functions as a redundancy penalty, validating the theoretical soundness of our IDPruner formulation.

# F Statement on the Use of AI Assistants

In accordance with the ACL submission policies, we hereby declare the use of AI assistants in the preparation of this manuscript. We utilized AI assistants for writing refinement, including grammar correction, vocabulary enhancement, and proofreading to improve readability. We emphasize that all scientific claims, experimental designs, core concepts, and logical arguments presented in this work are the original contributions of the authors. All AI-generated content was meticulously reviewed and verified by the authors to ensure accuracy and adherence to academic standards; the authors assume full responsibility for the content of this paper.

![](images/46b07e68248a24e8eae101e9b68c077091123dca608f243a56f7b1b2bf416a79.jpg)  
Figure 4: Visualization of retained visual tokens across different samples from MMBench. Columns from left to right: Original Image, DivPrune, VisionSelector, and IDPruner. DivPrune maintains global coverage but often neglects the semantic subject. VisionSelector clusters heavily on salient objects, resulting in redundancy and background loss. IDPruner achieves a superior balance, preserving intricate details of the subject while maintaining essential background context for global reasoning.

![](images/621194f42be4b5931bf79176609f68aade85e1e80a1028780b5dc4112b60b32b.jpg)  
Figure 5: Distribution of pairwise angles between visual tokens. We calculated the angles for all token pairs across 100 images from MMBench using Qwen2.5-VL7B. The distribution is entirely concentrated within the acute angle range $( < 9 0 ^ { \circ }$ ), peaking around $7 4 ^ { \circ }$ . The absence of obtuse angles $( > 9 0 ^ { \circ }$ , right of the red dashed line) guarantees that the cosine similarity metric remains strictly non-negative.