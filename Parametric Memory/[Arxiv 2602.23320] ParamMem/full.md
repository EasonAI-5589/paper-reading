# ParamMem: Augmenting Language Agents with Parametric Reflective Memory

Tianjun Yao 1 Yongqiang Chen 1 2 Yujia Zheng 2 Pan Li 3 Zhiqiang Shen 1 Kun Zhang 1 2

# Abstract

Self-reflection enables language agents to iteratively refine solutions, yet often produces repetitive outputs that limit reasoning performance. Recent studies have attempted to address this limitation through various approaches, among which increasing reflective diversity has shown promise. Our empirical analysis reveals a strong positive correlation between reflective diversity and task success, further motivating the need for diverse reflection signals. We introduce ParamMem, a parametric memory module that encodes cross-sample reflection patterns into model parameters, enabling diverse reflection generation through temperature-controlled sampling. Building on this module, we propose ParamAgent, a reflection-based agent framework that integrates parametric memory with episodic and cross-sample memory. Extensive experiments on code generation, mathematical reasoning, and multi-hop question answering demonstrate consistent improvements over state-of-the-art baselines. Further analysis reveals that ParamMem is sample-efficient, enables weak-to-strong transfer across model scales, and supports self-improvement without reliance on stronger external model, highlighting the potential of ParamMem as a effective component for enhancing language agents. Code and data can be found at: https://github. com/tianyao-aka/ParamAgent.

# 1. Introduction

Large language models (LLMs) (Brown et al., 2020; Chowdhery et al., 2023; Touvron et al., 2023) have exhibited remarkable progress in complex reasoning tasks. A key insight driving recent advances is test-time scaling, i.e., allocating additional computation during inference to improve reasoning (Wei et al., 2022; Wang et al., 2022; Madaan et al., 2023; Yao et al., 2023a; Shinn et al., 2023; Snell et al., 2024).

![](images/28674cc83de102a4f296b124d2b1ee95854792d39ccc6203c5e3f7d3485130ac.jpg)  
Figure 1. Correlation between reflective diversity (measured by average pairwise cosine distance) and task performance across five datasets using LLaMA-3.1-8B under Reflexion, DoT, and DoTbank.

Among these approaches, reflection-based frameworks have proven particularly effective, where agents verbally reflect on task feedback and accumulate self-reflections in episodic memory to guide subsequent trials (Shinn et al., 2023; Madaan et al., 2023; Yao et al., 2023a). Such reflection mechanisms have been successfully applied to programming (Shinn et al., 2023), mathematical reasoning (Lightman et al., 2023), decision-making (Yao et al., 2023b), and multi-agent systems (Wu et al., 2023; Hong et al., 2024).

However, recent studies have identified limitations in selfreflection, showing that it often produces repetitive and inaccurate outputs (Huang et al., 2023; Yao et al., 2023c; Lingam et al., 2025; Ozer et al., 2025), which hinders the effectiveness of self-reflection. Among these works, Lingam et al. (2025) attempts to increase reflective diversity through prompt-level modifications (DoT) and by incorporating cross-sample trajectories (DoT-bank), demonstrating preliminary success. In this work, we first explore how reflective diversity relates to final performance. Specifically, we conduct experiments on five datasets using LLaMA-3.1-8B, computing the pairwise cosine distance across multi-round reflection logs for each sample under Reflexion, DoT, and DoT-bank, and averaging these distances. As illustrated in Figure 1, the average Pearson correlation coefficient across the five datasets is 0.76, indicating a strong positive relationship between reflective diversity and task performance. While prompt-based approaches to diversifying reflections sometimes yield limited improvements, incorporating reasoning trajectories from similar samples often enhances diversity and the final performance.

Despite its effectiveness, the retrieval-based approach like DoT-bank relies on embedding similarity to retrieve crosssample trajectories, which has limited capacity for capturing compositional patterns (Nguyen & Yates, 2023; Weller et al., 2025); moreover, learned embeddings are prone to collapse into low-rank subspaces, reducing retrieval diversity (Guo et al., 2023). This naturally raises our question:

How can we further expand reflective diversity to achieve stronger reasoning performance?

To address this challenge, we introduce $\textcircled { < }$ ParamMem, a new form of reflective memory that provides diversity through a fundamentally different mechanism. Unlike approaches that rely on prompt variations and retrieval-based methods that explicitly utilize similar samples, ParamMem operates by fine-tuning a lightweight parametric module on an auxiliary reflection dataset $\mathcal { D } = \{ ( x _ { i } , r _ { i } ^ { g } ) \} _ { i = 1 } ^ { n }$ . Through training, the module encodes cross-sample patterns into its parameters; at inference time, it generates reflections by generalizing from these learned patterns rather than retrieving existing examples.

Contribution. We propose a new paradigm for enhancing reflective diversity to improve reasoning in language agents. Central to our approach is ParamMem, a parametric memory module that internalizes cross-sample reflection patterns. ParamMem targets diversity, is lightweight, and can seamlessly integrate into existing reflection-based frameworks. Building upon ParamMem, we propose ParamAgent and its enhanced variant ParamAgent-plus, which unify parametric reflective memory with episodic and crosssample memory within a coherent framework. Through extensive empirical evaluation, our method exhibits several notable advantages: $\textcircled{1}$ Substantial performance gains. Our approach achieves consistent improvements across programming, mathematical reasoning, and multi-hop question answering, outperforming state-of-the-art baselines significantly. $\textcircled{2}$ Sample efficiency. ParamMem requires only ${ \sim } 5 0 0$ training samples to deliver strong performance, highlighting its effectiveness in low-data regimes. This makes ParamMem practical for deployment in resourceconstrained settings. $\textcircled{3}$ Self-improvement. Even without relying on stronger external models, ParamMem can enhance reflective diversity using data generated by the base LLM itself, leading to improved performance for ParamAgent and ParamAgent-plus. This highlights the potential of ParamMem as a self-contained, annotationfree module for continual agent improvement. $\textcircled{4}$ Weak-tostrong transfer. Even when ParamMem is trained using a weaker LLM, its generated reflective signals still enhance ParamAgent built on stronger LLMs. This indicates that even a weaker model can effectively increase the reflective diversity of a stronger model.

# 2. Preliminaries

We consider a pretrained language model $p _ { \theta }$ that generates output $y$ given input $x$ . We use $r _ { 1 } , \ldots , r _ { k }$ to denote self-reflections accumulated up to $k$ iterations, and use $r _ { k } ^ { g }$ to denote the model-based outputs (e.g., reflections in programming and math tasks) sampled from the parametric memory module $\mathcal { M } _ { g }$ .

Reflexion Framework. Reflexion (Shinn et al., 2023) enables iterative reasoning through four components: (1) an actor $p _ { \theta }$ that generates candidate solutions, (2) an evaluator that provides task-specific feedback (e.g., test results, correctness signals), (3) a self-reflection module $p _ { \theta }$ that converts feedback into natural language reflections diagnosing errors, and (4) an episodic memory $\mathcal { M }$ that stores reflections from prior iterations. At iteration $k$ , the actor generates candidate solutions conditioned on accumulated reflections from episodic memory:

$$
y _ { k } \sim p _ { \theta } ( \cdot \mid x , r _ { 1 : k - 1 } ) .
$$

Cross-Sample Memory. Cross-sample memory, which leverages past experiences or external logs, has been proposed in recent studies to enhance agent reasoning capabilities (Borgeaud et al., 2022; Shi et al., 2023; Wang et al., 2023; Zhong et al., 2024; Wang et al., 2024b). As recent studies have identified limited diversity in self-reflections, cross-sample memory is adopted to store reasoning trajectories from previously solved problems, thereby enriching the diversity of reflective inputs, which has proven effective in improving agentic reasoning. Given a new task, relevant trajectories are retrieved from the memory bank and incorporated into the prompt:

$$
y \sim p _ { \theta } ( \cdot \mid x , r _ { 1 : k } , \mathrm { R E T R I E V E } ( \mathcal { B } , x ) ) ,
$$

where $\boldsymbol { B }$ denotes the trajectory bank. In this study, we propose ParamMem, a parametric memory module $p _ { \phi } ( \cdot )$ complementary to episodic memory and cross-sample memory, which further promotes the diversity of reflective inputs. Based on ParamMem, we propose ParamAgent and ParamAgent-plus. In ParamAgent, the actor generates solutions conditioned on both episodic memory and parametric memory:

$$
y _ { k } \sim p _ { \boldsymbol \theta } \big ( \cdot \mid x , r _ { 1 : k - 1 } , r _ { k } ^ { g } \big ) ,
$$

where $r _ { k } ^ { g } \sim p _ { \phi } ( \cdot \mid x )$ denotes the reflection sampled from ParamMem at the $k$ -th iteration. ParamAgent-plus further incorporates cross-sample memory, conditioning on all three memory sources:

$$
y _ { k } \sim p _ { \theta } ( \cdot \mid x , r _ { 1 : k - 1 } , \mathrm { R E T R I E V E } ( B , x ) , r _ { k } ^ { g } ) .
$$

![](images/8cf7df6790d39082fba37f4ccaf581c6881fba9f063541c175c09fffeae13bec.jpg)  
Figure 2. Comparison of memory mechanisms across different frameworks. Pale Mint denotes episodic memory only, as in Reflexion and DoT. Lavender Gray indicates episodic memory combined with cross-sample memory, as in DoT-bank. Soft Sand represents the full integration of episodic, cross-sample, and parametric memory, as in ParamAgent-plus. When using only episodic and parametric memory, the framework reduces to ParamAgent.

An architectural comparison of these frameworks is illustrated in Figure 2.

# 3. Augmenting Language Agents with ParamMem

In this section, we first describe how to construct ParamMem, and then present how to incorporate it into the proposed framework ParamAgent.

# 3.1. Building

# ParamMem

The core idea of ParamMem is to implicitly capture crosssample regularities via training dynamics. Through finetuning, the module learns to generalize reflection patterns to unseen examples, rather than relying on prompt-based instructions or retrieving similar samples. While promptbased methods are constrained by fixed instruction templates, and retrieval-based methods are limited by embedding similarity to existing examples, ParamMem can generate novel reflections by interpolating and extrapolating from learned patterns, therefore providing an additional source of diversity. The building process begins with constructing an auxiliary dataset for finetuning. Specifically, we curate a dataset $\mathbf { \mathcal { D } } = \{ ( x _ { i } , r _ { i } ^ { g } ) \} _ { i = 1 } ^ { n }$ , where $x _ { i }$ denotes the input sample (e.g., a programming task), and $r _ { i } ^ { g } \ = \ f _ { \phi } ( x _ { i } ; \mathcal { P } )$ is obtained by prompting an LLM $f _ { \phi }$ with a task-specific prompt $\mathcal { P }$ to generate auxiliary supervision for $x _ { i }$ . We then fine-tune a pretrained LLM on $\mathcal { D }$ using LoRA (Hu et al., 2022) to obtain the parametric module $\mathcal { M } _ { g }$ .

For programming and math tasks, $r _ { i } ^ { g }$ takes the form of reflective feedback that enumerates potential mistakes and buggy implementations. For multi-hop QA, directly providing all supporting passages would consume excessive tokens. Inspired by cognitive chunking (Miller, 1956; Baddeley, 2020) and least-to-most prompting (Zhou et al., 2022), we instead prompt the LLM to decompose the query into compact semantic units and potential reasoning sub-tasks. An example for programming and multi-hop QA is illustrated in Figure 3. Further details on dataset construction are provided in Appendix B.2.

# 3.2. Incorporating ParamMem into Reflexion-based Framework

Once the parametric module $\mathcal { M } _ { g }$ is obtained, we incorporate it into the Reflexion-based framework. The integration is straightforward: at the $k$ -th iteration, when providing $\{ r _ { 1 } , \ldots , r _ { k - 1 } \}$ to the actor, we additionally sample a modelbased output $r _ { k } ^ { g } \sim p _ { \psi } ( \cdot \mid x )$ from $\mathcal { M } _ { g }$ and concatenate it with the self-reflections:

$$
y _ { k } \sim p _ { \boldsymbol \theta } \big ( \cdot \mid x , r _ { 1 : k - 1 } , r _ { k } ^ { g } \big ) ,
$$

where $r _ { k } ^ { g }$ denotes the global-level reflection for programming and math, and denotes decomposed semantic unit and sub-tasks for multi-hop QA. We refer to this framework as ParamAgent. We further introduce ParamAgent-plus, a more powerful variant that additionally retrieves reasoning trajectories $\{ \tau _ { 1 } , \ldots , \tau _ { k } \}$ from a memory bank $\boldsymbol { B }$ of previously solved tasks. To ensure a fair comparison, the retrieval mechanism follows DoTbank (Lingam et al., 2025). The actor then conditions on both parametric and cross-sample signals:

$$
y _ { k } \sim p _ { \theta } ( \cdot \mid x , r _ { 1 : k - 1 } , r _ { k } ^ { g } , \tau _ { 1 : j } ) .
$$

The pseudocode is provided in Algorithm 1. Similar to prior approaches (Yao et al., 2023c; Lingam et al., 2025), ParamMem does not directly interact with the environment during inference. Instead, by conditioning the actor on model-based feedback $r _ { k } ^ { g }$ , the output distribution is shaped (a) An output example on programming task.

![](images/d6c295321845eaddb95d6f1c04edc7718d948ea3d6ee4d31a46d49b7007c3ade.jpg)  
Figure 3. Illustration of the output produced by ParamMem.

![](images/03c4efd74d83b0c4b5f3587ab75f4a0f30630084c0648f528eeb8dd23c916d57.jpg)

(b) An output example on multi-hop QA task.

# Algorithm 1 Pseudocode for the proposed method

Require: Dataset $\mathcal { D }$ , base LM $p _ { \theta }$ , episodic memory $\mathcal { M }$ , parametric module $\mathcal { M } _ { g }$ parametrized by $p _ { \psi }$ , cross-sample memory bank $\boldsymbol { B }$ , Failed task set $\mathcal { F }$ , max iterations $T _ { \mathrm { m a x } }$ . 1: $M \gets \emptyset$ , $B  \emptyset$ , $F \gets \emptyset$ Phase 1: ParamAgent 2: for $x \in \mathcal { D }$ do   
3: for $t = 1$ to $T _ { \mathrm { m a x } }$ do   
4: $T \gets 0 . 2$ if $t = 1$ , else 1.0   
5: $r _ { t } ^ { g } \sim p _ { \psi } ( \cdot \mid x ; T )$ ▷ Sample from $\mathcal { M } _ { g }$   
6: $r _ { 1 : t - 1 } $ RETRIEVEREFLECTIONS $( { \mathcal { M } } , x )$   
7: $y _ { t } \sim p _ { \theta } \big ( \cdot \mid x , r _ { 1 : t - 1 } , r _ { t } ^ { g } \big )$   
8: if EVALUATE $( y _ { t } , x )$ then   
9: $B  B \cup \{ ( x , \tau ) \}$ ; break $\triangleright$ Store trajectory   
10: else   
11: $\begin{array} { r l } & { { r _ { t } } \gets \mathrm { G E N E R A T E S E L F R E F L E C T I O N } ( y _ { t } ) } \\ & { \mathrel { \mathcal { M } } \gets \mathcal { M } \cup \{ ( x , r _ { t } ) \} } \end{array}$   
12:   
13: end if   
14: end for   
15: if not solved then ${ \mathcal { F } } \gets { \mathcal { F } } \cup \{ x \}$   
16: end for Phase 2: ParamAgent-plus ▷ Reattempt with cross-sample memory   
17: for $x \in { \mathcal { F } }$ do   
18: $\tau _ { 1 : j } \gets$ RETRIEVESIMILAR $( B , x , j )$ ▷ Retrieve j trajectories   
19: Repeat Phase 1 with $y _ { t } \sim p _ { \theta } ( \cdot \mid x , r _ { 1 : t - 1 } , r _ { t - 1 } ^ { g } , \tau _ { 1 : j } )$   
20: end for

effectiveness of ParamMem by analyzing how it promotes reflective diversity in both static and dynamic settings. We then perform comprehensive ablation studies to examine several key properties: (1) whether ParamAgent can achieve self-improvement without relying on stronger external models, (2) whether smaller parametric modules can enhance agents built on stronger LLMs (weak-to-strong transfer), and (3) the sample efficiency of ParamMem. More experimental results are included in Appendix B, including experiments with 70B scale LLMs and cost analysis.

# 4.1. Setup

Datasets We evaluate our framework across three domains. For programming, we use HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021). We also use Live-CodeBench (Jain et al., 2025), a more challenging dataset for additional empirical evaluation. For math reasoning, we adopt the MATH dataset (Hendrycks et al., 2021b), which covers competition-level problems of varying difficulty across seven subjects. For multi-hop QA, we use HotpotQA (Yang et al., 2018) and 2WikiMultiHopQA (Ho et al., 2020), which require reasoning across multiple passages. Further details about each dataset, as well as how we perform dataset splits are provided in Appendix B.

by parametric knowledge, which subsequently influences the generation of new self-reflections. This feedback loop enables ParamMem to indirectly participate in the dynamic interaction process.

Evaluation For programming tasks, we report $\mathrm { P a s s } @ 1$ During generation, only visible or synthetic test cases are used, while final evaluation is conducted on hidden test cases; a score of 1 is assigned if all tests pass and 0 otherwise. For math reasoning and multi-hop QA, we report 0–1 accuracy on subsampled testsets.

# 4. Experiments

In this section, we detail our experimental setup and present results across programming, math reasoning, and multi-hop QA. We then conduct more in-depth empirical analyses of our proposed method. Specifically, we first validate the

Baselines We compare against: (1) Base, the underlying LLM agent without reflection; (2) Reflexion (Shinn et al., 2023), which uses episodic self-reflections; (3) Retroformer (Yao et al., 2023c), which also employs a parametric reflective module but trains it via policy gradient optimization to improve reflection accuracy rather than diversity, which serves as a direct comparison with ParamAgent; (4) DoT (Lingam et al., 2025), which augments Reflexion with prompt-level diversity; (5) DoT-bank (Lingam et al., 2025), which further incorporates a memory bank to enrich the reflective feedbacks.

To ensure a comprehensive evaluation, we employ three backbone LLMs with varying levels of reasoning capability: (1) Llama-3.1-8B (Dubey et al., 2024), a strong opensource reasoning model; (2) Mistral-7B-v0.2 (Jiang et al., 2023), a competitive medium-sized model with efficient inference; and (3) Qwen2-1.5B-instruct (Bai et al., 2023). This selection of backbones allows us to examine how our approach performs across different model sizes and reasoning strengths. We also provide results with stronger base LLMs in Appendix B.3, showing that even with a smaller backbone as the parametric module $\mathcal { M } _ { g }$ , it can still provide noticable gains to agents built on 70B-scale LLMs.

Implementation details Across all experiments, we fix the number of iterations to 5 for both baseline methods and our proposed approach. For ParamAgent and its variants, we set the sampling temperature to $T = 0 . 2$ during the first iteration, and $T = 1 . 0$ in the subsequent iterations to promote diversity. Unless otherwise specified, the parametric model is instantiated using Llama3.1-8B-Instruct, and is finetuned via LoRA. For LoRA finetuning, we use a rank of $r = 1 2 8$ , scaling factor $\alpha = 3 2$ , a learning rate of $2 e - 5$ and train for 3 epochs.

# 4.2. Experimental Results

In this section, we first present the main results across 3 domains (Observation $\textcircled{1}$ ). We then analyze how ParamMem promotes reflective diversity in both static and dynamic settings (Observations $\textcircled{2}$ ), followed by a case study explaining why increased reflective diversity leads to performance gains (Observation $\textcircled{3}$ ). We then conduct ablation studies examining our framework without relying on stronger external models to generate datasets (Observation $\textcircled{4}$ ), iterative self-teaching (Observation $\textcircled{5}$ ), weak-to-strong transfer where smaller parametric modules enhance stronger agents (Observation $\textcircled{6}$ ), and the advantage of sample efficiency (Observation $\textcircled{7}$ ).

Observation $\textcircled{1}$ : ParamMem consistently enhances Reflexion-based frameworks across all domains. As shown in Tables 1, both ParamAgent and ParamAgent-plus achieve remarkable performance across the three domains. We note that ParamAgent differs from Reflexion and DoT solely through the incorporation of ParamMem, while ParamAgent-plus extends DoT-bank by augmenting its episodic and cross-sample memory with our parametric module.

On programming benchmarks, ParamAgent achieves significant improvements over all baselines even without the cross-sample memory component. Similarly, on multi-hop QA, ParamAgent substantially outperforms most prior methods, highlighting the standalone effectiveness of ParamMem. For mathematical reasoning, while ParamAgent improves upon Reflexion and DoT, we observe that cross-sample trajectories play a more critical role. This aligns with the intuition that mathematical problemsolving benefits from exposure to analogous problems and solution patterns, akin to how humans learn to solve math problems. Nevertheless, ParamAgent-plus still outperforms DoT-bank by incorporating ParamMem, demonstrating the complementary value of model-based reflective memory. Notably, Retroformer excels on MATH, where reflection accuracy may matter more than diversity. However, it underperforms on programming and multi-hop QA despite also using parametric encoding. We attribute this to distribution shift, as the training data may not align well with test data, causing accuracy-focused optimization to overfit. In contrast, ParamMem’s objective is diversity-driven, implying that diversity-focused parametric memory generalizes better across distributions.

Observation $\textcircled{2}$ : ParamMem induces an additional layer of reflective diversity beyond episodic and cross-sample memory. We hypothesize that the parametric module $\mathcal { M } _ { g }$ introduces an additional source of diversity through the training dynamics. To verify this, we conduct the following analysis on programming tasks. We fine-tune Llama-3.1-8B as the parametric module using synthetic datasets generated by either GPT-4o-mini or Llama-3.1-8B itself. For each task in HumanEval, we sample 10 reflections at temperature $T = 1 . 0$ , embed the outputs, and compute the mean value of pairwise cosine distances $D _ { m e a n }$ , and the distribution across all samples. As illustrated in Figure 4a, the parametric module trained on GPT-4o-mini data yields the highest diversity. Notably, even when using the same LLM for both data generation and fine-tuning, the resulting diversity still exceeds that of the unfinetuned Llama-3.1-8B. This finding also explains why ParamAgent and ParamAgent-plus remain effective in self-improvement settings (Table 2).

The above analysis characterizes diversity in a static setting. We further examine whether this diversity persists when ParamMem is incorporated into the Reflexion-based framework and interacts with the environment via $p _ { \theta } ( \cdot \ |$ $x , r _ { 1 : k - 1 } , r _ { k } ^ { g } )$ . Specifically, we maintain the complete reflection history for each sample on HumanEval and embed all reflections using OpenAI text-embedding-3-small model (OpenAI, 2024). We then perform $K$ -means clustering (Lloyd, 1982) over all reflections and apply the elbow method (Tibshirani et al., 2001) to determine the optimal number of clusters $K ^ { * }$ . As shown in Figure 4b, ParamAgent achieves $K ^ { * } = 3 9$ , substantially larger than Reflexion, DoT, and DoT-bank, indicating that ParamMem introduces significantly richer and more varied reflective signals. Moreover, the silhouette scores of ParamAgent are consistently higher across all $K$ , confirming superior clustering quality and semantic coherence of the generated reflections, as illustrated in Figure 4c. In conclusion, these analyses demonstrate that ParamMem introduces a complementary source of reflective diversity, thereby enriching the feedback signals available to the agent throughout iterative refinement.

Table 1. Performance on HumanEval/MBPP, MATH, HotpotQA, and 2WikiMultiHopQA. Bold denotes the best result, and underline marks the second best. $\uparrow$ and $\downarrow$ indicate the absolute improvement or decrease relative to the Base method. For clarity, the prompt token usage of the Base method is normalized to 1. Score is $\mathrm { P a s s } @ 1$ for HumanEval/MBPP and Accuracy for MATH/QA.   

<table><tr><td>Domain</td><td>Dataset</td><td>Method</td><td colspan="2">Llama-3.1-8B</td><td colspan="2">Mistral-7B-v0.2</td><td colspan="2">Qwen2-1.5B</td></tr><tr><td rowspan="10"></td><td rowspan="10"></td><td></td><td>Score</td><td>#Prompt Tokens</td><td>Score</td><td>#Prompt Tokens</td><td>Score</td><td>#Prompt Tokens</td></tr><tr><td>Base</td><td>59.15</td><td>1.00</td><td>32.93</td><td>1.00</td><td>41.46</td><td>1.00</td></tr><tr><td>Reflexion</td><td>76.22 ↑ 17.07</td><td>9.29</td><td>51.22 ↑ 18.29</td><td>28.54</td><td>49.39 ↑ 7.93</td><td>18.30</td></tr><tr><td>Retroformer</td><td>67.68 8.53</td><td>11.28</td><td>42.94  10.01</td><td>38.37</td><td>46.34 ↑ 4.88</td><td>12.77</td></tr><tr><td>DoT</td><td>73.17 14.02</td><td>17.45</td><td>46.95 ↑ 14.02</td><td>43.06</td><td>56.56  15.10</td><td>15.26</td></tr><tr><td>DoT-bank ParamAgent</td><td>79.56  20.41</td><td>24.71</td><td>54.26 ↑21.33</td><td>61.62</td><td>60.10 18.64</td><td>31.28</td></tr><tr><td></td><td>| 82.93 ↑ 23.78</td><td>19.18</td><td>| 67.07↑ 34.14</td><td>70.38</td><td>66.46 ↑25.00</td><td>33.45</td></tr><tr><td rowspan="5">MBPP</td><td>Base</td><td>47.61</td><td>1.00</td><td>24.94</td><td>1.00</td><td>42.06</td><td>1.00</td></tr><tr><td>Reflexion</td><td>58.69 11.08</td><td>37.18</td><td>28.46 ↑ 3.52</td><td>14.02</td><td>47.61 ↑ 5.55</td><td>26.95</td></tr><tr><td>Retroformer</td><td>42.82 ↓4.79</td><td>8.64</td><td>21.66 ↓ 3.8</td><td>12.08</td><td>31.49 ↓ 10.57</td><td>23.70</td></tr><tr><td>DoT</td><td>61.21 ↑ 13.60</td><td>51.83</td><td>19.79 ↓ 5.15</td><td>25.45</td><td>47.37 ↑ 5.31</td><td>21.48</td></tr><tr><td>DoT-bank</td><td>64.82 ↑ 17.21</td><td>69.41</td><td>24.68 ↓ 0.26</td><td>60.09</td><td>53.38 1.32</td><td>60.95</td></tr><tr><td rowspan="7"></td><td rowspan="7"></td><td>ParamAgent</td><td>67.00  19.39</td><td>86.39</td><td>51.64 ↑26.70</td><td>36.88</td><td>54.90  12.84</td><td>66.86</td></tr><tr><td>Base Reflexion</td><td>48.20</td><td>1.00</td><td>12.23</td><td>1.00</td><td>8.99</td><td>1.00</td></tr><tr><td></td><td>58.99  10.79</td><td>23.33</td><td>19.78 ↑ 7.55</td><td>27.67</td><td>21.94 ↑ 12.95</td><td>18.39</td></tr><tr><td>Retroformer</td><td>63.67 ↑ 15.47</td><td>17.09</td><td>43.53 ↑31.30</td><td>35.67</td><td>33.09 ↑24.10</td><td>30.12</td></tr><tr><td>DoT</td><td>64.38 ↑ 16.18</td><td>34.17</td><td>23.25 ↑ 11.02</td><td>40.51</td><td>22.30 ↑ 13.31</td><td>31.99</td></tr><tr><td>DoT-bank</td><td>73.02 ↑ 24.82</td><td>83.92</td><td>35.61 ↑23.38</td><td>122.92</td><td>24.37 ↑ 15.38</td><td>76.71</td></tr><tr><td>ParamAgent</td><td>67.99  19.79</td><td>57.01</td><td>28.06 ↑ 15.83</td><td>92.91</td><td>22.30  13.31</td><td>70.07</td></tr><tr><td rowspan="9">QA</td><td rowspan="6">HotpotQA</td><td>ParamAgent-plus</td><td>75.45 ↑ 27.25</td><td>111.32</td><td>38.96 ↑ 26.73</td><td>196.18</td><td>25.97 16.98</td><td>144.25</td></tr><tr><td>Base Reflexion</td><td>57.67</td><td>1.00</td><td>45.00</td><td>1.00</td><td>43.66</td><td>1.00</td></tr><tr><td>Retroformer</td><td>71.33 ↑ 3.66 73.00 15.3</td><td>4.13</td><td>62.33 ↑ 17.3</td><td>4.67</td><td>50.03 ↑6.37</td><td>6.22</td></tr><tr><td>DoT</td><td>666679.00</td><td>2.77</td><td>67.33 ↑2.33</td><td>4.59</td><td>47.70 ↑ 4.04</td><td>9.17</td></tr><tr><td>DoT-bank</td><td></td><td>7.10</td><td>58.33 ↑13.33</td><td>8.97</td><td>49.32 ↑ 5.66</td><td>58.05</td></tr><tr><td>ParamAgent</td><td>72.00  14.33</td><td>13.28</td><td>66.33 ↑ 21.33</td><td>19.35</td><td>52.02 ↑8.36</td><td>109.54</td></tr><tr><td></td><td>| 78.33 ↑ 20.66</td><td>22.25</td><td>| 69.67↑ 24.67</td><td>34.99</td><td>64.66 ↑21.00</td><td></td><td>14.69</td></tr><tr><td rowspan="5">2WikiMultiHopQA</td><td>Base</td><td>40.33</td><td>1.00</td><td>21.00</td><td>1.00</td><td>40.33</td><td>1.00</td></tr><tr><td>Reflexion</td><td>78.67 ↑ 38.34</td><td>5.47</td><td>61.33 ↑ 40.33</td><td>5.86</td><td>51.00 10.67</td><td>6.56</td></tr><tr><td>Retroformer</td><td>77.00  36.67</td><td>5.90</td><td>71.00 ↑ 50.00</td><td>5.33</td><td>67.66 ↑27.33</td><td>3.68</td></tr><tr><td>DoT</td><td>66.67 ↑ 26.34</td><td>7.03</td><td>52.13 ↑31.13</td><td>6.40</td><td>47.83 ↑ 7.50</td><td>30.55</td></tr><tr><td>DoT-bank</td><td>80.33 ↑40.00</td><td>12.49</td><td>74.66 ↑ 53.6</td><td>8.10</td><td>50.49  10.16</td><td>54.92</td></tr><tr><td></td><td>ParamAgent</td><td>88.67 ↑48.34</td><td>10.41</td><td>81.33  60.33</td><td>14.43</td><td>63.33 ↑23.00</td><td>17.39</td></tr></table>

Observation $\textcircled{3}$ : Diverse reflections enlarge the hypothesis space for error diagnosis. To understand the reason behind diversity-driven gains, we conduct a case study on MBPP, focusing on instances where ParamAgent succeeds but Reflexion and DoT fail (Figure 8 in Appendix B.5). We observe that self-reflections often fail to identify the core source of errors and mislead the agent away from correct implementations. While ParamAgent is not immune to such failure modes, the increased diversity of reflective feedback provides the agent with a broader set of diagnostic hypotheses, thereby increasing the likelihood of encountering the correct cue for successful refinement. This also explains why ParamAgent and ParamAgent-plus occasionally incur higher token consumption in certain datasets.

Table 2. Self-improvement results using Llama-3.1-8B as both the agent and data generator Bold denotes the best, underline the second best. $\Theta$ denotes Llama-3.1-8B-Instruct as the data generator.   

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>HumanEval HotpotQA</td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>59.15      57.67</td></tr><tr><td rowspan=1 colspan=1>Reflexion</td><td rowspan=1 colspan=1>76.22 ↑ 17.07 71.33↑ 13.66</td></tr><tr><td rowspan=1 colspan=1>DoT</td><td rowspan=1 colspan=1>73.17 ↑14.02 66.67 ↑ 9.00</td></tr><tr><td rowspan=1 colspan=1>DoT-bank</td><td rowspan=1 colspan=1>79.56↑20.41 72.00 ↑ 14.33</td></tr><tr><td rowspan=1 colspan=1>ParamAgent</td><td rowspan=1 colspan=1>78.05↑18.90 76.33↑ 18.66</td></tr><tr><td rowspan=1 colspan=1>ParamAgent-plus</td><td rowspan=1 colspan=1>86.59 ↑27.4483.33 ↑ 25.66</td></tr></table>

Observation $\textcircled{4}$ : ParamMem supports agent selfimprovement without dependence on stronger external

![](images/7331313a79e1cd6c8eedd30ad2842579302d9fc32c781c5b378fea2c92da6d59.jpg)  
(a) Pairwise cosine distance distribution.

![](images/d038d0e161688939c3636bbf25636cb6d4e50520c030871a320956c29e48a70e.jpg)  
(b) Clustering analysis.

![](images/c81db767633ebcdf0d0de8ab74bcd29e72b04058c1ff4d2cdcab9335e976e322.jpg)  
(c) Silhouette score across $K$ .

Figure 4. Reflection diversity induced by $\textcircled{9}$ ParamMem. (a) Higher pairwise distance indicates more diverse outputs. (b) Higher optimal $K$ and silhouette scores confirm greater semantic variation. (c) Silhouette score as a function of cluster number $K$ .

![](images/c9097c1112f4a08182ec5e32454cd36b0ddce422bb9102b8ebdb9825de012c39.jpg)  
Figure 5. The performance of ParamAgent and ParamAgent-plus in HumanEval dataset, with 3 iterative process.

models. Recent work on self-improving agents seeks to enhance reasoning capabilities without relying on external stronger models (Wei et al., 2022; Zelikman et al., 2022; Shinn et al., 2023; Zeng et al., 2024; Snell et al., 2025; Muennighoff et al., 2025). ParamAgent exhibits a similar property: even when ParamMem is fine-tuned on synthetic data generated by the base model itself, it still yields consistent gains. Specifically, we use Llama-3.1-8B to generate synthetic data and fine-tune the same base model as the parametric memory module. As shown in Table 2, ParamAgent and ParamAgent-plus improve significantly over DoT and DoT-bank, respectively. This demonstrates that ParamMem can enhance reasoning through diversified reflections without relying on stronger external models.

Observation $\textcircled{5}$ : Iterative self-teaching further enhances ParamAgent. We investigate iterative self-teaching on HumanEval: starting from Llama-3.1-8B-Instruct, we finetune ParamMem on 1,000 randomly sampled examples. After training, we use the resulting model to generate new targets for the same inputs, yielding an updated dataset $\mathcal D ^ { \prime } = \{ ( x _ { i } , \tilde { r } _ { i } ^ { g } ) \} _ { i = 1 } ^ { 1 0 0 0 }$ where $\tilde { r } _ { i } ^ { g } \sim p _ { \phi } ( \cdot \mid x _ { i } )$ are freshly sampled reflections. We then fine-tune on $\mathcal { D } ^ { \prime }$ and repeat this process for 3 iterations. As shown in Figure 5, ParamAgent improves steadily across iterations, suggesting that ParamMem progressively produces more diverse reflections. In contrast, ParamAgent-plus shows only marginal gains, we hypothesize that with cross-sample trajectories, the model already approach the diversity ceiling.

Table 3. Weak-to-strong transfer on LiveCodeBench, HumanEval, and HotpotQA with Qwen3-Next-80B-A3B-Instruct as the base LLM in the agent. $\odot$ and $\Join$ denote ParamMem instantiated by Llama-3.1-8B-Instruct and Qwen3-Next-30B-A3B-Instruct.   

<table><tr><td>Method</td><td>LiveCodeBench HumanEval HotpotQA</td><td></td><td></td></tr><tr><td>Simple</td><td>52.00</td><td>90.24</td><td>71.33</td></tr><tr><td>Reflexion</td><td>62.00  10.00</td><td>96.34 ↑6.10</td><td>82.33 11.00</td></tr><tr><td>DoT</td><td>60.67 ↑8.67</td><td>96.34 ↑6.10</td><td>79.67 ↑ 8.34</td></tr><tr><td>DoT-bank</td><td>62.00 ↑ 10.00</td><td>96.95 ↑ 6.71</td><td>83.33↑12.00</td></tr><tr><td>ParamAgent</td><td>61.33 ↑ 9.33</td><td>97.56 ↑ 7.32</td><td>79.67↑ 8.34</td></tr><tr><td>ParamAgent-plus</td><td>63.33 ↑11.33</td><td>98.17 ↑ 7.93</td><td>85.00 ↑ 13.67</td></tr><tr><td>ParamAgent</td><td>64.67 ↑ 12.67</td><td></td><td>76.33 5.00</td></tr><tr><td>ParamAgent-plus</td><td>68.00↑16.00</td><td></td><td>83.67  12.34</td></tr></table>

This resembles STaR (Zelikman et al., 2022) and recent variants (Hosseini et al., 2024; Zeng et al., 2024), but with a key difference: those methods filter for correct samples at each iteration, implicitly performing reward-driven bootstrapping. Our approach requires no filtering, since ParamMem targets diversity rather than correctness.

Observation $\textcircled{6}$ : ParamAgent enables weak-to-strong reasoning augmentation. The previous observation demonstrates that ParamMem supports self-improvement. Here we examine whether a weaker model serving as ParamMem can enhance a stronger agent. We evaluate on Live-CodeBench and HotpotQA using Qwen3-Next-80B-A3B-Instruct (Team, 2025) as the base model in the agent, with ParamMem instantiated by either Llama-3.1-8B or Qwen3- Next-30B-A3B-Instruct (Team, 2025). As shown in Table 3, both configurations consistently outperform all baselines. For coding, the Qwen3-30B-based ParamMem proves more effective, improving over the strongest baseline by $9 . 7 \%$ . Interestingly, for multi-hop QA, the smaller Llama-3.1-8Bbased ParamMem outperforms its Qwen3-30B counterpart, achieving a $2 . 0 \%$ relative improvement over the best baseline method. These results confirm that ParamMem enables weak-to-strong transfer: even smaller models can provide diverse reflective signals that benefit stronger agents.

Table 4. Sample efficiency of $\textcircled { < }$ ParamMem. Models are trained on 500 diverse samples via $K$ -means clustering. Bold denotes the best result, underline the second best.   

<table><tr><td>Method</td><td>HumanEval</td><td>MBPP</td></tr><tr><td>DoT</td><td>73.17</td><td>61.21</td></tr><tr><td>DoT-bank</td><td>79.56</td><td>64.82</td></tr><tr><td>ParamAgent ( 8000 samples)</td><td>82.93</td><td>67.00</td></tr><tr><td>ParamAgent (500 samples)</td><td>81.71</td><td>64.99</td></tr><tr><td>ParamAgent-plus (500 samples)</td><td>86.59</td><td>65.49</td></tr></table>

Observation $\textcircled{7}$ : ParamMem is sample-efficient. We examine how many samples are needed for an effective ParamMem. We apply $K$ -means clustering to the GPT-4o-mini synthetic data and sample 500 diverse examples across clusters for fine-tuning. As shown in Table 4, ParamAgent and ParamAgent-plus retain strong performance on HumanEval and MBPP even with this reduced training set; Notably, ParamAgent-plus with only 500 training samples outperforms ParamAgent trained on the dataset of over 8000 samples, demonstrating the effective synergy of episodic, cross-sample, and parametric memory.

# 5. Related Work

LLM Reasoning and Diversity LLMs perform multistep reasoning through techniques like CoT prompting (Wei et al., 2022), Self-Consistency (Wang et al., 2022), and Re-Act (Yao et al., 2023b). Self-Consistency improves CoT by sampling multiple reasoning paths and aggregating them via majority voting, demonstrating that diversity in reasoning traces leads to more robust outputs. Iterative self-feedback methods (Madaan et al., 2023; Shinn et al., 2023) and testtime compute scaling (Snell et al., 2025; OpenAI et al., 2024; Guo et al., 2025) further improve reasoning by allocating additional computation during inference. To enhance diversity, structured exploration methods like Tree of Thoughts (Yao et al., 2023a) and Graph of Thoughts (Besta et al., 2024) enable deliberate search over reasoning states. These methods highlight that exploring diverse solution paths is crucial for solving complex problems. Most relevant to our work, DoT (Lingam et al., 2025) addresses repetitive selfreflections via prompt-level interventions and cross-sample memory. We extend this line of research by proposing ParamMem, which provides an orthogonal source of reflective diversity beyond episodic and cross-sample memory through parametric encoding.

Improving Reflection in LLM Agents Recent work has explored diversity-driven approaches to improve reflection. DoT (Lingam et al., 2025) addresses repetitive selfreflections through prompt-level interventions and crosssample memory retrieval. Beyond diversity-driven methods, other approaches improve reflection through different mechanisms. Retroformer (Yao et al., 2023c) uses policy gradient optimization to learn a retrospective model that refines prompts based on environment feedback, enabling the agent to improve its reflection accuracy over time. Self-RAG (Asai et al., 2024) trains special reflection tokens directly into the generative model, enabling self-critique of generation and retrieval decisions during inference. ExpeL (Zhao et al., 2024) extracts generalizable insights from successful trajectories and stores them for cross-task transfer. These approaches primarily focus on improving reflection quality or accuracy through various mechanisms. While ParamMem also adopts parametric encoding of reflections like Retroformer and Self-RAG, it differs in both purpose and design: rather than optimizing reflection accuracy or self-critique capability, ParamMem aims to enhance reflective diversity by unifying episodic memory, cross-sample memory, and parametric memory within a single framework.

Self-improving Language Agents Self-improvement in language models enables agents to enhance their reasoning capabilities through iterative learning from self-generated data. STaR introduced bootstrapped reasoning, where models generate rationales and fine-tune on those leading to correct answers (Zelikman et al., 2022). This paradigm was extended by ReST (Gulcehre et al., 2023) and ReST-EM (Singh et al., 2023), which demonstrated that selfgenerated training data can surpass human-annotated data when verifiable feedback is available. More recent work has eliminated the need for external reward models entirely: Self-Rewarding Language Models (Yuan et al., 2024) use LLM-as-a-Judge (Zheng et al., 2023) to generate preference data, while Meta-Rewarding (Wu et al., 2025) adds metajudgment capabilities. SPIN (Chen et al., 2024) demonstrates that self-play fine-tuning can iteratively convert a weak language model into a stronger one by having the model compete against earlier versions of itself, without any additional human-annotated data. In contrast to these approaches, ParamMem enables self-improvement by progressively diversifying reflective feedback, thereby strengthening reflection-based frameworks. Notably, ParamMem does not require any external reward signal or human annotation; it leverages the base model’s own generated reflections as training data, making the self-improvement process both scalable and practical.

# 6. Conclusions and Limitations

We propose $\textcircled { < } \textcircled { < }$ ParamMem, a parametric memory module that internalizes reflections into model parameters, inducing an additional layer of diversity beyond episodic and crosssample memory. Building upon ParamMem, we introduce ParamAgent and ParamAgent-plus, which augment reflection-based reasoning frameworks with ParamMem. Across 3 domains, our methods deliver substantial performance gains over state-of-the-art baselines, highlighting the potential of parametric memory as a lightweight plug-in module for building language agents. Despite these advantages, our approach has limitations. A notable one is the increased token consumption in certain scenarios, which is an inherent cost of the additional reflective diversity. In future work, we aim to address this trade-off by exploring more token-efficient integration strategies.

# Impact Statement

Reflection-based reasoning is widely adopted in language agent systems to improve task performance across diverse domains. This work studies a fundamental limitation of selfreflection: the lack of diversity in generated reflections, and proposes a novel approach to address it. Our method has a positive impact on advancing the understanding and capability of language agents. We do not foresee any potential negative societal impact arising from this work.

# References

An, S., Ma, Z., Lin, Z., Zheng, N., Lou, J.-G., and Chen, W. Learning from mistakes makes llm better reasoner. arXiv preprint arXiv:2310.20689, 2023.   
Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Selfrag: Learning to retrieve, generate, and critique through self-reflection. 2024.   
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.   
Baddeley, A. Working memory. Memory, pp. 71–111, 2020.   
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 17682–17690, 2024.   
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pp. 2206–2240. PMLR, 2022.   
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,

Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,

Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Xu, H., Ding, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Chen, J., Yuan, J., Tu, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., You, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Zhou, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo,

Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx. doi.org/10.1038/s41586-025-09422-z.

Guo, X., Pan, J., Wang, X., Chen, B., Jiang, J., and Long, M. On the embedding collapse when scaling up recommendation models. arXiv preprint arXiv:2310.04400, 2023.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., and Steinhardt, J. Measuring coding challenge competence with apps. NeurIPS, 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021b.

Ho, X., Nguyen, A.-K. D., Sugawara, S., and Aizawa, A. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.

Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., Wang, J., Wang, Z., Yau, S. K. S., Lin, Z., et al. Metagpt: Meta programming for a multi-agent collaborative framework. International Conference on Learning Representations, ICLR, 2024.

Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., and Agarwal, R. V-star: Training verifiers for selftaught reasoners. arXiv preprint arXiv:2402.06457, 2024.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id $=$ chfJJYC3iL.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.- A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. Mistral 7b, 2023. URL https: //arxiv.org/abs/2310.06825.

Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Lingam, V., Omidvar-Tehrani, B., Sanghavi, S., Gupta, G., Ghosh, S., Liu, L., Huan, L., and Deoras, A. Enhancing language model agents using diversity of thoughts. 2025.

Lloyd, S. Least squares quantization in pcm. IEEE transactions on information theory, 28(2):129–137, 1982.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

Miller, G. A. The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological review, 63(2):81, 1956.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E., and Hashimoto, T. B. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20286– 20332, 2025.

Nguyen, T. and Yates, A. Generative retrieval as dense retrieval. arXiv preprint arXiv:2306.11397, 2023.

OpenAI. New embedding models and API updates. https://openai.com/index/ new-embedding-models-and-api-updates/, January 2024.

OpenAI, :, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., Iftimie, A., Karpenko, A., Passos, A. T., Neitz, A., Prokofiev, A., Wei, A., Tam, A., Bennett, A., Kumar, A., Saraiva, A., Vallone, A., Duberstein, A., Kondrich, A., Mishchenko, A., Applebaum, A., Jiang, A., Nair, A., Zoph, B., Ghorbani, B., Rossen, B., Sokolowsky,

B., Barak, B., McGrew, B., Minaiev, B., Hao, B., Baker, B., Houghton, B., McKinzie, B., Eastman, B., Lugaresi, C., Bassin, C., Hudson, C., Li, C. M., de Bourcy, C., Voss, C., Shen, C., Zhang, C., Koch, C., Orsinger, C., Hesse, C., Fischer, C., Chan, C., Roberts, D., Kappler, D., Levy, D., Selsam, D., Dohan, D., Farhi, D., Mely, D., Robinson, D., Tsipras, D., Li, D., Oprica, D., Freeman, E., Zhang, E., Wong, E., Proehl, E., Cheung, E., Mitchell, E., Wallace, E., Ritter, E., Mays, E., Wang, F., Such, F. P., Raso, F., Leoni, F., Tsimpourlas, F., Song, F., von Lohmann, F., Sulit, F., Salmon, G., Parascandolo, G., Chabot, G., Zhao, G., Brockman, G., Leclerc, G., Salman, H., Bao, H., Sheng, H., Andrin, H., Bagherinezhad, H., Ren, H., Lightman, H., Chung, H. W., Kivlichan, I., O’Connell, I., Osband, I., Gilaberte, I. C., Akkaya, I., Kostrikov, I., Sutskever, I., Kofman, I., Pachocki, J., Lennon, J., Wei, J., Harb, J., Twore, J., Feng, J., Yu, J., Weng, J., Tang, J., Yu, J., Candela, J. Q., Palermo, J., Parish, J., Heidecke, J., Hallman, J., Rizzo, J., Gordon, J., Uesato, J., Ward, J., Huizinga, J., Wang, J., Chen, K., Xiao, K., Singhal, K., Nguyen, K., Cobbe, K., Shi, K., Wood, K., Rimbach, K., Gu-Lemberg, K., Liu, K., Lu, K., Stone, K., Yu, K., Ahmad, L., Yang, L., Liu, L., Maksin, L., Ho, L., Fedus, L., Weng, L., Li, L., McCallum, L., Held, L., Kuhn, L., Kondraciuk, L., Kaiser, L., Metz, L., Boyd, M., Trebacz, M., Joglekar, M., Chen, M., Tintor, M., Meyer, M., Jones, M., Kaufer, M., Schwarzer, M., Shah, M., Yatbaz, M., Guan, M. Y., Xu, M., Yan, M., Glaese, M., Chen, M., Lampe, M., Malek, M., Wang, M., Fradin, M., McClay, M., Pavlov, M., Wang, M., Wang, M., Murati, M., Bavarian, M., Rohaninejad, M., McAleese, N., Chowdhury, N., Chowdhury, N., Ryder, N., Tezak, N., Brown, N., Nachum, O., Boiko, O., Murk, O., Watkins, O., Chao, P., Ashbourne, P., Izmailov, P., Zhokhov, P., Dias, R., Arora, R., Lin, R., Lopes, R. G., Gaon, R., Miyara, R., Leike, R., Hwang, R., Garg, R., Brown, R., James, R., Shu, R., Cheu, R., Greene, R., Jain, S., Altman, S., Toizer, S., Toyer, S., Miserendino, S., Agarwal, S., Hernandez, S., Baker, S., McKinney, S., Yan, S., Zhao, S., Hu, S., Santurkar, S., Chaudhuri, S. R., Zhang, S., Fu, S., Papay, S., Lin, S., Balaji, S., Sanjeev, S., Sidor, S., Broda, T., Clark, A., Wang, T., Gordon, T., Sanders, T., Patwardhan, T., Sottiaux, T., Degry, T., Dimson, T., Zheng, T., Garipov, T., Stasi, T., Bansal, T., Creech, T., Peterson, T., Eloundou, T., Qi, V., Kosaraju, V., Monaco, V., Pong, V., Fomenko, V., Zheng, W., Zhou, W., McCabe, W., Zaremba, W., Dubois, Y., Lu, Y., Chen, Y., Cha, Y., Bai, Y., He, Y., Zhang, Y., Wang, Y., Shao, Z., and Li, Z. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., and Gonzalez, J. Memgpt: Towards llms as operating systems. 2023.

Park, J. S., O’Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., and Yih, W.-t. Replug: Retrievalaugmented black-box language models. arXiv preprint arXiv:2301.12652, 2023.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Garcia, X., Liu, P. J., Harrison, J., Lee, J., Xu, K., et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Snell, C. V., Lee, J., Xu, K., and Kumar, A. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

Team, Q. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.

Tibshirani, R., Walther, G., and Hastie, T. Estimating the number of clusters in a data set via the gap statistic. Journal of the royal statistical society: series b (statistical methodology), 63(2):411–423, 2001.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36:74530–74543, 2023.

Ozer, O., Wu, G., Wang, Y., Dosti, D., Zhang, H., and De La Rue, V. Mar: Multi-agent reflexion improves reasoning abilities in llms. arXiv preprint arXiv:2512.20845, 2025.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Wang, Y., Gao, Y., Chen, X., Jiang, H., Li, S., Yang, J., Yin, Q., Li, Z., Li, X., Yin, B., et al. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024a.

Wang, Z. Z., Mao, J., Fried, D., and Neubig, G. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024b.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Weller, O., Boratko, M., Naim, I., and Lee, J. On the theoretical limitations of embedding-based retrieval. arXiv preprint arXiv:2508.21038, 2025.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., and Wang, C. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 3(4), 2023.

Wu, T., Yuan, W., Golovneva, O., Xu, J., Tian, Y., Jiao, J., Weston, J. E., and Sukhbaatar, S. Meta-rewarding language models: Self-improving alignment with llm-asa-meta-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 11548–11565, 2025.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023a.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023b.

Yao, W., Heinecke, S., Niebles, J. C., Liu, Z., Feng, Y., Xue, L., Murthy, R., Chen, Z., Zhang, J., Arpit, D., et al. Retroformer: Retrospective large language agents with policy gradient optimization. arXiv preprint arXiv:2308.02151, 2023c.

Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., and Weston, J. E. Self-rewarding language models. In Forty-first International Conference on Machine Learning, 2024.

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Zeng, W., Huang, Y., Zhao, L., Wang, Y., Shan, Z., and He, J. B-star: Monitoring and balancing exploration and exploitation in self-taught reasoners. arXiv preprint arXiv:2412.17256, 2024.

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., and Huang, G. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36: 46595–46623, 2023.

Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.

Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.

# Appendix

# A. More Related Work

Memory Systems for Language Agents. Memory architectures have been extensively studied to enhance agent capabilities. Generative Agents (Park et al., 2023) introduced the memory stream with retrieval based on recency, importance, and relevance. MemGPT (Packer et al., 2023) applies OS-inspired virtual memory management, while MemoryBank (Zhong et al., 2024) incorporates Ebbinghaus forgetting curves for human-like memory decay. For cross-sample learning, ExpeL (Zhao et al., 2024) maintains experience pools and abstracted insight stores. Notably, nearly all existing memory systems are retrieval-based, relying on embedding similarity to access stored experiences. While retrieval-based approaches like DoT-bank (Lingam et al., 2025) have shown promise in diversifying reflections through cross-sample trajectories, they suffer from limited capacity for capturing compositional patterns (Nguyen & Yates, 2023) and embedding collapse into low-rank subspaces (Guo et al., 2023).

Parametric Approaches to Learning from Experience. A growing line of work explores encoding experiences and reflection capabilities directly into model parameters. Retroformer (Yao et al., 2023c) trains a dedicated retrospective model via policy gradient reinforcement learning to generate improved verbal feedback, demonstrating that learned reflection can outperform prompt-based approaches; however, it requires expensive online RL with environment interaction. Self-RAG (Asai et al., 2024) trains LLMs to generate special reflection tokens that enable adaptive retrieval decisions and selfcritique of generation quality, representing a hybrid between prompting and parametric learning, though it focuses primarily on factual verification rather than reasoning diversity. LEMA (An et al., 2023) fine-tunes LLMs on mistake-correction pairs generated by stronger models, achieving strong results on mathematical reasoning by parametrically encoding error patterns. SCoRe (Kumar et al., 2024) trains models for intrinsic self-correction via multi-turn reinforcement learning, demonstrating significant improvements without external feedback but at considerable computational cost. MemoryLLM (Wang et al., 2024a) integrates a self-updatable memory pool within the LLM’s latent space, enabling parametric storage without full finetuning. These works collectively suggest that parametric learning can be more effective than retrieval for capturing complex patterns, though often at significant computational cost or with narrow task focus. Our work introduces ParamMem, a lightweight parametric memory module that specifically targets reflective diversity, encoding cross-sample reflection patterns into parameters via efficient supervised fine-tuning to enable diverse reflection generation, which supports reflection-based framework to unifies various forms of memories.

# B. More Experimental Details Results

# B.1. Dataset Statistics

Programming. For programming tasks, we evaluate on HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021). HumanEval consists of 164 hand-written Python programming problems, each accompanied by hidden unit tests and a small number of visible test cases. We additionally consider MBPP, which provides 974 crowd-sourced Python problems; following prior work, we use the 397 problems from the filtered evaluation split.

Math. For mathematical reasoning, we adopt the MATH dataset (Hendrycks et al., 2021b), which contains competitionstyle math problems spanning seven subjects including Algebra, Geometry, Number Theory, Counting and Probability, and Precalculus. We randomly sample a balanced subset across categories for evaluation.

Multi-hop QA. For multi-hop question answering, we use HotpotQA (Yang et al., 2018) and 2WikiMultiHopQA (Ho et al., 2020). In HotpotQA, we stratify by difficulty level and randomly sample 100 examples from each category (easy, medium, hard), yielding a total of 300 evaluation samples. For 2WikiMultiHopQA, we stratify by question type and randomly sample 75 examples from each of four categories (bridge comparison, comparison, compositional, inference), again yielding 300 samples in total. These stratified subsets ensure balanced evaluation across different reasoning styles.

# B.2. Finetuning the Parametric Module

Programming For programming tasks, we curate a dataset by sampling 4000 coding problems from the APP dataset (Hendrycks et al., 2021a) at introductory level. In addition, we synthesize 4200 problems using GPT-4o-mini, covering a diverse range of programming domains. The code templates and prompt used for data generation are provided in

Table 5. Datasets used for Programming, Math, and Multi-hop QA tasks.   

<table><tr><td>Task Type</td><td>Dataset Name</td><td>Size</td><td>Metric</td></tr><tr><td>Programming</td><td>HumanEval</td><td>164 problems, ~3 visible test cases/problem</td><td>Pass@ 1</td></tr><tr><td>Programming</td><td>MBPP</td><td>397 sampled problems</td><td>Pass@1</td></tr><tr><td>Math</td><td>MATH</td><td>278 sampled problems across 7 subjects</td><td>0-1 Acc</td></tr><tr><td>Multi-hop QA</td><td>HotpotQA</td><td>300 sampled problems (100 per difficulty)</td><td>0-1 Acc</td></tr><tr><td>Multi-hop QA</td><td>2WikiMultiHopQA</td><td>300 sampled problems (75 per type)</td><td>0-1 Acc</td></tr></table>

Table 6. Performance on HumanEval. Bold denotes the best result, and underline marks the second best. $\uparrow$ and $\downarrow$ indicate absolute change relative to the Base method. For clarity, the prompt token usage of the Base method is normalized to 1.   

<table><tr><td>Dataset</td><td>Method</td><td>Llama-3.1-70B-Instruct Pass@1</td><td>#Prompt Tokens</td><td>Qwen2.5-72B-Instruct Pass@1</td><td>#Prompt Tokens</td></tr><tr><td></td><td>Base</td><td>80.49</td><td>1.00</td><td>82.92</td><td>1.00</td></tr><tr><td></td><td>Model-based Reflection</td><td>87.80 ↑7.31</td><td>6.39</td><td>89.64 ↑ 6.72</td><td>3.48</td></tr><tr><td></td><td>Reflexion</td><td>90.24 ↑ 9.75</td><td>4.31</td><td>88.41 ↑5.49</td><td>3.48</td></tr><tr><td>HumanEval</td><td>DoT</td><td>90.85↑ 10.36</td><td>7.51</td><td>87.80 ↑4.88</td><td>6.05</td></tr><tr><td></td><td>DoT-bank</td><td>92.68 ↑ 12.19</td><td>9.14</td><td>90.24 ↑ 7.32</td><td>8.17</td></tr><tr><td></td><td>ParamAgent</td><td>92.07 11.58</td><td>11.90</td><td>93.90 ↑ 10.98</td><td>8.93</td></tr><tr><td></td><td>ParamAgent-plus</td><td>95.03↑14.54</td><td>19.47</td><td>95.12 ↑ 12.20</td><td>16.81</td></tr></table>

Figure 6. For each problem, GPT-4o-mini is further asked to produce potential mistakes along with buggy implementations. This yields a dataset of reflective signals and corresponding erroneous code examples. We then finetune LLaMA-3.1-8B with LoRA on this dataset to obtain the programming-specific parametric module $M _ { r }$ .

Math For mathematical reasoning, we leverage the MATH training set (Hendrycks et al., 2021b). From each subject area, we randomly sample 800 problems and adopt the same pipeline as in programming: GPT-4o-mini is prompted to produce reflective feedback and buggy derivations for each sampled problem. The resulting dataset is used to LoRA-finetune LLaMA-3.1-8B to instantiate $M _ { r }$ for math reasoning.

Multi-hop QA For multi-hop QA, we randomly sample 10000 instances from the HotpotQA (Yang et al., 2018) and 2WikiMultiHopQA (Ho et al., 2020) training sets respectively. GPT-4o-mini is prompted to output structured semantic units (e.g., entities, relations, constraints, answer types, and sub-questions) for each example. We then apply LoRA finetuning to LLaMA-3.1-8B on this dataset to build the parametric module $M _ { p }$ .

Across all domains, during dataset construction we provide one carefully designed demonstration example in the prompt to GPT-4o-mini. This ensures that the generated outputs (reflective feedback, buggy code, or semantic units) adhere to the required format, making the synthetic supervision more reliable.

# B.3. How does ParamAgent perform with stronger base LLMs?

We further study the performance of ParamAgent when paired with stronger base models of around 70B parameters. Specifically, we use Llama-3.1-70B and Qwen2.5-72B-Instruct as the underlying LLMs, while keeping the parametric module fixed as Llama-3.1-8B. We evaluate on HumanEval for programming and HotpotQA for multi-hop QA. The results are reported in Table 6 and Table 7 respectively.

Results. Across tasks, ParamAgent achieves performance that is on par with, or even surpasses, state-of-the-art baselines. Moreover, ParamAgent-plus consistently outperforms the best baseline methods by a large margin, highlighting the effectiveness of the parametric module. It is worth noting that our parametric module itself is only an 8B model, yet it integrates effectively with base LLMs as large as 70B. This demonstrates the strong potential of our approach when scaled further.

Table 7. Performance on HotpotQA dataset. Bold denotes the best result, and underline marks the second best. $\uparrow$ and $\downarrow$ indicate the absolute improvement or decrease relative to the Base method. For clarity, the prompt token usage of the Base method is normalized to 1.   

<table><tr><td>Dataset</td><td>Method</td><td colspan="2">Llama-3.1-70B-Instruct #Prompt Tokens</td><td colspan="2">Qwen2.5-72B-Instruct Acc #Prompt Tokens</td></tr><tr><td></td><td>Base</td><td>Acc 70.00</td><td>1.00</td><td>73.33</td><td>1.00</td></tr><tr><td rowspan="4">HotpotQA</td><td>Model-based CoT</td><td>73.67 ↑ 3.67</td><td>1.43</td><td>74.10 ↑1.05</td><td>1.44</td></tr><tr><td>Reflexion</td><td>82.33 ↑ 12.33</td><td>3.02</td><td>82.67 ↑ 9.34</td><td>2.81</td></tr><tr><td>DoT</td><td>73.67 ↑ 3.67</td><td>3.43</td><td>80.67 ↑ 7.34</td><td>4.30</td></tr><tr><td>DoT-bank</td><td>80.0010.00</td><td>5.24</td><td>82.33 9.00</td><td>7.87</td></tr><tr><td></td><td>ParamAgent</td><td>84.00↑14.00</td><td>7.70</td><td>81.00 ↑ 7.67</td><td>7.90</td></tr><tr><td></td><td>ParamAgent-plus</td><td>89.67 ↑ 19.67</td><td>13.69</td><td>84.67 ↑11.34</td><td>15.43</td></tr></table>

Table 8. Token usage and cost on HumanEval and HotpotQA datasets with Llama3.1-8B as backbone LLM. Best and second-best metrics are in bold and underline respectively.   

<table><tr><td rowspan="2">Method</td><td colspan="4">HumanEval</td><td colspan="4">HotpotQA</td></tr><tr><td>#Prompt Tokens</td><td>#Completion Tokens</td><td>Total Cost ($)</td><td>Pass@1 (%)</td><td>#Prompt Tokens</td><td>#Completion Tokens</td><td>Total Cost ($)</td><td>Acc (%)</td></tr><tr><td>Base</td><td>37,463</td><td>13,506</td><td>0.00917</td><td>59.15</td><td>164,013</td><td>1,801</td><td>0.02985</td><td>57.67</td></tr><tr><td>Model-based Reflection</td><td>342,805</td><td>82,280</td><td>0.07652</td><td>78.05</td><td>236,548</td><td>1,212</td><td>0.04280</td><td>61.67</td></tr><tr><td>Reflexion</td><td>348,068</td><td>73,538</td><td>0.07589</td><td>76.22</td><td>703,192</td><td>68,612</td><td>0.13892</td><td>71.33</td></tr><tr><td>DoT</td><td>653,981</td><td>169,986</td><td>0.14831</td><td>72.56</td><td>1,164,812</td><td>106,806</td><td>0.22889</td><td>66.67</td></tr><tr><td>DoT-bank</td><td>926,047</td><td>233,016</td><td>0.20863</td><td>79.88</td><td>2,179,148</td><td>195,283</td><td>0.42740</td><td>72.00</td></tr><tr><td>ParamAgent</td><td>814,627</td><td>163,257</td><td>0.17602</td><td>82.93</td><td>3,649,598</td><td>128,010</td><td>0.67997</td><td>78.33</td></tr></table>

# B.4. Cost Analysis

Table 8 reports prompt/completion tokens and costs using Llama-3.1-8B. Costs are computed with TogetherAI pricing as of Aug 20, 2025 (\$0.18 per million tokens). We can see that Model-based Reflection (CoT) is highly efficient, achieving strong accuracy with far fewer tokens than reflection-heavy methods like DoT-bank. By contrast, ParamAgent delivers the best results on both HumanEval and HotpotQA, at higher but still moderate cost, this highlights the advantages of incorporating various forms of memory modules.

# B.5. A Case Study

We present a case study from the MBPP dataset, where both Reflexion and DoT fail to generate the correct implementation, while ParamAgent succeeds. To better understand this difference, we analyze the reflective history of all three methods and highlight the gists, as illustrated in Figure 8.

From the analysis, we observe that Reflexion and DoT often produce unhelpful sometimes even misleading reflections, which push the agent further away from the correct solution. In contrast, ParamAgent generates fewer such misleading reflections. We hypothesize that this advantage arises from the parametric knowledge encoded in $M _ { r }$ , which helps ParamAgent avoid unhelpful or error-prone reflective signals.

# B.6. Prompt Templates

We provide prompt templates used in ParamAgent across different domains. The 1-shot reflective example for programming tasks can be found in Figure 9, and the corresponding math reasoning template in Figure 10. For multi-hop QA, the semantic decomposition 1-shot example is shown in Figure 11.

Instruction templates for generating actions for the programming is shown in Figure 12, the math reasoning instruction in Figure 13, and the multi-hop QA instruction in Figure 14.

![](images/fc4b343da297d764022bc7e4cb0e3545d8c81e5aedde65866e85a117abf324a6.jpg)  
Figure 6. Schema of categories for synthesizing programming tasks used in our parametric module construction.

![](images/0752f0cb5459ef830247362ef7c2928c72e15b56ea163d0d11865944c9bb44f1.jpg)  
Figure 7. Prompt for synthesizing programming tasks

# Reflexion.

Problem: "The implementation failed the test case where it tries to return a single element that is not a list. This issue arises because the ^ operator in the set difference operation returns a set, and then we explicitly convert it to a list..." Problem: "The implementation failed the test where it is expected to find a difference in the lists because the list to check for difference is a subset of the original list... This would return the union of the two lists, not the actual difference."

Problem: "The implementation failed the test where the difference should be a single element because it uses the symmetric difference operation which is not what’s intended here... When converting the set back to a list using the list function, the order of elements is lost."

Problem: "When converting the set back to a list using the list function, the order of elements is lost."

# DoT.

Problem: "The implementation failed the test due to a mismatch in the type of the function return value and the expected output."

Problem: "The function implementation does not handle the case where the input lists are not of the same length or contain duplicate elements."

Problem: "The function implementation does not correctly calculate the symmetric difference between two lists."

Problem: "The implementation failed due to a mismatch in the type of the function return value and the expected output, but the proposed fix in the previous reflection is incomplete."

Problem: "The function implementation does not correctly handle the case where the input lists are not of the same length or contain duplicate elements...

Problem: "The function implementation does not correctly remove duplicates from the input lists, leading to incorrect results."

# ParamAgent.

Problem: "The function implementation does not handle the case of empty lists correctly." Fix: "Add a check at the beginning of the function to return an empty list if either of the input lists is empty..."

Problem: "The function implementation does not preserve the original order of elements in the lists." Fix: "Instead of using set operations..., use a list comprehension to create the difference list..."

Problem: "The function implementation does not handle non-integer values in the lists correctly." Fix: "Add a check at the beginning... ensure that both input lists are lists of integers..."

Problem: "The function implementation does not handle non-list types correctly." Fix: "Add a check... ensure that both input arguments are lists..."

Problem: "The function implementation does not handle duplicates correctly." Fix: "Use a list comprehension... and preserve the original order..."

Problem: "The function implementation does not handle the case where the lists are identical." Fix: "Return an empty list if the two input lists are identical."

Figure 8. Selected reflections on the mbpp_769_Diff problem. Red $=$ critical errors that directly invalidate the implementation; Orange $=$ conceptual pitfalls or misleading reasoning patterns; Green $=$ key insights leading to a working solution; Gray $=$ background noise (irrelevant or stylistic).

![](images/cb9a212e8bbeff5a5b793205e5a138a0abf35361fdfb850aca12d8a548a103b1.jpg)  
Figure 9. 1-shot example for reflective dataset construction for programming task.

Question. Circle $O$ is located on the coordinate plane with center at $( 2 , 3 )$ . One endpoint of a diameter is at $( - 1 , - 1 )$ What are the coordinates of the other endpoint of this diameter? Express your answer as an ordered pair.

# Pitfalls & Potential Mistakes

1. Confusing the center with an endpoint. Assuming the center is an endpoint leads to an incorrect reflection point.   
2. Incorrect use of the midpoint formula. Forgetting that the center is the midpoint of the diameter, or solving $( x + x _ { 2 } ) / 2 = \mathrm { c e n t e r } _ { x }$ incorrectly.   
3. Using the wrong coordinates for the midpoint. Plugging endpoint coordinates in place of the center (or vice versa) yields the wrong unknowns.   
4. Arithmetic errors. Sign or algebra mistakes when solving, e.g. $2 = ( - 1 + x ) / 2 \Rightarrow x = 3$ (incorrect) instead of $x = 5$ .   
5. Switching $x$ and $y$ . Mixing $x \cdot$ - and $y$ -midpoint formulas, or using $x$ values to solve for $y$ .   
6. Incorrect interpretation of the diameter. Thinking the diameter extends in the same direction from the center; doubling the vector or reflecting in the wrong direction.

Figure 10. 1-shot example for reflective dataset construction in math reasoning.

# Example 1

Question. Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

# Question Parsing and Intent Extraction

# Key Components

• Entity A: Anatoly Maltsev — mathematician/logician; contributions in mathematical logic and abstract algebra.   
• Entity B: Valentin Turchin — computer scientist/philosopher; work in cybernetics and philosophy of science.   
• Implied Relationship: Comparative inquiry: which individual is more closely associated with mathematics.   
• Answer Type Expected: Person name (e.g., “Anatoly Maltsev”).   
• Reasoning Type: Comparative factual reasoning.   
• Required Background: Biographical profiles or retrieved professional records.

# Inference Trace

1. Retrieve factual data about Maltsev’s and Turchin’s primary academic domains.   
2. Classify Maltsev as a mathematician (core contributions to mathematical logic).   
3. Classify Turchin as mainly in cybernetics and philosophy.   
4. Eliminate Turchin as the primary mathematician.   
5. Conclude: Anatoly Maltsev.

# Disambiguation Note

Nationality (Russia) does not help differentiate them.

Figure 11. 1-shot example used in ParamAgent for semantic decomposition dataset construction in multi-hop QA.

You are an AI Python assistant. You will be given some potential pitfalls and several flawed implementations for the coding challenge, as well as your previous implementation of a function, a series of unit-test results, and your self-reflection on your previous implementation. Try to avoid the errors from your previous implementation and the listed pitfalls.

Instruction: ALWAYS WRITE your full implementation (restate the function signature).

Figure 12. Instruction prompt used by ParamAgent to generate next-round solutions for programming tasks.

You are revising your previous answer to a mathematics problem.

You will receive: (1) the original question, (2) potential mistakes and pitfalls, (3) your last answer, (4) feedback (Right or Wrong) explaining why that answer was unsatisfactory, and (5) your brief self-reflection on the mistake.

Respond with: 1. Reasoning: updated step-by-step thoughts.

2. Answer: the corrected final result.

Formatting: The final answer should be simplified to its simplest form, e.g., 25, $2 5 _ { 1 6 }$ , $\scriptstyle { \frac { 1 } { 3 6 } }$ , etc.

Figure 13. Instruction prompt used by ParamAgent to generate next-round solutions for math reasoning.

You are revising your previous answer to a multi-hop QA question. You will receive: (1) the original question, (2) some key points, the underlying intent, and possible inference patterns that facilitate answering this question, (3) your last answer, (4) supporting context, (5) feedback (Right or Wrong) explaining why that answer was unsatisfactory, (6) your brief self-reflection on the mistake.

Instruction: Based on the inputs, produce a new single-phrase answer that resolves the error and fully answers the question. Output only the answer — no commentary, no code.

Figure 14. The prompt of ParamAgent to generate next-round answers for multi-hop QA tasks.