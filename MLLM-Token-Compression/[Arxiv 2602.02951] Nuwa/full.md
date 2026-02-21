# NUWA ¨ : MENDING THE SPATIAL INTEGRITY TORN BY VLM TOKEN PRUNING

Yihong Huang1,2, Fei ${ \bf M } { \bf a } ^ { 1 \ast }$ ∗, Yihua $\mathbf { S h a o ^ { 3 } }$ , Jingcai $\mathbf { G u o ^ { 3 } }$ , Zitong $\mathbf { Y u } ^ { 4 }$ , Laizhong ${ \bf { C u i } } ^ { 5 }$ , Qi Tian1,6

1Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)   
2School of Artificial Intelligence, Xidian University   
3The Hong Kong Polytechnic University 4Great Bay University   
5Shenzhen University 6Huawei   
huangyihong@stu.xidian.edu.cn,mafei@gml.ac.cn

# ABSTRACT

Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM’s processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens’ positional information. Motivated by these findings, we propose Nuwa, a two-stage token pruning framework ¨ that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that Nuwa achieves SOTA performance on multiple ¨ VQA benchmarks (from $94 \%$ to $9 5 \%$ ) and yields substantial improvements on visual grounding tasks (from $7 \%$ to $47 \%$ ).

Code: https://github.com/Man-PaperRejected/Nuwa

![](images/dbc876586b8bd01b54a0564fdc67dd3e8452eecf4ee4f2b7991700cab3da6fc7.jpg)  
Figure 1: Nuwa Performance On VQA and VG tasks, preserving¨ $9 5 \%$ and $47 \%$ under $8 8 . 9 \%$ reduction of vision tokens. (a) Our Nuwa outperforms current efficient VLMs on 10 VQA benchmarks; ¨ (b) On 3 visual grounding benchmarks, Nuwa also achieves SOTA results. ¨

# 1 INTRODUCTION

VLMs (Liu et al., 2024a; Bai et al., 2025; Zhu et al., 2025) exhibit strong multimodal capabilities through pre-training on massive image-text pairs. However, the large number of vision tokens generated during inference leads to substantial computational overhead and reduced throughput. Recent visual token pruning methods aim to accelerate inference while preserving model performance. These include approaches based on visual-semantic similarity (Yang et al., 2024; Li & Shin, 2024; Jeddi et al., 2025), textual semantic filtering (Chen et al., 2024; Xing et al., 2024; Endo et al., 2024), and multi-stage pruning (Zhang et al., 2025a; Liu et al., 2025b), which improve inference throughput. Additionally, token merging (Bolya et al., 2023), a technique within token pruning, increases token sparsity in VLMs during inference. Fundamentally, reducing the number of tokens enhances sparsity and thereby improves VLM’s inference throughput.

![](images/538f0f9c5a3106638b933c1739a85b22a1e13a21bd8a19f7369e4cbecdf6a4ce.jpg)  
Figure 2: The left panel contrasts our Nuwa framework with prior token pruning methods. (a) ¨ Pruning at the vision encoder stage; (b) Text-guided pruning within the LLM; (c) Our two-stage approach: initial spatial-aware pruning via local aggregation that preserves global anchors in the vision encoder, followed by text-guided refinement in the LLM.

Nevertheless, recent studies (Wen et al., 2025; Endo et al., 2024) have questioned the effectiveness of existing pruning methods (Chen et al., 2024; Zhang et al., 2025b). In particular, random pruning and pooling-based merging can achieve competitive performance, yet these methods exhibit substantial degradation on visual grounding (VG) tasks compared with visual question and answering (VQA) tasks (Long et al., 2025; Shao et al., 2025a). To assess whether these issues are widespread, we systematically categorize existing pruning methods and compare them with simpler baselines across multiple datasets. Our experiments confirm that these limitations persist. These findings raise fundamental questions: $\bullet$ Why do existing pruning methods exhibit significant task-dependent degradation? ❷ How is vision information encoded and utilized within the VLM’s processing pipeline? $\cdot$ How to Mend grounding performance gaps in VLM’s token pruning setting?

Through systematic experimental analysis, we uncover that VLMs employ a multi-stage visual processing pipeline that progresses from global to fine-grained integration, with task-specific requirements. In particular, grounding tasks depend on preserving global spatial reference frames, which are constructed from token position information and can be disrupted by token pruning. Informed by these insights, we introduce Nuwa ¨ , as shown in Figure 2, a two-stage spatial-aware token pruning framework, patching up the torn spatial integrity. The first stage operates in the visual semantic space to reduce token redundancy while maintaining spatial topology. It employs a Boids-inspired algorithm (Reynolds, 1998) with three operations: (1) Separation: partitioning the token map into localized regions; (2) Alignment: selecting representative tokens based on their alignment with the global context and information density; and (3) Aggregation: merging features of neighboring tokens around representatives using semantic similarity. The second stage performs text-guided refinement in the intermediate layers of the LLM after multimodal feature alignment, using textual semantics to guide further pruning.

Nuwa ¨ demonstrates significant improvements, as shown in Figure 1, on VG benchmarks $( 7 \%  4 7 \%$ , $18 \%  7 5 \%$ ) across multiple pruning configurations in LLaVA-1.5, alongside enhancements in VQA benchmarks, including image reasoning and understanding performance $( 9 4 \% \to 9 5 \%$ ), and validates its effectiveness across additional models.

Our contributions are as follows:

1. Task-specific Analysis: We systematically examine VLM’s processing pipelines and show that current pruning methods fail on grounding tasks by overlooking task-specific requirements and disrupting spatial structure. Position reconstruction experiments confirm that spatial perception arises from the integrity of the global reference frame.

2. Nuwa Framework: ¨ We propose a two-stage spatial-aware pruning framework that retains global spatial anchors through separation and adaptive region aggregation, thereby preserving both spatial and semantic integrity. It further leverages textual information in the LLM for multimodal alignment-based pruning.

3. Performance Validation: Our approach yields superior results across 13 datasets and multiple VLMs, establishing new SOTA on VQA $9 5 \%$ performance retention) and VG $4 7 . 2 \%$ performance retention) tasks while achieving $89 \%$ reduction in TFLOPs and $62 \%$ reduction in prefill time with a $8 8 . 9 \%$ tokens reduction.

# 2 DISSECTING THE VISUAL PROCESSING PIPELINE: FROM SEMANTIC FLOW TO SPATIAL INTEGRITY

In this section, we first perform a systematic analysis $( { \mathrm { S e c } } 2 . 1 )$ of existing pruning methods to address two key questions. We then examine the visual information processing pipeline (Sec 2.2) in VLMs through two analytical experiments, tracing the progression from global attention mechanisms to local processing paradigms. Finally, position reconstruction experiments (Sec 2.3) uncover the root causes of performance degradation in grounding tasks, thereby providing insights for the design of pruning methods.

2.1 EVALUATING COMPETITIVE ADVANTAGES: SIMPLE BASELINES VERSUS ADVANCED PRUNING METHODS

Recent research (Wen et al., 2025; Endo et al., 2024) has questioned the effectiveness of existing visual token pruning methods. To investigate two key aspects — (1) Generalization: whether advanced methods consistently outperform simple baselines, and (2) Robustness: whether performance remains stable across tasks with diverse requirements, we conduct a comprehensive crosstask evaluation.

Experimental Setup We conduct a comprehensive evaluation across 12 datasets, covering a broad spectrum of capabilities including image grounding, fine-grained understanding, and complex reasoning. To facilitate a systematic comparison, we categorize mainstream visual token pruning methods into three distinct families based on their architectural placement and operation stage: Vision Encoder-Side Pruning, which focuses on reducing redundancy within or at the output of the vision encoder to save memory early on (e.g., VisionZip (Yang et al., 2024), PruMerge (Shang et al., 2024)); LLM Single-Layer Pruning, which applies a one-time, fixed-ratio pruning operation at specific layers within the LLM (e.g., FastV (Chen et al., 2024)); and LLM Multi-Layer Pruning, which dynamically identifies and removes non-essential vision tokens across consecutive LLM layers (e.g., PyramidDrop (Xing et al., 2024), SparseVLM (Zhang et al., 2025b)). To ensure a fair and rigorous assessment, we benchmark these sophisticated methods against two simple yet effective baselines, random sampling and average pooling, to determine the value of complex pruning designs.

Table 1: Performance comparison of various vision token pruning methods on LLAVA1.5-7B. Including LLM Single-Layer Pruning, LLM Multi-Layer Pruning, and Vision Encoder-Side Pruning.   

<table><tr><td>Method</td><td>Source</td><td>GQA</td><td>MMB</td><td>MMMU</td><td>MME</td><td>VQAv2</td><td>VQAtext</td><td>POPE</td><td>SQA</td><td>MMVet</td><td>Avg (%)</td></tr><tr><td>Vanilla</td><td>CVPR&#x27;24</td><td>61.9</td><td>64.7</td><td>36.3</td><td>1862</td><td>78.5</td><td>58.2</td><td>85.9</td><td>69.5</td><td>31.1</td><td>100.0</td></tr><tr><td>FastV</td><td>ECCV&#x27;24</td><td>46.1 (74.5%)</td><td>48.0 (74.2%)</td><td>34.0 (93.7%)</td><td>1255 (67.4%)</td><td>55.0 (70.1%)</td><td>47.8 (82.1%)</td><td>59.6 (69.4%)</td><td>68.7 (98.8%)</td><td>23.3 (74.9%)</td><td>78.3</td></tr><tr><td>Random</td><td></td><td>51.2 (82.6%)</td><td>41.8 (64.5%)</td><td>34.1 (94.0%)</td><td>1351 (72.6%)</td><td>65.4 (83.3%)</td><td>44.9 (77.1%)</td><td>61.1 (71.1%)</td><td>66.8 (96.1%)</td><td>16.9 (54.3%)</td><td>77.3</td></tr><tr><td>Poling</td><td></td><td>52.22(84.4%)</td><td>48.7 (75.3%)</td><td>34.0 (93.7%)</td><td>1380 74.1%)</td><td>69.1 (88.0%)</td><td>45.3 (77.9%)</td><td>67.8 (78.9%)</td><td>67.9 (97.7%)</td><td>16.3 (52.4%)</td><td>80.3</td></tr><tr><td>PDrop</td><td>CVPR&#x27;25</td><td>41.9 (67.7%)</td><td>33.3 (51.5%)</td><td>26.5 (73.0%)</td><td>1092 (58.6%)</td><td>57.3 (73.0%)</td><td>45.9 (78.9%)</td><td>55.9 (65.1%)</td><td>69.2 (99.6%)</td><td>24.9 (80.1%)</td><td>72.0</td></tr><tr><td>SparseVLM</td><td>ICML&#x27;25</td><td>53.8 (86.9%)</td><td>60.1 (92.9%)</td><td>35.4 (97.6%)</td><td>1589 (853%)</td><td>68.2 (86.9%)</td><td>53.4 (9.8%)</td><td>77.5 (90.2%)</td><td>69.8 (100.4%)</td><td>24.9 (80.1%)</td><td>90.2</td></tr><tr><td>Random</td><td></td><td>51.5 (83.2%)</td><td>46.0 (71.2%)</td><td>34.1 (94.0%)</td><td>1342 (72.1%)</td><td>67.1 (85.5%)</td><td>46.7 (80.2%)</td><td>71.8 (83.6%)</td><td>68.1 (98.0%)</td><td>23.1 (74.3%)</td><td>82.5</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>55.1 (89.0%)</td><td>60.1 (92.9%)</td><td>36.2 (99.7%)</td><td>1690 (90.8%)</td><td>72.4 (92.2%)</td><td>55.5 (95.4%)</td><td>77.0 (89.6%)</td><td>69.0 (99.3%)</td><td>31.7 (101.9%)</td><td>94.5</td></tr><tr><td>PruMerge+</td><td>ICCV&#x27;25</td><td>55.4 (89.5%)</td><td>59.6 (92.1%)</td><td>35.8 (98.6%)</td><td>1616 (86.8%)</td><td>71.3 (90.8%)</td><td>52.0 (89.3%)</td><td>75.7 (88.1%)</td><td>69.5 (100.0%)</td><td>28.0 (90.0%)</td><td>91.7</td></tr><tr><td>Random</td><td></td><td>54.3 (87.7%)</td><td>51.1 (79.0%)</td><td>34.0 (93.7%)</td><td>1410 (75.7%)</td><td>66.2 (84.3%)</td><td>46.5 (79.9%)</td><td>68.2 (79.3%)</td><td>65.5 (94.2%)</td><td>21.1 (68.0%)</td><td>82.4</td></tr><tr><td>Pooling</td><td></td><td>51.5 (83.1%)</td><td>44.4 (68.6%)</td><td>32.1 (88.4%)</td><td>1151 (61.8%)</td><td>68.1 (86.8%)</td><td>42.9 (73.8%)</td><td>68.0 (79.2%)</td><td>64.7 (93.1%)</td><td>18.7 (60.1%)</td><td>77.2</td></tr></table>

Our results reveal key patterns across task types. On general-purpose VQA benchmarks (Table 1), simple baselines achieve competitive performance, often matching advanced pruning methods. In contrast, on object-centric grounding tasks (Table 2), all methods show systematic

Table 2: Performance comparison on RefCOCO series datasets.   

<table><tr><td>Avg Tokens</td><td>Method</td><td>Refcoco-test</td><td>Refcoco+-testA</td><td>Refcoco+-testB</td><td>Refcocog-test</td></tr><tr><td>576</td><td>LLaVA</td><td>58.30</td><td>59.43</td><td>38.88</td><td>48.50</td></tr><tr><td rowspan="4">128</td><td>FastV</td><td>10.34</td><td>8.53</td><td>9.83</td><td>8.87</td></tr><tr><td>SparseVLM</td><td>6.27</td><td>55.79</td><td>4.22</td><td>6.35</td></tr><tr><td>VisionZip</td><td>4.49</td><td>4.6</td><td>4.86</td><td>3.50</td></tr><tr><td>Pooling</td><td>23.01</td><td>24.37</td><td>15.04</td><td>19.69</td></tr><tr><td rowspan="4">64</td><td>FastV</td><td>2.73</td><td>1.17</td><td>1.02</td><td>2.19</td></tr><tr><td>SparseVLM</td><td>1.04</td><td>0.96</td><td>1.28</td><td>0.61</td></tr><tr><td>VvisionZip</td><td>4.04</td><td>3.73</td><td>3.86</td><td>3.38</td></tr><tr><td>Poling</td><td>112.01</td><td>12.20</td><td>7.55</td><td>11.40</td></tr></table>

performance degradation, regardless of design complexity. Notably, average pooling yields the best results among pruning approaches, likely because it partially preserves spatial structural features.

Finding 1 Advanced pruning methods provide limited benefits over simple baselines on VQA tasks, whereas all methods suffer systematic degradation on grounding tasks, with average pooling achieving the best performance.

# 2.2 UNVEILING TASK-DEPENDENT VISUAL PROCESSING PIPELINE

Building on the task-dependent performance degradation observed in Sec. 2.1, prior explainability studies on LLMs and VLMs (Selvaraju et al., 2016; Ding et al., 2017; Zhang et al., 2024; Yin et al., 2025) have not sufficiently explored how visual processing adapts to shifts in task focus, such as from VQA to VG. To address this, we conduct two analytical experiments: visualizing attention flows from the final token to vision tokens during decoding, and applying gradient-weighted attribution methods to trace critical visual information pathways across tasks. Additionally, we evaluate the model’s object-centric perception at different stages using two fine-grained metrics.

![](images/0c3cf5d10290a1e23c30ac3c5d90fb8fb3ff3c55a80e24269f337529c3ea254d.jpg)  
Figure 3: (a) to (d) show different types of attention flows (First row) and gradient-weighted attention flows (Second row), where A-to-B means the degree of attention A pays to B. (e) shows the differences in Last-to-Vision attention maps across different tasks. VLMs exhibit a two-stage visual processing pipeline, with task-independent multimodal interactions in early layers and task-specific processing in middle layers.

Figure 3 depicts task-dependent characteristics of visual processing in VLMs. Panels (a) and (b) at the first row show that attention flows exhibit distinct early and mid-stage phases. However, gradientweighted analysis (Second row) reveals a pronounced task-dependent divergence in the mid-stage, underscoring the model’s sensitivity to task requirements during visual integration — with VG tasks showing greater reliance on vision tokens. Panel (c) highlights a task-independent aspect: early multimodal interactions, suggesting universal visual processing in initial stages. Panel (d) illustrates task-varying differences in text information handling. Further experiments on attention blocking (Appendix B.5) indicate that, in VG tasks, textual cues extract critical visual details, resulting in unique last-to-text attention patterns.

Visual Attention Entropy And Object-Centric Cohesion: Attention flows offer insights into the model’s information processing dynamics. To further quantify the multi-stage visual processing pipeline identified in the prior analysis and its task-dependent characteristics, we introduce two finegrained metrics: Visual Attention Entropy (VAE) and Object-Centric Cohesion (OCC). VAE measures the distribution of information in the visual self-attention mechanism by computing the average Shannon entropy across visual tokens (Eq. (1)). High VAE values indicate diffuse, global attention patterns, whereas low values reflect concentrated, local focus. Complementing this, OCC assesses object-level feature cohesion by calculating the Intersection over Union (IoU) between ground-truth object tokens and the top- $k$ tokens most similar to the object’s center token (Eq. (2)). Higher OCC scores denote stronger localization of features to relevant objects, capturing fine-grained processing.

$$
H ( v _ { i } ) = - \sum _ { j = 1 } ^ { i - 1 } p ( v _ { j } | v _ { i } ) \log _ { 2 } p ( v _ { j } | v _ { i } ) , \quad { \mathrm { V A E } } = { \frac { 1 } { N - 1 } } \sum _ { i = 2 } ^ { N } H ( v _ { i } )
$$

$$
\mathrm { O C C } ( { \mathcal { O } } ) = \frac { | V _ { k } ^ { \mathrm { m o d e l } } \cap V _ { \mathcal { O } } | } { | V _ { k } ^ { \mathrm { m o d e l } } \cup V _ { \mathcal { O } } | }
$$

![](images/6111a36ba7ee141adac2f15c8b654c928be06ddc395d3af6898157c5b442a2e5.jpg)  
Figure 4: Visualization of VLM’s Two-Stage Vision Tokens Processing: (a) Layer-wise Analysis of VAE and OCC Metrics; (b) Layer-wise Instance Heatmap Visualization. Both demonstrate finegrained feature extraction at the mid-stage.

As shown in Figure 4, the VAE of the ViT encoder exhibits a decreasing trend in the middle stage, indicating a gradual shift from global context integration to fine-grained feature extraction. In contrast, the VAE of the LLM decoder fluctuates after a sharp initial increase, suggesting a more complex process of reorganizing visual features and integrating them into the textual semantic space. The OCC scores provide a clearer explanation — they peak in the middle stage of both ViT and LLM, signifying the formation of object-level representations. This phenomenon also effectively explains the earlier observation: why grounding tasks demand such high levels of visual information at this stage.

Finding 2 Visual processing in VLMs unfolds through a multi-stage pipeline, progressing from global semantic integration to fine-grained object-centric focus, with task-specific reliance on vision tokens. Grounding tasks require heightened visual integration during middle stages for spatial reasoning, in contrast to the reduced demands in image understanding tasks.

# 2.3 SPATIAL INTEGRITY: RECONSTRUCTING THE GLOBAL REFERENCE FRAME

Building on the mid-stage visual integration demands in Sec. 2.2, where pruning disrupts taskspecific vision reliance, we hypothesize that spatial integrity — via the Global Spatial Reference Frame from position embeddings — is essential for spatial perception, as pooling methods’ superior grounding performance indicates. To validate this, we design experiments restoring integrity through modified position embedding strategies.

# 2.3.1 A TAXONOMY OF POSITION EMBEDDING STRATEGIES

To rigorously test our hypothesis, we first deconstruct the implicit position embedding (PE) handling strategies within existing pruning methods, as shown in Figure 5, abstracting them into three distinct paradigms:

Position Embedding Range Compression (PERC): Compresses the PE of pruned tokens into a tiny range, missing the global reference frame, like Visionzip.

Position Embedding Sparse Preservation (PESP): Retains the original PE for each pruned token, forming a sparse subset within an incomplete spatial frame, like FastV.

Relative Position Mapping Extension (RPME): Preserves the relative spatial distance of the pruned tokens and extends their PE via linear mapping, to span the entire original range and retain the spatial integrity.

![](images/0c2b57cc99297491dc110ec70108a12da5b7838b72bf228b920ad80ac384b82c.jpg)  
Figure 5: Sketch of different Position Embedding Strategies. RPME retains the spatial integrity.

Experiment Setup We select two representative methods, VisionZip (PERC) and FastV (PESP), replacing their PE strategy with RPME, and then evaluate the performance of these “fixed” models on visual grounding benchmarks.

Table 3: Position Reconstruction Experiment on Refcoco series and VQA Benchmarks. The symbols $\cdot _ { + } \cdot$ and $\cdot _ { - } ,$ indicate changes relative to the pre-reconstruction values showed in Table 2.   

<table><tr><td>Method</td><td>Refcoco -test</td><td>Refcoco -val</td><td>Refcoco+ -testA</td><td>Refcoco+ -testB</td><td>Refcoco+ -val</td><td>Refcocog -test</td><td>Refcocog -val</td><td>GQA</td><td>MMB</td><td>VQAv2</td><td>MME</td></tr><tr><td colspan="10">56.42</td><td>78.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Average 64 Tokens</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">10.50 (+6.69) 9.27 (+5.54) 7.57 (+3.71) 8.62 (+5.12) 4.31 (+1.90)</td></tr><tr><td>Fastv-fix</td><td>11.57 (+7.53) 4.52 (+1.79)</td><td>4.11 (+2.10)</td><td>3.84 (+2.67)</td><td>2.75 (+1.73)</td><td>8.19 (+4.81) 4.17 (+1.98)</td><td></td><td>8.31 (+5.10) 4.22 (+2.21)</td><td>55.6 (+0.5) 46.2 (+0.1)</td><td>61.8 (+1.7) 47.8 (-0.2)</td><td>70.6 (-1.8) 54.1 (-0.9)</td><td>1700 (+10) 1247 (-8)&#x27;</td></tr><tr><td>Pooling</td><td>12.01</td><td>11.84</td><td>12.20</td><td>7.55</td><td>10.50</td><td>11.40</td><td>9.85</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Average 128 Tokens</td></tr><tr><td>VisionZip-fix 21.39 (+16.90)</td><td></td><td>21.04 (+16.93)</td><td>19.96 (+15.90)</td><td>13.45 (+8.59)</td><td>16.10 (+12.22)</td><td>15.69 (+12.19)</td><td>15.52 (+12.04)</td><td>58.5 (+0.9)</td><td>63.4 (+1.4)</td><td>74.3 (-1.3)</td><td>1751 (-10)</td></tr><tr><td>Fastv-fix</td><td>13.41 (+3.07)</td><td>13.24 (+3.11)</td><td>11.69 (+3.16)</td><td>12.29 (+2.46)</td><td>14.55 ((+3.31)</td><td>12.02 (+3.15)</td><td>11.87 (+3.45)</td><td>51.3 (+0.8)</td><td>57.7 (+1.6)</td><td>60.3 (-1.5)</td><td>1494 (+4)</td></tr><tr><td>Pooling</td><td>23.01</td><td>22.67</td><td>24.37</td><td>15.04</td><td>17.88</td><td>19.69</td><td>19.03</td><td></td><td></td><td></td><td></td></tr></table>

Results in Table 3 show that RPME yields notable improvements across benchmarks: VisionZip achieves gains of $5 . 6 \%$ and $1 3 . 4 \%$ in two settings, while FastV sees more modest increases of $1 . 8 \%$ and $3 . 2 \%$ . These differences confirm our analysis: PERC in VisionZip eliminates positional information, whereas PESP in FastV preserves absolute coordinates but disrupts spatial continuity. Gains grow with larger token budgets, underscoring the increasing importance of complete spatial frameworks for richer visual organization. Pooling methods outperform others consistently by aggregating features on coarse grids that implicitly maintain global topology, reinforcing that reconstructing continuous spatial coordinates is vital for grounding tasks. This strategy has a negligible impact on image understanding and reasoning benchmarks, indicating broad applicability.

Finding 3 The degradation of VLMs on grounding tasks is principally driven by the loss of Global Spatial Reference Frame within token pruning strategies, which can be restored by preserving global position embedding.

# 3 METHODOLOGY

Our analysis reveals that existing token pruning methods fail on spatial localization tasks by disrupting the global spatial reference frame. This motivates three core design principles for effective visual token compression: (1) preserving spatial uniformity to ensure consistent coverage; (2) aggregating redundant information in a vision-centric, cohesive manner while retaining local salience (Stage1); and (3) applying text-modulated fine-grained filtering to select task-relevant tokens based on textual semantics (Stage2). We apply these principles in Nuwa, a two-stage pruning framework. ¨

3.1 STAGE 1: SPATIAL COHESION PRUNING IN THE VISION ENCODER

This stage reduces the initial $N ^ { 2 }$ visual tokens in the vision encoder to a dense, spatial-preserving sequence via three sequential operations.

# 3.1.1 SEPARATION VIA GRID PARTITIONING

To maintain spatial integrity, we partition the input token grid $\mathcal { T } = \{ t _ { 1 } , t _ { 2 } , \dots , t _ { N ^ { 2 } } \}$ into $M \times M$ non-overlapping local regions $\mathcal { R } _ { i , j }$ . Subsequent selection and aggregation occur at the region level, enabling a complete global coordinate system.

# 3.1.2 ALIGNMENT VIA SALIENCE IDENTIFICATION

Within each region $\mathcal { R }$ , we select representative benchmark tokens as aggregation centers. These tokens should exhibit high global salience; we initially use attention scores from the [CLS] token. However, analysis indicates sparse distributions in deeper vision encoder layers. To mitigate this, we incorporate information capacity, defined as the L2-norm of the token’s key vector $( \left. \mathbf { k } _ { i } \right. _ { 2 } )$ , as a secondary criterion. The resulting salience score $S ( t _ { i } )$ for token $t _ { i }$ is the product of its global attention score and information capacity:

$$
S ( t _ { i } ) = \alpha _ { \mathrm { c l s } , i } \cdot | \mathbf { k } _ { i } | _ { 2 }
$$

where $\alpha _ { \mathrm { c l s } , i }$ is the attention weight from the [CLS] token. In each local region $\mathcal { R } _ { k }$ , we select the $k$ tokens with the highest salience scores to form the Benchmark Token set $\mathcal { T } _ { B }$ .

![](images/f0ae295dee778356f05136ec3252088236cb62b42911a1c8fcbc490298bd465c.jpg)  
Figure 6: The Framework of Nuwa: (a) Stage 1 Pruning regarding Separation, Alignment and Cohe- ¨ sion; (b) Layer-wise 2D visualization of text-visual token similarity during LLM; (c) Stage2 pruning based on text semantics at LLM mid-stage; (e) prefill time of Nuwa across different scales. ¨

# 3.1.3 AGGREGATION VIA SPATIAL PROXIMITY

This operation merges features from other tokens into the benchmark set $\mathcal { T } _ { B }$ , guided by role assignment and spatial proximity, yielding a semantically rich and spatially complete token sequence.

Role Assignment: Pillars and Collectors. We differentiate benchmark tokens in $\mathcal { T } _ { B }$ by information capacity. Recent works (Darcet et al., 2024; Lappe & Giese, 2025) identify high-norm tokens in ViTs as registers — frequently attended during decoding and often task-agnostic. Modifications to these can shift feature distributions and affect predictions. Thus, we classify tokens with $\| \mathbf { k } _ { i } \| _ { 2 }$ in the top quartile as Pillar Tokens $( \mathcal { T } _ { P } )$ , whose features remain unmodified. The rest are Collector Tokens $( \mathcal { T } _ { C } )$ , which aggregate from spatial neighbors.

$$
\begin{array} { r } { \mathcal { T } _ { P } = \big \{ t _ { i } \in \mathcal { T } _ { B } \ | \ | \mathbf { k } _ { i } | _ { 2 } \geq \mathrm { Q u a n t i l e } \big ( \{ | \mathbf { k } _ { j } | _ { 2 } \} _ { t _ { j } \in \mathcal { T } _ { B } } , 0 . 7 5 \big ) \big \} ; \quad \mathcal { T } _ { C } = \mathcal { T } _ { B } \setminus \mathcal { T } _ { P } } \end{array}
$$

Weighted Aggregation. High semantic similarity does not imply aggregability; relying solely on it for global features is inadequate, as it risks disrupting object-centric representations by merging

spatially distant tokens. Thus, we balance it with spatial proximity to form a weight matrix ${ \textbf { W } } \in$ RK×N2, where $K = | T _ { B } |$ , combining semantic and proximity matrices.

Semantic Similarity Matrix (A): We consider only positively correlated semantic information. Element $A _ { i j }$ is defined as Eq. (5):

$$
A _ { i j } = \operatorname { R e L U } \left( \sin ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) \right) = \operatorname { R e L U } \left( { \frac { \mathbf { v } _ { i } \cdot \mathbf { v } _ { j } } { | \mathbf { v } _ { i } | | \mathbf { v } _ { j } | } } \right)
$$

Spatial Proximity Matrix $\mathbf { \Pi } ^ { ( \mathbf { P } ) }$ : To penalize long range aggregation, we define a proximity matrix allowing each benchmark token to aggregate features within an extended local neighborhood, enabling limited cross-region interaction. Element $P _ { i j }$ is computed as Eq. (6):

$$
P _ { i j } = 1 - \operatorname* { m a x } \left( 1 , \frac { d ( p _ { i } , p _ { j } ) } { d _ { \mathrm { t h r e s h } } } \right)
$$

where $d ( p _ { i } , p _ { j } )$ is the Euclidean distance between $p _ { i }$ and $p _ { j }$ , and $d _ { \mathrm { t h r e s h } }$ is a predefined threshold.

Based on role assignment, the final aggregation weight $W _ { i j }$ is defined as:

$$
\begin{array} { r } { W _ { i j } = \left\{ \begin{array} { l l } { \delta _ { i j } } & { \mathrm { i f ~ } t _ { i } \in \mathcal { T } _ { P } \mathrm { ( P i l l a r ~ T o k e n ) } } \\ { A _ { i j } \cdot P _ { i j } } & { \mathrm { i f ~ } t _ { i } \in \mathcal { T } _ { C } \mathrm { ( C o l l e c t o r ~ T o k e n ) } } \end{array} \right. } \end{array}
$$

where $\delta _ { i j }$ is the Kronecker delta, ensuring Pillar Tokens only aggregate from themselves.

The weight $\hat { \mathbf { W } }$ is row-normalized from W, the original feature matrix is $\mathbf { V } \in \mathbb { R } ^ { N ^ { 2 } \times D }$ . The updated feature matrix for benchmark tokens, $\mathbf { V } _ { B } ^ { \prime } \in \mathbb { R } ^ { K \times D }$ , is computed as $\mathbf { V } _ { B } ^ { \prime } = \hat { \mathbf { W } } \mathbf { V }$ .

# 3.2 STAGE 2: TEXT-MODULATED PRUNING IN THE LLM

Following Stage 1, the aggregated vision tokens $\mathbf { V } _ { B } ^ { \prime }$ are fed into the LLM for multimodal feature interaction. We apply a second round of task-oriented pruning at an intermediate layer, after initial multimodal alignment (Shukor & Cord, 2024), where textual and visual features converge in a shared space. To guide this pruning, we first derive a holistic textual query vector $\bar { \bf q }$ by average-pooling the embeddings $\left\{ \mathbf { q } _ { 1 } , \dots , \mathbf { q } _ { K } \right\}$ of text tokens:

$$
\bar { \mathbf { q } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbf { q } _ { k }
$$

We calculate a relevance score $R _ { i }$ for each visual token $t _ { i } ^ { \prime }$ (with updated feature vector $\mathbf { v } _ { i } ^ { \prime }$ , the $i$ -th token of $\mathbf { V } _ { B } ^ { \prime }$ ) by measuring its cosine similarity to the query vector in the shared embedding space:

$$
R _ { i } = \mathrm { s i m } ( \mathrm { p r o j } ( \mathbf { v } _ { i } ^ { \prime } ) , \bar { \mathbf { q } } ) = \frac { \mathrm { p r o j } ( \mathbf { v } _ { i } ^ { \prime } ) \cdot \bar { \mathbf { q } } } { | \mathrm { p r o j } ( \mathbf { v } _ { i } ^ { \prime } ) | \cdot | \bar { \mathbf { q } } | }
$$

where $\mathrm { p r o j } ( \cdot )$ denotes the multimodal projection layer mapping visual features into the common text-vision embedding space. Finally, we retain only the top- $K _ { \mathrm { f i n a l } }$ visual tokens with the highest relevance scores $R _ { i }$ , passed to subsequent LLM layers for final reasoning and response generation.

# 4 EXPERIMENT

Experimental Setup: To validate the generality and effectiveness of our method, we conduct experiments on multiple VLMs and diverse benchmarks for image understanding and visual grounding tasks. The evaluated models are LLaVA-1.5, LLaVA-NeXT. We use 10 VQA benchmarks (e.g., GQA, TextVQA) and 3 VG benchmarks (RefCOCO, etc.). All experiments are run on NVIDIA A100-40G GPUs. Detailed configurations are in the Appendix B.2.

# 4.1 MAIN RESULT

Performance on VQA Tasks: We apply Nuwa during the inference stage of LLaVA-1.5-7B. More ¨ results in Table 5 demonstrate that Nuwa achieves optimal performance across nearly all bench- ¨ marks, with average performance further improving upon existing SOTA methods. On more VLM models with different scales, such as LLaVA-NeXT-7B, Nuwa consistently demonstrates perfor- ¨ mance gains, establishing its strong generalizability. Results can be found in Appendix B.2. Performance on Visual Grounding Tasks: Visual grounding tasks are highly sensitive to spatial information in tokens, constituting a critical evaluation dimension for compression methods. On RefCOCO series visual grounding benchmarks, as shown in Table 6, our method substantially outperforms alternative approaches, achieving approximately $3 5 \%$ performance improvement over previous methods under 64 average tokens configuration. When retaining 192 tokens, our method maintains $79 \%$ of the original model’s performance.

Efficiency Analysis: As shown in Table 4, we evaluate efficiency from two dimensions: theoretical computational complexity and actual prefill latency. Nuwa introduces negligi- ¨ ble computational overhead, with TFLOPs increasing by only 0.01 and prefill stage latency increasing by $1 \ \mathrm { m s }$ , compared with previous SOTA. Nuwa’s design requires executing atten- ¨ tion computation only once on tokens from the final layer of the vision encoder, enabling seamless FlashAttention compatibility through simple code modifications.

Table 4: Comparison of Model Efficiency. “main” and “metric” mean the standard Transformer pipeline and the additional computational load of pruning metric.   

<table><tr><td>Method</td><td>Avg Token</td><td>main (TFLOPs)</td><td>metric (MFLOPs)</td><td>Prefill-Time (ms)</td></tr><tr><td>Vanilla</td><td>576</td><td>5.9730</td><td>0</td><td>124</td></tr><tr><td>FastV</td><td>64</td><td>0.8341</td><td>4.7185</td><td>92 ↓26%</td></tr><tr><td>SparseVLM</td><td>64</td><td>0.8141</td><td>5.5050</td><td>104 16%</td></tr><tr><td> VisionZip</td><td>64</td><td>.6461</td><td>8.9128</td><td>46 63%</td></tr><tr><td>N</td><td>64</td><td>.6476</td><td>117.5636</td><td>62%</td></tr></table>

# 4.2 ABLATION STUDY

Ablation on Spatial Proximity Threshold To enable aggregation based on spatial proximity, we define local neighborhoods via a distance threshold $\tau$ . Empirical evaluation (Table 7) shows that performance peaks at $\tau = 2 6 \%$ of the maximum distance. Smaller values restrict aggregation scope, leading to suboptimal results, while larger values incorporate noise from distant regions, also degrading performance. These results confirm the effectiveness of localized aggregation in preserving spatial integrity. Ablation on Key Components Experimental results in Table 8 show that region partitioning is essential for grounding tasks, as it implements a more precise RPME strategy, but has negligible effects on VQA tasks. The L2-norm criterion positively enhances baseline token selection across all tasks, consistent with our analysis in Sec. 3.1.3. For two-stage pruning, gains over random pruning remain modest. Notably, combining random pruning with region partitioning substantially degrades performance, as the partitioning introduces potentially task-irrelevant tokens that random selection may retain.

Table 5: VQA performance comparison On LLava-1.5 7B. Best and second-best results are highlighted.   

<table><tr><td>Method</td><td>Source</td><td>GQA</td><td>MMB</td><td>MMMU</td><td>MME</td><td>VQAv2</td><td>VQAtext</td><td>POPE</td><td>SQA</td><td>SEED</td><td>MMVet</td><td>avg</td></tr><tr><td>Vanilla</td><td>CVPR&#x27;24</td><td>61.9</td><td>64.7</td><td>36.3</td><td>1862</td><td>78.5</td><td>58.2</td><td>85.9</td><td>69.5</td><td>58.6</td><td>31.1</td><td>100%</td></tr><tr><td colspan="10">Average Token 192 ↓ 66.7%</td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>ECCV&#x27;24</td><td>52.7</td><td>61.2</td><td>34.3</td><td>1612</td><td>67.1</td><td>52.5</td><td>64.8</td><td>67.3</td><td>57.1</td><td>27.7</td><td>89.53%</td></tr><tr><td>PDrop</td><td>CVPR&#x27;25</td><td>57.1</td><td>63.2</td><td>34.1</td><td>1766</td><td>74.9</td><td>56.1</td><td>82.3</td><td>70.2</td><td>54.7</td><td>30.5</td><td>95.7%</td></tr><tr><td>parseVLM</td><td>ICML.25</td><td>57.6</td><td>62.5</td><td>33.8</td><td>1721</td><td>75.6</td><td>56.1</td><td>883.6</td><td>69.11</td><td>55.8</td><td>31.5</td><td>96.11%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>59.3</td><td>63.0</td><td>36.6</td><td>1782</td><td>76.8</td><td>57.3</td><td>85.3</td><td>68.9</td><td>56.4</td><td>31.7</td><td>98.26%</td></tr><tr><td> Nuwa</td><td>-</td><td>60.9</td><td>64.3</td><td>35.5</td><td>1834</td><td>75.9</td><td>57.4</td><td>86.4</td><td>68.2</td><td>59.7</td><td>30.5</td><td>98.80%</td></tr><tr><td colspan="10">Average Token 128 ↓ 77.8%</td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>ECCV&#x27;24</td><td>49.6</td><td>56.1</td><td>34.9</td><td>1490</td><td>61.8</td><td>50.6</td><td>59.6</td><td>60.2</td><td>55.9</td><td>28.1</td><td>85.04%</td></tr><tr><td>Drop</td><td>CVPR&#x27;25</td><td>56.00</td><td>61.1</td><td>34.2</td><td>1664</td><td>73.5</td><td>55.1</td><td>82.3</td><td>69.9</td><td>53.3</td><td>30.8</td><td>94.32%</td></tr><tr><td> parseVLM</td><td>ICML.25</td><td>56.0</td><td>6.0</td><td>33.8</td><td>1696</td><td>73.8</td><td>54.9</td><td>80.5</td><td>67.1</td><td>53.4</td><td>300</td><td>93.6%</td></tr><tr><td> VissionZip</td><td>CPR&#x27;25</td><td>57.6</td><td>62.0</td><td>37.9</td><td>11761</td><td>75.6</td><td>568 </td><td>83.2</td><td>68.9</td><td>54.9</td><td>32.6</td><td>97.63%%</td></tr><tr><td>PruMerge</td><td>ICCV&#x27;25</td><td>57.8</td><td>59.6</td><td>36.2</td><td>11712</td><td>74.7</td><td>54.3</td><td>81.5</td><td>67.6</td><td>-</td><td>30.4</td><td>95.06%</td></tr><tr><td>uwa</td><td>-</td><td>660.2</td><td>63.4</td><td>35.8</td><td>1828</td><td>75.1</td><td>57.0</td><td>85.5</td><td>67.8</td><td>58.7</td><td>29.8</td><td>97.87%</td></tr><tr><td colspan="10">Average Token 64 ↓ 88.9%</td><td></td><td></td><td></td></tr><tr><td>FastV</td><td>ECCV&#x27;24</td><td>46.1</td><td>48.0</td><td>34.0</td><td>1256</td><td>55.0</td><td>47.8</td><td>59.6</td><td>51.1</td><td>51.9</td><td>25.8</td><td>79.36%</td></tr><tr><td>Drop</td><td>CVPR&#x27;25</td><td>41.9</td><td>33.3</td><td>26.5</td><td>1092</td><td>57.3</td><td>45.9</td><td>55.9</td><td>69.2</td><td>40.0</td><td>24.9</td><td>71.56%</td></tr><tr><td>SparseVLM</td><td>IML.&#x27;25</td><td>53.8</td><td>60.1</td><td>35.44</td><td>1589</td><td>68.2</td><td>53.4</td><td>77.5</td><td>669.8</td><td>551.1</td><td>24.9</td><td>89.93%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>55.1</td><td>60.1</td><td>36.2</td><td>1690</td><td>72.4</td><td>55.5</td><td>77.0</td><td>69.0</td><td>52.2</td><td>31.7</td><td>93.99%%</td></tr><tr><td>PruMerge</td><td>ICCV&#x27;25</td><td>55</td><td>56</td><td>35.8</td><td>1616</td><td>71.3</td><td>52.0</td><td>75.7</td><td>69.55</td><td></td><td>28.0</td><td>91.71%</td></tr><tr><td>Nuwa</td><td></td><td>58.3</td><td>62.0</td><td>36.4</td><td>1706</td><td>72.8</td><td>54.9</td><td>83.0</td><td>67.5</td><td>56.44</td><td>28.2</td><td>94.91%</td></tr></table>

Table 6: Performance comparison on the RefCOCO series benchmark On LLava-1.5 7B. Best and second-best results are highlighted.   

<table><tr><td>Method</td><td>Source</td><td>Refcoco-test</td><td>Refcoco-val</td><td>Refcoco+-testA</td><td>Refcoco+-testB</td><td>Refcoco+-val</td><td>Refcocog-test</td><td>Refcocog-val</td><td>avg</td></tr><tr><td colspan="10">58.30 56.42 59.43</td></tr><tr><td></td><td></td><td></td><td></td><td>Average Tokens 192 ↓ 66.7 %</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">24.7</td></tr><tr><td>Nüwa</td><td>ICCV&#x27;25</td><td>27.7 47.91</td><td>46.12</td><td>43.18</td><td>31.86</td><td>37.68</td><td>27.2 37.64</td><td>37.90</td><td>48.38% 79..29%</td></tr><tr><td colspan="10">Average Tokens 128 ↓ 77.8%</td></tr><tr><td colspan="10">8.53</td></tr><tr><td>Fastv SparseVLM</td><td>ECCV&#x27;24</td><td>10.34</td><td>10.13</td><td></td><td>9.83</td><td>8.16</td><td>8.87</td><td>9.10</td><td>18.55%</td></tr><tr><td> Visionip</td><td>CML.25</td><td>6.27</td><td>6.17</td><td>5.79 </td><td>4.22</td><td>9.85 3.88</td><td>6.35 3.50</td><td>66.47 3.48</td><td>12.84% 8.1%</td></tr><tr><td>Nüwa</td><td>CVPR&#x27;25</td><td>4.49 45.09</td><td>4.11 43.69</td><td>4.06 42.63</td><td>4.86 28.98</td><td>35.32</td><td>36.59</td><td>36.00</td><td>75.20%</td></tr><tr><td colspan="10">Average Tokens 64 ↓ 88.9%</td></tr><tr><td colspan="10"></td></tr><tr><td>Fastv SparseVLM</td><td>ECCV&#x27;24</td><td>2.73</td><td>2.01</td><td>1.17</td><td>1.02</td><td>2.41</td><td>2.19</td><td>2.01</td><td>3.81%</td></tr><tr><td> VisinZip</td><td>ICML.25 CVPR&#x27;25</td><td>1.04</td><td>1.01</td><td>0.96</td><td>1.28</td><td>0.96</td><td>0.61 3.38</td><td>0.66 3.21</td><td>1.88% 7.28%</td></tr><tr><td>Nüwa</td><td></td><td>4.04 29.43</td><td>3.81 28.60</td><td>3.73 28.22</td><td>3.86 17.47</td><td>3.50 22.22</td><td>21.81</td><td>21.42</td><td>47.19%</td></tr></table>

Table 7: Ablation Study On cohesion distance. The best-performing result in each column is bolded, and the second-best is underlined.   

<table><tr><td>Config</td><td>GQA</td><td>MMB</td><td>MME</td><td>Refcoco -test</td><td>Refcoco+ -testA</td><td>Refcoco+ -testB</td><td>Refcocog -test</td></tr><tr><td>dist18</td><td>0.5784</td><td>60.2852</td><td>1695.37</td><td>0.2783</td><td>0.2818</td><td>0.1655</td><td>0.2018</td></tr><tr><td>dist148</td><td>0.5853</td><td>61.6838</td><td>1704.981</td><td>0.2922</td><td>0.2936</td><td>0.1730</td><td>0.2189</td></tr><tr><td>dist280</td><td>0.5833</td><td>62.0275</td><td>1706.869</td><td>0.2943</td><td>0.2822</td><td>0.1747</td><td>0.2181</td></tr><tr><td>dist412</td><td>0.5826</td><td>62.1134</td><td>1711.202</td><td>0.2879</td><td>0.2705</td><td>0.1698</td><td>0.2135</td></tr><tr><td>dist544</td><td>0.5811</td><td>62.0275</td><td>1702.986</td><td>0.2834</td><td>0.2637</td><td>0.1651</td><td>0.2100</td></tr><tr><td>dist676</td><td>0.5810</td><td>61.9416</td><td>1696.32</td><td>0.2801</td><td>0.2590</td><td>0.1636</td><td>0.2083</td></tr><tr><td>dist808</td><td>0.5808</td><td>61.8557</td><td>1706.869</td><td>0.2769</td><td>0.2607</td><td>0.1632</td><td>0.2071</td></tr><tr><td>dist940</td><td>0.5799</td><td>61.9416</td><td>1705.986</td><td>0.2774</td><td>0.2572</td><td>0.1622</td><td>0.2059</td></tr><tr><td>dist1058</td><td>0.5799</td><td>61.7698</td><td>1704.486</td><td>0.2765</td><td>0.2560</td><td>0.1630</td><td>0.2054</td></tr></table>

Table 8: Ablation Study on each design. Include Pillar-token selecting, Stage2 Random Pruning and Region Separation.   

<table><tr><td>region</td><td>pillar token</td><td>random S2</td><td>GQA</td><td>MMB</td><td>MME</td><td>Refcoco test</td><td>Refcoco+ testA</td><td>Refcoco+ testB</td><td>Refcocog test</td></tr><tr><td>×</td><td>×</td><td>×</td><td>58.84</td><td>58.18</td><td>1791</td><td>6.83</td><td>6.54</td><td>4.50</td><td>5.58</td></tr><tr><td>×</td><td>×</td><td>v</td><td>57.07</td><td>56.43</td><td>1736</td><td>6.72</td><td>6.48</td><td>4.38</td><td>5.25</td></tr><tr><td>×</td><td>v</td><td>x</td><td>59.62</td><td>62.98</td><td>1807</td><td>6.35</td><td>6.12</td><td>4.65</td><td>5.60</td></tr><tr><td>x</td><td>v</td><td>v</td><td>58.43</td><td>61.71</td><td>1771</td><td>7.01</td><td>6.81</td><td>4.42</td><td>4.92</td></tr><tr><td>v</td><td>×</td><td>x</td><td>57.94</td><td>56.68</td><td>1742</td><td>43.50</td><td>39.85</td><td>26.10</td><td>34.20</td></tr><tr><td>v</td><td>×</td><td>v</td><td>57.35</td><td>56.10</td><td>1724</td><td>43.17</td><td>38.94</td><td>25.58</td><td>33.74</td></tr><tr><td>v</td><td>v</td><td>x</td><td>60.18</td><td>63.40</td><td>1828</td><td>45.09</td><td>42.63</td><td>28.98</td><td>36.59</td></tr><tr><td>v</td><td>v</td><td>v</td><td>59.03</td><td>62.14</td><td>1791</td><td>44.30</td><td>41.20</td><td>27.50</td><td>35.80</td></tr></table>

# 5 CONCLUSION

In this paper, we identify limitations in existing token pruning methods for visual grounding tasks and perform a systematic analysis of VLMs’ multi-stage visual processing pipelines. Results reveal task-specific demands, where grounding relies on global spatial reference frames disrupted by pruning. To mitigate this, we propose Nuwa, a two-stage framework with Boids-inspired aggregation ¨ and text-guided refinement to preserve spatial integrity. Extensive experiments across 13 datasets and multiple VLMs show state-of-the-art performance on VQA $9 5 \%$ retention) and VG $4 7 . 2 \%$ retention), with $89 \%$ TFLOPs and $62 \%$ prefill reductions via $8 8 . 9 \%$ token pruning.

# REPRODUCIBILITY STATEMENT

To ensure the reproducibility of the results presented in this paper, we take the following steps. For the experiment of Sec. 2.1, we set up simple baselines for each category of methods to conduct comparative experiments. Implementation details are provided in the Appendix B.3. For the analytical experiments in Sec. 2.2, no additional configurations are applied. For the Experiment of Sec 2.3, we conduct position re-estimation experiments. The algorithm implementation for RPME is provided in the Appendix B.4. All experiments are based on LLAVA-1.5 7B, with the same environment configuration, dataset, and model weights claimed in Appendix B.1. For the Main Experiment, we provide the algorithm implementation of the complex Stage-1 pruning in the Appendix B.2. The Stage-2 implementation is analogous to FASTV.

# REFERENCES

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversitybased visual token pruning for large multimodal models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9392–9401, 2025. URL https://api. semanticscholar.org/CorpusID:276775957.

Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models, 2024. URL https://arxiv.org/abs/2408.10945.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani S. Marathe, Yonatan Bitton, Samir Yitzhak Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. ArXiv, abs/2308.01390, 2023. URL https://api.semanticscholar.org/ CorpusID:261043320.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster, 2023. URL https://arxiv.org/abs/ 2210.09461.

Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. ArXiv, abs/2405.17430, 2024. URL https://api.semanticscholar.org/ CorpusID:270063538.

Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13817–13827, 2023. URL https://api.semanticscholar. org/CorpusID:266174127.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, 2024. URL https://api. semanticscholar.org/CorpusID:268358224.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. ArXiv, abs/2305.06500, 2023. URL https: //api.semanticscholar.org/CorpusID:258615266.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. ArXiv, abs/2307.08691, 2023. URL https://api.semanticscholar.org/ CorpusID:259936734.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R’e. Flashattention: Fast and memory-efficient exact attention with io-awareness. ArXiv, abs/2205.14135, 2022. URL https: //api.semanticscholar.org/CorpusID:249151871.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers, 2024. URL https://arxiv.org/abs/2309.16588.

Yanzhuo Ding, Yang Liu, Huanbo Luan, and Maosong Sun. Visualizing and understanding neural machine translation. In Annual Meeting of the Association for Computational Linguistics, 2017. URL https://api.semanticscholar.org/CorpusID:27930067.

Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. ArXiv, abs/2412.13180, 2024. URL https: //api.semanticscholar.org/CorpusID:274789102.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. URL https://api.semanticscholar.org/CorpusID:259243928.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. International Journal of Computer Vision, 127:398 – 414, 2016. URL https://api. semanticscholar.org/CorpusID:8081284.

Lianyu Hu, Fanhua Shang, Wei Feng, and Liang Wan. Lightvlm: Acceleraing large multimodal models with pyramid token merging and kv cache compression, 2025. URL https://arxiv. org/abs/2509.00419.

Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaoshen Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification. ArXiv, abs/2412.00876, 2024. URL https: //api.semanticscholar.org/CorpusID:274437635.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6693–6702, 2019. URL https://api.semanticscholar. org/CorpusID:152282269.

Ahmadreza Jeddi, Negin Baghbanzadeh, Elham Dolatabadi, and Babak Taati. Similarity-aware token pruning: Your vlm but faster. ArXiv, abs/2503.11549, 2025. URL https://api. semanticscholar.org/CorpusID:277043961.

Alexander Lappe and Martin A. Giese. Register and cls tokens yield a decoupling of local and global features in large vits, 2025. URL https://arxiv.org/abs/2505.05892.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023a. URL https://arxiv. org/abs/2307.16125.

Dylan Li and Gyungin Shin. Promerge: Prompt and merge for unsupervised instance segmentation. In European Conference on Computer Vision (ECCV), 2024.

Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. ArXiv, abs/2407.02392, 2024. URL https://api.semanticscholar.org/CorpusID:270878717.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, 2023b. URL https://api.semanticscholar.org/ CorpusID:258740697.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296, 2024a. doi: 10.1109/CVPR52733.2024.02484.

Jizhihui Liu, Feiyi Du, Guangdao Zhu, Niu Lian, Jun Li, and Bin Chen. Hiprune: Training-free visual token pruning via hierarchical attention in vision-language models, 2025a. URL https: //arxiv.org/abs/2508.00553.

Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision token dropping: Towards efficient multimodal large language model, 2024b. URL https: //arxiv.org/abs/2411.10803.

Xuyang Liu, Ziming Wang, Junjie Chen, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Linfeng Zhang, Siteng Huang, and Honggang Chen. Global compression commander: Plug-and-play inference acceleration for high-resolution large vision-language models. 2025b. URL https: //api.semanticscholar.org/CorpusID:275405970.

Yuanzhan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? ArXiv, abs/2307.06281, 2023. URL https: //api.semanticscholar.org/CorpusID:259837088.

Xinwei Long, Kai Tian, Peng Xu, Guoli Jia, Jingxuan Li, Sa Yang, Yihua Shao, Kaiyan Zhang, Che Jiang, Hao Xu, et al. Adsqa: Towards advertisement video understanding. arXiv preprint arXiv:2509.08621, 2025.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. ArXiv, abs/2209.09513, 2022. URL https://api. semanticscholar.org/CorpusID:252383606.   
Craig W. Reynolds. Flocks, herds, and schools: a distributed behavioral model, pp. 273–282. Association for Computing Machinery, New York, NY, USA, 1998. ISBN 158113052X. URL https://doi.org/10.1145/280811.281008.   
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Journal of Computer Vision, 128:336 – 359, 2016. URL https: //api.semanticscholar.org/CorpusID:15019293.   
Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. ArXiv, abs/2403.15388, 2024. URL https: //api.semanticscholar.org/CorpusID:268667281.   
Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang, Ziyang Yan, Ao Ma, et al. Eventvad: Training-free event-aware video anomaly detection. arXiv preprint arXiv:2504.13092, 2025a.   
Zhenwei Shao, Mingyang Wang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Ning Mao, Wei Chen, and Jun Yu. Growing a twig to accelerate large vision-language models. ArXiv, abs/2503.14075, 2025b. URL https://api.semanticscholar.org/ CorpusID:277103933.   
Mustafa Shukor and Matthieu Cord. Implicit multimodal alignment: On the generalization of frozen llms to multimodal inputs, 2024. URL https://arxiv.org/abs/2405.16700.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8309–8318, 2019. URL https://api. semanticscholar.org/CorpusID:85553602.   
Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, and Ruixuan Li. Flowcut: Rethinking redundancy via information flow for efficient vision-language models, 2025. URL https://arxiv.org/abs/2505.19536.   
Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gregory Gabriel, Peter Grasch, Oncel Tuzel, and Hadi Pouransari. Fastvlm: Efficient vision encoding for vision language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19769–19780, 2024. URL https: //api.semanticscholar.org/CorpusID:274822212.   
Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and Linfeng Zhang. Token pruning in multimodal large language models: Are we solving the right problem? In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 15537–15549, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.802. URL https://aclanthology.org/2025.findings-acl.802/.   
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. ArXiv, abs/2309.17453, 2023. URL https://api. semanticscholar.org/CorpusID:263310483.   
Long Xing, Qidong Huang, Xiao wen Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. Pyramiddrop: Accelerating your large visionlanguage models via pyramid visual redundancy reduction. ArXiv, abs/2410.17247, 2024. URL https://api.semanticscholar.org/CorpusID:273507889.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19792–19802, 2024. URL https://api.semanticscholar.org/CorpusID:274514545.

Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24972–24982, 2024. URL https://api. semanticscholar.org/CorpusID:274436316.

Hao Yin, Guangzong Si, and Zilei Wang. Lifting the veil on visual information flow in mllms: Unlocking pathways to faster inference. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9382–9391, 2025. URL https://api.semanticscholar. org/CorpusID:277104433.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions. ArXiv, abs/1608.00272, 2016. URL https://api. semanticscholar.org/CorpusID:1688357.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. ArXiv, abs/2308.02490, 2023. URL https://api.semanticscholar.org/ CorpusID:260611572.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9556–9567, 2023. URL https://api.semanticscholar.org/ CorpusID:265466525.

Ce Zhang, Kaixin Ma, Tianqing Fang, Wenhao Yu, Hongming Zhang, Zhisong Zhang, Yaqi Xie, Katia P. Sycara, Haitao Mi, and Dong Yu. Vscan: Rethinking visual token reduction for efficient large vision-language models. ArXiv, abs/2505.22654, 2025a. URL https://api. semanticscholar.org/CorpusID:278959299.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning, 2025b.

Zhenyu (Allen) Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark W. Barrett, Zhangyang Wang, and Beidi ´ Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models. ArXiv, abs/2306.14048, 2023. URL https://api.semanticscholar.org/ CorpusID:259263947.

Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19781–19791, 2024. URL https://api.semanticscholar. org/CorpusID:274306239.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

# A RELATED WORK

# A.1 EFFICIENT LARGE VISION-LANGUAGE MODELS

LLMs and VLMs face significant computational efficiency challenges, particularly with extended sequences. LLMs grapple with the growing key-value (KV) cache during autoregressive inference, leading to the development of token reduction strategies like StreamingLLM (Xiao et al., 2023) and H2O (Zhang et al., 2023). However, VLMs confront amplified complexity due to the quadratic growth of visual tokens with image resolution or video frames, making their computational costs prohibitive and necessitating modality-specific optimizations. Two main architectural approaches address these computational constraints. One involves architectural compression, where modules like Q-Former (InstructBLIP (Dai et al., 2023)), perceiver resampler (OpenFlamingo (Awadalla et al., 2023)), and Locality-enhanced Abstractor (Honeybee (Cha et al., 2023)) distill high-dimensional visual inputs into compact representations, reducing the sequence length processed by expensive attention mechanisms. The other pathway utilizes hardware-aware optimization strategies, such as FlashAttention (Dao et al., 2022; Dao, 2023), which optimize memory access patterns for accelerated self-attention computation without altering token quantities, achieving performance gains through algorithmic refinements and efficient resource utilization.

# A.2 TOKEN PRUNING IN LARGE VISION-LANGUAGE MODELS

A complementary approach to VLM efficiency focuses on reducing computational overhead through token sequence optimization. The quadratic computational complexity of Transformer attention mechanisms becomes particularly problematic when processing the extensive visual token sequences typical in VLMs. Consequently, vision token pruning has emerged as a critical research direction, which can be systematically categorized along multiple dimensions. Token reduction approaches can be classified based on their training requirements into training-free and training-based methods. Regarding implementation stages, these techniques operate across four primary phases: (1) visual encoder preprocessing, (2) LLM internal processing, (3) KV cache optimization, and (4) hybrid multi-stage approaches. Each pruning strategy involves two fundamental decisions: identifying which tokens to retain and aggregating useful features from discarded tokens.

Token Pruning At Vision Encoder ToME (Bolya et al., 2023) establishes the foundation for training-free token merging at the visual encoder stage, demonstrating effective feature-based token consolidation that influences subsequent works, including VisionZip (Yang et al., 2024), DivPrune (Alvar et al., 2025), LLaVA-PruMerge (Shang et al., 2024), and so on (Tong et al., 2025; Liu et al., 2025a). These methods leverage visual feature similarity to merge redundant tokens before they enter the language model, thereby reducing the computational burden on downstream processing stages.

Token Pruning Within LLM FastV (Chen et al., 2024) pioneers attention score-based token pruning within the LLM processing pipeline, establishing a training-free paradigm that guides later developments such as SparseVLM (Zhang et al., 2025b), PyramidDrop (Xing et al., 2024), FastVLM (Vasu et al., 2024), and so on (Liu et al., 2025a; Arif et al., 2024). These approaches dynamically identify and remove less informative tokens based on attention patterns during inference, maintaining model performance while significantly reducing computational requirements.

Multi-Stage Optimization Strategies Comprehensive efficiency improvements have been achieved through multi-stage approaches that simultaneously optimize visual encoding, LLM prefill, and KV cache management during decoding. Representative methods include MustDrop (Liu et al., 2024b), LightVLM (Hu et al., 2025), and GlobalCom2 (Liu et al., 2025b), which coordinate token reduction across multiple pipeline stages to maximize computational savings while preserving model capabilities.

Training-Based Methods While training-based approaches may exhibit reduced generalizability compared to training-free methods, they demonstrate superior performance preservation through pruning-aware optimization. Methods such as ${ \bf { M } } ^ { 3 }$ (Cai et al., 2024), ATP-LLaVA (Ye et al., 2024), Dynamic-LLaVA (Huang et al., 2024), TokenPacker (Li et al., 2024), and TwigVLM (Shao et al.,

2025b) achieve competitive or superior performance compared to their full-token baselines through specialized training procedures that adapt the model to operate effectively with reduced token sequences.

# B DETAILED EXPERIMENT SETUP

In this section, we provide detailed experimental setups and algorithm implementations, along with supplementary experiments. These include the main experiment, position reconstruction experiments, and attention blocking experiments.

# B.1 IMPLEMENT DETAILS

To ensure the reproducibility of the results presented in this paper, we have provided reproducible explanations for the key experiments in the paper.

Table 9: Important packages in the Conda Environment.   

<table><tr><td>Name</td><td>Version</td></tr><tr><td>datasets</td><td>4.0.0</td></tr><tr><td>llava</td><td>1.2.2.post1</td></tr><tr><td>lmms-eval</td><td>0.3.4</td></tr><tr><td>qwen-vl-utils</td><td>0.0.14</td></tr><tr><td>sentencepiece</td><td>0.2.0</td></tr><tr><td>tokenizers</td><td>0.21.4</td></tr><tr><td>torch</td><td>2.6.0 + cu124</td></tr><tr><td>torchaudio</td><td>2.6.0 + cu124</td></tr><tr><td>torchvision</td><td>0.21.0 + cu124</td></tr><tr><td>transformers</td><td>4.54.0.dev0</td></tr></table>

Models The model weights used are sourced from the Hugging Face community, specifically as follows:

1. LLAVA-1.5 7B   
2. LLAVA-Next 7B   
3. QWEN-2.5 VL 7B

Datesets The dataset used originates from the lmm-lab datasets.

# B.2 MAIN EXPERIMENT

Nuwa’s stage-1 pruning algorithm: ¨ To more clearly illustrate the proposed method, particularly the complex first-stage cropping, we provide pseudocode for the algorithm implementation here Algorithm 1.

A detailed setup for different VLMs: Based on different models, we adjust Nuwa’s framework¨ configuration, which still revolves around a two-stage process. The configuration calculation for Average Token $( \bar { R } _ { v , \mathrm { L L M } } )$ is as follows:

$$
\bar { R } _ { v , \mathrm { L L M } } = \frac { 1 } { L _ { l } } \sum _ { i = L _ { v } } ^ { L _ { v } + L _ { l } - 1 } r _ { v } ^ { ( i ) }
$$

where $L _ { l }$ is the number of LLM layers and $r _ { v } ^ { ( i ) }$ is the number of visual tokens entering each LLM layer for processing. Our setting is shown in Table 10.

LLAVA-Next 7B Performance evaluation of our method on LLAVA-Next 7B, including VG task at Table 11 and VQA task at Table 12. Nuwa achieves state-of-the-art performance across multiple ¨ datasets.

# Algorithm 1 Nuwa Stage-1 Pruning ¨

# Require:

Input images X ∈ RB×C×H×W   
Vision tower model (use CLIP VIT-Large here)   
Penalty threshold percentile $\tau = 0 . 2 5$   
Region configuration: top- $\boldsymbol { n }$ tokens per $g \times g$ region   
Target tokens per image $k$

# Ensure:

Aggregated tokens $T \in \mathbb { R } ^ { B \times k \times D }$ (selected and aggregated features)   
Benchmark indices $I \in \mathbb { R } ^ { B \times k }$ (sorted positions of selected tokens)   
1: $B \gets | X | ; \quad N \gets H \times W ; \quad g , n , k \gets P r u n e _ { c f g }$ $\triangleright$ Get setting based on configuration   
2: $H _ { g }$ , $W _ { g } \gets H / g , W / g$   
3: $H _ { s } , A \gets V i s i o n t o w e r ( X )$ $\triangleright$ Output hidden states and attentions   
4: $\begin{array} { r } { H  H _ { s } [ L ] \quad A _ { L }  A [ L ] } \end{array}$ ▷ Select layer $L = - 2$ hidden states, shape $B \times ( 1 + N ) \times D$   
5: $M \gets \sum \bar { A _ { L } } [ : , : , 0 , 1$ :] $\triangleright$ Metric map from CLS attention, shape $B \times N$   
6: $M _ { 2 D } $ Reshape $M$ to 2D $\triangleright$ based on grid separation $H _ { g } \times W _ { g }$   
7: Step1: Region-based Candidate Selection:   
8: Unfold $M _ { 2 D }$ into regions of $g \times g$ patches $\triangleright \mathrm { S h a p e } \ B \times H _ { g } \times W _ { g } \times ( g \times g )$   
9: for each region (i,j) do   
10: C.append $\because \dot { \langle \textit { T o p } _ { k } ( M _ { 2 D } [ : , i , j , : ] ) \rangle }$ ▷ Select top- $\mathbf { \nabla } \cdot n$ indices of Local coordinates within region   
11: end for   
12: $S _ { C } $ Gather scores for $C$ from $M$   
13: Select top- $k$ from $S _ { C }$ : $I $ Benchmark indices ▷ Sorted   
14: Step2: Spatial Proximity And Similarity Construction:   
15: Create $\hat { P _ { \mathbf { \lambda } } } \in \mathbb { R } ^ { H _ { g } \times W _ { g } }$ ▷ Patch grid Spatial Proximity Matrix based on distance threshold   
16: $\bar { H } $ Average hidden states from mid-stage $\triangleright$ Shape $B \times ( 1 + N ) \times D$   
17: $P _ { t } \gets \bar { H } [ : , 1 : , : ]$ $\triangleright$ Patch tokens, normalized   
18: $S i m \gets \bar { P } _ { t } \cdot P _ { t } ^ { \bar { T } }$ $\triangleright$ Similarity matrix, $B \times N \times N$   
19: $W \gets \mathrm { R e L U } ( \bar { S } i m ) \odot P$ ▷ Aggregation weights with distance penalty   
20: Step3: Select Pillar Token:   
21: $S _ { I } $ Gather scores for $I$ from $L 2 _ { n o r m }$   
22: $I _ { p i l l a r }  Q u a n t i l e ( S _ { I } , \tau )$   
23: $\dot { W } \gets S e t V a l u e s ( W , I _ { p i l l a r } , 0 )$ $\triangleright$ for Pillar Tokens, set 0 value for aggregation Weight $W$   
24: Step4: Aggregation:   
25: $W _ { I } $ Gather $W$ for benchmark indices $I$ ▷ Shape $B \times k \times N$   
26: Normalize $W _ { I }$ $\triangleright$ Sum to 1 per row, self-weight 1   
27: $T \gets W _ { I } \cdot H [ : , 1 : , : ]$ $\triangleright$ Aggregate to benchmark tokens   
28: return T, I

Table 10: Nuwa two-stage pruning setting on different VLMs. ¨   

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stage 1    Stage 2</td><td rowspan=1 colspan=1>Stage 1     Stage 2</td><td rowspan=1 colspan=1>Stage 1     Stage 2</td></tr><tr><td rowspan=2 colspan=1>LLAVA-1.5</td><td rowspan=1 colspan=1>Average Token 64</td><td rowspan=1 colspan=1>Average Token 128</td><td rowspan=1 colspan=1>Average Token 192</td></tr><tr><td rowspan=1 colspan=1>112        16</td><td rowspan=1 colspan=1>224        32</td><td rowspan=1 colspan=1>336        48</td></tr><tr><td rowspan=2 colspan=1>LLAVA-Next</td><td rowspan=1 colspan=1>Average Token 160 (5.6%)</td><td rowspan=1 colspan=1>Average Token 320 (11.1%)</td><td rowspan=1 colspan=1>Average Token 640 (22.2%)</td></tr><tr><td rowspan=1 colspan=1>9%       1%</td><td rowspan=1 colspan=1>16%       2%</td><td rowspan=1 colspan=1>32%       4%</td></tr><tr><td rowspan=2 colspan=1>Qwen2.5-VL</td><td rowspan=1 colspan=1>Average Token 25%</td><td rowspan=1 colspan=1>Average Token 50%</td><td rowspan=1 colspan=1>Average Token 75%</td></tr><tr><td rowspan=1 colspan=1>35%      42%</td><td rowspan=1 colspan=1>60%       66%</td><td rowspan=1 colspan=1>85%       70%</td></tr></table>

Table 11: Refcoco series Benchmarks performance comparison On LLaVA-Next 7B. Best and second-best results are highlighted.   

<table><tr><td>Method</td><td>Refcoco-test</td><td>Refcoco+-testA</td><td>Refcoco-testB</td><td>Refcocog-test</td><td>avg</td></tr><tr><td>Vanilla</td><td>77.73</td><td>76.34</td><td>57.25</td><td>71.05</td><td>100%</td></tr><tr><td colspan="8">Average Tokens 160 ↓ 94.4%</td></tr><tr><td>Nüwa</td><td>18.67</td><td>15.44</td><td>10.45</td><td>17.04</td><td>21.62%</td></tr><tr><td colspan="8">Average Tokens 320 ↓ 88.9%</td></tr><tr><td>Nüwa</td><td>38.67</td><td>33.86</td><td>25.36</td><td>31.49</td><td>45.68%</td></tr><tr><td colspan="8">Average Tokens 640 ↓ 77.8%</td></tr><tr><td>Nüwa</td><td>68.17</td><td>66.91</td><td>49.54</td><td>59.72</td><td>86.48%</td></tr></table>

Table 12: VQA Benchmarks performance comparison On LLaVA-Next 7B. Best and second-best results are highlighted.   

<table><tr><td>Methods</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>TextVQA</td><td>avg</td></tr><tr><td>Vanilla</td><td>64.2</td><td>67.9</td><td>1846</td><td>86.4</td><td>73.2</td><td>61.3</td><td>100%</td></tr><tr><td colspan="8">Average Tokens 160 ↓ 94.4%</td></tr><tr><td>SparseVLM</td><td>51.2</td><td>52.1</td><td>1542</td><td>72.7</td><td>67.5</td><td>46.4</td><td>79.80%</td></tr><tr><td> VisonZip</td><td>55.5</td><td>60.1</td><td>1628</td><td>74.8</td><td>68.3</td><td>56.2</td><td>87.60%</td></tr><tr><td>Nüwa</td><td>60.0</td><td>60.4</td><td>1684</td><td>83</td><td>67.5</td><td>56.3</td><td>92.29%</td></tr><tr><td colspan="8">Average Tokens 320 ↓ 88.9%</td></tr><tr><td>SparseVLM</td><td>57.7</td><td>63.2</td><td>1685</td><td>82.2</td><td>67.3</td><td>55.9</td><td>91.20%</td></tr><tr><td> VisionZip</td><td>59.22</td><td>63.1</td><td>1702</td><td>821</td><td>67.3</td><td>558.9</td><td>93.00%</td></tr><tr><td>Nüwa</td><td>62.3</td><td>63.2</td><td>1813</td><td>86.0</td><td>68.2</td><td>58.5</td><td>96.10%</td></tr><tr><td colspan="8">Average Tokens 640 ↓ 77.8%</td></tr><tr><td>SparseVLM</td><td>60.3</td><td>65.8</td><td>1773</td><td>84.2</td><td>67.7</td><td>57.8</td><td>95.30%</td></tr><tr><td>VisionZip</td><td>61.3</td><td>66.2</td><td>11787</td><td>85.9</td><td>68.1</td><td>60.2</td><td>96..90%</td></tr><tr><td>T Nuwa</td><td>663.4</td><td>65.4</td><td>1879</td><td>87.2</td><td>668.6</td><td>59.55</td><td>98.10%</td></tr></table>

QWEN-2.5-VL 7B Performance evaluation of our method on QWEN-2.5 VL 7B, including VG task at Table 13 and VQA task at Table 14. Nuwa achieves state-of-the-art performance across ¨ multiple datasets.

Table 13: Refcoco series Benchmarks performance comparison On QWEN-2.5 VL 7B. Best and second-best results are highlighted.   

<table><tr><td>Method</td><td>Refcoco-testA</td><td>Refcoco-testB</td><td>Refcoco+-testA</td><td>Refcoco+-testB</td><td>Refcocog-test</td><td>avg</td></tr><tr><td>Vanilla</td><td>92.56</td><td>85.16</td><td>89.02</td><td>79.15</td><td>87.24</td><td>100%</td></tr><tr><td colspan="7">Average Tokens 75%</td></tr><tr><td>Nüwa</td><td>91.76</td><td>84.37</td><td>87.98</td><td>77.18</td><td>86.87</td><td>98.8%</td></tr><tr><td colspan="7">Average Tokens 50%</td></tr><tr><td>Nüwa</td><td>90.04</td><td>82.85</td><td>86.74</td><td>72.65</td><td>85.49</td><td>96.4%</td></tr><tr><td colspan="7"></td></tr><tr><td>Nüwa</td><td>80.71</td><td>72.83</td><td>Average Tokens 25% 73.57</td><td>62.4</td><td>73.96</td><td>83.8%</td></tr></table>

Table 14: VQA Benchmarks performance comparison On QWEN-2.5 VL 7B. Best and second-best results are highlighted.   

<table><tr><td>Methods</td><td>GQA</td><td>POPE</td><td>SQAimg</td><td>MMB-en</td><td>MME</td><td>VQA_text</td><td>avg</td></tr><tr><td>Vanilla</td><td>61.9</td><td>87.9</td><td>77.8</td><td>83.5</td><td>2347</td><td>82.2</td><td>100.0%</td></tr><tr><td colspan="8">Average Tokens 75%</td></tr><tr><td>Nüwa</td><td>60.41</td><td>87.52</td><td>77.98</td><td>83.13</td><td>2340</td><td>77.35</td><td>98.5%</td></tr><tr><td colspan="8">Average Tokens 50%</td></tr><tr><td>Nüwa</td><td>59.93</td><td>87.46</td><td>78.82</td><td>83.02</td><td>2330</td><td>76.03</td><td>98.1%</td></tr><tr><td colspan="8">Average Tokens 25%</td></tr><tr><td>Nüwa</td><td>58.4</td><td>87.06</td><td>78.58</td><td>82.47</td><td>2313</td><td>73.81</td><td>96.9%</td></tr></table>

# B.3 SIMPLE BASELINE COMPARISON EXPERIMENT

To provide a more accurate comparison, we make minor modifications to the baseline settings for each category of methods. Specifically, we only perform simple replacements on the pruning part. For random, we use torch.randprem with a random seed set to seed(44). For pooling, we employ adaptive pooling as shown in Algorithm 2 to accommodate dynamic token inputs and cropping rates. The pseudocode is as follows:

# Algorithm 2 Adaptive Token Pooling Compress

# Require:

Input visual token sequence $X \in \mathbb { R } ^ { N \times d }$   
Original grid dimensions $( H _ { i n } , W _ { i n } )$ (such that $H _ { i n } \times W _ { i n } = N _ { i n } )$   
Token retention ratio $\rho \in [ 0 , 1 ]$

# Ensure:

Pooled visual token sequence $V ^ { \prime } \in \mathbb { R } ^ { N _ { o u t } \times d }$

1: $k _ { t a r g e t } = \left\lfloor N _ { i n } \times \rho \right\rfloor$ ▷ Calculate target token   
2: $\alpha = \hat { H } _ { i n } / W _ { i n }$   
3: $W _ { o u t } = \mathrm { r o u n d } ( \sqrt { k _ { t a r g e t } / \alpha } )$ ; $H _ { o u t } = \mathrm { r o u n d } ( \alpha \times W _ { o u t } )$ ▷ Calculate New grid based original   
aspect ratio   
4: $N _ { o u t } = H _ { o u t } \times W _ { o u t }$ ▷ Final number of output tokens   
5: $F _ { g r i d } = \mathrm { R e s h a p e } ( X , ( 1 , d , H _ { i n } , W _ { i n } ) )$ ▷ Reshape to 2D feature map   
6: $F _ { p o o l e d } = \mathrm { P o o l i n g } ( F _ { g r i d } , \mathbf { o } { }$ utput size = (Hout, Wout)) ▷ This operation automatically handles   
scaling and pooling   
7: X′ = Flatten(Fpooled)   
8: return $X ^ { \prime }$

# B.4 POSITION RECONSTRUCTION EXPERIMENT

Here, we conduct a detailed exploration of the positional reconstruction experiment. For methods like Visionzip that employ PERC, the positional embedding is regenerated for the pruned complete sequence. Assuming ideal pruning, where the target is perfectly extracted, PERC effectively crops the image target and feeds it into an LLM for grounding. This approach naturally cannot output a bounding box based on the original image coordinate system. For PESP used in methods like FastV and SparseVLM, the original position embedding is employed, which performs relatively well but still compromises spatial integrity unless the target is located at the bottom-right corner of the image. This explains why such methods excel in grounding tasks. In contrast, RPME effectively scales the target to restore its original coordinate system, aiding localization of larger objects. However, it still fails with small objects, explaining why Visionzip’s localization accuracy improves marginally after adopting RPME. Here, we provide the pseudocode for RPME:

# Algorithm 3 Relative Position Mapping Extension

Require: Pruned visual token Indices $I \in \mathbb { R } ^ { 1 \times k }$ (sorted ascending, $k$ is the number of pruned tokens) Original Vision Position ID $( s , e )$ (Start and End indices of full visual range)   
Ensure: New Position Embedding $V ^ { \prime } \in \mathbb { R } ^ { k \times d }$ (remapped embeddings for pruned tokens)   
1: $m a x _ { I } \gets I [ k - 1 ]$ $\triangleright$ Last index as reference span   
2: $n e w \_ p o s \gets [ s ]$ $\triangleright$ Anchor first pruned token to start   
3: for $j = 1$ to $k - 1$ do   
4: $\begin{array} { r } { \dot { s } c a l e d \gets c e i l \left( I [ j ] \times \frac { e - s } { m a x _ { I } } \right) + s } \end{array}$   
5: new pos.append(scaled)   
6: end for   
7: $V ^ { \prime } $ get position embedding(new pos) ▷ Retrieve embeddings for new positions; shape $k \times d$   
8: return V ′

# B.5 ATTENTION BLOCK EXPERIMENT

To better understand the multimodal information interaction process within VLMs, we conduct attention blocking experiments on three datasets, as shown in Table 15. We divid tokens into four main categories: (1) System tokens, (2) Visual tokens, (3) Text tokens, and (4) Last tokens, which represent the token that will be used to predict. The attention-blocking experiment is conducted based on LLAVA-1.5 7B. The model is divided into four equal phases according to the decoder layers, with attention blocking applied to each phase.

Table 15: Attention Blocking Experiments on LLAVA-1.5 7B.   

<table><tr><td rowspan="2"></td><td colspan="3">Vision to Vision</td><td colspan="3">Text to Vision</td><td colspan="3">Last to Vision</td></tr><tr><td>Blocked GQA Layer</td><td>MMB</td><td>Refcoco-test</td><td>GQA</td><td>MMB</td><td>Refcoco-test</td><td>GQA</td><td>MMB</td><td>Refcoco-test</td></tr><tr><td>Original</td><td>61.9</td><td>64.7</td><td>58.30</td><td>61.9</td><td>64.7</td><td>58.30</td><td>61.9</td><td>64.7</td><td>58.30</td></tr><tr><td>0-7</td><td>60.5</td><td>61.62</td><td>18.17</td><td>39.38</td><td>51.63</td><td>54.75</td><td>62.55</td><td>64.26</td><td>57.85</td></tr><tr><td>8-15</td><td>55.2</td><td>57.98</td><td>2.64</td><td>53.32</td><td>46.39</td><td>14.11</td><td>62.47</td><td>64.08</td><td>2.01</td></tr><tr><td>16-23</td><td>58.44</td><td>62.37</td><td>19.27</td><td>61.04</td><td>62.45</td><td>15.94</td><td>61.32</td><td>64.26</td><td>64.52</td></tr><tr><td>24-31</td><td>58.81</td><td>62.37</td><td>19.31</td><td>6.12</td><td>63.05</td><td>15.08</td><td>62.12</td><td>64.17</td><td>59.10</td></tr></table>

# B.5.1 BLOCKING ATTENTION FROM VISION TO VISION

In this setting, we block self-attention between visual tokens in each phase. This disrupts the model’s ability to construct global and local visual contexts within the LLM decoder.

Impact on QA Tasks (GQA, MMB): Performance shows a certain degree of decline, but it is not catastrophic. For example, on GQA, even when blocking the most critical layers 8-15 (performance drops from 61.9 to 55.2), the model retains most of its performance. Layer-wise Sensitivity:

Disabling intermediate layers (8-15) causes the most significant performance degradation. This indicates that intermediate layers in the visual encoder are crucial for integrating low-level features extracted by earlier layers (e.g., edges, textures) into semantically meaningful object-level representations. Disabling early layers (0-7) or deep layers (16-31) has relatively minor impacts, likely because early features contain redundancy while deep features are highly abstracted.

Impact on grounding tasks (Refcoco): Performance crashes dramatically. Particularly when blocking layers 8-15, performance plummets from 58.30 to 2.64. Layer sensitivity: Similar to VQA tasks, intermediate layers (8-15) are absolutely critical, validating our previous experimental observations

Summary: In stark contrast to VQA tasks, localization tasks are extremely dependent on spatial and contextual relationships between visual features. This can be explained by our spatial integrity hypothesis and the global coordinate reference system. For instance, to understand “the cup to the left of the table”, the model must first establish spatial adjacency between “table” and “cup” via vision-to-vision attention. Blocking vision-to-vision attention reduces the image to a collection of disjointed, unconnected visual patches, preventing the model from forming a coherent understanding of the scene structure — thus causing the grounding task to fail completely.

# B.5.2 BLOCKING ATTENTION FROM LAST TO VISION

In this setting, we block cross-attention from last tokens to vision tokens in each phase. This directly prevents the model from extracting visual information during decoding.

Impact on VQA tasks (GQA, MMB): Minimal impact on VQA Tasks. GQA performance varies slightly from 61.9 to 62.55 (within error margins, with a marginal improvement), and MMB decreased marginally from 64.7 to 64.26.

Impact on grounding tasks (Refcoco): Catastrophic failure on VG tasks, particularly when blocking layers 8-15, where performance drops to 2.01. Moreover, blocking attention in subsequent stages actually improves accuracy.

Summary: The results demonstrate that answer generation in VQA tasks depends minimally on vision, as disconnecting visual inputs at any stage has a negligible impact on performance. VQA tasks primarily leverage global semantic information extracted from early multimodal interactions. In contrast, the result of grounding tasks contrasts with initial expectations. Integrating Text-toVision attention and gradient-weighted flows from Sec. 2.2 (last-to-text), we hypothesize that visual grounding tasks rely more on spatially aware visual information processed in the model’s midstage. The process then continually leverages textual cues to extract features, enabling integration of complex visual semantics within the visual modality (as analyzed in Sec. 2.2). Furthermore, blocking the Last-to-Vision attention in subsequent stages actually improves accuracy. We attribute this to the fact that subsequent visual information requires higher-level semantic integration to be transferred to the output space, which disrupts the spatial information in later stages.

# B.5.3 BLOCKING ATTENTION FROM TEXT TO VISION

In this setting, we block cross-attention from text tokens to vision tokens in each phase. This blocks multimodal information extraction.

Impact on VQA tasks (GQA, MMB): Performance degrades substantially, especially when blocking early layers (0-7), with GQA dropping from 61.9 to 39.38. Recovery occurs gradually as the blocked layer depth increases.

Impact on grounding tasks (Refcoco): Performance suffers severely across all blocked layers, particularly in mid-to-late stages (8-15, 16-23). This intriguing result highlights that grounding requires sustained multimodal interactions, yet early ones have minimal influence — consistent with anomalies in text-to-vision attention observed in Sec. 2.2.

Summary: These findings indicate that VQA tasks rely on text extracting global abstract features from visual inputs during initial processing for comprehension, with subsequent response generation depending little on vision. In contrast, visual grounding tasks exhibit a different reliance: they depend more on ongoing visual feature extraction in later stages, particularly when spatial understanding emerges, rather than early features, which lack enough spatial information.

Finding 4 Attention blocking experiments further reveal that grounding tasks primarily rely on two sets of information: 1. Abstract semantic information continuously extracted from text tokens to visual tokens; 2. Precise positional information was extracted from the model’s midstage. Meanwhile, most VQA tasks predominantly depend on abstract semantic information extracted from text tokens to visual tokens during the model’s early to mid-stages.

# B.6 BENCHMARKS

We utilize several benchmarks that evaluate a model’s ability to understand static images, including their content, context, and associated textual queries.

GQA (Hudson & Manning, 2019) The GQA benchmark is structured around three core components: scene graphs, questions, and images. It enriches visual content with comprehensive spatial information and object-level attributes. Questions are specifically designed to evaluate models’ ability to comprehend visual scenes and reason about diverse aspects of images.

MMBench (Liu et al., 2023) The MMBench Benchmark evaluates models through three hierarchical levels of abilities. The first level (L-1) focuses on fundamental perception and reasoning capabilities. The second level (L-2) expands into six distinct sub-abilities, while the third level (L-3) further refines these into 20 specific dimensions. This structure enables a granular and comprehensive assessment of a model’s various capabilities. Additionally, MMB-CN is the Chinese version of the benchmark.

MME (Fu et al., 2023) MME comprehensively evaluates models’ perceptual and cognitive abilities across 14 subtasks. By employing manually constructed instruction-answer pairs and concise instructions, it effectively minimizes data leakage and ensures fair assessment of model performance.

MMMU (Yue et al., 2023) MMMU evaluates multimodal models on complex tasks requiring college-level knowledge and reasoning. It includes 11.5K curated questions from exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. Covering 30 image types like charts, diagrams, and chemical structures, MMMU challenges models with advanced perception and domain-specific reasoning.

MMVet (Yu et al., 2023) MMVet defines six core vision-and-language (VL) capabilities: recognition, OCR, knowledge, language generation, spatial awareness, and math. These capabilities integrate to address a range of complex multimodal tasks. MMVet evaluates 16 specific integrations of these capabilities through quantitative assessments.

POPE (Li et al., 2023b) The POPE benchmark systematically evaluates object hallucination in models through a series of binary questions about object presence in images. Using accuracy, recall, precision, and F1 score as metrics, it precisely measures hallucination levels across different sampling strategies.

ScienceQA (Lu et al., 2022) ScienceQA encompasses a wide array of domains, including natural, language, and social sciences, with questions hierarchically organized into 26 topics, 127 categories, and 379 skills. Through this comprehensive structure, it offers diverse scientific questions that effectively evaluate multimodal understanding, multi-step reasoning capabilities, and interpretability.

SEEDBench (Li et al., 2023a) SEEDBench comprises 19,000 human-annotated multiple-choice questions evaluating models across 12 distinct aspects. It comprehensively assesses capabilities in recognizing spatial and temporal patterns within both images and videos.

TextVQA (Singh et al., 2019) TextVQA focuses on the integration of textual information within images. It evaluates a model’s ability to interpret visual elements and embedded text in images through tasks, requiring both visual and textual comprehension to answer questions accurately.

VQA-V2 (Goyal et al., 2016) VQA-V2 evaluates models’ visual perception capabilities through open-ended questions about 265,016 real-world scene images. Each question contains 10 humanannotated ground truth answers, enabling thorough assessment of a model’s ability to interpret and respond to visual queries.

These benchmarks are specifically designed to evaluate a model’s ability to locate textual descriptions of specific objects or regions within an image.

RefCOCO, RefCOCO+, RefCOCOg (Yu et al., 2016) These datasets are standard benchmarks for evaluating referring expression comprehension.

• RefCOCO and $\mathbf { R e f C O C O + }$ provide a large number of images annotated with referring expressions for objects. $\operatorname { R e f C O C O + }$ expands on RefCOCO by increasing the diversity of expressions and objects.

• RefCOCOg is an extension that includes more natural, grammatically complex, and longer referring expressions, challenging models to understand more nuanced linguistic descriptions and their grounding in complex scenes. These datasets typically evaluate models on their ability to accurately locate the target object described by the expression.

# C DETAILED FLOPS CALCULATION

We provide a detailed derivation of the Floating Point Operations (FLOPs) for a VLM with a flexible, layer-wise token pruning mechanism.

# C.1 PRELIMINARIES AND NOTATIONS

We first define the key hyperparameters of the underlying LLM, which follows a standard Transformer architecture. These parameters, summarized in Table 16, form the basis of our calculations.

Table 16: Model architecture parameters used for FLOPs calculation. Example of LLAVA-1.5 7B.   

<table><tr><td>Symbol</td><td>Value</td><td>Description</td></tr><tr><td>L</td><td>32</td><td>Total number of Transformer layers</td></tr><tr><td>H</td><td>4096</td><td>Hidden dimension of the model</td></tr><tr><td>I</td><td>11008</td><td>Intermediate dimension of the Feed-Forward Network (FFN)</td></tr><tr><td>S(l)</td><td>Variable</td><td>Sequence length (number of tokens) at layer l</td></tr></table>

Our calculation assumes that one FLOP corresponds to one multiply-accumulate (MAC) operation, which involves two floating-point operations (a multiplication and an addition). This is a standard convention in analyzing the computational cost of neural networks.

# C.2 FLOPS OF A STANDARD TRANSFORMER LAYER

The computational cost of a single Transformer layer is dominated by two components: the MultiHead Self-Attention (MHA) block and the Feed-Forward Network (FFN). We formulate the FLOPs for each, given a sequence of length $S$ .

Multi-Head Self-Attention (MHA). The MHA mechanism involves four primary matrix multiplication steps:

1. Q, K, V Projection: Projecting the input sequence of shape $( S , H )$ into Query, Key, and Value matrices. This involves three separate weight matrices of size $( H , H )$ .

$$
\mathrm { F L O P s } _ { \mathrm { Q K V } } = 3 \times ( 2 \cdot S \cdot H \cdot H ) = 6 S H ^ { 2 }
$$

2. Attention Score Calculation: Computing the dot product between Query and Key matrices, $( S , H ) \times ( H , S )$ .

$$
\mathrm { F L O P s } _ { \mathrm { S c o r e s } } = 2 \cdot S ^ { 2 } \cdot H
$$

3. Value Aggregation: Multiplying the attention scores (after softmax) with the Value matrix, $( S , S ) \times \bar { ( S , H ) }$ .

$$
\mathrm { F L O P s } _ { \mathrm { V a l u e s } } = 2 \cdot S ^ { 2 } \cdot H
$$

4. Output Projection: Projecting the aggregated value matrix back to the hidden dimension via a weight matrix of size $( H , H )$ .

$$
\mathrm { F L O P s } _ { \mathrm { P r o j } } = 2 \cdot S \cdot H \cdot H = 2 S H ^ { 2 }
$$

The total FLOPs for one MHA block is the sum of these components:

$$
\mathrm { F L O P s } _ { \mathrm { a t t n } } ( S ) = 8 S H ^ { 2 } + 4 S ^ { 2 } H
$$

Feed-Forward Network (FFN). The FFN block consists of two linear transformations with a nonlinear activation in between.

1. Up-Projection: Expanding the hidden dimension from $H$ to $I$ .

$$
\mathrm { F L O P s } _ { \mathrm { u p } } = 2 \cdot S \cdot H \cdot I
$$

2. Down-Projection: Reducing the dimension from $I$ back to $H$ .

$$
\mathrm { F L O P s _ { d o w n } } = 2 \cdot S \cdot I \cdot H
$$

The total FLOPs for one FFN block is:

$$
\mathrm { F L O P s } _ { \mathrm { f f n } } ( S ) = 4 S H I
$$

Thus, the total computational cost for a single Transformer layer is:

$$
\mathrm { F L O P s } _ { \mathrm { l a y e r } } ( S ) = \mathrm { F L O P s } _ { \mathrm { a t t n } } ( S ) + \mathrm { F L O P s } _ { \mathrm { f f n } } ( S )
$$

# C.3 GENERALIZING TO DYNAMIC TOKEN PRUNING

Token pruning methods dynamically alter the sequence length $S$ as data propagates through the model’s layers. A given pruning strategy can be formally described by a pruning schedule, which specifies the sequence length at each layer. Let $S ^ { ( l ) }$ denote the number of tokens present at the input of layer $l$ , where $l \in \{ 0 , 1 , \bar { \ldots } , \mathcal { L } - 1 \}$ .

For a baseline model without pruning, $S ^ { ( l ) }$ is constant for all layers. For a model with pruning, $S ^ { ( l ) }$ becomes a piece-wise constant function that is non-increasing with $l$ . For instance, a method that prunes tokens before layer $l _ { p }$ would have $S ^ { ( l ) } = S _ { \mathrm { i n i t i a l } }$ for $l < l _ { p }$ and $S ^ { ( l ) } = S _ { \mathrm { p r u n e d } }$ for $l \geq l _ { p }$ .

To calculate the total FLOPs for the entire LLM under a specific pruning schedule, we sum the FLOPs of each layer using its corresponding input sequence length. The total computational cost is given by:

$$
\mathrm { F L O P s } _ { \mathrm { T o t a l } } = \sum _ { l = 0 } ^ { \mathcal { L } - 1 } \mathrm { F L O P s } _ { \mathrm { l a y e r } } ( S ^ { ( l ) } )
$$

This generalized Eq. (14) provides a robust framework to accurately quantify the computational savings of any token pruning method, regardless of where or how many times pruning is applied. Our analysis primarily accounts for matrix multiplications, which dominate the computational cost in large Transformers. The cost of other operations, such as LayerNorm, activation functions, and softmax, is considered negligible, consistent with prior work in the field (e.g., Kaplan et al., 2020). By applying this methodology, we ensure a fair and insightful comparison across all evaluated methods in our experiments.

# C.4 COMPUTATIONAL OVERHEAD OF PRUNING METRICS

A comprehensive analysis of token pruning methods must account for the computational overhead incurred by the pruning decision logic itself. While pruning reduces the overall FLOPs of the main computational graph, the process of selecting which tokens to prune introduces an additional, nonnegligible cost. This overhead is critical for a holistic comparison, as a method with a computationally expensive metric might offset the gains from pruning. In this subsection, we formalize the FLOPs required for various pruning metrics.

First, we define the computational cost for common metrics used by the methods under evaluation. Let $S _ { v }$ be the number of vision tokens and $S _ { q }$ be the number of query tokens (e.g., a CLS or text token, where $S _ { q } = 1$ ).

• Attention Score-based Metric: This involves computing the dot product between $S _ { q }$ query vectors and $S _ { v }$ key vectors, both of dimension $H$ . The cost is:

$$
{ \mathrm { F L O P s } } _ { \mathrm { a t t n . s c o r e } } ( S _ { q } , S _ { v } ) = 2 \cdot S _ { q } \cdot S _ { v } \cdot H
$$

• Cosine Similarity Metric: This metric requires computing the pairwise cosine similarity between $S _ { v }$ vision tokens. The dominant cost is the dot product for each of the $\frac { S _ { v } ( S _ { v } - 1 ) } { 2 }$ pairs. The cost is approximated by:

$$
\mathrm { F L O P s } _ { \mathrm { c o s i n e } } ( S _ { v } ) = ( S _ { v } ) ^ { 2 } \cdot H
$$

• L2-Norm Metric: Calculating the L2-norm for $S _ { v }$ vectors of dimension $H$ requires approximately:

$$
\mathrm { F L O P s } _ { \mathrm { n o r m } } ( S _ { v } ) = 2 \cdot S _ { v } \cdot H
$$

Using these formulations, we summarize the total computational overhead for each pruning method in Table 17. The total FLOPs of a method is the sum of the main computation FLOPs (Eq. 14) and the overhead FLOPs detailed here.

Table 17: Summary of Computational Overhead for Different Token Pruning Methods. $\boldsymbol { S _ { v , i } }$ denotes the number of vision tokens at the input of pruning stage $i$ , and $S _ { v , i } ^ { \prime }$ denotes the number of remaining vision tokens after stage $i$ . The initial number of vision tokens is 576.   

<table><tr><td>Method</td><td>Pruning Stage</td><td>Metric Used &amp; Token Count</td><td>Overhead FLOPs Formula</td></tr><tr><td>FSATV</td><td>Stage 1</td><td>Last token attention to vision tokens (Sv,1 = 576)</td><td>FLOPSatt.score(1, 576)</td></tr><tr><td>Pdrop &amp;</td><td>Stage 1</td><td>Text token atention to vision tokens (Sv,1 = 576 → Sv,1 = 66)</td><td>FLOPsattn_score(1, 576)</td></tr><tr><td>SparseVLM</td><td>Stage 2</td><td>Text tokeetio  ii toen (v, = 66 →  =30)</td><td>FLOPsattn_score(1, 66)</td></tr><tr><td></td><td>Stage 3</td><td>Text token attention to vision tokens (Sv,3 = 30 → Sv,3 = 17)</td><td>FLOPSattn_score(1, 30)</td></tr><tr><td>VisionZip</td><td>Stage 1</td><td>CLS token attention to vision tokens (Sv,1 = 576 → Sv,1 = 64)</td><td>FLOPsattn_score(1, 576)</td></tr><tr><td></td><td>Stage 2</td><td>Cosine similarity among remaining vision tokens (Sv,1 = 64)</td><td>+FLOPscosine(64)</td></tr><tr><td>Nüwa</td><td>Stage 1</td><td>CLS token attention to vision tokens (Sv,1 = 576 → Sv,1 = 112)</td><td>FLOPsattn_score(1, 576)</td></tr><tr><td></td><td>Stage 2</td><td>Cne siarity &amp; L2-r on main toens (, = 112)</td><td>+2 * FLOPSscosine(112)</td></tr><tr><td></td><td>Stage 3</td><td>Text-Vision Cosine similarity (Sv = 112)</td><td>FLOPs_cosine(112+N_text)</td></tr></table>

# D VISUALIZE RESULTS

In this section, we supplement with additional visualization results, including some case studies, VAE and OCC visualizations from visual encoders (CLIP and SIGLIP2) and more visual representations of the conclusions in the paper.

# D.1 LLAVA

Attention flow Here, we present the complete attention flow, encompassing all token types.

![](images/6ad9a42849857b86b42eeacc251a66245dd0f3f06096f9f7e865aba7411cdbdd.jpg)  
Figure 7: Complete Attention Flow for VQA and VG Tasks on LLAVA-1.5 7B.

Text-Vsion Alignment Figures 8 and 9 illustrate the layer-by-layer alignment process of vision and text tokens within the LLM. Complementing this, we present the layer-wise similarity heatmap, which together demonstrate the alignment status of multimodal data across different stages.

![](images/b6186a421f357ea608b2173d9f51afb89ad5e99b8418e8015b826dd6a787b18b.jpg)  
Figure 8: Two-Dimensional Visualization of Vision-Text Similarity.

![](images/b37f24c7eb722b4cfdab1a200e80b52b88cd4b74a6b684105aea689226667213.jpg)  
Figure 9: Visualization results of Layer-wise Vision-Text Similarity Heatmap.

Pillar Token We present three types of heatmaps in the Figure 10. The second row displays a heatmap with token L2 norm values, with the top and bottom rows showing tokens that made positive contributions during decoding for the VQA task and grounding task, respectively.

![](images/eda20186665815cdc72ad561da79cc52b0b63490faec3c0712906beccb583dba.jpg)  
Figure 10: Visualization of “register” token and making significant contributions to the prediction (gradient-weighted attention values).

Here, we conduct further visual analysis on the “register” token. As mentioned in Section 3.1.3, these high-norm tokens receive significant attention in both VQA and grounding tasks. We calculate the overlap rate between the top-3 tokens with the highest L2 norm in the ”register” token and the top-3 tokens with the highest weighted attention scores in the VQA and VG tasks, finding a high overlap rate of $72 \%$ in VQA and $6 6 \%$ in VG.

Pruning Results We present some cropped visualization results Figure 11. Our approach preserves the integrity of the global space.

![](images/762aef683d5531727fa073adc8a76629c8555791a39ed84f2a68b09aeccfb080.jpg)  
Figure 11: Visualization results of Pruning results.

# D.2 CLIP

We provide additional visualizations of attention maps and object-centric similarity maps during the CLIP processing (Figure 12).

![](images/8111bf0c9a088bb3509b884b54b1ef5535aa8a7f4ad1cc684c10a79a50931114.jpg)  
Figure 12: More visualization results processed by CLIP-VIT.

# D.3 SIGLIP

Since SIGLIP lacks a CLS token, the attention map visualization is based on the attention values received by each token. We provide some sample visualization results (Figure 14) and the metric — VAE and OCC (Figure 13). By examining visual results and metrics such as OOC, it can be observed that compared to CLIP, SIGLIP exhibits a later phase of fine-grained feature extraction, and its overall trend is less pronounced than that of CLIP.

![](images/17782b930fe4a7b5a1e753f986b2d4583027f0c4f89653917d39260ee48b9b87.jpg)  
Figure 13: The VAE and OCC metric form SIGLIP2.

![](images/5ea61f5154c2f26901ef48d89acc9b58ce08fc710f811d08df61661ced20cd68.jpg)  
Figure 14: More visualization results processed by SIGLIP2.

# E CASE STUDY

In this subsection, we present the results of vision token pruning and localization under several configurations of the Nuwa framework. Specifically, we evaluate three primary settings: 1) token selec- ¨ tion without regional partitioning, 2) employment of Relative Position Mapping Extension (RPME), and 3) the Nuwa with full configuration. Furthermore, we analyze a failure case wherein localization ¨ inaccuracies arise from the model’s inherent misinterpretation of the text and the designated task.

![](images/def7647ca543592963d3397770a2fb87b1872bbc939aa71fd498e3bb73e888e4.jpg)  
Figure 15: A comparison between the original Nuwa setting and the token selection setting without ¨ regional partitioning. Red and green bounding boxes represent the prediction and ground truth, respectively.

Regional partitioning provides a significant advantage As illustrated in Figure 15, the configuration employing regional partitioning demonstrates superior preservation of spatial integrity during token pruning, enabling the observation of all four corners of the image. Conversely, the failure to retain sufficient spatial information appears to impair the model’s perception of global positioning, resulting in predictions that are systematically biased towards the top-left corner. The underlying reasons for this phenomenon are discussed in Section 2.3.

![](images/e9157e170222e9ab30c65ae51697377a82eb1b512bd8461a30fc66e7ddfcbb4b.jpg)  
Figure 16: A comparison between the token selection setting without regional partitioning and the setting with RPME. Red and green bounding boxes represent the prediction and ground truth, respectively.

RPME can mitigate the loss of spatial information. As shown in Figure 16, the prediction results from the RPME-enabled setting are more accurate. Specifically, the localization no longer exhibits a consistent bias towards the top-left corner. However, while the overall position is correct, the shape (i.e., aspect ratio) of the predicted box is problematic. This may be attributed to the image ‘distortion’ introduced by the stretching operations in RPME, which could interfere with the model’s understanding of the object’s geometry.

![](images/2dd615f45db388b810250b2d3f5f802f30f3600a699b6f9831ed48c3ff7ec128.jpg)  
Figure 17: Localization failure attributable to the model’s comprehension. Red and green bounding boxes represent the prediction and ground truth, respectively.

Localization Failures Attributable to Model comprehension. As shown in Figure 17, another potential failure case for localization is presented, which arises from the VLM’s incorrect understanding of the target. In the successful case on the right, we observe a clear positive correlation between the model’s predicted bounding box and the regions where it assigns high attention to the vision tokens. In the failure case on the left, however, the model mistakenly identifies the person in the distance wearing a baseball glove as the target, assigns extremely high attention to that region, and consequently produces an incorrect localization.