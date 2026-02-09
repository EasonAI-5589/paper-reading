# CRAVE: An Evaluation Framework for LLM-Generated Video Chapters

**Yichen Guo** · Nanyang Technological University


---

## §1 Introduction: Why Video Chaptering Matters

The rapid growth of long-form video content has made efficient content navigation increasingly important. **Video chaptering** — automatically segmenting videos into semantically coherent sections with descriptive titles — enables users to quickly locate relevant content, improves search and recommendation systems, and enhances overall viewing experience.

Despite recent progress, **systematically evaluating** LLM-generated chapters remains an open problem:

| Challenge | Description |
|---|---|
| **Temporal ambiguity** | Chapter boundaries are inherently ambiguous — the same video can be validly divided at different levels of granularity |
| **Semantic relevance** | Models tend to produce generic or repetitive descriptions that fail to capture distinctive content |
| **Hallucination** | LLMs generate plausible but factually incorrect titles that are difficult to detect with reference-based metrics alone |

These compounding challenges motivate the integration of **human feedback** into the evaluation and training pipeline.


---

## §1 Related Work: Timeline of Video Chaptering Research

![Timeline of video chaptering research](figures/fig1-related-work.png)

Recent advances have progressively scaled video chaptering from short clips to hour-long content:

| Year | Work | Key Contribution |
|---|---|---|
| 2015 | **CIDEr** (CVPR) | Consensus-based captioning metric via TF-IDF n-gram similarity |
| 2020 | **SODA** (ECCV) | Story-oriented dense captioning evaluation with temporal matching |
| 2023 | **VidChapters-7M** (NeurIPS) | First large-scale chaptering benchmark: 817K videos, 7M chapters |
| 2025 | **Chapter-Llama** (CVPR) | LLM-based pipeline for hour-long videos with ASR + frame captioning |
| 2025 | **ARC-Chapter** (arXiv) | MLLM + GRPO, F1: 45.3→59.3, CIDEr: 100.9→186.6, introduced GRACE metric |

**The gap:** Despite this progress, **no existing work has incorporated human evaluation** into the video chaptering pipeline.

We propose **CRAVE** (**C**ritique-**R**eward **A**ligned **V**ideo Chapter **E**valuation), the first framework to integrate human feedback into video chaptering, combining a Critique-Based Reward Model with GRPO to close the loop between automatic metrics and human judgment.


---

## §2 Evaluation Metrics: Three Dimensions

![Evaluation framework: three dimensions with five metrics](figures/fig2-technical-indicators.png)

To systematically assess LLM-generated chapters, we evaluate along **three complementary dimensions**:

| Dimension | What It Measures | Downstream Impact |
|---|---|---|
| ⚙️ **Quality** | Whether chapter boundaries are temporally accurate and structurally coherent | A prerequisite for reliable **navigation** |
| 🎯 **Relevance** | Whether chapter titles faithfully and specifically describe the corresponding content | Directly affects **search and discoverability** |
| 👤 **Utility** | Whether chapters actually improve navigation, engagement, and content consumption | Captures the **real user** impact |


---

## §2.1 Quality: Temporal Segmentation Metrics

Quality metrics measure the temporal overlap between predicted chapter boundaries and ground-truth segments.

### tIoU (temporal Intersection over Union)

Greedily matches predicted segments to ground-truth:

$$\text{tIoU} = \frac{1}{|M|}\sum_{(p_i, g_j) \in M} \frac{|p_i \cap g_j|}{|p_i \cup g_j|}$$

> Per-chapter temporal accuracy. Simple, intuitive, but sensitive to granularity mismatch.

### F1@IoU

Averages the harmonic mean of precision and recall across IoU thresholds:

$$\text{F1} = \frac{1}{|\Theta|}\sum_{\theta \in \Theta} \frac{2 P_\theta R_\theta}{P_\theta + R_\theta}, \quad \Theta = \{0.5, 0.55, \ldots, 0.95\}$$

> Strict: a prediction counts as correct only if overlap ≥ θ. Penalizes both missing and spurious chapters.


---

## §2.2 Relevance: Captioning and Semantic Quality Metrics

Relevance metrics evaluate the semantic quality of generated chapter titles — how faithfully and specifically they describe the video content.

### CIDEr

TF-IDF weighted n-gram cosine similarity, rewarding **specificity**:

$$\text{CIDEr}(c, R) = \sum_{n=1}^{N} w_n \cdot \frac{\mathbf{g}^n(c) \cdot \mathbf{g}^n(R)}{\|\mathbf{g}^n(c)\| \; \|\mathbf{g}^n(R)\|}$$

> "iPhone Camera Night Mode" scores higher than "Phone Review" — TF-IDF up-weights rare, informative n-grams.

### SODA

Dynamic programming for temporally optimal one-to-one matching, then F-measure over METEOR:

$$P = \frac{\sum_{(p,g) \in M^*} \text{METEOR}(p,g)}{|\mathcal{P}|}, \quad R = \frac{\sum_{(p,g) \in M^*} \text{METEOR}(p,g)}{|\mathcal{G}|}, \quad \text{SODA} = \frac{2PR}{P+R}$$

> Penalizes both **redundant** chapters (|P| ≫ |G|) and **missing** chapters.

### GRACE (Granularity-Robust)

Addresses granularity ambiguity — different annotators segment at different levels of detail. Many-to-one matching via DTW:

$$\text{GRACE}(P, G) = \sum_{(P_i, G_i) \in M^*} \phi(P_i, G_i) \cdot \text{BERTScore}(P_i, G_i)$$

where $\phi$ is the mean temporal IoU within each matched group. Jointly evaluates temporal alignment and semantic similarity — the most comprehensive single metric.


---

## §2.3 Utility: Human-Centered Evaluation

Behavioral signals (CTR, watch time) cannot accurately assess individual chapter quality. Therefore, following ARC-Chapter's richer structural output (title, abstract, timestamps), we enrich the chapter representation and introduce a protocol where **human reviewers directly rate** generated chapters across standardized dimensions.

### The fundamental limitation of automatic metrics:

| ✅ Auto metrics CAN | ❌ Auto metrics CANNOT |
|---|---|
| Measure temporal overlap (tIoU, F1) | **Watch the video** and verify titles against actual visual/audio content |
| Compare titles to reference text (CIDEr, SODA) | **Detect hallucinations** — fluent titles score high but describe non-existent content |
| Handle granularity variation (GRACE) | **Judge navigation quality** — is the structure intuitive for a human? |

> All existing metrics are **text-vs-text**. None performs **cross-modal verification** (text vs. video). A hallucinated title using correct vocabulary will fool every automatic metric. This motivates our CRAVE framework.


---

## §3 CRAVE Framework: The Base Pipeline

![Current offline pipeline (ARC-Chapter)](figures/fig3-offline-pipeline.png)

We build upon **ARC-Chapter**, adopting its two-stage training pipeline as our foundation:

- **Input:** Video frames (frozen ViT) + Timestamped ASR (Whisper)
- **Model:** Trainable MLLM (Qwen2.5-VL-7B)
- **Stage 1:** Supervised fine-tuning (SFT) on chaptering data
- **Stage 2:** GRPO reinforcement learning with automatic temporal reward GRACE(φ)

**Output types:**
- Short titles (e.g., "Camera Night Mode Test")
- Structural chapters (title + abstract + timestamps)
- Timestamped video descriptions

**The limitation:** The model learns *where* to place boundaries but has **no signal for *what* to call them** — no semantic quality feedback, no hallucination penalty.


---

## §3.2 CRAVE: Critique-Based Reward Model + GRPO

![CRAVE framework: closed-loop with human feedback](figures/fig4-rlhf-loop.png)

CRAVE introduces **human review** and a **Critique-Based Reward Model** to create a closed-loop training system. The combined reward integrates quality, relevance, and utility:

$$R = \alpha \cdot R_{\text{quality+relevance}}(\text{GRACE}(\phi)) + \beta \cdot R_{\text{utility}}(\text{critique + score})$$

| Component | What It Handles | Why |
|---|---|---|
| **α · R_auto** (GRACE) | Temporal + semantic precision | Objective, scalable — machines handle this well |
| **β · R_human** (Critique RM) | Hallucination, informativeness, user satisfaction | Only humans can verify cross-modally |

**Initial weights:** α = 0.4, β = 0.6 (tuned empirically). KL regularization prevents catastrophic forgetting.

### Why Critique-Based? (following MM-RLHF, ICML 2025)

| Traditional Scalar RM | Critique-Based RM |
|---|---|
| Input → black-box score | Input → **evaluation reasoning** → score |
| Not interpretable | **Explainable**: inspect *why* a set was rated low |
| Overfits to surface patterns | Reasoning constrains scoring → better calibrated |


---

## §3.3 Human Review Protocol

For each sampled video, the model generates **2–3 candidate chapter sets**. Reviewers evaluate through three tasks:

| Task | What Reviewers Do | Training Signal |
|---|---|---|
| **A. Dimension Scoring** | Rate on 5-point anchored scale: temporal accuracy, semantic relevance, structural completeness | Absolute quality scores |
| **B. Preference Ranking** | Rank chapter sets best-to-worst, producing pairs $(y_w \succ y_l)$ | **Preference pairs** — primary RM signal |
| **C. Written Critique** | Explain reasoning ≥50 words + tag error types (hallucination, inaccurate segmentation, etc.) | **Critique text** for RM training |

### Consistency Mechanisms (adapted from MM-RLHF)

| Mechanism | Implementation | Frequency |
|---|---|---|
| **Calibration sessions** | All reviewers score same videos, discuss disagreements | Weekly |
| **Golden set** | 10% known-answer items to detect annotator drift | Continuous |
| **Cohen's κ > 0.7** | Minimum inter-rater agreement threshold | Monthly |
| **Expert audit** | Domain experts review flagged data | Continuous |

### Scoring Anchor Example (Semantic Relevance):

| Score | Example |
|---|---|
| **5** | "iPhone 15 Pro Camera Night Mode Demo" — precise, information-rich |
| **3** | "Phone Review Part 3" — partially relevant, generic |
| **1** | "Unboxing" (no unboxing in video) — hallucinated |


---

## §3.4 Common LLM Failures in Video Chaptering

LLM-generated chapters are susceptible to three characteristic failure modes:

| Failure Mode | Description | Why Auto Metrics Miss It |
|---|---|---|
| 🔴 **Hallucination** | Generating plausible but factually incorrect titles unsupported by the video content | Fluent text scores high on CIDEr; only cross-modal verification catches it |
| 🟠 **Inaccurate Segmentation** | Over-segmenting coherent topics or under-segmenting distinct topics | tIoU averages mask structural problems |
| 🟡 **Generic Titling** | Defaulting to vague labels ("Part 1") that provide no descriptive value | Low specificity not always penalized by all metrics |

Since these errors can escape or even **inflate** automatic metrics, human evaluation is essential for both detecting them and providing the corrective signal that drives CRAVE's continuous improvement loop.

### How Errors Close the Loop:

Error tags from reviewers (Task C) → Critique-Based RM training → **error-type-specific tracking** (e.g., "hallucination rate ↓40% after iteration 3") → informed sampling priorities for next cycle.


---

## §4 Conclusion

| Component | Role | Innovation |
|---|---|---|
| **§2 Evaluation Dimensions** | Define *what* to evaluate | Impact-driven: Quality → Navigation, Relevance → Search, Utility → UX |
| **§2 Five Metrics** | Define *how* to measure | tIoU, F1@IoU, CIDEr, SODA, GRACE |
| **§3 CRAVE Framework** | Close the loop with humans | **First human feedback pipeline for video chaptering** |
| **§3 Error Taxonomy** | Ground the loop in failure modes | Hallucination, segmentation, generic titling → RM training labels |

### Key Contributions:

1. 🏆 **First** to integrate human evaluation into video chaptering — confirmed by literature search across 2024–2025
2. 🔗 **Unified closed-loop system** — evaluation dimensions, metrics, and RLHF as one pipeline
3. 👁️ **Cross-modal verification** — humans verify titles against actual video content (what no auto metric can do)
4. ⚖️ **Combined Reward GRPO:** $R = \alpha \cdot R_{\text{auto}}(\text{GRACE}) + \beta \cdot R_{\text{human}}(\text{Critique RM})$
5. 🏷️ **Critique-Based RM** — explainable reward model with evaluation reasoning before scoring


---

## Thank You

**Yichen Guo** · yichen013@e.ntu.edu.sg · Nanyang Technological University

### Key References
- **ARC-Chapter** (arXiv 2025) — MLLM + GRPO + GRACE, F1: 45.3→59.3
- **MM-RLHF** (ICML 2025) — Critique-Based RM, 120K preference pairs
- **VidChapters-7M** (NeurIPS 2023) — 817K videos, 7M chapters
- **Chapter-Llama** (CVPR 2025) — LLM-based chaptering pipeline
- **DeepSeek-R1** (2025) — GRPO: Group Relative Policy Optimization

