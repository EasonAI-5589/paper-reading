[← 返回 README](../README.md)

# 3. Methods

## 📌 预览
实验设计：250 万患者数据预训练 16 个模型（4 架构 × 多上下文长度），用 EHRSHOT 14 个二分类任务评估，按三个 EHR 属性分层分析。

---

Our goal is to measure how non-transformer architectures, context length, and the unique properties of EHR data impact performance on clinical prediction tasks. We pretrain 16 models across four architectures and six context lengths on the structured EHR data of 2.5M patients. We evaluate each model on 14 binary classification tasks from the EHRSHOT benchmark (Wornow et al., 2023), as detailed in Section 3.2. We stratify our results on the degree to which each patient exhibits 3 EHR-specific properties – token repetition due to copy-forwarding, irregularity of time intervals between tokens, and increased complexity of tokens due to disease progression – which we hypothesize may influence the efficacy of longer context models.

> 💡 **实验规模**: 16 个模型 = 4 架构 × 4-6 个上下文长度。关键设计是所有模型统一 ~120M 参数，隔离上下文长度和架构的影响。

---

## 3.1 Model Training

### 3.1.1 Problem Setup

In this paper, we focus exclusively on the structured data within a longitudinal (i.e. full-length) EHR – i.e., diagnoses, medications, lab tests, procedures, visits, and other observational data. Our dataset consists of $n$ patients $X = \{X_1, ..., X_n\}$. For each patient $i$ we have their structured EHR data $X_i$, which is composed of a sequence of chronologically ordered clinical events $X_{ij}$:

$$X_i = \{X_{i1}, X_{i2}, ..., X_{i|X_i|}\}$$

We refer to $X_i$ as a "patient timeline", where each clinical event is a tuple of the form $(t_{ij}, c_{ij}, v_{ij})$. Here, $t_{ij}$ is the timestamp, $c_{ij} \in \mathcal{C}$ is a medical code drawn from a fixed medical ontology ($\mathcal{C}$), and $v_{ij} \in \mathcal{V}_c \cup \mathcal{V}_n \cup \emptyset$ is an optional value, either categorical ($v_c$) or numeric ($v_n$):

$$X_{ij} = (t_{ij}, c_{ij}, v_{ij})$$

Events are sorted by time such that $t_{ij} \le t_{i(j+1)} \forall j$. This formulation of EHR data is also referred to as the "event stream format" (McDermott et al., 2023).

> 💡 **数据形式化**: 每个临床事件 = (时间戳, 医学编码, 可选值)。值可以是分类型（如：阳性/阴性）或数值型（如：血压 120mmHg）。这种"事件流"格式本质上就是一个时间戳序列，非常适合自回归建模。

For our experiments, we use a dataset of deidentified longitudinal EHRs sourced from an academic medical center that have been formatted under the OMOP Common Data Model (Sciences & Informatics, 2021). We refer to this dataset as EHR-OMOP. We use **2.5M** patients (covering 3.5B clinical events) for training, and hold out **0.5M** patients as a validation set. The average patient has 1,364 total and 237 unique events. Additional information can be found in Appendix Section A.

> 💡 **数据规模**:
> - 训练：250 万患者，35 亿临床事件
> - 验证：50 万患者
> - 平均每患者 1,364 事件，237 个不重复事件
> - 数据来自单一学术医学中心（Stanford），OMOP CDM 格式

---

### 3.1.2 Tokenization

Given a patient timeline $X_i$, we must convert it into a sequence of tokens $T_i$ that our models can ingest. Thus, we must map each $X_{ij} = (t_{ij}, c_{ij}, v_{ij})$ to some set of token(s) $T_{ij} = \{T_{ij1}, ..., T_{ijk}\}$. We use the same vocabulary used by the prior SOTA model on the benchmark we use for evaluation, EHRSHOT (Wornow et al., 2023). Each clinical "event" in a patient's timeline has a single "code" associated with it. Each "code" then gets converted into a single "token" within our vocabulary via the following process. First, all unique codes $c \in \mathcal{C}$ that occur at least once in our training dataset are assigned a unique token. Second, all codes that are associated with categorical values are assigned a unique token for each possible associated categorical value. Third, all codes associated with numerical values are assigned a unique token for each decile within the range of values attained in our training dataset. After sorting all tokens by their information content, the top 39811 tokens were kept as our vocabulary, and all models share this same vocabulary. Please see Appendix Section D for additional details on the token generation and selection process.

> 💡 **Tokenization 策略**:
> - 每个临床事件 → 1 个 token
> - 数值型值：按十分位分桶（如贫血 lab → 10 个不同 token 代表不同严重程度）
> - 分类型值：每个类别一个 token
> - 词表大小：39,818（39,811 + 7 特殊 token）
> - 与 CLMBR-t-base 使用完全相同的词表，确保公平比较
> 
> 注意：这里**没有显式编码时间信息**——时间通过位置编码隐式捕获。后面会实验 Artificial Time Tokens (ATT) 但发现效果更差。

---

### 3.1.3 Architectures

We evaluate four models – GPT (Brown et al., 2020), Llama (Team, 2024), Mamba (Gu & Dao, 2024), and Hyena (Poli et al., 2023a) – at the 120 million parameter scale using their default HuggingFace implementations. (see Appendix Section C for details on each architecture and Appendix Table 6 for exact configurations). We evaluate each model across various context lengths $L \in \mathcal{L}$, with $\mathcal{L} = \{512, 1k, 2k, 4k\}$ for the transformer-based models (GPT and Llama) and $\mathcal{L} = \{1k, 4k, 8k, 16k\}$ for the subquadratic models (Mamba and Hyena). The ranges are different given the poor computational scaling of transformers and our limited compute.

> 💡 **架构对比**:
> | 架构 | 类型 | 上下文范围 | 位置编码 | 复杂度 |
> |------|------|-----------|---------|--------|
> | GPT | Transformer | 512-4k | Absolute | O(n²) |
> | Llama | Transformer | 512-4k | RoPE | O(n²) |
> | Mamba | SSM | 1k-16k | 无 | O(n) |
> | Hyena | Long Conv | 1k-16k | Hyena PE | O(n log n) |
> 
> 注意 Transformer 只到 4k（因为 O(n²) 太贵），亚二次架构从 1k 开始到 16k。

For pretraining, we employ an autoregressive next-token prediction objective with cross entropy loss. We sample one subsequence of min{L, |T_i|} tokens from each patient $i$'s timeline per epoch and train each model for 2 billion tokens.

> 💡 **训练策略**: 标准 next-token prediction。每个 epoch 从每个患者的 timeline 中**随机采样**一段长 L 的子序列。总训练量 2B tokens。注意这里采样的是随机子序列而非最后 L 个 token（推理时用的是最后 L 个）。

---

## 3.2 Evaluation

We use the EHRSHOT clinical prediction benchmark for all of our downstream evaluations (Wornow et al., 2023). EHRSHOT consists of 15 clinical prediction tasks based on a dataset of 7k patients' longitudinal EHRs. The primary evaluation metric is AUROC, and Brier scores are also reported. We only consider binary classification tasks, thus we exclude the multilabel Chest X-Ray Findings task. We use the remaining 14 tasks from the EHRSHOT benchmark for our evaluations, which are broadly grouped into three categories:

- **Operational Outcomes**: ICU Transfer, 30-day Readmission, Long Length-of-Stay
- **Anticipating Lab Test Results**: Thrombocytopenia, Hyperkalemia, Hypoglycemia, Hyponatremia, Anemia
- **Assignment of New Diagnoses**: Hypertension, Hyperlipidemia, Pancreatic Cancer, Celiac Disease, Lupus, Acute MI

> 💡 **EHRSHOT Benchmark**:
> - 14 个二分类任务，3 大类
> - 7k 患者的纵向 EHR
> - 主指标：AUROC；辅助指标：Brier Score
> - **不做微调**：用 frozen 表征 + logistic regression head
> - 这意味着模型性能完全取决于预训练学到的表征质量

For our evaluations, we use the same context length that was used during pretraining. We thus sample the last min{L, |T_i|} tokens for each patient prior to the relevant prediction time for a task, then take the embedding of the last token in that sequence as our representation for that patient. We evaluate our models under the zero-shot, few-shot, and "All" data setting, with detailed results for zero- and few-shot evaluation provided in Appendix Sections G and H. All EHRSHOT scores reported in the main results use the "All" data setting. To be consistent with the original EHRSHOT benchmark, we do not finetune our base models – instead, we train a logistic regression head on top of the frozen representations created for each patient. Additional details are in Appendix Section A.

> 💡 **评估方法要点**:
> - 推理时取 prediction time 之前的最后 L 个 token
> - 用最后一个 token 的 embedding 作为患者表征
> - frozen 表征 + logistic regression（不微调 base model）
> - 这和 CLMBR-t-base 的评估方式完全一致

---

## 3.3 EHR-Specific Properties

In the following subsections, we define metrics to quantify three properties of EHR data that distinguish it from modalities such as natural language – repetitiveness due to copy-forwarding, irregular intervals of time between events, and a natural trend towards increased token complexity over time due to disease progression. Please see Figure 1c for an overview.

For all three metrics, we first apply them to the EHR-OMOP validation dataset to measure the extent to which a large corpus of real-world EHR data exhibits these properties. Second, we apply two of the EHR-specific metrics – repetitiveness and irregularity – to the EHRSHOT dataset to stratify individual patients based on how much they exhibit each property.

### 3.3.1 Copy-Forwarding Leads to Noisy Token Repetition

**EHR v. NLP.** Copy-forwarding refers to the practice of recording the same diagnosis across multiple visits, typically for chronic conditions or billing purposes (Thornton et al., 2013; Calder et al., 2024; Weis & Levy, 2014). This leads to higher levels of event repetition within the EHR. We hypothesize that repetition could worsen model performance by crowding information out of a limited context window. A long context model might be better equipped to handle this range of possibilities.

**Metrics.** To quantify the prevalence of copy-forwarding in a sequence, we calculate its n-gram repetition rate (RR), i.e., the proportion of n-grams in the sequence that are repeated at least once. A higher RR implies a more repetitive sequence.

> 💡 **Copy-Forwarding 的影响机制**: 假设上下文窗口 512 token，如果一个慢性病诊断在每次就诊时都被重复记录 50 次，那么实际有效信息只有 512-49=463 token。长上下文可以缓解这个问题——16k 窗口中即使有 50 个重复也只占 0.3%。

---

### 3.3.2 Time Intervals Between Events Are Highly Irregular

**EHR v. NLP.** In natural language, consecutive tokens uniformly have the same "distance" of 1 position. In EHR data, however, a patient might wait days, weeks, or even years between visits to the hospital (McDermott et al., 2023). This means consecutive EHR events can have vastly different "distances" in time. We hypothesize that patients with more "irregular" sequences, i.e., a greater variety of inter-event time intervals, are more difficult to model as they present a more complex mix of timespans over which a model must reason.

**Metrics.** We quantify irregularity as the standard deviation of time intervals between every pair of consecutive events. A higher standard deviation implies a more irregular sequence.

> 💡 **不规则时间的挑战**: 大多数患者的事件间隔标准差在 115 天到 3.2 年之间。这意味着在同一个上下文窗口内，模型需要同时理解"同一天内的多次检查"和"间隔三年的两次就诊"——时间尺度差异极大。
> 
> 对 agent memory 的启示：agent 的记忆也面临类似问题——有些记忆间隔几秒，有些间隔几个月。

---

### 3.3.3 Disease Progression Causes Increased Token Complexity Over Time

**EHR v. NLP.** Disease progression refers to the evolving nature of a patient's health over time. As people age, they experience an increase in the variety, frequency, and complexity of diseases they experience due to declining immunity and the increased likelihood of developing comorbidities (Fabbri et al., 2015). In natural language, earlier tokens tend to help in predicting later tokens, and thus perplexity is inversely correlated with a token's position in a prompt (Kaplan et al., 2020). Since disease becomes more complex over time, however, it was unclear if this trend holds for EHR data.

**Metrics.** To quantify disease complexity over time, we apply our trained EHR FMs to calculate the median perplexity at each token position across a sample of 20,000 patients from the EHR-OMOP validation set.

> 💡 **Perplexity 趋势的根本差异**:
> - **NLP**: 后面 token 的 perplexity↓（更多上下文 → 更好预测）
> - **EHR**: 后面 token 的 perplexity↑（疾病复杂度↑ → 更难预测）
> 
> 这是一个深刻的发现：在 EHR 中，看到更多历史并不能让预测更容易，因为疾病本身在变得更复杂。这与"更多记忆 = 更好决策"的直觉相矛盾。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 训练患者数 | 2.5M |
| 临床事件数 | 3.5B |
| 验证患者数 | 0.5M |
| 模型参数 | ~120M |
| 词表大小 | 39,818 |
| 评估任务数 | 14（二分类） |
| 训练量 | 2B tokens |

### 核心洞察
1. 实验设计通过统一参数量（~120M）隔离了架构和上下文长度的影响
2. Tokenization 不显式编码时间，依赖位置编码
3. 三个 EHR 属性的量化指标简单直观：n-gram RR、时间间隔标准差、perplexity by position
