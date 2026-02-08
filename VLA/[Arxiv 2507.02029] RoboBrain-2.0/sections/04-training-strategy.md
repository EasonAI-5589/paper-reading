# 4 Training Strategy

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

RoboBrain 2.0 achieves embodied capabilities (spatial understanding, temporal modeling, and chain-of-thought reasoning) through a progressive three-phase training strategy, as shown in Table 1. Starting from a robust vision-language foundation, we introduce escalating complexity in embodied supervision, enabling the model to evolve from static perception to dynamic reasoning and actionable planning in real-world environments.

> 💡 **三阶段渐进训练**:
> ```
> Stage 1: Foundational Spatiotemporal Learning (基础)
>    → 4.8M samples, SFT, full model
>
> Stage 2: Embodied Spatiotemporal Enhancement (增强)
>    → 224K samples, SFT, full model
>
> Stage 3: Chain-of-Thought Reasoning (推理)
>    → Phase 1: CoT-SFT (195K)
>    → Phase 2: RFT/RLVR with GRPO (45K)
> ```

---

### Table 1: Training Configuration

| | Stage-1 SFT | Stage-2 SFT | Stage-3 CoT-SFT | Stage-3 RFT (RLVR) |
|---|---|---|---|---|
| Dataset #Samples | Foundation 4.8M | Embodied 224K | Embodied (Phase 1) 195K | Embodied (Phase 2) 45K |
| Trainable Part | Full Model | Full Model | Full Model | Full Model |
| #Tunable Params | 8.29B or 33.45B | 8.29B or 33.45B | 8.29B or 33.45B | 8.29B or 33.45B |
| Per-device Batch | 2 | 2 | 4 | 1 |
| Gradient Accum | 2 | 2 | 2 | 2 |
| LR | 1×10⁻⁴ | 1×10⁻⁵ | 1×10⁻⁵ | 1×10⁻⁶ |
| Epoch | 1 | 1 | 1 | 3 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| DeepSpeed | — | — | Zero3 | Zero3 |
| Weight Decay | 0.1 | 0.1 | 0.1 | 0.0 |
| Warmup Ratio | 0.01 | 0.01 | 0.03 | 0.00 |
| LR Schedule | Cosine | Cosine | Cosine | Cosine |
| Max Seq Length | 16384 | 16384 | 32768 | 32768 |
| Max Compl Length | — | — | — | 1024 |
| Num Completions | — | — | — | 8 |
| GPU Nums | 16/64 × 8 | 16/64 × 8 | 4 × 8 | 4 × 8 |

> 💡 **Table 1 批读**:
> ```
> 训练规模递减，精度递增:
> Stage 1: 4.8M samples, LR=1e-4, 128/512 GPUs → 大规模基础训练
> Stage 2: 224K samples, LR=1e-5 → 精细化 embodied 数据
> Stage 3: 195K+45K, LR=1e-5→1e-6 → CoT+RL 微调
>
> 关键细节:
> - 全程 full model fine-tune（不是 LoRA!）
> - Stage 3 seq length 翻倍: 16384 → 32768（因为 CoT 输出更长）
> - RFT 阶段: 8 completions, epoch=3（采样多个回答 + 多轮优化）
> - Stage 1-2 不用 DeepSpeed Zero3, Stage 3 才用（内存需求更大）
> ```

---

### 4.1 Stage 1: Foundational Spatiotemporal Learning

The first stage focuses on building general capabilities in spatial perception and temporal understanding. We fine-tune the model on large-scale multimodal datasets covering dense captioning, object localization, interleaved image-text documents, and basic video QA, along with referring expression comprehension. These datasets span common physical scenes and interaction patterns, helping the model develop fundamental grounding for objects, spatial relations, and motion events. This stage lays the groundwork for understanding egocentric video streams and spatially anchored instructions.

> 💡 **Stage 1**: 4.8M 基础数据，建立视觉-语言-空间的基础能力。
> 任务: dense captioning, object localization, image-text interleaving, video QA, referring expression
> 这一阶段的数据来自 Section 3.1 (General VQA) + 部分 Section 3.2 (Spatial)

---

### 4.2 Stage 2: Embodied Spatiotemporal Enhancement

To better align the model with embodied tasks, we introduce a carefully curated collection of high-resolution, multi-view, and egocentric video datasets, along with instruction-augmented navigation and interaction data. Tasks include viewpoint-aware referring expressions, 3D affordance estimation, and object-centric scene graph construction. This stage of training emphasizes the modeling of long-horizon temporal dependencies, enabling the model to reason over extended sequences of actions and observations. Additionally, it incorporates multi-agent coordination scenarios, where the model learns to interpret and predict the behaviors of other agents in shared environments. To support these capabilities, we employ extended sequence lengths and multi-camera input encoding, allowing the model to process and fuse visual information from multiple viewpoints simultaneously. Through this training stage, the model can integrate historical visual cues with current instructions, fostering more coherent long-horizon planning, robust scene understanding, and adaptive decision-making in dynamic, interactive settings.

> 💡 **Stage 2**: 224K embodied 数据，精细化空间+时序能力。
> ```
> 新增能力:
> ├── Viewpoint-aware referring
> ├── 3D affordance estimation
> ├── Scene graph construction
> ├── Long-horizon temporal dependencies
> └── Multi-agent coordination
> ```
> 数据来自 Section 3.2 (Spatial) + Section 3.3 (Temporal) 的核心子集

---

### 4.3 Stage 3: Chain-of-Thought Reasoning in Embodied Contexts

In the third stage, we augment the model's high-level reasoning capabilities using Chain-of-Thought (CoT) methodology, following the two-phase framework of Reason-RFT [62]: CoT-based Supervised Fine-Tuning (CoT-SFT) and Reinforcement Fine-Tuning (RFT). We leverage multi-turn reasoning examples from both synthetic and real-world embodied scenarios, encompassing long-horizon task planning, manipulation prediction, closed-loop interaction, spatiotemporal understanding, and multi-robot collaboration, sourced from Section 3. Specifically, (1) CoT-SFT Phase: We annotate 10% of the constructed training data with CoT rationales annotated by GPT-4o [22] with custom prompts, then perform supervised fine-tuning for initial model from Stage 2. (2) RFT Phase: An additional 10% of the constructed training data is sampled to collect model's responses, with incorrect answers curated into a reformatted training set (e.g., multiple-choice questions or LaTeX/numerical answers). Optimization employs Group Relative Policy Optimization (GRPO) [17], guided by a composite reward function evaluating both answer accuracy and format correctness.

> 💡 **Stage 3: CoT + RLVR (两个 Phase)**:
> ```
> Phase 1: CoT-SFT (195K samples)
> ├── 取 10% 训练数据
> ├── GPT-4o 标注 CoT rationales
> └── SFT 微调 Stage 2 的模型
>
> Phase 2: RFT/RLVR (45K samples)
> ├── 取另外 10% 训练数据
> ├── 收集模型回答，过滤错误答案
> ├── 重新格式化为多选题 / LaTeX 答案
> └── GRPO 优化: reward = accuracy + format correctness
> ```
> **关键设计**:
> - **Reason-RFT framework [62]**: 先 CoT-SFT 再 RFT，渐进式引入推理
> - **只用 10%+10% 的数据**: 不是全量数据都加 CoT，这很务实（GPT-4o 标注成本高）
> - **GRPO (DeepSeek 的方法)**: Group Relative Policy Optimization，不需要 reward model
> - **Composite reward**: accuracy + format correctness（确保输出格式正确）

---

## 💡 Section 总结

### 训练路线图
```
Qwen2.5-VL (pretrained)
    │
    ▼ Stage 1: 4.8M samples, LR=1e-4
Foundation Model (spatial + temporal basics)
    │
    ▼ Stage 2: 224K samples, LR=1e-5
Embodied Model (3D affordance, scene graph, multi-agent)
    │
    ▼ Stage 3a: 195K CoT-SFT, LR=1e-5
CoT-Enhanced Model
    │
    ▼ Stage 3b: 45K GRPO, LR=1e-6, 3 epochs
RoboBrain 2.0 Final
```

### 核心洞察
1. **Full fine-tune 全程**: 不用 LoRA（和 1.0 的 A-LoRA/T-LoRA 完全不同）
2. **数据量递减、精度递增**: 4.8M → 224K → 195K → 45K
3. **CoT 是关键升级**: Stage 3 让模型学会 "think step by step"，这在 embodied 场景中特别重要
4. **GRPO 而非 PPO**: 更简单的 RL 方法，不需要单独的 reward model
