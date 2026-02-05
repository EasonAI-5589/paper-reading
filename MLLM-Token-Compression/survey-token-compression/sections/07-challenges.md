# 7. Open Challenges and Future Work

## 7.1 Lack of Theoretical Understanding

> Most existing approaches remain largely **experience-driven** and lack rigorous theoretical grounding.
>
> ==问题：大多数方法是经验驱动，缺乏理论基础==

> They rely on heuristic intuition and limited empirical validation. Consequently, they often exhibit **poor transferability** across datasets, architectures, and modalities.
>
> ==后果：跨数据集/架构/模态的迁移性差==

**核心问题：**

> A key weakness lies in the absence of a **principled theory of token importance**. Current practices—such as ranking tokens by attention weights, similarity, or mutual information—lack causal or generalization-based justification.
>
> ==缺乏 token 重要性的原理性理论==

> These metrics indicate **correlation rather than necessity**, offering little explanation of whether the retained tokens are truly sufficient for the downstream objective.
>
> ==现有指标显示相关性而非必要性==

**未来方向：**
- 连接 token 选择与充分性、因果性、鲁棒性
- 从 ad-hoc heuristics 走向 principled understanding

---

## 7.2 Lack of Task- and Content-Aware Adaptivity

> Most existing strategies operate in a **task-agnostic and content-agnostic** manner, applying a fixed compression ratio regardless of task type or visual complexity.
>
> ==问题：固定压缩率，不考虑任务和内容==

**关键观察 (M³):**

> For most benchmarks crafted from natural scenes (such as COCO), can be handled well with only **9 tokens per image**. In contrast, tasks like document understanding or OCR require **144~576 tokens per image**.
>
> ==自然场景：~9 tokens/image；OCR/文档：144-576 tokens/image==

**问题：**
- 简单任务保留冗余 tokens → 低效
- 复杂任务丢弃关键细节 → 性能下降

**未来方向：**
- Task-aware compression：根据任务语义调整压缩
- Content-aware compression：根据视觉复杂度调整压缩
- 代表探索：VisionThink (RL-based 自适应决策)

---

## 7.3 Performance Degradation in Practical Tasks

> Although many methods demonstrate competitive results on general Visual QA tasks, maintaining accuracy even at 1/3 or 1/4 compression, this **does not generalize** to real-world applications.
>
> ==通用 VQA 上表现好，但实际应用中泛化差==

**易受影响的任务：**
- OCR
- 文档理解
- 结构化视觉布局的密集推理

> These scenarios demand **precise localization, text recognition, and structural alignment**, where the loss of subtle spatial or semantic cues becomes detrimental.
>
> ==这些任务需要精确定位、文本识别、结构对齐，微妙线索丢失会导致严重性能下降==

**核心问题：**
> Current compression schemes prioritize **average efficiency** rather than **task-specific fidelity**.
>
> ==现有方案优先考虑平均效率而非任务特定保真度==

---

## 7.4 Limitations of Existing Evaluation

### 不统一的评估标准

> Different works select varying subsets of benchmarks, making **fair comparison difficult**.
>
> ==各论文选不同 benchmark，难以公平比较==

### 缺乏标准化效率指标

> Efficiency metrics (FLOPs, latency, memory) are reported **inconsistently** across works, complicating direct comparison.
>
> ==效率指标报告不一致==

### 忽略真实世界部署场景

> Most evaluations focus on academic benchmarks rather than **practical deployment scenarios** with diverse hardware constraints.
>
> ==大多数评估关注学术 benchmark，忽略实际部署场景==

---

## 总结：四大挑战

| 挑战 | 描述 | 未来方向 |
|------|------|---------|
| **理论基础缺失** | 大多数方法是 heuristic | 建立 token 重要性的因果/充分性理论 |
| **缺乏自适应** | 固定压缩率，不考虑任务/内容 | Task-aware + Content-aware 自适应压缩 |
| **细粒度任务掉点** | OCR、文档理解等任务性能下降 | 任务特定的压缩策略 |
| **评估标准不统一** | 各论文选不同 benchmark | 建立标准化评估协议 |
