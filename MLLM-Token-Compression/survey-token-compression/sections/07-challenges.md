# 7. Open Challenges and Future Work

---

## 7.1 Lack of Theoretical Understanding

> Most existing approaches remain largely experience-driven and lack rigorous theoretical grounding. They often exhibit poor transferability across datasets, architectures, and modalities.
>
> ==现有方法多是经验驱动，缺乏理论基础，跨数据集/架构/模态迁移差==

**核心问题：**
> A key weakness lies in the absence of a principled theory of token importance. Current practices—such as ranking tokens by attention weights, pairwise similarity, or mutual information—lack causal or generalization-based justification.
>
> ==缺乏 token 重要性的理论基础：attention/similarity/MI 只表示相关性，不能解释因果性和充分性==

**未来方向：**
> By connecting token selection to sufficiency, causality, and robustness, future work can move beyond ad-hoc heuristics toward principled understanding.
>
> ==将 token 选择与充分性、因果性、鲁棒性联系起来==

---

## 7.2 Lack of Task- and Content-Aware Adaptivity

> Most existing token compression strategies operate in a task-agnostic and content-agnostic manner, applying a fixed compression ratio regardless of task type or visual complexity.
>
> ==大多数方法是任务无关和内容无关的，固定压缩率不适应不同场景==

**关键发现 (M³)：**
> For most benchmarks (especially natural scenes like COCO), can be handled well with only **9 tokens per image**. In contrast, dense visual perception tasks such as document understanding or OCR require **144~576 tokens per image**.
>
> ==自然场景只需 ~9 tokens/图像，OCR/文档需要 144~576 tokens/图像！==

**问题：**
- 简单任务保留冗余 tokens → 效率低
- 复杂任务丢弃关键细节 → 性能降

**未来方向：**
> Future research should explore task- and content-aware compression, where the model dynamically determines the degree and manner of token reduction.
>
> ==探索任务和内容感知的自适应压缩==

**代表作探索：**
- PAR, QG-VTC, VCM: 根据文本 query 或视觉内容复杂度调整
- VisionThink: 强化学习决定是否需要高分辨率输入

---

## 7.3 Performance Degradation in Practical Tasks

> Although many token compression methods demonstrate competitive results on general Visual QA tasks, this performance stability does not generalize well to real-world applications.
>
> ==在通用 VQA 上表现好，但实际应用中性能下降严重==

**受影响的任务：**
| 任务类型 | 问题 |
|----------|------|
| OCR | 文本识别准确性下降 |
| Document Understanding | 结构化布局信息丢失 |
| Dense Reasoning | 视觉布局的精细推理受损 |
| Grounding | 定位精度降低 |

> These scenarios demand precise localization, text recognition, and spatial reasoning—capabilities that are highly sensitive to token-level information loss.
>
> ==这些任务需要精确定位、文本识别和空间推理，对 token 级信息丢失非常敏感==

---

## 7.4 Limitations of Existing Evaluation

> ==现有评估体系的局限性==

**问题：**
1. **Benchmark 选择不统一**：各论文选不同 benchmark，难以公平比较
2. **评估维度单一**：主要关注 Accuracy，忽略 Efficiency 的多维度
3. **缺乏细粒度任务评估**：通用 VQA 掩盖了细粒度任务的性能问题
4. **压缩率与位置的交互影响未充分研究**

**未来方向：**
- 建立统一的评估 protocol
- 增加细粒度任务的 benchmark
- 综合考虑 Efficiency 的多个维度（FLOPs, Latency, Memory）
- 评估不同压缩位置的实际影响

---

## 💡 挑战总结

| # | 挑战 | 核心问题 | 未来方向 |
|---|------|----------|----------|
| 1 | 缺乏理论基础 | Token 重要性定义不清 | 因果性 + 充分性分析 |
| 2 | 缺乏自适应 | 固定压缩率不适应场景变化 | 任务/内容感知的动态压缩 |
| 3 | 细粒度任务性能下降 | OCR/文档/定位等任务受损 | 保留关键细节的压缩策略 |
| 4 | 评估标准不统一 | 难以公平比较 | 统一 protocol + 细粒度 benchmark |

---

*[返回论文目录](../README.md)*
