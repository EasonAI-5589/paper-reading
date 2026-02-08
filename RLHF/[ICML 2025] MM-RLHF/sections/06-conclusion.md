# 6. Conclusion and Future Work

> 来源: MM-RLHF (ICML 2025)

---

## 📄 原文

In this work, we introduced MM-RLHF, a high-quality, fine-grained dataset specifically designed to advance the alignment of MLLMs. Unlike prior works that focus on specific tasks, our dataset and alignment approach aim to holistically improve performance across diverse dimensions. Even with preliminary improvements to reward modeling and optimization algorithms, we observed significant and consistent gains across almost all evaluation benchmarks, underscoring the potential of comprehensive alignment strategies.

> 💡 **核心贡献总结**: MM-RLHF 证明了一件事——MLLM alignment 可以全面提升模型能力，不仅限于特定任务。

Looking ahead, we see great opportunities to further unlock the value of our dataset. Its rich annotation granularity, such as per-dimension scores and ranking rationales, remains underutilized in current alignment algorithms. Future work will focus on leveraging this granularity with advanced optimization techniques, integrating high-resolution data to address limitations in specific benchmarks, and scaling the dataset efficiently using semi-automated strategies. We believe these efforts will not only push MLLM alignment to new heights but also set a foundation for broader, more generalizable multimodal learning frameworks.

> 💡 **未来方向**:
> 1. **更好地利用 annotation granularity**: 当前算法只用了 ranking 信息，per-dimension scores 和 rationales 还没充分利用
> 2. **高分辨率数据**: 弥补当前在 high-resolution benchmarks 上的不足
> 3. **半自动化标注 scaling**: Human-RM 协作框架降低成本

---

## 💡 Section 总结

### 本文的历史地位
- **首个系统性 MLLM alignment 工作**: 覆盖 image/video/safety, 10 维度, 27 benchmarks
- **首个大规模人工标注的 MLLM preference 数据集**: 120K pairs, 远超之前的 <10K
- **首个在 MLLM 中探索 critique-based RM**: 从 scalar reward 到可解释 reward
- **首个在 MLLM 中验证 dynamic β adjustment**: 并指出 LLM 的方法不能直接迁移

### 对 Apple Assignment 的整体价值
| 方面 | 可引用的要点 |
|------|-------------|
| Human Annotation Protocol | 50+ annotators, 8 experts, 2 months, 3 dimensions, scoring+ranking+explanation, Web UI |
| Preference Data Collection | 10M→30K→120K pipeline, CLIP clustering, 4:5:1 resampling, tie handling strategies |
| Critique-Based RM | Dual-head (critique + scoring), GPT-4o annotation enhancement, teacher-forcing |
| Dynamic Reward Scaling | β(δ) = β_ori(1+w(1-e^(-kδ))), bounded, uses external RM |
| Inter-rater Reliability | **未明确报告** (no Krippendorff's α / Cohen's κ / Fleiss' κ) — 但有 expert review + re-annotation |
| Video Understanding | 3 benchmarks (VideoChatGPT, Video-MME, VideoDC), moderate improvements, data source: SharedGPT-4 video |

### 论文局限性（可在 assignment 中讨论）
1. **No inter-rater reliability metrics**: 50+ annotators 的 agreement 没有量化报告
2. **Video data 占比小** (~14%): 导致 video 提升有限
3. **高分辨率数据不足**: 特定场景没有覆盖
4. **Annotation granularity 未充分利用**: 只用了 ranking，per-dimension scores 浪费了
5. **成本问题**: 2 months × 50+ annotators，scalability 依赖未来的 human-RM 协作
