[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览
总结 ARC-Chapter 的三大贡献和未来方向。

---

In this report, we introduced ARC-Chapter, a scalable and robust framework for structuring long-form videos into semantically coherent chapters and hierarchical summaries. ARC-Chapter leverages a large-scale dataset of millions of long video chapters and employs a semi-automatic annotation pipeline. These innovations advance the state of the art in video chaptering and summary generation. We also proposed the GRACE metric, which addresses the limitations of existing evaluation methods by providing a granularity-robust assessment of chapter boundaries. Experimental results show that ARC-Chapter achieves superior performance across multiple benchmarks, video durations, and languages. These findings demonstrate the framework's effectiveness and generalizability. ARC-Chapter has strong potential to facilitate efficient content navigation, retrieval, and understanding as long-form video content continues to grow rapidly.

> 💡 **结论批读**:
> ARC-Chapter 的三大核心贡献回顾：
> 1. **VidAtlas 数据集**：41 万视频，层级标注，双语——打破数据瓶颈
> 2. **模型框架**：Qwen2.5-VL + modality dropout + GRPO——简洁有效
> 3. **GRACE 指标**：many-to-one 匹配——更适合 chaptering 评估
>
> **论文没有明确讨论的局限性**：
> - 数据来源依赖有 chapter markers 的视频（selection bias）
> - 只用了 7B 模型，更大模型是否有更大提升？
> - 标注管线依赖 LLM 质量，LLM 的 hallucination 是否影响标注？
> - GRPO 只用 video 训练，如果用 video+ASR 训练 RL 会怎样？

---

## 🔖 Section 总结

### 核心洞察
1. Video chaptering 是一个 data-driven 的任务，大规模高质量数据是关键
2. ARC-Chapter 证明了 "数据 > 架构" 的范式：简单架构 + 大数据 = SOTA
3. GRACE 指标的 many-to-one 设计可能对其他 temporal grounding 任务也有启发
