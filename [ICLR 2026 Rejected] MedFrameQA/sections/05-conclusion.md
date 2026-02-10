[← 返回 README](../README.md)

# 5. Conclusion and Limitations

## 📌 预览
总结贡献，承认局限性（方法改进探索不足）。

---

This paper introduces MEDFRAMEQA, a multi-image medical visual question answering benchmark, comprising 2851 multi-image multi-choice questions, sourced from 3420 medical videos of 114 keywords and covering over 43 organs. We also propose an automated pipeline to generate high-quality multi-image VQA data from YouTube while ensuring semantic progression and contextual consistency across frames. Unlike existing datasets that rely on single-image inputs or lack detailed reasoning about the answer, MEDFRAMEQA has both multi-image question answering pairs and a detailed reasoning process, containing 2-5 images input and 3.24 images input per question.

> 💡 **贡献总结**: (1) 首个多图医学 VQA benchmark；(2) 从视频自动构建 VQA 的 pipeline；(3) 同时提供多图 QA 和推理过程。

We comprehensively benchmark ten state-of-the-art models, presenting accuracies predominantly below 50%. While MEDFRAMEQA reveals clear evidence of current MLLMs' inability in handling multi-image questions of clinical reasoning, effective strategies to enhance their multi-image reasoning capabilities remain underexplored. Future work will focus on developing and evaluating methods to improve such capabilities.

> 💡 **关键局限**: 论文只构建了 benchmark 和做了评测，**没有提出任何改进方法**。这可能是被拒的重要原因之一——ICLR 通常期望不仅发现问题，还要至少尝试解决问题。
>
> 其他潜在局限（论文未提但应该关注）：
> 1. **GPT-4o 偏差**: 数据构建全程依赖 GPT-4o，可能引入系统性偏差
> 2. **难度过滤偏差**: 用 GPT 系列模型过滤导致 benchmark 对这些模型特别不利
> 3. **caption→VQA 的 gap**: 问题从 caption 生成但模型看图片回答，可能存在不一致
> 4. **YouTube 视频质量参差**: 教育视频质量不统一，可能有错误内容

We believe MEDFRAMEQA will serve as a valuable resource for advancing research in multimodal medical AI and fostering the development of more capable diagnostic reasoning systems.

---

## 🔖 Section 总结

### 核心洞察
1. **论文定位**: 纯 benchmark 论文，贡献在于揭示问题而非解决问题
2. **被拒原因推测**: (1) 缺少改进方法；(2) 过度依赖 GPT-4o 的 pipeline 可能被质疑数据质量；(3) 实验设计中的难度过滤偏差
3. **对 Agent Memory 课题的启示**: 多步推理中的信息遗漏和错误传播是共性挑战，值得用 Agent Memory 的思路来应对
