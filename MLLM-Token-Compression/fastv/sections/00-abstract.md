# Abstract

## 📄 原文

> In this study, we identify the **inefficient attention phenomena** in Large Vision-Language Models (LVLMs), notably within prominent models like LLaVA-1.5, QwenVL-Chat, and Video-LLaVA.
>
> ==核心发现：LVLM 中存在"低效 attention"现象==

> We find that the **attention computation over visual tokens is extremely inefficient in the deep layers** of popular LVLMs, suggesting a need for a sparser approach compared to textual data handling.
>
> ==具体发现：深层 visual tokens attention 极度稀疏，需要更稀疏的处理方式==

> To this end, we introduce **FastV**, a versatile plug-and-play method designed to optimize computational efficiency by **learning adaptive attention patterns in early layers and pruning visual tokens in subsequent ones**.
>
> ==方法：FastV — 在浅层学习 attention 模式，在后续层剪枝 visual tokens==

> Our evaluations demonstrate FastV's ability to dramatically reduce computational costs (e.g., a **45% reduction in FLOPs** for LLaVA-1.5-13B) without sacrificing performance in a wide range of image and video understanding tasks.
>
> ==效果：LLaVA-1.5-13B 减少 45% FLOPs，性能无损==

> The computational efficiency and performance trade-off of FastV are highly customizable and **Pareto-efficient**. It can compress the FLOPs of a 13B-parameter model to achieve a lower cost than that of a 7B-parameter model while still maintaining superior performance.
>
> ==特点：Pareto-efficient，13B 压缩后成本低于 7B 但性能更好==

---

## 💡 Key Takeaways

1. **发现**：Visual tokens 在深层 attention 极度低效
2. **方法**：Plug-and-play，浅层学习 + 深层剪枝
3. **效果**：45% FLOPs 减少，性能无损
4. **价值**：边缘设备部署、商业模型加速

---

*[返回论文目录](../README.md)*
