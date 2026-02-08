[← 返回 README](../README.md)

# Abstract

## 📌 预览
FastV 的核心发现：LVLM 深层对视觉 token 的注意力极其低效，因此可以在浅层之后剪枝大量视觉 token，实现 45% FLOPs 减少且几乎不损失性能。

---

In this study, we identify the inefficient attention phenomena in Large Vision-Language Models (LVLMs), notably within prominent models like LLaVA-1.5, QwenVL-Chat, and Video-LLaVA. We find that the attention computation over visual tokens is extremely inefficient in the deep layers of popular LVLMs, suggesting a need for a sparser approach compared to textual data handling. To this end, we introduce FastV, a versatile plug-andplay method designed to optimize computational efficiency by learning adaptive attention patterns in early layers and pruning visual tokens in subsequent ones. Our evaluations demonstrate FastV's ability to dramatically reduce computational costs (e.g., a 45% reduction in FLOPs for LLaVA-1.5-13B) without sacrificing performance in a wide range of image and video understanding tasks. The computational efficiency and performance tradeoff of FastV are highly customizable and Pareto-efficient. It can compress the FLOPs of a 13B-parameter model to achieve a lower cost than that of a 7B-parameter model while still maintaining superior performance. We believe FastV has practical value for the deployment of LVLMs in edge devices and commercial models. Code is released at github.com/pkunlpicler/FastV.

> 💡 **Abstract 批读**:
> - **问题**: LVLM 深层对 visual token 的注意力计算效率很低
> - **方法**: FastV — plug-and-play，浅层学自适应注意力模式，深层剪枝 visual token
> - **效果**: LLaVA-1.5-13B 减少 45% FLOPs，性能基本不变
> - **亮点**: 13B + FastV 的 FLOPs 可以低于 7B 模型，同时性能更好
> - **适用范围**: 图像和视频理解任务均有效

---

## 🔖 Section 总结

### 核心洞察
1. LVLM 深层的视觉注意力极其低效 → 可以大幅剪枝
2. FastV 是 plug-and-play 方法，无需重训练
3. 效率-性能 trade-off 是 Pareto 最优且可定制
