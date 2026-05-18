[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要其实就是回答四个问题：现有 MLLM 推理有什么毛病？我们换什么思路？方法长啥样？效果怎么样？读完这段你要带走的关键词是「test-time」「latent think tokens」「confidence-guided」「dynamic visual injection」。

---

Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception–reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual–textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency.

> 💡 **Abstract 拆解**:
>
> | 句子 | 在讲什么 | 关键词 |
> |---|---|---|
> | "Recent advancements ... in the semantic space." | 现状：MLLM 已经能做 CoT 推理。 | semantic-space CoT |
> | "Building upon this ... explicit image generation." | 现状的延伸：Think-with-Image（外部工具/生图）让推理也能动用视觉。 | visual CoT, tool-augmented |
> | "However, these methods remain ... computational overhead." | **痛点**: 3 个——必须显式逐步推理、感知-推理交互不稳定、计算开销大。 | explicit, unstable, expensive |
> | "Inspired by human cognition ... within the mind." | **核心观点**: 人脑不是线性思考，而是「推理 ⇋ 感知」在脑内交错。 | dynamic interleaving |
> | "Motivated by this ... in-depth reasoning." | **方法主轴 A**: DMLR = test-time + 置信度引导的策略梯度，优化 latent think tokens。 | DMLR, latent policy gradient |
> | "Furthermore, a Dynamic Visual Injection Strategy ..." | **方法主轴 B**: DVI 在每个 latent token 处取相关视觉特征，动态更新 best patches。 | DVI |
> | "Experiments ... high inference efficiency." | **结论**: 7 个 benchmark、多种 backbone 都涨；既快又准。 | 7 benchmarks |

> 💡 **关键区分**: "Test-time" 表示**不需要训练**——预训练 MLLM 参数固定，只在推理过程优化注入的 latent embedding。这是 DMLR 与 CoCoNut[14]、Latent Visual Reasoning[16] 等 *训练式* latent reasoning 的根本区别。

§ Project Page: https://mllm-dmlr.github.io/

---

## 🔖 Section 小结

- **方法名**: DMLR (Dynamic Multimodal Latent Reasoning)
- **属性**: test-time，training-free，可叠在任意 MLLM 上
- **两条腿**:
  1. confidence-guided latent policy gradient（refine 内部「想法」）
  2. dynamic visual injection（按需「再看一眼图」）
- **可量化亮点**: 7 个 benchmark 上 reasoning 和 perception **同涨**，效率不降
