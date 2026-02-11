[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两条线：(1) Multimodal CoT — 在推理链中嵌入视觉信息；(2) Latent Reasoning in LLMs — 用 hidden state / 特殊 token 做隐式推理。Mirage 结合两者。

---

## 2.1 Multimodal Chain-of-Thought

Chain-of-Thought (CoT) prompting was first shown to elicit step-by-step reasoning in LLMs by supplying a few worked examples that include intermediate rationales [Feng et al., 2023, Zhang et al., 2024a, Wei et al., 2023]. Recent extensions of CoT to multimodal settings embed visual evidence directly into the reasoning trajectory. ICoT [Zhang et al., 2024b] interleaves attention-selected image crops with text tokens, yielding significant VQA gains, while Visual CoT [Shao et al., 2024a] supplies 438k bounding-box-grounded rationales to train VLMs that emit explicit visual tokens and improve spatial grounding. Recent works [Hu et al., 2024, Zhou et al., 2024, Yang et al., 2025, Gao et al., 2024, Wu et al., 2025, Chern et al., 2025, Fang et al., 2025, Cheng et al., 2025a, Su et al., 2025] further leverage external tools to supply visual cues that enrich multimodal CoT reasoning.

> 💡 **批注**: Multimodal CoT 的发展脉络：
> - **ICoT**: 从输入图片中裁剪关键区域嵌入推理链（外部视觉信息）
> - **Visual CoT**: 输出 bounding box 作为视觉 token（显式空间标记）
> - **Tool-augmented**: 调用外部工具（如画图、标注）生成视觉线索
> - 这些方法都依赖**显式视觉信息**（裁剪/bbox/工具输出），Mirage 则在 latent space 操作

---

Recent works [Chen et al., 2025, Wang et al., 2025] like Chameleon [Team, 2025, Chern et al., 2024] trains a unified token-based model that can emit arbitrary sequences of text and image tokens, but at the cost of large-scale pixel-level supervision and heavier decoding. MVoT [Li et al., 2025a] further trains a unified model to directly produce image and text interleaving trajectories, but absent of reasoning thoughts. In contrast, our Mirage framework differs by emitting compact latent visual tokens rather than real image patches or pixels, avoiding heavy image generation while still allowing fully interleaved visual–text reasoning.

> 💡 **批注**: 
> - **Chameleon/Anole**: 统一 tokenizer，能输出图片 token → 但需要大规模像素级预训练，且实验证明推理能力下降
> - **MVoT**: 生成 interleaved image+text，但"absent of reasoning thoughts"——只有行动和状态图片交替，没有文字推理
> - **Mirage 的定位**: latent token（不是 pixel token），保留推理能力且支持 interleaved

---

## 2.2 Latent Reasoning in LLMs

Much recent work has highlighted the importance of intermediate hidden representations in Large Language Models (LLMs) [Biran et al., 2024, Yang et al., 2024a]. To better guide the latent reasoning process, several approaches introduce specialized tokens into the input sequence. Wang et al. [2023] incorporate discrete `<plan>` tokens to control reasoning stages, while Goyal et al. [2023] propose inserting a `<pause>` token during pretraining to stabilize multi-step reasoning.

> 💡 **批注**: 特殊 token 的先驱工作：
> - `<plan>` token: 控制推理阶段（结构化推理）
> - `<pause>` token: 给模型"停顿思考"的时间（额外计算步骤）
> - Mirage 的 `<latent>` token: 触发视觉 embedding 生成（跨模态推理）

---

Another line of work seeks to internalize reasoning behavior by distilling chain-of-thought rationales into latent representations. Deng et al. [2023] trains models to mimic CoT-style reasoning implicitly through hidden states, and [Deng et al., 2024] further improves inference efficiency by removing explicit intermediate steps altogether. Yu et al. [2024] proposes to distill latent reasoning capabilities into a model by supervising it with data generated for complex reasoning. More recently, Hao et al. [2024] go further by replacing CoT tokens with continuous latent embeddings, enabling unconstrained reasoning in the latent space to explore on complex tasks including math and logical reasoning. While prior work primarily focuses on enhancing efficiency or structural planning within the LLM's latent space, our approach takes a different perspective: we treat latent tokens as a bridge for exploring visual information into the model.

> 💡 **批注**:
> - **Implicit CoT (Deng et al.)**: 把 CoT 蒸馏进 hidden state → 效率导向
> - **Coconut (Hao et al., 2024)**: 最接近 Mirage 的工作——用连续 latent embedding 替代 CoT token，在 latent space 推理。但只针对纯文本 LLM（数学/逻辑）
> - **Mirage 的区别**: 把 latent reasoning 从文本域扩展到多模态域——latent token 不是为了省 token，而是为了引入视觉信息
> - **关键 insight**: "we treat latent tokens as a bridge for exploring visual information" — latent token 是视觉信息的桥梁，不是推理效率的优化

---

## 🔖 Section 总结

### 核心洞察
1. **两条研究线的交汇**: Multimodal CoT（视觉+推理）+ Latent Reasoning（隐式推理）= Mirage（隐式视觉推理）
2. **与 Coconut 的关系**: Coconut 证明了 LLM 可以在 latent space 推理；Mirage 证明 VLM 也可以，且 latent token 可以编码视觉信息
3. **与统一模型的区别**: Chameleon 系列在像素空间操作（重），Mirage 在 embedding 空间操作（轻）
