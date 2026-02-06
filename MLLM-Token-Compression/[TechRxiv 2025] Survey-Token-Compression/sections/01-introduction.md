# 1. Introduction

## 📄 原文

> Multimodal Large Language Models (MLLMs) rapidly advanced the frontier of vision-language joint perception, alignment, reasoning, and generation. By integrating the remarkable language understanding capabilities of Large Language Models (LLMs) with comprehensive visual perception abilities from vision encoders, contemporary systems such as LLaVA, Qwen-VL and GPT-4o exhibit strong performance on diverse tasks spanning open-ended visual question answering, document understanding, and multi-step visual reasoning, among others.
>
> ==MLLMs = LLM语言能力 + Vision Encoder视觉感知能力，代表作：LLaVA、Qwen-VL、GPT-4o==

> However, these advanced cross-modal capabilities incur substantial computational costs. High-resolution images and long videos can generate hundreds to thousands of visual tokens, while multi-turn dialogue and chain-of-thought reasoning further extend the historical context. As sequence lengths increase, the quadratic complexity of attention in Transformer-based MLLMs results in prohibitive memory consumption and latency, limiting both scalability and deployment.
>
> ==问题核心：高分辨率图像/长视频 → 成百上千 visual tokens → O(n²) attention → 内存爆炸 + 延迟高==

> This tension between multimodal effectiveness and computational efficiency has made compressing multimodal token sequences an urgent research focus.
>
> ==矛盾：多模态效果 vs 计算效率 → Token Compression 成为热点==

---

## Token Compression 定义

> To build more efficient MLLMs, token compressing multimodal token sequences refers to methods that reduce the number of tokens processed by MLLMs while preserving critical cross-modal semantics.
>
> ==Token Compression = 减少 tokens 数量 + 保留跨模态语义==

> Conceptually, compression targets redundancy in:
> - **spatial structure** (e.g., repetitive background regions)
> - **temporal continuity** (e.g., frame-to-frame similarities)
> - **modality alignment** (e.g., text-conditioned visual irrelevance)
>
> yielding shorter sequences with minimal essential information degradation.
>
> ==三种冗余来源：空间冗余（背景重复）、时间冗余（帧间相似）、模态对齐冗余（与文本无关的视觉信息）==

> Historically, token compression originated in unimodal vision through patch dropping, token merging, and dynamic sparsification in Vision Transformers. These approaches have since been extended to multimodal settings, where compression can operate on visual streams, textual streams, or their fusion.
>
> ==历史：从 ViT 的 patch dropping / token merging 发展而来，现扩展到多模态==

---

## 三个核心问题

### Q1: Where and how should token compression be applied?

> Different modules in MLLMs, including the vision encoder, projector, and large language model, introduce distinct architectural characteristics, information bottlenecks, and computational trade-offs. The placement of compression strongly influences the preservation of visual semantics, the quality of cross-modal alignment, and downstream reasoning capability.
>
> ==不同模块（Vision Encoder / Projector / LLM）各有特点，压缩位置影响语义保留、跨模态对齐、推理能力==

### Q2: Which compression mechanism best suits specific deployment scenarios?

> The commonly-adopted design space spans:
> - token merging versus pruning
> - text-guided versus purely visual compression
> - objectives for training versus inference acceleration
> - plug-in modules versus end-to-end retraining
>
> Each paradigm offers distinct benefits and limitations that must be aligned with application-specific constraints.
>
> ==设计空间：Merging vs Pruning / Text-guided vs Purely-visual / Training vs Inference / Plug-in vs Retraining==

### Q3: What are the remaining open challenges and future directions?

> We discuss key challenges including:
> - lack of theoretical foundations
> - adaptation to dynamic compression requirements
> - efficiency-effectiveness trade-offs in fine-grained tasks (e.g., chart understanding and OCR)
> - the need for more rigorous evaluation protocols
>
> ==开放挑战：缺乏理论基础、动态压缩适配、细粒度任务 trade-off、评估协议不完善==

---

## 本文贡献

> **(i) Taxonomy of token compression by MLLM architectural placement (§3).**
> We introduce a systematic taxonomy that categorizes token compression methods by their application location within MLLMs—vision encoder, projector, or large language model—clarifying how architectural placement interacts with compression objectives and how hybrid strategies can synergistically combine approaches across different modules.
>
> ==贡献1：按 MLLM 模块位置分类的 Taxonomy==

> **(ii) Methodological analysis and design roadmap (§4).**
> We dissect critical design dimensions, including text-guided versus vision-only compression, token pruning versus merging, modular plug-ins versus end-to-end retraining, and training-centric versus inference-centric optimization. Based on this methodological breakdown, we provide a selection roadmap to guide researchers in choosing the optimal compression techniques.
>
> ==贡献2：方法分析 + 选择路线图（覆盖多个设计维度）==

---

## 💡 Key Takeaways

1. **问题**：高分辨率/长视频 → visual tokens 爆炸 → O(n²) 复杂度
2. **冗余来源**：空间（背景）、时间（帧间）、模态对齐（与 query 无关）
3. **三个核心问题**：Where / How to select / Open challenges
4. **本文贡献**：Taxonomy (按位置) + Selection Roadmap (按策略)

---

*[返回论文目录](../README.md)*
