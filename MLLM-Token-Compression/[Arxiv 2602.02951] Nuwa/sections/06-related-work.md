[← 返回 README](../README.md)

# Appendix A: Related Work

## 📌 预览
Related Work 放在 Appendix 中，涵盖两个方面：(A.1) 高效 VLM 的架构压缩与硬件优化；(A.2) Token pruning 的四个维度分类——vision encoder 侧、LLM 内部、KV cache、多阶段。

---

## A.1 Efficient Large Vision-Language Models

LLMs and VLMs face significant computational efficiency challenges, particularly with extended sequences. LLMs grapple with the growing key-value (KV) cache during autoregressive inference, leading to the development of token reduction strategies like StreamingLLM (Xiao et al., 2023) and H2O (Zhang et al., 2023). However, VLMs confront amplified complexity due to the quadratic growth of visual tokens with image resolution or video frames, making their computational costs prohibitive and necessitating modality-specific optimizations. Two main architectural approaches address these computational constraints. One involves architectural compression, where modules like Q-Former (InstructBLIP (Dai et al., 2023)), perceiver resampler (OpenFlamingo (Awadalla et al., 2023)), and Locality-enhanced Abstractor (Honeybee (Cha et al., 2023)) distill high-dimensional visual inputs into compact representations, reducing the sequence length processed by expensive attention mechanisms. The other pathway utilizes hardware-aware optimization strategies, such as FlashAttention (Dao et al., 2022; Dao, 2023), which optimize memory access patterns for accelerated self-attention computation without altering token quantities, achieving performance gains through algorithmic refinements and efficient resource utilization.

> 💡 **批注**: 两条路线——架构压缩（Q-Former 等 learned downsampling）vs 硬件优化（FlashAttention）。Nüwa 属于第三条路线：token pruning，介于两者之间，不需训练但也不修改底层计算。

---

## A.2 Token Pruning in Large Vision-Language Models

A complementary approach to VLM efficiency focuses on reducing computational overhead through token sequence optimization. The quadratic computational complexity of Transformer attention mechanisms becomes particularly problematic when processing the extensive visual token sequences typical in VLMs. Consequently, vision token pruning has emerged as a critical research direction, which can be systematically categorized along multiple dimensions. Token reduction approaches can be classified based on their training requirements into training-free and training-based methods. Regarding implementation stages, these techniques operate across four primary phases: (1) visual encoder preprocessing, (2) LLM internal processing, (3) KV cache optimization, and (4) hybrid multi-stage approaches. Each pruning strategy involves two fundamental decisions: identifying which tokens to retain and aggregating useful features from discarded tokens.

Token Pruning At Vision Encoder ToME (Bolya et al., 2023) establishes the foundation for training-free token merging at the visual encoder stage, demonstrating effective feature-based token consolidation that influences subsequent works, including VisionZip (Yang et al., 2024), DivPrune (Alvar et al., 2025), LLaVA-PruMerge (Shang et al., 2024), and so on (Tong et al., 2025; Liu et al., 2025a). These methods leverage visual feature similarity to merge redundant tokens before they enter the language model, thereby reducing the computational burden on downstream processing stages.

Token Pruning Within LLM FastV (Chen et al., 2024) pioneers attention score-based token pruning within the LLM processing pipeline, establishing a training-free paradigm that guides later developments such as SparseVLM (Zhang et al., 2025b), PyramidDrop (Xing et al., 2024), FastVLM (Vasu et al., 2024), and so on (Liu et al., 2025a; Arif et al., 2024). These approaches dynamically identify and remove less informative tokens based on attention patterns during inference, maintaining model performance while significantly reducing computational requirements.

Multi-Stage Optimization Strategies Comprehensive efficiency improvements have been achieved through multi-stage approaches that simultaneously optimize visual encoding, LLM prefill, and KV cache management during decoding. Representative methods include MustDrop (Liu et al., 2024b), LightVLM (Hu et al., 2025), and GlobalCom2 (Liu et al., 2025b), which coordinate token reduction across multiple pipeline stages to maximize computational savings while preserving model capabilities.

Training-Based Methods While training-based approaches may exhibit reduced generalizability compared to training-free methods, they demonstrate superior performance preservation through pruning-aware optimization. Methods such as ${ \bf { M } } ^ { 3 }$ (Cai et al., 2024), ATP-LLaVA (Ye et al., 2024), Dynamic-LLaVA (Huang et al., 2024), TokenPacker (Li et al., 2024), and TwigVLM (Shao et al.,

2025b) achieve competitive or superior performance compared to their full-token baselines through specialized training procedures that adapt the model to operate effectively with reduced token sequences.

> 💡 **批注**: Related Work 的分类与 Introduction 一致但更细。值得注意的分类维度：
> - **按位置**: Encoder 侧 → LLM 侧 → KV cache → 多阶段
> - **按训练**: Training-free (Nüwa, FastV, VisionZip) vs Training-based (M³, ATP-LLaVA)
> - Nüwa 属于 **training-free + multi-stage (encoder + LLM)** 的交叉类别
> - 所有 training-free 方法都面临 VG 退化问题，这是 Nüwa 的独特贡献
