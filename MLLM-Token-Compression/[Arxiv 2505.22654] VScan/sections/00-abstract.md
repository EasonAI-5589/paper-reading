[← 返回 README](../README.md)

# Abstract

## 📌 预览
VScan 提出两阶段 training-free 视觉 token 压缩：visual encoding 阶段用 global+local scan 互补选 token 并 merge，LLM decoding 阶段在中间层做 text-aware pruning。

---

Recent Large Vision-Language Models (LVLMs) have advanced multi-modal understanding by incorporating finer-grained visual perception and encoding. However, such methods incur significant computational costs due to longer visual token sequences, posing challenges for realtime deployment. To mitigate this, prior studies have explored pruning unimportant visual tokens either at the output layer of the visual encoder or at the early layers of the language model. In this work, we revisit these design choices and reassess their effectiveness through comprehensive empirical studies of how visual tokens are processed throughout the visual encoding and language decoding stages. Guided by these insights, we propose VScan, a twostage visual token reduction framework that addresses token redundancy by: (1) integrating complementary global and local scans with token merging during visual encoding, and (2) introducing pruning at intermediate layers of the language model. Extensive experimental results across four LVLMs validate the effectiveness of VScan in accelerating inference and demonstrate its superior performance over current state-of-the-arts on sixteen benchmarks. Notably, when applied to LLaVA-NeXT-7B, VScan achieves a $2.91\times$ speedup in prefilling and a $10\times$ reduction in FLOPs, while retaining $95.4\%$ of the original performance. Code is available at https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan.

> 💡 **Abstract 批读**:
> - **问题**：高分辨率/多图/视频输入导致 visual token 序列太长，self-attention 二次复杂度
> - **现有方法两类**：text-agnostic（visual encoder 输出层剪枝）和 text-aware（LLM 早期层剪枝）
> - **VScan 创新**：两阶段都做，而且都选了更优的位置——
>   - Stage 1 不只看 output layer，还看 shallow layer（捕获局部细节）
>   - Stage 2 不在 early layer 剪，而在 middle layer（避免位置偏差）
> - **关键数字**：LLaVA-NeXT-7B 上 2.91× prefill 加速，10× FLOPs 降低，保留 95.4% 性能
> - 相比之前方法（FastV 只做 Stage 2，VisionZip 只做 Stage 1），VScan 两阶段互补是核心卖点
