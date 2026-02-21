[← 返回 README](../README.md)

# Abstract

## 📌 预览
FSR 提出仿人类视觉感知的三阶段 token 剪枝框架：Focus（锁定关键局部证据）→ Scan（扫描互补全局上下文）→ Refine（聚合细节到锚点），在多个 VLM 和 benchmark 上实现 SOTA 的精度-效率权衡。

---

Vision-language models (VLMs) often generate massive visual tokens that greatly increase inference latency and memory footprint; while training-free token pruning offers a practical remedy, existing methods still struggle to balance local evidence and global context under aggressive compression. We propose Focus-Scan-Refine (FSR), a human-inspired, plug-and-play pruning framework that mimics how humans answer visual questions: focus on key evidence, then scan globally if needed, and refine the scanned context by aggregating relevant details. FSR first focuses on key evidence by combining visual importance with instruction relevance, avoiding the bias toward visually salient but queryirrelevant regions. It then scans for complementary context conditioned on the focused set, selecting tokens that are most different from the focused evidence. Finally, FSR refines the scanned context by aggregating nearby informative tokens into the scan anchors via similarity-based assignment and score-weighted merging, without increasing the token budget. Extensive experiments across multiple VLM backbones and vision-language benchmarks show that FSR consistently improves the accuracyefficiency trade-off over existing state-of-the-art pruning methods. The source codes can be found at https://github.com/ILOT-code/FSR

> 💡 **摘要批注**:
> - **核心问题**: 现有 training-free token pruning 在高压缩率下难以平衡局部证据 vs 全局上下文
> - **方法灵感**: 仿人类视觉问答的认知过程 — focus → scan → refine
> - **三阶段设计**:
>   1. Focus: 双通道评分（visual saliency + instruction relevance）选关键局部 token
>   2. Scan: 条件采样，选与 Focus 集最不同的互补 token
>   3. Refine: 将丢弃 token 信息聚合到 Scan 锚点，不增加 budget
> - **关键卖点**: plug-and-play、training-free、动态分配局部/全局 budget

Keywords: Vision–Language Models, Human-Inspired Visual Processing, Visual Token Pruning, Efficient Multimodal Inference
