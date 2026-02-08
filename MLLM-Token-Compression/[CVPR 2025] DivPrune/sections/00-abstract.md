[← 返回 README](../README.md)

# Abstract

## 📌 预览
DivPrune 将视觉 token 剪枝建模为 Max-Min Diversity Problem (MMDP)，通过最大化所选 token 的多样性来减少冗余，在不需要微调的情况下实现高压缩比下的优异性能。

---

Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected tokens is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models. The code is available here⋄.

> 💡 **Abstract 批读**:
> - **问题**: LMM 中视觉 token 数量巨大（常数千个），导致 LLM 推理延迟高
> - **现有方法缺陷**: (1) 需要校准/微调，成本高；(2) 基于 attention score 的重要性度量不够优，导致保留 token 冗余
> - **核心创新**: 将 token pruning 建模为 **Max-Min Diversity Problem (MMDP)**——选择子集使其多样性最大化
> - **关键优势**: training-free、calibration-free、plug-and-play；高压缩比（如 90%）下仍保持有效性能
> - **实验规模**: 16 个数据集，涵盖图像和视频理解任务

---

## 🔖 Section 总结

### 核心洞察
1. DivPrune 的核心思路是「选多样的 token」而非「选重要的 token」，这是与 FastV、PruMerge 等方法的根本区别
2. 方法不需要任何训练或校准数据，是真正的 plug-and-play 方案
3. 在 16 个数据集上达到 SOTA，同时减少延迟和显存
