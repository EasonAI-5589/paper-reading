# 2. Related Work

## 2.1 Multimodal Large Language Models

MLLMs (LLaVA, BLIP-2, MiniGPT-4 等) 通过整合 vision 和 text 处理实现多模态能力。但视觉数据处理面临挑战：
- 高分辨率 token 的冗余性和低信息密度
- Attention 的二次方复杂度

例如 LLaVA 将高分辨率图像编码为数千个 token，视频模型更多。Gemini、LWM 等通过优化 token 效率和扩展上下文长度来解决。

## 2.2 Visual Token Compression

现有方法分为几类：

- **需要训练的**：LLaMA-VID (Q-Former + context tokens), DeCo (adaptive pooling) — 修改模型结构，增加训练成本
- **Training-free 的**：
  - ToMe: 在 ViT 中加 token merge module，但破坏 LLM 中的 cross-modal interaction
  - FastV: 用 attention scores 选 important tokens
  - SparseVLM: 通过 cross-modal attention 引入 text guidance
  
这些方法的共同问题：放弃 FlashAttention，且只关注 token importance，忽视 token duplication。

DART 的定位：保持硬件加速兼容性 (FlashAttention)，关注 token duplication 这个被忽视的关键因素。

> 💡 Related work 的组织很清晰地把 DART 与前人工作区分开。但我注意到这里没有讨论 DivPrune (CVPR 2025) 和 PyramidDrop (CVPR 2025) 这两个同样考虑了 diversity 的工作。DivPrune 实际上也用了 diversity-based selection，但方式不同（conditional diversity maximization）。这是一个值得注意的遗漏。

> 💡 ToMe 的问题描述值得注意：在 ViT 阶段做 token merge 会 "disrupt early cross-modal interactions in language models"。这和 HiDivDrop 的 Late Injection 思想有共鸣——浅层的 cross-modal interaction 可能不是那么重要。
