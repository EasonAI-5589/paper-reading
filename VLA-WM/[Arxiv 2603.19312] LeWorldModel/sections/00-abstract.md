[← 返回 README](../README.md)

# 0. Abstract

## 📌 预览

JEPA 虽然概念简洁，但现有方法极易表征坍缩，依赖复杂的多项损失、EMA、预训练编码器等 heuristics。LeWM 是首个仅用两项损失（预测 + SIGReg 正则化）就能从像素端到端稳定训练的 JEPA，15M 参数单 GPU 可训，规划比 foundation-model WM 快 48 倍。

---

## 📄 原文

> Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pre-trained encoders, or auxiliary supervision to avoid representation collapse.

> 💡 **JEPA 的核心矛盾**: 理论上优雅（在紧凑 latent space 中学习世界模型），实践中脆弱（各种 heuristics 防坍缩，一个没调好就崩了）。

> In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative.

> 💡 **核心卖点: 6→1 超参数**。PLDM 需要调 6 个损失系数（搜索复杂度 O(n⁶)），LeWM 只需调 1 个 λ（搜索复杂度 O(log n)，二分法即可）。这不是小改进，是数量级的简化。

> With 15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks.

> 💡 **15M 参数 + 单 GPU + 几小时**: 这是真正的"小而美"。对比 DINO-WM 需要冻结一个 DINOv2 大模型做编码器（~300M 参数），LeWM 把入门门槛降到了实验室级别。

> Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

> 💡 不仅能做控制，还能做物理理解——latent space 编码了位置、角度等物理量，能检测"不合物理规律"的事件（Violation of Expectation）。
