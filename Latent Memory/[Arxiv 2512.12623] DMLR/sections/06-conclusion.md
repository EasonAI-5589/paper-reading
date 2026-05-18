[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
一段简洁的总结：DMLR 是个 test-time 多模态 latent reasoning 框架，融合 confidence-guided 优化和 dynamic visual injection，跨任务跨架构稳定提升，且不需要训练。

---

In this work, we analyze how MLLMs utilize visual information and confidence during reasoning. Based on these observations, we introduce DMLR, a test-time multimodal latent reasoning framework that integrates confidence-guided latent optimization with dynamic visual injection. This method enables models to refine their reasoning, retrieve visual evidence only when need without training. Extensive experiments across various tasks show that DMLR consistently boosts both reasoning and perception tasks, offering a stable and training-free alternative to other methods.

> 💡 **Conclusion 拆三层**:
> 1. **观察层 (Section 3)**: MLLM 在推理时怎么用视觉、怎么用置信度——视觉是稀疏的、置信度是多重一致的信号。
> 2. **方法层 (Section 4)**: DMLR = confidence-guided 优化 + DVI；test-time + training-free。
> 3. **结果层 (Section 5)**: reasoning & perception 同涨；稳定；跨架构有效。

---

## 🔖 全文总结

### 论文的"三个一句话"

1. **它解决什么问题？**
   多模态 CoT 推理要么纯文字（视觉接地差）、要么靠工具（开销大）、要么需要训练（不灵活）——缺一个既动态又免训练的方案。

2. **它怎么解决？**
   把推理变成"在 latent 空间做置信度引导的策略梯度优化 + 按需选最相关视觉 patch"，让模型像人一样按需"瞄一眼图"。

3. **它有多好？**
   7 个 benchmark、6 个 backbone（3B-8B）上 95%+ 任务最优；reasoning + perception 同涨；效率不损失。

### 这篇论文留给读者的"延伸思考"

| 问题 | 我的批读看法 |
|---|---|
| 为什么不直接做训练式（监督）latent reasoning？ | 训练式不灵活，且需要标注的"good latent"；DMLR 用 reward signal 摆脱标注依赖 |
| Confidence reward 可不可靠？会被 hack 吗？ | Section 3.2 实证 confidence ⇔ faithful reasoning，所以**不易**被 spurious chain hack（但理论上没保证 100%） |
| DMLR 跟 RLHF / inference-time scaling 关系？ | DMLR 本质是 **inference-time RL** 的一个特例：reward (entropy) 无需训练 reward model；优化的是 latent 而非 policy weights |
| 失败模式可能在哪？ | (1) 模型 confidence 校准很差时（如基座模型未做指令微调）；(2) 视觉信号本身不足时；(3) 任务对 deep multi-step 依赖（4 个 latent token 容量有限） |
| 跟 Soft Thinking / LatentSeek 等家族的本质差异？ | DMLR = (LatentSeek 的 test-time policy gradient) + (multimodal visual injection)，是"两个独立工作的合体"，但合得很自然 |

### 一句话评价

> DMLR 是把 "test-time latent policy gradient (LatentSeek)" 和 "interleaved-modal latent injection (ICoT)" 合二为一的工作。它的优雅在于：**用 confidence reward 把两件事统一到一个目标下**——不是简单叠加两个 trick，而是用同一个 reward 同时驱动 latent 优化和视觉 patch 选择。Section 3 那个三连观察 (Obs 1-3) 是把这个 reward 选择正当化的关键。
