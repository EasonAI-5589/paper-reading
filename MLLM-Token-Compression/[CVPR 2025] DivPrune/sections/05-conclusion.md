[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 DivPrune 的核心贡献和实验发现。

---

In this paper, we proposed a token pruning method based on a max-min diversity problem, called DivPrune. In the proposed method, maximum diversity is achieved among the selected tokens, resulting in reduced redundancy. By ensuring high diversity, the selected tokens provide a more representative subset of the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments were conducted with multiple LMMs on image and video understanding tasks across 16 datasets. The results show that DivPrune achieves state-of-the-art accuracy on the tested datasets. DivPrune generalizes well to different model sizes and architectures, while also improving memory consumption and end-to-end latency for the tested LMMs.

> 💡 **Conclusion 批读**:
> - DivPrune 的核心卖点简洁明了：**多样性 → 低冗余 → 更好的表示 → 高压缩比下仍保持性能**
> - 关键特性：training-free、calibration-free、plug-and-play
> - 在 16 个数据集上验证，涵盖 image + video 理解
> - 泛化性好：不同模型大小（7B/13B）和架构（LLaVA 1.5/1.6/NeXT-Video）都有效
>
> **未提及的局限性**:
> - 只在 LLaVA 系列上测试，未测试其他 LMM（如 Qwen-VL、InternVL）
> - 没有讨论与 KV cache compression 方法的结合
> - 贪心算法的理论保证（近似比）未分析
> - 对于需要细粒度空间理解的任务（如 OCR），多样性选择是否仍然最优？

---

## 🔖 Section 总结

### 核心洞察
1. DivPrune 是一个简洁优雅的方法：一个清晰的数学建模（MMDP）+ 一个高效的贪心求解
2. 论文的实验非常全面（16 数据集、4 个模型、5 个 baseline、多种消融）
3. 最大的亮点是在极端压缩（~90%）下仍保持性能，这对实际部署非常有价值
