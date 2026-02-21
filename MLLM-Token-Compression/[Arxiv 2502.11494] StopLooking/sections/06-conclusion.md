[← 返回 README](../README.md)

# 6 Conclusion & 7 Limitations

## 📌 预览
Conclusion 总结 DART 的贡献与发现；Limitations 讨论方法局限（不适用于黑盒模型）。

---

# 6 Conclusion

---

The pursuit of efficient token reduction in MLLMs has traditionally focused on token "importance", often measured by attention scores, but sometimes performs worse than random pruning. This study introduces DART, which targets token duplication, removing tokens similar to others and achieving better balance between performance and latency across multiple benchmarks and MLLMs (Tab. 1, 2, 3, 4, 5, 7, 9 and Fig. 4). Our exploration yields surprising insights: distinct retained token sets, with under $50 \%$ overlap, deliver similarly strong performance (§5.2). Moreover, token pruning may reduce hallucinations (§5.3). These findings expose limits of importance-based methods and offer insights into vision tokens in MLLMs.

> 💡 **批注**: 总结非常精炼。三个核心 takeaway：
> 1. Duplication > Importance 作为 pruning 指标
> 2. 不存在唯一最优 token 集合（<50% 重叠仍等效）
> 3. Token pruning 可能减少幻觉——开辟了新的研究方向

---

# 7 Limitations

---

Similar to many other methods aimed at improving efficiency, such as network pruning, quantization, distillation, model merging, and speculative decoding, one of the limitations of our work is that it cannot be applied to black-box models like the GPT (e.g. GPT 3.5 and more advanced versions) and Claude series, as we are unable to access their encoded tokens during the inference process. Moreover, due to space limitations in the main text, we had to move some experimental results that we believe are particularly insightful and interesting to the appendix. These include, for example, our investigation of strategies for pivot token selection, a more detailed analysis of the impact of the number of pivot tokens, and validations of our method on larger-scale models, which may slightly affect the overall reading experience.

> 💡 **批注**: 
> - **黑盒限制**：需要访问中间层 token 表示 → 不适用于 API-only 模型。这是所有 token pruning 方法的共同限制。
> - **未讨论的局限**：(1) 理论分析中的 Lipschitz 假设可能较松；(2) pivot 数量的最优选择依赖经验（4-8）；(3) 对于信息密度极高的图像（如文档OCR），duplication 可能低 → DART 优势可能缩小（从 OCRBench 数据看确实如此）。
