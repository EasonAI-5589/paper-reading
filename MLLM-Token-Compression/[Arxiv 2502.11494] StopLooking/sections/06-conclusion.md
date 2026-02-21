[← 返回 README](../README.md)

# 6 Conclusion & 7 Limitations

## 📌 预览
总结 DART 的核心贡献和局限性。

---

## 6 Conclusion

The pursuit of efficient token reduction in MLLMs has traditionally focused on token "importance", often measured by attention scores, but sometimes performs worse than random pruning. This study introduces DART, which targets token duplication, removing tokens similar to others and achieving better balance between performance and latency across multiple benchmarks and MLLMs (Tab. 1, 2, 3, 4, 5, 7, 9 and Fig. 4). Our exploration yields surprising insights: distinct retained token sets, with under 50% overlap, deliver similarly strong performance (§5.2). Moreover, token pruning may reduce hallucinations (§5.3). These findings expose limits of importance-based methods and offer insights into vision tokens in MLLMs.

> 💡 **结论批注**: 论文最后强调了两个 surprising insights：(1) <50% overlap 的不同 token 子集都能达到好性能；(2) token pruning 可能减少 hallucination。这两个发现比 DART 方法本身更有长远价值——它们揭示了 MLLM 中 vision tokens 的本质冗余性。

---

## 7 Limitations

Similar to many other methods aimed at improving efficiency, such as network pruning, quantization, distillation, model merging, and speculative decoding, one of the limitations of our work is that it cannot be applied to black-box models like the GPT (e.g. GPT 3.5 and more advanced versions) and Claude series, as we are unable to access their encoded tokens during the inference process. Moreover, due to space limitations in the main text, we had to move some experimental results that we believe are particularly insightful and interesting to the appendix. These include, for example, our investigation of strategies for pivot token selection, a more detailed analysis of the impact of the number of pivot tokens, and validations of our method on larger-scale models, which may slightly affect the overall reading experience.

> 💡 **局限性**: (1) 无法用于黑盒模型（GPT、Claude）——这是所有 token-level 方法的共同限制；(2) 空间限制导致部分结果在附录——这不算真正的 limitation。**真正未提及的限制**: DART 需要在 LLM 推理的某一层介入，这对于 quantized 模型或 compiled graphs (torch.compile) 可能需要额外工程适配。

---
