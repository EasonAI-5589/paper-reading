[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
总结 SparseVLM 的核心贡献和关键数字。

---

This paper introduced a text-aware training-free token optimization approach called SparseVLM which significantly decreased the test-time computations of various VLMs. Unlike prior methods, SparseVLM optimized VLMs without introducing extra parameters and fine-tuning costs. We achieved a more compact visual representation by employing the rank of attention matrices to determine pruning ratios and by recycling the pruned tokens via the reconstruction mechanism to reduce the information loss. Experiments demonstrated that e.g. the LLaVA when equipped with SparseVLM achieved 37.0% reduction in latency with a compression ratio of 77.8% while maintaining 97% of the original accuracy. Moreover, our method exceeded FastV accuracy by 14.7% in video understanding tasks. Our SparseVLM can provide practical benefits for deploying off-the-shelf VLMs on edge devices and in the cloud setting.

> 💡 **Conclusion 批读**:
> - **核心卖点**: training-free + text-aware + 即插即用
> - **最佳数字**: 77.8% 压缩率，latency -37%，精度 97%
> - **应用场景**: 边缘设备部署、云端推理加速
> - **vs FastV**: 视频任务上领先 14.7%

---

## Acknowledgments

This work was supported by the National Science and Technology Major Project (No. 2022ZD0117800) and by the National Natural Science Foundation of China under Grant 62472008.

## Impact Statement

Our SparseVLM provides practical advantages for deploying off-the-shelf large vision-language models on edge devices and cloud platforms. While our work does not present any evident societal implications, we believe it is unnecessary to emphasize this aspect in the current context.
