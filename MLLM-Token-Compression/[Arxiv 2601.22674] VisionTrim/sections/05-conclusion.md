[← 返回 README](../README.md)

# 5 Conclusion and Limitations

## 📌 预览
总结 VisionTrim 的贡献，坦诚指出 "not entirely without loss" 的局限。

---

In this paper, we proposed VisionTrim, a unified training-free framework for MLLM acceleration through comprehensive vision token compression. We presented two effective plug-and-play modules that accelerated both vision encoding and LLM decoding stages. By integrating the DVTS module, which selects tokens based on global semantics and local spatial continuity, with the TGVC module, which performs text-guided visual token complement, our approach consistently surpassed previous methods across various reduction ratios in both image and video understanding tasks.

> 💡 **总结批注**: 一句话概括 VisionTrim = "DVTS 选 + TGVC 补"，在 image 和 video 上全面 SOTA。

---

**Limitations.** Although VisionTrim achieves 98.8% of the original performance with an 88.9% reduction ratio in token count without additional training costs, it is not entirely without loss. We are committed to advancing our research to further explore the redundancy of visual tokens and developing lossless methods to enhance the efficiency of visual understanding with MLLMs.

> 💡 **Limitations 批注**:
> - 坦诚承认不是 lossless（98.8% ≠ 100%）
> - 未来方向：lossless visual token compression
> - **我的补充局限**:
>   1. TGVC 依赖 CLIP text encoder，对非 CLIP 架构（如 SigLIP）需要适配
>   2. Text-guided 在多轮对话中可能有问题（每轮问题不同，但 ViT 端只做一次）
>   3. 没有和需要训练的方法做公平比较（如 TokenPacker, DeCo）
>   4. LTAM 的 dual-kernel 增加了额外计算开销，文中没有量化

---

## 🔖 Section 总结

### 核心洞察
1. VisionTrim 是一个 training-free 的有损压缩方法，性能保持非常好但不是 lossless
2. 多轮对话场景下的 text-guided 策略需要进一步探索
