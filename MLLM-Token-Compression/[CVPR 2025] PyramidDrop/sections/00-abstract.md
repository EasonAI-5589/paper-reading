[← 返回 README](../README.md)

# Abstract

## 📌 预览
PyramidDrop 提出了一种视觉 token 渐进式丢弃策略，在 LVLM 的不同层逐步减少图像 token 数量，实现训练和推理加速，且性能几乎无损。

---

In large vision-language models (LVLMs), images serve as inputs that carry a wealth of information. As the idiom "A picture is worth a thousand words" implies, representing a single image in current LVLMs can require hundreds or even thousands of tokens. This results in significant computational costs, which grow quadratically as input image resolution increases, thereby severely impacting the efficiency. Previous approaches have attempted to reduce the number of image tokens either before or within the early layers of LVLMs. However, these strategies inevitably result in the loss of crucial image information. To address this challenge, we conduct an empirical study revealing that all visual tokens are necessary for LVLMs in the shallow layers, and token redundancy progressively increases in the deeper layers. To this end, we propose PyramidDrop, a visual redundancy reduction strategy for LVLMs to boost their efficiency in both inference and training with neglectable performance loss. Specifically, we partition the LVLM into several stages and drop part of the image tokens at the end of each stage with a pre-defined ratio. The dropping is based on a lightweight similarity calculation with a negligible time overhead. Extensive experiments demonstrate that PyramidDrop can achieve over 40% training time reduction and 55% inference FLOPs acceleration on leading LVLMs like LLaVA-NeXT, maintaining comparable multimodal performance. Besides, PyramidDrop can also serve as a plug-and-play strategy to accelerate inference in a free way, with better performance and lower inference cost than counterparts. This project is available at https://github.com/Cooperx521/PyramidDrop to serve as a pivotal resource for advancing the community.

> 💡 **Abstract 批读**:
> - **问题**：LVLM 中图像表示需要大量 token（数百~数千），计算成本随分辨率二次增长
> - **现有方法缺陷**：在 LLM 之前或浅层压缩 token → 不可避免地丢失关键信息
> - **核心发现**：浅层需要所有视觉 token，深层冗余逐渐增加
> - **方案**：PyramidDrop — 将 LVLM 分成多个 stage，每个 stage 末尾按预定比例丢弃部分图像 token
> - **关键数字**：训练时间减少 40%+，推理 FLOPs 加速 55%+，性能几乎无损
> - **额外优势**：可作为 plug-and-play 推理加速策略，免训练直接使用

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 训练时间减少 | >40% |
| 推理 FLOPs 加速 | >55% |
| 基准模型 | LLaVA-NeXT |
| 性能影响 | 可忽略 |
