[← 返回 README](../README.md)

# 5. Related Work

## 📌 预览
Related Work 简要回顾 VLM 的发展和 visual token 长度增长的趋势。

---

**Vision-Language Models.** Building on the success of large language models (LLMs) [1, 2, 49], recent vision-language models (VLMs) [8, 30, 32, 48] advance multimodal generation by processing extensive visual token sequences. Higher resolutions require exponentially more tokens; for example, LLaVA-NeXT processes 672×672 images into 2304 tokens [32]. Handling videos or multiple images increases token requirements, as seen in Video-LLaVA [31] and Video-ChatGPT [39]. Hence, it's essential to discuss more efficient ways to extract information from visual tokens, rather than merely increasing their length. The additional related work is shown in Appendix C.

> 💡 **批注**: Related Work 非常简短（主论文只有一段），详细版在 Appendix C。核心观点：VLM 的趋势是越来越多的 visual tokens（更高分辨率、视频），但这条路的效率问题日益严重。VisionZip 提出的方向是"提取更好的特征"而非"堆更多 token"。

---

## 🔖 Section 总结

### 核心洞察
1. VLM 的趋势是增加 visual tokens（更高分辨率、更多帧），但带来严重的效率问题
2. 论文暗示：与其无脑增加 token 长度，不如解决 token 冗余——这是 VisionZip 的哲学
