[← 返回 README](../README.md)

# 5. Conclusions

## 📌 预览
总结贡献、讨论局限性和未来方向。

---

We presented Chapter-Llama, an approach that leverages LLMs for hour-long video chaptering by mapping video to text using speech transcripts and efficiently captioning video frames sampled with a speech-based frame selector. Our results on VidChapters-7M consequently improved the state of the art by a large margin. We experimentally demonstrated the benefits of our components through an extensive ablation study. One limitation of our approach is that it relies on the accuracy of the ASR and the visual captioner. Future work can explore hierarchical chaptering with several granularities and consider the audio modality beyond speech. We also note that the LLM, the visual captioner, and speech transcription models are trained on large Web datasets, which can contain biases that can lead to inaccurate chaptering, especially for videos depicting underrepresented topics.

> 💡 **局限性**:
> 1. 依赖 ASR 和 captioner 的准确性（这两个模块的错误会级联传播）
> 2. 没有利用 speech 之外的音频信息（如音乐、音效、环境声）
> 3. Web 数据训练的模型可能存在偏见
>
> **未来方向**:
> - 层次化 chaptering（不同粒度的章节）
> - 利用更丰富的音频模态（不只是语音）

Acknowledgements. This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011014696 made by GENCI. This work was funded in part by the ANR project CorVis ANR-21-CE23-0003-01, a research gift from Google, the French government under management of Agence Nationale de la Recherche as part of the "France 2030" program, reference ANR-23-IACL-0008 (PR[AI]RIE-PSAI projet), and the ANR project VideoPredict ANR-21-FAI1-0002-01. Cordelia Schmid would like to acknowledge the support by the Korber European Science Prize. The authors would also like to thank Guillaume Astruc, Nikos Athanasiou, Hyolim Kang, and Nicolas Dufour for their feedback.

---

## 🔖 Section 总结

### 核心洞察
1. "视频→文本→LLM" 范式在 video chaptering 上效果极佳
2. 主要局限在于 ASR/captioner 的准确性上限
3. 未来可以探索多粒度 chaptering 和更丰富的音频特征
