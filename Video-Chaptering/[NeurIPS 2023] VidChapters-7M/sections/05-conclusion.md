[← 返回 README](../README.md)

# 5 Conclusion, Limitations, and Societal Impacts

## 📌 预览
总结全文贡献，讨论数据集偏差和潜在社会影响。

---

In this work, we presented VidChapters-7M, a large-scale dataset of user-chaptered videos. Furthermore, we evaluated a variety of baselines on the tasks of video chapter generation with and without ground-truth boundaries and video chapter grounding. Finally, we investigated the potential of VidChapters-7M for pretraining video-language models and demonstrated improved performance on the dense video captioning tasks. VidChapters-7M thus provides a new resource to the research community that can be used both as a benchmark for the video chapter generation tasks and as a powerful means for pretraining generic video-language models.

> 💡 **双重价值**: VidChapters-7M 既是 benchmark（评测三个任务），又是预训练资源（提升 dense captioning）。这种双重定位增加了数据集的影响力。

**Limitations.** As it is derived from YT-Temporal-180M [117], VidChapters-7M inherits the biases in the distribution of video categories reflected in this dataset.

> 💡 **局限性**: 只提到了继承 YT-Temporal-180M 的类别偏差，但实际还有更多局限：
> - 93% 英语 → 多语言场景下效果存疑
> - YouTube 平台依赖 → 视频可能下线导致数据集萎缩
> - 用户自发标注质量参差不齐（3% 无关内容，14% 仅结构性标注）
> - Moment-DETR 没有利用语音信息 → grounding 任务还有很大提升空间

**Societal Impacts.** The development of video chapter generation models might facilitate potentially harmful downstream applications, e.g., video surveillance. Moreover, models trained on VidChapters7M might reflect biases present in videos from YouTube. It is important to keep this in mind when deploying, analysing and building upon these models.

> 💡 **社会影响**: 视频理解技术的两面性——帮助用户导航的同时，也可能被用于监控。模型会继承 YouTube 视频中的偏差。

---

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2023-A0131011670 made by GENCI. The work was funded by Antoine Yang's Google PhD fellowship, the French government under management of Agence Nationale de la Recherche as part of the "Investissements d'avenir" program, reference ANR-19-P3IA-0001 (PRAIRIE 3IA Institute), the Louis Vuitton ENS Chair on Artificial Intelligence, the European Regional Development Fund under project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15 003/0000468). We thank Jack Hessel and Rémi Lacroix for helping with collecting the dataset, and Antoine Miech for interesting discussions.

---

## 🔖 Section 总结

### 核心洞察
1. VidChapters-7M 是一个集 benchmark + 预训练数据于一体的资源
2. 主要局限在于数据偏差（语言、类别、平台依赖）
3. 未来方向：多模态 grounding、多语言扩展、更大规模的 scaling
