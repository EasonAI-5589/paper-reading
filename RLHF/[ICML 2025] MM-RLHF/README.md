# MM-RLHF: The Next Step Forward in Multimodal LLM Alignment

**作者**: Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, et al.  
**机构**: KuaiShou, CASIA, NJU, USTC, PKU, Alibaba, Meta AI  
**会议**: ICML 2025  
**链接**: [项目页面](https://mm-rlhf.github.io/)

## 一句话总结
提出 120k 规模的细粒度人工标注多模态偏好数据集 MM-RLHF，以及 Critique-Based Reward Model 和 Dynamic Reward Scaling (MM-DPO)，在 10 个维度 27 个 benchmark 上全面提升 MLLM 对齐效果（对话 +19.5%，安全 +60%）。

## 核心贡献
1. **MM-RLHF 数据集**: 120k 人工标注偏好对比对，覆盖图像/视频/安全，标注包含三维评分 + 排名 + 文字解释
2. **Critique-Based Reward Model**: 先生成 critique 再打分，7B 模型超越 72B 开源模型
3. **MM-DPO (Dynamic Reward Scaling)**: 根据 reward margin 动态调整 DPO 的 $\beta$，高质量样本获得更大权重
4. **两个 Benchmark**: MM-RLHF-RewardBench（RM 评估）+ MM-RLHF-SafetyBench（安全评估）
5. **重要发现**: 小规模 MLLM 自我改进目前不可行（与 LLM 领域不同）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、方法概览、关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 动机、核心问题、三大贡献 + Figure 1 (Pipeline) |
| [02 - Dataset](sections/02-dataset.md) | 数据收集、聚类采样、标注流程 + Figure 2, Table 1 |
| [03 - Reward Model](sections/03-reward-model.md) | Critique-Based RM 架构与训练 + Figure 3, Eq 1-3 |
| [04 - MM-DPO](sections/04-mm-dpo.md) | Dynamic Reward Scaling + Figure 4-5, Eq 4-5 |
| [05 - Experiments](sections/05-experiments.md) | 全面评估、消融、自我改进分析 + Table 2-5, Figure 6 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与未来方向 |
| [07 - Related Work](sections/07-related-work.md) | MLLM 发展、对齐、评估三方向综述 |
| [08 - Appendix](sections/08-appendix.md) | 标注规范、安全数据集、消融实验 + Table 6-7, Figure 7-12 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 数据集规模 | 120k 偏好对比对 (30k queries) |
| 初始数据池 | 10M 样本 |
| 标注投入 | 50+ 标注员 + 8 专家，2 个月 |
| 响应生成模型 | GPT-4o, Claude 3.5, Qwen2-VL-72B, LLaVA-OV-72B |
| 评估维度 | 10 维度，27 benchmarks |
| 对话能力提升 | +19.5% (LLaVA-OV-7B) |
| 安全性提升 | +60% (LLaVA-OV-7B) |
| RM 模型规模 | 7B (超越 72B 开源模型) |
| MM-DPO 默认超参 | $\beta_{\text{ori}}=0.1$, $w=0.5$, $k=0.5$ |
| 训练硬件 | 32× H800 (80G) |

---

## BibTeX

```bibtex
@inproceedings{DBLP:conf/icml/0004Y0FLZX0ZW0H25,
  author       = {Yifan Zhang and
                  Tao Yu and
                  Haochen Tian and
                  Chaoyou Fu and
                  Peiyan Li and
                  Jianshu Zeng and
                  Wulin Xie and
                  Yang Shi and
                  Huanyu Zhang and
                  Junkang Wu and
                  Xue Wang and
                  Yibo Hu and
                  Bin Wen and
                  Tingting Gao and
                  Zhang Zhang and
                  Fan Yang and
                  Di Zhang and
                  Liang Wang and
                  Rong Jin},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {{MM-RLHF:} The Next Step Forward in Multimodal {LLM} Alignment},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  volume       = {267},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/zhang25cs.html}
}
```
