# VisionZip: Longer is Better but Not Necessary in Vision Language Models

**作者**: Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia  
**单位**: CUHK, HKUST, HITSZ  
**会议**: CVPR 2025  
**链接**: [arXiv](https://arxiv.org/abs/2412.04467) | [GitHub](https://github.com/dvlabresearch/VisionZip)

## 一句话总结

VisionZip 通过选取 vision encoder 中少量高注意力"dominant tokens"并合并剩余 token，在 **text-agnostic** 的前提下将视觉 token 压缩至 10%，仍保留 95% 性能，且 prefilling 加速 8×。

## 核心贡献

1. **发现视觉 token 冗余现象**：CLIP/SigLIP 的 -2 层输出中，绝大多数 token 注意力接近零，信息高度集中于少数 dominant tokens
2. **提出 VisionZip 方法**：Dominant Token Selection + Contextual Token Merging，training-free 即可使用，可选 30 分钟 projector fine-tuning 进一步提升
3. **Text-agnostic 优势**：与 FastV/SparseVLM 等 text-relevant 方法不同，VisionZip 在多轮对话场景更优，且兼容所有 LLM 加速算法
4. **全面实验验证**：在 LLaVA-1.5/NeXT、Mini-Gemini、Video-LLaVA 上全面超越 SOTA，13B 模型可比 7B 更快且更好

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 动机 + Figure 1 & 2 + 贡献概述 |
| 2. VisionZip | [02-method.md](sections/02-method.md) | 方法：Preliminary → 冗余观测 → Token Selection & Merging → Efficient Tuning |
| 3. Experiments | [03-experiments.md](sections/03-experiments.md) | 图像/视频理解实验 + 效率分析 |
| 4. Analysis | [04-analysis.md](sections/04-analysis.md) | 冗余原因分析 + 为什么优于 text-relevant 方法 + 部署优势 |
| 5. Related Work | [05-related-work.md](sections/05-related-work.md) | VLM 相关工作 |
| 6. Conclusion | [06-conclusion.md](sections/06-conclusion.md) | 总结与展望 |
| Appendix | [07-appendix.md](sections/07-appendix.md) | 补充实验、可视化、Non-CLS encoder 详解 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5 576→64 tokens | 95.2% 性能保留 (VisionZip‡) |
| LLaVA-NeXT prefilling 加速 | 7.8× |
| LLaVA-NeXT 总时间加速 | 3.0× |
| Video-LLaVA 2048→136 tokens | 93.2% 性能 (training-free) |
| Efficient Tuning 耗时 | 30 min on 8×A800 |
| Efficient Tuning 数据量 | 1/10 LLaVA-1.5 dataset |

---

## BibTeX

```bibtex
@InProceedings{Yang_2025_CVPR,
  author={Yang, Senqiao and Chen, Yukang and Tian, Zhuotao and Wang, Chengyao and Li, Jingyao and Yu, Bei and Jia, Jiaya},
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2025},
  pages={19792--19802}
}
```
