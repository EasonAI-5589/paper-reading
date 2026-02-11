# MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

**作者**: Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, Ser-Nam Lim  
**单位**: University of Maryland · Meta · University of Central Florida  
**会议**: CVPR 2024  
**链接**: [项目主页](https://boheumd.github.io/MA-LMM/)

## 一句话总结

在 BLIP-2 的 Q-Former 中插入 dual memory bank（visual + query），实现在线逐帧处理视频，通过 memory compression 保持恒定长度，解决 LLM 上下文长度和 GPU 显存限制，在长视频理解任务上达到 SOTA。

## 核心贡献

1. **Long-term Memory Bank**: 提出 visual memory bank（存原始视觉特征，跨 Q-Former block 共享）+ query memory bank（存 learned query，每层独立），以 key/value 形式嵌入 cross-/self-attention
2. **Online Auto-regressive Processing**: 逐帧处理视频，最终时刻的 Q-Former 输出包含所有历史信息，LLM 输入 token 数从 N×T 降为 N（32 tokens）
3. **Memory Bank Compression (MBC)**: 基于相邻帧 token 级余弦相似度，合并最冗余的相邻帧特征，保持时序不变 + 恒定长度，优于 FIFO
4. **Plug-and-play**: 可零训练直接插入 InstructBLIP 等现有模型，off-the-shelf 即提升长视频性能（ActivityNet +7.3%, LVU +9.2%）
5. **SOTA**: LVU 63.0%（+3.8%），Breakfast 93.0%（+2.3%），COIN 93.2%（+2.4%）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：在线处理 + memory bank 解决长视频 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 + Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | Image/Video-language models, Long-term video models |
| [03 - Method](sections/03-method.md) | **核心**: Visual/Query Memory Bank + MBC 压缩 + Text Decoding |
| [04 - Experiments](sections/04-experiments.md) | LVU/Breakfast/COIN/VQA/Captioning + 消融实验 + 可视化 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 |
| [06 - Appendix](sections/06-appendix.md) | 额外消融 + 与 TESTA/MovieChat/Chat-UniVi 对比 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LVU 平均 Top-1 | **63.0%** (+3.8% vs S5) |
| Breakfast Top-1 | **93.0%** (+2.3% vs S5) |
| COIN Top-1 | **93.2%** (+2.4% vs S5) |
| Q-Former 输出 token/帧 | 32 |
| Memory bank 最佳长度 | 10-20（100 帧输入） |
| 视觉编码器 | ViT-G/14 (EVA-CLIP) |
| LLM | Vicuna-7B |
| GPU | 4× A100 |

## 💡 与相关工作的对比

| 方法 | Memory 类型 | 压缩方式 | 在线处理 | 基座 |
|------|------------|----------|---------|------|
| **MA-LMM** | Token-level (visual + query) | MBC: 相邻帧 token 相似度合并 | ✅ 自回归 | InstructBLIP |
| VisMem | Latent memory (短期视觉+长期语义) | 特殊 token 按需调用 | ❌ | Qwen2-VL |
| MemGen | 生成式隐式记忆 | 推理-记忆交织 | ❌ | LLM |
| MeMViT | Feature bank | FIFO / learnable pooling | ✅ | MViT |
| MovieChat | Raw visual features | Token merging (帧级) | ❌ | Video-LLaMA |

## BibTeX

```bibtex
@inproceedings{he2024malmm,
  author       = {Bo He and
                  Hengduo Li and
                  Young Kyun Jang and
                  Menglin Jia and
                  Xuefei Cao and
                  Ashish Shah and
                  Abhinav Shrivastava and
                  Ser{-}Nam Lim},
  title        = {{MA-LMM:} Memory-Augmented Large Multimodal Model for Long-Term Video
                  Understanding},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2024, Seattle, WA, USA, June 16-22, 2024},
  pages        = {13504--13514},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/CVPR52733.2024.01282},
  doi          = {10.1109/CVPR52733.2024.01282}
}
```
