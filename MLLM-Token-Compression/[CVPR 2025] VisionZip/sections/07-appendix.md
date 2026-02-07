# Appendix

> 来源: VisionZip (CVPR 2025)

---

## 📄 概览

> 💡 **Appendix 概览**: 附录包含更多实验细节和可视化，重点关注：Non-CLS encoder 支持（A.2）、13B 和训练模式结果（B.1）、视频方向展望（B.2）、效率细节（B.3）、冗余可视化（D）。

---

## A. Further Discussion

### A.1 Comparison with Text-relevant Efficient VLM

VisionZip 的三个维度优势：
1. **Better Performance** — vision encoder 预聚合信息到 proxy token，text-relevant 方法选到的 "语义相关" token 信息量不足
2. **More Efficient** — 在 LLM 之前就减少 token，避免浅层的无效计算；text-relevant 方法甚至可能增加中间计算的显存
3. **More Application Scenarios** — 兼容所有 LLM 加速算法，适合多轮对话

### A.2 VisionZip for Non-CLS Vision Encoders (SigLIP)

> 💡 **批注**: 对于没有 CLS token 的 SigLIP:
> ```python
> # 计算每个 token 被其他 token attend 的平均程度
> attn_rec = attn.mean(dim=1).mean(dim=1)  # (B, S)
> # 选 Top-K
> _, topk_idx = attn_rec.topk(K, dim=1)
> ```
> 思路完全一致，只是把 "CLS 关注谁" 换成 "谁被大家关注最多"。

---

## B. Additional Experiments

### B.1 Image Understanding

#### Token 数量配比

| 模型 | 保留 64 | 保留 128 | 保留 192 |
|------|---------|---------|---------|
| LLaVA-1.5 | 54D + 10C | 108D + 20C | 162D + 30C |
| LLaVA-NeXT | 135D + 25C | 270D + 50C | 540D + 100C |

> 💡 **批注**: Dominant : Contextual ≈ 5.4 : 1（约 85% dominant + 15% contextual）。说明大部分信息确实集中在 dominant token 中，contextual 只是补充。

#### 13B 模型结果
- 保留 192 tokens: **97.9%** (training-free), 98.7% (fine-tuned)
- 保留 64 tokens: **93.7%** (training-free), 94.8% (fine-tuned)
- 13B 比 7B 在同等压缩下性能更好 → LLM 越大越能利用压缩后的 token

#### Training Mode 结果
- 保留 192 tokens 训练: 性能 **提升 0.6%**！
- 保留 128 tokens 训练: 保留 99.6%

> 💡 **批注**: 训练时就用 VisionZip 不仅省时间，还可能提升性能——因为去掉了干扰 token，模型能更专注学习。

### B.2 Video Understanding

> 💡 **批注**: 视频方向的展望很有价值：
> ```
> 当前: 8 帧 × 256 tokens = 2048 tokens → 受限于显存
> VisionZip: 8 帧 × 17 tokens = 136 tokens
> → 同样显存可以处理 80 帧！
> → 1 小时视频 → 5-10 小时视频
> ```

### B.3 Efficiency Analysis

#### CUDA Memory (LLaVA-NeXT 13B, 保留 320 tokens)

| 配置 | Memory | SQA Avg |
|------|--------|---------|
| Vanilla 13B | 36,721 Mb | 100% |
| VisionZip 13B | 28,810 Mb (-21%) | 94.7% |
| VisionZip-8bit | 16,632 Mb (-55%) | 95.0% |
| VisionZip-4bit | 10,176 Mb (-72%) | 94.0% |

> 💡 **批注**: VisionZip + 4bit 量化 = 只需 10GB 显存跑 13B，单张 3090 就够了！

#### Training Time (LLaVA-NeXT 7B, 保留 640 tokens)
- Vanilla: 33.8h, 63,558 Mb
- VisionZip-Train: **15.9h** (-53%), 35,326 Mb (-44%), 性能 99.0%

#### Inference Time (LLaVA-NeXT 13B)

| Token 数 | Prefilling | Total (TextVQA) | Avg 性能 |
|---------|-----------|-----------------|---------|
| 2880 | 129.4ms | 2506s | 100% |
| 640 | 48.2ms | 1219s | 97.5% |
| 320 | 30.3ms | 995s | 94.7% |
| 160 | 23.9ms | 888s | 91.3% |

> 💡 **批注**: 13B + 640 tokens: prefilling 48ms, total 1219s → 比 7B vanilla (54ms, 1598s) **更快更好**！

---

## D. Visualization

### D.1 冗余可视化

![Figure 9](../images/b0473076761b34c2f9bb8a84d09c9b69925c234b898ace594da91c65ba549d2f.jpg)
*Figure 9: CLIP 模型的冗余可视化 — COCO 数据集示例*

![Figure 10](../images/4cfffac1d39701a813267b48f08c17fb31ca2473a42bfbfcd5d1d05f2c7f2b2c.jpg)
*Figure 10: CLIP 模型的冗余可视化 — 更多示例*

![Figure 11](../images/3c37af7f5a957c19f51ee2ec4af3fe409d59a1ca69a504d265e6414f4b7d14a8.jpg)
*Figure 11: SigLIP 模型的冗余可视化 — 同样存在 attention 集中现象*

> 💡 **批注**: 三张图共同说明——无论 CLIP 还是 SigLIP，无论什么图片内容，attention 集中在少数 token 的现象是 **普遍的**。

### D.2 Attention 分布变化

![Figure 12](../images/97aea90df4f541e6a581a67eb5ab9593481b19495565f4201ef86cbf434e8e67.jpg)
*Figure 12: Attention 从浅层到深层的变化过程（示例 1）*

![Figure 13](../images/2a2cff853b0af85c74e7539176ed4a00fefc7cb950153d529291198e38fddb19.jpg)
*Figure 13: Attention 从浅层到深层的变化过程（示例 2）*

### D.3 Feature Misalignment 可视化

![Figure 14](../images/b4c7340a87b1697cdaf0977bf65108357ac0914b165bc96c5cc65a5972c42394.jpg)
*Figure 14: Feature Misalignment 可视化。红点是选中的 token，heatmap 显示其 attention 不在语义相关位置，而在 dominant/proxy token 上。*

> 💡 **Figure 14 批读**: 这张图直接证明了为什么 text-relevant 方法会失败——你以为选了 "人" 上面的 token 就包含人的信息，但实际上人的信息已经被转移到角落的 proxy token 了。

---

## 💡 Appendix 总结

### 关键补充信息
1. SigLIP 的适配方案简单有效（均值替代 CLS）
2. 13B 模型在 VisionZip 下表现更好（LLM 越大越能利用精选 token）
3. 训练时用 VisionZip 不仅省时间还可能提性能
4. VisionZip + 4bit 量化 = 10GB 跑 13B
5. 视频方向：同样显存处理 5-10× 更多帧
