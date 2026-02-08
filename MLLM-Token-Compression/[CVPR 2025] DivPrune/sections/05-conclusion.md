# 5. Conclusion & Supplementary

> 来源: DivPrune (Arxiv 2503.02175)

---

## 📄 Conclusion

In this paper, we proposed a token pruning method based on a max-min diversity problem, called DivPrune. In the proposed method, maximum diversity is achieved among the selected tokens, resulting in reduced redundancy. By ensuring high diversity, the selected tokens provide a more representative subset of the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments were conducted with multiple LMMs on image and video understanding tasks across 16 datasets. The results show that DivPrune achieves state-of-the-art accuracy on the tested datasets. DivPrune generalizes well to different model sizes and architectures, while also improving memory consumption and end-to-end latency for the tested LMMs.

> 💡 **批注**: 结论简洁，没有过度 claim。不过文章没有讨论局限性，这里补充一些：
> ```
> 可能的局限性:
> 1. 只在 LLaVA 系列上测试，没有测 Qwen-VL, InternVL 等
> 2. 多样性 ≠ 重要性：可能保留了多样但不重要的 token
>    （如背景区域的多样 token vs 关键目标区域的相似 token）
> 3. 贪心算法是近似解，不保证全局最优
> 4. 没有和 SparseVLM, PyramidDrop, SwiftVLM 等新方法对比
> 5. 固定剪枝率，没有自适应剪枝
> ```

---

## 📄 Supplementary Highlights

### 更多数据集验证 (Table 5)

| Method | TextVQA | VizWiz | VQAv2 |
|--------|---------|--------|-------|
| Original | 46.08 | 54.24 | 76.65 |
| FastV | 8.21 | 50.48 | 41.71 |
| VTW | 8.22 | 50.13 | 42.13 |
| **Ours** | **35.97** | **57.41** | **71.55** |

> 💡 **批读**: TextVQA 和 VQAv2 上 DivPrune 比 FastV/VTW 高 ~28%。VizWiz 上 DivPrune 甚至超过原始模型（57.41 vs 54.24）！

### LLaVA 1.5-13B 不同 TFLOP 比较

![Figure 4](../images/4fc544eb0479a378ab844c6c939d1049abb157b0d5a80d788765179628f30cef.jpg)
*Figure 4: LLaVA 1.5-13B 在不同 TFLOP ratio 下的性能对比。DivPrune 在高压缩比下显著领先。*

> 💡 **Figure 4 批读**: 和 7B 模型的趋势一致——DivPrune 在极端压缩下优势更大，13B 上也成立。

### 更多 t-SNE 可视化

![Figure 5](../images/0c42ef2378298f665807624c9f87e95548248a109adbdc23131131e40c9d658b.jpg)
*Figure 5: SeedBench 和 GQA 数据集上的 t-SNE 可视化及 Max-Min 距离直方图。*

> 💡 **批读**: 在不同数据集上都观察到同样的模式——DivPrune 的 token 均匀分散，FastV 的 token 扎堆。

### 定性示例

![Figure 6](../images/5460de59a4287de0f07636b2725cc12665c9a974a1421c9d27ebbfed2dfac578.jpg)
*Figure 6: Image captioning 定性对比。DivPrune 生成的 caption 与原始模型高度一致，而 FastV/VTW 生成完全不相关的描述。*

> 💡 **Figure 6 批读**:
> ```
> 浴室图片示例:
> - GT: "A bathroom with a bath tub near windows"
> - DivPrune: "A bathroom with a large window and a bathtub" ✓
> - FastV: "A person is standing in front of a white wall" ✗ （完全离谱）
> - VTW: "A person is standing in front of a painting of a forest" ✗
>
> 这说明 FastV/VTW 在高压缩比下丢失了关键视觉信息
> ```

![Figure 7](../images/cc0ece77412cf027d35fd185e2d29890179a4e530b314f61963f33806d3aeed8.jpg)
*Figure 7: VQA 定性对比。DivPrune 能正确回答问题，而 baseline 方法生成错误答案。*

### 超参数

| 方法 | 关键超参数 |
|------|-----------|
| DivPrune | 剪枝率 90.2% |
| FastV (7B) | K=3, R=0.001 |
| FastV (13B) | K=3, R=0.023 |
| VTW | K=4 (LLaVA 1.5), K=3 (LLaVA 1.6) |
| M³ | S=56 |
| FitPrune | 剪枝率 90% |

---

## 💡 全文总结

### DivPrune 核心思想
```
传统: token 重要性 → 选最重要的 → 冗余高 → 高压缩比下崩溃
DivPrune: token 多样性 → 选最不同的 → 冗余低 → 高压缩比下依然强

数学工具: Max-Min Diversity Problem (MMDP)
算法: 贪心 Farthest Point Sampling
距离: 余弦距离
```

### 对我们研究的启示
1. **多样性是个好的代理目标**: 不需要看 attention score，纯粹基于 token 表示的多样性就能做好选择
2. **高压缩比是关键赛道**: 50% 压缩大家都差不多，80%+ 才见真章
3. **即插即用很重要**: 不需要微调就能用，实际部署价值高
4. **和 SparseVLM/PyramidDrop/SwiftVLM 的对比缺失**: 这些方法没有被比较，可能是发表时间较近

### 与已有方法的定位
| 方法 | 策略 | 需要训练？ | 高压缩比表现 |
|------|------|-----------|-------------|
| FastV | Attention score | ✗ | 差 |
| SparseVLM | 文本引导选择 | ✗ | 未比较 |
| PyramidDrop | 渐进式剪枝 | ✗ | 未比较 |
| SwiftVLM | 早期退出+剪枝 | 需要 | 未比较 |
| PruMerge | Attention+聚类合并 | ✗ | 中等 |
| FitPrune | 校准优化 | 需要校准 | 较好 |
| M³ | 微调嵌套表示 | 需要 | 好 |
| **DivPrune** | **MMDP 多样性** | **✗** | **最好** |
