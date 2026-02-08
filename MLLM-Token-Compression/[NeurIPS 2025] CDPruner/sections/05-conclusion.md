# 5 Conclusion & Appendix

> 来源: Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs

---

## 📄 Conclusion

In this paper, we introduce a novel training-free visual token pruning method CDPruner, for MLLM inference acceleration. Specifically, it first defines the conditional similarity between visual tokens based on the instruction, and then reformulates the token pruning problem with DPP to maximize the conditional diversity of the selected subset.

Extensive experiments on diverse image and video benchmarks demonstrate that CDPruner achieves state-of-the-art performance across various MLLM architectures, including the LLaVA series and the advanced Qwen2.5-VL.

> 💡 **一句话总结**: CDPruner = Conditional Similarity + DPP → 同时优化多样性和指令相关性，training-free SOTA。

---

## Appendix 要点

### A. Fast Greedy MAP Inference

> 💡 **算法细节（大白话）**:
> ```
> DPP 的 MAP 推断是 NP-hard，但有贪心近似算法：
>
> 每轮迭代：
> 1. 对于还没选的每个 token i，计算"加入 i 后行列式增加多少"
> 2. 选增加最多的那个（用 Cholesky 分解加速）
> 3. 更新 Cholesky 因子（增量更新，不用重新算）
> 重复 m 轮
>
> 复杂度: O(nm²)，CUDA 并行化后 < 10ms
> 近似保证: (1 - 1/e) ≈ 63.2% of optimal
> ```

### C. Additional Results

**LLaVA-1.5-13B / LLaVA-NeXT-13B**: 更大 LLM 对剪枝更鲁棒，CDPruner 仍然一致最优。

**InternVL3-8B**: DivPrune 在此模型上显著下降（因为不考虑指令），CDPruner 在 90% 剪枝下保持 83.9%，比 FastV 高 3%。

**Balance factor θ 消融**: 可以通过 θ 调节多样性和相关性的权重，但默认版本（不调 θ）已经很强。

### E. Limitations

> 💡 **局限性**:
> 1. **只适用于开源模型**——无法应用于 ChatGPT/Gemini/Claude 等黑盒模型
> 2. **高级架构更敏感**——Qwen2.5-VL、InternVL3 因为内置了 token 压缩，再剪枝时性能下降更明显

### F. Broader Impacts

- 降低 MLLM 部署成本和延迟
- 使边缘设备部署成为可能
- 不能防止恶意使用

---

## 💡 全文总结

### CDPruner 在方法图谱中的位置
```
Visual Token Pruning 方法图谱:
├── Attention-based: FastV → PyramidDrop → SparseVLM
│   特点: 考虑指令，但保留重复 token
├── Vision-based: LLaVA-Prumerge → VisionZip
│   特点: 不需要 attention，但忽略指令
├── Similarity-based: DART → DivPrune (MMDP)
│   特点: 保证多样性，但忽略指令
└── Conditional Diversity: CDPruner (DPP) ⭐
    特点: 多样性 + 指令相关性，全面最优
```

### 对我们研究的启发
1. **DPP 是一个强大的子集选择工具**，可以考虑用在其他需要"多样+相关"的场景
2. **条件化是关键创新**——从 DPP 到 Conditional DPP，简单但有效
3. **高分辨率/视频场景最适合 token pruning**——冗余天然更多
4. **合理剪枝可能减少幻觉**——POPE 实验的有趣发现
5. **与 STAR-Pro 的关联**: CDPruner 的 token 剪枝思路可以作为我们模型的推理加速方案

### 代码
https://github.com/Theia-4869/CDPruner
