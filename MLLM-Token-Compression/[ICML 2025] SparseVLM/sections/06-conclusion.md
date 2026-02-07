# 6. Conclusion

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

This paper introduced a text-aware training-free token optimization approach called SparseVLM which significantly decreased the test-time computations of various VLMs. Unlike prior methods, SparseVLM optimized VLMs without introducing extra parameters and fine-tuning costs.

> 💡 **总结**:
> - LLaVA: 37% latency 降低，77.8% 压缩率，保持 97% 准确率
> - 比 FastV 好 14.7%（视频任务）
> - 可部署到边缘设备和云端

---

## 💡 我的整体评价

### 优点
1. **动机清晰**: text-guided 是自然且合理的改进方向
2. **Training-free**: 实用性强，可直接用于任何 VLM
3. **实验充分**: 3 个 VLM，8 个图像 benchmark，4 个视频 benchmark
4. **消融到位**: 每个组件都有独立验证

### 局限性
1. **Rank 计算开销**: 每层算 rank 需要 SVD，对大 token 序列可能不便宜
2. **FlashAttention 兼容**: 需要额外提取 attention 矩阵，实际部署可能有工程挑战
3. **只测了 7B 模型**: 没有在更大模型（13B/70B）上验证
4. **聚类超参数**: τ, θ, k 需要调优，论文没详细讨论敏感性

### 与其他工作的关系
```
Token Compression 方法谱系:
├── Training-required
│   ├── VoCo-LLaMA (CVPR 2025) — 训练 pruning 网络
│   └── Matryoshka (ICLR 2025) — 多尺度训练
│
└── Training-free
    ├── FastV (ECCV 2024) — attention-based, text-agnostic
    ├── ToMe (ICLR 2023) — token merging
    ├── PyramidDrop (CVPR 2025) — 层级递减
    └── SparseVLM (ICML 2025) — text-guided + recycling ⭐
```

### 对 STAR-Pro 的启示
- Text-guided 的思路可以借鉴：视频理解中，不同问题应关注不同帧/区域
- Token recycling 可能对长视频理解有帮助
- Training-free 的方式如果能和 training-based 结合，效果可能更好
