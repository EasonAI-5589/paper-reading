# 5. Conclusion & 评价

> 来源: PyramidDrop (CVPR 2025)

---

## 💡 我的整体评价

### 优点
1. **实证驱动**: 先发现规律（浅层需要、深层冗余），再设计方法
2. **训练 + 推理双加速**: 这是比 FastV/SparseVLM 都强的独特卖点
3. **简洁有效**: 只用 last instruction token 做排序，FlashAttention 兼容
4. **实验充分**: 16 个 benchmark，多个模型，多个设置

### 局限性
1. **只用 last instruction token**: 可能丢失多关键词查询的信息（SparseVLM 的 text rater 更细致）
2. **固定 stage 划分**: 32 层固定分 4 段，不一定对所有架构最优
3. **没有 token recycling**: 直接丢弃，可能丢失信息（SparseVLM 的回收机制更好）
4. **高分辨率文档理解敏感**: DocVQA 在 λ=0.4 时明显下降

### 与其他方法的完整对比

```
Token Compression 方法总览:

FastV (ECCV 2024):
  ✅ 简单、training-free
  ❌ Layer 2 就剪太激进、text-agnostic、性能差

SparseVLM (ICML 2025):
  ✅ Text-guided、token recycling、性能好
  ❌ 需要提取完整 attention 矩阵（FlashAttn 不兼容）、不能加速训练

PyramidDrop (CVPR 2025):
  ✅ 渐进式、训练+推理双加速、FlashAttn 兼容、性能最好
  ❌ 没有 text rater、没有 token recycling

理想方法 = PyramidDrop 的渐进式 + SparseVLM 的 text guidance + token recycling
```

### 对 STAR-Pro 的启示
- 渐进式剪枝对视频理解很有参考价值：前几层保留所有帧信息，深层只保留关键帧 token
- 训练时使用 PyramidDrop 可以加速 SFT 阶段
- 和 STAR-Pro 的 spatial-temporal reasoning 可能有结合点：空间维度用 PyramidDrop 压缩
