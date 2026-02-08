[← 返回 README](../README.md)

# 8. Conclusion

## 📌 预览
总结全文，强调 token compression 从单模块到多模块、从固定到自适应、从图像到视频的演进趋势。

---

MLLMs represent a significant advancement in cross-modal understanding, yet computational efficiency remains a critical bottleneck. Token compression emerges as a promising solution by reducing redundancy across MLLM components, enhancing both training and inference efficiency while alleviating long-context reasoning complexity.

> 💡 **Token compression 的定位**: 不是可选的优化技巧，而是 MLLM 规模化部署的"必要条件"。

The field has evolved from single-module to multi-module compression, from fixed-rate to adaptive dynamic approaches, and from static images to complex video sequences.

> 💡 **三个演进维度**:
> 1. **单模块 → 多模块**: 从只在 VE 或 LLM 压缩 → 全 pipeline 协同压缩
> 2. **固定率 → 自适应**: 从统一 4x 压缩 → 根据内容/任务动态调整
> 3. **图像 → 视频**: 从空间压缩 → 时空联合压缩

However, key challenges persist: the absence of unified evaluation frameworks for token compression, limited integration with mainstream training or inference acceleration libraries, and insufficient synergy with other MLLM efficiency techniques.

> 💡 **剩余挑战**: (1) 评测标准不统一；(2) 与 Flash Attention 等工程框架的兼容性不足；(3) 与量化、蒸馏等其他效率技术的协同不够。

This survey provides a systematic foundation for advancing efficient, scalable, and practically deployable multimodal large language models through strategic token compression methodologies.

---

## 🔖 Section 总结

### 全文核心 Takeaways
1. **Token compression 是 MLLM 效率优化的核心方向**
2. **按位置分类**: Vision Encoder / Projector / LLM / Hybrid — 各有优劣
3. **五个决策维度**: 时序增强、Visual vs. Text-guided、Merging vs. Dropping、Plug-in vs. Re-training、Training vs. Inference
4. **趋势**: 多模块协同 + 自适应 + 视频场景
5. **挑战**: 理论基础、自适应性、细粒度任务性能、评估标准
