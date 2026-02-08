[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 CDPruner 的核心方法和实验验证。

---

In this paper, we introduce a novel training-free visual token pruning method CDPruner, for MLLM inference acceleration. Specifically, it first defines the conditional similarity between visual tokens based on the instruction, and then reformulates the token pruning problem with DPP to maximize the conditional diversity of the selected subset. Extensive experiments on diverse image and video benchmarks demonstrate that CDPruner achieves state-of-the-art performance across various MLLM architectures, including the LLaVA series and the advanced Qwen2.5-VL. Efficiency analysis further shows that CDPruner significantly reduces inference latency and memory usage while maintaining competitive performance, facilitating the practical deployment of MLLMs in real-world applications.

> 💡 **总结**: CDPruner = Conditional Diversity + DPP，training-free + model-agnostic，在多种 MLLM 架构和 benchmark 上 SOTA。

---

## 🔖 全文总结

### CDPruner 方法一句话
用**行列式点过程（DPP）** 在指令条件下最大化视觉 token 子集的多样性，同时保证与用户指令的相关性。

### 核心创新
1. **条件 kernel 矩阵**: $\tilde{L} = \text{diag}(\tilde{r}) \cdot L \cdot \text{diag}(\tilde{r})$ — 将指令相关性优雅地融入 DPP
2. **统一框架**: diversity + relevance 在 log-det 中自然分解为两项
3. **即插即用**: 不需要训练、不依赖特定模型、不需要 attention scores

### 局限性
- 仅适用于开源模型（需访问 visual tokens）
- 在已有内置压缩的模型上（如 Qwen2.5-VL）效果打折
- VizWiz 等指令模糊的任务上优势有限

### 未来方向
- 适当剪枝减少幻觉（POPE 实验的启示）
- 在更先进的架构上优化（克服预压缩带来的挑战）
