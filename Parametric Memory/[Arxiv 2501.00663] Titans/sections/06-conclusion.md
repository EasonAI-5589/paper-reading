[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览
总结全文贡献和主要发现。

---

In this paper, we present a neural long-term memory that, as a meta in-context learner, learns to memorize at test time. The neural memory module is a recurrent model in nature, and is adaptively memorizing tokens that are more surprising or are close to surprising tokens. Comparing to modern recurrent models, it has more expressive memory update and storing mechanism. Using this memory, we present Titans architectures, and its three variants, in which we suggest to incorporate the memory module as (1) a context, (2) gating, and (3) a layer. Our experimental evaluation on diverse tasks tasks validate that Titans are more effective than Transformers and recent modern linear recurrent models, specifically for long context. That is, Titans can scale to larger than 2M context window size with better accuracy than baselines.

Titans are implemented in Pytorch and JAX and we intend to make the code we used to train and evaluate our models available soon.

> 💡 **Conclusion 批读**:
> - Titans 的核心创新：**测试时学习的神经记忆**，通过 surprise metric + momentum + weight decay 实现
> - 三种架构变体（MAC/MAG/MAL）提供了不同的效率-效果 trade-off
> - 关键结果：超越 Transformer 和所有现代线性循环模型，扩展到 2M+ 上下文
> - 代码承诺开源（PyTorch + JAX）

> 💡 **个人思考**:
> 1. **优化器即记忆** 是一个极其优雅的 insight，把 surprise → gradient, memory → parameters 的映射做到了极致
> 2. 后续重要问题：如何为不同任务自动选择记忆深度？MAC/MAG/MAL 能否动态切换？
> 3. 这篇工作对 Agent Memory 领域的启示：parametric memory（将经验编码到参数中）比 explicit memory（存储原始数据）可能更高效

---

## 🔖 Section 总结

### 全文核心洞察
1. **记忆视角统一框架**: 所有序列模型 = 记忆结构 + 读写操作
2. **Surprise = Gradient**: 人类记忆的"意外更好记"直觉有了数学形式
3. **优化器 = 记忆系统**: SGD+momentum+weight decay 就是完整的记忆更新机制
4. **并行融合 > 串行堆叠**: MAC/MAG 优于 MAL，对 hybrid 模型设计有指导意义
5. **测试时学习**: Neural memory 在推理时仍更新参数，这是与所有其他模型的根本区别
