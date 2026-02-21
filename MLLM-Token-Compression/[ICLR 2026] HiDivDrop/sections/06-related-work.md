[← 返回 README](../README.md)

# B Related Work (Appendix)

## 📌 预览
Related Work 将视觉 token 压缩方法分为 Pre-LLM、In-LLM 和 Joint 三类，HiDivDrop 属于 In-LLM 类别中的 training-based + differentiable 方向。

---

A distinctive property of MLLMs is that vision tokens are far more numerous yet information-sparse compared to text tokens (Marr, 2010), making them the primary source of redundancy and motivating research on token compression. Most prior work is training-free, pruning vision tokens during inference via heuristic rules (Chen et al., 2024b; Zhang et al., 2024b; Yang et al., 2025; Liu et al., 2024c). While effective in reducing computation, these methods introduce a train–inference mismatch. To address this issue, training-based approaches learn token reduction end-to-end, achieving alignment between training and inference and enhancing adaptability.

> 💡 **Training-free vs Training-based**:
> - Training-free：推理时用启发式剪枝 → 优点是即插即用，缺点是训练-推理不匹配
> - Training-based：端到端学习剪枝 → 训练推理一致，更适配
> - HiDivDrop 属于 training-based

---

Among training-based methods, previous studies can be grouped into Pre-LLM, In-LLM, and joint approaches, according to where the reduction is applied. (1) Pre-LLM approaches compress tokens before the LLM via compact projectors (Cha et al., 2024; Li et al., 2024b) or encoder-side modules (Hu et al., 2024; Song et al., 2025; Zhang et al., 2025). Such approaches remain disconnected from the LLM's internal reasoning, preventing compression from adapting to cross-modal interactions. (2) In-LLM approaches integrate compression into the LLM, enabling strategies for token selection, aggregation, or reduction. Some methods perform representation compression by replacing vision tokens with latent tokens (Ye et al., 2024b) or by pooling operations (Chen et al., 2024a), while others adopt selection-based pruning, either through heuristic schedules (Xing et al., 2024; Shao et al., 2025) or adaptive strategies (Ye et al., 2024a). However, most pruning approaches rely on non-differentiable Top- $k$ operators, hindering end-to-end optimization. Dynamic-LLaVA (Huang et al., 2024) relaxes this with soft gating but still provides only approximate gradients, whereas our differentiable Top- $k$ yields a continuous relaxation with stable gradient flow. (3) Joint approaches combine the strengths of both Pre-LLM and In-LLM strategies, e.g., FocusLLaVA (Zhu et al., 2024), which applies vision-guided pre-LLM compression and text-guided pruning inside the LLM. While such hybrid designs demonstrate the potential of combining both perspectives, their two-stage pipeline increases architectural complexity and prevents unified end-to-end optimization. Our work instead focuses on the In-LLM setting, aiming to achieve effective compression with a fully differentiable and text-aware token selection strategy.

> 💡 **三类方法对比**:
> | 类别 | 压缩位置 | 代表方法 | 优点 | 缺点 |
> |------|----------|----------|------|------|
> | Pre-LLM | LLM 之前 | Honeybee, TokenPacker, MQT, LeSS, LLaVA-Mini | 不影响 LLM | 无法感知 LLM 内部交互 |
> | In-LLM | LLM 内部 | FastV, PDrop, TwigVLM, VoCo-LLaMA, ATP-LLaVA, **HiDivDrop** | 感知跨模态交互 | 需要修改 LLM 前向 |
> | Joint | 前+内 | FocusLLaVA | 两者优势兼得 | 架构复杂，难以端到端优化 |
> 
> **HiDivDrop 的定位**：In-LLM + training-based + fully differentiable
> - 与 Dynamic-LLaVA 的区别：DTop-K 提供精确连续松弛 vs soft gating 的近似梯度

---

## 🔖 Section 总结

### Citation Landscape 中的关键对比
| 方法 | 类型 | 可微 | 调度 | 压缩率 |
|------|------|------|------|--------|
| FastV | In-LLM, training-free | ✗ | 单次 | ~50% |
| PDrop | In-LLM, training-based | ✗ | 均匀渐进 | ~47% |
| TwigVLM | In-LLM, training-based | ✗ | 两阶段 | ~86-92% |
| VoCo-LLaMA | In-LLM, training-based | ✗ | 压缩到 1 token | ~99.8% |
| Dynamic-LLaVA | In-LLM, training-based | 近似 | 自适应 | 可变 |
| **HiDivDrop** | **In-LLM, training-based** | **✓** | **层级自适应** | **~89-92%** |
