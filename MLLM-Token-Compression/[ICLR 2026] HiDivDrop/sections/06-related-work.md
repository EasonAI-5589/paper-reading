[← 返回 README](../README.md)

# B. Related Work (Appendix)

## 📌 预览
Related Work在Appendix B中，按Pre-LLM、In-LLM、Joint三类组织，HiDivDrop属于In-LLM类别。

---

A distinctive property of MLLMs is that vision tokens are far more numerous yet information-sparse compared to text tokens (Marr, 2010), making them the primary source of redundancy and motivating research on token compression. Most prior work is training-free, pruning vision tokens during inference via heuristic rules (Chen et al., 2024b; Zhang et al., 2024b; Yang et al., 2025; Liu et al., 2024c). While effective in reducing computation, these methods introduce a train–inference mismatch. To address this issue, training-based approaches learn token reduction end-to-end, achieving alignment between training and inference and enhancing adaptability.

> 💡 **Training-free vs Training-based**:
> - Training-free: 简单但有train-inference mismatch
> - Training-based: 端到端学习，训练推理一致

---

Among training-based methods, previous studies can be grouped into Pre-LLM, In-LLM, and joint approaches, according to where the reduction is applied.

**(1) Pre-LLM approaches** compress tokens before the LLM via compact projectors (Cha et al., 2024; Li et al., 2024b) or encoder-side modules (Hu et al., 2024; Song et al., 2025; Zhang et al., 2025). Such approaches remain disconnected from the LLM's internal reasoning, preventing compression from adapting to cross-modal interactions.

> 💡 **Pre-LLM的局限**: 在进入LLM之前就压缩，无法利用LLM内部的跨模态信息来指导压缩决策。

**(2) In-LLM approaches** integrate compression into the LLM, enabling strategies for token selection, aggregation, or reduction. Some methods perform representation compression by replacing vision tokens with latent tokens (Ye et al., 2024b) or by pooling operations (Chen et al., 2024a), while others adopt selection-based pruning, either through heuristic schedules (Xing et al., 2024; Shao et al., 2025) or adaptive strategies (Ye et al., 2024a). However, most pruning approaches rely on non-differentiable Top-k operators, hindering end-to-end optimization. Dynamic-LLaVA (Huang et al., 2024) relaxes this with soft gating but still provides only approximate gradients, whereas our differentiable Top-k yields a continuous relaxation with stable gradient flow.

> 💡 **In-LLM方法分类**:
> - **表示压缩**: VoCo-LLaMA（latent token替换）、LLaVA-PruMerge（pooling）
> - **选择剪枝**: FastV、PDrop（heuristic）、ATP-LLaVA（adaptive）
> - **HiDivDrop的定位**: In-LLM + differentiable selection + 层级感知

**(3) Joint approaches** combine the strengths of both Pre-LLM and In-LLM strategies, e.g., FocusLLaVA (Zhu et al., 2024), which applies vision-guided pre-LLM compression and text-guided pruning inside the LLM. While such hybrid designs demonstrate the potential of combining both perspectives, their two-stage pipeline increases architectural complexity and prevents unified end-to-end optimization.

Our work instead focuses on the In-LLM setting, aiming to achieve effective compression with a fully differentiable and text-aware token selection strategy.

> 💡 **方法定位图谱**:
> ```
> Token Compression
> ├── Training-free: FastV, SparseVLM, VisionZip
> └── Training-based
>     ├── Pre-LLM: Honeybee, TokenPacker, LLaVA-Mini
>     ├── In-LLM: PDrop, TwigVLM, VoCo-LLaMA, HiDivDrop✓
>     └── Joint: FocusLLaVA
> ```

---

## 🔖 Section 总结
HiDivDrop是In-LLM类别中唯一同时具备层级感知（三段式）和可微分token选择（DTop-K）的方法。
