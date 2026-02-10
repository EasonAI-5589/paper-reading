[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 VisMem 的贡献和实验验证。

---

To address "visual processing bottleneck" of VLMs that impairs advanced visual capacities, we propose VisMem in this work, a cognitively inspired framework embedding dynamic latent vision memory, which integrates dual specialized memory formers guided by human patterns, with a non-intrusive memory invocation mechanism. Extensive experiments validate VisMem achieves an obvious performance improvement across various benchmarks, and exhibits strong cross-domain generalization, catastrophic forgetting mitigation, compatibility, and efficient inference, unlocking comprehensive and advanced visual potentials.

> 💡 **总结**: VisMem 的核心贡献用一句话概括——**将认知心理学的短期/长期记忆二分法实例化为 VLM 的双路 latent vision memory，通过特殊 token 按需调用、两阶段 RL 训练，在不破坏原始能力的前提下全面提升视觉理解/推理/生成**。
>
> **局限性**（论文未显式讨论）:
> 1. 训练成本：两阶段 GRPO 训练，每阶段 16 个 group trajectory → 计算量较大
> 2. 仅在 VLM benchmark 上验证，未在真实 agent 场景中测试
> 3. 记忆容量有限（8/16 个 latent token），对于极长上下文可能不够
> 4. 未与 MemGen 联合使用——理论上 VisMem（视觉记忆）+ MemGen（经验记忆）可以互补

---

## 🔖 Section 总结

### 核心洞察
1. VisMem 展示了 latent vision memory 作为新范式的潜力
2. 非侵入式设计（LoRA + 特殊 token）是实用性的关键
3. 与同组工作 MemGen 形成天然互补：VisMem 管视觉记忆，MemGen 管经验记忆
