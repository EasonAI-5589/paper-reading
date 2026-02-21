# 5. Analysis and Discussion

## 5.1 Efficiency Analysis

- DART: 2.99× prefill speedup, 1.99× total speedup, POPE 仅降 2.4%
- FLOPs 不能代表实际加速：SparseVLM FLOPs 只多 2.8% 但 speedup 低 21.6%（多阶段串行处理）
- Performance-Latency trade-off (Figure 4): 在 LLaVA-Next-7B 上，一些现有方法甚至不如 random token retention
- DART 无缝集成 FlashAttention，token reduction overhead < 0.08s

## 5.2 Influence from Selection of Pivot Tokens

Table 6 (Appendix) 评估了多种 pivot 选择策略：
- Max attention score (♠), Min attention score (♡)
- Max K-norm, Min K-norm, Max V-norm, Min V-norm
- Random selection

**关键发现**：
1. 所有策略都达到 vanilla 模型 94.9%+ 的性能
2. Random pivot 仅比最佳策略低 1.2%，但仍超越 importance-based methods 2.1%
3. "Important" pivots (attention ♠) 仅比 "unimportant" pivots (attention ♡) 好 0.2%
4. K-norm♠ 和 K-norm♡ 保留的 token 集合 overlap < 50%，但性能相当

> 💡 **Pivot 选择的鲁棒性是 DART 最强的 ablation 之一**。它直接证明了论文的核心论点：duplication 比 importance 更关键。不管你用什么标准选 pivot，只要去除 duplicate tokens，效果都好。

> 💡 这里有一个深层 insight：overlap < 50% 说明存在多个等效的 "minimal information-preserving" token 子集。这与 compressed sensing 中的 RIP (Restricted Isometry Property) 有类似的味道——稀疏表示不唯一，但都能近似还原原始信号。

## 5.3 Pruning Layer (推测，基于默认配置)

默认在 layer 2 后 pruning。这与 FastV 的发现一致：浅层处理后 visual tokens 已经形成了稳定的表示，适合做 pruning 决策。

> 💡 Layer 2 作为 pruning point 的选择与 HiDivDrop 的 "Late Injection" 形成有趣对比。HiDivDrop 认为浅层是 "passive" 的，不需要 visual tokens；DART 则认为 layer 2 后的表示足够成熟来判断 duplication。两者观察到的现象一致（浅层不重要），但解决方案不同。
