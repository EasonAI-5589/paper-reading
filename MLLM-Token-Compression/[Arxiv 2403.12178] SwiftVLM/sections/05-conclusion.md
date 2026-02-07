# 5. Conclusion

> 来源: SwiftVLM

---

## 📄 原文

In this work, we revisit visual token pruning in VLMs and reveal that visual token importance varies substantially across layers. This observation explains why existing drop-based pruning methods, which rely on early selection decisions, often struggle on tasks requiring fine-grained visual reasoning.

> 💡 **批注**: 全文的核心洞察浓缩成一句话——token 重要性跨层变化，所以早期 drop 不靠谱。

To better preserve visual information, we introduce a novel pruning strategy, termed bypass, and integrate it into our proposed pruning framework, SwiftVLM. This design allows each pruning layer to perform token selection in a relatively independent manner.

> 💡 **批注**: "独立决策"是 bypass 的核心优势——每次剪枝都基于当前层的完整信息，不被之前的错误决策绑架。

Experimental results demonstrate that bypass consistently outperforms drop, suggesting its potential as a promising pruning paradigm.

---

## 💡 全文总结

### 一句话总结
SwiftVLM 提出 bypass 范式，让被"判不重要"的 visual token 走旁路到后面的层重新评估，避免了早期剪枝的不可逆信息丢失，在细粒度任务上大幅领先现有方法。

### 方法三件套
```
1. Layer Selection (动态规划)
   └── 找到最擅长区分 token 的层

2. Bypass (旁路保留)
   └── 不丢弃，保留原始 token 到下一个剪枝层

3. Token Alignment (offset 对齐)
   └── 用合并 token 的变化量近似旁路 token 的更新
```

### 与 Eason 研究方向的关联
- **Token compression for MLLM**: SwiftVLM 是 training-free 的 token pruning 方法，适合作为推理加速的 baseline
- **关键思路**: bypass 的"不丢弃、给二次机会"思路可以借鉴到其他 token reduction 方法中
- **局限性**: 只验证了 LLaVA 系列，没测更大模型；DP 选层需要少量数据；只有两次剪枝

### 值得思考的问题
1. Bypass 能不能扩展到 3 次以上的剪枝？
2. 能不能跟 training-based 方法结合？
3. 在视频理解（更多 token）场景下效果如何？
