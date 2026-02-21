[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
总结 ToDRE 的核心贡献、实验结果和方法意义。

---

In this work, we systematically analyze redundancy in LVLM inference and identify two key inefficiencies: (1) redundant visual tokens that inflate intra-modal computation, and (2) tokens that contribute little cross-modal information during decoding. To address these inefficiencies, we propose TODRE, a training-free, architecture-agnostic framework that first selects a maximally diverse subset of visual tokens via a greedy max-sum diversification algorithm, then removes all remaining visual tokens once cross-modal attention fades. Experiments on twelve image- and videolanguage benchmarks show that ToDRE prunes up to $90 \%$ of visual tokens while preserving $9 5 . 0 \%$ of the original performance, achieving $2 . 6 \times$ faster inference and $1 4 . 5 \%$ lower memory usage than uncompressed baselines.

> 💡 **总结批注**:
> - 两类低效：intra-modal 冗余 + cross-modal 无关
> - 两阶段应对：diversity selection + relevance reduction
> - Training-free + architecture-agnostic → 实用性强
>
> **未提及的局限性**:
> - Greedy diversification 的 O(k·n) 复杂度在超高分辨率图像（n 很大）时可能不忽略
> - Stage 2 的阈值 τ 和检测层位置是超参数，虽然消融显示不敏感但仍需手动设置
> - 没有在更大模型（70B+）上验证
> - 没有与需要训练的方法（如 TokenPacker、LLaVA-KD）做性能对比
> - 缺少 OCR-heavy 任务（如 DocVQA、ChartQA）的评估，这类任务对空间细节更敏感

---

## 🔖 Section 总结

### ToDRE 一句话
分离 token diversity（intra-modal）和 task relevance（cross-modal）两个正交维度，两阶段无训练压缩 90% visual token，保持 95% 性能。
