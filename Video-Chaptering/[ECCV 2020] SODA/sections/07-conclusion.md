[← 返回 README](../README.md)

# 7 Conclusion

## 📌 预览
总结全文贡献：揭示现有框架不足 → 提出 SODA → 实验验证有效性。

---

In this paper, we demonstrated that the current evaluation framework, which is the official evaluation framework utilized in ActivityNet Challenge, is inadequate for evaluating the performance of video story description systems. Then, we proposed a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), to perform better evaluations. To match generated and reference captions considering temporal ordering, SODA first finds the optimal matching that maximizes the sum of the IoU by using dynamic programming. Then, it computes F-measure based on the METEOR scores for the matched pairs.

> 💡 **一句话总结 SODA**: DP 最优匹配（保序一对一）+ F-measure（惩罚冗余/不足）+ IoU 加权（时间定位质量直接参与评分）。

Evaluation results obtained on the ActivityNet Captions dataset showed that we can detect inadequate captions and too many or too few captions by utilizing SODA, which cannot be detected by using the current evaluation framework. Furthermore, we demonstrated that SODA gives lower scores to captions with incorrect ordering and inconsistent story descriptions, than the current evaluation framework. We also showed that SODA is superior to the current framework in detecting appropriate captions and in detecting captions with incorrect temporal order via manual evaluation.

> 💡 **SODA 的三个优势**:
> 1. 能检测不合适的字幕数量（太多/太少）
> 2. 对错误顺序更敏感（惩罚力度是 Current 的 2 倍）
> 3. 与人类判断一致性更高（0.76/0.94 vs 0.66/0.72）

---

## 🔖 Section 总结

### 对 Video Chaptering 研究的启示
1. **评估框架决定研究方向**: 如果评估不惩罚冗余，系统就会被优化成"生成越多越好"
2. **故事性是核心**: 视频描述不只是事件级别的字幕，而是一个连贯的叙事
3. **SODA 的设计思想可迁移**: 时序最优匹配 + F-measure 的框架可以用于任何需要有序描述的评估任务
4. **局限性**: 论文没有讨论 SODA 在更长视频（电影级别）或更多事件的情况下的表现
