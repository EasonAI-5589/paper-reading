[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览
总结全文：发现问题（视觉注意力低效）→ 提出方案（FastV）→ 验证有效（多任务多模型）。

---

In this paper, we propose FastV, a plug-and-play inference cost optimization method for Large Vision-Language Models. Our insight for FastV arises from our observation that the attention computation over visual tokens is of extreme inefficiency in the deep layers of popular LVLMs though they take up a large portion of input tokens. FastV prunes out the unnecessary visual tokens according to the attention score ranking, which results in significant inference cost reduction without sacrificing performance.

> 💡 **Conclusion 批读**:
> - 一句话：发现 LVLM 深层视觉注意力低效 → 提出 FastV 根据 attention score 剪枝 → 大幅减少计算量不损性能
> - **未提及的局限性**：
>   1. 只验证了 LLaVA 系列和 QwenVL — 对 GPT-4V 等闭源模型？
>   2. 只在推理时剪枝 — 能否在训练中也利用这个发现？
>   3. 固定 K 和 R — 能否自适应地决定每个样本的剪枝比例？
> - **对后续工作的启发**：
>   - Token merging (而非简单 pruning) 可能更好
>   - 可以扩展到其他模态（audio tokens in speech models?）
>   - 与 KV cache 压缩结合可能有协同效果

---

## 🔖 Section 总结

### 全文核心要点回顾
1. **发现**: LVLM 深层对 image token 的 attention 效率极低（system prompt 的 1/472）
2. **机制**: 浅层通过 self-attention 将视觉信息聚合到 anchor token → 深层不再需要原始 image token
3. **方法**: FastV — 在第 K 层按 attention score 排序，剪掉 R% 最不重要的 image token
4. **效果**: K=2, R=50% → 45% FLOPs 减少，性能几乎不变
5. **泛化**: 多模型、多任务有效；视频场景效果更好
