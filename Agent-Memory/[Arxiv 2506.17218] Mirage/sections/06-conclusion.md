[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览
总结 + 三个局限性：合成数据质量、未扩展到统一模型、仅评估空间推理任务。

---

In this work, mimicking human mental imagery, we propose Mirage, a lightweight framework that interleaves compact latent visual tokens with text so a vision–language model can reason multimodally without ever generating pixel-level images. Specifically, our framework is trained in two stages: a joint supervision stage that anchors latent tokens to visual embeddings while learning the surrounding text, followed by a text-only supervision stage that lets those tokens adapt freely to support answer generation. A brief reinforcement-learning refinement further aligns the entire trajectory with task goals. Across four spatial-reasoning benchmarks, Mirage consistently outperforms text-only baselines, underscoring the effectiveness and potential of latent visual reasoning for multimodal models.

> 💡 **批注**: 总结精炼——三句话概括全文：(1) latent visual token 机制；(2) 两阶段训练 + RL；(3) 四个 benchmark 一致提升。

---

**Limitations and Future Works.** While effective, our framework has certain limitations:

- **Synthetic Data Quality**: The performance of our interleaved reasoning depends on the quality of the generated multimodal trajectories. Carefully curating high-quality datasets for unified reasoning models is an important next step.
- **Extend to Unified Models**: Our framework explores the latent space within a reasoning model, whereas unified models jointly align the latent space through image and text token generation during training. Despite current limitations in interleaved generation performance, whether the aligned feature space of unified models can be leveraged to further improve latent reasoning design remains an open question.
- **Task Scale beyond Spatial Reasoning**: Currently, our evaluation is limited to spatial-reasoning benchmarks. How to extend our framework to broader multimodal or purely textual tasks remains an open direction.

> 💡 **局限性批读**:
> 1. **数据质量**: 这是最大的瓶颈。helper image 的质量和合成推理链的质量直接限制上限
> 2. **与统一模型的结合**: 有趣的方向——如果 base model 本身就能生成图片（如 Chameleon），latent token 是否可以用其对齐过的特征空间？
> 3. **任务范围窄**: 只评估了空间推理（迷宫、拼图、视角变换、几何）。更广泛的 VQA、视觉常识推理等未测试
> 
> **对我们项目的启示**:
> - Mirage 的 latent token 机制简洁有效，但任务范围有限
> - 与 VisMem 对比：VisMem 有显式的 memory 系统（短期+长期），适用性可能更广
> - Mirage 作为 baseline：需要关注其在更多任务上的泛化能力

---

## 🔖 Section 总结

### 核心洞察
1. **Mirage 是轻量级方案**: 不需要额外模块（VisMem 需要 Memory Former），不需要像素解码
2. **局限性明确**: 任务范围窄 + 数据质量瓶颈
3. **与 VisMem 的互补性**: Mirage 强在简洁和可微优化，VisMem 强在 memory 系统设计和更广泛的适用性
