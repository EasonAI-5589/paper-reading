[← 返回 README](../README.md)

# Abstract

## 📌 预览
Mirage 框架概览：VLM 通过 latent visual token 进行 interleaved 多模态推理，无需生成像素级图片。两阶段训练 + RL。

---

Vision-language models (VLMs) excel at multimodal understanding, yet their text-only decoding forces them to verbalize visual reasoning, limiting performance on tasks that demand visual imagination. Recent attempts train VLMs to render explicit images, but the heavy image-generation pre-training often hinders the reasoning ability. Inspired by the way humans reason with mental imagery—the internal construction and manipulation of visual cues—we investigate whether VLMs can reason through interleaved multimodal trajectories without producing explicit images. To this end, we present a Machine Mental Imagery framework, dubbed as Mirage, which augments VLM decoding with latent visual tokens alongside ordinary text. Concretely, whenever the model chooses to "think visually", it recasts its hidden states as next tokens, thereby continuing a multimodal trajectory without generating pixel-level images. Begin by supervising the latent tokens through distillation from ground-truth image embeddings, we then switch to text-only supervision to make the latent trajectory align tightly with the task objective. A subsequent reinforcement learning stage further enhances the multimodal reasoning capability. Experiments on diverse benchmarks demonstrate that Mirage unlocks stronger multimodal reasoning without explicit image generation.

> 💡 **Abstract 批读**:
> - **核心问题**: VLM 只能输出文本 → 视觉推理被迫"语言化"，在空间推理等任务上受限
> - **现有方案的问题**: 训练 VLM 生成图片 → image generation pre-training 损害推理能力（像素生成 vs 逻辑推理是两个目标）
> - **Mirage 的解**: 不生成图片，而是在 hidden state 层面插入 latent visual token — "recasts hidden states as next tokens"
> - **关键机制**: 模型选择"视觉思考"时，把当前 hidden state 直接当作下一个 token（跳过 language head），继续多模态推理
> - **训练策略**: 两阶段 SFT + RL
>   - Stage 1: distillation from ground-truth image embeddings（锚定视觉空间）
>   - Stage 2: text-only supervision（让 latent token 自由适应任务）
>   - Stage 3: RL (GRPO) 进一步提升
> - **与 VisMem 的关系**: Mirage 是 VisMem 论文中唯一的 latent space baseline。区别在于 Mirage 复用 LLM 自身 hidden state，而 VisMem 用独立的 Memory Former 模块

---

## 🔖 Section 总结

### 核心洞察
1. **Mental imagery 类比**: 人类推理时不会生成照片级画面，而是用简化的心理草图 → Mirage 用 latent embedding 模拟这一过程
2. **Hidden state 复用**: 最关键的设计——不需要额外的 visual decoder，直接把 LLM hidden state 当 visual token 用
3. **两阶段训练的必要性**: 先用 GT embedding 锚定（否则 latent token 没有意义），再用 text loss 放松（否则过度约束）
