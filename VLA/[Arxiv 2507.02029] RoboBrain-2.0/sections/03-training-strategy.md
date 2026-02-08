# 4. Training Strategy

> 来源: RoboBrain 2.0 Technical Report

---

## 📄 原文

> 💡 **Section 概览**: 三阶段训练 + RLVR 强化学习，比 v1 的 6 阶段更简洁。

---

### Stage 1: Foundational Spatiotemporal Learning

- 基础空间感知 + 时序理解
- 大规模多模态数据：dense captioning, object localization, video QA, referring expression
- **4.8M 样本，全模型训练**

### Stage 2: Embodied Spatiotemporal Enhancement

- 高分辨率、多视角、第一人称视频
- 任务：viewpoint-aware referring, 3D affordance, scene graph construction
- 长程时序依赖 + 多智能体协调
- **224K 样本，全模型训练**

### Stage 3: Chain-of-Thought Reasoning

> 💡 **这是最大的创新 — 两阶段 Reason-RFT**:
> ```
> Phase 1: CoT-SFT
> ├── 取 10% 训练数据
> ├── GPT-4o 标注 CoT rationales
> ├── SFT 训练初始 CoT 能力
> └── 195K 样本
> 
> Phase 2: RLVR (Reinforcement Fine-Tuning)
> ├── 再取 10% 训练数据
> ├── 采样模型回答，收集错误答案
> ├── 重新格式化为 MCQ / LaTeX 答案
> ├── GRPO 优化 (Group Relative Policy Optimization)
> ├── Composite reward: 答案准确性 + 格式正确性
> └── 45K 样本
> ```

> 💡 **Table 1 批读 (训练配置)**:
> ```
> Stage 1: 4.8M, LR=1e-4, seq=16384, GPU=16/64×8
> Stage 2: 224K, LR=1e-5, seq=16384, GPU=16/64×8
> Stage 3 CoT-SFT: 195K, LR=1e-5, seq=32768, GPU=4×8
> Stage 3 RFT: 45K, LR=1e-6, epoch=3, GPU=4×8, completions=8
> ```
> - Stage 3 序列长度翻倍到 32768（因为 CoT 推理链很长）
> - RFT 阶段 LR 降到 1e-6，3 个 epoch，每个问题采样 8 个回答

---

## 💡 Section 总结

### 训练策略对比
| 维度 | v1 | v2 |
|------|----|----|
| 阶段数 | 6 (4 SFT + 2 LoRA) | 3 (2 SFT + 1 CoT/RFT) |
| 总数据量 | ~12M | ~5.3M |
| 基座 | 从零训 Projector | 从 Qwen2.5-VL 继续训 |
| LoRA | A-LoRA + T-LoRA | ❌ (统一全模型) |
| 强化学习 | ❌ | GRPO (Reason-RFT) |
| CoT | ❌ | ✅ GPT-4o distillation + RLVR |
| GPU | 最多 22×8 A800 | 最多 64×8 |

### 核心洞察
1. **v2 的训练流程更简洁** — 去掉了 LoRA 分支，统一全模型训练
2. **Reason-RFT (CoT + RLVR) 是核心创新** — 这是从 DeepSeek-R1 借鉴的思路
3. **从 Stage 1 到 Stage 3 是"基础→专项→推理"的渐进** — 符合课程学习范式
4. **GPU 需求更大** — 32B 模型需要 64×8 GPU，这是 BAAI 的资源优势
