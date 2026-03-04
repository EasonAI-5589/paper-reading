[← 返回 README](../README.md)

# 06 - Conclusion & Appendix

## 📌 预览
Conclusion 简要总结贡献。Appendix 内容丰富，包含详细训练配置、数据集细节、更多实验（通用 benchmark、不同架构/LLM backbone、消融实验）、可视化结果（good/bad cases）和数据集标注模板。

---

# 6. Conclusion

In this paper, we introduce ShareRobot, a high-quality dataset that labels multi-dimensional information, including task planning, object affordance, and end-effector trajectory. We also present RoboBrain, an MLLM-based model that integrates robotic and general multi-modal data, employs a multi-stage training strategy, and leverages long videos and high-resolution images to enhance robotic manipulation. Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various robotic tasks, underscoring its potential to significantly advance robotic capabilities.

> 💡 **结论评价**: 结论部分非常简洁，基本是摘要的压缩版。没有讨论局限性和未来工作（这些放在了 Appendix F 中）。

# Acknowledgments

This work was supported by the National Natural Science Foundation of China 62476011, 72225011 and 72434005.

---

# Appendix 概要

Appendix 内容较长（约占论文一半篇幅），以下为各部分概要总结：

## A. Details of Models and Training
- **模型细节**: SigLIP-so400m-patch14-384（27 层，729 tokens/image）、2 层 MLP Projector、Qwen2.5-7B-Instruct（28 层，128K context）
- **LoRA 设置**: rank=64，加在 Projector 和 LLM 的 FFN 层，冻结所有非 LoRA 参数
- **完整训练配置**: Table 4 给出了比 Table 1 更详细的版本，包括梯度累积、优化器（AdamW）、warmup ratio（0.03）、cosine LR schedule、最大序列长度等

> 💡 **关键补充信息**: Stage 3 使用 22×8=176 张 GPU，Stage 4 仅需 4×8=32 张。最大序列长度从 Stage 1 的 8192 增至后续的 32768（Stage 4 降回 4096）。

## B. Details of Training Dataset

![](../images/9a2e1765737482708b869b38f3f86aa0fa716d00479a44131128e52a79944c0d.jpg)  
Figure 7. The distribution of the entire training dataset.

> 💡 **Figure 7 解读**: 训练数据分布饼图，展示各阶段数据来源。LCS-558K 用于对齐，Image-4M/SI-3.2M/OV-1.6M 用于通用训练，RoboVQA-800K + ScanView-318K + ShareRobot-200K 构成机器人训练数据。

- **LCS-558K**: LAION/CC/SBU 子集，用于 Stage 1 视觉-语言对齐
- **Image-4M**: 8 个数据源（BLIP558K, COCO118K, CC3M 等），Stage 1.5
- **SI-3.2M**: Cambrian + Cauldron + UReader 等，Stage 2 单图
- **OV-1.6M**: 800K 重采样 + M4-Instruct + 视频数据，Stage 2 OneVision
- **RoboVQA-800K**: 5,246 长序 + 92,948 中序 episodes
- **ScanView-318K**: MMScan-224K + 3RScan-43K + ScanQA-25K + SQA3D-26K

## C. Complementary Experiments

### C.1 More Results on General Benchmarks (Table 5)

![](../images/f60aea46a1d4c1df023f6b4e0c1014dface205bc9e16416227d576ae7017e325.jpg)

> 💡 **通用 Benchmark 表现**: RoboBrain 在 12 个通用 benchmark 上与 LLaVA-OV-7B、GPT-4V 表现相当甚至更好，说明机器人训练没有显著损害通用能力。特别是 RealWorldQA 上超越了 GPT-4o（68.89 vs 58.6），展示了强大的真实世界理解能力。

### C.2 More Results on Robotic Benchmarks (Table 6)

![](../images/bfb786939e14a4d883a0880c3fb9c66a85941fa8638242f0ac20d20e3ad5909b.jpg)

> 💡 **完整机器人 Benchmark**: RoboVQA 上 BLEU-1~4 全面领先约 30%。OpenEQA 上各子类别表现均衡。ShareRobot Eval 上在 DISCRIMINATIVE（99.02）和 PLANNING-WITH（91.95）上表现尤为突出。

### C.3 Effectiveness of ShareRobot (Table 7 部分)
- 对比有/无 ShareRobot 训练：有 ShareRobot 时 ShareRobot Eval 分数 63.11 vs 27.03，证明 ShareRobot 数据的关键作用

### C.4 Effectiveness of Robot Data Proportion (Table 7 完整)

![](../images/2439ff8dc43bb4a5f64c7612db8213c790035b8220814649db832a47253111bb.jpg)

> 💡 **数据配比消融**: 测试了 3:7 到 7:3 五种比例。4:6（机器人:通用）达到最佳平衡（平均 62.48），既保持通用能力又获得机器人能力。过多机器人数据反而在通用 benchmark 上略有下降。

### C.5 Different Architecture and MLLMs (Table 8a)

![](../images/b73db68861f07da6ae07e60d52912d986bf0b2420cccaf075cbb1194225951be.jpg)

> 💡 **不同架构对比**: LLaVA-OV-7B、Qwen2VL-7B、OpenVLA-7B 三种架构加入 ShareRobot 训练后均获得显著提升。Qwen2VL-7B 在 RoboVQA 上提升最大（24.05 → 58.94）。

### C.6 Different LLM Backbones (Table 8b)
- 测试了 Qwen2.5、LLaMA、Vicuna、Mistral 四种 LLM backbone，均从 ShareRobot 数据中受益

### C.7 Ablation Studies of Different Stages (Table 9)

![](../images/d697cc7abf5b08c46d424abd88a27360fd704211d2e4e7c146bc5500d0c1b96e.jpg)

> 💡 **分阶段消融**: 从 S1.5 到 S4，每个 Stage 都带来了明确的能力提升。S3 是 planning 能力的关键跳跃（RoboVQA 31.81 → 62.96），S4-A 和 S4-T 分别提升 affordance（7.14 → 27.1）和 trajectory（1.00 → 0.09）。

## D. More Qualitative Results
- **D.1 Planning 可视化 (Fig. 8)**: 4 个案例（3 good + 1 bad），展示了 RoboBrain 在浇花、放锅、分类积木等任务上的规划能力。Bad case（清理桌面）暴露了物体识别错误、关键步骤遗漏、动作决策偏差等问题
- **D.2 Affordance 可视化 (Fig. 9)**: 多种物体的 affordance 预测，包括成功案例（瓶盖识别）和失败案例（噪声环境下的误识别）
- **D.3 Trajectory 可视化 (Fig. 10)**: 多样的 2D 轨迹预测，RoboBrain 生成的轨迹往往比 GT 更平滑高效。失败案例包括杯子定位失败、未考虑铰链约束、未考虑可变形物体属性

## E. Details of ShareRobot Dataset
- **E.1 Prompts**: Gemini 标注的完整 prompt 模板
- **E.2 Templates**: 10 种问题类型各 5 个模板
- **E.3 高层描述示例**: 40 个最频繁的 high-level descriptions（Closing/Opening drawer 最多）
- **E.4 低层指令示例**: 40 个最频繁的 low-level instructions（Grasp/Reach/Lift 最多）

## F. Future Work
计划增强空间理解、具身推理、工具使用、长文本理解能力，并关注模型效率和安全性。

---

## 🔖 Section 总结
Appendix 提供了大量补充实验和细节：
- **通用能力未被牺牲** — 12 个通用 benchmark 上与 LLaVA-OV 持平甚至更好
- **ShareRobot 数据有效** — 有/无 ShareRobot 在自家 benchmark 上差距 36 分
- **4:6 数据配比最优** — 过多机器人数据损害通用能力
- **跨架构/LLM 普适性** — 多种模型均从 ShareRobot 受益
- **渐进训练有效** — 每个 Stage 都有明确的能力增量
- **可视化揭示局限** — 物体识别错误、空间感知不足、物理约束缺失是主要失败原因
