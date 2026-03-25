[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

Related Work 分两条路线：VLA 策略 vs WAM / 视频机器人策略。重点是定位 Fast-WAM 的独特角色——不是提出新 WAM，而是研究 WAM 增益来源。

---

## 2.1 Vision-Language-Action Policies (VLA)

> Recent progress in embodied foundation models has been driven by VLA policies, which directly map visual observations and language instructions to robot actions using large pretrained vision-language backbones.

**代表工作**: OpenVLA [9], π0 [10], π0.5 [11], GR00T N1 [12], RT-2 [13], RDT-1B [14], SmolVLA [16], Gemini Robotics [17], DexVLA [19]

> However, standard VLA pretraining is largely based on static image-text data and does not explicitly model how the physical world evolves under action.

> 💡 **VLA 的局限**: 预训练基于静态图像-文本对，不建模物理动力学。Fast-WAM 保留了 VLA 式的推理接口（直接 obs→action），但通过视频联合训练获得了 WAM 式的世界理解。本质上是 **VLA 的推理效率 + WAM 的训练质量**。

---

## 2.2 World Action Models 和视频机器人策略

> A parallel line of work studies robot control through future visual prediction, using video generation as a way to model environment dynamics and infer actions.

两大范式：

| 范式 | 描述 | 代表工作 |
|------|------|---------|
| **Joint modeling** | 视频+动作在共享生成过程中联合建模 | WAM [4], Motus [5], Unified WM [6] |
| **Imagine-then-execute** | 先生成未来视频，再基于未来预测动作 | LingBot-VA [3], ViDAR [7], Du et al. [8] |

> 💡 **本文的定位**: 不是提出又一个 imagine-then-execute WAM，而是**研究 WAM 的增益到底从何而来**。

### 与 VPP / UVA 的关系

> Our work is also related to recent efforts that exploit video modeling for action prediction while reducing or bypassing explicit test-time video synthesis. VPP [34] conditions robot policies on predictive visual representations extracted from a video diffusion model, while UVA [35] jointly models video and action and skips video decoding at test time for faster inference.

> 💡 **与 VPP/UVA 的区别**: 它们也尝试减少推理时视频合成，但**没有做控制变量实验**来分离训练 vs 推理的贡献。Fast-WAM 的核心价值在于实验设计，而不仅仅是架构设计。
