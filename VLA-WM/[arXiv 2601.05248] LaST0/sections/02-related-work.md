[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 覆盖两个方向：(1) VLA 模型的发展脉络，(2) Latent CoT 在通用 VLM 和具身 AI 中的应用。

---

## VLA 模型

> VLA models are primarily driven by scaling robot demonstration data and adapting pretrained VLMs for robotic control. To improve expressivity for continuous actions, recent VLA research has increasingly employed continuous generative policy heads. Diffusion-based VLA models complex action distributions through iterative denoising, while flow-matching formulations offer an alternative that can improve sampling efficiency and stability.

> 💡 **VLA 三代演进**：
> | 世代 | 做法 | 代表 |
> |------|------|------|
> | 1.0 | 离散 token 输出 action | RT-2, OpenVLA |
> | 2.0 | Diffusion/Flow Matching action head | π₀, π₀.₅, CogACT |
> | 2.5（CoT） | 推理再 act | CoT-VLA, LaST₀ |

---

> Recent research equips VLA models with "reason-before-act" components to improve physical world reasoning. Some adopt textual chain-of-thought (CoT) generation for future task planning. Subsequent work extends generative text planning to future image prediction.

> 💡 **"Reason before act" 的进化**：文本 CoT → 视觉预测 CoT → **Latent CoT（LaST₀）**

---

## Latent CoT

> Recent work of VLM in the general domain has explored latent CoT reasoning to address the limitations of explicit CoT on ineffable visual-spatial matching and high-cost generation. These methods perform multi-step inference directly in continuous latent spaces.

> 💡 **通用 VLM 的 latent CoT 趋势**：LLM 社区已在探索把中间推理步骤"内化"到连续 latent 中，不生成可见文本。LaST₀ 把这个思路带到机器人操作领域。

---

> Beyond general-purpose VLMs, similar approaches have been adopted in the embodied intelligence domain:
> - **LCDrive**: 用 action-aligned latent rollout 替代语言解释，应用于自动驾驶
> - **Thinkact**: 把中间运动规划压缩成紧凑表征

> 💡 **LaST₀ 的独特性**：不是简单套用通用 latent CoT，而是专为机器人操作设计了"物理接地"的 latent 空间——同时编码语义意图（视觉）、几何结构（点云）和机器人状态（本体感知）。
