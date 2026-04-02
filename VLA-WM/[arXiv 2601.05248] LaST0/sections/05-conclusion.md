[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
Conclusion 总结 LaST₀ 的三个核心贡献，并指出其代表的研究方向——更"物理接地"的推理基础模型。

---

> We introduced LaST₀, a dual-system VLA model that enables efficient reason-before-act behavior for robotic manipulation through a Latent Spatio-Temporal Chain-of-Thought (LaST CoT). By shifting reasoning from explicit traces to a compact latent space, LaST₀ overcomes the latency and representational bottlenecks inherent in prior CoT VLA approaches, while preserving the ability to model fine-grained physical dynamics essential for closed-loop control.

> 💡 **核心贡献再总结**：
> 1. **LaST CoT**：推理从语言/像素 → latent，解决延迟和表征两个瓶颈
> 2. **MoT 双系统**：快慢解耦，协调慎思推理与实时响应
> 3. **混合频率训练**：一次训练，推理时自适应频率

---

> Central to our framework is a token-efficient spatio-temporal latent representation that autoregressively captures future semantic, geometric, and proprioceptive dynamics.

> 💡 **关键设计选择的合理性**：
> - 每模态 1 token（足够紧凑）
> - 三模态互补（语义+几何+运动学）
> - 时序延展 4 步（覆盖关键阶段）

---

> We believe LaST₀ represents a step toward more physically grounded reasoning in robotic foundation models.

> 💡 **研究意义（Last-WAM 视角）**：
> LaST₀ 的 latent 世界状态预测是 WAM（World Action Model）的高效 latent 版本。
> - WAM（DreamZero）在像素空间预测 → 计算代价高
> - LaST₀ 在 latent 空间预测 → 高效
> - **Last-WAM 的潜在方向**：以 LaST₀ 的 latent 时空表征作为世界模型，实现兼顾效率与物理感知的 WAM

---

## 局限性（论文未明确提但值得注意）

- 训练时需要点云数据（需要深度传感器采集 demo），部署时虽不需要，但采集成本仍存在
- Janus-Pro（1.5B）基座相对较小，在开放世界语义泛化上可能有上限
- 长时序（3+ 次连续操作）仍有下降趋势，只是比基线慢
