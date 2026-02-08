# 5. Infrastructure + 7. Conclusion

> 来源: RoboBrain 2.0 Technical Report

---

## 📄 Infrastructure (Section 5)

> 💡 **基础设施要点** — 大规模训练的工程经验：
> ```
> 训练框架: FlagScale (BAAI 开源)
> 
> 关键优化:
> ├── Non-uniform Pipeline Parallelism
> │   └── ViT 在第一个 pipeline stage，减少 LLM 层数以平衡
> ├── Separate Recompute
> │   └── ViT 开 recompute (省内存)，LLM 关 recompute (省计算)
> ├── Pre-Allocate Memory
> │   └── 第一步 pad 到最大长度，预分配显存，避免碎片化 OOM
> ├── JSON-only Preprocessing
> │   └── 只预处理 JSON，图片延迟加载 → 预处理时间减少 90%
> └── Distributed Data Loading
>     └── PP 只有首末 stage 加载数据，TP 只有一个 GPU 加载 → 减少冗余 IO
> 
> 推理优化:
> ├── Mixed-bit quantization: ViT 全精度 + LLM 8-bit weights + 16-bit activations
> └── 推理延迟降低 ~30%
> 
> 强化学习: VeRL 框架 + GRPO 算法
> ```

> 💡 **Pre-Allocate Memory 这个技巧值得学习**: 
> 变长输入导致 PyTorch 显存碎片化 → OOM。
> 解决方案不是 empty_cache()（太慢），而是第一步 pad 到最大长度让显存分配一次到位。

---

## 📄 Conclusion (Section 7)

### 未来方向

> 💡 **两个方向**:
> 1. **Embodied VLM-powered VLA** — 把 VLM 的感知推理能力接入 VLA 框架，增强动作生成的通用性
> 2. **System-Level Integration** — 与机器人操作系统深度集成，serverless 部署，技能注册，"智能体 App Store"

---

## 💡 全文总结

### RoboBrain 2.0 的核心贡献
1. **从"三合一"到"多合一"** — v1 做 planning+affordance+trajectory，v2 增加 pointing, spatial referring, placement, close-loop, multi-agent, scene graph
2. **Pseudo-3D 空间数据 pipeline** — 从 2D 图片自动构建 3D scene graph + 31 种空间概念
3. **Reason-RFT** — CoT SFT + GRPO 强化学习，让模型学会推理
4. **多机器人协作** — 基于 RoboOS 的 44K 协作数据，覆盖 1659 种任务
5. **工程优化** — FlagScale, Pre-Allocate Memory, Mixed-bit quantization

### 局限性
1. **没有真机实验** — 和 v1 一样，全是 benchmark 评测
2. **32B 模型资源门槛高** — 训练需要 64×8 GPU
3. **部分任务 7B > 32B** — 说明大模型不总是最优，可能存在训练不充分
4. **Trajectory 性能可能退步** — 统一模型的 trajectory vs 专门 LoRA 的 trade-off
5. **大量依赖 GPT-4o/DeepSeek 生成数据** — 数据质量受限于这些模型的能力

### 与其他工作的关系
- **v1 → v2**: 架构换代 (LLaVA → Qwen2.5-VL)，能力扩展 (空间+时序)，训练升级 (+RLVR)
- **v2 → v2.5**: v2.5 加入深度感知和更深的时序理解
- **vs pi0 系列**: pi0 是 VLA（直接输出动作），RoboBrain 是 VLM（输出计划/坐标/推理），定位不同
- **vs Gemini Robotics**: Google 的方向类似但闭源，RoboBrain 开源是优势
- **谭桦杰的角色**: 核心作者，同时主导了 RoboOS、Reason-RFT 等配套工作

### 对我们研究的启发
1. **Pseudo-3D pipeline** 可以用于其他需要 3D 理解的任务
2. **Reason-RFT (CoT + GRPO)** 可以借鉴用于 video understanding 的推理增强
3. **数据配比和阶段训练** 的经验值得参考
