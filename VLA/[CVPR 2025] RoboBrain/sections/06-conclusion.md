# 6. Conclusion + Appendix Highlights

> 来源: RoboBrain (CVPR 2025)

---

## 📄 Conclusion

In this paper, we introduce ShareRobot, a high-quality dataset that labels multi-dimensional information, including task planning, object affordance, and end-effector trajectory. We also present RoboBrain, an MLLM-based model that integrates robotic and general multi-modal data, employs a multi-stage training strategy, and leverages long videos and high-resolution images to enhance robotic manipulation.

> 💡 **总结就一段话**，核心就两件事：ShareRobot 数据集 + RoboBrain 模型。

---

## 📄 Appendix 重点摘录

### C.3 ShareRobot 的有效性

> 💡 **消融结果 (Table 7)**:
> ```
> 有 ShareRobot (Exp A): Average 62.48
> 无 ShareRobot (Exp B): Average 55.66
> → ShareRobot 带来 +6.82 的提升
> 
> 特别是在 ShareRobot 测试集上: 63.11 vs 27.03
> → 没有 ShareRobot 训练数据，模型根本不会做 fine-grained planning
> ```

### C.4 Robot Data 比例的影响

> 💡 **最佳比例 (Table 7)**:
> ```
> Robot:General 比例实验 (总量固定 200K):
> ├── 3:7 → 61.22 avg
> ├── 4:6 → 62.48 avg ⭐ 最佳
> ├── 5:5 → 61.92
> ├── 6:4 → 62.07
> └── 7:3 → 62.14
> ```
> **4:6 是甜点**: robot 数据太多会伤害通用能力，太少则 robot 能力不足。

### C.5 不同架构 & MLLM 的效果 (Table 8)

> 💡 **ShareRobot 对所有架构都有效**:
> ```
> LLaVA-OV-7B:  RoboVQA 36.29 → 43.63 (+7.3)
> Qwen2-VL-7B:  RoboVQA 24.05 → 58.94 (+34.9) ⭐ 提升最大
> OpenVLA-7B:   RoboVQA  4.11 → 54.79 (+50.7)
> ```
> 说明 ShareRobot 是通用的训练数据，不依赖特定架构。

### C.7 各阶段消融 (Table 9)

> 💡 **逐阶段提升**:
> ```
> Stage 1.5: RoboVQA  2.60, ShareRobot  9.81  ← 纯通用模型
> Stage 2-si: RoboVQA 28.90                   ← 加 single-image
> Stage 2-ov: RoboVQA 31.81                   ← 加 video
> Stage 3:    RoboVQA 62.96, ShareRobot 65.05 ← 加 robot data, 跳升!
> Stage 4-A:  Affordance 27.1                 ← A-LoRA 生效
> Stage 4-T:  Trajectory 0.09 (HD)            ← T-LoRA 生效
> ```
> Stage 3 是关键转折点：加入 robot data 后 RoboVQA 从 31.81 跳到 62.96。

---

## 💡 全文总结

### 优点
1. **数据贡献实在** — ShareRobot 是目前最大的开源机器人规划数据集，100 万+ QA pairs
2. **训练策略系统** — 多阶段训练 + 数据配比的消融实验很完整
3. **"Abstract to Concrete" 框架清晰** — Planning → Affordance → Trajectory 的级联逻辑自然
4. **泛化性验证** — 在不同架构/LLM 上都有效

### 局限性
1. **模型架构无创新** — 就是 LLaVA + LoRA，没有为 robot 场景设计专门的模块
2. **Affordance 只用 bbox** — 太粗糙，不如 segmentation mask 或 keypoint
3. **2D 轨迹 vs 3D 操作** — 实际机器人需要 3D 轨迹，2D waypoints 的实用性有限
4. **未在真实机器人上部署** — 所有实验都是 benchmark 评测，没有 real-robot 实验
5. **ShareRobot 测试集的自我评价** — 在自己的测试集上刷高分，参考价值有限

### 与后续工作的关系
- **RoboBrain 2.0** (2507.02029): 32B 模型，扩展到空间 + 时序任务
- **RoboBrain 2.5** (2601.14352): 加入深度感知和时序理解
- 系列工作的演进方向：更大模型 + 更多感知维度

### 对我们 STAR-Pro 的启发
- ShareRobot 的 "从 OXE 精选 + 重标注" 思路值得借鉴
- 多阶段训练 + 数据配比的经验可以参考
- Robot:General = 4:6 这个比例也许对视频理解任务有参考意义
