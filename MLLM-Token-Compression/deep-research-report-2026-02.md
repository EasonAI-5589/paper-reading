# MLLM Visual Token Compression — 深度调研报告

> 📅 调研日期：2026-02-14
> 🎯 目标：补充用户已有 100+ 篇论文列表中可能遗漏的最新工作
> 📊 新发现论文：**23 篇**

---

## 1. 新发现的论文

### 1.1 Vision Encoder 阶段

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **LaCo** | [2507.02279](https://arxiv.org/abs/2507.02279) / ICLR 2026 | **VE 内部中间层**压缩（非后端），Layer-wise pixel-shuffle + 残差学习非参数 shortcut，训练效率 +20%，推理 +15% | - |
| **FSR (Focus-Scan-Refine)** | [2602.05809](https://arxiv.org/abs/2602.05809) | Training-free，仿人类视觉三阶段：聚焦关键局部 → 扫描全局上下文 → 相似度 merging 精炼，64-192 token 预算下 SOTA | [ILOT-code/FSR](https://github.com/ILOT-code/FSR) |
| **PIO-FVLM** | [2602.04657](https://arxiv.org/abs/2602.04657) | Training-free 两阶段混合：pre-filtering + objective-driven refinement，支持有/无 VE 两种模式，多模型多预算 SOTA | - |
| **VisPruner** *(用户已有但补充 ICCV 2025 接收信息)* | [2412.01818](https://arxiv.org/abs/2412.01818) / ICCV 2025 | CLS attention + cosine similarity 去重，91% FLOPs 降低 | [Theia-4869/VisPruner](https://github.com/Theia-4869/VisPruner) |

### 1.2 Projector 阶段

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **Magic-MM-Embedding** | [2602.05275](https://arxiv.org/abs/2602.05275) | **无参数空间插值** 75% visual token 压缩 + 三阶段渐进训练，图像/文档检索 SOTA | - |
| **C&C (Compress & Cache)** | NeurIPS 2025 | **近无损**压缩：LLM 双前向 bottleneck 生成 summary tokens，解耦压缩与推理，生成任务 2x 压缩率 SOTA，检索任务也 SOTA | [OpenReview](https://openreview.net/forum?id=nGEq3D6FFX) |

### 1.3 LLM Prefilling 阶段

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **HiDivDrop** | ICLR 2026 | **Late Injection** 跳过被动浅层 + **凹金字塔剪枝** (Concave Pyramid) + Early Exit + 可微 top-k，~90% 压缩，训练 1.72x 加速 | [OpenReview](https://openreview.net/forum?id=2baJBgfr9S) |
| **VisionSelector** | [2510.16598](https://arxiv.org/abs/2510.16598) | 端到端可学习 scorer (12.85M params)，可微 Top-K + 课程退火，30% 保留率 100% MME 精度，10% 保留率超 prior 12.14% | - |
| **SwiftVLM** | [2602.03134](https://arxiv.org/abs/2602.03134) | **Bypass 范式**：未选中 token 走旁路到后续层重新评估（非不可逆丢弃），跨层独立剪枝，动态规划选择最优剪枝层 | - |
| **DyVTE** | [2411.19628](https://arxiv.org/abs/2411.19628) / NeurIPS 2025 | **动态视觉 token 退出**：轻量超网络评估文本 token 状态，自适应移除所有视觉 token（一次性退出而非逐 token），与 token-wise 压缩正交互补 | - |
| **EPIC** | NeurIPS 2025 | **渐进一致性蒸馏** 解决压缩训练困难：token-wise + layer-wise 双维度蒸馏，平滑学习轨迹 | [ZichenWen1/EPIC](https://github.com/ZichenWen1/EPIC) |
| **VisionTrim** | [2601.22674](https://arxiv.org/abs/2601.22674) / ICLR 2026 | Training-free 即插即用：DVTS (Dominant Vision Token Selection) 全局-局部剪枝 + TGVC (Text-Guided Vision Complement) 文本对齐 merging | - |

### 1.4 KV Cache 阶段

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **AirCache** | ICCV 2025 | **跨模态相关性**：self-attention 找文本 elite tokens → cross-attention 计算视觉 token 重要性 + 固定点消除 + 自适应层级预算，10% KV 近无损 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_AirCache_Activating_Inter-modal_Relevancy_KV_Cache_Compression_for_Efficient_Large_ICCV_2025_paper.html) |
| **MixKV** | [2510.20707](https://arxiv.org/abs/2510.20707) | **重要性 + 多样性混合**：per-head 自适应混合比例（冗余 head 偏多样性，非冗余偏重要性），极端压缩 +5.1%，GUI +8-9% | - |
| **VL-Cache** | [2412.04652](https://arxiv.org/abs/2412.04652) | **层自适应稀疏感知** budget 分配 + **模态感知** token 评分，10% KV 保持精度，解码 7.08x 加速 | - |
| **LightKV** | ICLR 2026 submission | **Prompt-aware 跨模态消息传递**压缩视觉 token embedding，50% 视觉 token → KV 减半 + 计算 -40% | [OpenReview](https://openreview.net/forum?id=U9OKWwkxuz) |

### 1.5 视频理解专用

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **CaCoVID** | [2602.01649](https://arxiv.org/abs/2602.01649) / AAAI 2026 | **强化学习**策略网络，直接优化 token 对正确预测的贡献（非 attention score），组合策略优化减少探索空间 | - |
| **MARC** | [2510.07915](https://arxiv.org/abs/2510.07915) / ICLR 2026 | **Retrieve-then-Compress**：Visual Memory Retriever 选关键片段 + C-GRPO 蒸馏推理，95% token 压缩，72% GPU 内存降低 | [Gimlettt/MARC](https://github.com/Gimlettt/MARC) |
| **EntropySelect** | ICLR 2026 submission | Training-free，**局部邻域熵** 排序 + 梯度显著性融合 + 网格配额，35-50% 保留即超全 token 性能（"压缩增强"现象） | [OpenReview](https://openreview.net/forum?id=1fvPaaFuy2) |
| **Dynamic-VLM** | [2412.09530](https://arxiv.org/abs/2412.09530) / ICCV 2025 | 动态视觉 token 压缩器适配不同长度视频 + 合成数据集，VideoMME +2.7%，MuirBench +10.7% | [Hon-Wong/ByteVideoLLM](https://github.com/hon-wong/bytevideollm) |

### 1.6 新应用领域

| 方法 | arXiv / 会议 | 核心创新 | GitHub |
|------|-------------|----------|--------|
| **Compressor-VLA** | [2511.18950](https://arxiv.org/abs/2511.18950) | **机器人 VLA 专用**：Semantic Task Compressor (cross-attn) + Spatial Refinement Compressor (local attn)，指令引导，59% FLOPs 降低，3x token 压缩 | - |
| **FastDriveVLA** | [2507.23318](https://arxiv.org/abs/2507.23318) / AAAI 2026 | **自动驾驶 VLA 专用**：ReconPruner (MAE-style 重建 + 对抗前景/背景策略)，75% 剪枝 7.5x 计算降低 | - |
| **FocusUI** | [2601.03928](https://arxiv.org/abs/2601.03928) | **UI Grounding 专用**：指令条件显著性 + UI-graph 评分 + PosPad 位置连续性保持，30% token 保留仅 -3.2% | [showlab/FocusUI](https://github.com/showlab/FocusUI) |

### 1.7 安全性分析

| 方法 | arXiv | 核心发现 | 备注 |
|------|-------|----------|------|
| **CAA (Compression-Aware Attack)** | [2601.12042](https://arxiv.org/abs/2601.12042) | 压缩导致 token 重要性排序不稳定 → 小扰动翻转排序 → 黑盒 Transfer CAA 攻击 | 用户已有 CAGE (2601.21531)，此为互补视角 |

---

## 2. 新兴趋势和方向

### 🔥 趋势 1：Bypass / 非不可逆剪枝
- **SwiftVLM** 的 bypass 范式：被剪枝 token 不丢弃，走旁路到后续层重新评估
- **DyVTE** 的整体退出：不是逐 token 剪枝，而是在某层一次性移除所有视觉 token
- **启示**：传统 "剪了就没了" 的范式正在被挑战

### 🔥 趋势 2：VE 内部中间层压缩
- **LaCo** 首次在 VE 中间层做 pixel-shuffle 压缩（非 VE 后端）
- 比 post-encoder 方法训练效率高 20%+
- **启示**：压缩位置的粒度进一步细化

### 🔥 趋势 3：RL/学习驱动的视频 token 选择
- **CaCoVID** (RL 策略网络) 和 **MARC** (C-GRPO) 都用强化学习优化 token 选择
- 直接优化"对正确预测的贡献"而非 proxy metric (attention score)
- **启示**：端到端优化 > 启发式规则

### 🔥 趋势 4：Domain-Specific 压缩
- **自动驾驶**: FastDriveVLA (前景/背景分离)
- **机器人操作**: Compressor-VLA (语义+空间双压缩器), TEAM-VLA
- **UI 理解**: FocusUI (位置连续性保持)
- **启示**：通用压缩 → 领域专用压缩

### 🔥 趋势 5：Late Injection / 跳过浅层
- **HiDivDrop** 发现浅层对视觉 token 是"被动"的，直接跳过注入到 active fusion 层
- 与 V-Skip、Skip-Vision 等层跳过工作形成互补
- **启示**：不是所有层都需要视觉 token

### 🔥 趋势 6："压缩增强"现象
- **EntropySelect** 在 35-50% 保留时超过全 token 性能
- 说明冗余 token 不仅浪费计算，还可能**干扰**推理
- **启示**：支持 "less is more" 假设

### 🔥 趋势 7：KV Cache 跨模态感知
- **AirCache** (跨模态相关性), **VL-Cache** (模态感知评分), **LightKV** (跨模态消息传递), **MixKV** (多样性+重要性)
- 与纯 LLM KV 压缩方法区别：**视觉 token 和文本 token 需要区别对待**

---

## 3. 与 STAR-Pro 的潜在关联

| 关联点 | 相关论文 | 关系 |
|--------|----------|------|
| **Attention bias / inconsistency** | SwiftVLM (非单调层间差异), FEATHER (位置偏差), HoloV | 支持 STAR-Pro 的 inconsistency 发现 |
| **渐进式压缩** | HiDivDrop (凹金字塔), PyramidDrop | STAR-Pro Progressive stage 的 baseline/对比 |
| **Text-guided 压缩** | VisionTrim (TGVC), FSR (指令相关性), Nüwa | STAR-Pro text guidance 机制的对比 |
| **Training-free** | FSR, PIO-FVLM, SwiftVLM, VisionTrim, EntropySelect | 如果 STAR-Pro 也是 training-free，这些都是直接竞争者 |
| **"压缩增强"** | EntropySelect | 可以用来论证 STAR-Pro 的 motivation |
| **安全性** | CAA (2601.12042), CAGE (2601.21531) | STAR-Pro 可以讨论压缩鲁棒性 |
| **Bypass 范式** | SwiftVLM | 如果 STAR-Pro 涉及 token 恢复/重用，SwiftVLM 是相关工作 |
| **视频扩展** | CaCoVID, MARC, EntropySelect | STAR-Pro 如果扩展到视频 |

---

## 4. 推荐阅读优先级

### ⭐⭐⭐ 必读 (直接竞争者 / 新范式)

1. **HiDivDrop** (ICLR 2026) — Late injection + 凹金字塔，新 SOTA，直接对比目标
2. **SwiftVLM** (2602.03134) — Bypass 范式，挑战传统不可逆剪枝
3. **LaCo** (ICLR 2026) — VE 内部中间层压缩，新压缩位置
4. **FSR** (2602.05809) — Training-free 三阶段 SOTA，直接竞争者
5. **DyVTE** (NeurIPS 2025) — 整体退出范式，与 token-wise 压缩正交

### ⭐⭐ 重要 (新方向 / 强 baseline)

6. **VisionSelector** (2510.16598) — 可微 Top-K 学习，10% 保留新 SOTA
7. **VisionTrim** (ICLR 2026) — Training-free DVTS+TGVC，图像+视频通用
8. **CaCoVID** (AAAI 2026) — RL 驱动 video token 选择
9. **MARC** (ICLR 2026) — Retrieve-then-compress 视频理解
10. **AirCache** (ICCV 2025) — 跨模态 KV Cache 压缩 SOTA
11. **MixKV** (2510.20707) — 重要性+多样性混合 KV 压缩
12. **EntropySelect** (ICLR 2026 sub) — 局部熵，"压缩增强"现象

### ⭐ 参考 (特定领域 / 补充视角)

13. **EPIC** (NeurIPS 2025) — 渐进一致性蒸馏解决训练问题
14. **C&C** (NeurIPS 2025) — 近无损 summary token 压缩
15. **Magic-MM-Embedding** (2602.05275) — 无参数插值检索
16. **PIO-FVLM** (2602.04657) — 两阶段 training-free
17. **FastDriveVLA** (AAAI 2026) — 自动驾驶专用
18. **Compressor-VLA** (2511.18950) — 机器人操作专用
19. **FocusUI** (2601.03928) — UI 理解专用
20. **VL-Cache** / **LightKV** — KV Cache 多模态感知
21. **Dynamic-VLM** (ICCV 2025) — 视频动态压缩
22. **CAA** (2601.12042) — 压缩安全性分析
23. **DualSpeed** 的互补视角 — EPIC 解决类似的训练问题

---

## 5. 快速参考：完整 arXiv ID 列表

```
2602.05275  Magic-MM-Embedding
2510.16598  VisionSelector  
2601.22674  VisionTrim (ICLR 2026)
2601.12042  CAA (Security Pitfalls)
N/A         HiDivDrop (ICLR 2026, OpenReview: 2baJBgfr9S)
2602.05809  FSR (Focus-Scan-Refine)
2602.03134  SwiftVLM
2602.01649  CaCoVID (AAAI 2026)
2510.07915  MARC (ICLR 2026)
2507.02279  LaCo (ICLR 2026)
2411.19628  DyVTE (NeurIPS 2025)
N/A         EPIC (NeurIPS 2025, OpenReview: gZjPllL9jM)
N/A         C&C (NeurIPS 2025, OpenReview: nGEq3D6FFX)
N/A         AirCache (ICCV 2025)
2510.20707  MixKV
2412.04652  VL-Cache
N/A         LightKV (ICLR 2026 sub, OpenReview: U9OKWwkxuz)
2511.18950  Compressor-VLA
2507.23318  FastDriveVLA (AAAI 2026)
2601.03928  FocusUI
N/A         EntropySelect (ICLR 2026 sub, OpenReview: 1fvPaaFuy2)
2602.04657  PIO-FVLM
2412.09530  Dynamic-VLM (ICCV 2025)
```

---

*Generated by 1号机 deep research subagent, 2026-02-14*
