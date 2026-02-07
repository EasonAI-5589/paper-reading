# 5. Conclusion & Appendix

> 来源: PyramidDrop (CVPR 2025)

---

## 📄 原文

We introduce PyramidDrop, a simple yet effective strategy to reduce visual token redundancy in LVLMs, for boosting efficiency without performance loss. PyramidDrop helps to reduce the redundancy and concentrate more on valuable visual information for efficient deployment in realistic world. Our empirical study reveals that all visual tokens are necessary in the shallow layers of LVLMs, and token redundancy progressively increases in deeper layers. Experiments demonstrate that PyramidDrop can achieve up to 1.82× and 2.22× acceleration for training and inference respectively.

> 💡 **批注**: 最终数字：训练 1.82× 加速（≈45%减少），推理 2.22× 加速（≈55%减少）。

---

## 💡 全文总评

### 论文优点
1. **Observation 扎实**: 逐层冗余递增的发现有充分的实验支撑（Figure 1），直觉上也说得通
2. **方法极简**: 零额外参数，复用 QK 矩阵，兼容 FlashAttention，工程实现友好
3. **实验充分**: 16 个 benchmark、2 个模型、训练+推理、图像+视频，覆盖全面
4. **即插即用**: 推理时不需要重训就能用，实用性强

### 论文局限
1. **只在 7B 模型验证**: 没有 13B、70B 的实验，不确定规模效应
2. **Stage 均分**: 固定均分 LLM 层，可能不是最优划分（不同层的冗余增速不同）
3. **仅用 last instruction token**: 多轮对话场景下，instruction 变化可能需要重新评估 token 重要性
4. **Video 实验浅**: 视频任务上所有方法差异不大，没有充分展示 PyramidDrop 的独特优势

### 核心贡献排序
```
1. 🥇 逐层视觉冗余递增的 empirical finding
2. 🥈 渐进式金字塔 token 丢弃策略
3. 🥉 训练+推理双场景适用的统一框架
```

### 与 MLLM Token Compression 领域的关系
```
PyramidDrop 在 token 压缩方法谱系中的位置:
├── 压缩位置: LLM 内部（多阶段）
├── 压缩策略: 渐进式丢弃（vs 一次性丢弃）
├── 重要性度量: Attention-based（instruction-guided）
└── 适用场景: 训练 + 推理（vs 仅推理）
```

### 可以借鉴的点
- **渐进式思想**: 不要一刀切，根据模型的"理解进度"动态调整压缩强度
- **复用已有计算**: QK 矩阵复用是很聪明的设计，避免了额外开销
- **训练即压缩**: 在训练中使用压缩可以让模型学会更紧凑的表示
