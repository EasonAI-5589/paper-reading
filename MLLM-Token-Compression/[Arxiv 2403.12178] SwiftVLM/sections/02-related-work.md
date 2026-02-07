# 2. Related Work

> 来源: SwiftVLM

---

## 📄 原文

> 💡 **Section 概览**: 相关工作分两大类——Text-agnostic（不看问题直接压缩）和 Text-aware（根据问题选择性压缩）。SwiftVLM 属于后者中的 training-free 阵营。

---

### 2.1 Text-agnostic 方法

> 💡 **要点预览**: 这类方法不看用户问了什么，纯粹基于视觉特征来压缩 token。

- **Qwen2.5-VL**: 每 4 个相邻 token 合并成 1 个
- **ToMe**: 基于相似度的 token merging
- **VisionZip**: 保留高 [CLS]-attention 的 token，其余按相似度合并
- **VoCo-LLAMA**: 把所有视觉信息压缩到一个可学习 token

> 💡 **批注**: 这些方法的共同问题——不看问题就压缩，如果问题问的是图中不显眼的细节（比如背景里的小字），这些方法很可能直接把它压没了。

> 💡 **2.1 小结**:
> - 优点: 简单高效，不依赖文本
> - 缺点: 无法保留 query-relevant 的视觉细节

---

### 2.2 Text-aware 方法

> 💡 **要点预览**: 利用文本信息来指导视觉 token 的选择/压缩。分为需要训练和不需要训练两派。

**需要训练的:**
- **Q-Former (BLIP-2)**: 训练交叉模态模块，压缩到少量可学习 token
- **ATP-LLaVA**: 在 VLM 内部加可训练模块，基于 attention 打分剪枝

> 💡 **批注**: 需要额外训练 = 额外成本，而且换个模型就得重新训。

**Training-free 的 (SwiftVLM 的直接对手):**

| 方法 | 策略 | 问题 |
|------|------|------|
| **FastV** (ECCV'24) | 浅层激进剪枝 | 丢掉后续重要 token |
| **PDrop** (CVPR'25) | 逐层渐进 drop | 仍然是不可逆 drop |
| **FEATHER** (ICCV'25) | 去除 RoPE 影响 + drop | 计算开销大，仍是 drop |
| **SparseVLM** (ICML'25) | 自适应逐层剪枝 | 假设早期 drop 的后面也不重要 |

> 💡 **批注**: 这四个方法的共同假设是"浅层不重要的 token 在深层也不重要"——但 Figure 2 已经证明这个假设是错的！SwiftVLM 通过 bypass 打破了这个假设。

> 💡 **2.2 小结**:
> - Training-free 方法都用 T-V attention 打分
> - 核心缺陷: drop 是不可逆的，一旦丢了就没了
> - SwiftVLM 的定位: training-free + bypass (可逆)

---

## 💡 Section 总结

### 方法谱系
```
Visual Token Reduction
├── Text-agnostic (不看问题)
│   ├── Merge: ToMe, Qwen-VL, VisionZip
│   └── Compress: VoCo-LLAMA
└── Text-aware (看问题)
    ├── 需要训练: Q-Former, ATP-LLaVA
    └── Training-free
        ├── Drop: FastV, PDrop, SparseVLM
        ├── Drop + RoPE fix: FEATHER
        └── Bypass: SwiftVLM ⭐ (本文)
```

### SwiftVLM 的差异化
- 唯一的 bypass 范式（vs 所有对手都是 drop）
- Training-free（vs Q-Former, ATP-LLaVA）
- 用动态规划选层（vs FastV 固定层、PDrop 均匀分布）
