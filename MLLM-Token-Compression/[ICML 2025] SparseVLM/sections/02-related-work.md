# 2. Related Work

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

> 💡 **Section 概览**: 两个子方向 — VLM 架构发展 + Visual token 压缩方法

---

### Vision-Language Models

高分辨率 → token 爆炸：

| 模型 | 分辨率 | Visual Tokens |
|------|--------|---------------|
| LLaVA | 336×336 | 576 |
| LLaVA-1.5 | 672×672 | 2880 |
| Mini-Gemini-HD | 1536×1536 + 672×672 | 2880 |
| VideoLLaVA | 多帧 | 数千 |

> 💡 **批注**: 分辨率越高，token 越多，推理越慢。视频更恐怖，每帧都有几百个 token，叠加起来就是几千。

---

### Visual Compression for VLMs

**两大方向**:

```
视觉 token 压缩
├── 方向 1: 修改 vision tower / projector
│   ├── LLaMA-VID: Q-Former + context token
│   └── DeCo: adaptive pooling 下采样
│
└── 方向 2: LLM 解码阶段稀疏化
    ├── FastV: attention score 低的直接剪 ← text-agnostic
    ├── VoCo-LLaMA: 训练 pruning 网络
    └── SparseVLM (本文): text-guided, training-free ← 改进
```

> 💡 **批注**: 方向 2 的现有方法都缺乏 text guidance。SparseVLM 补上了这个空缺。注意 SparseVLM 和 FastV 都是方向 2（解码阶段），但 SparseVLM 不需要训练。

---

## 💡 Section 总结

### 核心洞察
1. 视觉 token 数量是 LLM 推理的瓶颈
2. 压缩可以在 encoder 端或 decoder 端做，本文选择 decoder 端
3. 关键创新点：**text-guided + training-free** 的组合在之前没人做过
