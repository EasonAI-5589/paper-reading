# 2. Preliminaries

## 2.1 MLLM架构

现代MLLM采用**三组件架构**：

```
输入图像/视频 → [Vision Encoder] → [Projector] → [LLM] → 文本输出
                                        ↑
                              文本指令 → [Tokenizer] ──┘
```

### 三个核心组件

#### (1) Vision Encoder $\mathcal{E}_v$

- 常基于 SigLIP 或 CLIP
- 将视觉输入转换为dense visual token序列：
  $$\mathbf{Z}^v = \mathcal{E}_v(\mathcal{X}^v) \in \mathbb{R}^{n_v \times d_v}$$
- $n_v$: 视觉token数量，$d_v$: 特征维度

#### (2) Projector $\mathcal{P}$

- 桥接视觉与语言模态
- 将视觉特征从 $d_v$ 维映射到LLM的嵌入空间 $d_l$ 维：
  $$\mathbf{H}^v = \mathcal{P}(\mathbf{Z}^v) \in \mathbb{R}^{n_v \times d_l}$$

#### (3) LLM $\mathcal{G}$

- 处理拼接后的视觉+文本token序列：
  $$\mathbf{Y} = \mathcal{G}([\mathbf{H}^v; \mathcal{E}_t(\mathcal{X}^t)])$$

### 计算复杂度

对于序列长度 $n$、隐藏维度 $d$、FFN中间维度 $m$ 的单层Transformer：
$$\text{Layer FLOPs} = 4nd^2 + 2n^2d + 2ndm$$

$L$ 层的总FLOPs:
$$\text{Total FLOPs} = L \times (4nd^2 + 2n^2d + 2ndm)$$

其中 $n = n_t + n_v$（文本token + 视觉token）。

**关键瓶颈**: $2n^2d$ 项（注意力机制的二次复杂度）在 $n_v$ 很大时主导计算开销。

## 2.2 Token压缩

### 形式化定义

- 原始token总数: $N = n_t + n_v$
- 压缩后token数: $M < N$
- 压缩函数: $\mathbf{H}_{\text{comp}} = \mathcal{C}(\mathbf{H}) \in \mathbb{R}^{M \times d_l}$

### 压缩率

$$R_{\text{comp}} = \frac{N}{M}$$

- $4\times$, $8\times$ 等表示压缩级别
- 越高 = 更紧凑 + 更高效 + 可能更多信息损失

### 两种核心冗余来源

| 冗余类型 | 描述 | 利用方式 |
|---------|------|--------|
| **Intra-Visual** (视觉内部) | 背景重复、相邻patch相似、帧间冗余 | 空间/时序聚合 |
| **Cross-Modal** (跨模态) | 与文本query无关的视觉token | 文本引导筛选 |

### 关键观察

- 视觉token数量通常远超文本token（可达20倍以上）
- 因此大多数压缩方法**主要针对减少 $n_v$**

---

## 个人笔记

<!-- 在此添加对Preliminaries的理解和思考 -->

