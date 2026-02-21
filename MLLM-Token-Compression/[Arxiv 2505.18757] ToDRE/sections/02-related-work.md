[← 返回 README](../README.md)

# 2 Related Work

## 2.1 Large Vision-Language Models

LVLMs employ a vision encoder to extract visual features, which are subsequently projected into the LLM's embedding space via a visual projector (e.g., Q-Former or MLP). To process real-world high-resolution images, subsequent studies adopt **dynamic tiling**, which partitions images into regions and encodes each region independently using a shared vision encoder. However, dynamic tiling can yield thousands of visual tokens, significantly increasing computational overhead. This issue becomes even more pressing in video-based LVLMs.

> 💡 **批注**：Dynamic tiling (AnyRes) 是 visual token 爆炸的主要原因之一。LLaVA-NeXT 的 2880 tokens 就是这样来的。

## 2.2 Token Compression for LVLMs

Several earliest attempts modify model components and introduce additional training costs. More recently, training-free token compression methods have been widely adopted. These methods can be categorized into two main groups:

**(1) Token compression location:**
- **Vision encoder**: ToMe — merges redundant tokens via bipartite soft matching
- **LLM decoder**: FastV, SparseVLM — attention-based; VTW, FitPrune — output divergence-based
- **Both**: FiCoCo, MustDrop, FocusLLaVA

**(2) Token compression in LLM embedding space:**
- FasterVLM — [CLS]-to-visual cross-attention
- DivPrune — diversity-based pruning

> 💡 **批注**：ToDRE 的定位是"LLM embedding space + LLM decoder"双阶段，与 FiCoCo 类似的双阶段思路，但具体方法完全不同。DivPrune 是最接近的 baseline——同样使用 diversity，但 ToDRE 加了 Stage 2 的 relevance pruning。
