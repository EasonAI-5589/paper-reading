# 6. Conclusion

> 来源: VisionZip (CVPR 2025)

---

## 📄 原文

In this paper, we analyze popular VLM models, noting that while increasing the length of visual tokens can improve performance, there is significant redundancy in current visual tokens. We propose a simple method, VisionZip, which reduces the number of visual tokens substantially while preserving model performance, thereby greatly enhancing computational efficiency. This method is broadly applicable to image and video understanding tasks and is suitable for multi-turn dialogue in practical applications. VisionZip also suggests a future direction to develop vision encoders with lower redundancy capabilities to further improve VLM performance and handle longer video sequences.

> 💡 **批注**: 论文最后一句指出了未来方向——与其在 LLM 端做 token compression 的后处理，不如从根本上改进 vision encoder，让它输出更少但更好的 token。这是一个很有价值的 insight。

---

## 💡 全文总结

### VisionZip 一句话总结
**在 vision encoder 端用 [CLS] attention 选 dominant tokens + similarity-based merging 生成 contextual tokens，实现 text-agnostic 的视觉 token 压缩，性能和效率全面优于 text-aware 方法。**

### 核心贡献
1. **发现**: Vision encoder 输出的视觉 token 存在严重冗余（Softmax 马太效应导致信息集中到少数 proxy token）
2. **方法**: VisionZip = Dominant token selection + Contextual token merging（简单、有效、text-agnostic）
3. **洞察**: Text-relevant 方法因 feature misalignment 而选错 token（语义位置 ≠ 信息位置）
4. **效果**: 10% token 保留 95% 性能，prefilling 加速 8×，13B 比 7B 更快更好

### 方法优势 vs 局限
| 优势 | 局限 |
|------|------|
| Text-agnostic → 多轮对话友好 | 依赖 vision encoder 的 attention 分布规律 |
| 在 encoder 端压缩 → 效率最高 | 如果 vision encoder 不存在 attention 集中现象则可能失效 |
| 兼容量化、LLM 加速技术 | Token merging 的效果取决于 Key 相似度的质量 |
| 30min 微调即可进一步提升 | 极端压缩（如 <32 tokens）性能可能快速下降 |

### 对 Eason 研究的启示
1. **Token compression 方向**: VisionZip 证明了 text-agnostic 路线的可行性和优越性
2. **Vision encoder 分析**: Attention sink / proxy token 现象值得深入研究
3. **实际部署**: VisionZip + 量化 是一个非常实用的组合
4. **与 STAR-Pro 的关系**: 如果 STAR-Pro 涉及视觉 token 处理，VisionZip 的思路可以借鉴
5. **未来方向**: 设计低冗余 vision encoder 可能比后处理压缩更根本
