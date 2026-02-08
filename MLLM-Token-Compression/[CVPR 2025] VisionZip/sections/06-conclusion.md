[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
总结 VisionZip 的贡献，展望低冗余 vision encoder 的未来方向。

---

In this paper, we analyze popular VLM models, noting that while increasing the length of visual tokens can improve performance, there is significant redundancy in current visual tokens. We propose a simple method, VisionZip, which reduces the number of visual tokens substantially while preserving model performance, thereby greatly enhancing computational efficiency. This method is broadly applicable to image and video understanding tasks and is suitable for multi-turn dialogue in practical applications. VisionZip also suggests a future direction to develop vision encoders with lower redundancy capabilities to further improve VLM performance and handle longer video sequences.

> 💡 **Conclusion 批读**:
> - **核心贡献回顾**: 发现冗余 → 提出 VisionZip → 全面验证
> - **未来方向**: 开发低冗余的 vision encoder——不是在下游压缩，而是从源头减少冗余
> - 这个方向很有价值：如果 vision encoder 本身就输出更紧凑的表示，VLM 的效率问题从根本上解决

---

## 🔖 Section 总结

### 核心洞察
1. VisionZip 是一个"下游补丁"式的方案——在现有 encoder 输出上做压缩
2. 论文指向的更根本方向是设计低冗余 vision encoder
3. 对于长视频理解，VisionZip 可以让模型在同样显存下处理 5-10× 更多帧
