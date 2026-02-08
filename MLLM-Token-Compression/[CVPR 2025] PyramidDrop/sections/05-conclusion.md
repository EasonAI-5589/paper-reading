[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 PyramidDrop 的核心贡献和关键数字。

---

We introduce PyramidDrop, a simple yet effective strategy to reduce visual token redundancy in LVLMs, for boosting efficiency without performance loss. PyramidDrop helps to reduce the redundancy and concentrate more on valuable visual information for efficient deployment in realistic world. Our empirical study reveals that all visual tokens are necessary in the shallow layers of LVLMs, and token redundancy progressively increases in deeper layers. Experiments demonstrate that PyramidDrop can achieve up to 1.82× and 2.22× acceleration for training and inference respectively.

> 💡 **Conclusion 批读**:
> - **核心发现**：浅层需要所有 token，深层冗余递增
> - **方法**：PyramidDrop — 渐进式视觉 token 丢弃
> - **最终加速比**：训练 1.82×，推理 2.22×
> - 方法简单有效，可 plug-and-play，适用于图像和视频 LVLM
