[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 + 局限性讨论（在线处理的时间代价 + 未来改进方向）。

---

In this paper, we introduce a long-term memory bank designed to augment current large multimodal models, equipping them with the capabilities to effectively and efficiently model long-term video sequences. Our approach processes video frames sequentially and stores historical data in the memory bank, addressing LLMs' context length limitation and GPU memory constraints posed by the long video inputs. Our long-term memory bank is a plug-and-play module that can be easily integrated into existing large multimodal models in an off-the-shelf manner. Experiments on various tasks have demonstrated the superior advantages of our method. We believe our MA-LMM offers valuable insights for future research in the long-term video understanding area.

> 💡 **批注**: 简洁的总结。核心卖点再次强调：(1) plug-and-play (2) 解决 context length + GPU memory (3) 多任务 SOTA。

---

Acknowledgements. This project was partially funded by NSF CAREER Award (#2238769) to AS.

---

## 🔖 Section 总结

### 核心洞察
1. MA-LMM 的三大优势：memory bank 设计、在线处理效率、plug-and-play 通用性
2. 局限性（见 Appendix E）：在线逐帧处理 **增加了处理时间**（虽然省了 GPU memory）
3. 未来方向：video encoder 替代 image encoder、大规模 video-text 预训练、更强的 LLM
