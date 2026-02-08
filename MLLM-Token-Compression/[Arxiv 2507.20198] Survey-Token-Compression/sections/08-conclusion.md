[← 返回 README](../README.md)

# 8 Conclusion

## 📌 预览
总结全文贡献，指出核心挑战和未来方向。

---

This paper presents the first structured survey of token compression techniques for Multimodal Large Language Models (MLLMs), establishing a taxonomy based on modality-specific redundancy and underlying compression mechanisms. While current methods demonstrate promising efficiency gains, several critical challenges remain on the path toward scalable and robust MLLMs. Future research must move beyond simple redundancy reduction to address the preservation of cross-modal alignment under high compression ratios and the maintenance of causal reasoning capabilities in temporal sequences. Furthermore, the field necessitates the development of specialized benchmarks designed to rigorously evaluate multi-frame comprehension and long-term context retention. We hope this survey serves as a roadmap, guiding the community to tackle these open problems and push the boundaries of processing increasingly complex multimodal data.

> 💡 **结论要点**:
> 1. 首篇 MLLM token 压缩系统综述，建立了模态×机制的 taxonomy
> 2. 核心挑战：高压缩率下保持跨模态对齐 + 时间序列的因果推理能力
> 3. 需要专用 benchmark 评估多帧理解和长上下文保持
> 4. 定位：领域 roadmap，引导社区解决 open problems

---

# 9 Acknowledgment

This paper is supported by Young Scientists Fund of the National Natural Science Foundation of China (NSFC) (No. 62506305), Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (No. 2024R01007), Key Research and Development Program of Zhejiang Province (No. 2025C01026), Scientific Research Project of Westlake University (No. WU2025WF003), Chinese Association for Artificial Intelligence (CAAI) & Ant Group Research Fund - AGI Track (No. 2025CAAI-ANT-13). It is also supported by the research funds of National Talent Program and Hangzhou Municipal Talent Program.

> 💡 **资助信息**: 国自然青年基金 + 浙江省 + 西湖大学 + CAAI-蚂蚁 AGI 基金。通讯作者 Huan Wang 在西湖大学。
