[← 返回 README](../README.md)

# 6 Discussion and Future Work

## 📌 预览
讨论局限性和未来方向。

---

Real-time chunking is an inference-time algorithm for asynchronous execution of action chunking policies that demonstrates speed and performance across simulation and real-world experiments, including under significant inference delays. However, this work is not without limitations: it adds significant computational overhead compared to methods that sample directly from the base policy, and it is applicable only to diffusion- and flow-based policies. Additionally, while our real-world experiments cover a variety of challenging manipulation tasks, there are more dynamic settings that could benefit even more from real-time execution. One example is legged locomotion, which is represented in our simulated benchmark but not our real-world results.

> 💡 **局限性分析**:
> 1. **计算开销**: 每步 denoising 需要反向传播，延迟增加 ~28%（76ms → 97ms）。虽然是异步的不影响控制频率，但 GPU 占用更高
> 2. **只适用于 diffusion/flow-based**: 对 autoregressive VLA（如 RT-2 [8]、OpenVLA [30] 的 token 预测部分）不适用。不过可以通过 FAST [47] 等方法将 autoregressive 转换为 flow-based
> 3. **真实实验缺少腿足运动**: 仿真中有但真实世界没做。腿足运动对实时性要求更高（摔倒不可逆），是最好的应用场景之一
> 
> **我的补充**:
> - 没有讨论 **多模态 observation 延迟**（如相机帧延迟、状态估计延迟），这在真实部署中是个问题
> - 没有分析 **不同 VLA 架构**的适用性差异（如 Transformer vs MLP-Mixer 的 Jacobian 计算效率）
> - **跟 System 1/2 的结合**是最有前景的方向——在 System 2（大模型）上用 RTC 减少高层延迟

---

## 🔖 Section 总结

### 未来方向
1. 扩展到**腿足运动**的真实世界实验
2. 降低计算开销（如近似 Jacobian、更少的 guidance steps）
3. 与 System 1/2 架构结合
4. 适配 autoregressive VLA（需要新的 inpainting 形式）
