[← 返回 README](../README.md)

# 6 Discussion and Future Work

## 📌 预览

简短的 Discussion，点出局限性和未来方向。

---

Real-time chunking is an inference-time algorithm for asynchronous execution of action chunking policies that demonstrates speed and performance across simulation and real-world experiments, including under significant inference delays. However, this work is not without limitations: it adds significant computational overhead compared to methods that sample directly from the base policy, and it is applicable only to diffusion- and flow-based policies. Additionally, while our real-world experiments cover a variety of challenging manipulation tasks, there are more dynamic settings that could benefit even more from real-time execution. One example is legged locomotion, which is represented in our simulated benchmark but not our real-world results.

> 💡 **局限性分析**:
> 1. **计算开销**：+28% 延迟（需要每步去噪的反向传播），虽然比 BID 少，但不是零成本
> 2. **只适用于 diffusion/flow 策略**：不能用于 autoregressive（如 RT-2）、VQ（如 VQBET）、BPE（如 FAST）策略
> 3. **真实世界缺少高动态任务**：只测了操作任务，没测腿足运动（模拟中有，但真实中没有）
> 
> **未来方向**（隐含的）：
> - 扩展到非 diffusion/flow 策略（需要新的 inpainting 方法）
> - 腿足机器人上的 RTC
> - 与 System 1/2 架构结合
> - 与推理加速方法（consistency, quantization）叠加

---

## Appendix 亮点

### A.2 Guidance Weight Clipping ($\beta$)

> 💡 **$\beta$ ablation**:
> - $\beta \geq 5$ 后无明显提升
> - 过高的 $\beta$ 导致 action chunk 发散（少步去噪时尤其严重）
> - $n=5$ 时 $\beta=5$ 最优，$\beta=150$ 时加速度异常高（out-of-distribution actions）

### A.3 Latency Measurements

> 💡 **延迟对比表**:
> 
> | 方法 | 延迟 |
> |------|------|
> | Vanilla π₀.₅ | 76ms |
> | **RTC** | **97ms** (+28%) |
> | BID (no forward, N=16) | 115ms |
> | BID (shared backbone, N=16) | 169ms |
> | BID (full, N=16) | 223ms (2.3x RTC) |
> 
> RTC 延迟增加来自：每步去噪增加 VJP → 35ms vs 14ms per step (2.5x)，但固定开销（image encoder, LLM prefill）不变。

### A.5 Hyperparameters

> 💡 **超参数总结**:
> 
> | 参数 | 模拟 | 真实 |
> |------|------|------|
> | $n$（去噪步数） | 5 | 5 |
> | $H$（prediction horizon） | 8 | 50 |
> | $s_\text{min}$（最小 execution horizon） | — | 25 |
> | $\beta$（guidance clip） | 5 | 5 |
> | $b$（delay buffer size） | — | 10 |

---

## 🔖 Section 总结

### 核心洞察
1. RTC 是**推理时方法**：零训练成本，但有推理成本（+28%）
2. 只适用于 diffusion/flow 策略——这是最大的限制
3. 与推理加速、System 1/2 等方法**正交**，可以叠加
4. 腿足运动等高动态场景是未来的重要方向
