[← 返回 README](../README.md)

# 7 Conclusion

## 📌 预览
结论部分回收全文主张：A2C2 是一个轻量、可插拔、与 base VLA 正交的 correction mechanism。作者最后真正想落下来的，不只是“这个方法在 benchmark 上有效”，而是“未来更大的 VLA 很可能需要显式的 step-level correction 才能现实部署”。 

---

In this paper, we propose Asynchronous Action Chunk Correction (A2C2), which introduces a lightweight action correction head by augmenting a large base policy, such as VLAs. A2C2 addresses the challenge of preserving reactivity under inference delays and long execution horizons of action chunking policies. The correction head is trained on the same dataset as the base policy and, in principle, can be added to any off-the-shelf VLAs. Our experiments in both the Kinetix simulation suite and the LIBERO Spatial benchmark demonstrated that Asynchronous Action Chunk Correction (A2C2) consistently maintained high success rates, even in settings where naive or RTC degraded significantly.

While our approach adds minimal overhead compared to full model inference, further work is needed to validate its scalability to richer language instructions, out-of-distribution settings, and more dynamic tasks beyond those in LIBERO Spatial. Addressing these challenges would broaden the applicability of action chunk correction and strengthen its role as a general mechanism for enhancing reactivity in large policy architectures.

> 💡 **定位与局限性**:
> - 作者先把最核心的 claim 再收一遍：A2C2 解决的是 action chunking policy 在 delay 和 long horizon 下的反应性损失
> - 它的工程定位依然很克制：轻量、可插拔、沿用 base policy 数据、不要求重训大模型
> - 这里其实是在强调，A2C2 最有吸引力的地方不是“提出了一个新大模型”，而是“用一个很小的附加模块解决了一个会越来越普遍的问题”
> - richer language、OOD、以及比 LIBERO 更动态的任务，仍然需要进一步验证

Recently, Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated improved generality through parameter scaling, as established by neural scaling laws (Kaplan et al., 2020). Since recent VLA policies are built upon these models, it is reasonable to expect that future VLAs will continue to grow in size to support deployment across diverse environments and tasks. Our work can be viewed as a step toward enabling such scaled VLAs to operate in real time without sacrificing responsiveness by introducing a lightweight correction mechanism that mitigates the effects of inference latency.

Moreover, inference of models with billions of parameters already exceeds the computational capacity of on-board processors on most robotic platforms. In practice, this motivates client-server architectures where the VLA runs on a remote server and the robot queries it over a network. In this setting, by explicitly treating communication delay as part of the inference latency in our formulation, our framework naturally extends to client-server architectures where large VLAs are executed remotely. Thus, our framework provides a pathway to leverage the generalization benefits of large-scale VLAs while still maintaining reactivity in real-world deployments, enabling the design of next-generation VLA systems that combine scalability with responsiveness.

> 💡 **系统层面的真正落点**:
> - 作者把 scaling law、远程推理、client-server 架构放进结论，不是为了补背景，而是在说明这个问题未来只会更严重
> - 如果大模型继续变大、越来越依赖远程 GPU 推理，那么“每个 control step 都重跑大模型”只会越来越不现实
> - 在这个意义上，A2C2 想提示的是一种更长期的系统范式：**chunk 级规划 + step 级校正**

## 🔖 Section 总结

### 核心洞察
1. **短期价值很明确**: A2C2 是一个对现有 chunking policy 很实用的 plug-in 补丁。
2. **作者承认边界仍然存在**: richer language、OOD 和更动态任务还没有被充分覆盖。
3. **长期判断更重要**: 未来大模型机器人控制很可能需要显式的双时间尺度结构，也就是 chunk 级规划配合 step 级纠偏。
