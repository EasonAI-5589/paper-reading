[← 返回 README](../README.md)

# 0. Abstract

## 📌 预览

WAM 的 imagine-then-execute 范式带来严重推理延迟，但显式未来想象是否真的必要？Fast-WAM 解耦训练时视频建模与推理时未来生成，发现：训练时视频建模是 WAM 的核心价值来源，推理时未来想象并非必需。190ms 实时推理，比现有 WAM 快 4 倍以上。

---

## 📄 原文

> World Action Models (WAMs) have emerged as a promising alternative to Vision-Language-Action (VLA) models for embodied control because they explicitly model how visual observations may evolve under action. Most existing WAMs follow an imagine-then-execute paradigm, incurring substantial test-time latency from iterative video denoising, yet it remains unclear whether explicit future imagination is actually necessary for strong action performance.

> 💡 **开门见山指出 WAM 的核心矛盾**: imagine-then-execute 范式直觉上合理，但带来了巨大的推理延迟。更重要的是——**没人验证过显式想象未来是否真的必要**。

> In this paper, we ask whether WAMs need explicit future imagination at test time, or whether their benefit comes primarily from video modeling during training. We disentangle the role of video modeling during training from explicit future generation during inference by proposing Fast-WAM, a WAM architecture that retains video co-training during training but skips future prediction at test time.

> 💡 **核心问题一句话**: 训练时的视频建模 vs 推理时的未来生成，哪个才是 WAM 的"真功夫"？Fast-WAM 的设计直接回答这个问题。

> Fast-WAM achieves competitive results with state-of-the-art methods both on simulation benchmarks (LIBERO and RoboTwin) and real-world tasks, without embodied pretraining. It runs in real time with 190 ms latency, over 4× faster than existing imagine-then-execute WAMs. These results suggest that the main value of video prediction in WAMs may lie in improving world representations during training rather than generating future observations at test time.

> 💡 **关键数据点**: 190ms 时延（vs 810ms for IDM 变体），4× 加速。无 embodied pretraining 就能达到 SOTA 水平。结论明确但措辞谨慎（"may lie"）——这是好的学术表述。
