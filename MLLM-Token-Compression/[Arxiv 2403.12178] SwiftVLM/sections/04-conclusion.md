# 5. Conclusion & 评价

> 来源: SwiftVLM (Arxiv 2403.12178)

---

## 💡 我的整体评价

### 优点
1. **Bypass 是真正的创新** — 不是 merge 不是 drop，是新范式
2. **定位任务大幅领先** — 解决了现有方法在 fine-grained 任务上的短板
3. **分析透彻**: 为什么 bypass works，为什么 bypass > drop，都有实证
4. **DP 选层有理论支撑** — 不是拍脑袋选的

### 局限性
1. **只测了推理加速** — 没有 PyramidDrop 的训练加速能力
2. **额外计算**: bypass + alignment + 二次评估 → 比 FastV 慢
3. **选层需要 calibration**: 需要用 6 个 dataset 各 1000 samples 来选层
4. **只测了 LLaVA 系列**: 没有在 Qwen-VL 等其他架构上验证

### Token Compression 方法完整图谱

```
                    ┌──────────────────────────────────────┐
                    │     Token Compression 方法总览        │
                    └──────────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
        Training-required     Training-free            Hybrid
              │                     │                      │
        VoCo-LLaMA           ┌─────┼──────┐          PyramidDrop
        ATP-LLaVA            │     │      │          (train+infer)
        Q-Former         Text-agnostic  Text-aware
                              │           │
                          ToMe(merge)  ┌──┼──────┐
                          VisionZip    │  │      │
                                    FastV  PDrop  SparseVLM
                                   (drop) (prog)  (recycle)
                                              │
                                          SwiftVLM
                                         (bypass) ⭐

进化路径:
FastV → PDrop (渐进式) → SparseVLM (text-guided + recycle)
                      → SwiftVLM (bypass + DP 选层)
```

### 对 STAR-Pro 的启示
- **Bypass 思想**: 视频理解中，浅层判定不重要的帧 token 可能在深层变重要
  → 不应过早丢弃时间维度的 token
- **非单调 selection 能力**: 选择合适的层做 temporal reasoning 很重要
- **定位任务的关键性**: 如果 STAR-Pro 涉及空间定位（如 grounding），bypass 会很有帮助
