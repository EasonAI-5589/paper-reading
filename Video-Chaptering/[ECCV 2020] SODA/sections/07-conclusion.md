# 7. Conclusion

## 主要贡献

1. **证明现有评测不足**: ActivityNet Challenge 的评测框架无法正确评估视频故事描述
2. **提出 SODA**: 考虑时序顺序的评测框架
3. **引入 F-measure**: 惩罚过多或过少的 caption
4. **开源代码**: https://github.com/fujiso/SODA

## 实验验证

| 方面 | Current | SODA |
|------|---------|------|
| 检测冗余 caption | ❌ 无法区分 | ✅ 低分 |
| 检测顺序错误 | ⚠️ 轻微下降 | ✅ 显著下降 |
| 与人工评估一致性 | 0.66-0.72 | **0.76-0.94** |

## 局限性

- SODA 仍依赖 METEOR 指标（继承其局限性）
- 只验证了 ActivityNet Captions 数据集

## 影响

SODA 已被后续工作广泛采用：
- **VidChapters-7M (NeurIPS 2023)**: 使用 SODA_c 作为主要评测指标
- **Chapter-Llama (CVPR 2025)**: 使用 SODA_c

## 代码

```bash
git clone https://github.com/fujiso/SODA
```

---

## 个人理解

### 为什么 SODA 重要？

1. **评测驱动研究方向**: 错误的评测会导致错误的优化目标（生成更多 caption 而不是更好的 caption）
2. **故事性是核心**: Video Chaptering 的目标是帮助人理解视频，顺序和数量都很重要
3. **简单但有效**: DP + F-measure，思路清晰，容易实现

### 对 Video Chaptering 研究的启示

- 评测时一定要用 SODA_c，不能只看 BLEU/METEOR
- 生成的章节数量应该接近 ground truth
- 顺序很重要，乱序会被严重惩罚
