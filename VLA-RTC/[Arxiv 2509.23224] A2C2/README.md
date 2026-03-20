# A2C2: Leave No Observation Behind — Real-time Correction for VLA Action Chunks

> **A2C2** — 异步 Action Chunk 校正

| 属性 | 值 |
|------|-----|
| arXiv | [2509.23224](https://arxiv.org/abs/2509.23224) |
| 日期 | 2025-09 |
| 作者 | Tatsuya Matsushima et al. |
| 课题 | VLA 实时推理加速 |

## 核心思想

Action chunking 提升效率但损害反应性。A2C2 引入一个**轻量校正头**，在每个控制步骤对 VLA 输出的 action chunk 进行实时修正。

## 方法

- **Asynchronous Action Chunk Correction (A2C2)**
- 输入：最新观测 + VLA 预测的 base action + chunk 内位置编码 + base policy 特征
- 输出：每步校正量（per-step correction）
- **不需要重训练 base policy**，与 RTC 等异步方案正交

## 关键结果

| 基准 | 提升 |
|------|------|
| Kinetix (12 tasks) | **+23%** vs RTC |
| LIBERO Spatial | **+7%** vs RTC |

- 校正头小而快，相比大 VLA 推理几乎无额外开销
- 即使零延迟，长 horizon 场景也有提升
- **Plug-in 机制**，可叠加在任何 chunking policy 上

---

*待深度阅读* 📖
