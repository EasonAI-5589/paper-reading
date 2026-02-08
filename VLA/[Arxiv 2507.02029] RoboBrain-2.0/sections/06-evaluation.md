# 6 Evaluation Results

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

We conducted a comprehensive evaluation of RoboBrain-2.0, focusing on its performance across spatial and temporal reasoning capabilities on embodiment. To ensure consistency and rigor in evaluation, we adopted FlagEvalMM [20], our flexible framework for systematic multimodal model assessment. Evaluations on spatial reasoning benchmarks (e.g., CV-bench [67], Blink [15], Where2Place [77], ShareRobot-Bench [23]), presented in Section 6.1, underscore the model's strengths in embodied spatial reasoning. An in-depth analysis of multi-robot collaboration [61] and long-horizon planning (e.g., EgoPlan2 [9], RoboBench) capabilities is provided in Section 6.2, highlighting the model's advancements in temporal reasoning tasks. Qualitative examples and prompt details are provided in Section A and Section B, respectively.

> 💡 **评测框架**: FlagEvalMM (BAAI 自研)。分两大类: Spatial (9 benchmarks) + Temporal (3 benchmarks)。

---

### 6.1 Spatial Reasoning Capability

RoboBrain-32B-2.0 and RoboBrain-7B-2.0 demonstrate exceptional performance across nine spatial reasoning benchmarks: BLINK, CV-Bench, EmbSpatial, RoboSpatial, and RefSpatial-Bench (Table 2), as well as SAT, VSI-Bench, Where2Place, and ShareRobot-Bench (Table 3). Below is a detailed analysis highlighting their state-of-the-art (SOTA) achievements and near-SOTA competitive results.

#### Table 2: Performance across five spatial reasoning benchmarks

| Models | BLINK Dep. | BLINK Spa. | BLINK All↑ | CV-Bench All↑ | EmbSpatial All↑ | RoboSpatial All↑ | RefSpatial Loc. | RefSpatial Pla. | RefSpatial All↑ |
|--------|-----------|-----------|-----------|--------------|----------------|-----------------|----------------|----------------|----------------|
| **General Baselines** | | | | | | | | | |
| Gemini-2.5-Pro | 79.03 | 84.62 | 81.83 | 84.59 | 78.74 | 59.87 | 44.58 | 31.73 | 38.16 |
| GPT-o4-mini | 79.03 | 88.11 | 83.57 | 85.21 | 78.29 | 51.25 | 15.00 | 19.58 | 17.29 |
| GPT-4o | 72.58 | 83.22 | 77.90 | 78.63 | 71.92 | 44.42 | 8.00 | 9.55 | 8.78 |
| Claude-Sonnet-4 | 75.81 | 80.42 | 78.12 | 78.43 | 64.26 | 51.26 | 5.00 | 10.37 | 7.69 |
| Qwen2.5-VL-32B | 77.42 | 85.31 | 81.37 | 81.59 | 74.45 | 52.16 | 16.83 | 10.60 | 13.72 |
| Qwen2.5-VL-72B | 74.19 | 78.32 | 76.26 | 82.68 | 73.30 | 48.33 | 23.50 | 15.83 | 19.67 |
| **Embodied Baselines** | | | | | | | | | |
| Cosmos-Reason1-7B | 63.71 | 73.43 | 68.57 | 74.71 | 65.22 | 38.81 | 9.84 | 1.04 | 5.44 |
| VeBrain-8B | 78.23 | 81.12 | 79.68 | 78.57 | 70.52 | 42.48 | 0.03 | 0.57 | 0.30 |
| Magma-8B | 65.32 | 66.43 | 65.88 | 60.98 | 64.59 | 33.71 | 1.00 | 8.00 | 4.50 |
| RoboBrain-7B-1.0 | 75.81 | 78.32 | 77.07 | 76.22 | 68.13 | 51.53 | 14.43 | 5.41 | 9.92 |
| **RoboBrain-7B-2.0** | **84.68** | 83.22 | **83.95** | **85.75** | 76.32 | 54.23 | 36.00 | 29.00 | 32.50 |
| **RoboBrain-32B-2.0** | 79.84 | **87.41** | 83.63 | 83.92 | **78.57** | **72.43** | **54.00** | **54.00** | **54.00** |

> 💡 **Table 2 批读**:
> ```
> SOTA 排行 (All):
>
> BLINK:        RoboBrain-7B-2.0 (83.95) > GPT-o4-mini (83.57) > RoboBrain-32B (83.63)
> CV-Bench:     RoboBrain-7B-2.0 (85.75) > GPT-o4-mini (85.21) > Gemini-2.5-Pro (84.59)
> EmbSpatial:   Gemini-2.5-Pro (78.74) ≈ RoboBrain-32B (78.57) > GPT-o4-mini (78.29)
> RoboSpatial:  RoboBrain-32B (72.43) >> Gemini-2.5-Pro (59.87) 🔥 大幅领先
> RefSpatial:   RoboBrain-32B (54.00) >> Gemini-2.5-Pro (38.16) 🔥 碾压级
> ```
>
> **关键发现**:
> 1. **7B 在 BLINK/CV-Bench 上比 32B 好**——小模型在这两个 benchmark 上过拟合？或者 7B 训练更充分？
> 2. **RoboSpatial 和 RefSpatial 是 embodied-specific benchmarks**，通用模型表现很差（GPT-4o 只有 8.78）
> 3. **RefSpatial 是最有区分度的 benchmark**: RoboBrain-32B (54.00) vs 最好通用模型 Gemini (38.16) vs GPT-4o (8.78)

#### Table 3: Performance across four spatial reasoning benchmarks

| Models | SAT All↑ | VSI-Bench All↑ | Where2Place Seen | Where2Place Unseen | Where2Place All↑ | Afford.↑ | Traj.(DFD↓) |
|--------|---------|---------------|-----------------|-------------------|-----------------|----------|-------------|
| **General Baselines** | | | | | | | |
| Gemini-2.5-Pro | 79.33 | 47.81 | 42.92 | 41.13 | 42.38 | 10.26 | 0.7666 |
| GPT-o4-mini | 82.00 | 41.96 | 26.63 | 26.49 | 26.59 | 8.27 | 0.5726 |
| Qwen2.5-VL-72B | 58.67 | 35.51 | 35.74 | 49.65 | 39.92 | 23.80 | 0.5034 |
| **Embodied Baselines** | | | | | | | |
| RoboBrain-7B-1.0 | 59.33 | 31.12 | 54.58 | 49.45 | 53.04 | 10.20 | 0.6248 |
| **RoboBrain-7B-2.0** | 75.33 | 36.10 | 64.33 | 61.88 | 63.59 | 28.05 | 0.5512 |
| **RoboBrain-32B-2.0** | **86.67** | 42.69 | **73.95** | **72.74** | **73.59** | **35.28** | **0.2368** |

> 💡 **Table 3 批读**:
> ```
> SAT:          RoboBrain-32B (86.67) >> GPT-o4-mini (82.00) 🔥
> VSI-Bench:    Gemini-2.5-Flash (48.83) > Gemini-2.5-Pro (47.81) > RoboBrain-32B (42.69)
> Where2Place:  RoboBrain-32B (73.59) >> RoboBrain-1.0 (53.04) >> Qwen-72B (39.92) 🔥🔥
> Affordance:   RoboBrain-32B (35.28) > RoboBrain-7B (28.05) > Qwen-72B (23.80)
> Trajectory:   RoboBrain-32B (0.2368) >> Qwen-72B (0.5034) 🔥 DFD 降低 53%
> ```
>
> **关键发现**:
> 1. **VSI-Bench 是唯一没赢的**: Gemini 系列在 visual-spatial integration 上更强
> 2. **Where2Place 提升巨大**: 1.0→2.0 从 53.04→73.59 (+20.55)，说明 spatial referring 数据有效
> 3. **Trajectory DFD 0.2368**: 最低 = 最好，远超所有 baseline

---

### 6.2 Temporal Reasoning Capability

RoboBrain-32B-2.0 and RoboBrain-7B-2.0 exhibit outstanding performance across three critical measures of temporal reasoning benchmarks: Multi-Robot Planning, Ego-Plan2, and RoboBench, as shown in Table 4. Below is a detailed analysis highlighting their state-of-the-art (SOTA) achievements and near-SOTA results.

#### Table 4: Performance across three temporal reasoning benchmarks

| Models | Multi-Robot Super. | Rest. | House. | All↑ | EgoPlan2 Daily. | Hobbies. | Rec. | Work. | All↑ | RoboBench Plan.↑ |
|--------|-------------------|-------|--------|------|----------------|----------|------|-------|------|-----------------|
| **General Baselines** | | | | | | | | | | |
| Gemini-2.5-Pro | 63.51 | 54.77 | 78.39 | 65.39 | 44.19 | 43.05 | 46.45 | 39.60 | 42.85 | 63.49 |
| GPT-4o | 77.89 | 67.34 | 79.40 | 74.50 | 47.38 | 40.00 | 44.81 | 35.64 | 41.79 | 68.60 |
| Qwen2.5-VL-32B | 67.84 | 61.81 | 75.38 | 68.00 | 64.46 | 51.53 | 57.92 | 50.00 | 56.25 | 45.92 |
| Qwen2.5-VL-72B | 77.39 | 68.34 | 79.40 | 74.67 | 60.36 | 48.14 | 63.39 | 46.29 | 53.75 | 66.94 |
| **Embodied Baselines** | | | | | | | | | | |
| VeBrain-8B | 41.70 | 35.67 | 39.69 | 38.83 | 31.79 | 35.31 | 31.19 | 34.43 | 27.30 | 46.77 |
| RoboBrain-7B-1.0 | 4.52 | 7.04 | 5.03 | 5.50 | — | — | — | — | — | 38.93 |
| **RoboBrain-7B-2.0** | **83.92** | **77.39** | **84.42** | **81.50** | 39.41 | 32.20 | 33.88 | 26.98 | 33.23 | **72.16** |
| **RoboBrain-32B-2.0** | 84.42 | 72.36 | 85.43 | **80.33** | **64.01** | **53.22** | **57.92** | **52.48** | **57.23** | 68.33 |

> 💡 **Table 4 批读**:
> ```
> Multi-Robot Planning:
> ├── RoboBrain-7B-2.0 (81.50) ≈ RoboBrain-32B (80.33) 🔥🔥
> ├── >> GPT-4o (74.50) >> Qwen-72B (74.67)
> └── vs RoboBrain-1.0: 5.50 → 81.50 (+76!!) 质的飞跃
>
> EgoPlan2:
> ├── RoboBrain-32B (57.23) > Qwen-32B (56.25) > Qwen-72B (53.75)
> ├── RoboBrain-7B (33.23) 比较弱（低于通用模型）
> └── 说明 EgoPlan2 需要更强的通用理解能力
>
> RoboBench:
> ├── RoboBrain-7B (72.16) > Claude-Sonnet-4 (70.21) > GPT-o4-mini (70.01)
> └── 7B 比 32B (68.33) 好，有趣
> ```
>
> **关键发现**:
> 1. **Multi-Robot Planning 是最大亮点**: 1.0→2.0 从 5.50→81.50，这是因为 1.0 根本没有 multi-robot 训练数据
> 2. **7B 在 Multi-Robot 和 RoboBench 上比 32B 好**: 可能是因为 7B 训练更充分，或者这些 benchmark 不需要太大模型
> 3. **EgoPlan2 是 7B 的弱项**: 需要更强的通用视频理解能力，7B 参数量不够
> 4. **RoboBrain-1.0 在 temporal benchmarks 上几乎为零**: 说明 1.0 没有 temporal 能力，2.0 是从无到有的突破

---

## 💡 Section 总结

### SOTA 汇总（12 个 Benchmark）
| Benchmark | SOTA Model | Score | 优势幅度 |
|-----------|-----------|-------|----------|
| **BLINK** | RoboBrain-7B-2.0 | 83.95 | +0.38 vs GPT-o4-mini |
| **CV-Bench** | RoboBrain-7B-2.0 | 85.75 | +0.54 vs GPT-o4-mini |
| EmbSpatial | Gemini-2.5-Pro | 78.74 | RoboBrain-32B 78.57 (near) |
| **RoboSpatial** | RoboBrain-32B | 72.43 | +12.56 vs Gemini 🔥 |
| **RefSpatial** | RoboBrain-32B | 54.00 | +15.84 vs Gemini 🔥 |
| **SAT** | RoboBrain-32B | 86.67 | +4.67 vs GPT-o4-mini |
| VSI-Bench | Gemini-2.5-Flash | 48.83 | RoboBrain-32B 42.69 |
| **Where2Place** | RoboBrain-32B | 73.59 | +31.21 vs Qwen-72B 🔥🔥 |
| ShareRobot-Afford | RoboBrain-32B | 35.28 | +11.48 vs Qwen-72B |
| ShareRobot-Traj | RoboBrain-32B | 0.2368 | -0.2666 vs Qwen-72B |
| **Multi-Robot-Plan** | RoboBrain-7B-2.0 | 81.50 | +6.83 vs Qwen-72B 🔥 |
| **EgoPlan2** | RoboBrain-32B | 57.23 | +0.98 vs Qwen-32B |
| **RoboBench** | RoboBrain-7B-2.0 | 72.16 | +1.95 vs Claude-Sonnet-4 |

**SOTA 数**: 9/12 (标粗的)，near-SOTA: 2/12，非 SOTA: 1/12 (VSI-Bench)

### 核心洞察
1. **Embodied-specific benchmarks 优势最大**: RoboSpatial, RefSpatial, Where2Place, Multi-Robot-Plan 这些 embodied 专用 benchmark 上碾压通用模型
2. **通用 benchmark 也不差**: BLINK, CV-Bench, SAT 上也是 SOTA，说明没有牺牲通用能力
3. **7B vs 32B 分化**: 7B 在 BLINK/CV-Bench/Multi-Robot/RoboBench 更好，32B 在其余更好
4. **VSI-Bench 是唯一明显弱点**: visual-spatial integration 需要更强的视觉理解，Gemini 的视觉编码器可能更好
