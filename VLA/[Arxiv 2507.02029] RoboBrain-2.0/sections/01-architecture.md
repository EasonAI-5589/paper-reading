# 2. Architecture

> 来源: RoboBrain 2.0 Technical Report

---

## 📄 原文

> 💡 **Section 概览**: 架构从 LLaVA 换成了 Qwen2.5-VL，支持更丰富的输入模态和输出格式。

---

![Figure 3](../images/93df68325fc81b8640962a5376b2650b4c748c881b59d7ca7d573614181eda36.jpg)
*Figure 3: RoboBrain 2.0 架构 — 支持多图/长视频/高分辨率 + 复杂指令 + 场景图*

> 💡 **Figure 3 批读**:
> ```
> 架构组成:
> ├── Vision Encoder: ~689M 参数
> │   ├── 动态分辨率 + windowed attention
> │   ├── Multi-dimensional RoPE (时空编码)
> │   └── 多视角: head cam, wrist cam, multi-view
> │
> ├── MLP Projector: visual tokens → LLM token space
> │
> └── LLM Decoder: Qwen2.5-VL (7B 或 32B)
>     ├── 输入: text tokens + visual tokens + scene graph tokens
>     └── 输出: free-form text / spatial coords / reasoning traces
> ```

### 2.1 Input Modalities

> 💡 **四种输入模态** — 比 v1 丰富很多:
> ```
> 1. Language instructions: 高层("把苹果搬到最近的桌子") + 低层("导航到桌子")
> 2. Scene graph: JSON 格式的环境结构 (物体、位置、机器人配置)  ← 新增!
> 3. Multi-view images: 头部/腕部相机、多视角投影
> 4. Video frames: 带时间戳的第一人称视频
> ```
> **Scene graph 输入是关键创新** — 让模型能理解整个场景的拓扑结构，而不只是看图片。
> 这对多机器人协作和长程规划至关重要。

### 2.2 Vision Encoder

- 动态分辨率 + windowed attention（继承 Qwen2.5-VL）
- Frame-wise tokenization + multi-dimensional RoPE（时空编码）
- 多视角序列化 + view-specific positional identifiers

> 💡 **vs v1**: v1 用 SigLIP (patch14-384)，v2 直接用 Qwen2.5-VL 的 vision encoder，
> 支持任意分辨率，不再受 384×{6×6} 的限制。

### 2.3 LLM Decoder and Output

> 💡 **三种输出格式**:
> ```
> 1. Free-form text: 任务分解、场景图更新、对话
> 2. Spatial coordinates: 点坐标、bounding box、轨迹
> 3. Reasoning traces (可选): Chain-of-Thought 长推理链
> ```
> **vs v1**: v1 需要额外的 A-LoRA 和 T-LoRA 来做 affordance 和 trajectory，
> v2 统一在一个模型里通过不同 prompt 模板输出，更简洁。

---

## 💡 Section 总结

### 架构对比
| 组件 | v1 | v2 |
|------|----|----|
| Vision Encoder | SigLIP-so400m (~400M) | Qwen2.5-VL encoder (~689M) |
| Projector | 2-layer MLP | MLP |
| LLM | Qwen2.5-7B-Instruct | Qwen2.5-VL-7B/32B |
| Affordance/Traj | 独立 LoRA 模块 | 统一解码（prompt 控制） |
| Scene graph | ❌ | ✅ JSON 输入 |
| 多视角 | ❌ | ✅ 多相机拼接 |

### 核心洞察
1. **从 LLaVA 架构迁移到 Qwen2.5-VL** — 继承了动态分辨率和 RoPE 时空编码
2. **取消了 LoRA 分支**，统一用一个模型处理所有任务 — 更优雅但需要更大模型 (32B)
3. **Scene graph 输入**是亮点 — 提供结构化的环境理解，支持多机器人协作
