# 7. Open Challenges and Future Work

## 7.1 Lack of Theoretical Understanding (缺乏理论基础)

### 现状
- 大多数方法凭**经验/直觉**设计，缺少严格理论支撑
- 少数例外: DeCo [105], DART [183] 分析了压缩如何影响MLLM内部的表示学习

### 核心问题
当前常用的token重要性指标（attention权重、pairwise相似度、互信息）：
- 仅指示**相关性 (correlation)** 而非**必要性 (necessity)**
- 无法解释保留的token是否真正足够 → 好性能可能只是巧合

### 未来方向
- 建立token选择与**充分性 (sufficiency)、因果性 (causality)、鲁棒性 (robustness)** 之间的理论联系
- 从ad-hoc启发式 → 有原则的、可泛化的压缩理论

---

## 7.2 Lack of Task- and Content-Aware Adaptivity (缺乏任务/内容自适应性)

### 现状
- 多数方法采用**固定压缩率/固定启发式规则**，不考虑任务类型或视觉内容复杂度
- M³ [91] 观察: 自然场景（如COCO）仅需9个token即可处理，但文档理解/OCR需要144-576个token

### 问题
| 场景 | 固定策略的问题 |
|------|-------------|
| 简单任务 + 高保留率 | 保留了冗余token → 效率浪费 |
| 复杂任务 + 低保留率 | 丢弃了关键信息 → 性能下降 |
| 简单图像 + 统一压缩 | 忽略场景复杂度差异 |
| 信息密集图像 + 统一压缩 | 忽略内容丰富度差异 |

### 已有探索
- PAR [100], QG-VTC [101], VCM [186]: 引入自适应机制，根据文本query或视觉内容调整压缩
- VisionThink [190]: 基于强化学习，让模型自主决定是否需要高分辨率输入

### 未来方向
- **Task-aware compression**: 根据任务认知复杂度动态调整压缩程度和方式
- **Content-aware compression**: 根据视觉内容的信息密度和复杂度自适应

---

## 7.3 Performance Degradation in Practical Tasks (实际任务性能下降)

### 现状
- 即使保留1/3或1/4的视觉token，在通用Visual QA上仍可保持comparable accuracy
- **但**在需要fine-grained perception的实际任务上表现下降

### 受影响严重的任务
| 任务类型 | 具体场景 | 为什么压缩会影响 |
|---------|---------|----------------|
| **OCR** | 文本识别、文档解析 | 文本细节的精确定位依赖高分辨率token |
| **Document Understanding** | 表格/图表理解 | 结构化视觉布局需要空间精度 |
| **Dense Reasoning** | 空间关系、属性推理 | 需要精确的空间和语义线索 |

### 核心矛盾
> 当前压缩方案优先考虑**平均效率**而非**任务特定保真度** → 在需要高分辨率理解或domain-level精度的场景中适用性受限

### 未来方向
- 设计任务感知的压缩策略，在不同任务上实现efficiency-fidelity的最优平衡
- 探索可恢复/可逆压缩机制

---

## 7.4 Limitations of Existing Evaluation (现有评估的局限)

### 三个关键限制

#### (1) Lack of Systematic Task Categorization
- Benchmarks按粗粒度分类（如"Image Perception"、"Reasoning"）
- 无法揭示压缩对**具体视觉理解能力**的影响（如空间关系推理 vs 物体追踪 vs 表格解析）

#### (2) Inefficient Evaluation Processes
- 典型工作使用10+个benchmark，各含数千样本
- Benchmarks之间存在大量重叠 → 冗余评估 → 资源浪费

#### (3) Absence of Consistent Evaluation Standards
- 不同工作使用不同benchmark/指标组合
- 各工作强调不同strengths → 跨方法公平比较困难

### 已有努力
- [296]: 引入更具挑战性的评估设置
- 但系统化、标准化的评估框架仍然缺失

---

## 挑战总结与关键insight

| 挑战 | 当前状态 | 需要的突破 |
|------|---------|----------|
| 理论基础 | 经验驱动 | 因果性/充分性理论 |
| 自适应性 | 固定策略 | 任务/内容感知的动态压缩 |
| 实际任务性能 | 通用QA表现好，细粒度任务下降 | 任务特定的保真度保证 |
| 评估体系 | 碎片化、不一致 | 统一标准化框架 |

---

## 个人笔记

<!-- 在此添加对未来方向的思考 -->

### 我认为最有价值的研究方向
- TODO

### 可能的突破口
- TODO

### 与我的研究可结合的方向
- TODO

