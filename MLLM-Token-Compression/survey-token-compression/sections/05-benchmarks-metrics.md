# 5. Benchmarks and Metrics

## 5.1 Benchmarks (Table 8)

### 图像理解 Benchmarks

| Benchmark | 题型 | 评估指标 | 样本数 | 测评能力 |
|-----------|------|---------|--------|---------|
| GQA-testdev-balanced | Open | Accuracy | 12,578 | 通用图像感知 |
| VQA-v2-testdev | Open | Accuracy | 107,394 | 通用图像感知 |
| VizWiz-val | Open | Accuracy | 4,319 | 通用图像感知 |
| POPE | Y/N | F1-Score | 3,000 | 通用图像感知 |
| TextVQA-val | Open | Accuracy | 5,000 | OCR |
| ScienceQA-Image-test | MQA, Y/N | Accuracy | 2,017 | 知识 |
| MathVista-testmini | MQA, Open | Accuracy | 1,000 | 知识/推理 |
| MathVerse-testmini | MQA, Open | Accuracy | 3,940 | 知识/推理 |
| MMMU | MQA, Open | Accuracy | 11,550 | 知识/推理 |
| MME | Y/N | Perception Score | 2,374 | 综合 |
| MMBench-en-dev | MQA | Accuracy | 4,329 | 综合 |
| MM-Vet | Open | GPT-Score | 218 | 综合 |
| SeedBench-Image | MQA | Accuracy | 14,280 | 综合 |
| LLaVA-Bench^W | Open | GPT-Score | 60 | 综合 |

### 视频理解 Benchmarks

| Benchmark | 题型 | 评估指标 | 样本数 | 测评能力 |
|-----------|------|---------|--------|---------|
| ActivityNet-QA-test | Open | Accuracy, GPT-Score | 8,000 | 综合 |
| MVBench | MQA | Accuracy | 4,000 | 时序理解 |
| EgoSchema | MQA | Accuracy | 5,063 | 长视频 |
| LongVideoBench-val | MQA | Accuracy | 1,337 | 长视频 |
| MLVU-dev | MQA, Open | Accuracy, GPT-Score | 2,593 | 长视频综合 |
| Next-QA-MC-test | MQA | Accuracy | 8,564 | 综合 |
| Video-ChatGPT | Open | GPT-Score | 3,493 | 综合 |
| Video-MME | MQA | Accuracy | 2,700 | 综合 |

### Benchmark分类

**图像理解**:
- **General Image Perception**: 基本视觉识别 (物体、场景、属性、空间关系)
- **OCR**: 识别和解释视觉中的文本内容
- **Knowledge**: 视觉感知 + 领域/世界知识
- **Reasoning**: 基于视觉内容的逻辑推理和问题解决
- **Integrated**: 感知+推理综合评估

**视频理解**:
- **Temporal Understanding**: 动作序列、运动模式、事件定位
- **Long Video Understanding**: 长视频处理和推理
- **Integrated**: 多维度综合评估

---

## 5.2 Metrics

### 5.2.1 Effectiveness (效果)

| 指标 | 描述 |
|------|------|
| **Accuracy** | 模型预测是否匹配ground-truth (大多数benchmark的主指标) |
| **GPT-Score** | GPT对MLLM开放式回答的数值评分 (用于image captioning等无唯一答案的任务) |

### 5.2.2 Efficiency (效率)

| 指标 | 描述 | 注意事项 |
|------|------|---------|
| **Token Retention Count/Ratio** | 压缩后保留的视觉token绝对数量/相对百分比 | 相同保留率≠相同推理延迟（压缩位置影响很大） |
| **Prefilling/Decoding FLOPs** | 预填充和解码的理论计算量 | 硬件无关 |
| **Prefilling/Decoding Latency** | 实际墙钟时间 | 硬件相关 |
| **Memory Usage** | 推理时峰值内存 | 对资源受限设备部署至关重要 |

---

## 评估体系的问题 (§7.4)

论文指出当前评估存在三个关键限制：

1. **缺乏系统性任务分类**: benchmarks按粗粒度分类，无法揭示压缩对具体能力的影响 (如空间关系推理 vs 物体追踪)
2. **低效评估流程**: 通常使用10+个benchmark，各含数千样本，overlap大，冗余严重
3. **缺乏统一评估标准**: 不同工作使用不同benchmark和指标组合 → 跨方法公平比较困难

---

## 个人笔记

<!-- 在此添加对评估体系的思考 -->

### 我最关注的Benchmarks
- TODO

### 评估中需要注意的坑
- TODO

