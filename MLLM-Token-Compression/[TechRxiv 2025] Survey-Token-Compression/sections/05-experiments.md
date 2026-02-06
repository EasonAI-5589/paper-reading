# 5. Benchmarks and Metrics

## 5.1 Benchmarks

### 图像理解 Benchmarks

| Benchmark | 类型 | 指标 | 关注点 |
|-----------|------|------|--------|
| GQA-testdev | Open | Accuracy | General Image Perception |
| VQA-v2-testdev | Open | Accuracy | General Image Perception |
| VizWiz-val | Open | Accuracy | General Image Perception |
| POPE | Y/N | F1-Score | General Image Perception |
| TextVQA-val | Open | Accuracy | OCR |
| ScienceQA-Image | MQA,Y/N | Accuracy | Knowledge |
| MathVista-testmini | MQA,Open | Accuracy | Knowledge, Reasoning |
| MathVerse-testmini | MQA,Open | Accuracy | Knowledge, Reasoning |
| MMMU | MQA,Open | Accuracy | Knowledge, Reasoning |
| MME | Y/N | Perception Score | Integrated |
| MMBench-en-dev | MQA | Accuracy | Integrated |
| MM-Vet | Open | GPT-Score | Integrated |
| SeedBench-Image | MQA | Accuracy | Integrated |
| LLaVA-BenchW | Open | GPT-Score | Integrated |

**图像理解评估维度：**
- **General Image Perception**: 基础视觉识别（物体、场景、属性、空间关系）
- **OCR**: 文本识别和理解
- **Knowledge**: 视觉感知 + 领域知识整合
- **Reasoning**: 基于视觉内容的逻辑推理
- **Integrated**: 综合多维度评估

### 视频理解 Benchmarks

| Benchmark | 类型 | 指标 | 关注点 |
|-----------|------|------|--------|
| ActivityNet-QA | Open | Accuracy, GPT-Score | Integrated |
| MVBench | MQA | Accuracy | Temporal Understanding |
| EgoSchema | MQA | Accuracy | Long Video |
| LongVideoBench | MQA | Accuracy | Long Video, Integrated |
| MLVU-dev | MQA,Open | Accuracy, GPT-Score | Long Video, Integrated |
| Next-QA-MC | MQA | Accuracy | Integrated |
| Video-ChatGPT | Open | GPT-Score | Integrated |
| Video-MME | MQA | Accuracy | Integrated |

**视频理解评估维度：**
- **Temporal Understanding**: 时间动态（动作序列、运动模式、事件定位）
- **Long Video Understanding**: 长视频处理和推理
- **Integrated Video Understanding**: 视频感知 + 推理综合评估

---

## 5.2 Metrics

### 5.2.1 Effectiveness (效果)

| 指标 | 说明 |
|------|------|
| **Accuracy** | 预测是否匹配 ground-truth |
| **GPT-Score** | 开放式任务的 GPT 评分 |

### 5.2.2 Efficiency (效率)

| 指标 | 说明 |
|------|------|
| **Token Retention Count/Ratio** | 压缩后保留的 token 数量/比例 |
| **Prefilling/Decoding FLOPs** | 理论计算量 |
| **Prefilling/Decoding Latency** | 实际耗时（依赖硬件） |
| **Memory Usage** | 峰值内存占用 |

> Note: Identical retention levels do not guarantee equal inference latency, as factors such as compression position can significantly influence runtime.
>
> ==注意：相同保留率 ≠ 相同推理延迟，压缩位置等因素影响实际运行时间==

---

## 💡 Key Takeaways

1. **图像评估**：综合评估需覆盖 Perception + OCR + Knowledge + Reasoning
2. **视频评估**：时序理解和长视频处理是关键维度
3. **效率指标**：Token 保留率只是参考，实际延迟需要测量
4. **压缩位置影响**：同样压缩率，不同位置的实际效率不同

---

*[返回论文目录](../README.md)*
