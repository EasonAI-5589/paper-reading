# C.7. LLM variants

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

We conduct experiments with different variants of the Llama model family. All our previous results use Llama-3.1-8BInstruct, and we now compare it against the more recent

Table A.8. Llama variants: Model size has a significant impact on performance on Llama3.2 family. Llama-3.1-8B remains our choice due to its competitive performance with manageable computational complexity.   

<table><tr><td>Llama</td><td>Speech</td><td>Captions</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td rowspan="2">Llama-3.2-1B</td><td>;</td><td>-</td><td>23.5</td><td>58.3</td><td>6.9</td><td>23.9</td></tr><tr><td></td><td>✓</td><td>24.6</td><td>58.6</td><td>7.4</td><td>28.0</td></tr><tr><td rowspan="2">Llama-3.2-3B</td><td>;</td><td>;</td><td>35.2</td><td>66.7</td><td>10.5</td><td>52.5</td></tr><tr><td></td><td></td><td>34.7</td><td>65.2</td><td>12.5</td><td>63.6</td></tr><tr><td rowspan="2">Llama-3.2-11B</td><td>\</td><td>;</td><td>39.8</td><td>67.9</td><td>14.8</td><td>71.1</td></tr><tr><td></td><td></td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td>;</td><td></td><td>38.5</td><td>68.1</td><td>13.9</td><td>67.3</td></tr><tr><td></td><td>;</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

Llama-3.2 model in three sizes: 1B, 3B, and 11B parameters.

As shown in Tab. A.8, model size has a significant effect on chaptering quality. Using speech only, the F1 score improves substantially from 23.5 to 35.2 to 38.5 as we scale from 1B to 3B to 8B parameters, with only a minor additional gain to 39.8 when scaling to 11B parameters. This trend holds across all metrics. Llama-3.1-8B performs similar to Llama-3.2-11B, which we use in our final model due to reduced computational complexity. Note that we were unable to run Llama-3.2-11B on our final model combining speech and captions due to hardware constraints.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- 无图表

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
