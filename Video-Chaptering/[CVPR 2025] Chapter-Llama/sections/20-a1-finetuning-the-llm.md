# A.1. Finetuning the LLM

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

As mentioned in Sec. 3, for all experiments, we finetune Llama-3.1-8B-Instruct model [21] using LoRA [36] with rank $r = 8$ and target modules $\mathrm { Q }$ and $\mathrm { v }$ projections. LoRA [36] hyperparameters are set to $\alpha { = } 3 2$ and dropout $= 0 . 0 4$ . We use a batch size of 1 and a learning rate of $1 0 ^ { - 4 }$ , and train for 1 epoch using the AdamW optimizer. The training process takes

40 minutes using 4 NVIDIA H100 GPUs, and inference on 100 short videos takes 30 minutes using the same hardware.

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
