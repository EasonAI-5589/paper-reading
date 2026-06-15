# FOMC Hawkish-Dovish

**论文**: Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis<br>
**发表**: ACL 2023 Main Conference<br>
**arXiv**: [2305.07972](https://arxiv.org/abs/2305.07972)<br>
**代码/数据**: [gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish)<br>
**许可**: CC BY-NC 4.0

基于 FOMC 演讲、会议纪要和新闻发布会文本的货币政策立场数据集，任务是识别鹰派、鸽派或中性表述。ODA-Fin-RL 以 weighted F1 评估，在该项取得 61.0；该结果低于其 SFT 模型的 63.9，说明 RL 并非对所有分类任务都有增益。

**研究价值**: 适合检查金融推理后训练是否保留细粒度政策语义，而不仅是提升数值题正确率。

> 当前仅完成基础收录。
