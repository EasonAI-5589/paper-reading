# 6. Application Scenarios

## 6.1 Image Understanding

### Medical Image Processing (医学影像)

- **场景**: 临床数据的快速准确解读，平衡效率与准确性
- **挑战**: 高分辨率医学影像处理能力仍有限
- **机会**: token压缩算法可加速高分辨率医学影像处理
- **参考**: [279], [280]

### Multi-page Document Understanding (多页文档)

- **场景**: 处理长文档，生成摘要或有意义的解答
- **挑战**: 文档长度持续增加，上下文长度受限
- **关联方法**: mPLUG-DocOwl2 [124], mPLUG-Owl3 [148] 等高分辨率cross-attention技术可直接应用
- **机会**: 类似图像高分辨率处理的加速技术可迁移到文档理解

### Satellite and Remote Sensing Imagery (卫星/遥感)

- **场景**: 解读高分辨率卫星和遥感图像
- **挑战**: 图像包含丰富结构信息但计算资源受限
- **进展**: [283], [284] 已探索token压缩策略，实现更高分辨率输入的高效处理
- **意义**: 对工业部署极为重要

---

## 6.2 Video Understanding

### Embodied AI (具身智能)

- **场景**: 机器人学习和具身AI，需要实时响应连续视频输入
- **关键需求**: 高效捕获空间和时序信息 → 实时fine-grained video understanding
- **代表方法**: EgoPrune [74]
- **意义**: 使机器人和具身智能体更适合实际部署

### Streaming Video Understanding (流式视频)

- **场景**: 处理连续视频流，以最小延迟提供实时响应
- **技术路线**:
  - 利用高时序冗余（1-10 FPS）进行token压缩 [57], [285]-[287]
  - 通过记忆机制存储紧凑历史表示
  - 推理时高效检索query-relevant的KV caches
- **关键要求**: 维持响应性和准确性的同时有效管理计算资源

### Instructional Video Summary (教学视频摘要)

- **场景**: 会议总结、讲座关键点提取
- **核心思想**: 选择性保留信息性token + 丢弃冗余信息
- **参考**: [5], [6]

---

## 6.3 Other Applications

### Mitigating Visual Hallucinations (减少视觉幻觉)

- **问题**: MLLMs可能生成与视觉输入不一致的文本（视觉幻觉）
- **解决思路**: 通过选择性token pruning → 引导模型关注最相关的图像/视频区域 → 过滤背景噪声和无关对象
- **效果**: 改善模型输出与实际视觉上下文的对齐 [289], [290]
- **意义**: 提升模型可靠性和可信度

---

## 个人笔记

<!-- 在此添加对应用场景的思考 -->

### 我最关注的应用场景
- TODO

### 潜在的新应用方向
- TODO

