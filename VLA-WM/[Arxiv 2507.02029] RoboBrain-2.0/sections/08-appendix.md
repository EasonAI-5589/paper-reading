[← 返回 README](../README.md)

# Appendix

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 概要总结

Appendix 包含两个部分：

### A. Qualitative Examples（定性示例）

展示 RoboBrain 2.0 在各种具身 AI 任务上的可视化结果，共 7 类任务：

| 子节 | 任务 | Figure 范围 | 说明 |
|------|------|------------|------|
| A.1 | Pointing（指向） | Figure 5-20 | 蓝色点表示模型的空间指代预测，包含不同 Reasoning Step（1-5步推理）|
| A.2 | Affordance（可操作性） | 后续图 | 物体部件功能识别（如杯子的把手、瓶盖、勺子等）|
| A.3 | Trajectory（轨迹） | 后续图 | 未来轨迹预测的可视化 |
| A.4 | EgoPlan2 | 后续图 | 第一人称视角活动规划示例 |
| A.5 | Close-Loop Interaction | 后续图 | 闭环交互的 Observation-Thought-Action 序列 |
| A.6 | Multi-Robot Planning | 后续图 | 多机器人协作规划示例 |
| A.7 | Synthetic Benchmarks | 后续图 | 合成基准测试示例 |

> 💡 **Appendix A 要点**:
> - Pointing 示例最多（Figure 5-20），展示了从简单（"指出橙色盒子"）到复杂（"指出离电视最远的黑色物体"）的空间推理
> - Reasoning Step 从 1 到 5 不等，说明模型可以进行多步推理
> - Affordance 示例展示了功能性理解（"杯子哪个部分用来喝水？" → 指向杯口）
> - Close-Loop 示例展示了完整的 OTA 链

### B. Prompts Details（Prompt 详情）

列出了各任务使用的 prompt 模板：

| 子节 | 任务 | 说明 |
|------|------|------|
| B.1 | Pointing（坐标） | "Point out all instances of {label} in the image" 等 28 个模板 |
| B.2 | Trajectory（坐标） | 轨迹预测的 prompt 格式 |
| B.3 | Affordance（Bbox） | "Which part of X can be used to Y?" 格式 |
| B.4 | General Spatial Analysis | 自由问答形式的空间分析 |
| B.5 | Long-horizon Planning | 长程规划的 prompt + scene graph 输入格式 |
| B.6 | Closed Loop Conversation | 闭环交互的多轮对话格式 |
| B.7 | Multi-Robot Planning | 多机器人协作的结构化 prompt |

> 💡 **Appendix B 要点**:
> - Prompt 设计体现了任务的层次性：坐标输出 vs 文本输出 vs 结构化输出
> - Multi-Robot Planning 的 prompt 包含完整的 scene graph JSON + 机器人规格 + 工具列表

---

## 📄 原文（完整 Appendix）

# Appendix

# A Qualitative examples

This section provides a comprehensive set of qualitative examples that illustrate the capabilities of RoboBrain 2.0 in various embodied AI tasks. These examples demonstrate the model’s proficiency in spatial reasoning, temporal planning, and interactive reasoning, showcasing its potential for real-world applications.

# A.1 Examples for Pointing

In the pointing task, RoboBrain 2.0 is required to identify and point to specific objects within an image based on complex spatial instructions. For instance, given the instruction “Please point out the orange box,” the model accurately identifies the orange box in the image. Similarly, for more complex instructions such as “Please point out the brown box on the shelf,” RoboBrain 2.0 demonstrates its ability to understand spatial relationships and accurately points to the correct object. The model’s proficiency in this task is further exemplified by its performance on a variety of pointing examples, as shown in Figure 5-Figure 20. These examples highlight the model’s robust spatial reasoning capabilities, enabling it to handle a wide range of pointing tasks with high precision. Whether the instructions involve simple object identification or more intricate spatial relationships, RoboBrain 2.0 consistently demonstrates its ability to accurately locate and point to the specified objects. This capability is crucial for applications in robotics and automation, where precise object localization is essential for effective interaction with the physical environment.

![](../images/5199e0fb9c1ba6a4bd429003f1e686291b66a6a67c9d6010473d6603fab118c7.jpg)  
Figure 5 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

![](../images/7c676fbee81cf6b3b551c5244bf6c3055d4176edf198477d948f9a26419d72bf.jpg)  
Reasoning $S t e p = 1$

Please point to the farthest white cabinet.

![](../images/20f85a78c4d1dc6a851ce65d3bd544b5679cfa72b3b109131a31889bb4f8c6a5.jpg)  
Reasoning Step = 2

Please point to the top piece of paper on the white table.

![](../images/72a221505e4044bce69008d45bfcb8c347196d05107208ee0e81ea7b55648c05.jpg)  
Reasoning Step $= 2$

Please point to the left pillow on the sofa

![](../images/0529ec1c72ca54dde490c67a76688dcf48aac24609a28fd5d0f1d15b4957e047.jpg)  
Reasoning Step $= 3$

![](../images/34228a7114eced7df07fc7f5316cb5272c8c6ce888e309d89f3209603ca79e6c.jpg)  
Reasoning Step = 2

Please point out the leftmost black object on the same platform as the micro-wave oven.

![](../images/a5c3fc5c47a657565c93a50134ad18262db6de90665fbe9a37f6528c084f901d.jpg)  
Reasoning Step = 3

Please point out the orange box on the white table on the left.

Please point out the white cup on the shelf behind the chair.

![](../images/660df230436329f985d7d9a0feb5b530770b2f005a3134e762e2b7d0b8f39389.jpg)

# Reasoning Step $= 2$

![](../images/99d2b419d00bea52610a211410b05f002d54d46d362061d10769b8a041ee8868.jpg)  
Reasoning Step $= 3$

Please point to the rightmost blue box on the refrigerator.

Please point out the cardboard box under the bed which is closest box to the viewer.

![](../images/f4712c9675dd52fdc4ccb9764d1906fca61a3b3b77b1f7125a93e4d472007af8.jpg)  
Reasoning Step = 2   
Figure 6 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the blue object on the table.

![](../images/bb960538bea1431e3dffa4ec5f5f01c5dafeb3ca8aae7d06d080dac125b0e9b4.jpg)  
Reasoning $S t e p = 3$

Please point to the white bottle on the table that is closest to the green bottle on the left.

![](../images/3f238d8c9f7985b0634c30390b230a5921f9399495c1bdf86fa7c9b9127f5c06.jpg)  
Reasoning $S t e p = 2$

Please point out the second object from the left to the right on the nearest platform.

![](../images/da42c6c96ead10ed2388e038a94169f48bed7efecbce5ce66cdff28801e423d6.jpg)  
Reasoning Step = 2

Please point out the object on the right of the shovels.

![](../images/7ffc5265b97e7a918dc1c50a457b032424e09799904f091ad938a8002e69f292.jpg)  
Reasoning Step = 2

![](../images/a969622356830a650573479e6b0c66b9f491370e1da8a9e59f83e1ad8de392e9.jpg)  
Reasoning Step = 2

Please point to the pillow closest to the right nightstand.

![](../images/4f95dcf470e23908d0cf99362f35253518dd12ebce274d1cc340f4dc10e79904.jpg)  
Reasoning Step $= 3$ Please point out the object between the white box and the farthest black pot.

Please point to the pillow closest to the remote controller.

![](../images/218b0113a0683ff227bd5fa1ecf4b74f19834921829f012c6d8f7fd306a78f8d.jpg)

# Reasoning Step $= 3$

![](../images/869e1b0c5bf0fa89230d98d6f9ef83d18ea466a1a3f957877059a8aa558ce3af.jpg)  
Reasoning Step = 2

Please point out the black object that is on the same platform as the TV.

Please point out the vase closest to the TV.

![](../images/e6b8d0678f6ca273da3a82961a3a2f3aadaeca0940d834cceefd3763a87ea024.jpg)  
Reasoning Step = 3   
Figure 7 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point to the rightmost box at the bottom of the shelf.

![](../images/346632dafce3b56380528f7496d0c735756a8c2e13cfaed78289fb057065d622.jpg)  
Reasoning $S t e p = 1$

![](../images/c87e220ab9f1211445767999f9a17663741323509d01c741ed34ccfbbdd2b6d4.jpg)  
Reasoning Step = 1

Please point out the second silver box from left to right.

![](../images/84b7826ca3616d7874da33ecd2a2f1aa82669e423b87fb9d48f4c077c4827300.jpg)  
Reasoning Step = 3

Please point to the wooden plate on the far left.

Please point out the black framed painting on the right of the lamp.

![](../images/780e7ff7df7e629e3f6d22177d83275e5a232ace4e20b5d745cb32ce938e6d5b.jpg)  
Reasoning Step = 1

![](../images/1dcf7ce854c9049283a7f55079ad0f402c2255faa0a30199b98e7fafcad3674c.jpg)  
Reasoning $S t e p = 2$ Please point out the green towel on the upper right with yellow object on top.

Please point out the chair closest from the viewer.

![](../images/d854956af48302da3fde8aa79b14f381d123c7dceaf764c692cd1410db0efe1b.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please point out the brown sofa, which is the closest sofa to the viewer.

![](../images/d616ec9e0c0214bbc5fe55ce8cbcd6b78282a0408ca91d51037f3c65a4c34a30.jpg)

# Reasoning Step $= 2$

Please point out the object on the windowsill farthest from the viewer.

![](../images/274e1e30339975a21c782463f66b7d974255382a14df6c4243f8e2d95aab9717.jpg)  
Reasoning Step = 1

# Reasoning Step = 2

![](../images/04b65904ef358c5b9eba784b100cc7cd4d56d8ad88b0da81cd00941157ad802b.jpg)  
Figure 8 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the black object which is farthest from the shelf.

Please point out the paper tube closest to the viewer.

![](../images/d5d73c3707af4f2fa0095a28fcd3f24fcba66a8c068e82ef35ed77fc40bb208a.jpg)

# Reasoning $S t e p = 1$

Please point out the sofa on the right side that is closest to the viewer.

![](../images/72d2832c735673f73c58c8165e377a0943e75266fa31be2276e9aa08cad70bb7.jpg)  
Reasoning Step = 1

Please point out the sofa farthest from the viewer.

![](../images/26ee5b59217b71b66bfc46088b6570b9140372e7dcae79f9373fb4d2fa3a08ae.jpg)  
Reasoning Step = 2

Plea se point out the painting hanging on the wall.

![](../images/fc522065660aa4ea17d76ecff9952df0ac592e43e96f4df02e3b7bbb08831fbb.jpg)  
Reasoning $S t e p = 2$

Please point out the blue toothbrush farthest from the faucet.

![](../images/3093855e686184466ada056399eae290a8c7c6e3a8967d5330bbc9cddd6aa22d.jpg)  
Reasoning Step = 2

Please point out the card closest to the wooden door.

![](../images/efbc75654e0f2ef2da249c1776c933e014dab4b28cb7d3c282d901b455e50e8d.jpg)  
Reasoning Step ${ \tt = } 2$

Please point to the third card from right to left on the cabinet.

![](../images/76bf601fb4b9e4bdf514ca8b8b42006f153543d57420cb9020efc342fa133631.jpg)  
Reasoning Step = 1

Please point out the brown object farthest from the viewer.

![](../images/2e7d9432a4a2b671bc9a562ecd59fd79272b981dd9b03cadb220e2b03720e332.jpg)  
Reasoning Step $= 2$

Please point out the white object on the cabinet farthest from the viewer.

![](../images/9bb4d42d348f3d11f5f22f39248fed07f766d55808b473ebb4b232d2922ad5bc.jpg)  
Reasoning $S t e p = 3$

Please point out the white object adjacent to the left side of the picture frame on the cabinet

# Reasoning Step = 2

![](../images/812d01d5e80fd9e44bdbac2e2cbda5e3a61ecbad4eaabb41a9475508b45f0046.jpg)

Please point out the closest red box to the blue box.

# Reasoning Step = 1

![](../images/233eb4cde66bfae36e1eced93a370d83ed01aaa9b580d889f2b167dbaf5ae8c6.jpg)  
Figure 9 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

![](../images/0b9e035eb257e663d2ae3cc6b23517e883bfda610ae6ab56cb2d6a18e4be8515.jpg)  
Reasoning Step = 2

Please point out the second closest cup to the viewer.

Please point out the stool which is farthest from the white table.

![](../images/5a17671ef46332c2e17d7b11074856b453b097b6c2b0396889517a630ee99962.jpg)

# Reasoning $S t e p = 2$

Please point out the free space on the second shelf of the wooden shelf.

![](../images/0b96cef1661433e79db16b4515a212f694cb1c46702935eeaba5c54c1f259964.jpg)  
Reasoning Step $= 3$

Please point out the free space in front of the brown object on the shelf.

![](../images/19cb3ea0ef6dbb806bf2b468c46e8ca29bb5fc5d47520d5cee70df841f56bd24.jpg)  
Reasoning Step $= 3$

Please point out the free space between toilet and shelf.

![](../images/fa48011ccbe99cc1a421d53cc6957d95e11b179df69bc3e75731b04c5c77ef57.jpg)  
Reasoning Step = 2

![](../images/2e59be75deb0c1369c634cf228d8cdc3d201660d8c797de790488bf077191ad3.jpg)  
Reasoning Step = 2

Please point out the free space on the white table at center.

![](../images/20baad4d23e0d3b40dda3ee8affe131941964b27efb7320fcca31110264c96cd.jpg)  
Reasoning Step $= 4$

Please point out the empty area to the right of the leftmost stool.

Please point out the free space in front of the blue box which is on the top of the shelf.

![](../images/377e8abf3bd5e2f50ce5cec1970a786bb4447b37c73fe944013fd5abdb02e887.jpg)

# Reasoning Step $= 4$

Please point out the free space in front of the white vase which is on the top of the shelf.

![](../images/10cf27b0134865d42ee0403d23443dec7c5aebdc79474c8fd7611b805fbec5d0.jpg)  
Reasoning Step = 2

![](../images/656deda54d364b80b6fd54f37f9bbbf130a4853eecffdc3f22f295cf352bdfce.jpg)  
Figure 10 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

# Reasoning Step = 3

Please point out the free space between the cat tree and litter box.

Please point out the free space in front of the litter box.

![](../images/d9626387faddc54790396e0567e5975e7ad0f40c9e43b987b4c28e3573458f4d.jpg)  
Reasoning $S t e p = 2$

Please point out the free space on the lowest shelf of the shelf.

![](../images/10f1b60abc7615e8b0bc642fcb57bbcd132b580b6573c56fdf9b1fd6ab757b46.jpg)  
Reasoning Step = 3 Please point out the free space between the black water bottle and the pot lid.

![](../images/d087d40e62bef6581c985a632b9d165032fe5b261a32658817868dcbbbb93fd4.jpg)  
Reasoning Step = 4

Please point out the free space between the black water bottle, the pot lid, and the scissors.

![](../images/8a9802b14a244b9cdec7c335258d26097e0335b9cb3d0e524c49941c47063288.jpg)  
Reasoning Step = 2

![](../images/5ffe28621fd80529b876525b7b73e0e3772e5bac13baf5eec2fa5707e8beab44.jpg)  
Reasoning Step = 2

Please point out the free space on the right of the farthest pot.

![](../images/34b9a672095565478dc50491a766b8eb4b996014e06fe7833244ebb770569756.jpg)  
Reasoning Step $= 4$

Please point out the free space inside the closest pot.

Please point out the free space between the black plate, blue can and closest water glass.

![](../images/6e683f1cde0f6228506e4783fd0216ac8906783ad863fe8975c4ef4befbf26bb.jpg)

# Reasoning Step $= 2$

![](../images/405041e1618f3109291f9d9dfcaaca528af0ca18fe03021914f788ca0304437e.jpg)  
Reasoning $S t e p = 3$

Please point out the free space in the top corner of the table.

Please point out the free area on the table in facing direction of the second chair from the front on the right side.

![](../images/d2a9dd38c78f24aae2f30260ec10d81dceb222af25aaefd28f33d47eb8530697.jpg)  
Reasoning Step $= 3$   
Figure 11 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free area on the table that the second chair from the front on the left side is directly facing.

![](../images/c05da10c6165bf5ed748802ffc864a2754fd3c8d90ffd4642e62258f0a287a4b.jpg)  
Reasoning $S t e p = 3$

![](../images/1adf84568dc3c72fc7b4ae7c1934260974b61cdfe29ec9c307d5402f2c710f55.jpg)  
Reasoning $S t e p = 4$

Please point out the free space between the scissors and the microwave.

![](../images/b28212aee4cc59fb3e04b75535410d76556a474ef7664b680eed1a640f1c28c2.jpg)  
Reasoning Step $= 5$

Please point out the free space between the headphones farthest from the monitor and the keyboard.

Please point out the free space between the black cloth box to the bottom-right of the monitor and the keyboard.

![](../images/12528f8ff2f2dd42069a78828a3c17820b356338bc768147b966bff03b379af0.jpg)  
Reasoning $S t e p = 3$

![](../images/c6924a90cec151b0efb34c1c823454eaf36379041db608280e70b4a4ba477838.jpg)  
Reasoning Step = 3

Please point out the free area on the table that the first chair from the front on the left side is directly facing.

![](../images/54025dc1f2ed63efb116c243434331349b1d870d52138e9b571d25f8c7304384.jpg)  
Reasoning Step $= 2$

Please point out the free area on the table that the first chair from the front on the right side is directly facing.

Please point out the free area in the top-left corner of the table.

![](../images/78191decd8e115a80b7b60b47b2e2806f624515a9f99d34df983b12567a3a90b.jpg)

# Reasoning Step $= 3$

Please point out the free space on the stovetop in front of the black pot suitable for placing another pot.

# Reasoning Step $= 3$

![](../images/de64e717be5f51888b61b7a5a7c3af05bfd8cbf587613db839ea2d44c1658503.jpg)  
Reasoning Step = 2

![](../images/ec58e71bffeb6e27b17ad5e11f9a81830803c935ed2cb2abfef288498c4dfd89.jpg)  
Figure 12 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free area between the black container for spatulas and the black object on its right side.

Please point out the free space in front of the cat tree.

![](../images/1b59768bcd84c887ce232cc23b92f987a6a252ebc98ace40379f89f00250f823.jpg)

# Reasoning $S t e p = 4$

Please point out the free space on the table between the keyboard and the viewer.

![](../images/9a4965072e397b8873b2011e9aedceb22774a4e37254e42163f92ac579128fd9.jpg)  
Reasoning Step = 4 Please point out the free space on the toilet between the blue bottle and the red can.

![](../images/daa317a851acd8507e8ef24db0c6ff304c50f9b1dd767f8ac7f7a46ea4ca5479.jpg)

# Reasoning Step = 3

Please point out the free space between the bathtub and the toilet.

![](../images/7bd5708235422a193851d26be2d7c3456eae4c94bd12e9cb35749849b537e910.jpg)  
Reasoning Step = 3

![](../images/923ca3d42ebccba6691debf9a371ce500791a98ee4aa4e34269b4f02014ce4a2.jpg)  
Reasoning Step $= 2$

Please point out the free space between the cat tree and the chair.

![](../images/936f18cd94fa7fe8e41207eb516e001aed6953ddeac437ab1808363da6dd7ebc.jpg)  
Reasoning Step $= 3$

Please point out the free space on the lowest shelf of the cat tree.

Please point out the free space between the purple vacuum cleaner and the cabinet on the left.

![](../images/b497ed114dd5a49e4f118991a34185075227308fa111568ef2a78b1c957dbe3b.jpg)

# Reasoning Step $= 2$

![](../images/b9c3632261546fa6b8ae2a2984f69c4c6a947c479fe8d6b4785ab463a81af502.jpg)  
Reasoning Step = 2

Please point out the free space below the table.

Please point out the free space on the sofa cushion.

![](../images/98669d61aed0695ef8e32f3e5f656b624a7d2fae5e4c44c2510b4fb1ec89c682.jpg)  
Reasoning Step = 3   
Figure 13 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free space between the table and the sofa.

![](../images/bbedbf71e81f9a1f0ce9efc2adbbfdd5e168ad896bcaff3b747869d7aacd2735.jpg)  
Reasoning $S t e p = 3$ Please point out the free space on the cabinet to in front of the brown vase.

![](../images/27e17638395190b84b3c2fb26282633952c32f63e2ca55929ea50212b8cdf9a0.jpg)  
Reasoning $S t e p = 4$ Please point out the free space on the cabinet between the brown vase and white bottle.

![](../images/25f132244e363ff2ad00a40367925c24668923e14a9db2dcb1854073b5ff6512.jpg)  
Reasoning Step = 3

Please point out the free spot between the blue water kettle and the orange.

![](../images/97b32ae92f59629d653e3b6984969eb35b178a21ed3fbde26966850a7e0070da.jpg)  
Reasoning $S t e p = 5$

![](../images/6b9adf1ee5d6e43f365dbd9e701c2059049e9a3d45e174bbb91ee50927076397.jpg)  
Reasoning Step = 2

Please point out the free space on the table between the speaker to the right of the monitor and the mouse.

![](../images/e4b094a901dd9dca4022ecd8e9f609354313ef0deee1c84408facf1a3be41c7e.jpg)  
Reasoning Step $= 5$

Please point out the free space on the corner of the black table that is closest to the viewer.

Please point out the free space on the right part of the table between the mouse and the picture frame.

![](../images/4a197e9c1a2220cb39ef38728d8a72452de6b9d1ea10fa4a52a01eb51f0bda45.jpg)

# Reasoning Step $= 4$

Please point out the free space on the table between the pillow and the brown bowl.

![](../images/4695e9e5418be261779016c3005604ba00639c1d54dca8ef3b758c20f2a7b3cb.jpg)  
Reasoning Step = 2

![](../images/2ffe0037ce09e33f641551e31ed2210d0745feeac53229afddd31919d000e367.jpg)  
Figure 14 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

# Reasoning Step = 3

Please point out the free area on the stovetop to the left of the pot.

Please point out the free space on the left of the brown shelf.

![](../images/cf4607d88761bf82166488040551b7129fd28cc1ce4c07e7d5bf648645a23f0e.jpg)  
Reasoning Step $= 5$

Please point out the free spot on the table to the left of the two pink dumbbells where another dumbbell can be placed at an equal distance.

![](../images/712a75d18fa5fa33c4519a5791e078610cc92d127457c83089198a870c4f8274.jpg)

# Reasoning Step = 4

Please point out the free spot, equidistant from both the blue bowl and the red bowl, and between them, where another bowl can be placed.

![](../images/b4c594d0968d07bd89ea95e0bd88b8671cc7a35ef845b45a65c9f8476a07fa63.jpg)

# Reasoning Step = 4

Please point out the free spot behind the pink cup, such that distance to the pink cup is equal to distance from the pink cup to the red bowl.

![](../images/776034ded50e4fa9054fefb7e23e331d6969a5d01a98b0a80a9b0e8fdc43f432.jpg)  
Reasoning $S t e p = 4$

Please point out the free space midway between the first and second green cups from the left.

![](../images/5079420708891471edc619810ad295c09cf9f945e54c6b6927f1f52bba21f252.jpg)  
Reasoning Step $= 2$

![](../images/b2d55d15aaf60494e144fb91b21abac941b93e7adcd8229cf446427a1d02c206.jpg)  
Reasoning Step $= 3$

Please point out the free area in the direction of the handle of the rightmost green cup.

Please point out the free space between the mouse and the green cup.

![](../images/00571b014fa32d3c7718093f98e7592422303c47de2079c83a6b73c688ce5ddc.jpg)  
Reasoning Step $= 2$

Please point out the free space in the direction of the handle of the second closest cup to the viewer.

![](../images/b88a27519b8f5d6d285775d2b21fba1e49a5b60003fc542a6eb9135f1e7b2fae.jpg)

Reasoning Step = 4 Please point out the free spot to the right of the cloth bag, where an object of the same size can be placed at an equal.

![](../images/2fb7656a0c55516d749087a7314c09be1936700236388490707e52c1eeacd33a.jpg)  
Reasoning Step = 2

Please point out the free space in the direction of the handle of the transparent glass cup.

![](../images/f18af154471e74e5dbd81683674010308c33cbbba2b428ae6ea12ec24eb688ce.jpg)

# Reasoning Step = 2

Please point out the free space in the facing direction of the purple bag.

![](../images/b3743968424f1b7f1ead8848c8b4f93eca4766cf167e3ad9432242c487e65298.jpg)  
Reasoning Step = 2

Please point out the free space in the facing direction of the orange box.

![](../images/1a1cff70700c17f435f25de7f911722e8a2b1d8fa5296279f1e09460b6d63f72.jpg)  
Reasoning Step $= 3$   
Figure 15 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free space between the red box on the left and the black box.

![](../images/208466a467142ba6b47a616ecfeacb4e2fbe7c7cc03bebada73ac8e00cb09310.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What part of a mug holds the liquid inside for drinking?

![](../images/8474bc73150c49de08542d596819a30b7a0ecd18f8c9cfeafb2e3b43c715f111.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

Which part of the yellow bottle can be removed to access its content?

![](../images/f28e36ad2211b4800ce37aeb8bb001339fe33b8c59766ee27e97187acee61e32.jpg)

# Reasoning Step = 1

What utensil can be used to scoop and transfer food to your mouth?

![](../images/89900d5acc9a67199144177873e6dd701f930fd5eb4e418410e4ec9e5b7520e4.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What part of a mug should be gripped to lift it?

![](../images/91571c139fc5e9fa876f637aa38328ca86dbca1d63e870b017d2799c7a03bd91.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What object can be used to hold and drink beverages?

![](../images/99ed4554e299d3dbc75c8ba8452440f8b07d2d08bcf536ca2d5b0081005578f2.jpg)  
Reasoning Step $\sp { \bullet 1 }$

What object can be used to input text and commands into a computer by pressing its keys?

![](../images/8db46d8ce64f23b1ad878ebea221d50444f17e70cb4f6088c70dce455cd9af74.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](../images/33321b465884e9a5eeeca152b89c410a534513a4eec319da52cbb251b829c2e1.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](../images/b3f498ce384605a6d57e23c007b963d90c94a46ccbc975d074677f74fce30db9.jpg)  
Reasoning Step $\sp { \bullet 1 }$

What part of the cabinet should be pulled to open it?

Which part of the bowl can be filled with soup or salad?

What part of a mug should be gripped to lift it?

![](../images/4ef5bcf478b1c66c268446bc68388d440597a8c75969e4d8e5d742f73e5e64d2.jpg)

# Reasoning Step = 1

![](../images/793b79b51509128360ee4961786fb3c97285b2c9f7f0ff8c43486f43397ace5d.jpg)

# Reasoning Step = 1

# Reasoning Step = 1

![](../images/d31d6dce16aafc18ac70d256e4c12702bf7d3b870043bd8819e9c7af955dd8d5.jpg)  
Figure 16 Pointing Examples of RoboBrain 2.0. The objects or their parts are pointed according to their affordances queried in the instruction.

What object can be stacked or used as a building block in a structure?

What part of pan should be gripped to lift it safely?

What part of a faucet should be turned to control the water flow?

![](../images/45974ac92d1d2f67592302af9e720aaca7bb5f301aeba1ed246ea522810223c9.jpg)  
Reasoning Step $= 2$

Mark the yellow bottle nearest to the bowl in the image.

![](../images/ef89c8415a19caad13a2587286082748dfe3c024cd11c6fdc9090d0c70fea79f.jpg)  
Reasoning Step = 2

Determine several points on plate nearest to the cup in the image.

![](../images/4000a4857c8b04d2145201b975246ce2e5839476df9375324473151e8b24920f.jpg)  
Reasoning Step $= 2$

Indicate the plate nearest to the cup in the image.

![](../images/738e0edabd72c97e6b7ba8974d6984ed3b027d852c4e605a647167cd01229edf.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](../images/765b4319069643ed3bf9b1bbe88d66b99226b275bc844438bca96221579edf37.jpg)  
Reasoning Step = 1

![](../images/6ebb79fce885d420d1c54ec6e9eedbc5afdf5d11ec451bd613d2fed47d1ba63c.jpg)  
Reasoning Step = 2

Highlight the middle mug in the image.

Find several points on the front can in the image.

Identify the bowl right to the peach in the image.

![](../images/b1517e4e1f7ad92b90da692a226cf8f0e4f1421af6bf898abffb835ced86cdb9.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

Mark the left sink in the image.

![](../images/29c0d5d3c90c9297c9442709197b95a455ad765c7820ff7b1d7115a57c8bfd77.jpg)  
Reasoning Step $= 2$

Highlight several points on the notebook near to the plastic container in the image.

![](../images/4b746af7c2970b5e03a706eed74a52d49d7d6ecb6a9f3a0208de02733ee9372a.jpg)

# Reasoning Step = 2

Spot several points on the can left next to the apple in the image.

![](../images/a6164ffc1535728e1646ebb190fa25ec3578e5b61131776fe47e82c96a057cf5.jpg)

# Reasoning Step = 3

![](../images/aa2dae4db9580866b6a6b1339ee00f958fb5f4f7d0adb33f3f18f2e845c50df8.jpg)  
Reasoning Step = 1

# Reasoning Step = 2

Identify the cup right to the bowl below the cabinet in the image.

![](../images/2d8d2baa711dd980b0b3ebe742a242102d8bc7346ecd6e81d7be6b1dccce5c87.jpg)  
Figure 17 Pointing Examples of RoboBrain 2.0. The objects referred by spatial relations or object attributes are pointed out.

Highlight the rightmost fruit in the image.

Determine several points on the plate in the right sink in the image.

![](../images/6654ef4c14e72db1f45081625cab3ada7202e410deebc370ad6af62d5e13f5f1.jpg)  
Reasoning Step $= 2$

![](../images/c6392a9fc2fd66752ed4b333ff73ac23b4cd287aa741d88a3df675b72c44a31b.jpg)  
Reasoning Step = 2

![](../images/9d2f6802874b25b9344eeb72758e2d3195146882b197de0c4d57dc1cbf8e9551.jpg)  
Reasoning Step $\sp { \bullet 1 }$

Highlight several points on the bowl right to the orange in the image.

Highlight the object right to can in the image.

Mark the rightmost object in the image.

![](../images/c89c1273f16f2af926f63f70c77bf7e0dfef3504da70fec3ef312967139597c7.jpg)  
Reasoning Step = 2

![](../images/d97c6260a768e19fc5d30648ac7569dc25a8847d31d09573f8442ec9c9b7262e.jpg)  
Reasoning Step = 2

![](../images/91f3f9dba21a4a0a31dcd8a6602a344093ddbc79c01a94cc54fc4280fb3e0b28.jpg)  
Reasoning Step $= 2$

Pinpoint the chair right to the table in the image.

Recognize several points on the box nearest to the pan in the image.

Pinpoint the bottle in front of the can in the image.

![](../images/a4520bd41eb446bcaf2fc3407b78c2060e2f21382e2467f0471a55055fa2a767.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](../images/be0c837bb6142ac97ee31b82c6a87c0a978e07f85c37af217bf4f20535be7adb.jpg)  
Reasoning Step = 2

![](../images/a78c52568b6c856fd8658ffd3a0262066f7adac312960b3da796b308bcf1973b.jpg)  
Reasoning Step $= 2$

Identify several points in the front sink in the image.

Pinpoint the box left to the yellow can in the image.

Highlight the bottle left to the lamp in the image.

![](../images/e3f59a69b500cd5d9fda4c4d57386a67f011883adc3098faec77a139e60dfc60.jpg)

# Reasoning Step = 1

![](../images/d0750768e71d16c463d61997496dde17f9b7b09616e2d3034d3d0d5b687721ea.jpg)  
Reasoning Step = 2

Indicate several points on the top drawer in the image.

Highlight the box behind the pan in the image.

![](../images/f383b56ad07f75a3c2407abe7ac50e00bfd48ba65dfa8f5422e8f6d9940f53b6.jpg)  
Reasoning Step $= 2$   
Figure 18 Pointing Examples of RoboBrain 2.0. The objects referred by spatial relations or object attributes are pointed out.

Locate several points on the building block in the bottom-right corner in the image.

![](../images/ae4e1297c2d35ba4609728f8922c793644e5601ab0035832c504f32cf49e5701.jpg)  
Reasoning Step $= 3$

Identify several spots within the vacant space that’s between the two mugs.

![](../images/c68d64a814c8b611ad32dc8e11c476e2b3ad658c913c1a9dd88b765add517eb5.jpg)  
Reasoning $S t e p = 2$

Locate several points within the vacant space positioned to the left of the yellow mustard bottle.

![](../images/b00028429695de20acf045db5aed70c1bdc0e16ef3baea37dc7780aa2a56e7f5.jpg)  
Reasoning Step $= 3$

Locate several points within the vacant area that is situated to the right of the teal plate.

![](../images/a89108e0db2f6b3e6cbf97f45b1479de6c93d87704a4867ce339ec37d3d7d619.jpg)  
Reasoning Step $= 3$

Select one or more locations within the vacant area that is in front of the mug in the middle.

![](../images/eaaab94989e37cdc2da82afa6203607f2b9f04928aa31ef3e88ec818452bdeab.jpg)  
Reasoning Step $= 3$ Locate a few places in the free space between the orange and the plastic cup.

![](../images/361fcdec395c8086a2e42bc0cda22fc6408b1493878d45634146f9049df749ab.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the back side of the stove.

![](../images/f106902f300fc2df636e41440cbb7dc6cc9e3a6a6e122b795563c1310b6c0639.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the front portion of the stove.

![](../images/9c75e9df1e1613d4874a941e75a1de70465ba403226898096ad94883196f728c.jpg)  
Reasoning Step = 2

Locate a few spots within the unoccupied space behind the mouse.

![](../images/5a26a6bf0040741de9e0b0ad162b68445243efdadb214ae5d1aa6c4b64b11779.jpg)  
Reasoning Step $= 2$

Locate a few spots within the unoccupied area inside the cabinet.

![](../images/284b2af54d1499fda91d975ff857ea3503116b4e61c7e0f90619354a53e4cdcd.jpg)

# Reasoning Step = 2

Locate several spots within the unoccupied area beneath the apple.

![](../images/bb275058c0bb1535fcfa8fee246f4f2a1381b908be3cac631ec79c7005c66067.jpg)  
Reasoning Step $= 3$

Locate several spots within the vacant area that is in front of the bowl on the left.

![](../images/6be89d46e68d375b855f6f32ad67f0a105fea897e961f80e49a8d7a8fda81daa.jpg)  
Figure 19 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

# Reasoning Step $= 2$

Locate several spots within the vacant area that is in front of the teal bowl.

![](../images/f934257620fd7f453e1690d1dab7c8c10dd8ddadedbf2d42e3a85b667e865aee.jpg)  
Reasoning Step $= 2$

Locate several points within the vacant area that lies before the plastic container.

![](../images/1c87a7a89ffba5c2581218c799ba5d9878dcbb0ed6384a21d0aba0d1ce71f709.jpg)  
Reasoning $S t e p = 2$

Locate several points within the vacant area that is in front of the blue cup.

![](../images/4a1a2e4d093a131cd3170cad6891a0542ce84c2f9f54251fcb11ca15f366b514.jpg)  
Reasoning Step $= 3$

Locate several spots within the vacan space situated above the leftmost item.

![](../images/e75d8117ad9be2b30691bd13b20f131247280ae96ab00eacceb05c6e22d169b0.jpg)  
Reasoning Step $= 2$

![](../images/9adeecfa3d13694079c5ed3d53225262b8f3cc25e4b8b448ce87a11d78eab6b8.jpg)

Pinpoint several spots within the vacant area located to the righthand of the green container.

Reasoning $S t e p = 4$ Identify several points within the vacant area that lies between the blue cup and the teal bowl on the table.

![](../images/e69fa32e2a17aee56d476abb13cc283874881e7b8421de0bb3aead34d9da360b.jpg)  
Reasoning Step $= 4$

Locate a few points within the unoccupied space that lies before the leftmost fruit on the table.

![](../images/b12a40413993ce71de829347aee23e8418590bc0936e7b718f3e23b6eb844e50.jpg)  
Reasoning Step $= 2$

Locate a few points within the vacant space to the left of the frying pan.

![](../images/1d33dac865d2a229ebbb1038c5d60e642b7023cdd9be4c4a7559bd151be223d3.jpg)  
Reasoning Step $= 2$

Locate several spots within the vacant area situated to the left side of the orange.

![](../images/be25417eeb223411bacdc2aa03e1d7f41985f84d3210beaec3bb0a70f923777b.jpg)  
Reasoning Step = 3

Locate several spots in a vacant area next to the white mug.

![](../images/1cc81ca6de542e1d13136beaec42398fb9179b68449a6b63c068450e4b7c1d1d.jpg)

# Reasoning Step $= 2$

Locate several points within the vacant space situated on the left part of the cabinet shelf.

![](../images/0f686052a0aa1b0426207d6736c2de2ff9cc99ddddd7557770bffb91fc496f85.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the front side of the table.

![](../images/33d8c7ad387f87e68a8e07ec548e590574fe7d9372dd88e86d4b76ba3a9f4827.jpg)  
Reasoning Step $= 3$ 3   
Figure 20 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

Find a few points in the free space in front of the window on the left.

![](../images/7d015b2968a5e65ddfbf19fd3dadd546b89413ee9636e26f0950515f1b81e1a8.jpg)  
Figure 21 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

# A.2 Examples for Affordance

The affordance task assesses RoboBrain 2.0’s understanding of object functionalities and interaction possibilities. For example, when asked “What part of a mug holds the liquid for drinking?” the model correctly identifies the interior of the mug as the part that holds the liquid. In another example, the instruction “Which part of a handbag can be grasped to carry it?” is accurately answered by identifying the handle of the handbag. These examples showcase the model’s ability to reason about object affordances, making it capable of understanding how objects can be interacted with in the real world. As shown in Figure 22-Figure 23, the model demonstrates its proficiency in identifying functional parts of objects and their potential uses.

![](../images/5b21a6bf6b55f0bc41bef58800e481d789a8e2ff8bf72cfbc825bdbf742ec72b.jpg)  
Figure 22 Affordance Examples of RoboBrain 2.0. The purple bounding boxes denote the actionable affordance areas for specific tasks.

![](../images/97d4eaf53f2a5149983bf6487c3c30e33550c4b13c182fcbd62978188c0edb94.jpg)

# Reasoning Step = 2

Please identify the affordance area for pouring the wine glass.

![](../images/aed2977d5b0da1552cf63ec2ecc21f534a40194105d1a539066feb85b3a8c775.jpg)

# Reasoning Step = 3

Please identify the affordance area for lying on the bench.

![](../images/cb92c079dee680f94a3394a9b0abec3908f33b4573fef94198414d53b16f2437.jpg)

# Reasoning Step = 2

Please identify the affordance area for holding the knife.

![](../images/e51ccceb693cd95c99f9a0832b23f798d21d1487080528a72eccbbce3685219e.jpg)

![](../images/a6bff0fffa697e707af85745bfe5a184274d61241c6e8af3ffeca2da4c8db38e.jpg)  
Reasoning Step = 2

![](../images/6f2487039a92aae3a5c10fd1c51c38f11a4bf9f781c37f6d7bd76ed3bfeb6b0c.jpg)

Please identify the affordance area for opening the bottle.

# Reasoning Step = 2

Please identify the affordance area for the pen to write.

# Reasoning Step = 2

Please identify the affordance area for picking up the suitcase.

![](../images/72198db976d5b7f7accc0a3bff276c6f22c9f27f47a2278db47025a1ef207421.jpg)

# Reasoning Step = 2

Please identify the affordance area for sipping the cup.

![](../images/32caf0cd855e56136514f9a96f2fecb5f35af8ca440ac3e9e10544842b0630cd.jpg)

# Reasoning Step = 2

Please identify the affordance area for holding the cup.

![](../images/1b8f6081cb764bf4d65be0dacd377a997cea71988d402c793cea2ef856f666f1.jpg)

# Reasoning $S t e p = 2$

Please identify the affordance area for opening the refrigerator.

![](../images/263c2ddabe052c8ed938e3a9a6dd9457362ac7458b7f5269311dcf5412235ccf.jpg)

# Reasoning Step = 2

![](../images/4738db6832f0c2a932c5ca6af3344064dc50fdbe23c82395888673788406d304.jpg)

Please identify the affordance area for holding the cup.

![](../images/dc90a282db2224cad89e59a1c66351afa9a7b34b5d031e42140c9c6a6cfc7c4f.jpg)  
Figure 23 Affordance Examples of RoboBrain 2.0. The purple bounding boxes denote the actionable affordance areas for specific tasks.

# Reasoning Step = 2

Please identify the affordance area for sitting on the bicycle.

# A.3 Examples for Trajectory

The trajectory task evaluates the model’s ability to predict and navigate paths based on given instructions. For instance, given the instruction “Please provide the trajectory to move the robot arm to grasp the apple,” RoboBrain 2.0 generates a smooth and efficient path for the robot arm to follow. The model’s trajectory predictions are accurate and take into account the spatial constraints and obstacles in the environment, demonstrating its proficiency in spatial and temporal reasoning for navigation tasks. As shown in Figure 24- Figure 25, the model effectively plans and executes trajectories that are both optimal and collision-free.

![](../images/d599d972d714beca9108e9dd24eedfd3373cf3c08f9dd3e7bac9cb3b5d20c55c.jpg)  
Figure 24 Trajectory Examples of RoboBrain 2.0. The blue trajectories, composed of key trajectory points, represent the model-predicted paths for task completion.

![](../images/d6af0d9ec885820005f04ce2e69aff671b45ddccdc0236f348d065ae91ea169b.jpg)  
Reasoning Step $= 2$

Please predict the key trajectory points for moving green cube to the top of yellow cube.

![](../images/50fdf987306e7eb33f8c023118c2a306e63943e5878b5f8860a1baf15a900c52.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please predict the key trajectory points for closing middle drawer.

![](../images/099d9968a6819418cc5e6813d25bb2d50a355fcab9376374a936968f0556be4f.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please predict the key trajectory points for closing top drawer.

![](../images/dd37b766d0ffc6d6f8a1e92a446f36841d40ef3b36ace7b48cc8799dd753d4fc.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$ Please predict the key trajectory points for destacking purple yellow cube.

![](../images/fb797afaca32c4fb97dec44fc1415745322f392e4b6f499093f8d298e2a7f2ea.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for making a cup of coffee with the Keurig machine.

![](../images/9bd2760d826b96d633d3fc0ff0cbbfade1d903cf1a825db1ab37c4183952aca4.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for moving red circle closer towards blue cube.

![](../images/e5bdc5f9c8aca07ae00d3cfad284810efcdb16de88a74f8781247f6d52c4f854.jpg)  
Reasoning $S t e p = 2$ Please predict the key trajectory points for moving red spoon to just below green towel.

![](../images/b465194017fdc7a5f761c84b54d2de9676c733f7bb5f507d8c911e7428f21061.jpg)

# Reasoning $S t e p = 2$

![](../images/22146398b2af83273eb4eb81cee90ed3a910665047e1c016585f3387f5943c5b.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for moving rxbar chocolate near coke can.

Please predict the key trajectory points for moving red spoon to upper right corner of table.

# Reasoning $S t e p = 2$

![](../images/3f0a234fd8692acac90ed5c1508c6c9ea190cb4e6dec3e6cbac772375bd046c7.jpg)

Please predict the key trajectory points for moving spatula onto the towel.

![](../images/9eeae4fd4d44fd01828b81f20bc491f6ecc4fc5350f9d786d8fc4715b7eff68b.jpg)

# Reasoning $S t e p = 2$

![](../images/b493d51a180d40c0b2de6d7ad43ff07fe7987bfa5a5cd659ca7f053ef6193ef2.jpg)  
Figure 25 Trajectory Examples of RoboBrain 2.0. The blue trajectories, composed of key trajectory points, represent the model-predicted paths for task completion.

Please predict the key trajectory points for moving blue triangle below the green star

# Reasoning $S t e p = 2$

Please predict the key trajectory points for picking green can from middle shelf of fridge.

# A.4 Examples for EgoPlan2

The EgoPlan2 task focuses on planning daily activities from an egocentric perspective. For instance, given the instruction “Plan the steps to prepare a cup of coffee,” RoboBrain 2.0 outlines a detailed sequence of actions, including locating the coffee machine, fetching the coffee beans, and following the steps to brew the coffee. The model’s ability to break down complex tasks into actionable steps demonstrates its proficiency in task decomposition and sequential planning. As shown in Figure 26-Figure 28, the model effectively plans and executes multi-step tasks, showcasing its capabilities in long-horizon planning and task execution.

![](../images/a4fc9add15e40b08408460b3454e1d6b9a27e4ba1d78f05d472b3c32a2efb61a.jpg)

Question: I am tasked with Move the saucepan and light up the candle. The task's progress is demonstrated in the   
provided video. What should be my next action?   
(A). move saucepan (B). close drawer (C). adjust hob (D). open drawer

Thinking: From the visual input, I observe an open drawer containing various utensils such as spoons and a whisk. My focus remains on completing the task of moving the saucepan and lighting a candle. The current step involves handling objects from this drawer. Based on this, the logical next action is to close the drawer after retrieving the necessary item (likely utensils) for cooking or handling the saucepan, ensuring workspace orderliness and safety.

The available options include: (A) Moving the saucepan, (B) Closing the drawer, (C) Adjusting the hob, and (D) Opening another drawer. Given that the drawer is already open and no further items appear necessary at the moment, closing it aligns with maintaining organization and preventing obstruction during subsequent steps.

Therefore, the most reasonable action is option (B), closing the drawer.

Answer: (B). close drawer

![](../images/d678186c89fbb824a180b94cc3042a281e2073c4c5983bd9797c9d2b6d81a941.jpg)

Question: My current task is to Wash and rinse two mugs. The task's progress is demonstrated in the provided video. What should be my next action?   
(A). put mug (B). rinse mug (C). pick up mug (D). close tap

Thinking: From the visual input, I observe that I am positioned near a sink within what appears to be a kitchen environment. My task involves washing and rinsing two mugs. Currently, I am holding one red mug and a sponge, actively cleaning it. Other items like a blue plate are visible on the counter nearby, suggesting this is part of a larger cleaning task.

The sink contains various other dishware, including another mug and additional kitchen utensils, indicating my current focus on washing multiple mugs. Given the visual context, the next logical step after cleaning the mug would be to rinse it thoroughly before placing it aside or moving on to wash the second mug.

Considering the options provided—(A) put mug, (B) rinse mug, (C) pick up mug, (D) close tap—the most immediate and sensible action aligns with rinsing the mug to ensure it is properly cleaned and ready for further steps, such as drying or placing it away.

Therefore, based on my visual analysis of the scene and the task at hand, the appropriate next action is to rinse the mug, which corresponds to option (B).

Answer: (B). rinse mug

Figure 26 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

![](../images/3540ff9fc0fe72fe929909024cda61c617f0899308776e719f297a0afda0eb50.jpg)

Question: I am in charge of Take mozzarella, open the bag, spread it onto the dough and add salt, and the video reveals the task's advancement, What is the most rational next move? (A). spread mozzarella onto dough (B). put down knife (C). take mozzarella (D). open mozzarella bag

Thinking: From the visual input, I recognize that my task involves preparing ingredients for a dish on a wooden countertop. I observe a loaf of mozzarella, which is currently being held by me, and an open packet containing additional mozzarella slices placed near the edge of the countertop. A flat, baked surface with sliced mozzarella arranged neatly appears to be the target area for spreading the cheese.

Given this setup, my next logical step is to apply the cheese directly onto the prepared flat surface with the slices of mozzarella already laid out, ensuring even distribution over the crust. This aligns with the objective of adding cheese as part of the preparation process.

# Comparing my options:

- Option (A) "spread mozzarella onto dough" directly addresses the current task at hand, as it involves positioning the cheese appropriately on the target area.   
- Option (B) "put down knife" doesn\'t fit the immediate need since the task focuses on spreading rather than cutting. - Option (C) "take mozzarella" is redundant, as I already have the mozzarella in hand.   
- Option (D) "open mozzarella bag" is unnecessary since the bag is already open, and the focus is on applying the cheese.

Thus, based on the visual scene analysis and task requirements, the most rational next move is to spread the mozzarella onto the dough.Therefore, the selected action is (A) "spread mozzarella onto dough.

Answer: (A). spread mozzarella onto dough

![](../images/9de168dab6d43efd6f8ef18adf6ec2655ac2eaf7f37a0953cfdc42ce4c9223c6.jpg)

Question: My assignment is to Preparing onion and boiling water, with the task's progress evident in the video. What should I do next? Please provide a reasonable and succinct recommendation.   
(A). take onion (B). open container (C). take knife (D). open tap

Thinking: From the visual input, the bowl containing onions is held and is the primary focus of the scene. The onion pieces are visible inside the bowl, indicating that it is already opened and ready for use. The surrounding context includes kitchen items such as a spatula, a knife on the counter, and a grater nearby, but these are not directly related to the task here. My task involves preparing an onion, which requires handling or slicing the onion contents. Given this, the logical next step would be to focus on utilizing the already accessible onion by taking it out of the bowl or using it for cooking.

The options provided include (A) taking the onion itself, (B) opening a container (which has been done), (C) taking a knife (not relevant at this stage since the onion is already handled), and (D) opening the tap (irrelevant as no water preparation is observed). Among these, option (A) aligns most directly with the immediate need to prepare the onion.

Therefore, based on the observed situation, the most reasonable next action is to take the onion from the bowl, ensuring readiness for further steps in the cooking process.

Answer: (A). take onion

Figure 27 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

![](../images/4d620b409cadd9ca5cdcf217125f8e4f577464053d5c0eda270eeb352faa6f3b.jpg)  
Question: As I am in the process of Prepare and slice more onions, with my progress visible in the video. what do you suggest as my next move? (A). take knife (B). turn knife (C). throw onion skin into bin (D). continue cutting onion

Thinking: From the visual input, the person is focused on slicing an onion on a green cutting board. The onion's white layers have been separated, and some skin or layers are visible on the cutting surface. To maintain safety and efficiency, continuing to slice the onion is logical. The person holds the onion steady, and their fingers are positioned to maintain control during further cuts.

Analyzing the current task, the objective remains to chop the onion neatly without causing any unnecessary mess. The presence of a knife suggests readiness for this action. With the onion already sliced partially, the next logical step is to complete the task by continuing to slice.

Considering the options provided:   
(A) indicates taking the knife again, which might be redundant given its current proximity.   
(B) suggests turning the knife, which may not be necessary unless there's an issue with the orientation.   
(C) proposes throwing away the onion skin, which could happen after cutting but not as part of the ongoing slicing process.   
(D) confirms the need to continue cutting the onion, aligning well with the observed task progression.

Based on the visual scene and task context, option (D) to continue cutting the onion is the appropriate choice, as it supports the process of preparing and slicing the onion efficiently.

Answer: (D). continue cutting onion

![](../images/17bd2eedaaeccaf681a96e11c8ffee9258cb04e7c2822c8e198fde819257254e.jpg)

Question: Given my responsibility to Clean the sink and counter, the video shows the progress What is the most logical   
next step?   
(A). take spoon (B). rinse spoon (C). take cloth (D). rinse cloth

Thinking: From the visual input, I am observing a kitchen environment where cleaning activities are taking place. The focus is on a sink area that contains various cleaning tools, including utensils and dishes. A spoon is being held over the sink, which is filled with water.The faucet is nearby, suggesting its use for rinsing or further cleaning actions.

Given the current task of cleaning, the spoon appears to be in the process of being washed, as it is already oriented towards the water source. The fact that this spoon is already being handled indicates readiness for further actions, such as rinsing or placing it elsewhere after cleaning. The surrounding environment includes a drying rack holding other utensils, reinforcing the notion of completing cleaning tasks sequentially.

Analyzing the sequence and progression of cleaning actions, the spoon is currently within a position conducive to rinsing under running water. This action is the next logical step before moving on to other items or utensils, to ensure all surfaces are appropriately cleaned for safe handling or placement. The other options—such as taking a cloth or additional utensils—do not align with the immediate visual context, as the spoon is already in an appropriate state for this activity.

Therefore, based on the visual analysis, the most logical next step is to rinse the spoon (Option B). This action ensures proper cleaning and aligns with the observed process within the image.

Answer: (B). rinse spoon

Figure 28 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

# A.5 Examples for Close-Loop Interaction

Close-loop interaction examples showcase RoboBrain 2.0’s ability to engage in interactive reasoning with feedback. For example, in a scenario where the model is asked to “Find a muff cup and pour coffee into it,” it not only needs to navigate and search for the mug multiple times within the task environment but also must operate the coffee machine based on feedback to complete the pouring process. This iterative process highlights the model’s capability to refine its actions based on real-time feedback, ensuring more accurate and reliable performance in interactive tasks. As shown in Figure 29-Figure 32, the model demonstrates its ability to adapt and improve its responses through iterative feedback loops.

![](../images/79d710b126705eb90857298015cd17ba3d6567da7879f140f12951e8111bb9ed.jpg)  
Task: Find a pen and place it to box, and then find a pillow, place it to arm chair.   
Figure 29 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](../images/5aa21544a6ac70a225234e28029faf546eb1346b5e247b91cb34d91f11b6529f.jpg)  
Figure 30 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](../images/04b217dc891ec5ee0e828ae5c6b57ce50da45420aab8aa9aa78a16d3e6ff4d72.jpg)  
Task: Find a muff cup and place it to sink, and then find a potato, place the potato into Fridge, and then pick up the egg from fridge and place it to Garbage.   
Figure 31 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](../images/119c9833b455e51b7a653ea3c2aea5ee27d4595a4ae8be3b236b690d1eff2274.jpg)  
Task: Find an egg and heat it with microwave, and then find a muff cup, pour coffee into it and pick it up.   
Figure 32 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

# A.6 Examples for Multi-Robot Planning

In multi-robot planning scenarios, RoboBrain 2.0 coordinates the actions of multiple robots to achieve a common goal. For example, in a supermarket scenario, the model plans the movements of multiple robots to efficiently restock shelves. The planning involves assigning specific tasks to each robot, coordinating their movements to avoid collisions, and ensuring that the overall goal is achieved in a timely manner. These examples highlight the model’s advanced capabilities in multi-agent coordination and long-horizon planning.

As shown in Figure 33, the model demonstrates its ability to orchestrate complex multi-robot activities with high precision and efficiency. In the restaurant setting (Figure 33(a)), a Unitree G1 humanoid and Agilex dual-arm robot collaborate on burger preparation and delivery for the command “I’m hungry and order a normal burger,” with RoboBrain 2.0 performing scene-aware task decomposition. The household scenario (Figure 33(b)) features a Realman single-arm and Agilex dual-arm robot executing commands like “Give me an orange and a knife.” In the supermarket (Figure 33(c)), RoboBrain 2.0 assists customers with gift selection by analyzing dimensions and bag compatibility, coordinating the Realman robot for gift placement and the Agilex executing VLA-cerebellum skills like “open the gift bag.” Please refer to RoboOS [61] for more details.

![](../images/65483ad8bd31e04858177c969a40dc30721d49fa23ff119539f7347774a485af.jpg)  
(c) Global Task: I want to give a small gift to my friend, please help me to choose one. [Supermarket]   
Figure 33 We showcase multi-robot collaboration in three scenarios: (a) Restaurant: Unitree G1 and Agilex robots prepare burgers. (b) Household: Realman and Agilex robots fetch items. (c) Supermarket: Robots coordinate gift selection and packaging.

# A.7 Examples for Synthetic Benchmarks

Synthetic benchmarks are used to evaluate RoboBrain 2.0’s performance on a variety of spatial and temporal reasoning tasks. For instance, in the BLINK benchmark, which assesses depth perception and spatial relation understanding, the model achieves high accuracy in identifying the relative positions and distances of objects. In the CV-Bench benchmark, which evaluates 3D spatial understanding, RoboBrain 2.0 demonstrates its ability to accurately process and reason about 3D scenes. These synthetic benchmarks provide a comprehensive evaluation of the model’s capabilities across different reasoning dimensions. As shown in Figure 34-Figure 35, the model consistently performs well across various synthetic benchmarks, showcasing its robust abilities.

![](../images/a0185fb659793abef858b65aea1a37853e132722c355af76d8f2a2bfbdcc0abb.jpg)  
Figure 34 CVbench Benchmark Examples of RoboBrain 2.0. The solid circle in the diagram represents the selected point.

![](../images/1ae575a6ffdc31d5976997d5f23f92994603808e220e6e9f8f2329c66ef81439.jpg)  
Figure 35 BLINK Benchmark Examples of RoboBrain 2.0. The solid circle in the diagram represents the selected object.

# B Prompts Details

This section outlines the system prompts for various spatial understanding and planning tasks assigned to a robot with advanced visual and analytical capabilities. Each task requires simulating visual-spatial reasoning, leveraging visual inputs as if directly perceiving the scene, and generating step-by-step reasoning processes within <think></think> tags, with answers in <answer></answer> tags. Reasoning is kept concise (200–500 words) and follows a five-stage process tailored to each task. Instructions emphasize direct visual-spatial language, avoiding abstract references to input data (e.g., “based on the description”) and maintaining the robot’s role.

# B.1 Spatial Understanding: Coordinates -- Pointing

The robot is tasked with identifying specific points within an image based on visual criteria, such as locating points in a vacant area on a delineated plane. The reasoning process includes:

Object Analysis Observe the object’s shape, size, and spatial relationships (e.g., a red rectangular border delineating a plane with vacant patches).

Capability Assessment Relate visual processing capabilities to identifying vacant regions and pinpointing coordinates.

Contextual Relevance Focus on the task requirement to select points within the vacant area.

Verification Ensure selected coordinates lie within boundaries and are distinct.

nt Conclusion Output coordinates as a list of tuples, justified by visual analysi

Example Prompt: Locate points within a vacant area on a plane outlined by a red border. Your answer should be formatted as a list of tuples, i.e. $[ ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , . . . ]$ , where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image.

Example Output w/ Thinking: <think> . . . </think><answer>[(296, 282), (321, 256), . . . ]</answer>.

# B.2 Spatial Understanding: Coordinates -- Trajectory

The robot predicts a sequence of key trajectory points to achieve a goal, such as reaching an object. The reasoning process includes:

Object Analysis Identify the target object’s properties and spatial relationships (e.g., a banana on a plate with potential obstacles nearby).

Capability Assessment Use joint control to plan smooth end-effector paths, avoiding obstacles.

ontextual Relevance Ensure the trajectory aligns with the goal (e.g., reaching the banana).

Verification Confirm the path avoids obstacles and reaches the target.

Trajectory Conclusion Output trajectory points as $[ [ x _ { 1 } , y _ { 1 } ] , [ x _ { 2 } , y _ { 2 } ] , \dots ]$ , justified by visual and kinodynamic analysis.

Example Prompt: You are a robot using the joint control. The task is “Reach for a banana on a plate”. Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. $[ [ x _ { 1 } , y _ { 1 } ] , [ x _ { 2 } , y _ { 2 } ] , \dots ]$ , where each tuple contains the x and y coordinates of a point."

Example Output w/ Thinking: <think> . . . </think><answer>[[116, 114], [153, 97], . . . ].</answer>.

# B.3 Spatial Understanding: Bounding Box -- Affordance

The robot identifies an affordance area for interaction with an object, such as grasping a handle. The reasoning process includes:

Object Analysis Describe the object’s shape, size, and material properties (e.g., a blue coffee mug with a handle, inferred as ceramic from sheen).

Capability Assessment Assess gripper compatibility with the object’s features (e.g., handle size vs. gripper opening).

Contextual Relevance Align with the task goal (e.g., preparing coffee by grasping the mug).

erification Confirm the affordance area suits the interaction and is within reach.

Affordance Conclusion Output the bounding box as $[ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x } ]$ , justified by visual compatibility.

Example Prompt: You are a robot using the joint control. The task is “hold a coffee mug”. Please predict a possible affordance area of the end effector.

Example Output w/ Thinking: <think> . . . </think><answer>[915, 408, 1109, 533].</answer>.

# B.4 Spatial Understanding: Freeform Q&A -- General Spatial Analysis

The robot answers questions about spatial relationships or action outcomes based on one or more images. The reasoning process includes:

Scene Perception Detail prominent features and their spatial arrangement (e.g., a metallic gripper above a green book on a shelf).

Task Objective Interpretation Clarify the question’s focus (e.g., predicting the outcome of a gripper’s trajectory).

Focused Visual Analysis Examine relevant scene elements or project actions (e.g., a yellow trajectory toward a lower shelf).

Relational Reasoning Synthesize observations to form a hypothesis, evaluating provided options.

Conclusion Derivation Output the answer, justified by visual evidence and logical reasoning.

Example Prompt: Predict the outcome of a gripper following a yellow trajectory. Options: (A) place book on lower shelf; (B) place book on upper shelf.

Example Output w/ Thinking: <think> . . . </think><answer>(A)</answer>.

# B.5 Temporal Understanding: Long-horizon Planning

The robot determines the next action in a task (e.g., cooking) based on a sequence of images and the current view. The reasoning process includes:

Task Progress Analysis Interpret completed actions from the sequence (e.g., onions peeled and sliced on a cutting board).

Current Scene Analysis Describe the current view’s objects and state (e.g., frying pan on hob, oil container nearby).

ontextual Relevance Align with the task goal (e.g., cook onions by preparing the pan).

Action Option Evaluation Assess options for suitability (e.g., pour oil vs. peel onion, considering onions are already prepared).

Next Action Conclusion Output the next action, justified by visual evidence and task flow.

Example Prompt: Prepare and cook onions; choose the next action (options: pour oil, turn up hob, etc.).   
Example Output w/ Thinking: <think> . . . </think><answer>Pour oil.</answer>.

# B.6 Temporal Understanding: Closed Loop Conversation

The robot answers a question within a conversation history, leveraging prior visual inputs and responses. The reasoning process includes:

Task Progress Recall Recap previous actions and their outcomes (e.g., opened the fridge to access ingredients).   
Initial Analysis Focus on current visual input relevant to the question (e.g., a coffee machine on the countertop).   
Contextual Relevance Align with the current task goal (e.g., flipping the coffee machine switch).

Action Option Evaluation Assess options for logical progression based on history and current state.

Next Action Conclusion Output the action, justified by visual evidence and conversation context.

Example Prompt: The task is “Flip the coffee machine switch after opening the fridge.” After you have finishe <action $>$ , you can see <image $>$ , and the feedback of final action is xxx. What is your next action?

Example Output w/ Thinking: <think> . . . </think><answer>Toggle on Coffee Machine.</answer>.

# B.7 Temporal Understanding: Multi-Robot Planning

The robot coordinates actions with other robots to achieve a common goal, devided into global task decomposition and agent-based tool-calling.

Example Prompt for Global Task Decomposition: Please Refer to Figure 36.   
Example Prompt for Agent-based Tool-calling: Please Refer to Figure 37.   
Example Output w/ Thinking: <think> . . . </think><answer>Graph of TaskFlow</answer>.

# System Prompt for Global Task Decomposition

# # You are a robotics expert specializing in task decomposition.

Your role is to decompose tasks into subtasks based on the task description and assign them to different robots for execution.

# ## Example 1:

Current Robot: realman_1, singlearm_1, doublearm_1   
Current Task: All the robots go to the table and bring an apple to the fridge respectively.   
Your answer:   
\`\`\`json   
[ {{'robot_id': 'realman_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 " \}$ , {{'robot_id': 'singlearm_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 \%$ , {{'robot_id': 'doublearm_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 " \}$ ,

# ## Example 2:

Current Robot: realman_1, doublearm_1   
Current Task: realman take the basket from table_1 to table_2, then doublearm take the apple into basket in table_2, then realman take the   
basket back to table_1.   
Your answer:   
\`\`\`json {{'robot_id': 'realman_1', 'task': 'bring the basket from table_1 to table_2.', 'task_order': '0'}}, {{'robot_id': 'doublearm_1', 'task': 'pick an apple into the basket.', 'task_order': '1'}}, {{'robot_id': 'realman_1', 'task': 'bring the basket from table_2 to table_1.', 'task_order': '2'}},   
\`\`\`

## Note: 'subtask_order' means the order of the sub-task.

If the tasks are not sequential, please set the same 'task_order' for the same task. For example, if two robots are assigned to the two tasks, both of which are independance, they should share the same 'task_order’. If the tasks are sequential, the 'task_order' should be set in the order of execution. For example, if the task_2 should be started after task_1, they should have different 'task_order'.

# # Now it's your turn !!!

We will provide more scenario information and robot information. Based on the following robot information and scene information, please break down the given task into sub-tasks, each of which cannot be too complex, make sure that a single robot can do it. It can't be too simple either, e.g. it can't be a sub-task that can be done by a single step robot tool. Each sub-task in the output needs a concise name of the sub-task, which includes the robots that need to complete the sub-task. Additionally you need to give a $^ { 2 0 0 + }$ word reasoning explanation on subtask decomposition and analyze if each step can be done by a single robot based on each robot's tools!

# ## The output format is as follows, in the form of a JSON structure:

{ "reasoning_explanation": xxx, "subtask_list": [ {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, ]   
}

# ## Robot Information:

Robot in Scene: {Robot List}. Robot positional states:{Robotic Memory}. Robot available tools:{Robotic Tool Libraries

# ## Scene Information:

# {Scene Graph}

# The task to be completed is:{Global Task}. Your output answer:

Figure 36 Prompt for global task decomposition.

# System Prompt for Agent-based Tool Calling

# # You are an expert assistant who can solve any task using tool calls.

You will be given a task to solve as best you can. To do so, you have been given access to some tools.

The tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".   
This Action/Observation can repeat N times, you should take several steps when needed.

You can use the result of the previous action as input for the next action. The observation will always be a string: it can represent a file, like "image_1.jpg". Then you can use it as input for the next action. You can do it for instance as follows:

Observation: "image_1.jpg"

Action:

{{ "name": "image_transformer", "arguments": {{"image": "image_1.jpg"}}}}

To provide the final answer to the task, use an action blob with "name": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:

Action:

{{"name": "final_answer", "arguments": {{"answer": "insert your final answer here"}}}} "arguments": "image.png"}}

# # Here are a few examples using notional tools:

Task: "What is the result of the following operation: $5 + 3 + 1 2 9 4 . 6 7 8 ?$ “

Action:

$\{ \{$ "name": "python_interpreter", "arguments": {{"code": "5 + 3 + 1294.678"}}}} Observation: 1302.678

Action:

{{ "name": "final_answer", "arguments": "1302.678" }} # Above example were using notional tools that might not exist for you. You only have access to these tools

{%- for tool in tools.values() %}   
- {{ tool.name }}: {{ tool.description }}   
Takes inputs: {{tool.inputs}}   
Returns an output of type: {{tool.output_type}}   
$\{ \% \}$ - endfor %}

# # Here are the rules you should always follow to solve your task:

1. ALWAYS provide a tool call, else you will fail.   
2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.   
3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.   
If no tool call is needed, use final_answer tool to return your answer.   
4. Never re-do a tool call that you previously did with the exact same parameters.

# # Now Begin! If you solve the task correctly, you will receive a reward of \$1,000,000.

Task: {Subtask}   
The tool you have used are: {Tool-Calling_History}   
Observation: {Observation}   
Your next action is:

Figure 37 Prompt for agent-based tool calling.