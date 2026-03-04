# APPENDIX

## A. Contributions

The project was started by HW and DD. DD, HW, and JTS designed the video encoder and short-horizon memory system. KP designed the long-horizon memory system. MT, KP, HW, KV, SN, and ME designed the evaluation suite and performed experimental evaluations, including model post-training. DD, KP, AR, and QV developed the pre-training recipe and data mixture, and ran large-scale model training. HW, KD, KS, and MT developed the infrastructure for long-context VLA training. JT optimized inference of the MEM VLA. SL and CF provided advice throughout the project. MT, KP, DD, BI, KV, SN, SL, and CF worked on writing, illustrations, and the video.

> 💡 **作者分工**：这是一个大团队项目。Video encoder 由 Danny Driess (DD)、Homer Walke (HW) 和 Jost Tobias Springenberg (JTS) 设计；Language memory 由 Karl Pertsch (KP) 设计。Chelsea Finn (CF) 和 Sergey Levine (SL) 提供全程指导。

## B. Task Details

### 1) Long-horizon Tasks (Section IV-A):

**a) Preparing the items for a recipe:** The goal of the task is for the robot to take out all of the items for a recipe. The items are placed in various, randomized locations throughout the kitchen (fridge, cabinet, drawers, stove ...). The policy's prompt specifies which items should be assembled, where they should be placed (countertop, stove, sink), and their respective initial location in the kitchen. The robot will receive 1 point per item retrieved and placed in the requested position. Most recipes consist of 6 to 7 points. We collected data on over 40 recipes with a large variety of objects, scenes, appliances, prompts. The evaluation is done on a variety of unseen scenes, with seen recipes.

> 💡 **评分机制**：每个正确取出并放置的食材 +1 分。大多数食谱包含 6-7 个子项。训练了 40+ 个食谱，在新场景中评估已知食谱。

**Example prompts:**

**Mashed potatoes:**
> Take out supplies for the recipe. Remove the lid from the pot on the stove and place the lid on the countertop to the left of the sink, and pick up the pot and place it in the sink. Get the potatoes, butter, milk from the fridge and place them on the countertop to the left of the sink. Get the masher from the top drawer to the left of the sink and place it on the countertop to the left of the stove.

**Fried rice:**
> Take out the supplies for the recipe. Get the bag of rice and the spam from the cabinet on the bottom right of the dishwasher and place them on the countertop to the left of the sink. Get the soy sauce from the fridge and place it on the same section of the countertop to the left of the sink. Get the pan from the bottom cabinet to the right of the dishwasher and put it on the stovetop. Get the spatula from the top drawer to the right of the dishwasher and place it on the pan on the stovetop.

**Pizza:**
> Take out the supplies for the recipe. Get the baking tray from the bottom cabinet on the left of the dishwasher and place it on the countertop to the left of the sink. Get the pizza dough, package of pepperoni, and bag of cheese from the fridge and place them on the countertop to the left of the sink. Get the dough roller from the top drawer to the right of the stove and place it on the countertop to the left of the sink.

> 💡 **Prompt 很长且详细**：注意这些 prompt 的长度和精确度——指定了每个物品的位置（"冰箱"、"洗碗机左下方的柜子"）和放置目标（"水槽左侧的台面"）。这本身就对模型的语言理解能力提出了很高的要求。

**b) Cleaning the kitchen:** This task requires cleaning a kitchen by wiping the countertop with the sponge, drying the countertop with a paper towel, storing any food items left in the countertop inside the fridge, putting all the dishes from the dishrack into the cabinets, and washing all of the dishes from the sink. The scoring consists of +1 point per subtask done and on average episodes consist of 8 subtasks.

**Clean kitchen example prompt:**
> Clean up the kitchen. Wipe the countertop with the sponge and dry it with the paper towel and throw the towels in the trash can under the sink. Put the mustard in the fridge. Put the dishes in the bottom cabinet on the left of the fridge. Wash the blue plate and black plate and place them in the dishrack.

### 2) In-Context Adaptation Tasks (Section IV-B):

**a) Chopstick Pick Up:** The policy is tasked with picking up a chopstick on a variable height table. During data collection, the chopstick is placed at random locations on the table with random table heights in the upper half of the table height range. We collect human interventions whenever the policy mis-grasps, and then train both memory and non-memory policies on this data and evaluate them with the chopstick randomly placed on the table at the lowest table height setting. The final policies are scored +1 for picking up the chopstick and +1 for placing it in the bin, and success is a score of 2.

> 💡 **OOD 测试设计**：训练时桌高在上半范围，测试时设为最低高度 → 强制模型在 OOD 条件下依赖 in-context adaptation 而不是记忆固定策略。

**b) Open Refrigerator:** The goal of the task is to open a fridge in front of the robot. The fridge does not have obvious visual cues indicating which side the door hinge is on, leading policies to often attempt to open it in the wrong direction. We collect human interventions whenever the robot opens the fridge in the wrong direction. For evaluation, an episode is defined as successful if it takes ≤4 grasps to open the door, to test for intentional strategy switching rather than repeated random sampling.

> 💡 **≤4 次尝试**的约束排除了"随机成功"的可能——要求模型真正学会策略切换。

### 3) Analysis Experiments (Section IV-C):

**a) Three-way swap mugs:** Place three coffee mugs under a coffee machine sequentially. Two mugs start at random positions on the table and a third mug starts under the coffee machine. The scoring consists of obtaining +1 point for each mug that goes on the coffee machine without tipping and without repeating mugs.

> 💡 **需要空间记忆**：记住哪些杯子已经放过，当前杯子的位置。

**b) Find object:** Retrieve the hidden object in a cabinet with four drawers. A person places the object in a random drawer. The robot needs to remember which drawer the person placed the object into, then open the correct drawer and retrieve the item. One point for success without opening any incorrect drawer.

> 💡 **纯 partial observability 测试**：看到人放东西 → 记住 → 开对的抽屉。无记忆的 VLA = 25% 随机猜。

**c) Unpack groceries:** Retrieve all items from a grocery bag without missing any. The inside of the bag is not observable from the third person camera. The robot will need to remember how many objects are left in the bag from its past wrist camera observations.

**d) Scoop coffee:** Put exactly two scoops of coffee beans in a grinder. The robot needs to remove the lid, pick up a coffee scoop, put two scoops from the coffee package into the grinder, then put the lid back on. Success if exactly two scoops are added and the lid is put back on.

> 💡 **计数能力测试**：记住已经加了几勺。

**e) Grilled cheese:** Prepare a grilled cheese sandwich. Assemble in pan, wait for cooking, flip, wait for other side, plate. +1 point for assembling, cooking correct time (30s-3min per side), flipping, and plating.

> 💡 **时间记忆测试**：记住每面煎了多久（30秒到3分钟之间）。

**f) Window cleaning:** Wipe a window door. Spray windex, rip paper towel, wipe whole window, throw paper towel in trash. Must remember which parts have been wiped.

> 💡 **空间覆盖记忆**：记住窗户哪些部分已经擦过。

**g) Table bussing:** Sort 12 objects from a tabletop into a bussing bin (utensils/tableware) or trash can (plastic/bottles/paper). +1 per correctly placed item.

**h) Shirt folding:** Fold a shirt on a tabletop into a rectangle without seams sticking out or excessive wrinkles.

**i) Clean up counter:** Clean items from a counter into a drawer. Open drawer, place items, close it.

**j) Make bed:** Tidy a bed by straightening blanket and placing pillows at the head.

**k) Kitchen cleanup:** Place multiple plates, bowls, and cutting board from counter into sink.

**l) Batch folding:** Take clothing from hamper, flatten, fold, stack. May contain t-shirts and shorts.

**m) Box building:** Fold a flattened cardboard cutout into a box. Requires precise bimanual manipulation.

> 💡 **任务多样性**：13 个任务覆盖了单臂、双臂、移动机器人；从简单记忆（开抽屉）到精细操作（折衣服、组装纸箱）；从秒级记忆（数咖啡勺）到分钟级记忆（做三明治）。评估非常全面。

## C. Video encoder with Space-Time separable attention

We describe how we can adjust a layer in a given ViT image encoder to instantiate our video-encoding scheme.

Let $\mathbf{z}_{p,t}^{l-1}$ denote the input embeddings to layer $l$ for spatial patch $p$ and timestep $t$ (where $t \in [-K, 0]$). We first modify the embedding by adding a sinusoidal position embedding based on $t$, denote the output of this step with

$$
\hat{\mathbf{z}}_{p,t}^{l-1} = \mathbf{z}_{p,t}^{l-1} + e(t),
$$

where $e(t)$ denotes the sinusoidal position embedding and we set the boundary condition $e(0) = 0$.

> 💡 **时间位置编码**：$e(0) = 0$ 这个边界条件保证了 K=1（单帧）时编码结果与原始 ViT 完全一致。

We then re-use the ViT's standard query key and value projections which in our case are given as

$$
\begin{array}{r}
\mathbf{q}_{p,t}^{l,a} = W_Q^{l,a} \mathbf{LN}(\hat{\mathbf{z}}_{p,t}^{l-1}), \\
\mathbf{k}_{p,t}^{l,a} = W_K^{l,a} \mathbf{LN}(\hat{\mathbf{z}}_{p,t}^{l-1}), \\
\mathbf{v}_{p,t}^{l,a} = W_V^{l,a} \mathbf{LN}(\hat{\mathbf{z}}_{p,t}^{l-1}),
\end{array}
$$

where $a$ indexes the attention head and LN denotes a layer-norm implementation (we use RMSNorm but this choice is dependent on the ViT we start from).

> 💡 **复用 ViT 的 QKV 投影**：不引入新参数，直接用预训练 ViT 的权重。

Next, we can define the general attention mechanism (ignoring normalization for brevity of notation) as the softmax over the query and key outer product:

$$
\alpha_{p,t}^{l,a}(S = \{1, \ldots N\}, \mathcal{T} = \{1, \ldots, T\})[\hat{\mathbf{z}}] = \mathrm{SM}\Big((\mathbf{q}_{p,t}^{l,a})^T \cdot \big[\mathbf{k}_{0,0}^{l,a} \{\mathbf{k}_{p',t'}^{l,a}\}_{p' \in S, t' \in \mathcal{T}}\big]\Big),
$$

where SM denotes the softmax operation and $S = \{1, \ldots N\}$ and $\mathcal{T} = \{1, \ldots, T\}$ denote the space and time indices we want to perform attention over respectively. With this general definition we can define the space only attention mechanism as instantiating $\alpha_{p,t}^{l,a}(S = \{1, \dots N\}, \mathcal{T} = \{\})$ and the time only attention mechanism as instantiating $\alpha_{p,t}^{l,a}(S = \{\}, T = \{1, \dots, T\})$. Thus the attention mechanism used in our video encoder is precisely given as

$$
\alpha_{p,t}^{l,a}(S = \{1, \ldots N\}, T = \{\})[\alpha_{p,t}^{l,a}(S = \{1, \ldots N\}, T = \{1, \ldots, T\})[\hat{\mathbf{z}}]].
$$

And thereafter we follow the standard computation of a transformer layer, i.e. we compute the outputs of the transformer by first combining attention weights $\alpha$ with values $\mathbf{v}$ and passing the corresponding outputs to a MLP.

> 💡 **Factorized Attention 的数学形式**：
> - 先做 spatial+temporal joint attention（内层）
> - 再做 spatial-only attention（外层）
> - 这不是简单的先 spatial 后 temporal，而是一个嵌套结构
> - 复杂度从 $O(n^2K^2)$ 降到 $O(Kn^2 + nK^2)$
>
> 💡 **与 TimeSformer 的区别**：TimeSformer 用的是 divided space-time attention（交替做），MEM 用的是嵌套结构。这种设计让 temporal 信息能更深地融入 spatial 表示中。
