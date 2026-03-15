# RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation

Chao $\mathrm { Y u } ^ { 1 2 }$ , Yuanqing Wang34, Zhen $\mathrm { G u o } ^ { 3 }$ , Hao Lin3, Si ${ \mathrm { X u } } ^ { 3 }$ , Hongzhi Zang1, Quanlu Zhang3, Yongji $\mathrm { \sf W u } ^ { 5 }$ , Chunyang $\mathrm { Z h u } ^ { 3 }$ , Junhao $\mathrm { H u } ^ { 3 }$ , Zixiao Huang1, Mingjie Wei2, Yuqing Xie1, Ke Yang2, Bo $\mathrm { D a i } ^ { \dag }$ , Zhexuan ${ \mathrm { X u } } ^ { 1 }$ , Jiakun $\mathrm { D u } ^ { 1 }$ , Xiangyuan Wang4, Xu $\mathrm { F u } ^ { 3 }$ , Letong $\mathrm { S h i } ^ { 3 }$ , Zhihao Liu2, Kang Chen42, Weilin $\mathrm { L i u } ^ { 3 }$ , Gang $\mathrm { L i u ^ { 1 } }$ , Boxun $\mathrm { L i } ^ { 3 }$ , Jianlei Yang6, Zhi Yang4, Guohao Dai73, Yu Wang1∗

1Tsinghua University 2Zhongguancun Academy 3Infinigence AI 4Peking University 5UC Berkeley 6Beihang University 7Shanghai Jiaotong University ∗Corresponding Author: yu-wang@tsinghua.edu.cn GitHub Repo: https://github.com/RLinf/RLinf

# Abstract

Reinforcement learning (RL) has demonstrated immense potential in advancing artificial general intelligence, agentic intelligence, and embodied intelligence. However, the inherent heterogeneity and dynamicity of RL workflows often lead to low hardware utilization and slow training on existing systems. In this paper, we present RLinf, a high-performance RL training system based on our key observation that the major roadblock to efficient RL training lies in system flexibility. To maximize flexibility and efficiency, RLinf is built atop a novel RL system design paradigm called macro-to-micro flow transformation (M2Flow), which automatically breaks down high-level, easy-to-compose RL workflows at both the temporal and spatial dimensions, and recomposes them into optimized execution flows. Supported by RLinf worker’s adaptive communication capability, we devise context switching and elastic pipelining to realize M2Flow transformation, and a profiling-guided scheduling policy to generate optimal execution plans. Extensive evaluations on both reasoning RL and embodied RL tasks demonstrate that RLinf consistently outperforms state-of-the-art systems, achieving $1 . 0 7 \times \sim 2 . 4 3 \times$ speedup in end-to-end training throughput.

# 1 Introduction

The rapid progress of large language models (LLMs) has reached a point where further scaling model alone yields diminishing returns. To push intelligence beyond pretraining, reinforcement learning (RL) has emerged as a crucial paradigm. Recent advances such as RLHF [6, 33], GRPO [42], and RL for embodied agents [19, 20] and Deep Research [30, 59] all rely on RL to align LLMs with human preferences, improve reasoning, and enable autonomous interaction with complex environments. OpenAI and others predict that RL workloads will soon consume more computational resources than LLM pretraining [32], making RL training efficiency the most critical system concern.

However, efficient RL training for various scenarios such as reasoning, agentic and embodiment at the scale of modern large models is challenging, which combines highly heterogeneous components with diverse workload characteristics and resource demands, such as LLM generation, inference and training, reward models, critic models, agent tooling and embodied environment simulators. For instance, LLM training consumes more accelerator (e.g., GPU) memory than LLM generation and inference (prefill-only generation) to maintain gradients and optimizer states, while LLM generation shows high dynamicity in response lengths, leading to low accelerator utilization. Moreover, components like LLM training support diverse parallelization strategies (e.g., data, tensor, pipeline parallelism), whereas others scale only via instance replication and may yield computation workloads distinct from common tensor computation in LLM, e.g., embodied simulators [8,47] that require CPU for physics simulation and GPU graphics pipeline for 3D rendering.

Single execution mode of existing RL training systems fails to capture this diversity, leading to suboptimal efficiency. Collocated execution, where components sequentially occupy accelerators [44], suffers from the long-tail problem due to varying generation lengths, leaving accelerators idle. Disaggregated pipelining, where components run concurrently on separate accelerators with pipelining [11], mitigates the long-tail issue but introduces memory and computation imbalance (§2.2). Neither mode is universally optimal. Many RL workloads demand hybrid scheduling of the components to maximize efficiency, i.e., mixing collocation and pipelining in a more flexible way. However, supporting such flexible execution modes for a single programmed workflow is a significant challenge, as they often require different program structures and communication patterns. Also, identifying the right scheduling for a given workflow usually requires considerable manual tunning.

In this paper, we present RLinf, an RL training system that maximizes system flexibility to achieve efficient execution of a logically programmed RL workflow. At its core is a new paradigm called macro-to-micro flow transformation (M2Flow), i.e., macro logical flow with micro execution flow, which decouples the logical programming of RL workflows from their physical execution planning. With M2Flow, developers program RL workflows imperatively, using a natural programming interface to define how components communicate and synchronize at a coarse granularity. RLinf then automatically transforms this logical flow into a fine-grained execution plan tailored to the workload and hardware at both spatial and temporal dimensions. By decoupling program semantics from execution modes, M2Flow lets developers preserve clean, intuitive workflows while the system explores a vast scheduling space, including temporal multiplexing, spatial pipelining, and hybrid scheduling.

RLinf achieves this through three key mechanisms. First, a worker abstraction that encapsulates each RL component for flexible placement, and built-in adaptive communication that allows direct and efficient communication between components regardless of worker and data placement. Second, elastic pipelining and automatic context switching that enable M2Flow transformation and expand the scheduling space, achieving pipeline granularity tunning and temporal accelerator multiplexing without modifying the logical workflow. Third, a profiling-guided scheduling policy that automatically selects efficient execution modes, balancing utilization across heterogeneous components. Together, these capabilities deliver both high flexibility, efficiency and programmability.

We implement RLinf using Ray for cluster management and worker process launch on remote nodes. Apart from the core mechanisms, RLinf also provides rich support for common RL components, algorithms, and models to facilitate RL workflow programming (§4). We extensively evaluate RLinf on both reasoning and embodied RL training across diverse models (e.g., Qwen2.5-1.5/7/32B [35], Qwen3 [50], Open-VLA [19], OpenVLA-OFT [18]) and varying scales. Our evaluation shows that RLinf improves end-to-end throughput (i.e., tokens per second) by up to $1 . 7 \times$ compared to stateof-the-art RL training systems in reasoning RL, and by up to $2 . 4 3 \times$ in embodied RL. We have open-sourced the full codebase of RLinf to accelerate RL innovations in LLM era.

This paper makes the following contributions:

• Analysis of representative RL algorithms and scenarios, identifying characteristics of modern RL workflows and highlighting inefficiencies in current systems. • A novel RL system paradigm M2Flow that decouples logical workflow programming from execution planning, enabling intuitive programming with flexible execution. • A cohesive set of mechanisms, i.e., worker abstraction, elastic pipelining, context switching, adaptive communication, and a profiling-guided scheduler, that jointly realize M2Flow and enable efficient RL training. Comprehensive evaluations of RLinf across various RL workloads show that RLinf significantly improves efficiency and flexibility compared to existing approaches.

![](images/1d112a959da308ae0ec0304d40f1efecefbdeeb104ee551c03f910994027ebd0.jpg)  
Figure 1: Diverse RL workflows in various scenarios.

# 2 Background and Motivation

# 2.1 RL Workflows in LLM Era

Various RL Algorithms and Scenarios. With the slowdown of scaling gains in large language models, reinforcement learning has become increasingly important for advancing LLM intelligence. Unlike traditional RL, RL in the LLM era often involves multiple LLMs in the loop. Given the scale of modern models (tens to hundreds of billions of parameters), fitting RL training into available accelerators (e.g., GPUs) is already challenging. Figure 1 illustrates four representative RL workflows across different scenarios and algorithms.

The simplest is GRPO [42], an RL algorithm designed to reduce reliance on reward models. It involves a single LLM that generates multiple responses, e.g., 8, for a query (i.e., Generation), computes logarithmic probabilities for these responses (i.e., Inference), and uses the results as training data to update the same model (i.e., Training). The updated weights are then synchronized back for inference and generation, completing one training iteration.

In contrast, the RLHF [33] workflow adopts PPO [41], resulting in a more complex design involving four LLMs. The actor model serves as the core policy, generating responses to queries. The reference model remains fixed to constrain the actor from drifting too far from its initialization. The reward model assigns scalar rewards to generated responses, while the critic model estimates expected rewards to stabilize training. Actor and critic are trainable, whereas reference and reward models are frozen. These components interact closely, as shown in the figure.

Beyond algorithms, RL workflows also vary by application scenario. In embodied intelligence [19, 20], RL relies on simulators that simulates the physical world. An LLM interacts with the simulator by generating actions and receiving feedback, producing trajectories that serve as training data. Similarly, in Deep Research [30, 59], RL improves model performance through interaction with a search server that retrieves online information. The resulting rollout results are fed into training, while inference follows the GRPO workflow.

![](images/ed06220cf3c3cea3e28ca74629152e7f224fc27ffff42a755c155005152a60aa.jpg)  
Figure 2: The distribution of response lengths and the number of unfinished responses over time in the generation phase of a math RL experiment.

Characteristics of RL Workflows. RL workflows consist of heterogeneous components with distinct demands on GPU memory, computation cores, accelerator types, and parallelization strategies. For example, training requires substantially more GPU memory than inference to maintain gradients and optimizer states. Unlike training, generation often underutilizes GPU cores, as its matrix and vector multiplications are bottlenecked by memory bandwidth. Some components (e.g., simulator) run on CPUs, or use GPUs for non-tensor computations (e.g., 3D rendering). Parallelization also differs significantly, e.g., LLM training exploits data, tensor, and pipeline parallelism, whereas simulators typically scale only through multiple instances. Maximizing overall utilization across such heterogeneous components is a great challenge.

Further, RL workflows exhibit complex dependencies, primarily through data flow and weight updates. Data flow can occur at different granularity, e.g., per response between generation and inference, or at least a micro-batch of responses between generation and training. Some workflows even introduce cyclic data flows, such as in embodied RL and Deep Research (Figure 1), which further complicates coordination. Weight updates, in contrast, act as barriers that synchronize generation and training. These complex dependencies require more fine-grained scheduling.

# 2.2 Inefficiencies in Diverse RL Workflows

We analyzed different RL workflows to identify the source of inefficiencies as follows.

Dynamicity in Rollout Wastes Computation. The rollout phase is inherently dynamic. Lengths fluctuate across responses of the same query, and even more so across different queries. Embodied tasks such as grasping can take different numbers of steps. In deep research, generating a report may involve varying numbers of search interactions. Since rollouts are executed in batches, these variations create a long-tail problem where a few slow queries block the entire phase from proceeding to inference or training. The problem commonly exists in the collocated mode (e.g., veRL [44]), where generation, inference, and training sequentially share GPUs by swapping between CPU and GPU memory. We conducted an experiment of a 7B math reasoning RL training [35] on 8 nodes with 8 H100 GPUs each, as shown in Figure 2. The number of unfinished responses quickly shrinks to less than $5 \%$ , where a very small set of long-tail responses stalls the generation, leaving many GPUs underutilized or idle. Scaling out with more GPUs worsens the problem as idle time grows.

![](images/df9701cc8d87758ddfe9e3235e134b4a5616282d2dab505aeb4156cc6b80564a.jpg)  
Figure 3: The execution time of generation and simualtor with different batch sizes respectively, batch size in simulator is the number of environments.

Pipelining alleviates this by allocating fewer GPUs to generation and leaving the rest for inference and training. In this setup, inference and training start once partial samples are ready. However, pipelining introduces its own inefficiency since inference and training must wait for the first batch to be generated. Thus, neither approach is universally optimal, and supporting both collocated mode and pipelining within a single framework remains a significant challenge.

# Simple Execution Modes Cannot Fit Diverse Components.

Collocated and pipelined modes are two extremes, i.e., all components on the same GPUs versus fully disaggregated GPUs. Some RL workloads, however, do not fit neatly into either mode. Their diverse component characteristics require more flexible orchestration. Take embodied RL (Figure 1) as an example. Figure 3 shows computation profiles of generation and simulator. The execution time of simulator increases slightly with the number of environments and its GPU utilization remains low (i.e., ${ < } 2 4 \%$ ), while its memory usage grows linearly with the number of environments. In contrast, generation scales linearly in both runtime and memory with batch size, while keeping GPU cores highly utilized (i.e., $> 7 0 \%$ ). Training consumes more memory, but its execution time is only one-third as long as generation’s.

This profile rules out simple execution modes. The simulator should scale with as many parallel environments as possible to reduce its total runtime. However, this prevents collocation with generation due to memory contention. A better choice is running on disaggregated GPUs with pipelining for higher efficiency. Training, in contrast, would waste compute if fully disaggregated, so it should share GPUs. After rollout, simulator and generation are swapped to CPU, and training takes over the GPU. The result is a hybrid mode that combines collocation and pipelining to balance efficiency.

![](images/53354ee4edab854608d4884e5ec12a828f69bd0af9168909b1ce2cc0531e2e60.jpg)  
Figure 4: The architecture of RLinf.

Identifying Suitable Orchestration is Challenging. The orchestration of the hybrid mode depends on the analysis of the components. However, finding the most suitable orchestration for a given RL workflow is challenging, as the characteristics are diverse and the dependencies are complex. Manually enumerating options is tedious, time-consuming, and risks overlooking better choices. Moreover, no clear guidelines exist to identify the most suitable orchestration.

# 2.3 Flexibility as a Key to Efficiency

Maximizing computation efficiency for an RL workload requires flexible orchestration that aligns with component and workflow characteristics. However, adjusting the execution mode without changing the programmed workflow is challenging. Collocated and disaggregated pipelining modes differ significantly. Collocated mode operates at coarse-grained, phase-level execution. Each phase starts after the previous phase is complete. However, disaggregated pipelining runs at fine-grained, batch-level with precise timing to minimize pipeline bubbles. Mixing these modes further increases complexity. We advocate a system design that bridges this gap, enabling RL developers to maintain an intuitive, logically organized workflow while achieving high execution efficiency with flexible execution modes.

# 3 RLinf Design

# 3.1 Overview

In pursuit of efficient, flexible, and intuitive RL systems, we propose a new design paradigm termed M2Flow, i.e., macro logical flow executed with micro execution flow. In this paradigm, developers program the complex RL workflow by imperatively specifying the logical communication flow among the RL components at a coarse granularity (macro logical flow), while the system automatically transforms the workflow into a fine-grained execution flow (micro execution flow). Essentially, M2Flow decouples programmable code logic from the physical execution and scheduling of the individual RL components, so as to maximize the efficiency while minimizing the programming complexity.

Figure 4 shows the architecture of RLinf that realizes M2Flow. As illustrated, RLinf provides an easy-to-use procedural programming interface for users to construct RL workflows imperatively, which describe the data communication and interaction among the RL components. RL components are then encapsulated as workers in RLinf, each implementing the main logic of this component. Workers are equipped with communication functionality to freely communicate with each other, as well as resource offloading mechanism to enable temporal multiplexing of hardware resources. This worker abstraction enables RLinf to retain substantial scheduling flexibility at both the spatial and temporal dimensions, while following the procedural workflow. Spatial scheduling assigns workers to accelerators, temporal scheduling determines their execution periods, and spatio–temporal scheduling governs their pipelined execution granularity.

To produce a desirable micro execution flow across these scheduling dimensions, the core of the scheduler module is a profiling-guided scheduling policy, which utilizes runtime profiling of worker characteristics to search for the optimal execution mode of each worker. Based on the determined execution mode, the Controller assigns workers to accelerators, manages inter-worker connections, and orchestrates the execution flow by dispatching function invocations to the corresponding workers. To realize this, two mechanisms named elastic pipelining and context switch are devised to enable spatial and temporal orchestration of workers, respectively. Adaptive communication utilities such as point-to-point communication and data channel act as the data plane to support scalable worker interactions. The entire system leverages Ray [28] to remotely launch and control workers.

# 3.2 Workflow Construction Interface

The design philosophy of RLinf is to maximize system flexibility to achieve high efficiency, which is also the guideline for RLinf’s programming interface design. To this end, unlike traditional graph-based declarative programming [1] that sacrifices control flow flexibility, debuggability and transparency for optimization opportunity, RLinf adopts a procedural programming paradigm that enables developers to flexibly express workflows imperatively. An example workflow based on this interface is shown in Figure 5. As shown, an RL program based on RLinf consists of two parts: (1) worker programs that define the logic of each RL component (e.g., simulator, LLM generation, actor training), and (2) a workflow runner that orchestrates the overall workflow by invoking the workers’ core functions and defining inter-worker interactions.

![](images/1aa68e143ebdfd20a7cecc390fee358bf9bf8779c06c2ca80da1cf1179c3fa7e.jpg)  
Figure 5: RLinf workflow programming interface.

Figure 5a demonstrates a typical worker implementation atop RLinf. The base Worker class provides communication primitives such as send and recv for inter-worker communication. All workers inherit from the base class automatically gains the capability to communicate with other workers (§3.5), which is also the foundation for higher-level communication facilities like the data channel. To manage limited device resources like GPU memory, all workers are required to implement resource management functions (onload and offload) for acquiring and releasing the resources.

After implementing workers, developers can compose the overall RL workflow as in Figure 5b. First, the runner launches workers on a cluster of nodes and devices in an SPMD manner. The scheduler module decides the placement of each worker process before its launch, which can also be manually specified if desired. All processes of the same worker are collectively managed via the WorkerGroup abstraction of RLinf (e.g., rollout_group), which automatically assumes the public functions defined in the worker class and dispatches them to all (or a selective portion) of the worker processes if invoked. The functions of a WorkerGroup is inherently asynchronous and returns a result handle, whose wait primitive provides synchronization barriers, enabling the computations on certain granularity of data. For example, in GRPO training, rollout can proceed per query, but normalization must aggregate all responses for a query, which pauses the pipeline at this step until normalization completes. The core facility for connecting the data flow among distributed worker groups is the data channel (detailed in $\ S 3 . 5 )$ , which decouples the control and data flows of dependent components, enabling highly flexible programming while exposing a broad optimization space, as shown below.

![](images/a72a527588ec7225e0d2b36d80786dfaf71755dba9d6687799018cc7cf00c1a3.jpg)  
Figure 6: The M2Flow execution logic.

# 3.3 M2Flow Transformation

The programming interface shown in Figure 5b offers a flowlike programming model, with which developers describe the high-level, logical control and data flows among workers. Having obtained the logical flows, RLinf follows the M2Flow paradigm to transform the logical flows (i.e., how should the workers run) into the concrete execution flow, i.e., where (spatial) and when (temporal) should the workers run. In this section, we focus on two enabling mechanisms of M2Flow transformation and flexible scheduling—elastic pipelining and context switching. In $\ S 3 . 4$ , we will describe the scheduling policy that determines the optimal execution flow.

Based on the programmed workflow, the key idea of M2Flow transformation is to control the spatial and temporal scheduling of workers by throttling their data processing granularity and concurrent resource accesses, respectively.

Spatial Scheduling via Elastic Pipelining. For spatial scheduling, workers can be executed in a pipelined manner with different number of accelerators/devices. To maximize pipeline flexibility, RLinf introduces elastic pipelining to enable workers to flexibly process data at different granularity with the given device resources. Elastic pipelining builds upon our insight that in RL training and agentic scenarios, most workers follow the SPMD pattern, allowing execution across varying batch sizes. For instance, LLM serving engines like SGLang and vLLM can process a single prompt or a list of prompts at a time, and inference (i.e., prefill-only computation of a model) similarly supports single-batch or multi-batch execution. This flexibility enables the Execution Flow Manager of RLinf to achieve flexible pipelining of a worker task via dynamic data granularity—output data can be forwarded once a configured size of data batch is ready, allowing downstream workers to start earlier with smaller batches or later with larger batches. Notably, the scheduling space is further affected by individual worker’s internal computation semantics. For example, training workers operate with both the micro-batch and global-batch concepts—the microbatch defines forward/backward units, while the global-batch determines when model updates occur.

![](images/bee2085bd1e931dd723d6190ae95ec1a707412b1ea66722bad3b9d4ab4890468.jpg)  
Figure 7: Spatial and temporal scheduling of workers.

Temporal Scheduling via Automatic Context Switching. Beyond spatial scheduling, RLinf also supports natural temporal multiplexing of devices via automatic context switching, further expanding the scheduling space. Context switching enables workers that cannot co-reside in the same accelerators with limited device resources (e.g., GPU memory) to share devices by executing sequentially. In RLinf, this is realized via a distributed device lock of the data channel facility, i.e., device_lock as shown in Figure 5a. This lock serves as the primitive to throttle concurrent resource access by multiple workers on the same devices and have data flow dependencies, i.e., producers and consumers of the same channel. Before using device resources, a worker must acquire the lock, whose state is globally consistent to all workers and can only be changed atomically, thus ensuring exclusive access to the resources. Upon lock acquiring, the worker’s device resources are automatically loaded by calling the onload function of the worker if the resources have been offloaded. After completing its task, the worker releases the lock and offloads its resources via the offload function to free up device resources for children workers in the workflow. Unlike conventional lock, the device lock leverages the data channel’s data dependency information to define lock acquiring priority, i.e., children workers that depend on the outputs of parent workers can only acquire the lock after the parents have enqueued data and released the lock, so as to avoid lock contention and deadlock. Also, it utilizes device placement information from the Controller to avoid unnecessary resource loading and offloading when workers are placed on different devices.

M2Flow Execution. Figure 6 shows how M2Flow manages execution. The workflow is written imperatively, i.e., a for loop iterating over main logic, with three workers, i.e., rollout, inference, and training. A rollout task such as generate(data, ch1) processes data and enqueues results to channel ch1. The Execution Flow Manager can divide the input data into smaller chunks, allowing workers to process outputs at a smaller data granularity. Alternatively, tasks can be coalesced into fewer, larger sub-tasks with larger data chunks to realize different temporal scheduling. Meanwhile, the device lock enables automatic resource management among workers with data dependencies, enabling spatial scheduling via context switching.

With M2Flow, user-defined workflows can be orchestrated across a complete spatial–temporal scheduling space. Figure 7 illustrates several representative execution modes suited to different RL workloads and configurations.

The left part of Figure 7 shows pure temporal scheduling, where each worker occupies all accelerators. Once a worker completes its task, it is swapped out and the next worker is swapped in. For the illustrated case, since inference and training share the same model weights, no offloading or reloading is required. This mode is particularly useful when a worker must use all devices to be runnable, such as a large model’s training. However, it suffers from GPU idleness due to longtail effects, e.g., the longest response length in the rollout stage determines the overall completion time of the stage.

The middle part illustrates spatial scheduling, where workers are assigned to separate GPUs. Because workers depend on one another, pipelining is applied to mitigate idle time. Achieving efficient pipelining requires balancing resources across workers so that their execution times align.

Finally, RL workflows are often too complex for purely temporal or spatial scheduling to remain efficient. M2Flow therefore supports hybrid scheduling, as shown on the right of Figure 7. Some workers are distributed across GPUs with pipelined execution, but once the workers’ stage completes, they can be swapped out and replaced by successors to continue the workflow.

Algorithm 1: Worker scheduling policy.   

<table><tr><td></td><td>Input : Workflow graph G, execution time estimation function E for each component, and number of devices N. Output : A worker schedule Sbest and its estimated time Tbest.</td></tr><tr><td></td><td>1 Dtable ← {};// graph map to (time, schedule)</td></tr><tr><td></td><td>2 Gdag ←ConvertCircleToNode (G);</td></tr><tr><td></td><td>3 best, Sbest ←FindSchedule (Gag, N, able);</td></tr><tr><td></td><td>4 Function FindSchedule (G, N, Dtable):</td></tr><tr><td>5</td><td>if (G,N ) in Dable then</td></tr><tr><td>6</td><td>return Dtable[(G,N)];</td></tr><tr><td>7</td><td>end</td></tr><tr><td>8</td><td>if G is a node then</td></tr><tr><td>9</td><td>return Enode, Snode;</td></tr><tr><td>10</td><td>end</td></tr><tr><td>11</td><td>Tbest ← +inf; Sbest ←None;</td></tr><tr><td>12</td><td>for Gs, Gt in TraverseStCuts (G) do</td></tr><tr><td>13</td><td>/* Gs and Gt share the same gpus */</td></tr><tr><td>14</td><td>, ←FindSchedule (, , able);</td></tr><tr><td>15</td><td>,S ←FindSchedule (G, , able);</td></tr><tr><td>16</td><td>able[(Gs,N)]←(, );</td></tr><tr><td>17</td><td>ale[(G,N)]←(, S);</td></tr><tr><td>18</td><td>if best &gt;  +  then</td></tr><tr><td>19</td><td>Tbest ←+; Sbest ←shared(,  );</td></tr><tr><td>20</td><td>end</td></tr><tr><td>21</td><td>/* Gs and Gt use different devices */</td></tr><tr><td>22</td><td>for Ns, Nt in TraverseGpuNum (N) do</td></tr><tr><td>23</td><td>/* Ns+Nt equals N */</td></tr><tr><td>24</td><td>, ←FindSchedule (, s, able);</td></tr><tr><td>25</td><td>, ←FindSchedule (, t, able);</td></tr><tr><td>26</td><td>table[(G,Ns)]←(, );</td></tr><tr><td>27</td><td>table[(Gt,Nt)]←(,  );</td></tr><tr><td>28</td><td>if Tbest &gt;PipeliningTime (s, T) then</td></tr><tr><td>29</td><td>Tbest ←PipeliningTime (s, t);</td></tr><tr><td>30</td><td></td></tr><tr><td></td><td>Sbes ←pipeline(, );</td></tr><tr><td>31</td><td>end</td></tr><tr><td>32</td><td>end</td></tr><tr><td>33</td><td>end</td></tr><tr><td>34</td><td>return Tbest, Sbest;</td></tr><tr><td>35 end</td><td></td></tr></table>

# 3.4 Scheduling Policy

RLinf offers a large scheduling space through flexible orchestration of workers, thus finding the most efficient execution mode is challenging. To this end, RLinf introduces two modules: the profiler and the scheduler. The profiler is used to measure and estimate the execution characteristics of each component under different numbers of GPUs. The scheduler then utilizes this information to composite an overall execution plan, containing specific GPU assignments and pipelining configurations.

Profiler. The profiler measures each component’s execution time and memory usage under different data parallel sizes, because data parallel size tend to have positive (sometimes nearly linear) relation with the component throughput as data is consumed faster with the increasing data parallel size. For non-model components like simulators, the data parallel size is the number of instances. For model training and generation components, the profiler requires users to provide model parallel configuration to decide the data parallel size, which can usually be determined by the model size and GPU memory. With the profiled data, the profiler extrapolates the execution time and memory usage for larger data parallel sizes using polynomial extrapolation, outputting an execution time estimation function $E$ for each component. This function is then fed to the scheduler for it to estimate each component’s execution time and whether it can be fitted into memory given a selected number of GPUs. Moreover, during profiling, the overall RL workflow is also captured into a workflow graph in a just-in-time manner, by tracing the data flow among workers through the communication primitives.

Scheduler. The scheduling policy of the scheduler is shown in Algorithm 1. The input of the policy includes the workflow graph $G$ , the execution time estimation function $E$ produced by the profiler, and the total number of GPU devices. Specifically, the scheduling algorithm recursively partitions the workflow graph into two subgraphs, $G _ { s }$ and $G _ { t }$ , connected by directed edges known as the $s { - } t$ cuts [9]. For each partition, it evaluates the time cost of both the temporal and spatial scheduling policies. In the temporal scheduling, $G _ { s }$ and $G _ { t }$ share the same set of devices (e.g., GPUs)— $G _ { s }$ processes its batch, and upon completion $G _ { t }$ consumes its output. In the spatial scheduling, $G _ { s }$ and $G _ { t }$ are assigned to disjoint device sets and executed in a pipelined fashion. The scheduler uses the profiling results to find the optimal device allocation and data processing granularity for each subgraph. The algorithm then selects the most performant scheduling, and applies this process recursively until each subgraph reduces to a single node, at which point the node returns its profiled execution time under the assigned placement.

Modeling the execution time of $G _ { s }$ and $G _ { t }$ is the key to finding the optimal scheduling. In the temporal scheduling where workers share devices, the cost is the sum of $G _ { s }$ and $G _ { t }$ plus any resource offloading and reloading overhead. In the spatial scheduling, the runtime is estimated as

$$
T _ { \mathrm { c r i t i c a l } } + \left( M / m - 1 \right) \times T _ { \mathrm { b o t t l e n e c k } } ,
$$

where $T _ { \mathrm { c r i t i c a l } }$ is the pipeline warm up and cool down time, Tbottleneck is the runtime of the slowest subgraph, $M$ is the total batch size, and $m$ is the data processing granularity.

Before invoking FindSchedule, the workflow graph is preprocessed to collapse cycles into single nodes. When recursion reaches such a node, its computation is evenly partitioned across GPUs. This avoids exhaustive partition enumeration while still achieving near-optimal performance.

# 3.5 Adaptive Communication

To support flexible worker orchestration, RLinf’s communication layer needs to realize two key design goals. (1) Flexible. Any two workers should be able to communicate with each other, regardless of the worker placement and program logic. (2) Adaptive. Communication primitives should be able to adapt to arbitrary data living in different devices (CPUs and GPUs across nodes), while achieving maximum throughput of the underlying communication links.

However, existing collective communication libraries for GPU and CPU data, e.g., NCCL [31], Gloo [34], and MPI [10], fail to meet our goals as they are mostly built for standard communication patterns across a fixed number of processes in traditional model training and serving scenarios. In contrast, RL components manifest considerable spatial and temporal dynamicity. Spatially, components can collocate on the same device, or being distributed across different devices. Temporally, components can be launched or terminated at arbitrary times. However, state-of-the-art libraries like NCCL does not support flexible rank scaling and efficient intra-device communications. Furthermore, the data communicated between components can be complex data structures beyond standard contiguous GPU/CPU data buffer, e.g., a composition of multiple data buffer with varying sizes. Efficient handling of such data dynamicity is also missing in existing libraries.

Communication Protocol and Primitives. To achieve the design goals, we devise worker and data placement-aware communication protocol and primitives, which enhances existing CPU/GPU communication libraries in terms of both flexibility and performance in the RL scenario.

At the protocol level, RLinf features transparent connection lifecycle management, which avoids manual connection management as in traditional communication libraries, and handles dynamic worker placement and scaling automatically. Specifically, upon launch, each worker’s placement, IP and port information will be registered into a global worker manager. Connections among workers are then established with the information lazily when workers invoke communication primitives to reduce connection overhead and enhance scalability. When a group of workers establish a connection, the connection metadata is maintained both locally by workers and globally by a connection manager. When a worker is terminated, the connection manager will notify all connected workers to teardown the connection and release resources.

At the primitive level, like existing libraries, RLinf offers both synchronous and asynchronous send and recv primitives for point-to-point communication, as well as collective communication primitives like broadcast. Differently, RLinf’s primitives automatically exploit the worker and data placement information of the communicating workers to select the most efficient communication backend, e.g., NCCL for GPU-GPU communication, zero-copy cudaIPC for intra-GPU communication, and Gloo for CPU communication. For data dynamicity, both the asynchronous and synchronous primitives support arbitrary Python objects as the communication payload, which are serialized and deserialized in a structure-aware manner, i.e., data buffers are extracted from the objects and communicated directly without serialization/deserialization overhead. Also, data structure information is piggybacked in the communication metadata to facilitate efficient deserialization at the receiver side.

Load-Balancing Data Channel. Atop the above communication primitives, we further build a high-level FIFO queue-like communication facility (termed data channel) for producer-consumer worker communication in the workflow. This enables decoupling of both the control and data flows of producer and consumer workers, which is essential for flexible pipelining. The data channel maintains its data queue in a special channel worker process, and can be accessed by any other worker processes by passing the channel handle. The channel supports both CPU and GPU data, and can be configured for offloading GPU data to CPU to reduce GPU memory consumption. Furthermore, the data channel is enhanced with load-balancing capability. Each item enqueued to the channel can be assigned a weight value, which is used to balance the load across multiple consumers dequeuing from the channel. The consumers can also define custom load-balancing policies, which are invoked by the channel upon each dequeue operation to select the desired items from the channel.

# 4 Implementation

RLinf is implemented in 20K lines of Python code. Among them, 5K lines are for the core worker, controller, and scheduler components. 2K for common workers such as rollout workers based on serving engines like SGLang, vLLM and HuggingFace Transformers, training actors based on Megatron and FSDP, and embodied simulators, which can be used out of the box for any future RL models and workflows. The remaining 13K lines are mostly rich supports for various RL algorithms like PPO and GRPO. Notably, a typical workflow runner like the LLM reasoning RL workflow implementation is less than 100 lines of code, and requires no code changes to be scheduled both temporally and spatially.

Currently, RLinf supports not only traditional LLM-based reasoning RL, but also agentic RL with tooling, and embodied RL involving complex workloads like 3D rendering, physics simulation, and robotic control. For RL algorithms, RLinf supports popular algorithms like PPO [41], GRPO [42], DAPO [52], and REINFORCE $^ { + + }$ [13], as well as some of their off-policy asynchronous versions. For models, we have implemented support for language models like Qwen [35], multi-modal models like VLM [3], and embodied models like VLA [18, 19] and Pi0 [48]. The rich RL workflow, algorithm and model support not only accelerates the development of new RL workflows with RLinf, but also demonstrates its generality, versatility and extensibility in practice.

Cluster Management and Device Allocation. RLinf leverages Ray to realize cluster management, launching worker processes on remote nodes, and dispatching worker function executions. RLinf does not rely on Ray for device allocation to workers, because Ray only supports rigid packed-style (i.e., consecutive devices) or spread-style (i.e., spread-across-node first) resource allocation [28], which are not flexible enough for today’s complex and dynamic RL workflows. Instead, RLinf offers a flexible device allocation strategy, which allows each worker process to be allocated with any device or devices of any node across the cluster, by simply specifying the target devices’ global IDs. Also, hardware devices beyond accelerators such as robot arms are also abstracted and managed as schedulable devices in the same manner as accelerators. This enables RLinf to scale on all kinds of hardware that foundation models can interact with.

Performance Profiling. Performance profiling support is not only crucial for developers to understand system bottlenecks, but also serves as the key to our scheduling policy. Thus, RLinf provides worker-group-level timer for every public function invoked remotely by the workflow runner. The timer automatically captures the execution time of a worker function, whose value can be retrieved via the asynchronous handle returned by the corresponding worker group function; the values of all processes in the worker group will be reduced to a single value via a specified reduction method (e.g., mean, max, min). Beyond this, developers can also create custom timers for more fine-grained profiling of any code region, and retrieve the timer values similarly.

# 5 Evaluation

We extensively evaluate RLinf across math-reasoning and embodied RL workloads, covering four different models of different sizes (i.e., Qwen2.5 [35], Qwen3-MoE [50], Open-VLA [19], OpenVLA-OFT [18]), two RL algorithms (i.e., GRPO [42], PPO [41]), and multiple cluster scales. Overall, our key findings include:

• RLinf consistently outperforms state-of-the-art RL systems veRL [44] and Slime [60] by $1 . 0 7 \times \sim 1 . 7 0 \times$ on a variety of math-reasoning RL settings. The results also show that different RL settings favor different execution modes. • RLinf demonstrates higher training throughput on embodied RL tasks in both ManiSkill [47] and LIBERO [22]. On LIBERO, RLinf achieves a $1 . 0 5 \times \sim 2 . 4 3 \times$ speedup over SimpleVLA-RL [20]; on ManiSkill, its hybrid mode yields up to $1 . 8 7 \times$ improvement over other strategies. • The scheduling policy identifies the best execution mode within $7 \times 1 0 ^ { - 4 } \sim 5 . 9 8 s$ on the clusters of 8 to 1024 GPUs. • The models trained with RLinf achieve SOTA or better benchmark scores after RL fine-tuning.

![](images/98eca6059a5d2c022dcf6cd70a716bfb1e020eaf29369265d64ece576b168d17.jpg)  
Figure 8: GRPO training throughput of Qwen2.5 on RLinf and veRL under different cluster scales and model sizes.

# 5.1 End-to-End Experiments

The end-to-end experiments run on a 32-node cluster, where each node is equipped with 8 NVIDIA H100-80GB GPUs, 2 Intel Xeon Platinum 8558 CPUs (2.1 GHz, 48 cores), and ${ } ^ { 2 } \mathrm { T B }$ of memory. Intra-node communication uses NVLink, while inter-node communication uses 8 Mellanox ConnectX-7 RDMA NICs per node, each providing 400 Gbps bandwidth with RoCEv2.

# 5.1.1 Reasoning RL Training

Experimental setup. We evaluate using Qwen2.5 Models [35] distilled by DeepSeek-R1,covering sizes from 1.5B to 32B parameters and Qwen3-30B-A3B [50] which is an MoE model, and using AReaL-boba-Data dataset [49], which integrates multiple standard datasets including Deep-ScaleR, Open-Reasoner-Zero, Light-R1, etc. We compare RLinf with two state-of-the-art open-source RL systems, i.e., veRL v0.5 [44] and Slime v0.1 [60]. To ensure fairness, all systems use SGLang [58] for rollouts and Megatron-LM [45] for training, with the same parallelism setting (e.g., tensor parallelism configuration). Training speed is reported in tokens/sec, defined as the total number of prompt and response tokens in a global batch divided by the iteration time. All performance results are averaged over 10 training iterations after warm-up.

Qwen2.5 with GRPO. We evaluate the Qwen2.5 1.5B, 7B, and 32B dense models on 64, 128, and 256 GPUs, respectively, using a rollout batch size of 512 and maximum sequence length 28672. As shown in Figure 8, RLinf in temporal mode consistently outperforms veRL across all model scales, achieving $1 . 1 0 \times$ to $1 . 5 8 \times$ speedups. Since RLinf ’s temporal mode is similar to veRL’s design, the gains primarily stem from faster rollouts enabled by larger KV-cache allocations (RLinf has better GPU memory management) and reduced synchronization overhead between upstream and downstream stages of the middle inference stage, as illustrated in Figure 9.

Figure 8 also shows that veRL scales poorly as GPU number increases, largely because inference becomes a growing bottleneck, with its proportion of total execution time rising from $1 5 . 2 \%$ to $1 9 . 9 \%$ when scaling from 64 to 256 GPUs. In addition, veRL’s unoptimized rollout engine leads to excessive peak memory usage, forcing smaller KV-cache allocations and causing rollout time to decrease only sublinearly.

![](images/c4976a592f1f9712ba431cb70fdcf4412d36013b1de2299e01d01b6c9f993bd6.jpg)  
Figure 9: Latency breakdown of Qwen2.5 7B model training.

![](images/c990787a7021244f1e8934e840028fa0081d1320c145c353d65161e70b53702c.jpg)  
Figure 10: PPO training throughput of Qwen2.5 on RLinf and veRL under different cluster scales and model sizes.

We also tested spatial mode, but it underperforms veRL by $4 4 . 3 \substack { - 6 8 . 6 \% }$ for the 7B mode, because rollout, inference, and training run on disjoint smaller sets of GPUs, slowing both rollout and training. With the long sequence length (i.e., 28672), the training stage also waits longer for the first batch of rollouts to be generated. In addition, the rollout batch size is relatively smaller, reducing the overlapping time.

Qwen2.5 with PPO. We train PPO with Qwen2.5 as both the actor and critic, using model sizes of 1.5B, 7B, and 14B, scaling from 16 to 256 GPUs. We use a rule-based reward, a maximum sequence length of 12288, and a rollout batch size ranging from 256 to 1024 depending on GPU count. RLinf-Spatial allocates GPUs in a 4:1:1:1:1 ratio for rollout, actor inference, actor training, critic inference, and critic training. Figure 10 presents the results. For the 1.5B model, RLinf-Spatial outperforms veRL by $6 9 . 6 \%$ , $3 5 . 0 \%$ , and $3 5 . 6 \%$ on 16, 32, and 64 GPUs, respectively. In contrast to the GRPO experiment, RLinf-Spatial surpasses RLinf-Temporal by $3 9 . 8 \%$ $1 9 . 4 \%$ , and $1 8 . 6 \%$ .

For the 7B model, RLinf-Spatial outperforms both veRL and RLinf-Temporal by $3 8 . 7 \mathrm { - } 6 0 . 7 \%$ and $1 9 . 0 { - } 4 4 . 8 \%$ , respectively. As shown in Figure 11, RLinf-Temporal exceeds veRL mainly due to speedups in “Others”, which includes contextswitch overhead (offload/onload), parameter resharding, advantage computation, and inter-stage synchronization. RLinf-Spatial performs faster than RLinf-Temporal because rollout, inference, and training overlap effectively. Although rollout in spatial mode is $3 9 . 3 \%$ slower than in temporal mode, inference and training finish quickly once rollout completes. The 14B model shows similar trends with smaller gains, i.e., RLinf-

Spatial is $2 7 . 2 \substack { - 5 6 . 5 \% }$ and $1 7 . 7 \mathrm { - } 2 5 . 7 \%$ faster than veRL and RLinf-Temporal, respectively.

Qwen3-30B-A3B with GRPO. We evaluate the MoE model Qwen3-30B-A3B on 32, 64, and 128 GPUs with a rollout batch size of 1536 and sequence length 20480. We disable logprob recomputation, a common configuration in RL, which removes the inference stage. Since Slime is optimized for MoE RL training, we compare against two variants: Slime (spatial mode without pipelining, as Slime does not support pipelining) and Slime-Colocate (temporal mode). As shown in Figure 12, Slime is the slowest because rollout and training run on disjoint GPU sets with no execution overlap. For 32 and 64 GPUs, RLinf-Temporal performs similarly to Slime-Colocate, while RLinf-Spatial (1:1 rollout-to-training GPUs) is $3 1 . 2 \%$ and $7 . 2 \%$ faster than Slime-Colocate, respectively.

Figure 13 presents the performance breakdown. Spatial mode achieves shorter rollout time because temporal mode suffers from memory contention between rollout and training, forcing much smaller parallelism (i.e., max_running_requests of 128 vs. 256 in SGLang). In addition, RLinf-Spatial achieves good rollout–training overlap. On 128 GPUs, RLinf-Temporal outperforms Slime-Colocate by $3 . 7 \%$ due to better scaling from decoupled system design; both stages run slightly faster. RLinf-Spatial underperforms because rollout and training do not overlap well, where training continues for 80 seconds after rollout completes.

# 5.1.2 Embodied RL Training

Experimental setup. We evaluate on OpenVLA [19] and OpenVLA-OFT [18], two vision-language-action models that are supervised-finetuned using RL4VLA [23] and SimpleVLA-RL [20], respectively. We train OpenVLA on ManiSkill [47] and OpenVLA-OFT on LIBERO [22], two widely used embodied environments that emulate physical tasks such as pick-and-place. On LIBERO, we compare RLinf with SimpleVLA-RL (commit d001d) [20], which is built on veRL. On ManiSkill, no distributed RL baseline exists, so we compare different execution modes of RLinf. Training speed is reported in steps/sec, computed as the total number of environment steps divided by the iteration time.

ManiSkill environment. We train OpenVLA on the “PutCarrotOnPlateInScene-v2” task [23] using 256 parallel ManiSkill environments, each stepping for 80 steps per iteration. Figure 14(a) reports end-to-end throughput across three execution modes in RLinf on 8, 16, and 32 GPUs. RLinf-Hybrid achieves $5 2 . 2 \substack { - 6 9 . 1 \% }$ higher throughput than RLinf-Temporal and $6 0 . 7 \mathrm { - } 8 7 . 2 \%$ higher than RLinf-Spatial. For ManiSkill environment, the per-step environment time remains nearly constant as parallelism increases (Figure 3b) and scaling parallel environments is primarily limited by GPU memory. Thus, dedicating GPUs to environment execution is advantageous. As rollout (i.e., environment and generation)

![](images/5e37a5acc23b4f5570c0f36e2b70b2277346c4a0de2e786a22608a3770e31bb5.jpg)  
Figure 11: Latency breakdown of Qwen2.5 7B with PPO on 32 GPUs. The width of bar presents the number of GPUs.

![](images/2ef92664b11d14821fec384f15998affe1eea3f901d270dece4e16ab5fcb899a.jpg)  
Figure 12: RL training throughput of Qwen3-30B-A3B on RLinf and Slime under different cluster scales.

![](images/35b4434927f47ecc0a103dce9e709624330b09cdc498d99c0b2f050741ffda2d.jpg)  
Figure 13: Latency breakdown of Qwen3- 30B-A3B with GRPO on 32 GPUs. The width of bar is proportional to the number of GPUs.

![](images/2e7e3d2cf2f998424d245f76938f9586169f401bc72fee0c72e4e2e59095bbb0.jpg)  
Figure 14: End-to-end throughput of RLinf and SimpleVLA-RL under different cluster scales.

![](images/0af3034d555c547e8debb0b17400c3a75d5e83b8d0b15851d628aca346de24a9.jpg)  
Figure 15: Latency breakdown of ManiSkill and LIBERO.

and training all require GPU memory, temporally sharing GPUs between rollout and training is more effective. Thus, RLinf-Hybrid performs much better than the other two. In RLinf-Temporal, environments and generation must coexist in memory, constraining both the number of environments and the batch size, ultimately reducing GPU utilization. RLinf-Spatial splits GPUs across the components, forcing training to retain a sufficient number of GPUs to be runnable, which leaves too few GPUs for rollout and significantly slows it down.

LIBERO environment. We train OpenVLA-OFT on the public benchmark task groups provided by LIBERO, using 512 parallel environments, each running 64 steps per iteration. The results in Figure 14(b) show that RLinf-Temporal is $3 7 . 8 \%$ , $4 2 . 6 \%$ , $1 4 3 . 4 \%$ faster than SimpleVLA-RL on 8, 16, 32 GPUs respectively. As shown in Figure 15, both rollout and training are faster in RLinf-Temporal than in SimpleVLA-RL due to (1) the elimination of redundant environment initialization during rollout, and (2) the use of a single forward pass to compute both action and log probability, with only a modest memory increase. RLinf-Temporal also scales better thanks to its decoupled system design and more efficient implementation. In contrast to ManiSkill, RLinf-Hybrid performs worse than RLinf-Temporal because LIBERO environment is CPUintensive; allocating environments on a subset of GPUs limits the utilization of CPU cores, making its rollout stage even slower than that of SimpleVLA-RL.

![](images/44a26d62dd8b8de648a8fb1af2795debbdbacccfeb9fa54b45424c1a83550b1c.jpg)  
Figure 16: (a) The real and predicted latency in different cases. (b) The search overhead under different cluster scales.

# 5.2 Effectiveness of Search Policy

We evaluate the search policy along two dimensions: (a) the accuracy of end-to-end throughput estimates derived from profiling data, and (b) the search speed as the number of GPUs increases.

Estimation accuracy. Figure 16(a) compares real and predicted iteration times across four Qwen2.5–GRPO settings. Cases 1 and 2 use the 1.5B model on 128 GPUs with temperatures 0.6 and 1.0, while Cases 3 and 4 use the 7B model on 64 GPUs with the same temperature settings. Temporal mode achieves estimation errors below $2 \%$ because its end-to-end time is well approximated by summing profiled worker times. Spatial mode with pipelining shows slightly higher errors $( < 5 \% )$ , mainly due to response-length variability that causes pipeline imbalance. These errors are small enough that they do not change the ranking of feasible execution modes, allowing the search policy to consistently select the most efficient mode for the above end-to-end experiments.

Table 1: Evaluation scores of RLinf-math-1.5B/7B and opensource models. GPQA represents GPQA-diamond.   

<table><tr><td>Model</td><td>AIME24</td><td>AIME25</td><td>GPQA</td><td>Avg.</td></tr><tr><td colspan="5">1.5B models</td></tr><tr><td>Base model</td><td>28.33</td><td>24.90</td><td>27.45</td><td>26.89</td></tr><tr><td>DeepScaleR [25]</td><td>40.41</td><td>30.93</td><td>27.54</td><td>32.96</td></tr><tr><td>AReaL [11]</td><td>44.42</td><td>34.27</td><td>33.81</td><td>37.50</td></tr><tr><td>FastCuRL [46]</td><td>43.65</td><td>32.49</td><td>35.00</td><td>37.05</td></tr><tr><td>RLinf</td><td>48.44</td><td>35.63</td><td>38.46</td><td>40.84</td></tr><tr><td colspan="5">7B models</td></tr><tr><td>Base model</td><td>54.90</td><td>40.20</td><td>45.48</td><td>46.86</td></tr><tr><td>AReaL [11]</td><td>61.66</td><td>49.38</td><td>46.93</td><td>52.66</td></tr><tr><td>Skywork [12]</td><td>66.87</td><td>52.49</td><td>44.43</td><td>54.60</td></tr><tr><td>Polaris [2]</td><td>68.55</td><td>51.24</td><td>43.88</td><td>54.56</td></tr><tr><td>AceMath [24]</td><td>67.30</td><td>55.00</td><td>45.57</td><td>55.96</td></tr><tr><td>RLinf</td><td>68.33</td><td>52.19</td><td>48.18</td><td>56.23</td></tr></table>

Table 2: OpenVLA model success rate results on ManiSkill3.   

<table><tr><td>Model</td><td>Vision</td><td>Semantic</td><td>Position</td><td>Avg.</td></tr><tr><td>RL4VLA [23]</td><td>80.47%</td><td>75.00%</td><td>81.77%</td><td>79.15%</td></tr><tr><td>RLinf</td><td>82.03%</td><td>78.35%</td><td>85.42%</td><td>81.93%</td></tr></table>

Search speed. Search complexity is dominated by the number of workflow-graph nodes and available GPUs. RL workflows typically have fewer than ten nodes, allowing our dynamic-programming search to finish quickly. For a graph with three nodes (e.g., Qwen2.5-GRPO or ManiSkill) and GPU counts ranging from 8 to 1024 (Figure 16(b)), search time grows exponentially with GPU count but remains under five seconds, demonstrating the efficiency of our algorithm.

# 5.3 Model Performance

We comprehensively evaluate the performance of the models produced by the reasoning and the embodied RL training, respectively, to demonstrate the effectiveness of RLinf.

Reasoning RL Training. We applied GRPO training, a built-in RL algorithm in RLinf, on two base models, i.e., DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B, to improve their math reasoning ability. As shown in Table 1, the models trained with RLinf achieve the best average performance compared to the baseline systems. RLinf 1.5B model outperforms the baselines across all the three benchmarks (i.e., AIME 24 [54], AIME 25 [55], GPQAdiamond [39]), improving by up to 20 points over its base model. RLinf 7B model achieves the highest performance on GPQA-diamond.

Embodied RL Training. We also evaluate the performance of the models after embodied RL training using RLinf. Table 2 and Table 3 present the evaluation results on the ManiSkill and LIBERO tasks by training OpenVLA and OpenVLA-OFT models respectively, using the PPO algorithm. For ManiSkill, the trained OpenVLA achieves higher success rate than the RL4VLA [23] baseline system. For LIBERO, we further train the publicly released OpenVLA-OFT model (one-trajectory fine-tuned version) from SimpleVLA-RL [20]. By comparing its task success rates with those of the model RL trained using RLinf, we find that RLinf significantly improves OpenVLA-OFT’s performance on the LIBERO tasks.

Table 3: OpenVLA-OFT success rate after RL on LIBERO.   

<table><tr><td>Model</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>OpenVLA-OFT(one-traj) [20]</td><td>56.45%</td><td>25.60%</td><td>45.59%</td><td>9.68%</td><td>34.33%</td></tr><tr><td>RLinf</td><td>98.99%</td><td>98.99%</td><td>98.99%</td><td>94.35%</td><td>97.83%</td></tr></table>

# 6 Related Works

RL Training Frameworks. RL training frameworks adopt varying system designs to support large-scale alignment tasks [6, 33], typically falling into either task-colocated or task-separated execution modes. For task-colocated systems, DeepSpeed Chat [51] and veRL [44] put all training phases on shared GPUs for simplified orchestration. In contrast, task-separated systems like NeMo-Aligner [43] and Open-RLHF [14] divide components across devices to improve modularity and scalability. AReal [11] further introduces asynchronous model updates algorithm upon task-separated systems to increase training throughput. Unlike these frameworks, RLinf provides a more flexible component-to-device placement mechanism, enabling practitioners to explore and deploy more efficient execution configurations tailored to workload characteristics.

Distributed Training Systems for LLM. Distributed training frameworks for LLMs [21, 56, 57] and systemlevel optimizations for memory [5, 15, 27] and communication [4, 17, 26] have significantly advanced large-model scaling. Megatron-LM [45] combines tensor, pipeline, and data parallelism to improve scalability, while DeepSpeed [38] introduces the ZeRO family of techniques [36, 37, 40] that shard optimizer states, gradients, and activations to reduce GPU memory usage. Although distributed RL systems face similar multi-device scaling challenges, RL workflows are more dynamic due to interactive data generation and asynchronous updates, making system design and optimization substantially more complex.

Dataflow System. Traditional dataflow systems [7, 16, 29, 53] excel at large-scale data processing through static task graphs and centralized scheduling, making them efficient for batch and streaming workloads with predictable structures. In contrast, RL training pipelines feature dynamic task graphs and asynchronous components (e.g., data collection, policy updates), requiring more flexible coordination than existing systems. Modern frameworks such as Ray [28] fill this gap with actor-based execution and decentralized control, making them better suited for orchestrating complex RL workflows.

# 7 Conclusion

Reinforcement learning is poised to surpass pretraining as the driving force behind LLM progress, but its workflows are too diverse and dynamic for rigid execution models. RLinf shows that by decoupling workflow logic from execution through the novel macro-to-micro transformation mechanism, we can unlock both efficiency and programmability. Beyond RL, we see this approach as a blueprint for future AI runtimes: systems that flexibly orchestrate heterogeneous components, e.g., training, inference, simulation, and reasoning, under one unified execution framework. We believe RLinf marks an early step toward the operating system for AI workloads.

# References

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: a system for large-scale machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16), pages 265–283, 2016.

[2] Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. Polaris: A posttraining recipe for scaling reinforcement learning on advanced reasoning models, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[4] Chang Chen, Xiuhong Li, Qianchao Zhu, Jiangfei Duan, Peng Sun, Xingcheng Zhang, and Chao Yang. Centauri: Enabling efficient scheduling for communicationcomputation overlap in large model training via communication partitioning. In Proceedings of ASPLOS, pages 178–191, 2024.

[5] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

[6] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 2017.

[7] Jeffrey Dean and Sanjay Ghemawat. Mapreduce: simplified data processing on large clusters. Communications of the ACM, 51(1):107–113, 2008.

[8] Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and Cheston Tan. A survey of embodied ai: From simulators to research tasks. Proceedings of IEEE, 2022.

[9] Lester Randolph Ford and Delbert Ray Fulkerson. Flows in networks. 2015.

[10] Message Passing Interface Forum. Mpi: A messagepassing interface standard. https://www.mpi-forum. org, 2025.

[11] Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. arXiv preprint arXiv:2505.24298, 2025.

[12] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Yang Liu, and Yahui Zhou. Skywork open reasoner series. https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680, 2025. Notion Blog.

[13] Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

[14] Jian Hu, Xibin Wu, Zilin Zhu, Weixun Wang, Dehao Zhang, Yu Cao, et al. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

[15] Zixiao Huang, Junhao Hu, Hao Lin, Chunyang Zhu, Yueran Tang, Quanlu Zhang, Zhen Guo, Zhenhua Li, Shengen Yan, Zhenhua Zhu, Guohao Dai, and Yu Wang. Stalloc: Enhancing memory efficiency in large-scale model training with spatio-temporal planning, 2025.

[16] Michael Isard, Mihai Budiu, Yuan Yu, Andrew Birrell, and Dennis Fetterly. Dryad: distributed data-parallel programs from sequential building blocks. In Proceedings of EuroSys, pages 59–72, 2007.

[17] Anand Jayarajan, Jinliang Wei, Garth Gibson, Alexandra Fedorova, and Gennady Pekhimenko. Priority-based parameter propagation for distributed dnn training. Proceedings of MLSys, 1:132–145, 2019.

[18] Moo Jin Kim, Chelsea Finn, and Percy Liang. Finetuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[20] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning, 2025.

[21] Shenggui Li, Hongxin Liu, Zhengda Bian, Jiarui Fang, Haichen Huang, Yuliang Liu, Boxiang Wang, and Yang You. Colossal-ai: A unified deep learning system for large-scale parallel training. In Proceedings of ICPP, pages 766–775, 2023.

[22] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[23] Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What can RL bring to VLA generalization? an empirical study. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[24] Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint, 2024.

[25] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radiob75.notion.site/DeepScaleR-Surpassing-O1- Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2, 2025. Notion Blog.

[26] Kshiteej Mahajan, Ching-Hsiang Chu, Srinivas Sridharan, and Aditya Akella. Better together: Jointly optimizing ml collective scheduling and execution planning using syndicate. In Proceedings of NSDI, pages 809– 824, 2023.

[27] Chen Meng, Minmin Sun, Jun Yang, Minghui Qiu, and Yang Gu. Training deeper models by gpu memory optimization on tensorflow. In Proceedings of ML Systems Workshop in NIPS, volume 7, 2017.

[28] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I Jordan, et al. Ray: A distributed framework for emerging $\{ \mathrm { A I } \}$ applications. In 13th USENIX symposium on operating systems design and implementation (OSDI 18), pages 561–577, 2018.

[29] Derek G Murray, Frank McSherry, Rebecca Isaacs, Michael Isard, Paul Barham, and Martín Abadi. Naiad: a timely dataflow system. In Proceedings of SOSP, pages 439–455, 2013.

[30] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[31] NVIDIA. Nvidia collective communications library (nccl). https://developer.nvidia.com/ nccl, 2025.

[32] OpenAI. Learning to reason with llms. https://openai.com/index/learning-toreason-with-llms/, 2025.

[33] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[34] PyTorch. Gloo: Collective communications library. https://github.com/pytorch/gloo, 2025.

[35] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

[36] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In Proceedings of SC, pages 1–16, 2020.

[37] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking the gpu memory wall for extreme scale deep learning. In Proceedings of SC, pages 1–14, 2021.

[38] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of SIGKDD, pages 3505–3506, 2020.

[39] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduatelevel google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

[40] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. Zero-offload: Democratizing billion-scale model training. In Proceedings of ATC, pages 551–564, 2021.

[41] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[42] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[43] Gerald Shen, Zhilin Wang, Olivier Delalleau, Jiaqi Zeng, Yi Dong, Daniel Egert, Shengyang Sun, Jimmy Zhang, Sahil Jain, Ali Taghibakhshi, et al. Nemo-aligner: Scalable toolkit for efficient model alignment. arXiv preprint arXiv:2405.01481, 2024.

[44] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

[45] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

[46] Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, and Feng Zhang. Fastcurl: Curriculum reinforcement learning with stage-wise context scaling for efficient training r1-like reasoning models, 2025.

[47] Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao

Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.

[48] Physical Intelligence team. openpi holds open-source models and packages for robotics. https://github. com/Physical-Intelligence/openpi, 2025.

[49] Huggingface xDAN-datasets (xDAN Back). xdandatasets/areal-boba-data. https://huggingface. co/datasets/xDAN-datasets/AReaL-boba-Data, 2025.

[50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[51] Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, et al. Deepspeed-chat: Easy, fast and affordable rlhf training of chatgpt-like models at all scales. arXiv preprint arXiv:2308.01320, 2023.

[52] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[53] Matei Zaharia, Reynold S Xin, Patrick Wendell, Tathagata Das, Michael Armbrust, Ankur Dave, Xiangrui Meng, Josh Rosen, Shivaram Venkataraman, Michael J Franklin, et al. Apache spark: a unified engine for big data processing. Communications of the ACM, 59(11):56–65, 2016.

[54] Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.

[55] Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

[56] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

[57] Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P Xing, et al. Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning. In Proceedings of OSDI, pages 559–578, 2022.

[58] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs, 2024.

[59] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. arXiv preprint arXiv:2504.03160, 2025.

[60] Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025. GitHub repository. Corresponding author: Xin Lv.