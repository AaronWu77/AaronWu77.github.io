# Navigation World Models

**Author:** Amir Bar et al.  
**Date:** 2024

## 1. Abstract

Navigation is a fundamental skill of agents with visual-motor capabilities. We introduce a Navigation World Model (NWM), a controllable video generation model that predicts future visual observations based on past observations and navigation actions. To capture complex environment dynamics, NWM employs a Conditional Diffusion Transformer (CDiT), trained on a diverse collection of egocentric videos of both human and robotic agents, and scaled up to 1 billion parameters. In familiar environments, NWM can plan navigation trajectories by simulating them and evaluating whether they achieve the desired goal. Unlike supervised navigation policies with fixed behavior, NWM can dynamically incorporate constraints during planning. Experiments demonstrate its effectiveness in planning trajectories from scratch or by ranking trajectories sampled from an external policy. Furthermore, NWM leverages its learned visual priors to imagine trajectories in unfamiliar environments from a single input image, making it a flexible and powerful tool for next-generation navigation systems.

NWM采用了条件扩散Transformer(CDiT, Conditional Diffusion Transformer)，在包含人类和机器人智能体的多样化第一视角视频数据集上进行训练，模型参数规模扩展至 10 亿。
- 熟悉环境中：NWM可以通过模拟导航轨迹并评估其是否实现预期目标来进行规划。
- 不熟悉环境中：NWM能够利用其学习到的视觉先验知识，从单张输入图像中想象轨迹。

有监督的导航策略(Supervised Navigation Policies):
- 代表模型：GNM(Generative Navigation Model), NoMaD, SF(Speak Followers), NvEM
- 概述：通过学习输入（视觉语言信息）与输出（导航动作）之间的映射关系来实现导航任务。监督式训练的核心是构建损失函数，最小化模型预测与专家数据的差异（损失函数：回归损失，分类损失，轨迹匹配损失等）。

监督主要来源于：
- 人工标注的导航轨迹
- 专家决策数据
- 高质量的仿真模型（比如Habitat）

有监督导航策略的核心目标：让智能体在与训练数据相似的环境中，直接根据观测信息输出符合专家逻辑的导航动作，无需通过试错（如强化学习的 “奖励 - 惩罚” 反馈）优化策略。

---

## 2. Introduction

When human agents plan, they often imagine their fu-ture trajectories considering constraints and counterfactu-als. On the other hand, current state-of-the-art robotics navigation policies [53, 55] are “hard-coded”, and after training, **new constraints cannot be easily introduced** (e.g. “no left turns”). Another limitation of current supervised visual navigation models is that they **cannot dynamically allocate more computational resources to address hard problems**. We aim to design a new model that can mitigate these issues.


人类在规划的时候，常常会结合符合约束条件，以及反设事实(Conterfactuals)来想象未来的轨迹。但是目前最先进的机器导航策略是“硬编码”的，训练完成后，新的约束条件无法被轻易引入（例如“禁止左转”）。另外一个问题是：现有有监督视觉导航模型的另一局限在于，无法为解决复杂问题动态分配更多计算资源。本文所提出的NVM旨在解决这个问题。

---

In this work, we propose a Navigation World Model (NWM), **trained to predict the future representation of a video frame based on past frame representation(s) and action(s)**(see Figure 1(a)). NWM is trained on video footage and navigation actions collected from various robotic agents. After training, NWM is used to plan novel navigation trajectories by simulating potential navigation plans and verifying if they reach a target goal (see Figure 1(b)). 

![alt text](image.png)

本文提出的 NWM 的训练目标是基于过往帧表征和动作预测视频帧的未来表征（见图 1 (a)）。NWM 采用来自多种机器人智能体的视频片段和导航动作进行训练。训练完成后，NWM 可通过模拟潜在导航方案并验证其是否抵达目标，从而规划全新导航轨迹（见图 1 (b)）。

---

NWM is conceptually similar to recent diffusion-based world models for **offline model-based reinforcement learning**, such as DIAMOND [1] and GameNGen [66]. However, unlike these models, NWM is trained across a wide range of environments and embodiments, leveraging the diversity of navigation data from robotic and human agents. This allows us to train a large diffusion transformer model capable of scaling effectively with model size and data to adapt to multiple environments. Our approach also shares similarities with **Novel View Synthesis (NVS)** methods like NeRF [40], Zero-1-2-3 [38], and GDC [67], from which we draw inspiration. However, unlike NVS approaches, our goal is to train a single model for navigation across diverse environments and model temporal dynamics from natural.

NWM 在概念上类似于基于扩散的离线模型强化学习世界模型（如 DIAMOND [1] 和 GameNGen [66]）。

NWM 相比于这些模型的优势：利用了大量来自于机器人和人类智能体的多样化导航数据，在各种环境和体现形式中进行训练。使得 NWM 能够适应多种环境。

NWM 还与新颖视图合成（NVS， Novel View Sythensis）方法有相似之处，但不同的是，NWM 的目标是训练一个单一模型，在多样化环境中进行导航，并从自然场景中建模时间动态。

---

To learn a NWM, we propose a novel Conditional Diffusion Transformer (CDiT), trained to predict the next image state given past image states and actions as context. Unlike a DiT [44], CDiT’s computational complexity is **linear with respect to the number of context frames**, and it scales favorably for models trained up to 1B parameters across diverse environments and embodiments, **requiring 4x fewer FLOPs compared to a standard DiT** while achieving better future prediction results. 

构建 NWM 我们使用的是 CDiT，其训练目标是基于过往图像状态和动作作为上下文，预测下一图像状态。与 DiT 不同，CDiT 的计算复杂度与上下文帧数量呈线性关系，在多种环境和智能体形态下，模型参数可顺利扩展至 10 亿，与标准 DiT 相比，计算量减少 4 倍，同时未来预测效果更优。

---

In unknown environments, our results show that NWM benefits from training on unlabeled, action- and reward-freevideo data from Ego4D. Qualitatively, we observe improved video prediction and generation performance on single images (see Figure 1(c)). Quantitatively, with additional unlabeled data, NWM produces more accurate predictions when evaluated on the held-out Stanford Go [24] dataset

在未知环境中，实验结果表明，NWM 能从 Ego4D 的无标签、无动作、无奖励视频数据中获益。定性分析显示，NWM 在单图像上的视频预测和生成性能得到提升（见图 1 (c)）；定量分析表明，加入额外无标签数据后，NWM 在预留的斯坦福 Go 数据集上的预测精度更高。

---

## 3. Navigation World Models

### 3.1. Formulation

We are given an egocentric video dataset together with agent navigation actions **$D = {(x_0, a_1, ..., x_T, a_T)}_{i=1}^n$** such that $x_i\in R^{H\times W\times 3}$ is an image and $a_i = (u, \phi)$ is a
navigation command given by translation parameter $u\in R^2$ that controls the change in forward/backward and right/left motion, as well as $\phi \in R$ that controls the change in yaw
rotation angle

$D = {(x_0, a_1, ..., x_T, a_T)}_{i=1}^n$
- **$x_i\in R^{H\times W\times 3}$**：表示图像信息，H、W 分别为图像的高度和宽度
- **$a_i = (u, \phi)$**：表示导航指令
  - **$u\in R^2$**：平移参数，控制前后和左右运动的变化
  - **$\phi \in R$**：控制偏航旋转角度的变化

上述的这个模型可以轻易的扩展到三维，$u\in R^3$ 和 $\phi \in R^3$ 分别表示 偏航角(yaw), 俯仰角(pitch) 和 翻滚角(roll)。

---

The navigation actions $a_i$ can be fully observed (as in Habitat [49]), e.g. moving forward towards a wall will trigger a response from the environment based on physics, which will lead to the agent staying in place, whereas in other environments the navigation actions can be approximated based on the change in the agent’s location.

导航动作 $a_i$ 可完全观测：例如，向墙壁前进时，环境会基于物理规律产生反馈，导致智能体保持静止；而在其他环境中，导航动作可通过智能体位置变化近似获取。

---

Our goal is to learn a world model $F$ , $a$ stochastic mapping from previous latent observation(s) $s_\tau$ and action $a_\tau$ to future latent state representation $s_{t+1}$ :
$$
s_i = enc_\theta(x_i)
$$
$$
s_{\tau+1} \sim F_\theta(s_{\tau + 1} | s_\tau, a_\tau)
$$

Where $s_\tau = (s_\tau, ..., s_{\tau-m})$ are the past m visual observations encoded via a pretrained VAE [4]. Using a VAE has the benefit of working with compressed latents, allowing to decode predictions back to pixel space for visualization.
Due to the simplicity of this formulation, it can be naturally shared across environments and easily extended to more complex action spaces, like controlling a robotic arm.Different than [20], we aim to train a single world model across environments and embodiments, without using task or action embeddings like in [22].

我们的目标是要构建世界模型 $F$ ，其核心作用是 “模拟环境规律”—— 让模型根据历史的环境观测和智能体的动作，预测未来的环境状态。在本文中，世界模型 F 被定义为 “随机映射”，这意味着它预测的未来状态不是唯一确定的（符合真实环境中存在噪声、动态障碍物等不确定性的特点），而非像确定性模型那样输出固定结果。例如：机器人在室内导航时，根据当前看到的 “门口” 图像（观测 s）和 “向前走 1 米” 的动作（a），世界模型 F 不会只预测 “下一步看到客厅” 这一种结果，而是会给出 “看到客厅”“看到客厅且有行人经过” 等多种可能状态的概率分布，以此应对环境中的不确定性。

$$
s_i = enc_\theta(x_i)
$$

编码器 $enc_\theta$ 的作用是将高维图像 $x_i$ 压缩为低维潜在表征 $s_i$，以降低后续世界模型的计算复杂度。本文采用预训练的 VAE 作为编码器。这么做的目的是：原始图像数据冗余度高（如相邻像素信息重复），直接用于后续预测会导致计算量过大；而潜在变量 si​ 能提取图像的 “核心语义特征”（如 “桌子的形状”“墙壁的颜色”），既减少计算成本，又能聚焦关键信息。

$$
s_\tau = (s_\tau, ..., s_{\tau-m})
$$

历史状态整合：为了让模型具备 “记忆” 能力，NWM 会将过去 m 帧的潜在表征整合为历史状态 $s_\tau$，以便更全面地反映环境动态。

$$
s_{\tau+1} \sim F_\theta(s_{\tau + 1} | s_\tau, a_\tau)
$$

在给定 “过去 m 个观测的潜在变量 $s_\tau$​” 和 “当前动作 $a_\tau$​” 的条件下，未来状态 $s_{\tau+1}$​ 的概率分布由世界模型 $F_\theta$ 生成。

---

The formulation in Equation 1 models action but does not allow control over the temporal dynamics. We extend this formulation with a time shift input $k \in [T_{min} , T_{max} ]$, setting $a_\tau = (u, \phi, k)$, thus now $a_\tau$ specifies the time change k, used to determine how many steps should the model move into the future (or past). Hence, given a current state $s_\tau$ , we can randomly choose a timeshift k and use the corresponding time shifted video frame as our next state $s_{\tau + 1}$ .The navigation actions can then be approximated to be asummation from time $m=\tau+k+1$:
$$
u_{\tau\rightarrow m}=\sum_{t=\tau}^{m} u_t\ \ \ \ \   ,\ \ \ \ \   \phi_{\tau\rightarrow m}=\sum_{t=\tau}^{m} \phi_t\ \  mod\  2\ \pi
$$
This formulation allows learning both navigation actions,but also the environment temporal dynamics. In practice, we allow time shifts of up to ±16 seconds. 


公式（1）虽能关联 “观测 - 动作 - 未来状态”，但无法控制 “预测的时间跨度”—— 例如模型只能预测 “下一帧（约 0.25 秒后）” 的状态，若需要预测 “1 秒后”“2 秒后” 的状态，传统方法需多次迭代预测（先预测 0.25 秒后，再以该结果为输入预测 0.5 秒后，以此类推），不仅计算效率低，还会累积误差（每一步预测的微小偏差，多步后会导致结果严重偏离实际）

时间偏移 k:
- $k \in [T_{min} , T_{max} ]$: 是人为设定的 “时间跨度参数”，例如 k=4 代表预测 “4 帧后（1 秒后）” 的状态，k=−2 代表回溯 “2 帧前（0.5 秒前）” 的状态（可用于修正历史预测误差），本文中 k 的范围最高为 ±16 秒；
- $a_\tau = (u, \phi, k)$: 将时间偏移 k 纳入动作向量 a 中，使得模型在预测未来状态时，能同时考虑 “动作” 和 “时间跨度” 两个因素；
- 引入了时间偏移 k 后，模型可以直接根据当前状态 $s_\tau$​、动作 $a_\tau$​ 和时间偏移 k，预测任意时间点的未来状态 $s_{\tau+k+1}$​，无需多次迭代，大大提升了计算效率，且减少了误差累积。

导航动作的近似运算：
当时间跨度为k的时候，模型需要计算k步内的累计动作而不是单步动作
$$
u_{\tau\rightarrow m}=\sum_{t=\tau}^{m} u_t\ \ \ \ \   ,\ \ \ \ \   \phi_{\tau\rightarrow m}=\sum_{t=\tau}^{m} \phi_t\ \  mod\  2\ \pi
$$
第一个表示平移累计，第二个表示旋转累计。
通过这两个累积计算，模型能准确判断 “连续动作组合后，智能体的最终位置和朝向”，进而更精准地预测对应的未来视觉状态。

---

One challenge that may arise is the entanglement of actions and time. For example, if reaching a specific location always occurs at a particular time, the model may learn to rely solely on time and ignore the subsequent actions, or vice versa. In practice, the data may contain natural counterfactuals—such as reaching the same area at different times. To encourage these natural counterfactuals, we sample multiple goals for each state during training. We further explore this approach in Section 4.

在训练数据中，可能存在 “特定动作与特定时间强关联” 的情况。例如，若训练数据中 “到达餐厅” 这一状态总是出现在 “导航开始后 10 秒”，模型可能会学到错误规律 —— 仅根据 “时间达到 10 秒” 就预测 “到达餐厅”，而忽略 “是否实际执行了‘走向餐厅’的动作”；反之，也可能仅根据 “执行了走向餐厅的动作”，就预测 “立即到达餐厅”，而忽略 “实际需要 10 秒才能走到” 的时间规律。这种 “过度依赖单一因素（时间或动作）而忽略另一因素” 的现象，就是 “动作与时间的纠缠”，会导致模型在真实场景中失效（如 “10 秒时若因障碍物绕路，实际未到达餐厅，模型仍会错误预测到达”）。

解决方案：
- 利用数据中的自然反设事实（counterfactuals）：例如，训练数据中可能包含 “10 秒到达餐厅” 和 “15 秒到达餐厅” 两种情况，模型可通过学习这些多样化的时间 - 动作组合，避免过度依赖单一因素；
- 训练时为每个状态采样多个目标：通过为同一状态提供不同的时间 - 动作目标，强制模型学习综合考虑两者，从而减轻纠缠问题。   

---

### 3.2 Diffusion Transformer as World Model