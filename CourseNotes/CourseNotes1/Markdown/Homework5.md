# Homework 5

## Ex.1 (P108, T29)
**题目：** 设 $T \in \mathcal{L}(V)$ 且 $\dim \operatorname{range} T = k$. 证明 $T$ 至多有 $k+1$ 个不同的本征值.

**证明：**
假设 $T$ 至少有 $k+2$ 个不同的本征值。
其中至多有一个本征值为 $0$，因此 $T$ 至少有 $k+1$ 个不同的非零本征值，设为 $\lambda_1, \lambda_2, \dots, \lambda_{k+1}$。
对于每个 $\lambda_i$，存在相应的非零本征向量 $v_i \in V$，使得 $T v_i = \lambda_i v_i$。
因为对应于不同本征值的本征向量是线性无关的，所以这 $k+1$ 个向量 $v_1, v_2, \dots, v_{k+1}$ 是线性无关的。
另一方面，由于 $\lambda_i \neq 0$，我们可以写出 $v_i = \frac{1}{\lambda_i} T v_i = T(\frac{1}{\lambda_i} v_i)$。
这说明对于所有的 $i = 1, 2, \dots, k+1$，都有 $v_i \in \operatorname{range} T$。
从而，我们在 $\operatorname{range} T$ 中找到了 $k+1$ 个线性无关的向量，这意味着 $\dim \operatorname{range} T \ge k+1$。
这与已知条件 $\dim \operatorname{range} T = k$ 矛盾。
因此，假设不成立，$T$ 至多有 $k+1$ 个不同的本征值。


## Ex.1 (P108, T30)
**题目：** 设 $T \in \mathcal{L}(\mathbf{R}^3)$ 且 $-4, 5, \sqrt{7}$ 均为 $T$ 的本征值. 证明存在 $x \in \mathbf{R}^3$ 使得 $Tx - 9x = (-4, 5, \sqrt{7})$.

**证明：**
考察线性算子 $T - 9I \in \mathcal{L}(\mathbf{R}^3)$。
由于 $-4, 5, \sqrt{7}$ 均为 $T$ 的本征值，且 $\mathbf{R}^3$ 为 3 维空间，这三个值即为 $T$ 的所有本征值。
由本征值的性质，$T - 9I$ 的本征值为 $\lambda - 9$，其中 $\lambda$ 为 $T$ 的本征值。
因此 $T - 9I$ 的本征值为 $-4-9=-13$，$5-9=-4$，以及 $\sqrt{7}-9$。
显然，这些本征值均不为零，即 $0$ 不是 $T - 9I$ 的本征值。
这就意味着 $\ker(T - 9I) = \{0\}$，即 $T - 9I$ 是单射。
因为 $T - 9I$ 是有限维向量空间 $\mathbf{R}^3$ 上的线性算子，由线性代数基本定理的推论（单射等价于满射），可知 $T - 9I$ 是满射（即双射）。
因此，对于 $\mathbf{R}^3$ 中的任意向量，包括 $(-4, 5, \sqrt{7})$，都存在唯一的原像。
所以存在 $x \in \mathbf{R}^3$ 使得 $(T - 9I)x = (-4, 5, \sqrt{7})$，即 $Tx - 9x = (-4, 5, \sqrt{7})$。


## Ex.1 (P109, T35)
**题目：** 设 $V$ 是有限维的, $T \in \mathcal{L}(V)$, $U$ 在 $T$ 下不变. 证明 $T/U$ 的每个本征值均为 $T$ 的本征值.

**证明：**
设 $\lambda$ 是商算子 $T/U$ 的一个本征值。
由定义，存在非零向量 $v + U \in V/U$（即 $v \in V$ 且 $v \notin U$），使得 $(T/U)(v + U) = \lambda(v + U)$。
根据商算子的定义，有 $Tv + U = \lambda v + U$，这等价于 $(T - \lambda I)v = Tv - \lambda v \in U$。
记 $u = (T - \lambda I)v$，则 $u \in U$。

现在考察算子 $T - \lambda I$ 在不变子空间 $U$ 上的限制，即 $(T - \lambda I)|_U \in \mathcal{L}(U)$。
分两种情况讨论：
1. 若 $(T - \lambda I)|_U$ 不是单射：
那么存在非零向量 $w \in U$，使得 $(T - \lambda I)w = 0$。即 $Tw = \lambda w$ 且 $w \neq 0$。这直接证明了 $\lambda$ 是 $T$ 的本征值。

2. 若 $(T - \lambda I)|_U$ 是单射：
因为 $U$ 是有限维的，且 $(T - \lambda I)|_U$ 是 $U$ 上的单射线性算子，所以它在 $U$ 上必是满射。
由于前文已知 $u \in U$，由满射性可知，存在 $w \in U$ 使得 $(T - \lambda I)w = u$。
现在考察向量 $x = v - w \in V$。
首先，因为 $v \notin U$ 且 $w \in U$，所以 $v - w \notin U$，从而 $x = v - w \neq 0$。
其次，计算 $(T - \lambda I)x$：
$(T - \lambda I)x = (T - \lambda I)(v - w) = (T - \lambda I)v - (T - \lambda I)w = u - u = 0$。
即 $Tx = \lambda x$ 且 $x \neq 0$。这也证明了 $\lambda$ 是 $T$ 的本征值。

综上所述，无论哪种情况，$\lambda$ 都是 $T$ 的本征值。证明完毕。